# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""First-time glove calibration for the DC GUI.

A glove that has never been calibrated has no
``config/dexumi_exoskeleton/DEXUMI-<Side>.yaml``. The calibration script needs an
existing config to read the glove's joint directions and fingertip polynomial
map from, so this module first copies the glove's design template
(``config/dexumi_exoskeleton/templates/DEXUMI-<Side>.template.yaml``), then runs

    calibrate_exo.py --device DEXUMI-<Side> --version v4

as a child process. With the calibration tool the script asks two questions: ``Press
Enter when ready to record``, once the operator has seated the glove in the tool,
and ``Save to <config>? [y/N]`` once it has read the pose. The GUI shows each
moment and answers it from a button.

A new glove is the only board plugged in (the boards are identical over USB, which
the script also enforces), and a board that a calibration already records is not a
new glove. A glove that is already calibrated can be recalibrated (``start(side,
recalibrate=True)``): the script opens its board by the serial its calibration
records, so other gloves may stay plugged in, and the run starts from the existing
file. The old file is kept as ``.bak`` and put back unless the run saves.

Calibrations belong to a glove, identified by its board's USB serial, but the
recorder works with one file per side (``DEXUMI-Left.yaml``, ``DEXUMI-Right.yaml``).
Every glove's calibration is therefore also kept in a library,
``config/dexumi_exoskeleton/library/DEXUMI-<Side>_<serial>.yaml``. A glove calibrated
for a side that holds another glove's calibration archives that one first, so it
stays available; and a plugged-in glove that only the library records has its
calibration loaded into its side's file (``activate_saved``), the one it replaces
being archived first. The side's previous file is also kept as
``DEXUMI-<Side>.yaml.bak`` during a run and put back if the run does not finish.
"""

import codecs
import collections
import os
import re
import shutil
import signal
import subprocess
import sys
import threading

from gui.dc_ui.core.composer import EXO_CONFIG_DIR, repo_path
from gui.dc_ui.core.devices import _calibration_issue, pico_ports, read_calibration
from gui.dc_ui.core.kinds import REPO_ROOT
from gui.dc_ui.core.supervisor import ANSI_RE

SIDES = ("Left", "Right")
TOOL_VERSION = "v4"
CALIBRATE_SCRIPT = "examples/hand/exoskeleton/calibrate_exo.py"
TEMPLATE_DIR = f"{EXO_CONFIG_DIR}/templates"
LIBRARY_DIR = f"{EXO_CONFIG_DIR}/library"
_LIBRARY_NAME = re.compile(r"^DEXUMI-(Left|Right)_([0-9A-Za-z]+)\.yaml$")
# The two prompts of the calibration-tool route: calibrate_zero_with_tool asks the
# first, save_config the second (only "y" writes the config).
PROMPT_MARKER = "Press Enter when ready to record"
SAVE_MARKER = "[y/N]"
LOG_RING = 400
# SIGINT lets the script print "Calibration cancelled" and release the board;
# the later signals are for a child that ignores it.
STOP_SIGNALS = ((signal.SIGINT, 5.0), (signal.SIGTERM, 3.0), (signal.SIGKILL, 2.0))

(
    IDLE,
    STARTING,
    WAITING,
    RECORDING,
    REVIEW,
    SAVING,
    DONE,
    FAILED,
    CANCELLED,
) = (
    "idle",
    "starting",
    "waiting",
    "recording",
    "review",
    "saving",
    "done",
    "failed",
    "cancelled",
)
LIVE_STATES = (STARTING, WAITING, RECORDING, REVIEW, SAVING)


class CalibrationRefused(RuntimeError):
    """The calibration cannot start (or cannot take this action) right now.

    ``code`` optionally names the refusal for the caller.
    """

    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


def glove_name(side):
    return f"DEXUMI-{side}"


def config_rel(side):
    return f"{EXO_CONFIG_DIR}/{glove_name(side)}.yaml"


def template_rel(side):
    return f"{TEMPLATE_DIR}/{glove_name(side)}.template.yaml"


def library_rel(side, serial):
    return f"{LIBRARY_DIR}/{glove_name(side)}_{serial}.yaml"


def _copy_atomic(source, target):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = f"{target}.tmp"
    shutil.copyfile(source, tmp)
    os.replace(tmp, target)


def _same_file(a, b):
    try:
        with open(a, "rb") as fa, open(b, "rb") as fb:
            return fa.read() == fb.read()
    except OSError:
        return False


def library_entries():
    """The saved calibrations: ``{side, serial, path, calibrated}`` each."""
    folder = repo_path(LIBRARY_DIR)
    entries = []
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        match = _LIBRARY_NAME.match(name)
        if not match:
            continue
        side, serial = match.groups()
        config = read_calibration(f"{LIBRARY_DIR}/{name}")
        recorded = bool(
            config
            and str(config.get("device_serial") or "") == serial
            and config.get("zero_position_angle") is not None
        )
        entries.append(
            {
                "side": side,
                "serial": serial,
                "path": f"{LIBRARY_DIR}/{name}",
                "calibrated": recorded and _calibration_issue(config) is None,
            }
        )
    return entries


def archive(side):
    """Keep the side's current calibration in the library, under its glove's serial.

    Returns the serial archived, or None when the side holds no calibration of a glove.
    """
    status = glove_status(side)
    if not status["serial"]:
        return None
    source = repo_path(status["config_path"])
    target = repo_path(library_rel(side, status["serial"]))
    if not _same_file(source, target):
        _copy_atomic(source, target)
    return status["serial"]


def activate_saved(serials):
    """Load, for each plugged-in board no active calibration records, the library's
    calibration of that board into its side's file; returns ``[{side, serial}]``.

    The calibration it replaces is archived first, so nothing is lost. A board is
    left alone when the library does not know it, knows it for both sides, or when
    two plugged-in boards want the same side or the side's file belongs to a
    plugged-in glove: the operator is not guessed for.
    """
    serials = {str(s) for s in serials}
    for (
        side
    ) in SIDES:  # the library keeps every glove's calibration, however it was made
        archive(side)
    active = {g["serial"] for g in map(glove_status, SIDES) if g["serial"]}
    library = [e for e in library_entries() if e["calibrated"]]
    wanted = {}
    for serial in sorted(serials - active):
        sides = {e["side"] for e in library if e["serial"] == serial}
        if len(sides) == 1:
            wanted.setdefault(sides.pop(), []).append(serial)
    done = []
    for side, boards in wanted.items():
        current = glove_status(side)
        if len(boards) > 1 or current["serial"] in serials:
            continue
        serial = boards[0]
        archive(side)
        _copy_atomic(repo_path(library_rel(side, serial)), repo_path(config_rel(side)))
        done.append({"side": side, "serial": serial})
    return done


def glove_status(side):
    """One side's calibration: whether its config records a board and zeros."""
    config = read_calibration(config_rel(side))
    issue = _calibration_issue(config) if config else None
    recorded = bool(
        config
        and config.get("device_serial")
        and config.get("zero_position_angle") is not None
    )
    return {
        "side": side,
        "name": glove_name(side),
        "config_path": config_rel(side),
        "calibrated": recorded and issue is None,
        "serial": str(config["device_serial"]) if recorded else None,
        "issue": issue if recorded else None,
        "has_template": os.path.isfile(repo_path(template_rel(side))),
    }


def connected_boards():
    """USB serials of the Pico-family boards plugged in (one entry per board)."""
    import serial.tools.list_ports

    serials = set()
    for port in pico_ports(serial.tools.list_ports.comports()):
        if port.get("serial_number"):
            serials.add(str(port["serial_number"]))
    return sorted(serials)


def boards_report(serials=None):
    """Each connected board and what calibrates it.

    ``recorded_by`` is the active calibration (DEXUMI-<Side>) that records the board;
    ``calibrated`` says that calibration is usable; ``saved_sides`` are the sides the
    library holds a calibration of this board for (a board with no ``recorded_by``
    but a saved side is loaded by ``activate_saved``).
    """
    recorded = {}
    for side in SIDES:
        status = glove_status(side)
        if status["serial"]:
            recorded[status["serial"]] = status
    if serials is None:
        serials = connected_boards()
    library = [e for e in library_entries() if e["calibrated"]]
    return [
        {
            "serial": s,
            "recorded_by": recorded[s]["name"] if s in recorded else None,
            "side": recorded[s]["side"] if s in recorded else None,
            "calibrated": bool(s in recorded and recorded[s]["calibrated"]),
            "saved_sides": sorted({e["side"] for e in library if e["serial"] == s}),
        }
        for s in serials
    ]


def saved_gloves(serials=None):
    """Calibrations of gloves that are not plugged in: the active files and the
    library, each glove once (``{side, serial, active}``)."""
    plugged = {str(s) for s in (connected_boards() if serials is None else serials)}
    seen = {}
    for side in SIDES:
        status = glove_status(side)
        if status["serial"]:
            seen[(side, status["serial"])] = True
    for entry in library_entries():
        if entry["calibrated"]:
            seen.setdefault((entry["side"], entry["serial"]), False)
    return [
        {"side": side, "serial": serial, "active": active}
        for (side, serial), active in sorted(seen.items())
        if serial not in plugged
    ]


def default_argv(side):
    return [
        sys.executable,
        CALIBRATE_SCRIPT,
        "--device",
        glove_name(side),
        "--version",
        TOOL_VERSION,
    ]


def _child_env():
    # Restrict PYTHONPATH to this checkout so inherited packages cannot shadow
    # dependencies.
    return {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(REPO_ROOT),
    }


class CalibrationRunner:
    """One calibration at a time, driven through the script's stdin and stdout."""

    def __init__(self, argv_for=None, boards=None):
        self._argv_for = argv_for or default_argv
        self._boards = boards or connected_boards
        self._lock = threading.Lock()
        self._reset()

    def _reset(self):
        self._state = IDLE
        self._side = None
        self._recalibrate = False
        self._proc = None
        self._lines = collections.deque(maxlen=LOG_RING)
        self._seq = 0
        self._pending = ""
        self._message = ""
        self._cancel_requested = False
        self._declined = False

    # -- queries ----------------------------------------------------------------

    def active(self):
        with self._lock:
            return self._state in LIVE_STATES

    def activate_saved(self):
        """Load the library's calibration for plugged-in boards that need it (idle only)."""
        with self._lock:
            if self._state in LIVE_STATES:
                return []
            try:
                return activate_saved(self._boards())
            except Exception:  # noqa: BLE001 - a port listing or copy failure must not break the page
                return []

    def boards(self):
        """The connected boards, each with the calibration that records it."""
        try:
            return boards_report(self._boards())
        except Exception:  # noqa: BLE001 - a failed port listing must not break the page
            return []

    def snapshot(self, since=0):
        with self._lock:
            return {
                "state": self._state,
                "side": self._side,
                "recalibrate": self._recalibrate,
                "name": glove_name(self._side) if self._side else None,
                "message": self._message,
                "seq": self._seq,
                "lines": [
                    {"seq": seq, "text": text}
                    for seq, text in self._lines
                    if seq > int(since)
                ],
                "pending": ANSI_RE.sub("", self._pending),
            }

    # -- actions ----------------------------------------------------------------

    def start(self, side, recalibrate=False):
        """Calibrate the glove that is plugged in alone as ``side``'s; or, with
        ``recalibrate``, redo the calibration the side already holds (its board
        must be plugged in; others may be, since the script opens it by serial)."""
        if side not in SIDES:
            raise CalibrationRefused(f"side must be one of {', '.join(SIDES)}")
        with self._lock:
            if self._state in LIVE_STATES:
                raise CalibrationRefused("a calibration is already running")
            boards = boards_report(self._boards())
            if recalibrate:
                holds = glove_status(side)
                if not holds["calibrated"]:
                    raise CalibrationRefused(
                        f"{glove_name(side)} holds no usable calibration to redo; "
                        "calibrate the glove first"
                    )
                if holds["serial"] not in {b["serial"] for b in boards}:
                    raise CalibrationRefused(
                        f"the {side} glove (board {holds['serial']}) is not plugged in"
                    )
                created, backup = False, self._backup_current(side)
            else:
                created, backup = self._claim_new_glove(side, boards)
            try:
                proc = subprocess.Popen(
                    self._argv_for(side),
                    cwd=str(REPO_ROOT),
                    env=_child_env(),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0,
                    start_new_session=True,
                )
            except Exception as exc:  # noqa: BLE001 - any launch failure must undo the config change
                self._undo(side, created, backup)
                raise CalibrationRefused(
                    f"cannot start the calibration: {exc}"
                ) from exc
            self._reset()
            self._state = STARTING
            self._side = side
            self._recalibrate = recalibrate
            self._proc = proc
        threading.Thread(
            target=self._pump, args=(proc, side, created, backup), daemon=True
        ).start()

    def _claim_new_glove(self, side, boards):
        """Checks and config for the glove calibrated for the first time."""
        if not boards:
            raise CalibrationRefused(
                "no glove board is connected; plug in the glove you are calibrating"
            )
        if len(boards) > 1:
            raise CalibrationRefused(
                f"{len(boards)} glove boards are connected; unplug every glove except the "
                "one you are calibrating (the boards look identical over USB, so the "
                "calibration can only recognise a glove that is plugged in alone)"
            )
        if boards[0]["recorded_by"]:
            raise CalibrationRefused(
                f"this glove is already calibrated ({boards[0]['recorded_by']} records "
                "its board); to redo it, use Recalibrate on its card"
            )
        holds = glove_status(side)
        if holds["calibrated"]:
            # Another glove's calibration: keep it in the library, where it
            # loads again whenever that glove is plugged in.
            archive(side)
        return self._prepare_config(side, replace=holds["calibrated"])

    def _backup_current(self, side):
        """Keep the side's calibration as ``.bak``: it is put back unless the run saves."""
        path = repo_path(config_rel(side))
        backup = f"{path}.bak"
        shutil.copy2(path, backup)
        return backup

    def confirm(self):
        """Answer the first prompt: the glove is seated in the tool, record it."""
        self._answer(
            WAITING,
            b"\n",
            RECORDING,
            "the calibration is not waiting for the glove to be seated",
        )

    def save(self, accept):
        """Answer the second prompt: write the calibration, or discard it."""
        with self._lock:
            self._declined = not accept
        self._answer(
            REVIEW,
            b"y\n" if accept else b"n\n",
            SAVING,
            "the calibration has no result to save yet",
        )

    def _answer(self, expected, data, next_state, refusal):
        with self._lock:
            if self._state != expected:
                raise CalibrationRefused(refusal)
            try:
                self._proc.stdin.write(data)
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise CalibrationRefused(
                    "the calibration process has already exited"
                ) from exc
            self._state = next_state

    def cancel(self):
        """Stop a running calibration; returns False when none is running."""
        with self._lock:
            if self._state not in LIVE_STATES:
                return False
            self._cancel_requested = True
            proc = self._proc
        threading.Thread(target=_terminate, args=(proc,), daemon=True).start()
        return True

    def shutdown(self):
        """Stop a running calibration and wait for it (GUI exit)."""
        with self._lock:
            proc = self._proc if self._state in LIVE_STATES else None
            if proc is not None:
                self._cancel_requested = True
        if proc is not None:
            _terminate(proc)

    # -- internals --------------------------------------------------------------

    def _prepare_config(self, side, replace):
        """Make the glove's config ready for the script; returns (created, backup).

        No config yet: copy the design template. A calibration that has to be
        replaced: keep it as ``.bak`` and start from the template. An existing
        config that records no board is used as it is.
        """
        path = repo_path(config_rel(side))
        exists = os.path.exists(path)
        if exists and not replace:
            return False, None
        template = repo_path(template_rel(side))
        if not os.path.isfile(template):
            raise CalibrationRefused(
                f"{glove_name(side)} has no design template at {template_rel(side)}"
            )
        backup = None
        if exists:
            backup = f"{path}.bak"
            shutil.copy2(path, backup)
        tmp = f"{path}.tmp"
        shutil.copyfile(template, tmp)
        os.replace(tmp, path)
        return True, backup

    def _undo(self, side, created, backup):
        """Put the glove's config back as it was before this run."""
        path = repo_path(config_rel(side))
        try:
            if backup:
                os.replace(backup, path)
            elif created:
                os.unlink(path)
        except OSError:
            pass

    def _add_line(self, text):
        self._seq += 1
        self._lines.append((self._seq, ANSI_RE.sub("", text)))

    def _pump(self, proc, side, created, backup):
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            text = decoder.decode(chunk).replace("\r\n", "\n").replace("\r", "\n")
            with self._lock:
                buffer = self._pending + text
                if self._state == STARTING and PROMPT_MARKER in buffer:
                    self._state = WAITING
                elif self._state == RECORDING and SAVE_MARKER in buffer:
                    self._state = REVIEW
                *complete, self._pending = buffer.split("\n")
                for line in complete:
                    self._add_line(line)
        code = proc.wait()
        status = glove_status(side)
        if backup and not created:
            # Only a confirmed save writes the calibration file.
            status["calibrated"] = status["calibrated"] and not _same_file(
                repo_path(config_rel(side)), backup
            )
        with self._lock:
            if self._pending:
                self._add_line(self._pending)
                self._pending = ""
            if status["calibrated"]:
                try:
                    archive(side)
                except (
                    OSError
                ) as exc:  # the glove is calibrated; only its library copy failed
                    self._add_line(f"Could not keep a library copy: {exc}")
                self._state = DONE
                self._message = (
                    f"Saved {glove_name(side)}.yaml for board {status['serial']}"
                )
            elif self._cancel_requested or self._declined:
                self._state = CANCELLED
                self._message = (
                    "Discarded; nothing was saved"
                    if self._declined
                    else "Calibration cancelled; nothing was saved"
                )
            else:
                self._state = FAILED
                self._message = self._failure_message(code)
        if not status["calibrated"]:
            self._undo(side, created, backup)

    def _failure_message(self, code):
        """The script's last line (a refusal or an exception prints its reason there), else its exit code."""
        tail = [text.strip() for _, text in self._lines if text.strip()]
        return tail[-1][:300] if tail else f"calibrate_exo.py exited with code {code}"


def _terminate(proc):
    """SIGINT, then SIGTERM, then SIGKILL the child's process group, waiting between."""
    for sig, grace in STOP_SIGNALS:
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue

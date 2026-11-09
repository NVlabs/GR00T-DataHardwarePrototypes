# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session folder naming: ``{root}/{YYYY-MM-DD}_dexL_dexR_hw{N}_{task}``.

The rig token names the exoskeleton boards the session is recorded with, left
side first: ``DEXUMI-Left`` + ``DEXUMI-Right`` -> ``dexL_dexR``. A base config
without real exoskeleton sources (the mock twin) gets no token, so its folders
are ``{date}_hw{N}_{task}``.

``hw{N}`` is the per-day hardware index for one task on this station: it names
one calibration of the boards. It is counted across rig tokens (``dexL_dexR``,
``dexL``, ``dexR`` or none). It starts at 1; the highest existing index N
decides the next launch (``next_session``):

- nothing recorded in hw{N} yet (a launch that failed before the first ``s``):
  hw{N} again. ``record_data.py`` creates the folder before streaming, and
  skipping it would leave an empty folder per failed launch. A launch with other boards shares that index but names its
  own folder; the other rig's empty folder is left in place.
- hw{N} holds episodes of these boards, and both hands' calibration files are
  byte-identical to the copies the recorder bundled into the folder: hw{N}
  again, so the relaunch continues that session.
- either hand's calibration changed, a copy is missing, or hw{N} was recorded
  with other boards: hw{N+1}.

The recorder may still take a ``_1`` copy of the continued folder when the rest
of the launch config changed (``session_slot.find_session_slot``, which the
server applies to this default). The folder it would append to is closed, and
hw{N+1} started (``start_next``), when its newest take is ``IDLE_CLOSE_S`` old,
or it holds an unfinished take kept for recovery (the server's
``_closed_reason``). ``root`` is returned as typed
(``~`` unexpanded) because the recorder writes ``recording.data_dir`` back into
the yaml in that form.
"""

import dataclasses
import datetime
import glob
import os
import re
import time

from gui.dc_ui.core.kinds import EXO_NAME_RE
from umi.common.recording.constants import PENDING_SUFFIX
from umi.common.recording.session_slot import folder_has_episode_history

# A session with no new take for 1 h is closed: it may already have been
# post-processed, and takes appended later would be left out of that result.
IDLE_CLOSE_S = 3600

# Left before Right; the config order is kept within a side.
_SIDE_ORDER = {"Left": 0, "Right": 1}
_SIDE_SUFFIX = {"Left": "L", "Right": "R"}
# The rig tokens rig_token() produces; any of them (or none) counts toward
# hw{N}.
_TOKEN_RE = r"(?:(dexL_dexR|dexL|dexR)_)?"


def today():
    """Station-local recording date (patched in tests)."""
    return datetime.date.today()


def _date_str(date):
    return date.isoformat() if hasattr(date, "isoformat") else str(date)


def rig_token(exo_names):
    """``dexL_dexR`` for the selected boards, left first.

    Names outside the form ``DEXUMI-Left`` / ``DEXUMI-Right`` contribute nothing;
    without any parseable name the token is empty and the folder carries none.
    """
    parsed = []
    for name in exo_names or ():
        match = EXO_NAME_RE.match(str(name or ""))
        if match:
            side = match.group(1)
            parsed.append((_SIDE_ORDER[side], f"dex{_SIDE_SUFFIX[side]}"))
    parts = [part for _, part in sorted(parsed, key=lambda item: item[0])]
    return "_".join(dict.fromkeys(parts))


def session_basename(date, n, task, token=""):
    rig = f"{token}_" if token else ""
    return f"{_date_str(date)}_{rig}hw{int(n)}_{task}"


def _pattern(date, task):
    # The optional trailing _N is resolve_session_slot's collision suffix (session_slot.with_counter).
    return re.compile(
        rf"^{re.escape(_date_str(date))}_{_TOKEN_RE}hw(\d+)_{re.escape(task)}(_\d+)?$"
    )


def existing_sessions(root, date, task):
    """{hw_index: [folder paths]} for this date + task under root (expanded), any rig token.

    A ``_N`` suffix counts only next to the un-suffixed folder: the recorder adds
    it after the base name is taken, so ``hw1_pour_2`` alone is the sibling task
    ``pour_2``, not a collision copy of ``pour``.
    """
    expanded = os.path.expanduser(root)
    if not os.path.isdir(expanded):
        return {}
    pattern = _pattern(date, task)
    names = {
        name
        for name in os.listdir(expanded)
        if os.path.isdir(os.path.join(expanded, name))
    }
    found = {}
    for name in sorted(names):
        match = pattern.match(name)
        if match is None:
            continue
        index, suffix = match.group(2), match.group(3)
        if suffix and name.removesuffix(suffix) not in names:
            continue
        found.setdefault(int(index), []).append(os.path.join(expanded, name))
    return found


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _left_first(stems):
    """Board names in rig_token's order: Left, Right, then anything else."""

    def side(stem):
        match = EXO_NAME_RE.match(stem)
        return _SIDE_ORDER[match.group(1)] if match else len(_SIDE_ORDER)

    return tuple(sorted(stems, key=side))


def changed_calibrations(folder, calibration_files):
    """Calibration files (stems) whose copy in ``folder`` is missing or differs from the live file.

    The recorder bundles every device config into the session folder by
    basename and compares the same bytes before it resumes a folder
    (session_slot.session_file_conflicts). A missing copy counts as changed:
    such a folder cannot show which calibration it was recorded with.
    """
    changed = []
    for path in calibration_files:
        try:
            with (
                open(path, "rb") as live,
                open(os.path.join(folder, os.path.basename(path)), "rb") as stored,
            ):
                same = live.read() == stored.read()
        except OSError:
            same = False
        if not same:
            changed.append(_stem(path))
    return changed


def pending_takes(folder):
    """Unfinished takes (``episode_N.pending``) left in ``folder``, e.g. kept for recovery."""
    pattern = os.path.join(os.path.expanduser(folder), f"episode_*{PENDING_SUFFIX}")
    return sorted(os.path.basename(path) for path in glob.glob(pattern))


def last_take_age(folder, now=None):
    """Seconds since the newest finished take in ``folder`` was written, or None without one."""
    written = [
        os.path.getmtime(path)
        for path in glob.glob(os.path.join(os.path.expanduser(folder), "episode_*"))
        if not path.endswith(PENDING_SUFFIX) and os.path.isdir(path)
    ]
    if not written:
        return None
    return (time.time() if now is None else now) - max(written)


def age_text(seconds):
    minutes = int(seconds // 60)
    return (
        f"{minutes // 60} h {minutes % 60:02d} min"
        if minutes >= 60
        else f"{minutes} min"
    )


@dataclasses.dataclass(frozen=True)
class SessionChoice:
    """The default session folder and why it was picked.

    ``kind``: ``first`` (no session of the task today), ``reuse`` (nothing
    recorded in hw{N} yet), ``continue`` (hw{N} holds episodes of these boards
    with the same calibration), ``recalibrated`` (``changed`` names the boards
    whose calibration differs from hw{N}'s copies), ``other_boards``, or
    ``closed`` (``start_next``: the folder named ``closed`` must not grow, for
    ``reason``). ``compared`` names the calibration files the rule compared.
    """

    path: str
    index: int
    kind: str
    changed: tuple = ()
    compared: tuple = ()
    closed: str = ""
    reason: str = ""

    @property
    def message(self):
        n = self.index
        if self.kind == "first":
            return f"hw{n}: the first session of this task today."
        if self.kind == "reuse":
            return f"hw{n} again: nothing has been recorded in hw{n} yet."
        if self.kind == "continue":
            if not self.compared:
                return f"Continues hw{n}: no exoskeleton calibration to compare."
            return (
                f"Continues hw{n}: {' and '.join(self.compared)} have the "
                f"calibration hw{n} was recorded with."
            )
        if self.kind == "recalibrated":
            return (
                f"New session hw{n}: the calibration of {' and '.join(self.changed)} "
                f"is not the one hw{n - 1} was recorded with."
            )
        if self.kind == "closed":
            return f"New session hw{n}: {self.closed} {self.reason}."
        return f"New session hw{n}: hw{n - 1} was recorded with other boards."


def _session_path(root, date, n, task, token):
    name = session_basename(date, n, task, token)
    return os.path.join(str(root).rstrip("/") or "/", name)


def start_next(choice, root, date, task, token, folder, reason):
    """hw{N+1} instead of ``choice``, because ``folder`` (the one it would append to) must not grow."""
    n = choice.index + 1
    return SessionChoice(
        _session_path(root, date, n, task, token),
        n,
        "closed",
        compared=choice.compared,
        closed=os.path.basename(str(folder).rstrip("/")),
        reason=reason,
    )


def next_session(root, date, task, token="", calibration_files=()):
    """The default folder for this launch; see the module docstring for the rule.

    ``calibration_files`` are the live calibration files of the selected
    boards (absolute paths); without any (the mock base) a folder of the same
    token is continued.
    """
    found = existing_sessions(root, date, task)
    compared = _left_first(_stem(path) for path in calibration_files)

    def choice(n, kind, changed=()):
        path = _session_path(root, date, n, task, token)
        return SessionChoice(path, n, kind, _left_first(changed), compared)

    if not found:
        return choice(1, "first")
    n = max(found)
    used = [path for path in found[n] if folder_has_episode_history(path)]
    if not used:
        return choice(n, "reuse")
    # This rig's hw{N} folder and the recorder's _1, _2 copies of it.
    own_name = re.compile(
        rf"{re.escape(session_basename(date, n, task, token))}(_\d+)?"
    )
    own = [path for path in used if own_name.fullmatch(os.path.basename(path))]
    if not own:
        return choice(n + 1, "other_boards")
    changes = [changed_calibrations(path, calibration_files) for path in own]
    if not all(changes):
        return choice(n, "continue")
    return choice(n + 1, "recalibrated", changes[0])

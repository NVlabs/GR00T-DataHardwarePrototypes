# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ManagedProcess + Supervisor for the DC GUI's recorder.

Each role is a child in its own session (``start_new_session=True`` ==
``setsid()``), so ``os.killpg`` reaches ``uv run``'s python grandchild and
anything it spawns. The child's stdout+stderr go straight to a log file under
``<home>/logs`` (not a pipe: an orphan keeps a valid stdout after a GUI restart
and the file keeps growing) and a tailer thread feeds a ring buffer with a
sequence number for incremental tails; readiness is the spec's ready regexes;
pidfiles under ``<home>/pids`` let a restarted GUI adopt still-running children.
Stop is SIGINT -> SIGTERM -> SIGKILL.
"""

import collections
import contextlib
import fcntl
import json
import os
import re
import signal
import subprocess
import threading
import time

from gui.dc_ui.core.kinds import LaunchSpec

LOG_RING = 5000
LOG_TAIL_ON_ADOPT = 500
DEFAULT_GRACE = (10, 5, 3)
# The first SIGINT interrupts recording and starts a drain of pending
# alignments, which may take 600 s. Apply the full shutdown grace period
# even to a process stopped during startup.
RECORDER_GRACE = (600, 5, 3)
DEFAULT_LOG_TAG = "dc_ui"
# print_red & co wrap lines in colour escapes; markers are matched on the
# stripped text while the ring buffer keeps the raw line.
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGKILL)
FOLLOW_PERIOD_S = 0.25
LIVE_STATES = ("starting", "ready", "adopted")
TERMINAL_STATES = ("done", "failed", "stopped")
SECRET_ENV_MARKERS = ("SECRET", "TOKEN", "PASSWORD", "ACCESS_KEY")
REDACTED = "<redacted>"
_NEW_LAUNCH = object()
STATUS_KEYS = (
    "pid",
    "exit_code",
    "uptime_s",
    "log_path",
    "last_line",
    "error",
    "last_error",
    "ready_at",
)


class StartRefused(RuntimeError):
    pass


def grace_for(role):
    return RECORDER_GRACE if role == "recorder" else DEFAULT_GRACE


def redact_env(env):
    out = {}
    for key, value in (env or {}).items():
        secret = any(marker in key.upper() for marker in SECRET_ENV_MARKERS)
        out[key] = REDACTED if secret and value is not None else value
    return out


def child_env(spec_env):
    env = dict(os.environ)
    for key, value in (spec_env or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)
    return env


def pid_alive(pid):
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/stat") as f:
            state = f.read().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"


def process_group_alive(pgid):
    if not pgid or pgid <= 0:
        return False
    # killpg(pgid, 0) also sees unreaped orphan zombies. They cannot hold
    # hardware and must not consume the entire stop budget.
    with os.scandir("/proc") as entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                with open(os.path.join(entry.path, "stat")) as stream:
                    fields = stream.read().rsplit(")", 1)[1].split()
                if int(fields[2]) == pgid and fields[0] != "Z":
                    return True
            except (OSError, IndexError, ValueError):
                continue
    return False


def _script_token(argv):
    for i, token in enumerate(argv):
        if token == "-m" and i + 1 < len(argv):
            return argv[i + 1]
        if token.endswith(".py"):
            return token
    return argv[-1] if argv else ""


def cmdline_matches(pid, argv):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmdline = f.read().replace(b"\0", b" ").decode("utf-8", errors="replace")
    except OSError:
        return False
    return _script_token(list(argv)) in cmdline


def _unlink_quietly(path):
    try:
        os.unlink(path)
    except OSError:
        pass


class ManagedProcess:
    def __init__(self, spec, home, log_tag=DEFAULT_LOG_TAG):
        self.spec = spec
        self.home = str(home)
        self.log_tag = log_tag
        self.state = "stopped"
        self.pid = None
        self.pgid = None
        self.exit_code = None
        self.error = None
        self.last_error = None
        self.started_at = None
        self.ready_at = None
        self.log_path = None
        self._proc = None
        self._reader = None
        self._adopted = False
        self._seq = 0
        self._lines = collections.deque(maxlen=LOG_RING)
        self._lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._ready_hits = [False] * len(spec.ready_regexes)
        self._group_exited = False

    # -- paths ---------------------------------------------------------------------

    def _pidfile(self):
        return os.path.join(self.home, "pids", f"{self.spec.role}.json")

    def _new_log_path(self):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.home, "logs", f"{self.spec.role}_{stamp}.log")

    # -- start ---------------------------------------------------------------------

    def start(self):
        role = self.spec.role
        if self._adopted:
            raise StartRefused(
                f"{role} was adopted from a pidfile (redacted spec) and cannot be "
                "relaunched from it; start it again with a fresh spec"
            )
        if self.is_alive():
            raise StartRefused(f"{role} is already running (pid {self.pid})")
        self._reset_run_state()
        os.makedirs(os.path.join(self.home, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.home, "pids"), exist_ok=True)
        self.log_path = self._new_log_path()
        with open(self.log_path, "ab") as log_fh:  # the child keeps its own copy
            offset = log_fh.tell()
            try:
                self._proc = subprocess.Popen(
                    list(self.spec.argv),
                    cwd=self.spec.cwd or None,
                    env=child_env(self.spec.env),
                    # Detached children use cv2 for operator input. If uv
                    # inherits a terminal here, it expects terminal SIGINT
                    # delivery and does not forward our Stop signal.
                    stdin=subprocess.DEVNULL,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as exc:
                self.state = "failed"
                self.error = f"launch failed: {exc}"
                self._append_line(f"[{self.log_tag}] {self.error}")
                log_fh.write(f"[{self.log_tag}] {self.error}\n".encode())
                return self.status()
        self.pid = self._proc.pid
        self.pgid = self.pid  # start_new_session: the child leads its own group
        self.started_at = time.monotonic()
        self.state = "starting"
        self._write_pidfile()
        self._start_tailer(offset)
        if not self.spec.ready_regexes:
            self._set_ready()
        return self.status()

    def _reset_run_state(self):
        self.pid = None
        self.pgid = None
        self.exit_code = None
        self.error = None
        self.last_error = None
        self.ready_at = None
        self._proc = None
        self._reader = None
        self._ready_hits = [False] * len(self.spec.ready_regexes)
        self._group_exited = False

    # -- output --------------------------------------------------------------------

    def _start_tailer(self, offset):
        self._reader = threading.Thread(
            target=self._follow_log, args=(offset,), daemon=True
        )
        self._reader.start()

    def _follow_log(self, offset):
        while self.is_alive():
            offset = self._read_new_log_lines(offset)
            time.sleep(FOLLOW_PERIOD_S)
        self._read_new_log_lines(offset, final=True)
        self._on_exit(self._proc.poll() if self._proc is not None else None)

    def _read_new_log_lines(self, offset, final=False):
        if not self.log_path or not os.path.isfile(self.log_path):
            return offset
        with open(self.log_path, "rb") as f:
            f.seek(offset)
            chunk = f.read()
        # whole lines only; a partial one waits for its newline (or the exit)
        end = len(chunk) if final else chunk.rfind(b"\n") + 1
        for raw in chunk[:end].splitlines():
            self._on_line(raw.decode("utf-8", errors="replace"))
        return offset + end

    def _append_line(self, line):
        with self._lock:
            self._seq += 1
            self._lines.append((self._seq, line))

    def _on_line(self, line):
        self._append_line(line)
        clean = ANSI_RE.sub("", line)
        if self.spec.done_regex and re.search(self.spec.done_regex, clean):
            self.state = "done"
            return
        if self.state not in LIVE_STATES:
            return
        for regex in self.spec.fail_regexes:
            if re.search(regex, clean):
                self._on_fail_line(clean)
                return
        for i, regex in enumerate(self.spec.ready_regexes):
            if not self._ready_hits[i] and re.search(regex, clean):
                self._ready_hits[i] = True
        if self.state == "starting" and all(self._ready_hits):
            self._set_ready()

    def _on_fail_line(self, line):
        text = line.strip()[:300]
        if self.state == "starting":
            self.state = "failed"
            self.error = text
            return
        # A background traceback can occur while the recorder is still running.
        # Expose it as last_error; process exit determines the final state.
        self.last_error = text

    def _set_ready(self):
        if self.state in ("starting", "adopted"):
            self.state = "ready"
            self.ready_at = time.time()

    def _on_exit(self, code):
        # A wrapper may exit before its workers. Keep their ownership record
        # until Stop has finished the group, and never clean a newer run.
        group_alive = self._group_alive()
        if not group_alive:
            self._cleanup_run_files()
        if self.state == "stopping":
            if not group_alive:
                self.state = "stopped"
        elif self.state == "starting":
            self.state = "failed"
            self.error = self.error or f"exited with code {code} before ready"
        elif self.state in TERMINAL_STATES:
            pass
        elif code in (0, None):
            self.state = "exited"
        else:
            self.state = "failed"
            self.error = f"exited with code {code}"
        if code is not None or self.exit_code is None:
            self.exit_code = code

    # -- stop ----------------------------------------------------------------------

    def stop(self, grace=DEFAULT_GRACE):
        # Concurrent Stop requests share one escalation. A second SIGINT can
        # interrupt the recorder's episode save.
        with self._stop_lock:
            leader_alive = self.is_alive()
            if not leader_alive and not self._group_alive():
                self._cleanup_run_files()
                return self.status()
            self.state = "stopping"
            if leader_alive:
                self._signal(signal.SIGINT)
                self._wait_exit(grace[0])
            group_stopped = self._sweep_group(grace)
            if self._reader is not None:
                self._reader.join(timeout=5)
            if group_stopped and not self.is_alive():
                # The tailer can have finished while descendants were alive.
                self._on_exit(self._proc.poll() if self._proc is not None else None)
            else:
                self.state = "failed"
                self.error = f"process group {self.pgid} did not exit after SIGKILL"
            return self.status()

    def _signal(self, sig):
        # SIGINT goes to the direct child only: `uv run` forwards exactly one
        # copy to its python, whereas killpg would deliver a second one that
        # can interrupt the recorder's cleanup and abort the episode;
        # grandchildren (ffmpeg writers) are closed by their parent.
        # SIGTERM / SIGKILL stay group-wide.
        try:
            if sig == signal.SIGINT:
                os.kill(self.pid, sig)
            else:
                os.killpg(self.pgid, sig)
        except ProcessLookupError:
            pass

    def _group_alive(self):
        if self._group_exited or self.pgid is None:
            return False
        alive = process_group_alive(self.pgid)
        if not alive:
            self._group_exited = True
        return alive

    def _sweep_group(self, grace):
        for sig, wait_s in zip(STOP_SIGNALS[1:], grace[1:]):
            if not self._group_alive():
                return True
            self._signal(sig)
            deadline = time.monotonic() + wait_s
            while self._group_alive() and time.monotonic() < deadline:
                time.sleep(0.05)
        return not self._group_alive()

    def _wait_exit(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self.is_alive():
                return True
            time.sleep(0.05)
        return not self.is_alive()

    def is_alive(self):
        if self._proc is not None:
            return self._proc.poll() is None
        return pid_alive(self.pid)

    # -- pidfile -------------------------------------------------------------------

    @contextlib.contextmanager
    def _pidfile_lock(self):
        os.makedirs(os.path.dirname(self._pidfile()), exist_ok=True)
        # Separate opens serialize ownership checks and writes across old/new
        # tailer threads, including adopted processes in another GUI instance.
        with open(self._pidfile() + ".lock", "a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _cleanup_run_files(self):
        if self.pid is None:
            return
        with self._pidfile_lock():
            try:
                with open(self._pidfile()) as stream:
                    owner = json.load(stream)
                owner_pid = int(owner["pid"])
                owner_pgid = int(owner.get("pgid") or owner_pid)
            except (OSError, ValueError, TypeError, KeyError):
                return
            if owner_pid != self.pid or owner_pgid != self.pgid:
                return
            _unlink_quietly(self._pidfile())

    def _write_pidfile(self):
        spec = self.spec.to_dict()
        spec["env"] = redact_env(spec.get("env") or {})
        data = {
            "pid": self.pid,
            "pgid": self.pgid,
            "started_at": time.time(),
            "log_path": self.log_path,
            "spec": spec,
        }
        with self._pidfile_lock():
            fd = os.open(self._pidfile(), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)

    # -- adoption ------------------------------------------------------------------

    @classmethod
    def adopt(cls, spec, data, home, log_tag=DEFAULT_LOG_TAG):
        proc = cls(spec, home, log_tag=log_tag)
        proc.pid = int(data["pid"])
        proc.pgid = int(data.get("pgid") or data["pid"])
        proc.log_path = data.get("log_path")
        age = max(0.0, time.time() - float(data.get("started_at") or time.time()))
        proc.started_at = time.monotonic() - age
        proc.state = "adopted"
        proc._adopted = True
        offset = proc._load_log_tail()
        proc._start_tailer(offset)
        proc._set_ready()  # past its startup: still running is ready
        return proc

    def _load_log_tail(self):
        """Adoption: the tail is history. Record it, but never let a past
        Traceback / Done. re-decide the state of a process that is still running."""
        if not self.log_path or not os.path.isfile(self.log_path):
            return 0
        with open(self.log_path, "rb") as f:
            data = f.read()
        for raw in data.splitlines()[-LOG_TAIL_ON_ADOPT:]:
            self._append_line(raw.decode("utf-8", errors="replace"))
        return len(data)

    # -- reporting -----------------------------------------------------------------

    def status(self):
        with self._lock:
            last = self._lines[-1][1] if self._lines else None
        alive = self.is_alive()
        uptime = time.monotonic() - self.started_at if self.started_at else None
        return {
            "role": self.spec.role,
            "state": self.state,
            "pid": self.pid,
            "exit_code": self.exit_code,
            "uptime_s": round(uptime, 1) if alive and uptime is not None else None,
            "log_path": self.log_path,
            "last_line": last,
            "error": self.error,
            "last_error": self.last_error,
            "ready": self.state == "ready",
            "ready_at": self.ready_at,
            "adopted": self._adopted,
        }

    def tail(self, since_seq=0, limit=500):
        with self._lock:
            selected = [(s, line) for s, line in self._lines if s > since_seq][-limit:]
            next_seq = self._seq
        return {"lines": [line for _, line in selected], "next_seq": next_seq}


def _stopped_status(role):
    status = {key: None for key in STATUS_KEYS}
    status.update(role=role, state="stopped", ready=False, adopted=False)
    return status


def launch_spec_changes(current, expected):
    """Compare launch settings without exposing secret environment values."""
    changes = []
    if list(current.argv) != list(expected.argv):
        changes.append("launch arguments changed")
    if current.cwd != expected.cwd:
        changes.append("working directory changed")
    if redact_env(current.env) != redact_env(expected.env):
        changes.append("environment changed")
    return changes


def adopt_pidfile(path, home, log_tag=DEFAULT_LOG_TAG):
    try:
        with open(path) as f:
            data = json.load(f)
        pid = int(data["pid"])
        spec = LaunchSpec.from_dict(data["spec"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        _unlink_quietly(path)
        return None
    if not pid_alive(pid) or not cmdline_matches(pid, spec.argv):
        _unlink_quietly(path)
        return None
    return ManagedProcess.adopt(spec, data, home, log_tag=log_tag)


class Supervisor:
    def __init__(self, home, roles, stop_order, log_tag=DEFAULT_LOG_TAG):
        self.home = str(home)
        self.roles = tuple(roles)
        self.stop_order = tuple(stop_order)
        self.log_tag = log_tag
        os.makedirs(self.home, exist_ok=True)
        self.procs = {}
        self._lock = threading.Lock()
        self._launch_state_lock = threading.Lock()
        self._launch_generation = 0
        self._shutdowns = 0

    def start(self, role, spec, *, launch_token=_NEW_LAUNCH):
        if launch_token is _NEW_LAUNCH:
            launch_token = self.capture_launch_token()
        if role not in self.roles:
            raise ValueError(f"unknown role {role!r}; roles: {self.roles}")
        if spec.role != role:
            raise ValueError(f"spec.role {spec.role!r} does not match role {role!r}")
        self.check_launch_token(launch_token)
        with self._lock:
            self.check_launch_token(launch_token)
            current = self.procs.get(role)
            if current is not None and (current.is_alive() or current._group_alive()):
                raise StartRefused(
                    f"{role} is already running (pid {current.pid}); stop it first"
                )
            proc = ManagedProcess(spec, self.home, log_tag=self.log_tag)
            status = proc.start()
            self.procs[role] = proc
            return status

    def capture_launch_token(self):
        """Capture before request validation so Stop all cancels pending work."""
        with self._launch_state_lock:
            return None if self._shutdowns else self._launch_generation

    def check_launch_token(self, token):
        with self._launch_state_lock:
            if self._shutdowns or token != self._launch_generation:
                raise StartRefused("launch cancelled by Stop all; retry after shutdown")

    def stop(self, role, grace=None):
        proc = self.procs.get(role)
        if proc is None:
            return _stopped_status(role)
        return proc.stop(grace or grace_for(role))

    def status_all(self, expected_specs=None):
        statuses = {
            role: self.procs[role].status()
            if role in self.procs
            else _stopped_status(role)
            for role in self.roles
        }
        for role, expected in (expected_specs or {}).items():
            status = statuses[role]
            proc = self.procs.get(role)
            changes = (
                launch_spec_changes(proc.spec, expected)
                if proc is not None and status["state"] in LIVE_STATES
                else []
            )
            status["restart_required"] = bool(changes)
            status["restart_reasons"] = changes
        return statuses

    def tail(self, role, since_seq=0, limit=500):
        proc = self.procs.get(role)
        if proc is None:
            return {"lines": [], "next_seq": 0}
        return proc.tail(since_seq, limit)

    def stop_all(self, grace=None):
        # Cancel queued launches before waiting for the in-flight start to
        # register its child. Status/log reads never acquire this lock and
        # remain available through startup and the full stop grace period.
        with self._launch_state_lock:
            self._launch_generation += 1
            self._shutdowns += 1
        try:
            with self._lock:
                return {
                    role: self.stop(role, grace)
                    for role in self.stop_order
                    if role in self.procs
                }
        finally:
            with self._launch_state_lock:
                self._shutdowns -= 1

    def adopt_orphans(self):
        adopted = []
        pids_dir = os.path.join(self.home, "pids")
        if not os.path.isdir(pids_dir):
            return adopted
        for name in sorted(os.listdir(pids_dir)):
            role = name[: -len(".json")] if name.endswith(".json") else None
            if role not in self.roles:
                continue
            current = self.procs.get(role)
            if current is not None and current.is_alive():
                continue
            proc = adopt_pidfile(
                os.path.join(pids_dir, name), self.home, log_tag=self.log_tag
            )
            if proc is not None:
                self.procs[role] = proc
                adopted.append(role)
        return adopted

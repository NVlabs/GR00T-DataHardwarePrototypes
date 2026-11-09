# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What this station is recording, and whether anything is still running.

State lives at the package root beside `devices.py` and `calibration.py`; the
route module is the HTTP boundary. That is the existing shape, and it is also
what lets `readiness` import this directly instead of reaching into a route from
inside a function to dodge an import cycle.

The key idea is that **a recording outlives the web process.**
`jobs.start` detaches the child with `start_new_session=True`, and this
registry (unlike the one `devices.registry_for` builds) has no shutdown
reaper, deliberately: ending a recording because somebody restarted the
console would be worse than the problem. So the console has to be able to find a
recorder it did not start, which is what the pid file and `adopt` are for, and
every question about "is one running" is answered from the process group rather
than from an object in memory that a restart erases.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import signal
import threading
import time
from dataclasses import dataclass, field

from station_ui import jobs as job_control
from station_ui import process_group
from station_ui.errors import RecordingInProgress, UnknownJob
from station_ui.jobs import JobRegistry

__all__ = [
    "KERNEL_BLOCKED",
    "KERNEL_BLOCK_S",
    "LOCK_MESSAGE",
    "PID_FILE",
    "RECORD_STEP",
    "RecordingInProgress",
    "RecordingState",
    "adopt",
    "current",
    "forget",
    "forget_last_recording",
    "in_progress",
    "kernel_block_age",
    "note_liveness",
    "remember",
    "request_stop",
    "require_idle",
    "state_for",
    "survivors",
]

logger = logging.getLogger(__name__)

# A stable step id, never a panel index: indices shift whenever the step list
# changes; a string does not.
RECORD_STEP = "record"

# Persist recorder process state outside the checkout alongside other
# station runtime state.
PID_FILE = "recorder.json"


@dataclass
class RecordingState:
    """Everything this station is running, and how far through it is.

    Attached to `app.state` on first use rather than built in `create_app`, so
    adding this page costs `app.py` one router include.
    """

    jobs: JobRegistry = field(default_factory=JobRegistry)
    steps_done: set[str] = field(default_factory=set)
    record_job_id: str | None = None
    # Adopted recorders have no job object or captured pipe, but their process
    # group still supports status and stop operations.
    pid: int | None = None
    pgid: int | None = None
    adopted: bool = False
    # Monotonic stamp of when Stop was pressed, or None. Drives the Force stop
    # countdown; see `routes/record.FORCE_AFTER_S`.
    stopping_since: float | None = None
    # Monotonic stamp of when the group was first seen blocked in the kernel and
    # has been ever since, or None. See `kernel_block_age`.
    blocked_since: float | None = None
    # Whether the last render observed a live recorder, and when that stopped
    # being true. Drives the Start cool-down; see `note_liveness`.
    was_live: bool = False
    went_quiet_at: float | None = None


_ATTACH = threading.Lock()


def state_for(app) -> RecordingState:
    with _ATTACH:
        state = getattr(app.state, "workflow", None)
        if state is None:
            state = RecordingState()
            app.state.workflow = state
        return state


def current(state_or_app):
    """The recording job, or None. Accepts the app or the state, as callers vary."""
    state = (
        state_or_app
        if isinstance(state_or_app, RecordingState)
        else state_for(state_or_app)
    )
    if not state.record_job_id:
        return None
    try:
        return state.jobs.get(state.record_job_id)
    except UnknownJob:  # pragma: no cover - the registry outlives every job
        return None


# ------------------------------------------------------------- the pid file


def _pid_path(app) -> pathlib.Path:
    return pathlib.Path(app.state.settings.state_dir) / PID_FILE


def remember(app, job) -> None:
    """Record which process this station's recorder is, on disk.

    The web process is not the recording's lifetime. `jobs.start` detaches the
    child, so a console restart (or a crash) leaves it running with nothing
    in memory that knows about it, and the page would come back offering Start
    at cameras another process is holding.

    `start_time` rides along because a pid is not an identity: Linux recycles
    them, and adopting on the pid alone would let the console SIGKILL an
    unrelated process.
    """
    payload = {
        "pid": job.pgid,
        "pgid": job.pgid,
        "start_time": process_group.start_time(job.pgid) if job.pgid else None,
        "job_id": job.id,
        "argv": list(job.argv),
    }
    path = _pid_path(app)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    except OSError:
        # Failure to persist process state disables later adoption; it must not
        # abort the current recording launch.
        logger.warning("could not write %s", path)


def forget(app) -> None:
    """Drop the pid file. Idempotent."""
    try:
        _pid_path(app).unlink()
    except FileNotFoundError:
        pass
    except OSError:  # pragma: no cover - a read-only state dir
        logger.warning("could not remove %s", _pid_path(app))


def adopt(app) -> str:
    """Re-attach to a recorder that outlived the console.

    Returns what happened, for the caller to log and the page to explain:
    `none`, `stale`, `recycled` or `adopted`.
    """
    try:
        record = json.loads(_pid_path(app).read_text())
    except (OSError, ValueError):
        return "none"
    try:
        pid = int(record.get("pid") or 0)
        pgid = int(record.get("pgid") or 0)
    except (TypeError, ValueError):
        forget(app)
        return "none"

    if not pid or not process_group.alive(pid):
        # A stale leader may leave camera-owning workers that must be collected.
        job_control.terminate_group(pgid)
        forget(app)
        return "stale"

    if process_group.start_time(pid) != record.get("start_time"):
        # This pid was reused. Signal nothing: the process wearing it now
        # belongs to somebody else.
        logger.warning("recorder pid %s was reused; not adopting", pid)
        forget(app)
        return "recycled"

    state = state_for(app)
    state.pid, state.pgid, state.adopted = pid, pgid, True
    # An adopted recorder's original pipe is gone; report unavailable captured
    # output instead of pretending it emitted nothing.
    state.record_job_id = None
    return "adopted"


# --------------------------------------------------------------- the predicate


def _pgid(state) -> int | None:
    """The recording's process group, from whichever source still knows it."""
    pgid = state.pgid
    if not pgid:
        job = current(state)
        pgid = job.pgid if job is not None else None
    return pgid or None


def survivors(app) -> list:
    """Every live process in the recording's group. The evidence, not a verdict.

    `in_progress` is this emptied to a bool. Callers that need more than
    yes/no -- the Record page asks what the group is *doing* -- take the list
    and read it themselves, rather than each recomputing the pgid and walking
    /proc again on the same render.
    """
    pgid = _pgid(state_for(app))
    return process_group.survivors(pgid) if pgid else []


def in_progress(app) -> bool:
    """Whether a recorder is alive right now. The lock predicate.

    Derived, never stored: a lock released by a stale flag is a worse dead end
    than the one this whole design removes.

    Read from the process group rather than the job state, and the difference is
    the point. A draining recorder is still reading the staging directory and
    still holds the boards long after `RUNNING` ends, and a crashed leader can
    be `FAILED` while a forked worker keeps an exclusive OAK. Both are "still
    running" for every purpose a caller has.
    """
    return bool(survivors(app))


# Processes in uninterruptible kernel sleep cannot act on pending signals
# until the kernel call returns. Return such survivors rather than falsely
# reporting a wedged USB/SDK process as terminated.
KERNEL_BLOCKED = "D"

# Report kernel blocking only after it persists across the threshold.
# Ordinary disk and network I/O briefly enters D state and must not be
# reported as a stalled drain.
KERNEL_BLOCK_S = 30.0


def kernel_block_age(app, procs) -> float | None:
    """Seconds the group has been continuously blocked in the kernel, or None.

    Stateful because the question is about persistence and each render sees one
    instant. Any process out of `D` resets the clock: the group is making
    progress, whatever the rest of it is doing.

    Advisory only: it changes what the page says, never what it does. Nothing
    is gated on it, so a console nobody is looking at (this only advances while
    the page renders) loses a hint and no behaviour.
    """
    state = state_for(app)
    if procs and any(proc.state == KERNEL_BLOCKED for proc in procs):
        if state.blocked_since is None:
            state.blocked_since = time.monotonic()
        return time.monotonic() - state.blocked_since
    state.blocked_since = None
    return None


def note_liveness(app, live: bool) -> float | None:
    """Seconds since a recorder the console was watching went away, else None.

    Exists because Start and Stop occupy the same place on the page, and the
    card swaps itself. An operator whose page still shows Stop (the recorder
    exited on the cv2 `q` key and the reload has not landed, or the stream
    dropped and it never will) presses it, htmx replaces the card with one
    whose first control is Start, and their second press (the first appeared to
    do nothing, so there is almost always a second) would launch a recorder:
    the operator pressed Stop twice and got a new cv2 window.

    Armed only on an observed live -> gone transition, which keeps it from being
    a nuisance: it is exactly the case where the page in front of somebody said
    Stop a moment ago. A console opened fresh after a recording ended never saw
    it alive, so it arms nothing and Start is live immediately.
    """
    state = state_for(app)
    if live:
        state.was_live = True
        state.went_quiet_at = None
        return None
    if state.was_live:
        state.was_live = False
        state.went_quiet_at = time.monotonic()
    if state.went_quiet_at is None:
        return None
    return time.monotonic() - state.went_quiet_at


def request_stop(app) -> bool:
    """SIGTERM an adopted recorder's leader. True if the signal went.

    The console-started case goes through `jobs.cancel`, which owns the job's
    escalation timers. An adopted recording has no job object (its pipe died
    with the process that started it), so this is the only way to ask it to
    stop, and without it the Record page would render a Stop button that does
    nothing at all.

    The leader alone, like `jobs.cancel`: the drain runs in forked workers that
    inherit both the pgid and `_stop_on_sigterm`, so a group signal kills the
    processes performing the drain and `BackgroundAligner.shutdown()` then
    early-returns on a dead worker, a silent no-op that still exits 0.
    """
    state = state_for(app)
    pid = state.pid
    if not pid or not process_group.alive(pid):
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:  # pragma: no cover - a kernel-level refusal
        logger.warning("could not signal adopted recorder %s", pid)
        return False
    return True


LOCK_MESSAGE = (
    "A recording is in progress, so this page is read-only. Stop the recording "
    "on Record before changing the rig."
)


def require_idle(app) -> None:
    """Refuse a rig mutation while a recorder is alive.

    Called from every mutating route on steps 1-4, not merely rendered around
    the buttons: not drawing a control hides the affordance, it does not remove
    the endpoint.
    """
    if in_progress(app):
        raise RecordingInProgress(LOCK_MESSAGE)


def forget_last_recording(app) -> bool:
    """Drop the finished recording the page still reports. True if it did.

    Everything else on `/record` is derived on each render (the folder, the
    task, the pre-flight guard all come back out of the config file), but the
    status badge, the log under it and the Record tick are read from here. So
    without this, an operator who changed a camera and applied, or who started
    a new session, would come back to a green "Stopped -- drained" over the
    previous rig's recording, its log, and a ticked step: the one thing on the
    page that is not about the session in front of them.

    Refuses while a recording is live, and that is the whole reason this is a
    function rather than two assignments at the call sites. Forgetting the job
    id of a running recorder does not stop it: `jobs.start` detaches the child,
    so it goes on writing episodes and holding every camera with the Stop
    button (which needs that id) gone from the page. The rig is genuinely
    busy; the caller gets False and leaves the status alone.
    """
    state = state_for(app)
    job = current(state)
    if job is not None and not job.finished:
        return False
    if state.record_job_id is None and RECORD_STEP not in state.steps_done:
        return False
    state.record_job_id = None
    state.steps_done.discard(RECORD_STEP)
    return True

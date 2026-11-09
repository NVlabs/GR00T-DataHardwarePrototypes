# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One model for every long-running subprocess the station starts.

Hardware opens have no timeout of their own, so calling one inline from a
request handler would pin an HTTP worker until the operator gives up. Every
long-running step therefore runs as a `Job`.

A `Job` captures merged stdout+stderr, reports the child's real exit code,
enforces a wall-clock cap, and can be cancelled, with `SIGTERM`, so
`record_data.py` still runs the `finally` block that drains queued
alignments. Escalation to `SIGKILL` happens only after a grace period and is
written to both the log and the job's own output, because a hard kill means
that drain was skipped and the operator needs to know.

Output is streamed as Server-Sent Events assembled by hand: the station install
(`uv export --only-group data_collection`) carries neither `websockets` nor
`sse-starlette`, and this must not add a dependency to get a progress bar.
"""

from __future__ import annotations

import codecs
import json
import logging
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

from station_ui import process_group
from station_ui.errors import UnknownJob

logger = logging.getLogger(__name__)

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"

TERMINAL_STATES = frozenset({DONE, FAILED, CANCELLED})

# Default timeout for scans and calibration. Long-running recording jobs
# must pass timeout=None explicitly.
DEFAULT_TIMEOUT_S = 600.0
# Default grace covers short jobs only. Recording callers must allow the
# 600 s alignment drain plus remaining cleanup before SIGKILL escalation.
DEFAULT_TERM_GRACE_S = 10.0

# Poll process-group exit without busy-walking /proc.
_SURVIVOR_POLL_S = 0.05
# Bound each job's captured text to approximately 2 MB. Report dropped
# lines when trimming old output.
DEFAULT_MAX_LINES = 20_000

_READ_SIZE = 65536
# Bound pipe draining after child exit; inherited write ends in descendants
# can otherwise prevent EOF indefinitely.
_DRAIN_S = 2.0
# SSE comment frame. Keeps idle connections (and any proxy in front of them)
# alive without looking like output.
KEEPALIVE_FRAME = ": keepalive\n\n"
KEEPALIVE_S = 15.0

# Treat carriage returns as progress updates so redraw-only output streams live.
_LINE_BREAK = re.compile(r"\r\n|\n|\r")


@dataclass
class Job:
    """A single child process and everything observable about it."""

    id: str
    argv: list[str]
    state: str = PENDING
    cwd: str = "."
    label: str = ""
    # Tag jobs by owning page. List, stream and cancel only that page's jobs;
    # otherwise an audit action could terminate an active recording.
    kind: str = ""
    exit_code: int | None = None
    error: str | None = None
    timeout: float | None = DEFAULT_TIMEOUT_S
    term_grace: float = DEFAULT_TERM_GRACE_S
    max_lines: int = DEFAULT_MAX_LINES
    # Signal only the recording parent for graceful stop: forked aligners
    # inherit its SIGTERM handler, and group-wide SIGTERM would abort the drain.
    # Other jobs may stop their whole group. SIGKILL escalation always reaches
    # the entire group to collect surviving workers.
    signal_group: bool = True
    # After cancellation, escalate on sustained output silence as well as the
    # overall grace ceiling. None disables the silence trigger; the maximum
    # grace still applies.
    stall_grace: float | None = None
    # Run completion callbacks once on the supervisor thread, after the child
    # and process group have exited. Jobs that fail before spawning have no
    # supervisor and do not invoke this callback.
    on_done: Callable[[Job], None] | None = None

    def __post_init__(self) -> None:
        # One condition guards output and state together, so "no new lines and
        # the job has ended" is a single consistent observation for a reader.
        self._cond = threading.Condition()
        self._lines: list[str] = []
        self._dropped = 0
        # Separate from _cond: signalling must not wait behind a reader.
        self._proc_lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._cancelled = False
        self._timed_out = False
        self._timers: list[threading.Timer] = []
        # Capture the process-group ID at spawn; it cannot be queried from a reaped
        # leader when surviving children still need cleanup.
        self.pgid: int | None = None
        # Monotonic time of the newest output line, shared by stall detection and
        # the shutdown progress display.
        self.last_output_at: float = time.monotonic()

    @property
    def finished(self) -> bool:
        return self.state in TERMINAL_STATES

    def lines(self) -> list[str]:
        """A snapshot of the captured output still in the buffer."""
        with self._cond:
            return list(self._lines)

    # -- internals ---------------------------------------------------------

    def _append(self, line: str) -> None:
        with self._cond:
            self._lines.append(line)
            self.last_output_at = time.monotonic()
            # Trim in blocks so the cost is amortised to O(1) per line rather
            # than a list shift on every append once the buffer is full.
            if len(self._lines) > self.max_lines * 2:
                drop = len(self._lines) - self.max_lines
                del self._lines[:drop]
                self._dropped += drop
            self._cond.notify_all()

    def _finish(self, state: str, exit_code=None, error=None) -> None:
        with self._cond:
            if self.finished:
                return
            self.state = state
            self.exit_code = exit_code
            self.error = error
            self._cond.notify_all()

    def _read_from(self, index: int, timeout: float):
        """Block until there is output past `index` or the job ends.

        Returns `(lines, next_index, dropped, finished)`. `dropped` is how many
        lines fell out of the buffer before this reader got to them.
        """
        with self._cond:
            if index >= self._dropped + len(self._lines) and not self.finished:
                self._cond.wait(timeout=timeout)
            dropped = 0
            if index < self._dropped:
                dropped = self._dropped - index
                index = self._dropped
            lines = self._lines[index - self._dropped :]
            return lines, index + len(lines), dropped, self.finished


class JobRegistry:
    """Every job this process has started, by id."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def add(self, job: Job) -> Job:
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise UnknownJob(f"no job {job_id!r}")
        return job

    def all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def shutdown(self, *, grace: float = 3.0, timeout: float = 15.0) -> None:
        """Stop everything still running. Safe to call twice."""
        running = [job for job in self.all() if not job.finished]
        for job in running:
            cancel(self, job.id, grace=grace)
        deadline = time.monotonic() + timeout
        for job in running:
            while not job.finished and time.monotonic() < deadline:
                time.sleep(0.02)


# --------------------------------------------------------------------- start


def start(
    registry: JobRegistry,
    argv: Sequence[str],
    *,
    cwd,
    timeout: float | None = DEFAULT_TIMEOUT_S,
    term_grace: float = DEFAULT_TERM_GRACE_S,
    stall_grace: float | None = None,
    max_lines: int = DEFAULT_MAX_LINES,
    label: str = "",
    kind: str = "",
    env: Mapping[str, str] | None = None,
    signal_group: bool = True,
    on_done: Callable[[Job], None] | None = None,
) -> Job:
    """Spawn `argv` and return immediately with a registered `Job`.

    A command that cannot be spawned at all comes back as a `failed` job rather
    than an exception: the caller is a route that has already promised the
    operator a job id, and "failed, here is why" renders where a 500 does not.
    """
    argv = [str(part) for part in argv]
    if not argv:
        raise ValueError("argv must name a command to run")

    job = Job(
        id=uuid.uuid4().hex[:12],
        argv=argv,
        cwd=str(cwd),
        label=label,
        kind=kind,
        timeout=timeout,
        term_grace=term_grace,
        stall_grace=stall_grace,
        max_lines=max_lines,
        signal_group=signal_group,
        on_done=on_done,
    )
    registry.add(job)

    child_env = dict(os.environ)
    # Request unbuffered Python output so piped child progress arrives before exit.
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env.update(env or {})

    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # Use unbuffered byte reads so output can stream as soon as it is available.
            bufsize=0,
            env=child_env,
            # Own session, so the child is a process-group leader and cancel
            # can signal the helpers it spawns as well as the child itself.
            start_new_session=True,
        )
    except OSError as exc:
        message = f"could not start {argv[0]!r}: {exc}"
        job._append(f"[job] {message}")
        job._finish(FAILED, error=message)
        return job

    job._proc = proc
    # A new-session child leads its own process group, so pgid equals pid.
    # Save it now for cleanup after the leader is reaped.
    job.pgid = proc.pid
    job.state = RUNNING

    pump = threading.Thread(
        target=_pump, args=(job,), name=f"job-{job.id}-pump", daemon=True
    )
    pump.start()
    threading.Thread(
        target=_supervise, args=(job, pump), name=f"job-{job.id}", daemon=True
    ).start()

    if timeout:
        job._timers.append(_after(timeout, _on_timeout, job))
    return job


def _after(delay: float, func, job: Job) -> threading.Timer:
    timer = threading.Timer(delay, func, args=(job,))
    timer.daemon = True
    timer.start()
    return timer


# ---------------------------------------------------------------- lifecycle


def _pump(job: Job) -> None:
    """Split the merged output stream into lines as it arrives."""
    stream_fd = job._proc.stdout.fileno()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buffered = ""
    while True:
        try:
            chunk = os.read(stream_fd, _READ_SIZE)
        except OSError:
            break
        if not chunk:
            break
        buffered += decoder.decode(chunk)
        parts = _LINE_BREAK.split(buffered)
        buffered = parts.pop()
        for part in parts:
            job._append(part)
    # A final write with no trailing newline is still output the operator needs
    # to see -- an `input()` prompt, for instance.
    tail = buffered + decoder.decode(b"", final=True)
    if tail:
        job._append(tail)
    try:
        job._proc.stdout.close()
    except OSError:
        pass


def _supervise(job: Job, pump: threading.Thread) -> None:
    """Own the child's exit. The exit code, not the pipe, ends the job."""
    code = job._proc.wait()
    pump.join(timeout=_DRAIN_S)
    for timer in job._timers:
        timer.cancel()

    if job._timed_out:
        job._finish(FAILED, exit_code=code, error=f"timed out after {job.timeout:g}s")
    elif job._cancelled:
        job._finish(CANCELLED, exit_code=code)
    elif code == 0:
        job._finish(DONE, exit_code=code)
    else:
        job._finish(FAILED, exit_code=code, error=f"exited with code {code}")

    # Collect surviving helpers after the leader exits. Native crashes and
    # SIGKILL bypass the recorder's finally block and can leave devices claimed.
    if job.pgid:
        left = terminate_group(job.pgid, term_wait=2.0, kill_wait=3.0)
        if left:
            pids = ", ".join(str(proc.pid) for proc in left)
            job._append(f"[job] {len(left)} process(es) survived after exit: {pids}")

    # Invoke completion only after setting final job state and collecting
    # writers; audit sealing must not race an active episode write. Callback
    # errors must not interrupt supervisor cleanup.
    if job.on_done is not None:
        try:
            job.on_done(job)
        except Exception:  # noqa: BLE001
            logger.exception("job %s: completion callback failed", job.id)


def _signal_group(job: Job, sig: int, *, group: bool | None = None) -> bool:
    """Signal the child, or its whole process group. False if it is gone.

    `group` overrides `job.signal_group` for one call, which is how escalation
    stays group-wide even for a job whose graceful stop is deliberately narrow.
    """
    proc = job._proc
    if proc is None:
        return False
    use_group = job.signal_group if group is None else group
    with job._proc_lock:
        # The session-leading child's pid is its pgid. Guard signaling after exit
        # to reduce the unavoidable PID-reuse race.
        if proc.returncode is not None:
            return False
        try:
            if use_group and hasattr(os, "killpg"):
                os.killpg(proc.pid, sig)
            else:
                proc.send_signal(sig)
        except (ProcessLookupError, PermissionError):
            return False
        except OSError as exc:
            logger.warning("job %s: could not signal %s: %s", job.id, sig, exc)
            return False
    return True


def _on_timeout(job: Job) -> None:
    if job.finished:
        return
    job._timed_out = True
    message = f"wall-clock cap of {job.timeout:g}s reached, terminating"
    logger.warning("job %s: %s (%s)", job.id, message, " ".join(job.argv))
    job._append(f"[job] {message}")
    _signal_group(job, signal.SIGTERM)
    job._timers.append(_after(job.term_grace, _escalate, job))


def _escalate(job: Job) -> None:
    """Last resort, and never quiet: a hard kill skipped the child's cleanup."""
    if job.finished:
        return
    message = (
        f"child ignored SIGTERM for {job.term_grace:g}s, escalating to SIGKILL "
        "(its shutdown handlers did not run)"
    )
    logger.warning("job %s: %s (%s)", job.id, message, " ".join(job.argv))
    job._append(f"[job] {message}")
    # Always the whole group, even when the graceful stop was narrowed: a job
    # that ignored SIGTERM may have left helpers running, and this is the path
    # that exists to guarantee nothing survives.
    _signal_group(job, signal.SIGKILL, group=True)


def _poll_until_empty(pgid: int, deadline: float) -> list:
    while time.monotonic() < deadline:
        left = process_group.survivors(pgid)
        if not left:
            return []
        time.sleep(_SURVIVOR_POLL_S)
    return process_group.survivors(pgid)


def terminate_group(pgid, *, term_wait: float = 5.0, kill_wait: float = 5.0) -> list:
    """End every process in `pgid`, verify it, and return what survived.

    Signals are fire-and-forget. Sending one and reporting success is how a
    console tells an operator the rig is free while a forked worker still
    holds an exclusive OAK, so each stage is followed by a poll of
    `process_group.survivors`, and the return value is the evidence.

    An empty list is the only success. A non-empty one is not a bug in this
    function: a task in uninterruptible sleep (state `D`, which is what a
    wedged USB or DepthAI ioctl produces) cannot be killed at all, by anyone.
    The caller has to say so rather than report the rig free; a new recorder
    could not have the cameras in that state anyway.
    """
    if not pgid or pgid <= 0:
        return []
    if not process_group.survivors(pgid):
        return []
    for sig, wait in ((signal.SIGTERM, term_wait), (signal.SIGKILL, kill_wait)):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            # Report actual survivors when the group has exited or signaling is not
            # permitted.
            return _poll_until_empty(pgid, time.monotonic() + wait)
        if not _poll_until_empty(pgid, time.monotonic() + wait):
            return []
    return process_group.survivors(pgid)


def kill_now(registry: JobRegistry, job_id) -> list:
    """Force stop: SIGKILL the whole group at once, and verify. What survived.

    The operator's escape hatch, so it never waits out a grace period and never
    asks politely first -- `cancel` already did that, and this exists because it
    did not work. Pending escalations are cancelled so a timer cannot later fire
    into a group that is already gone.
    """
    job = registry.get(job_id)
    job._cancelled = True
    for timer in job._timers:
        timer.cancel()
    job._timers.clear()
    survived = terminate_group(job.pgid, term_wait=0.0, kill_wait=5.0)
    if survived:
        pids = ", ".join(str(proc.pid) for proc in survived)
        job._append(f"[job] {len(survived)} process(es) survived SIGKILL: {pids}")
    return survived


def _stall_check(job: Job) -> None:
    """Escalate a cancelled child that has gone silent, not merely slow.

    Reschedules itself while the child is still talking, so a drain printing per
    episode is never truncated and one wedged in `stop_streaming` on a dead
    camera does not sit out the whole flat ceiling. See `Job.stall_grace`.
    """
    if job.finished or not job.stall_grace:
        return
    quiet = time.monotonic() - job.last_output_at
    if quiet >= job.stall_grace:
        job._append(f"[job] no output for {quiet:.0f}s while stopping; escalating")
        _escalate(job)
        return
    job._timers.append(_after(min(job.stall_grace, 5.0), _stall_check, job))


def cancel(registry: JobRegistry, job_id, *, grace: float | None = None) -> None:
    """Ask the job to stop, gracefully.

    Always `SIGTERM` first. `record_data.py` drains queued alignments in a
    `finally` block that `SIGKILL` would skip, so a hard kill
    is data loss. Returns as soon as the signal is delivered -- the escalation
    runs on a timer, so a route never blocks for the grace period.
    """
    job = registry.get(job_id)
    if job.finished:
        return
    # Arm one escalation timer; repeated cancellation must be idempotent.
    already_cancelled = job._cancelled
    job._cancelled = True
    if job._proc is None:
        job._finish(CANCELLED)
        return
    logger.info("job %s: cancelling (%s)", job.id, " ".join(job.argv))
    if _signal_group(job, signal.SIGTERM) and not already_cancelled:
        job._timers.append(
            _after(job.term_grace if grace is None else grace, _escalate, job)
        )
        # Use the silence trigger for an unresponsive child; retain the overall
        # grace as a final ceiling.
        if job.stall_grace:
            job._timers.append(_after(min(job.stall_grace, 5.0), _stall_check, job))


# ----------------------------------------------------------------- streaming


def data_frame(text: str) -> str:
    return f"data: {text}\n\n"


def _end_frame(job: Job) -> str:
    payload = json.dumps(
        {
            "id": job.id,
            "state": job.state,
            "exit_code": job.exit_code,
            "error": job.error,
        }
    )
    return f"event: end\ndata: {payload}\n\n"


def stream(
    registry: JobRegistry, job_id, *, keepalive: float = KEEPALIVE_S
) -> Iterator[str]:
    """Server-Sent Event frames for a job, from its first line to its last.

    Not itself a generator function: a generator would defer the lookup to the
    first `next()`, by which point `StreamingResponse` has already sent
    `200 OK` and an unknown id could never be answered with a 404.

    Each reader keeps its own cursor, so two tabs watching one recording both
    see every line, and a reader that attaches late replays the whole buffer.
    """
    job = registry.get(job_id)
    return _frames(job, keepalive)


def _frames(job: Job, keepalive: float) -> Iterator[str]:
    index = 0
    while True:
        lines, index, dropped, finished = job._read_from(index, keepalive)
        if dropped:
            yield data_frame(f"[job] {dropped} earlier output lines dropped")
        for line in lines:
            yield data_frame(line)
        if not lines and not dropped:
            if finished:
                break
            yield KEEPALIVE_FRAME
    yield _end_frame(job)

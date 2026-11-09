# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One model for every long-running subprocess the station starts.

Each job has captured output, a real exit code, a wall-clock cap, and a
cancellation that a child can survive long enough to clean up after itself --
rather than a fire-and-forget launch with no pid, no exit code and no output,
or hardware opened from a request handler with no timeout, which would pin an
HTTP worker forever.
"""

import json
import logging
import os
import signal
import sys
import textwrap
import threading
import time

import pytest

from station_ui import jobs, process_group
from station_ui.errors import UnknownJob
from station_ui.jobs import JobRegistry, cancel, start, stream

_DEADLINE = 30.0


@pytest.fixture
def registry():
    reg = JobRegistry()
    yield reg
    reg.shutdown()


def _py(source, *args):
    """A child that does not pass -u: real launchers do not either."""
    return [sys.executable, "-c", textwrap.dedent(source), *args]


def _wait_until_finished(job, timeout=_DEADLINE):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if job.finished:
            return
        time.sleep(0.02)
    raise AssertionError(f"job {job.id} still {job.state} after {timeout}s")


def _wait_for_line(job, needle, timeout=_DEADLINE):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(needle in line for line in job.lines()):
            return
        time.sleep(0.02)
    raise AssertionError(f"never saw {needle!r}; got {job.lines()!r}")


def _payloads(frames):
    return [
        f.removeprefix("data: ").removesuffix("\n\n")
        for f in frames
        if f.startswith("data: ")
    ]


# ----------------------------------------------------------- exit code + state


def test_a_successful_job_reports_done_and_a_zero_exit_code(registry):
    job = start(registry, _py("print('hello')"), cwd=".")
    _wait_until_finished(job)
    assert job.state == jobs.DONE
    assert job.exit_code == 0
    assert "hello" in job.lines()


def test_a_nonzero_exit_is_reported_as_failed_with_the_code(registry):
    """The exit code decides the outcome; nobody is asked whether it worked."""
    job = start(registry, _py("import sys\nsys.exit(3)"), cwd=".")
    _wait_until_finished(job)
    assert job.state == jobs.FAILED
    assert job.exit_code == 3


def test_stderr_is_captured_alongside_stdout(registry):
    """A traceback goes to stderr. Dropping it would leave a failed job with no
    explanation."""
    job = start(registry, _py("import sys\nraise RuntimeError('boom')"), cwd=".")
    _wait_until_finished(job)
    assert job.state == jobs.FAILED
    joined = "\n".join(job.lines())
    assert "RuntimeError" in joined and "boom" in joined


def test_the_job_runs_in_the_requested_directory(registry, tmp_path):
    (tmp_path / "marker.txt").write_text("x")
    job = start(
        registry, _py("import os\nprint(sorted(os.listdir('.')))"), cwd=tmp_path
    )
    _wait_until_finished(job)
    assert "marker.txt" in "\n".join(job.lines())


def test_start_returns_without_waiting_for_the_child(registry):
    started = time.monotonic()
    job = start(registry, _py("import time\ntime.sleep(5)"), cwd=".")
    assert time.monotonic() - started < 1.0, "start() blocked on the child"
    assert job.state == jobs.RUNNING
    cancel(registry, job.id)


def test_a_child_that_reads_stdin_fails_fast_instead_of_hanging(registry):
    """`record_data.py` and gripper calibration both call `input()`.
    Inheriting the server's stdin blocks the child forever with nothing able
    to answer it; a closed stdin turns that into a visible failure."""
    job = start(registry, _py("input('prompt: ')"), cwd=".")
    _wait_until_finished(job, timeout=10)
    assert job.state == jobs.FAILED
    assert "EOF" in "\n".join(job.lines())


def test_an_unstartable_command_fails_the_job_instead_of_raising(registry):
    job = start(registry, ["definitely-not-a-real-binary-xyz"], cwd=".")
    assert job.state == jobs.FAILED
    assert job.error
    frames = list(stream(registry, job.id))
    assert frames[-1].startswith("event: end")
    assert "definitely-not-a-real-binary-xyz" in "\n".join(frames)


def test_an_empty_argv_is_rejected(registry):
    with pytest.raises(ValueError):
        start(registry, [], cwd=".")


def test_an_unknown_job_id_raises_before_any_response_starts(registry):
    """`stream` must not itself be a generator function: a generator defers
    the lookup to the first `next()`, by which point StreamingResponse has
    already sent `200 OK` and the 404 can never be delivered."""
    with pytest.raises(UnknownJob):
        stream(registry, "nope")
    with pytest.raises(UnknownJob):
        cancel(registry, "nope")
    with pytest.raises(UnknownJob):
        registry.get("nope")


# ------------------------------------------------------------------ streaming


def test_stdout_is_streamed_as_data_frames_that_end_with_an_end_event(registry):
    job = start(registry, _py("print('alpha')\nprint('beta')"), cwd=".")
    frames = list(stream(registry, job.id))
    assert _payloads(frames) == ["alpha", "beta"]
    assert frames[-1].startswith("event: end\n")
    payload = json.loads(frames[-1].split("data: ", 1)[1])
    assert payload["state"] == jobs.DONE
    assert payload["exit_code"] == 0
    assert payload["id"] == job.id


def test_the_end_event_carries_the_failure(registry):
    job = start(registry, _py("import sys\nsys.exit(7)"), cwd=".")
    frames = list(stream(registry, job.id))
    payload = json.loads(frames[-1].split("data: ", 1)[1])
    assert payload["state"] == jobs.FAILED
    assert payload["exit_code"] == 7
    assert payload["error"]


def test_the_stream_is_incremental_rather_than_buffered(registry):
    job = start(
        registry,
        [
            sys.executable,
            "-u",
            "-c",
            "import time,sys\n"
            "print('first', flush=True)\n"
            "time.sleep(5)\n"
            "print('second', flush=True)",
        ],
        cwd=".",
    )
    frames = stream(registry, job.id)
    started = time.monotonic()
    assert "first" in next(f for f in frames if "first" in f)
    assert time.monotonic() - started < 3, "stream buffered instead of yielding"
    cancel(registry, job.id)


def test_a_child_that_does_not_pass_dash_u_still_streams_incrementally(registry):
    """No real launcher adds `-u`. A Python child writing to a pipe block-
    buffers stdout, so without `PYTHONUNBUFFERED` in the child's environment
    the first line only arrives at exit and every 'live' page is a lie."""
    job = start(registry, _py("import time\nprint('first')\ntime.sleep(5)\n"), cwd=".")
    frames = stream(registry, job.id)
    started = time.monotonic()
    assert "first" in next(f for f in frames if "first" in f)
    assert time.monotonic() - started < 3, "child stdout was block-buffered"
    cancel(registry, job.id)


def test_ten_thousand_lines_arrive_before_the_child_exits(registry):
    """The load-bearing assertion is the last one: the child sleeps after its
    final print, so seeing all 10k lines while the job is still RUNNING proves
    delivery is incremental and not a `communicate()` at exit. It also proves
    the pump keeps draining -- 10k lines overflow the 64 KiB pipe buffer, so a
    reader that waits would deadlock the child instead."""
    job = start(
        registry,
        _py("import time\nfor i in range(10000): print(f'line-{i}')\ntime.sleep(4)\n"),
        cwd=".",
    )
    started = time.monotonic()
    first_at = None
    seen = 0
    for frame in stream(registry, job.id):
        if frame.startswith("data: line-"):
            if first_at is None:
                first_at = time.monotonic() - started
            seen += 1
            if seen == 10000:
                break
    assert first_at is not None and first_at < 3, f"first frame took {first_at}s"
    assert seen == 10000
    assert job.state == jobs.RUNNING, "all 10k lines only landed once the child exited"
    cancel(registry, job.id)


def test_a_late_subscriber_replays_the_whole_output(registry):
    """A reload after the job ended must still show what it printed."""
    job = start(registry, _py("print('alpha')\nprint('beta')"), cwd=".")
    _wait_until_finished(job)
    assert _payloads(list(stream(registry, job.id))) == ["alpha", "beta"]


def test_two_readers_of_one_job_each_get_every_line(registry):
    """Two tabs watching one recording. A queue-based fan-out silently splits
    the output between them; each reader must own its own cursor."""
    job = start(
        registry,
        _py("import time\nfor i in range(200): print(i)\ntime.sleep(0.3)\n"),
        cwd=".",
    )
    results = []
    lock = threading.Lock()

    def read():
        payloads = _payloads(list(stream(registry, job.id)))
        with lock:
            results.append(payloads)

    threads = [threading.Thread(target=read) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=_DEADLINE)

    assert len(results) == 2
    assert results[0] == results[1] == [str(i) for i in range(200)]


def test_carriage_return_progress_output_becomes_its_own_frames(registry):
    """`tqdm` redraws with `\\r` and no newline (the audit runs tqdm scripts).
    Splitting only on `\\n` makes the whole progress bar one frame that
    arrives when the job ends -- i.e. no progress at all."""
    job = start(
        registry,
        _py(
            "import sys, time\n"
            "for i in range(3):\n"
            "    sys.stdout.write(f'step {i}\\r')\n"
            "    sys.stdout.flush()\n"
            "    time.sleep(0.05)\n"
            "print()\n"
        ),
        cwd=".",
    )
    _wait_until_finished(job)
    assert "step 0" in job.lines() and "step 2" in job.lines()


# --------------------------------------------------------------- cancellation


def test_cancel_delivers_sigterm_so_the_child_can_clean_up(registry, tmp_path):
    """SIGKILL passes a naive 'the process ended' assertion while skipping
    `record_data.py`'s `finally` block, which drains queued alignments. Assert the signal number the child actually received."""
    marker = tmp_path / "signal.txt"
    job = start(
        registry,
        _py(
            f"""
            import signal, sys, time
            def handler(signum, frame):
                open({str(marker)!r}, 'w').write(str(signum))
                sys.exit(0)
            signal.signal(signal.SIGTERM, handler)
            print('ready')
            time.sleep(30)
            """
        ),
        cwd=".",
    )
    _wait_for_line(job, "ready")
    cancel(registry, job.id)
    _wait_until_finished(job)

    assert marker.read_text() == str(int(signal.SIGTERM)) == "15"
    assert job.state == jobs.CANCELLED
    assert job.exit_code == 0, "the child was killed before its cleanup finished"


def test_a_child_without_a_handler_dies_of_sigterm_not_sigkill(registry):
    """The signal number again, from the other side: a process killed by
    SIGTERM reports -15 and one killed by SIGKILL reports -9."""
    job = start(registry, _py("import time\nprint('ready')\ntime.sleep(30)"), cwd=".")
    _wait_for_line(job, "ready")
    cancel(registry, job.id)
    _wait_until_finished(job)

    assert job.exit_code == -int(signal.SIGTERM) == -15
    assert job.exit_code != -int(signal.SIGKILL)
    assert job.state == jobs.CANCELLED


def test_cancel_reaches_a_grandchild_in_the_same_process_group(registry):
    """`record_data.py` spawns helpers. Signalling only the direct child
    leaves them running and holding the staging directory."""
    job = start(
        registry,
        _py(
            """
            import os, subprocess, sys, time
            child = subprocess.Popen([sys.executable, '-c',
                'import time\\nwhile True: time.sleep(0.1)'])
            print(child.pid)
            time.sleep(30)
            """
        ),
        cwd=".",
    )
    deadline = time.monotonic() + _DEADLINE
    grandchild = None
    while time.monotonic() < deadline and grandchild is None:
        for line in job.lines():
            if line.strip().isdigit():
                grandchild = int(line.strip())
                break
        time.sleep(0.02)
    assert grandchild is not None

    cancel(registry, job.id)
    _wait_until_finished(job)

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild, 0)
        except OSError:
            return
        time.sleep(0.05)
    raise AssertionError(f"grandchild {grandchild} outlived the cancel")


def test_sigkill_escalation_waits_for_the_grace_and_is_announced(registry, caplog):
    """Escalation is a last resort and must never be silent: an operator whose
    recording was hard-killed needs to know the drain was skipped."""
    job = start(
        registry,
        _py(
            """
            import signal, time
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            print('ready')
            while True:
                time.sleep(0.1)
            """
        ),
        cwd=".",
    )
    _wait_for_line(job, "ready")

    with caplog.at_level(logging.WARNING, logger="station_ui.jobs"):
        cancel(registry, job.id, grace=1.0)
        time.sleep(0.4)
        assert job.state == jobs.RUNNING, "escalated to SIGKILL with no grace period"
        _wait_until_finished(job, timeout=20)

    assert job.exit_code == -int(signal.SIGKILL)
    assert any("SIGKILL" in r.getMessage() for r in caplog.records), caplog.text
    assert any("SIGKILL" in line for line in job.lines()), job.lines()


def test_cancelling_a_finished_job_is_a_no_op(registry):
    job = start(registry, _py("print('done')"), cwd=".")
    _wait_until_finished(job)
    cancel(registry, job.id)
    assert job.state == jobs.DONE


def test_registry_shutdown_stops_every_running_job():
    reg = JobRegistry()
    job = start(reg, _py("import time\nprint('ready')\ntime.sleep(60)"), cwd=".")
    _wait_for_line(job, "ready")
    reg.shutdown()
    assert job.finished
    assert job.exit_code is not None


# ------------------------------------------------------------ wall-clock cap


def test_a_job_that_outruns_its_wall_clock_cap_is_failed(registry):
    """Hardware opens in this repo have no timeout of their own; inline in a
    route that pins a worker until the operator gives up."""
    job = start(registry, _py("import time\ntime.sleep(30)"), cwd=".", timeout=1.0)
    _wait_until_finished(job, timeout=20)
    assert job.state == jobs.FAILED
    assert "timed out" in (job.error or "")
    assert any("cap" in line for line in job.lines()), job.lines()


def test_the_timeout_stream_terminates_too(registry):
    """A stream that never ends holds the browser's connection open forever."""
    job = start(registry, _py("import time\ntime.sleep(30)"), cwd=".", timeout=1.0)
    frames = list(stream(registry, job.id))
    payload = json.loads(frames[-1].split("data: ", 1)[1])
    assert payload["state"] == jobs.FAILED
    assert "timed out" in payload["error"]


def test_every_job_carries_a_wall_clock_cap_by_default(registry):
    job = start(registry, _py("import time\ntime.sleep(30)"), cwd=".")
    assert job.timeout is not None and job.timeout > 0
    cancel(registry, job.id)


def test_a_cap_of_none_lets_a_long_recording_run(registry):
    """A recording legitimately lasts hours; the opt-out from the wall-clock
    cap has to be explicit rather than a default nobody notices."""
    job = start(
        registry,
        _py("import time\nprint('ready')\ntime.sleep(30)"),
        cwd=".",
        timeout=None,
    )
    _wait_for_line(job, "ready")
    time.sleep(0.5)
    assert job.state == jobs.RUNNING
    cancel(registry, job.id)


def test_a_job_that_finishes_inside_its_cap_is_not_marked_timed_out(registry):
    job = start(registry, _py("print('quick')"), cwd=".", timeout=10.0)
    _wait_until_finished(job)
    assert job.state == jobs.DONE
    assert job.error is None


# Verify process-group exit after signaling; sending a signal does not
# establish that device-owning helpers are gone.

# Wait for the child's handler-installed marker before signaling.
# Otherwise default SIGTERM exit could falsely pass an escalation test.
STUBBORN = """
    import signal, time
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print('armed', flush=True)
    time.sleep(30)
"""

# The orphan sleeps far longer than any deadline in this file, so the assertion
# below cannot pass by waiting for the worker to die of old age: only an actual
# reap satisfies it.
ORPHANER = """
    import subprocess, sys, time
    subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])
    time.sleep(0.2)
"""


def _wait_running(job, timeout=_DEADLINE):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if job.state == jobs.RUNNING:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job.id} never ran: {job.state}")


def _wait_group_populated(pgid, count, timeout=_DEADLINE):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(process_group.survivors(pgid)) >= count:
            return
        time.sleep(0.02)


def test_terminate_group_confirms_an_ordinary_child_died(registry):
    job = start(registry, _py("import time; time.sleep(30)"), cwd=".", timeout=None)
    _wait_running(job)
    assert jobs.terminate_group(job.pgid, term_wait=5.0, kill_wait=5.0) == []


def test_terminate_group_escalates_for_a_child_that_ignores_sigterm(registry):
    """SIGTERM alone cannot end this child. Reporting success after sending it
    is exactly the lie this function exists to stop telling."""
    job = start(registry, _py(STUBBORN), cwd=".", timeout=None)
    _wait_for_line(job, "armed")
    assert jobs.terminate_group(job.pgid, term_wait=0.5, kill_wait=5.0) == []


def test_a_worker_outliving_its_parent_is_reaped_when_the_job_ends(registry):
    """The orphan. `_supervise` finishes the job on the parent's exit code, so
    without a group reap a forked worker goes on holding every camera and
    the next Start fails to open a board with a message blaming the cable."""
    job = start(registry, _py(ORPHANER), cwd=".", timeout=None)
    pgid = job.pgid
    _wait_group_populated(pgid, 2)
    _wait_until_finished(job)
    # `_supervise` reaps the group after the parent exits. Bounded well below
    # the worker's own 600s lifetime, so passing means it was killed.
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and process_group.survivors(pgid):
        time.sleep(0.02)
    assert process_group.survivors(pgid) == [], "the orphaned worker survived"


def test_cancel_twice_does_not_stack_escalation_timers(registry):
    """Repeated presses must not each append another `_after(term_grace,
    _escalate)`: an operator mashing Stop would schedule a queue of kills
    instead of escalating."""
    job = start(registry, _py(STUBBORN), cwd=".", timeout=None, term_grace=30.0)
    _wait_for_line(job, "armed")
    cancel(registry, job.id)
    after_one = len(job._timers)
    cancel(registry, job.id)
    cancel(registry, job.id)
    assert len(job._timers) == after_one, "Stop stacked escalation timers"
    jobs.kill_now(registry, job.id)


def test_kill_now_kills_the_whole_group_immediately(registry):
    """Force stop never waits out a grace period: `cancel` already asked
    politely, and this exists because that did not work."""
    job = start(registry, _py(STUBBORN), cwd=".", timeout=None, term_grace=900.0)
    _wait_for_line(job, "armed")
    assert jobs.kill_now(registry, job.id) == []


def test_last_output_at_moves_only_when_the_child_prints(registry):
    job = start(
        registry,
        _py("import time; print('one', flush=True); time.sleep(30)"),
        cwd=".",
        timeout=None,
    )
    _wait_for_line(job, "one")
    stamped = job.last_output_at
    time.sleep(0.3)
    assert job.last_output_at == stamped, "the stamp moved with no output"
    jobs.kill_now(registry, job.id)


# ---- the unattended backstop -------------------------------------------------


def test_a_silent_cancelled_child_is_escalated_without_waiting_for_the_ceiling(
    registry,
):
    """`term_grace` is derived from the child's own drain bounds, and both
    drains are self-bounding -- so a healthy child cannot exceed them and the
    flat ceiling only ever fires for one wedged before or outside the drain,
    where the whole budget is dead time nobody benefits from."""
    job = start(
        registry,
        _py(STUBBORN),
        cwd=".",
        timeout=None,
        term_grace=900.0,
        stall_grace=0.3,
    )
    _wait_for_line(job, "armed")
    cancel(registry, job.id)
    _wait_until_finished(job, timeout=20.0)
    assert job.state == jobs.CANCELLED


def test_a_child_that_keeps_printing_is_not_escalated_by_the_stall_backstop(registry):
    """A drain printing per episode must never be truncated -- that is the data
    loss the SIGTERM path exists to prevent, arriving by another route."""
    chatty = """
        import signal, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print('up', flush=True)
        for _ in range(200):
            print('working', flush=True)
            time.sleep(0.05)
    """
    job = start(
        registry,
        _py(chatty),
        cwd=".",
        timeout=None,
        term_grace=900.0,
        stall_grace=0.5,
    )
    _wait_for_line(job, "up")
    cancel(registry, job.id)
    time.sleep(2.0)
    assert not job.finished, "a child that was still printing was killed"
    jobs.kill_now(registry, job.id)


# Run on_done on the supervisor thread after exit; the request that
# started the job must not wait for completion.


def test_the_completion_callback_runs_after_the_job_has_finished(registry):
    """It has to see a job whose state is final: a caller branches on it, and
    a callback that observed RUNNING would act on a job that had not finished."""
    seen = []
    job = start(
        registry,
        _py("print('hi')"),
        cwd=".",
        on_done=lambda finished: seen.append((finished.state, finished.exit_code)),
    )
    _wait_until_finished(job)
    deadline = time.monotonic() + _DEADLINE
    while not seen and time.monotonic() < deadline:
        time.sleep(0.01)
    assert seen == [(jobs.DONE, 0)]


def test_the_callback_is_told_about_a_failure_too(registry):
    """Not filtered here. Which terminal states are worth acting on is the
    caller's decision."""
    seen = []
    job = start(
        registry,
        _py("raise SystemExit(3)"),
        cwd=".",
        on_done=lambda f: seen.append(f.state),
    )
    _wait_until_finished(job)
    deadline = time.monotonic() + _DEADLINE
    while not seen and time.monotonic() < deadline:
        time.sleep(0.01)
    assert seen == [jobs.FAILED]


def test_a_raising_callback_does_not_take_the_supervisor_with_it(registry):
    """The supervisor also collects the child's surviving helpers and settles
    the registry's view of the job. Neither may depend on a caller's follow-up
    work being correct."""
    job = start(
        registry,
        _py("print('hi')"),
        cwd=".",
        on_done=lambda _job: 1 / 0,
    )
    _wait_until_finished(job)
    assert job.state == jobs.DONE
    assert job.exit_code == 0

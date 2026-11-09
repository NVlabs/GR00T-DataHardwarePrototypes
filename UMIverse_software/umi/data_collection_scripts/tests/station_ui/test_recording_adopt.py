# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A recording outlives the web server. The console has to cope with that.

`jobs.start` detaches with `start_new_session=True`, and the recorder's registry
is the one with no shutdown reaper -- deliberately, since ending a shift's
recording because somebody restarted the console would be worse than the
problem. So after a restart the recorder is still running, and the page must
not come back offering Start at cameras another process is holding.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time
import types

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

from station_ui import process_group, recording  # noqa: E402

SLEEPER = "import time; time.sleep(600)"
ABSENT_PID = 0x7FFFFFF0


def _app(tmp_path):
    return types.SimpleNamespace(
        state=types.SimpleNamespace(
            settings=types.SimpleNamespace(state_dir=tmp_path),
        )
    )


@pytest.fixture
def detached():
    """A recorder that outlived its console, as far as the console can tell."""
    proc = subprocess.Popen([sys.executable, "-c", SLEEPER], start_new_session=True)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not process_group.survivors(proc.pid):
        time.sleep(0.02)
    yield proc
    proc.kill()
    proc.wait(timeout=5)


def _write(tmp_path, **fields):
    (tmp_path / recording.PID_FILE).write_text(json.dumps(fields))


def test_no_pid_file_adopts_nothing(tmp_path):
    assert recording.adopt(_app(tmp_path)) == "none"
    assert recording.in_progress(_app(tmp_path)) is False


def test_a_live_recorder_is_adopted(tmp_path, detached):
    _write(
        tmp_path,
        pid=detached.pid,
        pgid=detached.pid,
        start_time=process_group.start_time(detached.pid),
        job_id="abc",
    )
    app = _app(tmp_path)
    assert recording.adopt(app) == "adopted"
    state = recording.state_for(app)
    assert state.pid == detached.pid and state.adopted is True
    assert state.record_job_id is None, "an adopted recording has no captured output"
    assert recording.in_progress(app) is True


def test_a_dead_pid_is_stale_and_the_file_goes(tmp_path):
    _write(tmp_path, pid=ABSENT_PID, pgid=ABSENT_PID, start_time=1, job_id="abc")
    app = _app(tmp_path)
    assert recording.adopt(app) == "stale"
    assert not (tmp_path / recording.PID_FILE).exists()
    assert recording.in_progress(app) is False


def test_a_recycled_pid_is_refused_and_nothing_is_signalled(tmp_path, detached):
    """A pid is not an identity -- Linux recycles them. Adopting on the pid
    alone would let the console SIGKILL an unrelated process after a reboot."""
    _write(
        tmp_path,
        pid=detached.pid,
        pgid=detached.pid,
        start_time=(process_group.start_time(detached.pid) or 0) + 999,
        job_id="abc",
    )
    app = _app(tmp_path)
    assert recording.adopt(app) == "recycled"
    assert recording.in_progress(app) is False
    time.sleep(0.2)
    assert detached.poll() is None, "an unrelated process was killed"


def test_in_progress_releases_once_the_adopted_recorder_exits(tmp_path, detached):
    """The lock's release path. Derived from the process group, so it cannot be
    left set by a flag nobody cleared."""
    _write(
        tmp_path,
        pid=detached.pid,
        pgid=detached.pid,
        start_time=process_group.start_time(detached.pid),
        job_id="abc",
    )
    app = _app(tmp_path)
    assert recording.adopt(app) == "adopted"
    assert recording.in_progress(app) is True

    detached.kill()
    detached.wait(timeout=5)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and recording.in_progress(app):
        time.sleep(0.02)
    assert recording.in_progress(app) is False, "the recording lock never released"


def test_remember_then_forget_round_trips(tmp_path):
    app = _app(tmp_path)
    job = types.SimpleNamespace(pgid=ABSENT_PID, id="abc", argv=["x"])
    recording.remember(app, job)
    assert (tmp_path / recording.PID_FILE).exists()
    recording.forget(app)
    assert not (tmp_path / recording.PID_FILE).exists()
    recording.forget(app)  # idempotent

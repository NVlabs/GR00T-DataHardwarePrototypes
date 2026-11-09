# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which processes are still alive in a process group, read from /proc.

`ps aux` was the obvious answer and it is the wrong one: a text match on a
command line also catches another station's recorder, a `tail -f` of the log, an
editor with the file open, and the grep itself. Killing on a text match is how
something unrelated gets killed. The pgid is an exact answer, because
`jobs.start` already passes `start_new_session=True`.
"""

from __future__ import annotations

import os
import pathlib
import signal
import subprocess
import sys
import time

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

from station_ui import process_group  # noqa: E402

# A leader that spawns one child and then sleeps -- the shape `record_data.py`
# has once `BackgroundAligner` has forked its multiprocessing worker.
LEADER = (
    "import subprocess,sys,time;"
    "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']);"
    "time.sleep(30)"
)

# Far above any real pid on a station, so it names nothing.
ABSENT_PID = 0x7FFFFFF0


@pytest.fixture
def group():
    """A live two-process group, torn down however the test left it."""
    proc = subprocess.Popen([sys.executable, "-c", LEADER], start_new_session=True)
    # `start_new_session` makes the child its own group leader, so pgid == pid.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and len(process_group.survivors(proc.pid)) < 2:
        time.sleep(0.02)
    yield proc
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover - the kill above worked
        pass


def test_survivors_finds_the_leader_and_its_forked_child(group):
    found = process_group.survivors(group.pid)
    assert len(found) == 2, f"expected the leader and its child, got {found}"
    assert group.pid in {proc.pid for proc in found}


def test_survivors_excludes_zombies(group):
    """After killpg the leader lingers as a zombie -- already dead, awaiting
    reap, with an empty cmdline. Counting it would make the console decide a
    recording is still running forever, and never unlock."""
    os.killpg(group.pid, signal.SIGKILL)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and process_group.survivors(group.pid):
        time.sleep(0.02)
    assert process_group.survivors(group.pid) == [], (
        "a zombie was counted as a live process"
    )


def test_survivors_never_includes_this_process():
    """The web server is in a group of its own, and a console that reported
    itself as a recorder to be killed would SIGKILL the server."""
    assert all(
        proc.pid != os.getpid() for proc in process_group.survivors(os.getpgrp())
    )


def test_survivors_of_an_unused_group_is_empty():
    assert process_group.survivors(ABSENT_PID) == []
    assert process_group.survivors(0) == []


def test_alive_and_start_time_agree_about_a_live_process(group):
    assert process_group.alive(group.pid) is True
    assert isinstance(process_group.start_time(group.pid), int)


def test_start_time_is_none_for_a_pid_that_names_nothing():
    assert process_group.start_time(ABSENT_PID) is None
    assert process_group.alive(ABSENT_PID) is False


def test_start_time_survives_a_command_name_containing_spaces_and_parens(tmp_path):
    """`comm` is the executable name in parentheses and may contain both, so a
    plain split() mis-indexes every field after it -- including pgrp, which is
    what decides whether a process gets killed."""
    weird = tmp_path / "my (odd) prog"
    weird.write_text("#!/bin/sh\nsleep 30\n")
    weird.chmod(0o755)
    proc = subprocess.Popen([str(weird)], start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not process_group.survivors(proc.pid):
            time.sleep(0.02)
        assert process_group.survivors(proc.pid), "the odd-named process was lost"
        assert isinstance(process_group.start_time(proc.pid), int)
    finally:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)

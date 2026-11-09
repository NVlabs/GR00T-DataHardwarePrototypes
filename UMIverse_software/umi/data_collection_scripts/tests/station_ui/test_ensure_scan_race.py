# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two requests, one scan child.

`ensure_scan` is called from a page load (`/config` auto-scan), from Refresh,
and from a second tab -- so two of them landing at once is ordinary traffic,
not a corner case. Duplicating the child is not a cosmetic bug: `jobs.start`
uses `start_new_session=True`, and an OAK open is exclusive, so the second
scanner is denied the camera the first one is holding and reports working
hardware as broken.

The window is widened deliberately rather than raced. Straight thread hammering
does not catch a check-then-act here -- measured, not assumed: a 20-thread test
caught a missing lock 0 times in 20. `_SlowJobSlot` makes the interleaving
certain instead of merely likely, exactly as `_SlowOwnerSlot` does in
`test_rig_state.py`, so any check outside the lock fails this deterministically
while every atomic implementation passes.

**No hardware.** The scan child is `python -c` throughout; `scan_argv()` is
never called.
"""

import sys
import threading
import time
from types import SimpleNamespace

import pytest

from station_ui import devices as devices_core
from station_ui.jobs import JobRegistry

# Long enough that both threads are certainly inside the check when the other
# one gets there; short enough that the file stays sub-second.
SLOW_S = 0.05
DEADLINE_S = 10.0

# A child that exits at once, for the "no scan yet" case, and one that stays up,
# for the case where a running scan must not be replaced.
INSTANT_ARGV = [sys.executable, "-c", "pass"]
SLEEPING_ARGV = [sys.executable, "-c", "import time; time.sleep(5)"]


class _SlowJobSlot(devices_core.DevicesState):
    """A devices state whose job slot takes a visible moment to read and write.

    Reading is what `ensure_scan` checks and writing is what it does about it,
    so slowing both is what turns "A and B could interleave" into "A and B
    do interleave". Nothing about the mechanism is constrained -- only that the
    check and the spawn cannot be pulled apart.
    """

    _SLOW_S = SLOW_S

    @property
    def job_id(self):
        time.sleep(self._SLOW_S)
        return self.__dict__.get("_job_id")

    @job_id.setter
    def job_id(self, value):
        time.sleep(self._SLOW_S)
        self.__dict__["_job_id"] = value


class _Router:
    """`registry_for` attaches a shutdown handler to whatever it is given."""

    def add_event_handler(self, *args, **kwargs) -> None:
        pass


def _app(state=None):
    """The smallest thing `ensure_scan` will accept: `.state` and `.router`."""
    return SimpleNamespace(
        state=SimpleNamespace(devices=state or _SlowJobSlot(), jobs=JobRegistry()),
        router=_Router(),
    )


@pytest.fixture
def app():
    built = _app()
    yield built
    built.state.jobs.shutdown()


def _race(app, argv, *, force=False, threads=2):
    """Run `ensure_scan` on *threads* threads released together.

    Returns the `(job, started)` pairs in completion-agnostic order.
    """
    barrier = threading.Barrier(threads)
    results = []
    guard = threading.Lock()

    def go():
        barrier.wait()
        outcome = devices_core.ensure_scan(
            app, argv, cwd=".", timeout=DEADLINE_S, label="fake scan", force=force
        )
        with guard:
            results.append(outcome)

    workers = [threading.Thread(target=go) for _ in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(DEADLINE_S)
    assert not any(worker.is_alive() for worker in workers), "ensure_scan deadlocked"
    return results


def _await_finished(job):
    deadline = time.monotonic() + DEADLINE_S
    while not job.finished and time.monotonic() < deadline:
        time.sleep(0.01)
    assert job.finished, "the fake scan child never exited"


def test_two_simultaneous_requests_start_exactly_one_scan(app):
    """Check-then-act must not let both callers past `current_job`."""
    results = _race(app, INSTANT_ARGV)

    spawned = app.state.jobs.all()
    assert len(spawned) == 1, (
        f"{len(spawned)} scan children for one scan — two of them fight over the "
        "same exclusive OAK connection and the loser blames a USB cable"
    )
    assert [started for _, started in results].count(True) == 1
    # And the caller that did not start it is handed the one that is running,
    # not None: the page it renders has to be able to stream that child.
    assert {job.id for job, _ in results} == {spawned[0].id}


def test_a_forced_refresh_racing_itself_still_starts_one_scan(app):
    """`force=True` is Refresh, and it is the path `/config` auto-scan uses.

    It skips the "already exists" arm for a finished job, so it re-enters the
    same window from the other side: two Refresh clicks on one finished scan
    must still leave one child, because the replacement is running by the time
    the second caller looks.
    """
    first, started = devices_core.ensure_scan(
        app, INSTANT_ARGV, cwd=".", timeout=DEADLINE_S, label="fake scan"
    )
    assert started
    _await_finished(first)

    _race(app, SLEEPING_ARGV, force=True)

    spawned = [job for job in app.state.jobs.all() if job.id != first.id]
    assert len(spawned) == 1, f"{len(spawned)} children for one Refresh"


def test_a_running_scan_is_never_replaced_even_when_forced(app):
    """Not a race -- the invariant the race would break, held on one thread."""
    running, started = devices_core.ensure_scan(
        app, SLEEPING_ARGV, cwd=".", timeout=DEADLINE_S, label="fake scan"
    )
    assert started

    again, started_again = devices_core.ensure_scan(
        app, SLEEPING_ARGV, cwd=".", timeout=DEADLINE_S, label="fake scan", force=True
    )
    assert (again.id, started_again) == (running.id, False)
    assert len(app.state.jobs.all()) == 1


def test_the_slot_is_actually_slow_enough_to_interleave():
    """Guards the guard: a `_SlowJobSlot` that stopped sleeping -- because
    `DevicesState` grew `__slots__`, say, or `job_id` was renamed -- would make
    every test above pass against the unfixed code and say nothing."""
    state = _SlowJobSlot()
    start = time.monotonic()
    state.job_id = "x"
    assert state.job_id == "x"
    assert time.monotonic() - start >= 2 * SLOW_S

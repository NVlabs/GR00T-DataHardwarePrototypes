# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stopping mid data-check must not delete the episode the operator just saved.

`w` does not finalize an episode. It writes `episode_N.pending/` and hands the
data check to a subprocess; only a passed verdict makes the main loop rename
that folder to `episode_N/`. Between the two there is a window -- seconds to
minutes on a large episode -- where a *saved* episode exists only as a pending
folder.

An unconditional `discard_staging()` in `_run_session`'s `finally` (that is
`shutil.rmtree`) would delete, on Stop in that window, an episode the console
had already reported as saved, and `check_executor.shutdown(wait=False)` would
abandon the check that would have kept it. End of shift -- press `w`, walk to
the browser, press Stop -- is exactly that sequence.

The rule these tests pin: on the shutdown path an episode that reached staging
is finalized or quarantined, never deleted. Quarantine is recoverable and
rmtree is not, and unlike the in-loop handler there is no operator watching a
red banner who could act on a check error.

Discarding the episode that was still *being recorded* stays correct and is not
what these tests are about -- an interrupted demonstration is not a valid one.
"""

from __future__ import annotations

import inspect
from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest

from real_scripts.data_collection import record_data
from umi.common.recording.constants import PENDING_SUFFIX

EPISODE = 12
EPISODE_KEY = f"episode_{EPISODE}{PENDING_SUFFIX}"


def _report(passed: bool, issues=()):
    return {"episodes": {EPISODE_KEY: {"passed": passed, "issues": list(issues)}}}


def _settled(future, manager=None):
    """Run the shutdown settle step against a stub manager and return it."""
    manager = manager or MagicMock()
    record_data._settle_pending_check(manager, future, EPISODE)
    return manager


@pytest.fixture(autouse=True)
def _no_real_waiting(monkeypatch):
    """Keep the timeout path instant. The real values are minutes by design."""
    monkeypatch.setattr(record_data, "SHUTDOWN_CHECK_WAIT_S", 0.05)
    monkeypatch.setattr(record_data, "SHUTDOWN_CHECK_HEARTBEAT_S", 0.01)


def test_no_pending_check_settles_nothing():
    """The ordinary stop: no episode was saved, so the discard below owns it."""
    manager = _settled(None)
    manager.finalize_episode.assert_not_called()
    manager.quarantine_staging.assert_not_called()


def test_a_check_that_passed_finalizes_the_episode():
    future = Future()
    future.set_result(_report(passed=True))

    manager = _settled(future)

    manager.finalize_episode.assert_called_once()
    manager.quarantine_staging.assert_not_called()


def test_a_check_that_failed_quarantines_with_its_issues():
    future = Future()
    future.set_result(_report(passed=False, issues=["gripper gap", "dropped frames"]))

    manager = _settled(future)

    manager.finalize_episode.assert_not_called()
    manager.quarantine_staging.assert_called_once()
    assert manager.quarantine_staging.call_args.kwargs["issues"] == [
        "gripper gap",
        "dropped frames",
    ]


def test_a_check_that_never_finishes_quarantines_rather_than_deleting():
    """The wait is bounded -- being SIGKILLed while waiting helps nobody -- but
    running out of patience must not turn into running out of episode."""
    manager = _settled(Future())  # never resolved

    manager.finalize_episode.assert_not_called()
    manager.quarantine_staging.assert_called_once()
    issues = manager.quarantine_staging.call_args.kwargs["issues"]
    assert len(issues) == 1 and "did not finish" in issues[0], issues


def test_a_check_that_errored_quarantines_rather_than_deleting():
    """Diverges from the in-loop handler, deliberately. There a check error
    discards because an operator sees a red banner and can act; on the shutdown
    path nobody is watching, so the unverified episode is kept for review."""
    future = Future()
    future.set_exception(RuntimeError("zarr store unreadable"))

    manager = _settled(future)

    manager.finalize_episode.assert_not_called()
    manager.quarantine_staging.assert_called_once()
    assert (
        "zarr store unreadable"
        in manager.quarantine_staging.call_args.kwargs["issues"][0]
    )


def test_a_verdict_for_some_other_episode_quarantines_rather_than_deleting():
    """A report that does not mention this episode is not a pass. The poll must
    not raise `KeyError` on it, and the episode must still be preserved."""
    future = Future()
    future.set_result({"episodes": {"episode_999": {"passed": True, "issues": []}}})

    manager = _settled(future)

    manager.finalize_episode.assert_not_called()
    manager.quarantine_staging.assert_called_once()


def test_a_pass_that_cannot_be_finalized_falls_through_to_quarantine():
    """`finalize_episode` returning False leaves the folder in staging, where
    the `discard_staging()` two lines later would rmtree it."""
    manager = MagicMock()
    manager.finalize_episode.return_value = False
    future = Future()
    future.set_result(_report(passed=True))

    _settled(future, manager)

    manager.quarantine_staging.assert_called_once()


def test_the_settle_runs_before_the_discard_in_the_shutdown_path():
    """The ordering is the point.

    Every test above exercises the helper in isolation, and all of them keep
    passing if the call is dropped from `_run_session` or lands after the
    discard -- by which point the folder is already gone. Nothing else can
    observe that ordering without standing up a full recording session, so it is
    pinned against the source.
    """
    source = inspect.getsource(record_data._run_session)
    # Inspect the outer shutdown finally block, not discard_staging in a
    # nested runtime error handler.
    shutdown = source[source.rindex("\n    finally:") :]
    settle = shutdown.index("_settle_pending_check(")
    discard = shutdown.index("recorder_manager.discard_staging()")
    assert settle < discard, (
        "discard_staging() runs before the pending data check is settled, so a "
        "`w`-saved episode is rmtree'd while its verdict is still in flight"
    )


def test_the_shutdown_wait_stays_under_the_console_stall_threshold():
    """The console SIGKILLs a recorder that prints nothing for `STALL_S`.

    Waiting longer than that for a verdict would get the wait itself hard-killed
    -- losing the episode in a new way -- so the bound and the heartbeat are
    pinned to the console's number rather than chosen independently.
    """
    from station_ui.routes.record import STALL_S

    assert record_data.SHUTDOWN_CHECK_WAIT_S < STALL_S
    assert record_data.SHUTDOWN_CHECK_HEARTBEAT_S < STALL_S

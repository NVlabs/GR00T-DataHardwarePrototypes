# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RateKeeper behaviour beyond the `with` contract in test_rate_keeper_skip.py:
the `deadline` replay loops use as target_time, and verbose overrun warnings.

Driven by a fake clock, like the contract tests, so the assertions are exact and
cannot flake on a loaded host.
"""

import re

import pytest

import umi.common.utility.rate_keeper as rate_keeper_module
from umi.common.utility.rate_keeper import RateKeeper


class _FakeClock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = _FakeClock()
    monkeypatch.setattr(rate_keeper_module, "time", fake)
    return fake


def test_deadline_is_grid_slot_on_schedule_and_never_stale_after_stall(clock):
    period = 0.05
    rk = RateKeeper(1.0 / period)
    for i in range(6):
        with rk:
            if i < 3:
                # Equals what a replay loop would compute from its index.
                assert rk.deadline == rk.start_time + (i + 1) * rk.dt
            elif i == 3:
                clock.now += 2.5 * period  # iteration 3 ends at slot 5.5
            else:
                # Next future slot on the same grid, never in the past.
                assert rk.deadline > clock.now
                assert rk.deadline == rk.start_time + (i + 2) * rk.dt


def _plain(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text)  # drop color codes


def _run(rk, clock, body_times):
    for body in body_times:
        with rk:
            clock.now += body  # simulated work inside the loop body


def test_verbose_warns_on_overrun_with_skipped_ticks(clock, capsys):
    period = 0.05
    rk = RateKeeper(1.0 / period, verbose=True, name="Replay")
    # Body takes 2.4 periods: ends past slot 2, so the slot-2 deadline is skipped.
    _run(rk, clock, [2.4 * period])
    out = _plain(capsys.readouterr().out)
    assert "[Replay] running slow" in out
    assert "skipped 1 tick(s)" in out


def test_verbose_is_silent_on_schedule_and_when_not_verbose(clock, capsys):
    period = 0.05
    _run(RateKeeper(1.0 / period, verbose=True), clock, [0.2 * period] * 20)
    _run(RateKeeper(1.0 / period, verbose=False), clock, [3.0 * period] * 5)
    assert "running slow" not in capsys.readouterr().out


def test_persistent_overrun_is_reported_about_once_a_second(clock, capsys):
    period = 0.05
    rk = RateKeeper(1.0 / period, verbose=True)
    # 60 iterations of 1.5 periods each: ~4.5 s of slow loop.
    _run(rk, clock, [1.5 * period] * 60)
    reports = capsys.readouterr().out.count("running slow")
    assert 3 <= reports <= 6


def test_catch_up_repays_missed_ticks_back_to_back(clock):
    period = 0.05
    rk = RateKeeper(1.0 / period, catch_up=True)
    starts = []
    for i in range(8):
        with rk:
            starts.append(clock.now)
            if i == 1:
                clock.now += 3.5 * period  # stall past three more deadlines
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    # Catch-up mode repays missed ticks immediately, then resumes the original timing
    # grid.
    assert gaps[2] == 0.0 and gaps[3] == 0.0
    assert rk.deadline == rk.start_time + 8 * rk.dt

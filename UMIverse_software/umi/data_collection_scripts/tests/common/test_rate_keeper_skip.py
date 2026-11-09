# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for RateKeeper missed-tick handling.

- An iteration that ends a full period or more behind schedule skips the
  missed ticks: the loop never bursts back-to-back iterations to catch up.
- Overruns smaller than one period are absorbed, so the average rate holds.

Driven by a fake clock (the keeper's own sleeps advance it), so the timing is
exact and cannot flake on a loaded CI runner.
"""

import numpy as np
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


def test_stall_skips_missed_ticks_instead_of_bursting(clock):
    period = 0.05
    stall_at = 5
    rk = RateKeeper(1.0 / period)
    starts = []
    for i in range(stall_at + 1 + 8):
        with rk:
            starts.append(clock.monotonic())
            if i == stall_at:
                # >= 5 periods behind; the fractional .2 keeps a grid-aligned
                # restart well clear of the 0.5-period bound.
                clock.sleep(5.2 * period)

    gaps = np.diff(starts[stall_at + 1 :])
    assert gaps.min() >= 0.5 * period, (
        f"burst after stall: gaps (ms) = {np.round(gaps * 1000, 1).tolist()}"
    )


def test_small_overruns_are_absorbed_average_rate_holds(clock):
    # Sub-period overruns must not shift the timing grid.
    period = 0.02
    rk = RateKeeper(1.0 / period)
    starts = []
    for i in range(60):
        with rk:
            starts.append(clock.monotonic())
            if i % 4 == 3:
                clock.sleep(1.4 * period)  # ends 0.4 period behind

    rate = (len(starts) - 1) / (starts[-1] - starts[0])
    target = 1.0 / period
    assert abs(rate - target) / target <= 0.05, f"rate {rate:.1f} Hz vs {target} Hz"

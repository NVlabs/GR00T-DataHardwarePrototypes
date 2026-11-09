# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time

from umi.common.utility.print_utils import print_yellow


class RateKeeper:
    def __init__(
        self, target_rate, verbose=False, sleep_alpha=0.0, name=None, catch_up=False
    ):
        """
        Args:
            target_rate (float): The desired frequency (Hz/FPS).
            verbose (bool): Warn (at most once a second) when iterations run past
                their deadline, with the number of ticks skipped as a result.
            sleep_alpha (float): Learning rate for sleep overhead (0.0 to 1.0).
                               Set to 0.0 to disable adaptive learning (default).
            name (str): Optional name for this rate keeper (useful for debugging).
            catch_up (bool): Repay missed ticks with back-to-back iterations instead
                of skipping them. Only for loops whose body blocks until new data
                arrives (recorders): catching up then records the frames that
                arrived after a stall, at most at the source rate.
        """
        self.target_rate = target_rate
        self.dt = 1.0 / target_rate
        self.verbose = verbose
        self.name = name
        self.catch_up = catch_up

        self.start_time = time.monotonic()
        self.iter = 0

        # Adaptive sleep compensation
        self.sleep_overhead = 0.001  # Initial guess: 1ms overhead
        self.alpha = (
            sleep_alpha  # How fast the OS sleep overhead is learned (0.0 = no learning)
        )

        # Actual rate tracking (EMA-based, thread-safe)
        self._last_timestamp = None
        self._actual_rate = 0.0
        # Unsmoothened dt of the latest tick exposes individual stalls, unlike the
        # exponential moving average used for actual_rate.
        self._rate_lock = threading.Lock()

        # Overrun warnings (verbose), aggregated between reports
        self._overruns = 0
        self._worst_overrun = 0.0
        self._skipped_ticks = 0
        self._last_overrun_report = float("-inf")

    def __enter__(self):
        # Preserve start_time across context-manager entries to maintain a fixed timing
        # grid.

        self.deadline = self.start_time + (self.iter + 1) * self.dt
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        now = time.monotonic()
        remaining_time = self.deadline - now

        if remaining_time > 0:
            adjusted_sleep = max(0.0, remaining_time - self.sleep_overhead)

            if adjusted_sleep > 0:
                t0 = time.monotonic()
                time.sleep(adjusted_sleep)
                t1 = time.monotonic()

                actual_sleep_duration = t1 - t0
                sleep_error = actual_sleep_duration - adjusted_sleep

                self.sleep_overhead = (1 - self.alpha) * self.sleep_overhead + (
                    self.alpha * sleep_error
                )

        now = time.monotonic()
        skipped = 0
        if not self.catch_up and now - self.deadline >= self.dt:
            # A full period or more behind: skip the missed ticks, re-anchoring
            # to the next future slot (same phase), instead of bursting
            # back-to-back iterations to repay them. Smaller lag is absorbed.
            next_iter = int((now - self.start_time) // self.dt)
            skipped = next_iter - self.iter - 1
            self.iter = next_iter
        else:
            self.iter += 1

        with self._rate_lock:
            if self._last_timestamp is not None:
                dt = now - self._last_timestamp
                if dt > 0:
                    instant_rate = 1.0 / dt
                    if self._actual_rate == 0.0:
                        self._actual_rate = instant_rate
                    else:
                        self._actual_rate = 0.9 * self._actual_rate + 0.1 * instant_rate
            self._last_timestamp = now

        # Only the body itself running past the deadline counts; sleep jitter
        # (waking up late) does not.
        if self.verbose and remaining_time <= 0:
            self._report_overrun(now - self.deadline, skipped, now)

    def _report_overrun(self, overrun, skipped, now):
        # Aggregate so a persistently slow loop prints once a second, not per
        # iteration (printing every tick would slow it down further).
        self._overruns += 1
        self._worst_overrun = max(self._worst_overrun, overrun)
        self._skipped_ticks += skipped
        if now - self._last_overrun_report < 1.0:
            return
        name_prefix = f"[{self.name}] " if self.name else ""
        print_yellow(
            f"{name_prefix}running slow: {self._overruns} iteration(s) past the "
            f"{self.dt * 1000:.1f} ms deadline (worst +{self._worst_overrun * 1000:.1f} ms), "
            f"skipped {self._skipped_ticks} tick(s)"
        )
        self._overruns = 0
        self._worst_overrun = 0.0
        self._skipped_ticks = 0
        self._last_overrun_report = now

    @property
    def actual_rate(self):
        """Returns the actual rate (Hz) based on recent iterations. Thread-safe."""
        with self._rate_lock:
            return self._actual_rate

    def reset(self):
        """Resets the internal clock and counters. Useful if you want to resync."""
        self.start_time = time.monotonic()
        self.iter = 0
        self._last_timestamp = None
        self._actual_rate = 0.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time


class RateKeeper:
    def __init__(self, target_rate, verbose=False, sleep_alpha=0.0, name=None):
        """
        Args:
            target_rate (float): The desired frequency (Hz/FPS).
            verbose (bool): Whether to print debug stats.
            sleep_alpha (float): Learning rate for sleep overhead (0.0 to 1.0).
                               Set to 0.0 to disable adaptive learning (default).
            name (str): Optional name for this rate keeper (useful for debugging).
        """
        self.target_rate = target_rate
        self.dt = 1.0 / target_rate
        self.verbose = verbose
        self.name = name

        self.start_time = time.monotonic()
        self.iter = 0

        # Adaptive sleep compensation
        self.sleep_overhead = 0.001  # Initial guess: 1ms overhead
        self.alpha = (
            sleep_alpha  # How fast we learn the OS sleep overhead (0.0 = no learning)
        )

        # Actual rate tracking (EMA-based, thread-safe), shown in verbose stats
        self._last_timestamp = None
        self._actual_rate = 0.0
        self._rate_lock = threading.Lock()

    def ready(self) -> bool:
        now = time.monotonic()
        deadline = self.start_time + (self.iter * self.dt)
        return now >= deadline

    def tick(self):
        """
        Advance the clock.
        This moves the deadline forward to the next slot.
        """
        now = time.monotonic()

        self.iter += 1

        with self._rate_lock:
            if self._last_timestamp is not None:
                duration = now - self._last_timestamp
                if duration > 0:
                    instant_rate = 1.0 / duration
                    if self._actual_rate == 0.0:
                        self._actual_rate = instant_rate
                    else:
                        self._actual_rate = 0.9 * self._actual_rate + 0.1 * instant_rate
            self._last_timestamp = now

        if self.verbose:
            deadline = self.start_time + self.iter * self.dt
            self._print_stats(now - deadline)

    def __enter__(self):
        # Preserve start_time across context-manager entries to maintain a fixed timing
        # grid.
        self.frame_start_time = time.monotonic()

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

        self.iter += 1

        now = time.monotonic()
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

        if self.verbose:
            self._print_stats(remaining_time)

    def _print_stats(self, remaining_time):
        """Helper to print verbose stats without cluttering __exit__"""
        frame_dur = time.monotonic() - self.frame_start_time
        fps = 1.0 / frame_dur if frame_dur > 0 else 0
        status = "Sleep" if remaining_time > 0 else "Lag"
        name_prefix = f"[{self.name}] " if self.name else ""
        print(
            f"{name_prefix}Iter: {self.iter} | "
            f"FPS: {fps:.1f} | "
            f"Actual: {self._actual_rate:.1f}Hz | "
            f"{status}: {abs(remaining_time):.4f}s | "
            f"Overhead: {self.sleep_overhead * 1000:.2f}ms"
        )

    def reset(self):
        """Resets the internal clock and counters. Useful if you want to resync."""
        self.start_time = time.monotonic()
        self.iter = 0
        self._last_timestamp = None
        self._actual_rate = 0.0

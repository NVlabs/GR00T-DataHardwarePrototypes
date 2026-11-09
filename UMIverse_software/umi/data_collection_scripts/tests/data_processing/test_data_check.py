# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for data_check.py validation functions.

Run with: uv run pytest tests/data_processing/test_data_check.py -v
"""

import os
import subprocess
import tempfile

import numpy as np
import yaml
import zarr

from real_scripts.data_processing.data_check import (
    check_config_device_names,
    check_consecutive_frame_intervals,
    check_consecutive_movement,
    check_device_name_consistency,
    check_fps_consistency,
    check_gripper_jumps,
    check_gripper_movement,
    check_gripper_range,
    check_gripper_serial_consistency,
    check_monotonic_timestamps,
    check_pose_changes,
    check_sustained_frame_drop,
    check_timestamp_coverage,
    check_tracking_pose_jumps,
    check_video_resolution,
    extract_episode_index,
    get_device_expected_fps,
    get_device_expected_resolutions,
    get_device_timestamps,
    get_latency_adjusted_timestamps,
    load_check_config,
)

# =============================================================================
# Timestamp Validation Tests
# =============================================================================


class TestCheckFpsConsistency:
    """Tests for check_fps_consistency function."""

    def test_exact_fps_passes(self):
        """Timestamps at exactly expected FPS should pass."""
        expected_fps = 45.0
        num_frames = 100
        # Generate timestamps at exactly 45 FPS
        timestamps = np.arange(num_frames) / expected_fps

        result = check_fps_consistency(timestamps, expected_fps, max_deviation_pct=10.0)

        assert result["passed"]
        assert abs(result["avg_fps"] - expected_fps) < 0.1
        assert result["deviation_pct"] < 1.0

    def test_within_tolerance_passes(self):
        """Timestamps slightly off expected FPS (within 10%) should pass."""
        expected_fps = 45.0
        actual_fps = 44.0  # ~2.2% deviation
        num_frames = 100
        timestamps = np.arange(num_frames) / actual_fps

        result = check_fps_consistency(timestamps, expected_fps, max_deviation_pct=10.0)

        assert result["passed"]
        assert abs(result["avg_fps"] - actual_fps) < 0.1

    def test_fps_too_low_fails(self):
        """Timestamps at much lower FPS should fail."""
        expected_fps = 45.0
        actual_fps = 30.0  # 33% deviation
        num_frames = 100
        timestamps = np.arange(num_frames) / actual_fps

        result = check_fps_consistency(timestamps, expected_fps, max_deviation_pct=10.0)

        assert not result["passed"]
        assert result["deviation_pct"] > 10.0

    def test_fps_too_high_fails(self):
        """Timestamps at much higher FPS should fail."""
        expected_fps = 45.0
        actual_fps = 60.0  # 33% deviation
        num_frames = 100
        timestamps = np.arange(num_frames) / actual_fps

        result = check_fps_consistency(timestamps, expected_fps, max_deviation_pct=10.0)

        assert not result["passed"]
        assert result["deviation_pct"] > 10.0

    def test_single_timestamp_fails(self):
        """Single timestamp should fail (not enough data)."""
        timestamps = np.array([0.0])

        result = check_fps_consistency(timestamps, expected_fps=45.0)

        assert not result["passed"]
        assert "error" in result

    def test_zero_duration_fails(self):
        """Two identical timestamps (zero duration) should fail."""
        timestamps = np.array([1.0, 1.0])

        result = check_fps_consistency(timestamps, expected_fps=45.0)

        assert not result["passed"]
        assert "error" in result


class TestCheckMonotonicTimestamps:
    """Tests for check_monotonic_timestamps function."""

    def test_strictly_increasing_passes(self):
        """Strictly increasing timestamps should pass."""
        timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        result = check_monotonic_timestamps(timestamps)

        assert result["monotonic"]
        assert result["num_violations"] == 0
        assert len(result["non_monotonic_indices"]) == 0

    def test_duplicate_timestamp_fails(self):
        """Duplicate timestamp should fail with 1 violation."""
        timestamps = np.array([0.0, 1.0, 2.0, 2.0, 3.0])

        result = check_monotonic_timestamps(timestamps)

        assert not result["monotonic"]
        assert result["num_violations"] == 1
        assert 2 in result["non_monotonic_indices"]

    def test_decreasing_timestamp_fails(self):
        """Decreasing timestamp should fail."""
        timestamps = np.array([0.0, 1.0, 3.0, 2.0, 4.0])

        result = check_monotonic_timestamps(timestamps)

        assert not result["monotonic"]
        assert result["num_violations"] == 1

    def test_multiple_violations_detected(self):
        """Multiple violations should all be detected."""
        timestamps = np.array([0.0, 1.0, 1.0, 3.0, 2.0])  # 2 violations

        result = check_monotonic_timestamps(timestamps)

        assert not result["monotonic"]
        assert result["num_violations"] == 2

    def test_single_element_passes(self):
        """Single element should pass (nothing to compare)."""
        timestamps = np.array([0.0])

        result = check_monotonic_timestamps(timestamps)

        assert result["monotonic"]


class TestCheckTimestampCoverage:
    """Tests for check_timestamp_coverage function."""

    def test_identical_ranges_passes(self):
        """Identical timestamp ranges should pass with ratio=1.0."""
        device_timestamps = {
            "device_a": np.array([0.0, 5.0, 10.0]),
            "device_b": np.array([0.0, 5.0, 10.0]),
        }

        result = check_timestamp_coverage(device_timestamps, threshold=0.95)

        assert result["passed"]
        assert result["overlap_ratio"] == 1.0

    def test_overlapping_ranges_passes(self):
        """Slightly offset ranges with good overlap should pass."""
        device_timestamps = {
            "device_a": np.array([0.0, 5.0, 10.0]),
            "device_b": np.array([0.2, 5.2, 10.2]),
        }

        result = check_timestamp_coverage(device_timestamps, threshold=0.95)

        # intersection = [0.2, 10.0] = 9.8, span = [0, 10.2] = 10.2
        # ratio = 9.8 / 10.2 ≈ 0.9608
        assert result["passed"]
        assert result["overlap_ratio"] >= 0.95 and result["overlap_ratio"] <= 0.97

    def test_partial_overlap_fails(self):
        """Poor overlap should fail."""
        device_timestamps = {
            "device_a": np.array([0.0, 5.0, 10.0]),  # range: 0-10
            "device_b": np.array([5.0, 10.0, 15.0]),  # range: 5-15
        }

        result = check_timestamp_coverage(device_timestamps, threshold=0.35)

        # intersection = [5, 10] = 5, span = [0, 15] = 15
        # ratio = 5 / 15 ≈ 0.333
        assert not result["passed"]
        assert result["overlap_ratio"] < 0.34 and result["overlap_ratio"] > 0.33

    def test_single_device_passes(self):
        """Single device should pass (no comparison needed)."""
        device_timestamps = {
            "device_a": np.array([0.0, 5.0, 10.0]),
        }

        result = check_timestamp_coverage(device_timestamps, threshold=0.95)

        assert result["passed"]

    def test_empty_dict_passes(self):
        """Empty device dict should pass."""
        result = check_timestamp_coverage({}, threshold=0.95)

        assert result["passed"]


class TestCheckConsecutiveFrameIntervals:
    """Tests for check_consecutive_frame_intervals function."""

    def test_consistent_intervals_passes(self):
        """Uniform intervals at expected FPS should pass."""
        expected_fps = 45.0
        num_frames = 100
        timestamps = np.arange(num_frames) / expected_fps

        result = check_consecutive_frame_intervals(
            timestamps, expected_fps, gap_min_frames=3.0
        )

        assert result["passed"]
        assert result["num_violations"] == 0

    def test_small_gap_passes(self):
        """Small gap (2x expected dt) should pass with threshold of 3x."""
        expected_fps = 45.0
        dt = 1.0 / expected_fps
        # Create timestamps with one 2x gap
        timestamps = np.array([0.0, dt, 2 * dt, 4 * dt, 5 * dt])  # gap at index 2->3

        result = check_consecutive_frame_intervals(
            timestamps, expected_fps, gap_min_frames=3.0
        )

        assert result["passed"]

    def test_large_gap_fails(self):
        """Large gap (5x expected dt) should fail with threshold of 3x."""
        expected_fps = 45.0
        dt = 1.0 / expected_fps
        # Create timestamps with one 5x gap
        timestamps = np.array([0.0, dt, 2 * dt, 7 * dt, 8 * dt])  # 5x gap at index 2->3

        result = check_consecutive_frame_intervals(
            timestamps, expected_fps, gap_min_frames=3.0
        )

        assert not result["passed"]
        assert result["num_violations"] >= 1

    def test_multiple_gaps_detected(self):
        """Multiple large gaps should all be detected."""
        expected_fps = 45.0
        dt = 1.0 / expected_fps
        # Create timestamps with two 5x gaps
        timestamps = np.array([0.0, 5 * dt, 6 * dt, 11 * dt])  # gaps at 0->1 and 2->3

        result = check_consecutive_frame_intervals(
            timestamps, expected_fps, gap_min_frames=3.0
        )

        assert not result["passed"]
        assert result["num_violations"] == 2

    # --- detection vs verdict ---------------------------------------------
    # A flagged interval is not by itself a failed episode. Each failure rule
    # must fire on its own, and none may change what is detected.

    @staticmethod
    def _stream(fps, duration_s, gaps_s):
        """A steady stream at `fps` with each gap in `gaps_s` spliced in."""
        dt = 1.0 / fps
        n = int(duration_s * fps)
        ts, gaps = [0.0], list(gaps_s)
        for i in range(1, n):
            ts.append(
                ts[-1]
                + (gaps.pop(0) if gaps and i % (n // (len(gaps_s) + 1)) == 0 else dt)
            )
        return np.array(ts)

    def test_one_short_gap_in_a_long_episode_is_reported_but_does_not_fail(self):
        """A single 233 ms hole in a 60 s episode.

        It is detected and carries its duration into the report, but on its own
        it is not a failure.
        """
        ts = self._stream(30.0, 60.0, [0.233])
        r = check_consecutive_frame_intervals(ts, 30.0, gap_min_frames=4.0)

        assert r["num_violations"] == 1, "the gap must still be DETECTED"
        assert r["passed"], "0.33% of an episode is not a failed episode"
        assert r["missing_pct"] < 1.0
        assert 0.2 < r["longest_gap_s"] < 0.3

    def test_a_very_long_gap_fails_on_the_duration_backstop(self):
        """Past a long enough hole, duration has to take over.

        The aggregate rule is disabled here on purpose so that the backstop is
        the only thing left that can fail this episode -- otherwise 2.5 s out of
        60 trips the 1% limit first and the test proves nothing about the
        backstop. No positions either, so there is no recoverability estimate.
        """
        ts = self._stream(30.0, 60.0, [2.5])
        r = check_consecutive_frame_intervals(
            ts, 30.0, gap_min_frames=4.0, gap_fail_max_missing_pct=99.0
        )

        assert r["longest_gap_s"] >= 2.0
        assert not r["passed"]

        # ... and the same hole passes when the backstop is lifted, which is
        # what shows the backstop is what fired.
        loose = check_consecutive_frame_intervals(
            ts,
            30.0,
            gap_min_frames=4.0,
            gap_fail_max_missing_pct=99.0,
            gap_fail_max_frames=999.0,
        )
        assert loose["passed"]

    # --- recoverability, the rule that actually decides ---------------------
    # Same gap, same duration, opposite verdicts -- decided only by whether
    # interpolation can put the motion back.

    @staticmethod
    def _positions(ts, turn_at=None, speed=0.02):
        """Straight-line motion at a realistic gripper speed (about 0.02 m/s
        across gaps in recorded data), optionally reversing at `turn_at`."""
        pos = np.zeros((len(ts), 3))
        for i, t in enumerate(ts):
            if turn_at is None or t < turn_at:
                pos[i, 0] = speed * t
            else:
                pos[i, 0] = speed * turn_at - speed * (t - turn_at)
        return pos

    def test_a_gap_through_steady_motion_is_recoverable_and_passes(self):
        """Constant velocity: a straight line interpolates back exactly."""
        ts = self._stream(30.0, 60.0, [0.233])
        r = check_consecutive_frame_intervals(
            ts, 30.0, gap_min_frames=4.0, positions=self._positions(ts)
        )

        assert r["num_violations"] == 1, "still DETECTED"
        assert r["interp_error_checked"]
        assert r["max_est_interp_error_mm"] < 1.0
        assert r["passed"]

    def test_the_same_gap_through_a_direction_change_fails(self):
        """Same 233 ms hole, but the gripper reverses inside it.

        Nothing about the duration changed -- only whether the poses in the hole
        are inferable from the ones around it, which is the whole question.
        """
        ts = self._stream(30.0, 60.0, [0.233])
        gap_start = ts[np.argmax(np.diff(ts))]
        r = check_consecutive_frame_intervals(
            ts,
            30.0,
            gap_min_frames=4.0,
            # About the 90th-percentile gripper speed in recorded data; a
            # reversal at 0.02 m/s is genuinely recoverable and should not fail.
            positions=self._positions(ts, turn_at=gap_start + 0.11, speed=0.15),
        )

        assert r["num_violations"] == 1, "the same single gap"
        assert r["longest_gap_s"] < 10 / 30.0, "duration rules would let this through"
        assert r["missing_pct"] < 1.0
        assert r["max_est_interp_error_mm"] > 5.0
        assert not r["passed"], "unrecoverable motion must fail on its own"

    def test_devices_without_positions_are_judged_on_duration_alone(self):
        """Cameras and grippers have no 3D pose; they must not be penalised."""
        ts = self._stream(30.0, 60.0, [0.233])
        r = check_consecutive_frame_intervals(ts, 30.0, gap_min_frames=4.0)

        assert not r["interp_error_checked"]
        assert r["max_est_interp_error_mm"] == 0.0
        assert r["passed"]

    def test_many_short_gaps_fail_on_the_total(self):
        """And the converse: no single gap is long, but they add up."""
        ts = self._stream(30.0, 60.0, [0.233] * 12)
        r = check_consecutive_frame_intervals(ts, 30.0, gap_min_frames=4.0)

        assert r["longest_gap_s"] < 0.5, "single-gap rule must NOT be what fires here"
        assert r["missing_pct"] > 1.0
        assert not r["passed"]

    def test_the_failure_thresholds_do_not_change_what_is_detected(self):
        """Loosening the verdict must never hide a gap from the report."""
        ts = self._stream(30.0, 60.0, [0.233] * 12)
        strict = check_consecutive_frame_intervals(
            ts, 30.0, 4.0, gap_fail_max_missing_pct=0.0, gap_fail_max_frames=0.001
        )
        loose = check_consecutive_frame_intervals(
            ts, 30.0, 4.0, gap_fail_max_missing_pct=99.0, gap_fail_max_frames=999.0
        )

        assert strict["num_violations"] == loose["num_violations"] > 0
        assert strict["violation_frames"] == loose["violation_frames"]
        assert not strict["passed"] and loose["passed"]

    def test_a_gap_of_exactly_the_limit_is_allowed(self):
        """`gap_fail_max_frames` is a maximum, so a gap of that length passes.

        Both failure rules fail only on exceeding their limit, the same
        convention as `gap_fail_max_missing_pct`.
        """
        dt = 1.0 / 30.0
        exact = self._stream(30.0, 60.0, [10 * dt])
        over = self._stream(30.0, 60.0, [10.5 * dt])

        r_exact = check_consecutive_frame_intervals(
            exact, 30.0, gap_min_frames=4.0, gap_fail_max_missing_pct=99.0
        )
        r_over = check_consecutive_frame_intervals(
            over, 30.0, gap_min_frames=4.0, gap_fail_max_missing_pct=99.0
        )

        assert r_exact["passed"], "a gap of exactly the maximum must be allowed"
        assert not r_over["passed"]

    def test_a_one_sample_device_returns_every_key(self):
        """The early return must be shape-compatible with the normal path.

        `dispatch` only formats these fields on failure and the early return
        passes, so nothing crashes today -- but a caller that logs
        `missing_pct` unconditionally would KeyError on any device with a single
        sample, and would do it in production rather than here.
        """
        full = check_consecutive_frame_intervals(
            self._stream(30.0, 60.0, [0.233]), 30.0, gap_min_frames=4.0
        )
        tiny = check_consecutive_frame_intervals(
            np.array([0.0]), 30.0, gap_min_frames=4.0
        )

        missing = set(full) - set(tiny)
        assert not missing, f"early return is missing {sorted(missing)}"
        assert tiny["passed"]

    def test_the_new_verdict_never_fails_what_the_old_one_passed(self):
        """The verdict only ever relaxes the "no flagged interval" rule.

        Compared with `passed == (num_violations == 0)`: when there are no
        violations there are no gaps, so total missing time, longest gap and
        worst interpolation error are all 0 -- and 0 clears every non-negative
        limit. So an episode with no flagged interval always passes.

        That argument is one refactor away from being wrong (a rule keyed on
        something other than the gap list, say a raw FPS or jitter term, would
        break it silently and start quarantining good data). Pinned by fuzzing
        rather than by re-stating the argument.
        """
        rng = np.random.default_rng(7)
        dt = 1 / 30.0
        checked = relaxed = 0
        for _ in range(3000):
            # Episode-length streams, not two-second ones: a single gap in a
            # 2 s stream is 10% of it and always trips the volume rule, so a
            # short fuzzer would only ever exercise the failing side.
            n = int(rng.integers(2, 2000))
            gaps = np.where(
                rng.random(n - 1) < 0.004, dt * rng.uniform(1, 15, n - 1), dt
            )
            ts = np.concatenate([[0.0], np.cumsum(gaps)])
            pos = np.cumsum(rng.normal(0, 0.004, size=(len(ts), 3)), axis=0)
            r = check_consecutive_frame_intervals(
                ts, 30.0, 4.0, positions=pos if rng.random() < 0.5 else None
            )
            checked += 1
            old_passed = r["num_violations"] == 0
            if old_passed:
                assert r["passed"], (
                    "an episode with no detected gaps must still pass: "
                    f"missing_pct={r['missing_pct']} longest={r['longest_gap_s']} "
                    f"est={r['max_est_interp_error_mm']}"
                )
            elif r["passed"]:
                relaxed += 1
        assert checked == 3000
        assert relaxed > 0, (
            "fuzzer never produced a relaxed case; it is not exercising the rules"
        )

    def test_single_timestamp_passes(self):
        """Single timestamp should pass (no intervals to check)."""
        timestamps = np.array([0.0])

        result = check_consecutive_frame_intervals(
            timestamps, expected_fps=45.0, gap_min_frames=3.0
        )

        assert result["passed"]
        assert "error" in result


# =============================================================================
# Per-Device Expected FPS Tests
# =============================================================================


class TestGetDeviceExpectedFps:
    """Tests for get_device_expected_fps helper."""

    def test_uses_per_source_fps(self):
        """Each source's record_fps should override the global default."""
        config = {
            "recording": {"record_fps": 30},
            "video_sources": [
                {"name": "cam_a", "record_fps": 60},
                {"name": "cam_b"},  # no override
            ],
            "numeric_sources": [
                {"name": "gripper", "record_fps": 45},
            ],
        }
        fps_map = get_device_expected_fps(config)

        assert fps_map["cam_a"] == 60
        assert fps_map["cam_b"] == 30  # falls back to global
        assert fps_map["gripper"] == 45

    def test_all_default(self):
        """When no per-source fps is set, all devices use the global default."""
        config = {
            "recording": {"record_fps": 45},
            "video_sources": [{"name": "cam_a"}, {"name": "cam_b"}],
            "numeric_sources": [{"name": "gripper"}],
        }
        fps_map = get_device_expected_fps(config)

        assert all(v == 45 for v in fps_map.values())

    def test_empty_config(self):
        """Empty config should return empty map."""
        assert get_device_expected_fps({}) == {}

    def test_missing_recording_section_uses_hardcoded_default(self):
        """If recording section is missing, uses 45 as hardcoded fallback."""
        config = {
            "video_sources": [{"name": "cam_a"}],
        }
        fps_map = get_device_expected_fps(config)

        assert fps_map["cam_a"] == 45  # hardcoded fallback


# =============================================================================
# Gripper Sensor Validation Tests
# =============================================================================


class TestCheckGripperMovement:
    """Tests for check_gripper_movement function."""

    def test_large_movement_passes(self):
        """Gripper with large movement range should pass."""
        distance = np.linspace(0.0, 50.0, 100)  # 0mm to 50mm

        result = check_gripper_movement(distance, threshold=10.0)

        assert result["moved"]
        assert result["range_mm"] == 50.0

    def test_threshold_edge_passes(self):
        """Movement just above threshold should pass."""
        distance = np.linspace(20.0, 35.0, 50)  # 15mm range > 10mm threshold

        result = check_gripper_movement(distance, threshold=10.0)

        assert result["moved"]
        assert result["range_mm"] == 15.0

    def test_no_movement_fails(self):
        """Constant distance (no movement) should fail."""
        distance = np.full(100, 30.0)  # constant 30mm

        result = check_gripper_movement(distance, threshold=10.0)

        assert not result["moved"]
        assert result["range_mm"] == 0.0

    def test_small_movement_fails(self):
        """Small movement below threshold should fail."""
        distance = np.linspace(25.0, 30.0, 50)  # 5mm range < 10mm threshold

        result = check_gripper_movement(distance, threshold=10.0)

        assert not result["moved"]
        assert result["range_mm"] == 5.0

    def test_empty_array_fails(self):
        """Empty distance array should fail."""
        distance = np.array([])

        result = check_gripper_movement(distance, threshold=10.0)

        assert not result["moved"]
        assert "error" in result


def _uniform_ts(n, fps=30.0):
    """Timestamps at a steady `fps`, so mm/frame and mm/s agree by `fps`."""
    return np.arange(n) / fps


class TestCheckGripperJumps:
    """Tests for check_gripper_jumps function."""

    def test_smooth_data_passes(self):
        """Smooth gradual changes should pass (no jumps)."""
        # Gradual changes, max diff ~10mm -> ~15 mm/s
        distance = np.linspace(10.0, 60.0, 100)

        result = check_gripper_jumps(
            distance, _uniform_ts(100), jump_threshold_mm_per_s=1500.0
        )

        assert not result["has_jumps"]
        assert result["num_jumps"] == 0

    def test_single_jump_fails(self):
        """Single large jump should be detected."""
        # 60mm jump at index 2->3, i.e. 1800 mm/s at 30 fps
        distance = np.array([10.0, 15.0, 20.0, 80.0, 85.0])

        result = check_gripper_jumps(
            distance, _uniform_ts(5), jump_threshold_mm_per_s=1500.0
        )

        assert result["has_jumps"]
        assert result["num_jumps"] == 1
        assert result["jump_frames"][0]["frame_index"] == 3

    def test_multiple_jumps_detected(self):
        """Multiple jumps should all be detected."""
        distance = np.array([10.0, 70.0, 75.0, 10.0, 15.0])  # jumps at 0->1 and 2->3

        result = check_gripper_jumps(
            distance, _uniform_ts(5), jump_threshold_mm_per_s=1500.0
        )

        assert result["has_jumps"]
        assert result["num_jumps"] == 2

    def test_single_element_passes(self):
        """Single element should pass (no consecutive pairs to check)."""
        distance = np.array([50.0])

        result = check_gripper_jumps(
            distance, _uniform_ts(1), jump_threshold_mm_per_s=1500.0
        )

        assert not result["has_jumps"]


class TestCheckGripperRange:
    """Tests for check_gripper_range function."""

    def test_all_in_range_passes(self):
        """All values within range should pass."""
        distance = np.linspace(10.0, 70.0, 100)  # within [0, 80]

        result = check_gripper_range(distance, min_val=0.0, max_val=80.0)

        assert result["in_range"]
        assert result["num_out_of_range"] == 0

    def test_below_range_fails(self):
        """Values below minimum should fail."""
        distance = np.array([10.0, 20.0, -5.0, 30.0, 40.0])  # -5 is below 0

        result = check_gripper_range(distance, min_val=0.0, max_val=80.0)

        assert not result["in_range"]
        assert result["num_out_of_range"] == 1

    def test_above_range_fails(self):
        """Values above maximum should fail."""
        distance = np.array([10.0, 20.0, 90.0, 30.0, 40.0])  # 90 is above 80

        result = check_gripper_range(distance, min_val=0.0, max_val=80.0)

        assert not result["in_range"]
        assert result["num_out_of_range"] == 1

    def test_both_violations_fails(self):
        """Values both below and above range should all be detected."""
        distance = np.array([-5.0, 20.0, 90.0, 30.0, 100.0])  # 3 violations

        result = check_gripper_range(distance, min_val=0.0, max_val=80.0)

        assert not result["in_range"]
        assert result["num_out_of_range"] == 3

    def test_default_max_accepts_a_fully_open_gripper(self):
        """The default has to agree with the configs every caller passes in.

        `dispatch` reads `range_max_mm` (90.0 in every production data-check
        config) and only falls back to this default, so a stale default is
        invisible until something calls the check bare -- and then it flags a
        healthy full-open gripper. 88.2 mm is the widest opening observed in
        recorded data."""
        distance = np.array([10.0, 60.0, 88.2])

        result = check_gripper_range(distance)

        assert result["in_range"]
        assert result["expected_range"][1] == 90.0


# =============================================================================
# iPhone Tracking Validation Tests
# =============================================================================


def _create_identity_pose():
    """Create a 4x4 identity pose matrix."""
    return np.eye(4)


def _create_translated_pose(x=0.0, y=0.0, z=0.0):
    """Create a 4x4 pose with translation."""
    pose = np.eye(4)
    pose[0, 3] = x
    pose[1, 3] = y
    pose[2, 3] = z
    return pose


def _create_rotated_pose(yaw=0.0):
    """Create a 4x4 pose rotated about z, with no translation."""
    pose = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    pose[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return pose


class TestCheckPoseChanges:
    """Tests for check_pose_changes function."""

    def test_significant_movement_passes(self):
        """Poses with significant translation should pass."""
        # Create poses moving 5cm in X direction
        poses = np.array([_create_translated_pose(x=i * 0.01) for i in range(10)])

        result = check_pose_changes(poses, threshold=0.01, downsample=1)

        assert result["has_movement"]
        assert result["max_distance"] >= 0.05  # at least 5cm movement

    def test_static_poses_fails(self):
        """All identical poses should fail."""
        poses = np.array([_create_identity_pose() for _ in range(10)])

        result = check_pose_changes(poses, threshold=0.01, downsample=1)

        assert not result["has_movement"]
        assert result["max_distance"] == 0.0

    def test_tiny_movement_fails(self):
        """Movement below threshold should fail."""
        # Create poses with only 5mm movement (below 1cm threshold)
        poses = np.array([_create_translated_pose(x=i * 0.001) for i in range(5)])

        result = check_pose_changes(poses, threshold=0.01, downsample=1)

        assert not result["has_movement"]
        assert result["max_distance"] < 0.01

    def test_empty_poses_fails(self):
        """Empty pose array should fail."""
        poses = np.zeros((0, 4, 4))

        result = check_pose_changes(poses, threshold=0.01, downsample=1)

        assert not result["has_movement"]
        assert "error" in result


class TestCheckConsecutiveMovement:
    """Tests for check_consecutive_movement function."""

    def test_continuous_movement_passes(self):
        """Poses with consistent movement each frame should pass."""
        # Each frame moves 2mm (> 0.5mm threshold)
        poses = np.array([_create_translated_pose(x=i * 0.002) for i in range(100)])

        result = check_consecutive_movement(
            poses, min_movement=0.0005, required_ratio=0.90
        )

        assert result["passed"]
        assert result["movement_ratio"] >= 0.90

    def test_mostly_static_fails(self):
        """Mostly static poses (50% moving) should fail 90% threshold."""
        # Alternate: moving frame, static frame
        poses = []
        for i in range(100):
            if i % 2 == 0:
                poses.append(_create_translated_pose(x=i * 0.001))
            else:
                poses.append(poses[-1].copy())  # duplicate previous
        poses = np.array(poses)

        result = check_consecutive_movement(
            poses, min_movement=0.0005, required_ratio=0.90
        )

        assert not result["passed"]
        assert result["movement_ratio"] < 0.90

    def test_all_static_fails(self):
        """All static poses should fail."""
        poses = np.array([_create_identity_pose() for _ in range(50)])

        result = check_consecutive_movement(
            poses, min_movement=0.0005, required_ratio=0.90
        )

        assert not result["passed"]
        assert result["movement_ratio"] == 0.0

    def test_two_poses_with_movement_passes(self):
        """Two poses with sufficient movement should pass."""
        poses = np.array(
            [
                _create_identity_pose(),
                _create_translated_pose(x=0.01),  # 1cm movement
            ]
        )

        result = check_consecutive_movement(
            poses, min_movement=0.0005, required_ratio=0.90
        )

        assert result["passed"]
        assert result["movement_ratio"] == 1.0


class TestCheckTrackingPoseJumps:
    """Tests for check_tracking_pose_jumps function."""

    def test_smooth_movement_passes(self):
        """Gradual continuous movement should pass with no jumps detected."""
        # Each frame moves 2mm (well below 100mm threshold)
        poses = np.array([_create_translated_pose(x=i * 0.002) for i in range(100)])

        result = check_tracking_pose_jumps(
            poses, _uniform_ts(len(poses)), jump_threshold_m_per_s=3.0
        )

        assert not result["has_jumps"]
        assert result["num_jumps"] == 0

    def test_single_jump_detected(self):
        """One sudden large position change should detect exactly 1 jump."""
        # Smooth movement, then a 150mm jump, then smooth again
        poses = [
            _create_translated_pose(x=0.0),
            _create_translated_pose(x=0.01),
            _create_translated_pose(x=0.02),
            _create_translated_pose(x=0.17),  # 150mm jump from previous
            _create_translated_pose(x=0.18),
        ]
        poses = np.array(poses)

        result = check_tracking_pose_jumps(
            poses, _uniform_ts(len(poses)), jump_threshold_m_per_s=3.0
        )

        assert result["has_jumps"]
        assert result["num_jumps"] == 1
        assert result["jump_frames"][0]["frame_index"] == 3

    def test_multiple_jumps_detected(self):
        """Multiple large jumps should all be detected and counted."""
        # Two jumps: at index 1 and index 3
        poses = [
            _create_translated_pose(x=0.0),
            _create_translated_pose(x=0.15),  # 150mm jump
            _create_translated_pose(x=0.16),
            _create_translated_pose(x=0.30),  # 140mm jump
            _create_translated_pose(x=0.31),
        ]
        poses = np.array(poses)

        result = check_tracking_pose_jumps(
            poses, _uniform_ts(len(poses)), jump_threshold_m_per_s=3.0
        )

        assert result["has_jumps"]
        assert result["num_jumps"] == 2

    def test_jump_below_threshold_passes(self):
        """Movement just below threshold should not trigger a jump."""
        # 90mm movement (below 100mm threshold)
        poses = [
            _create_translated_pose(x=0.0),
            _create_translated_pose(x=0.09),  # 90mm < 100mm threshold
        ]
        poses = np.array(poses)

        result = check_tracking_pose_jumps(
            poses, _uniform_ts(len(poses)), jump_threshold_m_per_s=3.0
        )

        assert not result["has_jumps"]
        assert result["num_jumps"] == 0

    def test_single_pose_passes(self):
        """Single element array should handle gracefully."""
        poses = np.array([_create_identity_pose()])

        result = check_tracking_pose_jumps(
            poses, _uniform_ts(len(poses)), jump_threshold_m_per_s=3.0
        )

        assert not result["has_jumps"]
        assert result["num_jumps"] == 0


# =============================================================================
# Gripper Serial Consistency Validation Tests
# =============================================================================


def _create_dataset_with_gripper_serials(episode_serials: list):
    """Helper to create a mock zarr dataset with gripper serial data.

    Args:
        episode_serials: List of dicts, each containing gripper serial info per episode.
            Example: [
                {"left_umi_gripper": {"device_name": "UMI-003", "serial": "ABC123"}},
                {"left_umi_gripper": {"device_name": "UMI-003", "serial": "ABC123"}},
            ]
            Use None for episodes without gripper serial data.

    Returns:
        zarr.Group representing the dataset
    """
    import zarr

    store = zarr.group()
    for idx, serial_data in enumerate(episode_serials):
        episode = store.create_group(f"episode_{idx}")
        if serial_data is not None:
            episode.attrs["gripper_serial"] = serial_data
    return store


class TestCheckGripperSerialConsistency:
    """Tests for check_gripper_serial_consistency function."""

    def test_consistent_serials_passes(self):
        """All episodes with same gripper serials should pass."""
        serial_data = {
            "left_umi_gripper": {
                "device_name": "UMI-003",
                "serial": "50443405000000B6",
            },
            "right_umi_gripper": {
                "device_name": "UMI-007",
                "serial": "50443405000000B5",
            },
        }
        data_store = _create_dataset_with_gripper_serials([serial_data] * 5)

        result = check_gripper_serial_consistency(
            data_store,
            ["episode_0", "episode_1", "episode_2", "episode_3", "episode_4"],
        )

        assert result["passed"]
        assert len(result["issues"]) == 0
        assert len(result["inconsistent_episodes"]) == 0

    def test_different_device_name_fails(self):
        """Episode with different device name should fail."""
        serial_data_1 = {
            "left_umi_gripper": {
                "device_name": "UMI-003",
                "serial": "50443405000000B6",
            },
        }
        serial_data_2 = {
            "left_umi_gripper": {
                "device_name": "UMI-004",
                "serial": "50443405000000B6",
            },
        }
        data_store = _create_dataset_with_gripper_serials(
            [serial_data_1, serial_data_1, serial_data_2]
        )

        result = check_gripper_serial_consistency(
            data_store, ["episode_0", "episode_1", "episode_2"]
        )

        assert not result["passed"]
        assert "episode_2" in result["inconsistent_episodes"]
        assert any("device_name mismatch" in issue for issue in result["issues"])

    def test_different_serial_fails(self):
        """Episode with different serial number should fail."""
        serial_data_1 = {
            "left_umi_gripper": {
                "device_name": "UMI-003",
                "serial": "50443405000000B6",
            },
        }
        serial_data_2 = {
            "left_umi_gripper": {
                "device_name": "UMI-003",
                "serial": "DIFFERENT_SERIAL",
            },
        }
        data_store = _create_dataset_with_gripper_serials(
            [serial_data_1, serial_data_2, serial_data_1]
        )

        result = check_gripper_serial_consistency(
            data_store, ["episode_0", "episode_1", "episode_2"]
        )

        assert not result["passed"]
        assert "episode_1" in result["inconsistent_episodes"]
        assert any("serial mismatch" in issue for issue in result["issues"])

    def test_missing_gripper_in_episode_fails(self):
        """Episode missing a gripper that exists in reference should fail."""
        serial_data_1 = {
            "left_umi_gripper": {"device_name": "UMI-003", "serial": "ABC123"},
            "right_umi_gripper": {"device_name": "UMI-007", "serial": "DEF456"},
        }
        serial_data_2 = {
            "left_umi_gripper": {"device_name": "UMI-003", "serial": "ABC123"},
            # right_umi_gripper is missing
        }
        data_store = _create_dataset_with_gripper_serials(
            [serial_data_1, serial_data_2]
        )

        result = check_gripper_serial_consistency(
            data_store, ["episode_0", "episode_1"]
        )

        assert not result["passed"]
        assert "episode_1" in result["inconsistent_episodes"]
        assert any("missing" in issue for issue in result["issues"])

    def test_no_gripper_serial_data(self):
        """Dataset with no gripper serial data in any episode should report issue."""
        data_store = _create_dataset_with_gripper_serials([None, None, None])

        result = check_gripper_serial_consistency(
            data_store, ["episode_0", "episode_1", "episode_2"]
        )

        assert "No gripper serial data found" in result["issues"][0]

    def test_empty_episode_list_passes(self):
        """Empty episode list should pass (nothing to check)."""
        data_store = _create_dataset_with_gripper_serials([])

        result = check_gripper_serial_consistency(data_store, [])

        # No episodes means no inconsistencies found
        assert len(result["inconsistent_episodes"]) == 0

    def test_single_episode_passes(self):
        """Single episode should pass (nothing to compare against)."""
        serial_data = {
            "left_umi_gripper": {"device_name": "UMI-003", "serial": "ABC123"},
        }
        data_store = _create_dataset_with_gripper_serials([serial_data])

        result = check_gripper_serial_consistency(data_store, ["episode_0"])

        assert result["passed"]
        assert result["gripper_serials"] == serial_data


# =============================================================================
# Device Name Consistency Validation Tests
# =============================================================================

# Expected device names are derived per session from its own recording config,
# so these tuples are just one conventional bimanual rig — not a canonical role
# list the check knows about.
BIMANUAL_VIDEO_DEVICES = (
    "left_tracking_camera",
    "left_umi_camera",
    "left_trigger_camera",
    "right_tracking_camera",
    "right_umi_camera",
    "right_trigger_camera",
)
BIMANUAL_NUMERIC_DEVICES = ("left_umi_gripper", "right_umi_gripper")
BIMANUAL_DEVICES = BIMANUAL_VIDEO_DEVICES + BIMANUAL_NUMERIC_DEVICES
LEFT_DEVICES = (
    "left_tracking_camera",
    "left_umi_camera",
    "left_trigger_camera",
    "left_umi_gripper",
)
RIGHT_DEVICES = (
    "right_tracking_camera",
    "right_umi_camera",
    "right_trigger_camera",
    "right_umi_gripper",
)


def _config_declaring(device_names):
    """Recording config that declares exactly `device_names` as its sources."""
    return {
        "video_sources": [
            {"type": "oak_camera", "name": name}
            for name in device_names
            if not name.endswith("_gripper")
        ],
        "numeric_sources": [
            {"type": "umi_gripper", "name": name}
            for name in device_names
            if name.endswith("_gripper")
        ],
    }


class TestCheckDeviceNameConsistency:
    """Tests for check_device_name_consistency function."""

    def test_all_declared_devices_passes_both_grippers(self):
        """Episode recording every declared device passes."""
        result = check_device_name_consistency(
            _config_declaring(BIMANUAL_DEVICES), BIMANUAL_DEVICES
        )

        assert result["passed"]
        assert len(result["missing_devices"]) == 0
        assert len(result["issues"]) == 0

    def test_missing_device_fails_both_grippers(self):
        """Episode missing one declared device should fail."""
        recorded = [d for d in BIMANUAL_DEVICES if d != "left_tracking_camera"]

        result = check_device_name_consistency(
            _config_declaring(BIMANUAL_DEVICES), recorded
        )

        assert not result["passed"]
        assert "left_tracking_camera" in result["missing_devices"]
        assert len(result["issues"]) == 1

    def test_missing_multiple_devices_fails_both_grippers(self):
        """Episode missing multiple declared devices lists all of them."""
        recorded = list(BIMANUAL_NUMERIC_DEVICES)

        result = check_device_name_consistency(
            _config_declaring(BIMANUAL_DEVICES), recorded
        )

        assert not result["passed"]
        assert sorted(result["missing_devices"]) == sorted(BIMANUAL_VIDEO_DEVICES)

    def test_extra_devices_not_declared_in_config(self):
        """Episode with a device the config never declared should fail."""
        recorded = list(BIMANUAL_DEVICES) + ["extra_sensor"]

        result = check_device_name_consistency(
            _config_declaring(BIMANUAL_DEVICES), recorded
        )

        assert not result["passed"]
        assert "extra_sensor" in result["extra_devices"]
        assert len(result["issues"]) == 1

    def test_empty_episode_fails_both_grippers(self):
        """Empty episode should fail with every declared device missing."""
        result = check_device_name_consistency(_config_declaring(BIMANUAL_DEVICES), [])

        assert not result["passed"]
        assert len(result["missing_devices"]) == 8

    def test_wrong_device_names_fails_both_grippers(self):
        """Episode with misspelled device names should fail both ways."""
        recorded = [
            "left_tracking_cam",  # Wrong: should be left_tracking_camera
            "right_tracking_cam",  # Wrong
            "left_camera",  # Wrong: should be left_umi_camera
            "right_camera",  # Wrong
            "gripper_left",  # Wrong: should be left_umi_gripper
            "gripper_right",  # Wrong
        ]

        result = check_device_name_consistency(
            _config_declaring(BIMANUAL_DEVICES), recorded
        )

        assert not result["passed"]
        assert len(result["missing_devices"]) == 8  # All declared are missing
        assert len(result["extra_devices"]) == 6

    def test_left_only_config_passes_with_left_devices(self):
        """A config declaring only left devices passes on a left-only episode."""
        result = check_device_name_consistency(
            _config_declaring(LEFT_DEVICES), LEFT_DEVICES
        )

        assert result["passed"]
        assert len(result["missing_devices"]) == 0
        assert len(result["extra_devices"]) == 0

    def test_left_only_config_fails_with_right_devices(self):
        """Right devices are extras, left devices missing, for a left config."""
        result = check_device_name_consistency(
            _config_declaring(LEFT_DEVICES), RIGHT_DEVICES
        )

        assert not result["passed"]
        # Right devices should be marked as extra
        assert "right_tracking_camera" in result["extra_devices"]
        assert "right_umi_camera" in result["extra_devices"]
        assert "right_umi_gripper" in result["extra_devices"]
        # Left devices should be marked as missing
        assert "left_tracking_camera" in result["missing_devices"]
        assert "left_umi_camera" in result["missing_devices"]
        assert "left_umi_gripper" in result["missing_devices"]

    def test_right_only_config_passes_with_right_devices(self):
        """A config declaring only right devices passes on a right-only episode."""
        result = check_device_name_consistency(
            _config_declaring(RIGHT_DEVICES), RIGHT_DEVICES
        )

        assert result["passed"]
        assert len(result["missing_devices"]) == 0
        assert len(result["extra_devices"]) == 0

    def test_right_only_config_fails_with_left_devices(self):
        """Left devices are extras, right devices missing, for a right config."""
        result = check_device_name_consistency(
            _config_declaring(RIGHT_DEVICES), LEFT_DEVICES
        )

        assert not result["passed"]
        # Left devices should be marked as extra
        assert "left_tracking_camera" in result["extra_devices"]
        assert "left_umi_camera" in result["extra_devices"]
        assert "left_umi_gripper" in result["extra_devices"]
        # Right devices should be marked as missing
        assert "right_tracking_camera" in result["missing_devices"]
        assert "right_umi_camera" in result["missing_devices"]
        assert "right_umi_gripper" in result["missing_devices"]

    def test_partial_left_devices_fails(self):
        """Episode missing some of the declared left devices should fail."""
        recorded = ["left_tracking_camera", "left_umi_gripper"]

        result = check_device_name_consistency(
            _config_declaring(LEFT_DEVICES), recorded
        )

        assert not result["passed"]
        assert "left_umi_camera" in result["missing_devices"]
        assert "left_trigger_camera" in result["missing_devices"]
        assert len(result["missing_devices"]) == 2

    def test_declared_head_camera_not_extra(self):
        """A head camera the session declares is not an extra device."""
        devices = list(BIMANUAL_DEVICES) + ["head_tracking_camera"]

        result = check_device_name_consistency(_config_declaring(devices), devices)

        assert result["passed"]
        assert "head_tracking_camera" not in result["extra_devices"]

    def test_optional_devices_are_allowed_without_being_declared(self):
        """`optional_devices` from the check config is an allow-list."""
        config = _config_declaring(BIMANUAL_DEVICES)
        config["device_name_consistency"] = {
            "optional_devices": ["head_umi_camera", "bottom_oak_camera"]
        }
        recorded = list(BIMANUAL_DEVICES) + ["head_umi_camera", "bottom_oak_camera"]

        result = check_device_name_consistency(config, recorded)

        assert result["passed"]
        assert len(result["extra_devices"]) == 0

    def test_optional_plus_unknown_device_fails(self):
        """An optional camera plus an unknown device should still fail."""
        config = _config_declaring(BIMANUAL_DEVICES)
        config["device_name_consistency"] = {"optional_devices": ["head_umi_camera"]}
        recorded = list(BIMANUAL_DEVICES) + ["head_umi_camera", "unknown_sensor"]

        result = check_device_name_consistency(config, recorded)

        assert not result["passed"]
        assert "unknown_sensor" in result["extra_devices"]
        assert "head_umi_camera" not in result["extra_devices"]

    # -- device_name_consistency config path tests --

    def test_device_name_consistency_episode_matching_passes(self):
        """Episode with all devices from device_name_consistency passes."""
        config = {
            "device_name_consistency": {
                "left_video_sources": ["left_tracking_camera", "left_umi_camera"],
                "left_numeric_sources": ["left_umi_gripper"],
            },
        }
        devices = ["left_tracking_camera", "left_umi_camera", "left_umi_gripper"]

        result = check_device_name_consistency(config, devices)

        assert result["passed"]

    def test_device_name_consistency_missing_device_fails(self):
        """Episode missing a device specified in device_name_consistency fails."""
        config = {
            "device_name_consistency": {
                "left_video_sources": ["left_tracking_camera", "left_umi_camera"],
                "left_numeric_sources": ["left_umi_gripper"],
            },
        }
        devices = [
            "left_tracking_camera",
            "left_umi_gripper",
        ]  # Missing left_umi_camera

        result = check_device_name_consistency(config, devices)

        assert not result["passed"]
        assert "left_umi_camera" in result["missing_devices"]

    def test_device_name_consistency_extra_device_fails(self):
        """Episode with a device not in device_name_consistency fails."""
        config = {
            "device_name_consistency": {
                "left_video_sources": ["left_tracking_camera"],
                "left_numeric_sources": ["left_umi_gripper"],
            },
        }
        devices = ["left_tracking_camera", "left_umi_gripper", "unknown_sensor"]

        result = check_device_name_consistency(config, devices)

        assert not result["passed"]
        assert "unknown_sensor" in result["extra_devices"]

    def test_device_name_consistency_custom_names_passes(self):
        """Custom device names from device_name_consistency are accepted."""
        config = {
            "device_name_consistency": {
                "left_video_sources": ["custom_cam"],
                "left_numeric_sources": ["custom_sensor"],
            },
        }
        devices = ["custom_cam", "custom_sensor"]

        result = check_device_name_consistency(config, devices)

        assert result["passed"]

    def test_check_config_names_win_over_the_session_config(self):
        """The check-config block pins the required names for rigs whose zarr
        groups are not one-to-one with recording-config sources."""
        config = {
            "video_sources": [{"type": "oak_camera", "name": "right_umi_camera"}],
            "numeric_sources": [],
            "device_name_consistency": {
                "right_video_sources": ["right_umi_camera"],
                "right_numeric_sources": ["ur5e", "right_hand_joint"],
            },
        }

        result = check_device_name_consistency(
            config, ["right_umi_camera", "ur5e", "right_hand_joint"]
        )

        assert result["passed"]
        assert result["expected_devices"] == [
            "right_hand_joint",
            "right_umi_camera",
            "ur5e",
        ]


class TestCheckConfigDeviceNames:
    """Tests for check_config_device_names function."""

    def test_derived_names_always_match_the_config(self):
        """Without a check-config name list there is nothing to validate the
        config against — expected names come from the config itself."""
        config = _config_declaring(BIMANUAL_DEVICES)

        result = check_config_device_names(config)

        assert result["passed"]
        assert result["mismatched_names"] == []
        assert result["expected_devices"] == sorted(BIMANUAL_DEVICES)

    def test_unconventional_names_are_not_a_config_mismatch(self):
        """A rig naming its cameras bottom_oak_camera / head_oak_camera is valid."""
        devices = [
            "bottom_oak_camera",
            "head_iphone_camera",
            "head_oak_camera",
            "right_umi_gripper",
        ]

        result = check_config_device_names(_config_declaring(devices))

        assert result["passed"]
        assert result["mismatched_names"] == []

    def test_oak_device_streams_are_reported_as_declared(self):
        """An unexpanded oak_device block declares its stream names."""
        config = {
            "video_sources": [
                {
                    "type": "oak_device",
                    "oak_device_name": "oak-FFC-05",
                    "streams": [
                        {"name": "right_trigger_camera"},
                        {"name": "right_umi_camera"},
                    ],
                }
            ],
            "numeric_sources": [{"type": "umi_gripper", "name": "right_umi_gripper"}],
        }

        result = check_config_device_names(config)

        assert result["passed"]
        assert result["expected_devices"] == [
            "right_trigger_camera",
            "right_umi_camera",
            "right_umi_gripper",
        ]

    def test_oak_device_block_with_both_name_and_streams_is_not_a_mismatch(self):
        """The expected set and the config set must come from one extractor: if
        they disagreed, every episode index would be stamped with
        config_device_name_mismatch."""
        config = {
            "video_sources": [
                {
                    "type": "oak_device",
                    "name": "left_oak",
                    "streams": [
                        {"name": "left_trigger_camera"},
                        {"name": "left_umi_camera"},
                    ],
                }
            ],
            "numeric_sources": [{"type": "umi_gripper", "name": "left_umi_gripper"}],
        }

        result = check_config_device_names(config)

        assert result["passed"], result
        assert result["mismatched_names"] == []
        assert "left_oak" not in result["config_devices"]

    # -- device_name_consistency config path tests --

    def test_device_name_consistency_matching_config_passes(self):
        """Config devices matching device_name_consistency should pass."""
        config = {
            "device_name_consistency": {
                "left_video_sources": ["left_tracking_camera", "left_umi_camera"],
                "left_numeric_sources": ["left_umi_gripper"],
                "optional_devices": ["left_trigger_camera"],
            },
            "video_sources": [
                {"name": "left_tracking_camera"},
                {"name": "left_umi_camera"},
                {"name": "left_trigger_camera"},
            ],
            "numeric_sources": [{"name": "left_umi_gripper"}],
        }
        result = check_config_device_names(config)

        assert result["passed"]
        assert len(result["mismatched_names"]) == 0

    def test_device_name_consistency_unknown_device_fails(self):
        """Config with device not in device_name_consistency should fail."""
        config = {
            "device_name_consistency": {
                "left_video_sources": ["left_tracking_camera"],
                "left_numeric_sources": ["left_umi_gripper"],
            },
            "video_sources": [{"name": "weird_camera"}],
            "numeric_sources": [{"name": "left_umi_gripper"}],
        }
        result = check_config_device_names(config)

        assert not result["passed"]
        assert "weird_camera" in result["mismatched_names"]

    def test_device_name_consistency_custom_names_passes(self):
        """Custom device names from device_name_consistency are accepted in config."""
        config = {
            "device_name_consistency": {
                "left_video_sources": ["my_camera"],
                "left_numeric_sources": ["my_sensor"],
            },
            "video_sources": [{"name": "my_camera"}],
            "numeric_sources": [{"name": "my_sensor"}],
        }
        result = check_config_device_names(config)

        assert result["passed"]


# =============================================================================
# Video Resolution Validation Tests
# =============================================================================


def _create_test_video(video_path: str, width: int, height: int, num_frames: int = 10):
    """Helper to create a test H.264 video file with specified resolution.

    Args:
        video_path: Path where video file will be created
        width: Video width in pixels
        height: Video height in pixels
        num_frames: Number of frames to create
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        "30",
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        "-preset",
        "ultrafast",
        str(video_path),
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    for _ in range(num_frames):
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    assert proc.returncode == 0, (
        f"ffmpeg H.264 encoding failed: {proc.stderr.read().decode()}"
    )


class TestCheckVideoResolution:
    """Tests for check_video_resolution function."""

    def test_correct_resolution_passes(self):
        """Video with correct resolution (1280, 720) should pass."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            video_path = tmp_file.name

        try:
            _create_test_video(video_path, width=1280, height=720)

            result = check_video_resolution(video_path, expected_resolution=(1280, 720))

            assert result["passed"]
            assert result["actual_resolution"] == (1280, 720)
            assert result["expected_resolution"] == (1280, 720)
            assert result["error"] is None
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)

    def test_wrong_resolution_fails(self):
        """Video with wrong resolution should fail."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            video_path = tmp_file.name

        try:
            _create_test_video(video_path, width=640, height=480)

            result = check_video_resolution(video_path, expected_resolution=(1280, 720))

            assert not result["passed"]
            assert result["actual_resolution"] == (640, 480)
            assert result["expected_resolution"] == (1280, 720)
            assert result["error"] is None
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)

    def test_wrong_width_fails(self):
        """Video with correct height but wrong width should fail."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            video_path = tmp_file.name

        try:
            _create_test_video(video_path, width=1920, height=720)

            result = check_video_resolution(video_path, expected_resolution=(1280, 720))

            assert not result["passed"]
            assert result["actual_resolution"] == (1920, 720)
            assert result["expected_resolution"] == (1280, 720)
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)

    def test_wrong_height_fails(self):
        """Video with correct width but wrong height should fail."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            video_path = tmp_file.name

        try:
            _create_test_video(video_path, width=1280, height=1080)

            result = check_video_resolution(video_path, expected_resolution=(1280, 720))

            assert not result["passed"]
            assert result["actual_resolution"] == (1280, 1080)
            assert result["expected_resolution"] == (1280, 720)
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)

    def test_missing_file_fails(self):
        """Missing video file should fail with error message."""
        video_path = "/nonexistent/path/video.mp4"

        result = check_video_resolution(video_path, expected_resolution=(1280, 720))

        assert not result["passed"]
        assert result["actual_resolution"] is None
        assert result["error"] is not None
        assert "not found" in result["error"].lower()

    def test_custom_expected_resolution(self):
        """Should work with custom expected resolution."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            video_path = tmp_file.name

        try:
            _create_test_video(video_path, width=640, height=480)

            result = check_video_resolution(video_path, expected_resolution=(640, 480))

            assert result["passed"]
            assert result["actual_resolution"] == (640, 480)
            assert result["expected_resolution"] == (640, 480)
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)

    def test_default_expected_resolution(self):
        """Default expected resolution should be (1280, 720)."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            video_path = tmp_file.name

        try:
            _create_test_video(video_path, width=1280, height=720)

            result = check_video_resolution(video_path)

            assert result["passed"]
            assert result["expected_resolution"] == (1280, 720)
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)

    def test_swapped_dimensions_fails(self):
        """Video with swapped width/height should fail."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            video_path = tmp_file.name

        try:
            _create_test_video(video_path, width=720, height=1280)

            result = check_video_resolution(video_path, expected_resolution=(1280, 720))

            assert not result["passed"]
            assert result["actual_resolution"] == (720, 1280)
            assert result["expected_resolution"] == (1280, 720)
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)


# =============================================================================
# Sustained Frame Drop Tests
# =============================================================================


class TestCheckSustainedFrameDrop:
    """Tests for check_sustained_frame_drop function."""

    def test_exact_fps_passes(self):
        """Timestamps at exactly expected FPS should pass."""
        expected_fps = 45.0
        timestamps = np.arange(200) / expected_fps

        result = check_sustained_frame_drop(timestamps, expected_fps)

        assert result["passed"]
        assert result["num_violations"] == 0

    def test_slight_dip_within_ratio_passes(self):
        """A slight FPS dip within the 70% ratio should pass."""
        expected_fps = 45.0
        timestamps = np.arange(200) / (expected_fps * 0.75)

        result = check_sustained_frame_drop(timestamps, expected_fps)

        assert result["passed"]

    def test_sustained_drop_below_ratio_fails(self):
        """A sustained gap causing FPS well below 70% threshold should fail."""
        expected_fps = 45.0
        good = np.arange(50) / expected_fps
        gap_start = good[-1] + 1 / expected_fps
        # 2-second gap with very few frames
        sparse = gap_start + np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        after_gap = sparse[-1] + np.arange(1, 50) / expected_fps
        timestamps = np.concatenate([good, sparse, after_gap])

        result = check_sustained_frame_drop(timestamps, expected_fps)

        assert not result["passed"]
        assert result["num_violations"] > 0

    def test_short_array_passes(self):
        """Array with fewer than 2 timestamps should pass with error note."""
        result = check_sustained_frame_drop(np.array([1.0]), 45.0)

        assert result["passed"]
        assert "error" in result

    def test_episode_shorter_than_window_passes(self):
        """Episode shorter than window duration should pass with reason."""
        timestamps = np.arange(10) / 45.0  # ~0.2s, shorter than 1s window

        result = check_sustained_frame_drop(timestamps, 45.0, window_s=1.0)

        assert result["passed"]
        assert "reason" in result

    def test_violations_capped_at_20(self):
        """Violations list should be capped at 20 entries."""
        expected_fps = 45.0
        # Very sparse timestamps over a long period → many violations
        timestamps = np.arange(0, 60, 1.0)  # 1 frame per second for 60s

        result = check_sustained_frame_drop(timestamps, expected_fps)

        assert not result["passed"]
        assert len(result["violations"]) <= 20
        assert result["num_violations"] > 20


# =============================================================================
# Device Expected Resolutions Tests
# =============================================================================


class TestGetDeviceExpectedResolutions:
    """Tests for get_device_expected_resolutions function."""

    def test_builds_resolution_map(self):
        """Should build name -> (w, h) map from video_sources."""
        config = {
            "video_sources": [
                {
                    "name": "left_umi_camera",
                    "type": "oak_camera",
                    "params": {"camera_resolution": [1280, 720]},
                },
                {
                    "name": "right_umi_camera",
                    "type": "oak_camera",
                    "params": {"camera_resolution": [640, 480]},
                },
            ]
        }

        result = get_device_expected_resolutions(config)

        assert result == {
            "left_umi_camera": (1280, 720),
            "right_umi_camera": (640, 480),
        }

    def test_iphone_camera_without_resolution_skipped(self):
        """iphone_camera entries without camera_resolution should be skipped."""
        config = {
            "video_sources": [
                {
                    "name": "tracking_cam",
                    "type": "iphone_camera",
                    "params": {},
                },
            ]
        }

        result = get_device_expected_resolutions(config)

        assert result == {}

    def test_empty_sources(self):
        """Empty video_sources should return empty dict."""
        result = get_device_expected_resolutions({"video_sources": []})
        assert result == {}

    def test_no_video_sources_key(self):
        """Missing video_sources key should return empty dict."""
        result = get_device_expected_resolutions({})
        assert result == {}


# =============================================================================
# Extract Episode Index Tests
# =============================================================================


class TestExtractEpisodeIndex:
    """Tests for extract_episode_index function."""

    def test_standard_episode_key(self):
        """Should parse 'episode_0' -> 0."""
        assert extract_episode_index("episode_0") == 0

    def test_large_index(self):
        """Should parse 'episode_42' -> 42."""
        assert extract_episode_index("episode_42") == 42

    def test_pending_suffix(self):
        """Should parse 'episode_3.pending' -> 3."""
        assert extract_episode_index("episode_3.pending") == 3

    def test_invalid_key_raises(self):
        """Should raise ValueError for invalid episode key."""
        import pytest

        with pytest.raises(ValueError):
            extract_episode_index("not_an_episode")


# =============================================================================
# Latency Adjusted Timestamps Tests (zarr-based)
# =============================================================================


class TestGetLatencyAdjustedTimestamps:
    """Tests for get_latency_adjusted_timestamps function."""

    def test_prefers_capture_time(self):
        """Should return capture_time when both exist."""
        root = zarr.group()
        dev = root.create_group("cam")
        dev.create_array("capture_time", data=np.array([1.0, 2.0]))
        dev.create_array("receive_time", data=np.array([3.0, 4.0]))

        result = get_latency_adjusted_timestamps(root, "cam")

        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_falls_back_to_receive_time(self):
        """Should return receive_time when capture_time is absent."""
        root = zarr.group()
        dev = root.create_group("cam")
        dev.create_array("receive_time", data=np.array([3.0, 4.0]))

        result = get_latency_adjusted_timestamps(root, "cam")

        np.testing.assert_array_equal(result, [3.0, 4.0])

    def test_returns_none_when_neither_exists(self):
        """Should return None when neither timestamp array exists."""
        root = zarr.group()
        root.create_group("cam")

        result = get_latency_adjusted_timestamps(root, "cam")

        assert result is None


# =============================================================================
# Device Timestamps Tests (zarr-based)
# =============================================================================


class TestGetDeviceTimestamps:
    """Tests for get_device_timestamps function."""

    def test_prefers_capture_time(self):
        """Should return capture_time when both exist."""
        root = zarr.group()
        dev = root.create_group("sensor")
        dev.create_array("capture_time", data=np.array([10.0, 20.0]))
        dev.create_array("receive_time", data=np.array([30.0, 40.0]))

        result = get_device_timestamps(root, "sensor")

        np.testing.assert_array_equal(result, [10.0, 20.0])

    def test_falls_back_to_receive_time(self):
        """Should return receive_time when capture_time is absent."""
        root = zarr.group()
        dev = root.create_group("sensor")
        dev.create_array("receive_time", data=np.array([30.0, 40.0]))

        result = get_device_timestamps(root, "sensor")

        np.testing.assert_array_equal(result, [30.0, 40.0])

    def test_returns_none_for_missing_device(self):
        """Should return None when device does not exist."""
        root = zarr.group()

        result = get_device_timestamps(root, "nonexistent")

        assert result is None

    def test_returns_none_when_no_timestamps(self):
        """Should return None when device has no timestamp arrays."""
        root = zarr.group()
        root.create_group("sensor")

        result = get_device_timestamps(root, "sensor")

        assert result is None


# =============================================================================
# Load Check Config Tests
# =============================================================================


class TestLoadCheckConfig:
    """Tests for load_check_config function."""

    def test_loads_yaml_file(self, tmp_path):
        """Should load and return config dict from YAML file."""
        config_data = {
            "timestamp": {"fps_deviation_pct": 10.0},
            "gripper": {"movement_threshold_mm": 10.0},
        }
        config_file = tmp_path / "check_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        result = load_check_config(str(config_file))

        assert result["timestamp"]["fps_deviation_pct"] == 10.0
        assert result["gripper"]["movement_threshold_mm"] == 10.0

    def test_default_path_loads(self):
        """Default config file should load successfully."""
        result = load_check_config()

        assert "timestamp" in result
        assert "gripper" in result
        assert "tracking" in result

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Jump checks score speed, not per-sample delta.

Sample gaps alternate ~22/~44 ms (45 Hz stream into a 30 Hz recording), so a
flat per-sample threshold would make the verdict depend on which gap a motion
straddled.
"""

import numpy as np

from real_scripts.data_processing.data_check import (
    check_gripper_jumps,
    check_tracking_pose_jumps,
)

GRIPPER_LIMIT = 2000.0  # mm/s
POSE_LIMIT = 3.0  # m/s
SHORT_DT = 1.0 / 45.0  # ~22.2 ms — recorder took one 45 Hz tick
LONG_DT = 2.0 / 45.0  # ~44.4 ms — recorder took two 45 Hz ticks


def _ts(*gaps):
    return np.concatenate([[0.0], np.cumsum(gaps)])


def _poses_at(*xs):
    out = np.tile(np.eye(4), (len(xs), 1, 1))
    out[:, 0, 3] = xs
    return out


class TestGripperJumps:
    def test_same_speed_scores_the_same_on_short_and_long_gaps(self):
        """Identical motion gets the same verdict whatever the sample gap."""
        # Both ~1125 mm/s — the same physical release speed.
        short = check_gripper_jumps(np.array([0.0, 25.0]), _ts(SHORT_DT))
        long = check_gripper_jumps(np.array([0.0, 50.0]), _ts(LONG_DT))

        assert short["has_jumps"] == long["has_jumps"] is False

    def test_fast_release_on_long_gap_passes(self):
        """52 mm over 44.4 ms is ~1171 mm/s — under the ceiling."""
        result = check_gripper_jumps(np.array([15.1, 67.1]), _ts(LONG_DT))

        assert not result["has_jumps"]

    def test_fastest_real_release_passes(self):
        """The fastest spring release measured in real demos: 77 mm / 45.5 ms.

        ~1697 mm/s. Physically achievable, so it must not be quarantined.
        """
        result = check_gripper_jumps(np.array([0.0, 77.1]), _ts(0.0455))

        assert not result["has_jumps"]

    def test_true_discontinuity_is_flagged(self):
        """A full-scale step across one short gap is ~3600 mm/s — impossible."""
        result = check_gripper_jumps(np.array([0.0, 80.0]), _ts(SHORT_DT))

        assert result["num_jumps"] == 1
        assert result["jump_frames"][0]["magnitude_mm_per_s"] > GRIPPER_LIMIT

    def test_fast_motion_hidden_by_a_short_gap_is_now_caught(self):
        """50 mm over 22.2 ms is ~2250 mm/s, over the ceiling; a per-sample
        delta threshold would miss it on a short gap.
        """
        result = check_gripper_jumps(np.array([0.0, 50.0]), _ts(SHORT_DT))

        assert result["has_jumps"]

    def test_broken_timestamps_are_skipped_not_flagged(self):
        """Non-monotonic stamps are a separate check; don't invent jumps."""
        result = check_gripper_jumps(
            np.array([0.0, 80.0, 80.5]), np.array([0.0, 0.0, LONG_DT])
        )

        assert not result["has_jumps"]

    def test_single_sample_passes(self):
        assert not check_gripper_jumps(np.array([50.0]), np.array([0.0]))["has_jumps"]


class TestTrackingPoseJumps:
    def test_dropped_frame_is_not_a_jump(self):
        """0.12 m across a 640 ms dropout is 0.19 m/s — ordinary hand speed."""
        result = check_tracking_pose_jumps(
            _poses_at(0.0, 0.12), _ts(0.64), jump_threshold_m_per_s=POSE_LIMIT
        )

        assert not result["has_jumps"]

    def test_discontinuity_across_a_short_gap_is_flagged(self):
        """0.11 m in one 30 fps frame is 3.3 m/s — past any real hand motion."""
        result = check_tracking_pose_jumps(
            _poses_at(0.0, 0.11), _ts(1 / 30), jump_threshold_m_per_s=POSE_LIMIT
        )

        assert result["num_jumps"] == 1
        assert result["jump_frames"][0]["magnitude_m_per_s"] > POSE_LIMIT

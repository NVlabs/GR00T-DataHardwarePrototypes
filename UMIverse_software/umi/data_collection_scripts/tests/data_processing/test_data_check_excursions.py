# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tracker-excursion (pose glitch) detector contract.

A relocalisation glitch moves the tracked pose 20-150 mm and brings it back
within ~100-300 ms while the hand itself barely moves. At 30 Hz a 50 mm glitch
is 1.5 m/s on one interval -- well under the 4.5 m/s `pose_jumps` ceiling -- so
the jump check alone does not catch it. Confirmed glitches are
translation-only: real fast motion rotates the hand, a glitch does not. These
tests pin the detector's output contract on synthetic poses, independent of how
the detector is built.

Run with: uv run pytest tests/data_processing/test_data_check_excursions.py -v
"""

import json

import numpy as np
import pytest

from real_scripts.data_processing.data_check.tracking import (
    check_tracking_excursions,
    check_tracking_pose_jumps,
)

FPS = 30.0
SHORT_DT = 1.0 / 45.0  # ~22 ms: recorder took one 45 Hz tick
LONG_DT = 2.0 / 45.0  # ~44 ms: recorder took two 45 Hz ticks

TOP_LEVEL_KEYS = {
    "detector_version",
    "num_samples",
    "max_dv_m_per_s",
    "num_nominated",
    "num_events",
    "num_candidates",
    "events",
}
EVENT_KEYS = {
    "shape",
    "i0",
    "i1",
    "t_out_s",
    "t_back_s",
    "duration_ms",
    "amplitude_mm",
    "max_step_mm",
    "edge_share",
    "return_mm",
    "return_tol_mm",
    "hand_speed_m_per_s",
    "dv_m_per_s",
    "edge_rotation_deg_per_m_out",
    "edge_rotation_deg_per_m_back",
    "rotation_quiet",
    "state",
    "reason",
}


# --------------------------------------------------------------------------
# synthetic trajectories
# --------------------------------------------------------------------------


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _poses(positions, yaws=None, rolls=None):
    n = len(positions)
    yaws = np.zeros(n) if yaws is None else yaws
    rolls = np.zeros(n) if rolls is None else rolls
    out = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        out[i, :3, :3] = _rot_z(yaws[i]) @ _rot_x(rolls[i])
        out[i, :3, 3] = positions[i]
    return out


def _uniform_ts(n, fps=FPS):
    return np.arange(n) / fps


def _alternating_ts(n):
    gaps = np.where(np.arange(n - 1) % 2 == 0, SHORT_DT, LONG_DT)
    return np.concatenate([[0.0], np.cumsum(gaps)])


def _resting_hand(ts):
    """Hand nearly at rest: ~5 mm/s drift, slow smooth orientation wobble."""
    positions = np.stack(
        [0.005 * ts, 0.002 * np.sin(0.7 * ts), np.full_like(ts, 0.3)], axis=1
    )
    yaws = 0.03 * np.sin(0.5 * ts)
    return positions, yaws


def _smooth_moderate_motion(ts):
    """Ordinary manipulation: a slow sway plus one brisk min-jerk reach that
    rotates the hand the way a real reach does."""
    x = 0.15 * np.sin(2 * np.pi * 0.3 * ts)
    y = np.zeros_like(ts)
    # Min-jerk reach: 0.3 m in 0.5 s starting at t=2 s (~1.1 m/s peak).
    tau = np.clip((ts - 2.0) / 0.5, 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    y = y + 0.3 * s
    positions = np.stack([x, y, np.full_like(ts, 0.3)], axis=1)
    yaws = 0.4 * np.sin(2 * np.pi * 0.3 * ts) + 0.6 * s
    return positions, yaws


def _inject_out_and_back(positions, start, length, offset_m, axis=1):
    out = positions.copy()
    out[start : start + length, axis] += offset_m
    return out


def _glitch_episode(
    amplitude_m=0.100, start=60, length=8, n=150, ts=None, rotation_rad=0.0
):
    ts = _uniform_ts(n) if ts is None else ts
    positions, yaws = _resting_hand(ts)
    positions = _inject_out_and_back(positions, start, length, amplitude_m)
    rolls = np.zeros(len(ts))
    rolls[start : start + length] += rotation_rad
    return _poses(positions, yaws, rolls), ts


def _assert_plain_json(obj, path="result"):
    """Exact builtin types only: np.float64 subclasses float and would slip
    through json.dumps, but the result is persisted into zarr attrs."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert type(k) is str, f"{path}: key {k!r} is {type(k)}"
            _assert_plain_json(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_plain_json(v, f"{path}[{i}]")
    else:
        assert obj is None or type(obj) in (str, int, float, bool), (
            f"{path} is {type(obj)}"
        )


def _out_and_back(result):
    return [e for e in result["events"] if e["shape"] == "out_and_back"]


# --------------------------------------------------------------------------
# result shape
# --------------------------------------------------------------------------


class TestResultShape:
    def test_top_level_keys_and_version(self):
        poses, ts = _glitch_episode()
        result = check_tracking_excursions(poses, ts)

        assert TOP_LEVEL_KEYS <= set(result)
        assert result["detector_version"] == 1
        assert result["num_samples"] == len(ts)
        assert result["num_events"] == len(result["events"])

    def test_every_event_carries_every_key(self):
        poses, ts = _glitch_episode()
        result = check_tracking_excursions(poses, ts)

        assert result["events"], "precondition: the glitch produced an event"
        for event in result["events"]:
            assert EVENT_KEYS <= set(event)

    def test_result_is_plain_json(self):
        poses, ts = _glitch_episode()
        result = check_tracking_excursions(poses, ts)

        _assert_plain_json(result)
        assert json.loads(json.dumps(result)) == result

    def test_empty_result_is_plain_json(self):
        ts = _uniform_ts(150)
        positions, yaws = _smooth_moderate_motion(ts)
        result = check_tracking_excursions(_poses(positions, yaws), ts)

        _assert_plain_json(result)
        json.dumps(result)

    def test_num_candidates_counts_candidate_events(self):
        poses, ts = _glitch_episode()
        result = check_tracking_excursions(poses, ts)

        assert result["num_candidates"] == sum(
            e["state"] == "candidate" for e in result["events"]
        )


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


class TestSmoothMotion:
    def test_smooth_moderate_motion_has_no_events(self):
        ts = _uniform_ts(150)
        positions, yaws = _smooth_moderate_motion(ts)
        result = check_tracking_excursions(_poses(positions, yaws), ts)

        assert result["events"] == []
        assert result["num_events"] == 0
        assert result["num_candidates"] == 0

    def test_resting_hand_has_no_events(self):
        ts = _uniform_ts(150)
        positions, yaws = _resting_hand(ts)
        result = check_tracking_excursions(_poses(positions, yaws), ts)

        assert result["events"] == []


class TestOutAndBackGlitch:
    START, LENGTH = 60, 8

    def _result(self, amplitude_m=0.100):
        poses, ts = _glitch_episode(amplitude_m, self.START, self.LENGTH)
        return check_tracking_excursions(poses, ts), ts

    def test_exactly_one_out_and_back_event(self):
        result, _ = self._result()

        assert len(result["events"]) == 1
        assert result["events"][0]["shape"] == "out_and_back"
        assert result["num_events"] == 1

    def test_translation_only_glitch_is_a_rotation_quiet_candidate(self):
        result, _ = self._result()
        event = result["events"][0]

        assert event["rotation_quiet"] is True
        assert event["state"] == "candidate"
        assert event["reason"] is None
        assert event["edge_rotation_deg_per_m_out"] <= 50.0
        # Recorded for out_and_back, even though it never gates.
        assert type(event["edge_rotation_deg_per_m_back"]) is float
        assert result["num_candidates"] == 1

    def test_amplitude_is_near_the_injected_offset(self):
        result, _ = self._result()

        assert result["events"][0]["amplitude_mm"] == pytest.approx(100.0, abs=15.0)

    def test_rows_bracket_the_injected_frames(self):
        """i0/i1 are the first/last row of the excursion; whether that is the
        first displaced row or the last clean one is left to the detector."""
        result, ts = self._result()
        event = result["events"][0]
        last_displaced = self.START + self.LENGTH - 1

        assert self.START - 1 <= event["i0"] <= self.START
        assert last_displaced <= event["i1"] <= last_displaced + 1
        assert type(event["i0"]) is int and type(event["i1"]) is int

    def test_times_are_the_input_clock_at_i0_and_i1(self):
        result, ts = self._result()
        event = result["events"][0]

        assert event["t_out_s"] == pytest.approx(float(ts[event["i0"]]))
        assert event["t_back_s"] == pytest.approx(float(ts[event["i1"]]))
        assert 100.0 <= event["duration_ms"] <= 400.0

    def test_times_are_absolute_not_episode_relative(self):
        """Timestamps on a station clock (epoch seconds), not starting at 0."""
        poses, ts = _glitch_episode(0.100, self.START, self.LENGTH)
        offset = 1_760_000_000.0
        result = check_tracking_excursions(poses, ts + offset)
        event = result["events"][0]

        assert event["t_out_s"] == pytest.approx(float(ts[event["i0"]] + offset))
        assert event["t_back_s"] == pytest.approx(float(ts[event["i1"]] + offset))

    def test_dv_and_step_reflect_the_glitch(self):
        result, _ = self._result()
        event = result["events"][0]

        # A 100 mm step in one 33 ms interval is ~3 m/s.
        assert event["dv_m_per_s"] > 1.0
        assert event["max_step_mm"] == pytest.approx(100.0, abs=15.0)
        assert event["hand_speed_m_per_s"] < 0.1
        assert result["max_dv_m_per_s"] >= event["dv_m_per_s"] - 1e-9
        assert result["num_nominated"] >= 1

    def test_a_50mm_glitch_is_caught_where_pose_jumps_is_blind(self):
        poses, ts = _glitch_episode(0.050, self.START, self.LENGTH)

        jumps = check_tracking_pose_jumps(poses, ts, jump_threshold_m_per_s=4.5)
        result = check_tracking_excursions(poses, ts)

        assert not jumps["has_jumps"], "precondition: 1.5 m/s is under 4.5 m/s"
        events = _out_and_back(result)
        assert len(events) == 1
        assert events[0]["amplitude_mm"] == pytest.approx(50.0, abs=10.0)
        assert events[0]["state"] == "candidate"

    def test_a_10mm_glitch_is_below_the_amplitude_floor(self):
        result, _ = self._result(amplitude_m=0.010)

        assert result["events"] == []

    def test_min_amplitude_is_a_parameter(self):
        poses, ts = _glitch_episode(0.050, self.START, self.LENGTH)

        result = check_tracking_excursions(poses, ts, min_amplitude_mm=80.0)

        assert _out_and_back(result) == []

    def test_glitch_on_a_moving_hand_is_still_caught(self):
        """The bridge removes the hand's own motion; the glitch stands out."""
        ts = _uniform_ts(150)
        positions = np.stack(
            [0.15 * ts, np.zeros_like(ts), np.full_like(ts, 0.3)], axis=1
        )
        positions = _inject_out_and_back(positions, self.START, self.LENGTH, 0.080)
        result = check_tracking_excursions(_poses(positions), ts)

        events = _out_and_back(result)
        assert len(events) == 1
        assert events[0]["amplitude_mm"] == pytest.approx(80.0, abs=15.0)


class TestRotationCoupledEdge:
    def test_rotation_rich_edge_is_rejected_for_rotation(self):
        """Real hand motion rotates: ~60 mm with 15 deg at the out edge is
        ~250 deg/m, far over the 50 deg/m quiet limit."""
        poses, ts = _glitch_episode(0.060, rotation_rad=np.radians(15.0))
        result = check_tracking_excursions(poses, ts)

        for event in result["events"]:
            assert event["rotation_quiet"] is False
            assert event["state"] == "rejected"
            assert event["reason"] == "rotation"
            assert event["edge_rotation_deg_per_m_out"] > 50.0
        assert result["num_candidates"] == 0

    def test_back_edge_rotation_is_recorded_but_never_gates(self):
        """Only the out edge decides rotation_quiet: a rotation-rich return is
        reported, not rejected."""
        start, length = 60, 8
        ts = _uniform_ts(150)
        positions, yaws = _resting_hand(ts)
        positions = _inject_out_and_back(positions, start, length, 0.060)
        rolls = np.zeros(len(ts))
        rolls[start + length :] += np.radians(15.0)  # turn only on the way back
        result = check_tracking_excursions(_poses(positions, yaws, rolls), ts)

        events = _out_and_back(result)
        assert len(events) == 1
        event = events[0]
        assert event["edge_rotation_deg_per_m_out"] <= 50.0
        assert event["edge_rotation_deg_per_m_back"] > 50.0
        assert event["rotation_quiet"] is True
        assert event["state"] == "candidate"

    def test_out_edge_between_old_and_new_limit_is_quiet(self):
        """~40 deg/m on the out edge: quiet under the default 50 deg/m limit."""
        start, length = 60, 8
        ts = _uniform_ts(150)
        positions, yaws = _resting_hand(ts)
        positions = _inject_out_and_back(positions, start, length, 0.100)
        rolls = np.zeros(len(ts))
        rolls[start : start + length] += np.radians(4.0)  # 4 deg / ~100 mm
        result = check_tracking_excursions(_poses(positions, yaws, rolls), ts)

        events = _out_and_back(result)
        assert len(events) == 1
        assert 35.0 < events[0]["edge_rotation_deg_per_m_out"] <= 50.0
        assert events[0]["rotation_quiet"] is True
        assert events[0]["state"] == "candidate"

        strict = check_tracking_excursions(
            _poses(positions, yaws, rolls), ts, edge_rotation_deg_per_m=35.0
        )
        assert _out_and_back(strict)[0]["state"] == "rejected"

    def test_rotation_only_twist_is_not_an_event(self):
        """Detection is on translation; a wrist twist with the hand in place
        is never an excursion."""
        poses, ts = _glitch_episode(0.0, rotation_rad=np.radians(30.0))
        result = check_tracking_excursions(poses, ts)

        assert result["events"] == []

    def test_rotation_limit_is_a_parameter(self):
        poses, ts = _glitch_episode(0.060, rotation_rad=np.radians(15.0))
        result = check_tracking_excursions(poses, ts, edge_rotation_deg_per_m=1e6)

        assert result["events"], "precondition: the out-and-back is detected"
        assert all(e["state"] == "candidate" for e in result["events"])


class TestStep:
    def test_non_returning_sharp_step_is_a_step(self):
        ts = _uniform_ts(150)
        positions, yaws = _resting_hand(ts)
        positions[60:, 1] += 0.100
        result = check_tracking_excursions(_poses(positions, yaws), ts)

        assert _out_and_back(result) == []
        steps = [e for e in result["events"] if e["shape"] == "step"]
        assert len(steps) == 1
        assert steps[0]["edge_rotation_deg_per_m_back"] is None
        assert 59 <= steps[0]["i0"] <= 60


class TestAlternatingCadence:
    """45 Hz stream into a 30 Hz recording: gaps alternate ~22/~44 ms."""

    def test_glitch_is_detected(self):
        ts = _alternating_ts(150)
        poses, ts = _glitch_episode(0.100, 60, 8, ts=ts)
        result = check_tracking_excursions(poses, ts)

        events = _out_and_back(result)
        assert len(events) == 1
        assert events[0]["amplitude_mm"] == pytest.approx(100.0, abs=15.0)
        assert events[0]["state"] == "candidate"

    def test_smooth_motion_has_no_events(self):
        ts = _alternating_ts(150)
        positions, yaws = _smooth_moderate_motion(ts)
        result = check_tracking_excursions(_poses(positions, yaws), ts)

        assert result["events"] == []


class TestMaxEvents:
    def _many_glitches(self, count=6):
        ts = _uniform_ts(int(1.5 * FPS) * (count + 1))
        positions, yaws = _resting_hand(ts)
        for k in range(count):
            positions = _inject_out_and_back(
                positions, int(1.5 * FPS) * (k + 1), 8, 0.080
            )
        return _poses(positions, yaws), ts

    def test_all_glitches_found_under_the_default_cap(self):
        poses, ts = self._many_glitches()
        result = check_tracking_excursions(poses, ts)

        assert len(_out_and_back(result)) == 6

    def test_cap_limits_the_event_list(self):
        poses, ts = self._many_glitches()
        result = check_tracking_excursions(poses, ts, max_events=3)

        assert len(result["events"]) == 3


# --------------------------------------------------------------------------
# degenerate input never raises
# --------------------------------------------------------------------------


class TestDegenerateInput:
    def test_empty(self):
        result = check_tracking_excursions(np.zeros((0, 4, 4)), np.zeros(0))

        assert result["events"] == []
        assert result["num_samples"] == 0
        assert result["detector_version"] == 1
        json.dumps(result)

    @pytest.mark.parametrize("n", [1, 2, 5, 12])
    def test_short(self, n):
        ts = _uniform_ts(n)
        positions, yaws = _resting_hand(ts)
        positions[n // 2 :, 1] += 0.1
        result = check_tracking_excursions(_poses(positions, yaws), ts)

        assert result["events"] == []
        assert result["num_samples"] == n
        json.dumps(result)

    def test_nan_poses(self):
        poses, ts = _glitch_episode()
        poses[10, :3, 3] = np.nan
        poses[100, :3, :3] = np.nan

        result = check_tracking_excursions(poses, ts)

        assert isinstance(result["events"], list)
        _assert_plain_json(result)
        json.dumps(result)

    def test_all_nan(self):
        ts = _uniform_ts(90)
        poses = np.full((90, 4, 4), np.nan)

        result = check_tracking_excursions(poses, ts)

        assert result["events"] == []

    def test_nan_timestamps(self):
        poses, ts = _glitch_episode()
        ts = ts.copy()
        ts[30] = np.nan

        result = check_tracking_excursions(poses, ts)

        assert isinstance(result["events"], list)
        json.dumps(result)

    def test_non_monotonic_timestamps_do_not_nominate(self):
        """Duplicate and backward stamps are a timestamp-check finding; they
        must not turn ordinary motion into an excursion."""
        ts = _uniform_ts(150)
        positions, yaws = _resting_hand(ts)
        bad_ts = ts.copy()
        bad_ts[40] = bad_ts[39]  # duplicate
        bad_ts[80] = bad_ts[78]  # backwards
        result = check_tracking_excursions(_poses(positions, yaws), bad_ts)

        assert result["events"] == []
        json.dumps(result)

    def test_non_monotonic_timestamps_with_a_glitch_do_not_raise(self):
        poses, ts = _glitch_episode()
        ts = ts.copy()
        ts[61] = ts[60]
        ts[64] = ts[62]

        result = check_tracking_excursions(poses, ts)

        assert isinstance(result["events"], list)
        _assert_plain_json(result)

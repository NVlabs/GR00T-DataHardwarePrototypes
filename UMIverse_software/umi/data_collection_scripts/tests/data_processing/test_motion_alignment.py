# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the arm-motion alignment check (motion_alignment.py).

Run with: uv run pytest tests/data_processing/test_motion_alignment.py -v
"""

import numpy as np
import pytest

from real_scripts.data_processing.data_check import (
    aperture_velocity_cells,
    check_motion_alignment,
    combine_flow_cells,
    compute_flow_cell_speeds,
    estimate_flow_offset,
    estimate_flow_offset_rank,
    motion_bursts,
    per_burst_offsets,
    pose_speeds,
    pose_stall_flags,
)

BURST_CENTERS = (5.0, 12.0, 19.0, 26.0)


def _speed_profile(t):
    """Bursty speed profile: Gaussian motion bursts on a quiet floor."""
    t = np.asarray(t, dtype=np.float64)
    s = np.full_like(t, 0.02)
    for tc in BURST_CENTERS:
        s = s + np.exp(-0.5 * ((t - tc) / 0.25) ** 2)
    return s


def _synthetic_pair(offset_s=0.06, duration_s=32.0, cam_fps=30.0, ref_fps=30.0):
    """Pose speed + flow speed where the visual motion is offset_s LATER."""
    ref_ts = np.arange(0.0, duration_s, 1.0 / ref_fps)
    cam_ts = np.arange(0.0, duration_s, 1.0 / cam_fps)
    return cam_ts, _speed_profile(cam_ts - offset_s), ref_ts, _speed_profile(ref_ts)


def _rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _poses_from_profile(ts, omega_profile, v_dir=(1.0, 0.0, 0.0)):
    """(N, 4, 4) poses rotating about z and translating along v_dir with the
    given angular-speed profile (translation speed = 0.1 * omega)."""
    dt = np.diff(ts, prepend=ts[0])
    theta = np.cumsum(omega_profile * dt)
    dist = np.cumsum(0.1 * omega_profile * dt)
    poses = np.zeros((len(ts), 4, 4))
    for i in range(len(ts)):
        poses[i, :3, :3] = _rotation_z(theta[i])
        poses[i, :3, 3] = dist[i] * np.asarray(v_dir)
        poses[i, 3, 3] = 1.0
    return poses


class TestPoseSpeeds:
    def test_constant_rotation_rate_recovered(self):
        ts = np.arange(0.0, 10.0, 1.0 / 30.0)
        poses = _poses_from_profile(ts, np.full(len(ts), 0.5))
        omega, v = pose_speeds(poses, ts)
        assert np.isnan(omega[0]) and np.isnan(v[0])
        assert np.allclose(omega[1:], 0.5, atol=1e-6)
        assert np.allclose(v[1:], 0.05, atol=1e-6)

    def test_non_finite_pose_yields_nan(self):
        ts = np.arange(0.0, 2.0, 1.0 / 30.0)
        poses = _poses_from_profile(ts, np.full(len(ts), 0.5))
        poses[10, 0, 0] = np.nan
        omega, _ = pose_speeds(poses, ts)
        assert np.isnan(omega[10]) and np.isnan(omega[11])
        assert np.isfinite(omega[12])

    def test_stall_flags(self):
        ts = np.arange(0.0, 4.0, 1.0 / 30.0).copy()
        poses = _poses_from_profile(ts, np.full(len(ts), 0.5))
        poses[20:26] = poses[20]  # frozen pose stream
        ts[60:] += 0.5  # delivery gap
        flags = pose_stall_flags(poses, ts)
        assert flags["max_gap_s"] > 0.5
        assert flags["n_gaps_over_threshold"] == 1
        assert flags["repeated_pose_frac"] > 0.03


class TestMotionBursts:
    def test_finds_all_bursts(self):
        ts = np.arange(0.0, 32.0, 1.0 / 30.0)
        bursts = motion_bursts(ts, _speed_profile(ts), min_speed=0.3)
        assert len(bursts) == len(BURST_CENTERS)
        for burst, tc in zip(bursts, BURST_CENTERS):
            assert abs(burst["t_peak"] - tc) < 0.1

    def test_idle_signal_has_no_bursts(self):
        ts = np.arange(0.0, 10.0, 1.0 / 30.0)
        assert motion_bursts(ts, np.full(len(ts), 0.01), min_speed=0.3) == []


class TestEstimateFlowOffset:
    def test_positive_offset_recovered(self):
        cam_ts, flow, ref_ts, ref = _synthetic_pair(0.06)
        est = estimate_flow_offset(cam_ts, flow, ref_ts, ref)
        assert est["valid"]
        assert abs(est["offset_s"] - 0.06) < 0.01
        assert est["peak_correlation"] > 0.9

    def test_negative_offset_recovered(self):
        cam_ts, flow, ref_ts, ref = _synthetic_pair(-0.08)
        est = estimate_flow_offset(cam_ts, flow, ref_ts, ref)
        assert est["valid"]
        assert abs(est["offset_s"] + 0.08) < 0.01

    def test_aligned_signals_measure_zero(self):
        cam_ts, flow, ref_ts, ref = _synthetic_pair(0.0)
        est = estimate_flow_offset(cam_ts, flow, ref_ts, ref)
        assert est["valid"]
        assert abs(est["offset_s"]) < 0.01

    def test_offset_correct_on_bimodal_frame_arrival(self):
        """Real UMI cameras deliver frames in a ~17ms/~33ms bimodal pattern
        averaging 45fps. The lag scan shifts indices, so it must convert with
        the mean spacing -- the median is 22% lower here, and using it would
        scale every reported offset down by that much.
        """
        duration, offset = 32.0, 0.1
        steps = np.tile([0.0172, 0.0333], int(duration / 0.0505) + 1)
        cam_ts = np.concatenate([[0.0], np.cumsum(steps)])
        cam_ts = cam_ts[cam_ts < duration]
        assert np.median(np.diff(cam_ts)) / np.mean(np.diff(cam_ts)) < 0.85
        ref_ts = np.arange(0.0, duration, 1.0 / 30.0)
        est = estimate_flow_offset(
            cam_ts,
            _speed_profile(cam_ts - offset),
            ref_ts,
            _speed_profile(ref_ts),
        )
        assert est["valid"]
        assert abs(est["offset_s"] - offset) < 0.015, (
            f"recovered {est['offset_s'] * 1000:.1f}ms of a {offset * 1000:.0f}ms offset"
        )

    def test_different_rates_handled(self):
        cam_ts, flow, ref_ts, ref = _synthetic_pair(0.05, cam_fps=30.0, ref_fps=60.0)
        est = estimate_flow_offset(cam_ts, flow, ref_ts, ref)
        assert est["valid"]
        assert abs(est["offset_s"] - 0.05) < 0.01

    def test_flat_signal_invalid(self):
        cam_ts, _, ref_ts, ref = _synthetic_pair(0.0)
        est = estimate_flow_offset(cam_ts, np.zeros(len(cam_ts)), ref_ts, ref)
        assert not est["valid"]


class TestEstimateFlowOffsetRank:
    def test_offset_recovered(self):
        cam_ts, flow, ref_ts, ref = _synthetic_pair(0.07)
        bursts = motion_bursts(ref_ts, ref, min_speed=0.3)
        est = estimate_flow_offset_rank(cam_ts, flow, ref_ts, ref, bursts)
        assert est["valid"]
        assert abs(est["offset_s"] - 0.07) < 0.015

    def test_monotone_warp_invariance(self):
        # A monotone response (square) shifts derivative shapes but not ranks.
        cam_ts, flow, ref_ts, ref = _synthetic_pair(0.07)
        bursts = motion_bursts(ref_ts, ref, min_speed=0.3)
        est = estimate_flow_offset_rank(cam_ts, flow**2, ref_ts, ref, bursts)
        assert est["valid"]
        assert abs(est["offset_s"] - 0.07) < 0.015

    def test_no_bursts_invalid(self):
        cam_ts, flow, ref_ts, ref = _synthetic_pair(0.05)
        est = estimate_flow_offset_rank(cam_ts, flow, ref_ts, ref, [])
        assert not est["valid"]


class TestCombineFlowCells:
    def test_selects_correlated_cells_only(self):
        rng = np.random.default_rng(0)
        cam_ts, flow, ref_ts, ref = _synthetic_pair(0.0)
        n = len(cam_ts)
        cells = rng.normal(0.0, 0.3, size=(n, 2, 2))
        cells[:, 0, 0] = flow + rng.normal(0.0, 0.05, n)
        cells[:, 1, 1] = flow + rng.normal(0.0, 0.05, n)
        combo = combine_flow_cells(cam_ts, cells, ref, min_cells=2)
        assert combo is not None
        assert combo["n_cells"] == 2
        assert combo["cell_corr"][0, 0] > 0.5
        assert abs(combo["cell_corr"][0, 1]) < 0.3

    def test_uncorrelated_cells_return_none(self):
        rng = np.random.default_rng(1)
        cam_ts, _, _, ref = _synthetic_pair(0.0)
        cells = rng.normal(0.0, 0.3, size=(len(cam_ts), 2, 2))
        assert combine_flow_cells(cam_ts, cells, ref) is None


class TestPerBurstOffsets:
    def test_median_matches_injected_offset(self):
        cam_ts, flow, ref_ts, ref = _synthetic_pair(0.06)
        bursts = motion_bursts(ref_ts, ref, min_speed=0.3)
        results = per_burst_offsets(cam_ts, flow, ref_ts, ref, bursts)
        assert len(results) == len(BURST_CENTERS)
        confident = [
            r["offset_s"]
            for r in results
            if r["valid"] and r["peak_correlation"] >= 0.5
        ]
        assert len(confident) >= 3
        assert abs(float(np.median(confident)) - 0.06) < 0.015


def _write_rotating_video(path, ts, omega_profile, size=(160, 120), fps=30):
    """Video of a textured plane rotating about the image center with the
    given angular-speed profile — flow magnitude tracks |omega|."""
    cv2 = pytest.importorskip("cv2")

    rng = np.random.default_rng(42)
    w, h = size
    big = rng.integers(0, 255, size=(h * 2, w * 2), dtype=np.uint8)
    big = cv2.GaussianBlur(big, (7, 7), 0)  # give Farneback gradients to grip
    dt = np.diff(ts, prepend=ts[0])
    theta = np.cumsum(omega_profile * dt)

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter cannot open mp4 output")
    for angle in theta:
        rot = cv2.getRotationMatrix2D((w, h), np.degrees(angle), 1.0)
        frame = cv2.warpAffine(big, rot, (w * 2, h * 2))
        crop = frame[h // 2 : h // 2 + h, w // 2 : w // 2 + w]
        writer.write(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR))
    writer.release()


class TestComputeFlowCellSpeeds:
    def test_flow_tracks_rotation_profile(self, tmp_path):
        fps = 30.0
        ts = np.arange(0.0, 20.0, 1.0 / fps)
        profile = _speed_profile(ts)
        video = tmp_path / "cam.mp4"
        _write_rotating_video(video, ts, profile, fps=int(fps))

        cells = compute_flow_cell_speeds(str(video), ts, downscale_width=160).motion
        assert cells.shape[0] == len(ts)
        mean_speed = np.nanmean(cells.reshape(len(ts), -1), axis=1)
        finite = np.isfinite(mean_speed)
        assert np.mean(finite) > 0.9
        corr = np.corrcoef(mean_speed[finite], profile[finite])[0, 1]
        assert corr > 0.8

    def test_frame_indices_with_repeats_and_skips(self, tmp_path):
        fps = 30.0
        ts_native = np.arange(0.0, 8.0, 1.0 / fps)
        profile = _speed_profile(ts_native)
        video = tmp_path / "cam.mp4"
        _write_rotating_video(video, ts_native, profile, fps=int(fps))

        indices = np.array([0, 2, 2, 5, 6, 9, 9, 9, 12, 20, 21, 30])
        # Aligned timelines are strictly increasing even where nearest-frame
        # matching repeats a native frame.
        ts = np.arange(len(indices)) / fps
        cells = compute_flow_cell_speeds(
            str(video), ts, frame_indices=indices, stride=1
        ).motion
        assert cells.shape[0] == len(indices)
        assert np.all(np.isnan(cells[0]))
        # Repeated native frames over a positive dt are exactly zero motion.
        assert np.nanmax(cells[2]) == 0.0
        assert np.nanmax(cells[6]) == 0.0
        assert np.isfinite(cells[1]).any()

    def test_stride_preserves_signal_timing(self, tmp_path):
        """Striding must not shift the signal in time.

        Samples are timestamped at their interval midpoint, so a strided run
        lines up with an unstrided one. Labelling them at the interval end
        instead would shift the signal by (stride - 1) / 2 samples, which this
        catches: 4 whole samples at stride 9.
        """
        fps = 30.0
        ts = np.arange(0.0, 20.0, 1.0 / fps)
        profile = _speed_profile(ts)
        video = tmp_path / "cam.mp4"
        _write_rotating_video(video, ts, profile, fps=int(fps))

        def mean_speed(stride):
            cells = compute_flow_cell_speeds(str(video), ts, stride=stride).motion
            x = np.nanmean(cells.reshape(len(ts), -1), axis=1)
            finite = np.isfinite(x)
            x[~finite] = np.interp(
                np.flatnonzero(~finite), np.flatnonzero(finite), x[finite]
            )
            return (x - x.mean()) / x.std()

        base = mean_speed(1)
        lags = np.arange(-6, 7)
        for stride in (2, 3, 5, 9):
            strided = mean_speed(stride)
            corr = np.array(
                [
                    np.mean(strided[k:] * base[: len(base) - k])
                    if k >= 0
                    else np.mean(strided[:k] * base[-k:])
                    for k in lags
                ]
            )
            j = int(np.argmax(corr))
            peak = float(lags[j])
            if 0 < j < len(corr) - 1:  # parabolic sub-sample refinement
                denom = corr[j - 1] - 2 * corr[j] + corr[j + 1]
                if abs(denom) > 1e-12:
                    peak += 0.5 * (corr[j - 1] - corr[j + 1]) / denom
            assert abs(peak) < 1.0, f"stride={stride} shifted by {peak:.2f} samples"
            assert corr[j] > 0.85, f"stride={stride} lost the signal"

    def test_invalid_stride_rejected(self):
        ts = np.arange(0.0, 2.0, 1.0 / 30.0)
        with pytest.raises(ValueError, match="stride"):
            compute_flow_cell_speeds("unused.mp4", ts, stride=0)


class TestCheckArmMotionAlignment:
    def _build_episode(self, tmp_path, video_delay_s):
        """Raw-mode episode: iPhone pose + wrist camera whose visual motion
        happens video_delay_s later than the pose says."""
        zarr = pytest.importorskip("zarr")
        fps = 30.0
        duration = 32.0
        ts = np.arange(0.0, duration, 1.0 / fps)
        profile = _speed_profile(ts)

        root = zarr.open(str(tmp_path), mode="w")
        root.attrs["latency"] = {"left_tracking_camera": 0.0, "left_umi_camera": 0.0}

        tracker = root.create_group("left_tracking_camera")
        tracker.create_array("pose", data=_poses_from_profile(ts, profile))
        tracker.create_array("capture_time", data=ts)
        tracker.create_array("receive_time", data=ts)

        cam = root.create_group("left_umi_camera")
        cam.create_array("receive_time", data=ts)
        _write_rotating_video(
            tmp_path / "left_umi_camera.mp4",
            ts,
            _speed_profile(ts - video_delay_s),
            fps=int(fps),
        )
        return zarr.open(str(tmp_path), mode="r")

    def test_clean_episode_passes(self, tmp_path):
        episode = self._build_episode(tmp_path, video_delay_s=0.0)
        result = check_motion_alignment(episode, str(tmp_path), "", {})
        assert result["mode"] == "raw"
        pair = result["cameras"]["left_umi_camera"]["pairs"]["left_tracking_camera"]
        assert pair["measurable"]
        assert pair["verified"]
        assert abs(pair["global"]["offset_s"]) < 0.03
        assert result["passed"]

    def test_delayed_video_warns_and_does_not_fail(self, tmp_path):
        """The offset limit is only validated on some hardware, so a measured
        offset is a warning, never a failure."""
        episode = self._build_episode(tmp_path, video_delay_s=0.15)
        result = check_motion_alignment(episode, str(tmp_path), "", {})
        pair = result["cameras"]["left_umi_camera"]["pairs"]["left_tracking_camera"]
        assert pair["measurable"]
        assert pair["global"]["offset_s"] > 0.1
        assert not pair["passed"]
        assert result["passed"]
        assert result["issues"] == []
        assert any("misalignment" in w for w in result["warnings"])

    def test_gross_delay_fails_via_pose_tracking_gate(self, tmp_path):
        # A delay far beyond max_lag_s leaves no offset the estimator can
        # measure, so the zero-lag gate is what catches it.
        episode = self._build_episode(tmp_path, video_delay_s=1.5)
        result = check_motion_alignment(episode, str(tmp_path), "", {})
        pair = result["cameras"]["left_umi_camera"]["pairs"]["left_tracking_camera"]
        assert pair["pose_tracking_corr"] is not None
        assert pair["pose_tracking_corr"] < 0.35
        assert pair.get("pose_not_tracking")
        assert not pair["passed"]
        assert not result["passed"]
        assert any("does not track" in i for i in result["issues"])

    def test_pose_tracking_gate_clean_on_a_synced_episode(self, tmp_path):
        episode = self._build_episode(tmp_path, video_delay_s=0.0)
        result = check_motion_alignment(episode, str(tmp_path), "", {})
        pair = result["cameras"]["left_umi_camera"]["pairs"]["left_tracking_camera"]
        assert pair["pose_tracking_corr"] > 0.35
        assert not pair.get("pose_not_tracking")

    def test_gate_ignores_cells_that_do_not_track_the_arm(self, tmp_path):
        """A wrist camera whose view is filled by an independently-moving
        object must not fail the gate. Cell selection drops those cells; the
        gate has to use the selected ones, not the all-cell mean.

        For example, a shirt covering most of the view can drag the all-cell
        mean to ~0.0 while the selected cells agree at 0.77-0.86.
        """
        episode = self._build_episode(tmp_path, video_delay_s=0.0)
        result = check_motion_alignment(episode, str(tmp_path), "", {})
        pair = result["cameras"]["left_umi_camera"]["pairs"]["left_tracking_camera"]
        assert pair["pose_tracking_basis"] == "selected_cells"
        assert not pair.get("pose_not_tracking")

    def test_gate_falls_back_to_all_cells_when_none_select(self, tmp_path):
        """No cells selecting is the dead-pose signature, so the gate must
        still produce a number rather than silently going quiet."""
        episode = self._build_episode(tmp_path, video_delay_s=1.5)
        result = check_motion_alignment(episode, str(tmp_path), "", {})
        pair = result["cameras"]["left_umi_camera"]["pairs"]["left_tracking_camera"]
        assert pair["pose_tracking_corr"] is not None
        assert pair.get("pose_not_tracking")

    def test_pose_tracking_gate_does_not_fire_on_a_clean_episode(self, tmp_path):
        # The gate must not be the thing that quarantines healthy data.
        episode = self._build_episode(tmp_path, video_delay_s=0.0)
        result = check_motion_alignment(episode, str(tmp_path), "", {})
        pair = result["cameras"]["left_umi_camera"]["pairs"]["left_tracking_camera"]
        assert not pair.get("pose_not_tracking")
        assert result["passed"]

    def test_no_tracking_device_skips(self, tmp_path):
        zarr = pytest.importorskip("zarr")
        root = zarr.open(str(tmp_path / "empty"), mode="w")
        cam = root.create_group("left_umi_camera")
        cam.create_array("receive_time", data=np.arange(10.0))
        episode = zarr.open(str(tmp_path / "empty"), mode="r")
        result = check_motion_alignment(episode, str(tmp_path / "empty"), "", {})
        assert result["skip_reason"] == "No tracking device with pose data"
        assert result["passed"]


def _write_gripper_video(path, aperture, fps=30):
    """Frames where a finger-colored bar fills a fraction of the view.

    ``aperture[i]`` in [0, 1] is how open the gripper is, so the finger bar
    covers ``1 - aperture`` of the frame -- fingers fill the view as it closes.
    """
    cv2 = pytest.importorskip("cv2")
    h, w = 120, 160
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter cannot open mp4 output")
    rng = np.random.default_rng(0)
    texture = rng.integers(0, 60, size=(h, w, 3), dtype=np.uint8)
    for a in aperture:
        frame = texture.copy()
        rows = int(np.clip(1.0 - a, 0.0, 1.0) * h)
        if rows:
            frame[h - rows :, :] = (200, 120, 40)  # BGR -> hue ~105, in range
        writer.write(frame)
    writer.release()


class TestGripperAlignment:
    def _build(self, tmp_path, gripper_delay_s):
        """Raw episode: a gripper that opens and closes, and a camera whose
        fingers move ``gripper_delay_s`` later than the sensor says."""
        zarr = pytest.importorskip("zarr")
        fps, duration = 30.0, 32.0
        ts = np.arange(0.0, duration, 1.0 / fps)

        def aperture(t):
            # Smooth open/close cycles. A real gripper takes a few hundred ms
            # to travel, and zero-lag cell selection needs the two velocity
            # signals to still overlap at the offset being measured.
            return 0.5 + 0.4 * np.sin(2 * np.pi * t / 4.0)

        root = zarr.open(str(tmp_path), mode="w")
        root.attrs["latency"] = {"left_umi_gripper": 0.0, "left_umi_camera": 0.0}
        grip = root.create_group("left_umi_gripper")
        grip.create_array("distance", data=aperture(ts) * 80.0)
        grip.create_array("receive_time", data=ts)
        cam = root.create_group("left_umi_camera")
        cam.create_array("receive_time", data=ts)
        _write_gripper_video(
            tmp_path / "left_umi_camera.mp4",
            aperture(ts - gripper_delay_s),
            fps=int(fps),
        )
        return zarr.open(str(tmp_path), mode="r")

    def test_aperture_velocity_flips_sign_of_finger_fraction(self):
        ts = np.arange(0.0, 4.0, 1.0 / 30.0)
        # fingers steadily filling the view == gripper closing == aperture falling
        cells = np.linspace(0.0, 1.0, len(ts))[:, None, None] * np.ones((1, 2, 2))
        vel = aperture_velocity_cells(cells, ts)
        assert np.all(vel[1:-1] < 0)

    def test_synced_gripper_passes(self, tmp_path):
        episode = self._build(tmp_path, gripper_delay_s=0.0)
        result = check_motion_alignment(episode, str(tmp_path), "", {})
        pair = result["cameras"]["left_umi_camera"]["gripper"]["left_umi_gripper"]
        assert pair["measurable"], pair.get("skip_reason")
        assert abs(pair["global"]["offset_s"]) < 0.05
        assert pair["passed"]

    def test_delayed_gripper_warns_and_does_not_fail(self, tmp_path):
        episode = self._build(tmp_path, gripper_delay_s=0.12)
        result = check_motion_alignment(episode, str(tmp_path), "", {})
        pair = result["cameras"]["left_umi_camera"]["gripper"]["left_umi_gripper"]
        assert pair["measurable"], pair.get("skip_reason")
        assert pair["global"]["offset_s"] > 0.08
        assert not pair["passed"]
        assert result["passed"]
        assert any("camera-gripper" in w for w in result["warnings"])

    def test_confidence_reported_on_a_good_capture(self, tmp_path):
        episode = self._build(tmp_path, gripper_delay_s=0.0)
        result = check_motion_alignment(episode, str(tmp_path), "", {})
        pair = result["cameras"]["left_umi_camera"]["gripper"]["left_umi_gripper"]
        conf = pair["confidence"]
        assert conf["trusted"], conf["reasons"]
        assert conf["n_usable_events"] >= 3
        assert conf["event_spread_s"] is not None

    def test_too_few_actuations_is_skipped_not_guessed(self, tmp_path):
        """One open/close cycle cannot support an offset, so the pair is
        reported and skipped rather than given a verdict."""
        zarr = pytest.importorskip("zarr")
        fps, duration = 30.0, 32.0
        ts = np.arange(0.0, duration, 1.0 / fps)
        # a single slow cycle over the whole capture
        aperture = 0.5 + 0.4 * np.sin(2 * np.pi * ts / duration)
        root = zarr.open(str(tmp_path), mode="w")
        root.attrs["latency"] = {"left_umi_gripper": 0.0, "left_umi_camera": 0.0}
        grip = root.create_group("left_umi_gripper")
        grip.create_array("distance", data=aperture * 80.0)
        grip.create_array("receive_time", data=ts)
        cam = root.create_group("left_umi_camera")
        cam.create_array("receive_time", data=ts)
        _write_gripper_video(tmp_path / "left_umi_camera.mp4", aperture, fps=int(fps))
        episode = zarr.open(str(tmp_path), mode="r")

        result = check_motion_alignment(episode, str(tmp_path), "", {})
        pair = result["cameras"]["left_umi_camera"]["gripper"]["left_umi_gripper"]
        assert not pair["measurable"]
        assert not pair["confidence"]["trusted"]
        assert pair.get("skip_reason")
        assert result["passed"]  # a skip must never fail the episode

    def test_no_gripper_group_is_not_an_error(self, tmp_path):
        episode = self._build(tmp_path, gripper_delay_s=0.0)
        result = check_motion_alignment(episode, str(tmp_path), "", {})
        assert "gripper" in result["cameras"]["left_umi_camera"]

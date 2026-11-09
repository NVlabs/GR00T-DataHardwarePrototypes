# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-frame state machine in CameraRelativeCalibrationCheck."""

from __future__ import annotations

import inspect
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from real_scripts.data_collection.pre_episode_checks.camera_calibration import (
    CameraRelativeCalibrationCheck,
)
from umi.common.geometry.pose_calibration import estimate_pose
from umi.common.recording.pre_episode_checks import PreEpisodeContext


def _make_check(
    cameras=("ref", "left", "right"),
    align_stable_required=3,
    loop_timeout_sec=10.0,
    run_every_n_episodes=10,
):
    return CameraRelativeCalibrationCheck(
        reference_camera="ref",
        cameras=list(cameras),
        run_every_n_episodes=run_every_n_episodes,
        marker={"mode": "aruco", "marker_size": 0.1875},
        align_t_mm=5.0,
        align_r_deg=0.5,
        align_stable_required=align_stable_required,
        loop_timeout_sec=loop_timeout_sec,
    )


def _make_ctx(episode_id=0):
    # `get_source` returning None skips the _pause_calibration_streams path in
    # tick() — keeps these state-machine tests focused on convergence
    # accounting without simulating a live recorder stream.
    rm = SimpleNamespace(
        episode_id=episode_id,
        last_calibration_data=None,
        get_latest_frames=lambda: {},
        get_source=lambda name: None,
        pause_source_streaming=lambda name: None,
        resume_source_streaming=lambda name: None,
        push_frame=lambda name, frame: None,
    )
    return PreEpisodeContext(latest_frames={}, cfg={}, recorder_manager=rm)


def test_is_due_initially_true_then_false_after_finalize():
    chk = _make_check()
    assert chk.is_due(0) is True
    chk._cached = {"ref_camera_name": "ref", "cameras": ["ref"]}
    chk._last_calib_episode = 0
    assert chk.is_due(0) is False
    assert chk.is_due(9) is False
    assert chk.is_due(10) is True


def test_start_session_resets_per_session_state():
    chk = _make_check(align_stable_required=3)
    chk._stable_count = {"left": 2, "right": 3}
    chk._latest_capture = {"ref": {"sentinel": 1}}
    chk._latest_calib = {"left": np.eye(4)}
    chk._session_iters = 42
    chk.start_session()
    assert chk.is_session_active() is True
    assert chk._stable_count == {"left": 0, "right": 0}
    assert chk._latest_capture == {}
    assert chk._latest_calib == {}
    assert chk._session_iters == 0


def test_tick_with_no_session_is_noop():
    chk = _make_check()
    ctx = _make_ctx()
    chk.tick(ctx)
    assert chk._session_iters == 0
    assert chk.is_session_active() is False


def test_tick_increments_iters_when_session_active():
    chk = _make_check()
    chk.start_session()
    ctx = _make_ctx()
    with patch.object(chk, "_capture_one", return_value=(None, "no marker")):
        chk.tick(ctx)
        chk.tick(ctx)
    assert chk._session_iters == 2


def test_is_session_timed_out_after_loop_timeout():
    chk = _make_check(loop_timeout_sec=0.05)
    chk.start_session()
    assert chk.is_session_timed_out() is False
    time.sleep(0.06)
    assert chk.is_session_timed_out() is True


def test_finalize_session_publishes_cache_and_marks_converged():
    chk = _make_check(align_stable_required=1)
    chk.start_session()

    def fake_capture(_name):
        return {"frame": SimpleNamespace(pose=None), "T_marker_optical": np.eye(4)}

    chk._latest_capture = {n: fake_capture(n) for n in chk.cameras}
    ctx = _make_ctx(episode_id=4)
    chk._finalize_session(ctx)
    assert chk._cached is not None
    assert chk._last_calib_episode == 4
    assert ctx.recorder_manager.last_calibration_data == [chk._cached]
    # is_session_active() flips to False on convergence, so later tick()
    # calls do nothing.
    assert chk.is_session_active() is False
    # is_session_converged() takes over so draw() can show the DONE overlay
    # while the operator is still in the CALIBRATION stage.
    assert chk.is_session_converged() is True
    assert chk._session_start is not None  # session window not yet closed


def test_tick_is_noop_after_finalize():
    """Once converged, tick() must short-circuit even if called again before
    reset_session() -- otherwise the DONE state would re-run the convergence
    loop on every visualizer frame."""
    chk = _make_check(cameras=("ref",), align_stable_required=1)
    chk.start_session()
    chk._converged = True  # simulate post-finalize state
    iters_before = chk._session_iters
    ctx = _make_ctx(episode_id=0)
    chk.tick(ctx)
    chk.tick(ctx)
    assert chk._session_iters == iters_before


def test_reset_session_clears_converged_flag():
    chk = _make_check()
    chk._session_start = 1.0
    chk._converged = True
    chk.reset_session()
    assert chk.is_session_converged() is False
    assert chk.is_session_active() is False


def test_run_is_pure_gate_no_blocking_loop():
    chk = _make_check()
    ctx = _make_ctx(episode_id=0)
    ok, msg = chk.run(ctx)
    assert ok is False
    assert "not ready" in msg.lower()
    chk._cached = {"ref_camera_name": "ref", "cameras": ["ref"]}
    chk._last_calib_episode = 0
    ok, msg = chk.run(ctx)
    assert ok is True
    assert ctx.recorder_manager.last_calibration_data == [chk._cached]


def test_reset_session_clears_active_session():
    chk = _make_check()
    chk.start_session()
    chk._stable_count = {"left": 2, "right": 1}
    chk.reset_session()
    assert chk.is_session_active() is False
    assert all(v == 0 for v in chk._stable_count.values())


def test_tick_converges_when_repeated_capture_is_stable():
    """End-to-end: tick() with controlled stable mock captures should hit
    align_stable_required and finalize the session."""
    chk = _make_check(cameras=("ref", "left"), align_stable_required=2)
    chk.start_session()
    ctx = _make_ctx(episode_id=0)

    fake_frame = SimpleNamespace(pose=None)
    stable_marker_pose = np.eye(4)
    stable_marker_pose[:3, 3] = [0.1, 0.2, 0.3]

    def fake_capture_one(_ctx, name):
        return {
            "frame": fake_frame,
            "T_marker_optical": stable_marker_pose.copy(),
        }, None

    with patch.object(chk, "_capture_one", side_effect=fake_capture_one):
        for _ in range(5):
            chk.tick(ctx)
            if not chk.is_session_active():
                break

    assert chk.is_session_active() is False, "expected finalize to clear session"
    assert chk._cached is not None
    assert chk._last_calib_episode == 0
    assert ctx.recorder_manager.last_calibration_data == [chk._cached]


def test_capture_one_skips_stale_frame():
    """A frame whose `capture_time` matches the last-processed value for the
    same camera should be reported as 'stale' and produce no new capture."""
    chk = _make_check()
    fake_frame = SimpleNamespace(pose=None, capture_time=42.0)
    ctx = SimpleNamespace(
        latest_frames={"ref": [fake_frame]},
        recorder_manager=SimpleNamespace(episode_id=0),
    )
    chk._last_capture_time["ref"] = 42.0
    cap, err = chk._capture_one(ctx, "ref")
    assert cap is None
    assert err == "stale frame"


def test_capture_one_processes_fresh_frame_then_marks_stale():
    """First call with a new capture_time runs detection; immediate second
    call with the same capture_time short-circuits."""
    chk = _make_check()
    fake_frame = SimpleNamespace(pose=None, capture_time=7.0)
    ctx = SimpleNamespace(
        latest_frames={"ref": [fake_frame]},
        recorder_manager=SimpleNamespace(episode_id=0),
    )
    with patch(
        "real_scripts.data_collection.pre_episode_checks.camera_calibration.estimate_pose",
        return_value=(np.eye(4), None),
    ):
        cap_first, _ = chk._capture_one(ctx, "ref")
        cap_second, err_second = chk._capture_one(ctx, "ref")
    assert cap_first is not None
    assert chk._last_capture_time["ref"] == 7.0
    assert cap_second is None
    assert err_second == "stale frame"


def _lens_cam(model, coeffs, capture_time=1.0):
    return SimpleNamespace(
        distortion_model=model, distortion=coeffs, pose=None, capture_time=capture_time
    )


def _lens_ctx(**cameras):
    """Context whose get_source returns objects carrying a lens, and whose
    latest_frames yields a frame for each (the non-paused capture path)."""
    return SimpleNamespace(
        latest_frames={n: [c] for n, c in cameras.items()},
        recorder_manager=SimpleNamespace(get_source=lambda n: cameras.get(n)),
    )


def _lens_passed_to_estimate_pose(chk, ctx, name):
    """The `camera` estimate_pose was handed for one _capture_one call."""
    with patch(
        "real_scripts.data_collection.pre_episode_checks.camera_calibration.estimate_pose",
        return_value=(np.eye(4), None),
    ) as spy:
        chk._capture_one(ctx, name)
    cam = spy.call_args.kwargs["camera"]
    return getattr(cam, "distortion_model", None), getattr(cam, "distortion", None)


def test_each_camera_solves_with_its_own_lens():
    """One check spans cameras with different lenses — the station console adds
    optional cameras to `cameras`, so a fisheye wrist cam and a rectilinear head
    cam share these params. Any single shared value would be wrong for one."""
    chk = _make_check(cameras=("ref", "head"))
    ctx = _lens_ctx(
        ref=_lens_cam("fisheye", np.array([-0.0134, -0.0046, 0.0, 0.0])),
        head=_lens_cam("pinhole", None),
    )
    # One call per camera: a second call for the same camera short-circuits on
    # the stale-frame guard and never reaches estimate_pose.
    model, coeffs = _lens_passed_to_estimate_pose(chk, ctx, "ref")
    assert model == "fisheye"
    assert coeffs is not None
    assert _lens_passed_to_estimate_pose(chk, ctx, "head") == ("pinhole", None)


def test_a_lens_in_the_marker_config_has_no_effect():
    """The lens is not configurable. A stray key must not reach the solver —
    silently mis-modelling a camera is the worst outcome available, since the
    wrong projection costs ~20 mm of board pose while raising nothing."""
    chk = _make_check(cameras=("ref",))
    chk.marker["distortion_model"] = "none"
    cam = _lens_cam("fisheye", np.array([-0.0134, -0.0046, 0.0, 0.0]))
    assert _lens_passed_to_estimate_pose(chk, _lens_ctx(ref=cam), "ref")[0] == "fisheye"
    assert "distortion_model" not in inspect.signature(estimate_pose).parameters, (
        "estimate_pose must take the lens from the camera, not a config key"
    )


def test_start_and_reset_session_clear_last_capture_time():
    chk = _make_check()
    chk._last_capture_time = {"ref": 1.0, "left": 2.0}
    chk.start_session()
    assert chk._last_capture_time == {}
    chk._last_capture_time = {"ref": 3.0}
    chk.reset_session()
    assert chk._last_capture_time == {}

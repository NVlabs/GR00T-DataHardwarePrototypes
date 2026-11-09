# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for umi.common.utility.reference_camera — reference camera sync."""

from umi.common.utility import reference_camera as ref_cam

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _calibration_params(cfg):
    for check in cfg.get("pre_episode_checks", []):
        if check["name"] == "camera.relative_calibration":
            return check["params"]
    return None


def _make_config(reference_camera, video_source_names, calibration_cameras=None):
    """Build a minimal recording config."""
    cfg = {
        "meta": {"gripper_pipeline": "v4"},
        "reference_camera": reference_camera,
        "video_sources": [{"name": n} for n in video_source_names],
    }
    if calibration_cameras is not None:
        cfg["pre_episode_checks"] = [
            {"name": "umi_device.uart_connected"},
            {
                "name": "camera.relative_calibration",
                "params": {
                    "reference_camera": reference_camera,
                    "cameras": list(calibration_cameras),
                },
            },
        ]
    return cfg


# ---------------------------------------------------------------------------
# sync_reference_camera_to_video_sources — core function
# ---------------------------------------------------------------------------


def test_head_mount_disabled_falls_back_to_right_tracking():
    cfg = _make_config(
        reference_camera="head_tracking_camera",
        video_source_names=[
            "left_tracking_camera",
            "right_tracking_camera",
            "left_umi_camera",
            "right_umi_camera",
        ],
        calibration_cameras=[
            "head_tracking_camera",
            "left_tracking_camera",
            "right_tracking_camera",
        ],
    )
    changed = ref_cam.sync_reference_camera_to_video_sources(cfg)
    assert changed is True
    assert cfg["reference_camera"] == "right_tracking_camera"
    params = _calibration_params(cfg)
    assert params["reference_camera"] == "right_tracking_camera"
    assert "head_tracking_camera" not in params["cameras"]
    assert "right_tracking_camera" in params["cameras"]
    assert "left_tracking_camera" in params["cameras"]


def test_already_valid_is_noop():
    cfg = _make_config(
        reference_camera="right_tracking_camera",
        video_source_names=["left_tracking_camera", "right_tracking_camera"],
        calibration_cameras=["left_tracking_camera", "right_tracking_camera"],
    )
    changed = ref_cam.sync_reference_camera_to_video_sources(cfg)
    assert changed is False
    assert cfg["reference_camera"] == "right_tracking_camera"


def test_prefers_a_head_oak_over_a_tracking_camera():
    cfg = _make_config(
        reference_camera="head_tracking_camera",
        video_source_names=[
            "left_tracking_camera",
            "right_tracking_camera",
            "head_oak_camera",
        ],
        calibration_cameras=[
            "head_tracking_camera",
            "left_tracking_camera",
            "right_tracking_camera",
        ],
    )
    ref_cam.sync_reference_camera_to_video_sources(cfg)
    assert cfg["reference_camera"] == "head_oak_camera"
    params = _calibration_params(cfg)
    assert "head_tracking_camera" not in params["cameras"]
    assert "head_oak_camera" in params["cameras"]


def test_calibration_camera_list_pruned():
    cfg = _make_config(
        reference_camera="right_tracking_camera",
        video_source_names=["left_tracking_camera", "right_tracking_camera"],
        calibration_cameras=[
            "right_tracking_camera",
            "left_tracking_camera",
            "head_tracking_camera",
        ],
    )
    changed = ref_cam.sync_reference_camera_to_video_sources(cfg)
    assert changed is True
    params = _calibration_params(cfg)
    assert "head_tracking_camera" not in params["cameras"]
    assert "left_tracking_camera" in params["cameras"]
    assert "right_tracking_camera" in params["cameras"]


def test_empty_config_is_noop():
    assert ref_cam.sync_reference_camera_to_video_sources({}) is False


def test_none_config_is_noop():
    assert ref_cam.sync_reference_camera_to_video_sources(None) is False


def test_no_pre_episode_checks_only_updates_reference_camera():
    cfg = {
        "reference_camera": "head_tracking_camera",
        "video_sources": [{"name": "right_tracking_camera"}],
    }
    changed = ref_cam.sync_reference_camera_to_video_sources(cfg)
    assert changed is True
    assert cfg["reference_camera"] == "right_tracking_camera"


# ------------------------------------------- oak_device streams are cameras


def _oak_ffc_config(reference_camera):
    """A rig with no source-level camera name at all.

    Two OAK boards, four cameras, every one of them a `streams[]` entry. No
    shipped recording config looks like this -- every rig also carries a phone
    or a USB camera, which does have a source-level `name`, so a reader that
    skipped streams would still find something.
    """
    return {
        "meta": {"gripper_pipeline": "v4"},
        "reference_camera": reference_camera,
        "video_sources": [
            {
                "type": "oak_device",
                "oak_device_name": "oak-FFC-10",
                "streams": [
                    {"name": "left_trigger_camera", "socket": "CAM_A"},
                    {"name": "left_umi_camera", "socket": "CAM_B"},
                ],
            },
            {
                "type": "oak_device",
                "oak_device_name": "oak-FFC-05",
                "streams": [
                    {"name": "right_trigger_camera", "socket": "CAM_A"},
                    {"name": "right_umi_camera", "socket": "CAM_B"},
                ],
            },
        ],
    }


def test_an_oak_devices_streams_are_camera_names():
    """They are what the mp4s, the zarr groups and `get_source` are keyed on."""
    names = ref_cam.video_source_names(_oak_ffc_config("right_umi_camera"))

    assert names == {
        "left_trigger_camera",
        "left_umi_camera",
        "right_trigger_camera",
        "right_umi_camera",
    }


def test_a_source_level_name_and_its_streams_both_count():
    cfg = {
        "video_sources": [
            {"type": "iphone_camera", "name": "right_tracking_camera"},
            {"type": "oak_device", "streams": [{"name": "right_umi_camera"}]},
        ]
    }

    assert ref_cam.video_source_names(cfg) == {
        "right_tracking_camera",
        "right_umi_camera",
    }


def test_a_wrist_camera_reference_survives_record_time_validation():
    """RecorderManager._validate_reference_camera aborts the session at
    construction when the reference names no recorded camera. An all-oak_device
    rig must count its streams as recorded cameras, or every reference such a
    config could carry would be rejected and the preset would be unusable."""
    from umi.common.recording.record_manager import RecorderManager

    rm = RecorderManager.__new__(RecorderManager)
    rm.config = _oak_ffc_config("right_umi_camera")

    rm._validate_reference_camera()  # must not raise


def test_a_reference_naming_no_camera_at_all_still_fails_loud():
    """The widening must not turn the guard off. Non-vacuity for the test above."""
    import pytest

    from umi.common.recording.record_manager import RecorderManager

    rm = RecorderManager.__new__(RecorderManager)
    rm.config = _oak_ffc_config("head_oak_camera")

    with pytest.raises(ValueError, match="head_oak_camera"):
        rm._validate_reference_camera()


def test_a_head_camera_outranks_the_arm_cameras_in_the_repair_preference():
    """The repair path has to agree with `ConfigAssembler._reference_rank`, which
    ranks the head camera first. Disagreeing would move the anchor to a camera
    the picker never offered."""
    assert (
        ref_cam.pick_reference_camera(
            ["left_umi_camera", "head_oak_camera", "right_tracking_camera"]
        )
        == "head_oak_camera"
    )


def test_a_rig_with_no_tracking_camera_repairs_to_a_wrist_not_alphabetically():
    """`bimanual_v4_vive`'s camera set: its pose comes from pucks, not cameras.
    Without wrist/trigger entries in the preference the fallback is
    `sorted(names)[0]` -- `left_trigger_camera`, i.e. alphabetical order
    standing in for a world frame."""
    assert (
        ref_cam.pick_reference_camera(
            [
                "left_trigger_camera",
                "left_umi_camera",
                "right_trigger_camera",
                "right_umi_camera",
            ]
        )
        == "right_umi_camera"
    )

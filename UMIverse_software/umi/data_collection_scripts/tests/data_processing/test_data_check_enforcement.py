# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real failed measurements survive warn-only policy and remain reportable."""

import numpy as np
import pytest
import zarr

from real_scripts.data_processing.data_check.dispatch import check_episode
from real_scripts.data_processing.data_check.report import data_check_findings


def _episode():
    episode = zarr.open_group(store=zarr.storage.MemoryStore(), mode="w")
    times = np.arange(90, dtype=float) / 10
    for name in ("left_gripper", "left_tracking_camera"):
        device = episode.create_group(name)
        device.create_array("receive_time", data=times)
        if "gripper" in name:
            device.create_array("distance", data=np.full(len(times), 100.0))
        else:
            device.create_array("pose", data=np.tile(np.eye(4), (len(times), 1, 1)))
    return episode


@pytest.mark.parametrize(
    "section", ["timestamp", "gripper", "tracking", "device_name_consistency"]
)
def test_failed_section_still_runs_but_does_not_reject_when_unchecked(section):
    episode = _episode()
    check_config = {section: {"enabled": True}}
    if section == "device_name_consistency":
        check_config[section]["right_numeric_sources"] = ["right_gripper"]
    enforced = check_episode(episode, {}, check_config)
    assert not enforced["passed"]
    assert enforced["issues"]
    relaxed = check_episode(
        episode, {"check_enforcement": {"data": {section: False}}}, check_config
    )
    assert relaxed["passed"]
    assert relaxed["issues"] == []
    assert relaxed["warnings"] == [
        f"[not enforced] {issue}" for issue in enforced["issues"]
    ]
    for key in ("timestamp_checks", "sensor_checks", "tracking_checks", "device_check"):
        np.testing.assert_equal(relaxed[key], enforced[key])
    assert data_check_findings(relaxed)["warnings"] == relaxed["warnings"]


def test_relaxing_one_section_preserves_failures_in_other_sections():
    result = check_episode(
        _episode(),
        {"check_enforcement": {"data": {"gripper": False}}},
        {"gripper": {"enabled": True}, "tracking": {"enabled": True}},
    )
    assert not result["passed"]
    assert result["issues"] and all(
        "left_tracking_camera" in s for s in result["issues"]
    )
    assert result["warnings"] and all("left_gripper" in s for s in result["warnings"])


def test_inline_data_check_enforce_false_matches_gui_policy():
    result = check_episode(
        _episode(), {}, {"timestamp": {"enabled": True, "enforce": False}}
    )
    assert result["passed"]
    assert not result["timestamp_checks"]["fps"]["left_gripper"]["passed"]
    assert result["warnings"]


def test_video_and_motion_failures_can_warn_independently(monkeypatch, tmp_path):
    from real_scripts.data_processing.data_check import dispatch, motion_alignment

    monkeypatch.setattr(
        dispatch,
        "get_device_expected_resolutions",
        lambda cfg: {"left_tracking_camera": (640, 480)},
    )
    monkeypatch.setattr(
        dispatch,
        "check_video_resolution",
        lambda *args, **kwargs: {
            "passed": False,
            "actual_resolution": (320, 240),
            "expected_resolution": (640, 480),
        },
    )
    monkeypatch.setattr(
        motion_alignment,
        "check_motion_alignment",
        lambda *args, **kwargs: {
            "passed": False,
            "mode": "test",
            "cameras": {},
            "tracking": {},
            "notes": [],
            "issues": ["camera motion mismatch"],
            "warnings": ["offset uncertain"],
        },
    )
    checks = {
        "video_resolution": {"enabled": True},
        "motion_alignment": {"enabled": True},
    }
    result = check_episode(
        _episode(),
        {"check_enforcement": {"data": {"video_resolution": False}}},
        checks,
        data_dir=str(tmp_path),
    )
    assert not result["passed"]
    assert result["issues"] == ["Motion: camera motion mismatch"]
    assert result["warnings"][0] == "Motion: offset uncertain"
    assert result["warnings"][1].startswith(
        "[not enforced] left_tracking_camera: resolution"
    )
    assert not result["resolution_checks"]["left_tracking_camera"]["passed"]
    result = check_episode(
        _episode(),
        {
            "check_enforcement": {
                "data": {"video_resolution": False, "motion_alignment": False}
            }
        },
        checks,
        data_dir=str(tmp_path),
    )
    assert result["passed"]
    assert not result["motion_alignment_checks"]["passed"]
    assert "[not enforced] Motion: camera motion mismatch" in result["warnings"]

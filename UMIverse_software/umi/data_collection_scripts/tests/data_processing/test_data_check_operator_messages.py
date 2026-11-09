# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every issue template check_episode can emit maps to an operator category.

The examples are the real templates from dispatch.py / numeric.py /
motion_alignment.py, filled with typical values. A new
template that matches no rule falls back to showing the raw text, which is
caught here so it gets an explicit HARDWARE/MOTION decision.
"""

import pytest

from real_scripts.data_processing.data_check.operator_messages import (
    HARDWARE,
    MOTION,
    explain_issue,
    overlay_hint,
    overlay_lines,
    terminal_lines,
    warning_lines,
)

MOTION_ISSUES = [
    "left_umi_gripper: did not move (range: 1.2mm)",
    "left_tracking_camera: static (max movement 3.1mm)",
    "left_tracking_camera: low consecutive movement (40.0%, need 60.0% > 1.0mm)",
    "left_hand.joint_pos: required axes did not move (axis 1: range 6.00 < 8.0)",
    "left_hand.joint_pos: no movement (max range 0.1 < 1.0)",
    "left_hand.joint_pos: start frame out of tolerance (3/5 axes in tol; worst axis 2: +0.40 > 0.10)",
]

HARDWARE_ISSUES = [
    "FPS deviation for left_umi_camera: 12.5%",
    "Non-monotonic timestamps for left_umi_camera: 3 violations",
    "Frame interval violations for left_umi_camera: 4 frames exceed 3x expected dt (1.20% of the episode missing, longest gap 180 ms) at frames [10, 11]",
    "Sustained frame drop for left_umi_camera: 1 window(s) of 2.0s with FPS below 80% of expected",
    "Low timestamp coverage: 82.0%",
    "Device consistency: Missing device names: ['right_umi_gripper']",
    "Device consistency: Non-standard device names: ['gripper_7']",
    "left_umi_camera: resolution (1280, 720) != expected (1280, 800)",
    "left_umi_camera: Failed to open video: /data/ep/left_umi_camera.mp4",
    "left_umi_camera: Video file not found: /data/ep/left_umi_camera.mp4",
    "left_umi_gripper: distance jumps detected (4 jumps > 800mm/s)",
    "left_umi_gripper: out of range (12 frames)",
    "head_tracking_camera: pose jumps detected (2 jumps > 4.5m/s)",
    "left_tracking_camera: confirmed tracker excursions (1 >= 20mm)",
    "Motion: left_umi_camera vs left_tracking_camera: camera-pose misalignment 80ms (expected 0ms on aligned timeline, residual 80ms > 50ms)",
    "Motion: left_umi_camera vs left_umi_gripper: camera-gripper misalignment 90ms (expected 0ms on aligned timeline, > 50ms)",
    "Motion: left_umi_camera vs left_tracking_camera: pose stream does not track the camera (zero-lag agreement 0.10 < 0.3) — the pose is not following the arm, so no time offset exists to measure",
    "left_hand.joint_pos: missing",
    "left_hand.joint_pos: registered but no chunks written (0/3)",
    "left_hand.joint_pos: 4 NaN values",
    "left_hand.joint_pos: out of range [-1.0, 1.0] (3 elements)",
    "left_hand.joint_pos: 2 jumps detected (threshold=0.5)",
    "left_hand.joint_pos: shape mismatch (expected (5,), got (6,))",
    "left_hand.joint_pos: start_within_tolerance config error (bad target)",
    "left_hand.joint_pos: start_within_tolerance requires both `target` and `tolerance`",
    "Check failed with error: KeyError: 'receive_time'",
]


@pytest.mark.parametrize("issue", MOTION_ISSUES)
def test_motion_issues(issue):
    explained = explain_issue(issue)
    assert explained.kind == MOTION
    assert explained.message != issue  # matched a rule, not the raw fallback


@pytest.mark.parametrize("issue", HARDWARE_ISSUES)
def test_hardware_issues(issue):
    explained = explain_issue(issue)
    assert explained.kind == HARDWARE
    assert explained.message != issue


def test_device_names_are_readable():
    e = explain_issue("left_umi_gripper: did not move (range: 1.2mm)")
    assert e.message == "left gripper never opened or closed"


def test_unknown_issue_is_shown_verbatim():
    e = explain_issue("data check did not finish within 60s")
    assert e.kind == HARDWARE
    assert e.message == "data check did not finish within 60s"


def test_terminal_groups_motion_before_hardware_and_keeps_raw():
    issues = [
        "left_umi_gripper: distance jumps detected (4 jumps > 800mm/s)",
        "left_tracking_camera: static (max movement 3.1mm)",
    ]
    text = "\n".join(terminal_lines(issues))
    assert text.index("MOTION issue") < text.index("HARDWARE issue")
    for raw in issues:
        assert f"(details: {raw})" in text


@pytest.mark.parametrize("issue", HARDWARE_ISSUES)
def test_hardware_action_is_check_then_swap(issue):
    action = explain_issue(issue).action
    assert action.startswith("Check ") and "if necessary" in action


def test_hardware_action_names_the_device():
    e = explain_issue("left_umi_gripper: distance jumps detected (4 jumps > 800mm/s)")
    assert e.action == "Check left gripper; swap it if necessary"


def test_check_name_matches_config_key():
    assert (
        explain_issue(
            "left_umi_gripper: distance jumps detected (4 jumps > 800mm/s)"
        ).check
        == "gripper.jumps"
    )
    assert (
        explain_issue("left_tracking_camera: static (max movement 3.1mm)").check
        == "tracking.pose_movement"
    )


def test_overlay_lines_are_tagged_and_hint_follows_kind():
    motion = ["left_tracking_camera: static (max movement 3.1mm)"]
    hardware = ["left_umi_gripper: out of range (12 frames)"]
    assert overlay_lines(motion + hardware)[0].startswith(
        "[MOTION] tracking.pose_movement: "
    )
    assert overlay_lines(hardware) == [
        "[HARDWARE] gripper.range: left gripper width reading out of range\n"
        "-> Check left gripper; swap it if necessary"
    ]
    assert "Redo the episode" in overlay_hint(motion)
    assert "swap it if necessary" in overlay_hint(motion + hardware)


def test_warning_lines_say_the_episode_was_not_failed_and_keep_raw():
    warnings = [
        "left_tracking_camera: confirmed tracker excursions (1 >= 20mm)",
        "Motion: left_umi_camera vs left_umi_gripper: camera-gripper misalignment 90ms (expected 0ms on aligned timeline, > 50ms)",
    ]
    text = "\n".join(warning_lines(warnings))
    assert "does not fail the episode" in text
    assert "[tracking.excursions]" in text and "[motion_alignment]" in text
    for raw in warnings:
        assert f"(details: {raw})" in text
    assert warning_lines([]) == []

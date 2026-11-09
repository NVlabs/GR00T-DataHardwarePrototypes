# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Board pose (T_board_<cam>) persisted in the calibration dict by
umi.common.geometry.pose_calibration.compute_relative_calibration.

Board pose = T_marker_optical (the camera's optical frame expressed in the board
frame), stored only for fixed depth cameras (frame carries `depth`); tracking
cameras (pose / no depth) are excluded.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from umi.common.geometry.pose_calibration import compute_relative_calibration


def _rigid(rx, ry, rz, t):
    """Small helper: rigid 4x4 from a rotation vector (deg-ish) and translation."""
    ca, cb, cc = np.cos([rx, ry, rz])
    sa, sb, sc = np.sin([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = t
    return T


def _depth_capture(T_marker_optical):
    # Fixed depth camera: no self-pose, carries depth.
    return {
        "frame": SimpleNamespace(pose=None, depth=np.zeros((2, 2), dtype=np.uint16)),
        "T_marker_optical": T_marker_optical,
    }


def _tracking_capture(T_marker_optical):
    # iPhone tracking camera: self-pose, no depth attribute at all.
    return {
        "frame": SimpleNamespace(pose=np.eye(4)),
        "T_marker_optical": T_marker_optical,
    }


def test_compute_relative_calibration_emits_board_pose_for_depth_cams():
    T_ref = _rigid(0.1, -0.2, 0.3, [0.1, 0.2, 0.9])
    T_left = _rigid(-0.3, 0.15, 0.05, [-0.4, 0.0, 1.0])
    captures = {
        "middle_depth_camera": _depth_capture(T_ref),
        "left_depth_camera": _depth_capture(T_left),
        "left_tracking_camera": _tracking_capture(np.eye(4)),
    }

    result = compute_relative_calibration("middle_depth_camera", captures)

    # Board pose for every depth camera, including the reference.
    assert "T_board_middle_depth_camera" in result
    assert "T_board_left_depth_camera" in result
    # Tracking camera excluded from board pose (but keeps its relative transform).
    assert "T_board_left_tracking_camera" not in result
    assert "T_left_tracking_camera" in result

    # Stored as-is: T_board_<cam> == that camera's T_marker_optical.
    np.testing.assert_allclose(result["T_board_middle_depth_camera"], T_ref)
    np.testing.assert_allclose(result["T_board_left_depth_camera"], T_left)


def test_board_pose_round_trips_to_relative_transform():
    """inv(T_board_ref) @ T_board_other reproduces the stored T_<other>."""
    T_ref = _rigid(0.2, 0.1, -0.1, [0.0, 0.3, 0.8])
    T_left = _rigid(0.05, -0.25, 0.2, [-0.5, 0.1, 1.1])
    captures = {
        "middle_depth_camera": _depth_capture(T_ref),
        "left_depth_camera": _depth_capture(T_left),
    }

    result = compute_relative_calibration("middle_depth_camera", captures)

    T_board_ref = np.asarray(result["T_board_middle_depth_camera"])
    T_board_other = np.asarray(result["T_board_left_depth_camera"])
    reconstructed = np.linalg.inv(T_board_ref) @ T_board_other

    np.testing.assert_allclose(
        reconstructed, np.asarray(result["T_left_depth_camera"]), atol=1e-9
    )

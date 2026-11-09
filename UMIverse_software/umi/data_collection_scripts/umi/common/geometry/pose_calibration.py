# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared multi-camera pose-calibration math.

Used by the `camera.relative_calibration` pre-episode check.

Naming convention: T_A_B = pose of B's origin/axes in A's coordinate frame
(so T_A_B @ x_B = x_A, and T_A_B @ T_B_C = T_A_C).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from umi.common.geometry.camera_position import (
    get_camera_position_and_orientation,
    get_camera_position_and_orientation_charuco,
)
from umi.common.utility.pose import invert_transformation

# ARKit camera axes use +Y up and viewing direction -Z; optical axes use
# +Y down and +Z forward. The frames share an origin and differ by a
# self-inverse 180-degree X rotation.
T_cam_optical: np.ndarray = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def has_self_pose(capture: dict) -> bool:
    """A capture has self-pose if its frame carries a non-None `pose` (ARKit-style)."""
    return getattr(capture["frame"], "pose", None) is not None


def compute_T_world_a_world_b(ca: dict, cb: dict) -> np.ndarray:
    """Compute T_w_a_w_b (pose of cam_b's world in cam_a's world) via the marker.

    Each camera has three relevant frames:
      - `w`:       the camera's persistent world frame.
                   self-pose (iPhone): ARKit tracking world.
                   otherwise:          same as `cam` (no separate world).
      - `cam`:     the camera frame as reported by self-tracking.
                   iPhone: Apple/ARKit convention (Y up).
                   otherwise: same as `optical` (OpenCV convention).
      - `optical`: the OpenCV-style optical sensor frame where marker detection
                   runs. `capture["T_marker_optical"]` = T_marker_optical.

    Frame relationships:
      T_w_cam       = `frame.pose` if self-pose else I       (live tracking)
      T_cam_optical = `T_cam_optical` if self-pose else I    (static axis convention)

    Formula (chains world → cam → optical → marker → optical → cam → world):
        T_w_a_w_b = T_w_a_cam_a
                    @ T_cam_a_optical_a
                    @ inv(T_marker_optical_a)
                    @ T_marker_optical_b
                    @ T_optical_b_cam_b
                    @ T_cam_b_w_b

    For two non-self-pose cameras, all `T_w_cam` and `T_cam_optical` factors
    collapse to identity and this reduces to plain optical-to-optical via
    the marker. For two iPhones, it gives the relative pose between their
    ARKit tracking worlds at the moment of capture.
    """
    eye4 = np.eye(4)
    T_w_a_cam_a = ca["frame"].pose if has_self_pose(ca) else eye4
    T_cam_b_w_b = invert_transformation(cb["frame"].pose) if has_self_pose(cb) else eye4
    T_cam_a_optical_a = T_cam_optical if has_self_pose(ca) else eye4
    T_optical_b_cam_b = (
        invert_transformation(T_cam_optical) if has_self_pose(cb) else eye4
    )
    return (
        T_w_a_cam_a
        @ T_cam_a_optical_a
        @ invert_transformation(ca["T_marker_optical"])
        @ cb["T_marker_optical"]
        @ T_optical_b_cam_b
        @ T_cam_b_w_b
    )


def estimate_pose(
    frame_data: Any, marker_cfg: dict, viz: bool = False, camera: Any = None
):
    """Dispatch to ArUco or Charuco estimator. Returns (T_marker_optical, vis_image).

    `marker_cfg.mode` is "aruco" (depth-based, requires depth) or "charuco"
    (RGB+intrinsics only). Other keys are passed through to the underlying
    detectors.

    `camera` is the device the frame came from. Its lens (`distortion_model` /
    `distortion`, from oak_device.FALLBACK_LENS) drives the ChArUco pose solve.
    The lens is deliberately not configurable: it is a property of the physical
    module, and one caller routinely spans modules with different lenses, so any
    single config value would be wrong for one of them.
    """
    mode = marker_cfg.get("mode", "aruco").lower()
    if mode == "aruco":
        return get_camera_position_and_orientation(
            frame_data,
            marker_size=marker_cfg["marker_size"],
            viz=viz,
        )
    if mode == "charuco":
        kwargs = dict(
            squares_x=marker_cfg.get("squares_x", 6),
            squares_y=marker_cfg.get("squares_y", 6),
            square_length=marker_cfg["square_length"],
            marker_length=marker_cfg["marker_length"],
            min_markers=marker_cfg.get("min_markers", 6),
            viz=viz,
        )
        model = getattr(camera, "distortion_model", None)
        if model:
            kwargs["distortion_model"] = model
            kwargs["dist_coeffs"] = getattr(camera, "distortion", None)
        if "aruco_dict" in marker_cfg:
            kwargs["aruco_dict"] = marker_cfg["aruco_dict"]
        return get_camera_position_and_orientation_charuco(frame_data, **kwargs)
    raise ValueError(f"Unknown marker.mode: {marker_cfg.get('mode')!r}")


def compute_relative_calibration(
    reference_camera: str,
    captures: dict[str, dict],
) -> dict:
    """Build a single-component consolidated calibration dict.

    `captures` maps camera name -> {"frame": Frame, "T_marker_optical": 4x4}.
    Returns a dict matching the existing zarr `attrs["calibration"]` element
    schema:
        {
          "ref_camera_name": str,
          "cameras": [str, ...],     # sorted, includes ref
          "T_<name>": [[...4x4...]], # one per non-ref camera; pose of <name>'s
                                     # world in the reference camera's world
                                     # (T_w_ref_w_other; see
                                     # compute_T_world_a_world_b for semantics).
          "T_board_<name>": [[4x4]], # one per fixed depth camera (incl. ref);
                                     # T_marker_optical, the camera's optical
                                     # frame expressed in the board frame.
        }

    All transforms are stored as nested Python lists (JSON-friendly).
    """
    if reference_camera not in captures:
        raise ValueError(
            f"reference_camera {reference_camera!r} missing from captures "
            f"(have: {sorted(captures)})"
        )
    ref_cap = captures[reference_camera]
    cameras = sorted(captures)
    result: dict = {
        "ref_camera_name": reference_camera,
        "cameras": cameras,
    }
    for name, cap in captures.items():
        if name == reference_camera:
            continue
        T = compute_T_world_a_world_b(ref_cap, cap)
        result[f"T_{name}"] = np.asarray(T).tolist()
    # Store fixed depth-camera poses as T_marker_optical: optical frames
    # expressed in the board frame. Exclude tracked iPhone camera poses.
    for name, cap in captures.items():
        if getattr(cap.get("frame"), "depth", None) is None:
            continue
        result[f"T_board_{name}"] = np.asarray(cap["T_marker_optical"]).tolist()
    return result

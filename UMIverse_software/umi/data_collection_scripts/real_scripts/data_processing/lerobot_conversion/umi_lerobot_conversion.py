#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "av",
#     "numpy",
#     "pandas",
#     "pyarrow",
#     "pyyaml",
#     # torchcodec is built against one torch release: upgrade the two together.
#     "torch==2.14.*",
#     "torchcodec==0.17.*",
#     "zarr>=3",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchcodec = { index = "pytorch-cpu" }
# ///
"""UMI V4 aligned-Zarr to LeRobot conversion, on local disk.

Reads aligned sessions from ``<root>/aligned/<task>/<session>`` and writes each
one to ``<root>/lerobot_format/<task>/<embodiment>.<session>``, where ``<root>``
is ``$UMI_OUTPUT_ROOT`` or else ``output/`` at the repository root
(--output-root overrides both). --input-dataset converts one session, to
--output-dataset if given. Nothing is downloaded or uploaded. The embodiment is
detected per session from the streams it recorded unless --embodiment-tag names
one.

The grippers are UMI V4. Their tracker->eef offset and full-open width are the
v2 gripper's values (UMI_T_TRACKER_EEF, GRIPPER_FULL_OPEN_MM).

Each dataset is a standard LeRobot-style folder with parquet, mp4, and meta
files.

A station has these dependencies already: the `data_collection` group in
pyproject.toml installs them (CPU-only torch), and the console's Data Audit page
runs this script with the station's interpreter ("Convert to LeRobot"). From a
station's repo checkout:

    uv run --group data_collection python \
        real_scripts/data_processing/lerobot_conversion/umi_lerobot_conversion.py \
        --task-names <task folder>

Anywhere else, `uv run <this file>` installs the dependencies above into an
environment of its own. Keep the two lists in step
(tests/station_ui/test_audit_page.py checks it).

Video streams (per RGB camera, unless --no-video-variants):
- ``<cam>.rgb``                    -- native resolution, as recorded
- ``<cam>_320x240_cropped.rgb``    -- center-cropped to 4:3, then resized to
  320x240 (undistorted, narrower field of view)
- ``<cam>_320x240_resized.rgb``    -- whole frame resized to 320x240 (full field
  of view, aspect ratio not preserved)
- ``<cam>_320x240_stereo_rectified.rgb`` -- fisheye (AR0234) streams only, unless
  --no-stereo-variant: stereographic reprojection from the native frame
  (square pixels, wide field of view)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import warnings
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import cpu_count, get_context
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
import yaml
import zarr


def spawn_process_pool(max_workers: int) -> ProcessPoolExecutor:
    """Create a process pool with isolated native-library state.

    Forked workers can inherit inconsistent native-library state (notably
    PyArrow), so conversion and statistics workers start in clean
    interpreters.
    """
    # Spawned workers inherit this and import torch lazily, so each one's
    # intra-op pool (stereo variant) gets a fair share of the cores, not all.
    os.environ.setdefault(
        "OMP_NUM_THREADS", str(max(1, cpu_count() // max(1, max_workers)))
    )
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=get_context("spawn"),
    )


# UMI V4 mount transform from the iPhone tracking-camera frame to the end effector.
UMI_T_TRACKER_EEF = np.array(
    [
        [1.0, 0.0, 0.0, -0.13462],
        [0.0, 1.0, 0.0, -0.039546],
        [0.0, 0.0, 1.0, -0.044184],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

# Vive tracker-to-end-effector mount transform, shared by both arms.
# Use it for SHARED_WORLD_TRACKING_EMBODIMENTS instead of the iPhone mount.
UMI_T_VIVE_TRACKER_EEF = np.array(
    [
        [0.003527, -0.999821, 0.018615, -0.006669],
        [0.999993, 0.003531, 0.000168, -0.078868],
        [-0.000235, 0.018614, 0.999827, -0.057893],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

VIDEO_KEYS_BY_EMBODIMENT = {
    "gripper_umi": [
        "left_umi_camera",
        "right_umi_camera",
    ],
    "gripper_umi_trigger_camera": [
        "left_umi_camera",
        "right_umi_camera",
        "left_trigger_camera",
        "right_trigger_camera",
    ],
    "gripper_umi_trigger_head_camera": [
        "left_umi_camera",
        "right_umi_camera",
        "left_trigger_camera",
        "right_trigger_camera",
        "head_tracking_camera",
    ],
    "gripper_umi_trigger_head_camera_w_rels": [
        "left_umi_camera",
        "right_umi_camera",
        "left_trigger_camera",
        "right_trigger_camera",
        "head_tracking_camera",
    ],
    # Passive head RGB stream with per-frame intrinsics and no head pose.
    "gripper_umi_trigger_head_oak_camera": [
        "left_umi_camera",
        "right_umi_camera",
        "left_trigger_camera",
        "right_trigger_camera",
        "head_oak_camera",
    ],
    # Track arms with Vive numeric pose groups instead of iPhone SLAM groups.
    "gripper_umi_trigger_head_oak_camera_vive": [
        "left_umi_camera",
        "right_umi_camera",
        "left_trigger_camera",
        "right_trigger_camera",
        "head_oak_camera",
    ],
    # Vive tracking with wrist and trigger video streams; no head camera.
    "gripper_umi_trigger_camera_vive": [
        "left_umi_camera",
        "right_umi_camera",
        "left_trigger_camera",
        "right_trigger_camera",
    ],
    # Passive upward-facing table camera (OV9782, 640x400). It has no pose
    # stream or AR0234 fisheye stereo variant.
    "gripper_umi_trigger_head_oak_bottom_oak_camera": [
        "left_umi_camera",
        "right_umi_camera",
        "left_trigger_camera",
        "right_trigger_camera",
        "head_oak_camera",
        "bottom_oak_camera",
    ],
    "gripper_umi_trigger_two_head_camera_w_rels": [
        "left_umi_camera",
        "right_umi_camera",
        "left_trigger_camera",
        "right_trigger_camera",
        "head_umi_camera",
    ],
}

# Reject near-zero rotation blocks during conversion; they represent dropped
# poses and are invalid inputs to scipy Rotation.
ZERO_ROTATION_NORM_TOL = 1e-8

TRACKING_TO_EEF = {
    "left_tracking_camera": "left_arm",
    "right_tracking_camera": "right_arm",
}

# Vive rigs record tracker poses as ``*_tracking_pose`` numeric groups (one per
# lighthouse puck) rather than ``*_tracking_camera`` SLAM groups.
VIVE_TRACKING_TO_EEF = {
    "left_tracking_pose": "left_arm",
    "right_tracking_pose": "right_arm",
}


RAW_TRACKER_KEY = {
    "left_arm": "left_tracker.eef",
    "right_arm": "right_tracker.eef",
    "head": "head_tracker.eef",
}

HEAD_RELATIVE_EMBODIMENTS = {
    "gripper_umi_trigger_head_camera",
    "gripper_umi_trigger_head_camera_w_rels",
    "gripper_umi_trigger_two_head_camera_w_rels",
}

ARM_TO_ARM_RELATIVE_EMBODIMENTS = {
    "gripper_umi",
    "gripper_umi_trigger_camera",
    "gripper_umi_trigger_head_camera_w_rels",
    "gripper_umi_trigger_head_oak_camera",
    "gripper_umi_trigger_head_oak_camera_vive",
    "gripper_umi_trigger_camera_vive",
    "gripper_umi_trigger_head_oak_bottom_oak_camera",
    "gripper_umi_trigger_two_head_camera_w_rels",
}

# Rigs whose trackers all report poses in one shared world (Vive lighthouse):
# arm-to-arm relative pose is a direct rebase, with no per-episode calibration
# transform to lift through.
SHARED_WORLD_TRACKING_EMBODIMENTS = {
    "gripper_umi_trigger_head_oak_camera_vive",
    "gripper_umi_trigger_camera_vive",
}

HEAD_RELATIVE_REQUIRED_EMBODIMENTS = {
    "gripper_umi_trigger_head_camera_w_rels",
    "gripper_umi_trigger_two_head_camera_w_rels",
}

GRIPPER_KEYS = {
    "left_umi_gripper": "left_gripper",
    "right_umi_gripper": "right_gripper",
}

# The only gripper this converter supports. Recorded in meta/episodes.jsonl and
# meta/info.json; the session's config.yaml is not consulted for it.
GRIPPER_VERSION = "v4"
# UMI wrist-to-grasp-site offset between the fingers.
GRASP_OFFSET_M = 0.1347
# Full-open UMI V4 gripper width used to normalize measured distances.
GRIPPER_FULL_OPEN_MM = 85.55

# Datasets are written under <root>/LOCAL_DATASETS_DIR/<task>/; the station
# console's Data Audit page looks for them there.
LOCAL_DATASETS_DIR = "lerobot_format"

# meta/info.json and meta/embodiment.json robot_type.
ROBOT_TYPE = "gripper_umi"


@dataclass
class SessionResult:
    task: str
    session: str
    status: str
    input_dataset: str | None = None
    local_output: str | None = None
    selected_episodes: list[str] | None = None
    error: str | None = None
    episode_clips: str | None = None


@dataclass(frozen=True)
class EpisodeConversionSpec:
    episode_name: str
    episode_index: int
    frame_start_index: int
    task_description: str
    task_index: int


# Per-episode half-open [clip_start, clip_end) intervals, computed by the
# sibling episode_clip.py module.
EPISODE_CLIPS_REL_PATH = "meta/episode_clips.jsonl"
# A task-text correction recorded beside an aligned session, and the record of
# applying it that ships with the dataset. See read_language_override.
LANGUAGE_OVERRIDE_FILENAME = "language_override.json"
LANGUAGE_CORRECTION_REL_PATH = "meta/language_correction.json"
LANGUAGE_FIELDS = ("task_description", "operator_instruction")
EPISODE_CLIPS_ARM_KEY = re.compile(r"^([a-z]+)_arm\.eef$")
SOURCE_EPISODE_RE = re.compile(r"^episode_(\d+)$")

# Per-episode identity marker, e.g. `meta/source_episodes/episode_000003.episode_5.json`.
# `episode_index` is a *position* in the selected episode list, so the same index
# means a different source episode whenever the selection changes. The marker's
# *name* records which source episode each output index was built from.
SOURCE_EPISODES_DIR = "meta/source_episodes"


def tracking_to_eef_for_embodiment(embodiment_tag: str) -> dict[str, str]:
    if embodiment_tag in SHARED_WORLD_TRACKING_EMBODIMENTS:
        return dict(VIVE_TRACKING_TO_EEF)
    tracking = dict(TRACKING_TO_EEF)
    if embodiment_tag in HEAD_RELATIVE_EMBODIMENTS:
        tracking["head_tracking_camera"] = "head"
    return tracking


def rgb_video_keys_for_embodiment(embodiment_tag: str) -> list[str]:
    """RGB video camera keys for an embodiment.

    These are the candidate cameras for per-frame intrinsics: the data-driven
    intrinsics path emits a ``{cam}.intrinsics`` column for any of these whose
    zarr group records an ``intrinsics`` array (the head OAK), and skips those
    without one (UMI / trigger).
    """
    return list(VIDEO_KEYS_BY_EMBODIMENT.get(embodiment_tag, []))


def normalize_episode_keys(raw: Any) -> set[str]:
    if not isinstance(raw, (list, tuple, set)):
        return set()
    out = set()
    for item in raw:
        text = str(item)
        out.add(text if text.startswith("episode_") else f"episode_{text}")
    return out


def source_episode_sort_key(name: str) -> tuple[int, str]:
    match = SOURCE_EPISODE_RE.match(str(name))
    if match:
        return int(match.group(1)), str(name)
    return sys.maxsize, str(name)


def normalize_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): normalize_metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_metadata(v) for v in value]
    if hasattr(value, "item"):
        try:
            return normalize_metadata(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return normalize_metadata(value.tolist())
        except Exception:
            pass
    return value


def infer_task_session(dataset: str) -> tuple[str, str]:
    parts = list(Path(dataset).expanduser().resolve().parts)
    if "aligned" in parts:
        aligned_index = len(parts) - 1 - parts[::-1].index("aligned")
        if len(parts) > aligned_index + 2:
            return parts[aligned_index + 1], parts[aligned_index + 2]
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return "dataset", parts[-1] if parts else "unknown"


def default_output_root() -> Path:
    """The station's output folder: ``$UMI_OUTPUT_ROOT``, else ``output/`` at the repo root.

    Mirrors ``umi.common.utility.output_layout.output_root``, which this
    standalone script does not import.
    """
    env = os.environ.get("UMI_OUTPUT_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent / "output"
    return Path.cwd() / "output"


def list_local_sessions(aligned_root: Path, task: str) -> list[str]:
    """Sessions of ``task`` under the local ``aligned/`` tree with at least one aligned episode."""
    task_dir = aligned_root / task
    if not task_dir.is_dir():
        raise FileNotFoundError(
            f"No aligned sessions for task {task!r}: {task_dir} does not exist"
        )
    sessions = []
    for path in sorted(task_dir.iterdir()):
        if not path.is_dir():
            continue
        if not (path / "zarr.json").is_file():
            print(
                f"Skipping {path}: no zarr.json, the session was never aligned",
                flush=True,
            )
        elif not any(path.glob("episode_*/zarr.json")):
            print(f"Skipping {path}: no aligned episodes", flush=True)
        else:
            sessions.append(path.name)
    return sessions


def filter_by_date(
    sessions: list[str], from_date: str | None, to_date: str | None
) -> list[str]:
    out = []
    for session in sessions:
        day = session[:10]
        if from_date and day < from_date:
            continue
        if to_date and day > to_date:
            continue
        out.append(session)
    return out


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim == 2:
        return matrix[:2, :].reshape(6)
    return matrix[:, :2, :].reshape(-1, 6)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    arr = np.asarray(rot6d, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, 6)
    rows = arr.reshape(-1, 2, 3)
    b1 = rows[:, 0]
    b2 = rows[:, 1]
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    b2 = b2 - (b1 * b2).sum(axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    out = np.zeros((arr.shape[0], 3, 3), dtype=np.float64)
    out[:, 0, :] = b1
    out[:, 1, :] = b2
    out[:, 2, :] = b3
    return out


def xyz_rot6d_to_transform(rows: Any) -> np.ndarray:
    arr = np.stack([np.asarray(row, dtype=np.float64) for row in rows])
    out = np.zeros((arr.shape[0], 4, 4), dtype=np.float64)
    out[:, :3, :3] = rot6d_to_matrix(arr[:, 3:9])
    out[:, :3, 3] = arr[:, :3]
    out[:, 3, 3] = 1.0
    return out


def transform_to_xyz_rot6d(transform: np.ndarray) -> list[list[float]]:
    return np.concatenate(
        [transform[:, :3, 3], matrix_to_rot6d(transform[:, :3, :3])],
        axis=1,
    ).tolist()


def arm_eef_to_grasp_eef(arm: np.ndarray) -> np.ndarray:
    """(N, 9) xyz + rot6d (first two ROWS of R) -> (N, 9) same convention, point moved 0.1347 m along local -Z."""
    arm = np.asarray(arm, dtype=np.float64)
    r1, r2 = arm[:, 3:6], arm[:, 6:9]
    r3 = np.cross(r1, r2)
    # translation in local frame t_local = (0, 0, -d); world offset = R @ t_local = -d * R[:, 2] (third COLUMN)
    col2 = np.stack([r1[:, 2], r2[:, 2], r3[:, 2]], axis=1)
    xyz = arm[:, :3] - GRASP_OFFSET_M * col2
    return np.concatenate([xyz, r1, r2], axis=1)  # rotation unchanged


def gripper_distance_to_normalized(distance: np.ndarray) -> np.ndarray:
    return np.asarray(distance, dtype=np.float64) / GRIPPER_FULL_OPEN_MM


def derived_columns(columns: Any) -> dict[str, list]:
    """``{side}_grasp.eef`` / ``{side}_gripper.distance_normalized`` from stored columns.

    ``columns`` is the episode's column mapping (the ``data`` dict being
    written), so the derived columns are computed from the same values that
    are stored next to them.
    """
    derived: dict[str, list] = {}
    for side in ("left", "right"):
        arm_key = f"{side}_arm.eef"
        if arm_key in columns:
            arm = np.asarray(list(columns[arm_key]), dtype=np.float64).reshape(-1, 9)
            derived[f"{side}_grasp.eef"] = arm_eef_to_grasp_eef(arm).tolist()
        distance_key = f"{side}_gripper.distance"
        if distance_key in columns:
            distance = np.asarray(list(columns[distance_key]), dtype=np.float64)
            derived[f"{side}_gripper.distance_normalized"] = (
                gripper_distance_to_normalized(distance).tolist()
            )
    return derived


def resolve_pose_interp_field(cam_group: Any) -> str | None:
    """Pick the tracker-pose field to read from a tracking-camera group.

    Prefers the aligned variant ``pose_interp_aligned`` over ``pose_interp``
    when present, so end-effector calculations use the pose in the calibrated
    frame. Returns ``None`` when neither is available.
    """
    if "pose_interp_aligned" in cam_group:
        return "pose_interp_aligned"
    if "pose_interp" in cam_group:
        return "pose_interp"
    return None


def verify_zarr_chunks_present(arr: Any) -> None:
    """Raise if any expected chunk file for a local zarr array is missing.

    Zarr silently fills missing chunks with ``fill_value`` (0.0 by default), so a
    truncated copy of a session becomes invisible at read time -- and a zeroed
    ``pose_interp`` matrix later crashes training in ``Rotation.from_matrix``.
    This turns the silent corruption into a loud conversion-time failure. No-op
    for non-local stores (nothing on disk to inspect).
    """
    store = getattr(arr, "store", None)
    store_root = getattr(store, "root", None)
    if store_root is None:
        return
    expected = 1
    for size, chunk in zip(arr.shape, arr.chunks, strict=False):
        expected *= math.ceil(size / chunk) if chunk else 1
    chunk_root = Path(store_root) / arr.path / "c"
    if not chunk_root.is_dir():
        if expected > 0:
            raise RuntimeError(
                f"Zarr array '{arr.path}': chunk dir missing at {chunk_root} "
                f"but expected {expected} chunk(s) for shape {arr.shape}."
            )
        return
    actual = sum(1 for path in chunk_root.rglob("*") if path.is_file())
    if actual < expected:
        raise RuntimeError(
            f"Zarr array '{arr.path}' (shape={arr.shape}, chunks={arr.chunks}): "
            f"expected {expected} chunk file(s), found {actual}. Missing chunks "
            f"would be silently filled with fill_value={arr.fill_value}."
        )


def verify_no_zero_pose_matrices(values: np.ndarray, qualified_key: str) -> None:
    """Raise if any 4x4 pose has a zero 3x3 rotation block (det=0).

    Catches tracker dropout / missing chunks written as zeros, which
    ``scipy.Rotation.from_matrix`` rejects during training. Points the failure
    at the exact source key/frames instead of surfacing 100k iterations later.
    """
    arr = np.asarray(values)
    if arr.ndim != 3 or arr.shape[-2:] != (4, 4):
        return
    rot = arr[:, :3, :3]
    rot_norm = np.linalg.norm(rot.reshape(rot.shape[0], -1), axis=1)
    bad = np.where(rot_norm < ZERO_ROTATION_NORM_TOL)[0]
    if bad.size:
        sample = bad[:10].tolist()
        raise RuntimeError(
            f"{qualified_key}: {bad.size}/{arr.shape[0]} frame(s) have a zero "
            f"rotation matrix (e.g. frames {sample}). Likely a missing zarr "
            f"chunk or upstream alignment failure."
        )


def tracker_eef_offsets(embodiment_tag: str) -> dict[str, np.ndarray]:
    """Tracker->eef transform per tracked body.

    Vive rigs use the puck mount's offset; every other rig the V4 gripper's
    tracking-camera offset.
    """
    if embodiment_tag in SHARED_WORLD_TRACKING_EMBODIMENTS:
        arm_offset = UMI_T_VIVE_TRACKER_EEF
    else:
        arm_offset = UMI_T_TRACKER_EEF
    offsets = {
        "left_arm": arm_offset,
        "right_arm": arm_offset,
    }
    if embodiment_tag in HEAD_RELATIVE_EMBODIMENTS:
        offsets["head"] = np.eye(4, dtype=np.float32)
    return offsets


def umi_geometry(embodiment_tag: str) -> dict[str, Any]:
    """The hardware geometry the pose and gripper columns are derived with.

    Written to meta/info.json.
    """
    if embodiment_tag in SHARED_WORLD_TRACKING_EMBODIMENTS:
        source = "vive tracker mount"
    else:
        source = GRIPPER_VERSION
    T_tracker_eef = tracker_eef_offsets(embodiment_tag)["left_arm"]
    return {
        "gripper_version": GRIPPER_VERSION,
        "T_tracker_eef": [
            [round(float(value), 7) for value in row] for row in T_tracker_eef
        ],
        "T_tracker_eef_source": source,
        "grasp_offset_m": GRASP_OFFSET_M,
        "gripper_full_open_mm": GRIPPER_FULL_OPEN_MM,
    }


def state_action_columns_for_embodiment(
    embodiment_tag: str,
) -> tuple[list[str], list[str]]:
    state = [
        "left_tracker.eef",
        "left_arm.eef",
        "left_grasp.eef",
        "left_gripper.distance",
        "left_gripper.distance_normalized",
        "right_tracker.eef",
        "right_arm.eef",
        "right_grasp.eef",
        "right_gripper.distance",
        "right_gripper.distance_normalized",
    ]
    action = [
        "left_arm.eef",
        "left_grasp.eef",
        "left_gripper.distance",
        "left_gripper.distance_normalized",
        "right_arm.eef",
        "right_grasp.eef",
        "right_gripper.distance",
        "right_gripper.distance_normalized",
    ]
    if embodiment_tag in HEAD_RELATIVE_EMBODIMENTS:
        state.extend(["head_tracker.eef", "head.eef"])
        action.append("head.eef")
    if embodiment_tag in HEAD_RELATIVE_REQUIRED_EMBODIMENTS:
        rels = [
            "left_arm_to_head.eef",
            "right_arm_to_head.eef",
            "right_arm_to_left_arm.eef",
        ]
        state.extend(rels)
        action.extend(rels)
    elif embodiment_tag in ARM_TO_ARM_RELATIVE_EMBODIMENTS:
        state.append("right_arm_to_left_arm.eef")
        action.append("right_arm_to_left_arm.eef")
    # Per-frame {cam}.intrinsics columns are observation-only and added
    # data-driven (see process_episode / finalize_dataset), not listed here.
    return state, action


def data_columns_for_embodiment(embodiment_tag: str) -> list[str]:
    state, _ = state_action_columns_for_embodiment(embodiment_tag)
    # ``.distance`` and ``.distance_normalized`` are both gripper columns.
    pose_columns = [col for col in state if ".distance" not in col]
    gripper_columns = [col for col in state if ".distance" in col]
    return [
        "timestamp",
        "frame_index",
        "episode_index",
        "task_index",
        *pose_columns,
        *gripper_columns,
        "index",
    ]


def verify_relative_pose_columns(
    embodiment_tag: str, data: dict[str, Any], episode_name: str
) -> None:
    """Raise if a schema-advertised relative pose column was not computed.

    The parquet writer keeps only columns present in ``data``, so a relative
    column that silently failed to compute (missing or unusable episode
    calibration) would ship a dataset whose ``modality.json`` advertises a
    feature the parquet does not contain.
    """
    state_columns, _ = state_action_columns_for_embodiment(embodiment_tag)
    missing = [col for col in state_columns if "_to_" in col and col not in data]
    if missing:
        raise ValueError(
            f"{episode_name}: missing required relative pose column(s) "
            f"{missing} for embodiment {embodiment_tag}; episode calibration "
            "is missing or unusable. The camera calibration is recorded with "
            "each episode: if Recording checks had Camera calibration set to "
            "warn only, the episode was recorded without it and must be "
            "recorded again."
        )


def get_task_description(group: zarr.Group) -> str:
    if "task_description" in group.attrs:
        return str(group.attrs["task_description"])
    arr = group.get("task_description")
    if arr is None:
        return ""
    value = np.asarray(arr)
    if value.shape == ():
        return str(value.item())
    if len(value):
        return str(value[0])
    return ""


def read_language_override(session_dir: Path) -> dict[str, Any] | None:
    """The session's task-text correction (``language_override.json``), or None.

    Raw and aligned keep the text the station recorded, even when it is wrong:
    manifests pin their sizes and any re-align copies raw back. A correction is
    recorded beside the aligned session instead, and the conversion applies it:
    exact ``old_text`` matches only, never to blank
    text. Raises for any other schema, or a file naming another task or session,
    rather than converting with text nobody vouched for.
    """
    path = session_dir / LANGUAGE_OVERRIDE_FILENAME
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    if not isinstance(doc, dict) or doc.get("schema") != 1:
        raise ValueError(f"{path}: expected a schema 1 language override")
    task, session = infer_task_session(str(session_dir))
    if doc.get("task_name") != task or doc.get("session_id") != session:
        raise ValueError(
            f"{path} names {doc.get('task_name')}/{doc.get('session_id')}, not {task}/{session}"
        )
    overrides = doc.get("overrides")
    if not isinstance(overrides, dict) or not overrides:
        raise ValueError(f"{path}: no overrides")
    for field, entry in overrides.items():
        old = entry.get("old_text") if isinstance(entry, dict) else None
        new = entry.get("new_text") if isinstance(entry, dict) else None
        if (
            field not in LANGUAGE_FIELDS
            or not isinstance(old, str)
            or not old.strip()
            or not isinstance(new, str)
            or not new.strip()
            or old == new
        ):
            raise ValueError(f"{path}: unusable override for {field!r}")
    return doc


def override_text(value: Any, field: str, override: dict[str, Any] | None) -> Any:
    """``value`` with the override applied: exact match on ``old_text`` only."""
    entry = (override or {}).get("overrides", {}).get(field)
    if entry and isinstance(value, str) and value == entry["old_text"]:
        return entry["new_text"]
    return value


def override_fields(
    values: dict[str, Any], override: dict[str, Any] | None
) -> dict[str, Any]:
    """A copy of ``values`` (zarr attrs, config ``meta``) with each field overridden."""
    if not override:
        return values
    out = dict(values)
    for field in LANGUAGE_FIELDS:
        if field in out:
            out[field] = override_text(out[field], field, override)
    return out


def recording_config_for_info(session_dir: Path) -> Any:
    """The session ``config.yaml`` as ``info.json`` records it, override applied."""
    config = yaml.safe_load((session_dir / "config.yaml").read_text())
    override = read_language_override(session_dir)
    if override and isinstance(config, dict) and isinstance(config.get("meta"), dict):
        config["meta"] = override_fields(config["meta"], override)
    return config


def write_language_correction(
    output_dir: Path,
    session_dir: Path,
    episode_names: list[str],
    override: dict[str, Any] | None,
) -> int:
    """Write ``meta/language_correction.json`` when the override changed this dataset.

    Counts what the conversion replaced, from the station's own values: the task
    string (``meta/tasks.jsonl``), each episode's ``tasks[0]`` and
    ``episode_attrs`` fields (``meta/episodes.jsonl``) and the recording config
    (``meta/info.json``). Removes any earlier record first, so a dataset carries
    one only when its text differs from aligned. Returns the replacement count.
    """
    path = output_dir / LANGUAGE_CORRECTION_REL_PATH
    path.unlink(missing_ok=True)
    if not override:
        return 0
    root = zarr.open_group(str(session_dir), mode="r")
    entries = override["overrides"]
    old_task = (entries.get("task_description") or {}).get("old_text")
    station_tasks = [get_task_description(root[name]) for name in episode_names]
    episodes = sum(task == old_task for task in station_tasks)
    for name in episode_names:
        attrs = dict(root[name].attrs)
        episodes += sum(
            attrs.get(field) == entry["old_text"] for field, entry in entries.items()
        )
    meta = (read_recording_config(session_dir) or {}).get("meta") or {}
    info = sum(meta.get(field) == entry["old_text"] for field, entry in entries.items())
    occurrences = {
        "meta/tasks.jsonl": int(old_task is not None and old_task in station_tasks),
        "meta/episodes.jsonl": episodes,
        "meta/info.json": info,
    }
    total = sum(occurrences.values())
    if not total:
        print(
            f"{LANGUAGE_OVERRIDE_FILENAME} present; no recorded text matched it",
            flush=True,
        )
        return 0
    record = {
        "schema": 1,
        "overrides": entries,
        "reason": override.get("reason"),
        "correction_record": override.get("correction_record"),
        "corrected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "corrected_by": "umi_lerobot_conversion",
        "occurrences": occurrences,
        "unchanged": "data/ parquets and videos/ (they carry only task_index)",
        "source_note": (
            f"aligned/raw keep the station text; see aligned/{override['task_name']}/"
            f"{override['session_id']}/{LANGUAGE_OVERRIDE_FILENAME}"
        ),
    }
    if "task_description" in entries:
        # Accepted keys in persisted source-episode records.
        record.update(field="task_description", **entries["task_description"])
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {LANGUAGE_CORRECTION_REL_PATH} ({total} replacements)", flush=True)
    return total


def get_selected_episodes(root: zarr.Group) -> list[str]:
    episodes = [str(k) for k in root.group_keys() if str(k).startswith("episode_")]
    invalid = normalize_episode_keys(root.attrs.get("invalid_episode_indices", []))
    skipped = set(invalid)

    selected = [episode for episode in episodes if episode not in skipped]
    return sorted(selected, key=source_episode_sort_key)


def decode_video_frames_with_torchcodec(
    input_video: Path, frame_indices: np.ndarray
) -> np.ndarray:
    try:
        import torchcodec.decoders
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "torchcodec is required for UMI video frame extraction; install the "
            "data_collection dependency group (scripts/setup_station.sh) or run this "
            "script with `uv run`"
        ) from exc

    decoder = torchcodec.decoders.VideoDecoder(
        str(input_video),
        device="cpu",
        dimension_order="NHWC",
        num_ffmpeg_threads=0,
    )
    return decoder.get_frames_at(
        indices=[int(index) for index in frame_indices.tolist()]
    ).data.numpy()


def center_crop_to_aspect(
    frame: np.ndarray, target_width: int, target_height: int
) -> np.ndarray:
    """Center-crop ``frame`` (H, W, C) to the ``target_width:target_height`` aspect ratio.

    Frames whose aspect ratio already matches (within float tolerance) are
    returned unchanged. Crop only -- no resize, no interpolation.
    """
    src_height, src_width = frame.shape[:2]
    src_aspect = src_width / src_height
    target_aspect = target_width / target_height
    if abs(src_aspect - target_aspect) <= 1e-6:
        return frame
    if src_aspect > target_aspect:
        new_width = int(src_height * target_aspect)
        crop_left = (src_width - new_width) // 2
        return frame[:, crop_left : crop_left + new_width, :]
    new_height = int(src_width / target_aspect)
    crop_top = (src_height - new_height) // 2
    return frame[crop_top : crop_top + new_height, :, :]


@dataclass(frozen=True)
class VideoVariant:
    """One derived, downscaled copy of a source RGB video stream.

    ``crop_to_aspect=True`` center-crops to the target aspect ratio before
    resizing (undistorted, narrower field of view); ``False`` resizes the whole
    frame straight to the target (full field of view, aspect ratio not
    preserved). Resizing uses PyAV/libswscale bilinear scaling, so the converter
    needs no OpenCV dependency.
    """

    name: str
    width: int
    height: int
    crop_to_aspect: bool

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if self.crop_to_aspect:
            frame = center_crop_to_aspect(frame, self.width, self.height)
        video_frame = av.VideoFrame.from_ndarray(
            np.ascontiguousarray(frame), format="rgb24"
        )
        return video_frame.reformat(width=self.width, height=self.height).to_ndarray(
            format="rgb24"
        )

    def transform(self, frames: np.ndarray) -> Iterable[np.ndarray]:
        return (self.apply(frame_array) for frame_array in frames)


# Generate 320x240 center-cropped and whole-frame-resized RGB variants.
# Cropping preserves geometry; resizing preserves field of view but changes aspect.
DEFAULT_320X240_VARIANTS: tuple[VideoVariant, ...] = (
    VideoVariant("320x240_cropped", 320, 240, crop_to_aspect=True),
    VideoVariant("320x240_resized", 320, 240, crop_to_aspect=False),
)

# AR0234 fisheye gripper cameras (UMI wrist + trigger) at 640x400, OpenCV
# fisheye (Kannala-Brandt) model. Used when the zarr carries no intrinsics.
AR0234_640X400_FISHEYE_K = np.array(
    [[198.367, 0.0, 320.001], [0.0, 197.878, 196.493], [0.0, 0.0, 1.0]]
)
AR0234_FISHEYE_D = (-0.013400, -0.004618, 0.0, 0.0)
STEREO_VARIANT_NAME = "320x240_stereo_rectified"
# Vertical FOV of the stereo view; matches OpenCV pinhole balance=0 on AR0234 640x400.
STEREO_VFOV_DEG = 115.6
# Frames per grid_sample call; bounds the float32 working set per worker.
STEREO_BATCH_FRAMES = 16


def build_stereo_grid(
    K, D, src_w, src_h, out_w=320, out_h=240, vfov_deg=STEREO_VFOV_DEG, ss=2
):
    """grid_sample grid (1, ss*out_h, ss*out_w, 2) mapping a stereographic view onto the fisheye frame.

    Stereographic projection r = 2 f tan(theta/2), square pixels, principal point at
    the image center, vertical FOV ``vfov_deg`` (default STEREO_VFOV_DEG).
    """
    import torch

    K = np.asarray(K, np.float64).reshape(3, 3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    half_v = math.radians(vfov_deg) / 2
    w, h = out_w * ss, out_h * ss
    f = (h / 2) / (2 * math.tan(half_v / 2))
    u, v = np.meshgrid(np.arange(w) - (w - 1) / 2, np.arange(h) - (h - 1) / 2)
    theta = 2 * np.arctan(np.hypot(u, v) / (2 * f))
    psi = np.arctan2(v, u)
    t2 = theta * theta
    td = theta * (1 + D[0] * t2 + D[1] * t2**2 + D[2] * t2**3 + D[3] * t2**4)
    mx = fx * td * np.cos(psi) + cx
    my = fy * td * np.sin(psi) + cy
    gx = 2 * mx / (src_w - 1) - 1
    gy = 2 * my / (src_h - 1) - 1
    return torch.from_numpy(np.stack([gx, gy], -1)[None].astype(np.float32))


def stereo_undistort_frames(frames: np.ndarray, grid: Any, ss: int = 2) -> np.ndarray:
    """frames: uint8 (N, H, W, 3) -> uint8 (N, out_h, out_w, 3)."""
    import torch
    import torch.nn.functional as F

    x = torch.as_tensor(frames).permute(0, 3, 1, 2).float()
    y = F.grid_sample(
        x,
        grid.expand(x.shape[0], -1, -1, -1),
        mode="bicubic",
        padding_mode="border",
        align_corners=True,
    )
    y = F.avg_pool2d(y, ss)
    return y.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()


def fisheye_lens_for_camera(
    recording_config: dict[str, Any] | None,
    cam: str,
    K: np.ndarray | None,
    distortion: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[float, ...]] | None:
    """(K, D) for the stereo variant of ``cam``, or None when it is not a known fisheye.

    ``distortion`` is the zarr ``{cam}/distortion`` array when recorded; it must
    match AR0234_FISHEYE_D, since only that lens model is supported.
    """
    entry = None
    for source in (recording_config or {}).get("video_sources") or []:
        if isinstance(source, dict) and source.get("name") == cam:
            entry = source
            break
    sensor_module = entry.get("sensor_module") if entry else None
    if sensor_module == "AR0234":
        if K is None:
            resolution = (entry.get("params") or {}).get("camera_resolution")
            if list(resolution or []) != [640, 400]:
                return None
            K = AR0234_640X400_FISHEYE_K
    elif (
        sensor_module is None
        and K is not None
        and np.allclose(K, AR0234_640X400_FISHEYE_K, atol=1e-3)
    ):
        # Infer the lens from recorded intrinsics when sensor_module is absent.
        pass
    else:
        return None
    if distortion is not None and not np.allclose(
        np.asarray(distortion, dtype=np.float64).reshape(-1, len(AR0234_FISHEYE_D)),
        AR0234_FISHEYE_D,
    ):
        raise ValueError(
            f"{cam} distortion {np.unique(np.asarray(distortion), axis=0).tolist()} "
            f"does not match the AR0234 fisheye model {AR0234_FISHEYE_D}"
        )
    return np.asarray(K, dtype=np.float64), AR0234_FISHEYE_D


@dataclass(frozen=True)
class StereoVideoVariant:
    """Stereographic reprojection of a fisheye stream at 320x240.

    Resampled from the native frame (2x supersampled bicubic, then 2x2 average),
    so it keeps more edge detail than resizing, square pixels, and a wider field
    of view than the center crop.
    """

    K: np.ndarray
    D: tuple[float, ...]
    name: str = STEREO_VARIANT_NAME
    width: int = 320
    height: int = 240

    def transform(self, frames: np.ndarray) -> Iterable[np.ndarray]:
        grid = build_stereo_grid(
            self.K, self.D, frames.shape[2], frames.shape[1], self.width, self.height
        )
        for start in range(0, len(frames), STEREO_BATCH_FRAMES):
            yield from stereo_undistort_frames(
                frames[start : start + STEREO_BATCH_FRAMES], grid
            )


def variant_video_key(video_key: str, variant_name: str) -> str:
    """Raw key of a derived variant stream, e.g. ``left_umi_camera_320x240_cropped``."""
    return f"{video_key}_{variant_name}"


def stereo_video_key(video_key: str) -> str:
    return variant_video_key(video_key, STEREO_VARIANT_NAME)


def output_video_keys(
    video_keys: list[str], video_variants: bool, stereo_keys: Iterable[str] = ()
) -> list[str]:
    """Source video keys plus, when enabled, one derived key per (RGB stream, variant).

    Only the fisheye streams in ``stereo_keys`` get the stereo variant.
    """
    keys = list(video_keys)
    stereo_keys = set(stereo_keys)
    if video_variants:
        for key in video_keys:
            keys.extend(
                variant_video_key(key, variant.name)
                for variant in DEFAULT_320X240_VARIANTS
            )
            if key in stereo_keys:
                keys.append(stereo_video_key(key))
    return keys


def read_recording_config(session_dir: Path) -> dict[str, Any] | None:
    config_path = session_dir / "config.yaml"
    if not config_path.exists():
        return None
    cfg = yaml.safe_load(config_path.read_text())
    return cfg if isinstance(cfg, dict) else None


def episode_stereo_lenses(
    recording_config: dict[str, Any] | None,
    group: Any,
    video_keys: list[str],
) -> dict[str, tuple[np.ndarray, tuple[float, ...]]]:
    """``{cam: (K, D)}`` for every RGB stream of one episode that gets the stereo variant.

    K comes from the zarr ``{cam}/intrinsics`` at the episode's aligned frames
    and must be constant within the episode; without one the config decides.
    """
    lenses = {}
    for key in video_keys:
        if key not in group:
            continue
        cam_group = group[key]
        K = None
        distortion = None
        if "aligned_frame_indices" in cam_group:
            aligned_idx = np.asarray(cam_group["aligned_frame_indices"])
            if "intrinsics" in cam_group:
                per_frame = np.asarray(cam_group["intrinsics"])[aligned_idx].reshape(
                    -1, 3, 3
                )
                if len(per_frame) and not np.allclose(per_frame, per_frame[0]):
                    raise ValueError(
                        f"{key}.intrinsics is not constant within the episode"
                    )
                K = per_frame[0] if len(per_frame) else None
            if "distortion" in cam_group:
                distortion = np.asarray(cam_group["distortion"])[aligned_idx]
        lens = fisheye_lens_for_camera(recording_config, key, K, distortion)
        if lens is not None:
            lenses[key] = lens
    return lenses


def session_stereo_keys(
    session_dir: Path, root: Any, episode_names: list[str], video_keys: list[str]
) -> list[str]:
    """RGB streams that get the stereo variant; must agree across the selected episodes."""
    recording_config = read_recording_config(session_dir)
    keys: list[str] | None = None
    for episode_name in episode_names:
        episode_keys = sorted(
            episode_stereo_lenses(recording_config, root[episode_name], video_keys)
        )
        if keys is not None and episode_keys != keys:
            raise ValueError(
                f"Stereo-variant cameras differ between episodes: {keys} vs "
                f"{episode_keys} in {episode_name}"
            )
        keys = episode_keys
    return keys or []


def encode_h264_video(
    frames: Iterable[np.ndarray], output_video: Path, fps: int, width: int, height: int
) -> None:
    out_container = av.open(str(output_video), mode="w")
    try:
        out_stream = out_container.add_stream("h264", rate=int(fps))
        out_stream.width = width
        out_stream.height = height
        out_stream.pix_fmt = "yuv420p"
        for frame_array in frames:
            out_frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in out_stream.encode(out_frame):
                out_container.mux(packet)
        for packet in out_stream.encode():
            out_container.mux(packet)
    finally:
        out_container.close()


def extract_video(
    input_video: Path,
    frame_indices: np.ndarray,
    output_video: Path,
    fps: int,
    variants: list[tuple[VideoVariant | StereoVideoVariant, Path]] | None = None,
) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    wanted = [int(i) for i in frame_indices.tolist()]
    if not wanted:
        raise ValueError(f"No frame indices for {input_video}")
    if wanted != sorted(wanted):
        raise ValueError(f"Frame indices must be nondecreasing for {input_video}")

    frames = decode_video_frames_with_torchcodec(input_video, frame_indices)
    if frames.shape[0] != len(wanted):
        raise RuntimeError(
            f"{input_video} returned {frames.shape[0]} frames for {len(wanted)} requested indices"
        )

    height, width = frames[0].shape[:2]
    encode_h264_video(frames, output_video, fps, width=width, height=height)
    # Every variant re-encodes from the same decoded frames: one decode, N encodes.
    for variant, variant_output in variants or []:
        variant_output.parent.mkdir(parents=True, exist_ok=True)
        encode_h264_video(
            variant.transform(frames),
            variant_output,
            fps,
            width=variant.width,
            height=variant.height,
        )


def episode_calibration(group: zarr.Group) -> dict[str, Any]:
    attrs = dict(group.attrs) if hasattr(group, "attrs") else {}
    calibration = attrs.get("calibration", [])
    if isinstance(calibration, list) and calibration:
        first = calibration[0]
        return first if isinstance(first, dict) else {}
    return {}


def add_calibrated_arm_relative_pose_columns(
    group: zarr.Group,
    data: dict[str, Any],
    T_world_eef: dict[str, np.ndarray],
) -> None:
    """Right arm pose in the left arm frame when the SLAM reference is an arm camera.

    The reference camera only decides which SLAM world the calibration
    transforms map into; the emitted feature is always the canonical
    ``right_arm_to_left_arm.eef``. Keying the column name on the reference arm
    would flip the feature's direction whenever the alignment picked
    ``right_tracking_camera``, so both arms are lifted into the reference world
    and rebased into the left arm.

    The reference does not have to be an arm camera: umiV3 rigs calibrate both
    trackers against ``head_oak_camera``, in which case each arm lifts into the
    reference world via its own ``T_{cam}`` transform.
    """
    calib = episode_calibration(group)
    ref_cam = calib.get("ref_camera_name")
    if not ref_cam:
        print("Warning: No reference camera found.", flush=True)
        return
    tracking = tracking_to_eef_for_embodiment("gripper_umi")

    T_ref_world_eef: dict[str, np.ndarray] = {}
    for cam, eef in tracking.items():
        if eef not in T_world_eef:
            continue
        if cam == ref_cam:
            T_ref_world_eef[eef] = T_world_eef[eef]
            continue
        T_key = f"T_{cam}"
        if T_key not in calib:
            continue
        T_Wref_Wcam = np.asarray(calib[T_key], dtype=np.float64)
        T_ref_world_eef[eef] = np.einsum("ij,njk->nik", T_Wref_Wcam, T_world_eef[eef])

    if any(eef not in T_ref_world_eef for eef in ("left_arm", "right_arm")):
        print(
            "Warning: missing arm pose or calibration transform; "
            "skipping arm-to-arm relative pose.",
            flush=True,
        )
        return
    T_left_right = np.einsum(
        "nij,njk->nik",
        np.linalg.inv(T_ref_world_eef["left_arm"]),
        T_ref_world_eef["right_arm"],
    )
    data["right_arm_to_left_arm.eef"] = transform_to_xyz_rot6d(T_left_right)


def add_shared_world_arm_relative_pose_columns(
    data: dict[str, Any],
    T_world_eef: dict[str, np.ndarray],
) -> None:
    """Right arm pose in the left arm frame when both trackers share one world.

    Vive lighthouse tracking reports every puck in the same base-station world,
    so unlike the SLAM-camera rigs there is no per-episode calibration to lift
    through -- rebase the right arm into the left arm directly:

        T_leftarm_rightarm = inv(T_world_leftarm) @ T_world_rightarm
    """
    if any(eef not in T_world_eef for eef in ("left_arm", "right_arm")):
        print(
            "Warning: missing arm tracker pose; skipping arm-to-arm relative pose.",
            flush=True,
        )
        return
    T_left_right = np.einsum(
        "nij,njk->nik",
        np.linalg.inv(T_world_eef["left_arm"]),
        T_world_eef["right_arm"],
    )
    data["right_arm_to_left_arm.eef"] = transform_to_xyz_rot6d(T_left_right)


def add_head_relative_pose_columns(
    group: zarr.Group,
    data: dict[str, Any],
    T_world_eef: dict[str, np.ndarray],
    embodiment_tag: str,
) -> None:
    calib = episode_calibration(group)
    ref_cam = calib.get("ref_camera_name")
    if not ref_cam:
        if embodiment_tag in HEAD_RELATIVE_REQUIRED_EMBODIMENTS:
            raise RuntimeError(
                "Missing calibration ref_camera_name for head-relative embodiment"
            )
        print(
            "Warning: No reference camera found; skipping head-relative poses.",
            flush=True,
        )
        return

    tracking = tracking_to_eef_for_embodiment(embodiment_tag)
    head_cam = "head_tracking_camera"
    if head_cam not in tracking or "head" not in T_world_eef:
        if embodiment_tag in HEAD_RELATIVE_REQUIRED_EMBODIMENTS:
            raise RuntimeError(
                "Missing head_tracking_camera pose data for head-relative embodiment"
            )
        print(
            "Warning: missing head_tracking_camera; skipping head-relative poses.",
            flush=True,
        )
        return

    T_ref_world_eef: dict[str, np.ndarray] = {}
    for cam, eef in tracking.items():
        if eef not in T_world_eef:
            continue
        if cam == ref_cam:
            T_ref_world_eef[cam] = T_world_eef[eef]
            continue
        T_key = f"T_{cam}"
        if T_key not in calib:
            continue
        T_Wref_Wcam = np.asarray(calib[T_key], dtype=np.float64)
        T_ref_world_eef[cam] = np.einsum("ij,njk->nik", T_Wref_Wcam, T_world_eef[eef])

    if head_cam not in T_ref_world_eef:
        if embodiment_tag in HEAD_RELATIVE_REQUIRED_EMBODIMENTS:
            raise RuntimeError("Missing calibration transform for head_tracking_camera")
        print(
            "Warning: calibration missing head camera transform; skipping head-relative poses.",
            flush=True,
        )
        return

    T_head_Wref = np.linalg.inv(T_ref_world_eef[head_cam])
    for cam, eef in tracking.items():
        if cam == head_cam or cam not in T_ref_world_eef:
            continue
        T_head_eef = np.einsum("nij,njk->nik", T_head_Wref, T_ref_world_eef[cam])
        data[f"{eef}_to_head.eef"] = transform_to_xyz_rot6d(T_head_eef)

    if embodiment_tag in HEAD_RELATIVE_REQUIRED_EMBODIMENTS:
        missing = [
            key
            for key in ("left_arm_to_head.eef", "right_arm_to_head.eef")
            if key not in data
        ]
        if missing:
            raise RuntimeError(
                "Missing head-relative arm poses for required embodiment: "
                + ", ".join(missing)
            )
        T_head_left = xyz_rot6d_to_transform(data["left_arm_to_head.eef"])
        T_head_right = xyz_rot6d_to_transform(data["right_arm_to_head.eef"])
        T_left_right = np.einsum(
            "nij,njk->nik", np.linalg.inv(T_head_left), T_head_right
        )
        data["right_arm_to_left_arm.eef"] = transform_to_xyz_rot6d(T_left_right)


def process_episode(
    session_dir: Path,
    spec: EpisodeConversionSpec,
    output_dir: Path,
    video_keys: list[str],
    fps: int,
    embodiment_tag: str,
    video_variants: bool = True,
    stereo_variant: bool = True,
) -> dict[str, Any]:
    root = zarr.open_group(str(session_dir), mode="r")
    group = root[spec.episode_name]
    timestamps = np.asarray(group["valid_timestamps"])
    length = len(timestamps)

    data: dict[str, Any] = {
        "timestamp": timestamps.tolist(),
        "frame_index": list(range(length)),
        "episode_index": [spec.episode_index] * length,
        "task_index": [spec.task_index] * length,
    }

    T_tracker_eef_by_body = tracker_eef_offsets(embodiment_tag)
    T_world_eef: dict[str, np.ndarray] = {}
    for cam, eef in tracking_to_eef_for_embodiment(embodiment_tag).items():
        if cam not in group:
            continue
        cam_group = group[cam]
        pose_field = resolve_pose_interp_field(cam_group)
        if pose_field is None:
            continue
        pose_array = cam_group[pose_field]
        verify_zarr_chunks_present(pose_array)
        T_world_tracker = np.asarray(pose_array)
        verify_no_zero_pose_matrices(T_world_tracker, f"{cam}.{pose_field}")
        if len(T_world_tracker) != length:
            raise ValueError(f"Length mismatch for {cam}.{pose_field}")
        data[RAW_TRACKER_KEY[eef]] = transform_to_xyz_rot6d(T_world_tracker)
        eef_pose = np.einsum("nij,jk->nik", T_world_tracker, T_tracker_eef_by_body[eef])
        T_world_eef[eef] = eef_pose
        data[f"{eef}.eef"] = transform_to_xyz_rot6d(eef_pose)

    if embodiment_tag in HEAD_RELATIVE_EMBODIMENTS:
        add_head_relative_pose_columns(group, data, T_world_eef, embodiment_tag)
    elif embodiment_tag in SHARED_WORLD_TRACKING_EMBODIMENTS:
        add_shared_world_arm_relative_pose_columns(data, T_world_eef)
    elif embodiment_tag in ARM_TO_ARM_RELATIVE_EMBODIMENTS:
        add_calibrated_arm_relative_pose_columns(group, data, T_world_eef)
    verify_relative_pose_columns(embodiment_tag, data, spec.episode_name)

    for raw, out in GRIPPER_KEYS.items():
        if raw in group and "distance_interp" in group[raw]:
            values = np.asarray(group[raw]["distance_interp"]).reshape(-1)
            if len(values) != length:
                raise ValueError(f"Length mismatch for {raw}.distance_interp")
            data[f"{out}.distance"] = values.tolist()
    data.update(derived_columns(data))

    # Emit per-frame intrinsics for each RGB camera with a recorded 3x3 K array.
    # Align them to the episode timeline without a camera-type allowlist.
    for column in expected_intrinsics_columns(group, embodiment_tag):
        cam = column[: -len(".intrinsics")]
        cam_group = group[cam]
        intrinsics_array = cam_group["intrinsics"]
        verify_zarr_chunks_present(intrinsics_array)
        K = np.asarray(intrinsics_array)
        aligned_idx = np.asarray(cam_group["aligned_frame_indices"])
        if len(aligned_idx) != length:
            raise ValueError(f"Length mismatch for {cam}.intrinsics")
        data[f"{cam}.intrinsics"] = K[aligned_idx].reshape(length, 9).tolist()

    data["index"] = list(range(spec.frame_start_index, spec.frame_start_index + length))
    # Static columns, plus any {cam}.intrinsics columns the data-driven loop
    # above emitted (sorted for a deterministic on-disk order).
    selected_keys = [
        key for key in data_columns_for_embodiment(embodiment_tag) if key in data
    ]
    selected_keys += sorted(
        key for key in data if key.endswith(".intrinsics") and key not in selected_keys
    )
    selected_data = {key: data[key] for key in selected_keys}
    data_path = output_dir / episode_data_rel_path(spec.episode_index)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected_data).to_parquet(data_path)

    stereo_lenses = {}
    if video_variants and stereo_variant:
        stereo_lenses = episode_stereo_lenses(
            read_recording_config(session_dir), group, video_keys
        )

    for key in video_keys:
        output_video = output_dir / episode_video_rel_path(spec.episode_index, key)
        if key not in group or "aligned_frame_indices" not in group[key]:
            raise ValueError(f"Missing video alignment for {spec.episode_name}/{key}")
        input_video = session_dir / spec.episode_name / f"{key}.mp4"
        indices = np.asarray(group[key]["aligned_frame_indices"])
        if len(indices) != length:
            raise ValueError(f"Length mismatch for {spec.episode_name}/{key}")
        variant_targets: list[tuple[VideoVariant | StereoVideoVariant, Path]] = []
        if video_variants:
            variant_targets = [
                (
                    variant,
                    output_dir
                    / episode_video_rel_path(
                        spec.episode_index, variant_video_key(key, variant.name)
                    ),
                )
                for variant in DEFAULT_320X240_VARIANTS
            ]
            if key in stereo_lenses:
                variant_targets.append(
                    (
                        StereoVideoVariant(*stereo_lenses[key]),
                        output_dir
                        / episode_video_rel_path(
                            spec.episode_index, stereo_video_key(key)
                        ),
                    )
                )
        extract_video(
            input_video, indices, output_video, fps=fps, variants=variant_targets
        )

    write_episode_source_marker(output_dir, spec)

    return episode_metadata(session_dir, spec, video_keys, length)


def episode_metadata(
    session_dir: Path,
    spec: EpisodeConversionSpec,
    video_keys: list[str],
    length: int,
) -> dict[str, Any]:
    root = zarr.open_group(str(session_dir), mode="r")
    group = root[spec.episode_name]
    episode_attrs = (
        normalize_metadata(dict(group.attrs)) if hasattr(group, "attrs") else {}
    )
    episode_attrs = override_fields(episode_attrs, read_language_override(session_dir))
    return {
        "episode_index": spec.episode_index,
        "tasks": [spec.task_description],
        "length": length,
        # The aligned episode this one was converted from, on this machine.
        "original_path": str(Path(session_dir).resolve() / spec.episode_name),
        "original_episode_index": spec.episode_name,
        "episode_attrs": episode_attrs,
        "gripper_version": GRIPPER_VERSION,
    }


def video_metadata(path: Path) -> dict[str, Any]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        codec = stream.codec_context
        rate = stream.average_rate or stream.base_rate
        fps = float(rate) if rate else None
        pixel_format = codec.format.name if codec.format is not None else None
        height = codec.height or stream.height
        width = codec.width or stream.width
    return {
        "dtype": "video",
        "shape": [int(height), int(width), 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": fps,
            "video.codec": codec.name,
            "video.pix_fmt": pixel_format,
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }


def column_feature(series: pd.Series) -> dict[str, Any]:
    first = np.asarray(series.iloc[0])
    if first.shape == ():
        return {"dtype": first.dtype.name, "shape": [1]}
    return {"dtype": first.dtype.name, "shape": list(first.shape)}


def stats_for_series(series: pd.Series) -> dict[str, Any]:
    values = [np.asarray(v, dtype=np.float32).reshape(-1) for v in series]
    arr = np.vstack(values)
    return {
        "max": np.max(arr, axis=0).tolist(),
        "min": np.min(arr, axis=0).tolist(),
        "mean": np.mean(arr, axis=0).tolist(),
        "std": np.std(arr, axis=0).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


def rot_6d_to_rot_mat(rot_6d: np.ndarray) -> np.ndarray:
    arr = np.asarray(rot_6d, dtype=np.float64).reshape(-1, 2, 3)
    if not np.allclose(np.sum(arr[:, 0] * arr[:, 1], axis=-1), 0):
        basis = np.zeros_like(arr)
        basis[:, 0] = arr[:, 0] / np.linalg.norm(arr[:, 0], axis=-1, keepdims=True)
        v = arr[:, 1]
        v = v - np.sum(v * basis[:, 0], axis=-1, keepdims=True) * basis[:, 0]
        basis[:, 1] = v / np.linalg.norm(v, axis=-1, keepdims=True)
        arr = basis
    rot_mat = np.zeros((arr.shape[0], 3, 3), dtype=np.float64)
    rot_mat[:, :2, :] = arr
    rot_mat[:, 2, :] = np.cross(arr[:, 0], arr[:, 1])
    return rot_mat


def compute_eef_delta(action: np.ndarray, proprio: np.ndarray) -> np.ndarray:
    horizon = action.shape[1]
    trans = action[:, :, :3].reshape(-1, 3)
    rot = rot_6d_to_rot_mat(action[:, :, 3:9].reshape(-1, 6))
    current = np.repeat(proprio[:, 0:1], horizon, axis=1)
    current_trans = current[:, :, :3].reshape(-1, 3)
    current_rot = rot_6d_to_rot_mat(current[:, :, 3:9].reshape(-1, 6))

    delta_rot = np.einsum("nij,njk->nik", np.swapaxes(current_rot, 1, 2), rot)
    delta_rot6d = matrix_to_rot6d(delta_rot).reshape(-1, horizon, 6)
    delta_trans = np.einsum(
        "nij,nj->ni",
        np.swapaxes(current_rot, 1, 2),
        trans - current_trans,
    ).reshape(-1, horizon, 3)
    return np.concatenate([delta_trans, delta_rot6d], axis=-1)


def build_padded_action_chunks_and_mask(
    data: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    trajectory_length = data.shape[0]
    is_scalar = data.ndim == 1
    if is_scalar:
        chunks = np.zeros((trajectory_length, horizon), dtype=data.dtype)
    else:
        chunks = np.zeros((trajectory_length, horizon, data.shape[1]), dtype=data.dtype)
    valid_mask = np.zeros((trajectory_length, horizon), dtype=bool)

    for start in range(trajectory_length):
        valid_len = min(horizon, trajectory_length - start)
        chunks[start, :valid_len] = data[start : start + valid_len]
        if valid_len < horizon and valid_len > 0:
            chunks[start, valid_len:] = data[start + valid_len - 1]
        valid_mask[start, :valid_len] = True
    return chunks, valid_mask


def process_episode_relative_stats(
    args: tuple[Path, list[str], list[str], int],
) -> dict[str, np.ndarray]:
    parquet_path, eef_columns, gripper_columns, max_action_chunk_size = args
    df = pd.read_parquet(parquet_path)
    trajectory_length = len(df)
    if trajectory_length < 2:
        return {}

    result: dict[str, np.ndarray] = {}
    for col in eef_columns:
        if col not in df.columns:
            continue
        eef_data = np.stack(df[col].values).astype(np.float64)
        actions, valid_mask = build_padded_action_chunks_and_mask(
            eef_data,
            max_action_chunk_size,
        )
        states = eef_data[:, None, :]
        delta = compute_eef_delta(actions, states)
        delta[~valid_mask] = np.nan
        result[col] = delta

    for col in gripper_columns:
        if col not in df.columns:
            continue
        gripper_data = np.asarray(df[col].values, dtype=np.float64)
        actions, valid_mask = build_padded_action_chunks_and_mask(
            gripper_data,
            max_action_chunk_size,
        )
        delta = actions - gripper_data[:, None]
        result[col] = np.where(valid_mask[:, :, None], delta[:, :, None], np.nan)
    return result


def calculate_relative_stats(
    output_dir: Path,
    *,
    max_action_chunk_size: int = 80,
    num_workers: int | None = None,
) -> None:
    info = json.loads((output_dir / "meta" / "info.json").read_text())
    features = info.get("features", {})
    eef_columns = [
        col
        for col in [
            "left_arm.eef",
            "right_arm.eef",
            "left_grasp.eef",
            "right_grasp.eef",
        ]
        if col in features
    ]
    gripper_columns = [
        col
        for col in [
            "left_gripper.distance",
            "right_gripper.distance",
            "left_gripper.distance_normalized",
            "right_gripper.distance_normalized",
        ]
        if col in features
    ]
    columns = eef_columns + gripper_columns
    if not columns:
        return

    total_episodes = int(info["total_episodes"])
    chunks_size = int(info["chunks_size"])
    template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    paths = [
        output_dir
        / template.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
        )
        for episode_index in range(total_episodes)
    ]
    paths = [path for path in paths if path.exists()]
    if not paths:
        return

    workers = min(num_workers or min(cpu_count(), 32), len(paths))
    args = [
        (path, eef_columns, gripper_columns, max_action_chunk_size) for path in paths
    ]
    if workers <= 1:
        results = [process_episode_relative_stats(arg) for arg in args]
    else:
        with spawn_process_pool(workers) as executor:
            results = list(executor.map(process_episode_relative_stats, args))

    deltas: dict[str, list[np.ndarray]] = {col: [] for col in columns}
    for result in results:
        for col, values in result.items():
            deltas[col].append(values)

    relative_stats = {}
    for col, values in deltas.items():
        if values:
            relative_stats[col] = stats_for_relative_array(
                np.concatenate(values, axis=0)
            )
    if not relative_stats:
        return

    stats_path = output_dir / "meta" / "relative_stats.json"
    stats_path.write_text(json.dumps(relative_stats, indent=2))


def stats_for_relative_array(values: np.ndarray) -> dict[str, Any]:
    with np.errstate(invalid="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return {
                "mean": np.nanmean(values, axis=0).tolist(),
                "std": np.nanstd(values, axis=0).tolist(),
                "min": np.nanmin(values, axis=0).tolist(),
                "max": np.nanmax(values, axis=0).tolist(),
                "q01": np.nanpercentile(values, 1, axis=0).tolist(),
                "q99": np.nanpercentile(values, 99, axis=0).tolist(),
                "q005": np.nanpercentile(values, 0.5, axis=0).tolist(),
                "q995": np.nanpercentile(values, 99.5, axis=0).tolist(),
            }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def video_modality_key(raw_key: str) -> str:
    """Map a raw zarr video key to its LeRobot modality key, ``{key}.rgb``.

    The stored mp4 path and ``info.json`` feature key keep using the raw key
    (the ``original_key``).
    """
    return f"{raw_key}.rgb"


def modality_state_dim(key: str) -> int:
    """Flattened width of a state/action column: 9 for eef poses and 3x3
    intrinsics, 1 for scalar gripper distances."""
    return 9 if key.endswith(".eef") or key.endswith(".intrinsics") else 1


def finalize_dataset(
    session_dir: Path,
    output_dir: Path,
    video_keys: list[str],
    episodes: list[dict[str, Any]],
    task_to_index: dict[str, int],
    embodiment_tag: str,
    fps: int,
    geometry: dict[str, Any] | None = None,
) -> None:
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(
        meta_dir / "tasks.jsonl",
        [
            {"task_index": idx, "task": task}
            for task, idx in sorted(task_to_index.items(), key=lambda x: x[1])
        ],
    )
    write_jsonl(meta_dir / "episodes.jsonl", episodes)

    first_parquet = output_dir / "data/chunk-000/episode_000000.parquet"
    first_df = pd.read_parquet(first_parquet)
    # Declare metadata for every intrinsics column emitted from recorded camera data.
    intrinsics_columns = sorted(
        c for c in first_df.columns if c.endswith(".intrinsics")
    )

    modality = {
        "state": {},
        "action": {},
        "video": {video_modality_key(key): {"original_key": key} for key in video_keys},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    state_columns, action_columns = state_action_columns_for_embodiment(embodiment_tag)
    for key in [*state_columns, *intrinsics_columns]:
        shape = modality_state_dim(key)
        entry: dict[str, Any] = {
            "original_key": key,
            "start": 0,
            "end": shape,
            "rotation_type": None,
            "absolute": True,
            "dtype": "float64",
            "range": None,
        }
        if key.endswith(".eef"):
            entry["rotation_type"] = "rotation_6d"
        modality["state"][key] = entry
    for key in action_columns:
        shape = modality_state_dim(key)
        entry = {
            "original_key": key,
            "start": 0,
            "end": shape,
            "rotation_type": None,
            "absolute": True,
            "dtype": "float64",
            "range": None,
        }
        if key.endswith(".eef"):
            entry["rotation_type"] = "rotation_6d"
        modality["action"][key] = entry

    (meta_dir / "modality.json").write_text(json.dumps(modality, indent=2))
    (meta_dir / "embodiment.json").write_text(
        json.dumps(
            {
                "robot_type": ROBOT_TYPE,
                "fps": fps,
                "num_episodes": len(episodes),
                "dataset_version": "1.0",
                "source_format": "zarr",
                "embodiment_tag": embodiment_tag,
            },
            indent=2,
        )
    )

    features = {col: column_feature(first_df[col]) for col in first_df.columns}
    for key in video_keys:
        features[key] = video_metadata(
            output_dir / f"videos/chunk-000/{key}/episode_000000.mp4"
        )

    total_frames = sum(ep["length"] for ep in episodes)
    info = {
        "codebase_version": "v2.1",
        "robot_type": ROBOT_TYPE,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(task_to_index),
        "total_videos": len(video_keys) * len(episodes),
        "total_chunks": len(episodes) // 1000 + 1,
        "chunks_size": 1000,
        "fps": float(fps),
        # Every episode is in the train split (LeRobot range syntax "start:end").
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    config_path = session_dir / "config.yaml"
    if config_path.exists():
        info["recording_config"] = recording_config_for_info(session_dir)
    if geometry is not None:
        info["umi_geometry"] = geometry
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

    parquet_paths = sorted((output_dir / "data").glob("*/*.parquet"))
    frames_by_path = [(path, pd.read_parquet(path)) for path in parquet_paths]
    verify_episode_frames_consistent(frames_by_path)
    df = pd.concat([frame for _, frame in frames_by_path], axis=0)
    stats = {col: stats_for_series(df[col]) for col in df.columns}
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2))


def load_episode_clip_module() -> Any:
    # Load the sibling clipping module by path for direct-script execution.
    # Register it in sys.modules before execution so dataclasses can resolve
    # annotations, and reuse an existing module under the same name.
    name = "episode_clip"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name("episode_clip.py")
        )
        if spec is None or spec.loader is None:
            raise ImportError("cannot load vendored episode_clip.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return module


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def compute_episode_clip_records(output_dir: Path) -> list[dict[str, Any]] | None:
    """Sidecar records for a finished dataset; None when it has no ``<side>_arm.eef``.

    Read the final episode Parquets in frame order and pass their end-effector
    poses and gripper widths to the sibling ``episode_clip.py`` module. Each
    record identifies its episode, frame window, and clipping rule version.
    """
    clip_rule = load_episode_clip_module()
    info = json.loads((output_dir / "meta/info.json").read_text())
    arm_names = [
        m.group(1) for k in info["features"] if (m := EPISODE_CLIPS_ARM_KEY.match(k))
    ]
    if not arm_names:
        return None
    episodes = read_jsonl_rows(output_dir / "meta/episodes.jsonl")
    chunks_size = info.get("chunks_size", 1000)
    records = []
    for ep in sorted(episodes, key=lambda e: e["episode_index"]):
        idx = ep["episode_index"]
        path = output_dir / info["data_path"].format(
            episode_chunk=idx // chunks_size, episode_index=idx
        )
        cols = ["timestamp", "frame_index"] + [f"{a}_arm.eef" for a in arm_names]
        grip_cols = [
            f"{a}_gripper.distance"
            for a in arm_names
            if f"{a}_gripper.distance" in info["features"]
        ]
        df = pd.read_parquet(path, columns=cols + grip_cols).sort_values("frame_index")
        if not np.array_equal(df["frame_index"].to_numpy(), np.arange(len(df))):
            raise ValueError(f"{path}: frame_index is not 0..N-1")
        t = df["timestamp"].to_numpy(dtype=float)
        arms = {a: np.stack(df[f"{a}_arm.eef"].to_numpy()) for a in arm_names}
        grippers = {
            c.split("_gripper")[0]: np.stack(df[c].to_numpy()).reshape(-1)
            for c in grip_cols
        }
        clip = clip_rule.compute_episode_clip(t, arms, grippers)
        records.append(
            {
                "episode_index": idx,
                **clip.to_json(),
                "rule": clip_rule.RULE_NAME,
                "rule_version": clip_rule.RULE_VERSION,
            }
        )
    return records


def validate_episode_clip_records(
    records: list[dict[str, Any]], episodes: list[dict[str, Any]]
) -> None:
    """Require one nonempty, in-bounds clip per episode with matching frame counts."""
    lengths = {}
    for ep in episodes:
        idx = int(ep["episode_index"])
        if idx in lengths:
            raise ValueError(f"episodes.jsonl lists episode {idx} twice")
        lengths[idx] = int(ep["length"])
    seen = set()
    for record in records:
        idx = int(record["episode_index"])
        start, end, num_frames = (
            int(record["clip_start"]),
            int(record["clip_end"]),
            int(record["num_frames"]),
        )
        if idx not in lengths:
            raise ValueError(f"clip for episode {idx}, which is not in episodes.jsonl")
        if idx in seen:
            raise ValueError(f"duplicate clip for episode {idx}")
        seen.add(idx)
        if num_frames != lengths[idx]:
            raise ValueError(
                f"episode {idx}: clip num_frames {num_frames} != episodes.jsonl length {lengths[idx]}"
            )
        if not 0 <= start < end <= lengths[idx]:
            raise ValueError(
                f"episode {idx}: clip [{start}, {end}) is outside [0, {lengths[idx]}) or empty"
            )
    if missing := sorted(set(lengths) - seen):
        raise ValueError(f"no clip for episodes {missing[:10]} ({len(missing)} total)")


def write_episode_clips(output_dir: Path, enabled: bool = True) -> str:
    """(Re)write ``meta/episode_clips.jsonl`` from the final parquets; returns the report status.

    Recomputed on every run from the final dataset. Clip metadata is optional,
    but stale windows can point to the wrong frames. Remove the previous file
    first and publish a sidecar only after computing and validating every
    record. On failure, report a warning and leave the dataset without a sidecar.
    """
    path = output_dir / EPISODE_CLIPS_REL_PATH
    path.unlink(missing_ok=True)
    if not enabled:
        return "disabled"
    try:
        records = compute_episode_clip_records(output_dir)
        if records is None:
            print(
                f"No <side>_arm.eef feature; not writing {EPISODE_CLIPS_REL_PATH}",
                flush=True,
            )
            return "skipped (no <side>_arm.eef feature)"
        validate_episode_clip_records(
            records, read_jsonl_rows(output_dir / "meta/episodes.jsonl")
        )
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
    except Exception as exc:
        path.unlink(missing_ok=True)
        print(
            f"WARNING: episode clips failed; dataset ships without {EPISODE_CLIPS_REL_PATH}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return f"failed ({type(exc).__name__}: {exc})"
    print(f"Wrote {EPISODE_CLIPS_REL_PATH} for {len(records)} episodes", flush=True)
    return f"written ({len(records)} episodes)"


def episode_chunk(episode_index: int) -> int:
    return episode_index // 1000


def episode_data_rel_path(episode_index: int) -> str:
    return f"data/chunk-{episode_chunk(episode_index):03d}/episode_{episode_index:06d}.parquet"


def episode_video_rel_path(episode_index: int, video_key: str) -> str:
    return f"videos/chunk-{episode_chunk(episode_index):03d}/{video_key}/episode_{episode_index:06d}.mp4"


def episode_source_rel_path(spec: EpisodeConversionSpec) -> str:
    return f"{SOURCE_EPISODES_DIR}/episode_{spec.episode_index:06d}.{spec.episode_name}.json"


def write_episode_source_marker(output_dir: Path, spec: EpisodeConversionSpec) -> Path:
    """Record which source episode this output index was built from."""
    path = output_dir / episode_source_rel_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"episode_index": spec.episode_index, "source_episode": spec.episode_name},
            indent=2,
        )
        + "\n"
    )
    return path


def expected_intrinsics_columns(episode_group: Any, embodiment_tag: str) -> list[str]:
    """``{cam}.intrinsics`` columns the current input yields for one episode.

    A camera whose ``intrinsics`` array is missing or has no
    ``aligned_frame_indices`` is skipped -- which is also why an aligned session
    copied without an array's ``zarr.json`` converts *without* that column
    rather than failing.
    """
    columns = []
    for cam in rgb_video_keys_for_embodiment(embodiment_tag):
        if cam not in episode_group:
            continue
        cam_group = episode_group[cam]
        if "intrinsics" not in cam_group or "aligned_frame_indices" not in cam_group:
            continue
        columns.append(f"{cam}.intrinsics")
    return columns


def verify_episode_frames_consistent(frames: list[tuple[Path, pd.DataFrame]]) -> None:
    """Every episode parquet must carry the same columns before they are stacked.

    ``stats_for_series`` otherwise fails deep inside numpy with a message that
    names a global row index and nothing else; name the offending episodes and
    the differing columns instead.
    """
    if not frames:
        return
    reference_path, reference = frames[0]
    reference_columns = list(reference.columns)
    mismatched = [
        f"{path.name}: missing {sorted(set(reference_columns) - set(frame.columns))}, "
        f"extra {sorted(set(frame.columns) - set(reference_columns))}"
        for path, frame in frames[1:]
        if list(frame.columns) != reference_columns
    ]
    if mismatched:
        raise ValueError(
            f"Episode parquet columns differ from {reference_path.name}: "
            + "; ".join(mismatched)
        )


def build_episode_specs(
    root: zarr.Group,
    episode_names: list[str],
    override: dict[str, Any] | None = None,
) -> tuple[list[EpisodeConversionSpec], dict[str, int]]:
    task_to_index: dict[str, int] = {"": 0}
    specs = []
    frame_start_index = 0
    for idx, episode_name in enumerate(episode_names):
        episode_group = root[episode_name]
        task_description = override_text(
            get_task_description(episode_group), "task_description", override
        )
        task_index = task_to_index.setdefault(task_description, len(task_to_index))
        episode_length = len(np.asarray(episode_group["valid_timestamps"]))
        specs.append(
            EpisodeConversionSpec(
                episode_name=episode_name,
                episode_index=idx,
                frame_start_index=frame_start_index,
                task_description=task_description,
                task_index=task_index,
            )
        )
        frame_start_index += episode_length
    return specs, task_to_index


def convert_session(
    session_dir: Path,
    output_dir: Path,
    embodiment_tag: str,
    fps: int,
    episode_workers: int = 1,
    video_variants: bool = True,
    stereo_variant: bool = True,
    episode_clips: bool = True,
    episode_clips_status: dict[str, str] | None = None,
) -> list[str]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing non-empty output: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(session_dir), mode="r")
    episode_names = get_selected_episodes(root)
    if not episode_names:
        raise RuntimeError(f"No episodes selected for conversion: {session_dir}")

    video_keys = video_keys_for_embodiment(embodiment_tag)
    # Source streams plus the derived 320x240 variants (when enabled). The
    # variants are conversion outputs like any other stream, so the final
    # metadata lists them too.
    stereo_keys = []
    if video_variants and stereo_variant:
        stereo_keys = session_stereo_keys(session_dir, root, episode_names, video_keys)
    output_keys = output_video_keys(video_keys, video_variants, stereo_keys)

    geometry = umi_geometry(embodiment_tag)
    language_override = read_language_override(session_dir)
    specs, task_to_index = build_episode_specs(root, episode_names, language_override)

    def episode_job_args(spec: EpisodeConversionSpec) -> tuple[Any, ...]:
        return (
            session_dir,
            spec,
            output_dir,
            video_keys,
            fps,
            embodiment_tag,
            video_variants,
            stereo_variant,
        )

    episodes: list[dict[str, Any]] = []
    worker_count = min(max(1, episode_workers), len(specs))
    if worker_count == 1:
        for spec in specs:
            print(f"Converting {session_dir.name}/{spec.episode_name}", flush=True)
            episodes.append(process_episode(*episode_job_args(spec)))
    else:
        print(
            f"Converting {len(specs)} episodes with {worker_count} workers",
            flush=True,
        )
        with spawn_process_pool(worker_count) as executor:
            future_to_spec = {
                executor.submit(process_episode, *episode_job_args(spec)): spec
                for spec in specs
            }
            try:
                for future in as_completed(future_to_spec):
                    spec = future_to_spec[future]
                    episodes.append(future.result())
                    print(
                        f"Finished {session_dir.name}/{spec.episode_name}", flush=True
                    )
            except Exception:
                for future in future_to_spec:
                    future.cancel()
                raise
    episodes.sort(key=lambda episode: episode["episode_index"])

    finalize_dataset(
        session_dir,
        output_dir,
        output_keys,
        episodes,
        task_to_index,
        embodiment_tag,
        fps,
        geometry=geometry,
    )
    write_language_correction(
        output_dir,
        session_dir,
        [episode["original_episode_index"] for episode in episodes],
        language_override,
    )
    # After finalize_dataset: the parquets and episodes.jsonl are final here.
    clips_status = write_episode_clips(output_dir, enabled=episode_clips)
    if episode_clips_status is not None:
        episode_clips_status["status"] = clips_status
    return episode_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert UMI V4 aligned Zarr sessions on local disk to LeRobot datasets."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "The station output folder <root>: aligned sessions are read from "
            "<root>/aligned and datasets written to <root>/lerobot_format. Default "
            "$UMI_OUTPUT_ROOT, else output/ at the repository root"
        ),
    )
    parser.add_argument(
        "--embodiment-tag",
        default="auto",
        help=(
            "Default 'auto': each session's tag is detected from the streams it recorded"
        ),
    )
    parser.add_argument(
        "--input-dataset",
        help="Local path of one aligned Zarr session to convert",
    )
    parser.add_argument(
        "--output-dataset",
        help=(
            "With --input-dataset: local path of the converted LeRobot dataset. "
            "Default <root>/lerobot_format/<task>/<embodiment>.<session>"
        ),
    )
    parser.add_argument(
        "--task-names",
        nargs="*",
        help="Convert every aligned session of these tasks under <root>/aligned",
    )
    parser.add_argument("--from-date")
    parser.add_argument("--to-date")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode-workers", type=int, default=1)
    parser.add_argument("--relative-stats-chunk-size", type=int, default=80)
    parser.add_argument("--no-relative-stats", action="store_true")
    parser.add_argument(
        "--no-video-variants",
        dest="video_variants",
        action="store_false",
        help="Do not emit the 320x240 cropped/resized variants of each RGB stream.",
    )
    parser.add_argument(
        "--no-stereo-variant",
        dest="stereo_variant",
        action="store_false",
        help="Do not emit the 320x240 stereographic variant of fisheye RGB streams.",
    )
    parser.add_argument(
        "--no-episode-clips",
        dest="episode_clips",
        action="store_false",
        help="Do not write meta/episode_clips.jsonl (per-episode clip windows for gr00t).",
    )
    args = parser.parse_args()
    args.output_root = (
        (args.output_root or default_output_root()).expanduser().resolve()
    )
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.embodiment_tag != "auto":
        video_keys_for_embodiment(args.embodiment_tag)

    task_mode = bool(args.task_names)
    direct_mode = bool(args.input_dataset)
    if task_mode == direct_mode:
        raise ValueError("Use exactly one input mode: --input-dataset or --task-names")
    if args.output_dataset and not direct_mode:
        raise ValueError("--output-dataset is only valid with --input-dataset")
    if args.episode_workers < 1:
        raise ValueError("--episode-workers must be at least 1")
    if args.relative_stats_chunk_size < 1:
        raise ValueError("--relative-stats-chunk-size must be at least 1")


def output_path(args: argparse.Namespace, task: str, dataset_name: str) -> Path:
    if args.output_dataset:
        return Path(args.output_dataset).expanduser()
    return args.output_root / LOCAL_DATASETS_DIR / task / dataset_name


def video_keys_for_embodiment(embodiment_tag: str) -> list[str]:
    video_keys = VIDEO_KEYS_BY_EMBODIMENT.get(embodiment_tag)
    if not video_keys:
        raise ValueError(
            f"Unsupported embodiment tag in standalone converter: {embodiment_tag}"
        )
    return video_keys


def detect_embodiment_tag(session_dir: Path) -> str:
    """The embodiment whose streams are exactly the ones a local session recorded.

    Looks at the first episode: its cameras with an mp4, except the arm
    tracking cameras (only their poses are converted), must be the tag's RGB
    streams, so no recorded camera is silently dropped. The tag's tracking
    groups must be there too, which tells an iPhone rig from a Vive one. Tags
    with the same streams resolve to the first in VIDEO_KEYS_BY_EMBODIMENT.
    """
    episodes = sorted(
        (path for path in session_dir.glob("episode_*") if path.is_dir()),
        key=lambda path: source_episode_sort_key(path.name),
    )
    if not episodes:
        raise ValueError(f"No episodes to detect the embodiment from in {session_dir}")
    groups = {path.name for path in episodes[0].iterdir() if path.is_dir()}
    cameras = {path.stem for path in episodes[0].glob("*.mp4")} - set(TRACKING_TO_EEF)
    if not cameras:
        raise ValueError(
            f"No camera videos in {episodes[0]}. Alignment is incomplete or failed. "
            "Run Check and Re-align on the raw session before converting. "
            "If a camera did not record, record the episode again; changing "
            "--embodiment-tag cannot restore missing data."
        )
    for tag in VIDEO_KEYS_BY_EMBODIMENT:
        if (
            set(rgb_video_keys_for_embodiment(tag)) == cameras
            and set(tracking_to_eef_for_embodiment(tag)) <= groups
        ):
            return tag
    raise ValueError(
        f"No embodiment tag matches the cameras {sorted(cameras)} recorded in "
        f"{episodes[0]}; pass --embodiment-tag to convert a subset of them"
    )


def sessions_to_convert(args: argparse.Namespace) -> list[tuple[str, str, Path]]:
    """``(task, session, aligned session dir)`` for each session this run converts."""
    if args.input_dataset:
        session_dir = Path(args.input_dataset).expanduser().resolve()
        task, session = infer_task_session(str(session_dir))
        return [(task, session, session_dir)]
    aligned_root = args.output_root / "aligned"
    return [
        (task, session, aligned_root / task / session)
        for task in args.task_names
        for session in filter_by_date(
            list_local_sessions(aligned_root, task), args.from_date, args.to_date
        )
    ]


def convert_one_session(
    args: argparse.Namespace,
    task: str,
    session: str,
    session_dir: Path,
) -> SessionResult:
    """Convert one aligned session; a failure is recorded in the result, not raised."""
    clips_status: dict[str, str] = {}
    output_dir = None
    try:
        embodiment_tag = args.embodiment_tag
        if embodiment_tag == "auto":
            embodiment_tag = detect_embodiment_tag(session_dir)
            print(f"{task}/{session}: detected embodiment {embodiment_tag}", flush=True)
        output_dir = output_path(args, task, f"{embodiment_tag}.{session}")
        if not (session_dir / "zarr.json").exists():
            raise RuntimeError(
                f"Input dataset is not an aligned Zarr root: {session_dir}"
            )
        selected = convert_session(
            session_dir,
            output_dir,
            embodiment_tag,
            args.fps,
            episode_workers=args.episode_workers,
            video_variants=args.video_variants,
            stereo_variant=args.stereo_variant,
            episode_clips=args.episode_clips,
            episode_clips_status=clips_status,
        )
        if not args.no_relative_stats:
            calculate_relative_stats(
                output_dir,
                max_action_chunk_size=args.relative_stats_chunk_size,
                num_workers=args.episode_workers,
            )
    except Exception as exc:
        print(f"FAILED {task}/{session}: {exc}", file=sys.stderr, flush=True)
        return SessionResult(
            task=task,
            session=session,
            status="failed",
            input_dataset=str(session_dir),
            local_output=str(output_dir) if output_dir else None,
            error=str(exc),
            episode_clips=clips_status.get("status"),
        )
    return SessionResult(
        task=task,
        session=session,
        status="ok",
        input_dataset=str(session_dir),
        local_output=str(output_dir),
        selected_episodes=selected,
        episode_clips=clips_status.get("status"),
    )


def report_filename_component(value: str | None) -> str:
    if not value:
        return ""
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip(".-_")
    return component[:180]


def report_filename(created_at: datetime, results: list[SessionResult]) -> str:
    timestamp = created_at.strftime("%Y%m%d_%H%M%S_%f")
    suffix = ""
    if len(results) == 1:
        result = results[0]
        suffix = report_filename_component(result.session)
    elif results:
        tasks = sorted({result.task for result in results if result.task})
        if len(tasks) == 1:
            suffix = report_filename_component(f"{tasks[0]}_{len(results)}-sessions")
        else:
            suffix = report_filename_component(f"{len(results)}-sessions")

    if suffix:
        return f"conversion_report_{timestamp}_{suffix}.json"
    return f"conversion_report_{timestamp}.json"


def main() -> int:
    args = parse_args()
    validate_args(args)
    report_dir = args.output_root / LOCAL_DATASETS_DIR / "_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    results = [
        convert_one_session(args, task, session, session_dir)
        for task, session, session_dir in sessions_to_convert(args)
    ]

    created_at = datetime.now(timezone.utc)
    report = {
        "created_at": created_at.isoformat(),
        "embodiment_tag": args.embodiment_tag,
        "episode_workers": args.episode_workers,
        "video_variants": args.video_variants,
        "stereo_variant": args.stereo_variant,
        "episode_clips": args.episode_clips,
        "results": [r.__dict__ for r in results],
    }
    report_path = report_dir / report_filename(created_at, results)
    report_path.write_text(json.dumps(report, indent=2))

    return 1 if any(r.status != "ok" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())

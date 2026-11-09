#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DexUMI aligned-Zarr session to a LeRobot dataset, on local disk.

Runs after alignment, on ``<session>_aligned``: first the robot-action
conversion (real_scripts/dexumi_pipeline/convert_to_robot_action.py) writes the
robot-hand joints into the aligned session (with its tactile variants unless
--no-tactile, the same default as ``./launch.sh convert``), then the dataset is
written to
``<session>_lerobot`` next to it (--output-dataset overrides). An existing
dataset is replaced only with --replace, and only once the new one is complete.
The DC GUI's "Convert for training" step runs this script with --replace.

    ./launch.sh lerobot --input-dataset output/<session>_aligned

Columns (state and action, one set per hand, ``<side>`` = left / right):

- ``<side>_tracker.eef``         -- the hand's iPhone tracker pose: xyz (m) +
  6D rotation (first two rows of R), 9 values. The raw iPhone pose: no
  glove-mount offset is applied.
- ``<side>_hand.joint_positions`` -- the robot-hand joint positions (rad) from
  the robot-action conversion: state from
  ``<side>_dexumi_exoskeleton_encoder_robot_state/state_interp``, action from
  ``..._robot_action/action_interp``.

After a tactile robot-action conversion (the groups' ``tactile`` attr), each
hand also gets an action-only column:

- ``<side>_hand.joint_torques`` -- per-joint torque from the fingertip forces
  (N*m), from ``..._robot_state/joints_torque_interp``.

When the session recorded the fingertip forces, each hand also gets the
state-only ``<side>_hand.tactile`` column: 30 values in newtons, fx, fy, fz at
measurement point 1 then 2 for each sensor 33..37 (thumb..pinky). The hand
column names and layouts follow the gr00t DexUMI conversion. Every episode of a
session must agree, so all its parquet files have the same columns.

Each wrist camera with recorded intrinsics also gets an observation-only
``<cam>.intrinsics`` column (3x3, flattened).

Video streams (per wrist camera, plus the head camera when it was recorded,
unless --no-video-variants):
- ``<cam>.rgb``                    -- native resolution, as recorded
- ``<cam>_320x240_cropped.rgb``    -- center-cropped to 4:3, then resized to
  320x240 (preserved aspect ratio, narrower field of view)
- ``<cam>_320x240_resized.rgb``    -- whole frame resized to 320x240 (full field
  of view, aspect ratio not preserved)
- ``<cam>_320x240_stereo_rectified.rgb`` -- only for a stream whose recorded
  intrinsics match the AR0234 fisheye lens, unless --no-stereo-variant

The iPhone tracking-camera videos are not exported; only their poses are.
This script was adapted from the GEAR UMI LeRobot converter
(``umi_lerobot_conversion.py``) and keeps its dataset layout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import warnings
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import cpu_count, get_context
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
import yaml
import zarr

# Expected config.yaml meta.embodiment and output robot_type.
EMBODIMENT_TAG = "bimanual_dexumi"
ROBOT_TYPE = "dexumi_hand"
SIDES = ("left", "right")
# Each hand's iPhone tracking camera: only its pose is exported.
TRACKING_CAMERAS = {side: f"{side}_tracking_camera" for side in SIDES}
# Wrist cameras every session must have, and the optional head camera.
REQUIRED_VIDEO_KEYS = ["left_umi_camera", "right_umi_camera"]
OPTIONAL_VIDEO_KEYS = ["head_umi_camera"]
# Groups the robot-action conversion (``./launch.sh convert``) writes per hand.
ROBOT_STATE_GROUP = "{side}_dexumi_exoskeleton_encoder_robot_state"
ROBOT_ACTION_GROUP = "{side}_dexumi_exoskeleton_encoder_robot_action"
ROBOT_STATE_ARRAY = ROBOT_STATE_GROUP + "/state_interp"
ROBOT_ACTION_ARRAY = ROBOT_ACTION_GROUP + "/action_interp"
# Arrays only a tactile robot-action conversion writes (groups' ``tactile`` attr).
ROBOT_TORQUE_ARRAY = ROBOT_STATE_GROUP + "/joints_torque_interp"
TACTILE_ONLY_ARRAYS = (
    ROBOT_TORQUE_ARRAY,
    ROBOT_ACTION_GROUP + "/action_torque_interp",
)
# Recorded fingertip forces: thumb..pinky sensors, raw counts in millinewtons.
TACTILE_GROUP = "{side}_dexumi_exoskeleton_tactile"
TACTILE_SENSOR_IDS = (33, 34, 35, 36, 37)
TACTILE_POINTS = (1, 2)
TACTILE_COMPONENTS = ("fx", "fy", "fz")
TACTILE_SCALE = 0.001  # newtons per recorded count
RERUN_HINT = "re-run ./launch.sh convert --aligned-dir <session>_aligned --sequential"
# Match the completion markers used by umi.common.data.write_state.
WRITE_STATE, WRITING = "write_state", "writing"
# Suffixes of the session folders: <session>, <session>_aligned, <session>_lerobot.
ALIGNED_SUFFIX = "_aligned"
LEROBOT_SUFFIX = "_lerobot"
SOURCE_EPISODE_RE = re.compile(r"^episode_(\d+)$")
# Per-episode identity marker, e.g. `meta/source_episodes/episode_000003.episode_5.json`.
SOURCE_EPISODES_DIR = "meta/source_episodes"


def tracker_column(side: str) -> str:
    return f"{side}_tracker.eef"


def joints_column(side: str) -> str:
    return f"{side}_hand.joint_positions"


def action_joints_column(side: str) -> str:
    """Parquet column for joint actions, stored separately from joint state."""
    return f"action.{joints_column(side)}"


def torque_key(side: str) -> str:
    return f"{side}_hand.joint_torques"


def tactile_column(side: str) -> str:
    return f"{side}_hand.tactile"


def tactile_channel_keys() -> list[str]:
    """Recorded tactile arrays in packing order: sensor, then point, then fx/fy/fz."""
    return [
        f"tactile_{sensor}_{component}_{point}_interp"
        for sensor in TACTILE_SENSOR_IDS
        for point in TACTILE_POINTS
        for component in TACTILE_COMPONENTS
    ]


@dataclass(frozen=True)
class SessionColumns:
    """The optional per-hand columns a session exports, decided once for all
    its episodes so every parquet file has the same columns."""

    tactile_actions: bool = False  # the robot-action conversion ran with tactile
    tactile_state: bool = False  # the session recorded the fingertip forces

    def state_keys(self) -> list[str]:
        keys = []
        for side in SIDES:
            keys += [tracker_column(side), joints_column(side)]
            if self.tactile_state:
                keys.append(tactile_column(side))
        return keys

    def action_keys(self) -> list[str]:
        keys = []
        for side in SIDES:
            keys += [tracker_column(side), joints_column(side)]
            if self.tactile_actions:
                keys.append(torque_key(side))
        return keys


def action_source_column(key: str) -> str:
    """The parquet column holding the action modality ``key``: hand actions are
    stored as ``action.<key>``, apart from the state of the same name."""
    if any(key.startswith(f"{side}_hand.") for side in SIDES):
        return f"action.{key}"
    return key


@dataclass(frozen=True)
class EpisodeConversionSpec:
    episode_name: str
    episode_index: int
    frame_start_index: int
    task_description: str
    task_index: int


def spawn_process_pool(max_workers: int) -> ProcessPoolExecutor:
    """Create a process pool with isolated native-library state.

    Forked workers can inherit inconsistent native-library state (notably
    PyArrow), so conversion and statistics workers start in clean
    interpreters.
    """
    # Divide CPU threads among workers for stereographic video processing.
    os.environ.setdefault(
        "OMP_NUM_THREADS", str(max(1, cpu_count() // max(1, max_workers)))
    )
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=get_context("spawn"),
    )


# Reject near-zero rotation blocks during conversion; they represent dropped
# poses and are invalid inputs to scipy Rotation.
ZERO_ROTATION_NORM_TOL = 1e-8


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


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim == 2:
        return matrix[:2, :].reshape(6)
    return matrix[:, :2, :].reshape(-1, 6)


def transform_to_xyz_rot6d(transform: np.ndarray) -> list[list[float]]:
    return np.concatenate(
        [transform[:, :3, 3], matrix_to_rot6d(transform[:, :3, :3])],
        axis=1,
    ).tolist()


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

    Zarr fills missing chunks with ``fill_value`` (0.0 by default), which can
    hide incomplete data. Check local chunk files before reading; non-local
    stores are skipped.
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

    Reject tracker dropout or missing chunks stored as zeros, and report the
    affected source key and frames before training.
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


def decode_video_frames_with_torchcodec(
    input_video: Path, frame_indices: np.ndarray
) -> np.ndarray:
    try:
        import torchcodec.decoders
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "torchcodec is required for video frame extraction; run `uv sync` at the "
            "repository root, or run this script with `uv run`"
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
    resizing (preserved aspect ratio, narrower field of view); ``False`` resizes
    the whole frame to the target (full field of view, changed aspect ratio).
    Resizing uses PyAV/libswscale bilinear scaling.
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


# AR0234 fisheye lens at 640x400 (the UMI wrist cameras), OpenCV fisheye
# (Kannala-Brandt) model. A DexUMI camera gets the stereo variant only when its
# recorded intrinsics match this K (its config names no sensor_module).
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


def episode_chunk(episode_index: int) -> int:
    return episode_index // 1000


def episode_data_rel_path(episode_index: int) -> str:
    return f"data/chunk-{episode_chunk(episode_index):03d}/episode_{episode_index:06d}.parquet"


def episode_video_rel_path(episode_index: int, video_key: str) -> str:
    return f"videos/chunk-{episode_chunk(episode_index):03d}/{video_key}/episode_{episode_index:06d}.mp4"


def verify_episode_frames_consistent(frames: list[tuple[Path, pd.DataFrame]]) -> None:
    """Check column consistency before stacking and report mismatched episodes."""
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


def verify_embodiment(session_dir: Path) -> dict[str, Any]:
    """The session's config.yaml, after checking it was recorded as bimanual DexUMI."""
    config = read_recording_config(session_dir)
    if config is None:
        raise ValueError(f"{session_dir} has no config.yaml; is it an aligned session?")
    embodiment = (config.get("meta") or {}).get("embodiment")
    if embodiment != EMBODIMENT_TAG:
        raise ValueError(
            f"{session_dir} was recorded as {embodiment!r}, not {EMBODIMENT_TAG!r}; "
            "this converter handles bimanual DexUMI sessions only"
        )
    return config


def session_video_keys(session_dir: Path, episode_names: list[str]) -> list[str]:
    """Wrist cameras plus the head camera when the session recorded it.

    Every episode must hold the same set: the dataset lists its streams once.
    """
    keys = None
    for name in episode_names:
        recorded = {path.stem for path in (session_dir / name).glob("*.mp4")}
        missing = [key for key in REQUIRED_VIDEO_KEYS if key not in recorded]
        if missing:
            raise ValueError(
                f"{name} has no {', '.join(missing)} video; alignment is incomplete"
            )
        episode_keys = REQUIRED_VIDEO_KEYS + [
            key for key in OPTIONAL_VIDEO_KEYS if key in recorded
        ]
        if keys is not None and episode_keys != keys:
            raise ValueError(
                f"Recorded cameras differ between episodes: {keys} vs {episode_keys} in {name}"
            )
        keys = episode_keys
    return keys or []


def verify_aligned_timestamps(root: zarr.Group, episode_names: list[str]) -> None:
    """Every episode needs the ``valid_timestamps`` alignment writes (one per row)."""
    missing = [name for name in episode_names if "valid_timestamps" not in root[name]]
    if missing:
        shown = ", ".join(missing[:4]) + (" ..." if len(missing) > 4 else "")
        raise ValueError(
            f"Alignment is incomplete: no valid_timestamps in {shown}. Align the "
            "session again before converting it."
        )


def verify_robot_actions(root: zarr.Group, episode_names: list[str]) -> None:
    """Every episode needs both hands' robot state and action, completely written."""
    problems = []
    for name in episode_names:
        for side in SIDES:
            for template in (ROBOT_STATE_ARRAY, ROBOT_ACTION_ARRAY):
                path = template.format(side=side)
                group_path = path.rsplit("/", 1)[0]
                if path not in root[name]:
                    problems.append(f"{name}/{path} is missing")
                elif root[name][group_path].attrs.get(WRITE_STATE) == WRITING:
                    problems.append(f"{name}/{group_path} is half-written")
    if problems:
        shown = "; ".join(problems[:4]) + (" ..." if len(problems) > 4 else "")
        raise ValueError(
            f"Robot actions are not ready ({shown}). Run the robot-action conversion "
            "first: ./launch.sh convert --aligned-dir <session>_aligned --sequential"
        )


def session_columns(root: zarr.Group, episode_names: list[str]) -> SessionColumns:
    """Decide the optional columns from the robot-action groups' ``tactile``
    attr and the recorded tactile groups; every episode and hand must agree."""
    converted: dict[bool, list[str]] = {}
    recorded: dict[bool, list[str]] = {}
    problems = []
    for name in episode_names:
        episode = root[name]
        for side in SIDES:
            for template in (ROBOT_STATE_GROUP, ROBOT_ACTION_GROUP):
                group_name = template.format(side=side)
                attrs = episode[group_name].attrs
                if "tactile" not in attrs:
                    problems.append(f"{name}/{group_name} has no tactile attr")
                    continue
                converted.setdefault(bool(attrs["tactile"]), []).append(
                    f"{name}/{group_name}"
                )
                if not attrs["tactile"]:
                    stale = [
                        path.format(side=side)
                        for path in TACTILE_ONLY_ARRAYS
                        if path.format(side=side).startswith(group_name + "/")
                        and path.format(side=side) in episode
                    ]
                    problems += [
                        f"{name}/{path} is left from a tactile conversion"
                        for path in stale
                    ]
            group_name = TACTILE_GROUP.format(side=side)
            recorded.setdefault(group_name in episode, []).append(
                f"{name}/{group_name}"
            )
    if len(converted) > 1:
        problems.append(
            "the robot-action groups mix tactile and kinematic conversions "
            f"(kinematic: {converted[False][:2]} ...)"
        )
    if problems:
        shown = "; ".join(problems[:4]) + (" ..." if len(problems) > 4 else "")
        raise ValueError(f"Robot actions are inconsistent ({shown}); {RERUN_HINT}")
    if len(recorded) > 1:
        raise ValueError(
            "Only some episodes or hands recorded tactile data (missing: "
            f"{recorded[False][:4]}); the dataset needs the same columns in "
            "every episode"
        )
    return SessionColumns(
        tactile_actions=next(iter(converted), False),
        tactile_state=next(iter(recorded), False),
    )


def read_hand_tactile(group: zarr.Group, side: str, length: int) -> list:
    """The hand's fingertip forces in newtons, one row per frame: 6 values (fx,
    fy, fz at both measurement points) per sensor, sensors ascending."""
    name = TACTILE_GROUP.format(side=side)
    channels = []
    for key in tactile_channel_keys():
        if key not in group[name]:
            raise ValueError(f"{name}/{key} is missing")
        array = group[name][key]
        verify_zarr_chunks_present(array)
        values = np.asarray(array, dtype=np.float64).reshape(-1)
        if len(values) != length:
            raise ValueError(
                f"{name}/{key} has {len(values)} frames, expected {length}"
            )
        channels.append(values)
    forces = np.stack(channels, axis=1) * TACTILE_SCALE
    if not np.isfinite(forces).all():
        raise ValueError(f"{name} has non-finite values")
    return forces.tolist()


def read_hand_joints(group: zarr.Group, template: str, side: str, length: int) -> list:
    array = group[template.format(side=side)]
    verify_zarr_chunks_present(array)
    values = np.asarray(array, dtype=np.float64)
    if values.ndim != 2 or len(values) != length:
        raise ValueError(
            f"{template.format(side=side)} has shape {values.shape}, expected "
            f"({length}, joints)"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{template.format(side=side)} has non-finite values")
    return values.tolist()


def expected_intrinsics_columns(episode_group: Any, video_keys: list[str]) -> list[str]:
    """``{cam}.intrinsics`` for each exported camera with recorded intrinsics."""
    return [
        f"{cam}.intrinsics"
        for cam in video_keys
        if cam in episode_group
        and "intrinsics" in episode_group[cam]
        and "aligned_frame_indices" in episode_group[cam]
    ]


def process_episode(
    session_dir: Path,
    spec: EpisodeConversionSpec,
    output_dir: Path,
    video_keys: list[str],
    fps: int,
    video_variants: bool = True,
    stereo_variant: bool = True,
    columns: SessionColumns = SessionColumns(),
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
    for side in SIDES:
        cam = TRACKING_CAMERAS[side]
        pose_field = resolve_pose_interp_field(group[cam]) if cam in group else None
        if pose_field is None:
            raise ValueError(f"{spec.episode_name}: no aligned pose for {cam}")
        pose_array = group[cam][pose_field]
        verify_zarr_chunks_present(pose_array)
        T_world_tracker = np.asarray(pose_array)
        verify_no_zero_pose_matrices(T_world_tracker, f"{cam}.{pose_field}")
        if len(T_world_tracker) != length:
            raise ValueError(f"Length mismatch for {cam}.{pose_field}")
        data[tracker_column(side)] = transform_to_xyz_rot6d(T_world_tracker)
        data[joints_column(side)] = read_hand_joints(
            group, ROBOT_STATE_ARRAY, side, length
        )
        data[action_joints_column(side)] = read_hand_joints(
            group, ROBOT_ACTION_ARRAY, side, length
        )
        if columns.tactile_actions:
            data[action_source_column(torque_key(side))] = read_hand_joints(
                group, ROBOT_TORQUE_ARRAY, side, length
            )
        if columns.tactile_state:
            data[tactile_column(side)] = read_hand_tactile(group, side, length)

    for column in expected_intrinsics_columns(group, video_keys):
        cam = column[: -len(".intrinsics")]
        intrinsics_array = group[cam]["intrinsics"]
        verify_zarr_chunks_present(intrinsics_array)
        K = np.asarray(intrinsics_array)
        aligned_idx = np.asarray(group[cam]["aligned_frame_indices"])
        if len(aligned_idx) != length:
            raise ValueError(f"Length mismatch for {cam}.intrinsics")
        data[column] = K[aligned_idx].reshape(length, 9).tolist()

    data["index"] = list(range(spec.frame_start_index, spec.frame_start_index + length))
    selected = ["timestamp", "frame_index", "episode_index", "task_index"]
    selected += columns.state_keys()
    for key in columns.action_keys():
        if action_source_column(key) not in selected:
            selected.append(action_source_column(key))
    selected += sorted(key for key in data if key.endswith(".intrinsics"))
    selected.append("index")
    frame = pd.DataFrame({key: data[key] for key in selected})
    data_path = output_dir / episode_data_rel_path(spec.episode_index)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(data_path)

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
    episode_attrs = normalize_metadata(dict(group.attrs))
    return {
        "episode_index": spec.episode_index,
        "tasks": [spec.task_description],
        "length": length,
        # The aligned episode this one was converted from, on this machine.
        "original_path": str(Path(session_dir).resolve() / spec.episode_name),
        "original_episode_index": spec.episode_name,
        "episode_attrs": episode_attrs,
    }


def write_episode_source_marker(output_dir: Path, spec: EpisodeConversionSpec) -> Path:
    """Record which source episode this output index was built from."""
    path = (
        output_dir
        / f"{SOURCE_EPISODES_DIR}/episode_{spec.episode_index:06d}.{spec.episode_name}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"episode_index": spec.episode_index, "source_episode": spec.episode_name},
            indent=2,
        )
        + "\n"
    )
    return path


def modality_entry(key: str, original_key: str, width: int) -> dict[str, Any]:
    return {
        "original_key": original_key,
        "start": 0,
        "end": width,
        "rotation_type": "rotation_6d" if key.endswith(".eef") else None,
        "absolute": True,
        "dtype": "float64",
        "range": None,
    }


def finalize_dataset(
    session_dir: Path,
    output_dir: Path,
    video_keys: list[str],
    episodes: list[dict[str, Any]],
    task_to_index: dict[str, int],
    fps: int,
    columns: SessionColumns = SessionColumns(),
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

    parquet_paths = sorted((output_dir / "data").glob("*/*.parquet"))
    frames_by_path = [(path, pd.read_parquet(path)) for path in parquet_paths]
    verify_episode_frames_consistent(frames_by_path)
    first_df = frames_by_path[0][1]

    def width(column: str) -> int:
        return int(np.asarray(first_df[column].iloc[0]).size)

    intrinsics_columns = sorted(
        c for c in first_df.columns if c.endswith(".intrinsics")
    )
    modality = {
        "state": {
            key: modality_entry(key, key, width(key))
            for key in [*columns.state_keys(), *intrinsics_columns]
        },
        "action": {
            key: modality_entry(
                key, action_source_column(key), width(action_source_column(key))
            )
            for key in columns.action_keys()
        },
        "video": {video_modality_key(key): {"original_key": key} for key in video_keys},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    (meta_dir / "modality.json").write_text(json.dumps(modality, indent=2))
    (meta_dir / "embodiment.json").write_text(
        json.dumps(
            {
                "robot_type": ROBOT_TYPE,
                "fps": fps,
                "num_episodes": len(episodes),
                "dataset_version": "1.0",
                "source_format": "zarr",
                "embodiment_tag": EMBODIMENT_TAG,
            },
            indent=2,
        )
    )

    features = {col: column_feature(first_df[col]) for col in first_df.columns}
    for key in video_keys:
        features[key] = video_metadata(
            output_dir / f"videos/chunk-000/{key}/episode_000000.mp4"
        )
    info = {
        "codebase_version": "v2.1",
        "robot_type": ROBOT_TYPE,
        "total_episodes": len(episodes),
        "total_frames": sum(ep["length"] for ep in episodes),
        "total_tasks": len(task_to_index),
        "total_videos": len(video_keys) * len(episodes),
        "total_chunks": len(episodes) // 1000 + 1,
        "chunks_size": 1000,
        "fps": float(fps),
        "splits": {"train": "0:100"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "recording_config": read_recording_config(session_dir),
        "dexumi_geometry": {
            "tracker_pose": "raw iPhone tracker pose; no glove-mount offset applied",
            "hand_joints": "robot-action conversion (kinematic), radians",
        },
    }
    if columns.tactile_actions:
        info["dexumi_geometry"]["hand_joint_torques"] = (
            "tau = B^T J^T f from the fingertip forces, N*m"
        )
    if columns.tactile_state:
        info["dexumi_geometry"]["hand_tactile"] = (
            f"fingertip forces in N, sensors {list(TACTILE_SENSOR_IDS)} "
            "(thumb..pinky), each fx, fy, fz at measurement point 1 then 2"
        )
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

    df = pd.concat([frame for _, frame in frames_by_path], axis=0)
    stats = {col: stats_for_series(df[col]) for col in df.columns}
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2))


def process_episode_relative_stats(
    args: tuple[Path, list[str], list[tuple[str, str, str]], int],
) -> dict[str, np.ndarray]:
    """Action-chunk deltas of one episode: SE(3) deltas for the tracker poses,
    plain differences for the hand joints, both relative to the current state.

    ``joint_pairs`` lists ``(modality key, state column, action column)``."""
    parquet_path, pose_columns, joint_pairs, max_action_chunk_size = args
    df = pd.read_parquet(parquet_path)
    if len(df) < 2:
        return {}
    result: dict[str, np.ndarray] = {}
    for col in pose_columns:
        state = np.stack(df[col].values).astype(np.float64)
        chunks, valid_mask = build_padded_action_chunks_and_mask(
            state, max_action_chunk_size
        )
        delta = compute_eef_delta(chunks, state[:, None, :])
        delta[~valid_mask] = np.nan
        result[col] = delta
    for key, state_col, action_col in joint_pairs:
        state = np.stack(df[state_col].values).astype(np.float64)
        action = np.stack(df[action_col].values).astype(np.float64)
        chunks, valid_mask = build_padded_action_chunks_and_mask(
            action, max_action_chunk_size
        )
        delta = chunks - state[:, None, :]
        delta[~valid_mask] = np.nan
        result[key] = delta
    return result


def calculate_relative_stats(
    output_dir: Path, *, max_action_chunk_size: int = 80, num_workers: int = 1
) -> None:
    """``meta/relative_stats.json``: statistics of the action chunks relative to the state."""
    info = json.loads((output_dir / "meta" / "info.json").read_text())
    pose_columns = [tracker_column(side) for side in SIDES]
    # Joint-position actions relative to the joint state. Torques are absolute
    # and get no entry.
    joint_pairs = []
    for side in SIDES:
        key = joints_column(side)
        if action_source_column(key) in info.get("features", {}):
            joint_pairs.append((key, key, action_source_column(key)))
    paths = sorted((output_dir / "data").glob("*/*.parquet"))
    if not paths or not info.get("total_episodes"):
        return
    args = [(path, pose_columns, joint_pairs, max_action_chunk_size) for path in paths]
    workers = min(max(1, num_workers), len(paths))
    if workers <= 1:
        results = [process_episode_relative_stats(arg) for arg in args]
    else:
        with spawn_process_pool(workers) as executor:
            results = list(executor.map(process_episode_relative_stats, args))
    deltas: dict[str, list[np.ndarray]] = {}
    for result in results:
        for col, values in result.items():
            deltas.setdefault(col, []).append(values)
    relative_stats = {
        col: stats_for_relative_array(np.concatenate(values, axis=0))
        for col, values in deltas.items()
    }
    if relative_stats:
        (output_dir / "meta" / "relative_stats.json").write_text(
            json.dumps(relative_stats, indent=2)
        )


def build_episode_specs(
    root: zarr.Group, episode_names: list[str]
) -> tuple[list[EpisodeConversionSpec], dict[str, int]]:
    task_to_index: dict[str, int] = {"": 0}
    specs = []
    frame_start_index = 0
    for idx, episode_name in enumerate(episode_names):
        episode_group = root[episode_name]
        task_description = get_task_description(episode_group)
        task_index = task_to_index.setdefault(task_description, len(task_to_index))
        specs.append(
            EpisodeConversionSpec(
                episode_name=episode_name,
                episode_index=idx,
                frame_start_index=frame_start_index,
                task_description=task_description,
                task_index=task_index,
            )
        )
        frame_start_index += len(np.asarray(episode_group["valid_timestamps"]))
    return specs, task_to_index


def convert_session(
    session_dir: Path,
    output_dir: Path,
    fps: int | None,
    episode_workers: int = 1,
    video_variants: bool = True,
    stereo_variant: bool = True,
) -> list[str]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing non-empty output: {output_dir}"
        )
    verify_embodiment(session_dir)
    root = zarr.open_group(str(session_dir), mode="r")
    episode_names = sorted(
        (str(k) for k in root.group_keys() if str(k).startswith("episode_")),
        key=source_episode_sort_key,
    )
    if not episode_names:
        raise RuntimeError(f"No aligned episodes in {session_dir}")
    verify_aligned_timestamps(root, episode_names)
    verify_robot_actions(root, episode_names)
    columns = session_columns(root, episode_names)
    print(
        f"Tactile action columns: {'yes' if columns.tactile_actions else 'no'}; "
        f"tactile state column: {'yes' if columns.tactile_state else 'no'}",
        flush=True,
    )
    video_keys = session_video_keys(session_dir, episode_names)
    stereo_keys = []
    if video_variants and stereo_variant:
        stereo_keys = session_stereo_keys(session_dir, root, episode_names, video_keys)
    output_keys = output_video_keys(video_keys, video_variants, stereo_keys)
    specs, task_to_index = build_episode_specs(root, episode_names)
    if fps is None:
        fps = session_fps(root, episode_names)
        print(
            f"Frame rate {fps} fps (measured from the aligned timestamps)", flush=True
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    job_args = (output_dir, video_keys, fps, video_variants, stereo_variant, columns)
    episodes: list[dict[str, Any]] = []
    worker_count = min(max(1, episode_workers), len(specs))
    if worker_count == 1:
        for spec in specs:
            print(f"Converting {session_dir.name}/{spec.episode_name}", flush=True)
            episodes.append(process_episode(session_dir, spec, *job_args))
    else:
        print(
            f"Converting {len(specs)} episodes with {worker_count} workers", flush=True
        )
        with spawn_process_pool(worker_count) as executor:
            futures = {
                executor.submit(process_episode, session_dir, spec, *job_args): spec
                for spec in specs
            }
            try:
                for future in as_completed(futures):
                    episodes.append(future.result())
                    print(
                        f"Finished {session_dir.name}/{futures[future].episode_name}",
                        flush=True,
                    )
            except Exception:
                for future in futures:
                    future.cancel()
                raise
    episodes.sort(key=lambda episode: episode["episode_index"])
    finalize_dataset(
        session_dir, output_dir, output_keys, episodes, task_to_index, fps, columns
    )
    return episode_names


def session_fps(root: zarr.Group, episode_names: list[str]) -> int:
    """The aligned frame rate: the index camera's rate (``record_fps``, 45 on the
    station rig), from the median spacing of ``valid_timestamps``."""
    steps = np.concatenate(
        [np.diff(np.asarray(root[name]["valid_timestamps"])) for name in episode_names]
    )
    steps = steps[steps > 0]
    if not steps.size:
        raise ValueError("Cannot measure the frame rate: too few aligned frames")
    return int(round(1.0 / float(np.median(steps))))


def rebuild_dataset(
    session_dir: Path, output_dir: Path, fps: int | None = None
) -> list[str]:
    """Rebuild a dataset when conversion is requested with ``--replace``.

    Build beside the existing output and replace it only after conversion and
    statistics succeed. A conversion failure or interrupt preserves the
    existing dataset.
    """
    import shutil

    partial = output_dir.with_name(output_dir.name + ".partial")
    previous = output_dir.with_name(output_dir.name + ".previous")
    for stale in (partial, previous):
        if stale.exists():
            shutil.rmtree(stale)
    try:
        episodes = convert_session(session_dir, partial, fps)
        calculate_relative_stats(partial)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    if output_dir.exists():
        output_dir.rename(previous)
    partial.rename(output_dir)
    shutil.rmtree(previous, ignore_errors=True)
    return episodes


def default_output_dataset(session_dir: Path) -> Path:
    """``<session>_lerobot`` next to ``<session>`` and ``<session>_aligned``."""
    name = session_dir.name.removesuffix(ALIGNED_SUFFIX)
    return session_dir.with_name(name + LEROBOT_SUFFIX)


def convert_robot_actions(session_dir: Path, tactile: bool = True) -> None:
    """Write the robot-hand state and action of every episode into the aligned
    session, with the tactile variants unless ``tactile`` is False (the same
    default as ``./launch.sh convert``); raises when any episode fails."""
    from real_scripts.dexumi_pipeline.convert_to_robot_action import (
        convert_all_episodes,
    )

    mode = "tactile" if tactile else "kinematics only"
    print(f"Converting the glove data to robot actions ({mode})", flush=True)
    results = convert_all_episodes(session_dir, tactile=tactile)
    failed = [f"episode_{idx}: {err}" for idx, ok, err in results if not ok]
    if failed:
        raise RuntimeError("robot-action conversion failed: " + "; ".join(failed))
    print(f"Robot actions written for {len(results)} episode(s)", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an aligned DexUMI session to robot actions and a LeRobot dataset."
    )
    parser.add_argument(
        "--input-dataset",
        "-d",
        required=True,
        type=Path,
        help="The aligned session (<session>_aligned)",
    )
    parser.add_argument(
        "--output-dataset",
        "-o",
        type=Path,
        help="Where to write the dataset. Default <session>_lerobot next to the aligned session",
    )
    parser.add_argument(
        "--fps",
        type=int,
        help="Frame rate of the dataset and its videos. Default: measured from the "
        "aligned timestamps (the rate of the alignment index camera)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Rebuild the dataset when one already exists (it is replaced only "
        "once the new one is complete)",
    )
    parser.add_argument(
        "--no-tactile",
        dest="tactile",
        action="store_false",
        help="Run the robot-action step kinematics only (as ./launch.sh convert "
        "--no-tactile), for a session without tactile recordings; the dataset "
        "then has no joint_torques columns. By default the torque actions are "
        "written into the aligned session and exported. The <side>_hand.tactile "
        "column is exported whenever the session recorded tactile data.",
    )
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
    args = parser.parse_args(argv)
    if args.fps is not None and args.fps < 1:
        parser.error("--fps must be at least 1")
    if args.episode_workers < 1:
        parser.error("--episode-workers must be at least 1")
    if args.relative_stats_chunk_size < 1:
        parser.error("--relative-stats-chunk-size must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    session_dir = args.input_dataset.expanduser().resolve()
    if not (session_dir / "zarr.json").is_file():
        print(f"Not an aligned Zarr session: {session_dir}", file=sys.stderr)
        return 1
    output_dir = (
        args.output_dataset or default_output_dataset(session_dir)
    ).expanduser()
    try:
        if output_dir.exists() and any(output_dir.iterdir()) and not args.replace:
            raise FileExistsError(
                f"{output_dir} already exists; pass --replace to rebuild it"
            )
        convert_robot_actions(session_dir, tactile=args.tactile)
        if args.replace:
            episodes = rebuild_dataset(session_dir, output_dir, args.fps)
        else:
            episodes = convert_session(
                session_dir,
                output_dir,
                args.fps,
                episode_workers=args.episode_workers,
                video_variants=args.video_variants,
                stereo_variant=args.stereo_variant,
            )
            if not args.no_relative_stats:
                calculate_relative_stats(
                    output_dir,
                    max_action_chunk_size=args.relative_stats_chunk_size,
                    num_workers=args.episode_workers,
                )
    except Exception as exc:
        print(f"FAILED {session_dir.name}: {exc}", file=sys.stderr, flush=True)
        return 1
    print(f"Wrote {len(episodes)} episode(s) to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

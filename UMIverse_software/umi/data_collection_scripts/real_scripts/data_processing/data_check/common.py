# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared across check modules (timestamps, pose math, episode listing).

These functions are imported by multiple per-check modules; pulling them here
avoids both duplication and circular imports.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zarr

from umi.common.utility.tracking import (
    tracking_device_for_side,
    tracking_device_to_arm,
    tracking_devices_in,
)

from .constants import PENDING_SUFFIX

__all__ = [
    "tracking_devices_in",
    "tracking_device_for_side",
    "tracking_device_to_arm",
    "get_device_timestamps",
    "get_device_expected_fps",
    "get_device_expected_resolutions",
    "get_latency_adjusted_timestamps",
    "_consecutive_position_distances",
    "_max_pairwise_position_distance",
    "extract_episode_index",
    "get_episode_keys",
    "get_episode_list",
]


def get_device_timestamps(
    episode_data: zarr.Group, device_name: str
) -> Optional[np.ndarray]:
    """Get timestamps for a device, preferring capture_time over receive_time.

    Args:
        episode_data: Zarr group for the episode
        device_name: Name of the device

    Returns:
        Numpy array of timestamps or None if not found
    """
    if device_name not in episode_data:
        return None

    device_data = episode_data[device_name]

    # Prefer capture_time, fall back to receive_time
    if "capture_time" in device_data:
        return np.array(device_data["capture_time"][:])
    elif "receive_time" in device_data:
        return np.array(device_data["receive_time"][:])
    return None


def get_device_expected_fps(config: Dict[str, Any]) -> Dict[str, float]:
    """Build a per-device expected FPS map from the recording config.

    Each video/numeric source can specify its own ``record_fps``.
    Devices without an explicit value inherit the global
    ``recording.record_fps`` default.

    Returns:
        Dict mapping device name -> expected FPS.
    """
    default_fps = config.get("recording", {}).get("record_fps", 45)
    fps_map: Dict[str, float] = {}
    for source in config.get("video_sources", []):
        name = source.get("name")
        if name:
            fps_map[name] = source.get("record_fps", default_fps)
    for source in config.get("numeric_sources", []):
        name = source.get("name")
        if name:
            fps_map[name] = source.get("record_fps", default_fps)
    return fps_map


def get_device_expected_resolutions(
    config: Dict[str, Any],
) -> Dict[str, Tuple[int, int]]:
    """Build device -> (width, height) from dataset config.yaml video_sources.

    Reads params.camera_resolution [w, h] for each video source. Devices without
    camera_resolution are omitted (no resolution check will run for them).
    """
    out: Dict[str, Tuple[int, int]] = {}
    for source in config.get("video_sources", []):
        name = source["name"]
        res = source["params"].get("camera_resolution")
        if source["type"] != "iphone_camera":
            assert res is not None, f"camera_resolution is required for {name}"
        if not name or not res or len(res) < 2:
            continue
        try:
            out[name] = (int(res[0]), int(res[1]))
        except (TypeError, ValueError):
            continue
    return out


def get_latency_adjusted_timestamps(
    episode_data: zarr.Group,
    device_name: str,
    latency: float = 0.0,
) -> np.ndarray:
    """Get latency adjusted timestamps for a device.

    Args:
        episode_data: Zarr group containing episode data
        device_name: Name of the device
    Returns:
        Numpy array of timestamps or None if not found
    """
    device_data = episode_data[device_name]
    if "capture_time" in device_data:
        return np.array(device_data["capture_time"][:])
    elif "receive_time" in device_data:
        return np.array(device_data["receive_time"][:])
    return None


def _consecutive_position_distances(poses: np.ndarray) -> np.ndarray:
    """Vectorized consecutive position distances (meters). Shape (N-1,)."""
    positions = poses[:, :3, 3].astype(np.float64)
    return np.linalg.norm(np.diff(positions, axis=0), axis=1)


def _max_pairwise_position_distance(positions: np.ndarray) -> float:
    """Max pairwise position distance over (n, 3) positions. Vectorized."""
    if len(positions) < 2:
        return 0.0
    diff = positions[:, None, :] - positions[None, :, :]
    return float(np.max(np.linalg.norm(diff, axis=2)))


def extract_episode_index(episode_key: str) -> int:
    """Extract episode index from episode key.

    Args:
        episode_key: Episode key (e.g., "episode_0" or "episode_0.pending")

    Returns:
        Episode index (always extractable for valid episode keys)

    Raises:
        ValueError: If episode index cannot be extracted from the key
    """
    try:
        if episode_key.endswith(PENDING_SUFFIX):
            idx_str = episode_key[len("episode_") : -len(PENDING_SUFFIX)]
            return int(idx_str)
        else:
            return int(episode_key.split("_")[1])
    except (ValueError, IndexError) as e:
        raise ValueError(
            f"Failed to extract episode index from key '{episode_key}'. "
            f"Expected format: 'episode_N' or 'episode_N.pending'"
        ) from e


def get_episode_keys(data_dir: str, pending: bool = False) -> List[str]:
    """Get all episode keys in a data directory.

    Args:
        data_dir: Path to the data directory
        pending: If True, return pending episodes (episode_N.pending) instead of finalized episodes.

    Returns:
        List of episode keys sorted by index (if extractable) or alphabetically
    """
    data_store = zarr.open(data_dir, mode="r")

    episode_keys = []
    for key in data_store.group_keys():
        if pending:
            if key.startswith("episode_") and key.endswith(PENDING_SUFFIX):
                episode_keys.append(key)
        else:
            if key.startswith("episode_") and not key.endswith(PENDING_SUFFIX):
                episode_keys.append(key)

    # Sort by index if extractable, otherwise alphabetically
    def sort_key(key: str) -> Tuple[int, int]:
        idx = extract_episode_index(key)
        if idx is not None:
            return (0, idx)  # Use 0 as first element to prioritize numeric sorting
        return (1, 0)  # Use 1 to put non-numeric keys after numeric ones

    episode_keys.sort(key=sort_key)
    return episode_keys


def get_episode_list(data_dir: str) -> List[Tuple[int, str]]:
    """List all episodes in a data directory with their indices and folder names.

    Args:
        data_dir: Path to the data directory

    Returns:
        List of tuples (episode_index, folder_name) sorted by index
    """
    episode_keys = get_episode_keys(data_dir, pending=False)

    episodes = []
    for key in episode_keys:
        try:
            idx = int(key.split("_")[1])
            episodes.append((idx, key))
        except ValueError:
            continue

    episodes.sort(key=lambda x: x[0])
    return episodes

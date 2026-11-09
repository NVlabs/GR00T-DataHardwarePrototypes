# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gripper sensor checks: movement, jumps, range, visualization, serial consistency."""

import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import zarr

__all__ = [
    "check_gripper_movement",
    "check_gripper_jumps",
    "check_gripper_range",
    "visualize_gripper_data",
    "check_gripper_serial_consistency",
]


def check_gripper_movement(
    distance: np.ndarray, threshold: float = 10.0
) -> Dict[str, Any]:
    """Check if gripper moved during episode (distance change > threshold mm).

    Args:
        distance: Array of gripper distance values in mm
        threshold: Minimum movement threshold in mm (default 10mm = 1cm)

    Returns:
        Dict with movement status and range information
    """
    if len(distance) == 0:
        return {
            "moved": False,
            "range_mm": 0.0,
            "min_mm": 0.0,
            "max_mm": 0.0,
            "error": "No distance data",
        }

    min_dist = float(np.min(distance))
    max_dist = float(np.max(distance))
    range_mm = max_dist - min_dist

    return {
        "moved": range_mm > threshold,
        "range_mm": round(range_mm, 2),
        "min_mm": round(min_dist, 2),
        "max_mm": round(max_dist, 2),
    }


def check_gripper_jumps(
    distance: np.ndarray,
    timestamps: np.ndarray,
    jump_threshold_mm_per_s: float = 2000.0,
) -> Dict[str, Any]:
    """Detect jump readings in gripper distance, scored as speed.

    Sample gaps alternate ~22/~44 ms (45 Hz stream into a 30 Hz recording), so a
    flat per-sample threshold is twice as strict on the long gaps.

    Args:
        distance: Array of gripper distance values in mm
        timestamps: Per-sample timestamps in seconds
        jump_threshold_mm_per_s: Speed ceiling in mm/s

    Returns:
        Dict with has_jumps, jump_frames, num_jumps, jump_threshold_mm_per_s
    """
    result = {
        "has_jumps": False,
        "jump_frames": [],
        "num_jumps": 0,
        "jump_threshold_mm_per_s": jump_threshold_mm_per_s,
    }

    if len(distance) < 2:
        return result

    diffs = np.abs(np.diff(distance))
    dt = np.diff(np.asarray(timestamps, dtype=float))
    # Non-positive gaps mean broken timestamps, which the timestamp checks
    # report separately; a speed across them is meaningless.
    speeds = np.divide(diffs, dt, out=np.zeros_like(diffs, dtype=float), where=dt > 0)

    for idx in np.where(speeds > jump_threshold_mm_per_s)[0]:
        result["jump_frames"].append(
            {
                "frame_index": int(idx + 1),  # +1 to point to the frame that jumped
                "before_value": round(float(distance[idx]), 2),
                "after_value": round(float(distance[idx + 1]), 2),
                "magnitude_mm": round(float(diffs[idx]), 2),
                "magnitude_mm_per_s": round(float(speeds[idx]), 2),
            }
        )

    result["has_jumps"] = len(result["jump_frames"]) > 0
    result["num_jumps"] = len(result["jump_frames"])

    return result


def check_gripper_range(
    distance: np.ndarray, min_val: float = 0.0, max_val: float = 90.0
) -> Dict[str, Any]:
    """Check if gripper distance values are within reasonable range.

    Args:
        distance: Array of gripper distance values in mm
        min_val: Minimum valid distance (default 0)
        max_val: Maximum valid distance (default 90mm, matching `range_max_mm`
            in every production data-check config and the fallback in
            `dispatch`. Most recorded episodes peak above 80mm at full open,
            and none above 90mm.)

    Returns:
        Dict with range status and out-of-range frames
    """
    if len(distance) == 0:
        return {"in_range": True, "out_of_range_frames": [], "num_out_of_range": 0}

    out_of_range_mask = (distance < min_val) | (distance > max_val)
    out_of_range_indices = np.where(out_of_range_mask)[0]

    out_of_range_frames = []
    for idx in out_of_range_indices:
        out_of_range_frames.append(
            {
                "frame_index": int(idx),
                "value": round(float(distance[idx]), 2),
            }
        )

    return {
        "in_range": len(out_of_range_frames) == 0,
        "out_of_range_frames": out_of_range_frames,
        "num_out_of_range": len(out_of_range_frames),
        "expected_range": [min_val, max_val],
        "fixable_indices": out_of_range_indices,
        "num_fixable": len(out_of_range_indices),
        "num_remaining_fixable": 0,
    }


def visualize_gripper_data(
    joint_positions: np.ndarray,
    distance: Optional[np.ndarray],
    episode_name: str,
    device_name: str,
    jump_frames: List[Dict[str, Any]],
    jump_threshold_mm_per_s: float = 2000.0,
    output_dir: Optional[str] = None,
) -> None:
    """Plot gripper joint positions and distance data with jump markers.

    Args:
        joint_positions: Array of gripper joint position values in degrees
        distance: Array of gripper distance values in mm (optional)
        episode_name: Name of the episode for the title
        device_name: Name of the gripper device
        jump_frames: List of detected jump frames from check_gripper_jumps
        jump_threshold_mm_per_s: Speed ceiling in mm/s used for detection
        output_dir: Directory to save plot. If None, displays interactively.
    """
    num_plots = 2 if distance is not None else 1
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots), sharex=True)

    if num_plots == 1:
        axes = [axes]

    frames = np.arange(len(joint_positions))

    ax_jp = axes[0]
    ax_jp.plot(frames, joint_positions, "b-", linewidth=0.8, label="Joint Positions")

    if jump_frames:
        jump_indices = [j["frame_index"] for j in jump_frames]
        jump_values = [joint_positions[i] for i in jump_indices]
        ax_jp.scatter(
            jump_indices,
            jump_values,
            c="red",
            s=50,
            zorder=5,
            label=f"Jumps (>{jump_threshold_mm_per_s:.0f} mm/s)",
        )

    ax_jp.set_ylabel("Joint Positions (degrees)")
    ax_jp.set_title(f"{episode_name} - {device_name}")
    handles, labels = ax_jp.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_jp.legend(by_label.values(), by_label.keys(), loc="upper right")
    ax_jp.grid(True, alpha=0.3)

    if distance is not None:
        ax_dist = axes[1]
        dist_frames = np.arange(len(distance))
        ax_dist.plot(dist_frames, distance, "g-", linewidth=0.8, label="Distance")

        if jump_frames:
            jump_indices = [j["frame_index"] for j in jump_frames]
            valid_indices = [i for i in jump_indices if i < len(distance)]
            if valid_indices:
                jump_dist_values = [distance[i] for i in valid_indices]
                ax_dist.scatter(
                    valid_indices,
                    jump_dist_values,
                    c="red",
                    s=50,
                    zorder=5,
                    label="Jumps",
                )

        ax_dist.set_ylabel("Distance (mm)")
        ax_dist.set_xlabel("Frame")
        handles, labels = ax_dist.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_dist.legend(by_label.values(), by_label.keys(), loc="upper right")
        ax_dist.grid(True, alpha=0.3)
    else:
        ax_jp.set_xlabel("Frame")

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{episode_name}_{device_name}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=100)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def check_gripper_serial_consistency(
    data_store: zarr.Group,
    episode_keys: List[str],
) -> Dict[str, Any]:
    """Check that gripper device names and serials are consistent across all episodes.

    Args:
        data_store: Zarr store containing the dataset
        episode_keys: List of episode keys to check

    Returns:
        Dict with consistency check results
    """
    result = {
        "passed": True,
        "issues": [],
        "gripper_serials": {},  # {gripper_name: {device_name, serial}}
        "inconsistent_episodes": [],
    }

    reference_serials = None

    for episode_key in episode_keys:
        if episode_key not in data_store:
            continue

        episode_data = data_store[episode_key]
        gripper_serial = episode_data.attrs.get("gripper_serial", {})

        if not gripper_serial:
            continue

        if reference_serials is None:
            reference_serials = gripper_serial
            result["gripper_serials"] = gripper_serial
        else:
            for gripper_name, ref_info in reference_serials.items():
                if gripper_name not in gripper_serial:
                    result["passed"] = False
                    result["issues"].append(
                        f"{episode_key}: missing {gripper_name} serial data"
                    )
                    result["inconsistent_episodes"].append(episode_key)
                    continue

                curr_info = gripper_serial[gripper_name]
                if curr_info.get("device_name") != ref_info.get("device_name"):
                    result["passed"] = False
                    result["issues"].append(
                        f"{episode_key}: {gripper_name} device_name mismatch "
                        f"(expected {ref_info.get('device_name')}, got {curr_info.get('device_name')})"
                    )
                    result["inconsistent_episodes"].append(episode_key)
                if curr_info.get("serial") != ref_info.get("serial"):
                    result["passed"] = False
                    result["issues"].append(
                        f"{episode_key}: {gripper_name} serial mismatch "
                        f"(expected {ref_info.get('serial')}, got {curr_info.get('serial')})"
                    )
                    if episode_key not in result["inconsistent_episodes"]:
                        result["inconsistent_episodes"].append(episode_key)

    if reference_serials is None:
        result["issues"].append("No gripper serial data found in any episode")

    return result

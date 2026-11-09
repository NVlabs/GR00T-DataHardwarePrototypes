# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data check package: the public API of the record-time data check.

Callers (record_data.py, the tests and the examples) import from
``real_scripts.data_processing.data_check``; every public symbol is re-exported
here so that import path stays stable.
"""

import logging

from .common import (
    _consecutive_position_distances,
    _max_pairwise_position_distance,
    extract_episode_index,
    get_device_expected_fps,
    get_device_expected_resolutions,
    get_device_timestamps,
    get_episode_keys,
    get_episode_list,
    get_latency_adjusted_timestamps,
)
from .config_loader import check_enabled, is_enabled, load_check_config
from .constants import (
    DEFAULT_CHECK_CONFIG_PATH,
    PENDING_SUFFIX,
)
from .device_names import (
    check_config_device_names,
    check_device_name_consistency,
    declared_device_names,
    expected_device_names,
)
from .dispatch import check_dataset, check_episode
from .gripper import (
    check_gripper_jumps,
    check_gripper_movement,
    check_gripper_range,
    check_gripper_serial_consistency,
    visualize_gripper_data,
)
from .motion_alignment import (
    aperture_velocity_cells,
    check_motion_alignment,
    combine_flow_cells,
    compute_flow_cell_speeds,
    estimate_flow_offset,
    estimate_flow_offset_rank,
    gripper_distance_reference,
    motion_bursts,
    per_burst_offsets,
    pose_speeds,
    pose_stall_flags,
    pose_tracking_agreement,
)
from .numeric import (
    check_numeric_array,
    check_numeric_values,
    run_numeric_value_checks,
)
from .operator_messages import (
    overlay_hint,
    overlay_lines,
    terminal_lines,
    warning_lines,
)
from .report import data_check_findings
from .timestamps import (
    check_consecutive_frame_intervals,
    check_fps_consistency,
    check_monotonic_timestamps,
    check_sustained_frame_drop,
    check_timestamp_coverage,
)
from .tracking import (
    check_consecutive_movement,
    check_pose_changes,
    check_tracking_excursions,
    check_tracking_pose_jumps,
)
from .video import check_video_resolution

# Module-level logger, available as `data_check.logger`.
logger = logging.getLogger("real_scripts.data_processing.data_check")

__all__ = [
    # JSON helpers
    # Config loader
    "load_check_config",
    "is_enabled",
    "check_enabled",
    # Constants
    "PENDING_SUFFIX",
    "DEFAULT_CHECK_CONFIG_PATH",
    # Helpers
    "get_device_timestamps",
    "get_device_expected_fps",
    "get_device_expected_resolutions",
    "get_latency_adjusted_timestamps",
    "_consecutive_position_distances",
    "_max_pairwise_position_distance",
    "extract_episode_index",
    "get_episode_keys",
    "get_episode_list",
    # Timestamp checks
    "check_fps_consistency",
    "check_monotonic_timestamps",
    "check_timestamp_coverage",
    "check_consecutive_frame_intervals",
    "check_sustained_frame_drop",
    # Gripper checks
    "check_gripper_movement",
    "check_gripper_jumps",
    "check_gripper_range",
    "visualize_gripper_data",
    "check_gripper_serial_consistency",
    # Tracking checks
    "check_pose_changes",
    "check_consecutive_movement",
    "check_tracking_pose_jumps",
    "check_tracking_excursions",
    # Device-name checks
    "declared_device_names",
    "expected_device_names",
    "check_config_device_names",
    "check_device_name_consistency",
    # Video resolution
    "check_video_resolution",
    # Numeric value checks
    "check_numeric_array",
    "check_numeric_values",
    "run_numeric_value_checks",
    # Arm-motion alignment check
    "compute_flow_cell_speeds",
    "pose_speeds",
    "pose_stall_flags",
    "motion_bursts",
    "combine_flow_cells",
    "estimate_flow_offset",
    "estimate_flow_offset_rank",
    "per_burst_offsets",
    "pose_tracking_agreement",
    "aperture_velocity_cells",
    "gripper_distance_reference",
    "check_motion_alignment",
    # Dispatchers
    "check_episode",
    "check_dataset",
    # Report
    "data_check_findings",
    # Operator-facing wording (HARDWARE vs MOTION)
    "terminal_lines",
    "overlay_lines",
    "overlay_hint",
    "warning_lines",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level orchestrators: check_episode (per-episode) and check_dataset (across all)."""

import logging
import os
import time
import traceback
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
import zarr
from tqdm import tqdm

from umi.common.utility.tracking import tracking_device_to_arm

from .common import (
    extract_episode_index,
    get_device_expected_fps,
    get_device_expected_resolutions,
    get_device_timestamps,
    get_episode_keys,
    tracking_devices_in,
)
from .config_loader import (
    check_enabled,
    is_enabled,
    load_check_config,
)
from .device_names import (
    check_config_device_names,
    check_device_name_consistency,
    expected_device_names,
)
from .gripper import (
    check_gripper_jumps,
    check_gripper_movement,
    check_gripper_range,
    check_gripper_serial_consistency,
)
from .numeric import run_numeric_value_checks
from .timestamps import (
    check_consecutive_frame_intervals,
    check_fps_consistency,
    check_monotonic_timestamps,
    check_sustained_frame_drop,
    check_timestamp_coverage,
)
from .tracking import (
    EXCURSION_DETECTOR_VERSION,
    check_consecutive_movement,
    check_pose_changes,
    check_tracking_excursions,
    check_tracking_pose_jumps,
    excursion_state_counts,
)
from .video import check_video_resolution

# Suppress zarr warnings about non-zarr files (e.g., .mp4 files in the store)
warnings.filterwarnings(
    "ignore", message=".*not recognized as a component of a Zarr hierarchy.*"
)

__all__ = ["check_episode", "check_dataset"]

# Episode warning prefix for a failure of a check the operator set to warn-only.
NOT_ENFORCED_PREFIX = "[not enforced]"


def _unenforced_sections(config: Dict[str, Any], check_config: Dict[str, Any]) -> set:
    """Data-check sections whose failures only warn: ``enforce: false`` in the
    data-check yaml, or ``check_enforcement.data.<section>: false`` in the
    recording config (the GUI's per-launch choice)."""
    relaxed = {
        section
        for section, block in check_config.items()
        if isinstance(block, dict) and block.get("enforce") is False
    }
    chosen = ((config or {}).get("check_enforcement") or {}).get("data") or {}
    relaxed.update(section for section, enforce in chosen.items() if enforce is False)
    return relaxed


def _relax_issues(results, section_starts, relaxed) -> None:
    """Turn the issues of warn-only sections into warnings.

    ``section_starts`` is ``[(section, index of its first issue), ...]`` in the
    order the sections ran, so a section owns the issues up to the next start.
    """
    if not relaxed or not section_starts:
        return
    issues = results["issues"]
    bounds = [start for _, start in section_starts] + [len(issues)]
    kept, moved = list(issues[: bounds[0]]), []
    for i, (section, _start) in enumerate(section_starts):
        target = moved if section in relaxed else kept
        target.extend(issues[bounds[i] : bounds[i + 1]])
    results["issues"] = kept
    if moved:
        results.setdefault("warnings", []).extend(
            f"{NOT_ENFORCED_PREFIX} {issue}" for issue in moved
        )


logger = logging.getLogger(__name__)


def _config_with_device_name_override(
    recording_config: Dict[str, Any], check_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Return a shallow copy of recording_config with `device_name_consistency`
    from check_config layered on top (minus the `enabled` flag).

    `expected_device_names` reads its `device_name_consistency` block from the
    dict it receives, but in practice the block is written in the data-check
    YAML, not the recording config. Without this shim, a check config that pins
    the expected names (including the session-local copy older sessions carry)
    would be silently ignored.
    """
    block = check_config.get("device_name_consistency")
    if not isinstance(block, dict):
        return recording_config
    merged = dict(recording_config)
    block_without_enabled = {k: v for k, v in block.items() if k != "enabled"}
    if block_without_enabled:
        merged["device_name_consistency"] = block_without_enabled
    return merged


# Detector kwargs read from `tracking.excursion_<kwarg>`.
_EXCURSION_DETECTOR_KEYS = (
    "dv_threshold_m_per_s",
    "min_amplitude_mm",
    "min_edge_share",
    "max_return_s",
    "return_floor_mm",
    "return_per_m_per_s_mm",
    "return_fraction",
    "context_s",
    "edge_rotation_deg_per_m",
    "max_events",
)
# Video-confirmation params: `tracking` key -> `confirm_excursions` param.
_EXCURSION_CONFIRM_KEYS = {
    "excursion_confirm_max": "confirm_max",
    "excursion_reject_min": "reject_min",
    "excursion_confirm_pad_s": "pad_s",
    "excursion_confirm_max_events": "max_events",
}


def _detect_excursions(
    episode_data: zarr.Group,
    tracking_device: str,
    poses: np.ndarray,
    tracking_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Excursion events for one tracking device. Never raises: a failure is
    recorded in the result, and the result alone never fails the episode."""
    try:
        return check_tracking_excursions(
            poses,
            get_device_timestamps(episode_data, tracking_device),
            **{
                key: tracking_config[f"excursion_{key}"]
                for key in _EXCURSION_DETECTOR_KEYS
                if f"excursion_{key}" in tracking_config
            },
        )
    except Exception as e:  # noqa: BLE001 - a measurement must not fail the check
        logger.warning("%s: excursion check failed: %s", tracking_device, e)
        return {
            "detector_version": EXCURSION_DETECTOR_VERSION,
            "error": f"{type(e).__name__}: {e}",
            "events": [],
        }


def _confirm_excursions(
    episode_data: zarr.Group,
    tracking_device: str,
    result: Dict[str, Any],
    tracking_config: Dict[str, Any],
    data_dir: str,
    episode_name: str,
    deadline: float,
) -> None:
    """Judge one device's candidates against the video, in place, and recount
    the states so the persisted counts agree with the events."""
    try:
        # Lazy import: pulls in cv2 only when there is something to judge
        from .excursion_confirm import confirm_excursions

        result["events"] = confirm_excursions(
            episode_data,
            data_dir,
            episode_name,
            tracking_device,
            result["events"],
            get_device_timestamps(episode_data, tracking_device),
            params={
                param: tracking_config[key]
                for key, param in _EXCURSION_CONFIRM_KEYS.items()
                if key in tracking_config
            },
            deadline=deadline,
        )
    except Exception as e:  # noqa: BLE001 - a measurement must not fail the check
        logger.warning("%s: excursion confirmation failed: %s", tracking_device, e)
        result["confirm_error"] = f"{type(e).__name__}: {e}"
    result.update(excursion_state_counts(result["events"]))


def check_episode(
    episode_data: zarr.Group,
    config: Dict[str, Any],
    check_config: Dict[str, Any],
    episode_name: str = "",
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run all validation checks on a single episode.

    Args:
        episode_data: Zarr group containing episode data
        config: Recording config dict
        check_config: Data check threshold config dict
        episode_name: Name of the episode for visualization titles
        data_dir: Path to data directory (for accessing video files)

    Returns:
        Dict with all validation results. ``issues`` fail the episode;
        ``warnings`` are reported alongside it and never fail it.
    """
    results = {
        "timestamp_checks": {
            "fps": {},
            "monotonic": {},
            "frame_intervals": {},
            "sustained_frame_drop": {},
            "coverage": {},
        },
        "sensor_checks": {},  # Per-device: {device_name: {movement, jumps, range}}
        "tracking_checks": {},  # Per-device: {device_name: {pose_movement, consecutive}}
        "device_check": {},  # Device name consistency check
        "resolution_checks": {},  # Per-device: {device_name: {resolution check}}
        "numeric_value_checks": {},  # Per (device, array_key): {exists, no_nan, in_range, jump, movement, shape}
        "motion_alignment_checks": {},  # Per-camera: pairs vs same-side pose
        "passed": True,
        "issues": [],
        "warnings": [],
    }

    device_fps_map = get_device_expected_fps(config)
    device_resolution_map = get_device_expected_resolutions(config)
    default_fps = config.get("recording", {}).get("record_fps", 45)

    ts_config = check_config.get("timestamp", {})
    gripper_config = check_config.get("gripper", {})
    tracking_config = check_config.get("tracking", {})

    devices = [key for key in episode_data.group_keys()]

    section_starts = []

    def _mark(section):
        section_starts.append((section, len(results["issues"])))

    # Read each pose array once and share it across checks to avoid repeated
    # decompression.
    _pose_cache: Dict[str, Optional[np.ndarray]] = {}

    def get_pose(device_name):
        if device_name not in _pose_cache:
            try:
                group = episode_data[device_name]
                _pose_cache[device_name] = (
                    np.array(group["pose"][:]) if "pose" in group else None
                )
            except (KeyError, AttributeError, TypeError):
                _pose_cache[device_name] = None
        return _pose_cache[device_name]

    # ==========================================================================
    # Timestamp Checks
    # ==========================================================================
    device_timestamps = {}

    _mark("timestamp")
    if is_enabled(check_config, "timestamp"):
        for device in devices:
            timestamps = get_device_timestamps(episode_data, device)
            if timestamps is not None and len(timestamps) > 0:
                device_timestamps[device] = timestamps

                # Per-device expected FPS (from source config, else global default)
                expected_fps = device_fps_map.get(device, default_fps)

                # FPS check
                fps_deviation_threshold = ts_config.get("fps_deviation_pct", 10.0)
                fps_result = check_fps_consistency(
                    timestamps, expected_fps, fps_deviation_threshold
                )
                results["timestamp_checks"]["fps"][device] = fps_result
                if not fps_result.get("passed", True):
                    results["issues"].append(
                        f"FPS deviation for {device}: {fps_result['deviation_pct']}%"
                    )

                # Monotonic check
                mono_result = check_monotonic_timestamps(timestamps)
                results["timestamp_checks"]["monotonic"][device] = mono_result
                if not mono_result["monotonic"]:
                    results["issues"].append(
                        f"Non-monotonic timestamps for {device}: {mono_result['num_violations']} violations"
                    )

                # gap_min_frames controls reporting; gap_fail_* thresholds control
                # failure.
                # Optional pose positions quantify interpolation error across a gap.
                # Accept max_missing_frames as a compatibility alias for saved session
                # configs.
                gap_min_frames = ts_config.get(
                    "gap_min_frames", ts_config.get("max_missing_frames", 3.0)
                )
                pose = get_pose(device)
                positions = None
                if pose is not None:
                    if pose.ndim == 3 and pose.shape[1:] == (4, 4):
                        positions = pose[:, :3, 3]
                    else:
                        # A pose in some other layout silently disables the
                        # fidelity rule, which would look like the rule passing.
                        logger.debug(
                            "%s: pose has shape %s, not (N, 4, 4) -- skipping the "
                            "interpolation-error rule for this device",
                            device,
                            pose.shape,
                        )
                interval_result = check_consecutive_frame_intervals(
                    timestamps,
                    expected_fps,
                    gap_min_frames,
                    gap_fail_max_missing_pct=ts_config.get(
                        "gap_fail_max_missing_pct", 1.0
                    ),
                    gap_fail_max_frames=ts_config.get("gap_fail_max_frames", 10.0),
                    positions=positions,
                    gap_fail_max_interp_error_mm=ts_config.get(
                        "gap_fail_max_interp_error_mm", 5.0
                    ),
                )
                results["timestamp_checks"]["frame_intervals"][device] = interval_result
                if not interval_result["passed"]:
                    results["issues"].append(
                        f"Frame interval violations for {device}: {interval_result['num_violations']} frames exceed {gap_min_frames}x expected dt "
                        f"({interval_result['missing_pct']:.2f}% of the episode missing, longest gap {interval_result['longest_gap_s'] * 1000:.0f} ms"
                        + (
                            f", est. interpolation error {interval_result['max_est_interp_error_mm']:.1f} mm"
                            if interval_result.get("interp_error_checked")
                            else ""
                        )
                        + ") "
                        f"at frames {[v['frame_index'] for v in interval_result['violation_frames']]}"
                    )

                # Sustained frame drop check: fail if any window (e.g. 1s) has FPS significantly below target
                frame_drop_window_s = ts_config.get("frame_drop_window_s", 1.0)
                frame_drop_min_ratio = ts_config.get("frame_drop_min_fps_ratio", 0.7)
                sustained_drop_result = check_sustained_frame_drop(
                    timestamps,
                    expected_fps,
                    window_s=frame_drop_window_s,
                    min_fps_ratio=frame_drop_min_ratio,
                )
                results["timestamp_checks"]["sustained_frame_drop"][device] = (
                    sustained_drop_result
                )
                if not sustained_drop_result.get("passed", True):
                    results["issues"].append(
                        f"Sustained frame drop for {device}: {sustained_drop_result['num_violations']} window(s) of {frame_drop_window_s}s with FPS below {frame_drop_min_ratio * 100:.0f}% of expected"
                    )

        # Coverage check
        coverage_threshold = ts_config.get("coverage_threshold", 0.95)
        coverage_result = check_timestamp_coverage(
            device_timestamps, coverage_threshold
        )
        results["timestamp_checks"]["coverage"] = coverage_result
        if not coverage_result.get("passed", True):
            results["issues"].append(
                f"Low timestamp coverage: {coverage_result['overlap_ratio'] * 100:.1f}%"
            )

    # ==========================================================================
    # Numeric Value Checks (generic exists / no_nan / in_range / jump / movement / shape)
    # ==========================================================================
    _mark("numeric_values")
    if is_enabled(check_config, "numeric_values"):
        # Preserve configured check order when removing enabled. Pass recording
        # config so rules can resolve from_recording_config references.
        nv_results, nv_issues = run_numeric_value_checks(
            episode_data, check_config["numeric_values"], recording_config=config
        )
        results["numeric_value_checks"] = nv_results
        results["issues"].extend(nv_issues)

    # ==========================================================================
    # Device Name Consistency Check
    # ==========================================================================
    _mark("device_name_consistency")
    if is_enabled(check_config, "device_name_consistency"):
        # Layer explicit device-name rules onto a config copy for the expected-name
        # extractor; do not mutate the caller's recording config.
        config_with_override = _config_with_device_name_override(config, check_config)
        device_check_result = check_device_name_consistency(
            config_with_override, list(episode_data.group_keys())
        )
        results["device_check"] = device_check_result
        if not device_check_result.get("passed", True):
            for issue in device_check_result.get("issues", []):
                results["issues"].append(f"Device consistency: {issue}")
    else:
        results["device_check"] = {"passed": True, "skipped": True}

    # ==========================================================================
    # Video Resolution Check (expected resolution from dataset config per device)
    # ==========================================================================
    _mark("video_resolution")
    if (
        is_enabled(check_config, "video_resolution")
        and data_dir is not None
        and device_resolution_map
    ):
        for device_name, expected_resolution in device_resolution_map.items():
            if device_name not in episode_data:
                continue
            video_path = os.path.join(data_dir, episode_name, f"{device_name}.mp4")
            resolution_result = check_video_resolution(
                video_path, expected_resolution=expected_resolution
            )
            results["resolution_checks"][device_name] = resolution_result
            if not resolution_result.get("passed", True):
                actual = resolution_result.get("actual_resolution")
                expected = resolution_result.get("expected_resolution")
                if actual:
                    results["issues"].append(
                        f"{device_name}: resolution {actual} != expected {expected}"
                    )
                else:
                    results["issues"].append(
                        f"{device_name}: {resolution_result.get('error', 'Unknown error')}"
                    )

    # ==========================================================================
    # Gripper Sensor Checks (handles multiple grippers)
    # ==========================================================================
    gripper_devices = [d for d in devices if "gripper" in d.lower()]

    _mark("gripper")
    if is_enabled(check_config, "gripper"):
        for gripper_device in gripper_devices:
            if gripper_device not in episode_data:
                continue

            gripper_data = episode_data[gripper_device]
            results["sensor_checks"][gripper_device] = {}

            distance = None
            if "distance" in gripper_data:
                distance = np.array(gripper_data["distance"][:])

            if distance is None:
                continue

            jump_threshold = gripper_config.get("jump_threshold_mm_per_s", 2000.0)
            if check_enabled(gripper_config, "jumps"):
                jump_result = check_gripper_jumps(
                    distance,
                    get_device_timestamps(episode_data, gripper_device),
                    jump_threshold_mm_per_s=jump_threshold,
                )

                results["sensor_checks"][gripper_device]["jumps"] = jump_result
                if jump_result["has_jumps"]:
                    results["issues"].append(
                        f"{gripper_device}: distance jumps detected ({jump_result['num_jumps']} jumps > {jump_threshold:.0f}mm/s)"
                    )

            # The shipped baseline disables per-gripper movement checks because a valid
            # unimanual task on a bimanual rig can leave one gripper still.
            if check_enabled(gripper_config, "movement"):
                movement_threshold = gripper_config.get("movement_threshold_mm", 10.0)
                movement_result = check_gripper_movement(distance, movement_threshold)
                results["sensor_checks"][gripper_device]["movement"] = movement_result
                if not movement_result.get("moved", True):
                    results["issues"].append(
                        f"{gripper_device}: did not move (range: {movement_result['range_mm']}mm)"
                    )

            if check_enabled(gripper_config, "range"):
                range_min = gripper_config.get("range_min_mm", 0.0)
                range_max = gripper_config.get("range_max_mm", 90.0)
                range_result = check_gripper_range(distance, range_min, range_max)
                results["sensor_checks"][gripper_device]["range"] = range_result
                if not range_result["in_range"]:
                    results["issues"].append(
                        f"{gripper_device}: out of range ({range_result['num_out_of_range']} frames)"
                    )

    # ==========================================================================
    # Tracking Checks (any pose source -- iPhone camera or Vive puck)
    # ==========================================================================
    _mark("tracking")
    tracking_devices = tracking_devices_in(devices)

    if not is_enabled(check_config, "tracking"):
        tracking_devices_for_loop = []
    else:
        tracking_devices_for_loop = tracking_devices

    min_movement = tracking_config.get("consecutive_min_movement_m", 0.0005)
    required_ratio = tracking_config.get("consecutive_required_ratio", 0.80)
    pose_jump_threshold = tracking_config.get("pose_jump_threshold_m_per_s", 4.5)
    # For multi-camera scenario, use multi_tracking_consecutive_required_ratio if specified
    multi_tracking_ratio = tracking_config.get(
        "multi_tracking_consecutive_required_ratio"
    )

    # One video-confirmation budget per episode, shared by every device.
    confirm_deadline = time.monotonic() + tracking_config.get(
        "excursion_confirm_budget_s", 3.0
    )

    for tracking_device in tracking_devices_for_loop:
        if tracking_device not in episode_data:
            continue

        tracking_data = episode_data[tracking_device]
        results["tracking_checks"][tracking_device] = {}

        if "pose" in tracking_data:
            poses = get_pose(tracking_device)

            # Pose movement check (N×N downsampled)
            pose_threshold = tracking_config.get("pose_movement_threshold_m", 0.01)
            pose_downsample = tracking_config.get("pose_downsample", 10)
            pose_result = check_pose_changes(poses, pose_threshold, pose_downsample)
            results["tracking_checks"][tracking_device]["pose_movement"] = pose_result
            if not pose_result.get("has_movement", True):
                results["issues"].append(
                    f"{tracking_device}: static (max movement {pose_result['max_distance'] * 1000:.1f}mm)"
                )

            pose_jump_result = check_tracking_pose_jumps(
                poses,
                get_device_timestamps(episode_data, tracking_device),
                jump_threshold_m_per_s=pose_jump_threshold,
            )
            results["tracking_checks"][tracking_device]["pose_jumps"] = pose_jump_result
            if pose_jump_result["has_jumps"]:
                results["issues"].append(
                    f"{tracking_device}: pose jumps detected ({pose_jump_result['num_jumps']} jumps > {pose_jump_threshold:.1f}m/s)"
                )

            if check_enabled(tracking_config, "excursions"):
                results["tracking_checks"][tracking_device]["excursions"] = (
                    _detect_excursions(
                        episode_data, tracking_device, poses, tracking_config
                    )
                )

            # Consecutive frame movement check
            consecutive_result = check_consecutive_movement(
                poses, min_movement, required_ratio
            )
            results["tracking_checks"][tracking_device]["consecutive_movement"] = (
                consecutive_result
            )
            # Note: Issues for consecutive movement are checked after all devices are processed

    # Confirm arm excursions before head excursions: arm poses drive actions
    # and have priority in the shared video-decode budget.
    excursion_devices = [
        device
        for device in tracking_devices_for_loop
        if "excursions" in results["tracking_checks"].get(device, {})
    ]
    for tracking_device in sorted(
        excursion_devices, key=lambda d: tracking_device_to_arm(d) is None
    ):
        excursions = results["tracking_checks"][tracking_device]["excursions"]
        if (
            excursions.get("num_candidates")
            and data_dir is not None
            and check_enabled(tracking_config, "excursion_confirm")
        ):
            _confirm_excursions(
                episode_data,
                tracking_device,
                excursions,
                tracking_config,
                data_dir,
                episode_name,
                confirm_deadline,
            )
        # Excursions are warn-only. Surface video-confirmed cases to the operator
        # and retain other detections in the report.
        num_confirmed = excursions.get("num_confirmed", 0)
        if num_confirmed:
            min_amplitude = tracking_config.get("excursion_min_amplitude_mm", 20.0)
            results["warnings"].append(
                f"{tracking_device}: confirmed tracker excursions ({num_confirmed} >= {min_amplitude:.0f}mm)"
            )

    # Every tracking camera must meet multi_tracking_consecutive_required_ratio
    # when configured; evaluate this after collecting all device results.
    if multi_tracking_ratio is not None and len(tracking_devices) > 0:
        effective_ratio = multi_tracking_ratio
    else:
        effective_ratio = required_ratio

    for tracking_device in tracking_devices:
        if tracking_device not in results["tracking_checks"]:
            continue
        consecutive_result = results["tracking_checks"][tracking_device].get(
            "consecutive_movement", {}
        )
        movement_ratio = consecutive_result.get("movement_ratio", 0.0)
        if movement_ratio < effective_ratio:
            results["issues"].append(
                f"{tracking_device}: low consecutive movement "
                f"({movement_ratio * 100:.1f}%, need {effective_ratio * 100:.1f}% > {min_movement * 1000:.1f}mm)"
            )

    # ==========================================================================
    # Motion Alignment Check (visual ego-motion vs ARKit pose)
    # ==========================================================================
    _mark("motion_alignment")
    if is_enabled(check_config, "motion_alignment") and data_dir is not None:
        # Lazy import: pulls in cv2 and video decoding only when enabled
        from .motion_alignment import check_motion_alignment

        motion_result = check_motion_alignment(
            episode_data,
            data_dir,
            episode_name,
            check_config["motion_alignment"],
        )
        results["motion_alignment_checks"] = {
            k: motion_result[k]
            for k in ("mode", "cameras", "tracking", "passed", "notes")
        }
        # A camera/pose mismatch fails the episode. Time offsets only warn because
        # the limits are hardware-dependent; estimator disagreements stay report-only.
        for issue in motion_result.get("issues", []):
            results["issues"].append(f"Motion: {issue}")
        for warning in motion_result.get("warnings", []):
            results["warnings"].append(f"Motion: {warning}")

    _relax_issues(results, section_starts, _unenforced_sections(config, check_config))

    results["passed"] = len(results["issues"]) == 0

    return results


def check_dataset(
    data_dir: str,
    episodes: Optional[List[int]] = None,
    check_config_path: Optional[str] = None,
    pending: bool = False,
) -> Dict[str, Any]:
    """Run validation checks on all episodes in a dataset.

    Args:
        data_dir: Path to the data directory
        episodes: Optional list of episode indices to process
        check_config_path: Path to check config file. If None, uses default path.
        pending: If True, check pending episodes (episode_N.pending folders) instead of
                 finalized episodes. Used for validation before finalizing a recording.

    Returns:
        Dict with full check report
    """
    from .constants import PENDING_SUFFIX

    config_path = os.path.join(data_dir, "config.yaml")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    check_config = load_check_config(check_config_path)

    data_store = zarr.open(data_dir, mode="r")

    all_episode_keys = get_episode_keys(data_dir, pending=pending)

    if episodes:
        requested_keys = set()
        for idx in episodes:
            if pending:
                key = f"episode_{idx}{PENDING_SUFFIX}"
            else:
                key = f"episode_{idx}"
            if key in all_episode_keys:
                requested_keys.add(key)
        episode_keys = [key for key in all_episode_keys if key in requested_keys]
    else:
        episode_keys = all_episode_keys

    report = {
        "data_dir": data_dir,
        "config": {
            "record_fps": config.get("recording", {}).get("record_fps", 45),
            "stream_fps": config.get("recording", {}).get("stream_fps", 60),
        },
        "check_config": check_config,  # Save the full check config used for validation
        "episodes": {},
        "dataset_checks": {},  # Dataset-level checks (e.g., gripper serial consistency)
        "summary": {
            "total_episodes": len(episode_keys),
            "passed_episodes": 0,
            "failed_episodes": 0,
            "failed_episode_indices": [],  # Episode indices that failed (for quick view)
            "issues_by_type": {},
        },
    }

    logger.info("Running data checks...")

    check_results = []

    for episode_key in tqdm(episode_keys, desc="Checking episodes"):
        if episode_key not in data_store:
            check_results.append((episode_key, None))
            continue

        episode_data = data_store[episode_key]
        try:
            episode_result = check_episode(
                episode_data,
                config,
                check_config,
                episode_name=episode_key,
                data_dir=data_dir,
            )
            check_results.append((episode_key, episode_result))
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            logger.exception("Episode %s check failed: %s", episode_key, msg)
            episode_result = {
                "passed": False,
                "issues": [f"Check failed with error: {msg}"],
                "warnings": [],
                "timestamp_checks": {
                    "fps": {},
                    "monotonic": {},
                    "frame_intervals": {},
                    "sustained_frame_drop": {},
                    "coverage": {},
                },
                "sensor_checks": {},
                "tracking_checks": {},
                "device_check": {},
                "resolution_checks": {},
                "_error_traceback": traceback.format_exc(),
            }
            check_results.append((episode_key, episode_result))

    for episode_key, episode_result in check_results:
        if episode_result is None:
            raise ValueError(f"Episode {episode_key} not found in data store")

        report["episodes"][episode_key] = episode_result

        episode_idx = extract_episode_index(episode_key)

        passed = episode_result.get("passed", False)
        if passed:
            report["summary"]["passed_episodes"] += 1
        else:
            report["summary"]["failed_episodes"] += 1
            report["summary"]["failed_episode_indices"].append(episode_idx)

        for issue in episode_result["issues"]:
            if ": " in issue:
                device, issue_detail = issue.split(": ", 1)
                if "pose jumps" in issue_detail.lower():
                    issue_type = f"{device}: pose jumps"
                elif "jumps" in issue_detail.lower():
                    issue_type = f"{device}: jumps"
                elif "did not move" in issue_detail.lower():
                    issue_type = f"{device}: no movement"
                elif "no movement" in issue_detail.lower():
                    # `numeric_values` movement check (distinct wording from
                    # check_gripper_movement's "did not move" above).
                    issue_type = f"{device}: no movement"
                elif "out of range" in issue_detail.lower():
                    issue_type = f"{device}: out of range"
                elif "static" in issue_detail.lower():
                    issue_type = f"{device}: static"
                elif "consecutive movement" in issue_detail.lower():
                    issue_type = f"{device}: low consecutive movement"
                elif "nan values" in issue_detail.lower():
                    issue_type = f"{device}: NaN values"
                elif "shape mismatch" in issue_detail.lower():
                    issue_type = f"{device}: shape mismatch"
                elif "no chunks written" in issue_detail.lower():
                    issue_type = f"{device}: no chunks written"
                elif "start frame out of tolerance" in issue_detail.lower():
                    issue_type = f"{device}: start frame out of tolerance"
                elif "start_within_tolerance" in issue_detail.lower():
                    issue_type = f"{device}: start_within_tolerance config error"
                elif issue_detail.strip().lower() == "missing":
                    issue_type = f"{device}: missing"
                else:
                    issue_type = device
            elif "FPS deviation" in issue:
                issue_type = "FPS deviation"
            elif "Frame interval violations" in issue:
                issue_type = "frame interval violations"
            elif "Sustained frame drop" in issue:
                issue_type = "sustained frame drop"
            elif "timestamp" in issue.lower():
                issue_type = "timestamp issues"
            elif "coverage" in issue.lower():
                issue_type = "low coverage"
            elif "device consistency" in issue.lower():
                issue_type = "device name mismatch"
            else:
                issue_type = issue.split(" ")[0]

            if issue_type not in report["summary"]["issues_by_type"]:
                report["summary"]["issues_by_type"][issue_type] = []
            report["summary"]["issues_by_type"][issue_type].append(episode_idx)

    # Dataset-level checks: validate device names for the session.
    config_with_override = _config_with_device_name_override(config, check_config)
    config_device_check = check_config_device_names(config_with_override)
    report["dataset_checks"]["config_device_names"] = config_device_check

    if not config_device_check["passed"]:
        if "config_device_name_mismatch" not in report["summary"]["issues_by_type"]:
            report["summary"]["issues_by_type"]["config_device_name_mismatch"] = []
        # This is a dataset-level issue, mark all episodes as affected
        for key in episode_keys:
            idx = extract_episode_index(key)
            if (
                idx
                not in report["summary"]["issues_by_type"][
                    "config_device_name_mismatch"
                ]
            ):
                report["summary"]["issues_by_type"][
                    "config_device_name_mismatch"
                ].append(idx)

    # Summarize per-episode device-name checks without adding duplicate issues.
    if is_enabled(check_config, "device_name_consistency"):
        expected = expected_device_names(config_with_override)
        device_consistency_summary = {
            "passed": True,
            "expected_devices": sorted(expected["all_devices"]),
            "video_sources": expected["video_sources"],
            "numeric_sources": expected["numeric_sources"],
            "episodes_with_missing_devices": {},
            "episodes_with_extra_devices": {},
        }

        for episode_key, episode_result in report["episodes"].items():
            device_check_result = episode_result.get("device_check", {})

            if not device_check_result.get("passed", True):
                device_consistency_summary["passed"] = False

                missing_devices = device_check_result.get("missing_devices", [])
                if missing_devices:
                    device_consistency_summary["episodes_with_missing_devices"][
                        episode_key
                    ] = missing_devices

                extra_devices = device_check_result.get("extra_devices", [])
                if extra_devices:
                    device_consistency_summary["episodes_with_extra_devices"][
                        episode_key
                    ] = extra_devices

        report["dataset_checks"]["device_name_consistency"] = device_consistency_summary
    else:
        report["dataset_checks"]["device_name_consistency"] = {
            "passed": True,
            "skipped": True,
        }

    gripper_serial_check = check_gripper_serial_consistency(data_store, episode_keys)
    report["dataset_checks"]["gripper_serial_consistency"] = gripper_serial_check

    if not gripper_serial_check["passed"]:
        if "gripper_serial_inconsistency" not in report["summary"]["issues_by_type"]:
            report["summary"]["issues_by_type"]["gripper_serial_inconsistency"] = []
        for idx in gripper_serial_check["inconsistent_episodes"]:
            if (
                idx
                not in report["summary"]["issues_by_type"][
                    "gripper_serial_inconsistency"
                ]
            ):
                report["summary"]["issues_by_type"][
                    "gripper_serial_inconsistency"
                ].append(idx)

    logger.info("Data checks complete")

    return report

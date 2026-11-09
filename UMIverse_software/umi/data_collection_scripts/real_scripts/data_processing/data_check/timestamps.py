# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Timestamp validation: FPS, monotonicity, coverage, frame intervals, sustained drops."""

from typing import Any, Dict, Optional

import numpy as np

__all__ = [
    "check_fps_consistency",
    "check_monotonic_timestamps",
    "check_timestamp_coverage",
    "check_consecutive_frame_intervals",
    "check_sustained_frame_drop",
]


def check_fps_consistency(
    timestamps: np.ndarray,
    expected_fps: float,
    max_deviation_pct: float = 10.0,
) -> Dict[str, Any]:
    """Check if average FPS matches expected FPS from config.

    Args:
        timestamps: Array of timestamps
        expected_fps: Expected FPS from config
        max_deviation_pct: Maximum allowed deviation percentage

    Returns:
        Dict with avg_fps, expected_fps, deviation_pct, and passed status
    """
    if len(timestamps) < 2:
        return {
            "avg_fps": 0.0,
            "expected_fps": expected_fps,
            "deviation_pct": 100.0,
            "passed": False,
            "error": "Not enough timestamps",
        }

    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        return {
            "avg_fps": 0.0,
            "expected_fps": expected_fps,
            "deviation_pct": 100.0,
            "passed": False,
            "error": "Invalid duration",
        }

    avg_fps = (len(timestamps) - 1) / duration
    deviation_pct = abs(avg_fps - expected_fps) / expected_fps * 100

    return {
        "avg_fps": round(avg_fps, 2),
        "expected_fps": expected_fps,
        "deviation_pct": round(deviation_pct, 2),
        "passed": deviation_pct < max_deviation_pct,
    }


def check_monotonic_timestamps(timestamps: np.ndarray) -> Dict[str, Any]:
    """Check if timestamps are monotonically increasing.

    Args:
        timestamps: Array of timestamps

    Returns:
        Dict with monotonic status and list of non-monotonic indices
    """
    if len(timestamps) < 2:
        return {"monotonic": True, "non_monotonic_indices": []}

    diffs = np.diff(timestamps)
    non_monotonic_mask = diffs <= 0
    non_monotonic_indices = np.where(non_monotonic_mask)[0].tolist()

    return {
        "monotonic": len(non_monotonic_indices) == 0,
        "non_monotonic_indices": non_monotonic_indices,
        "num_violations": len(non_monotonic_indices),
    }


def check_timestamp_coverage(
    device_timestamps: Dict[str, np.ndarray], threshold: float = 0.95
) -> Dict[str, Any]:
    """Check if timestamp ranges have at least threshold% same coverage.

    Args:
        device_timestamps: Dict mapping device names to timestamp arrays
        threshold: Minimum required overlap ratio (default 95%)

    Returns:
        Dict with overlap_ratio, ranges_per_device, and passed status
    """
    if len(device_timestamps) < 2:
        return {
            "overlap_ratio": 1.0,
            "ranges": {},
            "passed": True,
        }

    ranges = {}
    for device, timestamps in device_timestamps.items():
        if len(timestamps) > 0:
            ranges[device] = {
                "min": float(timestamps.min()),
                "max": float(timestamps.max()),
                "duration": float(timestamps.max() - timestamps.min()),
            }

    if len(ranges) < 2:
        return {
            "overlap_ratio": 1.0,
            "ranges": ranges,
            "passed": True,
        }

    all_mins = [r["min"] for r in ranges.values()]
    all_maxs = [r["max"] for r in ranges.values()]

    intersection_start = max(all_mins)
    intersection_end = min(all_maxs)
    intersection_duration = max(0, intersection_end - intersection_start)

    min_start = min(r["min"] for r in ranges.values())
    max_end = max(r["max"] for r in ranges.values())
    max_duration = max_end - min_start
    if max_duration <= 0:
        overlap_ratio = 0.0
    else:
        overlap_ratio = intersection_duration / max_duration

    return {
        "overlap_ratio": round(overlap_ratio, 4),
        "intersection_start": intersection_start,
        "intersection_end": intersection_end,
        "intersection_duration": round(intersection_duration, 4),
        "ranges": ranges,
        "passed": overlap_ratio >= threshold,
    }


def estimate_interpolation_error_mm(
    timestamps: np.ndarray,
    positions: np.ndarray,
    gap_index: int,
) -> float:
    """Estimated error, in mm, of interpolating the pose across this gap.

    Linear interpolation is exact at constant velocity and wrong in proportion
    to how much the velocity changes, so use the midpoint bound
    ``|v_after - v_before| * gap / 8``, taken from the samples bracketing the
    hole -- the only ones left once the middle is gone. Validated against
    held-out real Vive pose at r = 0.85.

    0.0 at the very start or end of a stream, where there is no bracketing pair.
    """
    i = int(gap_index)
    if i < 1 or i + 2 >= len(timestamps):
        return 0.0
    dt_before = timestamps[i] - timestamps[i - 1]
    dt_after = timestamps[i + 2] - timestamps[i + 1]
    if dt_before <= 0 or dt_after <= 0:
        return 0.0
    v_before = (positions[i] - positions[i - 1]) / dt_before
    v_after = (positions[i + 2] - positions[i + 1]) / dt_after
    gap = timestamps[i + 1] - timestamps[i]
    return float(np.linalg.norm(v_after - v_before) * gap / 8.0 * 1000.0)


def check_consecutive_frame_intervals(
    timestamps: np.ndarray,
    expected_fps: float,
    gap_min_frames: float = 3.0,
    gap_fail_max_missing_pct: float = 1.0,
    gap_fail_max_frames: float = 10.0,
    positions: Optional[np.ndarray] = None,
    gap_fail_max_interp_error_mm: float = 5.0,
) -> Dict[str, Any]:
    """Find the gaps in a timestamp stream and decide whether they matter.

    An interval longer than ``gap_min_frames`` frame periods is a gap; anything
    shorter is normal jitter. Every gap is reported with its duration whatever
    the verdict, so the three ``gap_fail_*`` limits never hide one; they only
    decide whether the episode passes.

    It fails if any of them is exceeded, because three different things can be
    wrong with a gap:

      * ``gap_fail_max_interp_error_mm``: interpolation across it cannot put
        the motion back. The rule that matters; duration is a poor proxy for it,
        since the same 233 ms hole reconstructs to 0.7 mm while coasting and
        47 mm through a direction change. Needs ``positions``.
      * ``gap_fail_max_missing_pct``: too much of the episode is synthetic,
        however recoverable each individual gap was.
      * ``gap_fail_max_frames``: one gap is simply too long to accept.

    Args:
        timestamps: Array of timestamps.
        expected_fps: Expected FPS from config; dt = 1/fps.
        gap_min_frames: An interval this many frame periods long counts as a
            gap. Sets what is reported, not what fails.
        gap_fail_max_missing_pct: Fail if total gap time (the excess beyond one
            frame period per gap) exceeds this percentage of the episode.
        gap_fail_max_frames: Fail if any single gap is longer than this many
            frame periods. A gap of exactly this length is allowed, matching
            `gap_fail_max_missing_pct`, which also fails only on exceeding.
        positions: Optional (N, 3) positions matching `timestamps`. Devices
            without them (cameras, grippers) skip the interpolation-error rule.
        gap_fail_max_interp_error_mm: Fail if the estimated interpolation error
            across any gap exceeds this. Ignored when `positions` is None.

    Returns:
        Dict with passed status, violation details, and statistics
    """
    if len(timestamps) < 2:
        # Use the normal result schema even with a single sample so report formatting
        # does not need a special case.
        return {
            "passed": True,
            "expected_dt": 0.0,
            "max_allowed_interval": 0.0,
            "num_violations": 0,
            "violation_frames": [],
            "missing_s": 0.0,
            "missing_pct": 0.0,
            "longest_gap_s": 0.0,
            "gap_min_frames": gap_min_frames,
            "gap_fail_max_missing_pct": gap_fail_max_missing_pct,
            "gap_fail_max_frames": gap_fail_max_frames,
            "gap_fail_max_gap_s": 0.0,
            "max_est_interp_error_mm": 0.0,
            "gap_fail_max_interp_error_mm": gap_fail_max_interp_error_mm,
            "interp_error_checked": False,
            "error": "Not enough timestamps",
        }

    expected_dt = 1.0 / expected_fps
    max_allowed_interval = gap_min_frames * expected_dt

    diffs = np.diff(timestamps)
    violation_mask = diffs > max_allowed_interval
    violation_indices = np.where(violation_mask)[0]

    have_positions = (
        positions is not None
        and len(positions) == len(timestamps)
        and np.ndim(positions) == 2
        and np.shape(positions)[1] == 3
    )

    violation_frames = []
    worst_error_mm = 0.0
    for idx in violation_indices:
        entry = {
            "frame_index": int(idx + 1),
            "time_diff": round(float(diffs[idx]), 6),
            "expected_dt": round(expected_dt, 6),
            "multiplier": round(float(diffs[idx] / expected_dt), 2),
        }
        if have_positions:
            err = estimate_interpolation_error_mm(timestamps, positions, idx)
            entry["est_interp_error_mm"] = round(err, 3)
            worst_error_mm = max(worst_error_mm, err)
        violation_frames.append(entry)

    # Excess over a normal frame period, not the whole gap.
    gap_lengths = diffs[violation_mask]
    missing_s = float(np.sum(gap_lengths - expected_dt)) if len(gap_lengths) else 0.0
    duration_s = float(timestamps[-1] - timestamps[0])
    missing_pct = (missing_s / duration_s * 100.0) if duration_s > 0 else 0.0
    longest_gap_s = float(gap_lengths.max()) if len(gap_lengths) else 0.0

    gap_fail_max_gap_s = gap_fail_max_frames * expected_dt
    passed = (
        missing_pct <= gap_fail_max_missing_pct and longest_gap_s <= gap_fail_max_gap_s
    )
    if have_positions:
        passed = passed and worst_error_mm <= gap_fail_max_interp_error_mm

    return {
        "passed": passed,
        "expected_dt": round(expected_dt, 6),
        "max_allowed_interval": round(max_allowed_interval, 6),
        "gap_min_frames": gap_min_frames,
        "num_violations": len(violation_frames),
        "violation_frames": violation_frames,
        # Include measured gap sizes and active thresholds to explain the verdict.
        "missing_s": round(missing_s, 4),
        "missing_pct": round(missing_pct, 4),
        "longest_gap_s": round(longest_gap_s, 4),
        "gap_fail_max_missing_pct": gap_fail_max_missing_pct,
        "gap_fail_max_frames": gap_fail_max_frames,
        "gap_fail_max_gap_s": round(gap_fail_max_gap_s, 6),
        # 0.0 also means "not checked" -- read with interp_error_checked.
        "max_est_interp_error_mm": round(worst_error_mm, 3),
        "gap_fail_max_interp_error_mm": gap_fail_max_interp_error_mm,
        "interp_error_checked": bool(have_positions),
    }


def check_sustained_frame_drop(
    timestamps: np.ndarray,
    expected_fps: float,
    window_s: float = 1.0,
    min_fps_ratio: float = 0.7,
) -> Dict[str, Any]:
    """Check that no time window of length window_s has FPS significantly below expected.

    If any 1-second (or configurable) window has effective FPS below
    expected_fps * min_fps_ratio, the source is considered delayed/dropping
    frames for that period and the check fails.

    Args:
        timestamps: Array of timestamps (seconds).
        expected_fps: Expected FPS from config.
        window_s: Window duration in seconds (e.g. 1.0 for "sustained for 1s").
        min_fps_ratio: Minimum allowed ratio of actual to expected FPS in any window (e.g. 0.7 = 70%).

    Returns:
        Dict with passed, window_s, min_fps_ratio, violations (list of low-FPS windows).
    """
    if len(timestamps) < 2:
        return {
            "passed": True,
            "window_s": window_s,
            "min_fps_ratio": min_fps_ratio,
            "num_violations": 0,
            "violations": [],
            "error": "Not enough timestamps",
        }

    t_min = float(timestamps[0])
    t_max = float(timestamps[-1])
    if t_max - t_min < window_s:
        return {
            "passed": True,
            "window_s": window_s,
            "min_fps_ratio": min_fps_ratio,
            "num_violations": 0,
            "violations": [],
            "reason": "Episode shorter than window",
        }

    min_frames_in_window = expected_fps * window_s * min_fps_ratio
    violations = []

    # Slide the window by time: sample start times so the windows cover the episode
    step_s = max(window_s / 4.0, 0.05)  # overlap so no bad window is missed
    t_start = t_min
    while t_start + window_s <= t_max:
        t_end = t_start + window_s
        # Count frames in [t_start, t_end]
        left = np.searchsorted(timestamps, t_start, side="left")
        right = np.searchsorted(timestamps, t_end, side="right")
        count = right - left
        if count < min_frames_in_window:
            violations.append(
                {
                    "window_start_s": round(t_start, 3),
                    "window_end_s": round(t_end, 3),
                    "frame_count": int(count),
                    "min_expected_frames": round(min_frames_in_window, 1),
                    "effective_fps": round(count / window_s, 2),
                }
            )
        t_start += step_s

    return {
        "passed": len(violations) == 0,
        "window_s": window_s,
        "min_fps_ratio": min_fps_ratio,
        "expected_fps": expected_fps,
        "num_violations": len(violations),
        "violations": violations[:20],  # cap for report size
    }

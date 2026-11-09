# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""iPhone/tracking-camera pose checks: movement, consecutive movement, jumps,
excursions."""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .common import _consecutive_position_distances, _max_pairwise_position_distance

__all__ = [
    "check_pose_changes",
    "check_consecutive_movement",
    "check_tracking_pose_jumps",
    "check_tracking_excursions",
    "excursion_state_counts",
]

EXCURSION_DETECTOR_VERSION = 1
# Degree of the polynomial bridge fitted across an excursion; it needs
# EXCURSION_BRIDGE_DEG + 2 context samples on each side.
EXCURSION_BRIDGE_DEG = 4
# Count key per event state.
EXCURSION_STATE_COUNTS = {
    "candidate": "num_candidates",
    "confirmed": "num_confirmed",
    "unverified": "num_unverified",
    "rejected": "num_rejected",
}


def excursion_state_counts(events) -> Dict[str, int]:
    """A count per event state, recomputed after video confirmation moves
    events out of "candidate" so the persisted counts match the events."""
    return {
        key: sum(e.get("state") == state for e in events)
        for state, key in EXCURSION_STATE_COUNTS.items()
    }


def check_pose_changes(
    poses: np.ndarray, threshold: float = 0.01, downsample: int = 10
) -> Dict[str, Any]:
    """Check if poses show meaningful movement using downsampled N×N comparison.

    Args:
        poses: Array of 4x4 homogeneous transformation matrices (N, 4, 4)
        threshold: Movement threshold in meters (default 0.01m = 1cm)
        downsample: Downsample factor for efficiency (default 10)

    Returns:
        Dict with movement status and max distance found
    """
    if len(poses) == 0:
        return {
            "has_movement": False,
            "max_distance": 0.0,
            "error": "No pose data",
        }

    # Downsample for efficiency
    sampled_poses = poses[::downsample]
    n = len(sampled_poses)

    if n < 2:
        return {
            "has_movement": False,
            "max_distance": 0.0,
            "num_samples": n,
        }

    # Vectorized: only position distance is needed; avoid O(n²) pose_distance calls
    positions = sampled_poses[:, :3, 3].astype(np.float64)
    max_dist = _max_pairwise_position_distance(positions)

    return {
        "has_movement": max_dist > threshold,
        "max_distance": round(float(max_dist), 4),
        "threshold": threshold,
        "num_samples": n,
        "downsample_factor": downsample,
    }


def check_consecutive_movement(
    poses: np.ndarray,
    min_movement: float = 0.0005,
    required_ratio: float = 0.90,
) -> Dict[str, Any]:
    """Check if 90% of consecutive frames show movement > 0.5mm.

    Args:
        poses: Array of 4x4 homogeneous transformation matrices (N, 4, 4)
        min_movement: Minimum movement threshold in meters (default 0.0005m = 0.5mm)
        required_ratio: Required ratio of frames with movement (default 0.90 = 90%)

    Returns:
        Dict with consecutive movement check results
    """
    if len(poses) < 2:
        return {
            "passed": False,
            "movement_ratio": 0.0,
            "required_ratio": required_ratio,
            "min_movement_m": min_movement,
            "error": "Not enough poses",
        }

    # Vectorized consecutive position distances (only position is used for this check)
    consecutive_distances = _consecutive_position_distances(poses)

    moving_frames = np.sum(consecutive_distances > min_movement)
    total_frames = len(consecutive_distances)
    movement_ratio = moving_frames / total_frames

    return {
        "passed": movement_ratio >= required_ratio,
        "movement_ratio": round(float(movement_ratio), 4),
        "required_ratio": required_ratio,
        "min_movement_m": min_movement,
        "moving_frames": int(moving_frames),
        "total_frames": total_frames,
        "avg_consecutive_dist_mm": round(
            float(np.mean(consecutive_distances) * 1000), 2
        ),
        "median_consecutive_dist_mm": round(
            float(np.median(consecutive_distances) * 1000), 2
        ),
    }


def check_tracking_pose_jumps(
    poses: np.ndarray,
    timestamps: np.ndarray,
    jump_threshold_m_per_s: float = 4.5,
) -> Dict[str, Any]:
    """Detect jump readings in tracking camera poses, scored as speed.

    Scoring against the real gap keeps a dropped frame from reading as a jump.

    Args:
        poses: Array of 4x4 homogeneous transformation matrices (N, 4, 4)
        timestamps: Per-sample timestamps in seconds
        jump_threshold_m_per_s: Speed ceiling in m/s (default 4.5)

    Returns:
        Dict with has_jumps, jump_frames, num_jumps, jump_threshold_m_per_s
    """
    result = {
        "has_jumps": False,
        "jump_frames": [],
        "num_jumps": 0,
        "jump_threshold_m_per_s": jump_threshold_m_per_s,
    }

    if len(poses) < 2:
        return result

    # Vectorized consecutive position distances (only position is used for jump detection)
    consecutive_distances = _consecutive_position_distances(poses)
    dt = np.diff(np.asarray(timestamps, dtype=float))
    # Non-positive gaps mean broken timestamps, which the timestamp checks
    # report separately; a speed across them is meaningless.
    speeds = np.divide(
        consecutive_distances,
        dt,
        out=np.zeros_like(consecutive_distances, dtype=float),
        where=dt > 0,
    )

    for idx in np.where(speeds > jump_threshold_m_per_s)[0]:
        result["jump_frames"].append(
            {
                "frame_index": int(idx + 1),  # +1 to point to the frame that jumped
                "magnitude_m": round(float(consecutive_distances[idx]), 4),
                "magnitude_mm": round(float(consecutive_distances[idx] * 1000), 2),
                "magnitude_m_per_s": round(float(speeds[idx]), 4),
            }
        )

    result["has_jumps"] = len(result["jump_frames"]) > 0
    result["num_jumps"] = len(result["jump_frames"])

    return result


def _excursion_bridge(
    t: np.ndarray, X: np.ndarray, i0: int, i1: int, context_s: float
) -> Optional[Tuple[float, float]]:
    """Polynomial in time through the `context_s` before sample i0 and after
    sample i1 (the excursion itself excluded), with a post-excursion offset.

    Returns (amplitude_mm, return_mm), or None without enough context.
    amplitude: how far the excursion samples sit off the bridge fitted without
    the offset. return: |offset|, ~0 when the trajectory after the excursion
    continues the one before it, ~S after a lasting step of S.
    """
    ar = np.arange(len(t))
    pre = np.flatnonzero((t >= t[i0] - context_s) & (ar < i0))
    post = np.flatnonzero((t <= t[i1] + context_s) & (ar > i1))
    if min(len(pre), len(post)) < EXCURSION_BRIDGE_DEG + 2:
        return None
    idx = np.concatenate([pre, post])
    tc = 0.5 * (t[i0] + t[i1])
    tt = t[idx] - tc
    powers = [tt**k for k in range(EXCURSION_BRIDGE_DEG, -1, -1)]
    with_offset = np.column_stack(powers + [(idx > i1).astype(float)])
    beta, *_ = np.linalg.lstsq(with_offset, X[idx], rcond=None)
    return_mm = float(np.linalg.norm(beta[-1]) * 1e3)
    beta, *_ = np.linalg.lstsq(np.column_stack(powers), X[idx], rcond=None)
    tr = t[i0 : i1 + 1] - tc
    fit = np.column_stack([tr**k for k in range(EXCURSION_BRIDGE_DEG, -1, -1)]) @ beta
    amp_mm = float(np.linalg.norm(X[i0 : i1 + 1] - fit, axis=1).max() * 1e3)
    return amp_mm, return_mm


def _edge_deg_per_m(step: np.ndarray, angle_deg: np.ndarray, k: int) -> float:
    """Rotation per metre of translation over edge interval k and its larger
    neighbour -- the 22/44 ms receive cadence can split one edge in two."""
    neighbours = [j for j in (k - 1, k + 1) if 0 <= j < len(step)]
    pair = [k] + ([max(neighbours, key=lambda j: step[j])] if neighbours else [])
    return float(angle_deg[pair].sum()) / max(float(step[pair].sum()), 1e-9)


def check_tracking_excursions(
    poses: np.ndarray,
    timestamps: np.ndarray,
    *,
    dv_threshold_m_per_s: float = 1.0,
    min_amplitude_mm: float = 20.0,
    min_edge_share: float = 0.5,
    max_return_s: float = 1.0,
    return_floor_mm: float = 15.0,
    return_per_m_per_s_mm: float = 25.0,
    return_fraction: float = 0.25,
    context_s: float = 0.3,
    edge_rotation_deg_per_m: float = 50.0,
    max_events: int = 20,
) -> Dict[str, Any]:
    """Find tracker excursions: the position jumps off its path and comes back.

    A tracker relocalisation glitch moves the recorded position 20-150 mm in
    a frame or two and returns within ~100-300 ms. That is under the
    `check_tracking_pose_jumps` speed ceiling, so that check does not see it.
    This uses a translation rule plus a rotation test:

    1. Nominate a sample where the velocity changes by more than
       `dv_threshold_m_per_s` between consecutive intervals.
    2. Grow the excursion one sample at a time for up to `max_return_s`, and
       fit a quartic bridge to the `context_s` either side. It is an
       out-and-back at the first end where the excursion sits at least
       `min_amplitude_mm` off the bridge, the largest single step carries at
       least `min_edge_share` of it, and the trajectory after it is back on
       the bridge within max(`return_floor_mm` + `return_per_m_per_s_mm` x
       pre-event hand speed, `return_fraction` x amplitude).
    3. A sharp edge that never returns is a "step"; a nomination with no sharp
       edge is a brisk real motion and is only counted.
    4. Confirmed glitches move the position but not the orientation, while a
       real hand moving that fast always rotates. An event whose out edge
       carries more than `edge_rotation_deg_per_m` of rotation per metre of
       translation is "rejected" (reason "rotation"); the rest stay
       "candidate"s for video confirmation (`excursion_confirm.py`). The
       back edge is recorded but not gated on: one confirmed artefact rotates
       there, and this pre-filter favours recall.

    Measures only; whether a confirmed event fails the episode is the
    caller's decision. Never raises: empty, short or non-finite input yields
    no events, and non-positive timestamp gaps nominate nothing (the
    timestamp checks report them).

    Args:
        poses: (N, 4, 4) homogeneous transformation matrices
        timestamps: (N,) seconds; event times are on this clock

    Returns:
        JSON-serialisable dict with counts and the per-event list.
    """
    result: Dict[str, Any] = {
        "detector_version": EXCURSION_DETECTOR_VERSION,
        "num_samples": 0,
        "max_dv_m_per_s": 0.0,
        "num_nominated": 0,
        "num_events": 0,
        **excursion_state_counts([]),
        "events": [],
    }
    try:
        poses = np.asarray(poses, dtype=np.float64)
        t = np.asarray(timestamps, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return result
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        return result
    n = min(len(poses), len(t))
    poses, t = poses[:n], t[:n]
    result["num_samples"] = int(n)
    if n < 3 or not (np.isfinite(poses).all() and np.isfinite(t).all()):
        return result

    X = poses[:, :3, 3]
    dt = np.diff(t)
    ok = dt > 0
    step = np.linalg.norm(np.diff(X, axis=0), axis=1)
    speed = np.divide(step, dt, out=np.zeros_like(step), where=ok)
    # A non-positive gap has no velocity; the NaN zeroes both dv values it touches.
    vel = np.diff(X, axis=0) / np.where(ok, dt, np.nan)[:, None]
    dv = np.zeros(n)
    dv[1:-1] = np.nan_to_num(np.linalg.norm(vel[1:] - vel[:-1], axis=1))
    result["max_dv_m_per_s"] = round(float(dv.max()), 3)

    # Geodesic angle between consecutive rotations.
    rotations = poses[:, :3, :3]
    rel_trace = np.einsum("nij,nij->n", rotations[:-1], rotations[1:])
    angle_deg = np.degrees(np.arccos(np.clip((rel_trace - 1.0) / 2.0, -1.0, 1.0)))

    nominated = np.flatnonzero(dv > dv_threshold_m_per_s)
    result["num_nominated"] = int(len(nominated))

    events = []
    taken_until = -1
    for k in nominated:
        if len(events) >= max_events:
            break
        if k <= taken_until:
            continue  # inside an event already reported
        # The excursion starts at the sample after the larger step around k.
        i0 = k if step[k - 1] >= step[k] else k + 1
        i0 = min(max(i0, 1), n - 2)
        pre = np.flatnonzero((t >= t[i0] - context_s) & (np.arange(n) < i0))
        hand_v = float(np.median(speed[pre[:-1]])) if len(pre) > 2 else 0.0
        base_tol = return_floor_mm + return_per_m_per_s_mm * hand_v
        best = last = None
        for i1 in range(i0, n - 1):
            if t[i1] - t[i0 - 1] > max_return_s:
                break
            fit = _excursion_bridge(t, X, i0, i1, context_s)
            if fit is None:
                continue
            amp, back = fit
            if amp < min_amplitude_mm:
                continue
            edge = float(step[i0 - 1 : i1 + 1].max() * 1e3) / amp
            tol = max(base_tol, return_fraction * amp)
            last = dict(i1=i1, amp=amp, back=back, edge=edge, tol=tol)
            # The first end back on the bridge closes the excursion: on the
            # plateau `back` is about the amplitude, so nothing passes early.
            if edge >= min_edge_share and back <= tol:
                best = last
                break
        if best is None and (last is None or last["edge"] < min_edge_share):
            continue  # no sharp edge: a brisk real motion, counted only
        c = best or last
        returned = best is not None
        i1 = c["i1"]

        rot_out = _edge_deg_per_m(step, angle_deg, i0 - 1)
        rot_back = _edge_deg_per_m(step, angle_deg, i1) if returned else None
        quiet = rot_out <= edge_rotation_deg_per_m
        events.append(
            {
                "shape": "out_and_back" if returned else "step",
                "i0": int(i0),
                "i1": int(i1),
                "t_out_s": float(t[i0]),
                "t_back_s": float(t[i1]),
                "duration_ms": round(float(t[i1 + 1] - t[i0 - 1]) * 1e3, 1),
                "amplitude_mm": round(c["amp"], 1),
                "max_step_mm": round(float(step[i0 - 1 : i1 + 1].max() * 1e3), 1),
                "edge_share": round(c["edge"], 2),
                "return_mm": round(c["back"], 1),
                "return_tol_mm": round(c["tol"], 1),
                "hand_speed_m_per_s": round(hand_v, 2),
                "dv_m_per_s": round(float(dv[k]), 2),
                "edge_rotation_deg_per_m_out": round(rot_out, 1),
                "edge_rotation_deg_per_m_back": (
                    None if rot_back is None else round(rot_back, 1)
                ),
                "rotation_quiet": bool(quiet),
                "state": "candidate" if quiet else "rejected",
                "reason": None if quiet else "rotation",
            }
        )
        taken_until = i1 + 1

    result["events"] = events
    result["num_events"] = len(events)
    result.update(excursion_state_counts(events))
    return result

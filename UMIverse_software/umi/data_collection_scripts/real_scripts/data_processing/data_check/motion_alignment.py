# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera↔sensor temporal-alignment check from visual motion.

Pairs each wrist camera against the same-side arm sensors — the tracking pose
and the gripper — and measures the time offset to each. That is the
video/action pairing training actually consumes. Both signals are placed on
the ``receive_time − latency`` clock (the same math ``align_data`` applies),
so the expected offset is zero and whatever is measured is residual desync.

Vision module
    Mean absolute intensity change between one analyzed frame and the
    previous, on a downscaled grayscale image, with finger-colored pixels
    excluded (fingers move independently of the camera). The per-cell mean
    over a coarse grid gives one motion-speed signal per cell, and cells are
    auto-selected per pose reference: a cell tracks the arm if its speed
    correlates with the pose speed at zero lag. This drops rigid-mounted
    hardware (no motion regardless of the arm) and textureless regions
    without any hand-tuned ROI.

Gripper module
    The finger mask that the vision module subtracts is itself the gripper's
    visual aperture: the fingers fill more of the frame as it closes. Its
    per-cell fraction is differentiated and correlated against the gripper
    sensor's ``distance`` velocity, reusing the same cell selection and
    matcher. It comes from the same decode pass, so it costs one resize per
    analyzed frame. Cells are selected at zero lag, which bounds the offset
    this can measure to roughly one finger-travel time; that is far above the
    tens of milliseconds it exists to catch.

Pose module
    Angular speed |ω| is the primary reference — rotation-induced flow is
    depth-independent, whereas translation-induced flow gain drifts with
    scene depth. A combined |ω|+|v| reference is the fallback for episodes
    that move without rotating.

Timelines
    aligned  ``pose_interp`` row i corresponds 1:1 to the video frame at
             ``aligned_frame_indices[i]`` on the episode's
             ``valid_timestamps`` (align_data resamples pose on the
             ``receive_time − latency`` clock). Expected offset is 0 and the
             measured offset is the residual desync that training consumes —
             an iPhone delay beyond the recorded latency attr survives into
             ``pose_interp``.
    raw      ``receive_time − latency`` clocks per device (the same math
             align_data applies), so the measured offset predicts the
             aligned-mode residual on raw recordings.

Matcher
    Z-scored speed signals are cross-correlated over lags (parabolic
    sub-sample peak). Positive ``offset_s`` means the visual motion occurs
    later than the pose motion. A misalignment verdict additionally requires
    the warp-invariant Spearman rank estimator (restricted to motion-burst
    windows, the analog of gripper close events) to agree — a verified
    misalignment is a warning; an unverifiable pair is report-only.
"""

import os
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "compute_flow_cell_speeds",
    "pose_speeds",
    "pose_stall_flags",
    "motion_bursts",
    "combine_flow_cells",
    "pose_tracking_agreement",
    "aperture_velocity_cells",
    "gripper_distance_reference",
    "gripper_confidence",
    "estimate_flow_offset",
    "estimate_flow_offset_rank",
    "per_burst_offsets",
    "check_motion_alignment",
]


class CellSignals(NamedTuple):
    """The two per-cell signals one decode pass produces.

    ``motion`` drives the camera-vs-pose check and ``finger`` the
    camera-vs-gripper one. They come from the same frames and the same HSV
    mask, so the second signal costs one extra resize per analyzed frame.
    """

    motion: np.ndarray  # (N, rows, cols) intensity change per second
    finger: np.ndarray  # (N, rows, cols) fraction of the cell that is finger


# Finger HSV range (OpenCV H in [0,180]).
DEFAULT_HSV_LOWER = (90, 80, 60)
DEFAULT_HSV_UPPER = (130, 255, 255)


def _zscore(x: np.ndarray) -> Tuple[np.ndarray, float]:
    std = float(np.std(x))
    if std < 1e-12:
        return np.zeros_like(x), std
    return (x - np.mean(x)) / std, std


def _fill_nan(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """Linear-interpolate NaNs; returns (filled, finite_fraction)."""
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    frac = float(np.mean(finite)) if len(x) else 0.0
    if frac == 0.0:
        return np.zeros_like(x), 0.0
    if frac < 1.0:
        idx = np.arange(len(x))
        x = x.copy()
        x[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    return x, frac


def _prepare_frame(frame, downscale_width, hsv_lower, hsv_upper, kernel):
    """Downscaled grayscale, plus the not-finger and finger masks."""
    import cv2

    h, w = frame.shape[:2]
    if w > downscale_width:
        frame = cv2.resize(frame, (downscale_width, int(h * downscale_width / w)))
    finger = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), hsv_lower, hsv_upper)
    finger = cv2.dilate(finger, kernel, iterations=2)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), finger == 0, finger > 0


def _cell_finger_fraction(finger, grid):
    """Fraction of each grid cell covered by finger-colored pixels.

    This is the gripper's visual aperture: the fingers fill more of the frame
    as the gripper closes. INTER_AREA of a 0/1 mask is the per-cell mean.
    """
    import cv2

    rows, cols = grid
    return cv2.resize(
        finger.astype(np.float32), (cols, rows), interpolation=cv2.INTER_AREA
    ).astype(np.float64)


def _cell_motion(prev_gray, gray, prev_valid, valid, grid, min_cell_valid_frac):
    """Mean inter-frame intensity change per grid cell, over non-finger pixels."""
    import cv2

    rows, cols = grid
    mag = cv2.absdiff(gray, prev_gray).astype(np.float32)
    both_valid = (valid & prev_valid).astype(np.float32)
    # INTER_AREA of masked magnitude / INTER_AREA of mask == per-cell mean
    # over unmasked pixels.
    num = cv2.resize(mag * both_valid, (cols, rows), interpolation=cv2.INTER_AREA)
    den = cv2.resize(both_valid, (cols, rows), interpolation=cv2.INTER_AREA)
    return np.where(den >= min_cell_valid_frac, num / np.maximum(den, 1e-9), np.nan)


def _resample_to_frame_grid(
    sample_times: List[float],
    sample_cells: List[np.ndarray],
    timestamps: np.ndarray,
    n_out: int,
    grid: Tuple[int, int],
    mid_interval: bool = True,
) -> np.ndarray:
    """Place motion samples on the per-frame grid the estimators expect.

    A sample measures motion over ``[t_prev, t_i]``, so it belongs at that
    interval's midpoint rather than at its end. Resampling those midpoints
    onto the grid of per-frame midpoints keeps the timing identical at any
    stride, and restores the sample density ``combine_flow_cells`` needs (it
    drops any cell that is less than half finite).
    """
    rows, cols = grid
    out = np.full((n_out, rows, cols), np.nan, dtype=np.float64)
    if not sample_times:
        return out

    src_t = np.asarray(sample_times, dtype=np.float64)
    src_v = np.asarray(sample_cells, dtype=np.float64).reshape(len(src_t), rows * cols)
    if mid_interval:
        # A rate spans two frames, so row i sits at that interval's midpoint.
        # Row 0 spans no interval and stays NaN, as it does without striding.
        grid_t = np.full(n_out, np.nan)
        grid_t[1:] = (timestamps[: n_out - 1] + timestamps[1:n_out]) / 2.0
    else:
        # A level is measured at the frame itself.
        grid_t = np.asarray(timestamps[:n_out], dtype=np.float64)

    flat = out.reshape(n_out, rows * cols)
    for c in range(rows * cols):
        finite = np.isfinite(src_v[:, c])
        if not finite.any():
            continue
        t_c = src_t[finite]
        # Only interpolate inside the span this cell actually measured.
        inside = np.isfinite(grid_t) & (grid_t >= t_c[0]) & (grid_t <= t_c[-1])
        flat[inside, c] = np.interp(grid_t[inside], t_c, src_v[finite, c])
    return out


def compute_flow_cell_speeds(
    video_path: str,
    timestamps: np.ndarray,
    frame_indices: Optional[np.ndarray] = None,
    grid: Tuple[int, int] = (6, 8),
    downscale_width: int = 96,
    hsv_lower: Sequence[int] = DEFAULT_HSV_LOWER,
    hsv_upper: Sequence[int] = DEFAULT_HSV_UPPER,
    min_cell_valid_frac: float = 0.5,
    stride: int = 3,
    max_frames: Optional[int] = None,
) -> np.ndarray:
    """Per-analyzed-frame, per-cell visual motion speed for a whole video.

    Motion is the mean absolute intensity change between one analyzed frame
    and the previous one, per grid cell, divided by the elapsed time. Only
    the magnitude matters downstream — cells are selected and combined by
    their correlation with the pose speed — so a frame difference carries the
    same information as dense optical flow at a fraction of the cost.

    Analyzed frames are all native frames when ``frame_indices`` is None,
    else the native frames at those indices (aligned mode; repeats from
    nearest-frame matching yield zero motion over a positive dt, which is
    correct). Finger-colored pixels are masked out of the per-cell mean; a
    cell with fewer than ``min_cell_valid_frac`` unmasked pixels is NaN.

    Args:
        timestamps: One timestamp per analyzed frame; speeds are magnitudes
            divided by the local dt (intensity/s at the downscaled resolution).
        frame_indices: Native video frame index per analyzed frame
            (non-decreasing), or None for all frames.
        stride: Analyze every ``stride``-th frame, decoding the rest without
            converting them. Samples are timestamped at their interval
            midpoint and resampled onto the per-frame grid, so the stride
            does not shift the signal in time.

    Returns:
        ``CellSignals``: two (N, rows, cols) float arrays, ``motion`` (for the
        pose check) and ``finger`` (for the gripper check), from one decode
        pass. Row 0 and unmeasured samples are NaN.
    """
    import cv2

    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    timestamps = np.asarray(timestamps, dtype=np.float64)
    wanted = (
        None if frame_indices is None else np.asarray(frame_indices, dtype=np.int64)
    )
    n_out = len(timestamps) if wanted is None else len(wanted)
    if max_frames is not None:
        n_out = min(n_out, max_frames)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    kernel = np.ones((3, 3), np.uint8)
    lower, upper = np.asarray(hsv_lower), np.asarray(hsv_upper)
    motion_times: List[float] = []  # mid-interval: motion spans two frames
    motion_cells: List[np.ndarray] = []
    finger_times: List[float] = []  # frame time: aperture is a level, not a rate
    finger_cells: List[np.ndarray] = []
    decoder_at = 0  # next native frame the decoder will return
    held = -1  # native index of the frame currently in gray/valid
    gray = valid = None
    prev_gray = prev_valid = None
    prev_i = None
    try:
        for out_i in range(0, n_out, stride):
            native = out_i if wanted is None else int(wanted[out_i])
            if native != held:
                while decoder_at < native:  # skip: decode without converting
                    if not cap.grab():
                        break
                    decoder_at += 1
                ok, frame = cap.read()
                if not ok:
                    break
                decoder_at += 1
                held = native
                gray, valid, finger = _prepare_frame(
                    frame, downscale_width, lower, upper, kernel
                )
            finger_times.append(float(timestamps[out_i]))
            finger_cells.append(_cell_finger_fraction(finger, grid))
            if prev_gray is not None:
                dt = timestamps[out_i] - timestamps[prev_i]
                if np.isfinite(dt) and dt > 0:
                    cell = _cell_motion(
                        prev_gray, gray, prev_valid, valid, grid, min_cell_valid_frac
                    )
                    motion_times.append((timestamps[prev_i] + timestamps[out_i]) / 2.0)
                    motion_cells.append(cell / dt)
            prev_gray, prev_valid, prev_i = gray, valid, out_i
    finally:
        cap.release()

    return CellSignals(
        motion=_resample_to_frame_grid(
            motion_times, motion_cells, timestamps, n_out, grid, mid_interval=True
        ),
        finger=_resample_to_frame_grid(
            finger_times, finger_cells, timestamps, n_out, grid, mid_interval=False
        ),
    )


def pose_speeds(
    poses: np.ndarray, timestamps: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Angular speed |ω| (rad/s) and translational speed |v| (m/s) per row.

    Args:
        poses: (N, 4, 4) homogeneous transforms.
        timestamps: (N,) seconds.

    Returns:
        (omega, v), each (N,) with entry i the speed between rows i-1 and i;
        entry 0 and entries with non-finite poses or dt <= 0 are NaN.
    """
    poses = np.asarray(poses, dtype=np.float64)
    ts = np.asarray(timestamps, dtype=np.float64)
    n = min(len(poses), len(ts))
    omega = np.full(n, np.nan)
    v = np.full(n, np.nan)
    if n < 2:
        return omega, v

    R = poses[:n, :3, :3]
    p = poses[:n, :3, 3]
    dt = np.diff(ts[:n])
    finite = np.isfinite(poses[:n].reshape(n, -1)).all(axis=1)

    # Relative rotation angle: ||log(R_{i-1}^T R_i)|| via the trace.
    rel = np.einsum("nij,nik->njk", R[:-1], R[1:])  # R_{i-1}^T @ R_i
    tr = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(tr)
    dist = np.linalg.norm(np.diff(p, axis=0), axis=1)

    ok = finite[:-1] & finite[1:] & np.isfinite(dt) & (dt > 0)
    omega[1:][ok] = ang[ok] / dt[ok]
    v[1:][ok] = dist[ok] / dt[ok]
    return omega, v


def pose_stall_flags(
    poses: np.ndarray,
    timestamps: np.ndarray,
    max_gap_s: float = 0.25,
    stall_eps: float = 1e-9,
) -> Dict[str, Any]:
    """Cheap secondary flags on the raw pose stream (report-only).

    Interpolation smooths through delivery stalls, so ``pose_interp`` hides
    them; the raw stream shows them as timestamp gaps or repeated poses.
    """
    ts = np.asarray(timestamps, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64)
    n = min(len(poses), len(ts))
    result: Dict[str, Any] = {"max_gap_s": None, "repeated_pose_frac": None}
    if n < 2:
        return result
    gaps = np.diff(ts[:n])
    result["max_gap_s"] = round(float(np.max(gaps)), 4)
    result["n_gaps_over_threshold"] = int(np.sum(gaps > max_gap_s))
    flat = poses[:n].reshape(n, -1)
    repeated = np.all(np.abs(np.diff(flat, axis=0)) <= stall_eps, axis=1)
    result["repeated_pose_frac"] = round(float(np.mean(repeated)), 4)
    return result


def motion_bursts(
    timestamps: np.ndarray,
    ref_speed: np.ndarray,
    min_speed: float = 0.3,
    percentile: float = 70.0,
    min_duration_s: float = 0.25,
    merge_gap_s: float = 0.4,
    max_bursts: int = 40,
) -> List[Dict[str, Any]]:
    """Sustained motion segments of the reference speed signal.

    The analog of gripper close events: per-burst offsets and the rank
    estimator's sample windows anchor here so idle stretches don't dilute
    the statistics. Threshold = max(``min_speed``, episode ``percentile``).

    Returns:
        List of {"t_start", "t_end", "t_peak", "peak_speed"}, time-ordered.
    """
    ts = np.asarray(timestamps, dtype=np.float64)
    speed, _ = _fill_nan(ref_speed)
    n = min(len(ts), len(speed))
    if n < 4:
        return []
    ts, speed = ts[:n], speed[:n]
    thresh = max(min_speed, float(np.percentile(speed, percentile)))
    above = speed >= thresh

    segments: List[Tuple[int, int]] = []
    start = None
    for i in range(n):
        if above[i] and start is None:
            start = i
        elif not above[i] and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, n - 1))

    merged: List[List[int]] = []
    for s, e in segments:
        if merged and ts[s] - ts[merged[-1][1]] <= merge_gap_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    bursts = []
    for s, e in merged:
        if ts[e] - ts[s] < min_duration_s:
            continue
        k = s + int(np.argmax(speed[s : e + 1]))
        bursts.append(
            {
                "t_start": float(ts[s]),
                "t_end": float(ts[e]),
                "t_peak": float(ts[k]),
                "peak_speed": round(float(speed[k]), 4),
            }
        )
    if len(bursts) > max_bursts:
        bursts = sorted(bursts, key=lambda b: -b["peak_speed"])[:max_bursts]
        bursts.sort(key=lambda b: b["t_start"])
    return bursts


def pose_tracking_agreement(
    camera_ts: np.ndarray,
    visual_speed: np.ndarray,
    ref_ts: np.ndarray,
    ref_speed: np.ndarray,
    bin_s: float = 1.0,
) -> Optional[float]:
    """Zero-lag agreement between visual motion and pose speed on coarse bins.

    Answers "is the pose stream moving when the camera is?", with no lag
    search at all. It catches the failure the offset estimator cannot: when a
    pose stream is dead or dropping most of its motion, no shift lines the two
    signals up, so a lag search returns an arbitrary one. A healthy pair
    scores well above 0.5; a dead stream scores near zero or negative.

    Uses the mean over all cells rather than the correlation-selected subset,
    so it still reports a number for pairs that are otherwise unmeasurable (no
    cells correlate, too few motion bursts), which is exactly when a pose
    stream is broken.

    Returns:
        Pearson correlation of the binned signals, or None if either signal
        has too few bins or no variation to correlate.
    """
    camera_ts = np.asarray(camera_ts, dtype=np.float64)
    ref_ts = np.asarray(ref_ts, dtype=np.float64)
    n = min(len(camera_ts), len(visual_speed))
    if n < 2 or len(ref_ts) < 2:
        return None
    visual = _fill_nan(np.asarray(visual_speed)[:n])[0]
    ref = _fill_nan(np.asarray(ref_speed)[: len(ref_ts)])[0]

    t0 = camera_ts[0]
    n_bins = int((camera_ts[n - 1] - t0) / bin_s)
    if n_bins < 5:
        return None
    edges = t0 + np.arange(n_bins + 1) * bin_s
    v_bins = np.array(
        [
            visual[(camera_ts[:n] >= a) & (camera_ts[:n] < b)].mean()
            for a, b in zip(edges[:-1], edges[1:])
        ]
    )
    r_bins = np.array(
        [
            ref[(ref_ts >= a) & (ref_ts < b)].mean()
            if ((ref_ts >= a) & (ref_ts < b)).any()
            else np.nan
            for a, b in zip(edges[:-1], edges[1:])
        ]
    )
    ok = np.isfinite(v_bins) & np.isfinite(r_bins)
    if ok.sum() < 5 or np.std(v_bins[ok]) < 1e-12 or np.std(r_bins[ok]) < 1e-12:
        return None
    return float(np.corrcoef(v_bins[ok], r_bins[ok])[0, 1])


def combine_flow_cells(
    camera_ts: np.ndarray,
    cell_speeds: np.ndarray,
    ref_on_cam: np.ndarray,
    cell_min_corr: float = 0.15,
    min_cells: int = 3,
    min_finite_frac: float = 0.5,
    signed: bool = False,
) -> Optional[Dict[str, Any]]:
    """Select the cells that track the reference and combine them.

    A cell tracks the reference if the zero-lag correlation between its
    (z-scored) signal and the reference is at least ``cell_min_corr``.
    Selected cells combine with correlation weights into one signal.
    Rigid-mounted hardware and textureless cells simply never correlate.

    ``signed`` picks which cells count. Visual motion magnitude can only grow
    with ego-motion, so for the pose pairing a negative correlation is noise
    and the default keeps positive cells only. Gripper aperture is signed:
    the fingers move into some cells as it closes and out of others, so a
    cell can legitimately anticorrelate and carries just as much timing.
    ``signed=True`` selects on |corr| and flips those cells before combining;
    on recorded gripper data this made 24 of 24 pairs measurable instead of
    11 of 24, at the same mean offset.

    Returns:
        Dict with ``flow_speed`` (N,), ``n_cells``, ``cell_corr``
        (rows, cols), ``max_cell_corr`` — or None when fewer than
        ``min_cells`` cells track the reference.
    """
    n = min(len(camera_ts), len(cell_speeds), len(ref_on_cam))
    if n < 10:
        return None
    cell_speeds = np.asarray(cell_speeds[:n], dtype=np.float64)
    ref_z, r_std = _zscore(_fill_nan(ref_on_cam[:n])[0])
    if r_std < 1e-12:
        return None

    rows, cols = cell_speeds.shape[1:]
    flat = cell_speeds.reshape(n, rows * cols)
    cell_corr = np.zeros(rows * cols, dtype=np.float64)
    cell_z: List[Optional[np.ndarray]] = [None] * (rows * cols)
    for i in range(rows * cols):
        filled, frac = _fill_nan(flat[:, i])
        if frac < min_finite_frac:
            continue
        z, std = _zscore(filled)
        if std < 1e-12:
            continue
        cell_corr[i] = float(np.mean(z * ref_z))
        cell_z[i] = z

    score = np.abs(cell_corr) if signed else cell_corr
    selected = [
        i
        for i in range(rows * cols)
        if score[i] >= cell_min_corr and cell_z[i] is not None
    ]
    if len(selected) < min_cells:
        return None

    combined = np.zeros(n, dtype=np.float64)
    total_w = 0.0
    for i in selected:
        weight = abs(cell_corr[i])
        combined += np.sign(cell_corr[i]) * weight * cell_z[i]
        total_w += weight
    combined /= total_w

    return {
        "flow_speed": combined,
        "n_cells": len(selected),
        "cell_corr": cell_corr.reshape(rows, cols),
        "max_cell_corr": float(np.max(np.abs(cell_corr) if signed else cell_corr)),
    }


def estimate_flow_offset(
    camera_ts: np.ndarray,
    flow_speed: np.ndarray,
    ref_ts: np.ndarray,
    ref_speed: np.ndarray,
    max_lag_s: float = 0.5,
    smooth_k: int = 3,
) -> Dict[str, Any]:
    """Cross-correlation offset of the visual speed relative to the pose speed.

    Both signals are speeds already (the derivative domain the gripper check
    works in), so they are lightly smoothed, z-scored, and lag-scanned
    directly with parabolic sub-sample refinement. Positive ``offset_s``
    means the visual motion occurs later than the pose motion.
    """
    n = min(len(camera_ts), len(flow_speed))
    camera_ts = np.asarray(camera_ts[:n], dtype=np.float64)

    result: Dict[str, Any] = {
        "valid": False,
        "offset_s": None,
        "peak_correlation": None,
        "lags_s": [],
        "correlations": [],
        "dt_s": None,
    }
    if n < 10 or len(ref_ts) < 10:
        result["error"] = "Not enough samples"
        return result
    # Convert lag samples to seconds using mean(dt), not median(dt). Bimodal
    # camera intervals make the median understate elapsed time and bias offsets.
    dt = float(np.mean(np.diff(camera_ts)))
    if not np.isfinite(dt) or dt <= 0:
        result["error"] = "Invalid camera timestamps"
        return result
    result["dt_s"] = dt

    def prep(x):
        filled, _ = _fill_nan(x)
        if smooth_k > 1:
            filled = np.convolve(filled, np.ones(smooth_k) / smooth_k, mode="same")
        return _zscore(filled)

    ref_on_cam = np.interp(camera_ts, ref_ts, _fill_nan(ref_speed)[0])
    visual_z, v_std = prep(flow_speed[:n])
    ref_z, r_std = prep(ref_on_cam)
    if v_std < 1e-12 or r_std < 1e-12:
        result["error"] = "Flat signal (no arm motion or no visual motion)"
        return result

    max_lag = max(1, int(round(max_lag_s / dt)))
    if n <= 2 * max_lag + 2:
        result["error"] = "Window shorter than lag range"
        return result

    # corr[k]: visual delayed by k samples relative to the pose reference
    # (visual_z[i] matched against ref_z[i - k]).
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = np.empty(len(lags), dtype=np.float64)
    for j, k in enumerate(lags):
        if k >= 0:
            a, b = visual_z[k:], ref_z[: n - k]
        else:
            a, b = visual_z[:k], ref_z[-k:]
        corrs[j] = float(np.mean(a * b))

    peak_j = int(np.argmax(corrs))
    peak_lag = float(lags[peak_j])
    if 0 < peak_j < len(lags) - 1:
        c0, c1, c2 = corrs[peak_j - 1], corrs[peak_j], corrs[peak_j + 1]
        denom = c0 - 2 * c1 + c2
        if abs(denom) > 1e-12:
            peak_lag += 0.5 * (c0 - c2) / denom

    result.update(
        valid=True,
        offset_s=float(peak_lag * dt),
        peak_correlation=float(corrs[peak_j]),
        lags_s=(lags * dt).tolist(),
        correlations=corrs.tolist(),
        at_boundary=peak_j in (0, len(lags) - 1),
    )
    return result


def estimate_flow_offset_rank(
    camera_ts: np.ndarray,
    flow_speed: np.ndarray,
    ref_ts: np.ndarray,
    ref_speed: np.ndarray,
    bursts: Sequence[Dict[str, Any]],
    max_lag_s: float = 0.5,
    window_pad_s: float = 0.5,
) -> Dict[str, Any]:
    """Warp-invariant offset estimate: Spearman rank correlation vs lag.

    Flow gain drifts with scene depth and the flow→speed response need not
    be linear; if the visual signal is any monotone function of the true
    camera speed plus noise, rank correlation still peaks at the true shift.
    Samples are restricted to motion-burst windows so idle stretches don't
    dilute the statistic (the analog of transition windows in the gripper
    check's rank estimator).
    """
    from scipy.stats import rankdata

    n = min(len(camera_ts), len(flow_speed))
    camera_ts = np.asarray(camera_ts[:n], dtype=np.float64)
    result: Dict[str, Any] = {
        "valid": False,
        "offset_s": None,
        "peak_rank_correlation": None,
        "n_samples": 0,
    }
    if not bursts or n < 10:
        return result

    mask = np.zeros(n, dtype=bool)
    for b in bursts:
        mask |= (camera_ts >= b["t_start"] - window_pad_s) & (
            camera_ts <= b["t_end"] + window_pad_s
        )
    idx = np.nonzero(mask)[0]
    if len(idx) < 50:
        return result
    result["n_samples"] = int(len(idx))

    dt = float(np.median(np.diff(camera_ts)))
    max_lag = max(1, int(round(max_lag_s / dt)))
    t_sel = camera_ts[idx]
    ref_filled, _ = _fill_nan(ref_speed)
    visual_z, v_std = _zscore(rankdata(_fill_nan(flow_speed[:n])[0][idx]))
    if v_std < 1e-12:
        return result

    lags = np.arange(-max_lag, max_lag + 1)
    rhos = np.empty(len(lags), dtype=np.float64)
    for j, k in enumerate(lags):
        # visual delayed by k samples => it matches the pose k*dt earlier
        s = np.interp(t_sel - k * dt, ref_ts, ref_filled)
        ref_z, r_std = _zscore(rankdata(s))
        rhos[j] = float(np.mean(visual_z * ref_z)) if r_std > 1e-12 else 0.0

    peak_j = int(np.argmax(rhos))
    peak_lag = float(lags[peak_j])
    if 0 < peak_j < len(lags) - 1:
        c0, c1, c2 = rhos[peak_j - 1], rhos[peak_j], rhos[peak_j + 1]
        denom = c0 - 2 * c1 + c2
        if abs(denom) > 1e-12:
            peak_lag += 0.5 * (c0 - c2) / denom

    result.update(
        valid=True,
        offset_s=round(float(peak_lag * dt), 4),
        peak_rank_correlation=round(float(rhos[peak_j]), 3),
        at_boundary=peak_j in (0, len(lags) - 1),
    )
    return result


def per_burst_offsets(
    camera_ts: np.ndarray,
    flow_speed: np.ndarray,
    ref_ts: np.ndarray,
    ref_speed: np.ndarray,
    bursts: Sequence[Dict[str, Any]],
    window_pad_s: float = 0.75,
    max_lag_s: float = 0.5,
) -> List[Dict[str, Any]]:
    """Local offset estimate around each motion burst (median-aggregated by
    the caller). One dict per burst: t_peak_s relative to episode start,
    offset_s, peak_correlation, valid."""
    n = min(len(camera_ts), len(flow_speed))
    camera_ts = np.asarray(camera_ts[:n], dtype=np.float64)
    t0 = float(camera_ts[0])
    results = []
    for b in bursts:
        lo = int(np.searchsorted(camera_ts, b["t_start"] - window_pad_s))
        hi = int(np.searchsorted(camera_ts, b["t_end"] + window_pad_s))
        est = estimate_flow_offset(
            camera_ts[lo:hi],
            flow_speed[lo:hi],
            ref_ts,
            ref_speed,
            max_lag_s=max_lag_s,
        )
        results.append(
            {
                "t_peak_s": round(b["t_peak"] - t0, 3),
                "offset_s": None
                if est["offset_s"] is None
                else round(est["offset_s"], 4),
                "peak_correlation": None
                if est["peak_correlation"] is None
                else round(est["peak_correlation"], 3),
                "valid": bool(est["valid"]),
            }
        )
    return results


def aperture_velocity_cells(
    finger_cells: np.ndarray, camera_ts: np.ndarray
) -> np.ndarray:
    """Per-cell rate of change of visual gripper aperture.

    Aperture is ``-finger_fraction``: the fingers fill more of the cell as the
    gripper closes, while the sensor's ``distance`` falls, so negating puts the
    two signals in the same direction and positive-only cell selection works
    unchanged. Both sides are differentiated because the levels drift (lighting,
    pose) while their rates of change do not.
    """
    n = min(len(finger_cells), len(camera_ts))
    cells = np.asarray(finger_cells[:n], dtype=np.float64)
    ts = np.asarray(camera_ts[:n], dtype=np.float64)
    if n < 3:
        return np.full_like(cells, np.nan)
    rows, cols = cells.shape[1:]
    flat = cells.reshape(n, rows * cols)
    out = np.full_like(flat, np.nan)
    for c in range(rows * cols):
        filled, frac = _fill_nan(flat[:, c])
        if frac > 0.0:
            out[:, c] = np.gradient(-filled, ts)
    return out.reshape(n, rows, cols)


def gripper_confidence(
    camera_ts: np.ndarray,
    aperture_speed: np.ndarray,
    grip_ts: np.ndarray,
    grip_vel: np.ndarray,
    peak_correlation: float,
    min_events: int = 3,
    max_spread_s: float = 0.030,
    window_pad_s: float = 1.0,
    event_min_correlation: float = 0.5,
) -> Dict[str, Any]:
    """Is this episode's gripper motion enough to time the camera against?

    A single actuation, or a gripper barely moving, produces an offset the
    estimator will still report. The check here is repeatability within the
    episode: measure the offset separately around each open/close event and
    see whether they agree. A real delay is the same at every event, so the
    scatter of those is the uncertainty, and it costs nothing to compute
    because the signals are already built.

    Returns a dict with ``trusted`` plus the numbers behind it, so a rejected
    pair says which condition failed rather than just disappearing.
    """
    # Use the 90th percentile and a 0.2 s minimum duration to select sustained
    # gripper travel rather than velocity noise.
    speed = np.abs(np.asarray(grip_vel, dtype=np.float64))
    events = motion_bursts(
        grip_ts,
        speed,
        min_speed=float(np.percentile(speed, 90.0)) if len(speed) else 0.0,
        percentile=90.0,
        min_duration_s=0.2,
        merge_gap_s=0.6,
    )
    per_event = per_burst_offsets(
        camera_ts,
        aperture_speed,
        grip_ts,
        grip_vel,
        events,
        window_pad_s=window_pad_s,
    )
    offsets = [
        b["offset_s"]
        for b in per_event
        if b["valid"]
        and b["offset_s"] is not None
        and (b["peak_correlation"] or 0.0) >= event_min_correlation
    ]
    spread = (
        float(np.median(np.abs(np.asarray(offsets) - np.median(offsets))))
        if len(offsets) >= 2
        else None
    )
    reasons = []
    if len(offsets) < min_events:
        reasons.append(f"only {len(offsets)} usable open/close events (< {min_events})")
    if spread is not None and spread > max_spread_s:
        reasons.append(
            f"per-event offsets scatter by {spread * 1000:.0f}ms "
            f"(> {max_spread_s * 1000:.0f}ms), so the offset is not repeatable"
        )
    if peak_correlation < 0.35:
        reasons.append(f"correlation {peak_correlation:.2f} too low to time against")
    return {
        "trusted": not reasons,
        "n_events": len(events),
        "n_usable_events": len(offsets),
        "median_event_offset_s": round(float(np.median(offsets)), 4)
        if offsets
        else None,
        "event_spread_s": None if spread is None else round(spread, 4),
        "reasons": reasons,
    }


def gripper_distance_reference(
    episode_data,
    device: str,
    latencies: Dict[str, float],
    valid_ts: Optional[np.ndarray],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(timestamps, distance velocity) for one gripper, on the analysis clock.

    Mirrors the pose reference: ``distance_interp`` on ``valid_timestamps`` in
    aligned mode, else raw ``distance`` on ``receive_time - latency``.
    """
    group = episode_data[device]
    keys = list(group.array_keys())
    if valid_ts is not None and "distance_interp" in keys:
        dist = np.asarray(group["distance_interp"][:], dtype=np.float64)
        ts = valid_ts[: len(dist)]
    elif "distance" in keys:
        dist = np.asarray(group["distance"][:], dtype=np.float64)
        ts = _adjusted_timestamps(episode_data, device, latencies)
        if ts is None:
            return None
    else:
        return None
    n = min(len(dist), len(ts))
    dist, ts = _fill_nan(dist[:n])[0], np.asarray(ts[:n], dtype=np.float64)
    if n < 3 or np.std(dist) < 1e-12:
        return None
    return ts, np.gradient(dist, ts)


def _device_side(device_name: str) -> Optional[str]:
    lowered = device_name.lower()
    for side in ("left", "right", "head"):
        if side in lowered:
            return side
    return None


def _adjusted_timestamps(
    episode_data, device: str, latencies: Dict[str, float]
) -> Optional[np.ndarray]:
    """Latency-adjusted timestamps, mirroring align_data's clock math."""
    group = episode_data[device]
    if "receive_time" in group and device in latencies:
        return np.asarray(group["receive_time"][:], dtype=np.float64) - float(
            latencies[device]
        )
    for key in ("capture_time", "receive_time"):
        if key in group:
            return np.asarray(group[key][:], dtype=np.float64)
    return None


def check_motion_alignment(
    episode_data,
    data_dir: str,
    episode_name: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the camera↔arm-pose alignment check for every same-side pair.

    Every camera with an mp4 next to the episode zarr is paired with the
    tracking device on its own side (the iPhone video pairs with its own
    pose — a sanity pair that stays clean even when the whole iPhone stream
    is late, which disambiguates iPhone delay from a broken flow estimate).
    A pose stream that does not track the camera fails the episode. A verified
    time offset is a warning, because ``max_residual_s`` has only been
    validated on some hardware generations. Pairs without enough motion,
    texture, or correlation are report-only.

    Args:
        episode_data: Zarr group for the episode.
        data_dir: Recordings dir containing ``{episode_name}/{camera}.mp4``.
        episode_name: Episode key; empty string when ``data_dir`` is the
            episode directory.
        params: ``motion_alignment`` block of the check config.

    Returns:
        Dict with mode, per-camera results (per-tracking-device pairs),
        per-tracking-device raw-stream stall flags, overall passed flag,
        issues list (fail the episode) and warnings list (do not).
    """
    p = dict(params or {})
    grid = tuple(p.get("grid", (6, 8)))
    hsv_lower = tuple(p.get("finger_hsv_lower", DEFAULT_HSV_LOWER))
    hsv_upper = tuple(p.get("finger_hsv_upper", DEFAULT_HSV_UPPER))
    downscale_width = p.get("downscale_width", 96)
    stride = p.get("stride", 3)
    grip_min_events = p.get("gripper_min_events", 3)
    grip_max_spread_s = p.get("gripper_max_spread_s", 0.030)
    cell_min_corr = p.get("cell_min_corr", 0.15)
    min_cells = p.get("min_cells", 3)
    max_lag_s = p.get("max_lag_s", 0.5)
    max_residual_s = p.get("max_residual_s", 0.06)
    min_correlation = p.get("min_correlation", 0.35)
    burst_min_omega = p.get("burst_min_omega", 0.3)
    burst_percentile = p.get("burst_percentile", 70.0)
    burst_min_duration_s = p.get("burst_min_duration_s", 0.25)
    burst_merge_gap_s = p.get("burst_merge_gap_s", 0.4)
    min_bursts = p.get("min_bursts", 2)
    burst_window_pad_s = p.get("burst_window_pad_s", 0.75)
    burst_min_correlation = p.get("burst_min_correlation", 0.5)
    rank_min_bursts = p.get("rank_min_bursts", 3)
    track_min_corr = p.get("pose_tracking_min_corr", 0.35)
    track_bin_s = p.get("pose_tracking_bin_s", 1.0)
    stall_note_gap_s = p.get("stall_note_gap_s", 0.25)
    stall_note_repeated_frac = p.get("stall_note_repeated_frac", 0.05)

    latencies = dict(episode_data.attrs.get("latency", {}))
    results: Dict[str, Any] = {
        "passed": True,
        "mode": None,
        "cameras": {},
        "tracking": {},
        "issues": [],  # pose not tracking the camera — these fail the episode
        "warnings": [],  # verified time offsets — reported, never a failure
        "notes": [],  # report-only observations (estimator disagreement)
    }

    aligned_mode = "valid_timestamps" in episode_data.array_keys()
    valid_ts = (
        np.asarray(episode_data["valid_timestamps"][:], dtype=np.float64)
        if aligned_mode
        else None
    )
    results["mode"] = "aligned" if aligned_mode else "raw"

    # ── Pose module: one reference per tracking device ──────────────────────
    references: Dict[str, Dict[str, Any]] = {}
    for device in episode_data.group_keys():
        group = episode_data[device]
        if "pose" not in group and "pose_interp" not in group:
            continue
        info: Dict[str, Any] = {"side": _device_side(device)}
        if aligned_mode and "pose_interp" in group:
            poses = np.asarray(group["pose_interp"][:])
            ref_ts = valid_ts[: len(poses)]
        elif "pose" in group:
            poses = np.asarray(group["pose"][:])
            ref_ts = _adjusted_timestamps(episode_data, device, latencies)
            if ref_ts is None:
                continue
            n = min(len(poses), len(ref_ts))
            poses, ref_ts = poses[:n], ref_ts[:n]
        else:
            continue
        omega, v = pose_speeds(poses, ref_ts)
        info["ts"] = ref_ts
        info["omega"] = omega
        info["v"] = v
        info["bursts"] = motion_bursts(
            ref_ts,
            omega,
            min_speed=burst_min_omega,
            percentile=burst_percentile,
            min_duration_s=burst_min_duration_s,
            merge_gap_s=burst_merge_gap_s,
        )
        # Raw-stream stall/gap flags (report-only) — interpolation hides these.
        if "pose" in group and "capture_time" in group:
            info["raw_stalls"] = pose_stall_flags(
                np.asarray(group["pose"][:]),
                np.asarray(group["capture_time"][:], dtype=np.float64),
            )
        references[device] = info
        results["tracking"][device] = {
            "n_bursts": len(info["bursts"]),
            "raw_stalls": info.get("raw_stalls"),
        }
        stalls = info.get("raw_stalls") or {}
        if (stalls.get("max_gap_s") or 0.0) > stall_note_gap_s or (
            stalls.get("repeated_pose_frac") or 0.0
        ) > stall_note_repeated_frac:
            results["notes"].append(
                f"{device}: raw pose stream irregular — max gap "
                f"{stalls.get('max_gap_s')}s, repeated poses "
                f"{(stalls.get('repeated_pose_frac') or 0.0) * 100:.1f}% "
                "(delivery stalls; interpolation smooths these away)"
            )

    # The gripper pairing needs no pose at all, so only bail when there is
    # nothing on either side to compare the camera against.
    has_gripper = any("umi_gripper" in d for d in episode_data.group_keys())
    if not references and not has_gripper:
        results["skip_reason"] = "No tracking device with pose data"
        return results

    # Report pose/pose angular-speed correlation: synchronized arm motion makes
    # cross-side attribution ambiguous.
    if len(references) >= 2:
        results["pose_sync"] = {}
        devs = sorted(references)
        for i in range(len(devs)):
            for j in range(i + 1, len(devs)):
                a, b = references[devs[i]], references[devs[j]]
                b_on_a = np.interp(a["ts"], b["ts"], _fill_nan(b["omega"])[0])
                az, a_std = _zscore(_fill_nan(a["omega"])[0])
                bz, b_std = _zscore(b_on_a)
                if a_std > 1e-12 and b_std > 1e-12:
                    results["pose_sync"][f"{devs[i]}~{devs[j]}"] = round(
                        float(np.mean(az * bz)), 3
                    )

    # ── Vision module: every camera with a video, paired same-side ──────────
    cameras = [
        d
        for d in episode_data.group_keys()
        if os.path.exists(os.path.join(data_dir, episode_name, f"{d}.mp4"))
        and "gripper" not in d.lower()
    ]
    # By default, gate same-side wrist-camera/pose pairs. Enable
    # umi_cameras_only: false for additional diagnostic camera pairs.
    if p.get("umi_cameras_only", True):
        cameras = [
            d
            for d in cameras
            if d.endswith("_umi_camera") and _device_side(d) in ("left", "right")
        ]

    for camera_device in cameras:
        cam_side = _device_side(camera_device)
        refs = {
            dev: info for dev, info in references.items() if info["side"] == cam_side
        }
        cross_side = False
        if not refs:
            # Without a same-side pose, compare the camera against all trackers.
            # Keep these pairs report-only because synchronized arm motion makes
            # per-cell attribution ambiguous.
            refs = dict(references)
            cross_side = True
        cam_result: Dict[str, Any] = {"passed": True, "pairs": {}}
        results["cameras"][camera_device] = cam_result

        cam_group = episode_data[camera_device]
        if aligned_mode and "aligned_frame_indices" in cam_group:
            frame_indices = np.asarray(cam_group["aligned_frame_indices"][:])
            camera_ts = valid_ts[: len(frame_indices)]
        else:
            camera_ts = _adjusted_timestamps(episode_data, camera_device, latencies)
            frame_indices = None
        if camera_ts is None:
            cam_result["skip_reason"] = "No timestamps"
            continue

        video_path = os.path.join(data_dir, episode_name, f"{camera_device}.mp4")
        signals = compute_flow_cell_speeds(
            video_path,
            camera_ts,
            frame_indices=frame_indices,
            grid=grid,
            downscale_width=downscale_width,
            hsv_lower=hsv_lower,
            hsv_upper=hsv_upper,
            stride=stride,
        )

        for ref_device, ref in refs.items():
            pair: Dict[str, Any] = {
                "self_pair": ref_device == camera_device,
                "cross_side": cross_side,
                "measurable": False,
                "n_bursts": len(ref["bursts"]),
            }
            cam_result["pairs"][ref_device] = pair

            # Compute zero-lag tracking agreement before offset estimation so a dead
            # pose
            # stream still reports a failure. Use selected cells when available to
            # exclude independent object motion; otherwise use all-cell motion as the
            # fallback signal.
            gate_omega = _fill_nan(ref["omega"])[0]
            gate_combo = combine_flow_cells(
                camera_ts,
                signals.motion,
                np.interp(camera_ts, ref["ts"], gate_omega),
                cell_min_corr=cell_min_corr,
                min_cells=min_cells,
            )
            if gate_combo is not None:
                gate_signal = gate_combo["flow_speed"]
                pair["pose_tracking_basis"] = "selected_cells"
            else:
                n_cam = min(len(camera_ts), len(signals.motion))
                gate_signal = np.nanmean(
                    np.asarray(signals.motion[:n_cam], dtype=np.float64).reshape(
                        n_cam, -1
                    ),
                    axis=1,
                )
                pair["pose_tracking_basis"] = "all_cells"
            tracking_corr = pose_tracking_agreement(
                camera_ts,
                gate_signal,
                ref["ts"],
                gate_omega,
                bin_s=track_bin_s,
            )
            pair["pose_tracking_corr"] = (
                None if tracking_corr is None else round(tracking_corr, 3)
            )
            if tracking_corr is not None and tracking_corr < track_min_corr:
                pair["pose_not_tracking"] = True
                pair["passed"] = False
                msg = (
                    f"{camera_device} vs {ref_device}: pose stream does not "
                    f"track the camera (zero-lag agreement {tracking_corr:.2f} "
                    f"< {track_min_corr}) — the pose is not following the arm, "
                    "so no time offset exists to measure"
                )
                if cross_side:
                    results["notes"].append(f"cross-side (report-only): {msg}")
                else:
                    cam_result["passed"] = False
                    results["issues"].append(msg)

            if pair.get("pose_not_tracking"):
                # Verdict already decided; the offset below cannot be trusted.
                continue
            if len(ref["bursts"]) < min_bursts:
                pair["skip_reason"] = (
                    f"Only {len(ref['bursts'])} motion bursts (< {min_bursts})"
                )
                continue

            # Rotation-first reference; combined |ω|+|v| fallback for
            # episodes that translate without rotating.
            candidates = [("angular", _fill_nan(ref["omega"])[0])]
            omega_z, o_std = _zscore(_fill_nan(ref["omega"])[0])
            v_z, v_std = _zscore(_fill_nan(ref["v"])[0])
            if o_std > 1e-12 and v_std > 1e-12:
                candidates.append(("combined", omega_z + v_z))

            def _evaluate(ref_ts_eval):
                found = None
                for ref_name, ref_speed in candidates:
                    ref_on_cam = np.interp(
                        camera_ts,
                        ref_ts_eval,
                        np.asarray(ref_speed, dtype=np.float64),
                    )
                    combo = combine_flow_cells(
                        camera_ts,
                        signals.motion,
                        ref_on_cam,
                        cell_min_corr=cell_min_corr,
                        min_cells=min_cells,
                    )
                    if combo is None:
                        continue
                    est = estimate_flow_offset(
                        camera_ts,
                        combo["flow_speed"],
                        ref_ts_eval,
                        ref_speed,
                        max_lag_s=max_lag_s,
                    )
                    if not est["valid"]:
                        continue
                    if (
                        found is None
                        or est["peak_correlation"] > found[3]["peak_correlation"]
                    ):
                        found = (ref_name, ref_speed, combo, est)
                    if est["peak_correlation"] >= min_correlation:
                        break  # rotation-first: keep |ω| when it already verifies
                return found

            best = _evaluate(ref["ts"])

            if best is None:
                pair["skip_reason"] = (
                    "No flow cells track the arm motion (textureless view, "
                    "all-rigid view, or camera fault)"
                )
                continue
            ref_name, ref_speed, combo, est = best
            pair["measurable"] = True
            pair["reference_used"] = ref_name
            pair["n_cells"] = combo["n_cells"]

            # Expected offset is 0 in both modes: aligned rows are nominally
            # simultaneous, and raw clocks already have latency applied.
            measured = float(est["offset_s"])
            residual = measured
            pair["global"] = {
                "offset_s": round(measured, 4),
                "expected_offset_s": 0.0,
                "residual_s": round(residual, 4),
                "peak_correlation": round(est["peak_correlation"], 3),
                "at_boundary": est.get("at_boundary", False),
            }
            pair["bursts"] = per_burst_offsets(
                camera_ts,
                combo["flow_speed"],
                ref["ts"],
                ref_speed,
                ref["bursts"],
                window_pad_s=burst_window_pad_s,
                max_lag_s=max_lag_s,
            )
            confident = [
                b["offset_s"]
                for b in pair["bursts"]
                if b["valid"]
                and b["peak_correlation"] is not None
                and b["peak_correlation"] >= burst_min_correlation
                and b["offset_s"] is not None
            ]
            pair["burst_summary"] = {
                "n_bursts": len(pair["bursts"]),
                "n_confident": len(confident),
                "median_offset_s": round(float(np.median(confident)), 4)
                if confident
                else None,
            }

            rank = estimate_flow_offset_rank(
                camera_ts,
                combo["flow_speed"],
                ref["ts"],
                ref_speed,
                ref["bursts"],
                max_lag_s=max_lag_s,
            )
            rank_residual = None
            if rank["valid"]:
                rank_residual = rank["offset_s"]
                pair["global"]["rank_offset_s"] = round(rank_residual, 4)
                pair["global"]["rank_residual_s"] = round(rank_residual, 4)
                pair["global"]["rank_correlation"] = rank["peak_rank_correlation"]

            # Verdict gating mirrors the gripper check: xcorr flags, the
            # warp-invariant rank estimator must not contradict it.
            verified = est["peak_correlation"] >= min_correlation
            pair["verified"] = verified
            xcorr_flags = verified and abs(residual) > max_residual_s
            rank_available = (
                rank_residual is not None and len(ref["bursts"]) >= rank_min_bursts
            )
            rank_confirms = (
                rank_available
                and abs(rank_residual) > max_residual_s
                and np.sign(rank_residual) == np.sign(residual)
            )
            if xcorr_flags and rank_available and not rank_confirms:
                pair["suspect_estimator_bias"] = True
                pair["passed"] = True
                results["notes"].append(
                    f"{camera_device} vs {ref_device}: xcorr suggests "
                    f"{residual * 1000:.0f}ms residual but the rank estimator "
                    f"measures {rank_residual * 1000:.0f}ms — estimator "
                    "disagreement, not flagged; inspect diagnostics"
                )
            elif xcorr_flags:
                pair["passed"] = False
                rank_note = (
                    f", rank estimator confirms {rank_residual * 1000:.0f}ms"
                    if rank_confirms
                    else ""
                )
                msg = (
                    f"{camera_device} vs {ref_device}: camera-pose "
                    f"misalignment {measured * 1000:.0f}ms (expected 0ms on "
                    f"{results['mode']} timeline, residual "
                    f"{residual * 1000:.0f}ms{rank_note} > "
                    f"{max_residual_s * 1000:.0f}ms)"
                )
                if cross_side:
                    results["notes"].append(f"cross-side (report-only): {msg}")
                else:
                    results["warnings"].append(msg)
            else:
                pair["passed"] = True

        # Reuse the decoded finger mask as visual gripper aperture, avoiding
        # another video decode.
        gripper_device = f"{cam_side}_umi_gripper" if cam_side else None
        if gripper_device and gripper_device in episode_data.group_keys():
            grip_pair: Dict[str, Any] = {"measurable": False}
            cam_result["gripper"] = {gripper_device: grip_pair}
            reference = gripper_distance_reference(
                episode_data, gripper_device, latencies, valid_ts
            )
            if reference is None:
                grip_pair["skip_reason"] = "No usable gripper distance signal"
            else:
                grip_ts, grip_vel = reference
                aperture = aperture_velocity_cells(signals.finger, camera_ts)
                combo = combine_flow_cells(
                    camera_ts,
                    aperture,
                    np.interp(camera_ts, grip_ts, grip_vel),
                    cell_min_corr=cell_min_corr,
                    min_cells=min_cells,
                    signed=True,
                )
                if combo is None:
                    grip_pair["skip_reason"] = (
                        "No cells track the gripper (fingers out of view, "
                        "gripper never actuated, or a stuck encoder)"
                    )
                else:
                    est = estimate_flow_offset(
                        camera_ts,
                        combo["flow_speed"],
                        grip_ts,
                        grip_vel,
                        max_lag_s=max_lag_s,
                    )
                    confidence = gripper_confidence(
                        camera_ts,
                        combo["flow_speed"],
                        grip_ts,
                        grip_vel,
                        est.get("peak_correlation") or 0.0,
                        min_events=grip_min_events,
                        max_spread_s=grip_max_spread_s,
                    )
                    grip_pair["confidence"] = confidence
                    if not est["valid"]:
                        grip_pair["skip_reason"] = est.get("error", "Not measurable")
                    elif not confidence["trusted"]:
                        # Reported, never a verdict: an offset the episode
                        # cannot support is worse than no offset.
                        grip_pair["skip_reason"] = "; ".join(confidence["reasons"])
                    else:
                        grip_pair["measurable"] = True
                        grip_pair["n_cells"] = combo["n_cells"]
                        grip_pair["global"] = {
                            "offset_s": round(float(est["offset_s"]), 4),
                            "expected_offset_s": 0.0,
                            "residual_s": round(float(est["offset_s"]), 4),
                            "peak_correlation": round(est["peak_correlation"], 3),
                            "at_boundary": est.get("at_boundary", False),
                        }
                        verified = est["peak_correlation"] >= min_correlation
                        grip_pair["verified"] = verified
                        if verified and abs(est["offset_s"]) > max_residual_s:
                            grip_pair["passed"] = False
                            results["warnings"].append(
                                f"{camera_device} vs {gripper_device}: "
                                f"camera-gripper misalignment "
                                f"{est['offset_s'] * 1000:.0f}ms (expected 0ms on "
                                f"{results['mode']} timeline, > "
                                f"{max_residual_s * 1000:.0f}ms)"
                            )
                        else:
                            grip_pair["passed"] = True

    # A pair's own `passed` still records a measured offset; only an issue
    # fails the episode.
    results["passed"] = not results["issues"]
    return results

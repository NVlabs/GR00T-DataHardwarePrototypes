# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Confirm tracker excursions from the cameras rigidly mounted with the tracker.

A tracker glitch moves the recorded position while the hand stays put, so a
camera on the same gripper -- the iPhone's own video
(``{side}_tracking_camera.mp4``) or the wrist fisheye (``{side}_umi_camera.mp4``)
-- sees no matching image motion. A real motion moves the camera with the
tracker.

The statistic is a gain ratio: image motion per mm of tracker step at the
event's edge frames, divided by the same in the surrounding context. A real
motion keeps the camera's px/mm gain at ~1; a glitch drives it towards 0. On
the confirmation study (72 brisk real motions, 110 injected glitches, 2
labelled real events) confirm < 0.20 / reject >= 0.40 confirmed 72% of glitches
and 0% of real motions. The middle band stays unverified.

Only the frames in ``[t_out - pad_s, t_back + pad_s]`` are decoded, so an
event costs ~0.1 s per camera. cv2 is imported lazily.
"""

import logging
import os
import time
from typing import Any, Dict, Optional

import numpy as np

from .motion_alignment import (
    DEFAULT_HSV_LOWER,
    DEFAULT_HSV_UPPER,
    _adjusted_timestamps,
    _prepare_frame,
)

__all__ = ["confirm_excursions"]

logger = logging.getLogger(__name__)

CONFIRM_METHOD = "video_gain_v1"
DEFAULT_PARAMS = {
    "confirm_max": 0.20,  # every usable camera under this -> confirmed
    "reject_min": 0.40,  # any usable camera at or above this -> rejected
    "pad_s": 0.3,  # context decoded either side of the excursion
    "max_events": 6,  # candidates decoded per device; the rest stay unverified
}
EDGE_TOL_S = 0.05  # camera frames this close to an edge are edge frames
DOWNSCALE_WIDTH = 240
MAX_CORNERS = 150
FB_MAX_PX = 1.0  # forward-backward LK error that keeps a feature
MIN_FEATURES = 6  # tracked features for a frame pair to count
MIN_TRACKED_FRAC = 0.8  # frame pairs that must count for the camera to be usable
FLOW_PERCENTILE = 90  # the median misses motion when rigid gripper parts fill the view
TRK_CTX_FLOOR_MM = 0.3  # a near-still context must not inflate the context gain


def _unverified(event: Dict[str, Any], skip_reason: str, cameras=None) -> None:
    event["state"] = "unverified"
    event["confirm"] = {
        "method": CONFIRM_METHOD,
        "score": None,
        "cameras": cameras or {},
        "skip_reason": skip_reason,
    }


def _sanitise_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """DEFAULT_PARAMS overridden by every value that is a usable number; a
    malformed one (wrong type, NaN, negative) falls back to its default."""
    p = dict(DEFAULT_PARAMS)
    for key, default in DEFAULT_PARAMS.items():
        value = (
            (params or {}).get(key, default) if isinstance(params, dict) else default
        )
        try:
            value = type(default)(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value >= 0 and not (key == "pad_s" and value == 0):
            p[key] = value
    return p


def _round(x: float, places: int) -> Optional[float]:
    return round(float(x), places) if np.isfinite(x) else None


def _decode_window(video_path: str, f0: int, f1: int):
    """Downscaled grayscale frames f0..f1 and their not-finger masks."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")
    kernel = np.ones((3, 3), np.uint8)
    lower, upper = np.asarray(DEFAULT_HSV_LOWER), np.asarray(DEFAULT_HSV_UPPER)
    grays, masks = [], []
    try:
        # Seeking lands exactly on these files; grab() covers a short landing.
        at = 0
        if f0 > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
            at = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        while at < f0:
            if not cap.grab():
                break
            at += 1
        for _ in range(f0, f1 + 1):
            ok, frame = cap.read()
            if not ok:
                break
            gray, valid, _finger = _prepare_frame(
                frame, DOWNSCALE_WIDTH, lower, upper, kernel
            )
            grays.append(gray)
            masks.append(valid)
    finally:
        cap.release()
    return grays, masks


def _frame_motion(grays, masks) -> np.ndarray:
    """p90 sparse-LK feature displacement (px) from the previous frame; NaN for
    frame 0 and for pairs without enough trackable features."""
    import cv2

    out = np.full(len(grays), np.nan)
    for i in range(1, len(grays)):
        a, b = grays[i - 1], grays[i]
        mask = (masks[i - 1] & masks[i]).astype(np.uint8) * 255
        p0 = cv2.goodFeaturesToTrack(
            a, maxCorners=MAX_CORNERS, qualityLevel=0.01, minDistance=5, mask=mask
        )
        if p0 is None or len(p0) < MIN_FEATURES:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            a, b, p0, None, winSize=(21, 21), maxLevel=3
        )
        p0r, _, _ = cv2.calcOpticalFlowPyrLK(
            b, a, p1, None, winSize=(21, 21), maxLevel=3
        )
        fb_err = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
        good = (st.ravel() == 1) & (fb_err < FB_MAX_PX)
        if good.sum() < MIN_FEATURES:
            continue
        disp = np.linalg.norm((p1 - p0).reshape(-1, 2)[good], axis=1)
        out[i] = float(np.percentile(disp, FLOW_PERCENTILE))
    return out


def _camera_gain(
    video_path: str,
    cam_ts: np.ndarray,
    pose_ts: np.ndarray,
    X: np.ndarray,
    t_out: float,
    t_back: float,
    pad_s: float,
) -> Dict[str, Any]:
    """One camera's evidence for one event: gain ratio plus its ingredients,
    or a skip_reason when the camera cannot judge."""
    f0 = max(int(np.searchsorted(cam_ts, t_out - pad_s)) - 1, 0)
    f1 = min(int(np.searchsorted(cam_ts, t_back + pad_s)), len(cam_ts) - 1)
    if f1 - f0 < 3:
        return {"frames": 0, "skip_reason": "event outside the video"}
    grays, masks = _decode_window(video_path, f0, f1)
    flow = _frame_motion(grays, masks)
    wt = cam_ts[f0 : f0 + len(grays)]
    result: Dict[str, Any] = {"frames": len(grays)}
    if len(grays) < 4:
        result["skip_reason"] = "too few frames decoded"
        return result

    # Tracker displacement (mm) between consecutive camera frame times.
    P = np.stack([np.interp(wt, pose_ts, X[:, k]) for k in range(3)], axis=1)
    steps = np.zeros(len(wt))
    steps[1:] = np.linalg.norm(np.diff(P, axis=0), axis=1) * 1e3

    edge = (np.abs(wt - t_out) <= EDGE_TOL_S) | (np.abs(wt - t_back) <= EDGE_TOL_S)
    context = (wt < t_out - EDGE_TOL_S) | (wt > t_back + EDGE_TOL_S)
    context[0] = False  # frame 0 has no previous frame to move from
    edge[0] = False

    tracked = np.isfinite(flow[1:])
    f_edge = flow[edge & np.isfinite(flow)]
    f_ctx = flow[context & np.isfinite(flow)]
    if tracked.mean() < MIN_TRACKED_FRAC or not len(f_edge) or not len(f_ctx):
        result["skip_reason"] = "too little texture to track"
        return result

    flow_edge = float(f_edge.max())
    flow_ctx = float(np.median(f_ctx))
    trk_edge = float(steps[edge].max())
    trk_ctx = float(np.median(steps[context]))
    gain_edge = flow_edge / max(trk_edge, 1e-3)
    gain_ctx = flow_ctx / max(trk_ctx, TRK_CTX_FLOOR_MM)
    result.update(
        gain_ratio=_round(gain_edge / max(gain_ctx, 1e-3), 3),
        flow_edge_px=_round(flow_edge, 2),
        flow_ctx_px=_round(flow_ctx, 2),
        trk_edge_mm=_round(trk_edge, 1),
        trk_ctx_mm=_round(trk_ctx, 2),
    )
    if result["gain_ratio"] is None:
        result["skip_reason"] = "non-finite gain"
    return result


def confirm_excursions(
    episode_data,
    data_dir: str,
    episode_name: str,
    tracking_device: str,
    events: list,
    timestamps,
    params: Optional[Dict[str, Any]] = None,
    deadline: Optional[float] = None,
) -> list:
    """Judge each "candidate" excursion against the same-side cameras.

    Returns new event dicts (the input is not mutated). "rejected" events pass
    through unchanged. Each candidate becomes "confirmed", "rejected" (reason
    "video") or "unverified", with ``confirm = {method, score, cameras}``:
    score is the largest usable gain ratio, None when no camera could judge.
    Any usable camera at or above ``reject_min`` rejects; all under
    ``confirm_max`` confirm; anything else, including no usable camera, is
    unverified. Never raises.

    Args:
        episode_data: Zarr group of a raw episode
        data_dir: Recordings dir holding ``{episode_name}/{camera}.mp4``
        episode_name: Episode key; empty when ``data_dir`` is the episode dir
        tracking_device: The pose device the events were measured on
        events: Events from ``check_tracking_excursions``
        timestamps: The pose timestamps the events were measured on
        params: Overrides for ``DEFAULT_PARAMS``
        deadline: ``time.monotonic()`` after which no further event is
            decoded (left "unverified"); None for no limit. The caller shares
            one deadline across an episode's devices.
    """
    out = [dict(e) for e in events or []]
    candidates = [e for e in out if e.get("state") == "candidate"]
    if not candidates:
        return out
    try:
        p = _sanitise_params(params)
        thresholds = (p["confirm_max"], p["reject_min"])
        pad_s, max_events = p["pad_s"], p["max_events"]
        cameras, X, pose_ts, why = _witnesses(
            episode_data, data_dir, episode_name, tracking_device
        )
    except Exception as exc:  # noqa: BLE001 - never fail the data check
        logger.warning(
            "%s: excursion confirmation setup failed: %s", tracking_device, exc
        )
        for e in candidates:
            _unverified(e, f"setup failed: {type(exc).__name__}: {exc}")
        return out
    if not cameras:
        for e in candidates:
            _unverified(e, why)
        return out

    for k, e in enumerate(candidates):
        if k >= max_events:
            _unverified(e, f"over the budget of {max_events} decoded events")
            continue
        try:
            if deadline is not None and time.monotonic() >= deadline:
                _unverified(e, "over the time budget")
                continue
            _judge(e, cameras, X, pose_ts, timestamps, pad_s, *thresholds)
        except Exception as exc:  # noqa: BLE001 - never fail the data check
            logger.warning(
                "%s: excursion confirmation failed: %s", tracking_device, exc
            )
            _unverified(e, f"failed: {type(exc).__name__}: {exc}")
    return out


def _witnesses(episode_data, data_dir, episode_name, tracking_device):
    """({camera: (video path, clock)}, tracker positions, tracker clock, why
    there is no camera), with pose and cameras on the latency-adjusted clock
    alignment uses."""
    if "valid_timestamps" in episode_data.array_keys():
        # Aligned episodes drop the iPhone video and resample the rest.
        return {}, None, None, "aligned episode"
    latencies = dict(episode_data.attrs.get("latency", {}))
    X = np.asarray(episode_data[tracking_device]["pose"][:], dtype=np.float64)
    pose_ts = _adjusted_timestamps(episode_data, tracking_device, latencies)
    if pose_ts is None:
        return {}, None, None, "no pose timestamps"
    n = min(len(X), len(pose_ts))
    X, pose_ts = X[:n, :3, 3], pose_ts[:n]

    side = tracking_device.split("_", 1)[0]
    cameras = {}
    for cam in (f"{side}_tracking_camera", f"{side}_umi_camera"):
        path = os.path.join(data_dir, episode_name, f"{cam}.mp4")
        if cam not in episode_data or not os.path.exists(path):
            continue
        cam_ts = _adjusted_timestamps(episode_data, cam, latencies)
        if cam_ts is not None and len(cam_ts) > 0:
            cameras[cam] = (path, cam_ts)
    return cameras, X, pose_ts, "no witness camera"


def _judge(e, cameras, X, pose_ts, timestamps, pad_s, confirm_max, reject_min):
    """Set one candidate's state, reason and confirm block in place."""
    n = len(pose_ts)
    # Edges on the pose rows, so pose and cameras share the adjusted clock.
    i0, i1 = e.get("i0"), e.get("i1")
    if i0 is None or i1 is None:
        ts = np.asarray(timestamps, dtype=np.float64)
        i0 = int(np.searchsorted(ts, e["t_out_s"]))
        i1 = int(np.searchsorted(ts, e["t_back_s"]))
    if not 0 < i0 <= i1 < n - 1:
        _unverified(e, "event outside the pose stream")
        return
    t_out = float(pose_ts[i0])
    t_back = float(pose_ts[i1 + 1])  # the back edge is the step after the last row
    per_cam = {
        cam: _camera_gain(path, cam_ts, pose_ts, X, t_out, t_back, pad_s)
        for cam, (path, cam_ts) in cameras.items()
    }
    gains = [c["gain_ratio"] for c in per_cam.values() if "skip_reason" not in c]
    if not gains:
        _unverified(e, "no usable camera", per_cam)
        return
    score = max(gains)
    if score >= reject_min:
        e["state"], e["reason"] = "rejected", "video"
    elif score < confirm_max:
        e["state"], e["reason"] = "confirmed", None
    else:
        e["state"], e["reason"] = "unverified", None
    e["confirm"] = {"method": CONFIRM_METHOD, "score": score, "cameras": per_cam}

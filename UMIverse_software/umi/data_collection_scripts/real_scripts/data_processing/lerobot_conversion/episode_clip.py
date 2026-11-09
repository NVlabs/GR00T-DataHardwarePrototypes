# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Episode-clipping rules shared by conversion outputs.
"""Per-episode start/end clip window for UMI gripper episodes (rule "S_g5+E_b0.5").

Drops the meaningless frames around the task -- idle, pedal-press drift/jiggle before the
first reach, and the retraction/settling after the last release -- and never touches
anything between the first grasp and the last release. Motion and gripper-width
thresholds are defined in ``ClipParams``. Without a detected gripper event, the
window follows the first and last activity; an inactive episode is kept whole.

``compute_episode_clip`` is a pure NumPy function with no I/O. The sibling
``umi_lerobot_conversion.py`` calls it when writing ``meta/episode_clips.jsonl``.
Windows use zero-based frame indices with an exclusive ``clip_end``; the full
episode data and videos remain intact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

RULE_NAME = "S_g5+E_b0.5"
RULE_VERSION = 1

# Distance-to-target signals are smoothed more than speeds (±3 frames): they gate on
# 1.5-3 cm changes, which single-frame tracker noise would otherwise trigger.
_DIST_SMOOTH_HALF = 3
# Gripper rest widths (episode start/end, and the local plateau around an event) are
# medians over 0.3 s windows.
_GRIPPER_REST_S = 0.3
# Walking back to a grasp onset by speed stops once the width is within 2 mm of its rest
# value, so slow pre-grasp drift is not counted as part of the event.
_GRIPPER_SETTLE_MM = 2.0
# A frame is on a plateau when its width is within 1 mm of the median of the adjacent 0.3 s.
_GRIPPER_PLATEAU_MM = 1.0


@dataclass(frozen=True)
class ClipParams:
    # activity thresholds (idle detection)
    active_speed_mps: float = 0.03
    active_rot_radps: float = 0.3
    active_gripper_mps: float = 0.005
    smooth_half_window: int = 2
    idle_margin_s: float = 0.2
    # gripper events
    gripper_event_min_change_m: float = 0.010
    # progress-gated start (S_e3) + jitter guard (g5)
    progress_drop_m: float = 0.03
    progress_hold_m: float = 0.015
    pause_speed_mps: float = 0.03
    pause_min_s: float = 0.17
    start_margin_s: float = 0.2
    jitter_max_translation_m: float = 0.05
    jitter_max_rotation_rad: float = 0.436
    start_cap_before_grasp_s: float = 0.3
    # end
    end_margin_after_release_s: float = 0.5
    end_floor_after_release_s: float = 0.3
    min_keep_frames: int = 2


@dataclass(frozen=True)
class EpisodeClip:
    clip_start: int
    clip_end: int
    num_frames: int
    start_rule: str
    end_rule: str
    first_grasp_frame: int | None
    last_release_frame: int | None
    last_release_end_frame: int | None

    def to_json(self) -> dict:
        return asdict(self)


def _window(n, half):
    idx = np.arange(n)
    return np.clip(idx - half, 0, n - 1), np.clip(idx + half, 0, n - 1)


def _rate(t, x, half):
    """Centred difference over ±half frames divided by the real dt (timestamps jitter 22/44 ms)."""
    a, b = _window(len(t), half)
    dt = t[b] - t[a]
    dt = np.where(dt > 0, dt, np.nan).reshape(-1, *([1] * (x.ndim - 1)))
    return np.nan_to_num((x[b] - x[a]) / dt)


def _smooth(x, half):
    """Box filter over ±half frames along axis 0, edge-truncated."""
    n = len(x)
    c = np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:]), x]), axis=0)
    a = np.clip(np.arange(n) - half, 0, n)
    b = np.clip(np.arange(n) + half + 1, 0, n)
    return (c[b] - c[a]) / (b - a).reshape(-1, *([1] * (x.ndim - 1)))


def _rot6d_to_matrix(r6):
    """LeRobot rot6d = first two rows of R; Gram-Schmidt them and complete with the cross product."""
    r1 = r6[:, :3] / np.linalg.norm(r6[:, :3], axis=1, keepdims=True)
    r2 = r6[:, 3:] - np.sum(r1 * r6[:, 3:], axis=1, keepdims=True) * r1
    r2 = r2 / np.linalg.norm(r2, axis=1, keepdims=True)
    return np.stack([r1, r2, np.cross(r1, r2)], axis=1)


def _angle(Ra, Rb):
    """Geodesic angle between rotation stacks (…, 3, 3)."""
    tr = np.einsum("...ij,...ij->...", Ra, Rb)  # trace(Ra^T Rb)
    return np.arccos(np.clip((tr - 1) / 2, -1, 1))


def _gripper_events(t, g_mm, p):
    """(first event onset, last event onset, last event completion) frame indices, or None if
    the gripper is unused.

    An event is the width moving >= gripper_event_min_change_m from its rest value. Its onset
    (completion) is the earlier (later) of two estimates, so a slow ramp is never cut short:
    the last (first) frame still on the plateau before (after) the event, and the frame where
    the width speed drops below active_gripper_mps walking back (forward).
    """
    delta = p.gripper_event_min_change_m * 1000
    vmin = p.active_gripper_mps * 1000
    g = _smooth(g_mm, p.smooth_half_window)
    gv = np.abs(_rate(t, g, p.smooth_half_window))
    n = len(g)
    g0 = np.median(g[t < t[0] + _GRIPPER_REST_S])
    ge = np.median(g[t > t[-1] - _GRIPPER_REST_S])

    def on_plateau(k, before):
        # width within _GRIPPER_PLATEAU_MM of the median of the 0.3 s before / after frame k
        if before:
            w = g[np.searchsorted(t, t[k] - _GRIPPER_REST_S) : k + 1]
        else:
            w = g[k : np.searchsorted(t, t[k] + _GRIPPER_REST_S, "right")]
        return abs(g[k] - np.median(w)) <= _GRIPPER_PLATEAU_MM

    def onset(i, guard):
        j = i
        while (
            j > 0
            and gv[j - 1] > vmin
            and (guard is None or abs(g[j - 1] - guard) > _GRIPPER_SETTLE_MM)
        ):
            j -= 1
        k = i - 1
        while k > 0 and not on_plateau(k, before=True):
            k -= 1
        return max(0, min(j, k))

    away = np.abs(g - g0) > delta
    if not away.any():
        return None
    i0 = onset(int(np.argmax(away)), guard=g0)
    # the last event is the last time the width is still far from its final rest value
    not_final = np.abs(g - ge) > delta
    if not not_final.any():
        return i0, i0, i0
    il = n - 1 - int(np.argmax(not_final[::-1]))
    i1 = onset(il, guard=None)
    j = il
    while j < n - 1 and gv[j] > vmin:
        j += 1
    k = il + 1
    while k < n - 1 and not on_plateau(k, before=False):
        k += 1
    i2 = min(n - 1, max(j, k))
    return i0, max(i0, i1), max(i0, i2)


def _active(t, arms, grippers, p):
    h = p.smooth_half_window
    a, b = _window(len(t), h)
    dt = np.maximum(t[b] - t[a], 1e-3)
    act = np.zeros(len(t), bool)
    for pos, R in arms.values():
        v = np.linalg.norm(_rate(t, pos, h), axis=1)
        w = np.nan_to_num(_angle(R[a], R[b])) / dt
        act |= (v > p.active_speed_mps) | (w > p.active_rot_radps)
    for g in grippers.values():
        act |= np.abs(_rate(t, g / 1000, h)) > p.active_gripper_mps
    return act


def _progress_start(t, ps, sp, i_a, iw, target_idx, n_pause, p):
    """Time the arm starts making lasting progress toward its first-grasp pose, or None.

    Progress = distance to the target falls progress_drop_m below its running max (since the idle
    cut) and never climbs back above that max - progress_hold_m. The start is the end of the last
    pause (>= pause_min_s below pause_speed_mps) before that point.
    """
    d = np.linalg.norm(ps[i_a : iw + 1] - ps[target_idx], axis=1)
    run = np.maximum.accumulate(d)
    suffix_max = np.maximum.accumulate(d[::-1])[::-1]
    hit = np.nonzero(
        (d < run - p.progress_drop_m) & (suffix_max <= run - p.progress_hold_m)
    )[0]
    if not len(hit):
        return None
    ip = i_a + int(hit[0])
    slow = sp[i_a : ip + 1] < p.pause_speed_mps
    # length of the slow run ending at each frame
    runlen = np.zeros(len(slow), int)
    for k, s in enumerate(slow):
        runlen[k] = runlen[k - 1] + 1 if s and k else int(s)
    ends = np.nonzero(runlen >= n_pause)[0]
    return t[i_a + int(ends[-1])] if len(ends) else t[i_a]


def compute_episode_clip(
    timestamps: np.ndarray,
    arms: dict[str, np.ndarray],
    grippers: dict[str, np.ndarray],
    params: ClipParams = ClipParams(),
) -> EpisodeClip:
    p = params
    t = np.asarray(timestamps, dtype=float)
    n = len(t)
    keep_all = EpisodeClip(0, n, n, "none", "none", None, None, None)
    if n < 2:
        return keep_all

    arm_sig = {}
    for name, eef in arms.items():
        eef = np.asarray(eef, dtype=float)
        arm_sig[name] = (eef[:, :3], _rot6d_to_matrix(eef[:, 3:9]))
    grip_sig = {k: np.asarray(v, dtype=float).reshape(n) for k, v in grippers.items()}

    act = _active(t, arm_sig, grip_sig, p)
    if not act.any():
        return keep_all
    first_act = int(np.argmax(act))
    last_act = n - 1 - int(np.argmax(act[::-1]))

    # idle cut: the floor for any start
    t_idle = max(t[0], t[first_act] - p.idle_margin_s)
    i_idle = int(np.searchsorted(t, t_idle))

    events = {
        k: ev
        for k, g in grip_sig.items()
        if (ev := _gripper_events(t, g, p)) is not None
    }
    if not events:
        end = int(np.searchsorted(t, t[last_act] + p.idle_margin_s))
        return _finish(i_idle, end, n, "idle", "idle", None, None, None, p)

    first_grasp = min(ev[0] for ev in events.values())
    # the last event is the one that completes last; its onset is reported alongside
    _, last_release, release_end = max(events.values(), key=lambda ev: ev[2])
    first_arm = min(events, key=lambda k: events[k][0])

    # progress-gated candidate (S_e3): earliest arm's last pause before its lasting approach
    n_pause = max(1, int(round(p.pause_min_s * (n - 1) / (t[-1] - t[0]))))
    h = p.smooth_half_window
    smoothed = {k: _smooth(pos, _DIST_SMOOTH_HALF) for k, (pos, _) in arm_sig.items()}
    starts = []
    for k, (ig, _, _) in events.items():
        if k not in smoothed or ig <= i_idle:
            continue
        # the second arm only counts progress made before the episode's first grasp
        iw = ig if k == first_arm else min(ig, first_grasp)
        if iw <= i_idle:
            continue
        ps = smoothed[k]
        sp = np.linalg.norm(_rate(t, ps, h), axis=1)
        s = _progress_start(t, ps, sp, i_idle, iw, ig, n_pause, p)
        if s is None and k == first_arm:
            s = t[max(i_idle, ig - 1)]  # target already close at the idle cut: no gate
        if s is not None:
            starts.append(s)

    # cap: never closer than start_cap_before_grasp_s to the first grasp (wins over the idle floor)
    cap = max(
        0,
        int(np.searchsorted(t, t[first_grasp] - p.start_cap_before_grasp_s, "right"))
        - 1,
    )
    start, start_rule = i_idle, "idle"
    if starts:
        cand = int(np.searchsorted(t, min(starts) - p.start_margin_s))
        cand = min(max(i_idle, cand), cap)
        # jitter guard (g5): only clip beyond the idle cut if no arm really moved in that span
        jitter = all(
            np.linalg.norm(smoothed[k][cand] - smoothed[k][i_idle])
            <= p.jitter_max_translation_m
            and _angle(R[i_idle], R[cand]) <= p.jitter_max_rotation_rad
            for k, (_, R) in arm_sig.items()
        )
        if jitter and cand > i_idle:
            start, start_rule = cand, "progress_gated"
    start = min(start, cap)

    # end is measured from the release completion (width settled), not its onset
    end = int(np.searchsorted(t, t[release_end] + p.end_margin_after_release_s))
    floor = int(np.searchsorted(t, t[release_end] + p.end_floor_after_release_s))
    return _finish(
        start,
        max(end, floor, release_end + 1),
        n,
        start_rule,
        "release",
        first_grasp,
        last_release,
        release_end,
        p,
    )


def _finish(
    start, end, n, start_rule, end_rule, first_grasp, last_release, release_end, p
):
    end = min(end, n)
    start = max(0, min(start, n - 1))
    keep = min(p.min_keep_frames, n)
    if end - start < keep:
        end = min(n, start + keep)
        start = end - keep
    return EpisodeClip(
        clip_start=int(start),
        clip_end=int(end),
        num_frames=int(n),
        start_rule=start_rule,
        end_rule=end_rule,
        first_grasp_frame=None if first_grasp is None else int(first_grasp),
        last_release_frame=None if last_release is None else int(last_release),
        last_release_end_frame=None if release_end is None else int(release_end),
    )

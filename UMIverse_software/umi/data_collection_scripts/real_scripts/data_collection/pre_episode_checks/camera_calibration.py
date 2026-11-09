# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-episode check for inter-camera relative calibration.

The recorder calibrates inline: in its CALIBRATION stage it calls
``start_session()`` and then ``tick()`` on every frame until the cameras
converge, and ``run()`` gates recording on the cached result.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from umi.common.geometry.pose_calibration import (
    compute_relative_calibration,
    compute_T_world_a_world_b,
    estimate_pose,
)
from umi.common.recording.pre_episode_checks import (
    PreEpisodeContext,
    register,
)
from umi.common.utility.data_visualizer import roi_for_tile

# BGR status colors: amber requires a board detection; green marks a calibrated camera.
AIM_COLOR = (0, 200, 255)
OK_COLOR = (0, 255, 0)

# Dim inactive camera tiles while leaving enough detail to detect a lost stream.
DIM_ALPHA = 0.6

# Inset the calibration border to preserve the camera tile's left/right identity border.
_BORDER_INSET = 8
_BORDER_THICKNESS = 8


def _pose_delta_metrics(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    """Translation (mm) and rotation (deg) between two 4x4 rigid poses."""
    T_delta = np.linalg.inv(T_a) @ T_b
    t_mm = float(np.linalg.norm(T_delta[:3, 3])) * 1000.0
    R = T_delta[:3, :3]
    cos_a = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    return t_mm, float(np.degrees(np.arccos(cos_a)))


def _draw_centered_panel(
    img,
    lines: list[tuple[str, float, tuple[int, int, int], int]],
    bg_alpha: float = 0.65,
    pad: int = 14,
    line_gap: int = 6,
) -> None:
    """Draw a centered semi-transparent panel with stacked text lines.

    `lines` is a list of (text, font_scale, color, thickness). All lines share
    a single background rectangle so the panel never fragments visually.
    """
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    metrics = [
        (text, scale, color, thickness, cv2.getTextSize(text, font, scale, thickness))
        for text, scale, color, thickness in lines
    ]
    panel_w = max(size[0][0] for *_, size in metrics) + 2 * pad
    panel_h = (
        sum(size[0][1] + size[1] for *_, size in metrics)
        + line_gap * (len(metrics) - 1)
        + 2 * pad
    )

    h, w = img.shape[:2]
    x0 = max(0, (w - panel_w) // 2)
    y0 = max(0, (h - panel_h) // 2)

    bg = img.copy()
    cv2.rectangle(bg, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(bg, bg_alpha, img, 1 - bg_alpha, 0, img)

    y = y0 + pad
    for text, scale, color, thickness, ((_, th), bl) in metrics:
        y += th
        cv2.putText(
            img, text, (x0 + pad, y), font, scale, color, thickness, cv2.LINE_AA
        )
        y += bl + line_gap


@register("camera.relative_calibration")
class CameraRelativeCalibrationCheck:
    """Inter-camera relative calibration via a ChArUco/ArUco marker.

    The recorder drives the inline ``start_session`` / ``tick`` state machine;
    ``run()`` / ``is_due()`` gate recording.
    """

    def __init__(
        self,
        reference_camera: str,
        cameras: list[str],
        run_every_n_episodes: int,
        marker: dict,
        align_t_mm: float = 5.0,
        align_r_deg: float = 0.5,
        align_stable_required: int = 5,
        loop_timeout_sec: float = 30.0,
        loop_fps: float | None = None,
    ):
        if reference_camera not in cameras:
            raise ValueError(
                f"reference_camera {reference_camera!r} must be in cameras={cameras}"
            )
        if run_every_n_episodes < 1:
            raise ValueError("run_every_n_episodes must be >= 1")
        if align_stable_required < 1:
            raise ValueError("align_stable_required must be >= 1")
        self.reference_camera = reference_camera
        self.cameras = list(cameras)
        self.run_every_n_episodes = int(run_every_n_episodes)
        self.marker = dict(marker)
        self.align_t_mm = float(align_t_mm)
        self.align_r_deg = float(align_r_deg)
        self.align_stable_required = int(align_stable_required)
        self.loop_timeout_sec = float(loop_timeout_sec)
        self.loop_fps = float(loop_fps) if loop_fps is not None else None

        # Cross-session cache (survives episodes within the refresh window).
        self._cached: dict | None = None
        self._last_calib_episode: int | None = None

        # Keep _session_start after convergence so the DONE overlay remains visible
        # until the operator advances. reset_session() clears both timing and
        # convergence state.
        self._session_start: float | None = None
        self._converged: bool = False
        self._session_iters: int = 0
        self._latest_capture: dict[str, dict] = {}
        self._latest_calib: dict[str, np.ndarray] = {}
        self._stable_count: dict[str, int] = {
            n: 0 for n in self.cameras if n != self.reference_camera
        }
        # Per-camera last-processed `capture_time`.
        self._last_capture_time: dict[str, float] = {}
        # Pause each calibration camera's stream thread so calibration is its only
        # reader. Retain the recorder manager to resume streams on reset or completion.
        self._paused_streams: set[str] = set()
        self._recorder_manager_ref = None

    # ------------------------------------------------------------------
    # Cache / due-ness
    # ------------------------------------------------------------------

    def is_due(self, episode_id: int) -> bool:
        if self._cached is None or self._last_calib_episode is None:
            return True
        return (episode_id - self._last_calib_episode) >= self.run_every_n_episodes

    # ------------------------------------------------------------------
    # Per-session state machine
    # ------------------------------------------------------------------

    def is_session_active(self) -> bool:
        """True only while actively converging. Becomes False on finalize,
        when `is_session_converged()` takes over."""
        return self._session_start is not None and not self._converged

    def is_session_converged(self) -> bool:
        """True from `_finalize_session()` until the next `reset_session()`.
        Used by `draw()` to render the post-convergence DONE overlay while
        the operator is still in the CALIBRATION stage."""
        return self._converged

    def is_session_timed_out(self) -> bool:
        if self._session_start is None or self._converged:
            return False
        return (time.monotonic() - self._session_start) >= self.loop_timeout_sec

    def start_session(self) -> None:
        """Reset per-session counters and mark the start time."""
        self._session_start = time.monotonic()
        self._converged = False
        self._session_iters = 0
        self._latest_capture = {}
        self._latest_calib = {}
        self._stable_count = {n: 0 for n in self.cameras if n != self.reference_camera}
        self._last_capture_time = {}

    def reset_session(self) -> None:
        """End the session (operator acknowledged success, abort, or timeout).
        Cache is left untouched."""
        self._resume_paused_streams()
        self._session_start = None
        self._converged = False
        self._session_iters = 0
        self._latest_capture = {}
        self._latest_calib = {}
        self._stable_count = {n: 0 for n in self.cameras if n != self.reference_camera}
        self._last_capture_time = {}

    def _pause_calibration_streams(self, ctx: PreEpisodeContext) -> None:
        """Pause the recorder stream thread for every camera in `self.cameras`
        so this check is the sole reader for the duration of the session.
        Cameras outside `self.cameras` (e.g. wrist cameras during a head-mount
        calibration) keep streaming normally.
        """
        for name in self.cameras:
            if name in self._paused_streams:
                continue
            if ctx.recorder_manager.get_source(name) is not None:
                ctx.recorder_manager.pause_source_streaming(name)
                self._paused_streams.add(name)

    def _resume_paused_streams(self) -> None:
        """Symmetric to `_pause_calibration_streams`. Safe to call multiple
        times."""
        if self._recorder_manager_ref is None or not self._paused_streams:
            self._paused_streams = set()
            return
        for name in list(self._paused_streams):
            self._recorder_manager_ref.resume_source_streaming(name)
        self._paused_streams = set()

    def tick(self, ctx: PreEpisodeContext) -> None:
        """Do ONE convergence iteration. No-op if no session is active.

        Called every visualizer frame while the operator is in the calibration
        stage. Auto-finalises the session (publishing `_cached`) the first
        iteration on which every non-reference camera has produced
        `align_stable_required` consecutive aligned T_w[ref]_w[i] estimates.
        """
        if not self.is_session_active():
            return
        # Cache the manager so reset_session() / _finalize_session() can
        # resume paused streams without needing a ctx of their own.
        self._recorder_manager_ref = ctx.recorder_manager
        # Take exclusive control of every calibration camera on the first
        # tick of the session. Subsequent ticks are no-ops.
        if not self._paused_streams:
            self._pause_calibration_streams(ctx)
        self._session_iters += 1

        ctx.latest_frames = ctx.recorder_manager.get_latest_frames()
        cur_detect: dict[str, dict] = {}
        for name in self.cameras:
            cap, _ = self._capture_one(ctx, name)
            if cap is not None:
                cur_detect[name] = cap
                self._latest_capture[name] = cap

        non_ref = [n for n in self.cameras if n != self.reference_camera]

        # 2. Update T_w[ref]_w[name] for any non-ref camera that detected the
        #    marker this tick (the formula also needs ref's current pose).
        if self.reference_camera in cur_detect:
            ref_cap = cur_detect[self.reference_camera]
            self._latest_calib[self.reference_camera] = np.eye(4)
            for name in non_ref:
                if name not in cur_detect:
                    continue
                T_new = compute_T_world_a_world_b(ref_cap, cur_detect[name])
                T_prev = self._latest_calib.get(name)
                if T_prev is None:
                    self._stable_count[name] = 0
                else:
                    t_mm, ang = _pose_delta_metrics(T_prev, T_new)
                    if t_mm <= self.align_t_mm and ang <= self.align_r_deg:
                        self._stable_count[name] += 1
                    else:
                        self._stable_count[name] = 0
                self._latest_calib[name] = T_new

        if non_ref:
            converged = all(
                self._stable_count[n] >= self.align_stable_required for n in non_ref
            )
        else:
            # Single-camera setup: one detection of the reference is enough.
            converged = self.reference_camera in self._latest_capture

        if converged:
            self._finalize_session(ctx)

    def _finalize_session(self, ctx: PreEpisodeContext) -> None:
        """Build the consolidated cache, publish it, mark session converged.

        Leaves `_session_start` set so the post-convergence DONE overlay
        keeps rendering until the operator advances out of the CALIBRATION
        stage and `reset_session()` is called.
        """
        self._cached = compute_relative_calibration(
            self.reference_camera, self._latest_capture
        )
        self._last_calib_episode = ctx.recorder_manager.episode_id
        ctx.recorder_manager.last_calibration_data = [self._cached]
        elapsed = time.monotonic() - (self._session_start or time.monotonic())
        print(
            f"[calibration] converged at episode {self._last_calib_episode} "
            f"({self._session_iters} iters in {elapsed:.1f}s)"
        )
        self._converged = True
        # Resume recorder streams after calibration; retain the DONE overlay until
        # the operator advances to INIT_POSE.
        self._resume_paused_streams()

    # ------------------------------------------------------------------
    # Pre-episode-check protocol
    # ------------------------------------------------------------------

    def run(self, ctx: PreEpisodeContext) -> tuple[bool, str]:
        """Pure gate; never blocks. Publishes the cache when ready."""
        episode_id = ctx.recorder_manager.episode_id
        if self._cached is not None and not self.is_due(episode_id):
            ctx.recorder_manager.last_calibration_data = [self._cached]
            next_run = self._last_calib_episode + self.run_every_n_episodes
            return True, f"calibration cached (next refresh @ episode {next_run})"
        return False, "calibration not ready (run the calibration stage first)"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _capture_one(self, ctx: PreEpisodeContext, name: str):
        # Select the pose projection from the camera's measured distortion model,
        # not a shared setting: optional and wrist cameras can have different lenses.
        get_source = getattr(ctx.recorder_manager, "get_source", None)
        cam = get_source(name) if get_source else None

        paused = name in self._paused_streams
        if paused:
            # Read a calibration-resolution frame while the producer is paused, then
            # republish it for the preview. The visualizer resizes tiles independently.
            if cam is None:
                return None, "no source"
            frame = cam.get_calibration_frame()
            if frame is None:
                return None, "no frame"
        else:
            frames = ctx.latest_frames.get(name)
            if not frames or frames[0] is None:
                return None, "no frame"
            frame = frames[0]
        # Skip if the camera hasn't produced a new frame since the last tick.
        capture_time = getattr(frame, "capture_time", None)
        if capture_time is not None and capture_time == self._last_capture_time.get(
            name
        ):
            return None, "stale frame"
        if capture_time is not None:
            self._last_capture_time[name] = capture_time
        T_marker_optical, vis_image = estimate_pose(
            frame, self.marker, viz=True, camera=cam
        )
        if paused:
            if vis_image is not None:
                frame.rgb = vis_image
            ctx.recorder_manager.push_frame(name, frame)
        if T_marker_optical is None:
            return None, "marker not detected"
        return {"frame": frame, "T_marker_optical": T_marker_optical}, None

    # ------------------------------------------------------------------
    # Overlay
    # ------------------------------------------------------------------

    def _tile_tag(self, name: str) -> tuple[str, tuple[int, int, int]]:
        """The tag and colour for one calibration camera's tile.

        The reference camera has no stable count of its own (the counts are of
        its pose against each other camera's), so it is tagged by role and
        goes green as soon as it has resolved the board at all.
        """
        if name == self.reference_camera:
            detected = name in self._latest_capture
            return "REF", (OK_COLOR if detected else AIM_COLOR)
        count = self._stable_count.get(name, 0)
        color = OK_COLOR if count >= self.align_stable_required else AIM_COLOR
        return f"AIM BOARD HERE  {count}/{self.align_stable_required}", color

    def _draw_tiles(self, frame, ctx: PreEpisodeContext) -> bool:
        """Dim every tile that is not a calibration camera; border and tag the
        ones that are. Returns whether it decorated anything.

        Draws nothing unless every source's tile can be located. With
        `layout: overlay` every source is blended into one pane, so there is no
        other tile to dim and dimming by ROI would black out the window; the
        centred panel is then the only signal, which the return value tells
        the caller.
        """
        order = getattr(ctx, "video_source_order", None) or []
        if not order:
            return False
        viz = ctx.cfg.get("visualization") or {}
        layout = viz.get("layout", "horizontal")
        h, w = frame.shape[:2]
        rois = [
            roi_for_tile(
                layout,
                len(order),
                i,
                w,
                h,
                grid_columns=viz.get("grid_columns"),
                grid_fill=viz.get("grid_fill", "rows"),
                labels=order,
            )
            for i in range(len(order))
        ]
        if any(roi is None for roi in rois):
            return False

        for name, (x, y, tw, th) in zip(order, rois):
            # Clip overlays to the actual composite size when it is smaller than the
            # configured layout.
            x, y = max(0, min(x, w - 1)), max(0, min(y, h - 1))
            tw, th = max(1, min(tw, w - x)), max(1, min(th, h - y))
            tile = frame[y : y + th, x : x + tw]

            if name not in self.cameras:
                cv2.addWeighted(
                    tile, 1.0 - DIM_ALPHA, np.zeros_like(tile), DIM_ALPHA, 0, tile
                )
                continue

            text, color = self._tile_tag(name)
            cv2.rectangle(
                tile,
                (_BORDER_INSET, _BORDER_INSET),
                (tw - 1 - _BORDER_INSET, th - 1 - _BORDER_INSET),
                color,
                _BORDER_THICKNESS,
            )
            self._draw_tile_tag(tile, text, color)

        return True

    @staticmethod
    def _draw_tile_tag(tile, text: str, color: tuple[int, int, int]) -> None:
        """A filled chip under the tile's name badge.

        Below the badge, not in a corner: `draw_overlay` owns the composite's
        top-left and `draw_task_display` owns the bottom strip, and a tag in
        either is a tag sitting under something else. Solid fill with near-black
        text, the treatment `_draw_camera_label` already uses, so it stays
        readable over a blown-out scene.
        """
        th_, tw_ = tile.shape[:2]
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = float(np.clip(tw_ / 700.0, 0.5, 0.9))
        thickness = 1 if scale < 0.75 else 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        pad_x, pad_y = round(scale * 14), round(scale * 10)
        box_w, box_h = text_w + pad_x * 2, text_h + baseline + pad_y * 2
        x1 = max(0, (tw_ - box_w) // 2)
        y1 = min(round(th_ * 0.16), max(0, th_ - box_h))
        x2, y2 = min(tw_, x1 + box_w), min(th_, y1 + box_h)
        cv2.rectangle(tile, (x1, y1), (x2, y2), (20, 20, 20), -1)
        cv2.rectangle(tile, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), color, -1)
        cv2.putText(
            tile,
            text,
            (x1 + pad_x, y1 + pad_y + text_h),
            font,
            scale,
            (11, 11, 11),
            thickness,
            cv2.LINE_AA,
        )

    def _panel_lines(self, episode_id: int, per_camera: bool):
        """The centred panel's lines, or None when there is nothing to say.

        `per_camera` asks for the stable counters. They are left out when the
        tiles show them: in a grid the panel lands on the composite's centre,
        which is the top edge of a middle tile, right over that tile's name
        badge. With `layout: overlay` no tile is decorated and the caller passes
        True, because the panel is then the only place to read them.
        """
        non_ref = [n for n in self.cameras if n != self.reference_camera]

        # Render when calibration refresh is due, or during inline test sessions.
        # When cached and not due, stay silent.
        if self.is_session_active():
            elapsed = time.monotonic() - (self._session_start or time.monotonic())
            remaining = max(0.0, self.loop_timeout_sec - elapsed)
            header = (
                f"[calibration] CONVERGING - {remaining:.0f}s left "
                f"(iter {self._session_iters})"
            )
            header_color = (0, 200, 255)
        elif self.is_session_converged():
            header = "[calibration] DONE - press s to continue"
            header_color = (0, 255, 0)
        elif self.is_due(episode_id):
            header = "[calibration] DUE - press s to start inline calibration"
            header_color = (0, 200, 255)
        else:
            return None

        lines: list[tuple[str, float, tuple[int, int, int], int]] = [
            (header, 0.6, header_color, 2),
        ]

        # Per-camera stable counters: only meaningful while converging.
        if per_camera and self.is_session_active():
            req = self.align_stable_required
            for name in non_ref:
                cnt = self._stable_count.get(name, 0)
                color = (0, 255, 0) if cnt >= req else (60, 60, 255)
                lines.append((f"  {name}: {cnt}/{req}", 0.5, color, 1))
            lines.append((f"  ref={self.reference_camera}", 0.45, (200, 200, 200), 1))

        return lines

    def draw(self, frame, ctx: PreEpisodeContext) -> None:
        episode_id = ctx.recorder_manager.episode_id
        if self._panel_lines(episode_id, per_camera=False) is None:
            return
        # Tiles first: the panel is the summary and must sit on top of them.
        decorated = self._draw_tiles(frame, ctx)
        _draw_centered_panel(
            frame, self._panel_lines(episode_id, per_camera=not decorated)
        )

    # Compatibility entry points

    def get_cached(self) -> dict | None:
        return self._cached


__all__ = ["CameraRelativeCalibrationCheck"]

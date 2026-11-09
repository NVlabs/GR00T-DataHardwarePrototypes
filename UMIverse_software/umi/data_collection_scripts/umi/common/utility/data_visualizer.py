# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data Visualization Utilities for Recording

Provides visualization configuration and display functionality for video and
numeric data during recording sessions.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Fallback screen dimensions when display detection is unavailable.
# Use a readable size that still fits a typical laptop panel.
DEFAULT_SCREEN_SIZE: Tuple[int, int] = (1920, 1080)

# Fraction of the detected screen the preview window may occupy. Leaves room
# for the terminal the operator drives recording from.
DEFAULT_SCREEN_FRACTION: float = 0.75

# Apply a readability minimum, but never enlarge the automatic window
# beyond the detected screen.
MIN_WINDOW_SIZE: Tuple[int, int] = (640, 360)

# Use shared BGR side accents across browser and native previews, distinct
# from green/amber/red status colors.
_SIDE_ACCENTS = (
    ("left", (255, 161, 78)),  # #4EA1FF blue
    ("right", (69, 159, 255)),  # #FF9F45 orange
    ("head", (255, 125, 199)),  # #C77DFF violet
    ("bottom", (182, 196, 46)),  # #2EC4B6 teal
)
# Anything that is not a left/right/head/bottom camera (wrist, static, extras).
_DEFAULT_ACCENT = (0, 185, 118)  # #76B900 primary green

# Error-overlay line colors (BGR) for data-check issue tags, see
# real_scripts/data_processing/data_check/operator_messages.py.
_ISSUE_TAG_COLORS = {
    "[MOTION]": (0, 220, 255),  # yellow: redo the episode
    "[HARDWARE]": (255, 160, 80),  # light blue: check the device
}


def _side_accent(label: str) -> tuple:
    """Accent color for a camera name, by the side it belongs to."""
    name = label.lower()
    for side, color in _SIDE_ACCENTS:
        if side in name:
            return color
    return _DEFAULT_ACCENT


def _screen_size_from_screeninfo() -> Tuple[int, int]:
    """Screen size via the optional ``screeninfo`` package (multi-monitor aware)."""
    from screeninfo import get_monitors

    monitors = get_monitors()
    primary = next((m for m in monitors if getattr(m, "is_primary", False)), None)
    monitor = primary if primary is not None else monitors[0]
    return int(monitor.width), int(monitor.height)


def _screen_size_from_tkinter() -> Tuple[int, int]:
    """Screen size via tkinter. Raises TclError when there is no display."""
    import tkinter

    root = tkinter.Tk()
    try:
        root.withdraw()
        return int(root.winfo_screenwidth()), int(root.winfo_screenheight())
    finally:
        root.destroy()


# Tried in order; the first one that returns a positive size wins.
_SCREEN_SIZE_BACKENDS = (_screen_size_from_screeninfo, _screen_size_from_tkinter)


def detect_screen_size(
    default: Tuple[int, int] = DEFAULT_SCREEN_SIZE,
) -> Tuple[int, int]:
    """Return the primary monitor size in pixels, or ``default`` if unknown.

    Never raises: recording must keep working headless (CI, ssh session with no
    X display), where every detection backend fails. It does warn, though — a
    silent fallback to 1920x1080 on a 4K station looks exactly like "the
    preview sizing is broken", with nothing to tell the two apart.
    """
    failures = []
    for backend in _SCREEN_SIZE_BACKENDS:
        name = getattr(backend, "__name__", repr(backend))
        try:
            width, height = backend()
        except Exception as exc:
            failures.append(f"{name} failed ({type(exc).__name__}: {exc})")
            continue
        if width and height and width > 0 and height > 0:
            return int(width), int(height)
        failures.append(f"{name} returned an unusable size {(width, height)}")

    logger.warning(
        "Could not detect the screen size (%s); assuming %dx%d for the preview "
        "window. Install the 'screeninfo' package (and the system python3-tk) "
        "to size the window to the real monitor.",
        "; ".join(failures) or "no detection backends configured",
        default[0],
        default[1],
    )
    return default


def compute_window_size(
    frame_size: Optional[Tuple[int, int]],
    screen_size: Optional[Tuple[int, int]] = None,
    fraction: float = DEFAULT_SCREEN_FRACTION,
    min_size: Tuple[int, int] = MIN_WINDOW_SIZE,
    override: Tuple[Optional[int], Optional[int]] = (None, None),
) -> Optional[Tuple[int, int]]:
    """Pick an initial preview-window size for a composed frame.

    The composed frame is often a wide strip (e.g. 1280x400 for two side-by-side
    camera feeds), which OpenCV would otherwise show at 1:1 pixel size — tiny on
    a 4K monitor. Scale it to a fraction of the detected screen instead, keeping
    the frame's aspect ratio so nothing is stretched.

    Args:
        frame_size: ``(width, height)`` of the composed frame, or None if unknown.
        screen_size: ``(width, height)`` of the monitor. Detected when None.
        fraction: Share of the screen the window may occupy.
        min_size: Lower bound on the result, itself capped by the screen.
        override: Explicit ``(width, height)`` from the config; either entry may
            be None. An override always wins over the automatic size — that is
            the operator saying the automatic choice does not suit them.

    Returns:
        ``(width, height)``, or None when there is nothing to go on (no frame
        size and no usable override), meaning "leave OpenCV's default sizing".
    """
    override_width, override_height = override
    override_width = int(override_width) if override_width else None
    override_height = int(override_height) if override_height else None

    aspect = None
    if frame_size is not None:
        frame_width, frame_height = frame_size
        if frame_width > 0 and frame_height > 0:
            aspect = frame_width / frame_height

    # An explicit config value wins outright; it is not clamped to the screen.
    if override_width and override_height:
        return override_width, override_height
    if aspect is not None:
        if override_width:
            return override_width, max(1, round(override_width / aspect))
        if override_height:
            return max(1, round(override_height * aspect)), override_height
    if aspect is None:
        return None

    if screen_size is None:
        screen_size = detect_screen_size()
    screen_width, screen_height = screen_size

    # Fit the frame into the allowed share of the screen (uniform scale).
    scale = min(
        (screen_width * fraction) / frame_width,
        (screen_height * fraction) / frame_height,
    )
    width = frame_width * scale
    height = frame_height * scale

    # Grow to the minimum, then shrink back if that overflowed the screen.
    grow = max(min_size[0] / width, min_size[1] / height, 1.0)
    width *= grow
    height *= grow
    shrink = min(screen_width / width, screen_height / height, 1.0)
    width *= shrink
    height *= shrink

    return max(1, int(round(width))), max(1, int(round(height)))


@dataclass
class VisualizationConfig:
    enabled: bool = True
    window_name: str = "Data Recording"
    show_fps: bool = True
    show_episode_id: bool = True
    layout: str = "horizontal"
    overlay_numeric: bool = True
    overlay_position: tuple = (10, 60)
    overlay_alpha: float = 0.5  # Default transparency for overlay mode
    overlay_per_source_alpha: Optional[List[float]] = None  # Per-source alpha values
    # Multiplier on overlay text font size and line spacing (numeric/pose overlay)
    text_scale: float = 1.0
    # Max numeric values per line before wrapping to a new line
    values_per_line: int = 8
    # WINDOW_NORMAL allows resizing; WINDOW_AUTOSIZE fixes the window to frame
    # dimensions.
    resizable: bool = False
    # Optional initial window size (pixels) applied once when resizable=True.
    # None leaves OpenCV's default sizing.
    window_width: Optional[int] = None
    window_height: Optional[int] = None
    # Share of the screen the window takes at launch (aspect preserved).
    # None = DEFAULT_SCREEN_FRACTION (0.75, leaving room for the terminal);
    # 1.0 = maximized to the screen, and the window is moved to the origin.
    window_fraction: Optional[float] = None
    # Fill spare grid cells in order with text, rendered arrays, or text/image
    # objects. Extra cells stay black; non-grid layouts ignore filler cells.
    filler_cells: Optional[List[Any]] = None
    # Render filler content in spare cells, a full-height side column, or
    # independent resizable windows. Side/window modes leave spare video cells black.
    filler_position: str = "cells"
    # Width (px, pre-scaling) of the side column when filler_position="side".
    side_panel_width: int = 720
    # Grid order: rows is row-major; columns fills top to bottom; sides groups
    # left/right camera-name tokens into matching columns.
    grid_fill: str = "rows"
    # Explicit column count for the video grid; None keeps ceil(sqrt(n)).
    grid_columns: Optional[int] = None


def side_columns(labels: Optional[List[str]], n: int) -> Optional[List[List[int]]]:
    """One column per arm, keyed off the ``left``/``right`` token in the name.

    Unsided cameras (``head_*``) append to the shorter column, keeping the
    grid two wide. Returns None unless both arms are present, so a unimanual
    rig falls back to the plain fill instead of half a window of black.
    """
    left: List[int] = []
    right: List[int] = []
    other: List[int] = []
    for i in range(n):
        label = labels[i] if labels and i < len(labels) else ""
        # Tokens, not substrings, so "bright_camera" is not a right camera.
        tokens = str(label).lower().split("_")
        bucket = left if "left" in tokens else right if "right" in tokens else other
        bucket.append(i)
    if not left or not right:
        return None
    for i in other:
        (left if len(left) <= len(right) else right).append(i)
    return [left, right]


def build_visualizer(
    viz_cfg: dict, default_window_name: str = "Data Recording"
) -> "DataVisualizer":
    """Build a preview visualizer from a ``visualization`` config dict.

    ``resizable`` defaults to True so the preview opens as a drag-resizable
    cv2.WINDOW_NORMAL sized to the operator's monitor. Without it, cv2 uses a
    WINDOW_AUTOSIZE window locked to the composed frame size (1280x400 on a 4K
    panel), and the YAML's ``window_width`` / ``window_height`` keys would have
    no effect.
    """
    return DataVisualizer(
        VisualizationConfig(
            enabled=viz_cfg.get("enabled", True),
            window_name=viz_cfg.get("window_name", default_window_name),
            show_fps=viz_cfg.get("show_fps", True),
            show_episode_id=viz_cfg.get("show_episode_id", True),
            layout=viz_cfg.get("layout", "horizontal"),
            overlay_numeric=viz_cfg.get("overlay_numeric", True),
            overlay_position=tuple(viz_cfg.get("overlay_position", [10, 60])),
            overlay_alpha=viz_cfg.get("overlay_alpha", 0.5),
            overlay_per_source_alpha=viz_cfg.get("overlay_per_source_alpha", None),
            # `or`-defaults also cover keys left blank in the YAML (null).
            text_scale=viz_cfg.get("text_scale") or 1.0,
            values_per_line=viz_cfg.get("values_per_line") or 8,
            resizable=viz_cfg.get("resizable", True),
            window_width=viz_cfg.get("window_width"),
            window_height=viz_cfg.get("window_height"),
            window_fraction=viz_cfg.get("window_fraction"),
            filler_cells=viz_cfg.get("filler_cells"),
            filler_position=viz_cfg.get("filler_position", "cells"),
            side_panel_width=viz_cfg.get("side_panel_width") or 720,
            grid_fill=viz_cfg.get("grid_fill", "rows"),
            grid_columns=viz_cfg.get("grid_columns"),
        )
    )


# Share normalized camera-tile dimensions with overlays so their regions
# match the composed frame.
TILE_W = 640
TILE_H = 400


def grid_shape(
    n_cameras: int,
    grid_columns: Optional[int] = None,
    grid_fill: str = "rows",
    labels: Optional[List[str]] = None,
) -> Tuple[int, int, bool, Optional[List[List[int]]]]:
    """``(cols, rows, column_major, columns)`` of the grid layout's video block.

    The one place the grid's shape is decided: `combine_frames` lays tiles out
    with it and `roi_for_tile` inverts it, so overlays always address the same
    tile the layout drew.

    ``columns`` is the per-arm grouping under ``grid_fill: "sides"``, else
    None. When set, tile order is not source order.
    """
    columns = side_columns(labels, n_cameras) if grid_fill == "sides" else None
    if columns is not None:
        return len(columns), max(len(col) for col in columns), True, columns
    cols = grid_columns or int(np.ceil(np.sqrt(max(n_cameras, 1))))
    rows = int(np.ceil(max(n_cameras, 1) / cols))
    return cols, rows, grid_fill == "columns", None


def roi_for_tile(
    layout: str,
    n_cameras: int,
    panel_idx: int,
    frame_w: int,
    frame_h: int,
    grid_columns: Optional[int] = None,
    grid_fill: str = "rows",
    labels: Optional[List[str]] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """Pixel ROI ``(x, y, w, h)`` of camera ``panel_idx`` in a composite.

    The inverse of `DataVisualizer.combine_frames`' tiling, and deliberately
    next to it: an overlay that wants to draw on one camera's tile has to
    reproduce this arithmetic, and a copy living in the overlay drifts the
    first time the layout changes.

    ``grid_columns`` / ``grid_fill`` / ``labels`` are the visualization-config
    keys deciding shape and tile order, and must be forwarded whenever the
    config sets them; ``sides`` groups by ``labels``, so omitting them
    silently falls back. With the defaults, the grid fills row by row in source
    order.

    Returns ``None`` (not the whole frame) when the caller's camera has no
    region of its own: an ``overlay`` layout blends every source into a single
    tile, and an unknown layout or an out-of-range index means the geometry is
    simply not known. A caller that dims "the other tiles" must treat None as
    "do not draw", because painting the full frame would cover the composite.
    """
    if n_cameras < 1 or panel_idx < 0 or panel_idx >= n_cameras:
        return None
    if n_cameras == 1:
        return (0, 0, frame_w, frame_h)

    layout = (layout or "horizontal").lower()

    if layout == "horizontal":
        tw = max(1, frame_w // n_cameras)
        return (panel_idx * tw, 0, tw, frame_h)

    if layout == "vertical":
        th = max(1, frame_h // n_cameras)
        return (0, panel_idx * th, frame_w, th)

    if layout == "grid_overlay":
        # np.hstack(cameras..., blended_merge) -- one extra slot at the end.
        tw = max(1, frame_w // (n_cameras + 1))
        return (panel_idx * tw, 0, tw, frame_h)

    if layout == "grid":
        cols, rows, column_major, columns = grid_shape(
            n_cameras, grid_columns, grid_fill, labels
        )
        if columns is not None:
            # Tile order is not source order here, so search rather than index.
            for col_i, col in enumerate(columns):
                if panel_idx in col:
                    row_i = col.index(panel_idx)
                    break
            else:
                return None
        elif column_major:
            col_i, row_i = divmod(panel_idx, rows)
        else:
            row_i, col_i = divmod(panel_idx, cols)
        return (col_i * TILE_W, row_i * TILE_H, TILE_W, TILE_H)

    # "overlay" and anything unrecognised: no per-camera region exists.
    return None


class DataVisualizer:
    """Handles visualization of video and numeric data during recording."""

    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.frame_count = 0
        self.start_time = time.monotonic()
        self._timer_start: Optional[float] = (
            None  # None = stopped, else monotonic start
        )
        self._timer_elapsed_at_stop: float = 0.0  # Last elapsed time when stopped
        self._shake_start: Optional[float] = None
        self._shake_duration: float = 0.0
        self._shake_intensity: float = 0.0
        self._timer_flash_until: Optional[float] = None  # green pass-flash expiry
        self._window_created: bool = False
        # filler_cells image cache: path -> decoded image (or None if unreadable)
        self._filler_image_cache: Dict[str, Optional[np.ndarray]] = {}
        self._aux_windows_created: bool = False
        # Show the recorder-provided layout action near the timer only when
        # available; hide it during recording.
        self.layout_hint: Optional[str] = None

    def flash_timer_pass(self, duration: float = 2.5) -> None:
        """Briefly outline the timer in green to signal a passing data check."""
        self._timer_flash_until = time.monotonic() + duration

    def start_timer(self) -> None:
        """Start the stopwatch (e.g. when recording starts with 's')."""
        self._timer_start = time.monotonic()

    def stop_timer(self) -> None:
        """Stop the stopwatch (e.g. when stopping/restarting with 'w' or 'a')."""
        if self._timer_start is not None:
            self._timer_elapsed_at_stop = time.monotonic() - self._timer_start
        self._timer_start = None

    def _get_timer_elapsed_seconds(self) -> float:
        """Return elapsed seconds since timer start, or 0 if stopped."""
        if self._timer_start is None:
            return 0.0
        return time.monotonic() - self._timer_start

    def reset_fps_counter(self):
        self.frame_count = 0
        self.start_time = time.monotonic()

    def get_fps(self) -> float:
        elapsed = time.monotonic() - self.start_time
        if elapsed > 0:
            return self.frame_count / elapsed
        return 0.0

    def draw_overlay(
        self,
        frame: np.ndarray,
        episode_id: int,
        is_recording: bool,
        numeric_data: Optional[Dict[str, Any]] = None,
        pose_data: Optional[Dict[str, Any]] = None,
        error_active: bool = False,
    ) -> np.ndarray:
        """Draw status overlay on video frame."""
        viz_frame = frame.copy()
        h, w = viz_frame.shape[:2]
        y_offset = 30

        elapsed = (
            self._get_timer_elapsed_seconds()
            if self._timer_start is not None
            else self._timer_elapsed_at_stop
        )
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        centisecs = int((elapsed % 1) * 100)
        timer_text = f"{mins}:{secs:02d}.{centisecs:02d}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        timer_scale = 1.3
        timer_thickness = 3
        (text_w, text_h), baseline = cv2.getTextSize(
            timer_text, font, timer_scale, timer_thickness
        )
        pad = 10
        margin_x = 12
        tx = w - text_w - pad - margin_x
        _task_bar_top = h - 55
        _gap_above_bar = 8
        baseline_y = max(
            text_h + pad + 5,
            min(h - 10, _task_bar_top - _gap_above_bar),
        )
        bg_x1 = max(0, tx - pad)
        bg_y1 = max(0, baseline_y - text_h - pad)
        bg_x2 = min(w, tx + text_w + pad)
        bg_y2 = min(h, baseline_y + baseline + pad)
        bg_overlay = viz_frame.copy()
        cv2.rectangle(bg_overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(bg_overlay, 0.5, viz_frame, 0.5, 0, viz_frame)
        timer_color = (0, 0, 255) if is_recording else (0, 255, 0)
        cv2.putText(
            viz_frame,
            timer_text,
            (tx, baseline_y),
            font,
            timer_scale,
            timer_color,
            timer_thickness,
            cv2.LINE_AA,
        )
        # Data-check result border around the timer box:
        #   red (persists while the blocking error is active),
        #   green (auto-clears) as a brief flash on a passing check.
        border_color = None
        if error_active:
            border_color = (0, 0, 255)  # red
        elif self._timer_flash_until is not None:
            if time.monotonic() < self._timer_flash_until:
                border_color = (0, 255, 0)  # green
            else:
                self._timer_flash_until = None
        if border_color is not None:
            cv2.rectangle(viz_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), border_color, 3)

        # Place layout hints on the timer baseline above the side panel; hide
        # them while their action is disabled during recording.
        if self.layout_hint and not is_recording:
            hint_scale, hint_thickness = 1.0, 2
            (hint_w, _), _ = cv2.getTextSize(
                self.layout_hint, font, hint_scale, hint_thickness
            )
            if self.config.filler_position == "side":
                cx = w - self.config.side_panel_width // 2
            else:
                cx = w // 2
            hx = max(10, min(cx - hint_w // 2, bg_x1 - 8 - hint_w))
            cv2.putText(
                viz_frame,
                self.layout_hint,
                (hx, baseline_y),
                font,
                hint_scale,
                (0, 165, 255),  # orange
                hint_thickness,
                cv2.LINE_AA,
            )

        def draw_text_with_bg(
            img, text, pos, font_scale, color, thickness, bg_alpha=0.5, padding=4
        ):
            """Draw bold text with a semi-transparent background."""
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )
            x, y = pos
            bg_x1 = max(0, x - padding)
            bg_y1 = max(0, y - text_h - padding)
            bg_x2 = x + text_w + padding
            bg_y2 = y + baseline + padding
            bg_overlay = img.copy()
            cv2.rectangle(bg_overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.addWeighted(bg_overlay, bg_alpha, img, 1 - bg_alpha, 0, img)
            cv2.putText(img, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)

        if self.config.show_episode_id:
            status = "REC" if is_recording else "IDLE"
            color = (0, 0, 255) if is_recording else (0, 255, 0)
            draw_text_with_bg(
                viz_frame,
                f"Episode: {episode_id} [{status}]",
                (10, y_offset),
                font_scale=0.7,
                color=color,
                thickness=2,
                bg_alpha=0.5,
            )
            y_offset += 30

        if self.config.show_fps:
            fps = self.get_fps()
            draw_text_with_bg(
                viz_frame,
                f"FPS: {fps:.1f}",
                (10, y_offset),
                font_scale=0.6,
                color=(255, 255, 255),
                thickness=1,
                bg_alpha=0.5,
            )
            y_offset += 25

        ts = max(0.1, float(self.config.text_scale))
        if self.config.overlay_numeric and numeric_data:
            y_offset = max(y_offset, self.config.overlay_position[1])
            for source_name, frames in numeric_data.items():
                if frames and frames[0] is not None:
                    frame_data = frames[0]

                    # TrackingFrame visualization (left/right/head poses)
                    _tracking_attrs = ("left_pose", "right_pose", "head_pose")
                    if all(hasattr(frame_data, a) for a in _tracking_attrs):
                        draw_text_with_bg(
                            viz_frame,
                            f"[{source_name}]",
                            (10, y_offset),
                            font_scale=0.55 * ts,
                            color=(0, 255, 255),
                            thickness=1,
                            bg_alpha=0.5,
                        )
                        y_offset += int(20 * ts)
                        for attr_name, label in [
                            ("left_pose", "L"),
                            ("right_pose", "R"),
                            ("head_pose", "H"),
                        ]:
                            pose = getattr(frame_data, attr_name, None)
                            if pose is not None and isinstance(pose, np.ndarray):
                                pos = pose[:3, 3]
                                from scipy.spatial.transform import Rotation

                                rpy = Rotation.from_matrix(pose[:3, :3]).as_euler(
                                    "xyz", degrees=True
                                )
                                draw_text_with_bg(
                                    viz_frame,
                                    f"  {label} pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]",
                                    (10, y_offset),
                                    font_scale=0.4 * ts,
                                    color=(220, 220, 220),
                                    thickness=1,
                                    bg_alpha=0.5,
                                )
                                y_offset += int(16 * ts)
                                draw_text_with_bg(
                                    viz_frame,
                                    f"  {label} rpy: [{rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}]",
                                    (10, y_offset),
                                    font_scale=0.4 * ts,
                                    color=(220, 220, 220),
                                    thickness=1,
                                    bg_alpha=0.5,
                                )
                                y_offset += int(16 * ts)
                        y_offset += int(5 * ts)
                        continue

                    draw_text_with_bg(
                        viz_frame,
                        f"[{source_name}]",
                        (10, y_offset),
                        font_scale=0.55 * ts,
                        color=(255, 200, 0),
                        thickness=1,
                        bg_alpha=0.5,
                    )
                    y_offset += int(20 * ts)

                    skip_attrs = {"capture_time", "receive_time", "frame_source"}
                    line_h = int(18 * ts)
                    vpl = max(1, int(self.config.values_per_line))

                    for attr_name in dir(frame_data):
                        if attr_name.startswith("_"):
                            continue
                        if attr_name in skip_attrs:
                            continue

                        try:
                            value = getattr(frame_data, attr_name)
                        except Exception:
                            continue

                        if value is None or callable(value):
                            continue

                        if isinstance(value, float):
                            text = f"  {attr_name}: {value:.3f}"
                        elif isinstance(value, (int, bool)):
                            text = f"  {attr_name}: {value}"
                        elif isinstance(value, (list, np.ndarray)):
                            arr = np.asarray(value).flatten()
                            if len(arr) <= vpl:
                                vals_str = ", ".join(f"{v:.2f}" for v in arr)
                                text = f"  {attr_name}: [{vals_str}]"
                            else:
                                draw_text_with_bg(
                                    viz_frame,
                                    f"  {attr_name}: ({len(arr)} values)",
                                    (10, y_offset),
                                    font_scale=0.45 * ts,
                                    color=(220, 220, 220),
                                    thickness=1,
                                    bg_alpha=0.5,
                                )
                                y_offset += line_h
                                for chunk_start in range(0, len(arr), vpl):
                                    chunk = arr[chunk_start : chunk_start + vpl]
                                    vals_str = ", ".join(f"{v:.2f}" for v in chunk)
                                    is_first = chunk_start == 0
                                    is_last = chunk_start + vpl >= len(arr)
                                    prefix = "[" if is_first else " "
                                    suffix = "]" if is_last else ","
                                    draw_text_with_bg(
                                        viz_frame,
                                        f"    {prefix}{vals_str}{suffix}",
                                        (10, y_offset),
                                        font_scale=0.45 * ts,
                                        color=(220, 220, 220),
                                        thickness=1,
                                        bg_alpha=0.5,
                                    )
                                    y_offset += line_h
                                continue
                        else:
                            continue

                        draw_text_with_bg(
                            viz_frame,
                            text,
                            (10, y_offset),
                            font_scale=0.45 * ts,
                            color=(220, 220, 220),
                            thickness=1,
                            bg_alpha=0.5,
                        )
                        y_offset += line_h

                    y_offset += int(5 * ts)

        if pose_data:
            for source_name, pose in pose_data.items():
                if pose is not None:
                    draw_text_with_bg(
                        viz_frame,
                        f"[{source_name} pose]",
                        (10, y_offset),
                        font_scale=0.55 * ts,
                        color=(0, 255, 255),
                        thickness=1,
                        bg_alpha=0.5,
                    )
                    y_offset += int(20 * ts)

                    if isinstance(pose, np.ndarray):
                        if pose.shape == (4, 4):
                            # 4x4 transformation matrix - extract position and rotation
                            pos = pose[:3, 3]
                            draw_text_with_bg(
                                viz_frame,
                                f"  pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]",
                                (10, y_offset),
                                font_scale=0.45 * ts,
                                color=(220, 220, 220),
                                thickness=1,
                                bg_alpha=0.5,
                            )
                            y_offset += int(18 * ts)
                            rot_diag = np.diag(pose[:3, :3])
                            draw_text_with_bg(
                                viz_frame,
                                f"  rot diag: [{rot_diag[0]:.2f}, {rot_diag[1]:.2f}, {rot_diag[2]:.2f}]",
                                (10, y_offset),
                                font_scale=0.45 * ts,
                                color=(220, 220, 220),
                                thickness=1,
                                bg_alpha=0.5,
                            )
                            y_offset += int(18 * ts)
                        elif pose.ndim == 1:
                            # 1D array (e.g., [x, y, z, qx, qy, qz, qw] or similar)
                            pose_str = ", ".join([f"{v:.3f}" for v in pose[:7]])
                            draw_text_with_bg(
                                viz_frame,
                                f"  [{pose_str}]",
                                (10, y_offset),
                                font_scale=0.45 * ts,
                                color=(220, 220, 220),
                                thickness=1,
                                bg_alpha=0.5,
                            )
                            y_offset += int(18 * ts)
                    y_offset += int(5 * ts)

        return viz_frame

    def trigger_shake(self, duration: float = 1.5, intensity: float = 20.0) -> None:
        """Start a decaying shake animation (e.g. when a blocking error occurs)."""
        self._shake_start = time.monotonic()
        self._shake_duration = duration
        self._shake_intensity = intensity

    def apply_shake(self, frame: np.ndarray) -> np.ndarray:
        """Shift the frame by a decaying sinusoidal offset. No-op when inactive."""
        if self._shake_start is None:
            return frame
        elapsed = time.monotonic() - self._shake_start
        if elapsed >= self._shake_duration:
            self._shake_start = None
            return frame
        decay = 1.0 - (elapsed / self._shake_duration)
        dx = int(self._shake_intensity * decay * np.sin(elapsed * 2 * np.pi * 12.0))
        dy = int(self._shake_intensity * decay * np.sin(elapsed * 2 * np.pi * 9.0))
        h, w = frame.shape[:2]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(frame, M, (w, h))

    def draw_error_overlay(
        self,
        frame: np.ndarray,
        title: str,
        issues: List[str],
        fix_hint: str,
        max_issues: int = 6,
    ) -> np.ndarray:
        """Draw a prominent blocking-error overlay with a pulsing red border.

        Renders a dark panel in the center of the frame containing the error
        title, a truncated issue list, and an action hint for the operator. An
        issue may span several lines ("\\n"): the first is its headline, the
        rest (e.g. what to do) are drawn indented beneath it.
        """
        viz_frame = frame.copy()
        h, w = viz_frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_scale, issue_scale, sub_scale, hint_scale = 1.3, 0.95, 0.8, 0.9
        issue_h, sub_h = 46, 38

        # Pulsing red border around the whole frame (2 Hz)
        pulse = (np.sin(time.monotonic() * 2 * np.pi * 2) + 1) / 2
        border_thickness = int(4 + 8 * pulse)
        cv2.rectangle(viz_frame, (0, 0), (w - 1, h - 1), (0, 0, 255), border_thickness)

        # Truncate issue list to avoid panel overflow
        display_issues = issues[:max_issues]
        if len(issues) > max_issues:
            display_issues = issues[: max_issues - 1] + [
                f"... and {len(issues) - max_issues + 1} more"
            ]
        entries = [issue.split("\n") for issue in display_issues]

        body_h = sum(issue_h + sub_h * (len(lines) - 1) for lines in entries)
        panel_h = 105 + body_h + 75
        panel_y1 = max(0, h // 2 - panel_h // 2)
        panel_y2 = min(h - 1, panel_y1 + panel_h)

        overlay = viz_frame.copy()
        cv2.rectangle(overlay, (0, panel_y1), (w, panel_y2), (20, 0, 0), -1)
        cv2.addWeighted(overlay, 0.88, viz_frame, 0.12, 0, viz_frame)
        cv2.rectangle(viz_frame, (2, panel_y1), (w - 3, panel_y2), (0, 0, 255), 2)

        (tw, _), _ = cv2.getTextSize(title, font, title_scale, 2)
        cv2.putText(
            viz_frame,
            title,
            (w // 2 - tw // 2, panel_y1 + 55),
            font,
            title_scale,
            (50, 50, 255),
            2,
            cv2.LINE_AA,
        )

        def fit(text: str, scale: float, x: int) -> str:
            (char_w, _), _ = cv2.getTextSize("abcdefghij", font, scale, 1)
            max_chars = max(30, (w - x - 30) * 10 // char_w)
            return text if len(text) <= max_chars else text[: max_chars - 3] + "..."

        # Color tagged issues by action: MOTION means repeat the demonstration;
        # HARDWARE means check the device. Preserve the default for untagged text.
        y = panel_y1 + 110
        for headline, *sub_lines in entries:
            color = next(
                (c for tag, c in _ISSUE_TAG_COLORS.items() if headline.startswith(tag)),
                (180, 130, 255),
            )
            cv2.putText(
                viz_frame,
                fit(f"- {headline}", issue_scale, 30),
                (30, y),
                font,
                issue_scale,
                color,
                2,
                cv2.LINE_AA,
            )
            for sub in sub_lines:
                y += sub_h
                cv2.putText(
                    viz_frame,
                    fit(sub, sub_scale, 60),
                    (60, y),
                    font,
                    sub_scale,
                    (225, 225, 225),
                    1,
                    cv2.LINE_AA,
                )
            y += issue_h

        (hw, _), _ = cv2.getTextSize(fix_hint, font, hint_scale, 2)
        cv2.putText(
            viz_frame,
            fix_hint,
            (w // 2 - hw // 2, panel_y2 - 26),
            font,
            hint_scale,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

        return viz_frame

    def _draw_camera_label(self, frame: np.ndarray, label: str) -> np.ndarray:
        """Name a camera tile: an accent border around it, a badge naming it.

        The border in the side's accent color turns the tiled preview into
        clearly bounded panes, and colour-matches the calibration page, so the
        operator can tell which pane is which side without reading anything.
        The badge sits top-center of its tile, like a pane header: the bottom
        of the composite belongs to the task bar and the recording timer, and
        the top-left corner to the status text ``draw_overlay`` puts there.
        """
        if not label:
            return frame
        h, w = frame.shape[:2]
        accent = _side_accent(label)
        # Shorten camera source names into readable tile labels while keeping
        # clear of the composite status overlay.
        words = [word for word in label.split("_") if word]
        if len(words) > 1 and words[-1].lower() in ("camera", "cam"):
            words = words[:-1]
        text = " ".join(words).upper()

        border = max(2, round(min(h, w) / 160))
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), accent, border)

        font = cv2.FONT_HERSHEY_DUPLEX
        scale = float(np.clip(w / 700.0, 0.6, 1.1))
        thickness = 1 if scale < 0.75 else 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

        pad_x, pad_y = round(scale * 16), round(scale * 12)
        margin = border + round(scale * 10)
        box_w = tw + pad_x * 2
        box_h = th + baseline + pad_y * 2
        x1, y1 = max(0, (w - box_w) // 2), margin
        x2, y2 = min(w, x1 + box_w), y1 + box_h

        # Use opaque accent labels so camera identity stays readable over bright scenes.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), -1)
        cv2.rectangle(frame, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), accent, -1)
        cv2.putText(
            frame,
            text,
            (x1 + pad_x, y1 + pad_y + th),
            font,
            scale,
            (11, 11, 11),
            thickness,
            cv2.LINE_AA,
        )
        return frame

    def combine_frames(
        self, frames: List[np.ndarray], labels: Optional[List[str]] = None
    ) -> np.ndarray:
        """Combine multiple video frames based on layout config.

        If ``labels`` is provided, each frame is annotated in its top-left
        corner with the corresponding camera name before compositing, so the
        label stays with its tile regardless of layout.
        """

        target_h = 400
        target_w = 640
        if not frames:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        if len(frames) == 1:
            frame = frames[0]
            if labels:
                frame = self._draw_camera_label(frame.copy(), labels[0])
            return frame

        resized = []
        for idx, each_frame in enumerate(frames):
            if each_frame.shape[0] > each_frame.shape[1]:
                each_frame = np.rot90(each_frame, k=-1, axes=(1, 0))
            each_frame = cv2.resize(each_frame, (target_w, target_h))
            if labels and idx < len(labels):
                each_frame = self._draw_camera_label(each_frame, labels[idx])
            resized.append(each_frame)

        if self.config.layout == "horizontal":
            return np.hstack(resized)
        elif self.config.layout == "vertical":
            return np.vstack(resized)
        elif self.config.layout == "overlay":
            return self._overlay_frames(resized)
        elif self.config.layout == "grid_overlay":
            overlay = self._overlay_frames(resized)
            return np.hstack(resized + [overlay])
        else:  # grid
            n = len(resized)
            # Shared with roi_for_tile so overlays find the same tile.
            cols, rows, column_major, columns = grid_shape(
                n, self.config.grid_columns, self.config.grid_fill, labels
            )
            fillers = self.config.filler_cells or []
            # In side/windows mode the fillers render outside the video grid,
            # so the spare cells stay black.
            external = self.config.filler_position in ("side", "windows") and (
                len(fillers) > 0
            )

            pad_count = 0  # filler_cells are consumed in visual order

            def _pad_cell() -> np.ndarray:
                nonlocal pad_count
                idx, pad_count = pad_count, pad_count + 1
                if external:
                    return np.zeros_like(resized[0])
                return self._render_filler_cell(idx, target_h, target_w)

            if columns is not None:
                # Pad per column: a flat append would push a short left
                # column's blanks into the right one.
                ordered: List[np.ndarray] = []
                for col in columns:
                    ordered.extend(resized[i] for i in col)
                    for _ in range(rows - len(col)):
                        ordered.append(_pad_cell())
                resized = ordered
            else:
                while len(resized) < rows * cols:
                    resized.append(_pad_cell())
            if column_major:
                grid_img = np.hstack(
                    [np.vstack(resized[c * rows : (c + 1) * rows]) for c in range(cols)]
                )
            else:
                grid_img = np.vstack(
                    [np.hstack(resized[r * cols : (r + 1) * cols]) for r in range(rows)]
                )
            if external and self.config.filler_position == "side":
                panel = self._render_side_panel(grid_img.shape[0])
                grid_img = np.hstack([grid_img, panel])
            return grid_img

    def _render_side_panel(self, height: int) -> np.ndarray:
        """Stack every filler cell into one full-height column (side mode).

        Splitting the video grid's height among the entries makes each cell
        taller than a video tile — the point of side mode: the operator reads
        these during collection, so they get more area than a spare cell.
        """
        fillers = self.config.filler_cells or []
        width = self.config.side_panel_width
        heights = [height // len(fillers)] * len(fillers)
        heights[-1] += height - sum(heights)
        return np.vstack(
            [self._render_filler_cell(i, h, width) for i, h in enumerate(heights)]
        )

    def _render_filler_cell(self, idx: int, h: int, w: int) -> np.ndarray:
        """Render the ``idx``-th spare grid cell from ``config.filler_cells``.

        A string entry is centered text; an ndarray entry is a pre-rendered
        panel (in-process callers, e.g. the recorder's placement grid); a dict
        entry may carry ``text``, ``image`` (path) and/or ``array`` (ndarray,
        beats ``image``), text drawn on top — plus ``title`` (read only by the
        windows mode) and ``header_lines``/``header_color`` (declared text
        lines to accent, e.g. section headers). No entry -> black.
        """
        cell = np.zeros((h, w, 3), dtype=np.uint8)
        fillers = self.config.filler_cells or []
        if idx >= len(fillers) or fillers[idx] is None:
            return cell
        spec = fillers[idx]
        if isinstance(spec, np.ndarray):
            spec = {"array": spec}
        elif isinstance(spec, str):
            spec = {"text": spec}

        array = spec.get("array")
        if isinstance(array, np.ndarray):
            # Letterbox rather than stretch: a pre-rendered panel (e.g. the
            # placement grid) keeps its aspect in a differently-shaped cell.
            fit = min(w / array.shape[1], h / array.shape[0])
            fw, fh = int(array.shape[1] * fit), int(array.shape[0] * fit)
            x0, y0 = (w - fw) // 2, (h - fh) // 2
            cell[y0 : y0 + fh, x0 : x0 + fw] = cv2.resize(array, (fw, fh))
        else:
            image_path = spec.get("image")
            if image_path:
                img = self._load_filler_image(image_path)
                if img is not None:
                    cell = cv2.resize(img, (w, h))

        text = spec.get("text")
        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Font tracks the cell height (spare grid cells are 400 tall), so
            # text in a big side-panel cell is proportionally bigger too.
            scale = 0.9 * self.config.text_scale * (h / 400.0)
            thickness = 2
            max_w = w - 20
            header_lines = set(spec.get("header_lines") or ())
            header_color = tuple(spec.get("header_color") or (100, 200, 255))
            # Wrap each instruction line to cell width while preserving its declared
            # color.
            lines = []
            for declared in str(text).split("\n"):
                color = header_color if declared in header_lines else (240, 240, 240)
                words = declared.split()
                if not words:
                    lines.append(("", color))
                    continue
                current = words[0]
                for word in words[1:]:
                    candidate = f"{current} {word}"
                    if cv2.getTextSize(candidate, font, scale, thickness)[0][0] > max_w:
                        lines.append((current, color))
                        current = word
                    else:
                        current = candidate
                lines.append((current, color))
            sizes = [cv2.getTextSize(ln, font, scale, thickness)[0] for ln, _ in lines]
            line_h = max(sz[1] for sz in sizes) + 16
            y = (h - line_h * len(lines)) // 2 + line_h
            for (ln, color), sz in zip(lines, sizes):
                x = max((w - sz[0]) // 2, 10)
                cv2.putText(
                    cell,
                    ln,
                    (x, y),
                    font,
                    scale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
                y += line_h
        return cell

    def _load_filler_image(self, path: str) -> Optional[np.ndarray]:
        # Cache successful and failed image loads to avoid per-frame disk I/O.
        # An unreadable image falls back to a plain cell.
        if path not in self._filler_image_cache:
            img = cv2.imread(os.path.expanduser(path))
            if img is None:
                logger.warning("filler_cells image not readable: %s", path)
            self._filler_image_cache[path] = img
        return self._filler_image_cache[path]

    def _overlay_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """Overlay multiple frames with transparency.

        Args:
            frames: List of frames to overlay (all same size)

        Returns:
            Combined frame with overlays applied
        """
        if not frames:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        if len(frames) == 1:
            return frames[0]

        result = frames[0].astype(np.float32)

        if self.config.overlay_per_source_alpha and len(
            self.config.overlay_per_source_alpha
        ) >= len(frames):
            alpha_values = self.config.overlay_per_source_alpha
        else:
            alpha_values = [1.0] + [self.config.overlay_alpha] * (len(frames) - 1)

        for i in range(1, len(frames)):
            alpha = alpha_values[i]
            overlay = frames[i].astype(np.float32)

            # Alpha blending: result = (1 - alpha) * result + alpha * overlay
            result = (1 - alpha) * result + alpha * overlay

        return np.clip(result, 0, 255).astype(np.uint8)

    def draw_warning_overlay(
        self,
        frame: np.ndarray,
        warnings: dict,
    ) -> np.ndarray:
        """Draw a non-blocking orange warning banner at the top of the frame.

        Used for transient issues (frame drops, frame delays) detected during
        active recording.  All recording controls remain functional.

        Args:
            frame: The video frame to draw on.
            warnings: Dict of {(source_name, warning_type): message} from
                BaseRecorder.get_active_warnings().
        """
        if not warnings:
            return frame

        viz_frame = frame.copy()
        h, w = viz_frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Pulsing orange border (1.5 Hz)
        pulse = (np.sin(time.monotonic() * 2 * np.pi * 1.5) + 1) / 2
        border_thickness = int(3 + 5 * pulse)
        cv2.rectangle(
            viz_frame, (0, 0), (w - 1, h - 1), (0, 140, 255), border_thickness
        )

        lines = []
        for (source_name, warning_type), message in warnings.items():
            type_label = "FRAME DROP" if warning_type == "frame_drop" else "FRAME DELAY"
            lines.append(f"{source_name} [{type_label}]: {message}")

        line_h = 20
        banner_h = 28 + len(lines) * line_h + 22

        overlay = viz_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 60, 120), -1)
        cv2.addWeighted(overlay, 0.88, viz_frame, 0.12, 0, viz_frame)
        cv2.rectangle(viz_frame, (0, 0), (w - 1, banner_h), (0, 140, 255), 2)

        n = len(warnings)
        title = f"WARNING: {n} source{'s' if n > 1 else ''} with data issues"
        (tw, _), _ = cv2.getTextSize(title, font, 0.62, 2)
        cv2.putText(
            viz_frame,
            title,
            (w // 2 - tw // 2, 20),
            font,
            0.62,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

        y = 34
        for line in lines:
            if len(line) > 90:
                line = line[:87] + "..."
            cv2.putText(
                viz_frame, line, (15, y), font, 0.42, (200, 200, 255), 1, cv2.LINE_AA
            )
            y += line_h

        hint = "Press A to discard episode or W to stop and save"
        (hw, _), _ = cv2.getTextSize(hint, font, 0.46, 1)
        cv2.putText(
            viz_frame,
            hint,
            (w // 2 - hw // 2, banner_h - 6),
            font,
            0.46,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )

        return viz_frame

    def draw_task_display(
        self, frame: np.ndarray, task_description: str, y_offset: int = 0
    ) -> np.ndarray:
        """Draw current task description on the frame.

        Args:
            frame: The video frame to draw on.
            task_description: The task description to display.
            y_offset: Y offset for positioning.

        Returns:
            Frame with task description overlay.
        """
        if not task_description:
            return frame

        viz_frame = frame.copy()
        h, w = viz_frame.shape[:2]

        y_pos = h - 30 + y_offset

        cv2.rectangle(viz_frame, (0, y_pos - 25), (w, y_pos + 10), (40, 40, 40), -1)

        display_text = f"Task: {task_description}"
        if len(display_text) > 80:
            display_text = display_text[:77] + "..."

        cv2.putText(
            viz_frame,
            display_text,
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (100, 200, 255),
            1,
        )

        return viz_frame

    def ensure_window(self, frame: Optional[np.ndarray] = None) -> None:
        """Create the named window once, resizable if configured.

        Public because callers that drive ``cv2.imshow`` themselves must
        still get the configured window, not the
        WINDOW_AUTOSIZE one ``imshow`` would auto-create.

        With ``resizable=True`` the window is created with ``cv2.WINDOW_NORMAL``
        so the operator can drag-resize / move it, and sized to a fraction of
        the detected monitor while keeping the composed frame's aspect ratio.
        Explicit ``window_width`` / ``window_height`` override that. Without
        ``resizable``, ``cv2.imshow`` auto-creates a WINDOW_AUTOSIZE window
        locked to the frame size.
        """
        if self._window_created or not self.config.resizable:
            return
        cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)

        frame_size = None
        if frame is not None and getattr(frame, "shape", None) is not None:
            height, width = frame.shape[:2]
            frame_size = (int(width), int(height))

        fraction = self.config.window_fraction or DEFAULT_SCREEN_FRACTION
        has_override = self.config.window_width or self.config.window_height
        if fraction >= 1.0 and not has_override:
            # Maximize the window to the monitor; let OpenCV scale content inside it
            # rather than aspect-fitting the outer window.
            screen_w, screen_h = detect_screen_size()
            cv2.resizeWindow(self.config.window_name, screen_w, screen_h)
            cv2.moveWindow(self.config.window_name, 0, 0)
        else:
            size = compute_window_size(
                frame_size,
                fraction=fraction,
                override=(self.config.window_width, self.config.window_height),
            )
            if size is not None:
                cv2.resizeWindow(self.config.window_name, size[0], size[1])
        self._window_created = True

    # Native render size of a filler window (windows mode). Same 1.6 aspect
    # as a video tile / the placement grid, so panels fill it without bars.
    AUX_WINDOW_SIZE = (960, 600)

    def _ensure_aux_windows(self) -> None:
        """One resizable window per filler entry (filler_position="windows").

        Content is static per launch, so each panel is rendered and imshow-ed
        once; the main window's waitKey pumps events for every window in this
        process, which also keeps all cv2 GUI calls on the caller's thread.
        """
        if self._aux_windows_created:
            return
        w, h = self.AUX_WINDOW_SIZE
        for i, spec in enumerate(self.config.filler_cells or []):
            title = (spec.get("title") if isinstance(spec, dict) else None) or (
                f"{self.config.window_name} panel {i + 1}"
            )
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, w, h)
            cv2.imshow(title, self._render_filler_cell(i, h, w))
        self._aux_windows_created = True

    def update_filler_cell(self, idx: int, spec) -> None:
        """Replace filler entry *idx* and refresh its display.

        cells/side modes re-read ``filler_cells`` every frame, so mutating
        the config suffices; an already-open windows-mode panel is
        re-rendered and re-shown (the ``title`` must not change, or a new
        window appears instead). Call from the thread that drives show().
        """
        fillers = list(self.config.filler_cells or [])
        while len(fillers) <= idx:
            fillers.append(None)
        fillers[idx] = spec
        self.config.filler_cells = fillers
        if self.config.filler_position == "windows" and self._aux_windows_created:
            w, h = self.AUX_WINDOW_SIZE
            title = (spec.get("title") if isinstance(spec, dict) else None) or (
                f"{self.config.window_name} panel {idx + 1}"
            )
            cv2.imshow(title, self._render_filler_cell(idx, h, w))

    def show(self, frame: np.ndarray, wait_ms: int = 1) -> int:
        """Display frame and return key press.

        Args:
            frame: The frame to display.
            wait_ms: Milliseconds to wait for key press. Higher values improve
                key capture but reduce max frame rate. Default 1ms.

        Returns:
            Key code (0-255) or 255 if no key pressed.
        """
        self.frame_count += 1
        self.ensure_window(frame)
        if self.config.filler_position == "windows":
            self._ensure_aux_windows()
        cv2.imshow(self.config.window_name, frame)
        return cv2.waitKey(wait_ms) & 0xFF

    def close(self):
        cv2.destroyAllWindows()

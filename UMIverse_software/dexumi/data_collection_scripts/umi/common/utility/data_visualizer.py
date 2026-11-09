# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data Visualization Utilities for Recording

Provides visualization configuration and display functionality for video and
numeric data during recording sessions.
"""

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

_TACTILE_FIELD = re.compile(r"tactile_(\d+)_(fx|fy|fz)_([12])")
_CAMERA_TILE_W = 640
_CAMERA_TILE_H = 400
_CAMERA_LABELS = {
    "left_tracking_camera": "Left Tracking",
    "right_tracking_camera": "Right Tracking",
    "left_umi_camera": "Left UMI (Wrist)",
    "right_umi_camera": "Right UMI (Wrist)",
    "head_umi_camera": "Head Camera",
    "head_camera": "Head Camera",
}


def bimanual_camera_layout(source_names):
    """Return canvas (width, height) and each source's tile ROI.

    Tracking is above wrist on each side; the head is centered between them.
    Missing sources leave their slots empty. Extra cameras get rows below.
    """
    w, h = _CAMERA_TILE_W, _CAMERA_TILE_H
    slots = {
        "left_tracking_camera": (0, 0, w, h),
        "left_umi_camera": (0, h, w, h),
        "head_umi_camera": (w, h // 2, w, h),
        "right_tracking_camera": (2 * w, 0, w, h),
        "right_umi_camera": (2 * w, h, w, h),
    }
    regions = {}
    extra = 0
    for name in source_names:
        if name in slots:
            regions[name] = slots[name]
        else:
            regions[name] = ((extra % 3) * w, (2 + extra // 3) * h, w, h)
            extra += 1
    return (3 * w, (2 + (extra + 2) // 3) * h), regions


def _tactile_overlay_rows(frame_data, components):
    if components is None:
        return set(), []
    fields = set()
    values = {}
    for name in dir(frame_data):
        match = _TACTILE_FIELD.fullmatch(name)
        if match is None:
            continue
        fields.add(name)
        sensor_id, component, point = match.groups()
        if component not in components:
            continue
        try:
            value = getattr(frame_data, name)
        except Exception:
            continue
        if isinstance(value, (int, float, np.number)):
            axes = values.setdefault(int(sensor_id), {})
            axes.setdefault(component, {})[point] = value

    rows = []
    for sensor_id, axes in sorted(values.items()):
        for component in components:
            if component not in axes:
                continue
            points = axes[component]
            readings = [
                f"{point}={points[point]:.3f}" if point in points else f"{point}=--"
                for point in ("1", "2")
            ]
            rows.append(f"  tactile_{sensor_id} {component}: " + "  ".join(readings))
    return fields, rows


@dataclass
class VisualizationConfig:
    enabled: bool = True
    window_name: str = "Data Recording"
    show_fps: bool = True
    show_episode_id: bool = True
    layout: str = "horizontal"
    # Collection dashboard: camera mosaic plus tactile, encoder and warning panels.
    dashboard: bool = False
    overlay_numeric: bool = True
    overlay_position: tuple = (10, 60)
    overlay_alpha: float = 0.5  # Default transparency for overlay mode
    overlay_per_source_alpha: Optional[List[float]] = None  # Per-source alpha values
    # Multiplier on overlay text font size and line spacing (numeric/pose overlay)
    text_scale: float = 1.0
    # Max numeric values per line before wrapping to a new line
    values_per_line: int = 8
    # Preview axes for tactile_<sensor>_<component>_<point> scalar channels.
    # Recorded frames retain every configured sensor's force component.
    # None preserves the generic overlay; collection opts into ("fz",).
    tactile_components: Optional[tuple] = None
    # Absolute text scale for tactile rows, independent of text_scale.
    tactile_text_scale: float = 1.5
    # WINDOW_NORMAL allows resizing; WINDOW_AUTOSIZE fixes the window to frame
    # dimensions.
    resizable: bool = False
    # Optional initial window size (pixels) applied once when resizable=True.
    # None leaves OpenCV's default sizing.
    window_width: Optional[int] = None
    window_height: Optional[int] = None

    def __post_init__(self):
        components = self.tactile_components
        if components is not None and (
            not isinstance(components, (list, tuple))
            or any(c not in ("fx", "fy", "fz") for c in components)
            or len(set(components)) != len(components)
        ):
            raise ValueError(
                "tactile_components must be a list of unique axes: fx, fy, fz"
            )
        if components is not None:
            self.tactile_components = tuple(components)
        self.tactile_text_scale = float(self.tactile_text_scale)
        if not np.isfinite(self.tactile_text_scale) or self.tactile_text_scale <= 0:
            raise ValueError("tactile_text_scale must be a finite positive number")


def _operator_error_issues(issues):
    encoder_pattern = re.compile(
        r"(left|right)_dexumi_exoskeleton_encoder\.(?:joint_angles|raw_encoder_angles): "
        r"\d+ NaN values \((J\d+: \d+(?:, J\d+: \d+)*)\)"
    )
    matches = [encoder_pattern.fullmatch(issue) for issue in issues]
    joints_by_side = {}
    for match in matches:
        if match:
            joints_by_side.setdefault(match[1], set()).update(
                int(joint) for joint in re.findall(r"J(\d+):", match[2])
            )
    rendered = []
    shown_sides = set()
    for issue, match in zip(issues, matches):
        if not match:
            rendered.append(issue)
            continue
        side = match[1]
        if side in shown_sides:
            continue
        shown_sides.add(side)
        joints = sorted(joints_by_side[side])
        numbers = ", ".join(str(joint) for joint in joints)
        if len(joints) == 1:
            rendered.append(f"{side.title()} encoder: joint {numbers} has NaN")
        else:
            rendered.append(f"{side.title()} encoder: joints {numbers} have NaN")
    return rendered


def _operator_error_issue(issue):
    if "NaN values" not in issue:
        return issue
    tactile = re.match(
        r"(left|right)_dexumi_exoskeleton_tactile\.tactile_(\d+)_(fx|fy|fz)_([12]): (.*)",
        issue,
    )
    if tactile:
        side, sensor, axis, point, detail = tactile.groups()
        return f"{side.title()} tactile {sensor} {axis.upper()} point {point}: {detail}"
    return re.sub(
        r"^(left|right)_dexumi_exoskeleton_encoder\.",
        lambda match: f"{match[1].title()} encoder ",
        issue,
    )


_HERSHEY_ASCII = str.maketrans(
    {"°": " deg", "–": "-", "—": "-", "→": "->", "±": "+/-", "×": "x"}
)


def _hershey(text):
    # OpenCV Hershey fonts support ASCII only; replace unsupported symbols
    # before measuring or drawing labels.
    return text.translate(_HERSHEY_ASCII).encode("ascii", "replace").decode("ascii")


def _wrap_error_issue(issue, width, font, scale):
    # Measure pixels rather than characters: a long source name or joint list
    # must remain readable even on the 640px fallback preview.
    lines = []
    remaining = _hershey(_operator_error_issue(issue))
    prefix = "- "
    while remaining:
        end = len(remaining)
        while (
            end > 1
            and cv2.getTextSize(prefix + remaining[:end], font, scale, 1)[0][0] > width
        ):
            end -= 1
        if end < len(remaining):
            space = remaining.rfind(" ", 0, end + 1)
            if space > 0:
                end = space
        lines.append(prefix + remaining[:end])
        remaining = remaining[end:].lstrip()
        prefix = "  "
    return lines or ["- "]


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
        self._timer_flash_until: Optional[float] = None  # green saved-flash expiry
        # Set when _ensure_window() creates the OpenCV window.
        self._window_created: bool = False
        self._warnings_active = False
        self._warning_page_start = self.start_time

    def flash_timer_saved(self, duration: float = 2.5) -> None:
        """Briefly outline the timer in green to signal a saved episode."""
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

    def dashboard_size(self, default):
        """Canvas size for a dashboard drawn into this window.

        Matches the live window so the dashboard fills it. A window smaller
        than 960 x 600 gets a uniformly larger canvas (at most 2x) that OpenCV
        scales down without distortion; a sliver that needs more is clamped
        per axis instead, so its canvas stays bounded.
        """
        width = self.config.window_width or default[0]
        height = self.config.window_height or default[1]
        if self._window_created:
            try:
                _, _, live_width, live_height = cv2.getWindowImageRect(
                    self.config.window_name
                )
                if live_width > 0 and live_height > 0:
                    width, height = live_width, live_height
            except cv2.error:
                pass
        factor = min(2.0, max(1.0, 960 / width, 600 / height))
        return max(960, round(width * factor)), max(600, round(height * factor))

    def draw_dashboard(
        self,
        camera_frame,
        episode_id,
        is_recording,
        numeric_data=None,
        task_description="",
        error_active=False,
        status_text=None,
        start_hint="S  Start recording",
        saved_data_hours=None,
        saved_episode_count=None,
        show_saved_count=False,
        warnings=None,
    ):
        from umi.common.utility.collection_dashboard import render_dashboard

        width, height = self.dashboard_size((1600, 900))
        now = time.monotonic()
        if warnings and not self._warnings_active:
            self._warning_page_start = now
        # A flickering source must not restart paging and starve later pages.
        self._warnings_active = bool(warnings)
        elapsed = (
            self._get_timer_elapsed_seconds()
            if self._timer_start is not None
            else self._timer_elapsed_at_stop
        )
        display_data = numeric_data if self.config.overlay_numeric else {}
        components = self.config.tactile_components or ()
        return render_dashboard(
            camera_frame,
            display_data,
            size=(width, height),
            episode_id=episode_id,
            is_recording=is_recording,
            elapsed=elapsed,
            fps=self.get_fps() if self.config.show_fps else None,
            show_fps=self.config.show_fps,
            components=components,
            tactile_text_scale=self.config.tactile_text_scale,
            task_description=task_description,
            error_active=error_active,
            timer_saved=(
                self._timer_flash_until is not None
                and time.monotonic() < self._timer_flash_until
            ),
            show_episode_id=self.config.show_episode_id,
            status_text=status_text,
            start_hint=start_hint,
            saved_data_hours=saved_data_hours,
            saved_episode_count=saved_episode_count,
            show_saved_count=show_saved_count,
            warnings={
                (_hershey(source), _hershey(kind)): _hershey(message)
                for (source, kind), message in (warnings or {}).items()
            },
            warning_page=int((now - self._warning_page_start) / 5),
        )

    def draw_overlay(
        self,
        frame: np.ndarray,
        episode_id: int,
        is_recording: bool,
        numeric_data: Optional[Dict[str, Any]] = None,
        pose_data: Optional[Dict[str, Any]] = None,
        error_active: bool = False,
        saved_episode_count: Optional[int] = None,
        show_saved_count: bool = False,
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
        # Border around the timer box:
        #   red (persists while the blocking error is active),
        #   green (auto-clears) as a brief flash after an episode is saved.
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
                (
                    f"Episode: {episode_id}"
                    if saved_episode_count is None and not show_saved_count
                    else f"Saved episodes: {'--' if saved_episode_count is None else saved_episode_count}"
                )
                + f" [{status}]",
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

                    tactile_fields, tactile_rows = _tactile_overlay_rows(
                        frame_data, self.config.tactile_components
                    )
                    tactile_ts = self.config.tactile_text_scale
                    draw_text_with_bg(
                        viz_frame,
                        f"[{source_name}]",
                        (10, y_offset),
                        font_scale=0.55 * ts,
                        color=(255, 200, 0),
                        thickness=1,
                        bg_alpha=0.5,
                    )
                    y_offset += int(20 * max(ts, tactile_ts if tactile_rows else ts))

                    for text in tactile_rows:
                        draw_text_with_bg(
                            viz_frame,
                            text,
                            (10, y_offset),
                            font_scale=0.45 * tactile_ts,
                            color=(220, 220, 220),
                            thickness=max(1, round(tactile_ts)),
                            bg_alpha=0.5,
                        )
                        y_offset += max(1, int(18 * tactile_ts))

                    skip_attrs = {
                        "capture_time",
                        "receive_time",
                        "frame_source",
                    } | tactile_fields
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
        max_issues: int = 8,
        page: int = 0,
    ) -> np.ndarray:
        """Draw a prominent blocking-error overlay with a pulsing red border.

        Renders wrapped issue text with at most ``max_issues`` rows per page.
        The collection loop handles [ / ] navigation; no issues are discarded.
        """
        viz_frame = frame.copy()
        h, w = viz_frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        title, fix_hint = _hershey(title), _hershey(fix_hint)

        # Pulsing red border around the whole frame (2 Hz)
        pulse = (np.sin(time.monotonic() * 2 * np.pi * 2) + 1) / 2
        border_thickness = int(4 + 8 * pulse)
        cv2.rectangle(viz_frame, (0, 0), (w - 1, h - 1), (0, 0, 255), border_thickness)

        line_h = 22
        text_width = max(1, w - 60)
        lines = [
            line
            for issue in _operator_error_issues(issues)
            for line in _wrap_error_issue(issue, text_width, font, 0.48)
        ]
        rows_per_page = max(1, min(max_issues, (h - 140) // line_h))
        page_count = max(1, (len(lines) + rows_per_page - 1) // rows_per_page)
        page %= page_count
        display_issues = lines[page * rows_per_page : (page + 1) * rows_per_page]
        panel_h = 100 + len(display_issues) * line_h + (25 if page_count > 1 else 0)
        panel_y1 = h // 2 - panel_h // 2
        panel_y2 = h // 2 + panel_h // 2

        overlay = viz_frame.copy()
        cv2.rectangle(overlay, (0, panel_y1), (w, panel_y2), (20, 0, 0), -1)
        cv2.addWeighted(overlay, 0.88, viz_frame, 0.12, 0, viz_frame)
        cv2.rectangle(viz_frame, (2, panel_y1), (w - 3, panel_y2), (0, 0, 255), 2)

        (tw, _), _ = cv2.getTextSize(title, font, 0.75, 2)
        title_scale = 0.75 * min(1.0, text_width / max(1, tw))
        (tw, _), _ = cv2.getTextSize(title, font, title_scale, 2)
        cv2.putText(
            viz_frame,
            title,
            (w // 2 - tw // 2, panel_y1 + 35),
            font,
            title_scale,
            (50, 50, 255),
            2,
            cv2.LINE_AA,
        )

        y = panel_y1 + 55
        for issue in display_issues:
            cv2.putText(
                viz_frame,
                issue,
                (30, y),
                font,
                0.48,
                (180, 130, 255),
                1,
                cv2.LINE_AA,
            )
            y += line_h

        if page_count > 1:
            cv2.putText(
                viz_frame,
                f"Page {page + 1}/{page_count}    [ previous    ] next",
                (30, panel_y2 - 40),
                font,
                0.48,
                (210, 210, 210),
                1,
                cv2.LINE_AA,
            )

        (hw, _), _ = cv2.getTextSize(fix_hint, font, 0.55, 1)
        hint_scale = 0.55 * min(1.0, text_width / max(1, hw))
        (hw, _), _ = cv2.getTextSize(fix_hint, font, hint_scale, 1)
        cv2.putText(
            viz_frame,
            fix_hint,
            (w // 2 - hw // 2, panel_y2 - 12),
            font,
            hint_scale,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )

        return viz_frame

    def _draw_camera_label(self, frame: np.ndarray, label: str) -> np.ndarray:
        """Draw a small camera-name label in the bottom-right corner of a frame.

        Bottom-right keeps the label clear of the top-left status overlay and
        the top-right hand-skeleton panels drawn by ``draw_overlay``.
        """
        if not label:
            return frame
        label = _CAMERA_LABELS.get(
            label,
            label.removesuffix("_camera")
            .replace("_", " ")
            .title()
            .replace("Umi", "UMI"),
        )
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.2 if self.config.dashboard else 0.65
        thickness = 2 if self.config.dashboard else 1
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        margin = 8
        if tw > w - 2 * margin:
            scale *= max(1, w - 2 * margin) / tw
            (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        x = w - tw - margin
        y = h - margin
        bg = frame.copy()
        cv2.rectangle(
            bg, (x - 4, y - th - 4), (x + tw + 4, y + baseline + 4), (0, 0, 0), -1
        )
        cv2.addWeighted(bg, 0.5, frame, 0.5, 0, frame)
        cv2.putText(
            frame, label, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA
        )
        return frame

    def combine_frames(
        self, frames: List[np.ndarray], labels: Optional[List[str]] = None
    ) -> np.ndarray:
        """Combine multiple video frames based on layout config.

        If ``labels`` is provided, each frame is annotated in its bottom-right
        corner with the corresponding camera name before compositing, so the
        label stays with its tile regardless of layout.
        """

        target_h = _CAMERA_TILE_H
        target_w = _CAMERA_TILE_W
        if not frames:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        bimanual = self.config.layout == "bimanual" and labels is not None
        if len(frames) == 1 and not bimanual:
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

        if bimanual:
            (width, height), regions = bimanual_camera_layout(labels)
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            for source, tile in zip(labels, resized, strict=True):
                x, y, w, h = regions[source]
                canvas[y : y + h, x : x + w] = tile
            return canvas
        elif self.config.layout == "horizontal":
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
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            while len(resized) < rows * cols:
                resized.append(np.zeros_like(resized[0]))
            grid_rows = []
            for r in range(rows):
                row_frames = resized[r * cols : (r + 1) * cols]
                grid_rows.append(np.hstack(row_frames))
            return np.vstack(grid_rows)

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
        """Draw the warning banner for the legacy camera-only preview.

        Used for transient stream issues and saved-episode quality warnings.
        All recording controls remain functional. The collection dashboard
        draws its warnings in reserved space through ``draw_dashboard``.

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
            type_label = {
                "frame_drop": "FRAME DROP",
                "frame_delay": "FRAME DELAY",
            }.get(warning_type, warning_type.replace("_", " ").upper())
            lines.append(_hershey(f"{source_name} [{type_label}]: {message}"))

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

    def draw_text_input_overlay(
        self,
        frame: np.ndarray,
        prompt: str,
        current_text: str,
        hint: str = "Enter to confirm, Esc to cancel",
    ) -> np.ndarray:
        """Draw a text input overlay on the frame.

        Args:
            frame: The video frame to draw on.
            prompt: The prompt text (e.g., "Task Description:").
            current_text: The current input text.
            hint: Hint text for user instructions.

        Returns:
            Frame with the text input overlay.
        """
        viz_frame = frame.copy()
        h, w = viz_frame.shape[:2]

        overlay = viz_frame.copy()
        cv2.rectangle(
            overlay, (20, h // 2 - 60), (w - 20, h // 2 + 60), (40, 40, 40), -1
        )
        cv2.addWeighted(overlay, 0.85, viz_frame, 0.15, 0, viz_frame)

        cv2.rectangle(
            viz_frame, (20, h // 2 - 60), (w - 20, h // 2 + 60), (100, 200, 100), 2
        )

        cv2.putText(
            viz_frame,
            prompt,
            (40, h // 2 - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 200, 100),
            2,
        )

        display_text = current_text + "_"
        cv2.putText(
            viz_frame,
            display_text,
            (40, h // 2 + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            viz_frame,
            hint,
            (40, h // 2 + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1,
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

    def _ensure_window(self) -> None:
        """Create the named window once, resizable if configured.

        With ``resizable=True`` the window is created with ``cv2.WINDOW_NORMAL``
        so the operator can drag-resize / move it; an optional initial size is
        applied. Without it, ``cv2.imshow`` auto-creates a WINDOW_AUTOSIZE window
        locked to the frame size (the original behavior).

        A dashboard is re-rendered at the window's size every frame, so its
        window lets the image fill the whole window (FREERATIO). The default
        KEEPRATIO would shrink the image area to the previous frame's aspect,
        ``getWindowImageRect`` would report that shrunken area, and the next
        frame would keep the old aspect with blank bars beside it.
        """
        if self._window_created or not self.config.resizable:
            return
        window_flags = cv2.WINDOW_NORMAL
        if self.config.dashboard:
            window_flags |= cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_FREERATIO
        cv2.namedWindow(self.config.window_name, window_flags)
        if self.config.window_width and self.config.window_height:
            cv2.resizeWindow(
                self.config.window_name,
                int(self.config.window_width),
                int(self.config.window_height),
            )
        self._window_created = True

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
        self._ensure_window()
        cv2.imshow(self.config.window_name, frame)
        return cv2.waitKey(wait_ms) & 0xFF

    def close(self):
        cv2.destroyAllWindows()

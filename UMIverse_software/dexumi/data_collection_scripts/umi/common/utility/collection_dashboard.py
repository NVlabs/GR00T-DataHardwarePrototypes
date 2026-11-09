# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure OpenCV renderer for the collection camera and tactile dashboard."""

import re
from collections import namedtuple

import cv2
import numpy as np

_FIELD = re.compile(r"tactile_(\d+)_(fx|fy|fz)_([12])")
_FINGERS = {33: "Thumb", 34: "Index", 35: "Middle", 36: "Ring", 37: "Pinky"}
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BG = (19, 17, 15)
_CARD = (31, 28, 25)
_LINE = (53, 47, 41)
_TEXT = (232, 229, 224)
_MUTED = (151, 144, 133)
_ACCENT = (200, 188, 124)
_GOOD = (115, 206, 149)
_BAD = (120, 143, 236)
_WARN = (64, 176, 236)  # amber: shown, never blocking
_TACTILE_ACTIVE = (255, 170, 80)


def _text(image, text, x, y, scale, color=_TEXT, *, width=None, right=False):
    text = str(text)
    if width is not None:
        if cv2.getTextSize(text, _FONT, scale, 1)[0][0] > width:
            while text and cv2.getTextSize(text + "...", _FONT, scale, 1)[0][0] > width:
                text = text[:-1]
            text += "..."
    text_width = cv2.getTextSize(text, _FONT, scale, 1)[0][0]
    if right:
        x -= text_width
    cv2.putText(image, text, (int(x), int(y)), _FONT, scale, color, 1, cv2.LINE_AA)


def _header_stat_width(label, value, scale):
    label_w = cv2.getTextSize(label, _FONT, 0.35 * scale, 1)[0][0]
    value_w = cv2.getTextSize(value, _FONT, 0.62 * scale, 1)[0][0]
    return label_w + round(10 * scale) + value_w


def _header_stat(image, label, value, x, baseline, scale, color=_TEXT):
    """Draw ``LABEL value`` on one header baseline; return the right edge."""
    _text(image, label, x, baseline, 0.35 * scale, _MUTED)
    value_x = x + cv2.getTextSize(label, _FONT, 0.35 * scale, 1)[0][0]
    value_x += round(10 * scale)
    _text(image, value, value_x, baseline, 0.62 * scale, color)
    return round(x + _header_stat_width(label, value, scale))


def _card(image, box, radius):
    x, y, w, h = box
    cv2.rectangle(image, (x + radius, y), (x + w - radius, y + h), _CARD, -1)
    cv2.rectangle(image, (x, y + radius), (x + w, y + h - radius), _CARD, -1)
    for cx, cy in (
        (x + radius, y + radius),
        (x + w - radius, y + radius),
        (x + radius, y + h - radius),
        (x + w - radius, y + h - radius),
    ):
        cv2.circle(image, (cx, cy), radius, _CARD, -1, cv2.LINE_AA)


def _warning_source(source):
    labels = {
        "left_tracking_camera": "Left Tracking",
        "right_tracking_camera": "Right Tracking",
        "left_umi_camera": "Left UMI (Wrist)",
        "right_umi_camera": "Right UMI (Wrist)",
        "head_umi_camera": "Head Camera",
        "head_camera": "Head Camera",
    }
    return labels.get(
        source, source.replace("_dexumi_exoskeleton", "").replace("_", " ").title()
    )


def _warning_lines(message, width, scale):
    """Wrap at most two rows; the last row is fitted by ``_text``."""
    remaining = " ".join(str(message).split())
    if cv2.getTextSize(remaining, _FONT, scale, 1)[0][0] <= width:
        return [remaining]
    end = len(remaining)
    while end > 1 and cv2.getTextSize(remaining[:end], _FONT, scale, 1)[0][0] > width:
        end -= 1
    space = remaining.rfind(" ", 0, end + 1)
    if space > 0:
        end = space
    return [remaining[:end], remaining[end:].lstrip()]


def _draw_warnings(image, box, warnings, page, scale, is_recording, error_active):
    x, y, w, h = box
    pad = max(7, round(18 * scale))
    _card(image, box, max(4, round(10 * scale)))
    items = sorted((warnings or {}).items())
    count = len(items)
    color = _WARN if count else _MUTED
    _text(
        image,
        f"WARNINGS  {count} active",
        x + pad,
        y + round(24 * scale),
        0.43 * scale,
        color,
    )
    if error_active:
        status = "Resolve error to continue"
    else:
        status = "Recording continues" if is_recording else "Not recording"
    if count:
        cv2.line(image, (x + 1, y + pad // 2), (x + 1, y + h - pad // 2), _WARN, 2)
        _text(
            image,
            status,
            x + w - pad,
            y + round(24 * scale),
            0.38 * scale,
            _MUTED,
            width=w * 0.3,
            right=True,
        )
    else:
        _text(
            image,
            "No active warnings",
            x + pad,
            y + round(62 * scale),
            0.48 * scale,
            _MUTED,
        )
        return

    # Bounded height regardless of warning count. All active warnings remain
    # available through automatic paging rather than covering other cards.
    page = int(page) % ((count + 1) // 2)
    first = page * 2
    visible = items[first : first + 2]
    if count > 2:
        _text(
            image,
            f"{first + 1}-{first + len(visible)} of {count} (cycles)",
            x + round(260 * scale),
            y + round(24 * scale),
            0.36 * scale,
            _MUTED,
            width=w * 0.32,
        )
    column_w = (w - 2 * pad - pad * (len(visible) - 1)) // len(visible)
    for index, ((source, kind), message) in enumerate(visible):
        column_x = x + pad + index * (column_w + pad)
        icon_x, icon_y = column_x + round(7 * scale), y + round(48 * scale)
        radius = max(4, round(7 * scale))
        triangle = np.array(
            [
                (icon_x, icon_y - radius),
                (icon_x - radius, icon_y + radius),
                (icon_x + radius, icon_y + radius),
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(image, triangle, _WARN, cv2.LINE_AA)
        cv2.line(image, (icon_x, icon_y - radius // 3), (icon_x, icon_y + 1), _CARD, 1)
        cv2.circle(image, (icon_x, icon_y + radius - 2), 1, _CARD, -1)
        text_x = column_x + round(24 * scale)
        text_w = column_w - round(24 * scale)
        label = {
            "frame_drop": "Frame drop",
            "frame_delay": "Frame delay",
        }.get(kind, kind.replace("_", " ").title())
        _text(
            image,
            f"{_warning_source(source)} / {label}",
            text_x,
            y + round(53 * scale),
            0.45 * scale,
            _WARN,
            width=text_w,
        )
        for row, line in enumerate(_warning_lines(message, text_w, 0.42 * scale)):
            _text(
                image,
                line,
                text_x,
                y + round((75 + row * 19) * scale),
                0.42 * scale,
                _TEXT,
                width=text_w,
            )


def _hand_data(numeric_data):
    hands = []
    for source, frames in (numeric_data or {}).items():
        frame = frames[0] if frames else None
        readings = {}
        for name in dir(frame) if frame is not None else ():
            match = _FIELD.fullmatch(name)
            if match is None:
                continue
            sensor, axis, point = match.groups()
            try:
                value = getattr(frame, name)
            except Exception:
                value = None
            readings[(int(sensor), axis, int(point))] = value
        if not readings and "tactile" not in source.lower():
            continue
        tokens = source.lower().split("_")
        side = next((side for side in ("left", "right") if side in tokens), None)
        title = (
            f"{side.upper()} HAND"
            if side
            else " ".join(
                token for token in tokens if token not in ("tactile", "exoskeleton")
            ).upper()
        )
        sensors = sorted(set(_FINGERS) | {key[0] for key in readings})
        hands.append((side, title or "TACTILE SENSOR", sensors, readings, source))
    return sorted(
        hands, key=lambda hand: (hand[0] != "left", hand[0] != "right", hand[1])
    )


def _reading(value):
    if isinstance(value, (int, float, np.number)) and np.isfinite(value):
        return f"{value:.1f}"
    return "--"


def _draw_hand(image, hand, box, components, scale, tactile_scale):
    side, title, sensors, readings, _source = hand
    x, y, w, h = box
    pad = max(7, round(18 * scale))
    _card(image, box, max(4, round(10 * scale)))
    accent = _ACCENT if side != "right" else (133, 174, 216)
    cv2.circle(
        image,
        (x + pad + 3, y + round(22 * scale)),
        max(2, round(3 * scale)),
        accent,
        -1,
        cv2.LINE_AA,
    )
    _text(
        image,
        title,
        x + pad + round(15 * scale),
        y + round(28 * scale),
        0.55 * scale,
        width=w - 2 * pad - round(70 * scale),
    )
    _text(
        image,
        "mN",
        x + w - pad,
        y + round(28 * scale),
        0.45 * scale,
        _MUTED,
        right=True,
    )

    inner_w = w - 2 * pad
    sensor_x = x + pad
    axis_x = x + pad + round(inner_w * 0.36)
    point1_right = x + pad + round(inner_w * 0.72)
    point2_right = x + w - pad
    heading_scale = min(0.43 * scale, inner_w / 620)
    table_top = 65
    headings_y = y + round((table_top - 12) * scale)
    _text(image, "Sensor", sensor_x, headings_y, heading_scale, _MUTED)
    _text(image, "Axis", axis_x, headings_y, heading_scale, _MUTED)
    _text(image, "Point 1", point1_right, headings_y, heading_scale, _MUTED, right=True)
    _text(image, "Point 2", point2_right, headings_y, heading_scale, _MUTED, right=True)
    table_y = y + round(table_top * scale)
    if not components:
        _text(
            image,
            "No force axes selected",
            sensor_x,
            table_y + round(28 * scale),
            0.48 * scale,
            _MUTED,
            width=inner_w,
        )
        return
    row_count = len(sensors) * len(components)
    row_h = min((y + h - pad - table_y) / row_count, 53 * scale)
    # The full glyph plus its baseline fits inside one row even with all axes.
    value_scale = min(0.74 * scale * tactile_scale, max(0.18, (row_h - 3) / 31))
    value_width = inner_w * 0.24
    # Fit finger labels and values from -999.9 to 9999.9 mN to the available
    # columns. Shrink out-of-range values individually rather than truncating them.
    value_scale = min(
        value_scale,
        # 1 px margin: getTextSize rounds, and a reading measured 1 px over the
        # column would be shrunk again below, unlike shorter readings.
        value_scale
        * (value_width - 1)
        / cv2.getTextSize("-000.0", _FONT, value_scale, 1)[0][0],
    )
    labels = [f"{_FINGERS.get(sensor, 'Sensor')} {sensor}" for sensor in sensors]
    label_scale = min(0.51 * scale, value_scale)
    label_width = max(
        cv2.getTextSize(text, _FONT, label_scale, 1)[0][0] for text in labels
    )
    label_scale = min(label_scale, label_scale * inner_w * 0.32 / label_width)
    for sensor_index, sensor in enumerate(sensors):
        group_y = table_y + sensor_index * len(components) * row_h
        if sensor_index:
            cv2.line(
                image,
                (sensor_x, round(group_y - 1)),
                (point2_right, round(group_y - 1)),
                _LINE,
                1,
                cv2.LINE_AA,
            )
        for axis_index, axis in enumerate(components):
            row_y = group_y + axis_index * row_h
            (__, text_h), baseline = cv2.getTextSize("0", _FONT, value_scale, 1)
            text_y = round(row_y + (row_h + text_h - baseline) / 2)
            if axis_index == 0:
                _text(
                    image,
                    labels[sensor_index],
                    sensor_x,
                    text_y,
                    label_scale,
                    width=inner_w * 0.34,
                )
            _text(
                image,
                axis.upper(),
                axis_x,
                text_y,
                min(label_scale, 0.43 * scale),
                _MUTED,
            )
            for point, right_edge in ((1, point1_right), (2, point2_right)):
                raw_value = readings.get((sensor, axis, point))
                value = _reading(raw_value)
                width = cv2.getTextSize(value, _FONT, value_scale, 1)[0][0]
                fitted_scale = min(
                    value_scale, value_scale * value_width / max(1, width)
                )
                value_color = _TEXT if value != "--" else _MUTED
                if value != "--" and raw_value > 1000:
                    value_color = _TACTILE_ACTIVE
                _text(
                    image,
                    value,
                    right_edge,
                    text_y,
                    fitted_scale,
                    value_color,
                    right=True,
                )


def _vector(value):
    if value is None:
        return []
    return np.asarray(value).reshape(-1).tolist()


def _encoder_data(numeric_data):
    encoders = []
    for source, frames in (numeric_data or {}).items():
        frame = frames[0] if frames else None
        if not hasattr(frame, "joint_angles") and "encoder" not in source.lower():
            continue
        tokens = source.lower().split("_")
        side = next((side for side in ("left", "right") if side in tokens), None)
        title = f"{side.upper()} ENCODERS" if side else source.replace("_", " ").upper()
        encoders.append(
            (side, title, source, _vector(getattr(frame, "joint_angles", None)))
        )
    return sorted(
        encoders, key=lambda item: (item[0] != "left", item[0] != "right", item[1])
    )


def _draw_encoder(image, encoder, box, scale):
    _side, title, source, current = encoder
    x, y, w, h = box
    pad = max(7, round(18 * scale))
    inner_w = w - 2 * pad
    _card(image, box, max(4, round(10 * scale)))
    _text(
        image, title, x + pad, y + round(27 * scale), 0.53 * scale, width=inner_w * 0.8
    )
    _text(
        image,
        "deg",
        x + w - pad,
        y + round(27 * scale),
        0.43 * scale,
        _MUTED,
        right=True,
    )
    count = max(15, len(current))
    summary_y = y + round(48 * scale)
    _text(
        image,
        "Live joint angles",
        x + pad,
        summary_y,
        0.42 * scale,
        _MUTED,
        width=inner_w,
    )
    headings_y = y + round(74 * scale)
    table_y = headings_y + round(10 * scale)
    rows = int(np.ceil(count / 3))
    row_h = (y + h - pad - table_y) / rows
    group_w = inner_w / 3
    value_scale = min(0.52 * scale, max(0.18, (row_h - 3) / 31), group_w / 430)
    labels = ("J", "Now")
    columns = (0.09, 0.37)
    cells = [
        [_reading(current[joint] if joint < len(current) else None)]
        for joint in range(count)
    ]
    # Shrink (never truncate) the values when a column's widest entry would
    # run into its left neighbour.
    gap = max(2, round(6 * scale))
    for column in range(1, len(labels)):
        widest = max(
            cv2.getTextSize(row[column - 1], _FONT, 1.0, 1)[0][0] for row in cells
        )
        room = (columns[column] - columns[column - 1]) * group_w - gap
        value_scale = min(value_scale, 0.97 * room / widest)
    for group in range(3):
        gx = x + pad + group * group_w
        if group:
            cv2.line(
                image,
                (round(gx - 5 * scale), headings_y - round(13 * scale)),
                (round(gx - 5 * scale), y + h - pad),
                _LINE,
                1,
                cv2.LINE_AA,
            )
        for label, position in zip(labels, columns, strict=False):
            _text(
                image,
                label,
                gx + group_w * position,
                headings_y,
                min(0.38 * scale, group_w / 480),
                _MUTED,
                right=True,
            )
    for joint in range(count):
        group, row = divmod(joint, rows)
        gx = x + pad + group * group_w
        (__, text_h), baseline = cv2.getTextSize("0", _FONT, value_scale, 1)
        baseline_y = round(table_y + row * row_h + (row_h + text_h - baseline) / 2)
        current_text = cells[joint][0]
        color = _TEXT if current_text != "--" else _MUTED
        _text(
            image,
            f"J{joint:02d}",
            gx + group_w * columns[0],
            baseline_y,
            min(0.38 * scale, value_scale),
            _MUTED,
            right=True,
        )
        _text(
            image,
            current_text,
            gx + group_w * columns[1],
            baseline_y,
            value_scale,
            color,
            right=True,
        )


_Geometry = namedtuple(
    "_Geometry",
    "scale pad gap header_h footer_h top upper_h encoder_h "
    "camera_w sidebar_x sidebar_w tactile_h hand_boxes warning_box",
)


def _geometry(width, height, hands, encoders, has_info):
    scale = min(width / 1600, height / 900)
    pad = max(4, round(20 * scale))
    gap = pad
    # One header row; every pixel it saves goes to the camera mosaic.
    header_h = round(48 * scale)
    footer_h = round(70 * scale)
    top = header_h + gap
    # Reserve the warning card even when clear, so transient warnings never
    # move or cover the camera, sensor tables, or task/keyboard footer.
    warning_h = round(104 * scale)
    warning_y = height - footer_h - gap - warning_h
    warning_box = (pad, warning_y, width - 2 * pad, warning_h)
    content_h = warning_y - gap - top
    encoder_h = round(230 * scale) if encoders else 0
    upper_h = content_h - encoder_h - gap if encoders else content_h
    horizontal_hands = len(hands) == 2 and bool(encoders)
    # Minimize sidebar width to leave room for video while preserving readable tactile
    # values.
    sidebar_w = round(width * (0.32 if horizontal_hands else 0.28))
    camera_w = width - 2 * pad - gap - sidebar_w
    sidebar_x = pad + camera_w + gap
    info_h = round(88 * scale) if has_info else 0
    tactile_h = upper_h - info_h - gap if info_h else upper_h
    hand_boxes = []
    if hands:
        card_h = (
            tactile_h
            if horizontal_hands
            else (tactile_h - gap * (len(hands) - 1)) // len(hands)
        )
        card_w = (sidebar_w - gap) // 2 if horizontal_hands else sidebar_w
        for index in range(len(hands)):
            hand_boxes.append(
                (
                    sidebar_x + index * (card_w + gap)
                    if horizontal_hands
                    else sidebar_x,
                    top if horizontal_hands else top + index * (card_h + gap),
                    card_w,
                    card_h,
                )
            )
    return _Geometry(
        scale,
        pad,
        gap,
        header_h,
        footer_h,
        top,
        upper_h,
        encoder_h,
        camera_w,
        sidebar_x,
        sidebar_w,
        tactile_h,
        hand_boxes,
        warning_box,
    )


def render_dashboard(
    camera_frame,
    numeric_data,
    *,
    size=(1600, 900),
    episode_id=0,
    is_recording=False,
    elapsed=0.0,
    fps=None,
    components=("fz",),
    tactile_text_scale=1.0,
    task_description="",
    error_active=False,
    timer_saved=False,
    show_episode_id=True,
    show_fps=True,
    status_text=None,
    start_hint="S  Start recording",
    saved_data_hours=None,
    saved_episode_count=None,
    show_saved_count=False,
    layout_info=None,
    warnings=None,
    warning_page=0,
):
    """Render a camera mosaic and tactile tables without changing source frames.

    ``size`` is ``(width, height)``; 960 x 600 or larger is recommended. Camera
    labels already drawn on the mosaic are preserved.
    Tactile readings are the device's original force values in millinewtons.
    Warnings always occupy a reserved bottom card; ``warning_page`` selects
    the visible pair when more than two warnings are active.
    """
    width, height = (int(value) for value in size)
    if layout_info is not None:
        layout_info.clear()
    if width < 1 or height < 1:
        raise ValueError("Dashboard dimensions must be positive")
    hands = _hand_data(numeric_data)
    encoders = _encoder_data(numeric_data)
    has_collection_info = saved_data_hours is not None
    geometry = _geometry(width, height, hands, encoders, has_collection_info)
    (
        scale,
        pad,
        gap,
        header_h,
        footer_h,
        top,
        upper_h,
        encoder_h,
        camera_w,
        sidebar_x,
        sidebar_w,
        tactile_h,
        hand_boxes,
        warning_box,
    ) = geometry
    canvas = np.full((height, width, 3), _BG, np.uint8)
    baseline = round(32 * scale)
    _text(canvas, "DexUMI", pad, baseline, 0.72 * scale)
    title_w = cv2.getTextSize("DexUMI", _FONT, 0.72 * scale, 1)[0][0]
    _text(
        canvas,
        "LIVE COLLECTION",
        pad + title_w + round(12 * scale),
        baseline,
        0.35 * scale,
        _MUTED,
    )

    status = (
        "Needs attention"
        if error_active
        else status_text or ("Recording" if is_recording else "Ready")
    )
    status_color = (
        (120, 143, 236)
        if error_active
        else (115, 206, 149)
        if is_recording
        else _ACCENT
    )
    status_x = round(width * 0.21)
    cv2.circle(
        canvas,
        (status_x, baseline - round(6 * scale)),
        max(2, round(4 * scale)),
        status_color,
        -1,
        cv2.LINE_AA,
    )
    _text(
        canvas,
        status,
        status_x + round(16 * scale),
        baseline,
        0.57 * scale,
        status_color,
        width=width * 0.24,
    )
    if show_episode_id:
        saved = show_saved_count or saved_episode_count is not None
        label = "SAVED EPISODES" if saved else "EPISODE"
        count = saved_episode_count if saved else episode_id
        _header_stat(
            canvas,
            label,
            "--" if count is None else f"{count:03d}",
            width * 0.55,
            baseline,
            scale,
        )
    centiseconds = max(0, int(round(elapsed * 100)))
    seconds, fraction = divmod(centiseconds, 100)
    timer = f"{seconds // 60}:{seconds % 60:02d}.{fraction:02d}"
    timer_color = _BAD if error_active or is_recording else _GOOD
    if not has_collection_info:
        timer_x = round(width * 0.71)
        timer_right = _header_stat(
            canvas, "ELAPSED", timer, timer_x, baseline, scale, timer_color
        )
        if error_active or timer_saved:
            cv2.rectangle(
                canvas,
                (timer_x - round(8 * scale), round(6 * scale)),
                (timer_right + round(8 * scale), header_h - round(6 * scale)),
                _BAD if error_active else _GOOD,
                max(1, round(2 * scale)),
                cv2.LINE_AA,
            )
    if show_fps:
        fps_text = "--" if fps is None or not np.isfinite(fps) else f"{fps:.1f}"
        _header_stat(
            canvas,
            "PREVIEW FPS",
            fps_text,
            width - pad - _header_stat_width("PREVIEW FPS", fps_text, scale),
            baseline,
            scale,
        )
    cv2.line(canvas, (pad, header_h), (width - pad, header_h), _LINE, 1, cv2.LINE_AA)

    info_h = round(88 * scale) if has_collection_info else 0
    camera_h = upper_h
    camera_box = (pad, top, camera_w, camera_h)
    _card(canvas, camera_box, max(3, round(10 * scale)))
    inset = max(2, round(8 * scale))
    if camera_frame is not None and camera_frame.size:
        available_w, available_h = camera_w - 2 * inset, camera_h - 2 * inset
        ratio = min(
            available_w / camera_frame.shape[1], available_h / camera_frame.shape[0]
        )
        display_w = max(1, round(camera_frame.shape[1] * ratio))
        display_h = max(1, round(camera_frame.shape[0] * ratio))
        display = cv2.resize(
            camera_frame,
            (display_w, display_h),
            interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR,
        )
        image_x = pad + (camera_w - display_w) // 2
        image_y = top + (camera_h - display_h) // 2
        canvas[image_y : image_y + display_h, image_x : image_x + display_w] = display
    else:
        _text(
            canvas,
            "Waiting for camera frames",
            pad + round(30 * scale),
            top + camera_h // 2,
            0.65 * scale,
            _MUTED,
            width=camera_w - 2 * pad,
        )

    if info_h:
        info_y = top + tactile_h + gap
        _card(canvas, (sidebar_x, info_y, sidebar_w, info_h), max(3, round(10 * scale)))
        saved = (
            f"{saved_data_hours:.4f} h"
            if saved_data_hours is not None and np.isfinite(saved_data_hours)
            else "--"
        )
        for index, (label, value, color) in enumerate(
            (
                ("Current episode", timer, timer_color),
                ("Saved this session", saved, _TEXT),
            )
        ):
            x = sidebar_x + pad + index * sidebar_w // 2
            cell_width = sidebar_w // 2 - 2 * pad
            _text(
                canvas,
                label,
                x,
                info_y + round(29 * scale),
                0.42 * scale,
                _MUTED,
                width=cell_width,
            )
            _text(
                canvas,
                value,
                x,
                info_y + round(64 * scale),
                0.75 * scale,
                color,
                width=cell_width,
            )
            if index == 0 and (error_active or timer_saved):
                cv2.rectangle(
                    canvas,
                    (x - pad // 2, info_y + round(8 * scale)),
                    (x + cell_width + pad // 2, info_y + info_h - round(8 * scale)),
                    _BAD if error_active else _GOOD,
                    max(1, round(2 * scale)),
                    cv2.LINE_AA,
                )

    if hands:
        for hand, box in zip(hands, hand_boxes, strict=True):
            _draw_hand(
                canvas,
                hand,
                box,
                tuple(components or ()),
                scale,
                tactile_text_scale,
            )
    else:
        _card(canvas, (sidebar_x, top, sidebar_w, tactile_h), max(3, round(10 * scale)))
        _text(canvas, "TACTILE", sidebar_x + pad, top + round(30 * scale), 0.55 * scale)
        _text(
            canvas,
            "Waiting for hand data",
            sidebar_x + pad,
            top + round(67 * scale),
            0.52 * scale,
            _MUTED,
            width=sidebar_w - 2 * pad,
        )
    if layout_info is not None:
        layout_info["warning_box"] = warning_box
    if encoders:
        encoder_w = (width - 2 * pad - gap * (len(encoders) - 1)) // len(encoders)
        for index, encoder in enumerate(encoders):
            box = (
                pad + index * (encoder_w + gap),
                top + upper_h + gap,
                encoder_w,
                encoder_h,
            )
            _draw_encoder(canvas, encoder, box, scale)

    _draw_warnings(
        canvas, warning_box, warnings, warning_page, scale, is_recording, error_active
    )

    footer_y = height - footer_h
    cv2.line(canvas, (pad, footer_y), (width - pad, footer_y), _LINE, 1, cv2.LINE_AA)
    _text(canvas, "TASK", pad, footer_y + round(25 * scale), 0.36 * scale, _MUTED)
    task_x = pad + round(62 * scale)
    _text(
        canvas,
        task_description or "Task not set",
        task_x,
        footer_y + round(25 * scale),
        0.47 * scale,
        width=width * 0.62 - task_x,
    )
    legend = (
        (">1000 mN", 0.66, _TACTILE_ACTIVE),
        ("Normal", 0.8, _TEXT),
        ("Missing", 0.88, _MUTED),
    )
    for text, position, color in legend:
        _text(
            canvas,
            text,
            width * position,
            footer_y + round(25 * scale),
            0.36 * scale,
            color,
        )
    hint_y = footer_y + round(54 * scale)
    _text(canvas, start_hint, pad, hint_y, 0.43 * scale, _MUTED, width=width * 0.24)
    hints = [
        ("A Discard" if is_recording else "T Task", 0.27),
        ("Q Quit", 0.93),
    ]
    for text, position in hints:
        if text:
            _text(canvas, text, width * position, hint_y, 0.43 * scale, _MUTED)
    return canvas

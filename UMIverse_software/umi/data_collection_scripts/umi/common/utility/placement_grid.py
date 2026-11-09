# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Object-placement grid for the recording preview.

Tasks are generated from templates, and each task's ``operator_instruction``
(task registry) lists the objects the operator must put on the table, e.g.::

    Place yellow cup, green cup, blue cup, 8 markers on the table

Not every listed object is used by the task, so the instruction alone does not
say *where* things go. This module renders the scene as a placement figure for
a spare preview cell: a rows x cols lattice with each object assigned to a
random intersection, so scene setup is reproducible per session without
hand-authoring a figure per task. The object list preferably comes structured
from the registry entry's ``objects`` field (the task generator emits it next
to the prose; supports containment — "2 markers in yellow cup"); the prose is
parsed only for tasks without it.

Positions are sampled fresh on every call — pass ``seed`` to pin them.
"""

import random
import re
from typing import List, Optional, Tuple

import cv2
import numpy as np

from umi.common.utility import setup_variants

# Color words recognized in object labels (BGR). An object whose name carries
# no color word gets _DEFAULT_COLOR; its text label is what tells it apart.
_COLOR_BGR = {
    "red": (40, 40, 230),
    "green": (60, 180, 60),
    "blue": (230, 130, 40),
    "yellow": (40, 210, 240),
    "orange": (30, 140, 255),
    "purple": (200, 60, 160),
    "pink": (180, 105, 255),
    "white": (235, 235, 235),
    "black": (90, 90, 90),
}
_DEFAULT_COLOR = (200, 200, 120)  # light mint

_DOT_RADIUS = 8  # one size for every object, so no dot reads as more important
# Containment ring around a container dot, drawn in the contents' color.
_RING_RADIUS = 13
_RING_THICKNESS = 2

_GRID_LINE = (70, 70, 70)
_AXIS_TEXT = (150, 150, 150)
_LABEL_TEXT = (235, 235, 235)

# Font scales for the 640x400 canvas; the preview upscales it, so these set
# the on-screen text size directly.
_LABEL_SCALE = 0.6
_AXIS_SCALE = 0.5


def parse_instruction_objects(instruction: str) -> List[Tuple[str, int]]:
    """Extract ``(label, count)`` pairs from a placement instruction.

    ``"Place yellow cup, green cup, 8 markers on the table"`` ->
    ``[("yellow cup", 1), ("green cup", 1), ("markers", 8)]``.

    Tolerant of a missing "Place" prefix / "on the table" suffix and of
    ``and`` as a separator, and of setup-variant groups (``{a|b}``, dropped).
    Returns [] when nothing parses.
    """
    text = setup_variants.strip(instruction).strip()
    match = re.search(
        r"place\s+(.*?)(?:\s+on\s+the\s+table)?\s*[.;]?\s*$",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    body = match.group(1) if match else text
    objects: List[Tuple[str, int]] = []
    for part in re.split(r",\s*|\s+and\s+", body):
        part = part.strip()
        if not part:
            continue
        count_match = re.match(r"(\d+)\s+(.+)", part)
        if count_match:
            objects.append((count_match.group(2).strip(), int(count_match.group(1))))
        else:
            objects.append((part, 1))
    return objects


def _object_color(label: str) -> Tuple[int, int, int]:
    for word, bgr in _COLOR_BGR.items():
        if word in label.lower():
            return bgr
    return _DEFAULT_COLOR


def _rects_intersect(a, b) -> bool:
    """Axis-aligned ``(x0, y0, x1, y1)`` overlap test."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


# Labels wider than this (px on the 640-wide canvas) fold into two lines.
_LABEL_FOLD_PX = 150


def _fold_label(label: str) -> list:
    """Fold a long label into two lines: at ' + ' (the containment
    separator) when present, else at the space nearest the middle."""
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, _LABEL_SCALE, 1)
    if tw <= _LABEL_FOLD_PX or " " not in label:
        return [label]
    if " + " in label:
        head, rest = label.split(" + ", 1)
        return [head, "+ " + rest]
    mid = len(label) / 2
    cut = min(
        (i for i, ch in enumerate(label) if ch == " "), key=lambda i: abs(i - mid)
    )
    return [label[:cut], label[cut + 1 :]]


def _object_label(obj: dict) -> str:
    """``{"name": "yellow cup", "contains": [{"name": "markers", "count": 2}]}``
    -> ``"yellow cup + 2 markers"``."""
    parts = [obj["name"]]
    for inner in obj.get("contains") or []:
        parts.append(f"{inner.get('count', 1)} {inner['name']}")
    return " + ".join(parts)


def _cell_code(r: int, c: int) -> str:
    """(row, col) indices -> the operator-facing coordinate, e.g. (2, 3) -> 'C4'."""
    return f"{chr(ord('A') + r)}{c + 1}"


def _cell_index(code: str) -> tuple:
    """'C4' -> (2, 3)."""
    return ord(code[0]) - ord("A"), int(code[1:]) - 1


def sample_layout(
    instruction: str,
    rows: int = 5,
    cols: int = 7,
    seed: Optional[int] = None,
    objects: Optional[List[dict]] = None,
    empty_ok: bool = False,
) -> Optional[dict]:
    """Sample object placements into a self-contained layout dict.

    This dict is the provenance format stamped into each episode's zarr
    attrs (``placement_layout``) and the input to :func:`render_layout` —
    storing it reproduces the figure later without re-running the sampler::

        {"grid": {"rows": 5, "cols": 7},
         "objects": [{"name": "yellow cup", "cell": "C4",
                      "contains": [{"name": "markers", "count": 1}]},
                     {"name": "markers", "cells": ["B3", "B5"]}]}

    ``objects`` is the task registry's structured scene list
    (``TaskSpec.objects``: ``{"name", "count", "contains": [...]}``) and is
    the preferred source; *instruction* prose is parsed only when it is
    absent (tasks without that field).

    Each top-level object instance lands on its own random interior
    intersection; the border rows/columns stay clear. Contained objects
    take no intersection; ``contains`` rides on the container's entry.
    Returns None when there are no objects (unless ``empty_ok``, which
    yields an object-less layout: the bare lattice for tasks without
    objects) or when the instances outnumber the interior intersections.
    """
    if not objects:
        objects = [
            {"name": name, "count": count}
            for name, count in parse_instruction_objects(instruction)
        ]
    total = sum(int(obj.get("count", 1)) for obj in objects)
    interior = [(r, c) for r in range(1, rows - 1) for c in range(1, cols - 1)]
    if (total == 0 and not empty_ok) or total > len(interior):
        return None

    rng = random.Random(seed)
    cells = iter(rng.sample(interior, total))
    entries = []
    for obj in objects:
        codes = [_cell_code(*next(cells)) for _ in range(int(obj.get("count", 1)))]
        entry: dict = {"name": obj["name"]}
        if len(codes) == 1:
            entry["cell"] = codes[0]
        else:
            entry["cells"] = codes
        contains = obj.get("contains") or []
        if contains:
            entry["contains"] = contains
        entries.append(entry)
    return {"grid": {"rows": rows, "cols": cols}, "objects": entries}


def render_placement_grid(
    instruction: str,
    rows: int = 5,
    cols: int = 7,
    width: int = 640,
    height: int = 400,
    seed: Optional[int] = None,
    objects: Optional[List[dict]] = None,
    empty_ok: bool = False,
) -> Optional[np.ndarray]:
    """Sample a layout and render it — see :func:`sample_layout` (the
    placement rules and provenance format) and :func:`render_layout` (the
    drawing). Callers that need to keep the layout (the recorder stamps it
    into episode attrs) call the two halves themselves."""
    layout = sample_layout(
        instruction, rows=rows, cols=cols, seed=seed, objects=objects, empty_ok=empty_ok
    )
    if layout is None:
        return None
    return render_layout(layout, width=width, height=height)


def render_layout(layout: dict, width: int = 640, height: int = 400) -> np.ndarray:
    """Draw a sampled (or stored) layout dict as the placement figure.

    Each object instance is a fixed-size dot at its cell with a text label
    beside it, moved above/below when the beside spot is occupied by another
    dot or label. A container's dot gains a thin ring in its contents' color
    and a combined label ("yellow cup + 2 markers"). Dot color comes from a
    color word in the name (_COLOR_BGR), light mint for colorless names.
    """
    rows = int(layout["grid"]["rows"])
    cols = int(layout["grid"]["cols"])
    img = np.full((height, width, 3), 25, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Lattice: `cols` vertical x `rows` horizontal lines; objects sit on the
    # intersections. Margins leave room for the A../1.. axis labels.
    margin_l, margin_t, margin_r, margin_b = 46, 30, 16, 24
    xs = np.linspace(margin_l, width - margin_r, cols).astype(int)
    ys = np.linspace(margin_t, height - margin_b, rows).astype(int)
    for x in xs:
        cv2.line(img, (x, ys[0]), (x, ys[-1]), _GRID_LINE, 1, cv2.LINE_AA)
    for y in ys:
        cv2.line(img, (xs[0], y), (xs[-1], y), _GRID_LINE, 1, cv2.LINE_AA)
    for i, x in enumerate(xs):
        cv2.putText(
            img,
            str(i + 1),
            (x - 6, ys[0] - 8),
            font,
            _AXIS_SCALE,
            _AXIS_TEXT,
            1,
            cv2.LINE_AA,
        )
    for i, y in enumerate(ys):
        cv2.putText(
            img,
            chr(ord("A") + i),
            (margin_l - 26, y + 4),
            font,
            _AXIS_SCALE,
            _AXIS_TEXT,
            1,
            cv2.LINE_AA,
        )

    # Draw and register every dot before labels so placement can avoid all
    # markers. Include container rings in the collision footprint.
    dots = []
    for entry in layout["objects"]:
        color = _object_color(entry["name"])
        contains = entry.get("contains") or []
        label = _object_label(entry)
        radius = _RING_RADIUS if contains else _DOT_RADIUS
        codes = [entry["cell"]] if "cell" in entry else entry.get("cells", [])
        for code in codes:
            r, c = _cell_index(code)
            point = (int(xs[c]), int(ys[r]))
            cv2.circle(img, point, _DOT_RADIUS, color, -1, cv2.LINE_AA)
            cv2.circle(img, point, _DOT_RADIUS, (15, 15, 15), 1, cv2.LINE_AA)
            if contains:
                cv2.circle(
                    img,
                    point,
                    _RING_RADIUS,
                    _object_color(contains[0]["name"]),
                    _RING_THICKNESS,
                    cv2.LINE_AA,
                )
            dots.append((point, label, radius))

    pad = 3
    occupied = [
        (px - radius - pad, py - radius - pad, px + radius + pad, py + radius + pad)
        for (px, py), _, radius in dots
    ]
    # Try labels beside, then above/below each dot, avoiding markers and prior
    # labels. If all positions collide, use the available side; wrap long labels
    # into two centered lines.
    for (px, py), label, radius in dots:
        lines = _fold_label(label)
        sizes = [cv2.getTextSize(ln, font, _LABEL_SCALE, 1)[0] for ln in lines]
        bw = max(w_ for w_, _ in sizes)
        th = max(h_ for _, h_ in sizes)
        lh = th + 6
        bh = lh * (len(lines) - 1)  # extra height below the first baseline
        beside = radius + 4
        # candidates are the first line's baseline origin
        candidates = [
            (px + beside, py + 4 - bh // 2),  # right of the dot, block centered
            (px - bw - beside, py + 4 - bh // 2),  # left
            (px - bw // 2, py - radius - 8 - bh),  # above
            (px - bw // 2, py + radius + 8 + th),  # below
        ]
        origin = candidates[0] if px + bw + beside <= width else candidates[1]
        for ox, oy in candidates:
            rect = (ox - pad, oy - th - pad, ox + bw + pad, oy + bh + pad)
            inside = (
                rect[0] >= 0 and rect[1] >= 0 and rect[2] <= width and rect[3] <= height
            )
            if inside and not any(_rects_intersect(rect, o) for o in occupied):
                origin = (ox, oy)
                break
        (ox, oy) = origin
        occupied.append((ox - pad, oy - th - pad, ox + bw + pad, oy + bh + pad))
        for i, (ln, (lw, _)) in enumerate(zip(lines, sizes)):
            cv2.putText(
                img,
                ln,
                (ox + (bw - lw) // 2, oy + i * lh),
                font,
                _LABEL_SCALE,
                _LABEL_TEXT,
                1,
                cv2.LINE_AA,
            )
    return img

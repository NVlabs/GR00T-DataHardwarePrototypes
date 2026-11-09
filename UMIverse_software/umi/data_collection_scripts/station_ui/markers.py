# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The image each camera is aimed at, and nothing about how it is decoded.

A QR code works poorly for this check on a fisheye lens. `UMI_VERIFY_LEFT` is
15 bytes, and because `_` is outside QR's alphanumeric charset it encodes as
BYTE, which at the default EC-M overflows version 1's 14-byte capacity, so it
renders as version 2: 25x25 modules, ~33 across with the quiet zone. A fisheye
maps a whole hemisphere onto a few hundred pixels of sensor, so a code held at
arm's length lands on maybe 40-60 of them, and a QR needs 3-4 pixels per module
to decode: two to three times under-sampled. Off-axis, barrel distortion curves
the finder patterns and the detector gives up entirely.

An ArUco marker is the right tool for this job. The check has to distinguish
exactly three things (about 1.5 bits), and QR is built to carry kilobytes.
A `DICT_4X4_50` marker is 4x4 data plus a one-module border: **6 modules across
against QR's 33**, so at the same size on screen every module is over five times
bigger, and ArUco's detector finds a quadrilateral by contour rather than
needing three finder patterns and a timing pattern to survive the lens.

The bit matrices are literals here, not generated. `station_ui.routes.calibration`
renders this into the page, and the web process deliberately does not import
cv2 (see `test_qr_rendering_does_not_import_cv2_into_the_web_process`):
pulling OpenCV and every camera SDK into the HTTP process at import time would
make the console slow and fragile. So the patterns are written down here and
pinned to OpenCV's own dictionary by a test that is allowed to import cv2. If
the two ever disagree, that test fails rather than an operator aiming at a
marker no camera will recognise.
"""

from __future__ import annotations

__all__ = [
    "MARKER_IDS",
    "marker_matrix",
    "marker_svg",
]

# Map verification sides to the first three ArUco IDs in DICT_4X4_50.
MARKER_IDS = {"left": 0, "right": 1, "head": 2}

# Include the black border in each 6x6 marker grid; the detector needs it
# to find the outer quad.
_MATRICES = {
    0: ("111111", "101001", "110101", "111001", "111011", "111111"),
    1: ("111111", "111111", "100001", "101101", "101011", "111111"),
    2: ("111111", "111001", "111001", "111011", "100101", "111111"),
}

# Add a white quiet zone so the marker's black outer border has a
# detectable edge against the dark page.
_QUIET = 1


def marker_matrix(side: str) -> tuple[str, ...]:
    """The 6x6 module grid for *side*. Raises KeyError on anything else."""
    return _MATRICES[MARKER_IDS[side]]


def marker_svg(side: str) -> str:
    """The marker for *side* as a standalone SVG document.

    Drawn in module coordinates with no width/height attributes, so the page
    decides how big it is: the single most effective thing that can be done for
    a fisheye is to make this fill the screen, and a size baked in here would
    cap that. `shape-rendering="crispEdges"` keeps the module boundaries hard
    under scaling: a resampled, softened edge is exactly what costs a marginal
    detection.
    """
    grid = marker_matrix(side)
    span = len(grid) + 2 * _QUIET
    rects = [
        f'<rect x="{x + _QUIET}" y="{y + _QUIET}" width="1" height="1"/>'
        for y, row in enumerate(grid)
        for x, module in enumerate(row)
        if module == "1"
    ]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {span} {span}" '
        f'shape-rendering="crispEdges" role="img" '
        f'aria-label="Alignment marker for the {side} cameras">'
        f'<rect width="{span}" height="{span}" fill="white"/>'
        f'<g fill="black">{"".join(rects)}</g>'
        "</svg>"
    )

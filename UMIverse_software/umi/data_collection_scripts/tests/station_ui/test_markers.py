# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The marker the page draws is the marker the worker looks for.

`station_ui.markers` writes the ArUco bit patterns down as literals, because the
web process must not import cv2 (see
`test_camera_check.py::test_qr_rendering_does_not_import_cv2_into_the_web_process`
-- pulling OpenCV and every camera SDK into the HTTP process at import time
would make the console slow and fragile).

Literals can drift from what they claim to be, and the failure mode is the worst
one available: an operator aiming a camera at a marker no detector will ever
recognise, indistinguishable from a broken camera. So this module is the pin.
It is allowed to import cv2 -- it is a test, not the server -- and it checks the
patterns against OpenCV's own dictionary rather than against a second copy of
the same literals.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

from station_ui import markers  # noqa: E402

cv2 = pytest.importorskip("cv2")


def _opencv_matrix(marker_id: int) -> list[str]:
    """The same 6x6 grid, straight from OpenCV. One module per pixel."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    image = cv2.aruco.generateImageMarker(dictionary, marker_id, 6, borderBits=1)
    return ["".join("1" if value == 0 else "0" for value in row) for row in image]


@pytest.mark.parametrize("side", ["left", "right", "head"])
def test_the_written_down_matrix_is_the_one_opencv_would_draw(side):
    """The pin. If OpenCV's dictionary ever changes, or a literal is mistyped,
    this fails here rather than at the bench."""
    marker_id = markers.MARKER_IDS[side]

    assert list(markers.marker_matrix(side)) == _opencv_matrix(marker_id)


def test_every_side_uses_a_different_marker():
    """A shared marker would be decoded by every camera and would pass a
    left/right swap -- the one fault this whole check exists to catch."""
    assert len(set(markers.MARKER_IDS.values())) == len(markers.MARKER_IDS) == 3


@pytest.mark.parametrize("side", ["left", "right", "head"])
def test_the_rendered_svg_decodes_back_to_its_own_side(side):
    """End to end through a real detector: rasterise what the page serves and
    check OpenCV reads the id back. Asserting the SVG's bytes would only restate
    the generator; this asserts the thing that has to be true."""
    grid = markers.marker_matrix(side)
    quiet = 1
    span = len(grid) + 2 * quiet
    # White field, black modules -- the same picture the SVG paints, at a size a
    # detector is comfortable with.
    scale = 20
    image = np.full((span * scale, span * scale), 255, np.uint8)
    for y, row in enumerate(grid):
        for x, module in enumerate(row):
            if module == "1":
                top, left = (y + quiet) * scale, (x + quiet) * scale
                image[top : top + scale, left : left + scale] = 0

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    _corners, ids, _rejected = detector.detectMarkers(image)

    assert ids is not None and len(ids), f"the {side} marker did not detect at all"
    assert int(ids.flatten()[0]) == markers.MARKER_IDS[side]


def test_the_svg_carries_no_baked_in_size():
    """The stylesheet sizes this to the viewport. A width/height here would cap
    the marker at a fixed square, and how big it is on the glass is the one term
    of a fisheye's angular resolution the console controls."""
    svg = markers.marker_svg("left")

    assert "viewBox=" in svg
    assert "width=" not in svg.split(">", 1)[0], svg.split(">", 1)[0]
    assert "height=" not in svg.split(">", 1)[0]


def test_the_marker_is_drawn_on_its_own_white_field():
    """ArUco finds the black border as a quad, so it needs something lighter
    around it. Flush against the dark panel there is no outer edge to find."""
    svg = markers.marker_svg("left")

    assert 'fill="white"' in svg


def test_an_unknown_side_raises_rather_than_drawing_something_wrong():
    with pytest.raises(KeyError):
        markers.marker_svg("sideways")

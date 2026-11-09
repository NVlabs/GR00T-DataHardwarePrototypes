# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for `roi_for_tile`, the inverse of `DataVisualizer.combine_frames`.

The point of these tests is that the helper stays honest about the tiling
`combine_frames` actually produces: every assertion here is cross-checked
against a real composite built by the visualizer itself, so the two cannot
drift apart silently.
"""

from __future__ import annotations

import numpy as np

from umi.common.utility.data_visualizer import (
    DataVisualizer,
    VisualizationConfig,
    roi_for_tile,
)

# The tile size `combine_frames` normalises every frame to.
TILE_W, TILE_H = 640, 400


def _composite(layout, n_cameras):
    """A real composite from the visualizer, so the expected geometry is
    measured rather than assumed."""
    viz = DataVisualizer(VisualizationConfig(layout=layout))
    frames = [np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8) for _ in range(n_cameras)]
    return viz.combine_frames(frames)


def test_horizontal_tiles_span_the_composite_width():
    combined = _composite("horizontal", 3)
    h, w = combined.shape[:2]
    assert (w, h) == (TILE_W * 3, TILE_H)

    assert roi_for_tile("horizontal", 3, 0, w, h) == (0, 0, TILE_W, TILE_H)
    assert roi_for_tile("horizontal", 3, 1, w, h) == (TILE_W, 0, TILE_W, TILE_H)
    assert roi_for_tile("horizontal", 3, 2, w, h) == (TILE_W * 2, 0, TILE_W, TILE_H)


def test_vertical_tiles_stack_down_the_composite():
    combined = _composite("vertical", 2)
    h, w = combined.shape[:2]
    assert (w, h) == (TILE_W, TILE_H * 2)

    assert roi_for_tile("vertical", 2, 0, w, h) == (0, 0, TILE_W, TILE_H)
    assert roi_for_tile("vertical", 2, 1, w, h) == (0, TILE_H, TILE_W, TILE_H)


def test_grid_wraps_six_cameras_into_three_columns():
    # The production bimanual rig: 6 sources, `layout: grid` -> 3 cols x 2 rows.
    combined = _composite("grid", 6)
    h, w = combined.shape[:2]
    assert (w, h) == (TILE_W * 3, TILE_H * 2)

    assert roi_for_tile("grid", 6, 0, w, h) == (0, 0, TILE_W, TILE_H)
    assert roi_for_tile("grid", 6, 2, w, h) == (TILE_W * 2, 0, TILE_W, TILE_H)
    # The two tracking cameras land on the bottom row, not side by side up top.
    assert roi_for_tile("grid", 6, 4, w, h) == (TILE_W, TILE_H, TILE_W, TILE_H)
    assert roi_for_tile("grid", 6, 5, w, h) == (TILE_W * 2, TILE_H, TILE_W, TILE_H)


def test_grid_pads_a_short_last_row():
    # 5 cameras -> 3 cols x 2 rows with one black pad tile. Index 4 is the only
    # tile on the bottom row and must not be stretched across it.
    combined = _composite("grid", 5)
    h, w = combined.shape[:2]
    assert (w, h) == (TILE_W * 3, TILE_H * 2)
    assert roi_for_tile("grid", 5, 4, w, h) == (TILE_W, TILE_H, TILE_W, TILE_H)


def test_grid_overlay_reserves_a_trailing_slot_for_the_blend():
    # np.hstack(cameras..., blended) -- one extra panel the cameras must not
    # be measured into.
    combined = _composite("grid_overlay", 2)
    h, w = combined.shape[:2]
    assert (w, h) == (TILE_W * 3, TILE_H)

    assert roi_for_tile("grid_overlay", 2, 0, w, h) == (0, 0, TILE_W, TILE_H)
    assert roi_for_tile("grid_overlay", 2, 1, w, h) == (TILE_W, 0, TILE_W, TILE_H)


def test_overlay_cannot_isolate_a_camera_and_says_so():
    # Every source is alpha-blended into one tile, so there is no per-camera
    # region. Callers that dim "the other tiles" must detect this and bail.
    assert roi_for_tile("overlay", 3, 1, 640, 400) is None


def test_single_camera_owns_the_whole_composite():
    assert roi_for_tile("grid", 1, 0, 640, 400) == (0, 0, 640, 400)


def test_out_of_range_index_has_no_roi():
    assert roi_for_tile("grid", 3, 3, 1920, 800) is None
    assert roi_for_tile("grid", 3, -1, 1920, 800) is None


def test_unknown_layout_has_no_roi():
    assert roi_for_tile("nonsense", 3, 1, 1920, 800) is None

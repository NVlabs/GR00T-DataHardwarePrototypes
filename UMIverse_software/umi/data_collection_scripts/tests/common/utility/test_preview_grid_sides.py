# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grid layout: the ``sides`` fill mode, and its fallbacks.

Tiles are solid colors so the composite can be read back cell-by-cell: the
point of the mode is *which camera lands where*.
"""

import numpy as np
import pytest

from umi.common.utility.data_visualizer import DataVisualizer, VisualizationConfig

TILE_H, TILE_W = 400, 640


def _viz(**kwargs):
    cfg = VisualizationConfig(
        window_name="test", layout="grid", grid_fill="sides", **kwargs
    )
    return DataVisualizer(cfg)


def _tiles(labels):
    """One flat frame per label, each a unique shade so cells are identifiable."""
    return [
        np.full((TILE_H, TILE_W, 3), 10 + 10 * i, dtype=np.uint8)
        for i in range(len(labels))
    ]


def _cell(grid, row, col):
    return grid[row * TILE_H : (row + 1) * TILE_H, col * TILE_W : (col + 1) * TILE_W]


def _read(grid, labels, rows, cols):
    """Recover the grid as a rows x cols table of label indices (None = blank)."""
    shades = {10 + 10 * i: i for i in range(len(labels))}
    table = []
    for r in range(rows):
        row = []
        for c in range(cols):
            # Sample a corner the label text never reaches.
            row.append(shades.get(int(_cell(grid, r, c)[-5, -5, 0])))
        table.append(row)
    return table


def test_six_camera_rig_mirrors_the_two_arms():
    """Source order interleaves the arms; a plain column fill would put
    right_trigger at the bottom of the left column."""
    labels = [
        "left_trigger_camera",
        "left_umi_camera",
        "right_trigger_camera",
        "right_umi_camera",
        "left_tracking_camera",
        "right_tracking_camera",
    ]
    grid = _viz().combine_frames(_tiles(labels), labels=labels)

    assert grid.shape == (3 * TILE_H, 2 * TILE_W, 3)
    assert _read(grid, labels, 3, 2) == [
        [0, 2],  # left_trigger  | right_trigger
        [1, 3],  # left_umi      | right_umi
        [4, 5],  # left_tracking | right_tracking
    ]


def test_four_camera_rig_fills_two_by_two():
    labels = [
        "left_umi_camera",
        "right_umi_camera",
        "left_tracking_camera",
        "right_tracking_camera",
    ]
    grid = _viz().combine_frames(_tiles(labels), labels=labels)

    assert grid.shape == (2 * TILE_H, 2 * TILE_W, 3)
    assert _read(grid, labels, 2, 2) == [[0, 1], [2, 3]]


def test_side_token_is_matched_as_a_word_not_a_substring():
    """``top_left_*`` is a left camera; ``bright_camera`` is not a right one."""
    labels = ["top_left_umi_camera", "wrist_human_camera_right", "bright_camera"]
    grid = _viz().combine_frames(_tiles(labels), labels=labels)

    # bright_camera is unsided, so it appends to the shorter column.
    assert grid.shape == (2 * TILE_H, 2 * TILE_W, 3)
    assert _read(grid, labels, 2, 2) == [[0, 1], [2, None]]


def test_head_camera_appends_and_the_grid_stays_two_wide():
    """An extra joins a column rather than opening a third one: a third
    column for one head camera costs ~16 points of screen height."""
    labels = [
        "left_umi_camera",
        "right_umi_camera",
        "head_camera",
        "right_tracking_camera",
    ]
    grid = _viz().combine_frames(_tiles(labels), labels=labels)

    # left=[0], right=[1,3]; head appends to the shorter (left) column.
    assert grid.shape == (2 * TILE_H, 2 * TILE_W, 3)
    assert _read(grid, labels, 2, 2) == [[0, 1], [2, 3]]


def test_the_full_v3_rig_with_a_head_mount_stays_two_wide():
    """The 7-tile case: optional_devices adds the head cam at Apply time."""
    labels = [
        "left_trigger_camera",
        "left_umi_camera",
        "right_trigger_camera",
        "right_umi_camera",
        "left_tracking_camera",
        "right_tracking_camera",
        "head_oak_camera",
    ]
    grid = _viz().combine_frames(_tiles(labels), labels=labels)

    assert grid.shape == (4 * TILE_H, 2 * TILE_W, 3)
    assert _read(grid, labels, 4, 2) == [
        [0, 2],  # left_trigger  | right_trigger
        [1, 3],  # left_umi      | right_umi
        [4, 5],  # left_tracking | right_tracking
        [6, None],  # head_oak   | ----
    ]


def test_short_column_pads_at_its_own_bottom_not_the_next_column():
    """A flat append would push the blank into the right column, shifting
    every right-hand camera up a row."""
    labels = ["left_umi_camera", "right_umi_camera", "right_tracking_camera"]
    grid = _viz().combine_frames(_tiles(labels), labels=labels)

    assert _read(grid, labels, 2, 2) == [[0, 1], [None, 2]]


@pytest.mark.parametrize(
    "labels",
    [
        ["left_umi_camera", "left_tracking_camera"],  # one-sided rig
        ["main_camera", "top_camera", "side_camera"],  # no sided camera at all
        None,  # caller passed no labels (umi_env.combine_frames)
    ],
)
def test_falls_back_to_the_plain_fill_when_the_rig_is_not_two_sided(labels):
    """Better a normal grid than a half-window of black."""
    names = labels or ["a", "b"]
    grid = _viz(grid_columns=2).combine_frames(_tiles(names), labels=labels)

    rows = -(-len(names) // 2)
    assert grid.shape == (rows * TILE_H, 2 * TILE_W, 3)


def test_sides_mode_leaves_the_other_fill_modes_alone():
    """The shared padding path still row-fills by default."""
    labels = ["left_umi_camera", "right_umi_camera", "head_camera"]
    cfg = VisualizationConfig(window_name="test", layout="grid", grid_columns=2)
    grid = DataVisualizer(cfg).combine_frames(_tiles(labels), labels=labels)

    assert _read(grid, labels, 2, 2) == [[0, 1], [2, None]]

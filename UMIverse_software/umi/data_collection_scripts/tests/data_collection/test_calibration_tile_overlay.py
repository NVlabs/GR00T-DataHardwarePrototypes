# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The CALIBRATION overlay marks which tiles the board has to be aimed at.

On a bimanual rig the composite behind the centred text panel is six tiles and
only two of them are calibration cameras -- and they land on the bottom row,
not side by side -- so naming the cameras in the panel alone would leave the
operator to map a name to a tile. These tests pin the tile treatment:
everything that is not a calibration camera is dimmed, and the ones that are
get a border and a tag.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from real_scripts.data_collection.pre_episode_checks.camera_calibration import (
    AIM_COLOR,
    OK_COLOR,
    CameraRelativeCalibrationCheck,
)
from umi.common.recording.pre_episode_checks import PreEpisodeContext

TILE_W, TILE_H = 640, 400

# A two-arm rig's sources in the order the recorder yields them, which is the
# order combine_frames tiles them.
ORDER = [
    "left_trigger_camera",
    "left_umi_camera",
    "right_trigger_camera",
    "right_umi_camera",
    "left_tracking_camera",
    "right_tracking_camera",
]
CAL_CAMERAS = ["right_tracking_camera", "left_tracking_camera"]
REF = "right_tracking_camera"

# Where each source sits in a 3x2 grid, as (col, row).
TILE_OF = {
    "left_trigger_camera": (0, 0),
    "left_umi_camera": (1, 0),
    "right_trigger_camera": (2, 0),
    "right_umi_camera": (0, 1),
    "left_tracking_camera": (1, 1),
    "right_tracking_camera": (2, 1),
}

BASE_GREY = 120


def _check(align_stable_required=5, run_every_n_episodes=10):
    return CameraRelativeCalibrationCheck(
        reference_camera=REF,
        cameras=list(CAL_CAMERAS),
        run_every_n_episodes=run_every_n_episodes,
        marker={"mode": "aruco", "marker_size": 0.1875},
        align_stable_required=align_stable_required,
    )


def _ctx(layout="grid", order=ORDER, episode_id=0):
    return PreEpisodeContext(
        latest_frames={},
        cfg={"visualization": {"layout": layout}},
        recorder_manager=SimpleNamespace(episode_id=episode_id),
        video_source_order=list(order) if order is not None else None,
    )


def _canvas(cols=3, rows=2):
    return np.full((TILE_H * rows, TILE_W * cols, 3), BASE_GREY, np.uint8)


def _tile(canvas, name):
    col, row = TILE_OF[name]
    return canvas[row * TILE_H : (row + 1) * TILE_H, col * TILE_W : (col + 1) * TILE_W]


def _corner(canvas, name):
    """A pixel on the tile's inset calibration border, clear of the centre
    panel and of the tile's own imagery."""
    col, row = TILE_OF[name]
    return canvas[row * TILE_H + 8, col * TILE_W + 8]


def _undimmed_region(canvas, name):
    """A patch of the tile that measures the scrim and nothing else: clear of
    the inset border, of the tag chip below the name badge, and of the centred
    panel that straddles the composite's middle rows."""
    col, row = TILE_OF[name]
    y, x = row * TILE_H, col * TILE_W
    return canvas[y + 200 : y + 280, x + 40 : x + 200]


def test_non_calibration_tiles_are_dimmed_and_calibration_tiles_are_not():
    chk = _check()
    canvas = _canvas()

    chk.draw(canvas, _ctx())

    for name in [
        "left_trigger_camera",
        "left_umi_camera",
        "right_trigger_camera",
        "right_umi_camera",
    ]:
        assert _undimmed_region(canvas, name).mean() < BASE_GREY * 0.75, (
            f"{name} is not a calibration camera and should have been dimmed"
        )
    for name in CAL_CAMERAS:
        assert _undimmed_region(canvas, name).mean() == BASE_GREY, (
            f"{name} is a calibration camera and must stay at full brightness"
        )


def test_a_calibration_camera_short_of_its_stable_count_borders_amber():
    chk = _check(align_stable_required=5)
    chk.start_session()
    chk._stable_count["left_tracking_camera"] = 3
    canvas = _canvas()

    chk.draw(canvas, _ctx())

    assert tuple(_corner(canvas, "left_tracking_camera")) == AIM_COLOR


def test_a_calibration_camera_at_its_stable_count_borders_green():
    chk = _check(align_stable_required=5)
    chk.start_session()
    chk._stable_count["left_tracking_camera"] = 5
    canvas = _canvas()

    chk.draw(canvas, _ctx())

    assert tuple(_corner(canvas, "left_tracking_camera")) == OK_COLOR


def test_the_reference_camera_borders_green_once_it_has_seen_the_board():
    chk = _check()
    chk.start_session()
    canvas = _canvas()

    chk.draw(canvas, _ctx())
    assert tuple(_corner(canvas, REF)) == AIM_COLOR, "not detected yet"

    chk._latest_capture[REF] = {"frame": None, "T_marker_optical": np.eye(4)}
    canvas = _canvas()
    chk.draw(canvas, _ctx())
    assert tuple(_corner(canvas, REF)) == OK_COLOR


def test_tile_tag_names_the_reference_camera_and_counts_the_others():
    chk = _check(align_stable_required=5)
    chk.start_session()
    chk._stable_count["left_tracking_camera"] = 3

    assert chk._tile_tag(REF) == ("REF", AIM_COLOR)
    assert chk._tile_tag("left_tracking_camera") == (
        "AIM BOARD HERE  3/5",
        AIM_COLOR,
    )

    chk._stable_count["left_tracking_camera"] = 5
    assert chk._tile_tag("left_tracking_camera") == (
        "AIM BOARD HERE  5/5",
        OK_COLOR,
    )


def test_panel_drops_the_per_camera_counters_once_the_tiles_carry_them():
    # In a grid the panel sits at the composite's centre, which is the top edge
    # of the middle-bottom tile -- exactly where that tile's name badge is.
    # Counters there would be unreadable, and the tags already say the same.
    chk = _check(align_stable_required=5)
    chk.start_session()
    chk._stable_count["left_tracking_camera"] = 3

    texts = [line[0] for line in chk._panel_lines(episode_id=0, per_camera=False)]

    assert len(texts) == 1
    assert "CONVERGING" in texts[0]


def test_panel_keeps_the_per_camera_counters_when_the_tiles_cannot():
    # `layout: overlay` blends every source into one pane, so no tile was
    # decorated and the panel is the only place the counts can be read.
    chk = _check(align_stable_required=5)
    chk.start_session()
    chk._stable_count["left_tracking_camera"] = 3

    texts = [line[0] for line in chk._panel_lines(episode_id=0, per_camera=True)]

    assert any("left_tracking_camera: 3/5" in t for t in texts)
    assert any(f"ref={REF}" in t for t in texts)


def test_draw_tiles_reports_whether_it_decorated_anything():
    chk = _check()
    canvas = _canvas()

    assert chk._draw_tiles(canvas, _ctx()) is True
    assert chk._draw_tiles(canvas, _ctx(layout="overlay")) is False
    assert chk._draw_tiles(canvas, _ctx(order=None)) is False


def test_overlay_layout_dims_nothing():
    # Every source is blended into one pane, so there is no "other tile" to
    # dim -- dimming by ROI would black out the whole window.
    chk = _check()
    canvas = _canvas(cols=1, rows=1)
    before = canvas.copy()

    chk.draw(canvas, _ctx(layout="overlay"))

    panel = canvas[TILE_H // 2 - 60 : TILE_H // 2 + 60]
    assert not np.array_equal(panel, before[TILE_H // 2 - 60 : TILE_H // 2 + 60]), (
        "the centre panel should still be drawn"
    )
    assert canvas[0:40, 0:200].mean() == BASE_GREY, "a tile was dimmed"


def test_no_video_source_order_dims_nothing():
    chk = _check()
    canvas = _canvas()

    chk.draw(canvas, _ctx(order=None))

    assert canvas[0:40, 0:200].mean() == BASE_GREY


def test_nothing_is_drawn_when_the_calibration_is_cached_and_not_due():
    chk = _check(run_every_n_episodes=10)
    chk._cached = {"stub": True}
    chk._last_calib_episode = 0
    canvas = _canvas()
    before = canvas.copy()

    chk.draw(canvas, _ctx(episode_id=1))

    assert np.array_equal(canvas, before)

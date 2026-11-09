# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`roi_for_tile` must invert `combine_frames` for the shipped configs.

A square row-major grid is the wrong assumption for templates that set
`grid_columns: 2` / `grid_fill: sides`, which puts tiles onto the operator
panel. These tests composite each config and read back where each camera
landed.
"""

import pathlib

import numpy as np
import pytest

yaml = pytest.importorskip("yaml")

from umi.common.utility.data_visualizer import (  # noqa: E402
    build_visualizer,
    roi_for_tile,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
CONFIGS = sorted(
    (REPO / "config" / "data_recording" / "production").glob("bimanual_v*.yaml")
)


def _camera_names(cfg) -> list:
    names = []
    for source in cfg.get("video_sources") or []:
        if source.get("type") == "oak_device":
            names += [s["name"] for s in source.get("streams") or []]
        else:
            names.append(source.get("name"))
    return names


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_the_ghost_lands_on_its_own_camera(path):
    cfg = yaml.safe_load(path.read_text())
    viz_cfg = dict(cfg.get("visualization") or {})
    names = _camera_names(cfg)
    if viz_cfg.get("layout") != "grid" or len(names) < 2:
        pytest.skip("not a multi-camera grid layout")

    viz_cfg["filler_cells"] = [{"text": "instruction"}, {"text": "placement"}]
    composite = build_visualizer(viz_cfg).combine_frames(
        # Blue channel carries index + 1; 0 means panel or padding.
        [np.full((400, 640, 3), (i + 1, 0, 0), np.uint8) for i in range(len(names))],
        labels=names,
    )
    h, w = composite.shape[:2]

    misplaced = []
    for i, name in enumerate(names):
        roi = roi_for_tile(
            viz_cfg["layout"],
            len(names),
            i,
            w,
            h,
            grid_columns=viz_cfg.get("grid_columns"),
            grid_fill=viz_cfg.get("grid_fill", "rows"),
            labels=names,
        )
        assert roi is not None, f"{name}: grid layout must give a per-camera ROI"
        x, y, tw, th = roi
        assert x + tw <= w and y + th <= h, f"{name}: ROI outside the composite"
        found = int(composite[y + th // 2, x + tw // 2, 0]) - 1
        if found != i:
            landed = names[found] if 0 <= found < len(names) else "operator panel"
            misplaced.append(f"{name} -> {landed}")

    assert not misplaced, f"{path.name}: {'; '.join(misplaced)}"


def test_callers_that_pass_no_layout_keys_keep_the_old_geometry():
    """Non-v3 callers are unaffected: square, row-major."""
    assert roi_for_tile("grid", 6, 2, 1920, 800) == (2 * 640, 0, 640, 400)
    assert roi_for_tile("grid", 6, 4, 1920, 800) == (640, 400, 640, 400)

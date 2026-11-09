# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Placement-layout provenance: the layout shown to the operator is stamped
into the episode's zarr attrs at save and reproduces the same figure later.

Covers the sample/render split in ``umi.common.utility.placement_grid`` (the
layout dict is the contract between the preview and the stored metadata) and
``RecorderManager._save_scene_provenance`` (episode attrs survive finalize).
"""

import time

import numpy as np
import zarr

from umi.common.data.frame import EncoderFrame
from umi.common.hardware_sdk.sensor.mock_sensor import MockSensor
from umi.common.recording.numeric_recorder import NumericRecorder
from umi.common.recording.record_manager import RecorderManager
from umi.common.utility.placement_grid import (
    render_layout,
    render_placement_grid,
    sample_layout,
)

INSTRUCTION = (
    "Place yellow cup, red cup, blue cup, 6 markers on the table. "
    "Place 1 markers in yellow cup."
)
OBJECTS = [
    {"name": "yellow cup", "count": 1, "contains": [{"name": "markers", "count": 1}]},
    {"name": "red cup", "count": 1},
    {"name": "blue cup", "count": 1},
    {"name": "markers", "count": 6},
]


def test_sample_layout_schema_and_determinism():
    layout = sample_layout(INSTRUCTION, objects=OBJECTS, seed=42)
    # Only grid size + object placements — no seed / version / extras.
    assert set(layout.keys()) == {"grid", "objects"}
    assert layout["grid"] == {"rows": 5, "cols": 7}
    by_name = {entry["name"]: entry for entry in layout["objects"]}
    assert set(by_name) == {"yellow cup", "red cup", "blue cup", "markers"}
    assert by_name["yellow cup"]["contains"] == [{"name": "markers", "count": 1}]
    assert len(by_name["markers"]["cells"]) == 6
    # All cells interior (border rows/cols A, E, 1, 7 stay clear) and unique.
    codes = [
        code
        for entry in layout["objects"]
        for code in ([entry["cell"]] if "cell" in entry else entry["cells"])
    ]
    assert len(codes) == len(set(codes)) == 9
    for code in codes:
        assert code[0] in "BCD" and 2 <= int(code[1:]) <= 6, code
    # Same seed -> same layout; the dict alone reproduces the drawn figure.
    assert layout == sample_layout(INSTRUCTION, objects=OBJECTS, seed=42)
    assert np.array_equal(
        render_layout(layout),
        render_placement_grid(INSTRUCTION, objects=OBJECTS, seed=42),
    )


def _run_episode(record_path, configure):
    source = MockSensor("right_arm")
    recorder = NumericRecorder(
        record_fps=60,
        stream_fps=60,
        record_path=record_path,
        numeric_sources=[source],
        frame_data_class=EncoderFrame,
        verbose=False,
        info_verbose=False,
        warning_verbose=False,
    )
    manager = RecorderManager(recorders=[recorder], record_path=record_path)
    configure(manager)
    manager.start_streaming()
    time.sleep(0.1)
    manager.reset_episode_recording()
    manager.start_recording()
    time.sleep(0.2)
    manager.stop_recording()
    manager.save_recordings(pending=False)  # finalizes -> episode_0/
    manager.stop_streaming()
    return dict(zarr.open(record_path, mode="r")["episode_0"].attrs)


def test_saved_episode_carries_layout_and_instruction(tmp_path):
    layout = sample_layout(INSTRUCTION, objects=OBJECTS, seed=7)

    def configure(manager):
        manager.operator_instruction = INSTRUCTION
        manager.placement_layout = layout

    attrs = _run_episode(str(tmp_path / "rec"), configure)
    assert attrs["operator_instruction"] == INSTRUCTION
    # zarr JSON round-trip preserves the dict exactly, and rendering the
    # stored dict reproduces the figure the operator staged the scene from.
    assert attrs["placement_layout"] == layout
    assert np.array_equal(
        render_layout(attrs["placement_layout"]), render_layout(layout)
    )


def test_legacy_task_saves_neither_attr(tmp_path):
    attrs = _run_episode(str(tmp_path / "rec"), lambda manager: None)
    assert "operator_instruction" not in attrs
    assert "placement_layout" not in attrs

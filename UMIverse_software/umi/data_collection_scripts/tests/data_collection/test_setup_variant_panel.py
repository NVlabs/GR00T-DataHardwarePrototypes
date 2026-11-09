# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Setup variants in the recorder: cell 0 shows this episode's option, and the
saved ``operator_instruction`` attr is the sentence the operator saw."""

import time

import zarr

from real_scripts.data_collection.record_data import _setup_panel
from umi.common.data.frame import EncoderFrame
from umi.common.hardware_sdk.sensor.mock_sensor import MockSensor
from umi.common.recording.numeric_recorder import NumericRecorder
from umi.common.recording.record_manager import RecorderManager
from umi.common.utility import setup_variants
from umi.common.utility.placement_grid import parse_instruction_objects

TOWEL = (
    "Place blue towel folded {once|twice|three times|four times}, tray on the table."
)
OPTIONS = ["once", "twice", "three times", "four times"]
UID = "2026-10-06_test_session"


def _cell0(raw, episode_id, warning=None):
    setup = setup_variants.resolve(raw, UID, episode_id)
    return _setup_panel(setup, "unfold the towel", warning)["text"]


def test_cell0_shows_one_option_per_episode():
    texts = [_cell0(TOWEL, e) for e in range(8)]
    for text in texts:
        assert "{" not in text and "TASK\nunfold the towel" in text
        assert sum(f"folded {o}," in text for o in OPTIONS) == 1
    for o in OPTIONS:
        assert sum(f"folded {o}," in t for t in texts) == 2
    # A discard or a 'g' re-roll keeps the episode id, hence the text.
    assert _cell0(TOWEL, 3) == texts[3]


def test_cell0_without_groups_is_unchanged():
    raw = "Place towel on the mat."
    assert _cell0(raw, 5) == "SETUP\nPlace towel on the mat.\n\nTASK\nunfold the towel"


def test_cell0_keeps_registry_warning_above_setup():
    text = _cell0(TOWEL, 1, warning="registry copy is stale")
    assert text.startswith("REGISTRY WARNING\nregistry copy is stale\n\nSETUP\n")


def test_prose_parsing_drops_groups():
    assert parse_instruction_objects(TOWEL) == [
        ("blue towel folded", 1),
        ("tray", 1),
    ]


def _manager(record_path):
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
    return RecorderManager(recorders=[recorder], record_path=record_path)


def _record(manager, pending):
    # What record_data's episode-advance hook does before each save.
    manager.operator_instruction = setup_variants.resolve(
        TOWEL, UID, manager.episode_id
    )
    manager.reset_episode_recording()
    manager.start_recording()
    time.sleep(0.2)
    manager.stop_recording()
    manager.save_recordings(pending=pending)


def test_saved_attr_is_the_resolved_sentence(tmp_path):
    record_path = str(tmp_path / "rec")
    manager = _manager(record_path)
    manager.start_streaming()
    time.sleep(0.1)
    try:
        _record(manager, pending=True)  # episode 0, then discarded
        assert manager.discard_staging()
        _record(manager, pending=False)  # episode 0 redone
        _record(manager, pending=False)  # episode 1
    finally:
        manager.stop_streaming()
    store = zarr.open(record_path, mode="r")
    for e in (0, 1):
        assert store[f"episode_{e}"].attrs["operator_instruction"] == (
            setup_variants.resolve(TOWEL, UID, e)
        )
    assert (
        store["episode_0"].attrs["operator_instruction"]
        != store["episode_1"].attrs["operator_instruction"]
    )

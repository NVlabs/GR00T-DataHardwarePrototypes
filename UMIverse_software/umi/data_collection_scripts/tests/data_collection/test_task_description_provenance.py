# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The task description is fixed for the life of a recording session.

``record_data.py`` sets ``RecorderManager.task_description`` once, from the
config the station UI built out of the task registry, and nothing else may
change it afterwards. A hotkey that edited it mid-session would let one
accidental keypress relabel every episode recorded after it.

Two guards: the recorder writes the same description onto every episode and
onto the dataset root, and the recorder UI has no key bound to changing it.
"""

import re
import time
from pathlib import Path

import zarr

from umi.common.data.frame import EncoderFrame
from umi.common.hardware_sdk.sensor.mock_sensor import MockSensor
from umi.common.recording.numeric_recorder import NumericRecorder
from umi.common.recording.record_manager import RecorderManager

DESCRIPTION = (
    "pour dice from the blue measuring cup into the bowl, stir them with the "
    "whisk, and lay the whisk on the tray"
)

RECORD_DATA = (
    Path(__file__).resolve().parents[2]
    / "real_scripts"
    / "data_collection"
    / "record_data.py"
)


def _record_episodes(record_path, count):
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
    manager.task_description = DESCRIPTION  # the one write record_data.py does
    manager.start_streaming()
    time.sleep(0.1)
    for _ in range(count):
        manager.reset_episode_recording()
        manager.start_recording()
        time.sleep(0.2)
        manager.stop_recording()
        manager.save_recordings(pending=False)
    manager.stop_streaming()
    return zarr.open(record_path, mode="r")


def test_every_episode_and_the_dataset_root_carry_the_startup_description(tmp_path):
    store = _record_episodes(str(tmp_path / "rec"), count=2)
    assert store["episode_0"].attrs["task_description"] == DESCRIPTION
    assert store["episode_1"].attrs["task_description"] == DESCRIPTION
    assert store.attrs["task_description"] == DESCRIPTION


def test_record_data_has_no_key_that_changes_the_description():
    source = RECORD_DATA.read_text()
    # The setter is called exactly once, at startup, with the config's value.
    assert source.count("recorder_manager.task_description = ") == 1
    assert "recorder_manager.task_description = initial_task" in source
    # No hotkey opens a text-entry mode: no `t` binding, no input buffer, and
    # no character accumulation from the cv2 key loop.
    assert 'ord("t")' not in source
    assert "task_input_mode" not in source
    assert not re.search(r"\+= chr\(key\)", source)

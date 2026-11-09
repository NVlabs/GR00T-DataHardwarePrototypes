# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GUI-side UMI gripper identity helpers."""

import sys
from pathlib import Path
from types import SimpleNamespace

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from station_ui.core import config_utils  # noqa: E402
from station_ui.core import umi_gripper_devices as grippers  # noqa: E402


def _device(device_id, registry_name=None):
    return SimpleNamespace(
        kind="gripper", device_id=device_id, registry_name=registry_name
    )


def test_a_registered_device_is_selectable_only_under_its_registry_name(
    monkeypatch,
):
    monkeypatch.setattr(grippers, "UMI_V4_GRIPPERS", {"UMI-V4-0001": "A" * 16})

    assert grippers.is_gripper_device_selectable(_device("A" * 16, "UMI-V4-0001"))
    assert not grippers.is_gripper_device_selectable(_device("A" * 16, "UMI-V4-0099"))


def test_pipeline_filter_accepts_well_formed_unregistered_serial():
    device = _device("DEADBEEFDEADBEEF")

    assert grippers.is_gripper_device_selectable(device)
    assert grippers.is_gripper_device_selectable(device, pipeline="v4")


def test_pipeline_filter_rejects_unusable_unregistered_identifier():
    assert not grippers.is_gripper_device_selectable(_device("serial-with-no-board-id"))


def test_raw_serial_resolves_to_calibration_port(monkeypatch):
    from umi.common.hardware_sdk.sensor.uart_reader import UartSensorReader

    seen = []

    def _find(serial):
        seen.append(serial)
        return "/dev/ttyACM1"

    monkeypatch.setattr(UartSensorReader, "find_port_by_serial", staticmethod(_find))

    assert (
        grippers.resolve_tty_path_for_device_name("deadbeefdeadbeef") == "/dev/ttyACM1"
    )
    assert seen == ["DEADBEEFDEADBEEF"]


def test_raw_identifier_normalization_is_case_insensitive():
    assert (
        grippers.normalize_umi_gripper_identifier("  deadbeefdeadbeef  ")
        == "DEADBEEFDEADBEEF"
    )
    assert grippers.normalize_umi_gripper_identifier(" UMI-V4-0001 ") == "UMI-V4-0001"


def test_v4_pipeline_uses_the_v4_registry(monkeypatch):
    monkeypatch.setattr(grippers, "UMI_V4_GRIPPERS", {"UMI-V4-0001": "D" * 16})

    assert grippers.list_umi_gripper_device_names_for_pipeline(pipeline="v4") == [
        "UMI-V4-0001"
    ]
    assert grippers.is_gripper_identifier_selectable("UMI-V4-0001", pipeline="v4")
    assert grippers.is_gripper_identifier_selectable("d" * 16, pipeline="v4")


def test_v4_name_round_trips_through_bimanual_data_dir(monkeypatch):
    """A `UMI-V4-NNNN` name must parse through the name regex: the generic
    16-char cut has no inverse, so falling through to it would leave the
    session folder unable to name its grippers."""
    monkeypatch.setattr(
        grippers,
        "UMI_V4_GRIPPERS",
        {"UMI-V4-0001": "A" * 16, "UMI-V4-0002": "B" * 16},
    )

    data_dir = config_utils.build_data_dir(
        "2026-10-19",
        "debug_task",
        "bimanual",
        left_gripper_num="UMI-V4-0001",
        right_gripper_num="UMI-V4-0002",
        session_uid="91e98c89fccb",
    )

    assert data_dir.endswith(
        "2026-10-19_umiV4_0001L_umiV4_0002R_debug_task_91e98c89fccb"
    )
    assert grippers.device_name_from_path_segment("V4_0001") == "UMI-V4-0001"
    assert grippers.device_name_from_path_segment("V4_0002") == "UMI-V4-0002"

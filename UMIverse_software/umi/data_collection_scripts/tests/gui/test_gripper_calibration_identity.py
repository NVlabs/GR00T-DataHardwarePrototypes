# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw gripper identity must survive the GUI-to-calibration handoff."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.umi_gripper_example import _uart_reader_kwargs  # noqa: E402


def test_calibration_example_uses_named_reader_without_raw_port():
    assert _uart_reader_kwargs("/dev/ttyACM1", "deadbeefdeadbeef") == {
        "uart_device_name": "DEADBEEFDEADBEEF",
        "baudrate": 1_000_000,
    }


def test_calibration_example_keeps_port_only_path_for_legacy_callers():
    assert _uart_reader_kwargs("/dev/ttyACM1", None) == {
        "port": "/dev/ttyACM1",
        "baudrate": 1_000_000,
    }


def test_generic_port_only_reader_still_rejects_unregistered_serial(monkeypatch):
    from umi.common.hardware_sdk.sensor import uart_reader

    reader = object.__new__(uart_reader.UartSensorReader)
    reader.port = "/dev/ttyACM1"
    reader.uart_device_name = None
    monkeypatch.setattr(reader, "get_serial_for_port", lambda _port: "C" * 16)
    monkeypatch.setattr(uart_reader, "UMI_V4_GRIPPERS", {})

    with pytest.raises(AssertionError, match="Unregistered device"):
        reader._verify_device_serial()


def test_generic_port_only_reader_accepts_a_registered_serial(monkeypatch):
    from umi.common.hardware_sdk.sensor import uart_reader

    serial = "C" * 16
    reader = object.__new__(uart_reader.UartSensorReader)
    reader.port = "/dev/ttyACM1"
    reader.uart_device_name = None
    monkeypatch.setattr(reader, "get_serial_for_port", lambda _port: serial)
    monkeypatch.setattr(
        uart_reader,
        "UMI_V4_GRIPPERS",
        {"UMI-V4-0001": serial},
    )

    reader._verify_device_serial()

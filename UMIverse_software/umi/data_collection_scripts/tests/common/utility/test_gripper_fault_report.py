# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from umi.common.utility import gripper_fault_report as report_module

PAYLOAD = {
    "check": "close_repeatability",
    "measured_deg": 7.31,
    "limit_deg": 2.0,
    "close_angles_deg": [132.1, 139.41, 133.02],
    "open_angle_deg": None,
    "attempts": 2,
    "device_name": "umi_gripper",
    "gear_radius_mm": 26.0,
}


@pytest.fixture
def station(monkeypatch):
    monkeypatch.setenv("UMI_STATION_ID", "gear-umi-04")


def test_build_adds_who_where_and_when(station):
    report = report_module.build(PAYLOAD, gripper_identifier="UMI-035")

    assert report["station"] == "gear-umi-04"
    assert report["gripper"] == "UMI-035"
    assert report["recorded_at"].endswith("Z")
    # The measurements survive intact.
    assert report["check"] == "close_repeatability"
    assert report["close_angles_deg"] == PAYLOAD["close_angles_deg"]


def test_build_falls_back_to_device_name(station):
    """Every gripper the example script builds is called "umi_gripper", so a run
    launched by port rather than by name still has to name something."""
    report = report_module.build(PAYLOAD, gripper_identifier=None)
    assert report["gripper"] == "umi_gripper"


def test_write_local_writes_parseable_json(monkeypatch, tmp_path, station):
    monkeypatch.setenv("UMI_OUTPUT_ROOT", str(tmp_path / "output"))
    report = report_module.build(PAYLOAD, gripper_identifier="UMI-035")

    path = report_module.write_local(report)

    assert path.parent == (tmp_path / "output").resolve() / "reports" / "gripper_faults"
    assert path.name.endswith("_UMI-035.json")
    assert json.loads(path.read_text())["measured_deg"] == pytest.approx(7.31)

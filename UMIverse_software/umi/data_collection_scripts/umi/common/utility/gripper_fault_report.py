# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The record a calibration leaves behind when the operator calls the gripper broken.

Separate from `umi/umi_device/umi_gripper.py` on purpose: a hardware driver has
no business knowing where reports are kept or the station's identity. The
driver raises `GripperHardwareFault` carrying measurements; this turns that into
a report and writes it to ``output/reports/gripper_faults/`` (see
`umi.common.utility.output_layout`).
"""

from __future__ import annotations

import json
import os
import pathlib
import socket
from datetime import datetime, timezone

from umi.common.utility.output_layout import output_root


def fault_dir() -> pathlib.Path:
    """Where fault reports go: beside the session reports in the output folder."""
    return output_root() / "reports" / "gripper_faults"


def _station() -> str:
    return os.environ.get("UMI_STATION_ID") or socket.gethostname()


def _timestamp() -> str:
    """UTC, with dashes for the time so it is safe as a filename."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def build(payload: dict, *, gripper_identifier: str | None = None) -> dict:
    """The exception's measurements plus who, where, and when.

    `gripper_identifier` is the registry name or USB serial the calibration was
    launched with; `device_name` alone will not do, because every gripper the
    example script builds is called "umi_gripper".
    """
    return {
        **payload,
        "gripper": gripper_identifier or payload.get("device_name") or "unknown",
        "station": _station(),
        "recorded_at": _timestamp(),
    }


def filename(report: dict) -> str:
    return f"{report['recorded_at']}_{report['gripper']}.json"


def write_local(report: dict) -> pathlib.Path:
    directory = fault_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename(report)
    path.write_text(json.dumps(report, indent=2))
    return path

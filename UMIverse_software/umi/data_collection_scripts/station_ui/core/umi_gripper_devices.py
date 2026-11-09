# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UMI gripper device names from the main UMI library (device_manager)."""

from __future__ import annotations

from umi.common.utility.gripper_pipeline import GripperPipeline
from umi.utils.device_manager import (
    UMI_V4_GRIPPERS,
    is_umi_gripper_serial,
    normalize_umi_gripper_identifier,
)

__all__ = [
    "device_name_from_path_segment",
    "gripper_path_segment_from_device_name",
    "is_gripper_device_selectable",
    "is_gripper_identifier_selectable",
    "is_umi_gripper_serial",
    "list_umi_gripper_device_names_for_pipeline",
    "normalize_umi_gripper_identifier",
    "resolve_tty_path_for_device_name",
]


def _gripper_registries() -> dict[GripperPipeline, dict]:
    """Build dynamically so tests can replace the module-level registries."""
    return {GripperPipeline.V4: UMI_V4_GRIPPERS}


def _registry_for_pipeline(*, pipeline=None):
    return _gripper_registries()[GripperPipeline.parse(pipeline)]


def list_umi_gripper_device_names_for_pipeline(*, pipeline=None) -> list[str]:
    """Return gripper names from the pipeline's registry."""
    return sorted(_registry_for_pipeline(pipeline=pipeline).keys())


def is_gripper_device_selectable(device, *, pipeline=None) -> bool:
    """Whether a scanned gripper is usable by the selected pipeline.

    A registered device must be in the pipeline's registry. An unregistered
    device has no registry entry to inspect, so only a well-formed raw Pico
    board id is admitted -- which keeps newly-flashed hardware selectable.
    """
    registry = _registry_for_pipeline(pipeline=pipeline)
    name = getattr(device, "registry_name", None)
    if name is not None:
        return name in registry
    return is_gripper_identifier_selectable(
        getattr(device, "device_id", ""), pipeline=pipeline
    )


def is_gripper_identifier_selectable(value: str, *, pipeline=None) -> bool:
    """Accept registered gripper names and well-formed raw serials."""
    text = normalize_umi_gripper_identifier(value)
    registry = _registry_for_pipeline(pipeline=pipeline)
    return text in registry or is_umi_gripper_serial(text)


def gripper_path_segment_from_device_name(name: str) -> str:
    """Folder token for data_dir.

    Examples:
        UMI-V4-0001 -> "V4_0001"

    Versioned names drop the ``UMI-`` prefix so ``umi{token}L`` renders
    without a duplicated ``UMI`` component.
    """
    if not name or not str(name).strip():
        return "1"
    s = str(name).strip()
    import re

    m = re.match(r"^UMI-(V4[-_]\d+)$", s, re.I)
    if m:
        return m.group(1).replace("-", "_").upper()
    return re.sub(r"[^0-9A-Za-z]", "_", s)[:16] or "1"


def device_name_from_path_segment(segment: str) -> str | None:
    """Map a data_dir token back to a registered gripper device name.

    Handles versioned segments such as ``V4_0001`` produced by
    :func:`gripper_path_segment_from_device_name`.
    """
    if not segment:
        return None
    s = str(segment).strip()
    import re

    m = re.match(r"^(V4)[-_](\d+)$", s, re.I)
    if m:
        candidate = f"UMI-{m.group(1).upper()}-{m.group(2)}"
        if candidate in UMI_V4_GRIPPERS:
            return candidate
    return None


def resolve_tty_path_for_device_name(device_name: str) -> str | None:
    """Resolve /dev/ttyACM* for gripper calibration (-p) from a logical UMI-xxx name."""
    if not device_name or not str(device_name).strip():
        return None
    try:
        from umi.common.hardware_sdk.sensor.uart_reader import UartSensorReader
        from umi.utils.device_manager import get_umi_gripper_serial

        sn = get_umi_gripper_serial(str(device_name).strip())
    except ValueError:
        return None
    return UartSensorReader.find_port_by_serial(sn)

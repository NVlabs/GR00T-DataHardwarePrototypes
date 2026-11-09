# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load the GEAR Neck's settings: drive ids, directions, gains, limits and the CAN link.

Everything that belongs to one particular neck lives in config/neck.yaml beside
the package; the rest of the package reads it through load_config() and
hard-codes none of it. The file documents each setting. Every key is required
and unknown keys are rejected, so a misspelled setting fails loudly instead of
being ignored.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from .gdk_protocol import ID_MAX, ID_MIN

# An editable install (uv sync) finds the file here; a regular pip install does
# not include it, so pass a copy with --config.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "neck.yaml"
JOINT_NAMES = ("yaw", "roll", "pitch")
INTERFACES = ("socketcan", "slcan")

_TOP_KEYS = ("joints", "max_speed_deg_s", "control_rate_hz", "feedback_timeout_s", "can")
_JOINT_KEYS = ("name", "id", "can_id", "reversed", "kp", "kd", "min_deg", "max_deg")
_CAN_KEYS = ("interface", "socketcan_channel", "slcan_port", "bitrate", "adapter_usb_id")


class ConfigError(ValueError):
    """The settings file is missing or invalid."""


class Ids(NamedTuple):
    id: int  # ID (register 7): what the drive takes frames on
    can_id: int  # CAN_ID (register 14): what it replies on


# Where every GDK drive leaves the factory.
FACTORY_IDS = Ids(1, 0)


@dataclass(frozen=True)
class Joint:
    name: str
    ids: Ids
    # The drive turns against the joint's positive direction, as mounted. Every
    # position, velocity and torque to and from it is negated.
    reversed: bool
    kp: float  # MIT stiffness, Nm/rad
    kd: float  # MIT damping, Nm*s/rad
    min_deg: float
    max_deg: float

    @property
    def direction(self) -> float:
        """Joint value = direction * drive value."""
        return -1.0 if self.reversed else 1.0


@dataclass(frozen=True)
class CanSettings:
    interface: str  # one of INTERFACES
    socketcan_channel: str
    slcan_port: str | None  # None finds the adapter by adapter_usb_id
    bitrate: int
    adapter_usb_id: tuple[int, int]  # USB vendor and product id


@dataclass(frozen=True)
class NeckConfig:
    joints: tuple[Joint, ...]  # in the order of the kinematic chain from the torso
    max_speed_deg_s: float
    control_rate_hz: float
    feedback_timeout_s: float
    can: CanSettings

    @property
    def joints_by_name(self) -> dict[str, Joint]:
        return {joint.name: joint for joint in self.joints}

    @property
    def neck_ids(self) -> dict[str, Ids]:
        return {joint.name: joint.ids for joint in self.joints}


def load_config(path: str | Path | None = None) -> NeckConfig:
    """Read and check a settings file; config/neck.yaml when ``path`` is None."""
    path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    try:
        text = path.read_text(encoding="utf8")
    except FileNotFoundError:
        hint = (
            "; a regular pip install does not include config/neck.yaml, so pass a copy "
            "of it with --config"
            if path == DEFAULT_CONFIG_PATH
            else ""
        )
        raise ConfigError(f"no settings file at {path}{hint}") from None
    try:
        return _parse(yaml.safe_load(text))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: not valid YAML: {exc}") from None
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from None


def resolve_config(config: NeckConfig | str | Path | None) -> NeckConfig:
    """A NeckConfig as given, or loaded from a path (None for the default file)."""
    return config if isinstance(config, NeckConfig) else load_config(config)


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        help="settings file (default: config/neck.yaml beside the package)",
    )


def _parse(data: Any) -> NeckConfig:
    top = _mapping(data, "the file", _TOP_KEYS)
    joints = _parse_joints(top["joints"])
    return NeckConfig(
        joints=joints,
        max_speed_deg_s=_number(top["max_speed_deg_s"], "max_speed_deg_s", positive=True),
        control_rate_hz=_number(top["control_rate_hz"], "control_rate_hz", positive=True),
        feedback_timeout_s=_number(top["feedback_timeout_s"], "feedback_timeout_s", positive=True),
        can=_parse_can(top["can"]),
    )


def _parse_joints(data: Any) -> tuple[Joint, ...]:
    if not isinstance(data, list):
        raise ConfigError("joints: expected a list with one entry per joint")
    joints = []
    for index, entry in enumerate(data):
        where = f"joints[{index}]"
        fields = _mapping(entry, where, _JOINT_KEYS)
        name = fields["name"]
        if name not in JOINT_NAMES:
            raise ConfigError(
                f"{where}.name: expected one of {', '.join(JOINT_NAMES)}, not {name!r}"
            )
        where = f"joints.{name}"
        joint = Joint(
            name=name,
            ids=Ids(
                _integer(fields["id"], f"{where}.id", ID_MIN, ID_MAX),
                _integer(fields["can_id"], f"{where}.can_id", ID_MIN, ID_MAX),
            ),
            reversed=_boolean(fields["reversed"], f"{where}.reversed"),
            kp=_number(fields["kp"], f"{where}.kp", non_negative=True),
            kd=_number(fields["kd"], f"{where}.kd", non_negative=True),
            min_deg=_number(fields["min_deg"], f"{where}.min_deg"),
            max_deg=_number(fields["max_deg"], f"{where}.max_deg"),
        )
        if joint.min_deg >= joint.max_deg:
            raise ConfigError(f"{where}: min_deg must be below max_deg")
        joints.append(joint)
    names = [joint.name for joint in joints]
    if sorted(names) != sorted(JOINT_NAMES):
        raise ConfigError(f"joints: expected each of {', '.join(JOINT_NAMES)} once, got {names}")
    every_id = [value for joint in joints for value in joint.ids]
    duplicates = sorted({value for value in every_id if every_id.count(value) > 1})
    if duplicates:
        raise ConfigError(f"joints: every ID and CAN_ID must differ; {duplicates} repeat")
    return tuple(joints)


def _parse_can(data: Any) -> CanSettings:
    fields = _mapping(data, "can", _CAN_KEYS)
    interface = fields["interface"]
    if interface not in INTERFACES:
        raise ConfigError(
            f"can.interface: expected one of {', '.join(INTERFACES)}, not {interface!r}"
        )
    usb_id = fields["adapter_usb_id"]
    if not isinstance(usb_id, list) or len(usb_id) != 2:
        raise ConfigError("can.adapter_usb_id: expected [vendor id, product id]")
    slcan_port = fields["slcan_port"]
    if slcan_port is not None:
        slcan_port = _string(slcan_port, "can.slcan_port")
    return CanSettings(
        interface=interface,
        socketcan_channel=_string(fields["socketcan_channel"], "can.socketcan_channel"),
        slcan_port=slcan_port,
        bitrate=_integer(fields["bitrate"], "can.bitrate", 1, 10_000_000),
        adapter_usb_id=(
            _integer(usb_id[0], "can.adapter_usb_id[0]", 0, 0xFFFF),
            _integer(usb_id[1], "can.adapter_usb_id[1]", 0, 0xFFFF),
        ),
    )


def _mapping(data: Any, where: str, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping with {', '.join(keys)}")
    missing = [key for key in keys if key not in data]
    if missing:
        raise ConfigError(f"{where}: missing {', '.join(missing)}")
    unknown = [str(key) for key in data if key not in keys]
    if unknown:
        raise ConfigError(f"{where}: unknown {', '.join(unknown)}; expected {', '.join(keys)}")
    return data


def _number(value: Any, where: str, *, positive: bool = False, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}: expected a number, not {value!r}")
    if positive and value <= 0:
        raise ConfigError(f"{where}: must be above 0")
    if non_negative and value < 0:
        raise ConfigError(f"{where}: must not be negative")
    return float(value)


def _integer(value: Any, where: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}: expected a whole number, not {value!r}")
    if not low <= value <= high:
        raise ConfigError(f"{where}: {value} is outside {low}-{high}")
    return value


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where}: expected true or false, not {value!r}")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where}: expected a name, not {value!r}")
    return value

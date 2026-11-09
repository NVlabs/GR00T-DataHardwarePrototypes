# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Select cameras by iphone_udid or oak_device_id. Select an exoskeleton by
# its side-specific calibration file, whose device_serial binds it to the
# calibrated USB board.

import re
from pathlib import Path

import yaml

# Device kinds of the identity contract (SensorInterface.identity()). "mock"
# sources carry synthetic ids.
DEVICE_KINDS = (
    "iphone",
    "oak",
    "dexumi_exoskeleton",
    "mock",
)


def normalize_device_id(value) -> str | None:
    """Comparable form of a device id: ``str(value).strip().upper()``.

    Config values and SDK reads differ in type and case (USB serials are hex
    in either case), so every identity comparison goes through this.
    Empty -> None.
    """
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


_REPO_ROOT = Path(__file__).resolve().parents[2]
EXOSKELETON_CONFIG_DIR = _REPO_ROOT / "config" / "dexumi_exoskeleton"
# Exact form only: DEXUMI-Left or DEXUMI-Right. The name selects a calibration
# file, so no fuzzy matching — a lookalike spelling must fail loudly rather than
# silently select a different hand's calibration.
_EXO_DEVICE_NAME_RE = re.compile(r"^DEXUMI-(Left|Right)$")


def _match_exoskeleton_device_name(device_name: str) -> re.Match:
    match = _EXO_DEVICE_NAME_RE.match(device_name)
    if not match:
        raise ValueError(
            f"Invalid DexUMI exoskeleton device name: {device_name!r}. Expected "
            "the exact form 'DEXUMI-Left' or 'DEXUMI-Right'."
        )
    return match


def get_exoskeleton_config_path(device_name: str, require_exists: bool = True) -> str:
    """Canonical calibration config path for a DexUMI exoskeleton device name.

    Each device name maps to the config file of the same name:
    DEXUMI-Right -> config/dexumi_exoskeleton/DEXUMI-Right.yaml
    """
    _match_exoskeleton_device_name(device_name)
    config_path = EXOSKELETON_CONFIG_DIR / f"{device_name}.yaml"
    if require_exists and not config_path.exists():
        raise FileNotFoundError(
            f"No calibration config for device {device_name!r} at {config_path}. "
            f"Create it with: uv run examples/hand/exoskeleton/calibrate_exo.py "
            f"--device {device_name} --manual"
        )
    return str(config_path)


def get_exoskeleton_template_path(device_name: str) -> str:
    """Design template a glove's first calibration starts from.

    It holds the values shared by every glove of that side (directions, joint
    limits, fingertip map, sensor ids); calibration adds the measured ones:
    DEXUMI-Right -> config/dexumi_exoskeleton/templates/DEXUMI-Right.template.yaml
    """
    _match_exoskeleton_device_name(device_name)
    return str(EXOSKELETON_CONFIG_DIR / "templates" / f"{device_name}.template.yaml")


def read_device_serial(config_path) -> str | None:
    """The ``device_serial`` a calibration file records, or None when it has none."""
    with open(config_path) as f:
        data = yaml.safe_load(f)
    serial = data.get("device_serial") if isinstance(data, dict) else None
    return str(serial).strip() or None if serial is not None else None


def get_exoskeleton_serial(device_name: str) -> str:
    """USB serial of the board an exoskeleton's calibration was made on.

    Raises ValueError when the calibration is missing, unreadable or records
    no ``device_serial``.
    """
    config_path = get_exoskeleton_config_path(device_name, require_exists=False)
    try:
        serial = read_device_serial(config_path)
    except FileNotFoundError:
        raise ValueError(
            f"No calibration for exoskeleton {device_name!r} at {config_path}. "
            f"Calibrate it with its board connected: uv run "
            f"examples/hand/exoskeleton/calibrate_exo.py --device {device_name} --manual"
        ) from None
    except (OSError, yaml.YAMLError) as e:
        raise ValueError(f"Cannot read {config_path}: {e}") from e
    if serial is None:
        raise ValueError(
            f"{config_path} records no device_serial, so no board can be matched "
            f"to {device_name!r}. Calibrate it with its board connected: uv run "
            f"examples/hand/exoskeleton/calibrate_exo.py --device {device_name}"
        )
    return serial


def exoskeleton_serials() -> dict[str, str]:
    """``{device name: device_serial}`` of every exoskeleton calibration that records a board.

    Unreadable files and files without a ``device_serial`` are skipped.
    """
    serials = {}
    for path in sorted(EXOSKELETON_CONFIG_DIR.glob("DEXUMI-*.yaml")):
        if not _EXO_DEVICE_NAME_RE.match(path.stem):
            continue
        try:
            serial = read_device_serial(path)
        except (OSError, yaml.YAMLError):
            continue
        if serial is not None:
            serials[path.stem] = serial
    return serials

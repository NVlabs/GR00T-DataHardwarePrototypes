# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loader for the data check YAML configuration."""

import logging
from typing import Any, Dict, Optional

import yaml

from .constants import DEFAULT_CHECK_CONFIG_PATH

__all__ = ["load_check_config", "is_enabled", "check_enabled"]

logger = logging.getLogger(__name__)


def load_check_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load data check configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default path.

    Returns:
        Dict with check configuration parameters
    """
    if config_path is None:
        config_path = DEFAULT_CHECK_CONFIG_PATH

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded check config from: {config_path}")
    return config


def is_enabled(check_config: Dict[str, Any], section: str) -> bool:
    """Opt-in gate: a check section runs iff its block has `enabled: true`.

    Default is False: a missing block disables the check. The default UMI YAML
    sets `enabled: true` on every section it uses; rig-specific YAMLs opt in to
    only the checks that make sense for their hardware. A section no check
    reads (for example in an older session's snapshot) is ignored.

    `enabled` may also be a mapping of per-check switches (see `check_enabled`),
    which runs the section: the mapping says which checks inside it are on, not
    whether the section itself is.
    """
    block = check_config.get(section)
    if block is None:
        return False
    if isinstance(block, dict):
        enabled = block.get("enabled", False)
        if isinstance(enabled, dict):
            return True
        return bool(enabled)
    return bool(block)


def check_enabled(
    section_config: Dict[str, Any], name: str, default: bool = True
) -> bool:
    """Is one named check inside a section switched on?

    `enabled: true` runs every check in the section, so a YAML without
    per-check switches runs them all. A mapping overrides checks by name::

        gripper:
          enabled:
            movement: false   # jumps and range stay on
          jump_threshold_mm: 50.0

    A check the mapping does not name keeps `default`, which is True: the
    opposite of `is_enabled`. Sections are opt-in so hardware-specific YAMLs
    enable only the sections they support; individual checks must be opt-out,
    or adding a new switch would silently disable that check for every
    existing config.

    Thresholds stay flat beside `enabled` rather than moving under each check's
    name. The recorder snapshots this YAML into every session and re-checks read
    it back, so a threshold that moved would resolve to the code default instead
    of the value the session was originally checked against.
    """
    block = section_config.get("enabled", False)
    if isinstance(block, dict):
        return bool(block.get(name, default))
    return bool(block)

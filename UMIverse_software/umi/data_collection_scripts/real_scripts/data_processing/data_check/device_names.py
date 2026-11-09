# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-name consistency checks (per episode and at recording-config level).

Expected device names come from the session's own recording config, not from a
fixed list of roles: a rig may carry optional head or bottom cameras that no
shipped config lists, and a fixed role list would quarantine every episode of
such a session. A data-check config may still pin the expected names explicitly
(see ``_VIDEO_NAME_KEYS`` / ``_NUMERIC_NAME_KEYS``): older sessions whose
session-local ``data_check_config.yaml`` carries a role list re-check the same
way, and any rig that needs to pin names can.
"""

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

__all__ = [
    "declared_device_names",
    "expected_device_names",
    "check_config_device_names",
    "check_device_name_consistency",
]


# Video and numeric required-name overrides are independent. An omitted
# category still derives its expected names from the recording config.
_VIDEO_NAME_KEYS = ("left_video_sources", "right_video_sources")
_NUMERIC_NAME_KEYS = ("left_numeric_sources", "right_numeric_sources")


def declared_device_names(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Device names the recording config declares, as (video, numeric).

    A multi-stream ``oak_device`` block declares its per-stream names, not the
    device's own name. Both config shapes resolve to the same names: the
    repository template carries a ``streams`` list, while the config.yaml saved
    into a session has already been rewritten by ``create_video_recorders`` into
    one entry per stream (same ``type``, ``name`` = stream name).
    """
    video: List[str] = []
    for source in config.get("video_sources") or []:
        streams = source.get("streams")
        if streams:
            video.extend(s["name"] for s in streams if s.get("name"))
        elif source.get("name"):
            video.append(source["name"])

    numeric = [
        source["name"]
        for source in config.get("numeric_sources") or []
        if source.get("name")
    ]
    return video, numeric


def _name_list(block: Dict[str, Any], key: str) -> List[str]:
    """Read `block[key]` as a list of device names ([] when unset).

    A scalar would iterate one character at a time, turning the expected set into
    gibberish and quarantining the session with an unreadable report, so it is
    rejected outright.
    """
    value = block.get(key)
    if not value:
        return []
    if isinstance(value, str):
        raise ValueError(
            f"device_name_consistency.{key} must be a list of device names, not "
            f"the string {value!r} — write it as a YAML list."
        )
    return list(value)


def _pinned_names(block: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[Set[str]]:
    """Names pinned by the check config for one category, or None if unpinned."""
    names = {name for key in keys for name in _name_list(block, key)}
    return names or None


def expected_device_names(config: Dict[str, Any]) -> Dict[str, Tuple[str, ...]]:
    """Get the device names this session is expected to have recorded.

    Video and numeric names are decided independently: each category is pinned if
    the check config names it, and derived from the session's recording config
    otherwise. Pinning one category must not blank the other — doing so would
    make every device of the unpinned category an extra device and quarantine the
    whole session.

    Args:
        config: Recording config dict. ``dispatch`` layers the data-check
            config's ``device_name_consistency`` block onto a copy of it, so an
            explicit name list authored in the check YAML shows up here.

    Returns:
        Dict with (sorted tuples, so the JSON report is deterministic):
            - video_sources: Required video device names
            - numeric_sources: Required numeric device names
            - all_devices: Combined required device names
            - optional_devices: Names allowed but not required
    """
    block = config.get("device_name_consistency")
    if not isinstance(block, dict):
        block = {}
    optional = set(_name_list(block, "optional_devices"))

    video = _pinned_names(block, _VIDEO_NAME_KEYS)
    numeric = _pinned_names(block, _NUMERIC_NAME_KEYS)
    if video is None or numeric is None:
        declared_video, declared_numeric = declared_device_names(config)
        if video is None:
            video = set(declared_video)
        if numeric is None:
            numeric = set(declared_numeric)

    return {
        "video_sources": tuple(sorted(video)),
        "numeric_sources": tuple(sorted(numeric)),
        "all_devices": tuple(sorted(video | numeric)),
        "optional_devices": tuple(sorted(optional)),
    }


def check_config_device_names(config: Dict[str, Any]) -> Dict[str, Any]:
    """Check that every device name in config.yaml is an expected name.

    Only meaningful when the expected names are pinned by a data-check config:
    when they are derived from the recording config itself the two sets agree by
    construction, and the check degenerates into a no-op. That is intended —
    without a canonical role list there is nothing to validate config names
    against.

    Args:
        config: Recording config dict

    Returns:
        Dict with:
            - passed: bool - True if all config device names are expected
            - expected_devices: List of expected device names
            - config_devices: List of device names found in config
            - mismatched_names: Config device names that don't match expected
            - missing_from_config: Expected names not in config (info only)
            - issues: List of issue descriptions
    """
    expected = expected_device_names(config)
    expected_devices = set(expected["all_devices"])

    # Extract configured and expected names identically when a source has both
    # name and streams fields.
    declared_video, declared_numeric = declared_device_names(config)
    config_devices = declared_video + declared_numeric
    config_devices_set = set(config_devices)

    # Find mismatches - only devices not in expected or optional list are errors
    allowed_devices = expected_devices | set(expected["optional_devices"])
    mismatched_names = config_devices_set - allowed_devices
    missing_from_config = expected_devices - config_devices_set

    issues = []
    if mismatched_names:
        issues.append(
            f"Config has non-standard device names: {sorted(mismatched_names)}"
        )
    # missing_from_config is informational only, not a failure

    return {
        "passed": len(mismatched_names) == 0,
        "expected_devices": sorted(expected_devices),
        "config_devices": sorted(config_devices),
        "mismatched_names": sorted(mismatched_names),
        "missing_from_config": sorted(missing_from_config),
        "issues": issues,
    }


def check_device_name_consistency(
    config: Dict[str, Any],
    recorded_devices: Iterable[str],
) -> Dict[str, Any]:
    """Check the devices an episode recorded against the ones it declared.

    Strict in both directions: an expected device the episode never recorded is
    a failure, and a recorded device the config never declared is a failure.
    Names listed under ``optional_devices`` are allowed but not required.

    Args:
        config: Recording config dict (with the data-check config's
            ``device_name_consistency`` block layered on, if any)
        recorded_devices: Device names found in the episode (the zarr group keys)

    Returns:
        Dict with:
            - passed: bool - True if the recorded devices match the expected set
            - expected_devices: List of expected device names
            - actual_devices: List of device names found in episode
            - missing_devices: Expected devices not in episode (errors)
            - extra_devices: Devices in episode but not expected (errors)
            - issues: List of issue descriptions
    """
    expected = expected_device_names(config)
    expected_devices = set(expected["all_devices"])

    actual_devices = set(recorded_devices)

    missing_devices = expected_devices - actual_devices
    allowed_devices = expected_devices | set(expected["optional_devices"])
    extra_devices = actual_devices - allowed_devices

    issues = []

    if extra_devices:
        issues.append(f"Non-standard device names: {sorted(extra_devices)}")

    if missing_devices:
        issues.append(f"Missing device names: {sorted(missing_devices)}")

    return {
        "passed": len(extra_devices) == 0 and len(missing_devices) == 0,
        "expected_devices": sorted(expected_devices),
        "actual_devices": sorted(actual_devices),
        "missing_devices": sorted(missing_devices),
        "extra_devices": sorted(extra_devices),
        "video_sources_expected": expected["video_sources"],
        "numeric_sources_expected": expected["numeric_sources"],
        "issues": issues,
    }

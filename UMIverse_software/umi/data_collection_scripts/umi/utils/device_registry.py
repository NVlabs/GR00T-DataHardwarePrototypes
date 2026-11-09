# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device registry: friendly device names -> hardware serials, from a local file.

Load order (the first file that exists wins):

1. ``$UMI_DEVICE_REGISTRY_FILE``
2. ``config/devices.json`` at the repository root -- this station's own registry
   (git-ignored; written by :func:`save_registry`)
3. ``umi/utils/devices_fallback.json`` -- the bundled registry, empty so a new
   station starts with no devices

Raw hardware serials work without any registry entry (see ``device_manager``).
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from umi.common.utility.output_layout import repo_root
from umi.utils.device_types import (
    BUCKET_KEYS,
    CAMERA_KEYS,
    GRIPPER_KEYS,
)

logger = logging.getLogger(__name__)

ENV_REGISTRY_FILE = "UMI_DEVICE_REGISTRY_FILE"
_FALLBACK_FILE = Path(__file__).resolve().parent / "devices_fallback.json"
_EMPTY = {"version": 0, "updated_at": "1970-01-01T00:00:00Z"}

_REGISTRY: Optional[dict] = None


def local_registry_path() -> Path:
    """This station's own registry file, and where :func:`save_registry` writes."""
    env = os.environ.get(ENV_REGISTRY_FILE, "").strip()
    if env:
        return Path(env).expanduser()
    return repo_root() / "config" / "devices.json"


def registry_path() -> Path:
    """The file the registry is read from: the station's own, else the snapshot."""
    path = local_registry_path()
    return path if path.is_file() else _FALLBACK_FILE


def load_registry() -> dict:
    """Read the registry file. Returns an empty registry if it cannot be read.

    Returns:
        Registry dict with keys: version, updated_at, iphone_devices,
        oak_devices, umi_v4_grippers, and device_metadata.
    """
    path = registry_path()
    try:
        data = json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001 - a broken file must not block boot
        logger.warning("Device registry %s unreadable (%s); using none", path, e)
        return dict(_EMPTY)
    if path == _FALLBACK_FILE:
        logger.info("Using the bundled device registry snapshot (%s)", path)
    return data


def save_registry(data: dict) -> Path:
    """Write *data* as this station's registry (atomically) and make it active."""
    global _REGISTRY
    problems = validate_registry(data)
    if problems:
        raise ValueError("invalid device registry: " + "; ".join(problems))
    path = local_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    _REGISTRY = data
    return path


def _get_registry() -> dict:
    """The lazily-loaded registry."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = load_registry()
    return _REGISTRY


def reload() -> dict:
    """Re-read the registry file, replacing the in-memory copy."""
    global _REGISTRY
    _REGISTRY = load_registry()
    return _REGISTRY


def get_bucket(key: str) -> dict:
    """One `{name: serial}` section, empty if this snapshot predates it.

    `.get`, never `[]`: a registry file written before a bucket existed must
    still load, so a missing section reads as empty. A station whose file is a
    version behind must still boot.

    An undeclared key raises, though: a typo that read empty forever is how a
    renamed bucket goes unnoticed.
    """
    if key not in BUCKET_KEYS:
        raise KeyError(f"{key!r} is not a declared device bucket")
    value = _get_registry().get(key, {})
    return value if isinstance(value, dict) else {}


def get_iphone_devices() -> dict:
    return get_bucket("iphone_devices")


def get_oak_devices() -> dict:
    return get_bucket("oak_devices")


def get_umi_v4_grippers() -> dict:
    """V4 flexure grippers. Empty in a registry without the bucket."""
    return get_bucket("umi_v4_grippers")


def get_gripper_cameras() -> dict:
    """Which camera is bolted to which gripper: `{gripper_name: camera_name}`.

    An edge table, not a nesting. Putting the camera inside the gripper's own
    entry would make `umi_v4_grippers` a `dict[str, dict]`, and several runtime
    paths compare those values as strings: `UartSensorReader` checks that the
    port's serial is in `.values()`, `device_discovery` reverse-scans by value
    to recover the friendly name, and
    `config_assembler.canonical_device_identity` would casefold a dict into
    its repr and go on comparing without raising.

    Both sides are friendly names. Absent, or absent for one gripper, means no
    camera is linked yet.

    Both ends are folded to the registry's own spelling on read, so a table
    hand-edited as `umi-v4-0008` still binds. Without that this map would be
    the module's only case-sensitive surface: `camera_for_gripper` looks up the
    bucket's canonical spelling, so a table differing only in case would report
    "no camera bound" with no error and no log.

    An entry naming a device in no bucket is passed through as written rather
    than dropped, so a dangling pairing stays visible to `validate_registry`
    instead of vanishing into an empty result.
    """
    value = _get_registry().get("gripper_cameras", {})
    if not isinstance(value, dict):
        return {}
    # Build the identity index once; rebuilding it for every entry would make
    # validation quadratic.
    index = _name_index()
    out = {}
    for gripper, camera in value.items():
        if not isinstance(gripper, str) or not isinstance(camera, str):
            continue
        # Unresolvable names pass through as written rather than vanishing --
        # a dangling entry must stay visible to validate_registry.
        out[index.get(gripper.strip().casefold(), gripper)] = index.get(
            camera.strip().casefold(), camera
        )
    return out


def _name_index() -> dict:
    """Case-folded name and serial -> the stored spelling of the name.

    Built over every bucket at once so a caller does not have to know which
    one a device is in.

    Rebuilt per call, deliberately not cached: `reload()` can replace the
    registry at any time, and every test that monkeypatches `_REGISTRY` would
    otherwise read a stale index. A registry is a few hundred entries, and
    this is one pass over it against `canonical_device_identity`'s seven
    separate scans (`station_ui/core/config_assembler.py`).
    """
    index: dict[str, str] = {}
    for key in BUCKET_KEYS:
        for name, serial in get_bucket(key).items():
            if not isinstance(name, str):
                continue
            index.setdefault(name.casefold(), name)
            if isinstance(serial, str) and serial:
                index.setdefault(serial.casefold(), name)
    return index


def name_for_identifier(value) -> Optional[str]:
    """The canonical registry name for a name or a raw hardware serial.

    Returns None for anything in no bucket: an unregistered device has no
    name, and inventing one would put a made-up name into a config.
    """
    text = str(value or "").strip()
    if not text:
        return None
    return _name_index().get(text.casefold())


def camera_for_gripper(value) -> Optional[str]:
    """The camera bound to this gripper, or None.

    None for an unbound gripper, an unregistered serial and junk alike -- by the
    same path.
    """
    name = name_for_identifier(value)
    if name is None:
        return None
    return get_gripper_cameras().get(name)


def get_registry_info() -> dict:
    """Return metadata about the current registry (version, updated_at, source)."""
    reg = _get_registry()
    path = registry_path()
    return {
        "version": reg.get("version", 0),
        "updated_at": reg.get("updated_at", "unknown"),
        "source": "bundled" if path == _FALLBACK_FILE else "local",
        "path": str(path),
    }


def validate_registry(data: dict) -> list[str]:
    """Every problem in a registry blob, as sentences. Empty means clean.

    Pure: it neither raises nor mutates, so a caller (``save_registry``) can
    report everything wrong and then decide, rather than stop at the first bad
    row.
    """
    problems: list[str] = []
    if not isinstance(data, dict):
        return ["the registry is not a JSON object"]

    names: dict[str, str] = {}  # folded name -> bucket
    serials: dict[str, tuple] = {}  # folded serial -> (bucket, "bucket/name")
    # Reuse only validated buckets; rereading raw values could let malformed
    # non-iterables bypass the shape check.
    buckets: dict[str, dict] = {}

    for key in BUCKET_KEYS:
        bucket = data.get(key, {})
        if not isinstance(bucket, dict):
            problems.append(f"{key} is not an object of name -> serial")
            continue
        buckets[key] = bucket
        for name, serial in bucket.items():
            if not isinstance(name, str) or not isinstance(serial, str):
                problems.append(f"{key}: {name!r} -> {serial!r} is not name -> serial")
                continue
            folded = name.casefold()
            if folded in names:
                problems.append(f"{name} is declared in both {names[folded]} and {key}")
            names[folded] = key
            owner = f"{key}/{name}"
            folded_serial = serial.casefold()
            # Allow aliases within one device-kind bucket. Reject the same serial
            # claimed by different hardware kinds.
            if folded_serial in serials:
                prev_key, prev_owner = serials[folded_serial]
                if prev_key != key:
                    problems.append(
                        f"serial {serial} is claimed by both {prev_owner} and {owner}"
                    )
            else:
                serials[folded_serial] = (key, owner)

    gripper_names = {
        name.casefold()
        for key in GRIPPER_KEYS
        for name in buckets.get(key, {})
        if isinstance(name, str)
    }
    camera_names = {
        name.casefold()
        for key in CAMERA_KEYS
        for name in buckets.get(key, {})
        if isinstance(name, str)
    }
    # Exact spellings too: the folded sets answer "does this device exist",
    # these answer "is it written the way the registry writes it".
    gripper_exact = {
        name
        for key in GRIPPER_KEYS
        for name in buckets.get(key, {})
        if isinstance(name, str)
    }
    camera_exact = {
        name
        for key in CAMERA_KEYS
        for name in buckets.get(key, {})
        if isinstance(name, str)
    }

    links = data.get("gripper_cameras", {})
    # Validate present sections by type, not truthiness: [], null and empty
    # strings are malformed even though falsey. Absent or empty mapping sections
    # are valid.
    if not isinstance(links, dict):
        problems.append("gripper_cameras is not an object of gripper -> camera")
        links = {}

    bound_cameras: dict[str, str] = {}
    for gripper, camera in (links or {}).items():
        if not isinstance(gripper, str) or not isinstance(camera, str):
            problems.append(
                f"gripper_cameras: {gripper!r} -> {camera!r} is not name -> name"
            )
            continue
        if gripper.casefold() not in gripper_names:
            problems.append(
                f"gripper_cameras: {gripper} is not a gripper in any bucket"
            )
        elif gripper not in gripper_exact:
            # Report alias spelling drift here; runtime normalization otherwise conceals
            # it.
            problems.append(
                f"gripper_cameras: {gripper} is not the canonical spelling of "
                "a registered gripper"
            )
        if camera.casefold() not in camera_names:
            problems.append(f"gripper_cameras: {camera} is not a camera in any bucket")
        elif camera not in camera_exact:
            problems.append(
                f"gripper_cameras: {camera} is not the canonical spelling of "
                "a registered camera"
            )
        folded_camera = camera.casefold()
        if folded_camera in bound_cameras:
            problems.append(
                f"gripper_cameras: {camera} is bound to both "
                f"{bound_cameras[folded_camera]} and {gripper}"
            )
        else:
            bound_cameras[folded_camera] = gripper

    return problems

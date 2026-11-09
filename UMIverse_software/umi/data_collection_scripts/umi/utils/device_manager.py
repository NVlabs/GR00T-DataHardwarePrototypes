# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Device-name mappings delegate to device_registry. Register devices in
# config/devices.json; retain the public dict and lookup APIs for callers.

from umi.utils.device_registry import (
    get_iphone_devices,
    get_oak_devices,
    get_umi_v4_grippers,
)


class _LazyDict(dict):
    """Dict that loads from the registry, and reloads when it is replaced.

    Tracks the registry snapshot it was filled from and refills when that
    object changes identity, so a long-lived process sees
    `device_registry.reload()`, and the order in which tests touch a map
    never decides its contents.

    Identity rather than equality, because two successive snapshots can compare
    equal while a reload has genuinely happened, and comparing whole registries
    on every attribute access would cost more than the load it is avoiding.
    A reference to the snapshot is held rather than its `id()`, since an id can
    be reused by a later object once the original is collected.
    """

    _UNLOADED = object()

    def __init__(self, loader):
        super().__init__()
        self._loader = loader
        self._source = self._UNLOADED

    def invalidate(self):
        """Force the next access to reload.

        Rarely needed, since a replaced registry is noticed on its own, but
        supported: callers that mutate the current snapshot in place change
        nothing about its identity, and this is how they say so.
        """
        self._source = self._UNLOADED
        self.clear()

    def _ensure_loaded(self):
        from umi.utils import device_registry

        # Resolve the registry lazily instead of loading station state at import.
        current = device_registry._get_registry()
        if self._source is current:
            return
        # Replace the cache instead of merging so removed registry entries disappear.
        self.clear()
        self.update(self._loader())
        self._source = current

    def _resolve_key(self, key):
        self._ensure_loaded()
        if not isinstance(key, str):
            return key

        folded_key = key.casefold()
        matches = [
            stored_key
            for stored_key in dict.__iter__(self)
            if isinstance(stored_key, str) and stored_key.casefold() == folded_key
        ]
        if len(matches) > 1:
            variants = ", ".join(matches)
            raise ValueError(
                f"Ambiguous device name '{key}'; registry contains "
                f"case-only duplicates: {variants}"
            )
        return matches[0] if matches else key

    def __getitem__(self, key):
        return super().__getitem__(self._resolve_key(key))

    def __contains__(self, key):
        return super().__contains__(self._resolve_key(key))

    def __iter__(self):
        self._ensure_loaded()
        return super().__iter__()

    def __len__(self):
        self._ensure_loaded()
        return super().__len__()

    def keys(self):
        self._ensure_loaded()
        return super().keys()

    def values(self):
        self._ensure_loaded()
        return super().values()

    def items(self):
        self._ensure_loaded()
        return super().items()

    def get(self, key, default=None):
        return super().get(self._resolve_key(key), default)

    def __repr__(self):
        self._ensure_loaded()
        return super().__repr__()

    def __bool__(self):
        self._ensure_loaded()
        return dict.__bool__(self)


IPHONE_DEVICES = _LazyDict(get_iphone_devices)
OAK_DEVICES = _LazyDict(get_oak_devices)
UMI_V4_GRIPPERS = _LazyDict(get_umi_v4_grippers)


def _find_device_name(devices: dict, device_name: str) -> str | None:
    """Return the stored spelling of a case-insensitive device name."""
    if not isinstance(device_name, str):
        return None

    folded_name = device_name.casefold()
    matches = [name for name in devices if name.casefold() == folded_name]
    if len(matches) > 1:
        variants = ", ".join(matches)
        raise ValueError(
            f"Ambiguous device name '{device_name}'; registry contains "
            f"case-only duplicates: {variants}"
        )
    return matches[0] if matches else None


def get_iphone_udid(device_name: str) -> str:
    """Get iPhone UDID by device name."""
    stored_name = _find_device_name(IPHONE_DEVICES, device_name)
    if stored_name is None:
        raise ValueError(
            f"Unknown iPhone device: {device_name}. "
            f"Available: {list(IPHONE_DEVICES.keys())}"
        )
    return IPHONE_DEVICES[stored_name]


def _looks_like_device_id(value: str) -> bool:
    """A depthai device id is a long hex string; registry names are not.

    e.g. an 18-character hex id vs 'oak-1'.
    """
    text = (value or "").strip()
    return len(text) >= 12 and all(c in "0123456789abcdefABCDEF" for c in text)


def get_oak_device_id(device_name: str) -> str:
    """Registry name to device id. A raw device id passes through unchanged.

    A config may legitimately name hardware the registry does not know: a new
    module, or one missing from this station's registry. The Station Console
    stores the device id in that case, so rejecting it here would turn a
    config that saved fine into a crash at record time.
    """
    stored_name = _find_device_name(OAK_DEVICES, device_name)
    if stored_name is not None:
        return OAK_DEVICES[stored_name]
    if _looks_like_device_id(device_name):
        return device_name
    raise ValueError(
        f"Unknown OAK device: {device_name}. Available: {list(OAK_DEVICES.keys())}"
    )


def is_umi_gripper_serial(value: str) -> bool:
    """Return whether *value* is a raw RP2040 USB serial number.

    UMI firmware exposes the Pico's eight-byte board id as 16 hexadecimal
    characters.  Keeping this check strict prevents a misspelled registry name
    from being treated as a hardware id while still allowing newly-flashed
    grippers to be used before the shared registry is updated.
    """
    text = (value or "").strip()
    return len(text) == 16 and all(c in "0123456789abcdefABCDEF" for c in text)


def normalize_umi_gripper_identifier(value: str) -> str:
    """Strip an identifier and canonicalize raw hexadecimal serials."""
    text = (value or "").strip()
    return text.upper() if is_umi_gripper_serial(text) else text


def get_umi_gripper_serial(device_name: str) -> str:
    """Get a UMI gripper serial from a registry name or a raw serial.

    A raw serial is normalized to uppercase so it compares directly with the
    serial reported by pyserial.  This mirrors the raw-id pass-through used for
    OAK devices and lets an attached, unregistered gripper retain the same
    port-resolution and per-frame identity checks as a registered one.
    """
    stored_name = _find_device_name(UMI_V4_GRIPPERS, device_name)
    if stored_name is not None:
        return UMI_V4_GRIPPERS[stored_name]
    if is_umi_gripper_serial(device_name):
        return normalize_umi_gripper_identifier(device_name)
    raise ValueError(
        f"Unknown UMI gripper device: {device_name}. "
        f"Available: {list(UMI_V4_GRIPPERS.keys())}"
    )

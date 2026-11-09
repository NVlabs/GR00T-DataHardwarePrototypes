# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What buckets the device registry has, declared once.

Everything that needs to know what a bucket is derives from here: the getters,
the offline fallback and the test fixtures. Unknown top-level keys in a
registry file are passed through rather than rebuilt, so buckets this
repository does not declare survive a round trip.

Deliberately free of I/O and of any import from `device_registry`, so the
registry loader can import the table without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "BUCKET_KEYS",
    "CAMERA_KEYS",
    "COMMENT_FIELD",
    "DEVICE_TYPES",
    "GRIPPER_KEYS",
    "FACTS_KEY",
    "METADATA_KEY",
    "RELATION_KEYS",
    "DeviceType",
    "infer_bucket",
]


@dataclass(frozen=True)
class DeviceType:
    """One bucket of `{friendly_name: hardware_serial}` in the registry JSON."""

    key: str
    kind: str  # "camera" | "gripper"
    label: str
    name_prefixes: tuple[str, ...]


DEVICE_TYPES: tuple[DeviceType, ...] = (
    DeviceType("oak_devices", "camera", "OAK", ("oak-",)),
    DeviceType("iphone_devices", "camera", "iPhone", ("iphone",)),
    # Validate only supported registry buckets and pass unknown top-level keys
    # through unchanged for compatibility.
    DeviceType("umi_v4_grippers", "gripper", "UMI v4 gripper", ("UMI-V4-",)),
)

BUCKET_KEYS: tuple[str, ...] = tuple(entry.key for entry in DEVICE_TYPES)
CAMERA_KEYS: tuple[str, ...] = tuple(
    entry.key for entry in DEVICE_TYPES if entry.kind == "camera"
)
GRIPPER_KEYS: tuple[str, ...] = tuple(
    entry.key for entry in DEVICE_TYPES if entry.kind == "gripper"
)

# Sections that are not `{name: serial}` and must never be walked as one.
# `gripper_cameras` is `{gripper_name: camera_name}`; the two fact sections are
# `{device: {field: value}}`.
RELATION_KEYS: tuple[str, ...] = ("gripper_cameras",)

# Write per-device facts under the name-keyed metadata section. Read the
# compatibility section without rewriting it, since serial-keyed writers
# may prune entries they cannot resolve.
FACTS_KEY = "device_facts"
METADATA_KEY = "device_metadata"

# List metadata fields consumed by this repository; preserve unrecognized
# fields rather than treating this as a closed schema.
COMMENT_FIELD = "comment"


def infer_bucket(name: str) -> str | None:
    """Which bucket a friendly name belongs in, by its prefix.

    Longest prefix wins, never declaration order: when one declared prefix
    starts another, an order-sensitive rule would file every device under the
    shorter one.
    """
    text = str(name or "").casefold()
    if not text:
        return None
    best_key, best_len = None, 0
    for entry in DEVICE_TYPES:
        for prefix in entry.name_prefixes:
            folded = prefix.casefold()
            if text.startswith(folded) and len(folded) > best_len:
                best_key, best_len = entry.key, len(folded)
    return best_key

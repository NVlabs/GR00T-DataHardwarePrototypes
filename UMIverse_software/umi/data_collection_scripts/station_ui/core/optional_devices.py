# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional camera descriptors.

Head and bottom cameras are declared here and nowhere else. The configuration
page renders sections from these rows and the config assembler emits sources
from them, so neither inspects a gripper generation to decide what a rig may
carry.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# Placement is recorded metadata, with separate choices for head-mounted
# and table cameras. Store space-free tokens for downstream conversion;
# use labels only for display. Blank means unspecified.
PLACEMENTS_BY_SECTION: dict[str, tuple[str, ...]] = {
    "head": ("", "head", "neck", "static"),
    "bottom": ("", "table"),
}

# Accept persisted placement aliases without offering them for new selections.
LEGACY_PLACEMENTS: tuple[str, ...] = (
    "left",
    "right",
    "front",
    "rear",
    "overhead",
    "table",
)


@dataclass(frozen=True)
class OptionalDeviceType:
    id: str
    section: str
    device_kind: str
    source_type: str
    label: str
    # The name the camera is written under in the config, which is also the
    # handle the GUI keys its row on.
    fixed_name: str
    # None means unlimited. The limit is per descriptor, not per section, so
    # a head may carry one iPhone and one OAK at the same time.
    max_count: int | None = None


@dataclass(frozen=True)
class OptionalDeviceAssignment:
    type_id: str
    logical_name: str
    device: str = ""
    placement: str = ""


# Descriptor order determines Add-button and camera-row order; list head OAK first.
OPTIONAL_DEVICE_TYPES: tuple[OptionalDeviceType, ...] = (
    OptionalDeviceType(
        id="head_oak",
        section="head",
        device_kind="oak",
        source_type="oak_camera",
        label="Head OAK",
        fixed_name="head_oak_camera",
        max_count=1,
    ),
    OptionalDeviceType(
        id="head_iphone",
        section="head",
        device_kind="iphone",
        source_type="iphone_camera",
        label="Head iPhone",
        fixed_name="head_iphone_camera",
        max_count=1,
    ),
    # A bottom camera is a static, upward-facing OAK. Use the shared OAK source
    # block while keeping its placement distinct from head-mounted cameras.
    OptionalDeviceType(
        id="bottom_oak",
        section="bottom",
        device_kind="oak",
        source_type="oak_camera",
        label="Bottom OAK",
        fixed_name="bottom_oak_camera",
        max_count=1,
    ),
)

_BY_ID = {t.id: t for t in OPTIONAL_DEVICE_TYPES}


def get_optional_type(type_id: str) -> OptionalDeviceType:
    return _BY_ID[type_id]


def types_for_section(section: str) -> tuple[OptionalDeviceType, ...]:
    return tuple(t for t in OPTIONAL_DEVICE_TYPES if t.section == section)


def allocate_logical_name(type_id: str) -> str:
    """The name a new row of *type_id* is keyed on: its descriptor's fixed name.

    Cardinality is not enforced here: callers must check can_add() first, or a
    singleton head device will hand back its fixed name a second time.
    """
    return get_optional_type(type_id).fixed_name


def matches_logical_name(descriptor: OptionalDeviceType, name) -> bool:
    """True when *name* is a name this descriptor owns.

    The only way to recognise an optional camera in a config file written by an
    earlier session, and the guard the assembler applies to each assignment.
    """
    return str(name or "") == descriptor.fixed_name


def type_for_logical_name(name) -> OptionalDeviceType | None:
    """The descriptor that owns *name*, or None if no descriptor claims it."""
    for descriptor in OPTIONAL_DEVICE_TYPES:
        if matches_logical_name(descriptor, name):
            return descriptor
    return None


def can_add(type_id: str, existing: Sequence[OptionalDeviceAssignment]) -> bool:
    descriptor = get_optional_type(type_id)
    if descriptor.max_count is None:
        return True
    present = sum(1 for item in existing if item.type_id == type_id)
    return present < descriptor.max_count


# --------------------------------------------------------------- placement


@dataclass(frozen=True)
class PlacementOption:
    """One entry of the picker: what is submitted, what is read, what is on."""

    value: str
    label: str
    selected: bool


def placements_for_section(section: str) -> tuple[str, ...]:
    """What *section* offers. Raises rather than returning () on a typo, which
    would silently render a picker with no options at all."""
    try:
        return PLACEMENTS_BY_SECTION[section]
    except KeyError:
        raise ValueError(
            f"no placements declared for section {section!r} "
            f"(known: {sorted(PLACEMENTS_BY_SECTION)})"
        ) from None


def normalize_placement(value) -> str:
    """The stored token for anything the operator or an old draft may hand over.

    "left front" and "Left-Front" are both the same placement as `left_front`,
    and only one of the three is what gets written.
    """
    text = str(value or "").strip().lower().replace("-", " ")
    return "_".join(text.split())


def placement_label(value: str) -> str:
    """What the operator reads. The inverse of the underscore in the token."""
    return str(value or "").replace("_", " ")


def placement_options(section: str, current: str = "") -> tuple[PlacementOption, ...]:
    """The picker for one camera, with *current* selected.

    A value this section does not offer -- a head camera saved by an older
    version, or a hand-edited config -- is appended rather than dropped. An
    operator's existing choice disappearing out of the picker reads as the
    console having lost it, and re-saving the page would then silently rewrite
    the config.
    """
    offered = placements_for_section(section)
    current = normalize_placement(current)
    options = [
        PlacementOption(value, placement_label(value), value == current)
        for value in offered
    ]
    if current and current not in offered:
        options.append(
            PlacementOption(current, f"{placement_label(current)} (older option)", True)
        )
    return tuple(options)


def validate_placement(value: str, section: str) -> str:
    """The stored token for *value* on a camera in *section*.

    Per-section: a head camera cannot be given a side-only placement, so the
    value that reaches the config is one that describes where the camera
    actually is. Legacy values are the one exception -- see LEGACY_PLACEMENTS.
    """
    offered = placements_for_section(section)
    value = normalize_placement(value)
    if value in offered or value in LEGACY_PLACEMENTS:
        return value
    raise ValueError(
        f"Unknown placement {value!r} for a {section} camera; expected one of "
        f"{offered!r}"
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supported recording setups, as data.

A setup preset is one row: mode, gripper generation, and the immutable
template it starts from. Adding a supported setup means adding a row and a
validated template — never a branch in camera or config code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# One production template per preset. Keep reference and mock configs
# outside the preset template directory.
_TEMPLATE_DIR = ("config", "data_recording", "production")


# Use fixed pose-tracker tokens in paths and metadata so spelling differences
# cannot split reports for the same rig type.
POSE_TRACKER_IPHONE = "iphone"
POSE_TRACKER_VIVE = "vive"
# Write an explicit absent-pose token for each slot without a source.
POSE_TRACKER_NONE = "notracker"

POSE_TRACKERS = (POSE_TRACKER_IPHONE, POSE_TRACKER_VIVE, POSE_TRACKER_NONE)


@dataclass(frozen=True)
class SetupPreset:
    id: str
    label: str
    operation_mode: str
    gripper_generation: str
    template_name: str
    # Require each preset to name its wrist pose type explicitly and match
    # its template declarations.
    pose_tracker: str


SETUP_PRESETS: tuple[SetupPreset, ...] = (
    # V4 flexure gripper: 15 mm pinion and approximately 148 degrees of travel.
    # Each arm uses a CAM_B wrist sensor, USB trigger camera and iPhone pose.
    # CAM_A is absent; the reported CAM_C sensor does not stream.
    SetupPreset(
        id="bimanual_v4",
        label="Bimanual (iPhone)",
        operation_mode="bimanual",
        gripper_generation="v4",
        template_name="bimanual_v4.yaml",
        pose_tracker=POSE_TRACKER_IPHONE,
    ),
    # Same V4 rig as the row above, with the wrist pose coming from Vive pucks
    # instead of the iPhones.
    SetupPreset(
        id="bimanual_v4_vive",
        label="Bimanual (Vive)",
        operation_mode="bimanual",
        gripper_generation="v4",
        template_name="bimanual_v4_vive.yaml",
        pose_tracker=POSE_TRACKER_VIVE,
    ),
)

_BY_ID = {p.id: p for p in SETUP_PRESETS}


def preset_ids() -> tuple[str, ...]:
    return tuple(p.id for p in SETUP_PRESETS)


def get_preset(preset_id: str) -> SetupPreset:
    return _BY_ID[preset_id]


def template_for(preset: SetupPreset) -> str:
    return preset.template_name


def template_path(preset: SetupPreset, repo_root: Path) -> Path:
    return Path(repo_root).joinpath(*_TEMPLATE_DIR, template_for(preset))

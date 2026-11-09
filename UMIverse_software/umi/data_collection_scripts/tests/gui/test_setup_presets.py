# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib

import pytest

from station_ui.core.setup_presets import (
    SETUP_PRESETS,
    SetupPreset,
    get_preset,
    preset_ids,
    template_for,
    template_path,
)

REPO = pathlib.Path(__file__).resolve().parents[2]


def test_exactly_two_presets_in_display_order():
    assert preset_ids() == ("bimanual_v4", "bimanual_v4_vive")


def test_labels_leave_out_the_gripper_generation():
    # The station records only V4 grippers, so the labels do not say so.
    assert [p.label for p in SETUP_PRESETS] == [
        "Bimanual (iPhone)",
        "Bimanual (Vive)",
    ]


def test_capture_rate_is_not_a_preset_concern():
    # The OAK capture rate is the template's `fps`, so the preset has no rate to
    # carry. Guard against reintroducing one.
    assert not hasattr(get_preset("bimanual_v4"), "oak_fps")


def test_the_vive_v4_records_the_v4_rig_from_its_own_file():
    """The V4 rig with pucks for phones: same mode and gripper generation as
    `bimanual_v4`, its own template, and Vive as the pose tracker."""
    vive, plain = get_preset("bimanual_v4_vive"), get_preset("bimanual_v4")

    assert (vive.operation_mode, vive.gripper_generation) == ("bimanual", "v4")
    assert vive.operation_mode == plain.operation_mode
    assert vive.gripper_generation == plain.gripper_generation
    assert vive.template_name != plain.template_name
    assert vive.pose_tracker == "vive"


def test_every_template_exists_on_disk():
    for preset in SETUP_PRESETS:
        assert template_path(preset, REPO).is_file()
        assert template_path(preset, REPO).name == template_for(preset)


def test_presets_are_immutable():
    with pytest.raises(Exception):
        SETUP_PRESETS[0].label = "nope"


def test_unknown_preset_id_raises():
    with pytest.raises(KeyError):
        get_preset("bimanual_v9")


def test_adding_a_preset_needs_no_code_branch():
    # A preset is data: constructing one requires no registration call.
    extra = SetupPreset(
        id="bimanual_v9",
        label="Bimanual V9",
        operation_mode="bimanual",
        gripper_generation="v9",
        template_name="bimanual_v9.yaml",
        pose_tracker="iphone",
    )
    assert template_for(extra) == "bimanual_v9.yaml"

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pipeline version as an enum rather than a string and a boolean.

The enum is the one place a config's value is checked against, so adding a
version does not mean editing every module that compares versions. V4 is the
only version.

Run from project root:
    uv run pytest tests/common/test_gripper_pipeline.py -q
"""

import pytest

from umi.common.utility.gripper_pipeline import GripperPipeline

# --------------------------------------------------------------------------
# behaves as the string it serializes to
# --------------------------------------------------------------------------


def test_compares_equal_to_its_yaml_string():
    """Existing code does `gp == "v4"`; that must keep working."""
    assert GripperPipeline.V4 == "v4"


def test_str_is_the_bare_value():
    """So an f-string in a config or a log line reads 'v4', not the repr."""
    assert str(GripperPipeline.V4) == "v4"
    assert f"{GripperPipeline.V4}" == "v4"


def test_value_is_what_gets_written_to_yaml():
    assert GripperPipeline.V4.value == "v4"


# --------------------------------------------------------------------------
# parse: a config must open even when its version is odd
# --------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["v4", "V4", " v4 ", GripperPipeline.V4])
def test_parse_accepts_the_forms_a_config_can_hold(raw):
    assert GripperPipeline.parse(raw) is GripperPipeline.V4


def test_parse_falls_back_rather_than_raising():
    """Refusing to load would strand an operator on an unusable station."""
    assert GripperPipeline.parse("v99") is GripperPipeline.default()
    assert GripperPipeline.parse(None) is GripperPipeline.default()
    assert GripperPipeline.parse("") is GripperPipeline.default()


def test_a_retired_version_falls_back_to_the_default():
    """A config recorded before the repository went V4-only still opens."""
    for retired in ("v1", "v2", "v3"):
        assert GripperPipeline.parse(retired) is GripperPipeline.default()


def test_parse_honours_an_explicit_fallback():
    assert GripperPipeline.parse("v99", GripperPipeline.V4) is GripperPipeline.V4


def test_is_known_separates_a_real_version_from_a_typo():
    """parse() is silent by design; callers warn off is_known()."""
    assert GripperPipeline.is_known("v4")
    assert GripperPipeline.is_known(GripperPipeline.V4)
    assert not GripperPipeline.is_known("v99")
    assert not GripperPipeline.is_known("v3")
    assert not GripperPipeline.is_known(None)


def test_default_is_v4():
    assert GripperPipeline.default() is GripperPipeline.V4


# --------------------------------------------------------------------------
# the single source of truth
# --------------------------------------------------------------------------


def test_values_lists_every_member():
    assert GripperPipeline.values() == [m.value for m in GripperPipeline]
    assert GripperPipeline.values() == ["v4"]


def test_every_member_round_trips_through_parse():
    """Adding a member must not need any other edit to stay parseable."""
    for member in GripperPipeline:
        assert GripperPipeline.parse(member.value) is member


# --------------------------------------------------------------------------
# V4: the flexure gripper's geometry
# --------------------------------------------------------------------------


def test_v4_knows_its_gear_radius_and_open_travel_band():
    """The driver converts encoder degrees to mm with 2*pi*r*deg/360*2, so the
    radius is the one number a wrong value corrupts every distance with."""
    assert GripperPipeline.V4.gear_radius_mm == 15.0
    assert GripperPipeline.V4.open_travel_band_deg == (120.0, 175.0)
    lo, hi = GripperPipeline.V4.open_travel_band_deg
    # A V4 prototype opened ~148 deg; the band must hold that with margin on
    # both sides and stay under the 180 deg seam of get_signed_angle_offset.
    assert lo < 148.0 < hi
    assert hi < 180.0

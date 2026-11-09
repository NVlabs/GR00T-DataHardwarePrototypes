# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`create_umi_device`: the saved calibration wins, unless it is for another gear.

Run from project root:
    uv run pytest tests/umi_device/test_umi_device_factory.py -q
"""

from unittest.mock import MagicMock

import pytest

from umi.umi_device.umi_device_interface import create_umi_device
from umi.umi_device.umi_gripper import UmiGripper


def _write(tmp_path, body: str):
    path = tmp_path / "right_umi_gripper_config.yaml"
    path.write_text(body)
    return str(path)


LEGACY_V3_FILE = (
    "device_name: right_umi_gripper\n"
    "directions: '+'\n"
    "distance_zero_angle: 173.0\n"
    "gear_radius: 26.0\n"
)
V4_FILE = (
    "device_name: right_umi_gripper\n"
    "directions: '+'\n"
    "distance_zero_angle: 25.093\n"
    "gear_radius: 15.0\n"
    "gripper_generation: v4\n"
)


def test_a_v4_setup_refuses_a_file_calibrated_for_the_26_mm_gear(tmp_path):
    """A calibration saved for a V3 gear must not load into a V4 preset: the
    file's 26 mm radius would scale every V4 distance by 1.73 for the whole
    session, and nothing downstream could tell."""
    cfg = _write(tmp_path, LEGACY_V3_FILE)

    with pytest.raises(ValueError, match="different gripper generation"):
        create_umi_device(
            UmiGripper,
            device_name="right_umi_gripper",
            reader=MagicMock(),
            config_file=cfg,
            gear_radius=15.0,
            gripper_generation="v4",
        )


def test_the_same_gear_loads_the_file_as_it_always_did(tmp_path):
    cfg = _write(tmp_path, LEGACY_V3_FILE)

    g = create_umi_device(
        UmiGripper,
        device_name="right_umi_gripper",
        reader=MagicMock(),
        config_file=cfg,
        gear_radius=26.0,
    )

    assert g.gear_radius == 26.0
    assert g.distance_zero_angle == 173.0


def test_a_legacy_file_without_a_generation_is_not_refused_for_lacking_one(tmp_path):
    """Only a stated disagreement is a disagreement: calibration files written
    before this key existed must keep loading under a V3 preset."""
    cfg = _write(tmp_path, LEGACY_V3_FILE)

    g = create_umi_device(
        UmiGripper,
        device_name="right_umi_gripper",
        reader=MagicMock(),
        config_file=cfg,
        gear_radius=26.0,
        gripper_generation="v3",
    )

    assert g.gripper_generation is None


def test_a_v4_file_under_a_v3_setup_is_refused_too(tmp_path):
    cfg = _write(tmp_path, V4_FILE)

    with pytest.raises(ValueError, match="different gripper generation"):
        create_umi_device(
            UmiGripper,
            device_name="right_umi_gripper",
            reader=MagicMock(),
            config_file=cfg,
            gear_radius=26.0,
            gripper_generation="v3",
        )


def test_a_v4_file_under_a_v4_setup_carries_its_generation(tmp_path):
    cfg = _write(tmp_path, V4_FILE)

    g = create_umi_device(
        UmiGripper,
        device_name="right_umi_gripper",
        reader=MagicMock(),
        config_file=cfg,
        gear_radius=15.0,
        gripper_generation="v4",
    )

    assert (g.gripper_generation, g.gear_radius) == ("v4", 15.0)
    assert (g.open_travel_min_deg, g.open_travel_max_deg) == (120.0, 175.0)


def test_without_a_file_the_params_build_the_gripper(tmp_path):
    g = create_umi_device(
        UmiGripper,
        device_name="right_umi_gripper",
        reader=MagicMock(),
        config_file=str(tmp_path / "missing.yaml"),
        gripper_generation="v4",
        distance_zero_angle=25.0,
        directions="+",
    )

    assert (g.gripper_generation, g.gear_radius) == ("v4", 15.0)

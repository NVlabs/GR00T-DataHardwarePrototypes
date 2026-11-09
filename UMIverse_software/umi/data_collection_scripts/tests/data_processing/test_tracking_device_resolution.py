# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tracking checks must behave identically whatever produced the pose.

An iPhone rig records `{side}_tracking_camera` (ARKit pose inside a camera
frame); a Vive rig records `{side}_tracking_pose` (a pose-only numeric source).
Both are tracking devices. These tests pin that, because the failure mode is
silent: a check that stops resolving appends no issue and leaves `passed`
untouched, so a green data check quietly means less than it did.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "real_scripts"))

from real_scripts.data_processing.data_check.common import (  # noqa: E402
    tracking_device_for_side,
    tracking_device_to_arm,
    tracking_devices_in,
)
from real_scripts.data_processing.data_check.dispatch import check_episode  # noqa: E402

# Straight from the module, not through `data_check.common`: its callers are the
# station console's Vive plumbing, and re-exporting it beside the data check's
# own three would advertise it to code that has no use for it.
from umi.common.utility.tracking import tracking_device_side  # noqa: E402

IPHONE = ["left_tracking_camera", "right_tracking_camera"]
VIVE = ["left_tracking_pose", "right_tracking_pose"]
MIXED = ["left_tracking_camera", "right_tracking_pose"]

N = 120
CHECK_CONFIG = {
    "tracking": {"enabled": True},
}


# --------------------------------------------------------------------------
# resolution rules
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "device,expected",
    [
        ("left_tracking_camera", "left_arm"),
        ("right_tracking_camera", "right_arm"),
        ("left_tracking_pose", "left_arm"),
        ("right_tracking_pose", "right_arm"),
        # A future pose source needs no table entry.
        ("left_tracking_vive", "left_arm"),
        # The head is not an arm, and must never resolve to one.
        ("head_tracking_camera", None),
        ("wrist_tracking_thing", None),
    ],
)
def test_arm_is_derived_from_the_side_token_not_a_table(device, expected):
    assert tracking_device_to_arm(device) == expected


@pytest.mark.parametrize(
    "device,expected",
    [
        ("left_tracking_pose", "left"),
        ("right_tracking_pose", "right"),
        ("left_tracking_camera", "left"),
        # Case and shape of the tail are irrelevant -- only the leading token.
        ("RIGHT_umi_gripper", "right"),
        ("left", "left"),
        # The head is not an arm here either, and neither is nothing at all.
        ("head_tracking_pose", None),
        ("", None),
    ],
)
def test_the_bare_side_is_the_same_rule_as_the_arm(device, expected):
    """Callers that want "left" rather than "left_arm" -- the console's Vive
    rows, form fields and per-arm prompts -- share this one helper instead of
    each splitting the name against ARM_SIDES themselves."""
    assert tracking_device_side(device) == expected


def test_the_side_and_the_arm_can_never_disagree():
    """Defined in terms of `tracking_device_to_arm`, so a change to the naming
    convention lands in one place. Pinned because "silently mapped to nothing"
    is what this module's header exists to prevent."""
    for device in IPHONE + VIVE + ["head_tracking_camera", "wrist_tracking_thing"]:
        side = tracking_device_side(device)
        arm = tracking_device_to_arm(device)
        assert (arm is None) == (side is None), device
        assert arm is None or arm == f"{side}_arm", device


def test_tracking_devices_are_found_by_substring_only():
    found = tracking_devices_in(
        ["left_tracking_pose", "left_umi_camera", "right_tracking_camera", "junk"]
    )
    assert found == ["left_tracking_pose", "right_tracking_camera"]


def test_for_side_is_the_inverse_of_to_arm():
    assert tracking_device_for_side("left", IPHONE) == "left_tracking_camera"
    assert tracking_device_for_side("left", VIVE) == "left_tracking_pose"
    assert tracking_device_for_side("right", MIXED) == "right_tracking_pose"


def test_an_ambiguous_side_refuses_to_guess():
    both = ["left_tracking_pose", "left_tracking_camera"]
    assert tracking_device_for_side("left", both) is None


# --------------------------------------------------------------------------
# end-to-end through check_episode
# --------------------------------------------------------------------------


def _episode(tracking_names):
    """Bimanual episode: two grippers that close, plus moving tracking poses."""
    root = zarr.open_group(store=zarr.storage.MemoryStore(), mode="w")
    ep = root.create_group("episode_0")
    ts = np.arange(N) / 30.0
    for side in ("left", "right"):
        g = ep.create_group(f"{side}_umi_gripper")
        d = np.concatenate(
            [np.full(60, 70.0), np.linspace(70, 20, 10), np.full(50, 20.0)]
        )
        g.create_array("distance", shape=d.shape, dtype="f8")[:] = d
        g.create_array("capture_time", shape=ts.shape, dtype="f8")[:] = ts
    for name in tracking_names:
        t = ep.create_group(name)
        poses = np.tile(np.eye(4), (N, 1, 1))
        poses[:, 0, 3] = np.linspace(0, 0.5, N)
        t.create_array("pose", shape=poses.shape, dtype="f8")[:] = poses
        t.create_array("capture_time", shape=ts.shape, dtype="f8")[:] = ts
        t.create_array("receive_time", shape=ts.shape, dtype="f8")[:] = ts
    return ep


@pytest.mark.parametrize(
    "label,names", [("iphone", IPHONE), ("vive", VIVE), ("mixed", MIXED)]
)
def test_pose_checks_run_for_every_pose_source(label, names):
    result = check_episode(
        _episode(names),
        config={},
        check_config=CHECK_CONFIG,
        episode_name="episode_0",
        data_dir="/tmp",
    )
    checks = result["tracking_checks"]
    assert sorted(checks) == sorted(names), label
    for name in names:
        assert "pose_movement" in checks[name], f"{label}: {name} had no pose check"

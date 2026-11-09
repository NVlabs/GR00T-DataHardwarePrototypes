# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-name consistency is derived from the session's own recording config.

There is no canonical list of camera roles: a rig may carry optional head or
side cameras that no shipped config enumerates. The check stays strict in both
directions that matter — a declared source that never recorded fails, and a
recorded device the config never declared fails — but it never compares a
session against a fixed role list.
"""

from pathlib import Path

import pytest
import yaml

from real_scripts.data_processing.data_check.device_names import (
    check_device_name_consistency,
    declared_device_names,
    expected_device_names,
)

BASE_SOURCES = [
    {"type": "iphone_camera", "name": "left_tracking_camera"},
    {"type": "iphone_camera", "name": "right_tracking_camera"},
    {"type": "oak_camera", "name": "left_umi_camera"},
    {"type": "oak_camera", "name": "right_umi_camera"},
]
BASE_NUMERIC = [
    {"type": "umi_gripper", "name": "left_umi_gripper"},
    {"type": "umi_gripper", "name": "right_umi_gripper"},
]
BASE_RECORDED = [
    "left_tracking_camera",
    "right_tracking_camera",
    "left_umi_camera",
    "right_umi_camera",
    "left_umi_gripper",
    "right_umi_gripper",
]


def _cfg(extra_video=()):
    return {
        "video_sources": [*BASE_SOURCES, *extra_video],
        "numeric_sources": list(BASE_NUMERIC),
    }


def test_an_optional_bottom_camera_does_not_fail_the_check():
    cfg = _cfg([{"type": "oak_camera", "name": "bottom_oak_camera"}])
    result = check_device_name_consistency(
        cfg,
        recorded_devices=[
            "left_tracking_camera",
            "right_tracking_camera",
            "left_umi_camera",
            "right_umi_camera",
            "bottom_oak_camera",
            "left_umi_gripper",
            "right_umi_gripper",
        ],
    )
    assert result["passed"] is True, result
    assert result["extra_devices"] == []


def test_optional_head_cameras_do_not_fail_the_check():
    cfg = _cfg(
        [
            {"type": "oak_camera", "name": "head_oak_camera"},
            {"type": "iphone_camera", "name": "head_iphone_camera"},
        ]
    )
    result = check_device_name_consistency(
        cfg,
        recorded_devices=[
            "left_tracking_camera",
            "right_tracking_camera",
            "left_umi_camera",
            "right_umi_camera",
            "head_oak_camera",
            "head_iphone_camera",
            "left_umi_gripper",
            "right_umi_gripper",
        ],
    )
    assert result["passed"] is True, result


def test_a_declared_source_that_never_recorded_still_fails():
    cfg = _cfg([{"type": "oak_camera", "name": "bottom_oak_camera"}])
    result = check_device_name_consistency(cfg, recorded_devices=list(BASE_RECORDED))
    assert result["passed"] is False
    assert "bottom_oak_camera" in result["missing_devices"]


def test_a_recorded_device_absent_from_the_config_still_fails():
    result = check_device_name_consistency(
        _cfg(),
        recorded_devices=[*BASE_RECORDED, "mystery_camera"],
    )
    assert result["passed"] is False
    assert "mystery_camera" in result["extra_devices"]


def test_zero_grippers_does_not_raise():
    cfg = {"video_sources": list(BASE_SOURCES), "numeric_sources": []}
    result = check_device_name_consistency(
        cfg,
        recorded_devices=[
            "left_tracking_camera",
            "right_tracking_camera",
            "left_umi_camera",
            "right_umi_camera",
        ],
    )
    assert isinstance(result["passed"], bool)
    assert result["passed"] is True, result


def test_three_grippers_does_not_raise():
    cfg = {
        "video_sources": list(BASE_SOURCES),
        "numeric_sources": [
            *BASE_NUMERIC,
            {"type": "umi_gripper", "name": "middle_umi_gripper"},
        ],
    }
    result = check_device_name_consistency(
        cfg, recorded_devices=[*BASE_RECORDED, "middle_umi_gripper"]
    )
    assert result["passed"] is True, result


# ---------------------------------------------------------------------------
# oak_device blocks: template form (streams) and saved-session form (expanded)
# ---------------------------------------------------------------------------


def test_oak_device_block_declares_its_stream_names_not_the_device_name():
    cfg = {
        "video_sources": [
            {
                "type": "oak_device",
                "oak_device_name": "oak-FFC-05",
                "streams": [
                    {"name": "right_trigger_camera", "socket": "CAM_A"},
                    {"name": "right_umi_camera", "socket": "CAM_B"},
                ],
            }
        ],
        "numeric_sources": [{"type": "umi_gripper", "name": "right_umi_gripper"}],
    }
    video, numeric = declared_device_names(cfg)
    assert sorted(video) == ["right_trigger_camera", "right_umi_camera"]
    assert numeric == ["right_umi_gripper"]

    result = check_device_name_consistency(
        cfg,
        recorded_devices=[
            "right_trigger_camera",
            "right_umi_camera",
            "right_umi_gripper",
        ],
    )
    assert result["passed"] is True, result
    assert "oak-FFC-05" not in result["expected_devices"]


def test_expanded_oak_device_entries_declare_their_own_name():
    """`create_video_recorders` rewrites an oak_device block into one entry per
    stream (same `type`, `name` = stream name) before config.yaml is saved."""
    cfg = {
        "video_sources": [
            {"type": "oak_device", "name": "right_trigger_camera"},
            {"type": "oak_device", "name": "right_umi_camera"},
        ],
        "numeric_sources": [{"type": "umi_gripper", "name": "right_umi_gripper"}],
    }
    result = check_device_name_consistency(
        cfg,
        recorded_devices=[
            "right_trigger_camera",
            "right_umi_camera",
            "right_umi_gripper",
        ],
    )
    assert result["passed"] is True, result


# ---------------------------------------------------------------------------
# The data-check config escape hatch (pinned device names)
# ---------------------------------------------------------------------------


def test_check_config_name_lists_still_override_the_session_config():
    """A check config may pin the expected names outright; dispatch layers that
    block on. Sessions recorded before the names became derived carry such a
    block in their session-local data_check_config.yaml."""
    cfg = {
        "video_sources": [{"type": "oak_camera", "name": "right_umi_camera"}],
        "numeric_sources": [],
        "device_name_consistency": {
            "right_video_sources": ["right_umi_camera"],
            "right_numeric_sources": ["ur5e", "ur5e_action", "right_hand_joint"],
        },
    }
    result = check_device_name_consistency(
        cfg,
        recorded_devices=[
            "right_umi_camera",
            "ur5e",
            "ur5e_action",
            "right_hand_joint",
        ],
    )
    assert result["passed"] is True, result


def test_optional_devices_are_allowed_but_never_required():
    cfg = _cfg()
    cfg["device_name_consistency"] = {"optional_devices": ["head_umi_camera"]}

    # Declared-plus-optional recorded: allowed.
    allowed = check_device_name_consistency(
        cfg, recorded_devices=[*BASE_RECORDED, "head_umi_camera"]
    )
    assert allowed["passed"] is True, allowed

    # Optional absent: still fine, and the required set is still the session's.
    absent = check_device_name_consistency(cfg, recorded_devices=list(BASE_RECORDED))
    assert absent["passed"] is True, absent
    assert expected_device_names(cfg)["all_devices"] == tuple(sorted(BASE_RECORDED))


def test_a_pinned_camera_absent_from_the_config_is_reported_missing():
    """Why a rig pins its cameras even though they are normally derived.

    Under `--allow-missing-devices`, `create_video_recorders` catches a per-source
    construction error, prints it and continues without appending to the
    expanded config, so a camera that failed to open is neither declared in the
    saved config.yaml nor recorded. Derived expected video would be empty and the
    check would pass; the pin is what turns a video-less episode into a
    quarantine. Without that flag the recording aborts outright, so the pin is
    the guard for the deliberate degraded path.
    """
    cfg = {
        # The camera failed to open, so it never made it into the saved config.
        "video_sources": [],
        "numeric_sources": [],
        "device_name_consistency": {
            "enabled": True,
            "right_video_sources": ["right_umi_camera"],
            "right_numeric_sources": ["ur5e", "right_hand_joint"],
        },
    }

    result = check_device_name_consistency(
        cfg, recorded_devices=["ur5e", "right_hand_joint"]
    )

    assert result["passed"] is False, result
    assert "right_umi_camera" in result["missing_devices"]

    # Without the pin the same session passes silently — that is the gap it closes.
    unpinned = dict(cfg)
    unpinned["device_name_consistency"] = {
        "enabled": True,
        "right_numeric_sources": ["ur5e", "right_hand_joint"],
    }
    assert expected_device_names(unpinned)["video_sources"] == ()
    assert (
        check_device_name_consistency(
            unpinned, recorded_devices=["ur5e", "right_hand_joint"]
        )["passed"]
        is True
    )


def test_pinning_only_the_numeric_names_leaves_the_cameras_derived():
    """A partial override must not blank the category it does not name — that
    would make every camera an extra device and quarantine the session."""
    cfg = _cfg()
    cfg["device_name_consistency"] = {
        "enabled": True,
        "right_numeric_sources": ["ur5e", "ur5e_action"],
    }

    expected = expected_device_names(cfg)
    assert expected["video_sources"] == (
        "left_tracking_camera",
        "left_umi_camera",
        "right_tracking_camera",
        "right_umi_camera",
    )
    assert expected["numeric_sources"] == ("ur5e", "ur5e_action")

    result = check_device_name_consistency(
        cfg,
        recorded_devices=[
            "left_tracking_camera",
            "right_tracking_camera",
            "left_umi_camera",
            "right_umi_camera",
            "ur5e",
            "ur5e_action",
        ],
    )
    assert result["passed"] is True, result
    assert result["extra_devices"] == []


def test_pinning_only_the_video_names_leaves_the_grippers_derived():
    cfg = _cfg()
    cfg["device_name_consistency"] = {
        "enabled": True,
        "right_video_sources": ["right_umi_camera"],
    }

    expected = expected_device_names(cfg)
    assert expected["video_sources"] == ("right_umi_camera",)
    assert expected["numeric_sources"] == ("left_umi_gripper", "right_umi_gripper")


def test_a_scalar_source_list_fails_loudly_instead_of_iterating_characters():
    cfg = _cfg()
    cfg["device_name_consistency"] = {"right_video_sources": "right_umi_camera"}

    with pytest.raises(ValueError, match="must be a list of device names"):
        expected_device_names(cfg)


def test_a_scalar_optional_devices_value_fails_loudly():
    cfg = _cfg()
    cfg["device_name_consistency"] = {"optional_devices": "head_umi_camera"}

    with pytest.raises(ValueError, match="must be a list of device names"):
        expected_device_names(cfg)


# ---------------------------------------------------------------------------
# The shipped check configs must stay on the derive branch
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("config_name", ["data_check_config.yaml"])
def test_shipped_check_config_does_not_enumerate_device_names(config_name):
    """The production invariant of this check: the shipped data-check YAML must
    not name devices, or every rig carrying a camera it does not list gets its
    whole session quarantined."""
    path = REPO_ROOT / "config" / "data_check" / config_name
    with open(path) as f:
        block = yaml.safe_load(f)["device_name_consistency"]

    assert set(block) <= {"enabled"}, (
        f"{config_name} re-enumerates device names: {sorted(set(block) - {'enabled'})}"
    )

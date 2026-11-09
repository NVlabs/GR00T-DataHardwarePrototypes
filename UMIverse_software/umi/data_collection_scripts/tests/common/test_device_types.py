# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The device-type table: the one declaration of what buckets exist."""

import pytest

import umi.utils.device_types as device_types
from umi.utils.device_types import (
    BUCKET_KEYS,
    CAMERA_KEYS,
    DEVICE_TYPES,
    FACTS_KEY,
    GRIPPER_KEYS,
    METADATA_KEY,
    RELATION_KEYS,
    DeviceType,
    infer_bucket,
)


def test_every_bucket_key_is_declared_once():
    keys = [entry.key for entry in DEVICE_TYPES]
    assert len(keys) == len(set(keys)), keys


def test_every_kind_is_camera_or_gripper():
    for entry in DEVICE_TYPES:
        assert entry.kind in ("camera", "gripper"), entry


def test_the_key_groups_partition_the_table():
    assert set(GRIPPER_KEYS) | set(CAMERA_KEYS) == set(BUCKET_KEYS)
    assert not set(GRIPPER_KEYS) & set(CAMERA_KEYS)
    assert set(BUCKET_KEYS) == {entry.key for entry in DEVICE_TYPES}


def test_the_relation_and_metadata_keys_are_not_device_buckets():
    """gripper_cameras holds name->name, not name->serial. A caller that walked
    BUCKET_KEYS and hit it would compare a camera name against a serial."""
    assert METADATA_KEY not in BUCKET_KEYS
    assert FACTS_KEY not in BUCKET_KEYS
    for key in RELATION_KEYS:
        assert key not in BUCKET_KEYS


@pytest.mark.parametrize(
    "name,expected",
    [
        ("iphone15p-04", "iphone_devices"),
        ("oak-FFC-08", "oak_devices"),
        ("oak-IMX-IN-0001", "oak_devices"),
        ("UMI-V4-0001", "umi_v4_grippers"),
    ],
)
def test_infer_bucket_reads_the_name_prefix(name, expected):
    assert infer_bucket(name) == expected


@pytest.mark.parametrize("name", ["UMI-044", "UMI-V2-0008", "UMI-V3-0001"])
def test_a_retired_gripper_name_is_filed_nowhere(name):
    """Only the V4 bucket is declared, so a V1-V3 name must not land in it."""
    assert infer_bucket(name) is None


def test_infer_bucket_prefers_the_longest_prefix(monkeypatch):
    """When one declared prefix starts another, declaration order must not
    decide which bucket wins, or every device lands under the shorter one. The
    real table has no such pair today, so a two-row table stands in for one --
    shorter prefix declared first."""
    monkeypatch.setattr(
        device_types,
        "DEVICE_TYPES",
        (
            DeviceType("short_grippers", "gripper", "short", ("UMI-",)),
            DeviceType("long_grippers", "gripper", "long", ("UMI-V4-",)),
        ),
    )
    assert infer_bucket("UMI-V4-0001") == "long_grippers"
    assert infer_bucket("UMI-005") == "short_grippers"


def test_infer_bucket_matches_only_at_the_start_of_the_name():
    """A contains-based rewrite passes every other test in this file. Only a
    name with a prefix in the middle distinguishes anchored from unanchored."""
    assert infer_bucket("station-iphone-01") is None
    assert infer_bucket("my-oak-cam") is None
    assert infer_bucket("rig-iphone15p-04") is None
    assert infer_bucket("rig-UMI-V4-0001") is None


def test_infer_bucket_is_case_insensitive():
    assert infer_bucket("umi-v4-0008") == "umi_v4_grippers"
    assert infer_bucket("OAK-FFC-08") == "oak_devices"


def test_infer_bucket_returns_none_for_an_unknown_name():
    assert infer_bucket("widget-01") is None
    assert infer_bucket("") is None

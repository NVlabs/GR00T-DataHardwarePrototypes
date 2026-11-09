# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gripper -> camera edge table, and resolving either spelling to it."""

import pytest

REGISTRY = {
    "version": 1,
    "oak_devices": {
        "oak-FFC-08": "1944301000000000AA",
        "oak-FFC-14": "1944301000000000AB",
    },
    "iphone_devices": {"iphone15p-04": "00008130-00000000000000A3"},
    "umi_v4_grippers": {
        "UMI-V4-0008": "30313134000000B0",
        "UMI-V4-0001": "32343239000000B4",
    },
    "gripper_cameras": {"UMI-V4-0008": "oak-FFC-08"},
}

# The same binding, hand-edited in the wrong case on both ends.
CASE_DRIFTED = {**REGISTRY, "gripper_cameras": {"umi-v4-0008": "oak-ffc-08"}}


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    import umi.utils.device_registry as reg

    monkeypatch.setattr(reg, "_REGISTRY", dict(REGISTRY))
    yield reg
    monkeypatch.setattr(reg, "_REGISTRY", None)


def test_the_table_loads(registry):
    assert registry.get_gripper_cameras() == {"UMI-V4-0008": "oak-FFC-08"}


def test_a_snapshot_without_the_section_loads_empty(registry, monkeypatch):
    """A registry with no gripper -> camera section loads as an empty table."""
    monkeypatch.setattr(registry, "_REGISTRY", {"version": 1})
    assert registry.get_gripper_cameras() == {}


def test_a_bound_gripper_resolves_to_its_camera(registry):
    assert registry.camera_for_gripper("UMI-V4-0008") == "oak-FFC-08"


def test_a_bound_gripper_resolves_from_its_raw_serial(registry):
    """The picker's box may hold a serial: an unregistered device is stored by
    id, and an operator reads ids off a screen."""
    assert registry.camera_for_gripper("30313134000000B0") == "oak-FFC-08"


def test_resolution_is_case_insensitive(registry):
    assert registry.camera_for_gripper("umi-v4-0008") == "oak-FFC-08"


@pytest.mark.parametrize(
    "value",
    ["UMI-V4-0009", "UMI-V4-0001", "not-a-device", "", None],
)
def test_everything_unbound_resolves_to_none(registry, value):
    """Every answer here is None, so a bug that returns a camera for any one of
    them is invisible unless all five are asserted: an unregistered name, a
    registered gripper with no camera bound, junk, empty, and None."""
    assert registry.camera_for_gripper(value) is None


def test_name_for_identifier_folds_either_spelling(registry):
    assert registry.name_for_identifier("UMI-V4-0008") == "UMI-V4-0008"
    assert registry.name_for_identifier("30313134000000B0") == "UMI-V4-0008"
    assert registry.name_for_identifier("oak-ffc-14") == "oak-FFC-14"
    assert registry.name_for_identifier("nonsense") is None
    assert registry.name_for_identifier("") is None


def test_a_case_drifted_table_key_still_binds(registry, monkeypatch):
    """This table is hand-edited, and every other identifier path in the
    module folds case. A key spelled `umi-v4-0008` against a bucket spelling
    `UMI-V4-0008` must still match, not report "no camera bound" silently,
    with no error and no log."""
    monkeypatch.setattr(registry, "_REGISTRY", dict(CASE_DRIFTED))
    assert registry.get_gripper_cameras() == {"UMI-V4-0008": "oak-FFC-08"}
    assert registry.camera_for_gripper("UMI-V4-0008") == "oak-FFC-08"
    assert registry.camera_for_gripper("30313134000000B0") == "oak-FFC-08"


def test_an_entry_naming_an_unregistered_device_is_kept(registry, monkeypatch):
    """Folding must not become dropping: a pairing naming a device in no
    bucket is passed through as written, so validate_registry can report it
    rather than reading an empty table and calling it clean."""
    monkeypatch.setattr(
        registry,
        "_REGISTRY",
        {**REGISTRY, "gripper_cameras": {"UMI-V4-9999": "oak-FFC-99"}},
    )
    assert registry.get_gripper_cameras() == {"UMI-V4-9999": "oak-FFC-99"}


def test_the_name_index_is_rebuilt_not_cached(registry, monkeypatch):
    """reload() can replace the registry at any time, and a cache would also
    outlive every test that swaps _REGISTRY. An @lru_cache on _name_index
    passes every other test in this file, so this is the only guard."""
    assert registry.name_for_identifier("30313134000000B0") == "UMI-V4-0008"
    monkeypatch.setattr(
        registry,
        "_REGISTRY",
        {"version": 1, "umi_v4_grippers": {"UMI-V4-0099": "30313134000000B0"}},
    )
    assert registry.name_for_identifier("30313134000000B0") == "UMI-V4-0099"

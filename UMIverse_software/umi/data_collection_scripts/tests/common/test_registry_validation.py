# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One validator for the registry blob, used by `save_registry`."""

import copy

import pytest

from umi.utils.device_registry import validate_registry

CLEAN = {
    "version": 1,
    "oak_devices": {"oak-FFC-08": "1944301000000000AA"},
    "iphone_devices": {},
    "umi_v4_grippers": {"UMI-V4-0008": "30313134000000B0"},
    "gripper_cameras": {"UMI-V4-0008": "oak-FFC-08"},
}


@pytest.fixture
def registry():
    return copy.deepcopy(CLEAN)


def test_a_clean_registry_has_no_problems(registry):
    assert validate_registry(registry) == []


def test_a_snapshot_with_no_binding_section_is_clean(registry):
    del registry["gripper_cameras"]
    assert validate_registry(registry) == []


def test_an_empty_binding_section_is_clean(registry):
    """Deliberately empty is not a fault, only absent-by-another-name."""
    registry["gripper_cameras"] = {}
    assert validate_registry(registry) == []


@pytest.mark.parametrize("falsy", [[], None, ""], ids=["list", "null", "string"])
def test_a_falsy_malformed_binding_section_is_reported(registry, falsy):
    """Falsy non-dict sections are reported too, not only truthy ones.

    A truthiness gate (`if links and not isinstance(links, dict)`) never fires
    for a falsy value, so `[]`, `null` and `""` would go unreported while a
    truthy `["a","b"]` was caught. The gap is not cosmetic: a writer that skips
    non-dict sections on purpose relies on a malformed blob being reported
    here; with nothing reported it would save the blob with the operator's
    whole pairing table thrown away, under exit 0.
    """
    registry["gripper_cameras"] = falsy
    problems = validate_registry(registry)
    assert any("gripper_cameras" in p for p in problems), problems


def test_a_bucket_that_is_not_a_mapping_is_reported(registry):
    registry["oak_devices"] = ["oak-FFC-08"]
    problems = validate_registry(registry)
    assert any("oak_devices" in p for p in problems), problems


@pytest.mark.parametrize("scalar", [5, True, 1.5])
def test_a_scalar_bucket_is_reported_rather_than_raising(registry, scalar):
    """A truthy non-iterable must not escape the shape guard.

    The guard `continue`s past a bad bucket, but the name sets below re-read
    the same key from `data`, so `5` could reach `for name in 5` and raise
    TypeError out of a function contracted never to raise. A list, a string,
    None and 0 are handled the same way. The blob is hand-editable JSON and
    both callers take it unfiltered.
    """
    registry["oak_devices"] = scalar
    problems = validate_registry(registry)
    assert any("oak_devices" in p for p in problems), problems


def test_a_non_string_serial_is_reported(registry):
    registry["oak_devices"]["oak-FFC-09"] = {"serial": "x"}
    problems = validate_registry(registry)
    assert any("oak-FFC-09" in p for p in problems), problems


def test_a_binding_naming_an_unknown_gripper_is_reported(registry):
    registry["gripper_cameras"]["UMI-V4-9999"] = "oak-FFC-08"
    problems = validate_registry(registry)
    assert any("UMI-V4-9999" in p for p in problems), problems


def test_a_binding_naming_an_unknown_camera_is_reported(registry):
    registry["gripper_cameras"]["UMI-V4-0008"] = "oak-FFC-99"
    problems = validate_registry(registry)
    assert any("oak-FFC-99" in p for p in problems), problems


def test_a_binding_naming_a_camera_as_the_gripper_is_reported(registry):
    """The two columns are easy to swap in an imported sheet."""
    registry["gripper_cameras"]["oak-FFC-08"] = "UMI-V4-0008"
    problems = validate_registry(registry)
    assert any("oak-FFC-08" in p for p in problems), problems


def test_a_binding_spelled_in_the_wrong_case_is_reported(registry):
    """`get_gripper_cameras` folds both ends on read, so a case-drifted table
    still binds at runtime -- which is exactly why the stored blob has to be
    cleaned here instead. Without this the drift is invisible forever."""
    registry["gripper_cameras"] = {"umi-v4-0008": "OAK-ffc-08"}
    problems = validate_registry(registry)
    assert any("umi-v4-0008" in p for p in problems), problems
    assert any("OAK-ffc-08" in p for p in problems), problems


def test_the_canonical_spelling_is_not_reported(registry):
    """Guard against the case check firing on correct data."""
    assert validate_registry(registry) == []


def test_one_camera_bound_to_two_grippers_is_reported(registry):
    registry["umi_v4_grippers"]["UMI-V4-0009"] = "30313134000000B1"
    registry["gripper_cameras"]["UMI-V4-0009"] = "oak-FFC-08"
    problems = validate_registry(registry)
    assert any("oak-FFC-08" in p for p in problems), problems


def test_a_duplicate_serial_across_buckets_is_reported(registry):
    """One serial filed as a gripper and as a camera is two kinds of device."""
    registry["oak_devices"]["oak-FFC-09"] = "30313134000000B0"
    problems = validate_registry(registry)
    assert any("30313134000000B0" in p for p in problems), problems


def test_a_duplicate_serial_WITHIN_one_bucket_is_not_reported(registry):
    """Deliberate aliasing, not an error.

    Two names for one serial in one bucket is how overlay OAKs are registered,
    and it is the transient state of every rename.
    """
    registry["oak_devices"]["oak-overlay-B"] = "1944301000000000AA"
    assert validate_registry(registry) == []


def test_the_validator_never_mutates_its_argument(registry):
    before = copy.deepcopy(registry)
    validate_registry(registry)
    assert registry == before

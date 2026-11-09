# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""V4 gripper registry and per-device metadata regressions."""

import pytest

V4_SERIAL_1 = "534A3035000000B8"
V4_SERIAL_2 = "534A3035000000B7"


def _sample_registry():
    return {
        "version": 1,
        "iphone_devices": {},
        "oak_devices": {},
        "umi_v4_grippers": {
            "UMI-V4-0001": V4_SERIAL_1,
            "UMI-V4-0002": V4_SERIAL_2,
        },
        # Use device-name keys; keep coverage for persisted serial-keyed banks.
        "device_metadata": {
            "UMI-V4-0001": {
                "comment": "Left gripper, bench 1",
            },
            "UMI-V4-0002": {
                "comment": "Right gripper, bench 1",
            },
        },
    }


def _reset_manager_maps(manager):
    for devices in (
        manager.IPHONE_DEVICES,
        manager.OAK_DEVICES,
        manager.UMI_V4_GRIPPERS,
    ):
        devices.invalidate()


@pytest.fixture(autouse=True)
def reset_registry():
    import umi.utils.device_manager as manager
    import umi.utils.device_registry as registry

    registry._REGISTRY = None
    _reset_manager_maps(manager)
    yield
    registry._REGISTRY = None
    _reset_manager_maps(manager)


def test_old_registry_without_v4_bucket_loads_empty():
    import umi.utils.device_registry as registry

    registry._REGISTRY = {"version": 1}

    assert registry.get_umi_v4_grippers() == {}


def test_runtime_resolves_v4_name_case_insensitively():
    import umi.utils.device_manager as manager
    import umi.utils.device_registry as registry

    registry._REGISTRY = _sample_registry()
    _reset_manager_maps(manager)

    assert manager.get_umi_gripper_serial("umi-v4-0001") == V4_SERIAL_1


def test_v4_bucket_resolves_through_the_same_lookups():
    """The flexure gripper's bucket is read by every gripper lookup: the
    registry getter, the serial lookup and the lazy map the UART identity
    check and the station picker read."""
    import umi.utils.device_manager as manager
    import umi.utils.device_registry as registry

    reg = _sample_registry()
    reg["umi_v4_grippers"] = {"UMI-V4-0001": "32343239000000B3"}
    registry._REGISTRY = reg
    _reset_manager_maps(manager)

    assert registry.get_umi_v4_grippers() == {"UMI-V4-0001": "32343239000000B3"}
    assert manager.get_umi_gripper_serial("umi-v4-0001") == "32343239000000B3"
    assert manager.UMI_V4_GRIPPERS["UMI-V4-0001"] == "32343239000000B3"


def test_a_registry_still_carrying_retired_gripper_buckets_loads():
    """A station registry written before the repository went V4-only keeps its
    `umi_grippers` / `umi_v2_grippers` / `umi_v3_grippers` sections. They are
    passed through unread: none of their names resolve, and the V4 bucket
    beside them still does."""
    import umi.utils.device_manager as manager
    import umi.utils.device_registry as registry

    reg = _sample_registry()
    reg["umi_grippers"] = {"UMI-044": "30313134000000B2"}
    reg["umi_v2_grippers"] = {"UMI-V2-0008": "30313134000000B0"}
    reg["umi_v3_grippers"] = {"UMI-V3-0001": "32343239000000B4"}
    registry._REGISTRY = reg
    _reset_manager_maps(manager)

    assert manager.get_umi_gripper_serial("UMI-V4-0001") == V4_SERIAL_1
    assert registry.name_for_identifier("UMI-V3-0001") is None
    with pytest.raises(ValueError, match="Unknown UMI gripper device"):
        manager.get_umi_gripper_serial("UMI-V2-0008")

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for umi.utils.device_registry and the device_manager shim."""

import copy
import json
from unittest.mock import patch

import pytest

SAMPLE_REGISTRY = {
    "version": 1,
    "updated_at": "2026-05-04T21:00:00Z",
    "iphone_devices": {
        "iphone16p-ry": "00008140-00000000000000A5",
        "iphone15p-01": "00008130-00000000000000A4",
    },
    "oak_devices": {
        "oak-W97-01": "1944301000000000A7",
        "oak-W97-02": "1944301000000000A6",
    },
    "umi_v4_grippers": {
        "UMI-V4-0001": "30313134000000B2",
    },
}


# Carries gripper -> camera edges, which SAMPLE_REGISTRY deliberately does not:
# the binding tests need both ends present, and the existing cases depend on
# SAMPLE_REGISTRY's exact shape.
BOUND_REGISTRY = {
    "version": 1,
    "updated_at": "2026-08-04T12:00:00Z",
    "iphone_devices": {},
    "oak_devices": {
        "oak-FFC-08": "OAKSERIAL08",
        "oak-FFC-09": "OAKSERIAL09",
    },
    "umi_v4_grippers": {
        "UMI-V4-0008": "G0008",
        "UMI-V4-0009": "G0009",
    },
    "gripper_cameras": {
        "UMI-V4-0008": "oak-FFC-08",
        "UMI-V4-0009": "oak-FFC-09",
    },
}


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the registry singleton between tests."""
    import umi.utils.device_registry as reg

    reg._REGISTRY = None
    yield
    reg._REGISTRY = None


def _write(path, data):
    path.write_text(json.dumps(data))
    return path


class TestLoadRegistry:
    def test_loads_the_file_named_by_the_env_var(self, monkeypatch, tmp_path):
        import umi.utils.device_registry as reg

        path = _write(tmp_path / "devices.json", SAMPLE_REGISTRY)
        monkeypatch.setenv(reg.ENV_REGISTRY_FILE, str(path))

        assert reg.registry_path() == path
        assert reg.load_registry() == SAMPLE_REGISTRY

    def test_the_local_file_defaults_to_config_devices_json(self, monkeypatch):
        import umi.utils.device_registry as reg
        from umi.common.utility.output_layout import repo_root

        monkeypatch.delenv(reg.ENV_REGISTRY_FILE, raising=False)

        assert reg.local_registry_path() == repo_root() / "config" / "devices.json"

    def test_an_unreadable_file_loads_an_empty_registry(self, monkeypatch, tmp_path):
        import umi.utils.device_registry as reg

        path = tmp_path / "devices.json"
        path.write_text("not valid json{{{")
        monkeypatch.setenv(reg.ENV_REGISTRY_FILE, str(path))

        result = reg.load_registry()

        assert result["version"] == 0
        assert reg.get_oak_devices() == {}


class TestLazySingleton:
    def test_get_registry_caches_result(self):
        import umi.utils.device_registry as reg

        with patch.object(
            reg, "load_registry", return_value=SAMPLE_REGISTRY
        ) as mock_load:
            r1 = reg._get_registry()
            r2 = reg._get_registry()

        mock_load.assert_called_once()
        assert r1 is r2

    def test_reload_rereads_the_file(self, monkeypatch, tmp_path):
        import umi.utils.device_registry as reg

        reg._REGISTRY = {"version": 0, "iphone_devices": {}}

        updated = dict(SAMPLE_REGISTRY)
        updated["version"] = 2
        path = _write(tmp_path / "devices.json", updated)
        monkeypatch.setenv(reg.ENV_REGISTRY_FILE, str(path))

        result = reg.reload()

        assert result["version"] == 2
        assert reg._REGISTRY["version"] == 2


class TestDeviceManagerShim:
    def test_iphone_devices_populated(self):
        import umi.utils.device_registry as reg

        with patch.object(reg, "load_registry", return_value=SAMPLE_REGISTRY):
            reg._REGISTRY = SAMPLE_REGISTRY

            from umi.utils.device_manager import IPHONE_DEVICES

            # Force re-init by resetting the _LazyDict
            IPHONE_DEVICES.invalidate()

            assert "iphone16p-ry" in IPHONE_DEVICES
            assert IPHONE_DEVICES["iphone16p-ry"] == "00008140-00000000000000A5"

    def test_get_iphone_udid(self):
        import umi.utils.device_registry as reg

        with patch.object(reg, "load_registry", return_value=SAMPLE_REGISTRY):
            reg._REGISTRY = SAMPLE_REGISTRY

            from umi.utils.device_manager import IPHONE_DEVICES, get_iphone_udid

            IPHONE_DEVICES.invalidate()

            assert get_iphone_udid("iphone16p-ry") == "00008140-00000000000000A5"

    @pytest.mark.parametrize(
        ("getter_name", "device_name", "expected"),
        [
            ("get_iphone_udid", "IPHONE16P-RY", "00008140-00000000000000A5"),
            ("get_oak_device_id", "OAK-w97-01", "1944301000000000A7"),
            ("get_umi_gripper_serial", "umi-v4-0001", "30313134000000B2"),
        ],
    )
    def test_device_name_lookup_is_case_insensitive(
        self, getter_name, device_name, expected
    ):
        import umi.utils.device_manager as manager
        import umi.utils.device_registry as reg

        reg._REGISTRY = SAMPLE_REGISTRY
        registries = [
            manager.IPHONE_DEVICES,
            manager.OAK_DEVICES,
            manager.UMI_V4_GRIPPERS,
        ]
        for devices in registries:
            devices.invalidate()

        assert getattr(manager, getter_name)(device_name) == expected

    def test_ambiguous_case_only_runtime_names_raise(self, monkeypatch):
        import umi.utils.device_manager as manager

        monkeypatch.setattr(
            manager,
            "UMI_V4_GRIPPERS",
            {
                "UMI-V4-0051": "SERIAL-A",
                "umi-v4-0051": "SERIAL-B",
            },
        )

        with pytest.raises(ValueError, match="Ambiguous device name"):
            manager.get_umi_gripper_serial("UMI-V4-0051")

    def test_ambiguous_case_only_lazy_dict_names_raise(self):
        from umi.utils.device_manager import _LazyDict

        devices = _LazyDict(
            lambda: {
                "UMI-V4-0051": "SERIAL-A",
                "umi-v4-0051": "SERIAL-B",
            }
        )

        with pytest.raises(ValueError, match="Ambiguous device name"):
            devices["UMI-V4-0051"]

    def test_get_iphone_udid_raises_on_unknown(self):
        import umi.utils.device_registry as reg

        with patch.object(reg, "load_registry", return_value=SAMPLE_REGISTRY):
            reg._REGISTRY = SAMPLE_REGISTRY

            from umi.utils.device_manager import IPHONE_DEVICES, get_iphone_udid

            IPHONE_DEVICES.invalidate()

            with pytest.raises(ValueError, match="Unknown iPhone device"):
                get_iphone_udid("nonexistent-phone")

    def test_umi_v4_grippers_resolve_case_insensitively(self):
        import umi.utils.device_registry as reg

        with patch.object(reg, "load_registry", return_value=SAMPLE_REGISTRY):
            reg._REGISTRY = SAMPLE_REGISTRY

            from umi.utils.device_manager import UMI_V4_GRIPPERS

            UMI_V4_GRIPPERS.invalidate()

            assert "UMI-V4-0001" in UMI_V4_GRIPPERS
            assert "umi-v4-0001" in UMI_V4_GRIPPERS
            assert UMI_V4_GRIPPERS["umi-v4-0001"] == "30313134000000B2"
            assert UMI_V4_GRIPPERS.get("umi-v4-0001") == "30313134000000B2"

    def test_get_umi_gripper_serial_accepts_raw_board_id(self):
        from umi.utils.device_manager import get_umi_gripper_serial

        assert get_umi_gripper_serial("deadbeefdeadbeef") == "DEADBEEFDEADBEEF"

    def test_normalize_umi_gripper_identifier_only_canonicalizes_raw_serial(self):
        from umi.utils.device_manager import normalize_umi_gripper_identifier

        assert (
            normalize_umi_gripper_identifier("  deadbeefdeadbeef ")
            == "DEADBEEFDEADBEEF"
        )
        assert normalize_umi_gripper_identifier(" UMI-V4-0005 ") == "UMI-V4-0005"

    @pytest.mark.parametrize(
        "value",
        [
            "not-a-device",
            "DEADBEEF",
            "DEADBEEFDEADBEEF00",
            "DEADBEEFDEADBEEG",
        ],
    )
    def test_get_umi_gripper_serial_rejects_invalid_raw_id(self, value):
        from umi.utils.device_manager import get_umi_gripper_serial

        with pytest.raises(ValueError, match="Unknown UMI gripper device"):
            get_umi_gripper_serial(value)


class TestBucketAccess:
    """Every bucket reads the same way, and a missing one is empty not fatal."""

    def test_get_bucket_returns_the_named_section(self):
        import umi.utils.device_registry as reg

        reg._REGISTRY = copy.deepcopy(SAMPLE_REGISTRY)
        assert reg.get_bucket("oak_devices")["oak-W97-01"] == "1944301000000000A7"

    def test_every_declared_bucket_loads_empty_when_the_snapshot_omits_it(self):
        """A snapshot written before a bucket existed has no section for it.
        Every getter, camera and gripper alike, must read that as empty rather
        than raise KeyError."""
        import umi.utils.device_registry as reg
        from umi.utils.device_types import BUCKET_KEYS

        reg._REGISTRY = {"version": 1}
        for key in BUCKET_KEYS:
            assert reg.get_bucket(key) == {}, key

    def test_the_named_getters_agree_with_get_bucket(self):
        import umi.utils.device_registry as reg

        reg._REGISTRY = copy.deepcopy(SAMPLE_REGISTRY)
        assert reg.get_oak_devices() == reg.get_bucket("oak_devices")
        assert reg.get_iphone_devices() == reg.get_bucket("iphone_devices")
        assert reg.get_umi_v4_grippers() == reg.get_bucket("umi_v4_grippers")

    def test_get_bucket_refuses_a_key_the_table_does_not_declare(self):
        """A typo must not silently read empty -- that is how a getter for a
        bucket that was renamed goes on returning nothing forever."""
        import umi.utils.device_registry as reg

        reg._REGISTRY = copy.deepcopy(SAMPLE_REGISTRY)
        with pytest.raises(KeyError):
            reg.get_bucket("oak_device")

    def test_a_section_no_bucket_declares_still_loads_and_validates(self):
        """A station's own registry can carry a section for hardware this build
        no longer declares. It must keep loading and validating -- carried along
        untouched, never an error that blocks boot or a save."""
        import umi.utils.device_registry as reg

        data = copy.deepcopy(SAMPLE_REGISTRY)
        data["retired_devices"] = {"retired-01": "123456"}
        reg._REGISTRY = data

        assert reg.validate_registry(data) == []
        assert reg.get_oak_devices() == SAMPLE_REGISTRY["oak_devices"]
        assert reg.name_for_identifier("retired-01") is None


class TestFallbackFile:
    def test_the_fallback_is_loaded_from_json_and_has_every_bucket(self):
        import umi.utils.device_registry as reg
        from umi.utils.device_types import BUCKET_KEYS

        fallback = json.loads(reg._FALLBACK_FILE.read_text())
        for key in BUCKET_KEYS:
            assert key in fallback, key
            assert isinstance(fallback[key], dict)

    def test_load_registry_falls_back_to_the_file_when_there_is_no_local_registry(
        self, monkeypatch, tmp_path
    ):
        import umi.utils.device_registry as reg

        monkeypatch.setenv(reg.ENV_REGISTRY_FILE, str(tmp_path / "absent.json"))

        assert reg.registry_path() == reg._FALLBACK_FILE
        loaded = reg.load_registry()
        assert loaded == json.loads(reg._FALLBACK_FILE.read_text())


class TestSaveRegistry:
    def test_writes_the_local_file_and_makes_it_active(self, monkeypatch, tmp_path):
        import umi.utils.device_registry as reg

        path = tmp_path / "nested" / "devices.json"
        monkeypatch.setenv(reg.ENV_REGISTRY_FILE, str(path))

        assert reg.save_registry(copy.deepcopy(BOUND_REGISTRY)) == path

        assert json.loads(path.read_text()) == BOUND_REGISTRY
        assert not path.with_name("devices.json.tmp").exists()
        assert reg.camera_for_gripper("umi-v4-0008") == "oak-FFC-08"
        assert reg.reload() == BOUND_REGISTRY

    def test_refuses_an_invalid_registry_and_writes_nothing(
        self, monkeypatch, tmp_path
    ):
        import umi.utils.device_registry as reg

        path = tmp_path / "devices.json"
        monkeypatch.setenv(reg.ENV_REGISTRY_FILE, str(path))
        bad = copy.deepcopy(BOUND_REGISTRY)
        bad["gripper_cameras"]["UMI-V4-9999"] = "oak-FFC-08"

        with pytest.raises(ValueError, match="UMI-V4-9999"):
            reg.save_registry(bad)

        assert not path.exists()


class TestRegistryInfo:
    def test_reports_the_bundled_snapshot(self, monkeypatch, tmp_path):
        import umi.utils.device_registry as reg

        monkeypatch.setenv(reg.ENV_REGISTRY_FILE, str(tmp_path / "absent.json"))

        info = reg.get_registry_info()

        assert info["source"] == "bundled"
        assert info["path"] == str(reg._FALLBACK_FILE)

    def test_reports_the_local_file(self, monkeypatch, tmp_path):
        import umi.utils.device_registry as reg

        path = _write(tmp_path / "devices.json", SAMPLE_REGISTRY)
        monkeypatch.setenv(reg.ENV_REGISTRY_FILE, str(path))

        assert reg.get_registry_info() == {
            "version": 1,
            "updated_at": "2026-05-04T21:00:00Z",
            "source": "local",
            "path": str(path),
        }

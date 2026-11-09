# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for umi.common.tasks.task_registry_store and the lazy task_registry shim."""

import json
from unittest.mock import patch

import pytest

SAMPLE_REGISTRY = {
    "version": 1,
    "updated_at": "2026-05-20T00:00:00Z",
    "tasks": {
        "plane_assembly_v9": {
            "name": "plane_assembly_v9",
            "short_description": "Assemble the plane",
            "display_description": "",
            "mode": "bimanual",
            "HW_version": "v1",
            "training_group": "plane_assembly_v9",
            "use_head_mount": True,
            "has_trigger_cameras": False,
            "data_check_overrides": {
                "duration": {
                    "default": {"min_duration_s": 190.0, "max_duration_s": 225.0},
                    "plane_assembly_v9": {
                        "min_duration_s": 190.0,
                        "max_duration_s": 225.0,
                    },
                },
                "tracking": {"multi_tracking_consecutive_required_ratio": 0.15},
            },
        },
        "pnp_cylinder_weave_g1": {
            "name": "pnp_cylinder_weave_g1",
            "short_description": "Pick cylinder, weave, and place",
            "display_description": "",
            "mode": "bimanual",
            "HW_version": "v1",
            "training_group": "pnp_cylinder_weave_g1",
            "use_head_mount": True,
            "has_trigger_cameras": False,
            "data_check_overrides": {
                "duration": {
                    "default": {"min_duration_s": 35.0, "max_duration_s": 50.0},
                    "pnp_cylinder_weave_g1": {
                        "min_duration_s": 35.0,
                        "max_duration_s": 50.0,
                    },
                },
                "gripper": {"movement_threshold_mm": -1.0},
            },
        },
    },
}


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the registry singleton between tests."""
    import umi.common.tasks.task_registry_store as store

    store._REGISTRY = None
    store._REGISTRY_SOURCE = None
    yield
    store._REGISTRY = None
    store._REGISTRY_SOURCE = None


@pytest.fixture
def no_local_registry(monkeypatch, tmp_path):
    """No env override and no ``config/tasks.json``: the bundled snapshot."""
    import umi.common.tasks.task_registry_store as store

    monkeypatch.delenv(store.ENV_REGISTRY_FILE, raising=False)
    monkeypatch.setattr(
        store, "local_registry_path", lambda: tmp_path / "absent" / "tasks.json"
    )


def _write(path, data):
    path.write_text(json.dumps(data))
    return path


class TestLoadRegistry:
    def test_loads_the_file_named_by_the_env_var(self, monkeypatch, tmp_path):
        import umi.common.tasks.task_registry_store as store

        path = _write(tmp_path / "tasks.json", SAMPLE_REGISTRY)
        monkeypatch.setenv(store.ENV_REGISTRY_FILE, str(path))

        assert store.load_registry() == SAMPLE_REGISTRY
        assert store.get_registry_info()["source"] == f"file:{path}"

    def test_the_local_file_defaults_to_config_tasks_json(self, monkeypatch):
        import umi.common.tasks.task_registry_store as store
        from umi.common.utility.output_layout import repo_root

        monkeypatch.delenv(store.ENV_REGISTRY_FILE, raising=False)

        assert store.local_registry_path() == repo_root() / "config" / "tasks.json"

    def test_reads_the_local_file_when_no_env_var_is_set(self, monkeypatch, tmp_path):
        import umi.common.tasks.task_registry_store as store

        path = _write(tmp_path / "tasks.json", SAMPLE_REGISTRY)
        monkeypatch.delenv(store.ENV_REGISTRY_FILE, raising=False)
        monkeypatch.setattr(store, "local_registry_path", lambda: path)

        assert store.load_registry() == SAMPLE_REGISTRY
        assert store.get_registry_info()["source"] == f"file:{path}"

    def test_fallback_when_there_is_no_local_file(self, no_local_registry):
        import umi.common.tasks.task_registry_store as store

        result = store.load_registry()

        # The bundled list is empty, so a new station starts with no tasks.
        assert result["version"] >= 1
        assert result.get("tasks") == {}


class TestLazySingleton:
    def test_get_registry_caches_result(self):
        import umi.common.tasks.task_registry_store as store

        with patch.object(
            store, "_resolve_registry", return_value=(SAMPLE_REGISTRY, "bundled")
        ) as mock_resolve:
            r1 = store._get_registry()
            r2 = store._get_registry()
        mock_resolve.assert_called_once()
        assert r1 is r2

    def test_reload_rereads_the_file(self, monkeypatch, tmp_path):
        import umi.common.tasks.task_registry_store as store

        store._REGISTRY = {"version": 0, "tasks": {}}
        updated = dict(SAMPLE_REGISTRY)
        updated["version"] = 2
        path = _write(tmp_path / "tasks.json", updated)
        monkeypatch.setenv(store.ENV_REGISTRY_FILE, str(path))

        result = store.reload()
        assert result["version"] == 2
        assert store._REGISTRY["version"] == 2

    def test_get_registry_info_reports_the_local_file(self, monkeypatch, tmp_path):
        """``source`` must reflect the file the data actually came from."""
        import umi.common.tasks.task_registry_store as store

        path = _write(tmp_path / "tasks.json", SAMPLE_REGISTRY)
        monkeypatch.setenv(store.ENV_REGISTRY_FILE, str(path))

        assert store.get_registry_info() == {
            "version": 1,
            "updated_at": "2026-05-20T00:00:00Z",
            "source": f"file:{path}",
            "task_count": 2,
        }

    def test_get_registry_info_reports_the_bundled_snapshot(self, no_local_registry):
        import umi.common.tasks.task_registry_store as store

        info = store.get_registry_info()

        assert info["source"] == "bundled"
        assert info["task_count"] == len(store.get_all_tasks()) == 0


class TestAddTask:
    """How a station's list grows: the Station Console saves a new task here
    instead of anyone registering it by hand."""

    def test_the_first_task_creates_the_file_from_the_list_already_offered(
        self, monkeypatch, tmp_path
    ):
        import umi.common.tasks.task_registry_store as store

        path = tmp_path / "config" / "tasks.json"
        monkeypatch.setenv(store.ENV_REGISTRY_FILE, str(path))
        # The shipped list is empty; give it tasks to show they carry over.
        monkeypatch.setattr(
            store, "FALLBACK_FILE", _write(tmp_path / "bundled.json", SAMPLE_REGISTRY)
        )
        offered = set(store.get_all_tasks())  # no file yet: the bundled snapshot
        assert offered

        assert store.add_task("cup_on_plate", "Put the cup on the plate") == path

        saved = json.loads(path.read_text())
        # The bundled tasks stay selectable beside the new one.
        assert set(saved["tasks"]) == offered | {"cup_on_plate"}
        assert saved["tasks"]["cup_on_plate"] == {
            "name": "cup_on_plate",
            "short_description": "Put the cup on the plate",
        }
        # Active at once, read back from the file rather than kept in memory.
        assert store.get_registry_info()["source"] == f"file:{path}"
        assert "cup_on_plate" in store.get_all_tasks()

    def test_later_tasks_are_appended_and_a_listed_task_is_left_alone(
        self, monkeypatch, tmp_path
    ):
        import umi.common.tasks.task_registry_store as store

        path = _write(tmp_path / "tasks.json", SAMPLE_REGISTRY)
        monkeypatch.setenv(store.ENV_REGISTRY_FILE, str(path))

        store.add_task("cup_on_plate", "Put the cup on the plate")
        store.add_task("plane_assembly_v9", "Something else entirely")

        saved = json.loads(path.read_text())
        assert set(saved["tasks"]) == set(SAMPLE_REGISTRY["tasks"]) | {"cup_on_plate"}
        assert (
            saved["tasks"]["plane_assembly_v9"]
            == SAMPLE_REGISTRY["tasks"]["plane_assembly_v9"]
        )
        assert saved["updated_at"] != SAMPLE_REGISTRY["updated_at"]

    def test_a_file_that_cannot_be_read_is_never_overwritten(
        self, monkeypatch, tmp_path
    ):
        """The loader falls back to the snapshot for an unreadable file, so
        writing what it offers would replace the station's own list with it."""
        import umi.common.tasks.task_registry_store as store

        path = tmp_path / "tasks.json"
        path.write_text("{ not json")
        monkeypatch.setenv(store.ENV_REGISTRY_FILE, str(path))

        with pytest.raises(ValueError):
            store.add_task("cup_on_plate", "Put the cup on the plate")
        assert path.read_text() == "{ not json"


class TestTaskRegistryShim:
    def test_lazy_dict_resolves_specs(self):
        import umi.common.tasks.task_registry_store as store

        with patch.object(store, "load_registry", return_value=SAMPLE_REGISTRY):
            store._REGISTRY = SAMPLE_REGISTRY

            from umi.common.tasks.task_registry import TASK_REGISTRY, TaskSpec

            assert "plane_assembly_v9" in TASK_REGISTRY
            spec = TASK_REGISTRY["plane_assembly_v9"]
            assert isinstance(spec, TaskSpec)
            assert spec.mode == "bimanual"
            assert spec.HW_version == "v1"

    def test_get_task_helpers(self):
        import umi.common.tasks.task_registry_store as store

        with patch.object(store, "load_registry", return_value=SAMPLE_REGISTRY):
            store._REGISTRY = SAMPLE_REGISTRY

            from umi.common.tasks.task_registry import (
                get_task,
                task_names,
            )

            assert task_names() == [
                "plane_assembly_v9",
                "pnp_cylinder_weave_g1",
            ]
            assert get_task("plane_assembly_v9").training_group == "plane_assembly_v9"
            assert get_task("nonexistent") is None


class TestRequiredHoursField:
    """`required_hours` must read as None when unset -- 0 is never a valid
    target, and an absent one must never render as "complete"."""

    def test_spec_from_entry_required_hours(self):
        from umi.common.tasks.task_registry import _spec_from_entry

        assert _spec_from_entry("t", {}).required_hours is None
        assert _spec_from_entry("t", {"required_hours": 80}).required_hours == 80.0
        assert _spec_from_entry("t", {"required_hours": "45.5"}).required_hours == 45.5
        assert _spec_from_entry("t", {"required_hours": None}).required_hours is None


class TestTrainingStageInOverrides:
    """The stage lives in ``data_check_overrides.training_stage`` as
    ``{name, version}`` and is recorded into the session's ``meta``. Absent
    reads as pretrain/v1 so untagged entries keep loading."""

    def test_stage_of_defaults_to_pretrain_v1(self):
        from umi.common.tasks.task_registry import stage_of

        assert stage_of(None) == ("pretrain", "v1")
        assert stage_of({}) == ("pretrain", "v1")
        assert stage_of({"training_stage": "posttrain"}) == ("pretrain", "v1")
        assert stage_of({"training_stage": {"name": "midtrain"}}) == ("midtrain", "v1")
        assert stage_of({"training_stage": {"name": "posttrain", "version": "v2"}}) == (
            "posttrain",
            "v2",
        )

    def test_spec_reads_stage_from_overrides(self):
        from umi.common.tasks.task_registry import _spec_from_entry

        spec = _spec_from_entry("t", {})
        assert spec.training_stage() == ("pretrain", "v1")
        spec = _spec_from_entry(
            "t", {"data_check_overrides": {"training_stage": {"name": "posttrain"}}}
        )
        assert spec.training_stage() == ("posttrain", "v1")

    def test_top_level_stage_fields_are_gone(self):
        """The stage is not a TaskSpec field: it rides data_check_overrides."""
        import dataclasses

        from umi.common.tasks.task_registry import TaskSpec

        declared = {f.name for f in dataclasses.fields(TaskSpec)}
        assert "training_stage" not in declared
        assert "training_stage_version" not in declared

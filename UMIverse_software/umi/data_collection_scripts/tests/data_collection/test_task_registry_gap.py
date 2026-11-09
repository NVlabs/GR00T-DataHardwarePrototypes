# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A session must not silently record without setup text.

A task can be missing from the task registry, or the recorder's copy of the
registry can be older than the task. Either way, episodes would be recorded
with no operator instruction and nothing would show it. So the lookup re-reads
the registry once on a miss, the recorder shows a warning on its panel and in
the terminal, and every episode records where its registry copy came from.
"""

import time
from types import SimpleNamespace

import zarr

from umi.common.data.frame import EncoderFrame
from umi.common.hardware_sdk.sensor.mock_sensor import MockSensor
from umi.common.recording.numeric_recorder import NumericRecorder
from umi.common.recording.record_manager import RecorderManager
from umi.common.tasks import registry_gap, task_registry, task_registry_store

SPEC = SimpleNamespace(
    short_description="fold the towel",
    operator_instruction="Place towel on the mat.",
    objects=[],
)
NO_TEXT = SimpleNamespace(short_description="fold", operator_instruction="", objects=[])


def _info(source="bundled"):
    return {"source": source, "version": 42, "updated_at": "2026-09-18T22:31:46Z"}


def test_a_miss_refetches_once_and_then_finds_the_task(monkeypatch):
    calls = {"get": 0, "reload": 0}

    def get_task(name):
        calls["get"] += 1
        return None if calls["get"] == 1 else SPEC

    def reload():
        calls["reload"] += 1

    monkeypatch.setattr(task_registry, "get_task", get_task)
    monkeypatch.setattr(task_registry_store, "reload", reload)
    monkeypatch.setattr(task_registry_store, "get_registry_info", lambda: _info())
    spec, info, retried = registry_gap.resolve_task_spec("fold_towel_v1")
    assert spec is SPEC and retried is True
    assert calls == {"get": 2, "reload": 1}
    assert registry_gap.registry_warning("fold_towel_v1", spec, info) is None


def test_a_hit_does_not_refetch(monkeypatch):
    monkeypatch.setattr(task_registry, "get_task", lambda name: SPEC)

    def reload():
        raise AssertionError("no reload on a hit")

    monkeypatch.setattr(task_registry_store, "reload", reload)
    monkeypatch.setattr(task_registry_store, "get_registry_info", lambda: _info())
    spec, _, retried = registry_gap.resolve_task_spec("fold_towel_v1")
    assert spec is SPEC and retried is False


def test_the_two_warnings_worst_first():
    missing = registry_gap.registry_warning("x_v1", None, _info("bundled"))
    assert missing and missing.startswith("TASK NOT IN REGISTRY: x_v1")
    assert "task list from bundled" in missing
    unexpected = registry_gap.registry_warning("x_v1", SPEC, _info("cache"))
    assert unexpected and unexpected.startswith(
        "REGISTRY CAME FROM AN UNEXPECTED SOURCE: cache"
    )
    # Setup text is optional (the Station Console can add tasks without it), so
    # a registered task without any gets no warning.
    assert registry_gap.registry_warning("x_v1", NO_TEXT, _info("bundled")) is None
    # No warning for a registered task from the bundled snapshot, or from a local
    # file (source label 'file:<path>').
    assert registry_gap.registry_warning("x_v1", SPEC, _info("bundled")) is None
    assert (
        registry_gap.registry_warning("x_v1", SPEC, _info("file:/tmp/tasks.json"))
        is None
    )
    s3 = registry_gap.registry_warning("x_v1", SPEC, _info("s3"))
    assert s3 and s3.startswith("REGISTRY CAME FROM AN UNEXPECTED SOURCE: s3")


def test_provenance_says_what_the_episode_was_recorded_under():
    assert registry_gap.registry_provenance(SPEC, _info("bundled"), False) == {
        "source": "bundled",
        "version": 42,
        "updated_at": "2026-09-18T22:31:46Z",
        "task_registered": True,
        "has_operator_instruction": True,
        "refetched": False,
    }
    p = registry_gap.registry_provenance(None, _info("cache"), True)
    assert (p["task_registered"], p["has_operator_instruction"], p["refetched"]) == (
        False,
        False,
        True,
    )


def test_the_episode_carries_the_registry_provenance(tmp_path):
    """The provenance is stamped even when the episode has no instruction and no
    layout, the case it is there to explain."""
    record_path = str(tmp_path / "rec")
    source = MockSensor("right_arm")
    recorder = NumericRecorder(
        record_fps=60,
        stream_fps=60,
        record_path=record_path,
        numeric_sources=[source],
        frame_data_class=EncoderFrame,
        verbose=False,
        info_verbose=False,
        warning_verbose=False,
    )
    manager = RecorderManager(recorders=[recorder], record_path=record_path)
    manager.registry_provenance = {"source": "cache", "task_registered": False}
    manager.start_streaming()
    time.sleep(0.1)
    manager.reset_episode_recording()
    manager.start_recording()
    time.sleep(0.2)
    manager.stop_recording()
    manager.save_recordings(pending=False)
    manager.stop_streaming()
    attrs = dict(zarr.open(record_path, mode="r")["episode_0"].attrs)
    assert attrs["task_registry"] == {"source": "cache", "task_registered": False}
    assert "operator_instruction" not in attrs


def test_the_panel_is_built_once_warning_above_instruction_or_alone():
    panel = {
        "text": "SETUP\nPlace towel.\n\nTASK\nfold",
        "title": "Operator Instruction",
        "header_lines": ["SETUP", "TASK"],
        "header_color": (100, 200, 255),
    }
    # No warning: the instruction panel is returned unchanged.
    assert registry_gap.panel_with_warning(panel, None) == panel
    # Warning and instruction: the warning goes above the text, the headers are
    # merged, and the instruction's colour is kept.
    both = registry_gap.panel_with_warning(panel, "REGISTRY IS A CACHE COPY")
    assert both["text"].startswith(
        "REGISTRY WARNING\nREGISTRY IS A CACHE COPY\n\nSETUP"
    )
    assert both["header_lines"] == ["REGISTRY WARNING", "SETUP", "TASK"]
    assert both["header_color"] == (100, 200, 255)
    assert panel["header_lines"] == ["SETUP", "TASK"]  # input not mutated
    # Warning without instruction: the warning is the whole panel. Colours are
    # OpenCV BGR, so (80, 80, 255) renders red, not blue.
    alone = registry_gap.panel_with_warning(None, "TASK NOT IN REGISTRY: x")
    assert alone["text"] == "REGISTRY WARNING\nTASK NOT IN REGISTRY: x"
    assert alone["header_lines"] == ["REGISTRY WARNING"]
    assert alone["header_color"] == (80, 80, 255)


def test_one_staleness_rule_shared_by_recorder_and_station_ui():
    """`is_stale_copy` is the one staleness rule. The recorder and the Station
    Console both use it, so they always agree on a source label."""
    from umi.common.tasks.task_registry_store import is_stale_copy

    assert is_stale_copy("cache")
    assert is_stale_copy("fallback")
    assert is_stale_copy("s3")
    assert is_stale_copy("etcd")  # any unrecognised live-looking label warns
    assert not is_stale_copy("bundled")
    assert not is_stale_copy("file:/home/gear/tasks.json")
    assert not is_stale_copy(None)
    assert not is_stale_copy("")
    assert not is_stale_copy("unknown")
    # The recorder banner and the station card follow it.
    assert registry_gap.registry_warning("x", SPEC, {"source": "cache"}).startswith(
        "REGISTRY CAME FROM AN UNEXPECTED SOURCE"
    )
    assert (
        registry_gap.registry_warning("x", SPEC, {"source": "file:/tmp/t.json"}) is None
    )


def test_a_failed_refetch_is_written_to_the_station_log(monkeypatch, caplog):
    """A failed registry re-read is logged with its exception text: through the
    `log` callback the recorder passes (its stdout goes to the station log), or
    through the module logger when there is no callback."""
    monkeypatch.setattr(task_registry, "get_task", lambda name: None)

    def reload():
        raise OSError("endpoint unreachable")

    monkeypatch.setattr(task_registry_store, "reload", reload)
    monkeypatch.setattr(
        task_registry_store, "get_registry_info", lambda: _info("cache")
    )
    lines = []
    spec, info, retried = registry_gap.resolve_task_spec(
        "fold_towel_v1", log=lines.append
    )
    assert spec is None and retried is True and info["source"] == "cache"
    assert len(lines) == 1
    assert "re-read for 'fold_towel_v1' failed" in lines[0]
    assert "endpoint unreachable" in lines[0]
    # Default sink: the module logger.
    with caplog.at_level("WARNING", logger="umi.common.tasks.registry_gap"):
        registry_gap.resolve_task_spec("fold_towel_v1")
    assert any("endpoint unreachable" in r.getMessage() for r in caplog.records)


def test_a_blank_or_missing_source_warns_nowhere():
    """A blank, `None` or missing source gives no warning in the recorder or in
    the Station Console: `is_stale_copy` treats no evidence as no warning."""
    from station_ui.routes import config as config_routes

    for info in ({"source": ""}, {"source": None}, {}):
        assert registry_gap.registry_warning("x", SPEC, info) is None
        assert (
            config_routes.task_registry_store.is_stale_copy(info.get("source")) is False
        )


def test_missing_source_is_normalised_in_one_place(monkeypatch):
    """`source_label` is the one definition of "no source". The recorder's stamp
    and the Station Console's card both use it, so `{"source": ""}` means the
    same to both: no stale warning, and the stamp says "unknown"."""
    from station_ui.routes import config as config_routes
    from umi.common.tasks.task_registry_store import source_label

    for raw in (None, "", "  ", "unknown"):
        assert source_label(raw) is None
    assert source_label(" bundled ") == "bundled"
    assert source_label("file:/tmp/t.json") == "file:/tmp/t.json"
    for raw in (None, "", "unknown"):
        monkeypatch.setattr(
            task_registry_store, "get_registry_info", lambda raw=raw: {"source": raw}
        )
        assert config_routes._registry_source() is None
        assert config_routes._registry_is_stale_copy() is False
        text = registry_gap.registry_warning("x", None, {"source": raw})
        assert (
            text and text.startswith("TASK NOT IN REGISTRY") and "from unknown," in text
        )

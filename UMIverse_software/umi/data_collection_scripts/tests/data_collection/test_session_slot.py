# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the session-slot resolver in ``umi.common.recording.session_slot``.

When the current folder already holds episodes recorded under a different
config/calibration, ``resolve_session_slot`` rolls the session: it mints a fresh
uid onto the session folder (``recording.data_dir``) and onto
``meta.session_uid``, and records there instead of mixing configs in one folder.
``meta.task_name`` is left untouched — it stays the operator's real task, so the
task folder stays the same; only the session identity moves.

The content check (``session_file_conflicts``) compares the full session content —
the raw config-file snapshot (``launch_config.yaml``), the data-check config, and
the device-calibration files — so a config-file edit, a data-check change, or a
device recalibration all count as "changed". Nothing branches on which of them
changed. A folder is reused (no roll) when its content matches or it holds no
episodes yet (so a pre-first-episode recalibration doesn't leave the base folder
empty).

A roll mints a uid rather than appending an incrementing ``_N`` suffix: the uid
is read by anchoring on the end of the folder name, so ``…_<uid>_1`` would
parse as having no uid at all.
"""

import os
import shutil

import pytest
import yaml

from umi.common.recording import session_seal
from umi.common.recording.session_slot import (
    data_check_yaml_bytes,
    device_config_entries,
    persist_session,
    resolve_session_slot,
    roll_session,
    save_launch_config,
    session_file_conflicts,
)
from umi.common.recording.session_uid import (
    is_valid_session_uid,
    session_uid_from_dir_name,
)


@pytest.fixture(autouse=True)
def _registry_returns_none_by_default(monkeypatch):
    """Default: nothing is registered, so any data-check registry lookup returns
    None and never touches the real registry. The data-check test overrides
    ``reg.get_task`` itself."""
    import umi.common.tasks.task_registry as reg

    monkeypatch.setattr(reg, "get_task", lambda name: None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(
    data_dir: str,
    task: str,
    device_file: str | None = None,
    fps=None,
    session_uid: str | None = None,
) -> dict:
    """A minimal config. ``fps`` sets a config-file field (``recording.record_fps``)
    so tests can vary the config-file content; ``device_file`` attaches a device
    calibration file whose *content* is part of the content check.
    ``session_uid`` makes it a console-generated config rather than a
    hand-written CLI one — the two differ in whether a roll has a uid to move."""
    cfg: dict = {
        "meta": {"task_name": task},
        "recording": {"data_dir": data_dir},
        "video_sources": [],
    }
    if session_uid is not None:
        cfg["meta"]["session_uid"] = session_uid
    if fps is not None:
        cfg["recording"]["record_fps"] = fps
    if device_file:
        cfg["numeric_sources"] = [
            {"type": "gripper", "params": {"config_file": str(device_file)}}
        ]
    return cfg


def _write_session(
    folder: str, cfg: dict, check_config: str | None = None, episodes: int = 0
) -> None:
    """Write the content a real recorded-session folder would have — the
    ``launch_config.yaml`` snapshot, the bundled device-calibration files, the
    data-check config (if any) — plus ``episodes`` dummy episode dirs, so the folder
    looks like it holds recorded data for the resolver's checks."""
    os.makedirs(folder, exist_ok=True)
    save_launch_config(folder, cfg)
    for src_path, basename in device_config_entries(cfg):
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(folder, basename))
    dc = data_check_yaml_bytes(cfg, check_config)
    if dc is not None:
        with open(os.path.join(folder, "data_check_config.yaml"), "wb") as f:
            f.write(dc)
    for i in range(episodes):
        os.makedirs(os.path.join(folder, f"episode_{i}"), exist_ok=True)


def _write_working(
    path: str, task: str, data_dir: str, session_uid: str | None = None
) -> None:
    meta: dict = {"task_name": task}
    if session_uid is not None:
        meta["session_uid"] = session_uid
    with open(path, "w") as f:
        yaml.dump({"meta": meta, "recording": {"data_dir": data_dir}}, f)


UID = "0123456789ab"


def test_enforcement_choices_reuse_an_existing_session(tmp_path):
    folder = str(tmp_path / f"session_{UID}")
    cfg = _cfg(folder, "cup", fps=30, session_uid=UID)
    _write_session(folder, cfg, episodes=1)
    working = tmp_path / "working.yaml"
    working.write_text(yaml.safe_dump(cfg))
    cfg["check_enforcement"] = {"data": {"tracking": False}}

    assert session_file_conflicts(folder, cfg) == []
    resolve_session_slot(cfg, None, str(working))
    assert cfg["recording"]["data_dir"] == folder
    assert cfg["meta"]["session_uid"] == UID

    cfg["recording"]["record_fps"] = 60
    assert session_file_conflicts(folder, cfg) == ["launch_config.yaml"]
    resolve_session_slot(cfg, None, str(working))
    assert cfg["recording"]["data_dir"] != folder


@pytest.mark.parametrize("stored", ["invalid: [", "- a list", "null"])
def test_malformed_snapshot_is_a_conflict_instead_of_crashing(tmp_path, stored):
    (tmp_path / "launch_config.yaml").write_text(stored)
    assert session_file_conflicts(str(tmp_path), _cfg(str(tmp_path), "cup")) == [
        "launch_config.yaml"
    ]


# ---------------------------------------------------------------------------
# roll_session — the remedy, independent of what triggered it
# ---------------------------------------------------------------------------


def test_roll_replaces_a_trailing_uid_rather_than_appending():
    """Rolling twice must not yield ``…_<uid>_<uid>``: the uid is read by
    anchoring on the end of the name, so only the last would ever be seen."""
    cfg = _cfg(f"/d/2026-08-10_umi1R_cup_{UID}", "cup", session_uid=UID)

    rolled = roll_session(cfg)

    assert rolled.startswith("/d/2026-08-10_umi1R_cup_")
    assert rolled.count("_cup_") == 1
    new_uid = session_uid_from_dir_name(rolled)
    assert is_valid_session_uid(new_uid)
    assert new_uid != UID


def test_roll_appends_when_there_is_no_uid_to_replace():
    """A hand-written CLI ``data_dir`` predates the uid and must still come out
    unique."""
    cfg = _cfg("/d/2026_cup", "cup")

    rolled = roll_session(cfg)

    assert rolled.startswith("/d/2026_cup_")
    assert is_valid_session_uid(session_uid_from_dir_name(rolled))


def test_roll_moves_folder_and_meta_session_uid_together():
    """The session is identified by its folder, so a config whose
    ``meta.session_uid`` names a different session cannot say where its own
    episodes went. Both have to move together."""
    cfg = _cfg(f"/d/2026_cup_{UID}", "cup", session_uid=UID)

    rolled = roll_session(cfg)

    assert cfg["recording"]["data_dir"] == rolled
    assert cfg["meta"]["session_uid"] == session_uid_from_dir_name(rolled)
    assert cfg["meta"]["session_uid"] != UID


def test_roll_leaves_a_uidless_config_without_one():
    """A hand-written CLI config has no session uid to keep in step, and
    inventing one would hand RecorderManager a console-minted session it never
    had."""
    cfg = _cfg("/d/2026_cup", "cup")

    roll_session(cfg)

    assert "session_uid" not in cfg["meta"]


def test_roll_only_touches_the_last_path_segment():
    """A parent directory that happens to end in 12 hex characters is not the
    session, and renaming through it would move the folder somewhere else."""
    cfg = _cfg(f"/d/{UID}/2026_cup", "cup")

    rolled = roll_session(cfg)

    assert rolled.startswith(f"/d/{UID}/2026_cup_")


def test_persist_session_writes_both_fields(tmp_path):
    working = tmp_path / "working.yaml"
    _write_working(str(working), "cup", f"/d/2026_cup_{UID}", session_uid=UID)
    cfg = _cfg(f"/d/2026_cup_{UID}", "cup", session_uid=UID)
    rolled = roll_session(cfg)

    persist_session(str(working), cfg)

    persisted = yaml.safe_load(working.read_text())
    assert persisted["recording"]["data_dir"] == rolled
    assert persisted["meta"]["session_uid"] == session_uid_from_dir_name(rolled)
    assert persisted["meta"]["task_name"] == "cup"  # untouched


# ---------------------------------------------------------------------------
# session_file_conflicts — full content check (config file + data-check + device)
# ---------------------------------------------------------------------------


def test_conflicts_fresh_folder_is_empty(tmp_path):
    cfg = _cfg(str(tmp_path / "cup"), "cup")
    assert session_file_conflicts(str(tmp_path / "cup"), cfg) == []


def test_conflicts_matching_config_is_empty(tmp_path):
    folder = str(tmp_path / "cup")
    cfg = _cfg(folder, "cup")
    _write_session(folder, cfg)
    assert session_file_conflicts(folder, cfg) == []


def test_conflicts_config_content_change(tmp_path):
    """Editing the config file (record_fps) is a conflict."""
    folder = str(tmp_path / "cup")
    _write_session(folder, _cfg(folder, "cup", fps=30))
    assert session_file_conflicts(folder, _cfg(folder, "cup", fps=60)) == [
        "launch_config.yaml"
    ]


def test_conflicts_device_recalibration_is_conflict(tmp_path):
    """A recalibration (device-file content change) is a conflict: the device
    files are part of the content check, even when the config file is unchanged."""
    folder = str(tmp_path / "cup")
    dev = tmp_path / "cal.yaml"
    dev.write_text("v: 1\n")
    cfg = _cfg(folder, "cup", device_file=dev)
    _write_session(folder, cfg)
    dev.write_text("v: 2\n")  # recalibrate
    assert session_file_conflicts(folder, cfg) == ["cal.yaml"]


def test_conflicts_missing_live_device_is_conflict(tmp_path):
    """A stored device config whose live source disappeared is a conflict."""
    folder = str(tmp_path / "cup")
    dev = tmp_path / "cal.yaml"
    dev.write_text("v: 1\n")
    cfg = _cfg(folder, "cup", device_file=dev)
    _write_session(folder, cfg)
    dev.unlink()
    assert session_file_conflicts(folder, cfg) == ["cal.yaml"]


# ---------------------------------------------------------------------------
# resolve_session_slot
# ---------------------------------------------------------------------------


def test_resolve_fresh_folder_no_roll(tmp_path):
    data_dir = str(tmp_path / "2026_cup")
    cfg = _cfg(data_dir, "cup")
    got = resolve_session_slot(cfg, None, str(tmp_path / "working.yaml"))
    assert got == os.path.expanduser(data_dir)
    assert cfg["meta"]["task_name"] == "cup"
    assert cfg["recording"]["data_dir"] == data_dir


def test_resolve_matching_config_resumes(tmp_path):
    data_dir = str(tmp_path / "2026_cup")
    cfg = _cfg(data_dir, "cup")
    _write_session(data_dir, cfg, episodes=1)
    got = resolve_session_slot(cfg, None, str(tmp_path / "working.yaml"))
    assert got == os.path.expanduser(data_dir)  # content matches -> resume
    assert cfg["recording"]["data_dir"] == data_dir


def test_resolve_recalibration_reuses_empty_folder(tmp_path):
    """A recalibration conflicts, but if the folder has no episodes yet it's reused
    (overwritten) — so the base folder isn't left empty by a calibrate-before-first-
    episode workflow."""
    data_dir = str(tmp_path / "2026_cup")
    dev = tmp_path / "cal.yaml"
    dev.write_text("v: 1\n")
    _write_session(data_dir, _cfg(data_dir, "cup", device_file=dev))  # 0 episodes
    dev.write_text("v: 2\n")  # recalibrate -> device conflict
    cfg = _cfg(data_dir, "cup", device_file=dev)
    got = resolve_session_slot(cfg, None, str(tmp_path / "working.yaml"))
    assert got == os.path.expanduser(data_dir)  # empty -> reused despite conflict
    assert cfg["recording"]["data_dir"] == data_dir


def test_resolve_recalibration_with_episodes_rolls(tmp_path):
    """A recalibration when the folder already holds episodes rolls onto a fresh
    session (don't mix calibrations in one folder). task_name stays clean."""
    data_dir = str(tmp_path / f"2026_cup_{UID}")
    dev = tmp_path / "cal.yaml"
    dev.write_text("v: 1\n")
    _write_session(
        data_dir, _cfg(data_dir, "cup", device_file=dev, session_uid=UID), episodes=1
    )
    dev.write_text("v: 2\n")  # recalibrate
    cfg = _cfg(data_dir, "cup", device_file=dev, session_uid=UID)
    working = tmp_path / "working.yaml"
    _write_working(str(working), "cup", data_dir, session_uid=UID)

    got = resolve_session_slot(cfg, None, str(working))

    assert got != os.path.expanduser(data_dir)
    assert cfg["meta"]["task_name"] == "cup"  # unchanged
    assert cfg["recording"]["data_dir"] == got
    assert session_uid_from_dir_name(got) == cfg["meta"]["session_uid"] != UID


def test_resolve_config_change_rolls_and_persists(tmp_path):
    """Editing the config file rolls the session (folder holds episodes); task
    stays, and both halves of the identity reach the working config."""
    data_dir = str(tmp_path / f"2026_cup_{UID}")
    _write_session(data_dir, _cfg(data_dir, "cup", fps=30, session_uid=UID), episodes=1)
    cfg = _cfg(data_dir, "cup", fps=60, session_uid=UID)
    working = tmp_path / "working.yaml"
    _write_working(str(working), "cup", data_dir, session_uid=UID)

    got = resolve_session_slot(cfg, None, str(working))

    assert got != os.path.expanduser(data_dir)
    assert cfg["meta"]["task_name"] == "cup"  # unchanged
    persisted = yaml.safe_load(working.read_text())
    assert persisted["meta"]["task_name"] == "cup"  # unchanged
    assert persisted["recording"]["data_dir"] == cfg["recording"]["data_dir"]
    assert persisted["meta"]["session_uid"] == session_uid_from_dir_name(got)


def test_resolve_rolled_name_still_parses_as_a_session(tmp_path):
    """A rolled folder keeps a readable uid: a suffixed ``…_<uid>_1`` would have
    none, and the console could not identify the folder its own recorder moved to."""
    data_dir = str(tmp_path / f"2026-08-10_umi1R_cup_{UID}")
    _write_session(data_dir, _cfg(data_dir, "cup", fps=30, session_uid=UID), episodes=1)
    cfg = _cfg(data_dir, "cup", fps=60, session_uid=UID)

    got = resolve_session_slot(cfg, None, str(tmp_path / "working.yaml"))

    assert is_valid_session_uid(session_uid_from_dir_name(got))
    # The recording date is still the first ten characters -- every date-range
    # query downstream reads it positionally.
    assert os.path.basename(got).startswith("2026-08-10_")


def test_resolve_strips_trailing_slash_before_rolling(tmp_path):
    base = str(tmp_path / "2026_cup")
    _write_session(base, _cfg(base, "cup", fps=30), episodes=1)
    cfg = _cfg(base + "/", "cup", fps=60)  # config changed + trailing slash
    got = resolve_session_slot(cfg, None, str(tmp_path / "working.yaml"))
    assert got.startswith(os.path.expanduser(base) + "_")
    assert "/_" not in got  # the uid landed on the basename, not after the slash
    assert cfg["meta"]["task_name"] == "cup"


def test_resolve_no_double_roll_on_restart(tmp_path):
    """Relaunching into the rolled folder with matching content resumes it."""
    rolled = str(tmp_path / f"2026_cup_{UID}")
    _write_session(rolled, _cfg(rolled, "cup", fps=60, session_uid=UID), episodes=1)
    cfg = _cfg(rolled, "cup", fps=60, session_uid=UID)  # content matches
    got = resolve_session_slot(cfg, None, str(tmp_path / "working.yaml"))
    assert got == os.path.expanduser(rolled)  # resumes, no roll
    assert cfg["recording"]["data_dir"] == rolled
    assert cfg["meta"]["session_uid"] == UID


def test_resolve_second_change_rolls_again(tmp_path):
    """A second drift rolls again rather than accumulating suffixes."""
    first = str(tmp_path / f"2026_cup_{UID}")
    _write_session(first, _cfg(first, "cup", fps=60, session_uid=UID), episodes=1)
    cfg = _cfg(first, "cup", fps=90, session_uid=UID)  # changed again

    got = resolve_session_slot(cfg, None, str(tmp_path / "working.yaml"))

    assert got != os.path.expanduser(first)
    # One uid, not two: the basename is still date_task_uid.
    assert os.path.basename(got).count("_") == os.path.basename(first).count("_")


def test_resolve_new_task_uses_fresh_folder(tmp_path):
    """Switching to a different task records into that task's own fresh folder."""
    _write_session(
        str(tmp_path / "2026_cup"),
        _cfg(str(tmp_path / "2026_cup"), "cup"),
        episodes=1,
    )
    stack_dir = str(tmp_path / "2026_stack")
    cfg = _cfg(stack_dir, "stack")
    got = resolve_session_slot(cfg, None, str(tmp_path / "working.yaml"))
    assert got == os.path.expanduser(stack_dir)  # fresh folder, no suffix
    assert cfg["meta"]["task_name"] == "stack"


def test_resolve_survives_an_unwritable_working_config(tmp_path):
    """Persistence is best-effort: the episodes are already going to the rolled
    folder, so a config the station cannot rewrite must not cost the recording."""
    data_dir = str(tmp_path / f"2026_cup_{UID}")
    _write_session(data_dir, _cfg(data_dir, "cup", fps=30, session_uid=UID), episodes=1)
    cfg = _cfg(data_dir, "cup", fps=60, session_uid=UID)

    got = resolve_session_slot(cfg, None, str(tmp_path / "no" / "such" / "file.yaml"))

    assert got == os.path.expanduser(cfg["recording"]["data_dir"])
    assert got != os.path.expanduser(data_dir)


# Task names ending in _N are distinct registry entries, not collision suffixes.


def test_data_check_uses_task_name_directly(tmp_path, monkeypatch):
    """The data-check overrides are looked up on the task name verbatim — a real
    task whose name ends in _N (cup_1) uses its own overrides, not a stripped
    base's."""
    import umi.common.tasks.task_registry as reg

    gcfg = tmp_path / "data_check.yaml"
    gcfg.write_text("threshold: 1\n")

    class _Spec:
        def __init__(self, value):
            self.data_check_overrides = {"threshold": value}

    specs = {"cup": _Spec(99), "cup_1": _Spec(7)}
    monkeypatch.setattr(reg, "get_task", lambda name: specs.get(name))

    cup = data_check_yaml_bytes({"meta": {"task_name": "cup"}}, str(gcfg))
    cup_1 = data_check_yaml_bytes({"meta": {"task_name": "cup_1"}}, str(gcfg))
    stack = data_check_yaml_bytes({"meta": {"task_name": "stack"}}, str(gcfg))

    assert b"99" in cup
    assert b"7" in cup_1 and b"99" not in cup_1
    assert b"threshold: 1" in stack


# ---------------------------------------------------------------------------
# The training stage is metadata: it does not select a data-check overlay.
# ---------------------------------------------------------------------------


def test_a_training_stage_changes_no_check_threshold(tmp_path, monkeypatch):
    """A tagged task is checked against the default plus its own overrides,
    exactly like an untagged one: the training stage selects no data-check
    overlay."""
    import umi.common.tasks.task_registry as reg

    gcfg = tmp_path / "data_check.yaml"
    gcfg.write_text("tracking:\n  enabled: true\n  pose_jump_threshold_m_per_s: 4.5\n")

    class _Spec:
        data_check_overrides = {
            "training_stage": {"name": "posttrain", "version": "v1"}
        }

    monkeypatch.setattr(reg, "get_task", lambda name: _Spec())

    raw = data_check_yaml_bytes({"meta": {"task_name": "any"}}, str(gcfg))
    assert yaml.safe_load(raw)["tracking"] == {
        "enabled": True,
        "pose_jump_threshold_m_per_s": 4.5,
    }


# Sealed sessions cannot be reused; every launch path must select a new UID.


def _seal(folder):
    folder.mkdir(parents=True, exist_ok=True)
    assert (
        session_seal.write_seal(folder, reason="operator", episode_count=1) is not None
    )


def test_sealed_folder_is_never_usable_even_when_empty(tmp_path):
    from umi.common.recording.session_slot import slot_is_usable

    folder = tmp_path / "2026-08-26_umiV3_0001R_vive_task_a1b2c3d4e5f6"
    _seal(folder)
    # No episodes and no config to conflict with: only the seal refuses it.
    assert slot_is_usable(str(folder), {}, None) is False


def test_unsealed_empty_folder_is_still_usable(tmp_path):
    from umi.common.recording.session_slot import slot_is_usable

    folder = tmp_path / "2026-08-26_umiV3_0001R_vive_task_a1b2c3d4e5f6"
    folder.mkdir()
    assert slot_is_usable(str(folder), {}, None) is True


def test_resolve_session_slot_rolls_off_a_sealed_folder(tmp_path):
    from umi.common.recording.session_slot import resolve_session_slot
    from umi.common.recording.session_uid import session_uid_from_dir_name

    folder = tmp_path / "2026-08-26_umiV3_0001R_vive_task_a1b2c3d4e5f6"
    _seal(folder)
    cfg = {
        "recording": {"data_dir": str(folder)},
        "meta": {"session_uid": "a1b2c3d4e5f6"},
    }

    resolved = resolve_session_slot(cfg, None, str(tmp_path / "config.yaml"))

    assert resolved != str(folder)
    assert session_uid_from_dir_name(resolved) != "a1b2c3d4e5f6"
    # Both halves of the identity move together.
    assert cfg["meta"]["session_uid"] == session_uid_from_dir_name(resolved)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for session-config bundling + per-episode snapshots in RecorderManager.

There is no write-time content lock: the launch-time session-slot resolver decides
which folder to record into (by comparing the raw ``launch_config.yaml`` snapshot),
so ``RecorderManager`` just writes the session's ``config.yaml`` /
``data_check_config.yaml`` / device-calibration files for provenance. Restarting
into an existing folder — whether the config, data-check, or a device file changed
— is allowed; the session copies are simply refreshed.

What does hold:
1. The bundled files are written to the session folder (provenance).
2. ``_save_episode_configs`` snapshots every config into an episode only from the
   session copy — never in-memory state or the live source — so a mid-session edit
   to the live inputs cannot leak into an already-recorded episode.
"""

import os
from pathlib import Path
from unittest.mock import patch

import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_init():
    """No-op the unrelated RecorderManager init helpers (keep device-config ones).

    Patches out config.yaml / data_check_config.yaml saving so the device-config
    lock can be exercised in isolation.
    """
    return patch.multiple(
        "umi.common.recording.record_manager.RecorderManager",
        _save_config=lambda _: None,
        _save_data_check_config=lambda _: None,
        _validate_reference_camera=lambda _: None,
        _save_meta_data=lambda _: None,
    )


def _patch_init_keep_yaml():
    """No-op only the hardware/git/zarr init helpers; let the file-saving run.

    Keeps ``_save_config`` / ``_save_data_check_config`` / ``_save_device_configs``
    real so the session-level YAML locks and per-episode snapshots can be tested.
    """
    return patch.multiple(
        "umi.common.recording.record_manager.RecorderManager",
        _validate_reference_camera=lambda _: None,
        _save_meta_data=lambda _: None,
    )


def _write_device_config(path: Path, content: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(content, f)
    return str(path)


def _config_with_device(config_file: str) -> dict:
    return {
        "numeric_sources": [
            {
                "name": "right_umi_gripper",
                "type": "umi_gripper",
                "params": {"config_file": config_file},
            }
        ]
    }


def _make_manager(
    record_path: str,
    config: dict | None,
    recorders=None,
    data_check_config_path: str | None = None,
    keep_yaml: bool = False,
):
    ctx = _patch_init_keep_yaml() if keep_yaml else _patch_init()
    with ctx:
        from umi.common.recording.record_manager import RecorderManager

        return RecorderManager(
            recorders=recorders or [],
            record_path=record_path,
            config=config,
            data_check_config_path=data_check_config_path,
        )


class _FakeRecorder:
    def __init__(self, episode_id: int = 0):
        self.episode_id = episode_id


# ---------------------------------------------------------------------------
# _save_device_configs — session content lock
# ---------------------------------------------------------------------------


def test_device_config_saved_to_session(tmp_path):
    cfg_file = _write_device_config(
        tmp_path / "src" / "exo.yaml", {"mapping": {0: [1]}}
    )
    record_path = str(tmp_path / "dataset")
    os.makedirs(record_path, exist_ok=True)

    _make_manager(record_path, _config_with_device(cfg_file))

    # Session copy exists with identical content.
    session_copy = Path(record_path) / "exo.yaml"
    assert session_copy.is_file()
    assert session_copy.read_bytes() == Path(cfg_file).read_bytes()


def test_restart_identical_config_is_allowed(tmp_path):
    cfg_file = _write_device_config(
        tmp_path / "src" / "exo.yaml", {"mapping": {0: [1]}}
    )
    record_path = str(tmp_path / "dataset")
    os.makedirs(record_path, exist_ok=True)

    _make_manager(record_path, _config_with_device(cfg_file))
    # Re-run into the same session with the same file — must not raise.
    _make_manager(record_path, _config_with_device(cfg_file))


def test_restart_device_content_edited_in_place_is_allowed(tmp_path):
    # A recalibration edits the device file in place; config.yaml is unchanged and
    # device content is not locked -> restart is allowed and the session copy is
    # refreshed to the recalibrated content (the folder is reused, not bumped).
    cfg_path = tmp_path / "src" / "exo.yaml"
    cfg_file = _write_device_config(cfg_path, {"mapping": {0: [1]}})
    record_path = str(tmp_path / "dataset")
    os.makedirs(record_path, exist_ok=True)

    _make_manager(record_path, _config_with_device(cfg_file))

    _write_device_config(cfg_path, {"mapping": {0: [999]}})
    _make_manager(record_path, _config_with_device(cfg_file))  # must not raise

    session_copy = Path(record_path) / "exo.yaml"
    assert session_copy.is_file()
    assert yaml.safe_load(session_copy.read_text()) == {"mapping": {0: [999]}}


def test_restart_device_config_path_change_is_allowed(tmp_path):
    # Session resolution handles config conflicts before writing. Saving again
    # in the selected folder refreshes its provenance snapshot.
    content = {"mapping": {0: [1]}}
    cfg_a = _write_device_config(tmp_path / "a" / "exo.yaml", content)
    cfg_b = _write_device_config(tmp_path / "b" / "exo.yaml", content)  # same content
    record_path = str(tmp_path / "dataset")
    os.makedirs(record_path, exist_ok=True)

    _make_manager(record_path, _config_with_device(cfg_a), keep_yaml=True)
    _make_manager(record_path, _config_with_device(cfg_b), keep_yaml=True)  # no raise

    saved = yaml.safe_load((Path(record_path) / "config.yaml").read_text())
    assert saved["numeric_sources"][0]["params"]["config_file"] == cfg_b  # refreshed


def test_restart_device_config_removed_is_allowed(tmp_path):
    # Removing the device config changes config.yaml, which is not locked;
    # re-running into the same folder just refreshes config.yaml.
    cfg_file = _write_device_config(
        tmp_path / "src" / "exo.yaml", {"mapping": {0: [1]}}
    )
    record_path = str(tmp_path / "dataset")
    os.makedirs(record_path, exist_ok=True)

    _make_manager(record_path, _config_with_device(cfg_file), keep_yaml=True)
    _make_manager(record_path, {"numeric_sources": []}, keep_yaml=True)  # no raise

    saved = yaml.safe_load((Path(record_path) / "config.yaml").read_text())
    assert saved.get("numeric_sources") == []  # refreshed


# ---------------------------------------------------------------------------
# _save_episode_configs — snapshot only from session copy
# ---------------------------------------------------------------------------


def test_episode_snapshot_uses_session_copy_not_edited_source(tmp_path):
    cfg_path = tmp_path / "src" / "exo.yaml"
    cfg_file = _write_device_config(cfg_path, {"mapping": {0: [1]}})
    record_path = str(tmp_path / "dataset")
    os.makedirs(record_path, exist_ok=True)

    mgr = _make_manager(
        record_path,
        _config_with_device(cfg_file),
        recorders=[_FakeRecorder(0)],
        keep_yaml=True,
    )

    # Edit the live source AFTER the session copy was written.
    original_bytes = (Path(record_path) / "exo.yaml").read_bytes()
    _write_device_config(cfg_path, {"mapping": {0: [999]}})

    from umi.common.recording.base_recorder import PENDING_SUFFIX

    episode_dir = Path(record_path) / f"episode_0{PENDING_SUFFIX}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    mgr._save_episode_configs()

    snap = episode_dir / "exo.yaml"
    assert snap.is_file()
    # The snapshot must match the session calibration, not the edited source.
    assert snap.read_bytes() == original_bytes
    assert snap.read_bytes() != cfg_path.read_bytes()


def test_episode_snapshot_raises_when_session_copy_missing(tmp_path):
    cfg_file = _write_device_config(
        tmp_path / "src" / "exo.yaml", {"mapping": {0: [1]}}
    )
    record_path = str(tmp_path / "dataset")
    os.makedirs(record_path, exist_ok=True)

    mgr = _make_manager(
        record_path,
        _config_with_device(cfg_file),
        recorders=[_FakeRecorder(0)],
        keep_yaml=True,
    )

    # Remove the session copy to simulate the unrecoverable provenance gap.
    os.remove(Path(record_path) / "exo.yaml")

    from umi.common.recording.base_recorder import PENDING_SUFFIX

    episode_dir = Path(record_path) / f"episode_0{PENDING_SUFFIX}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    import pytest

    with pytest.raises(FileNotFoundError):
        mgr._save_episode_configs()


# ---------------------------------------------------------------------------
# config.yaml / data_check_config.yaml — session lock
# ---------------------------------------------------------------------------


def test_config_yaml_identical_restart_ok(tmp_path):
    record_path = str(tmp_path / "dataset")
    os.makedirs(record_path, exist_ok=True)
    config = {"video_sources": [{"name": "cam"}]}

    _make_manager(record_path, config, keep_yaml=True)
    _make_manager(record_path, dict(config), keep_yaml=True)  # no raise
    assert (Path(record_path) / "config.yaml").is_file()


def test_config_yaml_drift_is_allowed(tmp_path):
    # config.yaml is not content-locked at write time: the resolver's
    # launch_config.yaml decides folder reuse before recording, so a restart into
    # the same folder just refreshes config.yaml instead of raising.
    record_path = str(tmp_path / "dataset")
    os.makedirs(record_path, exist_ok=True)

    _make_manager(record_path, {"video_sources": [{"name": "cam"}]}, keep_yaml=True)
    _make_manager(record_path, {"video_sources": [{"name": "cam2"}]}, keep_yaml=True)

    saved = yaml.safe_load((Path(record_path) / "config.yaml").read_text())
    assert saved["video_sources"][0]["name"] == "cam2"  # refreshed, not raised


def test_data_check_config_drift_is_allowed(tmp_path):
    record_path = str(tmp_path / "dataset")
    os.makedirs(record_path, exist_ok=True)
    dc_path = tmp_path / "global_data_check.yaml"
    with open(dc_path, "w") as f:
        yaml.dump({"threshold": 1}, f)
    config = {"video_sources": [{"name": "cam"}]}

    _make_manager(
        record_path, config, data_check_config_path=str(dc_path), keep_yaml=True
    )
    assert (Path(record_path) / "data_check_config.yaml").is_file()

    # Edit the global data-check config, then restart into the same session. Only
    # config.yaml is compared (unchanged here), so this is allowed and the session
    # data_check copy is refreshed in place.
    with open(dc_path, "w") as f:
        yaml.dump({"threshold": 2}, f)
    _make_manager(
        record_path, config, data_check_config_path=str(dc_path), keep_yaml=True
    )
    saved = yaml.safe_load((Path(record_path) / "data_check_config.yaml").read_text())
    assert saved == {"threshold": 2}  # refreshed, not raised


def test_episode_config_snapshot_from_session_not_inmemory(tmp_path):
    record_path = str(tmp_path / "dataset")
    os.makedirs(record_path, exist_ok=True)
    config = {"video_sources": [{"name": "cam"}]}

    mgr = _make_manager(
        record_path, config, recorders=[_FakeRecorder(0)], keep_yaml=True
    )
    session_bytes = (Path(record_path) / "config.yaml").read_bytes()

    # Mutate the in-memory config after the session file was pinned.
    mgr.config["video_sources"][0]["name"] = "mutated"

    from umi.common.recording.base_recorder import PENDING_SUFFIX

    episode_dir = Path(record_path) / f"episode_0{PENDING_SUFFIX}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    mgr._save_episode_configs()

    snap = (episode_dir / "config.yaml").read_bytes()
    # Episode snapshot matches the session copy, not the mutated memory.
    assert snap == session_bytes
    assert b"mutated" not in snap

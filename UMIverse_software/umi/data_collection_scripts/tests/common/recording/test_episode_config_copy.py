# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each saved episode carries a byte-for-byte copy of the session's config.yaml."""

import os
from pathlib import Path
from unittest.mock import patch

from umi.common.recording.base_recorder import PENDING_SUFFIX


class _FakeRecorder:
    def __init__(self, episode_id: int = 0):
        self.episode_id = episode_id

    def set_episode_id(self, episode_id: int) -> None:
        self.episode_id = episode_id


def _make_manager(record_path: str, config):
    with patch.multiple(
        "umi.common.recording.record_manager.RecorderManager",
        _validate_reference_camera=lambda _: None,
        _save_meta_data=lambda _: None,
    ):
        from umi.common.recording.record_manager import RecorderManager

        return RecorderManager(
            recorders=[_FakeRecorder(0)], record_path=record_path, config=config
        )


def _save_episode(mgr, record_path: str) -> Path:
    episode_dir = Path(record_path) / f"episode_{mgr.episode_id}{PENDING_SUFFIX}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    mgr._save_episode_configs()
    return episode_dir / "config.yaml"


def test_the_episode_copy_is_the_session_config_byte_for_byte(tmp_path):
    record_path = str(tmp_path / "dataset")
    os.makedirs(record_path)
    mgr = _make_manager(record_path, {"a": 1})
    session_bytes = (Path(record_path) / "config.yaml").read_bytes()

    assert _save_episode(mgr, record_path).read_bytes() == session_bytes

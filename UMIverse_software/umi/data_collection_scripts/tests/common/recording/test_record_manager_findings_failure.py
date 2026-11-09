# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A findings attr that cannot be written never costs the episode.

`data_check_findings` is report-only. If writing it fails, `finalize_episode`
must still rename the episode and queue its alignment, and `quarantine_staging`
must still quarantine it with its issues.

Run with: uv run pytest tests/common/recording/test_record_manager_findings_failure.py -v
"""

import json
import os
from unittest.mock import Mock, patch

import pytest
import zarr
from zarr.core.attributes import Attributes

from umi.common.utility.output_layout import output_root, session_dir

TASK = "unit_test_task"
SESSION = "2026-01-01_unit_test_task"
FINDINGS = {"v": 1, "tracking_excursions": {"left_tracking_camera": {"events": []}}}
ISSUES = ["left_tracking_camera: pose jumps detected (1 jumps > 4.5m/s)"]


class _FakeRecorder:
    def __init__(self):
        self.episode_id = 0

    def set_episode_id(self, episode_id):
        self.episode_id = episode_id

    def finalize_episode(self):
        self.episode_id += 1

    def discard_staging(self):
        pass


@pytest.fixture
def manager(monkeypatch):
    record_path = session_dir("raw", TASK, SESSION)
    os.makedirs(record_path)
    with patch.multiple(
        "umi.common.recording.record_manager.RecorderManager",
        _validate_reference_camera=lambda _: None,
        _save_meta_data=lambda _: None,
    ):
        from umi.common.recording.record_manager import RecorderManager

        mgr = RecorderManager(
            recorders=[_FakeRecorder()],
            record_path=record_path,
            config={"meta": {"task_name": TASK}},
        )
    zarr.open_group(record_path, mode="a")
    for name in (
        "_save_episode_length_to_dataset",
        "_save_num_episodes",
        "_save_total_data_time",
    ):
        monkeypatch.setattr(mgr, name, lambda *a, **k: None)
    mgr.background_aligner = Mock()

    # Every write of the findings attr fails; every other attr still works.
    real_setitem = Attributes.__setitem__

    def failing_setitem(self, key, value):
        if key == "data_check_findings":
            raise OSError("disk full")
        real_setitem(self, key, value)

    monkeypatch.setattr(Attributes, "__setitem__", failing_setitem)
    return mgr


def _stage(mgr):
    group = zarr.open_group(mgr.get_pending_episode_path(), mode="w")
    group.create_group("left_tracking_camera")


def _attrs(episode_dir):
    with open(os.path.join(episode_dir, "zarr.json")) as f:
        return json.load(f).get("attributes", {})


def test_finalize_still_finalizes(manager):
    _stage(manager)
    pending = manager.get_pending_episode_path()
    final = manager.get_final_episode_path()

    assert manager.finalize_episode(findings=FINDINGS) is True

    assert not os.path.exists(pending) and os.path.isdir(final)
    assert "data_check_findings" not in _attrs(final)
    manager.background_aligner.queue_alignment.assert_called_once_with(episode_idx=0)


def test_quarantine_still_quarantines_with_its_issues(manager):
    _stage(manager)
    pending = manager.get_pending_episode_path()

    assert manager.quarantine_staging(issues=ISSUES, findings=FINDINGS) is True

    dest = output_root() / "quarantine" / TASK / SESSION / "episode_0"
    assert not os.path.exists(pending) and dest.is_dir()
    attrs = _attrs(dest)
    assert attrs["data_check_issues"] == ISSUES
    assert "data_check_findings" not in attrs
    manager.background_aligner.queue_alignment.assert_not_called()

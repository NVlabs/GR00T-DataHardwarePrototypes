# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data-check findings persisted on the episode's zarr attrs.

`finalize_episode(findings=...)` and `quarantine_staging(issues, findings)`
write `attrs["data_check_findings"]` onto the episode group: the finalized
episode in `output/raw/{task}/{session}`, the quarantined one in
`output/quarantine/{task}/{session}`.

Exercised on a real RecorderManager over a tmp dataset, with the dataset-level
bookkeeping (episode length / count / total time) stubbed out -- those read
recorder state these tests do not build and are not what is under test.

Run with: uv run pytest tests/common/recording/test_record_manager_findings.py -v
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import zarr

from umi.common.utility.output_layout import output_root, session_dir

TASK = "unit_test_task"
SESSION = "2026-01-01_unit_test_task"

FINDINGS = {
    "v": 1,
    "tracking_excursions": {
        "left_tracking_camera": {
            "detector_version": 1,
            "num_samples": 150,
            "max_dv_m_per_s": 3.0,
            "num_nominated": 1,
            "num_events": 1,
            "num_candidates": 1,
            "events": [
                {
                    "shape": "out_and_back",
                    "i0": 60,
                    "i1": 67,
                    "t_out_s": 2.0,
                    "t_back_s": 2.2333,
                    "amplitude_mm": 100.0,
                    "edge_rotation_deg_per_m_back": None,
                    "rotation_quiet": True,
                    "state": "candidate",
                    "reason": None,
                }
            ],
        },
        "right_tracking_camera": {
            "detector_version": 1,
            "num_samples": 150,
            "max_dv_m_per_s": 0.1,
            "num_nominated": 0,
            "num_events": 0,
            "num_candidates": 0,
            "events": [],
        },
    },
}
ISSUES = ["left_tracking_camera: pose jumps detected (1 jumps > 4.5m/s)"]


class _FakeRecorder:
    def __init__(self, episode_id: int = 0):
        self.episode_id = episode_id

    def set_episode_id(self, episode_id: int) -> None:
        self.episode_id = episode_id

    def finalize_episode(self) -> None:
        self.episode_id += 1

    def discard_staging(self) -> None:
        pass


def _zarr_attrs(episode_dir) -> dict:
    """Attributes as they are on disk: straight from zarr.json."""
    with open(os.path.join(episode_dir, "zarr.json")) as f:
        return json.load(f).get("attributes", {})


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
            recorders=[_FakeRecorder(0)],
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
    return mgr


def _stage_pending_episode(mgr) -> Path:
    pending = Path(mgr.get_pending_episode_path())
    group = zarr.open_group(str(pending), mode="w")
    group.attrs["task_description"] = "pick up the thing"
    group.create_group("left_tracking_camera")
    return pending


# --------------------------------------------------------------------------
# finalize_episode
# --------------------------------------------------------------------------


class TestFinalizeEpisode:
    def test_findings_written_to_the_final_episode(self, manager):
        _stage_pending_episode(manager)
        final_path = manager.get_final_episode_path()

        assert manager.finalize_episode(findings=FINDINGS) is True

        attrs = _zarr_attrs(final_path)
        assert attrs["data_check_findings"] == FINDINGS
        # Existing attrs are kept.
        assert attrs["task_description"] == "pick up the thing"

    def test_findings_readable_through_zarr(self, manager):
        _stage_pending_episode(manager)
        final_path = manager.get_final_episode_path()

        manager.finalize_episode(findings=FINDINGS)

        group = zarr.open_group(final_path, mode="r")
        assert group.attrs["data_check_findings"] == FINDINGS

    def test_none_writes_no_attr(self, manager):
        _stage_pending_episode(manager)
        final_path = manager.get_final_episode_path()

        assert manager.finalize_episode(findings=None) is True

        assert "data_check_findings" not in _zarr_attrs(final_path)

    def test_old_call_without_findings_still_works(self, manager):
        _stage_pending_episode(manager)
        final_path = manager.get_final_episode_path()

        assert manager.finalize_episode() is True

        assert os.path.isdir(final_path)
        assert "data_check_findings" not in _zarr_attrs(final_path)

    def test_missing_pending_episode_still_fails_cleanly(self, manager):
        assert manager.finalize_episode(findings=FINDINGS) is False


# --------------------------------------------------------------------------
# quarantine_staging
# --------------------------------------------------------------------------


class TestQuarantineStaging:
    def _quarantined_episode(self, manager, episode_id=0):
        root = output_root() / "quarantine" / TASK / SESSION
        return str(root / f"episode_{episode_id}")

    def test_issues_and_findings_written_side_by_side(self, manager):
        _stage_pending_episode(manager)

        assert manager.quarantine_staging(issues=ISSUES, findings=FINDINGS) is True

        attrs = _zarr_attrs(self._quarantined_episode(manager))
        assert attrs["data_check_issues"] == ISSUES
        assert attrs["data_check_findings"] == FINDINGS
        assert attrs["data_check_passed"] is False

    def test_none_writes_no_findings_attr(self, manager):
        _stage_pending_episode(manager)

        manager.quarantine_staging(issues=ISSUES, findings=None)

        attrs = _zarr_attrs(self._quarantined_episode(manager))
        assert attrs["data_check_issues"] == ISSUES
        assert "data_check_findings" not in attrs

    def test_old_call_without_findings_still_works(self, manager):
        _stage_pending_episode(manager)

        assert manager.quarantine_staging(issues=ISSUES) is True

        attrs = _zarr_attrs(self._quarantined_episode(manager))
        assert attrs["data_check_issues"] == ISSUES
        assert "data_check_findings" not in attrs

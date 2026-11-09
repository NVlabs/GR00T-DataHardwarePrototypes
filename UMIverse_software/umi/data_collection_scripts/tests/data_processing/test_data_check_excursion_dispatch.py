# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Excursion check wiring in `check_episode`, and the findings it persists.

Policy: an excursion never fails an episode. A video-confirmed one is reported
as a warning; candidate, unverified and rejected ones are only recorded. The
episode's verdict and issue list must be exactly what they were without the
check.

Run with: uv run pytest tests/data_processing/test_data_check_excursion_dispatch.py -v
"""

import copy
import json

import numpy as np
import pytest
import zarr

from real_scripts.data_processing.data_check.dispatch import check_episode
from real_scripts.data_processing.data_check.report import data_check_findings

N = 150
FPS = 30.0
GLITCH_START, GLITCH_LEN = 60, 8
LEFT, RIGHT = "left_tracking_camera", "right_tracking_camera"

_CONFIRM_MODULE = "real_scripts.data_processing.data_check.excursion_confirm"
_DISPATCH_MODULE = "real_scripts.data_processing.data_check.dispatch"


def _poses(ts, glitch_m=0.0, glitch_rotation_rad=0.0):
    """Hand moving steadily at 5 cm/s (enough for the consecutive-movement
    check) with a smooth yaw wobble; optional out-and-back glitch on y."""
    poses = np.tile(np.eye(4), (len(ts), 1, 1))
    yaw = 0.03 * np.sin(0.5 * ts)
    poses[:, 0, 0] = np.cos(yaw)
    poses[:, 0, 1] = -np.sin(yaw)
    poses[:, 1, 0] = np.sin(yaw)
    poses[:, 1, 1] = np.cos(yaw)
    poses[:, 0, 3] = 0.05 * ts
    poses[:, 2, 3] = 0.3
    sl = slice(GLITCH_START, GLITCH_START + GLITCH_LEN)
    poses[sl, 1, 3] += glitch_m
    if glitch_rotation_rad:
        c, s = np.cos(glitch_rotation_rad), np.sin(glitch_rotation_rad)
        roll = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        poses[sl, :3, :3] = poses[sl, :3, :3] @ roll
    return poses


def _episode(left_glitch_m=0.100, left_glitch_rotation_rad=0.0):
    """Bimanual iPhone episode; only the left tracker glitches."""
    root = zarr.open_group(store=zarr.storage.MemoryStore(), mode="w")
    ep = root.create_group("episode_0")
    ts = np.arange(N) / FPS
    for name, glitch_m, rot in (
        (LEFT, left_glitch_m, left_glitch_rotation_rad),
        (RIGHT, 0.0, 0.0),
    ):
        group = ep.create_group(name)
        poses = _poses(ts, glitch_m, rot)
        group.create_array("pose", shape=poses.shape, dtype="f8")[:] = poses
        group.create_array("capture_time", shape=ts.shape, dtype="f8")[:] = ts
        group.create_array("receive_time", shape=ts.shape, dtype="f8")[:] = ts
    return ep


def _tracking(**overrides):
    return {"tracking": {"enabled": True, **overrides}}


def _run(episode, check_config, data_dir=None):
    return check_episode(
        episode,
        config={},
        check_config=check_config,
        episode_name="episode_0",
        data_dir=data_dir,
    )


def _patch_confirm(monkeypatch, state):
    """Replace video confirmation: every candidate becomes `state`."""
    calls = []

    def fake_confirm(*args, **kwargs):
        events = kwargs.get("events", args[4] if len(args) > 4 else None)
        device = kwargs.get("tracking_device", args[3] if len(args) > 3 else None)
        calls.append(device)
        out = []
        for event in events or []:
            event = dict(event)
            if event.get("state") == "candidate":
                event["state"] = state
                event["confirm"] = {
                    "method": "video_gain_v1",
                    "score": 0.05 if state == "confirmed" else None,
                    "cameras": {},
                }
            out.append(event)
        return out

    monkeypatch.setattr(f"{_CONFIRM_MODULE}.confirm_excursions", fake_confirm)
    # Also cover a module-level `from .excursion_confirm import confirm_excursions`.
    monkeypatch.setattr(
        f"{_DISPATCH_MODULE}.confirm_excursions", fake_confirm, raising=False
    )
    return calls


# --------------------------------------------------------------------------
# result placement
# --------------------------------------------------------------------------


class TestResultPlacement:
    def test_excursions_recorded_per_tracking_device(self):
        result = _run(_episode(), _tracking())

        for device in (LEFT, RIGHT):
            excursions = result["tracking_checks"][device]["excursions"]
            assert excursions["detector_version"] == 1
            assert isinstance(excursions["events"], list)

    def test_glitch_is_on_the_left_only(self):
        result = _run(_episode(), _tracking())
        checks = result["tracking_checks"]

        left = [e for e in checks[LEFT]["excursions"]["events"]]
        assert len(left) == 1
        assert left[0]["shape"] == "out_and_back"
        assert checks[RIGHT]["excursions"]["events"] == []

    def test_without_data_dir_events_stay_candidates(self):
        """No videos to look at, so nothing can be confirmed or rejected."""
        result = _run(_episode(), _tracking(), data_dir=None)

        events = result["tracking_checks"][LEFT]["excursions"]["events"]
        assert [e["state"] for e in events] == ["candidate"]

    def test_existing_pose_jump_rule_is_unchanged(self):
        """A 100 mm one-frame step is ~3 m/s: the 4.5 m/s rule still ignores it."""
        result = _run(_episode(), _tracking())

        assert result["tracking_checks"][LEFT]["pose_jumps"]["has_jumps"] is False

    def test_result_is_json_serialisable(self):
        result = _run(_episode(), _tracking())

        json.dumps(result["tracking_checks"][LEFT]["excursions"])


# --------------------------------------------------------------------------
# gating
# --------------------------------------------------------------------------


class TestGating:
    def test_switched_off_by_enabled_mapping(self):
        config = {"tracking": {"enabled": {"excursions": False}}}
        result = _run(_episode(), config)

        for device in (LEFT, RIGHT):
            checks = result["tracking_checks"][device]
            assert "pose_jumps" in checks, "precondition: tracking section ran"
            assert "excursions" not in checks

    def test_tracking_section_disabled_runs_nothing(self):
        result = _run(_episode(), {"tracking": {"enabled": False}})

        for checks in result["tracking_checks"].values():
            assert "excursions" not in checks

    def test_thresholds_are_read_from_prefixed_tracking_keys(self):
        """`excursion_min_amplitude_mm` reaches the detector."""
        result = _run(_episode(), _tracking(excursion_min_amplitude_mm=200.0))

        events = result["tracking_checks"][LEFT]["excursions"]["events"]
        assert [e for e in events if e["shape"] == "out_and_back"] == []

    def test_confirmation_skipped_when_switched_off(self, monkeypatch, tmp_path):
        calls = _patch_confirm(monkeypatch, "confirmed")
        config = {"tracking": {"enabled": {"excursion_confirm": False}}}

        result = _run(_episode(), config, data_dir=str(tmp_path))

        assert calls == []
        events = result["tracking_checks"][LEFT]["excursions"]["events"]
        assert [e["state"] for e in events] == ["candidate"]

    def test_confirmation_runs_with_a_data_dir(self, monkeypatch, tmp_path):
        calls = _patch_confirm(monkeypatch, "confirmed")

        result = _run(_episode(), _tracking(), data_dir=str(tmp_path))

        assert LEFT in calls
        events = result["tracking_checks"][LEFT]["excursions"]["events"]
        assert [e["state"] for e in events] == ["confirmed"]


# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------


class TestVerdict:
    def test_a_confirmed_excursion_never_changes_the_verdict(
        self, monkeypatch, tmp_path
    ):
        """Even a confirmed excursion adds no issue: the verdict is identical to
        the same episode with the check switched off."""
        _patch_confirm(monkeypatch, "confirmed")
        episode = _episode()

        with_check = _run(episode, _tracking(), data_dir=str(tmp_path))
        without = _run(
            episode,
            {"tracking": {"enabled": {"excursions": False}}},
            data_dir=str(tmp_path),
        )

        assert with_check["issues"] == without["issues"]
        assert with_check["passed"] == without["passed"]
        assert not any("excursion" in i for i in with_check["issues"])

    def test_a_confirmed_excursion_is_one_warning(self, monkeypatch, tmp_path):
        _patch_confirm(monkeypatch, "confirmed")
        result = _run(_episode(), _tracking(), data_dir=str(tmp_path))

        assert result["passed"] is True
        assert len(result["warnings"]) == 1
        warning = result["warnings"][0]
        assert warning.startswith(f"{LEFT}: confirmed tracker excursions (1 >= 20mm")

    def test_warning_quotes_the_configured_amplitude(self, monkeypatch, tmp_path):
        _patch_confirm(monkeypatch, "confirmed")
        result = _run(
            _episode(),
            _tracking(excursion_min_amplitude_mm=30.0),
            data_dir=str(tmp_path),
        )

        assert len(result["warnings"]) == 1
        assert ">= 30mm" in result["warnings"][0]

    def test_a_snapshot_still_asking_to_quarantine_only_warns(
        self, monkeypatch, tmp_path
    ):
        """Sessions recorded before excursions became warn-only carry
        `excursion_quarantine_confirmed: true` in their snapshot. Re-checking
        one must not fail on it."""
        _patch_confirm(monkeypatch, "confirmed")
        result = _run(
            _episode(),
            _tracking(excursion_quarantine_confirmed=True),
            data_dir=str(tmp_path),
        )

        assert result["passed"] is True
        assert result["issues"] == []
        assert len(result["warnings"]) == 1

    def test_default_config_is_clean_on_a_glitch(self):
        result = _run(_episode(), _tracking())

        assert result["passed"] is True
        assert result["issues"] == []
        assert result["warnings"] == []

    @pytest.mark.parametrize("state", ["unverified", "rejected"])
    def test_an_unconfirmed_excursion_adds_nothing(self, monkeypatch, tmp_path, state):
        _patch_confirm(monkeypatch, state)
        result = _run(_episode(), _tracking(), data_dir=str(tmp_path))

        assert result["passed"] is True
        assert result["issues"] == []
        assert result["warnings"] == []

    def test_bare_candidates_add_nothing(self):
        """No data_dir: candidates are never confirmed, so never a warning."""
        result = _run(_episode(), _tracking(), data_dir=None)

        events = result["tracking_checks"][LEFT]["excursions"]["events"]
        assert events, "precondition: the glitch was detected"
        assert result["passed"] is True
        assert result["warnings"] == []

    def test_rotation_rejected_events_add_nothing(self, monkeypatch, tmp_path):
        """A rotation-rich edge is rejected before video; the fake confirmer
        would confirm any candidate it is handed, so none may reach it."""
        _patch_confirm(monkeypatch, "confirmed")
        result = _run(
            _episode(left_glitch_m=0.060, left_glitch_rotation_rad=np.radians(15.0)),
            _tracking(),
            data_dir=str(tmp_path),
        )

        events = result["tracking_checks"][LEFT]["excursions"]["events"]
        assert all(e["state"] == "rejected" for e in events)
        assert result["passed"] is True
        assert result["warnings"] == []


# --------------------------------------------------------------------------
# data_check_findings
# --------------------------------------------------------------------------


def _excursions(events=()):
    return {
        "detector_version": 1,
        "num_samples": 150,
        "max_dv_m_per_s": 0.1,
        "num_nominated": 0,
        "num_events": len(events),
        "num_candidates": 0,
        "events": list(events),
    }


class TestDataCheckFindings:
    def test_shape(self):
        left = _excursions([{"shape": "out_and_back", "state": "candidate"}])
        episode_result = {
            "tracking_checks": {
                LEFT: {"pose_jumps": {"has_jumps": False}, "excursions": left},
                RIGHT: {"pose_jumps": {"has_jumps": False}},
            },
            "passed": True,
            "issues": [],
        }

        findings = data_check_findings(episode_result)

        assert findings == {"v": 2, "tracking_excursions": {LEFT: left}}

    def test_empty_event_list_is_still_included(self):
        episode_result = {"tracking_checks": {RIGHT: {"excursions": _excursions()}}}

        findings = data_check_findings(episode_result)

        assert findings == {"v": 2, "tracking_excursions": {RIGHT: _excursions()}}

    @pytest.mark.parametrize(
        "episode_result",
        [
            {},
            {"tracking_checks": {}},
            {"tracking_checks": {LEFT: {"pose_jumps": {"has_jumps": False}}}},
        ],
    )
    def test_none_without_any_excursions_result(self, episode_result):
        assert data_check_findings(episode_result) is None

    def test_does_not_mutate_the_episode_result(self):
        episode_result = {"tracking_checks": {LEFT: {"excursions": _excursions()}}}
        before = copy.deepcopy(episode_result)

        data_check_findings(episode_result)

        assert episode_result == before

    def test_from_a_real_check_episode_result(self):
        result = _run(_episode(), _tracking())

        findings = data_check_findings(result)

        assert findings["v"] == 2
        assert set(findings["tracking_excursions"]) == {LEFT, RIGHT}
        assert len(findings["tracking_excursions"][LEFT]["events"]) == 1
        assert findings["tracking_excursions"][RIGHT]["events"] == []
        json.dumps(findings)

    def test_warnings_travel_with_the_episode(self):
        warning = f"{LEFT}: confirmed tracker excursions (1 >= 20mm)"
        findings = data_check_findings({"warnings": [warning], "tracking_checks": {}})

        assert findings == {"v": 2, "warnings": [warning]}

    def test_none_when_the_check_was_switched_off(self):
        result = _run(_episode(), {"tracking": {"enabled": {"excursions": False}}})

        assert data_check_findings(result) is None

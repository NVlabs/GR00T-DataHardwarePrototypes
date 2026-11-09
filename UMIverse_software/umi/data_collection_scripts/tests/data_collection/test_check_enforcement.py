# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warn-only recording checks still measure failures without blocking Start."""

from pathlib import Path

import pytest

from real_scripts.data_collection import record_data
from umi.common.recording.pre_episode_checks import (
    REGISTRY,
    PreEpisodeContext,
    build_checks,
    run_pre_episode_checks,
)


@pytest.fixture
def failing_check(monkeypatch):
    class Check:
        name = "test.connection"

        def __init__(self):
            self.calls = 0

        def run(self, ctx):
            self.calls += 1
            return False, "USB disconnected"

    monkeypatch.setitem(REGISTRY, Check.name, Check)
    return Check.name


def test_mixed_prechecks_still_run_and_only_enforced_failure_blocks(failing_check):
    checks = build_checks(
        [
            {"name": failing_check, "id": "left", "enforce": False},
            {"name": failing_check, "id": "right"},
        ]
    )
    warnings = []
    ok, failures = run_pre_episode_checks(
        checks, PreEpisodeContext({}, {}, None), warnings
    )
    assert not ok
    assert failures == [("right", "USB disconnected")]
    assert warnings == [("left", "USB disconnected")]
    assert [check.calls for check in checks] == [1, 1]


def test_gui_policy_can_make_precheck_warn_only(failing_check):
    checks = build_checks([{"name": failing_check}], relaxed=[failing_check])
    warnings = []
    assert run_pre_episode_checks(
        checks, PreEpisodeContext({}, {}, None), warnings
    ) == (
        True,
        [],
    )
    assert warnings == [(failing_check, "USB disconnected")]
    assert checks[0].calls == 1


@pytest.mark.parametrize("field", ["enabled", "enforce"])
def test_string_boolean_is_not_treated_as_a_policy(field, failing_check):
    with pytest.raises(ValueError, match=f"{field} must be true or false"):
        build_checks([{"name": failing_check, field: "false"}])


def _calibration_check(enforced):
    config = {
        "pre_episode_checks": [
            {
                "name": "camera.relative_calibration",
                "params": {
                    "reference_camera": "ref",
                    "cameras": ["ref", "other"],
                    "run_every_n_episodes": 10,
                    "marker": {"mode": "aruco", "marker_size": 0.1875},
                },
            }
        ],
        "check_enforcement": {"pre": {"camera.relative_calibration": enforced}},
    }
    (check,) = build_checks(
        config["pre_episode_checks"],
        search_dirs=[Path(record_data.__file__).parent / "pre_episode_checks"],
        relaxed=record_data._relaxed_pre_checks(config),
    )
    return check


def test_warn_only_calibration_never_holds_the_recording_stage():
    optional = _calibration_check(False)
    assert optional.is_due(0)
    assert not record_data._calibration_due(optional, 0)
    assert record_data._calibration_due(_calibration_check(True), 0)

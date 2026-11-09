# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression: data_check output on plane_assembly_v11 matches recorded baselines.

The baselines were captured with an earlier implementation of the data check.
The raw set has 3 clean episodes; the quarantine set has 7 failing episodes
exercising the gripper-jump and pose-jump failure paths. Strict bit-for-bit
comparison (modulo legitimate normalizations).

`baseline_quarantine.json` was re-captured when the jump checks started scoring
speed. The diff was verified to be only these three -- every other check still
matched:

  episode_2  gripper jump gone, so the episode now passes (~1171mm/s)
  episode_1  pose jumps 5 -> 4
  episode_6  pose jumps 2 -> 1

`baseline_raw.json` is unchanged. The jump verdicts are pinned by
`test_rate_normalized_jump_issues_on_quarantine`.

Sessions are fetched from a private S3 bucket on first run (cached under
/tmp/umi_test_fixtures/). Without access the tests skip with a warning.
"""

import json
from pathlib import Path

import pytest

from real_scripts.data_processing.data_check import check_dataset

from ._fixture_data import (
    EXCURSION_REPORT_ONLY_KEYS,
    FRAME_INTERVAL_REPORT_ONLY_KEYS,
    JUMP_ISSUE_MARKERS,
    JUMP_REPORT_ONLY_KEYS,
    PLANE_V11_QUARANTINE,
    PLANE_V11_RAW,
    session_path_or_skip,
)
from ._report_json import NumpyEncoder

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "plane_assembly_v11"
CHECK_YAML = FIXTURES / "check_config.yaml"


def _normalize(report):
    """Match the normalization used by test_data_check_refactor_regression.py."""
    report = json.loads(json.dumps(report, cls=NumpyEncoder))
    report.pop("data_dir", None)
    # check_config echo legitimately drifts (e.g. the fixture YAML adds `enabled`
    # keys that the earlier implementation ignored but the current one requires).
    report.pop("check_config", None)
    # Normalize report-only gap metrics and thresholds while retaining exact
    # comparisons of passed, num_violations and violation_frames.
    _strip_keys(report, FRAME_INTERVAL_REPORT_ONLY_KEYS)
    # Same for the jump thresholds, which are expressed as speeds.
    _strip_keys(report, JUMP_REPORT_ONLY_KEYS)
    # Report-only and newer than the baseline; see _fixture_data.
    _strip_keys(report, EXCURSION_REPORT_ONLY_KEYS)
    _strip_keys(report, {"_viz_data", "_error_traceback"})
    _round_floats(report, places=9)
    return report


def _strip_keys(obj, keys):
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            if k in keys:
                del obj[k]
            else:
                _strip_keys(obj[k], keys)
    elif isinstance(obj, list):
        for v in obj:
            _strip_keys(v, keys)


def _round_floats(obj, places):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, float):
                obj[k] = round(v, places)
            else:
                _round_floats(v, places)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, float):
                obj[i] = round(v, places)
            else:
                _round_floats(v, places)


@pytest.fixture(scope="module")
def v11_raw_session():
    return session_path_or_skip(PLANE_V11_RAW)


@pytest.fixture(scope="module")
def v11_quarantine_session():
    return session_path_or_skip(PLANE_V11_QUARANTINE)


def test_refactored_matches_pre_refactor_on_raw(v11_raw_session):
    """Clean episodes — all pass. Strict bit-for-bit after normalization."""
    with open(FIXTURES / "baseline_raw.json") as f:
        baseline = json.load(f)
    actual = check_dataset(
        str(v11_raw_session),
        check_config_path=str(CHECK_YAML),
    )
    assert _normalize(actual) == _normalize(baseline)


def test_refactored_matches_pre_refactor_on_quarantine(v11_quarantine_session):
    """Failing episodes — exercise gripper-jump and pose-jump issue paths.
    Bit-for-bit match validates that failure detection and issue
    categorization are unchanged on real failure data."""
    with open(FIXTURES / "baseline_quarantine.json") as f:
        baseline = json.load(f)
    actual = check_dataset(
        str(v11_quarantine_session),
        check_config_path=str(CHECK_YAML),
    )
    assert _normalize(actual) == _normalize(baseline)


# Expected jump issues per episode with the rate-normalised jump checks. These
# pin what the baseline no longer covers for these two checks.
EXPECTED_QUARANTINE_JUMP_ISSUES = {
    "episode_0": {"head_tracking_camera: pose jumps detected (1 jumps > 4.5m/s)"},
    "episode_1": {"head_tracking_camera: pose jumps detected (4 jumps > 4.5m/s)"},
    "episode_2": set(),
    "episode_3": {"head_tracking_camera: pose jumps detected (1 jumps > 4.5m/s)"},
    "episode_4": {"head_tracking_camera: pose jumps detected (6 jumps > 4.5m/s)"},
    "episode_5": {"head_tracking_camera: pose jumps detected (2 jumps > 4.5m/s)"},
    "episode_6": {"head_tracking_camera: pose jumps detected (1 jumps > 4.5m/s)"},
}


def test_rate_normalized_jump_issues_on_quarantine(v11_quarantine_session):
    """Pin the rate-normalised jump verdicts the baseline no longer covers.

    episode_2's gripper release spans a ~44ms sample gap: more than 50mm in one
    sample, but well under the 2000mm/s ceiling as a speed, so it is not a jump.
    """
    report = check_dataset(
        str(v11_quarantine_session),
        check_config_path=str(CHECK_YAML),
    )

    actual = {
        name: {
            s
            for s in episode.get("issues", [])
            if any(m in s for m in JUMP_ISSUE_MARKERS)
        }
        for name, episode in report["episodes"].items()
    }

    assert actual == EXPECTED_QUARANTINE_JUMP_ISSUES

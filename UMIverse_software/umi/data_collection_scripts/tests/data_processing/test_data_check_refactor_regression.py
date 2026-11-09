# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test pinning check_dataset output to a recorded baseline.

The baseline JSON in fixtures/ was captured with an earlier implementation of
the data check. This test runs the current check_dataset against the same UMI
session and asserts the output is structurally identical after normalization.

The session is fetched from a private S3 bucket on first run (cached under
/tmp/umi_test_fixtures/). Without access the test skips with a warning.

Run with: uv run pytest tests/data_processing/test_data_check_refactor_regression.py -v
"""

import json
from pathlib import Path

import pytest

from real_scripts.data_processing.data_check import check_dataset

from ._fixture_data import (
    EXCURSION_REPORT_ONLY_KEYS,
    FRAME_INTERVAL_REPORT_ONLY_KEYS,
    JUMP_REPORT_ONLY_KEYS,
    PLANE_V10_RAW,
    session_path_or_skip,
)
from ._report_json import NumpyEncoder

FIXTURES = Path(__file__).resolve().parent / "fixtures"
BASELINE_JSON = FIXTURES / "data_check_baseline.json"
BASELINE_YAML = FIXTURES / "data_check_baseline_config.yaml"


def _normalize(report):
    """Strip fields that legitimately differ between runs.

    Round-trips through JSON so numpy types in the in-memory report become
    native Python primitives matching the JSON-loaded baseline.
    """
    report = json.loads(json.dumps(report, cls=NumpyEncoder))
    # Absolute path; depends on checkout location.
    report.pop("data_dir", None)
    # check_config is a verbatim echo of the YAML; it is expected to drift as
    # `enabled: true` keys are added for opt-in gating.
    report.pop("check_config", None)
    # _viz_data is scratch state the existing check_episode leaves in the
    # result dict; not a meaningful check output.
    _strip_keys(report, {"_viz_data", "_error_traceback"})
    # Normalize report-only gap metrics and thresholds while retaining exact
    # comparisons of passed, num_violations and violation_frames.
    _strip_keys(report, FRAME_INTERVAL_REPORT_ONLY_KEYS)
    # Same for the jump thresholds, which are expressed as speeds. No verdict
    # moves on this fixture.
    _strip_keys(report, JUMP_REPORT_ONLY_KEYS)
    # Report-only and newer than the baseline; see _fixture_data.
    _strip_keys(report, EXCURSION_REPORT_ONLY_KEYS)
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
def umi_v10_session():
    return session_path_or_skip(PLANE_V10_RAW)


def test_check_dataset_matches_pre_refactor_baseline(umi_v10_session):
    with open(BASELINE_JSON) as f:
        baseline = json.load(f)

    actual = check_dataset(
        str(umi_v10_session),
        check_config_path=str(BASELINE_YAML),
    )

    assert _normalize(actual) == _normalize(baseline)

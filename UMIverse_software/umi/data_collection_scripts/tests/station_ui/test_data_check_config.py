# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The data-check config the station hands the recorder is generated.

`config/data_check/data_check_config.yaml` is version-controlled. Writing an
operator's choices into it would dirty every station's checkout, conflict on the
next pull, and destroy the shipped defaults with no way to tell "what UMI
ships" from "what this bench chose". The recording config is generated into a
gitignored tree, and the data-check config is generated the same way.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

from station_ui import data_check_config as dcc

REPO = pathlib.Path(__file__).resolve().parents[2]


def test_the_generated_path_is_keyed_by_task(tmp_path):
    path = dcc.output_path(tmp_path, "plane_assembly_v4")
    # Rooted at the state dir, not the checkout: see settings.STATE_DIR_ENV.
    assert path == tmp_path / "data_check" / "plane_assembly_v4.yaml"


def test_a_task_name_cannot_escape_the_generated_directory(tmp_path):
    """Task names reach this from a text field."""
    path = dcc.output_path(tmp_path, "../../etc/passwd")
    generated = tmp_path / "data_check"
    assert generated in path.parents, path


def test_generate_writes_a_loadable_file_carrying_the_baseline(tmp_path, monkeypatch):
    _seed_repo(
        tmp_path,
        {"gripper": {"enabled": True, "range_min_mm": -2.0, "range_max_mm": 90.0}},
    )
    monkeypatch.setattr(dcc, "_task_spec", lambda name: None)

    path = dcc.generate(tmp_path, task_name="some_task")

    assert path.is_file()
    data = yaml.safe_load(path.read_text())
    assert data["gripper"]["range_max_mm"] == 90.0


def test_generating_never_touches_the_baseline(tmp_path, monkeypatch):
    """The whole point."""
    baseline = _seed_repo(
        tmp_path,
        {"gripper": {"enabled": True, "range_min_mm": -2.0, "range_max_mm": 90.0}},
    )
    before = baseline.read_bytes()
    monkeypatch.setattr(dcc, "_task_spec", lambda name: None)

    dcc.generate(tmp_path, task_name="some_task")

    assert baseline.read_bytes() == before


def test_the_shipped_baselines_are_never_a_generate_target():
    """The version-controlled file is an input, and only an input.

    Checked against the real repo, not a tmp_path fixture: a generated path
    that could land on the tracked baseline is the exact regression this whole
    module exists to prevent, and a seeded fake repo cannot catch it.
    """
    baseline = dcc.baseline_path(REPO)
    assert baseline.is_file(), f"{baseline} is missing from the checkout"

    # No task reaches the baseline: the task is reduced to a single safe path
    # segment under the generated directory.
    for task in ("data_check_config", "../data_check_config", "t", "../..", ""):
        assert dcc.output_path(REPO, task) != baseline


def test_a_registered_task_s_overrides_are_merged_in(tmp_path, monkeypatch):
    """Generating with the registry's own values folded in makes RecordManager's
    second merge idempotent, so the artifact says what will actually be enforced."""
    _seed_repo(
        tmp_path,
        {"gripper": {"enabled": True, "range_min_mm": -2.0, "range_max_mm": 90.0}},
    )

    class Spec:
        data_check_overrides = {"gripper": {"range_max_mm": 85.0}}

    monkeypatch.setattr(
        dcc, "_task_spec", lambda name: Spec() if name == "known" else None
    )

    path = dcc.generate(tmp_path, task_name="known")
    data = yaml.safe_load(path.read_text())

    assert data["gripper"]["range_max_mm"] == 85.0
    # Untouched baseline keys survive the merge.
    assert data["gripper"]["range_min_mm"] == -2.0


def test_an_unknown_task_gets_the_baseline_unchanged(tmp_path, monkeypatch):
    _seed_repo(
        tmp_path,
        {"gripper": {"enabled": True, "range_min_mm": -2.0, "range_max_mm": 90.0}},
    )
    monkeypatch.setattr(dcc, "_task_spec", lambda name: None)

    path = dcc.generate(tmp_path, task_name="never_heard_of_it")
    data = yaml.safe_load(path.read_text())

    assert data["gripper"]["range_max_mm"] == 90.0


def test_a_missing_baseline_is_a_clear_error_not_a_traceback(tmp_path):
    with pytest.raises(dcc.BaselineMissing) as excinfo:
        dcc.generate(tmp_path, task_name="t")
    assert "data_check_config.yaml" in str(excinfo.value)


def test_regenerating_replaces_rather_than_appends(tmp_path, monkeypatch):
    _seed_repo(
        tmp_path,
        {"gripper": {"enabled": True, "range_min_mm": -2.0, "range_max_mm": 90.0}},
    )
    monkeypatch.setattr(dcc, "_task_spec", lambda name: None)

    first = dcc.generate(tmp_path, task_name="t")
    dcc.generate(tmp_path, task_name="t")

    # One document, written whole each time -- not appended to.
    assert first.read_text().count("gripper:") == 1
    assert len(list(yaml.safe_load_all(first.read_text()))) == 1


def _seed_repo(root, baseline_body):
    """A fake repo carrying only the data-check baseline."""
    directory = root / "config" / "data_check"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "data_check_config.yaml"
    path.write_text(yaml.safe_dump(baseline_body))
    return path

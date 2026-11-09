# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The two artifacts the recorder is handed, and where they come from.

Apply on `/config` writes the recording config. It has to write the data-check
config too, in the same action: generated apart, the two can name different
task bounds, and the recorder would then be checked against settings that do
not match the rig it recorded with.
"""

from __future__ import annotations

import pathlib

import pytest

# Reuse browser-form Apply helpers from this test directory.
import test_config_page as config_page
import yaml

from station_ui import data_check_config as dcc
from umi.common.tasks import task_registry_store

REPO = pathlib.Path(__file__).resolve().parents[2]

# The version-controlled baseline. Nothing here may write to it, and
# `_untouched` below is what proves it.
BASELINES = (REPO / "config" / "data_check" / "data_check_config.yaml",)


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    """The config page's own deterministic task table.

    Patched at the store, so `_spec_from_entry` still runs and
    `data_check_config._task_spec` -- which reaches the registry through
    `task_registry.get_task` -- sees the same tasks this page does. No test may
    read the station's real registry.
    """
    monkeypatch.setattr(
        task_registry_store, "get_all_tasks", lambda: dict(config_page.TASKS)
    )


def _applied(tmp_path, task_name="plane_assembly_v9"):
    """A station that has pressed Apply on `/config`, exactly as a browser does."""
    repo = config_page._repo(tmp_path)
    client = config_page._client(repo)
    config_page._apply(client, task_name=task_name)
    return client.app, client, client.app.state.rig.draft


@pytest.fixture
def applied_client(tmp_path):
    return _applied(tmp_path)


def test_apply_writes_a_data_check_config_beside_the_recording_config(applied_client):
    """Both artifacts, one action. Generated apart, they can disagree."""
    app, _client, draft = applied_client
    path = dcc.output_path(app.state.settings.state_dir, draft.task_name)
    assert path.is_file(), f"Apply did not generate {path}"


def test_record_points_at_the_generated_file_not_the_tracked_baseline(applied_client):
    from station_ui.routes import record as record_routes

    app, _client, draft = applied_client
    arg = record_routes._check_config_arg(app.state.settings.state_dir, draft)

    # "Under the state dir" is what separates the two: the generated tree lives
    # in `~/.umi_station/` and the baselines under `repo_root`, so that is the
    # assertion.
    path = pathlib.Path(arg)
    assert path.is_absolute(), arg
    assert path.is_relative_to(app.state.settings.state_dir), arg
    assert not path.is_relative_to(app.state.settings.repo_root), arg
    assert path.name != "data_check_config.yaml"
    assert path.is_file()


def test_the_generated_config_carries_this_task_s_own_overrides(tmp_path):
    """Per task, and merged -- not a copy of the baseline under a new name.

    `a_30_second_task` declares a gripper range_max_mm of 85; no baseline in the
    tree says that, so the number can only have come from the registry being
    folded in on the Apply path.
    """
    app, _client, draft = _applied(tmp_path, task_name="a_30_second_task")
    path = dcc.output_path(app.state.settings.state_dir, draft.task_name)

    gripper = yaml.safe_load(path.read_text())["gripper"]
    assert gripper["range_max_mm"] == 85.0


def test_apply_never_writes_into_the_real_checkout(tmp_path):
    """Apply must not write to a tracked path.

    The station's repo root is its checkout, so anything Apply writes to a
    tracked path dirties it and conflicts on the next pull. Asserted on the
    real files rather than on the fixture copy: `task_metadata` falls back to
    `project_root()` whenever a repo root is not threaded through, and that
    fallback points here.
    """
    before = {path: path.read_bytes() for path in BASELINES}
    _applied(tmp_path, task_name="a_30_second_task")
    for path in BASELINES:
        assert path.read_bytes() == before[path], f"Apply wrote to {path}"


def test_a_station_missing_its_baselines_is_told_on_apply(tmp_path):
    """Not hours later, when someone presses Record.

    An incomplete checkout is a station install problem, and the operator has
    to hear about it while they are still on the page that could have created
    the file.
    """
    repo = config_page._repo(tmp_path)
    (repo / "config" / "data_check" / "data_check_config.yaml").unlink()

    client = config_page._client(repo)
    response = config_page._apply(client, expect=400)

    assert "data_check_config.yaml" in response.text, response.text[:400]
    draft = client.app.state.rig.draft
    assert not dcc.output_path(
        client.app.state.settings.state_dir, draft.task_name
    ).exists()


# Exercise generated check configs at all consumers: CLI validation,
# session snapshotting and episode checking. Avoid recorder construction
# because it opens hardware.


def _generated_for(applied) -> pathlib.Path:
    """The absolute path the recorder is handed, taken from the argv builder.

    `_check_config_arg` returns the string that goes on the argv. It is
    absolute because the generated tree lives in the station state dir, and
    there is no working directory that would make a relative form correct.
    Read from the builder rather than rejoined here, so a test never stops
    exercising the real join.
    """
    app, _client, draft = applied
    from station_ui.routes import record as record_routes

    return pathlib.Path(
        record_routes._check_config_arg(app.state.settings.state_dir, draft)
    )


def _saved_by_recorder(generated: pathlib.Path, task_name: str, session) -> dict:
    """What `_save_data_check_config` freezes into a session folder.

    Built with `__new__` and the four attributes that method reads: a real
    `RecorderManager` opens cameras and gripper UARTs in its constructor, and
    this is a config test.

    `_enforce_session_file_lock` is deliberately not stubbed. It is a no-op
    against a folder with no `data_check_config.yaml` in it yet, which is what
    a fresh session is, so leaving it in place keeps the real write path whole.
    """
    from umi.common.recording.record_manager import RecorderManager

    manager = RecorderManager.__new__(RecorderManager)
    manager.data_check_config_path = str(generated)
    manager.record_path = str(session)
    manager.config = {"meta": {"task_name": task_name}}
    manager.verbose = False

    manager._save_data_check_config()

    written = session / "data_check_config.yaml"
    assert written.is_file(), f"the recorder wrote no config into {session}"
    return yaml.safe_load(written.read_text())


def test_click_accepts_the_generated_path_as_check_config(applied_client):
    """`record_data.py` declares `--check-config` as `click.Path(exists=True)`,
    so a path it cannot stat is rejected before `main` ever runs -- the
    operator presses Start and the job dies a second later.

    Asserted through the real click type rather than an `is_file()` check here,
    because click is the thing that decides, and it decides on more than
    existence (a directory would fail here too).
    """
    import click

    app, _client, draft = applied_client
    from station_ui.routes import record as record_routes

    absolute = record_routes._check_config_arg(app.state.settings.state_dir, draft)

    assert click.Path(exists=True).convert(absolute, None, None) == absolute


def test_the_recorder_merges_the_generated_file_into_the_session_folder(
    applied_client, tmp_path
):
    """`_save_data_check_config` is what freezes the checked-against config into
    the session, and it is the artifact every later reader trusts.

    The stronger claim is the second assertion: the recorder's own merge leaves
    the generated file unchanged. `data_check_config.generate` already folded
    `data_check_overrides` in, and re-merging the same overrides is a no-op, so
    the numbers `/config` showed the operator are the numbers the session
    records. If the two ever fold in different things, this is where it shows.
    """
    _app, _client, draft = applied_client
    generated = _generated_for(applied_client)
    session = tmp_path / "session"

    saved = _saved_by_recorder(generated, draft.task_name, session)

    assert saved["gripper"]["range_max_mm"] == 88.0, saved["gripper"]
    assert saved == yaml.safe_load(generated.read_text()), (
        "the recorder's merge changed the config the operator was shown"
    )


def test_the_data_checker_can_load_the_generated_file(applied_client):
    """`check_dataset` receives this path as `check_config_path` and loads it
    with `load_check_config` (dispatch.py). If that loader rejects the
    generated shape, every episode fails its check at record time -- long after
    the operator walked away.

    Imported from `dispatch`, not from `config_loader`, so this follows the name
    the caller actually binds. Not stubbed: a stub would assert that a stand-in
    accepts the file, which is worth nothing.
    """
    from real_scripts.data_processing.data_check.dispatch import load_check_config

    generated = _generated_for(applied_client)
    loaded = load_check_config(str(generated))

    assert isinstance(loaded, dict) and loaded
    # A section the checker reads. A loader that returned an empty mapping, or
    # the file's text, would satisfy the line above.
    assert loaded["gripper"]["range_max_mm"] == 88.0


def test_the_registry_still_wins_the_merge(applied_client, tmp_path):
    """Pins precedence rather than changing it.

    `_save_data_check_config` calls
    `_deep_merge_dict(file_contents, spec.data_check_overrides)` and the second
    argument wins, so the registry beats whatever file was passed on the command
    line. That is invisible in normal use only because the file is generated
    with those overrides already folded in -- the two agree, so nothing can be
    seen to lose.

    Here they are made to disagree: the generated file is rewritten with a value
    no registry entry declares, and the registry's number is still what lands
    in the session folder. If anyone makes `/config` edit data-check values
    directly, this is the tripwire -- the operator's number would be silently
    replaced at record time, and this test fails instead.
    """
    _app, _client, draft = applied_client
    generated = _generated_for(applied_client)

    config = yaml.safe_load(generated.read_text())
    config["gripper"]["range_max_mm"] = 1.0
    generated.write_text(yaml.dump(config, sort_keys=False))

    saved = _saved_by_recorder(generated, draft.task_name, tmp_path / "session")

    assert saved["gripper"]["range_max_mm"] == 88.0, saved["gripper"]

    # Registry overrides are an overlay; unspecified file thresholds must survive.
    assert saved["tracking"] == config["tracking"]
    assert saved["gripper"] == {**config["gripper"], "range_max_mm": 88.0}

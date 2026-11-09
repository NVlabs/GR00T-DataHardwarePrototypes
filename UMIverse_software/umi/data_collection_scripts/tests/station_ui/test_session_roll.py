# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A config that drifts mid-session rolls onto a fresh session, automatically.

The rule lives in `umi.common.recording.session_slot` and does not care what
drifted: a swapped camera, a re-zeroed gripper and an edited data-check
override all mean the same thing, that the episodes already in the folder were
recorded with a different rig. `station_ui.session_roll` is how the console
applies it, and the property every test here is really about is that the page
and the disk never name different sessions.

Renaming the folder at recorder launch and writing only `recording.data_dir`,
without telling the console, would break that: `routes/config._reopen` is
gated on an untouched draft, so the console could not pick the new uid up.
"""

from __future__ import annotations

import pathlib

import pytest

# Build Apply payloads from the rendered form through the shared test helper.
import test_config_page as config_page
import yaml

from station_ui import readiness, session_roll
from umi.common.recording import session_seal
from umi.common.recording.session_slot import save_launch_config
from umi.common.recording.session_uid import (
    is_valid_session_uid,
    session_uid_from_dir_name,
)
from umi.common.tasks import task_registry_store


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    """The config page's deterministic task table. No test may reach a remote
    registry.

    Patched at the store, exactly as `test_config_page.registry` does it, so the
    real registry code still runs.
    """
    monkeypatch.setattr(
        task_registry_store, "get_all_tasks", lambda: dict(config_page.TASKS)
    )


def _generated(client) -> pathlib.Path:
    path = readiness.generated_config(client.app)
    assert path is not None and path.is_file(), "Apply wrote no config"
    return path


def _data_dir(client) -> str:
    """The applied folder, expanded -- the config stores it with a leading `~`."""
    cfg = yaml.safe_load(_generated(client).read_text())
    raw = str((cfg.get("recording") or {}).get("data_dir") or "")
    return str(pathlib.Path(raw).expanduser())


def _drift_the_generated_config(client) -> None:
    """Change the config without going through Apply.

    Which is exactly what a gripper re-zeroed on step 4 does to a config applied
    on step 2 -- the drift the Start-side roll exists for.
    """
    generated = _generated(client)
    cfg = yaml.safe_load(generated.read_text())
    cfg["recording"]["record_fps"] = int(cfg["recording"].get("record_fps", 30)) + 15
    generated.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))


def _record_an_episode(client) -> str:
    """Make the applied folder look like a session that has been recorded into.

    Writes exactly what `record_data.py` writes on launch -- the raw config
    snapshot the drift check compares against -- plus one episode directory,
    which is what makes the folder unsafe to reuse under a different config.
    """
    folder = pathlib.Path(_data_dir(client)).expanduser()
    cfg = yaml.safe_load(_generated(client).read_text())
    folder.mkdir(parents=True, exist_ok=True)
    save_launch_config(str(folder), cfg)
    (folder / "episode_0").mkdir(exist_ok=True)
    return str(folder)


# --------------------------------------------------------------------- Apply


def test_a_drifted_apply_opens_a_new_session(tmp_path):
    """The commonest case: a camera is swapped on a session that already holds
    episodes, and Apply moves to a fresh session rather than mixing two rigs
    under one id."""
    client = config_page._client(config_page._repo(tmp_path))
    config_page._apply(client)
    before_uid = client.app.state.rig.draft.session_uid
    before_dir = _record_an_episode(client)

    config_page._apply(client, cameras={"right_umi_camera": "oak-FFC-Z"})

    draft = client.app.state.rig.draft
    assert draft.session_uid != before_uid, "the console stayed on the old session"
    assert is_valid_session_uid(draft.session_uid)
    assert _data_dir(client) != before_dir
    # The page and the file agree -- the whole point.
    assert session_uid_from_dir_name(_data_dir(client)) == draft.session_uid
    # Seal departed folders outside the rollover branch so both config drift
    # and task-name changes close the old session.
    marker = session_seal.read_seal(before_dir)
    assert marker is not None, "the drifted Apply left the old folder unsealed"
    assert marker["reason"] == "dir_change"


def test_the_rolled_config_carries_the_new_uid_in_its_meta(tmp_path):
    """`meta.session_uid` moves with the folder. Writing only the folder would
    leave the config claiming a session it is no longer in."""
    client = config_page._client(config_page._repo(tmp_path))
    config_page._apply(client)
    _record_an_episode(client)

    config_page._apply(client, cameras={"right_umi_camera": "oak-FFC-Z"})

    cfg = yaml.safe_load(_generated(client).read_text())
    assert cfg["meta"]["session_uid"] == client.app.state.rig.draft.session_uid
    assert (
        session_uid_from_dir_name(cfg["recording"]["data_dir"])
        == (cfg["meta"]["session_uid"])
    )


def test_the_operator_is_told_the_session_changed(tmp_path):
    """A folder the operator did not choose, under a "Configuration saved."
    notice, is the console moving their data without saying so."""
    client = config_page._client(config_page._repo(tmp_path))
    config_page._apply(client)
    _record_an_episode(client)

    config_page._apply(client, cameras={"right_umi_camera": "oak-FFC-Z"})

    body = client.get("/config").text
    assert "new session" in body
    assert client.app.state.rig.draft.session_uid in body


def test_an_unchanged_re_apply_stays_in_the_same_session(tmp_path):
    """The normal per-episode case. Rolling here would scatter one session
    across a folder per press of Apply."""
    client = config_page._client(config_page._repo(tmp_path))
    config_page._apply(client)
    before_uid = client.app.state.rig.draft.session_uid
    before_dir = _record_an_episode(client)

    config_page._apply(client)

    assert client.app.state.rig.draft.session_uid == before_uid
    assert _data_dir(client) == before_dir


def test_a_drifted_apply_on_an_empty_folder_stays_put(tmp_path):
    """A rig changed before the first episode leaves nothing behind to protect,
    so the session is reused rather than abandoned empty."""
    client = config_page._client(config_page._repo(tmp_path))
    config_page._apply(client)
    before_uid = client.app.state.rig.draft.session_uid
    folder = pathlib.Path(_data_dir(client)).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    save_launch_config(str(folder), yaml.safe_load(_generated(client).read_text()))

    config_page._apply(client, cameras={"right_umi_camera": "oak-FFC-Z"})

    assert client.app.state.rig.draft.session_uid == before_uid


# Start-time rollover runs only after launch gates pass. Update the
# already-generated config and draft together without rebuilding the rig.


def test_rolling_the_generated_config_moves_the_file_and_the_draft(tmp_path):
    client = config_page._client(config_page._repo(tmp_path))
    config_page._apply(client)
    before_dir = _record_an_episode(client)
    draft = client.app.state.rig.draft
    before_uid = draft.session_uid
    _drift_the_generated_config(client)
    generated = _generated(client)

    rolled = session_roll.roll_generated_config(generated, "", draft)

    assert rolled is not None and rolled != before_dir
    cfg = yaml.safe_load(generated.read_text())
    assert cfg["recording"]["data_dir"] == rolled
    assert cfg["meta"]["session_uid"] == draft.session_uid != before_uid
    assert session_uid_from_dir_name(rolled) == draft.session_uid


def test_rolling_is_a_no_op_when_nothing_drifted(tmp_path):
    client = config_page._client(config_page._repo(tmp_path))
    config_page._apply(client)
    _record_an_episode(client)
    draft = client.app.state.rig.draft
    before_uid = draft.session_uid

    assert session_roll.roll_generated_config(_generated(client), "", draft) is None
    assert draft.session_uid == before_uid


def test_an_unreadable_config_is_not_a_reason_to_fork_a_session(tmp_path):
    """`config_drifted` answers False whenever the question cannot be settled:
    rolling on a transient read error would split a live session in two."""
    client = config_page._client(config_page._repo(tmp_path))
    config_page._apply(client)
    _record_an_episode(client)
    draft = client.app.state.rig.draft
    before_uid = draft.session_uid
    generated = _generated(client)
    generated.write_text("{ this is not: valid: yaml [")

    assert session_roll.roll_generated_config(generated, "", draft) is None
    assert draft.session_uid == before_uid

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The applied session has to survive the server being restarted.

The draft lives in `app.state.rig`, so without a restore every restart would
throw away Setup and Configuration -- while the artefact those two steps
produced is still on disk.

Every test here restarts the console the only way that proves anything: it
builds a second app from the same `Settings`, which is what a fresh process
does. Nothing is carried over in memory, so anything the new console knows it
learned from the disk.

The generated config is the source of truth, deliberately. It is what Apply
writes, what `_shell_context` ticks step 2 on, and what the station records
with -- so a second state file that could disagree with it would be worse than
no restore at all. Its path carries the one field the file itself does not:
`data_recording/{preset_id}.yaml`.
"""

from __future__ import annotations

import pathlib
import shutil

import pytest

# Reuse the rendered-form Apply helper from this test directory.
import test_config_page as config_page
import yaml
from fastapi.testclient import TestClient

REPO = pathlib.Path(__file__).resolve().parents[2]

from station_ui import session_store  # noqa: E402
from station_ui.app import create_app  # noqa: E402
from station_ui.core.config_assembler import ConfigAssembler, SessionSpec  # noqa: E402
from station_ui.settings import Settings  # noqa: E402
from umi.common.tasks import task_registry_store  # noqa: E402


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    """The config page's deterministic task table, not this station's own."""
    monkeypatch.setattr(
        task_registry_store, "get_all_tasks", lambda: dict(config_page.TASKS)
    )


def _applied(tmp_path, *, preset="bimanual_v4", task_name="plane_assembly_v9"):
    """A station that has chosen a setup and pressed Apply. Returns the repo."""
    repo = config_page._repo(tmp_path)
    client = config_page._client(repo, preset=preset)
    config_page._apply(client, task_name=task_name)
    return repo, client.app.state.rig.draft


def _restarted(repo, **kwargs):
    """The same station, in a new process. This is the whole test method."""
    return create_app(Settings(repo_root=repo, **kwargs))


def _generated(repo, preset="bimanual_v4") -> pathlib.Path:
    return ConfigAssembler(repo).output_path(
        SessionSpec(
            preset_id=preset,
            operator_name="",
            task_name="",
            session_uid="0" * 12,
            data_dir="",
        )
    )


def test_a_restarted_console_still_knows_the_session_that_was_applied(tmp_path):
    """/record must not say "Not ready to record" after a restart while the
    config it would record from is on disk."""
    repo, applied = _applied(tmp_path)
    assert _generated(repo).is_file(), "nothing was applied; the test proves nothing"

    app = _restarted(repo)
    draft = app.state.rig.draft

    assert draft.preset_id == applied.preset_id
    assert draft.task_name == applied.task_name
    assert draft.operator_name == applied.operator_name

    body = TestClient(app).get("/record").text
    assert "Not ready to record" not in body
    assert "Choose a setup on the Setup page first." not in body


def test_the_restored_session_keeps_its_uid_and_its_recording_date(tmp_path):
    """The uid is the last segment of `raw/{task}/{session}/`, and the date is
    the first ten characters of the folder, which is how the audit finds a
    day's sessions. Minting either afresh on a restart forks a session that is
    already half recorded across two folders -- and the config on disk still
    points at the old one, so nothing on screen would say the two disagreed."""
    repo, applied = _applied(tmp_path)

    draft = _restarted(repo).state.rig.draft

    assert draft.session_uid == applied.session_uid
    assert draft.date == applied.date


def test_the_restored_session_records_into_the_same_folder(tmp_path):
    """The claim behind the two fields above, made against the artefact: the
    restored draft has to resolve to the folder the generated config is armed
    with, or the console and the recorder disagree about where episodes land."""
    from station_ui import session as session_state

    repo, _applied_draft = _applied(tmp_path)
    on_disk = yaml.safe_load(_generated(repo).read_text())["recording"]["data_dir"]

    draft = _restarted(repo).state.rig.draft

    assert session_state.data_dir(draft) == on_disk


def test_the_devices_and_grippers_come_back_too(tmp_path):
    """Six serials is the cost of not restoring them, and the operator would be
    re-typing them from hardware they cannot see from the keyboard."""
    repo, applied = _applied(tmp_path)

    draft = _restarted(repo).state.rig.draft

    assert draft.grippers == applied.grippers != {}
    assert draft.required_cameras == applied.required_cameras != {}


# =============================================================================
# 2. What is deliberately NOT restored
# =============================================================================


def test_a_restarted_console_owns_nothing_until_a_browser_asks(tmp_path):
    """Rig ownership is a property of a live browser, not of the session. A
    restored owner token would belong to a tab that is no longer connected, and
    the operator sitting at the station would be told to Take control from
    nobody."""
    repo, _ = _applied(tmp_path)

    assert _restarted(repo).state.rig.owner_token is None


# =============================================================================
# 3. Degrading, never crashing
# =============================================================================


def test_a_corrupt_generated_config_leaves_an_empty_draft(tmp_path):
    """A console that cannot read the file starts with an empty session; it
    never fails to boot."""
    repo, _ = _applied(tmp_path)
    _generated(repo).write_text("meta: [this is not\n  a mapping: {")

    app = _restarted(repo)

    assert app.state.rig.draft.preset_id is None
    assert TestClient(app).get("/").status_code == 200
    assert TestClient(app).get("/record").status_code == 200


def test_an_unreadable_generated_config_leaves_an_empty_draft(tmp_path):
    """Not the same failure as corrupt YAML: this one raises OSError out of the
    read, before any parser is involved."""
    repo, _ = _applied(tmp_path)
    path = _generated(repo)
    path.unlink()
    path.mkdir()

    app = _restarted(repo)

    assert app.state.rig.draft.preset_id is None
    assert TestClient(app).get("/").status_code == 200


def test_a_config_whose_schema_moved_on_leaves_an_empty_draft(tmp_path):
    """A file this build can parse but cannot understand -- every key it reads
    holding the wrong type. Valid YAML, so the loader is happy; the console
    still has to boot."""
    repo, _ = _applied(tmp_path)
    _generated(repo).write_text(
        yaml.safe_dump(
            {
                "meta": "a string where a mapping used to be",
                "recording": 7,
                "video_sources": "not a list of sources",
                "numeric_sources": {"nor": "this"},
            }
        )
    )

    app = _restarted(repo)

    assert app.state.rig.draft.preset_id is None
    assert TestClient(app).get("/").status_code == 200


@pytest.mark.parametrize(
    # A setup this build has retired, and one it has never heard of.
    "stale_name",
    ["bimanual_v3.yaml", "bimanual_v9_from_the_future.yaml"],
)
def test_a_config_for_a_preset_this_build_no_longer_has_is_ignored(
    tmp_path, stale_name
):
    """Restoring it would put an id nothing declares into the draft, and every
    page that resolves a preset would then render its "that setup is gone"
    path -- on a console the operator never told to choose it."""
    repo, _ = _applied(tmp_path)
    good = _generated(repo)
    stale = good.with_name(stale_name)
    shutil.copy(good, stale)
    # Newer than the real one, so it would win if it were a candidate at all.
    stale.touch()

    draft = _restarted(repo).state.rig.draft

    assert draft.preset_id == "bimanual_v4"


def test_a_repo_that_has_never_applied_anything_starts_empty(tmp_path):
    """No generated directory at all -- a fresh checkout."""
    repo = config_page._repo(tmp_path)

    app = _restarted(repo)

    assert app.state.rig.draft.preset_id is None
    assert TestClient(app).get("/record").status_code == 200


# =============================================================================
# 4. Which config, when there is more than one
# =============================================================================


def test_the_most_recently_applied_config_is_the_one_restored(tmp_path):
    """Apply leaves one file per preset, and `_disarm` only ever
    removes the one for the session being left -- so a station that has used
    two setups has two files. The one the operator last applied is the one the
    station is set up to record with."""
    repo = config_page._repo(tmp_path)
    first = config_page._client(repo, preset="bimanual_v4_vive")
    config_page._apply(first, task_name="plane_assembly_v9")
    second = config_page._client(repo, preset="bimanual_v4")
    config_page._apply(second, task_name="a_30_second_task")
    assert _generated(repo, preset="bimanual_v4_vive").is_file(), (
        "only one file; no choice"
    )

    draft = _restarted(repo).state.rig.draft

    assert draft.preset_id == "bimanual_v4"
    assert draft.task_name == "a_30_second_task"
    assert draft.session_uid == second.app.state.rig.draft.session_uid


def test_a_session_the_operator_left_does_not_come_back(tmp_path):
    """Start a new session disarms the config it left behind (`_disarm`), which
    is what unticks step 2. A restore that resurrected it would re-arm a
    session the operator explicitly closed."""
    repo, applied = _applied(tmp_path)
    client = config_page._client(repo, preset=None)
    client.app.state.rig.draft.preset_id = applied.preset_id
    client.app.state.rig.draft.session_uid = applied.session_uid
    client.post("/config/session/uid", data={"confirm": "yes"}, follow_redirects=False)
    assert not _generated(repo).is_file(), "the config was not disarmed"

    assert _restarted(repo).state.rig.draft.preset_id is None


# =============================================================================
# 5. Two servers, one repo
# =============================================================================


def test_restoring_writes_nothing_at_all(tmp_path):
    """The reason two consoles on one repo are safe: the restore is a read.

    Two servers on different ports share this checkout, and a restore that
    persisted its own copy of the draft would have them racing to write it --
    which is the failure a second state file would introduce. Nothing is
    written, so there is nothing to race over.
    """
    repo, _ = _applied(tmp_path)
    before = {
        path: path.stat().st_mtime_ns
        for path in sorted(repo.rglob("*"))
        if path.is_file()
    }

    _restarted(repo)
    _restarted(repo)

    after = {
        path: path.stat().st_mtime_ns
        for path in sorted(repo.rglob("*"))
        if path.is_file()
    }
    assert after == before, "the restore touched the repo"


def test_two_consoles_on_one_repo_restore_the_same_session(tmp_path):
    """They read one artefact, so they agree -- including on the uid, which is
    what keeps two browsers on two ports from filing episodes under two
    folders."""
    repo, applied = _applied(tmp_path)

    one = _restarted(repo, port=8770).state.rig.draft
    two = _restarted(repo, port=8899).state.rig.draft

    assert one.session_uid == two.session_uid == applied.session_uid
    assert one.preset_id == two.preset_id
    assert one.task_name == two.task_name


# =============================================================================
# 6. The seam itself
# =============================================================================


def test_restore_returns_none_when_there_is_nothing_to_restore(tmp_path):
    """None, not an empty draft: the caller decides what an absent session
    looks like, and today that is `SessionDraft()`."""
    assert session_store.restore(tmp_path / "nowhere") is None
    assert session_store.restore(config_page._repo(tmp_path)) is None

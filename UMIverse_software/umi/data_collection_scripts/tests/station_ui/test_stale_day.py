# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A session dated before today must not pass for the current one.

`draft.date` is the first ten characters of the session folder, and that is
how the audit finds a day's sessions -- so recording into a stale one files a
whole shift under yesterday with nothing on screen to say so
(`session.start_new_session` says exactly this, and exists to prevent it).

Two ways the console gets there, and neither involves the operator doing
anything wrong:

* the station is left running overnight. `draft.date` is set at draft creation
  and by `start_new_session`, and by nothing else -- so it is yesterday's until
  somebody presses the button;
* the console is reopened the next morning. `session_store.restore` deliberately
  brings the uid and the date back unchanged, because re-minting them mid-session
  would fork a half-recorded session across two folders.

`station_ui.day_roll` fixes both before any handler runs -- see
`test_day_roll.py`. What remains here is the predicate, and the one case the
roll refuses: a live recorder, where stopping is a decision only the operator
can make and so the warning still has to do the whole job.
"""

from __future__ import annotations

import datetime as dt
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

import test_config_page as config_page  # noqa: E402

from station_ui import day_roll, recording  # noqa: E402
from station_ui import session as session_state  # noqa: E402
from umi.common.tasks import task_registry_store  # noqa: E402


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(
        task_registry_store, "get_all_tasks", lambda: dict(config_page.TASKS)
    )


def _yesterday() -> str:
    return (dt.date.today() - dt.timedelta(days=1)).isoformat()


def test_a_draft_dated_today_is_not_stale():
    draft = session_state.SessionDraft()
    assert session_state.recording_day_is_stale(draft) is False


def test_a_draft_dated_before_today_is_stale():
    draft = session_state.SessionDraft(date=_yesterday())
    assert session_state.recording_day_is_stale(draft) is True


def test_a_session_in_a_shift_named_folder_is_stale_even_today():
    draft = session_state.SessionDraft(
        legacy_folder="/data/2026-08-27_AM_umiV4_0002R_task_91e98c89fccb"
    )
    assert session_state.recording_day_is_stale(draft) is True


def test_start_new_session_stamps_today_and_leaves_a_shift_named_folder():
    draft = session_state.SessionDraft(
        date=_yesterday(),
        legacy_folder="/data/2026-08-27_AM_umiV4_0002R_task_91e98c89fccb",
    )
    session_state.start_new_session(draft)
    assert (draft.date, draft.legacy_folder) == (dt.date.today().isoformat(), "")
    assert session_state.recording_day_is_stale(draft) is False


def test_a_fresh_draft_is_dated_today():
    assert session_state.SessionDraft().date == dt.date.today().isoformat()


def test_the_configuration_page_says_the_session_is_from_another_day(
    tmp_path, monkeypatch
):
    """The station left running overnight. Without a warning, an operator
    pressing Apply would file a fresh shift under yesterday.

    `day_roll` corrects this before the handler runs, so the recorder guard is
    what keeps a stale draft reachable at all -- and it is the case where the
    warning still has to do the whole job, because stopping a recording is a
    decision only the operator can make. See `test_day_roll.py` for the roll
    itself."""
    client = config_page._client(config_page._repo(tmp_path))
    config_page._apply(client, head=False)
    client.app.state.rig.draft.date = _yesterday()
    monkeypatch.setattr(recording, "in_progress", lambda app: True)

    body = client.get("/config").text

    assert _yesterday() in body, "the stale date is not even shown"
    assert "from another day" in body, "nothing warns that the session is stale"


def test_the_warning_is_absent_on_a_session_dated_today(tmp_path):
    client = config_page._client(config_page._repo(tmp_path))
    config_page._apply(client, head=False)

    assert "from another day" not in client.get("/config").text


def test_starting_a_new_session_clears_the_warning(tmp_path, monkeypatch):
    """The manual button, on its own. `day_roll` is stubbed out for this one --
    it would otherwise roll the draft on the POST's way in and the assertion
    below would pass without the button being pressed at all."""
    client = config_page._client(config_page._repo(tmp_path))
    config_page._apply(client, head=False)
    client.app.state.rig.draft.date = _yesterday()
    monkeypatch.setattr(day_roll, "roll_if_new_day", lambda app: None)

    client.post("/config/session/uid", data={"config_form": "1", "confirm": "yes"})

    draft = client.app.state.rig.draft
    assert draft.date == dt.date.today().isoformat()
    assert "from another day" not in client.get("/config").text


def test_the_record_page_warns_too(tmp_path, monkeypatch):
    """Where the harm actually lands. An operator reaches Record without
    necessarily passing Configuration again, and a warning only on the page they
    have already left is not protective."""
    client = config_page._client(config_page._repo(tmp_path))
    config_page._apply(client, head=False)
    client.app.state.rig.draft.date = _yesterday()
    monkeypatch.setattr(recording, "in_progress", lambda app: True)

    body = client.get("/record").text

    assert "from another day" in body
    assert _yesterday() in body

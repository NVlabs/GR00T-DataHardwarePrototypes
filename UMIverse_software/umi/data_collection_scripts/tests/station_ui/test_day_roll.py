# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The console starts the new day's session by itself.

`draft.date` is the first ten characters of the session folder, and the audit
finds a day's sessions by it. A station left running overnight would therefore
file an entire morning under yesterday unless somebody pressed a button, and
asking the operator to press it every morning is not a solution.

What is asserted here is the shape of "doing it for them" -- because a console
that changes the session id behind somebody's back can get that wrong in four
expensive ways, and each has a test:

* rolling under a live recorder, which would leave the page naming one folder
  and the recorder writing into another;
* deleting the applied config, the way the manual button does. That file is this
  station's only persistence (`session_store` reads the draft back out of it),
  so a console restarted after an overnight roll would come up blank;
* re-assembling it instead of moving it, which re-resolves every camera against
  whatever device scan happens to be in hand and can hand back a different field
  of view for a rig nobody touched;
* doing any of it silently.
"""

from __future__ import annotations

import datetime as dt
import os
import pathlib

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]

import test_config_page as config_page  # noqa: E402

from station_ui import day_roll, recording, session_store  # noqa: E402
from station_ui import session as session_state  # noqa: E402
from station_ui.core import config_assembler  # noqa: E402
from umi.common.recording import session_seal  # noqa: E402
from umi.common.tasks import task_registry_store  # noqa: E402


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(
        task_registry_store, "get_all_tasks", lambda: dict(config_page.TASKS)
    )


def _today() -> str:
    return dt.date.today().isoformat()


def _yesterday() -> str:
    return (dt.date.today() - dt.timedelta(days=1)).isoformat()


def _applied(tmp_path):
    """A station set up and applied, exactly as an operator leaves it at night."""
    client = config_page._client(config_page._repo(tmp_path))
    config_page._apply(client, head=False)
    return client


def _config(client) -> pathlib.Path:
    return config_page._generated(client.app.state.settings.repo_root)


def _overnight(client) -> str:
    """Move the clock, not the code: the draft is what goes stale on its own."""
    client.app.state.rig.draft.date = _yesterday()
    return _yesterday()


# ------------------------------------------------------------------- the roll


def test_a_session_dated_today_is_left_alone(tmp_path):
    """The common case, on every request of every day. Rolling a session that is
    already current would mint a new folder every time a page loaded."""
    client = _applied(tmp_path)
    before = client.app.state.rig.draft.session_uid

    assert day_roll.roll_if_new_day(client.app) is None
    assert client.app.state.rig.draft.session_uid == before


def test_a_session_from_yesterday_is_moved_onto_today(tmp_path):
    client = _applied(tmp_path)
    left = client.app.state.rig.draft.session_uid
    _overnight(client)

    rolled = day_roll.roll_if_new_day(client.app)

    assert rolled is not None
    assert rolled.from_date == _yesterday()
    assert rolled.to_date == _today()
    assert rolled.left_behind == left
    draft = client.app.state.rig.draft
    assert draft.date == _today()
    assert draft.session_uid != left, "the same uid on a new day is the same session"


def test_the_rig_carries_over(tmp_path):
    """Everything the operator configured belongs to the bench, not to the
    session. Re-typing six device serials every morning is the cost this whole
    feature exists to avoid, and it would buy nothing: the hardware did not
    move overnight."""
    client = _applied(tmp_path)
    draft = client.app.state.rig.draft
    before = (
        draft.preset_id,
        draft.task_name,
        draft.operator_name,
        dict(draft.grippers),
        dict(draft.required_cameras),
        session_state.device_fingerprint(draft),
    )
    _overnight(client)

    day_roll.roll_if_new_day(client.app)

    draft = client.app.state.rig.draft
    assert (
        draft.preset_id,
        draft.task_name,
        draft.operator_name,
        dict(draft.grippers),
        dict(draft.required_cameras),
        session_state.device_fingerprint(draft),
    ) == before


def test_the_device_test_and_calibration_survive_the_roll(tmp_path):
    """They are claims about hardware, and the fingerprint they expire on
    deliberately excludes the date and the uid. A roll that expired them would
    put a fifteen-minute camera check between an operator and their first
    episode of the day, every day."""
    client = _applied(tmp_path)
    before = session_state.device_fingerprint(client.app.state.rig.draft)
    _overnight(client)

    day_roll.roll_if_new_day(client.app)

    assert session_state.device_fingerprint(client.app.state.rig.draft) == before


# -------------------------------------------------------- the session left behind


def _left_folder(client) -> pathlib.Path:
    """The folder the draft names right now, expanded and created."""
    folder = pathlib.Path(
        os.path.expanduser(session_state.data_dir(client.app.state.rig.draft))
    )
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def test_the_folder_left_behind_is_sealed(tmp_path, monkeypatch):
    """Nothing else says a rolled session is finished -- it simply stops
    growing. Sealed, `session_slot` refuses to record into it again.

    The uid asserted is the one left, which is the whole ordering: the folder
    has to be captured before `start_new_session` moves the draft, and sealed
    after.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    client = _applied(tmp_path)
    left = client.app.state.rig.draft.session_uid
    # After `_overnight`, because the date is the first ten characters of the
    # folder: yesterday's episodes are in yesterday's directory.
    _overnight(client)
    folder = _left_folder(client)
    (folder / "episode_0").mkdir()

    day_roll.roll_if_new_day(client.app)

    marker = session_seal.read_seal(folder)
    assert marker is not None, "the rolled session was left unsealed"
    assert marker["reason"] == "day_roll"
    assert marker["session_uid"] == left
    assert marker["episode_count"] == 1
    assert not session_seal.folder_is_sealed(_left_folder(client)), (
        "the roll sealed the session it just opened"
    )


def test_an_idle_night_seals_nothing(tmp_path, monkeypatch):
    """Most rolls leave an empty folder behind, because the bench was idle.
    A seal there would mark a session nobody recorded -- `write_seal` refuses
    it, and this is the assertion that it is still asked.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    client = _applied(tmp_path)
    _overnight(client)
    folder = _left_folder(client)

    day_roll.roll_if_new_day(client.app)

    assert session_seal.folder_is_sealed(folder) is False


# --------------------------------------------------------- the recording guard


def test_a_live_recorder_stops_the_roll(tmp_path, monkeypatch):
    """The recorder writes into the folder named in the file it was handed. Roll
    under it and the console names one folder while episodes land in another --
    which is the split `session_store` refuses to create."""
    client = _applied(tmp_path)
    left = client.app.state.rig.draft.session_uid
    _overnight(client)
    monkeypatch.setattr(recording, "in_progress", lambda app: True)

    assert day_roll.roll_if_new_day(client.app) is None
    assert client.app.state.rig.draft.session_uid == left
    assert client.app.state.rig.draft.date == _yesterday()


def test_the_old_warning_is_what_a_recording_station_still_gets(tmp_path, monkeypatch):
    """The one case the console cannot decide for the operator: stopping a
    recording is their call. So the stale-day banner is still there, for
    exactly this."""
    client = _applied(tmp_path)
    _overnight(client)
    monkeypatch.setattr(recording, "in_progress", lambda app: True)

    body = client.get("/record").text

    assert "from another day" in body
    assert _yesterday() in body
    assert "as soon as this recording stops" in body


# ------------------------------------------------------------- the config file


def test_the_applied_config_follows_the_session(tmp_path):
    """The station records from this file, not from the page. Left carrying
    yesterday's folder, the new session's first episode lands under the old id
    while every page shows a different one."""
    client = _applied(tmp_path)
    _overnight(client)

    day_roll.roll_if_new_day(client.app)

    draft = client.app.state.rig.draft
    cfg = yaml.safe_load(_config(client).read_text())
    assert cfg["meta"]["session_uid"] == draft.session_uid
    assert cfg["recording"]["data_dir"] == session_state.data_dir(draft)
    assert _today() in cfg["recording"]["data_dir"]


def test_nothing_but_the_session_moves(tmp_path):
    """The point of a re-stamp rather than a re-assemble.

    Re-running assembly would re-resolve every OAK against whatever device scan
    the console holds this morning, so a stream applied last night at its
    module's native aspect can come back on the 16:10 fallback -- a different
    field of view, for a rig nobody touched. Asserted as a diff, so any future
    key that starts moving with the session shows up here rather than at a
    bench."""
    client = _applied(tmp_path)
    before = _config(client).read_text().splitlines()
    _overnight(client)

    day_roll.roll_if_new_day(client.app)

    after = _config(client).read_text().splitlines()
    assert len(before) == len(after), "the roll added or dropped lines"
    changed = [b.strip() for b, a in zip(before, after) if b != a]
    assert len(changed) == 2, f"more than the session moved: {changed}"
    assert any(line.startswith("session_uid:") for line in changed), changed
    assert any(line.startswith("data_dir:") for line in changed), changed


def test_a_console_restarted_after_the_roll_comes_up_on_todays_session(tmp_path):
    """Why the file is re-stamped rather than deleted. It is this station's only
    persistence -- `session_store` reads the draft back out of it -- so the
    manual button's `_disarm` would mean a console restarted after an overnight
    roll came up with Setup and Configuration blank."""
    client = _applied(tmp_path)
    _overnight(client)

    day_roll.roll_if_new_day(client.app)

    draft = client.app.state.rig.draft
    restored = session_store.restore(client.app.state.settings.state_dir)
    assert restored is not None, "the roll left nothing to restore"
    assert restored.session_uid == draft.session_uid
    assert restored.date == _today()
    assert restored.task_name == draft.task_name


def test_a_config_belonging_to_another_session_is_not_touched(tmp_path):
    """Two consoles share one checkout. A file some other session applied is not
    this roll's to move -- the same guard `_disarm` has."""
    client = _applied(tmp_path)
    path = _config(client)
    cfg = yaml.safe_load(path.read_text())
    cfg["meta"]["session_uid"] = "abcdef123456"
    path.write_text(yaml.safe_dump(cfg))
    stranger = path.read_text()
    _overnight(client)

    day_roll.roll_if_new_day(client.app)

    assert path.read_text() == stranger


def test_a_config_that_cannot_be_moved_is_removed(tmp_path):
    """A config still armed for yesterday is the one outcome worse than an extra
    Apply: the Configuration step unticks itself with the file, which is what
    "apply this session before recording it" looks like."""
    client = _applied(tmp_path)
    path = _config(client)
    _overnight(client)

    def _boom(*args, **kwargs):
        raise OSError("the state dir went read-only")

    original = day_roll.restamp_session
    day_roll.restamp_session = _boom
    try:
        day_roll.roll_if_new_day(client.app)
    finally:
        day_roll.restamp_session = original

    assert not path.exists()
    assert client.app.state.rig.draft.date == _today(), "the draft still rolled"


def test_a_broken_date_is_not_a_new_day(tmp_path):
    """`04/08/2026` is not a session that aged, it is one whose date was written
    wrong -- `config_form.hydrate` reads a hand-edited config back verbatim.
    Rolling it would repair the symptom, mint a fresh uid over it, and swallow
    the field-adjacent 400 Apply answers today. Two faults; one is a new day."""
    client = _applied(tmp_path)
    client.app.state.rig.draft.date = "04/08/2026"
    left = client.app.state.rig.draft.session_uid

    assert day_roll.roll_if_new_day(client.app) is None
    assert client.app.state.rig.draft.date == "04/08/2026"
    assert client.app.state.rig.draft.session_uid == left


def test_a_session_never_applied_rolls_anyway(tmp_path):
    """No file to move is not a reason to leave the console on yesterday."""
    client = config_page._client(config_page._repo(tmp_path))
    _overnight(client)

    rolled = day_roll.roll_if_new_day(client.app)

    assert rolled is not None
    assert client.app.state.rig.draft.date == _today()


# ------------------------------------------------------------------- the pages


def test_the_page_that_triggers_the_roll_already_shows_it(tmp_path):
    """Ordering, which is the whole reason this is middleware rather than a
    context processor. A context processor runs when the template renders --
    after the handler has read the date off the draft -- so the page would warn
    about a session that had been rolled a microsecond earlier."""
    client = _applied(tmp_path)
    _overnight(client)

    body = client.get("/config").text

    assert "from another day" not in body, "the page warned about a session it rolled"
    assert "Started a new session" in body
    assert client.app.state.rig.draft.date == _today()


def test_a_session_reopened_in_a_shift_named_folder_rolls_the_same_day(
    tmp_path, monkeypatch
):
    """Folder names written by older versions carry the shift after the date. A
    console reopened on such a session would otherwise name a folder without it
    while the recorder kept writing into the old one, so it moves on before any
    page renders: the old folder sealed, the config re-stamped."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    client = _applied(tmp_path)
    left = client.app.state.rig.draft.session_uid
    head, name = os.path.split(session_state.data_dir(client.app.state.rig.draft))
    old = pathlib.Path(head) / f"{name[:10]}_AM{name[10:]}"
    config_assembler.restamp_session(
        _config(client), session_uid=left, data_dir=str(old)
    )
    (old / "episode_0").mkdir(parents=True)
    client.app.state.rig.draft = session_store.restore(
        client.app.state.settings.state_dir
    )

    body = client.get("/config").text

    draft = client.app.state.rig.draft
    assert draft.session_uid != left
    assert draft.date == _today()
    assert "_AM_" not in session_state.data_dir(draft)
    assert session_seal.read_seal(old)["session_uid"] == left
    cfg = yaml.safe_load(_config(client).read_text())
    assert cfg["recording"]["data_dir"] == session_state.data_dir(draft)
    assert "Started a new session" in body


def test_the_notice_names_both_days(tmp_path):
    client = _applied(tmp_path)
    _overnight(client)

    body = client.get("/config").text

    assert _today() in body
    assert _yesterday() in body


def test_the_notice_is_shown_once(tmp_path):
    """It reports something that already happened. Repeating it on every page
    for the rest of the day is a banner operators learn to read past."""
    client = _applied(tmp_path)
    _overnight(client)

    assert "Started a new session" in client.get("/config").text
    assert "Started a new session" not in client.get("/config").text
    assert "Started a new session" not in client.get("/record").text


def test_a_poll_does_not_swallow_the_notice(tmp_path):
    """Every poll on this console fetches a whole page and picks a fragment out
    of it client-side (`hx-select`), so a notice consumed on any render at all
    is eaten by the one-second poll on /record and never reaches a human."""
    client = _applied(tmp_path)
    _overnight(client)

    client.get("/record", headers={"HX-Request": "true"})

    assert "Started a new session" in client.get("/record").text


def test_the_roll_reaches_a_page_the_operator_never_opens(tmp_path):
    """An operator can walk in and go straight to Record. The roll does not
    depend on visiting the Configuration page."""
    client = _applied(tmp_path)
    _overnight(client)

    body = client.get("/record").text

    assert "Started a new session" in body
    assert client.app.state.rig.draft.date == _today()


def test_static_files_do_not_roll_the_session(tmp_path):
    """A stylesheet is not a reason to mint a session id."""
    client = _applied(tmp_path)
    left = client.app.state.rig.draft.session_uid
    _overnight(client)

    client.get("/static/app.css")

    assert client.app.state.rig.draft.session_uid == left


def test_a_failing_roll_does_not_take_the_console_down(tmp_path, monkeypatch):
    """A console that will not serve a page cannot record, and the stale-day
    banner plus its button are still there behind the failure."""
    client = _applied(tmp_path)
    _overnight(client)

    def _boom(app):
        raise RuntimeError("the clock exploded")

    monkeypatch.setattr(day_roll, "roll_if_new_day", _boom)

    response = client.get("/config")

    assert response.status_code == 200
    assert "from another day" in response.text
    assert "/config/session/uid" in response.text, "no manual way out of the failure"


# ---------------------------------------------------------------- the primitive


def test_restamp_refuses_a_uid_that_cannot_be_a_folder(tmp_path):
    """The uid reaches `raw/{task}/{session}/` as a folder name. `assemble`
    refuses an invalid one for the same reason."""
    client = _applied(tmp_path)
    path = _config(client)
    before = path.read_text()

    with pytest.raises(ValueError):
        config_assembler.restamp_session(
            path, session_uid="not-a-uid", data_dir="/tmp/whatever"
        )

    assert path.read_text() == before


def test_restamp_refuses_an_empty_folder(tmp_path):
    client = _applied(tmp_path)
    path = _config(client)
    before = path.read_text()

    with pytest.raises(ValueError):
        config_assembler.restamp_session(
            path, session_uid="abcdef123456", data_dir="  "
        )

    assert path.read_text() == before


def test_restamp_refuses_a_file_this_editor_did_not_write(tmp_path):
    """Adding the blocks would hand the recorder a file that is half this
    build's shape and half something else's."""
    path = tmp_path / "stranger.yaml"
    path.write_text("cameras: []\n")

    with pytest.raises(ValueError):
        config_assembler.restamp_session(
            path, session_uid="abcdef123456", data_dir="/data/session"
        )

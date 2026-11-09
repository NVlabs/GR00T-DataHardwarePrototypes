# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rig ownership and the session draft.

One physical rig, one owner. If app state were rebound process-wide on every
request, two browser tabs would silently reconfigure each other; on a recording
console that means two operators editing one config and one of them losing a
session. First claim wins, a second token is read-only, and control only ever
moves through an explicit `take_control`.
"""

import datetime as dt
import sys
import threading
import time

import pytest

from station_ui.core.session_identity import is_valid_session_uid
from station_ui.errors import FieldError, StaleOwner
from station_ui.rig import (
    RigState,
    claim,
    is_owner,
    release,
    require_owner,
    take_control,
)
from station_ui.session import (
    SessionDraft,
    data_dir,
    has_episodes,
    regenerate_uid,
)
from umi.common.utility.output_layout import output_root

# ------------------------------------------------------------------ ownership


def test_first_claim_wins_and_the_second_token_is_read_only():
    state = RigState()
    assert claim(state, "tab-a") is True
    assert claim(state, "tab-b") is False
    require_owner(state, "tab-a")
    with pytest.raises(StaleOwner):
        require_owner(state, "tab-b")


def test_the_owners_own_token_reclaims_idempotently():
    """A reload re-posts the same token. Answering False there would drop the
    operator into read-only on every refresh of the page they already own."""
    state = RigState()
    assert claim(state, "tab-a") is True
    assert claim(state, "tab-a") is True
    require_owner(state, "tab-a")
    assert state.owner_token == "tab-a"


def test_an_unowned_rig_refuses_a_mutating_call():
    """Nobody has claimed yet: a mutating call must not implicitly become the
    owner, or the first stray POST takes the rig without anyone asking."""
    state = RigState()
    with pytest.raises(StaleOwner):
        require_owner(state, "tab-a")
    assert state.owner_token is None


def test_take_control_transfers_and_the_previous_owner_goes_read_only():
    state = RigState()
    claim(state, "tab-a")
    take_control(state, "tab-b")
    require_owner(state, "tab-b")
    with pytest.raises(StaleOwner):
        require_owner(state, "tab-a")
    assert is_owner(state, "tab-b") and not is_owner(state, "tab-a")


def test_take_control_of_an_unclaimed_rig_simply_claims_it():
    state = RigState()
    take_control(state, "tab-a")
    require_owner(state, "tab-a")


def test_control_never_transfers_without_take_control():
    """The exact failure this module exists to prevent. Reading the rig,
    claiming it and being refused must all leave the first owner in place --
    no amount of activity from a second tab may transfer control silently."""
    state = RigState()
    claim(state, "tab-a")
    for _ in range(5):
        assert claim(state, "tab-b") is False
        assert is_owner(state, "tab-b") is False
        with pytest.raises(StaleOwner):
            require_owner(state, "tab-b")
        assert is_owner(state, "tab-a") is True
    assert state.owner_token == "tab-a"


def test_release_hands_the_rig_to_the_next_claim():
    state = RigState()
    claim(state, "tab-a")
    release(state, "tab-a")
    assert state.owner_token is None
    assert claim(state, "tab-b") is True
    require_owner(state, "tab-b")


def test_a_stale_token_cannot_release_the_current_owner():
    """Otherwise release is a silent takeover with an extra step: any tab
    could free the rig and immediately claim it."""
    state = RigState()
    claim(state, "tab-a")
    with pytest.raises(StaleOwner):
        release(state, "tab-b")
    assert is_owner(state, "tab-a")


def test_releasing_an_unowned_rig_is_a_no_op():
    state = RigState()
    release(state, "tab-a")
    assert state.owner_token is None


def test_a_blank_token_can_never_own_the_rig():
    """`owner_token` is falsy when blank, so a blank claim would read as
    unowned and hand every later tab a free takeover."""
    state = RigState()
    for bad in ("", "   ", None, 17):
        with pytest.raises(ValueError):
            claim(state, bad)
        with pytest.raises(ValueError):
            take_control(state, bad)
    assert state.owner_token is None
    # And a blank token is never mistaken for the (absent) owner.
    assert is_owner(state, "") is False
    with pytest.raises(StaleOwner):
        require_owner(state, "")


class _SlowOwnerSlot(RigState):
    """A rig whose owner slot takes a visible moment to read and to write.

    Straight thread racing is not enough here, and that is measured, not
    assumed: a lock-free `if owner is None: owner = token` passed 20 rounds of
    20 racing threads even at a 1 microsecond switch interval, because CPython
    almost never preempts between those two bytecodes. Widening the window
    makes the interleaving certain instead of merely likely, so any
    check-then-set fails this deterministically while every genuinely atomic
    implementation passes -- it constrains the property, not the mechanism.
    """

    _SLOW_S = 0.05

    @property
    def owner_token(self):
        time.sleep(self._SLOW_S)
        return self.__dict__.get("_owner")

    @owner_token.setter
    def owner_token(self, value):
        time.sleep(self._SLOW_S)
        self.__dict__["_owner"] = value


def test_claim_is_atomic_when_the_owner_slot_is_slow_enough_to_interleave():
    state = _SlowOwnerSlot()
    wins = []
    record = threading.Lock()
    barrier = threading.Barrier(8)

    def go(i):
        barrier.wait()
        if claim(state, f"tab-{i}"):
            with record:
                wins.append(i)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(wins) == 1, f"{len(wins)} tabs each think they own the rig"
    assert state.owner_token == f"tab-{wins[0]}"


def test_twenty_concurrent_claims_yield_exactly_one_owner():
    """The unwidened race, for real threads on the real object. Weaker than
    the test above on its own -- keep both."""
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for _ in range(20):
            state, wins = RigState(), []
            record = threading.Lock()
            barrier = threading.Barrier(20)

            def go(i, state=state, wins=wins, barrier=barrier, record=record):
                barrier.wait()
                if claim(state, f"tab-{i}"):
                    with record:
                        wins.append(i)

            threads = [threading.Thread(target=go, args=(i,)) for i in range(20)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            assert len(wins) == 1, f"{len(wins)} tabs each think they own the rig"
            assert state.owner_token == f"tab-{wins[0]}"
    finally:
        sys.setswitchinterval(previous)


def test_the_rig_starts_with_its_own_ready_draft():
    state = RigState()
    assert isinstance(state.draft, SessionDraft)
    assert is_valid_session_uid(state.draft.session_uid)
    # A shared class-level default would give every RigState the same draft.
    assert RigState().draft is not state.draft


# --------------------------------------------------------------- session uid


def test_a_new_draft_always_has_a_valid_uid():
    draft = SessionDraft()
    assert is_valid_session_uid(draft.session_uid)
    first = draft.session_uid
    regenerate_uid(draft)
    assert is_valid_session_uid(draft.session_uid)
    assert draft.session_uid != first


def test_every_draft_gets_its_own_uid():
    """A class-level default would share one uid across every session the
    station ever records -- exactly the session folder collision the uid
    exists to stop."""
    assert len({SessionDraft().session_uid for _ in range(50)}) == 50


def test_editing_the_rest_of_the_draft_leaves_the_uid_alone():
    """The uid names the session folder. Re-minting it while the operator types
    would point the second half of a session at a different folder."""
    draft = SessionDraft()
    original = draft.session_uid
    draft.operator_name = "runyu"
    draft.task_name = "plane_assembly_v9"
    draft.preset_id = "bimanual_v4"
    draft.grippers["right"] = "UMI-V4-0002"
    draft.required_cameras["right_umi_camera"] = "oak-W97-51"
    assert draft.session_uid == original


def test_the_draft_defaults_to_todays_date():
    recorded = dt.date.fromisoformat(SessionDraft().date)
    assert abs((recorded - dt.date.today()).days) <= 1


# ------------------------------------------------------------- folder preview


def _bimanual_draft(**overrides):
    kwargs = dict(
        preset_id="bimanual_v4",
        task_name="plane_assembly_v9",
        date="2026-08-04",
        grippers={"left": "UMI-V4-0001", "right": "UMI-V4-0002"},
    )
    kwargs.update(overrides)
    return SessionDraft(**kwargs)


def test_the_folder_preview_ends_with_the_drafts_uid():
    draft = _bimanual_draft()
    path = data_dir(draft)
    assert path.startswith(str(output_root() / "raw" / "plane_assembly_v9") + "/")
    assert path.endswith(draft.session_uid)
    assert "2026-08-04" in path
    assert "umiV4_0001L" in path and "umiV4_0002R" in path
    assert "plane_assembly_v9" in path


def test_regenerating_the_uid_moves_the_folder_preview():
    draft = _bimanual_draft()
    before = data_dir(draft)
    regenerate_uid(draft)
    after = data_dir(draft)
    assert after != before
    assert after.endswith(draft.session_uid)


def test_a_non_iso_date_is_a_field_error_not_a_500():
    """`build_data_dir` raises ValueError on a non-ISO date. Unvalidated that
    escapes the route as a 500 while the operator is still mid-keystroke."""
    draft = _bimanual_draft(date="08/04/2026")
    with pytest.raises(FieldError) as excinfo:
        data_dir(draft)
    assert excinfo.value.field == "date"
    assert "08/04/2026" in str(excinfo.value)


def test_a_missing_preset_is_a_field_error_not_a_guessed_mode():
    draft = SessionDraft(task_name="t", date="2026-08-04")
    with pytest.raises(FieldError) as excinfo:
        data_dir(draft)
    assert excinfo.value.field == "preset_id"


def test_an_unknown_preset_is_a_field_error():
    draft = _bimanual_draft(preset_id="bimanual_v9")
    with pytest.raises(FieldError) as excinfo:
        data_dir(draft)
    assert excinfo.value.field == "preset_id"
    assert "bimanual_v9" in str(excinfo.value)


def test_a_corrupted_uid_is_a_field_error_not_a_500():
    draft = _bimanual_draft()
    draft.session_uid = "not-a-uid"
    with pytest.raises(FieldError) as excinfo:
        data_dir(draft)
    assert excinfo.value.field == "session_uid"


def test_a_field_error_carries_both_the_field_and_a_readable_message():
    error = FieldError("date", "must be YYYY-MM-DD")
    assert error.field == "date"
    assert error.message == "must be YYYY-MM-DD"
    assert "date" in str(error) and "must be YYYY-MM-DD" in str(error)


# -------------------------------------------------------------- episode freeze


def test_has_episodes_flips_once_an_episode_directory_exists(tmp_path, monkeypatch):
    """The draft refuses to re-mint the uid after the first episode: renaming
    the folder then would fork the session across two folders."""
    monkeypatch.setenv("UMI_OUTPUT_ROOT", str(tmp_path / "output"))
    draft = _bimanual_draft()
    folder = (
        tmp_path
        / "output"
        / "raw"
        / "plane_assembly_v9"
        / data_dir(draft).rsplit("/", 1)[-1]
    )

    assert has_episodes(draft) is False
    folder.mkdir(parents=True)
    assert has_episodes(draft) is False
    (folder / "logs").mkdir()
    (folder / "config.yaml").write_text("x")
    assert has_episodes(draft) is False, "logs/ and config.yaml are not episodes"
    (folder / "episode_0").mkdir()
    assert has_episodes(draft) is True


def test_has_episodes_is_false_when_the_folder_cannot_even_be_computed():
    """Called while the operator is still typing the date, so it must answer
    'nothing recorded here' rather than raising out of a render."""
    assert (
        has_episodes(SessionDraft(preset_id="bimanual_v4", date="08/04/2026")) is False
    )
    assert has_episodes(SessionDraft()) is False

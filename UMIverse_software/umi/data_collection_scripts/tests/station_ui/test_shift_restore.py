# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reopening a session whose folder an older build named with a shift.

Older builds put the shift right after the date
(``2026-08-27_AM_umiV4_0002R_…``). This build names the same session without
it, so a restore keeps the recorded folder -- still the one the config is armed
with -- and marks the session stale, so the day roll moves on to a new session
instead of the console silently renaming a half-recorded folder.
"""

import pytest

from station_ui.config_form import hydrate
from station_ui.session import SessionDraft, data_dir, recording_day_is_stale

UID = "91e98c89fccb"


def _cfg(folder: str) -> dict:
    return {
        "meta": {"session_uid": UID},
        "recording": {"data_dir": f"~/Dev/UMI/local_data/{folder}"},
    }


@pytest.mark.parametrize("shift", ["AM", "PM", "OOS"])
def test_a_shift_named_folder_is_kept_until_the_day_roll(shift):
    cfg = _cfg(f"2026-08-27_{shift}_umiV4_0002R_task_{UID}")
    draft = SessionDraft()
    hydrate(draft, cfg)
    assert draft.date == "2026-08-27"
    assert data_dir(draft) == cfg["recording"]["data_dir"]
    assert recording_day_is_stale(draft) is True


def test_a_folder_named_without_a_shift_is_not_kept():
    draft = SessionDraft()
    hydrate(draft, _cfg(f"2026-08-27_umiV4_0002R_task_{UID}"))
    assert (draft.date, draft.legacy_folder) == ("2026-08-27", "")


def test_a_folder_that_is_only_a_date_is_not_kept():
    draft = SessionDraft()
    hydrate(draft, _cfg("2026-08-27"))
    assert (draft.date, draft.legacy_folder) == ("2026-08-27", "")


def test_a_config_with_no_folder_leaves_the_date_alone():
    draft = SessionDraft(date="2026-08-01")
    hydrate(draft, {"meta": {}, "recording": {}})
    assert (draft.date, draft.legacy_folder) == ("2026-08-01", "")

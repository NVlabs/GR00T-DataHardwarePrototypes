# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`seal_session`, the one function every station trigger calls.

A seal is the on-disk marker, and the reason every trigger writes the same one
is that no trigger writes it itself: they call this. `write_seal` is tested
where it lives; what is asserted here is the decision this module makes and
nothing else does -- which refusals collapse to `False` (all of them, which is
what makes calling it twice safe), and what the station fills in (the folder's
episode count and this station's id).

The wiring itself -- that "Start a new session" and the day roll actually call
this, with the folder they left rather than the one they just opened -- is
asserted against the real handlers in `test_config_page.py` and
`test_day_roll.py`. A stub here would pass against a trigger that never called
it.
"""

from station_ui import seal as seal_mod
from umi.common.recording import session_seal


def _session(tmp_path, episodes=1):
    folder = tmp_path / "2026-08-26_umiV3_0001R_vive_task_a1b2c3d4e5f6"
    folder.mkdir()
    for index in range(episodes):
        (folder / f"episode_{index}").mkdir()
    return folder


def test_seal_session_writes_the_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("UMI_STATION_ID", "bench-7")
    folder = _session(tmp_path, episodes=3)

    assert (
        seal_mod.seal_session(folder, reason="day_roll", session_uid="a1b2c3d4e5f6")
        is True
    )

    assert session_seal.folder_is_sealed(folder) is True
    marker = session_seal.read_seal(folder)
    assert marker["reason"] == "day_roll"
    assert marker["episode_count"] == 3
    assert marker["session_uid"] == "a1b2c3d4e5f6"
    assert marker["station_id"] == "bench-7"


def test_seal_session_does_nothing_for_an_empty_session(tmp_path):
    folder = _session(tmp_path, episodes=0)
    assert seal_mod.seal_session(folder, reason="day_roll") is False
    assert session_seal.folder_is_sealed(folder) is False


def test_seal_session_is_idempotent(tmp_path):
    folder = _session(tmp_path)
    assert seal_mod.seal_session(folder, reason="operator") is True
    assert seal_mod.seal_session(folder, reason="ttl") is False
    assert session_seal.read_seal(folder)["reason"] == "operator"


# Test the previous-folder and orphan-sealing decisions directly, including
# missing configs and unchanged paths. Route tests cover their Apply integration.


def test_previous_data_dir_reads_the_armed_config(tmp_path):
    from station_ui.routes import config as config_routes

    path = tmp_path / "generated.yaml"
    path.write_text("recording:\n  data_dir: /data/2026-08-26_x_a1b2c3d4e5f6\n")
    assert config_routes._previous_data_dir(path) == "/data/2026-08-26_x_a1b2c3d4e5f6"


def test_previous_data_dir_is_blank_when_there_is_no_config(tmp_path):
    from station_ui.routes import config as config_routes

    assert config_routes._previous_data_dir(tmp_path / "nope.yaml") == ""
    assert config_routes._previous_data_dir(None) == ""


def test_previous_data_dir_survives_broken_yaml(tmp_path):
    from station_ui.routes import config as config_routes

    path = tmp_path / "generated.yaml"
    path.write_text("recording: [unclosed")
    assert config_routes._previous_data_dir(path) == ""


def test_seal_if_orphaned_seals_a_folder_the_draft_left(tmp_path):
    from station_ui.routes import config as config_routes

    old = _session(tmp_path, episodes=2)
    new = tmp_path / "2026-08-26_umiV3_0001R_vive_other_task_ffffffffffff"
    new.mkdir()

    config_routes._seal_if_orphaned(str(old), str(new))

    assert session_seal.folder_is_sealed(old) is True
    assert session_seal.read_seal(old)["reason"] == "dir_change"


def test_seal_if_orphaned_does_nothing_when_the_folder_is_unchanged(tmp_path):
    from station_ui.routes import config as config_routes

    folder = _session(tmp_path, episodes=2)
    config_routes._seal_if_orphaned(str(folder), str(folder))
    assert session_seal.folder_is_sealed(folder) is False


def test_seal_if_orphaned_does_nothing_for_an_empty_previous_folder(tmp_path):
    from station_ui.routes import config as config_routes

    old = _session(tmp_path, episodes=0)
    config_routes._seal_if_orphaned(str(old), str(tmp_path / "new"))
    assert session_seal.folder_is_sealed(old) is False


def test_seal_if_left_behind_seals_the_folder_the_console_left(tmp_path):
    """The shared rule: whatever folder the console stopped naming is sealed."""
    left = _session(tmp_path)

    assert (
        seal_mod.seal_if_left_behind(
            str(left), str(tmp_path / "somewhere_else"), reason="config_drift"
        )
        is True
    )

    assert session_seal.folder_is_sealed(left) is True
    marker = session_seal.read_seal(left)
    assert marker["reason"] == "config_drift"
    # From the folder name, not a caller: the draft has already moved on.
    assert marker["session_uid"] == "a1b2c3d4e5f6"


def test_seal_if_left_behind_does_nothing_when_the_folder_did_not_move(tmp_path):
    folder = _session(tmp_path)

    assert (
        seal_mod.seal_if_left_behind(str(folder), str(folder), reason="config_drift")
        is False
    )

    assert session_seal.folder_is_sealed(folder) is False


def test_seal_if_left_behind_is_idempotent(tmp_path):
    """No caller checks whether the folder is already sealed, because
    `write_seal` refuses to overwrite. That refusal is the whole guard."""
    left = _session(tmp_path)
    other = str(tmp_path / "somewhere_else")

    assert seal_mod.seal_if_left_behind(str(left), other, reason="config_drift") is True
    assert seal_mod.seal_if_left_behind(str(left), other, reason="day_roll") is False, (
        "the second call overwrote a seal"
    )

    assert session_seal.read_seal(left)["reason"] == "config_drift"

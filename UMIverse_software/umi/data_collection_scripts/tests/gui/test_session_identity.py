# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session identity: uid shape, folder-name layout, and the episode freeze."""

import os
import pathlib
import re

import pytest

from station_ui.core.config_utils import build_data_dir
from station_ui.core.session_identity import (
    SESSION_UID_CHARS,
    SESSION_UID_LEN,
    build_session_dir_name,
    count_episodes,
    is_valid_session_uid,
    new_session_uid,
    session_has_episodes,
    session_uid_from_dir_name,
)
from umi.common.utility.output_layout import output_root


def test_uid_is_twelve_lowercase_hex_chars():
    uid = new_session_uid()
    assert len(uid) == SESSION_UID_LEN == 12
    assert set(uid) <= SESSION_UID_CHARS
    assert uid == uid.lower()


def test_uids_are_distinct():
    assert len({new_session_uid() for _ in range(500)}) == 500


def test_validator_rejects_wrong_length_and_uppercase_and_non_hex():
    assert is_valid_session_uid("0123456789ab") is True
    assert is_valid_session_uid("0123456789a") is False
    assert is_valid_session_uid("0123456789AB") is False
    assert is_valid_session_uid("0123456789ag") is False


def test_validator_is_total_on_non_strings():
    for value in (None, 123, b"0123456789ab", ["0123456789ab"], object()):
        assert is_valid_session_uid(value) is False


def test_date_stays_at_offset_zero():
    # The audit finds a day's sessions by the date prefix. The uid must never
    # move the date.
    name = build_session_dir_name(
        date="2026-08-03",
        task_name="plane_assembly_v9",
        mode="bimanual",
        left_gripper="UMI-V4-0001",
        right_gripper="UMI-V4-0002",
        session_uid="91e98c89fccb",
    )
    assert name[:10] == "2026-08-03"
    assert name.split("_", 1)[0] == "2026-08-03"
    assert re.match(r"^\d{4}-\d{2}-\d{2}_", name)


def test_the_grippers_sit_right_after_the_date():
    name = build_session_dir_name(
        date="2026-08-27",
        task_name="task",
        mode="unimanual",
        right_gripper="UMI-V4-0002",
        session_uid="91e98c89fccb",
    )
    assert name.startswith("2026-08-27_umiV4_0002R_")


def test_build_data_dir_puts_the_grippers_right_after_the_date():
    path = build_data_dir(
        "2026-08-27",
        "plane_assembly_v9",
        "unimanual",
        right_gripper_num="UMI-V4-0002",
        session_uid="91e98c89fccb",
    )
    assert "/2026-08-27_umiV4_0002R_" in path


def test_uid_is_the_final_segment():
    name = build_session_dir_name(
        date="2026-08-03",
        task_name="task",
        mode="unimanual",
        right_gripper="UMI-V4-0002",
        session_uid="91e98c89fccb",
    )
    assert name.endswith("_91e98c89fccb")
    assert session_uid_from_dir_name(name) == "91e98c89fccb"


def test_legacy_names_without_a_uid_still_parse():
    legacy = "2026-08-02_umiV3_0001L_umiV3_0002R_debug_task"
    assert session_uid_from_dir_name(legacy) is None


def test_bad_uid_is_rejected_by_the_builder():
    for bad in ("NOTHEX", "", "0123456789A B", "0123456789abc", None, 42):
        with pytest.raises(ValueError, match="session_uid"):
            build_session_dir_name(
                date="2026-08-03",
                task_name="task",
                mode="unimanual",
                right_gripper="UMI-V4-0002",
                session_uid=bad,
            )


def test_builder_defaults_grippers_and_is_total_on_none():
    unimanual = build_session_dir_name(
        date="2026-08-03",
        task_name="task",
        mode="unimanual",
        session_uid="91e98c89fccb",
    )
    assert unimanual == "2026-08-03_umi1R_task_91e98c89fccb"

    bimanual = build_session_dir_name(
        date="2026-08-03",
        task_name="task",
        mode="bimanual",
        session_uid="91e98c89fccb",
    )
    assert bimanual == "2026-08-03_umi1L_umi2R_task_91e98c89fccb"

    # Every field except `date` degrades on None / blank / non-string rather
    # than raising; `date` is load-bearing at offset 0 and is validated.
    for kwargs in (
        {"date": "2026-08-03", "task_name": None, "mode": "bimanual"},
        {"date": "2026-08-03", "task_name": "task", "mode": None},
        {"date": "2026-08-03", "task_name": "", "mode": ""},
        {"date": "  2026-08-03  ", "task_name": 7, "mode": object()},
    ):
        name = build_session_dir_name(session_uid="91e98c89fccb", **kwargs)
        assert isinstance(name, str)
        assert name.startswith("2026-08-03_")
        assert name.endswith("91e98c89fccb")
        assert "__" not in name


def test_builder_rejects_a_date_that_is_not_iso():
    # The whole module exists to keep LEFT(session_id, 10) and
    # split("_", 1)[0] on the date. A dropped or malformed date slot moves
    # every downstream parser's offset 0, so it fails loudly instead.
    for bad in (None, "", "  ", "2026-8-4", "04-08-2026", "2026-08-03_x", 20260803):
        with pytest.raises(ValueError, match="date"):
            build_session_dir_name(
                date=bad,
                task_name="task",
                mode="unimanual",
                right_gripper="UMI-V4-0002",
                session_uid="91e98c89fccb",
            )
        with pytest.raises(ValueError, match="date"):
            build_data_dir(
                date=bad,
                task_name="task",
                mode="unimanual",
                right_gripper_num="UMI-V4-0002",
                session_uid="91e98c89fccb",
            )

    ok = build_session_dir_name(
        date="2026-08-03",
        task_name="task",
        mode="unimanual",
        right_gripper="UMI-V4-0002",
        session_uid="91e98c89fccb",
    )
    assert ok.startswith("2026-08-03_")


def test_uid_extraction_is_total_and_suffix_anchored():
    assert session_uid_from_dir_name(None) is None
    assert session_uid_from_dir_name("") is None
    assert session_uid_from_dir_name(12345) is None
    # Uppercase hex is not a uid.
    assert session_uid_from_dir_name("2026-08-03_task_91E98C89FCCB") is None
    # Longer / shorter hex runs are not uids.
    assert session_uid_from_dir_name("2026-08-03_task_91e98c89fcc") is None
    assert session_uid_from_dir_name("2026-08-03_task_91e98c89fccbb") is None
    # A trailing separator or a full path still resolves.
    assert (
        session_uid_from_dir_name("/a/b/2026-08-03_task_91e98c89fccb/")
        == "91e98c89fccb"
    )
    assert (
        session_uid_from_dir_name(
            pathlib.PurePosixPath("/a/b/2026-08-03_task_91e98c89fccb")
        )
        == "91e98c89fccb"
    )


def test_uid_round_trips_through_the_builder():
    for _ in range(200):
        uid = new_session_uid()
        name = build_session_dir_name(
            date="2026-08-03",
            task_name="task",
            mode="bimanual",
            left_gripper="UMI-V4-0001",
            right_gripper="UMI-V4-0002",
            session_uid=uid,
        )
        assert session_uid_from_dir_name(name) == uid


def test_build_data_dir_requires_a_uid_and_appends_it():
    path = build_data_dir(
        date="2026-08-03",
        task_name="task",
        mode="bimanual",
        left_gripper_num="UMI-V4-0001",
        right_gripper_num="UMI-V4-0002",
        session_uid="91e98c89fccb",
    )
    assert path.endswith("_91e98c89fccb")
    assert "/raw/task/2026-08-03_" in path
    with pytest.raises(TypeError):
        build_data_dir(
            date="2026-08-03",
            task_name="task",
            mode="bimanual",
            left_gripper_num="UMI-V4-0001",
            right_gripper_num="UMI-V4-0002",
        )


def test_build_data_dir_rejects_a_malformed_uid():
    with pytest.raises(ValueError, match="session_uid"):
        build_data_dir(
            date="2026-08-03",
            task_name="task",
            mode="bimanual",
            left_gripper_num="UMI-V4-0001",
            right_gripper_num="UMI-V4-0002",
            session_uid="NOTHEX",
        )


def test_build_data_dir_keeps_the_elp_prefix_before_the_task():
    path = build_data_dir(
        date="2026-08-03",
        task_name="task",
        mode="unimanual",
        right_gripper_num="UMI-V4-0002",
        camera_type="elp",
        session_uid="91e98c89fccb",
    )
    assert path == str(
        output_root() / "raw" / "task" / "2026-08-03_umiV4_0002R_ELP_task_91e98c89fccb"
    )


def test_build_data_dir_drops_the_elp_marker_when_the_task_is_blank():
    # A blank task omits the ELP marker rather than leaving a dangling path segment.
    path = build_data_dir(
        date="2026-08-03",
        task_name="",
        mode="unimanual",
        right_gripper_num="UMI-V4-0002",
        camera_type="elp",
        session_uid="91e98c89fccb",
    )
    assert path == str(
        output_root() / "raw" / "untitled_task" / "2026-08-03_umiV4_0002R_91e98c89fccb"
    )
    assert "ELP" not in path


def test_build_data_dir_is_total_on_blank_fields():
    # The GUI calls this on every keystroke, so a half-filled form must not
    # raise and must not leak "None" into the folder name.
    for kwargs in (
        {"date": "2026-08-03", "task_name": None, "mode": None},
        {"date": "2026-08-03", "task_name": "", "mode": ""},
        {"date": "2026-08-03", "task_name": None, "mode": "bimanual"},
    ):
        path = build_data_dir(
            right_gripper_num=None,
            left_gripper_num=None,
            session_uid="91e98c89fccb",
            **kwargs,
        )
        assert path.startswith(str(output_root() / "raw" / "untitled_task") + os.sep)
        assert path.endswith("_91e98c89fccb")
        name = os.path.basename(path)
        assert "None" not in name
        assert "__" not in name


def test_session_has_episodes_detects_recorded_episodes(tmp_path):
    assert session_has_episodes(tmp_path) is False
    (tmp_path / "episode_0").mkdir()
    assert session_has_episodes(tmp_path) is True


def test_session_has_episodes_ignores_non_episode_dirs(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "config.yaml").write_text("{}")
    assert session_has_episodes(tmp_path) is False


def test_session_has_episodes_is_total_on_garbage(tmp_path):
    assert session_has_episodes(None) is False
    assert session_has_episodes("") is False
    assert session_has_episodes(tmp_path / "does_not_exist") is False
    assert session_has_episodes(123) is False
    assert session_has_episodes(object()) is False

    a_file = tmp_path / "not_a_dir"
    a_file.write_text("x")
    assert session_has_episodes(a_file) is False

    # Only directories named exactly episode_<digits> count.
    (tmp_path / "episode_").mkdir()
    (tmp_path / "episode_x").mkdir()
    (tmp_path / "episode_1_backup").mkdir()
    (tmp_path / "episode_2").write_text("a file, not an episode")
    assert session_has_episodes(tmp_path) is False

    (tmp_path / "episode_17").mkdir()
    assert session_has_episodes(tmp_path) is True


def test_session_has_episodes_accepts_str_paths(tmp_path):
    (tmp_path / "episode_0").mkdir()
    assert session_has_episodes(str(tmp_path)) is True


def test_session_has_episodes_sees_the_dir_build_data_dir_names():
    # build_data_dir returns an absolute <output>/raw/<task>/<session> string
    # -- exactly what the GUI holds in its data_dir field.
    data_dir = build_data_dir(
        date="2026-08-03",
        task_name="task",
        mode="unimanual",
        right_gripper_num="UMI-V4-0002",
        session_uid="91e98c89fccb",
    )
    assert os.path.isabs(data_dir)

    session = pathlib.Path(data_dir)
    session.mkdir(parents=True)
    assert session_has_episodes(data_dir) is False

    (session / "episode_0").mkdir()
    assert session_has_episodes(data_dir) is True


def test_session_has_episodes_expands_a_hand_written_home_path(tmp_path, monkeypatch):
    # A hand-written data_dir may start with "~", and Path("~/...") is a
    # *relative* dir named "~". Without expanduser the freeze guard silently
    # no-ops on it.
    monkeypatch.setenv("HOME", str(tmp_path))
    data_dir = "~/sessions/2026-08-03_umiV4_0002R_task_91e98c89fccb"

    session = pathlib.Path(os.path.expanduser(data_dir))
    session.mkdir(parents=True)
    assert session_has_episodes(data_dir) is False

    (session / "episode_0").mkdir()
    assert session_has_episodes(data_dir) is True


def test_session_has_episodes_counts_an_in_flight_pending_episode(tmp_path):
    # A pending episode means writing has begun. Freeze the session folder
    # before any finalized episode exists.
    (tmp_path / "episode_0.pending").mkdir()
    assert session_has_episodes(tmp_path) is True


def test_session_has_episodes_ignores_pending_lookalikes(tmp_path):
    (tmp_path / "episode_x.pending").mkdir()
    (tmp_path / "episode_0.pending.bak").mkdir()
    (tmp_path / "episode_0.pend").mkdir()
    # A *file* named like a pending episode is not a recording.
    (tmp_path / "episode_1.pending").write_text("not a directory")
    assert session_has_episodes(tmp_path) is False


def test_count_episodes_agrees_with_the_has_any_flag(tmp_path):
    """The audit page shows the count; the GUI freezes the folder name off the
    flag. Two counters would eventually disagree about what an episode is."""
    assert count_episodes(tmp_path) == 0
    assert session_has_episodes(tmp_path) is False

    (tmp_path / "episode_0").mkdir()
    (tmp_path / "episode_1.pending").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "episode_x").mkdir()
    (tmp_path / "episode_2").write_text("a file, not an episode")

    assert count_episodes(tmp_path) == 2
    assert session_has_episodes(tmp_path) is True


def test_count_episodes_is_total_on_garbage(tmp_path):
    assert count_episodes(None) == 0
    assert count_episodes("") == 0
    assert count_episodes(tmp_path / "does_not_exist") == 0
    assert count_episodes(object()) == 0


def test_session_has_episodes_stops_at_the_first_episode(tmp_path, monkeypatch):
    """It runs on every keystroke in the GUI's session field, against a folder
    that holds a full day of episodes, so it stops at the first episode rather
    than counting them all: with `count_episodes(...) > 0` the probe below
    would report twenty matches instead of one.
    """
    from station_ui.core import session_identity as identity

    for index in range(20):
        (tmp_path / f"episode_{index}").mkdir()

    examined = []
    real_match = identity._EPISODE_RE.match

    class _CountingRe:
        def match(self, name):
            examined.append(name)
            return real_match(name)

    monkeypatch.setattr(identity, "_EPISODE_RE", _CountingRe())

    assert session_has_episodes(tmp_path) is True
    assert len(examined) == 1, f"walked {len(examined)} entries to answer 'any?'"

    examined.clear()
    assert count_episodes(tmp_path) == 20
    assert len(examined) == 20


class TestPoseTrackerSegment:
    """The wrist pose source, stamped into the name from the selected preset.

    It goes between the grippers and the task, which is the only region of the
    name that is free: the date owns the first ten characters (date-range
    queries match on that prefix) and the uid owns the tail
    (`session_uid_from_dir_name` anchors on the end). Both invariants are
    re-asserted here with a tracker present, because a new segment in the
    middle is exactly the kind of change that quietly shifts one of them.
    """

    @staticmethod
    def _name(**over):
        kwargs = dict(
            date="2026-08-03",
            task_name="plane_assembly_v9",
            mode="bimanual",
            left_gripper="UMI-V4-0001",
            right_gripper="UMI-V4-0002",
            session_uid="91e98c89fccb",
        )
        kwargs.update(over)
        return build_session_dir_name(**kwargs)

    @pytest.mark.parametrize("tracker", ["vive", "iphone", "notracker"])
    def test_the_token_lands_between_the_grippers_and_the_task(self, tracker):
        assert self._name(pose_tracker=tracker) == (
            f"2026-08-03_umiV4_0001L_umiV4_0002R_{tracker}_plane_assembly_v9_91e98c89fccb"
        )

    @pytest.mark.parametrize("tracker", ["vive", "iphone", "notracker"])
    def test_the_date_and_uid_invariants_survive_it(self, tracker):
        name = self._name(pose_tracker=tracker)
        assert name[:10] == "2026-08-03"
        assert name.split("_", 1)[0] == "2026-08-03"
        assert session_uid_from_dir_name(name) == "91e98c89fccb"

    def test_a_caller_with_no_preset_still_mints_the_old_name(self):
        """The CLI path passes no tracker, and must not grow an empty segment.

        Blank parts are dropped rather than joined, so this is "…R_task_uid"
        and never a "__" hole that no parser expects.
        """
        assert self._name(pose_tracker=None) == self._name()
        assert self._name(pose_tracker="") == self._name()
        assert "__" not in self._name(pose_tracker="")

    def test_it_does_not_disturb_the_unimanual_shape(self):
        assert (
            self._name(mode="unimanual", pose_tracker="vive", left_gripper=None)
            == "2026-08-03_umiV4_0002R_vive_plane_assembly_v9_91e98c89fccb"
        )

    def test_a_blank_task_leaves_no_hole_around_the_tracker(self):
        name = self._name(task_name="", pose_tracker="vive")
        assert name == "2026-08-03_umiV4_0001L_umiV4_0002R_vive_91e98c89fccb"
        assert "__" not in name

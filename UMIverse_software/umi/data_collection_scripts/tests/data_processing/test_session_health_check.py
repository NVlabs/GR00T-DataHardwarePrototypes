# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from umi.common.utility.output_layout import counterpart, output_root, session_dir


def _make_dirs(parent, names):
    for name in names:
        os.makedirs(os.path.join(parent, name), exist_ok=True)


def _write_yaml(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f)


def test_find_sessions_returns_matching(tmp_path):
    from real_scripts.data_processing.session_health_check import (
        find_sessions_across_roots,
    )

    _make_dirs(
        tmp_path,
        [
            "2026-04-02_umi35L_taskA",
            "2026-04-02_umi36R_taskB",
            "2026-04-02_umi35L_taskA_aligned",
            "2026-04-02_umi35L_taskA_quarantine",
            "2026-04-02_umi35L_taskA_extra_suffix",
            "2026-04-01_umi35L_yesterday",
            "not_a_session",
        ],
    )
    _write_yaml(
        os.path.join(tmp_path, "2026-04-02_umi35L_taskA", "config.yaml"),
        {"meta": {"task_name": "taskA"}},
    )
    _write_yaml(
        os.path.join(tmp_path, "2026-04-02_umi36R_taskB", "config.yaml"),
        {"meta": {"task_name": "taskB"}},
    )
    # Same meta.task_name as raw session, but basename has appended segments → skipped.
    for suffix in ("_aligned", "_quarantine", "_extra_suffix"):
        _write_yaml(
            os.path.join(tmp_path, f"2026-04-02_umi35L_taskA{suffix}", "config.yaml"),
            {"meta": {"task_name": "taskA"}},
        )

    results = find_sessions_across_roots([str(tmp_path)], "2026-04-02")
    names = [os.path.basename(r) for r in results]
    assert "2026-04-02_umi35L_taskA" in names
    assert "2026-04-02_umi36R_taskB" in names
    assert "2026-04-02_umi35L_taskA_aligned" not in names
    assert "2026-04-02_umi35L_taskA_quarantine" not in names
    assert "2026-04-02_umi35L_taskA_extra_suffix" not in names
    assert "2026-04-01_umi35L_yesterday" not in names
    assert "not_a_session" not in names


def test_find_sessions_sorted(tmp_path):
    from real_scripts.data_processing.session_health_check import (
        find_sessions_across_roots,
    )

    _make_dirs(
        tmp_path,
        [
            "2026-04-02_umi36R_taskB",
            "2026-04-02_umi35L_taskA",
        ],
    )
    _write_yaml(
        os.path.join(tmp_path, "2026-04-02_umi35L_taskA", "config.yaml"),
        {"meta": {"task_name": "taskA"}},
    )
    _write_yaml(
        os.path.join(tmp_path, "2026-04-02_umi36R_taskB", "config.yaml"),
        {"meta": {"task_name": "taskB"}},
    )
    results = find_sessions_across_roots([str(tmp_path)], "2026-04-02")
    names = [os.path.basename(r) for r in results]
    assert names == sorted(names)


def test_find_sessions_missing_dir():
    from real_scripts.data_processing.session_health_check import (
        find_sessions_across_roots,
    )

    assert find_sessions_across_roots(["/nonexistent/path"], "2026-04-02") == []


def test_find_sessions_filters_by_task_name(tmp_path):
    from real_scripts.data_processing.session_health_check import (
        find_sessions_across_roots,
    )

    s_a = tmp_path / "2026-04-02_umi35L_plane_assembly_v10"
    s_b = tmp_path / "2026-04-02_umi36R_other_task"
    s_a.mkdir()
    s_b.mkdir()
    _write_yaml(s_a / "config.yaml", {"meta": {"task_name": "plane_assembly_v10"}})
    _write_yaml(s_b / "config.yaml", {"meta": {"task_name": "other_task"}})

    results = find_sessions_across_roots(
        [str(tmp_path)], "2026-04-02", task_name="plane_assembly_v10"
    )
    names = [os.path.basename(r) for r in results]
    assert names == ["2026-04-02_umi35L_plane_assembly_v10"]


def test_find_sessions_skips_when_basename_task_mismatch(tmp_path):
    """Folder name must end with _<meta.task_name>; misnamed sessions are ignored."""
    from real_scripts.data_processing.session_health_check import (
        find_sessions_across_roots,
    )

    bad = tmp_path / "2026-04-02_umi35L_wrong_slug"
    bad.mkdir()
    _write_yaml(bad / "config.yaml", {"meta": {"task_name": "plane_assembly_v10"}})

    assert find_sessions_across_roots([str(tmp_path)], "2026-04-02") == []


# --- load_session_config ---


def test_load_session_config_reads_yaml(tmp_path):
    from real_scripts.data_processing.session_health_check import load_session_config

    _write_yaml(tmp_path / "config.yaml", {"meta": {"task_name": "pouring"}})
    cfg = load_session_config(str(tmp_path))
    assert cfg["meta"]["task_name"] == "pouring"


def test_load_session_config_missing_raises(tmp_path):
    from real_scripts.data_processing.session_health_check import load_session_config

    with pytest.raises(FileNotFoundError):
        load_session_config(str(tmp_path))


# --- find_episodes ---


def test_find_episodes_returns_sorted_indices(tmp_path):
    from real_scripts.data_processing.session_health_check import find_episodes

    for name in ["episode_3", "episode_0", "episode_1", "config.yaml", "logs"]:
        p = tmp_path / name
        if name.startswith("episode_"):
            p.mkdir()
        else:
            p.touch()
    assert find_episodes(str(tmp_path)) == [0, 1, 3]


def test_find_episodes_empty(tmp_path):
    from real_scripts.data_processing.session_health_check import find_episodes

    assert find_episodes(str(tmp_path)) == []


def test_find_episodes_nonexistent_dir():
    from real_scripts.data_processing.session_health_check import find_episodes

    assert find_episodes("/nonexistent/session/path") == []


# --- check_alignment_status ---


def test_check_alignment_status_missing_dir(tmp_path):
    from real_scripts.data_processing.session_health_check import check_alignment_status

    # aligned_dir is empty — episode_0 dir doesn't exist
    ok, needs = check_alignment_status(str(tmp_path), [0, 1], "right_umi_camera")
    assert ok == []
    assert needs == [0, 1]


def test_check_alignment_status_valid_episode(tmp_path):
    from real_scripts.data_processing.session_health_check import check_alignment_status

    (tmp_path / "episode_0").mkdir()
    with patch(
        "real_scripts.data_processing.session_health_check._validate_aligned_episode",
        return_value=[],  # no problems
    ):
        ok, needs = check_alignment_status(str(tmp_path), [0], "right_umi_camera")
    assert ok == [0]
    assert needs == []


def test_check_alignment_status_invalid_episode(tmp_path):
    from real_scripts.data_processing.session_health_check import check_alignment_status

    (tmp_path / "episode_0").mkdir()
    with patch(
        "real_scripts.data_processing.session_health_check._validate_aligned_episode",
        return_value=["valid_timestamps missing"],
    ):
        ok, needs = check_alignment_status(str(tmp_path), [0], "right_umi_camera")
    assert ok == []
    assert needs == [0]


def _write_json(path, data):
    path = str(path)
    with open(path, "w") as f:
        json.dump(data, f)


# --- Report ---


def _make_report(overrides=None):
    base = dict(
        session_folder="/data/2026-04-02_test",
        episode_indices=[0, 1, 2],
        align_ok=[0, 2],
        align_remediated=[2],
        align_failed={},
    )
    if overrides:
        base.update(overrides)
    return base


def test_build_session_report_structure(tmp_path):
    from real_scripts.data_processing.session_health_check import build_session_report

    r = build_session_report(**_make_report())
    assert r["total_episodes"] == 3
    assert r["align_remediated"] == [2]
    assert "check_passed" not in r and "quarantined" not in r


def test_save_report_writes_json_to_the_sessions_reports_folder():
    """output/raw/{task}/{session} reports to output/reports/{task}/{session}."""
    from real_scripts.data_processing.session_health_check import (
        build_session_report,
        save_report,
    )

    name = "2026-04-02_umi35L_pouring"
    session = session_dir("raw", "pouring", name)
    r = build_session_report(**_make_report({"session_folder": session}))
    save_report(r, session, "2026-04-02")
    report_path = (
        output_root()
        / "reports"
        / "pouring"
        / name
        / "session_health_report_2026-04-02.json"
    )
    assert report_path.exists()
    loaded = json.loads(report_path.read_text())
    assert loaded["total_episodes"] == 3


def test_print_report_runs_without_error(capsys):
    from real_scripts.data_processing.session_health_check import (
        build_session_report,
        print_report,
    )

    r = build_session_report(**_make_report())
    print_report("2026-04-02", {"2026-04-02_test": r}, dry_run=False)
    out = capsys.readouterr().out
    assert "SESSION HEALTH CHECK" in out
    assert "2026-04-02_test" in out


# --- check_session integration tests ---


def _setup_session(*, episode_count=2, task_name="pouring", alignment_overrides=None):
    """Create a minimal session in the output folder, with config and episodes.

    The session lives at ``output/raw/{task}/{session}``; the returned aligned
    folder is the one ``check_session`` resolves for it on its own,
    ``output/aligned/{task}/{session}``. ``alignment_overrides`` is merged into
    the ``alignment`` block — use it to model a session recorded with live
    alignment OFF (``{"enabled": False}``).
    """
    session = Path(session_dir("raw", task_name, "2026-04-02_umi35L_pouring"))
    session.mkdir(parents=True)
    for i in range(episode_count):
        (session / f"episode_{i}").mkdir()
    aligned = output_root() / "aligned" / task_name / session.name
    aligned.mkdir(parents=True)
    config = {
        "meta": {"task_name": task_name},
        "alignment": {
            "index_sources": "right_umi_camera",
            "enabled": True,
            **(alignment_overrides or {}),
        },
    }
    _write_yaml(session / "config.yaml", config)
    return session, aligned


def _check_session_patches(
    aligned_dir,
    log_dir,
    *,
    alignment_valid=True,
):
    """Return a list of patch context managers for check_session.

    ``run_realignment`` is always LAST, so a test can replace it.

    Parameters
    ----------
    aligned_dir : path to the aligned directory
    log_dir : path used as _ALIGNMENT_LOG_DIR
    alignment_valid : if True, _validate_aligned_episode returns no problems
    """
    return [
        patch(
            "real_scripts.data_processing.align_data.resolve_index_source_from_config",
            return_value="right_umi_camera",
        ),
        patch(
            "real_scripts.data_processing.session_health_check._validate_aligned_episode",
            return_value=[] if alignment_valid else ["missing data"],
        ),
        patch(
            "real_scripts.data_processing.session_health_check._ALIGNMENT_LOG_DIR",
            str(log_dir),
        ),
        patch(
            "real_scripts.data_processing.session_health_check.run_realignment",
            return_value=([], {}),
        ),
    ]


def _apply_patches(patches):
    """Enter all patches and return a list of started mocks (for cleanup via additive)."""
    from contextlib import ExitStack

    stack = ExitStack()
    mocks = [stack.enter_context(p) for p in patches]
    return stack, mocks


def test_check_session_remediation_writes_source(tmp_path):
    """When check_session re-aligns, log entries have source='health_check'."""
    from real_scripts.data_processing.session_health_check import check_session

    session, aligned = _setup_session()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    session_name = session.name

    patches = _check_session_patches(aligned, log_dir, alignment_valid=False)
    # Override run_realignment to report success for all episodes
    patches[-1] = patch(
        "real_scripts.data_processing.session_health_check.run_realignment",
        return_value=([0, 1], {}),
    )

    stack, mocks = _apply_patches(patches)
    with stack:
        report = check_session(str(session), dry_run=False, date_str="2026-04-02")

    assert report["align_remediated"] == [0, 1]
    # Re-aligned into the session's own aligned folder.
    assert mocks[-1].call_args.args[1] == str(aligned)

    log_path = log_dir / "session_health_check_align_2026-04-02.json"
    assert log_path.exists()
    log_data = json.loads(log_path.read_text())
    for idx in [0, 1]:
        ep = log_data[session_name][f"episode_{idx}"]
        assert ep["source"] == "health_check"


def test_check_session_realigns_on_the_session_configs_grid(tmp_path):
    """alignment.uniform_fps reaches every re-aligned episode, so an episode the
    health check re-aligns lands on the same timeline as the rest of its
    session instead of silently falling back to legacy alignment."""
    from real_scripts.data_processing.session_health_check import check_session

    session, aligned = _setup_session(alignment_overrides={"uniform_fps": 30})
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    align_remote = MagicMock()

    patches = _check_session_patches(aligned, log_dir, alignment_valid=False)
    # The real run_realignment, with Ray replaced by the recorded remote calls.
    patches[-1] = patch(
        "real_scripts.data_processing.align_data._align_episode_remote", align_remote
    )
    patches += [
        patch("umi.utils.ray_utils.ray_session", lambda *a, **k: nullcontext()),
        patch("ray.get", return_value=[(0, True, None), (1, True, None)]),
    ]
    stack, _ = _apply_patches(patches)
    with stack:
        check_session(str(session), dry_run=False, date_str="2026-04-02")

    calls = align_remote.remote.call_args_list
    assert sorted(c.args[2] for c in calls) == [0, 1]
    assert all(c.args[-1] == 30 for c in calls)
    assert all(c.args[1] == str(aligned) for c in calls)


def test_check_session_aligns_when_live_alignment_was_off(tmp_path):
    """Live alignment off at record time sets alignment.enabled to false. The
    health check must still align what is missing, and flag the session as
    recorded without live alignment."""
    from real_scripts.data_processing.session_health_check import check_session

    session, aligned = _setup_session(alignment_overrides={"enabled": False})
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    patches = _check_session_patches(aligned, log_dir, alignment_valid=False)
    patches[-1] = patch(
        "real_scripts.data_processing.session_health_check.run_realignment",
        return_value=([0, 1], {}),
    )
    stack, mocks = _apply_patches(patches)
    with stack:
        report = check_session(str(session), dry_run=False, date_str="2026-04-02")

    run_realignment = mocks[-1]
    run_realignment.assert_called_once()
    assert run_realignment.call_args.args[2] == [0, 1]
    assert report["align_remediated"] == [0, 1]
    assert report["alignment_disabled_during_recording"] is True


def test_check_session_does_not_flag_when_live_alignment_was_on(tmp_path):
    from real_scripts.data_processing.session_health_check import check_session

    session, aligned = _setup_session()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    stack, _ = _apply_patches(_check_session_patches(aligned, log_dir))
    with stack:
        report = check_session(str(session), dry_run=False, date_str="2026-04-02")

    assert report["alignment_disabled_during_recording"] is False


def test_check_session_dry_run_with_live_alignment_off_aligns_nothing(tmp_path):
    """ "Check alignment" on the audit page is --dry-run; it must stay
    report-only even though the toggle does not gate the alignment."""
    from real_scripts.data_processing.session_health_check import check_session

    session, aligned = _setup_session(alignment_overrides={"enabled": False})
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    patches = _check_session_patches(aligned, log_dir, alignment_valid=False)
    stack, mocks = _apply_patches(patches)
    with stack:
        report = check_session(str(session), dry_run=True, date_str="2026-04-02")

    mocks[-1].assert_not_called()  # run_realignment
    assert report["alignment_disabled_during_recording"] is True


def test_print_report_notes_sessions_recorded_with_alignment_off(capsys):
    from real_scripts.data_processing.session_health_check import (
        build_session_report,
        print_report,
    )

    flagged = build_session_report(
        **_make_report(), alignment_disabled_during_recording=True
    )
    clean = build_session_report(**_make_report())
    print_report("2026-04-02", {"flagged": flagged, "clean": clean}, dry_run=False)
    out = capsys.readouterr().out
    assert out.count("live alignment was OFF") == 1


def test_check_session_second_run_is_noop(tmp_path):
    """After a fully successful first run, the second run triggers no remediation."""
    from real_scripts.data_processing.session_health_check import check_session

    session, aligned = _setup_session()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    # --- First run: alignment missing → remediate ---
    first_patches = _check_session_patches(aligned, log_dir, alignment_valid=False)
    first_patches[-1] = patch(
        "real_scripts.data_processing.session_health_check.run_realignment",
        return_value=([0, 1], {}),
    )

    stack, mocks = _apply_patches(first_patches)
    with stack:
        report1 = check_session(str(session), dry_run=False, date_str="2026-04-02")

    assert report1["align_remediated"] == [0, 1]
    assert report1["align_failed"] == {}

    # Simulate that realignment produced episode dirs in the aligned folder
    for i in range(2):
        (aligned / f"episode_{i}").mkdir(exist_ok=True)

    # --- Second run: alignment valid on disk ---
    second_patches = _check_session_patches(aligned, log_dir, alignment_valid=True)

    stack2, mocks2 = _apply_patches(second_patches)
    with stack2:
        report2 = check_session(str(session), dry_run=False, date_str="2026-04-02")

    assert report2["align_ok"] == [0, 1]
    assert report2["align_remediated"] == []
    assert report2["align_failed"] == {}

    # run_realignment should not have been called on the second run
    mocks2[-1].assert_not_called()


def test_find_sessions_accepts_the_uid_suffixed_layout_the_gui_writes(tmp_path):
    """Run health check finds sessions whose folder ends in a session uid.

    Session folders carry a 12-hex session uid last --
    `{date}_umi..L_umi..R_{task}_{uid}` (station_ui/core/session_identity.py:
    build_session_dir_name) -- so matching on `basename.endswith("_" +
    task_name)` would reject every session the console records, while the audit
    page, which lists on the date prefix alone, shows them all.
    """
    from real_scripts.data_processing.session_health_check import (
        find_sessions_across_roots,
    )

    session = "2026-04-02_umiV3_0001L_umiV3_0002R_plane_assembly_v10_a1b2c3d4e5f6"
    _make_dirs(tmp_path, [session, f"{session}_aligned", f"{session}_quarantine"])
    for name in (session, f"{session}_aligned", f"{session}_quarantine"):
        _write_yaml(
            os.path.join(tmp_path, name, "config.yaml"),
            {"meta": {"task_name": "plane_assembly_v10"}},
        )

    names = [
        os.path.basename(r)
        for r in find_sessions_across_roots([str(tmp_path)], "2026-04-02")
    ]

    assert session in names, "the uid-suffixed session the GUI writes was skipped"
    # The uid must not smuggle the derived dirs back in.
    assert f"{session}_aligned" not in names
    assert f"{session}_quarantine" not in names


def test_find_sessions_accepts_the_legacy_counter_the_old_resolver_appended(tmp_path):
    """Sessions written with an extra `_N` suffix must still be reachable.

    Older versions appended an incrementing `_N` to a folder that already ended
    in a uid when a config drifted mid-session -- `…_{task}_{uid}_1`. Those
    folders hold real episodes, and the audit page (which lists on the date
    prefix alone) shows them, so the health check has to cover them too.
    """
    from real_scripts.data_processing.session_health_check import (
        find_sessions_across_roots,
    )

    base = "2026-04-02_umiV3_0001L_umiV3_0002R_plane_assembly_v10_a1b2c3d4e5f6"
    legacy = f"{base}_1"
    _make_dirs(tmp_path, [base, legacy, f"{legacy}_aligned", f"{legacy}_quarantine"])
    for name in (base, legacy, f"{legacy}_aligned", f"{legacy}_quarantine"):
        _write_yaml(
            os.path.join(tmp_path, name, "config.yaml"),
            {"meta": {"task_name": "plane_assembly_v10"}},
        )

    names = [
        os.path.basename(r)
        for r in find_sessions_across_roots([str(tmp_path)], "2026-04-02")
    ]

    assert legacy in names, "the legacy _N-suffixed session was skipped"
    assert base in names, "the rolled session it was bumped from was skipped"
    # Peeling two suffixes must not smuggle the derived dirs back in.
    assert f"{legacy}_aligned" not in names
    assert f"{legacy}_quarantine" not in names


@pytest.mark.parametrize(
    "basename, expected",
    [
        # Today's layout: task then the 12-hex session uid.
        ("2026-04-02_umiV3_0001L_taskA_a1b2c3d4e5f6", True),
        # Accept session names without a UID.
        ("2026-04-02_umi35L_taskA", True),
        # A hand-added run number, with and without a uid.
        ("2026-04-02_umi35L_taskA_1", True),
        ("2026-04-02_umi35L_taskA_2", True),
        ("2026-04-02_umiV3_0001L_taskA_2_a1b2c3d4e5f6", True),
        # Parse a UID and collision suffix together so resumed sessions remain
        # discoverable by both health checks and the audit page.
        ("2026-04-02_umiV3_0001L_taskA_a1b2c3d4e5f6_1", True),
        ("2026-04-02_umiV3_0001L_taskA_a1b2c3d4e5f6_12", True),
        # Two run numbers is not a layout anything ever wrote -- the `_N` suffix
        # counted on the stem (`…_<uid>_1` then `…_<uid>_2`), so at most one.
        ("2026-04-02_umiV3_0001L_taskA_a1b2c3d4e5f6_1_2", False),
        # Exclude derived outputs even when the source session has no UID.
        ("2026-04-02_umiV3_0001L_taskA_a1b2c3d4e5f6_1_aligned", False),
        # Derived dirs carry the same meta.task_name and must stay out, in every
        # layout -- only a real uid or a run number is ever stripped.
        ("2026-04-02_umi35L_taskA_aligned", False),
        ("2026-04-02_umi35L_taskA_quarantine", False),
        ("2026-04-02_umiV3_0001L_taskA_a1b2c3d4e5f6_aligned", False),
        ("2026-04-02_umiV3_0001L_taskA_a1b2c3d4e5f6_quarantine", False),
        # Not this task, and not a uid/run suffix either.
        ("2026-04-02_umi35L_taskA_extra_suffix", False),
        ("2026-04-02_umi35L_taskB", False),
        # A uid is exactly 12 lowercase hex: neither of these may be stripped.
        ("2026-04-02_umi35L_taskA_A1B2C3D4E5F6", False),
        ("2026-04-02_umi35L_taskA_a1b2c3d4e5", False),
    ],
)
def test_basename_matches_config_task_across_every_session_layout(basename, expected):
    """One table for the four layouts that are real, and the ones that are not."""
    from real_scripts.data_processing.session_health_check import (
        basename_matches_config_task,
    )

    assert basename_matches_config_task(basename, "taskA") is expected


def test_a_blank_task_name_matches_nothing():
    from real_scripts.data_processing.session_health_check import (
        basename_matches_config_task,
    )

    assert basename_matches_config_task("2026-04-02_umi35L_taskA", "") is False


# --- quarantine: never looked at ---


def test_check_session_aligns_only_the_session_folder(tmp_path):
    """Quarantined episodes failed the record-time check and are never aligned,
    so the health check does not look at output/quarantine at all."""
    from real_scripts.data_processing.session_health_check import check_session

    session, aligned = _setup_session()
    quarantine = Path(counterpart(str(session), "quarantine"))
    for idx in (0, 1, 2):
        (quarantine / f"episode_{idx}").mkdir(parents=True)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    patches = _check_session_patches(aligned, log_dir, alignment_valid=False)
    stack, mocks = _apply_patches(patches)
    with stack:
        report = check_session(str(session), dry_run=False, date_str="2026-04-02")

    assert mocks[-1].call_args.args[2] == [0, 1]  # run_realignment
    assert report["total_episodes"] == 2


def test_check_session_with_no_episodes_aligns_nothing(tmp_path):
    from real_scripts.data_processing.session_health_check import check_session

    session, _aligned = _setup_session(episode_count=0)

    with patch(
        "real_scripts.data_processing.session_health_check.run_realignment"
    ) as run_realignment:
        report = check_session(str(session), dry_run=False, date_str="2026-04-02")

    run_realignment.assert_not_called()
    assert report["total_episodes"] == 0


# --- main(): --date/--end-date range scan ---


def _empty_report(session_folder):
    return {
        "session_folder": session_folder,
        "total_episodes": 0,
        "alignment_disabled_during_recording": False,
        "align_ok": [],
        "align_remediated": [],
        "align_failed": {},
    }


def test_main_scans_every_day_from_date_through_end_date(monkeypatch, tmp_path):
    import real_scripts.data_processing.session_health_check as shc

    calls = []

    def fake_find(scan_roots, date_str, *, task_name=None):
        return [str(tmp_path / f"{date_str}_session")]

    def fake_check_session(session_folder, dry_run, date_str):
        calls.append((os.path.basename(session_folder), date_str))
        return _empty_report(session_folder)

    monkeypatch.setattr(shc, "resolve_recordings_root", lambda explicit: str(tmp_path))
    monkeypatch.setattr(shc, "recordings_scan_roots", lambda primary: [str(tmp_path)])
    monkeypatch.setattr(shc, "find_sessions_across_roots", fake_find)
    monkeypatch.setattr(shc, "check_session", fake_check_session)

    shc.main(shc.Args(date="2026-08-04", end_date="2026-08-06", dry_run=True))

    assert calls == [
        ("2026-08-04_session", "2026-08-04"),
        ("2026-08-05_session", "2026-08-05"),
        ("2026-08-06_session", "2026-08-06"),
    ]


def test_main_defaults_end_date_to_date_for_a_single_day_scan(monkeypatch, tmp_path):
    import real_scripts.data_processing.session_health_check as shc

    calls = []

    def fake_find(scan_roots, date_str, *, task_name=None):
        return [str(tmp_path / f"{date_str}_session")]

    def fake_check_session(session_folder, dry_run, date_str):
        calls.append((os.path.basename(session_folder), date_str))
        return _empty_report(session_folder)

    monkeypatch.setattr(shc, "resolve_recordings_root", lambda explicit: str(tmp_path))
    monkeypatch.setattr(shc, "recordings_scan_roots", lambda primary: [str(tmp_path)])
    monkeypatch.setattr(shc, "find_sessions_across_roots", fake_find)
    monkeypatch.setattr(shc, "check_session", fake_check_session)

    shc.main(shc.Args(date="2026-08-04", dry_run=True))

    assert calls == [("2026-08-04_session", "2026-08-04")]


def test_main_rejects_an_end_date_before_the_start_date(monkeypatch, tmp_path):
    import real_scripts.data_processing.session_health_check as shc

    monkeypatch.setattr(shc, "resolve_recordings_root", lambda explicit: str(tmp_path))
    monkeypatch.setattr(shc, "recordings_scan_roots", lambda primary: [str(tmp_path)])

    with pytest.raises(SystemExit):
        shc.main(shc.Args(date="2026-08-06", end_date="2026-08-04", dry_run=True))

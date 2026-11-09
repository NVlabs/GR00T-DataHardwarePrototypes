# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The health check surfaces per-session exceptions.

A per-session `except Exception` that only printed a traceback and carried on
would let a run in which every session failed still exit 0, and any wrapper
(the station web UI's /audit page, a CI job) would see a success.

These tests pin the contract: an unhandled exception makes the process exit
non-zero and name the sessions, while an episode that could not be aligned --
the normal, expected output of an audit -- still exits 0.
"""

import json

import pytest

from real_scripts.data_processing import session_health_check
from umi.common.utility.output_layout import output_root, session_dir

CLEAN_REPORT = {
    "total_episodes": 1,
    "align_ok": [0],
    "align_remediated": [],
    "align_failed": {},
}


def _session(tmp_path, name="2026-08-04_umi1R_taskA_aaaaaaaaaaaa"):
    path = tmp_path / name
    path.mkdir()
    (path / "config.yaml").write_text("meta:\n  task_name: taskA\n")
    return path


# --------------------------------------------------------- session_health_check


def test_session_health_check_exits_nonzero_when_a_session_raises(
    tmp_path, monkeypatch, capsys
):
    session = _session(tmp_path)
    monkeypatch.setattr(
        session_health_check,
        "check_session",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("alignment log corrupt")),
    )

    with pytest.raises(SystemExit) as exit_info:
        session_health_check.main(
            session_health_check.Args(session_folder=str(session), dry_run=True)
        )

    assert exit_info.value.code != 0
    printed = capsys.readouterr().out
    assert "Traceback" in printed
    assert "RuntimeError: alignment log corrupt" in printed
    assert session.name in printed


def test_session_health_check_exits_zero_when_every_session_is_processed(
    tmp_path, monkeypatch, capsys
):
    """Guards the other direction: a blanket non-zero exit would make every
    audit look broken."""
    session = _session(tmp_path)
    monkeypatch.setattr(
        session_health_check, "check_session", lambda *a, **k: dict(CLEAN_REPORT)
    )

    session_health_check.main(
        session_health_check.Args(session_folder=str(session), dry_run=True)
    )

    assert "Traceback" not in capsys.readouterr().out


def test_a_failed_alignment_is_not_treated_as_a_failed_run(
    tmp_path, monkeypatch, capsys
):
    """Episodes that cannot be aligned are what an audit is for. Only an
    unhandled exception means the audit itself did not run."""
    session = _session(tmp_path)
    report = dict(
        CLEAN_REPORT, align_ok=[], align_failed={"episode_0": "no index source"}
    )
    monkeypatch.setattr(session_health_check, "check_session", lambda *a, **k: report)

    session_health_check.main(
        session_health_check.Args(session_folder=str(session), dry_run=True)
    )

    assert "no index source" in capsys.readouterr().out


def test_session_health_check_names_every_failed_session_in_its_summary(
    tmp_path, monkeypatch, capsys
):
    good = _session(tmp_path, "2026-08-04_umi1R_taskA_aaaaaaaaaaaa")
    bad = _session(tmp_path, "2026-08-04_umi1R_taskB_bbbbbbbbbbbb")

    def check(session_folder, **_):
        if session_folder.rstrip("/").endswith(bad.name):
            raise ValueError("zarr.json unreadable")
        return dict(CLEAN_REPORT)

    monkeypatch.setattr(session_health_check, "check_session", check)
    monkeypatch.setattr(
        session_health_check,
        "find_sessions_across_roots",
        lambda *a, **k: [str(good), str(bad)],
    )

    with pytest.raises(SystemExit):
        session_health_check.main(
            session_health_check.Args(
                data_dir=str(tmp_path), date="2026-08-04", dry_run=True
            )
        )

    # The run's last word says which sessions to look at -- naming all of them,
    # or none, would leave the operator scrolling a multi-session log.
    printed = capsys.readouterr().out
    summary = [line for line in printed.splitlines() if line.strip()][-1]
    assert bad.name in summary, summary
    assert good.name not in summary, summary


def test_session_health_check_reports_no_sessions_without_failing(tmp_path, capsys):
    """An empty scan is an ordinary answer, not an error."""
    session_health_check.main(
        session_health_check.Args(
            data_dir=str(tmp_path), date="2026-08-04", dry_run=True
        )
    )

    assert "No raw recording sessions found" in capsys.readouterr().out


def test_a_saved_health_report_is_valid_json():
    """save_report writes to the session's folder under output/reports."""
    name = "2026-08-04_umi1R_taskA_aaaaaaaaaaaa"
    session = session_dir("raw", "taskA", name)
    session_health_check.save_report(dict(CLEAN_REPORT), session, "2026-08-04")

    saved = (
        output_root()
        / "reports"
        / "taskA"
        / name
        / "session_health_report_2026-08-04.json"
    )
    assert json.loads(saved.read_text())["total_episodes"] == 1

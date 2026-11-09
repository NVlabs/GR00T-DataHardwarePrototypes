# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Session Health Check

Audits recording sessions on this station's disk for alignment completeness.
Every episode in a raw session passed the record-time data check; this
confirms each one has a valid aligned copy, and re-aligns what is missing or
broken (unless --dry-run is given). Alignment runs in the background while
recording, so a recorder or station stopped mid-queue leaves gaps that only
this finds.

Everything stays local. Sessions are read from
``output/raw/{task}/{session}/`` and the aligned copy goes to
``output/aligned/{task}/{session}/`` (see
``umi.common.utility.output_layout``). The report is written to
``output/reports/{task}/{session}/session_health_report_<date>.json``.

Usage:
    # Raw recording sessions from today (folder basename must end with _{meta.task_name})
    uv run real_scripts/data_processing/session_health_check.py

    # Only sessions whose config.yaml meta.task_name matches
    uv run real_scripts/data_processing/session_health_check.py --task-name <task_name>

    # Specific session
    uv run real_scripts/data_processing/session_health_check.py --session-folder /path/to/session

    # Specific date, dry run
    uv run real_scripts/data_processing/session_health_check.py --date 2026-04-01 --dry-run

Exit code:
    0  every session was audited (episodes that could not be aligned are the
       normal output of an audit and do not change this)
    1  at least one session raised. The cause is printed and named in the
       closing summary, so a UI wrapper (station_ui's /audit page) reports
       a failed job instead of a silent success.
"""

from __future__ import annotations

import json
import os
import re
import traceback
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Annotated, Optional

import tyro
import yaml

from umi.common.recording.session_uid import basename_matches_config_task
from umi.common.utility.alignment_settings import alignment_settings
from umi.common.utility.background_aligner import (
    _update_status,
    _validate_aligned_episode,
)
from umi.common.utility.output_layout import counterpart, output_root
from umi.common.utility.print_utils import Colors, print_blue

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CONFIG_PATH = os.path.join(
    _REPO_ROOT, "config", "data_recording", "config.yaml"
)
EPISODE_RE = re.compile(r"^episode_(\d+)$")


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------


def resolve_recordings_root(explicit: Optional[str]) -> str:
    """Root folder holding raw sessions: ``--data-dir``, else ``output/raw``."""
    if explicit:
        return os.path.expanduser(explicit)
    return str(output_root() / "raw")


def recordings_scan_roots(primary: str) -> list[str]:
    """Dirs to search for session folders: the root and each task folder in it.

    Sessions are recorded to ``output/raw/{task}/{session}``, so the folders
    one level down hold them. The root itself is scanned too, for a
    ``--data-dir`` that points straight at a folder of sessions.
    """
    root = os.path.abspath(os.path.expanduser(primary))
    if not os.path.isdir(root):
        return []
    roots = [root]
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return roots
    for entry in entries:
        path = os.path.join(root, entry)
        if os.path.isdir(path):
            roots.append(path)
    return roots


def read_meta_task_name(session_folder: str) -> Optional[str]:
    """Return ``meta.task_name`` from ``config.yaml`` if present; else None."""
    cfg_path = os.path.join(session_folder, "config.yaml")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        raw = (cfg.get("meta") or {}).get("task_name")
        if raw is None:
            return None
        s = str(raw).strip()
        return s if s else None
    except (OSError, yaml.YAMLError):
        return None


def find_sessions_across_roots(
    scan_roots: list[str],
    date_str: str,
    *,
    task_name: Optional[str] = None,
) -> list[str]:
    """Unique session paths for ``date_str`` prefix under any root.

    Keeps only directories that have ``config.yaml`` with a non-empty
    ``meta.task_name``, and whose folder name ends with ``_<that task name>``
    so siblings with appended suffixes (aligned export, quarantine, etc.) are
    excluded without listing those suffixes.

    When ``task_name`` is set, ``meta.task_name`` must also equal that filter.
    """
    want_task = task_name.strip() if task_name else None
    found: list[str] = []
    seen_abs: set[str] = set()

    for data_dir in scan_roots:
        data_dir = os.path.abspath(os.path.expanduser(data_dir))
        if not os.path.isdir(data_dir):
            continue
        try:
            entries = os.listdir(data_dir)
        except OSError:
            continue
        for entry in entries:
            if not entry.startswith(date_str):
                continue
            path = os.path.join(data_dir, entry)
            if not os.path.isdir(path):
                continue
            ap = os.path.abspath(path)
            if ap in seen_abs:
                continue
            tn = read_meta_task_name(path)
            if tn is None:
                continue
            if not basename_matches_config_task(entry, tn):
                continue
            if want_task is not None and tn != want_task:
                continue
            seen_abs.add(ap)
            found.append(ap)

    return sorted(found)


# ---------------------------------------------------------------------------
# Per-session helpers
# ---------------------------------------------------------------------------


def load_session_config(session_folder: str) -> dict:
    """Load config.yaml from the session folder (saved there by record_data.py)."""
    config_path = os.path.join(session_folder, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml not found in {session_folder}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def find_episodes(session_folder: str) -> list[int]:
    """Return sorted episode indices found as episode_N/ dirs on disk."""
    if not os.path.isdir(session_folder):
        return []
    indices = []
    for entry in os.listdir(session_folder):
        m = EPISODE_RE.match(entry)
        if m and os.path.isdir(os.path.join(session_folder, entry)):
            indices.append(int(m.group(1)))
    return sorted(indices)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def check_alignment_status(
    aligned_dir: str,
    episode_indices: list[int],
    index_source: str,
) -> tuple[list[int], list[int]]:
    """Check which episodes have valid alignment on disk. Returns (ok, needs_align)."""
    ok: list[int] = []
    needs_align: list[int] = []
    for idx in episode_indices:
        ep_dir = os.path.join(aligned_dir, f"episode_{idx}")
        if not os.path.isdir(ep_dir):
            needs_align.append(idx)
            continue
        problems = _validate_aligned_episode(aligned_dir, idx, index_source)
        if problems:
            needs_align.append(idx)
        else:
            ok.append(idx)
    return ok, needs_align


_ALIGNMENT_LOG_DIR = os.path.join(os.path.expanduser("~"), ".umi", "alignment_logs")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_session_report(
    session_folder: str,
    episode_indices: list[int],
    align_ok: list[int],
    align_remediated: list[int],
    align_failed: dict[int, str],
    *,
    alignment_disabled_during_recording: bool = False,
) -> dict:
    return {
        "session_folder": session_folder,
        "total_episodes": len(episode_indices),
        "alignment_disabled_during_recording": alignment_disabled_during_recording,
        "align_ok": align_ok,
        "align_remediated": align_remediated,
        "align_failed": {f"episode_{k}": v for k, v in align_failed.items()},
    }


def report_dir(session_folder: str) -> str:
    """Where a session's reports go: ``output/reports/{task}/{session}``."""
    return counterpart(session_folder, "reports")


def save_report(report: dict, session_folder: str, date_str: str) -> None:
    directory = report_dir(session_folder)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"session_health_report_{date_str}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {path}")


def print_report(
    date_str: str, session_reports: dict[str, dict], dry_run: bool
) -> None:
    tag = "  [DRY RUN]" if dry_run else ""
    sep = "═" * 56
    print(f"\n{Colors.BOLD}{sep}{Colors.RESET}")
    print(f"{Colors.BOLD}SESSION HEALTH CHECK — {date_str}{tag}{Colors.RESET}")
    print(f"{Colors.BOLD}{sep}{Colors.RESET}\n")

    total_episodes = total_align_fail = 0
    for session_name, r in session_reports.items():
        n = r["total_episodes"]
        total_episodes += n
        total_align_fail += len(r["align_failed"])

        n_align_remed = len(r["align_remediated"])
        align_str = f"{len(r['align_ok'])} ok" + (
            f"  ({n_align_remed} re-aligned)" if n_align_remed else ""
        )

        print(f"Session: {Colors.BOLD}{session_name}{Colors.RESET}  ({n} episodes)")
        if r.get("alignment_disabled_during_recording"):
            print(
                f"  {Colors.YELLOW}Note: live alignment was OFF during recording; "
                f"aligned backfilled here.{Colors.RESET}"
            )
        print(f"  Alignment:   {align_str},  {len(r['align_failed'])} failed")

        if r["align_failed"]:
            print(f"\n  {Colors.RED}Alignment failures:{Colors.RESET}")
            for ep_key, err in r["align_failed"].items():
                print(f"    {ep_key}  →  {err}")
        print()

    print(f"{Colors.BOLD}{sep}{Colors.RESET}")
    print(
        f"{Colors.BOLD}TOTAL: {len(session_reports)} session(s) | {total_episodes} episodes | "
        f"{total_align_fail} alignment failures"
        f"{Colors.RESET}"
    )
    print(f"{Colors.BOLD}{sep}{Colors.RESET}\n")


# ---------------------------------------------------------------------------
# Remediation
# ---------------------------------------------------------------------------


def run_realignment(
    session_folder: str,
    aligned_dir: str,
    episode_indices: list[int],
    index_source: str,
    copy_videos: bool,
    uniform_fps: float,
) -> tuple[list[int], dict[int, str]]:
    """Re-align episodes using Ray. Returns (succeeded, {failed_idx: error})."""
    import ray

    from real_scripts.data_processing.align_data import (
        _align_episode_remote,
        _write_zarr_json_index_source,
    )
    from umi.utils.ray_utils import ray_session

    os.makedirs(aligned_dir, exist_ok=True)
    _write_zarr_json_index_source(aligned_dir, index_source)

    project_root = _REPO_ROOT
    existing = os.environ.get("PYTHONPATH", "")
    extra_env = {
        "env_vars": {"PYTHONPATH": project_root + (f":{existing}" if existing else "")}
    }

    abs_session = os.path.abspath(session_folder)
    abs_aligned = os.path.abspath(aligned_dir)

    # Read RAY_NUM_WORKERS, falling back to NUM_WORKERS; -1 uses available CPUs.
    num_workers_str = os.environ.get(
        "RAY_NUM_WORKERS", os.environ.get("NUM_WORKERS", "-1")
    )
    try:
        num_workers = int(num_workers_str)
    except ValueError:
        num_workers = -1

    with ray_session(num_workers, extra_env=extra_env):
        futures = [
            _align_episode_remote.remote(
                abs_session,
                abs_aligned,
                idx,
                index_source,
                copy_videos,
                uniform_fps,
            )
            for idx in episode_indices
        ]
        results = ray.get(futures)

    succeeded: list[int] = []
    failed: dict[int, str] = {}
    for idx, ok, err in results:
        if ok:
            problems = _validate_aligned_episode(abs_aligned, idx, index_source)
            if problems:
                failed[idx] = "; ".join(problems)
            else:
                succeeded.append(idx)
        else:
            failed[idx] = err or "unknown alignment error"
    return succeeded, failed


# ---------------------------------------------------------------------------
# Per-session orchestration
# ---------------------------------------------------------------------------


def _log_realignment(
    session_name: str,
    date_str: str,
    remediated: list[int],
    failed: dict[int, str],
) -> None:
    """Record re-alignment results in the local alignment log."""
    try:
        os.makedirs(_ALIGNMENT_LOG_DIR, exist_ok=True)
        log_path = os.path.join(
            _ALIGNMENT_LOG_DIR, f"session_health_check_align_{date_str}.json"
        )
        for idx in remediated:
            _update_status(
                log_path,
                session_name,
                f"episode_{idx}",
                align_status="success",
                source="health_check",
            )
        for idx, err in failed.items():
            _update_status(
                log_path,
                session_name,
                f"episode_{idx}",
                align_status="failure",
                align_error=err,
                source="health_check",
            )
    except Exception:
        print(f"  {Colors.YELLOW}Warning: failed to update alignment log{Colors.RESET}")


def check_session(session_folder: str, dry_run: bool, date_str: str) -> dict:
    """Run the full health check pipeline for one session."""
    from real_scripts.data_processing.align_data import resolve_index_source_from_config

    session_name = os.path.basename(session_folder.rstrip(os.sep))
    print(f"\n{Colors.BLUE}── {session_name} ──{Colors.RESET}")

    cfg = load_session_config(session_folder)
    alignment_cfg = cfg.get("alignment", {})
    settings = alignment_settings(cfg)
    # Report whether live alignment was enabled, but always repair missing
    # alignments during this health check.
    alignment_disabled_during_recording = not alignment_cfg.get("enabled", True)
    aligned_dir = alignment_cfg.get("aligned_dir") or counterpart(
        session_folder, "aligned"
    )

    # Inspect finalized episodes only. Record-time failures are stored separately
    # in quarantine and must not enter alignment.
    episode_indices = find_episodes(session_folder)
    print_blue(f"Episodes found: {len(episode_indices)}")

    report_fields = dict(
        alignment_disabled_during_recording=alignment_disabled_during_recording,
    )

    if not episode_indices:
        return build_session_report(session_folder, [], [], [], {}, **report_fields)

    try:
        index_source = resolve_index_source_from_config(
            session_folder, list(settings.index_sources)
        )
        print_blue(f"Using index source: {index_source}")
    except ValueError as exc:
        print(f"  {Colors.RED}Cannot resolve index source: {exc}{Colors.RESET}")
        err = str(exc)
        return build_session_report(
            session_folder,
            episode_indices,
            [],
            [],
            {i: err for i in episode_indices},
            **report_fields,
        )

    align_already_ok, needs_align = check_alignment_status(
        aligned_dir, episode_indices, index_source
    )
    print_blue(
        f"Alignment: {len(align_already_ok)} ok, {len(needs_align)} need alignment"
    )
    align_remediated: list[int] = []
    align_failed: dict[int, str] = {}
    if needs_align:
        if dry_run:
            print(f"  [DRY RUN] Would re-align: {needs_align}")
        else:
            print_blue(f"Re-aligning {len(needs_align)} episode(s)...")
            align_remediated, align_failed = run_realignment(
                session_folder,
                aligned_dir,
                needs_align,
                index_source,
                settings.copy_videos,
                settings.uniform_fps,
            )
            _log_realignment(session_name, date_str, align_remediated, align_failed)
    align_ok = sorted(align_already_ok + align_remediated)

    return build_session_report(
        session_folder,
        episode_indices,
        align_ok,
        align_remediated,
        align_failed,
        **report_fields,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class Args:
    """Audit sessions for alignment completeness, and re-align what is missing."""

    session_folder: Annotated[
        Optional[str],
        tyro.conf.arg(aliases=["-s"]),
    ] = None
    """Direct path to a session. If given, --date is ignored."""

    date: Annotated[
        Optional[str],
        tyro.conf.arg(aliases=["-d"]),
    ] = None
    """Date (YYYY-MM-DD) to scan for sessions. Defaults to today."""

    end_date: Annotated[
        Optional[str],
        tyro.conf.arg(aliases=["-e"]),
    ] = None
    """End date (YYYY-MM-DD), inclusive, for scanning every day from --date
    through --end-date. Defaults to --date (single-day scan). Ignored when
    --session-folder is given."""

    task_name: Annotated[
        Optional[str],
        tyro.conf.arg(aliases=["-t"]),
    ] = None
    """When scanning by date, only include sessions whose ``config.yaml``
    ``meta.task_name`` equals this string. Date scans always require the folder
    basename to end with ``_<meta.task_name>`` so derived dirs with extra
    suffixes are skipped."""

    data_dir: Optional[str] = None
    """Raw sessions root for date scans (default ``output/raw``). Its task
    folders are scanned too."""

    dry_run: bool = False
    """Report only — no re-alignment, and no report file is written."""


def _date_range(start_str: str, end_str: str) -> list[str]:
    """Every ISO date from ``start_str`` through ``end_str``, inclusive."""
    day = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)
    out = []
    while day <= end:
        out.append(day.isoformat())
        day += timedelta(days=1)
    return out


def main(args: Optional[Args] = None) -> None:
    if args is None:
        args = tyro.cli(Args)

    start_str = args.date if args.date is not None else date.today().isoformat()
    end_str = args.end_date if args.end_date is not None else start_str

    if args.session_folder:
        sessions = [os.path.abspath(args.session_folder)]
        session_dates = {sessions[0]: start_str}
    else:
        if date.fromisoformat(end_str) < date.fromisoformat(start_str):
            print(
                f"{Colors.RED}--end-date {end_str} is before --date {start_str}{Colors.RESET}"
            )
            raise SystemExit(1)

        primary = resolve_recordings_root(args.data_dir)
        roots = recordings_scan_roots(primary)
        sessions = []
        session_dates = {}
        for day_str in _date_range(start_str, end_str):
            for sf in find_sessions_across_roots(
                roots, day_str, task_name=args.task_name
            ):
                sessions.append(sf)
                session_dates[sf] = day_str

        if not sessions:
            roots_txt = "\n  ".join(roots or [primary])
            range_txt = start_str if start_str == end_str else f"{start_str}..{end_str}"
            msg = (
                f"No raw recording sessions found for date range {range_txt!r} "
                f"(basename must end with _<meta.task_name> from config.yaml)."
            )
            if args.task_name:
                msg += f"\nTask filter: meta.task_name == {args.task_name!r}."
            msg += f"\nSearched:\n  {roots_txt}"
            print(f"{Colors.YELLOW}{msg}{Colors.RESET}")
            return

    session_reports: dict[str, dict] = {}
    crashed: list[str] = []
    for sf in sessions:
        sname = os.path.basename(sf.rstrip(os.sep))
        day_str = session_dates[sf]
        try:
            report = check_session(sf, dry_run=args.dry_run, date_str=day_str)
            session_reports[sname] = report
            if not args.dry_run:
                save_report(report, sf, day_str)
        except Exception:
            crashed.append(sname)
            print(f"\n{Colors.RED}ERROR processing {sname}:{Colors.RESET}")
            print(traceback.format_exc())

    range_label = start_str if start_str == end_str else f"{start_str} to {end_str}"
    print_report(range_label, session_reports, args.dry_run)

    if crashed:
        # Return failure status when session processing fails so callers do not
        # interpret an error-only report as success.
        print(
            f"{Colors.RED}FAILED: {len(crashed)} of {len(sessions)} session(s) "
            f"raised: {', '.join(crashed)}{Colors.RESET}"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()

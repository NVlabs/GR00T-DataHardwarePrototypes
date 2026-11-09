# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`/audit` -- each session's alignment and warnings, and the alignment check.

Two questions, both about sessions already on disk. Does every episode have an
aligned copy? Alignment runs in the background while recording, so a recorder
or station stopped mid-queue leaves gaps, and training needs that copy. And
did the data check leave warnings on it? A measured camera offset is a warning,
not a failure, because its limit is not validated on every rig -- the session
listing is where "every episode is 80 ms off" becomes visible.

The listing reads the disk. The one script, `session_health_check.py`, is a
`Job`: its merged output streams to the page, its exit code decides whether the
run reads as succeeded or failed, and a wall-clock cap means it cannot silently
pin the station forever. "Check alignment" runs it with `--dry-run` and changes
nothing; "Re-align missing" is the separate, deliberate button that writes.

"Convert to LeRobot" runs the LeRobot converter on one session's aligned copy,
as a job of the same kind, writing to `output/lerobot_format`.

"Finish and close session" ends the session the console is armed for: it seals
the folder and moves the console onto a fresh session, the same sequence as
Configuration's "Start a new session" (`next_session.start_next_session`). It
is here so finishing a session does not mean configuring the next one first.
"""

from __future__ import annotations

import datetime
import logging
import pathlib
import threading
from urllib.parse import urlencode

import yaml
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse

from station_ui import audit, jobs, next_session, readiness, recording, tokens
from station_ui import session as session_state
from station_ui.errors import FieldError, StaleOwner, UnknownJob
from station_ui.jobs import JobRegistry
from station_ui.rig import claim, is_owner, require_owner
from station_ui.workflow import render_page

logger = logging.getLogger(__name__)

router = APIRouter()

SEE_OTHER = 303
CONFLICT = 409
NOT_FOUND = 404

# One registry per app, created on first use rather than in `create_app`, so
# this page owns its wiring and merges with the sibling pages that add theirs.
_REGISTRY_LOCK = threading.Lock()

# Check audit ownership on every job route. An audit cancel must never
# terminate a recording that shares the registry.
KIND = "audit"


def registry_of(app) -> JobRegistry:
    existing = getattr(app.state, "jobs", None)
    if existing is not None:
        return existing
    with _REGISTRY_LOCK:
        existing = getattr(app.state, "jobs", None)
        if existing is None:
            existing = JobRegistry()
            app.state.jobs = existing
        return existing


def _recent_jobs(app) -> list:
    registry = getattr(app.state, "jobs", None)
    if registry is None:
        return []
    ours = [job for job in registry.all() if getattr(job, "kind", None) == KIND]
    return list(reversed(ours))[:10]


def _active_folder(app) -> str:
    """The folder the console is armed for -- the file Apply wrote, not the form.

    `routes/config.py` and `day_roll.py` both rely on this: the station records
    from this file, and `preflight` pulls `recording.data_dir` out of it at
    launch. Apply is its only writer, so the draft and the armed folder agree
    right up until somebody edits the form -- and an operator who applies,
    records, then starts typing the next task name has moved one and not the
    other. Keyed on the draft, the live session would lose its "Recording"
    badge, and its unaligned episodes -- still in the background queue -- would
    read as gaps.

    The draft is the fallback. `data_dir` raises `FieldError` for an invalid uid
    or a malformed date, both of which an operator can type, so the last resort
    still has to be a folder name wherever one can be had.
    """
    try:
        path = readiness.generated_config(app)
        if path is not None and path.is_file():
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            armed = str((cfg.get("recording") or {}).get("data_dir") or "")
            if armed:
                return armed
    except (OSError, ValueError, AttributeError, yaml.YAMLError):
        # Fall back to the visible session draft when generated config is unreadable;
        # do not report the station as inactive.
        pass
    try:
        return session_state.data_dir(app.state.rig.draft)
    except (FieldError, AttributeError):
        return ""


def _page(
    request: Request,
    token,
    *,
    status_code: int = 200,
    error: str = "",
    confirm: dict | None = None,
    notice: str = "",
) -> HTMLResponse:
    return render_page(
        request,
        "audit",
        _page_context(
            request,
            token,
            status_code=status_code,
            error=error,
            confirm=confirm,
            notice=notice,
        ),
        token=token,
        status_code=status_code,
    )


def _page_context(
    request: Request,
    token,
    *,
    status_code: int = 200,
    error: str = "",
    confirm: dict | None = None,
    notice: str = "",
) -> dict:
    rig = request.app.state.rig
    root = audit.recordings_root()
    return {
        "page_title": "Data Audit",
        "active": "/audit",
        "sessions": audit.list_sessions(
            audit.scan_roots(root), active=_active_folder(request.app)
        ),
        "today": datetime.date.today().isoformat(),
        "jobs": _recent_jobs(request.app),
        # Do not offer conversion while the session's recorder or aligner can still add
        # episodes.
        "recording_live": recording.in_progress(request.app),
        "can_edit": is_owner(rig, token),
        "error": error,
        "confirm": confirm,
        "notice": notice,
    }


@router.get("/audit", response_class=HTMLResponse)
async def audit_page(request: Request) -> HTMLResponse:
    """The listing, and a claim on the rig if it is going spare.

    Claiming on load rather than on the first post is what makes the read-only
    banner honest: a second browser is told before it clicks.
    """
    rig = request.app.state.rig
    token = tokens.token_from(request) or tokens.new_token()
    claim(rig, token)
    closed = request.query_params.get("closed", "")
    notice = (
        f"Closed {closed}. Convert it to LeRobot below. To record again, apply a "
        "configuration on the Configuration page."
        if closed
        else ""
    )
    response = _page(request, token, notice=notice)
    tokens.remember(response, token)
    return response


def _start(request: Request, argv, label: str, env: dict | None = None):
    """Own the rig, then spawn."""
    rig = request.app.state.rig
    token = tokens.token_from(request)
    try:
        require_owner(rig, token)
    except StaleOwner as exc:
        return _page(request, token, status_code=CONFLICT, error=str(exc))

    job = jobs.start(
        registry_of(request.app),
        argv,
        cwd=audit.REPO_ROOT,
        timeout=audit.TIMEOUT_S,
        label=label,
        kind=KIND,
        env=audit.script_env() if env is None else env,
    )
    return RedirectResponse(f"/audit/jobs/{job.id}", status_code=SEE_OTHER)


@router.post("/audit/health-check")
async def run_health_check(
    request: Request,
    date: str = Form(""),
    end_date: str = Form(""),
    task_name: str = Form(""),
    session_folder: str = Form(""),
    dry_run: str = Form(""),
):
    """`session_health_check.py`: check a day's alignment, or repair it
    (with `end_date`, every day from `date` through `end_date` inclusive).

    `dry_run` is set by the "Check alignment" button and left empty by
    "Re-align missing": a check lists the episodes with no valid aligned copy
    and changes nothing, a re-align writes those copies.
    """
    report_only = bool(dry_run)
    if session_folder:
        scope = pathlib.Path(session_folder.rstrip("/")).name
    elif end_date and end_date != date:
        scope = f"{date or 'today'} to {end_date}"
    else:
        scope = date or "today"
    return _start(
        request,
        audit.health_check_argv(
            date=date,
            end_date=end_date,
            task_name=task_name,
            session_folder=session_folder,
            dry_run=report_only,
        ),
        f"{'Alignment check' if report_only else 'Re-align missing'} — {scope}",
    )


@router.post("/audit/lerobot")
async def convert_to_lerobot(request: Request, session_folder: str = Form(...)):
    """Convert one session's aligned copy to LeRobot, into `output/lerobot_format`.

    Refused for the session the console is armed for, until "Finish and close
    session" has moved it on, and when the session has no aligned copy yet: the
    converter reads only aligned data. A session already converted is refused by the converter
    itself, which never overwrites a dataset; the job log says so.
    """
    token = tokens.token_from(request)
    folder = pathlib.Path(session_folder.rstrip("/"))
    aligned = audit.aligned_folder(folder)
    if not (aligned / "zarr.json").is_file():
        return _page(
            request,
            token,
            status_code=CONFLICT,
            error=f"{folder.name} has no aligned copy to convert yet ({aligned}).",
        )
    active = pathlib.Path(_active_folder(request.app).rstrip("/")).name
    if folder.name == active:
        return _page(
            request,
            token,
            status_code=CONFLICT,
            error=(
                f"{folder.name} is the console's current session, so more episodes "
                "may still arrive. Finish and close it, then convert it."
            ),
        )
    session = audit.Session(folder)
    if not session.aligned or session.missing_alignment:
        detail = next(iter(session.alignment_problems.values()), [])
        return _page(
            request,
            token,
            status_code=CONFLICT,
            error=(
                f"{folder.name}: {session.aligned} of {session.finalized} episodes "
                "have completed alignment. Run Check and Re-align before converting "
                "to LeRobot. Missing camera recordings must be recorded again. "
                + " ".join(detail)
            ),
        )
    missing = audit.missing_lerobot_modules()
    if missing:
        return _page(
            request,
            token,
            status_code=CONFLICT,
            error=(
                f"The LeRobot converter needs {', '.join(missing)}, which this station "
                "does not have yet. Run scripts/setup_station.sh to install them."
            ),
        )
    return _start(
        request,
        audit.lerobot_argv(aligned),
        f"LeRobot conversion — {folder.name}",
    )


@router.post("/audit/session/close")
async def close_session(
    request: Request, session_folder: str = Form(...), confirm: str = Form("")
):
    """Finish the session the console is armed for: seal it, move to a new one.

    Only the armed session: any other folder was already left behind (and
    sealed when it was). Refused while a recorder runs into it, and when the
    Configuration draft names a different folder than the armed one, because
    the sequence seals the draft's folder and would leave the armed one open.
    Confirmed by a second POST, like Configuration's "Start a new session": a
    sealed folder takes no more episodes.
    """
    rig = request.app.state.rig
    token = tokens.token_from(request)
    try:
        require_owner(rig, token)
    except StaleOwner as exc:
        return _page(request, token, status_code=CONFLICT, error=str(exc))
    folder = pathlib.Path(session_folder.rstrip("/"))
    if folder.name != pathlib.Path(_active_folder(request.app).rstrip("/")).name:
        return _page(
            request,
            token,
            status_code=CONFLICT,
            error=f"{folder.name} is not the console's current session; it is already closed.",
        )
    if recording.in_progress(request.app):
        return _page(
            request,
            token,
            status_code=CONFLICT,
            error=f"{folder.name} is still being recorded. Stop the recording, then close it.",
        )
    draft = rig.draft
    try:
        draft_folder = pathlib.Path(session_state.data_dir(draft)).name
    except FieldError:
        draft_folder = ""
    if draft_folder != folder.name:
        return _page(
            request,
            token,
            status_code=CONFLICT,
            error=(
                "The Configuration page has changes that were not applied, so it "
                f"names another session than {folder.name}. Apply or undo them, "
                "then close the session."
            ),
        )
    if not confirm:
        return _page(
            request,
            token,
            confirm={
                "action": "/audit/session/close",
                "label": "Finish and close session",
                "session_folder": str(folder),
                "message": (
                    f"Close {folder.name}? Its episodes stay where they are, but no "
                    "more can be recorded into it. The console moves to a new "
                    "session, which needs a configuration applied before recording."
                ),
            },
        )
    next_session.start_next_session(request.app, draft, reason="operator")
    return RedirectResponse(
        f"/audit?{urlencode({'closed': folder.name})}#sessions", status_code=SEE_OTHER
    )


def _audit_job(request: Request, job_id: str):
    """This page's job with that id, or None -- and another page's job is None.

    The single choke point for every route that takes an id, so a foreign job
    is unreachable from here whether the operator reads it, streams it or
    stops it. An untagged job fails the test too: a job of unknown provenance
    is treated as another page's, never as this page's.
    """
    try:
        job = registry_of(request.app).get(job_id)
    except UnknownJob:
        return None
    return job if job.kind == KIND else None


@router.get("/audit/jobs/{job_id}", response_class=HTMLResponse)
async def job_page(request: Request, job_id: str) -> HTMLResponse:
    """The whole log, server-rendered.

    Rendering it here rather than only over SSE is the no-JavaScript path: with
    the vendored htmx or this page's own script blocked, the operator still
    sees every line and a plain link that fetches the rest.
    """
    job = _audit_job(request, job_id)
    if job is None:
        return _page(
            request,
            tokens.token_from(request),
            status_code=NOT_FOUND,
            # Keep monitoring with the page that owns the job, including stale links.
            error=f"No audit job {job_id}. Jobs do not survive a restart.",
        )
    token = tokens.token_from(request)
    context = _page_context(request, token)
    context["active_audit"] = {
        "job": job,
        "lines": [audit.plain(line) for line in job.lines()],
        "can_edit": is_owner(request.app.state.rig, token),
    }
    return render_page(request, "audit", context, token=token)


@router.get("/audit/jobs/{job_id}/stream")
async def job_stream(request: Request, job_id: str):
    """Server-Sent Events, hand-rolled in `jobs.stream`.

    No `sse-starlette`: it is in this repo only as a transitive `mcp`
    dependency and is absent from the `data_collection` group a station
    installs, so importing it would work here and fail there.
    """
    # Validate job ownership before starting the streaming generator: after
    # HTTP headers are sent it is too late to return 404.
    if _audit_job(request, job_id) is None:
        return HTMLResponse(f"no audit job {job_id}", status_code=NOT_FOUND)
    try:
        frames = jobs.stream(registry_of(request.app), job_id)
    except UnknownJob:
        return HTMLResponse(f"no audit job {job_id}", status_code=NOT_FOUND)
    return StreamingResponse(
        # Strip ANSI codes for HTML consumers; preserve them in shared job output.
        (audit.plain(frame) for frame in frames),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Disable proxy buffering so event output is delivered incrementally.
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/audit/jobs/{job_id}/cancel")
async def cancel_job(request: Request, job_id: str):
    """Stop an audit -- and only an audit. `SIGTERM` first, see `jobs.cancel`.

    The kind check is the load-bearing one on this route. `/record` registers
    `record_data.py` in the same registry, and that process drains queued
    alignments in a `finally` block: stopping it from a button
    reading "Stop this audit" would end a session the operator believed was
    still recording.
    """
    rig = request.app.state.rig
    token = tokens.token_from(request)
    try:
        require_owner(rig, token)
    except StaleOwner as exc:
        return _page(request, token, status_code=CONFLICT, error=str(exc))
    if _audit_job(request, job_id) is None:
        return _page(
            request,
            token,
            status_code=NOT_FOUND,
            error=f"No audit job {job_id} to stop.",
        )
    jobs.cancel(registry_of(request.app), job_id)
    return RedirectResponse(f"/audit/jobs/{job_id}", status_code=SEE_OTHER)

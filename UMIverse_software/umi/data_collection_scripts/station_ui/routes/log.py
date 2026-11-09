# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`/log` -- what this console has failed at, and the file it wrote it to.

Deliberately absent from `NAV_ITEMS`: that nav is a six-step operator workflow
and this is not a step. It is reached from the Log link in the app bar, from the
crash banner, and from the id on the crash page.

What is on the page:

  1. the crash list
  2. the traceback, in a collapsed <details>, plus a plain-text download
  3. `station.log` -- tail, live stream, whole-file download

A station log is what you send when something is wrong, and the person
standing at the station is the one who has to send it. The file's PATH is not
shown: it is a filesystem layout an operator cannot act on, and the tail, the
stream and the download all work without ever naming it.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse

from station_ui import crash, log_tail, logging_setup

router = APIRouter()

PAGE = "/log"
NOT_FOUND = 404

_MB = 1024 * 1024


def _log_file(request: Request):
    return getattr(request.app.state, "log_file", None)


def _file_view(request: Request) -> dict:
    """The log itself, for everyone."""
    blank = {"log_lines": [], "log_size": "", "has_log_file": False}
    path = _log_file(request)
    if path is None:
        return blank
    try:
        size = path.stat().st_size
        lines = log_tail.tail(path)
    except OSError:
        return blank
    return {
        "log_lines": lines,
        "log_size": f"{size / _MB:.1f} MB",
        "has_log_file": True,
    }


@router.get(PAGE, response_class=HTMLResponse)
async def log_page(request: Request, error: str = "") -> HTMLResponse:
    """The crash list. It does not stream, and that is deliberate.

    Crashes arrive by push: the banner appears where the operator is standing
    and the app-bar count follows them to every other page. A live-updating list
    would be a second, weaker channel that has to be watched. So this gets a
    plain **Refresh** link, exactly as `audit_job.html` does -- and not
    `config_scan.html`'s self-repolling shape, whose whole virtue is that it
    stops when the finished fragment omits its attributes. A crash list never
    finishes.
    """
    app = request.app
    return app.state.templates.TemplateResponse(
        request,
        "log.html",
        {
            "page_title": "Station log",
            "active": PAGE,
            "crashes": crash.recent(app),
            # Use the error query parameter to select the crash banner's traceback.
            "open_error": error,
            "started_at": logging_setup.started_at().strftime("%H:%M"),
            "log_problem": getattr(app.state, "log_file_error", ""),
            **_file_view(request),
        },
    )


@router.get(f"{PAGE}/errors/{{error_id}}.txt", response_class=PlainTextResponse)
async def error_report(request: Request, error_id: str) -> PlainTextResponse:
    """One crash as plain text: the artefact that goes into a bug report."""
    from station_ui.app import _station_id

    entry = crash.find(request.app, error_id)
    if entry is None:
        return PlainTextResponse(
            f"No error {error_id} on this station. Errors are kept until the "
            "console restarts.\n",
            status_code=NOT_FOUND,
        )
    return PlainTextResponse(crash.report(entry, _station_id()))


@router.get(f"{PAGE}/file.txt", response_class=PlainTextResponse)
async def log_file(request: Request) -> PlainTextResponse:
    path = _log_file(request)
    if path is None:
        return PlainTextResponse(
            "This station has no log file. " + _problem(request), status_code=NOT_FOUND
        )
    try:
        return PlainTextResponse(path.read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        return PlainTextResponse(
            f"{path} could not be read: {exc}\n", status_code=NOT_FOUND
        )


@router.get(f"{PAGE}/file/stream")
async def log_stream(request: Request):
    """The tail as Server-Sent Events.

    Same shape as every other stream on this console (`record.py`, `audit.py`,
    `calibration.py`): resolve eagerly, so a file that cannot be opened is a
    real 404 rather than a 200 that dies two frames in.
    """
    path = _log_file(request)
    if path is None:
        return PlainTextResponse(
            "This station has no log file. " + _problem(request), status_code=NOT_FOUND
        )
    try:
        frames = log_tail.frames(path)
    except OSError as exc:
        return PlainTextResponse(
            f"{path} could not be read: {exc}\n", status_code=NOT_FOUND
        )
    return StreamingResponse(
        frames,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


def _problem(request: Request) -> str:
    return getattr(request.app.state, "log_file_error", "") or ""

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application factory for the station web UI."""

from __future__ import annotations

import collections
import html
import logging
import os
import socket

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import (
    HTMLResponse,
    Response,
)
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from station_ui import (
    crash,
    day_roll,
    highlight,
    logging_setup,
    origin_guard,
    readiness,
    recording,
    session_store,
)
from station_ui.rig import RigState
from station_ui.routes import audit as audit_routes
from station_ui.routes import calibration as calibration_routes
from station_ui.routes import config as config_routes
from station_ui.routes import identify as identify_routes
from station_ui.routes import log as log_routes
from station_ui.routes import preview as preview_routes
from station_ui.routes import record as record_routes
from station_ui.routes import setup as setup_routes
from station_ui.session import SessionDraft
from station_ui.settings import Settings
from station_ui.workflow import render_section

logger = logging.getLogger(__name__)

# The five workflow pages, in operator order. A disabled entry still renders, so
# the nav always shows the whole workflow.
NAV_ITEMS = (
    {"href": "/", "label": "Setup", "enabled": True},
    {"href": "/config", "label": "Configuration", "enabled": True},
    {"href": "/calibration", "label": "Calibration", "enabled": True},
    {"href": "/record", "label": "Record", "enabled": True},
    {"href": "/audit", "label": "Data Audit", "enabled": True},
)

# The numbered steps the sequential gate applies to, in `readiness._steps`'
# terms. Record and Data Audit are deliberately absent: Record refuses at the
# point of action through `readiness.blocking`, which is stricter than greying
# a tab, and Data Audit is about episodes recorded in the past.
NAV_GATED = frozenset({"/", "/config", "/calibration"})


def _step_blocker(request) -> object | None:
    """The outstanding step to warn about on this page, or None to stay quiet.

    The sequential rule is rendered, never enforced by navigation. `nav_open`
    greys the tabs out, but a bookmark, a back button, or a tab left open from
    before a step expired still opens the page, and the honest answer there is
    to say so, not to bounce the operator somewhere they did not ask to go. A
    redirect breaks the back button (press it, get thrown forward again) and
    arrives with no explanation; this is the pattern `/record` already uses,
    which shows the blocker and lets the page render.

    Advisory on purpose. What actually protects this station is at the point of
    action (`/record` refuses to start), and a banner here is a sign, not a
    lock.

    Only the numbered steps. The step that is the first outstanding one says
    nothing: it is not blocked, it is the work.
    """
    path = request.url.path
    if path not in NAV_GATED:
        return None
    try:
        if path in readiness.reachable(request.app):
            return None
        return readiness.blocking(request.app)
    except Exception:  # noqa: BLE001 - chrome on every page must not raise
        return None


def _station_id() -> str:
    """Which bench this is.

    `UMI_STATION_ID`, else the hostname. Read per request rather than captured
    at import, because a station that exports UMI_STATION_ID after the server is
    up would otherwise keep the hostname.
    """
    return os.environ.get("UMI_STATION_ID") or socket.gethostname()


# Read readiness through its module so navigation and recording controls
# use the same completion rules.


def _shell_context(request: Request) -> dict:
    """What the chrome around every page needs: identity, and real progress.

    A context processor rather than a template global because both values are
    per-request state. `nav_items` is deliberately not emitted here: it stays
    a Jinja global so a test can inject a row into it and see the shell render
    that row.
    """
    app = request.app

    return {
        "station": {"id": _station_id()},
        # The tick beside each numbered tab. Computed per request from the
        # draft and the file on disk, never from "the page was visited".
        "nav_done": readiness.done(app),
        # Lock configuration steps while recording. Expose the reason separately
        # from browser ownership conflicts.
        "recording_locked": recording.in_progress(app),
        # Set only on a step the operator has not reached yet -- the banner that
        # says so instead of redirecting them away. See `_step_blocker`.
        "step_blocker": _step_blocker(request),
        # Set once, on the first real page load after the console started today's
        # session by itself. The session id under the operator changed while they
        # were not looking, so they are told -- see `day_roll`.
        "day_rolled": day_roll.notice(request),
        # Which tabs may be opened at all. The workflow is sequential, so every
        # tab past the first outstanding step renders as the disabled span the
        # tab bar already has markup for.
        "nav_open": readiness.reachable(app),
        # The error count is shown on every page, so a failure stays visible
        # after the operator has moved on. Absent from the markup while it is zero.
        "error_count": len(crash.recent(app)),
    }


SERVER_ERROR = 500

# Expose error IDs as links, response headers and log prefixes so browser
# and command-line reports can locate the same traceback.
ERROR_ID_HEADER = "X-Station-Error-Id"


async def _unhandled(request: Request, exc: Exception) -> Response:
    """Every failure nothing else claimed, answered where the operator is.

    Two shapes, chosen by the header htmx sends. An htmx swap gets the banner
    fragment, because injecting a whole document into a banner slot would be
    worse than the silence it replaces; anything else (a plain form POST, a
    page load) gets the same words wrapped in the shell.

    The traceback is logged here rather than left to a handler on the root
    logger, and that is not belt and braces: under `uvicorn.run` the
    `uvicorn` logger has `propagate: False`, so the ASGI traceback never
    reaches a root handler at all.

    `ServerErrorMiddleware` re-raises after this response is sent, so uvicorn
    still prints its own copy to the terminal. That duplication is deliberate:
    swallowing the exception here would take the failure away from `TestClient`
    and from any server that watches for it.
    """
    entry = crash.record(request.app, exc, request=request)
    logger.error(
        "[%s] %s %s %s",
        entry.id,
        entry.method or "-",
        entry.path or "-",
        entry.exc_type,
        exc_info=exc,
    )

    fragment = request.headers.get("HX-Request", "").lower() == "true"
    headers = {ERROR_ID_HEADER: entry.id}
    try:
        return request.app.state.templates.TemplateResponse(
            request,
            "_partials/crash_banner.html" if fragment else "crash.html",
            {
                "page_title": entry.page_label,
                "active": entry.page_href,
                "crash": entry,
            },
            status_code=SERVER_ERROR,
            headers=headers,
        )
    except Exception:  # noqa: BLE001 - the handler must not be able to crash
        # Render this fallback without templates or shared context: either may
        # have caused the exception being reported.
        logger.exception("[%s] the crash page itself could not be rendered", entry.id)
        return HTMLResponse(
            _fallback_page(entry, fragment=fragment),
            status_code=SERVER_ERROR,
            headers=headers,
        )


def _fallback_page(entry, *, fragment: bool) -> str:
    if entry.is_action:
        headline = f"{entry.action} failed."
        recovery = "Try it again. If it fails twice, this station needs an engineer."
    else:
        headline = f"{entry.page_label} failed to load."
        recovery = "Reload the page. If it fails twice, this station needs an engineer."

    body = (
        f'<p role="alert"><b>{html.escape(headline)}</b> {recovery}</p>'
        f'<p>Error <a href="/log?error={entry.id}">{entry.id}</a> · '
        '<a href="/log">Open the station log</a></p>'
    )
    if fragment:
        return body
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{html.escape(headline)} — UMI Station</title></head>"
        f"<body>{body}</body></html>"
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. One factory so tests get an isolated instance per call."""
    settings = settings or Settings()

    app = FastAPI(title="UMI Station", docs_url=None, redoc_url=None)
    app.state.settings = settings

    # Attach persistent logging once per process. A log-file failure must not
    # prevent data collection.
    app.state.log_file, app.state.log_file_error = logging_setup.configure(settings)
    # Keep a process-local log index for /log; the file remains the durable record.
    app.state.crashes = collections.deque(maxlen=crash.MAX_RECORDS)

    # Create one owner and draft per app instance. Restore the draft from the
    # last applied config, falling back to an empty draft; browser ownership is
    # never restored.
    app.state.rig = RigState(
        draft=session_store.restore(settings.state_dir) or SessionDraft()
    )

    # One-shot notice that the recording day advanced while the console was running.
    app.state.day_rolled = None

    # Reattach surviving recorder processes before serving pages. A detached
    # recorder may still own cameras, so a server restart must not offer a
    # second launch. Reap stale process groups during recovery.
    outcome = recording.adopt(app)
    if outcome != "none":
        logger.info("recorder from a previous console: %s", outcome)

    # Use StaticFiles containment checks to prevent paths escaping the static root.
    app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")

    templates = Jinja2Templates(
        directory=settings.templates_dir, context_processors=[_shell_context]
    )
    templates.env.globals["nav_items"] = NAV_ITEMS
    templates.env.globals["nav_gated"] = NAV_GATED
    # Returns Markup and escapes as it goes, so `|ansi` replaces the autoescape
    # on that value rather than being applied after it. Nothing may reach it
    # already escaped (see `highlight.py`).
    templates.env.filters["ansi"] = highlight.ansi_to_html
    templates.env.globals["render_section"] = render_section
    app.state.templates = templates

    # Keep cross-origin requests disabled. OriginGuard also rejects mutations
    # from other localhost ports, which SameSite cookies alone cannot isolate.
    app.add_middleware(origin_guard.OriginGuard)

    # Refresh the recording date before handlers run so an overnight console
    # cannot continue filing episodes under yesterday's session.
    app.add_middleware(day_roll.DayRoll)

    app.include_router(setup_routes.router)
    app.include_router(config_routes.router)
    app.include_router(preview_routes.router)
    app.include_router(identify_routes.router)
    app.include_router(calibration_routes.router)
    app.include_router(record_routes.router)
    app.include_router(audit_routes.router)
    app.include_router(log_routes.router)

    # Register the fallback exception handler before the middleware stack is built.
    app.add_exception_handler(Exception, _unhandled)

    return app

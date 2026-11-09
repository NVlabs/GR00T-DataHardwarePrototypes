# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every unhandled failure, in the words the operator pressed.

Every mutating control is an htmx swap, and htmx does not swap a non-2xx
response, so without this module a failing handler would leave the UI doing
nothing at all after the operator pressed a button. This module is the index
that lets `/log` name what broke without parsing `station.log`, and lets the
crash page say "Apply configuration failed" rather than "an error occurred".

The deque is process memory only, the same property `jobs` has and for the same
reason: the file is the durable record, this is the index. It dies with the
process, which is why the empty state says "since the console started".
"""

from __future__ import annotations

import collections
import datetime
import traceback as traceback_module
import uuid
from dataclasses import dataclass

MAX_RECORDS = 50

# Store thread exceptions outside app state because threading.excepthook
# receives no request or application context.
_THREAD_CRASHES: collections.deque = collections.deque(maxlen=MAX_RECORDS)

# Map METHOD/route-pattern pairs to operator-facing actions. Parameterized
# routes share one label; unmapped actions fall back to the page label.
ACTIONS = {
    "POST /setup/preset": "Choose a setup",
    "POST /take-control": "Take control",
    "POST /config/task": "Set the task name",
    "POST /config/apply": "Apply configuration",
    "POST /config/devices/scan": "Refresh devices",
    "POST /config/devices/adopt": "Save what you typed on Configuration",
    "POST /config/devices/link/{arm}": "Choose a gripper",
    "POST /config/identify/start": "Open camera identification",
    "POST /config/identify/refresh": "Refresh identified devices",
    "POST /config/identify/{session_id}/scan": "Identify the gripper cameras",
    "POST /config/identify/{session_id}/stop": "Close camera identification",
    "POST /config/preview/start": "Open a camera preview",
    "POST /config/preview/stop/{job_id}": "Close the camera preview",
    "POST /config/vive/probe": "Test the Vive connection",
    "POST /config/session/uid": "Start a new session",
    "POST /config/optional/add": "Add a camera",
    "POST /config/optional/remove": "Remove a camera",
    "POST /devices/scan": "Start the device check",
    "POST /devices/scan/cancel": "Cancel the device check",
    "POST /devices/skip": "Skip the device check",
    "POST /calibration/check/start": "Start the camera check",
    "POST /calibration/check/stop": "Stop the camera check",
    "POST /calibration/vive/start": "Start the tracker check",
    "POST /calibration/vive/stop": "Stop the tracker check",
    "POST /calibration/calibrate-gripper": "Calibrate the gripper",
    "POST /record/start": "Start recording",
    "POST /record/stop": "Stop recording",
    "POST /record/force-stop": "Force stop the recording",
    "POST /record/pkill": "Force kill the recorder",
    "POST /audit/health-check": "Run the alignment check",
    "POST /audit/lerobot": "Convert to LeRobot",
    "POST /audit/session/close": "Finish and close the session",
    "POST /audit/jobs/{job_id}/cancel": "Stop the audit",
}

# Pages that are not workflow steps, and so are not in `NAV_ITEMS`.
EXTRA_PAGES = {"/log": "Station log"}

_UNFORMATTABLE = "(traceback could not be formatted)"


@dataclass(frozen=True)
class CrashRecord:
    """One failure, described in both readers' vocabularies at once."""

    id: str
    at: datetime.datetime
    source: str  # "request" | "thread"
    method: str
    path: str  # the concrete URL
    route: str  # the matched pattern
    action: str  # the operator's word
    page_href: str
    page_label: str
    exc_type: str
    exc_message: str
    exc_last_line: str
    traceback: str

    @property
    def is_action(self) -> bool:
        """Whether this was something the operator did.

        A POST names an act ("Apply configuration failed"); a GET names a page
        ("Configuration failed to load"). The two need different sentences,
        because "try it again" is wrong advice for a page that will not load.
        """
        return self.method not in ("", "GET", "HEAD")

    @property
    def time(self) -> str:
        return self.at.strftime("%H:%M:%S")

    @property
    def line_count(self) -> int:
        return len(self.traceback.splitlines())


def record(app, exc, *, request=None, thread_name: str = "") -> CrashRecord:
    """Index one failure. Never raises.

    A crash handler that crashes is the one outcome with no recovery left, so
    every part of this that could fail -- formatting a traceback, reading the
    route pattern off the scope, appending to a deque on an app that has none --
    is contained. A record with a missing traceback is still a record.
    """
    try:
        formatted = "".join(
            traceback_module.format_exception(type(exc), exc, exc.__traceback__)
        ).rstrip()
    except Exception:  # noqa: BLE001 - see the docstring
        formatted = _UNFORMATTABLE

    method, path, route = _request_shape(request)
    action, page_href, page_label = _words(method, route, path, thread_name)

    entry = CrashRecord(
        id=uuid.uuid4().hex[:12],
        at=datetime.datetime.now().astimezone(),
        source="request" if request is not None else "thread",
        method=method,
        path=path,
        route=route,
        action=action,
        page_href=page_href,
        page_label=page_label,
        exc_type=type(exc).__name__,
        exc_message=str(exc),
        exc_last_line=_last_line(formatted, exc),
        traceback=formatted,
    )

    try:
        if app is not None and hasattr(app.state, "crashes"):
            app.state.crashes.append(entry)
        else:
            _THREAD_CRASHES.append(entry)
    except Exception:  # noqa: BLE001 - see the docstring
        pass
    return entry


def recent(app) -> list[CrashRecord]:
    """Everything this console has failed at, newest first.

    Request and thread crashes merged: from the operator's chair they are one
    list of things that went wrong, and a thread crash is the one they are
    least likely to have noticed any other way.
    """
    merged = list(getattr(app.state, "crashes", ())) + list(_THREAD_CRASHES)
    merged.sort(key=lambda entry: entry.at, reverse=True)
    return merged


def find(app, error_id: str) -> CrashRecord | None:
    for entry in recent(app):
        if entry.id == error_id:
            return entry
    return None


def reset() -> None:
    """Forget the thread crashes. Tests only -- the process buffer is global."""
    _THREAD_CRASHES.clear()


def report(entry: CrashRecord, station_id: str) -> str:
    """The plain-text artefact an operator pastes into a bug report.

    Same rationale as `routes/calibration.py:job_log`: the thing that gets
    pasted has to carry its own context, because the person reading it is not
    standing at the station and cannot see which one it was.
    """
    header = [
        f"UMI station error {entry.id}",
        f"station    {station_id}",
        f"at         {entry.at.strftime('%Y-%m-%d %H:%M:%S%z')}",
        f"action     {entry.action}",
    ]
    if entry.method:
        header.append(f"request    {entry.method} {entry.path}")
    else:
        header.append("source     background thread")
    header.append(f"exception  {entry.exc_type}: {entry.exc_message}")
    return "\n".join(header) + "\n\n" + entry.traceback + "\n"


# ------------------------------------------------------------------ internals


def _request_shape(request) -> tuple[str, str, str]:
    """`(method, concrete path, route pattern)`: all three, or blanks.

    The pattern comes from `request.scope["route"].path`, which inside an
    exception handler is the matched route rather than the URL, so every job id
    and session uid collapses onto one key.
    """
    if request is None:
        return "", "", ""
    try:
        method = request.method
        path = request.url.path
    except Exception:  # noqa: BLE001 - a half-built scope must not win
        return "", "", ""
    route = getattr(request.scope.get("route"), "path", "") or path
    return method, path, route


def _pages() -> list[tuple[str, str]]:
    """`(href, label)` for every page, longest first.

    Imported inside the function because `station_ui.app` imports this module;
    the same lazy-import shape `app._generated_config` already uses.
    """
    from station_ui.app import NAV_ITEMS

    pages = {item["href"]: item["label"] for item in NAV_ITEMS}
    pages.update(EXTRA_PAGES)
    return sorted(pages.items(), key=lambda pair: len(pair[0]), reverse=True)


def _words(
    method: str, route: str, path: str, thread_name: str
) -> tuple[str, str, str]:
    """The operator's word, and the page to send them back to."""
    if not method:
        # A background thread. There is no page and no button -- the honest
        # destination is the log itself.
        label = f"Background work ({thread_name})" if thread_name else "Background work"
        return label, "/log", "Station log"

    page_href, page_label = "/", "Setup"
    for href, label in _pages():
        if path == href or path.startswith(href.rstrip("/") + "/"):
            page_href, page_label = href, label
            break

    return ACTIONS.get(f"{method} {route}") or page_label, page_href, page_label


def _last_line(formatted: str, exc) -> str:
    """The exception line, hoisted out of the traceback.

    Python prints most-recent-call-last, so the point of failure is the one
    line you cannot see without scrolling sixty lines of frames.
    """
    for line in reversed(formatted.splitlines()):
        if line.strip():
            return line.strip()
    return f"{type(exc).__name__}: {exc}"

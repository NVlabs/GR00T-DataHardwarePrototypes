# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`/calibration` -- the browser orchestrates, the station runs the hardware.

The gripper step ends in a terminal an operator has to be standing in front of:
it waits on `input()` prompts (see `umi/umi_device/umi_gripper.py`), so it needs
a keyboard attached to the station. That cannot move into a browser, this page
does not try, and it says so in the page copy.

The camera check runs from here. `--watch` opens every camera at once and
reports what each decodes, this page renders the marker itself, and the fold in
`station_ui/calibration.py` decides which side is current. One click checks the
whole rig, and the operator carries this browser to the rig.

The step logic itself is in `station_ui/calibration.py`; this module is only the
HTTP boundary. Shape follows `routes/setup.py`: 303 after a POST, 409 when
another browser holds the rig.
"""

from __future__ import annotations

import pathlib
import uuid

from fastapi import APIRouter, Form, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from station_ui import calibration as cal
from station_ui import (
    calibration_media,
    identification,
    jobs,
    markers,
    preview,
    recording,
    tokens,
    vive_check,
)
from station_ui import session as session_state
from station_ui.camera_access import serialize_start
from station_ui.errors import (
    FieldError,
    RecordingInProgress,
    StaleOwner,
    UnknownJob,
)
from station_ui.rig import is_owner, require_owner
from station_ui.routes import config as config_routes
from station_ui.workflow import render_page

router = APIRouter()


SEE_OTHER = 303
CONFLICT = 409
BAD_REQUEST = 400
NOT_FOUND = 404

PAGE = "/calibration"

_ROLE_WORDS = {
    "left": "Left",
    "right": "Right",
    "head": "Head",
    "umi": "UMI (wrist)",
    "tracking": "Tracking (iPhone)",
    "trigger": "Trigger",
}


def _role_label(role: str) -> str:
    """`left_umi` -> `Left UMI (wrist) camera`, with no import of the script.

    Importing `verify_device_assignment` for its `ROLE_LABELS` would pull cv2
    and every camera SDK into the web process at import time.
    """
    parts = [_ROLE_WORDS.get(part, part.title()) for part in role.split("_")]
    return " ".join(parts) + " camera"


# --------------------------------------------------------------------- render


def _check_view(request: Request) -> cal.CheckView:
    config = cal.layout_config(request.app)
    roles = cal.verification_roles(config)
    return cal.check_view(request.app, roles, _role_label)


def _vive_sections(app):
    sections = cal.vive_sections(app)
    for section in sections:
        view = section["view"]
        section["chart"] = None
        if view.job_id:
            try:
                job = cal.job_registry(app).get(view.job_id)
            except UnknownJob:
                continue
            section["chart"] = calibration_media.tracker_chart(
                vive_check.parse(job.lines())
            )
    return sections


def _step_views(request: Request):
    app = request.app
    registry = cal.job_registry(app)
    state = cal.calibration_state(app)
    draft = app.state.rig.draft

    try:
        config = cal.layout_config(app)
    except FieldError as exc:
        return None, str(exc)

    units = {
        cal.STEP_GRIPPER: [
            (
                side,
                f"{side.title()} gripper — {draft.grippers.get(side) or 'choose in Configuration'}",
            )
            for side in sorted(
                draft.grippers
                or config_routes._requirements(request, draft).gripper_arms
            )
        ],
    }
    actions = {
        cal.STEP_GRIPPER: (f"{PAGE}/calibrate-gripper", "side", "Calibrate"),
    }
    empty = {
        cal.STEP_GRIPPER: (
            "No grippers chosen yet — pick them on the Configuration page."
        ),
    }

    views = []
    # Use the rig-specific step list, including Vive checks only when required.
    for step in cal.steps_for(config):
        overall = cal.step_state(app, step.id)
        entry = {
            "step": step,
            "state": overall,
            "badge": cal.BADGES[overall],
        }
        if step.id == cal.STEP_VERIFY:
            roles = cal.verification_roles(config)
            if roles and set(roles) <= identification.verified_roles(app):
                continue
            # One scan for the whole rig, so there are no per-unit runs to
            # render: the step hands its body to the check panel, which derives
            # every camera's state from that one child's output.
            entry["check"] = True
            views.append(entry)
            continue
        if step.id == cal.STEP_VIVE:
            # Render the whole-rig tracker panel separately; per-unit action handlers
            # apply only to the remaining calibration steps.
            entry["vive"] = True
            views.append(entry)
            continue

        runs = state.runs(step.id)
        action, field_name, verb = actions[step.id]
        entry.update(
            {
                "action": action,
                "field": field_name,
                "verb": verb,
                "empty": empty[step.id],
                "runs": [
                    {
                        "view": (
                            view := cal.run_view(
                                registry,
                                key,
                                label,
                                runs.get(key),
                                # Key calibration URLs by side; each side displays its
                                # latest run.
                                plot_url=f"{PAGE}/gripper/{key}/plot.png",
                            )
                        ),
                        "badge": cal.BADGES[view.state],
                    }
                    for key, label in units[step.id]
                ],
            }
        )
        views.append(entry)
    return views, ""


def _page(request: Request, *, status_code: int = 200, error: str = "") -> HTMLResponse:
    return render_page(
        request,
        "calibration",
        _page_context(request, status_code=status_code, error=error),
        token=tokens.token_from(request),
        status_code=status_code,
    )


def _page_context(request: Request, *, status_code: int = 200, error: str = "") -> dict:
    rig = request.app.state.rig
    token = tokens.token_from(request)
    views, blocked = _step_views(request)
    # Poll while terminal-based gripper calibration runs; it has no SSE stream.
    # It cannot run concurrently with camera verification.
    gripper_running = any(
        entry["step"].id == cal.STEP_GRIPPER and entry["state"] == cal.RUNNING
        for entry in (views or [])
    )
    try:
        check = _check_view(request)
    except FieldError:
        # Same condition `blocked` already reports; the page renders the "go
        # choose a preset" card instead of any of this.
        check = None

    return {
        "page_title": "Calibration",
        "active": PAGE,
        "steps": views,
        "blocked": blocked,
        "gripper_running": gripper_running,
        "check": check,
        # Only while a watcher is alive: an EventSource opened against a
        # finished job would connect, replay, and close on every reload.
        "check_stream_url": (
            f"{PAGE}/check/stream/{check.job_id}"
            if check is not None and check.running
            else None
        ),
        "vive_sections": _vive_sections(request.app),
        # Stream only a live tracker job to avoid repeatedly replaying completed
        # output. At most one arm runs against the shared bridge at once.
        "vive_stream_url": _vive_stream_url(request.app),
        "can_edit": is_owner(rig, token),
        "error": error,
    }


@router.get(PAGE, response_class=HTMLResponse)
async def calibration_page(request: Request) -> HTMLResponse:
    return _page(request)


# --------------------------------------------------------------------- guards


async def _guard(request: Request):
    """Ownership first, then whether the rig is configured enough to run.

    Returns an error response, or `(config, None)`. A read-only browser must
    never start hardware, so the ownership check comes before anything that
    could touch a device.
    """
    rig = request.app.state.rig
    token = tokens.token_from(request)
    try:
        require_owner(rig, token)
    except StaleOwner as exc:
        return None, _page(request, status_code=CONFLICT, error=str(exc))
    if identification.running(request.app):
        return None, _page(request, status_code=CONFLICT, error=identification.MESSAGE)
    # Do not start checks while recording owns the cameras. Keep check/stop
    # available because stopping releases hardware.
    try:
        recording.require_idle(request.app)
    except RecordingInProgress as exc:
        return None, _page(request, status_code=CONFLICT, error=str(exc))
    refused = await config_routes.save_for_action(request)
    if refused is not None:
        return None, refused
    try:
        return cal.recording_config(request.app), None
    except FieldError as exc:
        return None, _page(request, status_code=CONFLICT, error=str(exc))


def _start(request: Request, step_id: str, *, key, label, argv, timeout, **extra):
    job = jobs.start(
        cal.job_registry(request.app),
        argv,
        cwd=cal.repo_root(),
        timeout=timeout,
        label=f"{step_id}:{key}",
    )
    state = cal.calibration_state(request.app)
    state.record(
        step_id,
        cal.Run(key=key, job_id=job.id, label=label, **extra),
        # Which rig this run is about. A result recorded against a different
        # configuration expires rather than ticking -- see `calibration.stale`.
        fingerprint=session_state.device_fingerprint(request.app.state.rig.draft),
    )
    return RedirectResponse(f"{PAGE}#step-{step_id}", status_code=SEE_OTHER)


# --------------------------------------------------------------- camera check


@router.get(f"{PAGE}/check/panel", response_class=HTMLResponse)
async def check_panel(request: Request) -> HTMLResponse:
    """The fragment the live page refetches. Also what a reload renders."""
    try:
        check = _check_view(request)
    except FieldError:
        # Return the panel's empty state when a background refetch has no config.
        check = cal.CheckView()
    # Stop the watcher once all cameras verify so it releases exclusive OAK
    # ownership and recording can proceed.
    if check.done and check.running and check.job_id:
        try:
            jobs.cancel(cal.job_registry(request.app), check.job_id)
        except UnknownJob:
            pass
    return request.app.state.templates.TemplateResponse(
        request,
        "_partials/check_panel.html",
        {
            "check": check,
            # Update the step badge out of band with the live panel.
            "check_badge": cal.BADGES[cal.step_state(request.app, cal.STEP_VERIFY)],
            "can_edit": is_owner(request.app.state.rig, tokens.token_from(request)),
        },
    )


@router.post(f"{PAGE}/check/start")
@serialize_start
async def start_check(request: Request):
    _config, blocked = await _guard(request)
    if blocked is not None:
        return blocked

    state = cal.check_state(request.app)
    registry = cal.job_registry(request.app)
    if state.job_id:
        try:
            existing = registry.get(state.job_id)
        except UnknownJob:
            existing = None
        if existing is not None and not existing.finished:
            # An OAK open is exclusive: a second watcher would fight the first
            # over the same board and both would report it unavailable.
            return _page(
                request,
                status_code=CONFLICT,
                error="A camera check is already running. Stop it before starting another.",
            )

    # A live preview from /config holds one of the cameras the check opens.
    if preview.running(request.app):
        await run_in_threadpool(preview.stop, request.app, wait=preview.RELEASE_WAIT_S)

    roles = cal.verification_roles(_config)
    pending = [
        role
        for role in roles
        if role not in identification.verified_roles(request.app, _config)
    ]
    if not pending:
        return RedirectResponse(f"{PAGE}#calibration", status_code=SEE_OTHER)

    directory = (
        pathlib.Path(request.app.state.settings.state_dir) / "calibration-preview"
    )
    directory.mkdir(parents=True, exist_ok=True)
    # The previous watcher has exited before a new run can reach here.
    if state.preview_path:
        pathlib.Path(state.preview_path).unlink(missing_ok=True)
        pathlib.Path(state.preview_path + ".part").unlink(missing_ok=True)
    frame_path = directory / f"{uuid.uuid4().hex}.jpg"
    job = jobs.start(
        registry,
        cal.check_argv(
            cal.recording_config_path(request.app),
            frame_path,
            roles=pending if pending != roles else None,
        ),
        cwd=cal.repo_root(),
        timeout=cal.CHECK_TIMEOUT_S,
        label="camera check",
    )
    state.job_id = job.id
    state.preview_path = str(frame_path)
    # Stamp the whole camera-check step because one child watches every role.
    cal.calibration_state(request.app).stamp(
        cal.STEP_VERIFY,
        session_state.device_fingerprint(request.app.state.rig.draft),
    )
    return RedirectResponse(f"{PAGE}#step-{cal.STEP_VERIFY}", status_code=SEE_OTHER)


@router.post(f"{PAGE}/check/stop")
async def stop_check(request: Request):
    rig = request.app.state.rig
    try:
        require_owner(rig, tokens.token_from(request))
    except StaleOwner as exc:
        return _page(request, status_code=CONFLICT, error=str(exc))

    # Cancel without blocking the request; watcher cleanup releases cameras
    # asynchronously. Repeated stops remain harmless.
    cal.stop_check(request.app)
    return RedirectResponse(f"{PAGE}#step-{cal.STEP_VERIFY}", status_code=SEE_OTHER)


@router.get(f"{PAGE}/check/stream/{{job_id}}")
async def stream_check(request: Request, job_id: str):
    """The watcher's output as SSE, scoped to this page's own check.

    Scoped deliberately: the registry also holds gripper and recording jobs,
    and an endpoint under /calibration/check/ that would happily stream a
    recording is an accident waiting to be linked to.
    """
    if job_id != cal.check_state(request.app).job_id:
        return PlainTextResponse("no such camera check", status_code=NOT_FOUND)
    try:
        frames = jobs.stream(cal.job_registry(request.app), job_id)
    except UnknownJob:
        return PlainTextResponse("no such camera check", status_code=NOT_FOUND)
    return StreamingResponse(
        frames,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@router.get(f"{PAGE}/check/frame/{{job_id}}.jpg")
async def camera_check_frame(request: Request, job_id: str):
    """Serve only the current check's frame, never an arbitrary filesystem path."""
    state = cal.check_state(request.app)
    if job_id != state.job_id or not state.preview_path:
        return Response(status_code=NOT_FOUND)
    try:
        job = cal.job_registry(request.app).get(job_id)
    except UnknownJob:
        return Response(status_code=NOT_FOUND)
    if job.finished:
        return Response(status_code=NOT_FOUND)
    try:
        content = await run_in_threadpool(pathlib.Path(state.preview_path).read_bytes)
    except FileNotFoundError:
        return Response(status_code=204, headers={"Cache-Control": "no-store"})
    return Response(
        content, media_type="image/jpeg", headers={"Cache-Control": "no-store"}
    )


# ------------------------------------------------------------- tracker check


def _vive_stream_url(app) -> str | None:
    """The running arm's SSE URL, or None when no arm is running."""
    arm = cal.vive_running_arm(app)
    if arm is None:
        return None
    return f"{PAGE}/vive/stream/{cal.vive_check_state(app).job_ids[arm]}"


@router.get(f"{PAGE}/vive/panel", response_class=HTMLResponse)
async def vive_panel(request: Request) -> HTMLResponse:
    """The fragment the live page refetches. Also what a reload renders."""
    sections = _vive_sections(request.app)
    # Stop polling after this arm confirms so the shared bridge is released
    # for the next arm and the result cannot later become a timeout.
    for section in sections:
        view = section["view"]
        if view.done and view.running and view.job_id:
            try:
                jobs.cancel(cal.job_registry(request.app), view.job_id)
            except UnknownJob:
                pass
    return request.app.state.templates.TemplateResponse(
        request,
        "_partials/vive_panel.html",
        {
            "vive_sections": sections,
            # Update the step badge together with the live panel.
            "vive_badge": cal.BADGES[cal.step_state(request.app, cal.STEP_VIVE)],
            "can_edit": is_owner(request.app.state.rig, tokens.token_from(request)),
        },
    )


@router.post(f"{PAGE}/vive/start")
@serialize_start
async def start_vive_check(request: Request):
    _config, blocked = await _guard(request)
    if blocked is not None:
        return blocked

    form = await request.form()
    arm = str(form.get("arm") or "")
    keys = cal.vive_tracker_keys(request.app)
    if arm not in keys:
        # Reject a stale form if its arm is no longer part of the applied config.
        return _page(
            request,
            status_code=CONFLICT,
            error=(
                f"This rig has no {arm or 'unnamed'} tracker any more — "
                "reload the page."
            ),
        )

    running = cal.vive_running_arm(request.app)
    if running is not None:
        # Allow one verification poller across all arms to limit shared REP-bridge load.
        return _page(
            request,
            status_code=CONFLICT,
            error=(
                f"The {running} tracker check is already running. "
                "Stop it before starting another."
            ),
        )

    # Enforce arm order server-side as well as disabling blocked controls.
    holding = next(
        section["blocked_by"]
        for section in cal.vive_sections(request.app)
        if section["arm"] == arm
    )
    if holding is not None:
        return _page(
            request,
            status_code=CONFLICT,
            error=(
                f"Confirm the {holding} tracker first — the arms are checked "
                "one at a time."
            ),
        )

    job = jobs.start(
        cal.job_registry(request.app),
        cal.vive_argv(cal.recording_config_path(request.app), arm),
        cwd=cal.repo_root(),
        timeout=cal.VIVE_TIMEOUT_S,
        label=f"{arm} tracker check",
    )
    cal.vive_check_state(request.app).job_ids[arm] = job.id
    # Stamp the step with the rig identity shared by both arm checks.
    cal.calibration_state(request.app).stamp(
        cal.STEP_VIVE,
        session_state.device_fingerprint(request.app.state.rig.draft),
    )
    return RedirectResponse(f"{PAGE}#tracker-heading-{arm}", status_code=SEE_OTHER)


@router.post(f"{PAGE}/vive/stop")
async def stop_vive_check_route(request: Request):
    rig = request.app.state.rig
    try:
        require_owner(rig, tokens.token_from(request))
    except StaleOwner as exc:
        return _page(request, status_code=CONFLICT, error=str(exc))

    # Keep Stop available without the start guard so it can always release
    # the bridge. At most one tracker check is active.
    arm = cal.vive_running_arm(request.app)
    cal.stop_vive_check(request.app)
    anchor = f"tracker-heading-{arm}" if arm else f"step-{cal.STEP_VIVE}"
    return RedirectResponse(f"{PAGE}#{anchor}", status_code=SEE_OTHER)


@router.get(f"{PAGE}/vive/stream/{{job_id}}")
async def stream_vive_check(request: Request, job_id: str):
    """This page's own tracker check, as SSE.

    Scoped to that one job deliberately: the registry also holds gripper,
    camera-check and recording jobs, and an endpoint under /calibration/vive/
    that would happily stream a recording is an accident waiting to be linked
    to.

    Unlike the camera check's stream, this one notices a closed tab. Starlette
    wraps a sync iterable in `iterate_in_threadpool`, so handing it a raw
    generator pins one worker thread until the job finishes -- for up to
    VIVE_TIMEOUT_S per abandoned tab. Same shape as /record/stream, which
    already solves it.
    """
    # Restrict tracker operations to tracker jobs, including both arms.
    if job_id not in cal.vive_check_state(request.app).job_ids.values():
        return PlainTextResponse("no such tracker check", status_code=NOT_FOUND)
    try:
        frames = jobs.stream(cal.job_registry(request.app), job_id)
    except UnknownJob:
        return PlainTextResponse("no such tracker check", status_code=NOT_FOUND)

    async def _stream():
        async for frame in iterate_in_threadpool(frames):
            if await request.is_disconnected():
                break
            yield frame

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@router.get(f"{PAGE}/check/qr/{{side}}.svg")
async def check_qr(side: str):
    if side not in cal.SIDES:
        return Response("no such side", status_code=NOT_FOUND, media_type="text/plain")
    return Response(
        # Serve ArUco at the stable qr URL; existing pages can keep using it,
        # and the verification worker accepts both formats.
        markers.marker_svg(side),
        media_type="image/svg+xml",
        # Cache the fixed marker for each side rather than refetching it on every panel
        # refresh.
        headers={"Cache-Control": "public, max-age=86400"},
    )


# --------------------------------------------------------- gripper calibration


@router.post(f"{PAGE}/calibrate-gripper")
@serialize_start
async def calibrate_gripper(request: Request, side: str = Form(...)):
    _config, blocked = await _guard(request)
    if blocked is not None:
        return blocked
    identifier = request.app.state.rig.draft.grippers.get(side)
    if not identifier:
        return _page(
            request,
            status_code=BAD_REQUEST,
            error=f"No {side} gripper selected.",
        )
    # Share a run ID across status, plot and summary so results cannot be mixed.
    run = cal.run_id()
    status_path = cal.gripper_status_path(side, run)
    plot_dir = cal.gripper_plot_dir(side, run)
    summary_path = cal.gripper_summary_path(side, run)
    log_path = cal.gripper_log_path(side, run)
    return _start(
        request,
        cal.STEP_GRIPPER,
        key=side,
        label=f"{side.title()} gripper — {identifier}",
        argv=cal.gripper_argv(
            side,
            identifier,
            status_path=status_path,
            plot_dir=plot_dir,
            summary_path=summary_path,
            log_path=log_path,
            generation=cal.gripper_generation_for(request.app.state.rig.draft),
        ),
        timeout=cal.GRIPPER_TIMEOUT_S,
        status_path=status_path,
        plot_dir=plot_dir,
        summary_path=summary_path,
        log_path=log_path,
    )


@router.get(f"{PAGE}/gripper/{{side}}/plot.png")
async def gripper_plot(request: Request, side: str):
    """The plot the last calibration of *side* saved.

    Served from the run directory rather than mounted as static: these live
    under `~/.umi_temp/`, outside the package, and a StaticFiles mount over a
    directory the console writes per run is a much wider door than one file
    lookup keyed on a side this rig actually has.

    Read-only, so a second browser watching the bench sees it too.
    """
    run = cal.calibration_state(request.app).run(cal.STEP_GRIPPER, side)
    path = cal.gripper_plot_file(run) if run is not None else None
    if path is None:
        return PlainTextResponse("no plot for this gripper", status_code=NOT_FOUND)
    return FileResponse(path, media_type="image/png")


# ------------------------------------------------------------------- job logs


@router.get(f"{PAGE}/jobs/{{job_id}}/log", response_class=PlainTextResponse)
async def job_log(request: Request, job_id: str):
    """The whole captured log, for a step whose card only shows the tail.

    Plain text on purpose: an operator pastes this into a bug report.
    """
    try:
        job = cal.job_registry(request.app).get(job_id)
    except UnknownJob as exc:
        return PlainTextResponse(str(exc), status_code=NOT_FOUND)
    return PlainTextResponse("\n".join(job.lines()) + "\n")

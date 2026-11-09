# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`/record` -- start a recording, watch it, and stop it without losing data.

Two mechanisms make this page possible.

**Stopping a recording from outside its process.** `install_stop_on_sigterm()`
in `record_data.py` routes SIGTERM into the same shutdown path as the cv2 `q`
key, including the `finally` block that drains queued alignments. Without it,
`kill <pid>` would terminate Python without unwinding -- data loss that looks
like a clean exit. `jobs.cancel` is SIGTERM-first, so Stop here runs the same
shutdown the `q` key does.

**Seeing the recorder's output.** `station_ui.jobs` captures the child's
stdout, which is what makes a status page possible; this module streams it as
hand-rolled Server-Sent Events, because a station install carries neither
`sse-starlette` nor `websockets`.

What deliberately does not move into the browser: per-episode control. The
`s` / `w` / `a` / `o` keys stay on the native cv2 window on the station, where
the operator is already watching the video they are judging.
"""

from __future__ import annotations

import pathlib
import signal
import subprocess
import sys
import time

import yaml
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from station_ui import calibration as cal_state
from station_ui import data_check_config as data_check
from station_ui import devices as device_state
from station_ui import (
    enforcement,
    gripper_pairing,
    identification,
    jobs,
    launch_review,
    preview,
    readiness,
    recording,
    session_roll,
    tokens,
)
from station_ui import seal as seal_state
from station_ui import session as session_state
from station_ui.camera_access import serialize_start

# The module, same form and same reason as station_ui/routes/setup.py: the
# preset table is read per request rather than bound at import time.
from station_ui.core import setup_presets
from station_ui.errors import FieldError, StaleOwner, StationUIError, UnknownJob
from station_ui.rig import claim, is_owner, require_owner
from station_ui.routes import config as config_routes
from station_ui.workflow import progress_header, render_page
from umi.common.recording import shutdown_progress

# Re-exported so the templates, tests and `readiness` keep one spelling for
# these while the state itself lives at the package root beside `devices.py`.
RECORD_STEP = recording.RECORD_STEP
RecordingState = recording.RecordingState
WorkflowState = recording.RecordingState  # Compatibility alias for existing imports.
forget_last_recording = recording.forget_last_recording
workflow_state = recording.state_for

router = APIRouter()

SEE_OTHER = 303
BAD_REQUEST = 400
NOT_FOUND = 404
CONFLICT = 409

RECORD_SCRIPT = "real_scripts/data_collection/record_data.py"

# Allow 600 s for background alignment plus 60 s for recorder and hardware
# cleanup before SIGKILL. The ordinary job grace would interrupt saved data.
ALIGNER_DRAIN_S = 600.0
RECORD_TERM_GRACE_S = ALIGNER_DRAIN_S + 60.0

# Delay Force stop briefly to show the shutdown phase and prevent accidental
# repeat clicks. Explicit confirmation is the safeguard: force-stop can
# interrupt alignment and requires checking staged raw data afterward.
FORCE_AFTER_S = 15.0

# Briefly disable Start after the watched recorder exits. Replacing Stop
# with Start under a pending second click could otherwise launch a new take.
START_COOLDOWN_S = 5.0

# Escalate after sustained shutdown silence to catch hangs before or
# outside the self-bounded alignment drain. Keep the overall grace ceiling.
STALL_S = 300.0

# Reject blank and template-placeholder task descriptions before launch.
# A blank would prompt on closed stdin; a placeholder would label every episode.
PLACEHOLDER_TASK_DESCRIPTION = "Description"

# Keep confirmation text on one line for the hx-confirm browser dialog.
PRE_RECORD_CONFIRMATION = "Start recording?"

STOP_CONFIRMATION = (
    "Stop recording? Queued alignments finish first, which can take a moment."
)


class PreflightError(StationUIError):
    """A reason not to start, phrased for the operator.

    Every pre-flight failure is something they can fix -- an unset task label
    or a missing config -- so it renders
    as a 400 with the message next to the button, never as a 500.
    """


# ------------------------------------------------------------------ pre-flight


def recording_config_path(app) -> pathlib.Path:
    """The YAML this session records from -- the file `/config` generated, never
    the template it was generated from.

    The shipped templates carry placeholder task metadata (`task_name: Enter
    Task Name Here`, `task_description: Description`) and placeholder hardware.
    Reading one here would show the placeholder task on the record page, feed
    the placeholder description to `preflight`, and hand the recorder cameras
    that are not on this bench.

    Read the one file Apply writes (`readiness.generated_config`), the same file
    the Configuration tick and the calibration scripts key on (see also
    `calibration.recording_config_path`). Absolute, since the generated tree
    lives in the station state dir, outside the checkout -- exactly like the
    data-check config.
    """
    from station_ui import readiness

    path = readiness.generated_config(app)
    if path is None:
        raise PreflightError("Choose a setup on the Setup page first.")
    return path


def _load_config(path: pathlib.Path) -> dict:
    if not path.is_file():
        raise PreflightError(
            "Complete Configuration. It is saved when you start calibration or recording."
        )
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PreflightError(f"Could not read {path.name}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise PreflightError(f"{path.name} does not contain a YAML mapping.")
    return loaded


def _state_dir(app) -> pathlib.Path | None:
    """This app's state dir, or None when there is no app to ask.

    `preflight` takes `app` optionally so the config chain stays testable
    without one, and `output_path` reads None as "resolve the default from the
    environment" -- which is the same directory a running console would use.
    """
    settings = getattr(getattr(app, "state", None), "settings", None)
    return getattr(settings, "state_dir", None)


def _check_config_arg(state_dir, draft) -> str:
    """The data-check config Apply generated for this session, absolute.

    Never the tracked baseline: those are read-only inputs, and handing one to
    the recorder would let an operator's settings dirty the station's checkout.
    Which baseline this was built from is decided at generation time by the
    preset's operation mode, so nothing here needs to ask again.

    Built through `output_path` rather than by joining the parts here, so the
    two halves of the handoff cannot disagree about the directory or about how
    a typed task name is reduced to a filename.

    Absolute because the generated tree lives under `~/.umi_station/`, outside
    the checkout: there is no working directory that makes a relative form
    correct.
    """
    return str(data_check.output_path(state_dir, draft.task_name))


def build_argv(config_arg: str, check_arg: str):
    """The command that launches the recorder.

    Run with the server's own interpreter (`sys.executable`), not `uv run`. The
    server already lives in the project venv, so its interpreter has every
    dependency `record_data.py` imports; `uv run` would re-resolve the whole
    project on launch, which on an offline station can fail with "Network is
    unreachable" before recording starts. Same reason `audit.py`, the device scanner and the
    calibration launcher all use the interpreter directly.
    """
    return [
        sys.executable,
        RECORD_SCRIPT,
        "--config",
        config_arg,
        "--check-config",
        check_arg,
    ]


def preflight(draft, repo_root: pathlib.Path, *, app=None):
    """Everything checked before a camera is opened. Returns the argv.

    `app` is optional only so a test can exercise the config chain without one;
    every route passes it. It is what makes the first check below possible: the
    camera check on `/calibration` holds every camera on the rig open, with no
    wall-clock cap, and a recording started under it fails to open its own
    cameras with nothing on either page saying why.

    Grippers are not checked here: `record_data.py` already calls
    `assert_all_uart_readers_connected` itself, and its output is visible in
    the browser, so a second copy of the device-registry lookup would only add
    a way for the two to disagree.
    """
    # First, and before anything that reads a file: this is the only failure
    # here that is not about the config at all, and reporting it as one would
    # send the operator to /config to fix something that is not broken.
    if app is not None and identification.running(app):
        raise PreflightError(identification.MESSAGE)
    if app is not None and cal_state.check_running(app):
        raise PreflightError(device_state.CHECK_HOLDS_THE_CAMERAS)

    config_path = recording_config_path(app)
    config = _load_config(config_path)
    if app is not None:
        found, _ = device_state.outcome_for(device_state.current_job(app))
        mismatches = gripper_pairing.config_mismatches(config, found)
        if mismatches:
            raise PreflightError(
                "Correct the device selections on Configuration before recording. "
                + " ".join(mismatches)
            )
    meta = config.get("meta") or {}

    task_name = str(meta.get("task_name") or "").strip()
    # The messages name the missing field and the page that fixes it, not the
    # YAML key path (`meta.task_description`) the console reads the value from,
    # which the operator has never seen.
    if not task_name:
        raise PreflightError("Set a task name on Configuration before recording.")

    description = str(meta.get("task_description") or "").strip()
    if not description:
        raise PreflightError(
            "This task has no description. Set one on Configuration, then start recording again."
        )
    if description == PLACEHOLDER_TASK_DESCRIPTION:
        raise PreflightError(
            f"The task description is still the placeholder "
            f"{PLACEHOLDER_TASK_DESCRIPTION!r}. Set a real one on Configuration."
        )

    # Absolute because the generated tree lives outside the checkout, so it is
    # checked as it stands rather than joined onto repo_root first.
    check_arg = _check_config_arg(_state_dir(app), draft)
    if not pathlib.Path(check_arg).is_file():
        raise PreflightError(
            "Complete Configuration, then start calibration or recording to save the check settings."
        )

    return build_argv(
        # Absolute, like `check_arg`: the generated config lives in the station
        # state dir, outside the checkout, so there is no cwd that makes a
        # relative form right.
        str(config_path),
        check_arg,
    )


# ------------------------------------------------------------------- rendering

_STATUS = {
    jobs.RUNNING: ("recording", "Recording", "ok"),
    jobs.PENDING: ("pending", "Starting", "warn"),
    # Distinguish graceful cancellation from hard termination, which skips the drain.
    jobs.DONE: ("stopped", "Stopped", "ok"),
    jobs.CANCELLED: ("stopped", "Stopped — alignments drained", "ok"),
    jobs.FAILED: ("failed", "Failed", "error"),
}

# A child killed by signal N reports -N.
_HARD_KILLED = -int(signal.SIGKILL)


def status_of(job, *, live: bool = False) -> dict:
    """Icon name, label and tone for the badge.

    Three fields, not one colour: brand green and healthy green are the same
    hue on this console, so the glyph and the word carry the meaning and the
    colour only reinforces it.

    The drained/not-drained distinction is read off the exit code rather than
    asserted from the state. `_escalate` does not clear `job._cancelled`, so a
    recorder that ignored SIGTERM and was hard-killed still finishes as
    `CANCELLED` -- and a static label would show a calm green "drained" for the
    one case where the drain provably did not happen.
    """
    if job is None:
        if live:
            # An adopted recorder has no local job object but can still own hardware.
            # Use its process group to block duplicate launches.
            return {
                "icon": "recording",
                "label": "Recording",
                "tone": "ok",
                "detail": (
                    "Adopted after a console restart. This recording is still "
                    "running and can be stopped, but its output is not "
                    "available — the log died with the process that started it."
                ),
            }
        return {"icon": "idle", "label": "Not recording", "tone": "", "detail": ""}
    if job.finished and job.exit_code == _HARD_KILLED:
        return {
            "icon": "failed",
            "label": "Stopped — shutdown cut short",
            "tone": "error",
            # The one genuinely alarming state on this page, so it stays
            # operator-facing -- in terms of the consequence and the action,
            # not the console's signal handling.
            "detail": (
                "Queued alignments did not finish. Check the staging directory "
                "before recording again."
            ),
        }
    icon, label, tone = _STATUS.get(job.state, ("idle", job.state, ""))
    detail = ""
    if job.state == jobs.FAILED:
        # A bare "exit code 1" is a number with no action attached, offered as
        # the entire explanation of a failed recording. When the child died
        # without a message the operator gets the badge and the log below it.
        detail = job.error or ""
    return {"icon": icon, "label": label, "tone": tone, "detail": detail}


def _current_job(state):
    return recording.current(state)


def _context(request: Request, token, *, error: str = "") -> dict:
    rig = request.app.state.rig
    state = workflow_state(request.app)
    job = _current_job(state)
    draft = rig.draft

    config_path = None
    config = {}
    guard = ""
    pending_config = False
    try:
        config_path = recording_config_path(request.app)
        if config_path.is_file():
            config = _load_config(config_path)
        else:
            # Start submits and validates Configuration before preflight. A new
            # session intentionally drops the old file; hiding Start here would
            # remove the very action that saves the next session's config.
            config = cal_state.layout_config(request.app)
            pending_config = True
    except (PreflightError, FieldError) as exc:
        guard = str(exc)

    meta = config.get("meta") or {}
    if pending_config:
        from station_ui.core.task_metadata import task_metadata

        meta = {
            "task_name": draft.task_name,
            "task_description": task_metadata(draft.task_name).description
            or draft.task_description,
        }
    preset = (
        setup_presets.get_preset(draft.preset_id)
        if draft.preset_id in setup_presets.preset_ids()
        else None
    )

    # Configuration can be saved by Start itself. Hardware checks must already
    # be complete, and the POST rechecks all readiness after the validated save.
    blocked = readiness.blocking(
        request.app, configuration_will_be_saved=pending_config
    )
    # Every outstanding item, not just the first blocker: what is missing and
    # which steps to redo, shown before Start is pressed.
    review_items = launch_review.review(
        request.app,
        config,
        has_preset=bool(draft.preset_id),
        config_saved=config_path is not None and config_path.is_file(),
        configuration_will_be_saved=pending_config,
        placeholder_description=PLACEHOLDER_TASK_DESCRIPTION,
    )

    stopping_for = (
        None
        if state.stopping_since is None
        else time.monotonic() - state.stopping_since
    )

    # Use process-group liveness: job state is absent after adoption and can
    # be finished while descendants still own devices.
    procs = recording.survivors(request.app)
    live = bool(procs)
    blocked_for = recording.kernel_block_age(request.app, procs)
    phase = shutdown_progress.phase_from_lines(job.lines()) if job else None
    # Track time since recorder exit to prevent an accidental Start after
    # the Stop card is replaced.
    quiet_for = recording.note_liveness(request.app, live)
    return {
        "page_title": "Record",
        "active": "/record",
        "can_edit": is_owner(rig, token),
        "error": error,
        "guard": guard,
        "blocked": blocked,
        "preset": preset,
        "task_name": meta.get("task_name", ""),
        "task_description": meta.get("task_description", ""),
        # Where the harm of a stale recording date actually lands: this page is
        # the one that writes into the folder its date names. See
        # `session.recording_day_is_stale`.
        "stale_day": session_state.recording_day_is_stale(draft),
        "draft_date": draft.date,
        "confirmation": PRE_RECORD_CONFIRMATION,
        # What the recorder will actually run, read from the saved config.
        "warn_only_checks": enforcement.warn_only_titles(config) if config else [],
        "review_items": review_items,
        "stop_confirmation": STOP_CONFIRMATION,
        "job": job,
        "status": status_of(job, live=live),
        "log_lines": job.lines()[-200:] if job else [],
        # Whether a recorder is alive, from the process group. What the Stop and
        # Force stop controls render on.
        "recording_live": live,
        # Display the reported shutdown phase: teardown releases hardware; drain
        # finishes alignment and may be silent for minutes. Do not infer a phase
        # when no marker is available.
        "shutdown_phase": phase,
        "drain_started": phase in (shutdown_progress.DRAIN, shutdown_progress.DONE),
        # Seconds the group has been continuously stuck in an uninterruptible
        # kernel call, or None. Force stop cannot clear this and neither can
        # anything else -- see `recording.kernel_block_age`.
        "blocked_for": blocked_for,
        "kernel_blocked": (
            blocked_for is not None and blocked_for >= recording.KERNEL_BLOCK_S
        ),
        # Keep the recording lock while descendants survive the recorder leader.
        "stragglers": procs if (live and job is not None and job.finished) else [],
        # Poll when state can change autonomously, excluding active jobs and
        # stopping cards that already have their own refresh stream.
        "needs_watch": (
            (live or (quiet_for is not None and quiet_for < START_COOLDOWN_S))
            and not (job is not None and not job.finished)
            and not (live and stopping_for is not None)
        ),
        # The recorder this page was watching has just gone. Start is held for
        # a moment so it cannot be pressed by the second half of a double-press
        # aimed at Stop.
        "just_stopped": quiet_for is not None and quiet_for < START_COOLDOWN_S,
        "start_ready_in": (
            None if quiet_for is None else max(1, int(START_COOLDOWN_S - quiet_for) + 1)
        ),
        "step_done": RECORD_STEP in state.steps_done,
        # The staged stop. `stopping_for` is None until Stop is pressed, so the
        # panel and the Force stop control render only while stopping.
        "stopping_for": stopping_for,
        "force_ready": stopping_for is not None and stopping_for >= FORCE_AFTER_S,
        "force_remaining": (
            None if stopping_for is None else max(0, int(FORCE_AFTER_S - stopping_for))
        ),
        # Show time since output alongside Force stop: a healthy alignment can be
        # silent for up to its per-episode timeout.
        "last_output_age": (
            None if job is None else int(time.monotonic() - job.last_output_at)
        ),
        "adopted": state.adopted,
    }


def _page(request, token, *, status_code=200, error="") -> HTMLResponse:
    return render_page(
        request,
        "record",
        _context(request, token, error=error),
        token=token,
        status_code=status_code,
    )


# Mark fragment responses explicitly so the client may display their 4xx
# errors. Do not enable swapping for every error response: other routes
# return complete documents that must not be inserted into a section.
FRAGMENT_HEADER = "X-Station-Fragment"


def _fragment(request, token, *, status_code=200, error="") -> HTMLResponse:
    context = _context(request, token, error=error)
    state = request.app.state
    notice = getattr(state, "config_notice", "")
    if notice:
        draft = state.rig.draft
        context["config_update"] = {
            "notice": notice,
            "warnings": bool(getattr(state, "config_warnings", [])),
            "warned_devices": config_routes._warned_devices(
                getattr(state, "config_warnings", [])
            ),
            "folder": session_state.data_dir(draft),
            "uid": draft.session_uid,
        }
        state.config_notice = ""
        state.config_warnings = []
    response = request.app.state.templates.TemplateResponse(
        request,
        "_partials/record_status.html",
        context,
        status_code=status_code,
    )
    response.headers[FRAGMENT_HEADER] = "1"
    response.headers["X-Workflow-Progress"] = progress_header(request, token)
    # Start saved the whole page (Configuration) and re-read every step; tell the
    # page to refresh the sections that show those results.
    response.headers["HX-Trigger-After-Swap"] = "workflow-refresh"
    return response


def _respond(request, token, *, status_code=200, error=""):
    """A fragment for htmx, a redirect or a full page for a plain browser.

    The `<form>` posts for real, so an operator whose vendored htmx failed to
    load still starts and stops recordings -- they just get the whole page back
    instead of the status card.
    """
    if request.headers.get("HX-Request"):
        return _fragment(request, token, status_code=status_code, error=error)
    if status_code == 200:
        return RedirectResponse("/record", status_code=SEE_OTHER)
    return _page(request, token, status_code=status_code, error=error)


# ---------------------------------------------------------------------- routes


@router.get("/record", response_class=HTMLResponse)
async def record_page(request: Request) -> HTMLResponse:
    """Render the pre-flight summary, the live status, or both.

    Claims a free rig on load, like `/`: a read-only tab has to know it is
    read-only *before* it presses a button that starts hardware.
    """
    rig = request.app.state.rig
    token = tokens.token_from(request) or tokens.new_token()
    claim(rig, token)
    response = _page(request, token)
    tokens.remember(response, token)
    return response


@router.post("/record/start")
@serialize_start
async def start_recording(request: Request):
    rig = request.app.state.rig
    token = tokens.token_from(request)
    try:
        require_owner(rig, token)
    except StaleOwner as exc:
        return _respond(request, token, status_code=CONFLICT, error=str(exc))

    state = workflow_state(request.app)
    # Check the whole process group before launch, including adopted recorders
    # and workers surviving a crashed leader.
    if recording.in_progress(request.app):
        return _respond(
            request,
            token,
            status_code=CONFLICT,
            error="A recording is already running on this station.",
        )

    # Refuse resource conflicts before saving or rolling the session.
    if identification.running(request.app):
        return _respond(
            request, token, status_code=BAD_REQUEST, error=identification.MESSAGE
        )
    if cal_state.check_running(request.app):
        return _respond(
            request,
            token,
            status_code=BAD_REQUEST,
            error=device_state.CHECK_HOLDS_THE_CAMERAS,
        )
    if cal_state.vive_running(request.app):
        return _respond(
            request,
            token,
            status_code=BAD_REQUEST,
            error="Stop the tracker check before starting a recording.",
        )
    refused = await config_routes.save_for_action(request)
    if refused is not None:
        return refused

    # A live preview from /config holds one of the cameras the recorder opens.
    if preview.running(request.app):
        await run_in_threadpool(preview.stop, request.app, wait=preview.RELEASE_WAIT_S)

    settings = request.app.state.settings
    try:
        argv = preflight(rig.draft, settings.repo_root, app=request.app)
    except PreflightError as exc:
        return _respond(request, token, status_code=BAD_REQUEST, error=str(exc))

    # Recheck workflow readiness at the launch endpoint, after immediate
    # preflight failures but before jobs.start. Disabled buttons do not prevent
    # direct POSTs, and valid config alone does not establish calibration.
    blocked = readiness.blocking(request.app)
    if blocked is not None:
        return _respond(request, token, status_code=BAD_REQUEST, error=blocked.message)

    # Check calibration/config drift again just before launch; calibration may
    # change bundled YAML after Apply. Roll only after launch gates pass, and
    # capture the original session before rewriting its config.
    armed_before = seal_state.armed_data_dir(recording_config_path(request.app))
    rolled = session_roll.roll_generated_config(
        recording_config_path(request.app),
        _check_config_arg(_state_dir(request.app), rig.draft),
        rig.draft,
    )
    # The rule Apply already obeys: seal whatever folder the console stopped
    # naming. A folder rolled off because it was already sealed needs no
    # guard -- `write_seal` refuses to overwrite.
    if rolled:
        seal_state.seal_if_left_behind(
            armed_before,
            rolled,
            reason="config_drift",
        )

    job = jobs.start(
        state.jobs,
        argv,
        cwd=settings.repo_root,
        # A session lasts a shift. The default wall-clock cap would terminate
        # it mid-recording; `jobs.start` makes the opt-out explicit for exactly
        # this caller.
        timeout=None,
        # Long enough for the drain this whole page exists to protect.
        term_grace=RECORD_TERM_GRACE_S,
        # The backstop that actually fires: see STALL_S. The ceiling above only
        # ever catches a child that is chatty but wedged.
        stall_grace=STALL_S,
        # Send graceful SIGTERM only to the recorder parent so inherited handlers
        # in forked aligners cannot abort the drain. SIGKILL escalation still
        # targets the complete group.
        signal_group=False,
        label="record_data.py",
    )
    state.record_job_id = job.id
    state.pid = state.pgid = job.pgid
    state.adopted = False
    state.stopping_since = None
    # To disk, so a console that restarts (or crashes) can find this process
    # again. In memory it would die with the web process while the detached
    # child recorded on.
    recording.remember(request.app, job)
    # Marked done on launch, by name (see RECORD_STEP), never by position in
    # the step list.
    state.steps_done.add(RECORD_STEP)
    return _respond(request, token)


@router.post("/record/stop")
async def stop_recording(request: Request):
    """Ask the recorder to stop. Always SIGTERM, never SIGKILL.

    `jobs.cancel` signals the child's process group with SIGTERM and only
    escalates after a grace period, loudly. With
    `install_stop_on_sigterm()` in `record_data.py` that lands in the same
    `except KeyboardInterrupt` / `finally` the `q` key uses, so queued
    alignments drain before the process exits.
    """
    rig = request.app.state.rig
    token = tokens.token_from(request)
    try:
        require_owner(rig, token)
    except StaleOwner as exc:
        return _respond(request, token, status_code=CONFLICT, error=str(exc))

    state = workflow_state(request.app)
    # Stamped before the signal, so the countdown starts when the operator
    # pressed the button rather than when the child got round to noticing.
    if state.stopping_since is None:
        state.stopping_since = time.monotonic()
    job = _current_job(state)
    if job is not None and not job.finished:
        jobs.cancel(state.jobs, job.id)
    else:
        # Signal the persisted recorder leader when no active job object exists;
        # adopted recorders and surviving groups still need Stop support.
        recording.request_stop(request.app)
    return _respond(request, token)


@router.post("/record/force-stop")
async def force_stop_recording(request: Request):
    """SIGKILL the recorder's whole process group, now, and say what survived.

    Deliberately ungated. A wedged rig blocks the whole bench, this is the only
    lever that clears it, and it is also the recording lock's sole escape hatch.

    Never claims success it cannot verify. `jobs.kill_now` returns what survived
    the SIGKILL, and processes can survive it: a task in uninterruptible sleep
    from a wedged USB or DepthAI ioctl cannot be killed by anyone. Reporting the
    rig free there sends the next recording at cameras nothing can release, so
    the surviving pids are named and the answer is a 409.
    """
    rig = request.app.state.rig
    token = tokens.token_from(request)
    try:
        require_owner(rig, token)
    except StaleOwner as exc:
        return _respond(request, token, status_code=CONFLICT, error=str(exc))

    state = workflow_state(request.app)
    if state.record_job_id:
        survived = jobs.kill_now(state.jobs, state.record_job_id)
    else:
        # An adopted recording: no job object, because its output died with the
        # console that started it -- but the process group is still this
        # console's to signal.
        survived = jobs.terminate_group(state.pgid, term_wait=0.0, kill_wait=5.0)
    recording.forget(request.app)
    state.stopping_since = None
    if survived:
        pids = ", ".join(str(proc.pid) for proc in survived)
        return _respond(
            request,
            token,
            status_code=CONFLICT,
            error=(
                f"{len(survived)} process(es) survived SIGKILL (pids {pids}). "
                "They are stuck in the kernel, so this station needs a reboot "
                "before it can record again."
            ),
        )
    return _respond(request, token)


@router.post("/record/pkill")
async def pkill_recording(request: Request):
    """Nuclear option: pkill -9 -f record_data, bypassing all job machinery.

    Use when force-stop cannot reach the process (e.g. a D-state helper that
    survived SIGKILL, or any state where the UI is stuck). Always clears the
    recording lock afterward so the Start button unblocks.

    Deliberately not owner-gated: tab ownership is irrelevant when the rig is
    stuck and the operator needs to unblock it from any tab.
    """
    token = tokens.token_from(request)
    subprocess.run(["pkill", "-9", "-f", "record_data"], check=False)
    state = workflow_state(request.app)
    recording.forget(request.app)
    state.stopping_since = None
    return _respond(request, token)


@router.get("/record/stream/{job_id}")
async def stream_recording(request: Request, job_id: str):
    """The child's stdout as Server-Sent Events.

    Hand-rolled `StreamingResponse`: `sse-starlette` is only a transitive `mcp`
    dependency and `uv export --frozen --group data_collection` -- what
    `uv sync --group data_collection` installs on a station -- has neither
    it nor `websockets`. `jobs.stream` resolves the id eagerly so an unknown
    one is still a 404 rather than a 200 that ends immediately.

    Deliberately not owner-gated, unlike start and stop: watching is not
    mutating, and a read-only second tab has a legitimate reason to follow a
    recording it is not allowed to stop.
    """
    state = workflow_state(request.app)
    try:
        frames = jobs.stream(state.jobs, job_id)
    except UnknownJob:
        return HTMLResponse("no such job", status_code=NOT_FOUND)

    async def guarded():
        # Read blocking job frames in a worker thread and stop on client disconnect.
        # Otherwise an unfinished job's keepalives can hold SSE open through shutdown.
        async for frame in iterate_in_threadpool(frames):
            if await request.is_disconnected():
                break
            yield frame

    return StreamingResponse(
        guarded(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Nothing may sit on this and buffer it into one delivery at exit.
            "X-Accel-Buffering": "no",
        },
    )

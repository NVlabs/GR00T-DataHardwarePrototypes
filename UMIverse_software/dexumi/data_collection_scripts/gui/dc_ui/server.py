# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DexUMI data-collection GUI server: thin FastAPI routes over gui.dc_ui.core.

    dexumi_dc [--host 127.0.0.1] [--port 8791] [--home ~/.cache/umi/dc_ui] \
        [--data-root <repo>/output] [--scan-every-launch] \
        [--no-open]

`uv sync` at the repository root installs `dexumi_dc`; `./launch.sh gui` starts
the same server. It opens the GUI in a browser unless given --no-open.

Launch it from a shell on the station's X session: record_data.py opens a cv2
window (DISPLAY is inherited) and takes its s / w / q keys there. The recorder
dies with the GUI (atexit / SIGTERM -> Supervisor.stop_all); a pidfile lets a
restarted GUI adopt a still-running recorder.

HTTP status convention: 400 = malformed request (unknown form
field / a Host header that is not this GUI); 403 = a state-changing
POST the browser reports as cross-site; 409 = state refuses it (no plan yet,
recorder already running, devices requested while the recorder owns them);
422 = the plan is not launchable (preflight blockers -> the full plan payload).
"""

import atexit
import errno
import os
import signal
import socket
import sys
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import click
import uvicorn
import yaml
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
)

from gui.dc_ui.core import (
    calibration,
    composer,
    device_binding,
    lerobot,
    preflight,
    session_dir,
)
from gui.dc_ui.core.camera_pairing import (
    CameraSession,
    board_png,
    check_streams,
    identify_pair,
    identify_pair_live,
    pairing_candidates,
)
from gui.dc_ui.core.commands import (
    RECORDER_BINDING_REFUSED_EXIT,
    recorder_spec,
    shell_line,
)
from gui.dc_ui.core.devices import (
    discover_devices,
    from_payload,
    payload,
)
from gui.dc_ui.core.kinds import (
    DEFAULT_DATA_ROOT,
    RECORDER_ROLE,
    REPO_ROOT,
    DcForm,
    Finding,
)
from gui.dc_ui.core.runtime_home import RuntimeHomeInUseError, claim_runtime_home
from gui.dc_ui.core.session import DcSession, json_safe
from gui.dc_ui.core.supervisor import StartRefused, Supervisor
from umi.common.recording.session_slot import (
    LAUNCH_CONFIG_NAME,
    find_session_slot,
    folder_has_episode_history,
    launch_config_changes,
    session_file_conflicts,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
ENV_HOME = "UMI_DC_UI_HOME"
ENV_STATION_ID = "UMI_STATION_ID"
DEFAULT_HOME = "~/.cache/umi/dc_ui"
DEFAULT_PORT = 8791
LOG_TAG = "dc_ui"
STATE_CHANGING_METHODS = ("POST", "PUT", "PATCH", "DELETE")
# Allow loopback hosts and the configured bind host. A wildcard bind also
# allows the machine hostname.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
WILDCARD_HOSTS = ("0.0.0.0", "::", "")
SAME_ORIGIN_FETCH_SITES = ("same-origin", "none")
# These findings depend on discovery, which Start refreshes. A stale failure
# must not prevent retrying after the operator reconnects or recalibrates.
REFRESHABLE_DEVICE_FINDINGS = {
    "detection_missing",
    "device_not_connected",
    "exo_side_mismatch",
    "exo_config_invalid",
}


def default_home():
    return os.path.abspath(os.path.expanduser(os.environ.get(ENV_HOME) or DEFAULT_HOME))


def station_id():
    return (os.environ.get(ENV_STATION_ID) or "").strip() or None


def display():
    return (os.environ.get("DISPLAY") or "").strip() or None


def make_supervisor(home=None):
    return Supervisor(
        home or default_home(),
        roles=(RECORDER_ROLE, lerobot.LEROBOT_ROLE),
        stop_order=(RECORDER_ROLE, lerobot.LEROBOT_ROLE),
        log_tag=LOG_TAG,
    )


# Host validation


def host_name(value):
    value = (value or "").strip()
    if not value:
        return ""
    if "://" not in value:
        value = "//" + value
    try:
        return (urlsplit(value).hostname or "").lower()
    except ValueError:
        return ""


def is_loopback_host(host):
    return host_name(host) in LOOPBACK_HOSTS


def allowed_hosts(host):
    allowed = set(LOOPBACK_HOSTS)
    if str(host or "").strip().strip("[]") in WILDCARD_HOSTS:
        allowed.add(socket.gethostname().lower())
        return frozenset(allowed)
    bound = host_name(host)
    if bound:
        allowed.add(bound)
    return frozenset(allowed)


def _trusted_request(request: Request):
    allowed = request.app.state.ui.allowed_hosts
    raw_host = request.headers.get("host")
    if host_name(raw_host) not in allowed:
        raise HTTPException(
            400,
            f"Host {raw_host!r} is not this GUI (allowed: {', '.join(sorted(allowed))}"
            "); refused (DNS-rebinding guard; a remote bind needs --allow-remote)",
        )
    if request.method not in STATE_CHANGING_METHODS:
        return
    site = request.headers.get("sec-fetch-site")
    if site and site not in SAME_ORIGIN_FETCH_SITES:
        raise HTTPException(
            403, f"cross-site {request.method} refused (Sec-Fetch-Site: {site})"
        )
    origin = request.headers.get("origin")
    if origin and host_name(origin) not in allowed:
        raise HTTPException(
            403,
            f"cross-origin {request.method} refused (Origin: {origin}); this API is "
            f"only for the page it serves on {', '.join(sorted(allowed))}",
        )


# -- state -------------------------------------------------------------------------------


class UiState:
    def __init__(
        self,
        session,
        supervisor,
        data_root=None,
        base_config=None,
        host="127.0.0.1",
        scan_every_launch=False,
        calibration_runner=None,
    ):
        self.session = session
        # First-time glove calibration: a child of its own, never alongside the recorder
        # (both open the glove boards).
        self.calibration = calibration_runner or calibration.CalibrationRunner()
        self.supervisor = supervisor
        self.home = supervisor.home
        # Camera identification is scoped to this GUI process; restarting requires a
        # new scan. Each launch compares selected device IDs against the scan.
        # With --scan-every-launch, a ready launch consumes the scan; startup failure
        # allows a retry. A device-binding refusal always invalidates the scan.
        # Only selections that all match the scan receive a device binding.
        self.identified = {}
        self.scan_every_launch = bool(scan_every_launch)
        # Reason the scan was invalidated: a ready launch consumed it, or device
        # binding was refused. A successful new scan clears both fields.
        self.identification_cleared_at = None
        self.identification_cleared_by = None
        # The recorder process launched with the current scan, until it
        # became ready or exited (settled by _settle_identification).
        self.identification_launch = None
        self.gui_run = device_binding.new_gui_run()
        self.data_root = data_root or DEFAULT_DATA_ROOT
        self.base_config = base_config or composer.BASE_CONFIG
        self.allowed_hosts = allowed_hosts(host)
        self.lock = threading.Lock()
        # Enumeration and identification must reserve the cameras atomically
        # with recorder launch. Keep the reservation through worker cleanup.
        self.camera_operation = None
        self.camera_diagnostics = []
        self.pairing_cancel = None
        self.camera_done = threading.Event()
        self.camera_done.set()
        # The Identify-cameras dialog keeps every connected camera open in a
        # preview session (live thumbnails, scans without rebooting). It holds
        # the same reservation as a one-shot scan until it is closed.
        self.camera_session = None
        self.shutting_down = False
        # Keep DetectedDevice objects for preflight checks and their JSON form for
        # persisted state.
        self.detection = from_payload(session.detection)

    def recorder_status(self):
        return self.supervisor.status_all()[RECORDER_ROLE]

    def converting(self):
        """A LeRobot conversion (step 05) runs, or has exited but its exit is not
        recorded yet: the page reads the outcome once this turns false."""
        proc = self.supervisor.procs.get(lerobot.LEROBOT_ROLE)
        if proc is None:
            return False
        if proc.is_alive() or proc._group_alive():
            return True
        return proc.status()["state"] in ("starting", "ready", "adopted", "stopping")

    def recorder_live(self):
        proc = self.supervisor.procs.get(RECORDER_ROLE)
        # A failure banner changes the displayed status before the process has
        # necessarily released its devices. Ownership lasts through group exit.
        return proc is not None and (proc.is_alive() or proc._group_alive())


def _form_from(body):
    try:
        form = DcForm.from_dict(body or {})
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return form


def _recorder_slot(form, base, folder):
    """(slot, cfg): the folder record_data.py records into when launched on ``folder``."""
    cfg = composer.compose_config(form, base, folder)
    return find_session_slot(cfg, base_dir=str(REPO_ROOT)), cfg


def _closed_reason(form, base, slot, now=None):
    """Why the recorder must not append to ``slot`` although the naming rule continues it, or None.

    Appending would destroy a take the recorder kept for recovery (it reuses
    that episode ID) or grow a session that may already have been processed.
    """
    if not folder_has_episode_history(slot):
        return None
    pending = session_dir.pending_takes(slot)
    if pending:
        return f"holds an unfinished take kept for recovery ({', '.join(pending)})"
    age = session_dir.last_take_age(slot, now)
    if age is None:
        return "has no finished take left to continue"
    if age >= session_dir.IDLE_CLOSE_S:
        return (
            f"has had no new take for {session_dir.age_text(age)}; a session idle "
            "for 1 h is treated as finished"
        )
    return None


def _slot_changes(folder, cfg):
    """What differs between ``folder``'s recorded config and this launch, as the page names it."""
    expanded = os.path.expanduser(folder)
    changes = []
    for name in session_file_conflicts(expanded, cfg, base_dir=str(REPO_ROOT)):
        if name == LAUNCH_CONFIG_NAME:
            changes += launch_config_changes(expanded, cfg) or ["recording config"]
        else:
            changes.append(name)
    return changes


def _default_folder(state, form, base, root):
    """(default_dir, notes, error) for the form's task.

    session_dir.next_session picks hw{N} (kept while both hands keep the
    calibration hw{N} was recorded with), then the recorder's own
    session_slot.find_session_slot names the folder record_data.py will use: a
    ``_1`` copy when hw{N}'s folder holds another launch config. A folder that
    must not grow (_closed_reason) moves the default to hw{N+1}.
    """
    if not form.task_name:
        return "", [], None
    token = session_dir.rig_token(composer.exo_device_names(form, base))
    probe = composer.compose_config(form, base, "")
    date = session_dir.today()
    try:
        choice = session_dir.next_session(
            root, date, form.task_name, token, composer.exo_calibration_files(probe)
        )
    except OSError as exc:
        error = Finding(
            "error", "data_root_unreadable", f"Cannot inspect data root {root!r}: {exc}"
        )
        return "", [], error
    try:
        slot, cfg = _recorder_slot(form, base, choice.path)
        reason = slot and _closed_reason(form, base, slot)
        if reason:
            choice = session_dir.start_next(
                choice, root, date, form.task_name, token, slot, reason
            )
            slot, cfg = _recorder_slot(form, base, choice.path)
        changes = (
            _slot_changes(choice.path, cfg) if slot and slot != choice.path else []
        )
    except Exception as exc:  # e.g. a corrupt snapshot; record_data.py would fail too
        message = (
            f"Cannot compare {choice.path} with this launch: {exc}. Check the "
            "session folder permissions and saved configuration, then retry."
        )
        notes = [Finding("info", "session_index", choice.message)]
        return (
            choice.path,
            notes,
            Finding("error", "session_folder_unreadable", message),
        )
    notes = [Finding("info", "session_index", choice.message)]
    if slot is None:
        message = (
            f"No free _N copy of {choice.path} is left; a config field may change on "
            "every launch. Check the recording configuration before retrying."
        )
        return choice.path, notes, Finding("error", "session_slot_full", message)
    if slot != choice.path:
        what = ", ".join(changes[:4]) + (", …" if len(changes) > 4 else "")
        verb = "continues" if folder_has_episode_history(slot) else "starts"
        message = (
            f"{os.path.basename(choice.path)} was recorded with a different "
            f"{what or 'configuration'}; the recorder {verb} "
            f"{os.path.basename(slot)}, still hw{choice.index}."
        )
        notes.append(Finding("info", "session_slot", message))
    return slot, notes, None


def _compose(state, form):
    """form -> (cfg, data_dir, findings, default_dir).

    Both folder values come from _default_folder, recomputed every time. Only
    automatic naming can select an existing session for continuation.
    """
    root = state.data_root
    base = composer.load_base(state.base_config)
    default_dir, notes, directory_error = _default_folder(state, form, base, root)
    data_dir = default_dir
    cfg = composer.compose_config(form, base, data_dir)
    findings = preflight.preflight(
        cfg,
        state.detection,
        base_config=state.base_config,
        # The default continuing hw{N} appends to its folder by design.
        resume=bool(data_dir) and folder_has_episode_history(data_dir),
    )
    _settle_identification(state)
    findings += notes
    findings.sort(key=lambda f: preflight.LEVEL_ORDER.get(f.level, 9))
    if directory_error:
        findings.insert(0, directory_error)
    return cfg, data_dir, findings, default_dir


def _record_identification(state, side, result, detection):
    """Remember what the scan assigned, with the raw device id it opened.

    The other side loses a camera it reused. A result without ``identities``
    (an older or stubbed scan) takes the id discovery reported for the name.
    """
    other = "right" if side == "left" else "left"
    assignments = (result or {}).get("assignments") or {}
    identities = (result or {}).get("identities") or {}
    scanned_at = device_binding.now_iso()
    with state.lock:
        # Settle the completed launch before inspecting the current scan so it
        # cannot invalidate identification from a newer launch.
        _settle_identification(state)
        for kind in ("iphone", "oak"):
            name = assignments.get(f"{side}_{kind}")
            if not name:
                continue
            identity = identities.get(f"{side}_{kind}") or {}
            device_id = (
                identity.get("device_id") if identity.get("name") == name else None
            )
            if device_id is None:
                device = device_binding.detected_device(detection, kind, name)
                device_id = device.device_id if device is not None else None
            reused = state.identified.get(f"{other}_{kind}")
            if reused and (
                reused["name"] == name
                or device_binding.same_id(reused.get("device_id"), device_id)
            ):
                del state.identified[f"{other}_{kind}"]
            state.identified[f"{side}_{kind}"] = {
                "kind": kind,
                "name": name,
                "device_id": device_id,
                "scanned_at": scanned_at,
            }
            state.identification_cleared_at = None
            state.identification_cleared_by = None


def _clear_identification(state, reason):
    """Drop the scan; ``reason`` is "launch" or "refusal". Caller holds state.lock."""
    state.identification_launch = None
    if not state.identified:
        return
    state.identified = {}
    state.identification_cleared_at = time.time()
    state.identification_cleared_by = reason


def _hold_identification_for(state, proc):
    """Remember the recorder launched with the current scan. Caller holds state.lock.

    ``_settle_identification`` decides what the launch does to the scan: the
    recorder's binding refusal clears it in every mode; with
    --scan-every-launch reaching the ready banner uses it up, while a recorder
    that fails to start (no stream, a crash) or is stopped before ready leaves
    it for the retry.
    """
    if not state.identified or proc is None:
        state.identification_launch = None
        return
    state.identification_launch = proc
    _settle_identification(state)


def _launch_exit_code(proc):
    code = proc.exit_code
    if code is None and getattr(proc, "_proc", None) is not None:
        code = proc._proc.poll()
    return code


def _settle_identification(state):
    """Apply the outcome of the launch that used the scan. Caller holds state.lock.

    The recorder's own refusal (exit RECORDER_BINDING_REFUSED_EXIT, before its
    ready banner: a scanned device changed, so the scan is stale) clears the
    scan in every mode. Reaching the ready banner (``ready_at`` set) uses it up
    with --scan-every-launch and keeps it otherwise. Any other end before
    ready keeps it; a launch still starting leaves everything as is.
    """
    proc = state.identification_launch
    if proc is None:
        return
    if proc.ready_at is not None:
        if state.scan_every_launch:
            _clear_identification(state, "launch")
        else:
            state.identification_launch = None
        return
    if proc.is_alive() or proc._group_alive():
        return
    if _launch_exit_code(proc) == RECORDER_BINDING_REFUSED_EXIT:
        _clear_identification(state, "refusal")
        return
    state.identification_launch = None


def _identification_payload(state):
    """What the page needs about the identification record. Caller holds state.lock."""
    _settle_identification(state)
    return {
        "identified": {slot: r["name"] for slot, r in state.identified.items()},
        "identified_devices": {slot: dict(r) for slot, r in state.identified.items()},
        "scan_every_launch": state.scan_every_launch,
        "identification_cleared_at": state.identification_cleared_at,
        "identification_cleared_by": state.identification_cleared_by,
    }


def _preview(state, cfg, data_dir, findings, default_dir=None):
    return {
        "config": cfg,
        "data_dir": data_dir,
        # The computed folder the page offers as default (None for a saved plan).
        "default_data_dir": default_dir,
        "active_slots": list(composer.active_slots(cfg)),
        "findings": [f.to_dict() for f in findings],
        "blocked": preflight.has_blockers(findings),
        "yaml": _yaml_text(cfg, state.base_config),
    }


def _yaml_text(cfg, base_config):
    header = (
        f"# Generated by dc_ui (DexUMI data collection GUI) from {base_config} "
        "-- preview; the launched file is written under <home>/generated/.\n"
    )
    return header + yaml.safe_dump(cfg, sort_keys=False)


def _plan(state, form):
    """Compose, preflight, write the yaml (+ device binding), build the spec, save.

    Caller holds state.lock. Returns (payload, blocked). A blocked plan keeps
    the form in the session so the operator can fix it after a reload, and
    writes no yaml. The binding is written only when every selected
    wrist/tracking camera comes from a current scan (device_binding.py).
    """
    cfg, data_dir, findings, default_dir = _compose(state, form)
    body = _preview(state, cfg, data_dir, findings, default_dir)
    state.session.update(form=form, findings=findings, data_dir=data_dir or None)
    if body["blocked"]:
        state.session.update(config_path=None, spec=None)
        state.session.save()
        body.update({"config_path": None, "spec": None, "command": None})
        return body, True
    stem = os.path.basename(data_dir)
    config_path = composer.write_generated_config(
        cfg, state.home, stem, base_config=state.base_config
    )
    scanned = device_binding.scanned_sources(
        preflight.selected_devices(cfg), state.identified, state.detection
    )
    binding = (
        device_binding.write_binding(config_path, scanned, state.gui_run)
        if scanned
        else None
    )
    spec = recorder_spec(config_path, device_binding=binding)
    state.session.update(config_path=config_path, spec=spec)
    state.session.save()
    body.update(
        {
            "config_path": config_path,
            "device_binding": binding,
            "spec": spec.to_dict(),
            "command": shell_line(spec),
        }
    )
    return body, False


def _refuse_if_live(state, what):
    if state.recorder_live():
        raise HTTPException(
            409,
            f"the recorder is running (pid {state.recorder_status()['pid']}); {what}",
        )


def _saved_plan(state):
    session = state.session
    if not session.config_path or not session.spec:
        return None
    try:
        with open(session.config_path) as f:
            cfg = yaml.safe_load(f)
    except (OSError, ValueError, yaml.YAMLError):
        return None
    if not isinstance(cfg, dict):
        return None
    body = _preview(state, cfg, session.data_dir, session.findings)
    argv = session.spec.argv
    flag = "--device-binding"
    body["device_binding"] = (
        argv[argv.index(flag) + 1]
        if flag in argv and argv.index(flag) + 1 < len(argv)
        else None
    )
    body.update(
        config_path=session.config_path,
        spec=session.spec.to_dict(),
        command=shell_line(session.spec),
    )
    return body


def _camera_blocked_plan(
    state, form, cfg, data_dir, findings, cancelled=False, default_dir=None
):
    state.session.update(
        form=form,
        findings=findings,
        data_dir=data_dir or None,
        config_path=None,
        spec=None,
    )
    state.session.save()
    result = _preview(state, cfg, data_dir, findings, default_dir)
    result.update(
        config_path=None,
        spec=None,
        command=None,
        camera_diagnostics=state.camera_diagnostics,
        cancelled=cancelled,
    )
    return JSONResponse(status_code=422, content=json_safe(result))


def _refuse_if_calibrating(state):
    if state.calibration.active():
        raise HTTPException(
            409,
            "a glove calibration is running; wait for it to finish or cancel it",
        )


def _refuse_if_camera_busy(state):
    if state.shutting_down:
        raise HTTPException(409, "the GUI is shutting down")
    if state.camera_operation:
        raise HTTPException(
            409,
            f"camera {state.camera_operation} is still running; wait for it to finish",
        )


def _release_cameras(state):
    """Caller holds state.lock and has proven every camera worker exited."""
    state.pairing_cancel = None
    state.camera_operation = None
    state.camera_done.set()


def _close_camera_session(state, session=None):
    """Close the preview session (blocking until its workers are reaped) and
    release the camera reservation. Idempotent; safe from several threads."""
    with state.lock:
        session = session or state.camera_session
        if session is None or state.camera_session is not session:
            return False
        if state.pairing_cancel is not None:
            state.pairing_cancel.set()  # a scan in flight returns as cancelled
    session.close()
    with state.lock:
        if state.camera_session is session:
            state.camera_session = None
            _release_cameras(state)
    return True


def _watch_camera_session(state, session):
    """Release the cameras when the page stops polling or the GUI shuts down."""
    while True:
        time.sleep(1.0)
        with state.lock:
            current = state.camera_session is session and not session.closed
            shutting_down = state.shutting_down
        if not current:
            return
        if shutting_down or session.expired():
            _close_camera_session(state, session)
            return


# -- app ---------------------------------------------------------------------------------


def create_app(
    session=None,
    supervisor=None,
    data_root=None,
    base_config=None,
    host="127.0.0.1",
    scan_every_launch=False,
    calibration_runner=None,
):
    supervisor = supervisor or make_supervisor()
    session = session or DcSession.load(supervisor.home)
    state = UiState(
        session,
        supervisor,
        data_root=data_root,
        base_config=base_config,
        host=host,
        scan_every_launch=scan_every_launch,
        calibration_runner=calibration_runner,
    )

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            with state.lock:
                state.shutting_down = True
                if state.pairing_cancel:
                    state.pairing_cancel.set()
            await run_in_threadpool(_close_camera_session, state)
            await run_in_threadpool(state.camera_done.wait)
            await run_in_threadpool(state.calibration.shutdown)
            await run_in_threadpool(supervisor.stop_all)

    app = FastAPI(
        title="DexUMI data collection GUI",
        docs_url=None,
        redoc_url=None,
        dependencies=[Depends(_trusted_request)],
        lifespan=lifespan,
    )
    app.state.ui = state
    _static_routes(app)
    _api_routes(app, state)
    _calibration_routes(app, state)
    _camera_routes(app, state)
    _process_routes(app, state)
    return app


def _static_routes(app):
    @app.get("/static/{filename:path}")
    def static_file(filename: str):
        path = (STATIC_DIR / filename).resolve()
        if not path.is_relative_to(STATIC_DIR) or not path.is_file():
            raise HTTPException(404, "Not found")
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(
            (STATIC_DIR / "index.html").read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-cache"},
        )


def _calibration_payload(state, since=0):
    runner = state.calibration
    # A plugged-in glove that only the library records gets its saved calibration
    # loaded first, so this page and the device list (Step 02) agree on it.
    if not state.recorder_live():
        loaded = runner.activate_saved()
    else:
        loaded = []
    gloves = [calibration.glove_status(side) for side in calibration.SIDES]
    boards = runner.boards()
    return {
        "tool_version": calibration.TOOL_VERSION,
        "gloves": gloves,
        "boards": boards,
        # Calibrations of gloves that are not plugged in; they load when it is.
        "saved": calibration.saved_gloves([b["serial"] for b in boards]),
        "loaded": loaded,
        "recorder_live": state.recorder_live(),
        "job": runner.snapshot(since),
    }


def _calibration_routes(app, state):
    @app.get("/api/calibration")
    def api_calibration(since: int = 0):
        return _calibration_payload(state, since)

    @app.post("/api/calibration/start")
    def api_calibration_start(body: dict | None = None):
        side = str((body or {}).get("side") or "")
        recalibrate = bool((body or {}).get("recalibrate"))
        # Held across the start so a recorder launch cannot slip in between the check
        # and the calibration taking the glove boards.
        with state.lock:
            if state.shutting_down:
                raise HTTPException(409, "the GUI is shutting down")
            _refuse_if_live(state, "calibrating would fight it for the glove boards")
            try:
                state.calibration.start(side, recalibrate=recalibrate)
            except calibration.CalibrationRefused as exc:
                raise HTTPException(
                    409,
                    {"message": str(exc), "code": exc.code} if exc.code else str(exc),
                ) from exc
        return _calibration_payload(state)

    @app.post("/api/calibration/confirm")
    def api_calibration_confirm():
        try:
            state.calibration.confirm()
        except calibration.CalibrationRefused as exc:
            raise HTTPException(409, str(exc)) from exc
        return _calibration_payload(state)

    @app.post("/api/calibration/save")
    def api_calibration_save(body: dict | None = None):
        try:
            state.calibration.save(bool((body or {}).get("save")))
        except calibration.CalibrationRefused as exc:
            raise HTTPException(409, str(exc)) from exc
        return _calibration_payload(state)

    @app.post("/api/calibration/cancel")
    def api_calibration_cancel():
        state.calibration.cancel()
        return _calibration_payload(state)


def _api_routes(app, state):
    @app.get("/api/state")
    def api_state():
        # Replace form, findings, path and launch spec atomically so readers see one
        # plan.
        with state.lock:
            return {
                "session": state.session.snapshot(),
                "plan": _saved_plan(state),
                "live": state.recorder_live(),
                "detection": payload(state.detection) if state.detection else None,
                "procs": state.supervisor.status_all(),
                "home": state.home,
                "identifying": state.pairing_cancel is not None,
                "camera_operation": state.camera_operation,
                "camera_preview": state.camera_session is not None,
                # Seconds since the dialog last polled; a reloaded page uses
                # this to tell an orphaned session from one another tab drives.
                "camera_preview_idle_s": (
                    None
                    if state.camera_session is None
                    or state.camera_session.last_poll is None
                    else round(time.monotonic() - state.camera_session.last_poll, 2)
                ),
                "camera_diagnostics": state.camera_diagnostics,
                **_identification_payload(state),
                "defaults": {
                    "data_root": state.data_root,
                    "base_config": state.base_config,
                    "today": session_dir.today().isoformat(),
                    "active_slots": list(
                        composer.active_slots(composer.load_base(state.base_config))
                    ),
                    "station": station_id() or socket.gethostname(),
                    "display": display(),
                },
            }

    @app.get("/api/devices")
    def api_devices():
        with state.lock:
            _refuse_if_live(state, "device enumeration would fight it for the cameras")
            _refuse_if_camera_busy(state)
            # Load known calibrations before listing connected gloves.
            state.calibration.activate_saved()
            state.camera_operation = "enumeration"
            state.camera_done.clear()
        try:
            result = discover_devices(20)
            with state.lock:
                state.detection = result
                state.session.update(detection=payload(result))
                state.session.save()
        finally:
            with state.lock:
                state.camera_operation = None
                state.camera_done.set()
        return payload(result)

    @app.post("/api/compose")
    def api_compose(body: dict | None = None):
        form = _form_from(body)
        with state.lock:
            cfg, data_dir, findings, default_dir = _compose(state, form)
        return _preview(state, cfg, data_dir, findings, default_dir)

    @app.post("/api/plan")
    def api_plan(body: dict | None = None):
        form = _form_from(body)
        with state.lock:
            _refuse_if_live(state, "stop it before planning another session")
            _refuse_if_camera_busy(state)
            result, blocked = _plan(state, form)
        if blocked:
            return JSONResponse(status_code=422, content=json_safe(result))
        return result


def _camera_routes(app, state):
    def valid_side(side):
        if side not in ("left", "right"):
            raise HTTPException(400, "side must be 'left' or 'right'")
        return side

    @app.get("/api/cameras/board/{side}.png")
    def camera_board(side: str):
        return Response(
            board_png(valid_side(side)),
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    def _real_camera_slots():
        active = composer.active_slots(composer.load_base(state.base_config))
        return {slot for slot in active if slot.endswith(("_iphone", "_oak"))}

    @app.post("/api/cameras/preview/start")
    def camera_preview_start():
        """Open every connected iPhone/OAK in preview workers for the dialog.

        Holds the camera reservation until /api/cameras/preview/stop, cancel,
        the idle watchdog or shutdown closes the session. Returns as soon as
        the workers are spawned; the page polls /api/cameras/preview.
        """
        with state.lock:
            _refuse_if_live(state, "stop it before identifying cameras")
            _refuse_if_camera_busy(state)
            if not _real_camera_slots():
                raise HTTPException(409, "the base config has no real camera slots")
            cancel = threading.Event()
            state.pairing_cancel = cancel
            state.camera_operation = "identification"
            state.camera_done.clear()
        session = None
        try:
            detection = discover_devices(20)
            with state.lock:
                state.detection = detection
                state.session.update(detection=payload(detection))
                state.session.save()
                if cancel.is_set() or state.shutting_down:
                    raise HTTPException(409, "camera preview was cancelled")
                candidates, excluded, errors = pairing_candidates(detection)
                session = CameraSession(
                    candidates, preview=True, excluded=excluded, errors=errors
                ).start()
                state.camera_session = session
            threading.Thread(
                target=_watch_camera_session,
                args=(state, session),
                name="camera-preview-watchdog",
                daemon=True,
            ).start()
            return {"session": session.snapshot(), "detection": payload(detection)}
        except BaseException:
            if session is not None:
                _close_camera_session(state, session)
            with state.lock:
                if state.camera_session is None:
                    _release_cameras(state)
            raise

    @app.get("/api/cameras/preview")
    def camera_preview():
        with state.lock:
            session = state.camera_session
        if session is None:
            return {"active": False, "camera_operation": state.camera_operation}
        session.touch()
        return {"active": True, **session.snapshot()}

    @app.get("/api/cameras/preview/{kind}/{device_id}.jpg")
    def camera_preview_image(kind: str, device_id: str):
        with state.lock:
            session = state.camera_session
        jpeg = session.preview_jpeg(kind, device_id) if session is not None else None
        if jpeg is None:
            raise HTTPException(404, "no preview frame for this camera")
        return Response(
            jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"}
        )

    @app.post("/api/cameras/preview/stop")
    def camera_preview_stop():
        # Blocks until every worker has exited: the page may start the
        # recorder as soon as this returns.
        closed = _close_camera_session(state)
        return {"stopped": closed, "camera_operation": state.camera_operation}

    @app.post("/api/cameras/identify")
    def camera_identify(body: dict | None = None):
        side = valid_side((body or {}).get("side"))
        with state.lock:
            session = state.camera_session
            if session is not None:
                # Scan on the live preview workers; the session keeps the
                # reservation, so nothing is released here.
                if state.shutting_down:
                    raise HTTPException(409, "the GUI is shutting down")
                if session.scanning:
                    raise HTTPException(
                        409, f"a {session.scanning} scan is already running"
                    )
                if not session.settled():
                    raise HTTPException(
                        409,
                        "cameras are still starting; wait until every camera "
                        "shows live or an error",
                    )
                with session.lock:
                    session.scanning = side
                cancel = state.pairing_cancel
                detection = state.detection
            else:
                _refuse_if_live(state, "stop it before identifying cameras")
                _refuse_if_camera_busy(state)
                if not {f"{side}_iphone", f"{side}_oak"}.issubset(_real_camera_slots()):
                    raise HTTPException(
                        409, f"the base config has no real {side} camera pair"
                    )
                cancel = threading.Event()
                state.pairing_cancel = cancel
                state.camera_operation = "identification"
                state.camera_done.clear()
        if session is not None:
            result = identify_pair_live(session, side, cancel)
            _record_identification(state, side, result, detection)
            return {**result, "detection": payload(detection) if detection else None}
        try:
            # A newly attached or disconnected camera must participate in the
            # ambiguity check. Cached discovery is only a UI convenience.
            detection = discover_devices(20)
            with state.lock:
                state.detection = detection
                state.session.update(detection=payload(detection))
                state.session.save()
            result = (
                {
                    "side": side,
                    "assignments": {},
                    "cameras": [],
                    "errors": [],
                    "cancelled": True,
                }
                if cancel.is_set()
                else identify_pair(detection, side, cancel)
            )
            _record_identification(state, side, result, detection)
            return {**result, "detection": payload(detection)}
        finally:
            # identify_pair returns only once all camera workers have exited,
            # even when a timeout, cancellation or SDK failure occurred. This
            # synchronous route runs entirely in FastAPI's threadpool: HTTP
            # task cancellation cannot run this finally ahead of the probes.
            with state.lock:
                _release_cameras(state)

    @app.post("/api/cameras/identify/cancel")
    def camera_cancel():
        with state.lock:
            if state.pairing_cancel:
                state.pairing_cancel.set()
            cancelling = state.pairing_cancel is not None
            session = state.camera_session
        if session is not None:
            # Close camera workers in the background so cancelling the dialog does not
            # block the HTTP request.
            threading.Thread(
                target=_close_camera_session,
                args=(state, session),
                name="camera-preview-cancel",
                daemon=True,
            ).start()
        return {"cancelling": cancelling}


def _process_routes(app, state):
    sup = state.supervisor

    def _role(role):
        if role not in sup.roles:
            raise HTTPException(404, f"unknown role {role!r}; roles: {list(sup.roles)}")

    @app.get("/api/procs")
    def api_procs():
        expected = {RECORDER_ROLE: state.session.spec} if state.session.spec else None
        statuses = sup.status_all(expected)
        with state.lock:
            identification = _identification_payload(state)
        return {
            "procs": statuses,
            "live": state.recorder_live(),
            "converting": state.converting(),
            "identifying": state.pairing_cancel is not None,
            "camera_operation": state.camera_operation,
            "camera_preview": state.camera_session is not None,
            "camera_diagnostics": state.camera_diagnostics,
            # Let the page detect when a launch consumed or invalidated camera
            # identification.
            **identification,
        }

    @app.post("/api/procs/{role}/start")
    def api_start(role: str, body: dict | None = None):
        _role(role)
        if role != RECORDER_ROLE:
            raise HTTPException(404, f"start {role} through its own step")
        token = sup.capture_launch_token()
        # The page sends the form it shows (and validated via /api/compose);
        # a body-less call (curl, CLI) launches the last generated plan.
        form = _form_from(body) if body else None
        with state.lock:
            _refuse_if_camera_busy(state)
            _refuse_if_calibrating(state)
            form = form or state.session.form
            if form is None:
                raise HTTPException(
                    409, "no plan yet: fill the form and build a plan first"
                )
            _refuse_if_live(state, "stop it before starting another")
            if state.converting():
                raise HTTPException(
                    409,
                    "a LeRobot conversion is running; wait for it to finish or "
                    "stop it in step 05 before recording",
                )
            cfg, data_dir, findings, default_dir = _compose(state, form)
            state.camera_diagnostics = []
            if preflight.has_blockers(
                [f for f in findings if f.code not in REFRESHABLE_DEVICE_FINDINGS]
            ):
                return _camera_blocked_plan(
                    state, form, cfg, data_dir, findings, default_dir=default_dir
                )
            selected = preflight.selected_devices(cfg)
            cameras = {
                slot: name
                for slot, name in selected.items()
                if slot.endswith(("_iphone", "_oak"))
            }
            if not cameras:
                return launch(form, token)
            # Keep the entire discovery -> probe -> launch transition exclusive.
            # The probes prove video is arriving, then release every camera
            # before the real recorder opens it with the unchanged base config.
            cancel = threading.Event()
            state.session.update(
                form=form,
                findings=findings,
                data_dir=data_dir,
                config_path=None,
                spec=None,
            )
            state.session.save()
            state.pairing_cancel = cancel
            state.camera_operation = "stream check"
            state.camera_done.clear()
        try:
            detection = discover_devices(20)
            with state.lock:
                state.detection = detection
                state.session.update(detection=payload(detection))
                cfg, data_dir, findings, default_dir = _compose(state, form)
                for slot, name in cameras.items():
                    kind = "iphone" if slot.endswith("_iphone") else "oak"
                    if not any(d.name == name for d in detection.get(kind, [])):
                        state.camera_diagnostics.append(
                            {
                                "kind": kind,
                                "name": name,
                                "device_id": None,
                                "frames": 0,
                                "hits": 0,
                                "error": "Camera is not detected. Reconnect it and refresh devices.",
                            }
                        )
                if preflight.has_blockers(findings) and not cancel.is_set():
                    return _camera_blocked_plan(
                        state, form, cfg, data_dir, findings, default_dir=default_dir
                    )
            if cancel.is_set():
                checked = {"cameras": [], "errors": [], "cancelled": True}
            else:
                checked = check_streams(detection, cameras, cancel)
            with state.lock:
                state.camera_diagnostics = checked["cameras"]
                cancelled = (
                    checked["cancelled"] or cancel.is_set() or state.shutting_down
                )
                if cancelled:
                    findings.append(
                        Finding(
                            "error",
                            "camera_check_cancelled",
                            "Camera stream check cancelled; recorder was not started.",
                        )
                    )
                findings.extend(
                    Finding("error", "camera_stream_failed", error)
                    for error in checked["errors"]
                )
                if preflight.has_blockers(findings):
                    return _camera_blocked_plan(
                        state, form, cfg, data_dir, findings, cancelled, default_dir
                    )
                return launch(form, token)
        except HTTPException:
            raise
        except Exception as exc:
            with state.lock:
                findings.append(
                    Finding(
                        "error",
                        "camera_stream_failed",
                        f"Camera check could not finish: {exc}. Refresh devices and retry.",
                    )
                )
                return _camera_blocked_plan(
                    state, form, cfg, data_dir, findings, default_dir=default_dir
                )
        finally:
            with state.lock:
                state.pairing_cancel = None
                state.camera_operation = None
                state.camera_done.set()

    def launch(form, token):
        # Caller holds state.lock. Re-plan after probes: hw{N} may have advanced.
        result, blocked = _plan(state, form)
        result["camera_diagnostics"] = state.camera_diagnostics
        if blocked:
            return JSONResponse(status_code=422, content=json_safe(result))
        try:
            status = sup.start(RECORDER_ROLE, state.session.spec, launch_token=token)
        except StartRefused as exc:
            raise HTTPException(409, str(exc)) from exc
        _hold_identification_for(state, sup.procs.get(RECORDER_ROLE))
        return {**status, "plan": result}

    @app.get("/api/lerobot")
    def api_lerobot():
        """Step 05: every session under the data root and its conversion state."""
        return {
            "data_root": state.data_root,
            "sessions": lerobot.session_rows(state.data_root),
            "converting": state.converting(),
        }

    @app.post("/api/lerobot/start")
    def api_lerobot_start(body: dict | None = None):
        """Convert one listed session: robot actions, then its LeRobot dataset."""
        name = str((body or {}).get("session") or "")
        token = sup.capture_launch_token()
        with state.lock:
            if state.recorder_live():
                raise HTTPException(
                    409,
                    "the recorder is running: quit it (q) before converting; "
                    "re-encoding videos would compete with the recording",
                )
            if state.converting():
                raise HTTPException(409, "a conversion is already running")
            rows = {row["name"]: row for row in lerobot.session_rows(state.data_root)}
            row = rows.get(name)
            if row is None:
                raise HTTPException(404, f"no session {name!r} under {state.data_root}")
            if row["blocker"]:
                raise HTTPException(
                    409,
                    f"{name}: {row['blocker']}. Start the recorder on this session "
                    "again to finish its alignment (or run ./launch.sh align), then "
                    "convert.",
                )
            try:
                return sup.start(
                    lerobot.LEROBOT_ROLE,
                    lerobot.lerobot_spec(row["aligned_path"]),
                    launch_token=token,
                )
            except StartRefused as exc:
                raise HTTPException(409, str(exc)) from exc

    @app.post("/api/procs/{role}/stop")
    def api_stop(role: str):
        _role(role)
        with state.lock:
            if state.camera_operation == "stream check" and state.pairing_cancel:
                state.pairing_cancel.set()
        # SIGINT: the recorder interrupts running alignments; they resume on its
        # next launch into the folder.
        return sup.stop(role)

    @app.get("/api/procs/{role}/logs")
    def api_logs(role: str, since: int = 0, limit: int = 500):
        _role(role)
        return sup.tail(role, since, limit)

    @app.get("/api/procs/{role}/logfile")
    def api_logfile(role: str):
        _role(role)
        log_path = sup.status_all()[role].get("log_path")
        if not log_path or not os.path.isfile(log_path):
            raise HTTPException(404, f"{role} has no log file yet")
        with open(log_path, "rb") as f:
            return PlainTextResponse(f.read().decode("utf-8", errors="replace"))


# -- entry point -------------------------------------------------------------------------


def _require_free_port(host, port):
    bind_host = host.strip("[]") if host else host
    family = socket.AF_INET6 if ":" in (bind_host or "") else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((bind_host, int(port)))
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                return  # not ours to judge (bad address, ...): uvicorn reports it
            raise click.UsageError(
                f"http://{host}:{port}/ is already in use: another DC GUI is running "
                "-- open that URL, or stop it first. Nothing was adopted or stopped."
            ) from exc


def browser_url(host, port):
    """The GUI's URL for a browser on this machine: loopback for a wildcard bind."""
    name = str(host or "").strip().strip("[]")
    if name in WILDCARD_HOSTS:
        name = "127.0.0.1"
    return f"http://[{name}]:{port}/" if ":" in name else f"http://{name}:{port}/"


def open_browser_when_serving(host, port, timeout_s=30.0):
    """Open the GUI in a browser as soon as the server accepts connections.

    Waits in a daemon thread because uvicorn binds the port only once it runs;
    a server that never comes up leaves no tab behind. Skipped on Linux without
    a graphical session, where webbrowser would fall back to a text browser in
    this terminal.
    """
    if sys.platform.startswith("linux") and not (
        display() or os.environ.get("WAYLAND_DISPLAY")
    ):
        return
    url = browser_url(host, port)
    address = (urlsplit(url).hostname, port)

    def wait_and_open():
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(address, timeout=0.5):
                    pass
            except OSError:
                time.sleep(0.2)
                continue
            webbrowser.open_new_tab(url)
            return

    threading.Thread(target=wait_and_open, name="open-browser", daemon=True).start()


@click.command()
@click.option(
    "--host",
    default=lambda: os.environ.get("UMI_DC_UI_HOST", "127.0.0.1"),
    show_default="127.0.0.1",
    help="Bind host (this server spawns the recorder: keep it local; or UMI_DC_UI_HOST)",
)
@click.option(
    "--port",
    default=lambda: int(os.environ.get("UMI_DC_UI_PORT", str(DEFAULT_PORT))),
    type=int,
    show_default=str(DEFAULT_PORT),
    help="Bind port (or UMI_DC_UI_PORT)",
)
@click.option(
    "--home",
    default=None,
    help=f"Runtime dir for generated yamls, logs, pidfiles, session.json (default {ENV_HOME} or {DEFAULT_HOME})",
)
@click.option(
    "--data-root",
    default=lambda: os.environ.get("UMI_DC_UI_DATA_ROOT", DEFAULT_DATA_ROOT),
    show_default=DEFAULT_DATA_ROOT,
    help="Parent folder of the dated session folders (recording.data_dir = <root>/<date>_dex<K>L_dex<M>R_hw<N>_<task>; the token names the selected left and right boards)",
)
@click.option(
    "--base-config",
    default=composer.BASE_CONFIG,
    show_default=True,
    help="Recording yaml every generated config is derived from (never modified)",
)
@click.option(
    "--allow-remote",
    is_flag=True,
    default=False,
    help="Required for a non-loopback --host: exposes an unauthenticated recorder-spawning API to the network",
)
@click.option(
    "--scan-every-launch",
    is_flag=True,
    default=False,
    help="Use up an Identify-cameras scan once its recorder launch is ready, so a later launch carries a device binding only after a new scan (by default a scan covers every launch of this GUI run; each Start still re-checks the scanned device ids)",
)
@click.option(
    "--open/--no-open",
    "open_browser",
    default=True,
    show_default=True,
    help="Open the GUI in a browser once it is serving (skipped without a graphical session)",
)
def main(
    host,
    port,
    home,
    data_root,
    base_config,
    allow_remote,
    scan_every_launch,
    open_browser,
):
    """Start the DexUMI data-collection GUI."""
    if not is_loopback_host(host) and not allow_remote:
        raise click.UsageError(
            f"--host {host} is not loopback: this GUI spawns the recorder without "
            "authentication. Pass --allow-remote to expose it anyway "
            f"(Host allowlist: {', '.join(sorted(allowed_hosts(host)))})."
        )
    # Claim the port BEFORE touching the supervisor: a second GUI that loses the
    # bind would otherwise adopt the running recorder from its pidfile and stop
    # it on its way out (atexit).
    _require_free_port(host, port)
    try:
        # A different port must not adopt a still-running GUI's recorder.
        # Keep ownership through cleanup, not merely through HTTP serving.
        with claim_runtime_home(home or default_home()) as home:
            _serve(
                host,
                port,
                home,
                data_root,
                base_config,
                allow_remote,
                scan_every_launch,
                open_browser,
            )
    except RuntimeHomeInUseError as exc:
        raise click.UsageError(str(exc)) from exc


def _serve(
    host,
    port,
    home,
    data_root,
    base_config,
    allow_remote,
    scan_every_launch=False,
    open_browser=False,
):
    # A `uv run ... &` parent may leave SIGINT ignored; the recorder inherits
    # that and would never see the Ctrl-C used by Stop.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    supervisor = make_supervisor(home)
    # Validate the config before adopting a surviving recorder. A bad launch
    # configuration must leave that recorder available for a corrected restart.
    app = create_app(
        DcSession.load(home),
        supervisor,
        data_root=data_root,
        base_config=base_config,
        host=host,
        scan_every_launch=scan_every_launch,
    )
    try:
        adopted = supervisor.adopt_orphans()
        atexit.register(supervisor.stop_all)
        atexit.register(app.state.ui.calibration.shutdown)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        lines = [
            f"\n{'=' * 60}",
            "  DexUMI data collection GUI",
            f"  URL:        http://{host}:{port}/",
            f"  home:       {home}",
            f"  data root:  {data_root}",
            f"  base yaml:  {base_config}",
            f"  DISPLAY:    {display() or 'unset (record_data.py needs the X session)'}",
            "  cameras:    picked in the page or filled by Identify cameras",
            "  scan scope: "
            + (
                "one recorder launch per Identify-cameras scan (--scan-every-launch)"
                if scan_every_launch
                else "every launch of this GUI run (scanned ids re-checked at each Start; "
                "identify again after moving a camera to another exoskeleton)"
            ),
        ]
        if adopted:
            lines.append(f"  adopted:    {adopted}")
        if allow_remote:
            lines.append(
                f"  REMOTE:     --allow-remote on {host}; unauthenticated API reachable by "
                f"Host {', '.join(sorted(allowed_hosts(host)))}"
            )
        lines.append(f"{'=' * 60}\n")
        print("\n".join(lines), flush=True)
        if open_browser:
            open_browser_when_serving(host, port)
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        # Uvicorn's lifespan normally stops the recorder. Also cover a failed
        # startup before lifespan entry, and finish before releasing server.lock.
        try:
            app.state.ui.calibration.shutdown()
            supervisor.stop_all()
        finally:
            atexit.unregister(supervisor.stop_all)
            atexit.unregister(app.state.ui.calibration.shutdown)


if __name__ == "__main__":
    main()

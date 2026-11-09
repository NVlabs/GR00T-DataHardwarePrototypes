# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the calibration steps *are*, with no HTTP in sight.

The camera check runs entirely from the browser: one child holds every camera
on the rig open and reports what each decodes, this module folds those events
into per-camera state, and the QR the operator aims at is served by
`/calibration/check/qr/{side}.svg`. Nothing is drawn on the station monitor.

The tracker check is the same shape and only a Vive rig is asked for it, which
is why the step list is per-rig (`steps_for`) rather than a fixed tuple. Its
fold lives next door in `station_ui/vive_check.py`; what this module owns is the
command, the bound, and the step's verdict.

Gripper calibration blocks on `input()` prompts, so it runs in a terminal on
the station that the operator has to be standing at. This module builds the
command, hands it to the job registry, and reads the exit code back.

Design rules:

* **No `sudo`.** `99-umi-gripper.rules` at the repo root and membership in the
  `dialout` group give persistent access to the gripper's serial port, so
  nothing here is privileged and nothing waits on a password prompt.
* **The exit code decides.** A step is done when its child exited 0, not when
  the operator clicks something.
* **The child's exit code, not the terminal's.** Gripper calibration keeps
  blocking `input()` calls (`umi/umi_device/umi_gripper.py`), so it must run
  somewhere with a keyboard attached: a terminal emulator. But
  `gnome-terminal` is a client to a session server and always exits 0, so the
  wrapper records the calibration's own status to a file and that is what is
  read back.
* **Step state is server state.** It lives on `app.state`, so a reload (or a
  second browser) sees the same progress.

`LAUNCHER` and `TERMINAL_ARGV` are module globals, not inlined literals, so a
test can replace the subprocess boundary and still exercise real jobs with real
pids and real exit codes without opening hardware.
"""

from __future__ import annotations

import json
import pathlib
import shlex
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field

import yaml

from station_ui import camera_roles, fast_yaml, identification, jobs
from station_ui import session as session_state
from station_ui.core import setup_presets
from station_ui.errors import FieldError, UnknownJob
from station_ui.jobs import FAILED as JOB_FAILED
from station_ui.jobs import JobRegistry
from umi.common.utility.tracking import tracking_device_side

__all__ = [
    "CHECK_TIMEOUT_S",
    "DONE",
    "FAILED",
    "MATCHED",
    "MISMATCHED",
    "OPEN_FAILED",
    "PENDING",
    "RUNNING",
    "SIDES",
    "STEPS",
    "STEP_GRIPPER",
    "STEP_VERIFY",
    "STEP_VIVE",
    "VIVE_STEP",
    "VIVE_TIMEOUT_S",
    "WAITING",
    "WATCH_PREFIX",
    "CalibrationState",
    "CheckState",
    "CheckView",
    "Run",
    "ViveCheckState",
    "calibration_state",
    "check_argv",
    "check_running",
    "check_state",
    "check_view",
    "current_side",
    "expected_keys",
    "fold_events",
    "gripper_argv",
    "gripper_generation_for",
    "GRIPPER_RECORD_S",
    "gripper_summary_path",
    "gripper_summary",
    "gripper_plot_file",
    "gripper_plot_dir",
    "gripper_log_path",
    "gripper_log",
    "job_registry",
    "parse_events",
    "payload_for_side",
    "recording_config",
    "repo_root",
    "run_state",
    "side_of_role",
    "stale",
    "step_state",
    "steps_for",
    "stop_vive_check",
    "verification_roles",
    "vive_argv",
    "vive_check_state",
    "vive_running",
    "vive_running_arm",
    "vive_sections",
    "vive_tracker_keys",
    "vive_view",
    "watcher_log",
]

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Run calibration with the server interpreter to use its installed
# dependencies without invoking network resolution or relying on uv on PATH.
LAUNCHER: tuple[str, ...] = (sys.executable,)


# Use --wait to keep the job alive for the terminal command, and -- to end
# terminal options.
TERMINAL_ARGV: tuple[str, ...] = ("gnome-terminal", "--wait", "--")
# Result files and recorded exit statuses. Namespaced per run below.
TEMP_DIR = "~/.umi_temp"

VERIFY_SCRIPT = "real_scripts/device_verification/verify_device_assignment.py"
VIVE_SCRIPT = "real_scripts/device_verification/verify_vive_assignment.py"
GRIPPER_SCRIPT = "examples/umi_gripper_example.py"
GRIPPER_CONFIG_DIR = "config/umi_gripper"

# Camera verification is operator-paced; do not impose a wall-clock timeout.
CHECK_TIMEOUT_S = None
# Bound Vive verification because the bridge serves a shared REP socket.
# A timeout stops an unattended check without reporting a hardware failure.
VIVE_TIMEOUT_S = 300.0
# Limit post-prompt recording to a short measurement pass; the example
# otherwise records until interrupted.
GRIPPER_RECORD_S = 5.0
# Allow time for operator prompts, then bound the calibration command overall.
GRIPPER_TIMEOUT_S = 600.0

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

STEP_VERIFY = "device_verify"
STEP_GRIPPER = "gripper_cal"
STEP_VIVE = "vive_check"


@dataclass(frozen=True)
class Step:
    id: str
    # Derive step numbers from the selected rig rather than storing duplicate ordering.
    title: str
    # True when the step opens something on the station the operator must watch.
    # Rendered as copy on the page: a browser-only operator otherwise waits
    # forever on a window they cannot see.
    watch: str
    summary: str


VIVE_STEP = Step(
    id=STEP_VIVE,
    title="Tracker check",
    watch=(
        "Carry this browser to the rig and move one gripper at a time, as "
        "prompted. Follow the live movement plot below."
    ),
    summary="",
)

# Define the complete step catalogue here; steps_for selects the subset for
# a rig. Keep state buckets for every step so preset changes invalidate
# verification for hardware no longer selected.
STEPS: tuple[Step, ...] = (
    Step(
        id=STEP_VERIFY,
        title="Camera check",
        watch="",
        summary="",
    ),
    VIVE_STEP,
    Step(
        id=STEP_GRIPPER,
        title="Calibrate grippers",
        watch="Follow the steps in the Terminal",
        summary="",
    ),
)


def steps_for(config: dict) -> tuple[Step, ...]:
    """The steps this rig has, in the operator's order.

    `STEPS` above stays the full catalogue. This is the per-rig list, and the
    distinction is the whole reason the function exists:
    `readiness.calibration_done` requires every step in the list to pass, so a
    tracker check in the catalogue-wide list would block Record on every rig
    behind a check with no trackers to look at.

    Keyed on whether the config declares a tracker at all, rather than on the
    arm-owned ones `expected_keys` counts: a rig recording a puck this console
    has no column for still has a bridge worth checking, and hiding the step
    would leave that recording unverified with nothing to say so.
    """
    has_vive = any(
        entry.get("type") == "vive_tracker"
        for entry in (config.get("numeric_sources") or [])
    )
    return tuple(step for step in STEPS if step.id != STEP_VIVE or has_vive)


# ------------------------------------------------------------------ run state


@dataclass
class Run:
    """One launch of one unit of a step: a role, a gripper side, a camera."""

    key: str
    job_id: str
    label: str = ""
    # Where the wrapper recorded the *child's* exit status, when the child ran
    # inside a terminal emulator whose own status is meaningless.
    status_path: str | None = None
    # Use per-run plot and summary paths so results cannot be confused with
    # an earlier calibration.
    plot_dir: str | None = None
    summary_path: str | None = None
    # Capture calibration output through tee; terminal-emulator stdout does
    # not carry the command's output.
    log_path: str | None = None


class CalibrationState:
    """The latest run of every unit of every step, for this rig.

    Deliberately "latest run wins": re-running a failed gripper replaces the
    failure, which is what an operator retrying expects to see.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, dict[str, Run]] = {step.id: {} for step in STEPS}
        # Stamp verification with the device fingerprint. Changing selected
        # hardware invalidates results obtained for the prior devices.
        self._stamps: dict[str, str] = {}

    def record(self, step_id: str, run: Run, fingerprint: str = "") -> None:
        with self._lock:
            self._runs.setdefault(step_id, {})[run.key] = run
            if fingerprint:
                self._stamps[step_id] = fingerprint

    def runs(self, step_id: str) -> dict[str, Run]:
        with self._lock:
            return dict(self._runs.get(step_id, {}))

    def run(self, step_id: str, key: str) -> Run | None:
        return self.runs(step_id).get(key)

    def stamp(self, step_id: str, fingerprint: str) -> None:
        with self._lock:
            self._stamps[step_id] = fingerprint

    def reset(self) -> None:
        """Forget every recorded run: a clean slate for a new rig.

        The stamps deliberately survive. They are the record of which rig each
        step was last verified against, and that is what `stale` reads to say
        "the devices changed, so these no longer describe this rig; run them
        again". Clearing them would leave the console unable to explain why the
        step unticked, which reads as lost work.
        """
        with self._lock:
            self._runs = {step.id: {} for step in STEPS}

    def stamp_of(self, step_id: str) -> str:
        with self._lock:
            return self._stamps.get(step_id, "")


_STATE_LOCK = threading.Lock()


def _shared(app, name: str, factory):
    """One instance per app, created on first use.

    Built lazily rather than in `create_app`, so the factory does not need to
    know about every page's state. The lock makes the first two concurrent
    requests agree.
    """
    existing = getattr(app.state, name, None)
    if existing is not None:
        return existing
    with _STATE_LOCK:
        existing = getattr(app.state, name, None)
        if existing is None:
            existing = factory()
            setattr(app.state, name, existing)
        return existing


# Share the shutdown-registration marker with devices.registry_for so
# app.state.jobs gets one cleanup handler. Keep it local to avoid loading
# device-discovery SDKs in the web process.
_HOOKED = "jobs_reaped_on_shutdown"


def job_registry(app) -> JobRegistry:
    """The registry every calibration child is started into.

    The shutdown hook is not optional. `jobs.start` detaches its children
    (`start_new_session=True`), so nothing else ever collects them: without the
    hook, a console that never ran a device scan would leave the camera check
    holding an exclusive OAK (and the Vive check polling a bridge every other
    station shares) after the server had exited.

    Checked on every call rather than only when this function created the
    registry, for the reason `devices.registry_for` spells out: a registry
    seeded from anywhere else would otherwise never be reaped. Two callers
    racing the flag attach the handler twice, which costs nothing:
    `JobRegistry.shutdown` is documented safe to call twice.
    """
    registry = _shared(app, "jobs", JobRegistry)
    if not getattr(app.state, _HOOKED, False):
        app.router.add_event_handler("shutdown", registry.shutdown)
        setattr(app.state, _HOOKED, True)
    return registry


def calibration_state(app) -> CalibrationState:
    return _shared(app, "calibration", CalibrationState)


# ------------------------------------------------------------------- config


def repo_root() -> pathlib.Path:
    return REPO_ROOT


def recording_config_path(app) -> pathlib.Path:
    """The recording YAML the calibration scripts read.

    The file `/config` generated, never the template it was generated from: the
    templates carry placeholder hardware, so a check run against them would
    open devices that are not on this bench and fail every role.

    Resolved through `readiness.generated_config` rather than rebuilt here, so
    there is exactly one answer to "which config does this station record from":
    the same one the Configuration tick keys on and the recorder is handed.

    Nothing on this page writes it: the calibration scripts only read it, and
    `/config` owns edits, so a step finishing mid-edit cannot clobber what
    somebody is typing.
    """
    draft = app.state.rig.draft
    if not draft.preset_id:
        raise FieldError("preset_id", "choose a setup preset first")
    try:
        setup_presets.get_preset(draft.preset_id)
    except KeyError:
        raise FieldError(
            "preset_id", f"unknown setup preset {draft.preset_id!r}"
        ) from None
    # Import readiness lazily to avoid the calibration/readiness import cycle.
    from station_ui import readiness

    path = readiness.generated_config(app)
    if path is None:
        raise FieldError(
            "preset_id",
            "Complete Configuration. It is saved when you start calibration or recording.",
        )
    return path


def recording_config(app) -> dict:
    path = recording_config_path(app)
    if not path.is_file():
        # Action forms save this before launch; short API calls require an
        # already generated config. A new session intentionally disarms it.
        raise FieldError(
            "preset_id",
            "Complete Configuration. It is saved when you start calibration or recording.",
        )
    try:
        return fast_yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        # Report malformed generated YAML as FieldError. Callers can mark calibration
        # incomplete without breaking navigation or the whole page.
        raise FieldError(
            "preset_id", f"Recording config could not be read: {exc}"
        ) from exc


def layout_config(app) -> dict:
    """Describe the current rig, including between recording sessions.

    A new session disarms the generated config without changing the hardware or
    its calibration. The fallback supplies the roster for controls and completed
    checks; launches still read the validated file after saving Configuration.
    """
    path = recording_config_path(app)
    if path.is_file():
        return recording_config(app)
    draft = app.state.rig.draft
    preset = setup_presets.get_preset(draft.preset_id)
    try:
        config = fast_yaml.safe_load(
            setup_presets.template_path(
                preset, app.state.settings.repo_root
            ).read_text()
        )
    except (OSError, yaml.YAMLError) as exc:
        raise FieldError(
            "preset_id", f"Setup template could not be read: {exc}"
        ) from exc
    from station_ui.core.optional_devices import get_optional_type

    head = False
    for assignment in draft.optional:
        descriptor = get_optional_type(assignment.type_id)
        head |= descriptor.section == "head"
        config.setdefault("video_sources", []).append({"name": assignment.logical_name})
    config.setdefault("meta", {})["use_head_mount"] = head
    return config


def _has_camera(video_sources, name: str) -> bool:
    for source in video_sources:
        if source.get("name") == name:
            return True
        if source.get("type") == "oak_device":
            for stream in source.get("streams") or []:
                if stream.get("name") == name:
                    return True
    return False


def verification_roles(config: dict) -> list[str]:
    """One QR-verification role for every camera the config actually declares.

    Keyed on `camera_roles.ROLE_SOURCE_NAMES`, which the `--watch` worker reads
    too, so both sides agree on which source name carries which role. With two
    copies, a camera the worker opened and decoded could be dropped by the fold
    below as belonging to a role the page never listed, and the step would be
    badged Done with that camera unverified.
    """
    video_sources = config.get("video_sources") or []
    use_head_mount = bool((config.get("meta") or {}).get("use_head_mount", False))
    roles = []
    for role, names in camera_roles.ROLE_SOURCE_NAMES.items():
        if role in camera_roles.HEAD_ROLES and not use_head_mount:
            continue
        if any(_has_camera(video_sources, name) for name in names):
            roles.append(role)
    return roles


# ------------------------------------------------------- the camera check fold

# Keep this event prefix local to avoid importing camera SDKs in the web
# process; it must match verify_device_assignment.py.
WATCH_PREFIX = "##station-ui-verify## "

WAITING = "waiting"
MATCHED = "matched"
MISMATCHED = "mismatched"
OPEN_FAILED = "open_failed"

# Operator order: the left rig, the right rig, then the head mount.
SIDES = ("left", "right", "head")


def side_of_role(role: str) -> str:
    if role in ("head", "head_umi"):
        return "head"
    return role.split("_", 1)[0]


def payload_for_side(side: str) -> str:
    return f"UMI_VERIFY_{side.upper()}"


def parse_events(lines, prefix: str = WATCH_PREFIX) -> list[dict]:
    """Every prefixed JSON line the child printed, oldest first.

    A line that is prefixed but not parseable is dropped rather than raised
    on: the alternative is a page that 500s because a camera driver wrote a
    partial line during teardown.

    `prefix` defaults to the camera check's so every existing caller is
    unchanged; the Vive check passes its own (`vive_check.parse`). The two
    prefixes are distinct precisely so neither check can fold the other's lines
    when both have run on the same page.
    """
    events = []
    for line in lines:
        if not line.startswith(prefix):
            continue
        try:
            payload = json.loads(line[len(prefix) :])
        except ValueError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def fold_events(roles, events) -> dict[str, str]:
    """Latest event wins per role, except that a matched role latches.

    "Latest wins" is what makes re-aiming a still-wrong camera the retry: an
    operator who pointed the left camera at the right code corrects it by
    pointing it at the correct one, with nothing to click.

    But a camera that has already decoded its own code is verified, and stays
    verified: once matched it latches. A later glimpse of another side's code
    (a reflection, or the operator moving the board past it while checking the
    next side) must not silently un-verify a camera the operator already
    confirmed. A waiting or mismatched camera is still fixable by aiming; a
    matched one is done.
    """
    folded = {role: WAITING for role in roles}
    for event in events:
        role = event.get("role")
        if role not in folded:
            continue
        # Verified latches -- a confirmed camera's answer stands.
        if folded[role] == MATCHED:
            continue
        if event.get("event") == "open_failed":
            folded[role] = OPEN_FAILED
        elif event.get("event") == "saw":
            folded[role] = (
                MATCHED
                if event.get("payload") == payload_for_side(side_of_role(role))
                else MISMATCHED
            )
    return folded


def current_side(folded) -> str | None:
    """The first side the operator still has work to do on.

    A camera that would not open is excluded: it is red, it is not fixable by
    aiming, and pinning the page to its side would strand an operator who can
    still check the rest of the rig. A mismatched camera does hold its side,
    because that one is fixable where they are standing.
    """
    for side in SIDES:
        unresolved = [
            role
            for role, state in folded.items()
            if side_of_role(role) == side and state in (WAITING, MISMATCHED)
        ]
        if unresolved:
            return side
    return None


@dataclass
class CheckState:
    """The one camera check this station has running, if any.

    A single job id rather than a dict of runs: there is one check for the
    whole rig.
    """

    job_id: str | None = None
    preview_path: str | None = None


def check_state(app) -> CheckState:
    return _shared(app, "camera_check", CheckState)


def check_running(app) -> bool:
    """Whether a camera check is holding this rig's cameras open right now.

    Asked before a device scan and by `/record` before a recording. The check
    opens every camera the config declares and holds them until somebody
    presses Stop (`CHECK_TIMEOUT_S` is None on purpose), and an OAK open is
    exclusive. Without this, a rescan started mid-check reports the whole rig
    `Unavailable` and blames the hardware for a UI collision; a recording
    started mid-check simply fails to open its cameras.

    Reads `app.state` directly rather than through the accessors above so that
    merely asking the question does not attach a check state, a job registry,
    or the registry's shutdown hook to an app that has never run a check.
    """
    state = getattr(app.state, "camera_check", None)
    registry = getattr(app.state, "jobs", None)
    if state is None or registry is None or not state.job_id:
        return False
    try:
        job = registry.get(state.job_id)
    except Exception:
        return False
    return not job.finished


def stop_check(app, *, wait: float = 0.0) -> None:
    """Release the rig's cameras by cancelling any running check.

    The check holds every camera open (an OAK open is exclusive), so whatever
    needs the cameras next (a device rescan, a recording) has to be able to
    take them back. Idempotent: no check, an unknown job, or an already-finished
    one is a no-op.

    `wait` blocks until the watcher has actually exited (up to `wait` seconds).
    `jobs.cancel` only delivers the SIGTERM; the child's `finally` still has to
    close the cameras, and a scan launched into that gap fights the dying
    watcher for the same exclusive OAK and reports the rig Unavailable, the very
    collision the caller is trying to avoid. Run it off the event loop when
    `wait` is non-zero.
    """
    state = check_state(app)
    if not state.job_id:
        return
    registry = job_registry(app)
    try:
        job = registry.get(state.job_id)
    except UnknownJob:
        return
    if job.finished:
        return
    jobs.cancel(registry, state.job_id)
    deadline = time.monotonic() + wait
    while not job.finished and time.monotonic() < deadline:
        time.sleep(0.02)


@dataclass
class ViveCheckState:
    """This station's tracker-check jobs, one per arm.

    A dict keyed by arm: each arm has its own button and its own child, so the
    left arm's result has to outlive the child that produced it. The operator
    confirms left, that run exits, and the panel must still say left is
    confirmed while they start the right one.

    Nothing extra is needed to make that work: `JobRegistry` keeps finished
    jobs, so folding the left job's line buffer answers "did left confirm?" long
    after its process is gone. What this dict adds is which buffer to fold.
    """

    job_ids: dict = field(default_factory=dict)


def vive_check_state(app) -> ViveCheckState:
    return _shared(app, "vive_check", ViveCheckState)


def _vive_job(app, arm):
    """That arm's job, or None. Never attaches anything to `app.state`."""
    state = getattr(app.state, "vive_check", None)
    registry = getattr(app.state, "jobs", None)
    if state is None or registry is None:
        return None
    job_id = state.job_ids.get(arm)
    if not job_id:
        return None
    try:
        return registry.get(job_id)
    except Exception:
        return None


def vive_running_arm(app) -> str | None:
    """The arm whose check is polling the bridge right now, if any.

    At most one: the bridge is a single `zmq.REP` thread shared by every station
    on the bench, so the start route refuses a second run rather than letting
    two pollers onto it. Returned rather than a bare bool because the page needs
    to know which section to point its SSE stream at.
    """
    state = getattr(app.state, "vive_check", None)
    if state is None:
        return None
    for arm in state.job_ids:
        job = _vive_job(app, arm)
        if job is not None and not job.finished:
            return arm
    return None


def vive_running(app) -> bool:
    """Whether any Vive check is polling the bridge right now.

    Asked by `/record` before a recording and by `reset` before it lets go, so
    it stays a whole-station question even though the checks are per arm.

    Reads `app.state` directly rather than through the accessor above, for the
    reason `check_running` does: merely asking the question must not attach a
    check state, a job registry, or the registry's shutdown hook to an app that
    has never run one.
    """
    return vive_running_arm(app) is not None


def stop_vive_check(app, arm: str | None = None, *, wait: float = 0.0) -> None:
    """Release the bridge by cancelling running Vive checks.

    The bridge is one `zmq.REP` thread for the whole bench, so whatever needs it
    next -- another station's check, a recording -- has to be able to take it
    back. Idempotent: no check, an unknown job, or an already-finished one is a
    no-op. `wait` blocks until the child has actually exited, up to that many
    seconds, exactly as `stop_check` does.

    `arm` stops one arm's check; the default stops every one. The default is
    what `reset` and the record preflight want -- they are releasing the bridge
    on behalf of the whole station and must not have to know which arm holds it.
    """
    state = vive_check_state(app)
    registry = job_registry(app)
    arms = [arm] if arm is not None else list(state.job_ids)
    stopping = []
    for name in arms:
        job_id = state.job_ids.get(name)
        if not job_id:
            continue
        try:
            job = registry.get(job_id)
        except UnknownJob:
            continue
        if job.finished:
            continue
        jobs.cancel(registry, job_id)
        stopping.append(job)
    deadline = time.monotonic() + wait
    while any(not job.finished for job in stopping) and time.monotonic() < deadline:
        time.sleep(0.02)


def reset(app) -> None:
    """Forget the calibration: stop the check and clear every recorded run.

    Called from Apply when the rig changed (`readiness.reset_verification_for_new_
    rig`). The stamps already expire the steps on a device change; what they do
    not do is release the hardware: the check watcher is still holding every
    camera open, for a rig this config no longer names, and its events still fold
    into the panel. So stop it, forget its job, and drop the runs and the skip.

    The Vive check goes with it, and for a sharper version of the same reason:
    its poller would go on hitting a bridge address the new config may no longer
    name, and that bridge is shared with every other station on the bench.

    The stamps themselves survive, so the operator is still told the devices
    changed rather than just that the step is not done. See `CalibrationState.reset`.
    """
    stop_check(app)
    check_state(app).job_id = None
    stop_vive_check(app)
    vive_check_state(app).job_ids.clear()
    calibration_state(app).reset()


@dataclass
class CheckView:
    rows: list = field(default_factory=list)  # (role, label, state)
    current_side: str | None = None
    # Highlight the roles for the side currently being verified.
    current_roles: frozenset = field(default_factory=frozenset)
    running: bool = False
    # Wait for camera readiness before showing a marker to aim.
    cameras_open: bool = False
    done: bool = False
    # A child that fails before emitting events is a failed check, not a
    # check that was never started. Treat operator cancellation separately.
    failed: bool = False
    error: str | None = None
    job_id: str | None = None
    # Expose human-readable watcher output for every outcome, excluding JSON
    # event lines.
    log: list[str] = field(default_factory=list)


def watcher_log(lines, prefix: str = WATCH_PREFIX) -> list[str]:
    """The watcher's output with its own event lines removed.

    Every `WATCH_PREFIX` line is a serialized event the panel above has already
    rendered as a camera's state -- and there is one per decode, so at 30fps
    they bury the driver messages that are the reason anyone opened this.

    `prefix` defaults to the camera check's so every existing caller is
    unchanged. The Vive check passes its own -- without that, every
    `##station-ui-vive##` line would be shown to the operator as narration.
    """
    return [line for line in lines if not line.startswith(prefix)]


def check_view(app, roles, label_for) -> CheckView:
    """Everything the panel renders, derived on each call and never stored."""
    state = check_state(app)
    registry = job_registry(app)
    job = None
    if state.job_id:
        try:
            job = registry.get(state.job_id)
        except Exception:
            job = None

    if stale(app, STEP_VERIFY):
        job = None
    events = parse_events(job.lines()) if job else []
    folded = fold_events(roles, events)
    try:
        verified = identification.verified_roles(app, recording_config(app))
    except FieldError:
        verified = identification.verified_roles(app)
    for role in set(roles) & verified:
        folded[role] = MATCHED
    running = job is not None and not job.finished
    # Highlight a side only while its check is running.
    side = current_side(folded) if running else None
    return CheckView(
        rows=[(role, label_for(role), folded[role]) for role in roles],
        current_side=side,
        current_roles=frozenset(
            role for role in roles if side is not None and side_of_role(role) == side
        ),
        running=running,
        # Ready, saw and open_failed events all establish that camera opening has
        # finished.
        cameras_open=any(
            event.get("event") in ("ready", "saw", "open_failed") for event in events
        ),
        done=bool(roles) and all(folded[role] == MATCHED for role in roles),
        failed=job is not None and job.state == JOB_FAILED,
        error=job.error if job is not None else None,
        job_id=job.id if job is not None else None,
        log=watcher_log(job.lines()) if job is not None else [],
    )


def vive_tracker_keys(app) -> dict[str, str]:
    """Each arm's tracker key, as the generated config names it.

    The leading side token of the source name, through the shared rule that
    `ConfigAssembler._requirements` and the check's own `plan_sources` apply. A
    puck that names no arm (a head mount) is skipped rather than given an
    invented one. Read off `numeric_sources` directly rather than through the
    assembler: this module answers questions about the generated config, and one
    loop does not justify making the step catalogue depend on the
    template-requirements machinery.

    Ordered left then right, not as the config declares them. The arms are
    checked one button at a time and the right one is refused until the left has
    passed, so this order is the workflow rather than a rendering detail: a
    config that happened to declare `right_tracking_pose` first would otherwise
    invert the gate and lock the operator out of the arm they are told to start
    with. `SIDES` is the console's existing answer to "in what order is a rig
    walked", and an arm it does not name (a puck on a third mount) sorts after
    the ones it does rather than being dropped.
    """
    try:
        config = layout_config(app)
    except FieldError:
        return {}
    keys: dict[str, str] = {}
    for entry in config.get("numeric_sources") or []:
        if entry.get("type") != "vive_tracker":
            continue
        arm = tracking_device_side(str(entry.get("name") or ""))
        if arm is None:
            continue
        params = entry.get("params")
        if not isinstance(params, dict):
            params = {}
        keys[arm] = str(params.get("tracker_key") or "")
    order = {side: rank for rank, side in enumerate(SIDES)}
    return {
        arm: keys[arm]
        for arm in sorted(keys, key=lambda name: (order.get(name, len(SIDES)), name))
    }


def vive_view(app, arm: str):
    """One arm's panel section, derived on each call and never stored.

    Per arm because each arm has its own button and its own child. The fold
    itself lives in `station_ui.vive_check`; this adds what only the job knows.
    Imported inside the function on purpose -- that module imports
    `parse_events` from this one, so a module-level import is a cycle.
    """
    from station_ui import vive_check

    job = _vive_job(app, arm)

    lines = job.lines() if job is not None else []
    # Seed each arm from config before its check starts. Filtering to one arm
    # changes the displayed verdict, not the child's view of both trackers
    # needed for swap detection.
    view = vive_check.fold(
        vive_check.parse(lines),
        seed={arm: vive_tracker_keys(app).get(arm, "")},
        only=[arm],
    )
    view.running = job is not None and not job.finished
    view.job_id = job.id if job is not None else None
    view.error = job.error if job is not None else None
    # Strip the Vive event prefix from human-readable output.
    view.log = watcher_log(lines, prefix=vive_check.VIVE_PREFIX)
    # Treat a job timeout as a stopped check, not a hardware failure. Jobs encode
    # timeouts as FAILED plus a reason string; match the timeout phrase anywhere
    # in that reason.
    view.timed_out = job is not None and "timed out" in str(job.error or "")
    # A confirmed tracker swap fails verification even if the child exits successfully.
    view.failed = (
        job is not None and job.state == JOB_FAILED and not view.timed_out
    ) or view.cross is not None
    return view


def vive_sections(app) -> list[dict]:
    """One entry per arm, in walk order: its view and what is holding it up.

    `blocked_by` is the earlier arm that has not confirmed yet, or None when
    this arm may be started. Sequential on purpose (the operator is walked
    left then right, one button at a time), and computed here rather than in
    the template so the route can refuse the POST with the same answer the
    button was drawn from. A disabled button is a hint; the rule is the route's,
    the same stance `routes/config.add_optional` takes.

    Built in one pass because every `vive_view` re-folds a job's whole line
    buffer, and the page asks this question for the panel, the badge and the
    forward control on every SSE tick.
    """
    sections = []
    blocked_by = None
    for arm in vive_tracker_keys(app):
        view = vive_view(app, arm)
        sections.append({"arm": arm, "view": view, "blocked_by": blocked_by})
        if blocked_by is None and not view.done:
            blocked_by = arm
    return sections


# ------------------------------------------------------------------- commands


def _script(name: str) -> str:
    return str(repo_root() / name)


def check_argv(config_path, preview_path=None, *, roles=None) -> list[str]:
    """The Station Console uses a file preview; direct script launches keep their CLI flags."""
    return [
        *LAUNCHER,
        _script(VERIFY_SCRIPT),
        "--watch",
        *(["--roles", *roles] if roles else []),
        *(["--preview-file", str(preview_path)] if preview_path else []),
        "--config",
        str(config_path),
    ]


def vive_argv(config_path, arm: str) -> list[str]:
    """One child per arm, reading the same generated config.

    `--arm` confirms that arm and returns, so the operator presses a button per
    gripper instead of being walked through both by one run. It does not narrow
    what the child watches: every puck is still read every tick, which is what
    keeps a swapped pair detectable from a single-arm run.

    The Station Console draws the sample events in the tab. Direct invocations
    of the script can still opt into the native window with `--plot`.
    """
    return [
        *LAUNCHER,
        _script(VIVE_SCRIPT),
        "--watch",
        "--arm",
        arm,
        "--config",
        str(config_path),
    ]


def gripper_config_path(side: str) -> str:
    return str(repo_root() / GRIPPER_CONFIG_DIR / f"{side}_umi_gripper_config.yaml")


def gripper_generation_for(draft) -> str:
    """The chosen setup's gripper generation (``v4``), or ``""``.

    Empty when no preset is chosen or the id no longer resolves; the calibration
    then runs with the gripper library's defaults, which are the V4's.
    """
    preset_id = str(getattr(draft, "preset_id", "") or "").strip()
    if not preset_id:
        return ""
    try:
        return str(setup_presets.get_preset(preset_id).gripper_generation or "")
    except KeyError:
        return ""


def gripper_argv(
    side: str,
    identifier: str,
    *,
    status_path: str,
    plot_dir: str,
    summary_path: str,
    log_path: str,
    generation: str = "",
) -> list[str]:
    """Run the calibration in a terminal, and bring back its status and results.

    Three things matter. The calibration blocks on four `input()` prompts, so
    it needs a keyboard: hence the terminal. `gnome-terminal` exits 0 whatever
    its child did, so the child's status is written to `status_path` and read
    from there (`run_state` prefers it over the job's). And the results have to
    land in files, because the console cannot read back a terminal it did not
    capture: the status, the plot, the summary, and the output, which is what
    `log_path` is for.

    `--duration` is what makes this step end. Without it the example runs
    `while True` at 30Hz after the prompts, printing "Press Ctrl+C to stop
    recording" into a terminal nobody is told to interrupt, so the step would
    sit there until the job timed out. The prompts are operator-paced and
    unbounded; only the recording pass is capped.

    `--plot-dir` makes `visualize_gripper_data` save the plot and return
    instead of opening an interactive matplotlib window (which would block
    process exit and strand the job), so the plot comes back to the browser
    rather than to a window on the station.

    Everything interpolated goes through `shlex.quote`, so a device name with a
    quote or a semicolon in it is an argument and not a second command.
    """
    inner = [
        "env",
        f"UMI_GRIPPER_AUTO_SAVE_PATH={gripper_config_path(side)}",
        # Use a noninteractive plotting backend; calibration must run without DISPLAY.
        "MPLBACKEND=Agg",
        # Disable stdout buffering so the session log receives progress while
        # calibration is running.
        "PYTHONUNBUFFERED=1",
        *LAUNCHER,
        _script(GRIPPER_SCRIPT),
        "--uart-device-name",
        identifier,
        # Pass gripper generation through calibration so gear radius and open-travel
        # validation match the hardware expected by the recorder.
        *(["--gripper-generation", generation] if generation else []),
        "--duration",
        str(GRIPPER_RECORD_S),
        "--plot-dir",
        plot_dir,
        "--summary-json",
        summary_path,
    ]
    # Capture terminal-command output with tee and its exit status with
    # PIPESTATUS[0]. The pipeline's $? reports tee, not the calibration result.
    script = (
        f"{shlex.join(inner)} 2>&1 | tee {shlex.quote(log_path)}; "
        "rc=${PIPESTATUS[0]}; "
        f"printf '%s' \"$rc\" > {shlex.quote(status_path)}; "
        'if [ "$rc" -ne 0 ]; then echo "calibration exited with status $rc"; fi; '
        'exit "$rc"'
    )
    return [*TERMINAL_ARGV, "bash", "-c", script]


def temp_dir() -> pathlib.Path:
    path = pathlib.Path(TEMP_DIR).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_id() -> str:
    return uuid.uuid4().hex[:12]


def gripper_status_path(side: str, run: str) -> str:
    return str(temp_dir() / f"gripper_cal_{side}_{run}.rc")


def gripper_plot_dir(side: str, run: str) -> str:
    """Where this run's plot is written. One directory per run, so re-running a
    side cannot show the previous attempt's picture beside the new verdict."""
    path = temp_dir() / f"gripper_cal_{side}_{run}_plot"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def gripper_summary_path(side: str, run: str) -> str:
    return str(temp_dir() / f"gripper_cal_{side}_{run}.json")


def gripper_log_path(side: str, run: str) -> str:
    return str(temp_dir() / f"gripper_cal_{side}_{run}.log")


# Carriage-return progress lines contain many redraws; retain only the final
# segment of each line.
def _collapse_redraws(text: str) -> list[str]:
    # Split only on newline so carriage returns survive until redraw collapsing.
    return [
        segment.rsplit("\r", 1)[-1] if "\r" in segment else segment
        for segment in text.rstrip("\n").split("\n")
    ]


def gripper_log(run: Run, limit: int = 200) -> list[str]:
    """What this calibration printed, newest end first dropped rather than oldest.

    Tail rather than head: a step is read after it has gone wrong, and the
    reason is at the end. Empty covers every way there is nothing to show -- no
    file, an unreadable one, a child that died before writing -- because all of
    them mean the same thing to the page.
    """
    if not run or not run.log_path:
        return []
    try:
        # Disable newline translation so carriage returns remain available to
        # _collapse_redraws. Path.read_text cannot select this behavior on Python 3.12.
        with open(run.log_path, encoding="utf-8", errors="replace", newline="") as fh:
            text = fh.read()
    except OSError:
        return []
    return _collapse_redraws(text)[-limit:]


# Match visualize_gripper_data's plot filename without importing plotting
# and data-check dependencies into the web process.
GRIPPER_PLOT_NAME = "Live Recording_umi_gripper.png"


def gripper_plot_file(run: Run) -> pathlib.Path | None:
    """This run's saved plot, if the child got as far as writing one."""
    if not run.plot_dir:
        return None
    path = pathlib.Path(run.plot_dir) / GRIPPER_PLOT_NAME
    try:
        return path if path.is_file() else None
    except OSError:
        return None


def gripper_summary(run: Run) -> dict | None:
    """The three validation checks the child recorded, or None.

    None covers every way there is nothing to show -- no file, a half-written
    one, a child that died before the recording pass -- because all of them mean
    the same thing to the page: say nothing rather than say zero.
    """
    if not run.summary_path:
        return None
    try:
        loaded = json.loads(pathlib.Path(run.summary_path).read_text())
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


# -------------------------------------------------------------------- status


def _recorded_status(run: Run) -> int | None:
    """The integer exit code the wrapper wrote to `status_path`, or None.

    None covers every "no clear answer" shape: no status_path at all, an
    unreadable or empty file, and a file whose contents are not an integer (a
    terminal torn down mid-write). Callers decide what None means for them.
    """
    if not run.status_path:
        return None
    try:
        recorded = pathlib.Path(run.status_path).read_text().strip()
    except OSError:
        return None
    if not recorded:
        return None
    try:
        return int(recorded)
    except ValueError:
        return None


def _job_exit_code(registry: JobRegistry, run: Run) -> int | None:
    try:
        job = registry.get(run.job_id)
    except Exception:
        return None
    return job.exit_code


def child_exit_code(registry: JobRegistry, run: Run) -> int | None:
    """The calibration's own status, falling back to the job's, carefully.

    The recorded file wins because a terminal emulator's exit code says nothing
    about what ran inside it. When the file is present but unparseable the run is
    a failure with no code (a terminal torn down mid-write), reported as None.

    The fallback to the job's own exit code is only for a run that recorded
    nothing at all, and even then a clean 0 is not a pass. The wrapper writes
    the status before it exits, so a missing status behind a 0-exit
    `gnome-terminal --wait` means the window was closed before the calibration
    finished (gnome-terminal exits 0 on a window-close). Only a job that failed
    loudly keeps its non-zero code; a bare 0 becomes None, so `run_state` reads
    a calibration nobody completed as failed rather than a pass.
    """
    if run.status_path:
        try:
            recorded = pathlib.Path(run.status_path).read_text().strip()
        except OSError:
            recorded = ""
        if recorded:
            try:
                return int(recorded)
            except ValueError:
                return None
        code = _job_exit_code(registry, run)
        return code if code else None
    return _job_exit_code(registry, run)


def run_state(registry: JobRegistry, run: Run) -> str:
    try:
        job = registry.get(run.job_id)
    except Exception:
        return PENDING
    # The recorded child status determines completion. A terminal may remain
    # open after the calibration has finished.
    recorded = _recorded_status(run)
    if recorded is not None:
        return DONE if recorded == 0 else FAILED
    if not job.finished:
        return RUNNING
    code = child_exit_code(registry, run)
    return DONE if code == 0 else FAILED


def expected_keys(app, step_id: str) -> tuple[str, ...]:
    """Every unit a step must clear before the step itself is done."""
    if step_id == STEP_GRIPPER:
        return tuple(sorted(app.state.rig.draft.grippers))
    try:
        config = layout_config(app)
    except FieldError:
        return ()
    if step_id == STEP_VERIFY:
        return tuple(verification_roles(config))
    if step_id == STEP_VIVE:
        # Build per-arm tracker checks through vive_tracker_keys so step summaries
        # and live panels use the same arm set.
        return tuple(vive_tracker_keys(app))
    return ()


def stale(app, step_id: str) -> bool:
    """Whether this step's result describes a rig the draft no longer names.

    A step that has never run is not stale -- it is pending, which the caller
    already renders. Only a recorded result can go out of date.
    """
    recorded = calibration_state(app).stamp_of(step_id)
    if not recorded:
        return False
    return recorded != session_state.device_fingerprint(app.state.rig.draft)


def step_state(app, step_id: str) -> str:
    """Derived, always, from what the child reported, never from a click.

    A result recorded against a different rig reads as `pending`: the operator
    has to run it again, and telling them it passed would be the console
    vouching for hardware nothing has looked at. See `stale`.
    """
    if step_id != STEP_VERIFY and stale(app, step_id):
        return PENDING
    # One child watches all cameras; derive per-role status from events.
    if step_id == STEP_VERIFY:
        try:
            roles = verification_roles(layout_config(app))
        except FieldError:
            return PENDING
        view = check_view(app, roles, lambda role: role)
        # All cameras verified means done even while the watcher is still releasing
        # its devices.
        if view.done:
            return DONE
        if view.running:
            return RUNNING
        folded = {role: state for role, _label, state in view.rows}
        if any(s in (MISMATCHED, OPEN_FAILED) for s in folded.values()):
            return FAILED
        # A child that exits before emitting events must report failure rather than
        # leave every camera looking unstarted.
        if view.failed:
            return FAILED
        return PENDING

    # Fold per-arm events: confirmation and operator cancellation can both exit
    # with code 0, so exit status alone cannot establish completion.
    if step_id == STEP_VIVE:
        arms = vive_tracker_keys(app)
        if not arms:
            return PENDING
        views = [vive_view(app, arm) for arm in arms]
        # Any confirmed cross-talk fails the rig; a confirmed arm cannot override
        # a swapped tracker on another arm.
        if any(view.cross for view in views):
            return FAILED
        if all(view.done for view in views):
            return DONE
        if any(view.running for view in views):
            return RUNNING
        # After reporting active arms, fail on device errors but treat timeouts as
        # stops.
        if any(view.failed and not view.timed_out for view in views):
            return FAILED
        return PENDING

    registry = job_registry(app)
    runs = calibration_state(app).runs(step_id)
    states = {key: run_state(registry, run) for key, run in runs.items()}
    if RUNNING in states.values():
        return RUNNING
    if FAILED in states.values():
        return FAILED
    keys = expected_keys(app, step_id)
    if keys and all(states.get(key) == DONE for key in keys):
        return DONE
    return PENDING


# ------------------------------------------------------------------ rendering


BADGES = {
    PENDING: ("○", "Not started", ""),
    RUNNING: ("●", "Running", "badge--warn"),
    DONE: ("✓", "Done", "badge--ok"),
    FAILED: ("✕", "Failed", "badge--error"),
}


@dataclass
class RunView:
    """Everything the template needs about one unit, already resolved.

    There is deliberately no "cancelled" here. This view serves only the
    gripper step, which the operator cannot cancel: it runs in a terminal, and
    every non-zero exit is a failure the exit code itself explains.
    """

    key: str
    label: str
    state: str = PENDING
    job_id: str | None = None
    # Show the job's failure reason when available. If only the child result is
    # known, expose its verdict and full log instead of inventing a cause.
    error: str = ""
    # Show the run's saved plot and validation summary on success and failure;
    # leave them empty until the files exist.
    plot_url: str = ""
    summary: dict | None = None
    # Read the calibration's tee log, not terminal-emulator output.
    log: list[str] = field(default_factory=list)


def _job_error(registry: JobRegistry, run: Run) -> str:
    try:
        return str(registry.get(run.job_id).error or "")
    except Exception:
        return ""


def run_view(
    registry: JobRegistry, key: str, label: str, run: Run | None, plot_url: str = ""
) -> RunView:
    if run is None:
        return RunView(key=key, label=label)
    state = run_state(registry, run)
    return RunView(
        key=key,
        label=label,
        state=state,
        job_id=run.job_id,
        error=_job_error(registry, run) if state == FAILED else "",
        # Offer a plot only after it exists. Include the job ID in its URL to avoid
        # showing cached results from another run.
        plot_url=f"{plot_url}?run={run.job_id}"
        if gripper_plot_file(run) is not None
        else "",
        summary=gripper_summary(run),
        log=gripper_log(run),
    )

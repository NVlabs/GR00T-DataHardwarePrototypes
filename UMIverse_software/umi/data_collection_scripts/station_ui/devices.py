# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the session draft declares, and what a scan actually found.

Two halves, both pure: `declared_devices` reads the draft, `statuses` folds a
scan result over it. Neither opens anything, so the page renders before, during
and after a scan without a single hardware call on the request path.

The four states are deliberate. `Connected`, `Unavailable` and `Failed` each
carry an icon *and* a word, because the brand green and the healthy green are
the same hue on this console. The fourth, `Not tested`, exists because a page
that has never scanned must not claim anything about the rig; it carries no
status colour at all.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import threading
from dataclasses import dataclass

from station_ui import jobs as job_control
from station_ui.core import optional_devices
from station_ui.errors import UnknownJob
from station_ui.jobs import CANCELLED as CANCELLED_JOB
from station_ui.jobs import FAILED as FAILED_JOB
from station_ui.jobs import JobRegistry
from station_ui.scan_worker import RESULT_PREFIX

__all__ = [
    "AMBIGUOUS",
    "CONNECTED",
    "EMPTY",
    "FAILED",
    "NOT_COVERED",
    "NOT_TESTED",
    "OK",
    "UNAVAILABLE",
    "DeclaredDevice",
    "DeviceStatus",
    "DevicesState",
    "current_job",
    "declared_devices",
    "describe_record",
    "detected_for",
    "ensure_scan",
    "identifier_for",
    "matches_token",
    "outcome_for",
    "parse_result",
    "record_for",
    "registry_for",
    "resolve_device",
    "state_for",
    "statuses",
    "unavailable_reason",
]

NOT_TESTED = "not_tested"
CONNECTED = "connected"
UNAVAILABLE = "unavailable"
FAILED = "failed"
# Distinguish a kind not covered by discovery from a covered kind that was
# not found; only the latter establishes unavailability.
NOT_COVERED = "not_covered"

# The word is the status. The badge class only tints something already legible
# in monochrome, and `.badge` with no modifier is deliberately uncoloured.
STATUS_TEXT = {
    NOT_TESTED: "Not tested",
    CONNECTED: "Connected",
    UNAVAILABLE: "Unavailable",
    FAILED: "Failed",
    NOT_COVERED: "Not checked",
}

# Include every kind enumerated by scan_worker, including UVC from v4l2-ctl.
SCANNED_KINDS = frozenset({"oak", "iphone", "gripper", "uvc"})


def scan_covers(kind: str) -> bool:
    """Whether the scan can enumerate *kind*.

    The single predicate behind "did the scan look at this". Read by
    `is_unscanned_kind`, by the configuration picker's decision to fall back to
    the device registry, and by the presence line under each box. Three answers
    to one question is how a row reads Connected on one page and Unavailable on
    the other.
    """
    return kind in SCANNED_KINDS


# Retain enumerated devices that could not open so the UI can distinguish
# connection failure from absence.
AVAILABLE = "available"


@dataclass(frozen=True)
class DeclaredDevice:
    """One device the draft says this session needs.

    `identifier` is whatever the config stores -- a registry name for known
    hardware, a raw device id for anything unregistered -- and may be empty
    while the operator is still filling the form.
    """

    label: str
    identifier: str
    kind: str | None = None


@dataclass(frozen=True)
class DeviceStatus:
    device: DeclaredDevice
    state: str
    detail: str = ""

    @property
    def text(self) -> str:
        return STATUS_TEXT[self.state]

    @property
    def is_connected(self) -> bool:
        return self.state == CONNECTED

    @property
    def is_unavailable(self) -> bool:
        return self.state == UNAVAILABLE

    @property
    def is_failed(self) -> bool:
        return self.state == FAILED


def is_unscanned_kind(device: DeclaredDevice) -> bool:
    """Whether the scan can say nothing about this device: a kind nothing knows
    how to look for at all.

    A device with `kind is None` (a template-required tracking camera) is
    not unscanned: it is scannable and must still report Connected, so the
    truthiness check keeps it blocking.
    """
    return bool(device.kind) and not scan_covers(device.kind)


def _arm_label(arm: str) -> str:
    return f"{arm.replace('_', ' ').strip().capitalize()} gripper".strip()


def _optional_kind(type_id: str) -> str | None:
    """The descriptor's device kind, or None for a type_id nothing declares.

    A draft carrying a stale type_id is an operator-visible listing problem,
    never a 500 -- so an unknown id degrades to "kind unknown" and the row
    still renders.
    """
    try:
        return optional_devices.get_optional_type(type_id).device_kind
    except KeyError:
        return None


def declared_devices(draft) -> list[DeclaredDevice]:
    """Every device the draft declares, in a stable operator-facing order.

    Grippers first because a missing gripper blocks recording outright, then
    the template's required cameras, then whatever optional hardware was added.

    No production caller, deliberately -- this is `statuses`' other half. See
    the note above `statuses` for why the pair is kept and what deleting it
    would cost.
    """
    devices = [
        DeclaredDevice(
            label=_arm_label(arm),
            identifier=str(identifier or "").strip(),
            kind="gripper",
        )
        for arm, identifier in sorted((draft.grippers or {}).items())
    ]
    devices += [
        DeclaredDevice(label=name, identifier=str(identifier or "").strip())
        for name, identifier in sorted((draft.required_cameras or {}).items())
    ]
    devices += [
        DeclaredDevice(
            label=item.logical_name,
            identifier=str(item.device or "").strip(),
            kind=_optional_kind(item.type_id),
        )
        for item in (draft.optional or [])
        # An enabled optional camera with no selection is a configuration gap,
        # not a discovery failure. Applying the config validates the missing selection.
        if str(item.device or "").strip()
    ]
    return devices


def parse_result(lines) -> list[dict] | None:
    """The scan's machine-readable payload, or None if it never produced one.

    The child prints human progress for the operator *and* one prefixed JSON
    line for this parser, so the diagnostics are captured with the job rather
    than lost on another process's stdout. Read from the end: a device name
    echoed in the progress text cannot be mistaken for the result.
    """
    for line in reversed(list(lines)):
        if not line.startswith(RESULT_PREFIX):
            continue
        try:
            payload = json.loads(line[len(RESULT_PREFIX) :])
        except ValueError:
            return None
        found = payload.get("devices") if isinstance(payload, dict) else None
        return found if isinstance(found, list) else None
    return None


def _match(found, identifier: str, kind: str | None = None):
    """Look up by registry name or device id, like `DeviceScanner.by_identifier`.

    Matching on the registry name alone would silently return nothing for every
    unregistered device (and, for example, give an IMX378 the wrong stream
    resolution; see `station_ui/core/device_scan.py`).

    Scoped to *kind* when the caller knows one, because identifiers are only
    unique within a kind. A UVC camera's id is its V4L2 index (`0`, `4`, `10`),
    and those are short enough to collide with anything: a slot of another
    kind holding `10` would match the UVC record at index 10 and report
    `Connected` for a device the scan never looked at.

    `kind=None` means "do not scope", and it is not merely the lazy default: a
    template-required tracking camera is declared with no kind at all (see
    `declared_devices`), and scoping those to None would match nothing and
    report every one of them missing.
    """
    if not identifier:
        return None
    for record in found:
        if not isinstance(record, dict):
            continue
        if kind and record.get("kind") != kind:
            continue
        if identifier in (record.get("registry_name"), record.get("device_id")):
            return record
    return None


def record_for(found, identifier: str, kind: str | None = None):
    """The scan record naming *identifier*, or None.

    Public because `/config` asks the same question `/devices` does -- "is the
    device this row names actually here?" -- and two answers to one question is
    how a row reads Connected on one page and Unavailable on the other. Both
    pass the kind for the same reason: see `_match`.
    """
    return _match(found or [], str(identifier or "").strip(), kind)


def unavailable_reason(found, identifier, kind: str | None = None):
    """Why the last scan refused *identifier*, or None if it did not refuse it.

    The one distinction the selection gate turns on, and it is three states
    rather than two. A device the scan enumerated and reported as anything
    other than `available` is refused: it is on this bench, something is
    holding it, and recording from it produces a session with a dead camera in
    it. A device the scan never mentioned is not refused, and neither is a
    device with no scan behind it at all: configs are prepped before the
    hardware reaches the bench, and a new station's `device_registry` is
    empty, so "connected or nothing" would empty the screen on a fresh
    install.

    The reason is the worker's own, because a permission error on
    /dev/ttyACM1 and a phone with the Record3D app closed need different
    actions and "unavailable" sends the operator to re-seat a cable in both
    cases. Never empty: a refusal the operator cannot act on is a dead end.
    """
    record = record_for(found, identifier, kind)
    if record is None or record.get("state") == AVAILABLE:
        return None
    return str(record.get("error") or "").strip() or "it would not open"


def unscanned_kind_detail(kind: str) -> str:
    """What to say about a kind the scan cannot see at all.

    Shared with the configuration page's device boxes, which render the same
    sentence about the same kinds, so the two always say the same thing.
    """
    return f"the scan does not cover {kind} cameras — check this one by hand"


def _missing_detail(device: DeclaredDevice) -> str:
    # Resolve unscanned kinds before examining discovery results.
    return "not found in the device scan"


def _unscanned_status(device: DeclaredDevice) -> DeviceStatus:
    """The verdict on a device the scan cannot enumerate: the console has
    nothing to report, and says so instead of reporting it as Unavailable."""
    return DeviceStatus(device, NOT_COVERED, unscanned_kind_detail(device.kind))


def statuses(devices, found=None, *, failure: str | None = None) -> list[DeviceStatus]:
    """Fold a scan result over the declared list.

    Total by construction: a declared device the scan never mentions is
    `Unavailable`, an unassigned slot is `Unavailable`, and a scan that failed
    outright makes every row `Failed` with the reason. Nothing here raises, so
    a rig half-configured mid-form still renders.
    """
    out = []
    for device in devices:
        # An empty device slot is unassigned regardless of scan state.
        if not device.identifier:
            out.append(DeviceStatus(device, UNAVAILABLE, "no device assigned"))
            continue
        # Discovery cannot establish failure for a device kind it never enumerated,
        # even when the scan itself failed.
        if is_unscanned_kind(device):
            out.append(_unscanned_status(device))
            continue
        if failure:
            out.append(DeviceStatus(device, FAILED, failure))
            continue
        if found is None:
            out.append(DeviceStatus(device, NOT_TESTED))
            continue
        record = _match(found, device.identifier, device.kind)
        if record is None:
            out.append(DeviceStatus(device, UNAVAILABLE, _missing_detail(device)))
        elif record.get("state") == AVAILABLE:
            out.append(DeviceStatus(device, CONNECTED))
        else:
            # Show the worker's specific open-failure reason so permission problems and
            # missing streams receive different corrective guidance.
            reason = str(record.get("error") or "").strip()
            out.append(DeviceStatus(device, FAILED, reason or "would not open"))
    return out


# ------------------------------------------------- typed text -> device name

EMPTY = "empty"
OK = "ok"
NONE = "none"
AMBIGUOUS = "ambiguous"


def _trailing_int(name: str):
    """Trailing integer of a canonical device name (UMI-044 -> 44)."""
    runs = re.findall(r"\d+", name)
    return int(runs[-1]) if runs else None


def matches_token(name: str, token: str) -> bool:
    """Whether a typed token identifies *name*.

    Operators type short tokens, so both rules apply: a pure-digit token
    matches the device's trailing number as an integer, so `8` finds
    `UMI-V4-0008` and `54` finds `oak-54`; anything with letters or separators
    matches as a case-insensitive suffix, so `A-01` finds `oak-A-01`. Typing
    more of the name is how an operator disambiguates, and that only works if
    both rules are here.
    """
    token = (token or "").strip()
    if not token:
        return False
    if token.isdigit():
        trailing = _trailing_int(name)
        return trailing is not None and trailing == int(token)
    return name.lower().endswith(token.lower())


def resolve_device(typed, options):
    """Resolve typed text to exactly one canonical device name.

    Returns ``(name | None, status)`` with status in
    ``{empty, ok, none, ambiguous}``. **Never fabricates a name**: the result
    is always an entry of *options*. A raw string passthrough would write `05`
    to the config as `05`, and two devices ending in `1` would silently become
    whichever the assembler happened to accept.
    """
    text = str(typed or "").strip()
    if not text:
        return None, EMPTY
    folded = text.lower()
    for name in options:
        if str(name).lower() == folded:
            return name, OK
    matches = [name for name in options if matches_token(str(name), text)]
    if len(matches) == 1:
        return matches[0], OK
    if not matches:
        return None, NONE
    return None, AMBIGUOUS


# ------------------------------------------------- one scan, read many ways


def identifier_for(record) -> str:
    """What a scan record is called in a config: registry name, else raw id.

    An unregistered device is stored by id rather than dropped, because a new
    station's `device_registry` is empty: under a strict "must be in the
    registry" rule a fresh install would offer nothing.
    """
    name = str(record.get("registry_name") or "").strip()
    if name:
        return name
    device_id = str(record.get("device_id") or "").strip()
    if record.get("kind") == "gripper" and device_id:
        from umi.utils.device_manager import normalize_umi_gripper_identifier

        return normalize_umi_gripper_identifier(device_id)
    return device_id


def detected_for(found, kind: str) -> list[dict]:
    """Scan records of one kind, in scan order.

    An empty list for a kind the scan cannot enumerate is the honest answer --
    see SCANNED_KINDS -- and the caller says so rather than reporting the
    hardware as missing.
    """
    if not found or not kind:
        return []
    return [
        record
        for record in found
        if isinstance(record, dict) and record.get("kind") == kind
    ]


def describe_record(record) -> str:
    """The metadata strip for one detected device.

    Everything a row shows besides the name: the raw id (which is what gets
    saved for unregistered hardware), and for an OAK the per-socket sensor
    module, because that is what decides whether a stream is given 640x400 or
    640x480.
    """
    parts = []
    device_id = str(record.get("device_id") or "").strip()
    if device_id and device_id != identifier_for(record):
        parts.append(f"id {device_id}")
    metadata = record.get("metadata") or {}
    sockets = metadata.get("sockets") or {}
    if sockets:
        parts.append(" ".join(f"{s}: {m}" for s, m in sorted(sockets.items())))
        # Warn when enumeration lacks the expected color sensor: a board can
        # enumerate successfully while a recording stream is unavailable.
        # These discovery warnings do not block config saving.
        if record.get("kind") == "oak" and "CAM_A" not in sockets:
            parts.append("no CAM_A — this board has no colour sensor to record from")
    ports = metadata.get("ports") or []
    if ports:
        parts.append(" ".join(str(p) for p in ports))
    if record.get("kind") == "uvc":
        # Model names cannot distinguish identical USB trigger cameras; show the
        # V4L2 device path rather than suggesting registry registration.
        return " · ".join(
            part
            for part in (
                *parts,
                str(metadata.get("name") or "").strip(),
                str(metadata.get("path") or "").strip(),
            )
            if part
        )
    if not record.get("registry_name"):
        parts.append("not in the device registry — the id is saved as-is")
    if record.get("state") != AVAILABLE:
        reason = str(record.get("error") or "").strip()
        parts.append(
            reason or "unavailable — check USB or close any app using it, then rescan"
        )
    return " · ".join(part for part in parts if part)


# ------------------------------------------------------------- per-app state


@dataclass
class DevicesState:
    """The scan this console last started.

    `/config` runs the only scan, and it certifies nothing: the result is
    inventory for the pickers and the config assembler.
    """

    job_id: str | None = None


# Use an RLock: ensure_scan calls helpers that reacquire this lock.
# Hold it across check-and-spawn to enforce one scan per app.
_ATTACH = threading.RLock()
# Register one shutdown handler per app-owned job registry.
_HOOKED = "jobs_reaped_on_shutdown"


def registry_for(app) -> JobRegistry:
    """This app's job registry, plus the shutdown that reaps its children.

    The reaping hook below is attached on every call, not only when this
    function created the registry, so a registry created anywhere else is
    reaped too. `jobs.start` uses `start_new_session=True`, so the scan child
    is detached; nothing else would ever collect it, and it would go on holding
    an exclusive OAK connection after the server has exited.

    The handler is per-app and bound to that app's own registry, so two apps in
    one process (which is every test session) cannot reap each other's
    children.
    """
    with _ATTACH:
        registry = getattr(app.state, "jobs", None)
        if registry is None:
            registry = JobRegistry()
            app.state.jobs = registry
        if not getattr(app.state, _HOOKED, False):
            app.router.add_event_handler("shutdown", registry.shutdown)
            setattr(app.state, _HOOKED, True)
        return registry


def state_for(app) -> DevicesState:
    with _ATTACH:
        state = getattr(app.state, "devices", None)
        if state is None:
            state = DevicesState()
            app.state.devices = state
        return state


def reset(app) -> None:
    """Hook called from Apply when the rig changed.

    Called through `readiness.reset_verification_for_new_rig`. There is nothing
    to reset: the scan result is inventory (which devices are attached to this
    bench, and the sensor module behind each OAK, which `config_assembler`
    reads to pick each stream's native resolution), and it is still true after
    a config edit.
    """


# ---------------------------------------------- the one scan both pages share


def current_job(app):
    """The scan this station last started, from whichever page started it.

    Every caller shares `DevicesState.job_id`. An OAK open is exclusive, so two
    scans running at once would fight over the same camera and both would
    report it unavailable, and the loser's error would name a USB cable that is
    fine.
    """
    state = state_for(app)
    if not state.job_id:
        return None
    try:
        return registry_for(app).get(state.job_id)
    except UnknownJob:
        # A replaced registry may have no current scan; return the empty state.
        return None


def outcome_for(job):
    """`(found, failure)` for one scan job: at most one is ever set.

    A finished job with no result line is a *software* fault (the child was
    replaced, or its output was truncated), so it is reported as a failure
    rather than as every device merely being absent, which would blame the rig.

    Every reason here is worded to read well both beside a per-device `Failed`
    badge, where the badge already supplies the verdict, and inside the
    configuration page's scan strip, where it is followed by "-- type device
    names by hand". So none of them names an act, and none repeats the banner
    at the top of the page.
    """
    if job is None or not job.finished:
        return None, None
    if job.state == CANCELLED_JOB:
        return None, "cancelled"
    found = parse_result(job.lines())
    if job.state == FAILED_JOB:
        return None, job.error or "no reason given"
    if found is None:
        return None, "no result, run it again"
    return found, None


def ensure_scan(
    app,
    argv,
    *,
    cwd,
    timeout: float,
    label: str = "device scan",
    force: bool = False,
):
    """Start a scan unless one already exists. Returns `(job, started)`.

    Never blocks and never opens hardware on the request path: `jobs.start`
    spawns a capped child and returns as soon as the pid exists. The
    already-exists arm is what makes an auto-scan on page load safe: a second
    tab, a reload, or another visit mid-scan all reuse the running child
    instead of racing it for an exclusive OAK connection.

    `force` is Refresh: re-run a scan that has already finished, because devices
    get attached after the page is open and a camera held by the viewer stays
    unavailable until that releases. A scan still running is never replaced,
    forced or not: the second child would be denied the same exclusive OAK
    connection the first one is holding, and would report it broken.

    The check and the spawn are one critical section. Split, they are a
    check-then-act: an auto-scan on page load plus a second tab plus a Refresh
    are ordinary simultaneous traffic, and two callers that both read "no scan
    running" both spawn one. The duplicate is exactly the failure the
    already-exists arm exists to prevent: `jobs.start` detaches the child with
    `start_new_session=True`, so the loser of the race for the exclusive OAK
    connection reports healthy hardware as broken, and the operator is sent to
    re-seat a cable that is fine.
    """
    with _ATTACH:
        existing = current_job(app)
        if existing is not None and (not force or not existing.finished):
            return existing, False
        state = state_for(app)
        job = job_control.start(
            registry_for(app), argv, cwd=cwd, timeout=timeout, label=label
        )
        state.job_id = job.id
        return job, True


# Configuration triggers module discovery; recording preflight uses the
# same shared scan and failure messages.

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# OAK discovery opens devices exclusively to read module details and may
# take tens of seconds for a full rig. Bound stalled scans at two minutes.
SCAN_TIMEOUT_S = 120.0

# Do not scan while camera verification owns the boards exclusively.
# Concurrent opens would report healthy cameras as unavailable.
CHECK_HOLDS_THE_CAMERAS = (
    "The camera check on Calibration is still running. Stop it there first."
)

# After cancelling verification, wait for the watcher to release its
# cameras before scanning. This is a maximum wait, not a fixed delay.
CHECK_RELEASE_WAIT_S = 6.0


def scan_argv() -> list[str]:
    """The command one device scan runs.

    `--probe` is not optional. Without it an iPhone is only *listed* by Record3D
    and a gripper is only a port in a directory listing, so a gripper whose
    `/dev/ttyACM*` is root-only (the single most common way a healthy looking
    rig fails at record time) would be reported connected.

    A function rather than a constant so tests can replace the entire child with
    a fake that sleeps, crashes or prints a canned result: no test in this suite
    may open real hardware.
    """
    return [sys.executable, "-m", "station_ui.scan_worker", "--probe"]

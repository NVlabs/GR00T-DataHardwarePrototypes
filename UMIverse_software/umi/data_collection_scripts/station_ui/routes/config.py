# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`/config` -- the session, the rig, and the file both of them produce.

This page is the production caller of `new_session_uid`,
`session_has_episodes`, `SETUP_PRESETS` and `ConfigAssembler`: it is what turns
an operator's choices into the recording config the station records from. The
module is deliberately thin -- it owns the HTTP shape and nothing else, and
delegates every rule to the module that already states it:

* which devices exist            -> `ConfigAssembler.template_requirements`
* what an optional camera is     -> `station_ui.core.optional_devices`
* what the session folder is     -> `station_ui.session`
* who may change any of it       -> `station_ui.rig`

Every mutating control is one submit button on one form, aimed with
`formaction`. That is what lets "add a head camera" carry the operator's
half-typed session fields with it: the whole form is posted, absorbed into the
draft, and only then is the row added. Nested forms are illegal, and a separate
little form per button would silently discard everything typed so far.
"""

from __future__ import annotations

import logging
import os
import pathlib
import unicodedata
from dataclasses import replace

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from jinja2 import TemplateNotFound
from starlette.concurrency import run_in_threadpool

from station_ui import calibration as cal
from station_ui import (
    config_form,
    device_summary,
    enforcement,
    gripper_pairing,
    identification,
    next_session,
    preview,
    readiness,
    recording,
    session_roll,
    tokens,
)
from station_ui import data_check_config as data_check
from station_ui import devices as device_state
from station_ui import seal as seal_state
from station_ui import session as session_state
from station_ui.camera_access import serialize_start
from station_ui.core import (
    device_detector,
    optional_devices,
    setup_presets,
    vive_probe,
)
from station_ui.core.config_assembler import (
    UNKNOWN_OPERATOR,
    UNKNOWN_TASK_DESCRIPTION,
    ConfigAssembler,
    DeviceSpec,
    SessionSpec,
    TemplateRequirements,
    optional_stream,
    resolve_hw_version,
)
from station_ui.core.device_scan import (
    DeviceScanner,
    resolution_for,
)
from station_ui.core.task_metadata import task_metadata
from station_ui.core.umi_gripper_devices import (
    list_umi_gripper_device_names_for_pipeline,
)
from station_ui.errors import (
    FieldError,
    RecordingInProgress,
    StaleOwner,
    stale_choice,
)
from station_ui.rig import claim, is_owner, require_owner
from station_ui.workflow import render_page
from umi.common.tasks import task_registry_store
from umi.utils import device_registry

logger = logging.getLogger(__name__)

router = APIRouter()

SEE_OTHER = 303
NO_CONTENT = 204
CONFLICT = 409
BAD_REQUEST = 400

# Only absorb requests marked as full forms; background scan requests
# omit fields and must not clear them.
FORM_MARKER = config_form.FORM_MARKER


def _assembler(request: Request, *, scanner=None) -> ConfigAssembler:
    """The assembler this request writes through.

    `scanner` is what stops every OAK reporting "not found in the device scan"
    on a bench where the operator has just scanned: see
    `ConfigAssembler.__init__`. Only Apply passes one, because Apply is the only
    caller that writes resolutions -- `output_path` and `template_requirements`
    do not look at hardware at all.
    """
    settings = request.app.state.settings
    return ConfigAssembler(
        settings.repo_root, state_dir=settings.state_dir, scanner=scanner
    )


def _requirements(request: Request, draft) -> TemplateRequirements:
    """What this preset's template demands. Empty if it cannot be read.

    A missing or unreadable template is a station install problem, not an
    operator mistake, so the page still renders and says so rather than 500ing
    on the way to telling them.
    """
    try:
        return _assembler(request).template_requirements(draft.preset_id)
    except (KeyError, OSError):
        return TemplateRequirements()


def _generated_path(request: Request, draft) -> pathlib.Path | None:
    """Where this preset's config is written, whether or not it exists yet.

    Asked of the assembler rather than rebuilt here, so the preview and the
    reopen can never point somewhere Apply does not write.

    The answer lives in `next_session`, which needs the same one from a job's
    supervisor thread where there is no request to hand: two derivations of the
    path Apply writes to would be two chances to disarm a file nobody is armed
    with. This is the request-shaped way in, and nothing more.
    """
    return next_session.generated_path(request.app, draft)


def _reopen(request: Request, draft) -> None:
    """Load the config already generated for this preset, if the draft is new.

    This is what "reopen" means to an operator, and it is the reason a session
    uid survives a server restart instead of being minted a second time into a
    second `raw/{task}/{session}/` prefix. Only an untouched draft is
    overwritten, so it can never discard work in progress.
    """
    touched = (
        draft.task_name
        or draft.operator_name
        or draft.grippers
        or draft.required_cameras
        or draft.optional
        or draft.relaxed_checks
    )
    if touched:
        return
    path = _generated_path(request, draft)
    if path is None or not path.is_file():
        return
    try:
        cfg = yaml.safe_load(path.read_text()) or {}
    except (OSError, ValueError, yaml.YAMLError):
        # If saved YAML is malformed, restore an editable draft so Apply can
        # regenerate a valid config.
        return
    if isinstance(cfg, dict):
        config_form.hydrate(draft, cfg)


# -------------------------------------------------------------- device fields

# Resolve registry lookups by kind at request time. UVC has no registry;
# that differs from a registry that contains no devices.
_REGISTRY_LOOKUP = {
    "oak": "get_available_oak_cameras",
    "iphone": "get_available_iphones",
}

# Use the same human-readable hardware-kind labels in fields and errors.
_KIND_LABEL = {
    "oak": "OAK camera",
    "iphone": "iPhone",
    "uvc": "UVC camera",
    "gripper": "UMI gripper",
}

_SOURCE_LABEL = {
    "oak_device": "OAK-FFC board",
    "oak_camera": "OAK camera",
    "iphone_camera": "iPhone",
    "uvc_camera": "UVC camera",
}

# Derive readable camera roles from source names, retaining an unknown
# role's words rather than exposing a raw source ID.
_ROLE_WORDS = {
    "umi": "UMI camera",
    "tracking": "tracking camera",
    "trigger": "trigger camera",
}

# Order grippers before wrist and tracking cameras. For multi-stream
# boards, use the highest-priority stream to label their shared device field.
# Unknown roles sort last without changing template order.
_ROLE_ORDER = ("gripper", "umi", "tracking", "trigger")

# Use descriptor hardware labels only to distinguish otherwise identical optional-camera
# rows.
_SECTION_WORDS = {
    "head": "Head camera",
    "bottom": "Bottom camera",
}


def _kind_label(kind: str) -> str:
    return _KIND_LABEL.get(kind, f"{kind} device" if kind else "device")


def _hardware_label(source_type: str, kind: str) -> str:
    return _SOURCE_LABEL.get(source_type) or _kind_label(kind)


def _arm_names(requirements) -> list[str]:
    """Every arm this template declares, in template order.

    Derived, never enumerated, and never re-sorted here. A gripper arm is a
    `numeric_sources[].name` minus `_umi_gripper`; a camera arm is a leading
    `<arm>_` segment that more than one source block shares. So a template that
    grows a third arm grows a third section, and a preset whose streams are
    named differently degrades to one shared group rather than to a wrong one.

    Order comes from the template because the page lays the arms out left on
    the left: the recording templates list the left arm's sources first (which
    is also what puts the left tile first in the cv2 preview), so the columns
    fall out of the file rather than out of a table of side names here. A
    hardcoded `("right", "left")` would put the two out of step the moment a
    template changed.
    """
    arms = list(requirements.gripper_arms)
    seen: dict[str, int] = {}
    for block in requirements.blocks:
        heads = {name.split("_", 1)[0] for name in block.stream_names if "_" in name}
        if len(heads) != 1:
            continue
        head = heads.pop()
        seen[head] = seen.get(head, 0) + 1
    # Two blocks deep before a prefix counts as an arm: one block sharing a word
    # with nothing else is a camera name, not a side of the rig.
    arms += [head for head, count in seen.items() if count > 1 and head not in arms]
    return arms


def _arm_of(names, arms) -> str:
    """The arm all of *names* belong to, or "" when they name no single one."""
    found = {
        arm
        for arm in arms
        for name in names
        if name == arm or name.startswith(f"{arm}_")
    }
    return found.pop() if len(found) == 1 else ""


def _arm_pos(arm: str, arms) -> int:
    """Where an arm's column sits. A device belonging to no single arm sorts
    last, which is where `_arm_groups` puts the Shared section too."""
    return arms.index(arm) if arm in arms else len(arms)


def _upper_first(text: str) -> str:
    """Capitalise without lowercasing the rest -- `str.capitalize` would turn
    "UMI camera" into "Umi camera"."""
    return text[:1].upper() + text[1:]


def _role_of(stream_name: str, arm: str) -> str:
    """What a stream name says it is, once the arm and the noun are stripped.

    `right_umi_camera` on the right arm is a "umi"; `right_extra_camera` is an
    "extra". Derived rather than looked up, so this never has to know the set of
    stream names any template happens to declare.
    """
    text = str(stream_name or "")
    if arm and text.startswith(f"{arm}_"):
        text = text[len(arm) + 1 :]
    return " ".join(word for word in text.split("_") if word and word != "camera")


def _role_rank(role: str) -> int:
    return _ROLE_ORDER.index(role) if role in _ROLE_ORDER else len(_ROLE_ORDER)


def _block_role(block, arms) -> str:
    """What one box's block IS, taken from the most important stream it feeds."""
    arm = _arm_of(block.stream_names, arms)
    roles = [_role_of(name, arm) for name in block.stream_names]
    # `min` is stable, so a block whose streams are all unknown falls back to
    # its first stream rather than to an arbitrary one.
    return min(roles, key=_role_rank) if roles else ""


def _umi_camera_slots(requirements) -> dict:
    """arm -> the block key of the camera that arm's gripper carries.

    Template-only: it reads `requirements`, never the draft, so the caller may
    compute it once and absorb may evaluate the binding after it has updated
    the grippers from the form.

    "The gripper's camera" is the block whose role is `umi`, derived from the
    stream names by `_block_role` -- so this never has to know that the board
    is an OAK, and a template that names its streams differently degrades to
    no link rather than to a wrong one.
    """
    arms = _arm_names(requirements)
    slots: dict[str, str] = {}
    for block in requirements.blocks:
        arm = _arm_of(block.stream_names, arms)
        if arm and _block_role(block, arms) == "umi":
            slots.setdefault(arm, block.key)
    return slots


def _link_state(linked: str, value: str) -> str:
    """`""` | `linked` | `overridden` for one camera row.

    `overridden` only once the operator's value differs -- an empty box on a
    bound gripper is about to be filled, not a deliberate re-pair.
    """
    if not linked:
        return ""
    if value and value.casefold() != linked.casefold():
        return "overridden"
    return "linked"


def _plain_label(block, arms) -> str:
    """One block's field label, in the words the operator uses for it.

    "Right UMI camera", not "Right OAK-FFC board serving right_trigger_camera
    and right_umi_camera". A data collector plugs a camera into an arm; the
    board it hangs off and the two streams it feeds are wiring, and wiring is
    not on this form.
    """
    arm = _arm_of(block.stream_names, arms)
    role = _block_role(block, arms)
    words = _ROLE_WORDS.get(role) or (f"{role} camera" if role else "camera")
    return _upper_first(f"{arm} {words}" if arm else words)


def _optional_label(assignment) -> tuple[str, str]:
    """`(plain label, the word that separates it from a twin)` for one row."""
    try:
        descriptor = optional_devices.get_optional_type(assignment.type_id)
    except KeyError:
        return "Camera", ""
    section = descriptor.section
    label = _SECTION_WORDS.get(section) or _upper_first(f"{section} camera")
    hardware = descriptor.label
    if hardware.lower().startswith(f"{section} "):
        hardware = hardware[len(section) + 1 :]
    if section == "head":
        # Include head-camera hardware type in its label so the required ID format is
        # clear.
        return f"{label} ({hardware})", hardware
    return label, hardware


def _distinct(views) -> list[dict]:
    """Make every plain label name exactly one box on this page.

    Plainer is not worth ambiguous: a head may carry an iPhone and an OAK at
    once, so two rows really can arrive at "Head camera". A box that could be
    either of two devices is a box the operator fills in wrong and the page
    cannot explain -- so a collision is resolved, first by the hardware word
    (which is what the operator is holding), then by counting.
    """

    def clashing() -> set:
        labels = [view["label"] for view in views]
        return {label for label in labels if labels.count(label) > 1}

    tied = clashing()
    for view in views:
        if view["label"] in tied and view.get("hardware"):
            view["label"] = f"{view['label']} ({view['hardware']})"
    tied = clashing()
    counted: dict[str, int] = {}
    for view in views:
        label = view["label"]
        if label in tied:
            counted[label] = counted.get(label, 0) + 1
            view["label"] = f"{label} {counted[label]}"
    return views


def _registry_names(kind: str, preset) -> list[str]:
    """Every device of this kind the device registry knows about.

    Station-wide, and deliberately still offered. A new station's registry is
    empty, so its options come from the scan (see `_options`); the scan is also
    what says which registered devices are actually here.
    """
    if kind == "gripper":
        return _gripper_choices(preset)
    attribute = _REGISTRY_LOOKUP.get(kind)
    lookup = getattr(device_detector, attribute, None) if attribute else None
    if lookup is None:
        return []
    try:
        return list(lookup())
    except (KeyError, ValueError, OSError):
        # If the registry cannot load, retain typed selections without labeling them
        # invalid.
        return []


def _options(kind: str, preset, found) -> list[str]:
    """What a typed value may resolve to: the registry, widened by the scan.

    The widening is not cosmetic. An attached device that is not in the
    registry is stored by its raw id, and without it in the option list the
    operator's own selection resolves to `none`.
    """
    names = _registry_names(kind, preset)
    seen = set(names)
    for record in device_state.detected_for(found, kind):
        identifier = device_state.identifier_for(record)
        if identifier and identifier not in seen:
            seen.add(identifier)
            names.append(identifier)
    return names


def _resolver(preset, found):
    """`(kind, typed) -> (name | None, status)`, closed over one scan."""
    cache: dict[str, list[str]] = {}

    def resolve(kind, typed):
        if kind not in cache:
            cache[kind] = _options(kind, preset, found)
        return device_state.resolve_device(typed, cache[kind])

    return resolve


def _registered_check(preset):
    """`(kind, name) -> bool`: whether the device registry knows this exact device.

    The device box widens its options with the scan (`_options`), so an attached
    device the registry has never heard of resolves cleanly to its raw serial
    with an OK status, and nothing else on screen would say it is unregistered.
    This is what the "attached but not a registered device" warning keys on, so
    a serial that will not carry a name downstream is called out rather than
    saved silently.
    """
    cache: dict[str, set[str]] = {}

    def is_registered(kind, name):
        if kind not in cache:
            cache[kind] = {str(n).lower() for n in _registry_names(kind, preset)}
        names = cache[kind]
        # An unavailable lookup cannot establish that a device is unregistered.
        if not names:
            return True
        return str(name).lower() in names

    return is_registered


def _scan_running(job) -> bool:
    """Whether a device scan is in flight right now.

    One spelling, because two things key off it and they must not disagree: the
    strip polls while it is true, and the pickers stay empty while it is (see
    `_choices`).
    """
    return job is not None and not job.finished


def _choices(kind: str, preset, found, scanning: bool) -> list[dict]:
    """The datalist behind one device box: the scan first, then the registry.

    One control per device, so this list carries both what was found and what
    is registered. Scan results lead because they are the devices actually on
    this bench; a device the scan could not open is listed rather than dropped,
    because "gone" and "here but wedged" need different actions from the
    operator and a vanished row says neither.

    `note` becomes the option's `label`, which is what a browser shows beside
    the value in the dropdown -- never the value itself, so what a click inserts
    into the box is exactly the identifier the config stores.

    The status line under the box says only "connected"; this note says more
    because it is read while choosing between two boards, and the sensor module
    on each socket is what decides whether that stream is written 640x400 or
    640x480 -- a 4:3 IMX378 given 16:10 loses 17% of its FOV
    (tests/station_ui/test_migration_parity.py, "the picker shows the sensor
    module on each socket"). Once a board is chosen, the strip below says
    "connected" and the modules stop being a choice the operator is making.
    """
    options = []
    seen = set()
    for record in device_state.detected_for(found, kind):
        identifier = device_state.identifier_for(record)
        if not identifier or identifier in seen:
            continue
        seen.add(identifier)
        available = record.get("state") == device_state.AVAILABLE
        detail = device_state.describe_record(record)
        state = "connected" if available else "unavailable"
        options.append(
            {"value": identifier, "note": f"{state} · {detail}" if detail else state}
        )
    # After discovery, offer connected devices rather than every registry entry.
    # Before the first scan, use registry choices; during a scan, wait for its
    # results. For kinds discovery does not cover, the registry remains the
    # only source of choices. Manual identifiers remain allowed.
    if not device_state.scan_covers(kind) or (found is None and not scanning):
        options += [
            {"value": name, "note": ""}
            for name in _registry_names(kind, preset)
            if name not in seen
        ]
    return options


def _presence(kind: str, value: str, found, record) -> tuple[str, str]:
    """`(state, text)` for a resolved device: is it actually on this bench?

    `state` is one of `""`, `ok`, `warn` -- it only ever selects an icon that
    sits beside the words, never replaces them, because brand green and healthy
    green are the same hue on this console.

    The scan's own words about a device (`describe_record`: the raw id and the
    sensor module on each socket) are left out. Standing under a box that is
    already filled in, they are strings the data collector cannot act on --
    "connected" is the whole of what they need to know.
    """
    if not value:
        return "", ""
    if record is not None:
        if record.get("state") == device_state.AVAILABLE:
            return "ok", "connected"
        return "warn", "not working — check the cable, then Refresh devices"
    if kind and not device_state.scan_covers(kind):
        return "", device_state.unscanned_kind_detail(kind)
    if found is None:
        return "", "not scanned yet"
    return "warn", "not connected"


def _device_view(
    *,
    field,
    label,
    kind,
    value,
    preset,
    found,
    scanning,
    errors,
    resolve,
    registered=None,
) -> dict:
    """One device row, resolved and described.

    Resolution runs at render time as well as on absorb, so the message is
    right after a redirect too: absorb rewrote every value it could resolve, so
    anything still unresolved here is unresolvable, and the row says so.
    """
    value = str(value or "")
    name, status = resolve(kind, value)
    record = device_state.record_for(found, value, kind)
    presence_state, presence_text = _presence(kind, value, found, record)

    kind_label = _kind_label(kind)
    message, message_state = presence_text, presence_state
    if status == device_state.AMBIGUOUS:
        message_state = "error"
        message = f"'{value}' matches more than one {kind_label} — type more of it."
    elif status == device_state.NONE:
        message_state = "warn"
        message = f"'{value}' is not a known {kind_label} — saved as typed."
    elif (
        status == device_state.OK
        and name
        and registered is not None
        and not registered(kind, name)
    ):
        # Accept a connected raw serial that has no registry name, with a warning
        # that recorded identity will lack the friendly alias.
        message_state = "warn"
        message = (
            f"'{name}' is attached but not a registered {kind_label} — "
            "recorded by its serial id."
        )
    elif status == device_state.EMPTY:
        message_state, message = "", ""

    view = {
        "field": field,
        "list_id": config_form.dom_id("choices", field),
        "message_id": config_form.dom_id("message", field),
        "link_id": config_form.dom_id("link", field),
        "label": label,
        "kind": kind,
        "kind_label": kind_label,
        "value": value,
        "resolved": name or "",
        "status": status,
        "choices": _choices(kind, preset, found, scanning),
        # Carry scan availability on each row so standalone optional-section fragments
        # can render their device lists consistently.
        "scanning": scanning,
        "message": message,
        "message_state": message_state,
        # Display the linked camera through the same resolver used when saving the
        # draft.
        "linked": "",
        "link_state": "",
        # Build the row's swap target with the same dom_id function as its card.
        "arm_id": "",
    }
    # Show assembly rejection ahead of device-connection status.
    view["error"] = errors.get(field, "")
    view["error_id"] = config_form.error_id(field)
    return view


def _preset_for(draft):
    """The draft's preset, or None. A preset id nothing declares is a listing
    problem the operator fixes on /, never a 500 on the way there."""
    if not draft.preset_id:
        return None
    try:
        return setup_presets.get_preset(draft.preset_id)
    except KeyError:
        return None


def _device_views(request: Request, draft, requirements, errors=None) -> list[dict]:
    """One view per device slot, in the order the page renders them.

    Shared by the renderer and by Apply's gate, so the row that reports a
    problem and the row that refuses to save can never disagree about which
    rows exist or what they resolve to.
    """
    errors = errors or {}
    preset = _preset_for(draft)
    job = device_state.current_job(request.app)
    found, _ = device_state.outcome_for(job)
    scanning = _scan_running(job)
    resolve = _resolver(preset, found)
    registered = _registered_check(preset)

    def view(field, label, kind, value, **extra):
        built = _device_view(
            field=field,
            label=label,
            kind=kind,
            value=value,
            preset=preset,
            found=found,
            scanning=scanning,
            errors=errors,
            resolve=resolve,
            registered=registered,
        )
        built.update(extra)
        return built

    arms = _arm_names(requirements)
    # Resolve gripper/camera links once for the page, using the same lookup as absorb.
    linked_by_block = {}
    for arm, block_key in _umi_camera_slots(requirements).items():
        camera = device_registry.camera_for_gripper(draft.grippers.get(arm, ""))
        if camera:
            linked_by_block[block_key] = camera
    required = [
        view(
            config_form.camera_field(block.key),
            _plain_label(block, arms),
            block.device_kind,
            draft.required_cameras.get(block.key, ""),
            slot="camera",
            name=block.key,
            arm=_arm_of(block.stream_names, arms),
            role=_block_role(block, arms),
            hardware=_hardware_label(block.source_type, block.device_kind),
            linked=linked_by_block.get(block.key, ""),
            link_state=_link_state(
                linked_by_block.get(block.key, ""),
                draft.required_cameras.get(block.key, ""),
            ),
        )
        for block in requirements.blocks
    ]
    required += [
        view(
            config_form.gripper_field(arm),
            f"{arm.capitalize()} UMI gripper",
            "gripper",
            draft.grippers.get(arm, ""),
            slot="gripper",
            arm=arm,
            arm_id=config_form.dom_id("arm", arm),
            role="gripper",
        )
        for arm in requirements.gripper_arms
    ]
    # Sort by arm, then role; preserve template order within equivalent roles.
    required.sort(
        key=lambda v: (_arm_pos(v.get("arm", ""), arms), _role_rank(v["role"]))
    )
    views = required
    views += [
        view(
            config_form.optional_field(assignment.logical_name, "device"),
            _optional_label(assignment)[0],
            _optional_kind(assignment.type_id),
            assignment.device,
            slot="optional",
            logical_name=assignment.logical_name,
            hardware=_optional_label(assignment)[1],
        )
        for assignment in optional_slots(draft)
    ]
    return _distinct(views)


def optional_slots(draft) -> list:
    """Every optional camera the page renders a box for: the ones that were added.

    Optional cameras are added and removed explicitly, so a camera that is not
    in the draft is not on the page. `draft.optional` is already in the order
    the operator built it, per section.
    """
    return list(draft.optional)


def _optional_kind(type_id: str) -> str:
    try:
        return optional_devices.get_optional_type(type_id).device_kind
    except KeyError:
        return ""


def _ambiguous(views) -> dict:
    """Field -> message for every row whose text names more than one device.

    Refused rather than written: an ambiguous token resolves to no name, and a
    form with no such gate would write the raw token instead -- `05` reaching a
    config as `05` is a session recorded from nothing.

    `none` deliberately does not block. It is unambiguous -- exactly what the
    operator typed -- and blocking it would make the page unusable whenever the
    device registry is empty, as it is on a new station.
    """
    return {
        view["field"]: view["message"]
        for view in views
        if view["status"] == device_state.AMBIGUOUS
    }


# ------------------------------------------------------------- the shared scan


def _scan_context(request: Request, token) -> dict:
    """Everything the scan card renders from, for the page and the poller."""
    rig = request.app.state.rig
    job = device_state.current_job(request.app)
    found, failure = device_state.outcome_for(job)
    running = _scan_running(job)
    return {
        "can_edit": is_owner(rig, token),
        "scan_running": running,
        "scan_failure": failure,
        "scan_count": len(found) if found is not None else None,
        # Include the scan card state in full-page and polling-fragment responses.
        "scan_error": "",
        # Auto-scan only before the first attempt; failures and cancellations must
        # not trigger a loop of exclusive camera opens.
        "scan_auto": job is None,
        "device_summary": _device_summary(found, failure, running),
    }


def _device_summary(found, failure, running) -> dict:
    """The scan's devices grouped by gripper, for the table above the boxes.

    Built from the scan this page already runs, never a second one. Inside a
    try: a record shaped in a way the grouping did not expect must cost the
    operator this table, not the page.
    """
    rows = None
    if found is not None:
        try:
            rows = device_summary.rows(found)
        except Exception:
            logger.exception("device summary could not be built")
            failure = "the scan result could not be read"
    return {
        "rows": rows,
        "failure": failure,
        "running": running,
        "empty": device_summary.empty_message(found),
        "columns": device_summary.COLUMNS,
        "port_heading": device_summary.PORT_HEADING,
        "unavailable": device_summary.UNAVAILABLE,
        # The columns whose ids open a live preview (`routes/preview.py`).
        "previewable": preview.KINDS,
    }


def _start_scan(request: Request):
    """This page's inventory scan: what is attached to the bench right now.

    It fills the pickers and hands the assembler each OAK's sensor module. It
    runs through the one shared scan job (`devices.ensure_scan`), because an OAK
    open is exclusive and two scans would fight over one board and both report
    it unavailable.

    The result is not a device test. The scan fires from the strip's own
    `hx-trigger="load"` on a page the operator has only opened, and from the
    Refresh they press because they have just swapped hardware, so nothing here
    marks the rig as tested.
    """
    identification.forget_verification(request.app)
    return device_state.ensure_scan(
        request.app,
        device_state.scan_argv(),
        cwd=device_state.REPO_ROOT,
        timeout=device_state.SCAN_TIMEOUT_S,
        label="device scan",
        force=True,
    )


def _device_lists(views) -> list[dict]:
    """Every `<datalist>` the page needs, once each, in row order.

    Deduplicated on the id rather than trusting the rows to be distinct: two
    streams of one OAK board share a field and therefore a list id, and two
    `<datalist>` elements with one id is a document where the browser picks.
    """
    lists = []
    seen = set()
    for view in views:
        list_id = view.get("list_id")
        if not list_id or list_id in seen:
            continue
        seen.add(list_id)
        lists.append(
            {
                "id": list_id,
                "choices": view.get("choices") or [],
                "scanning": bool(view.get("scanning")),
            }
        )
    return lists


def _arm_groups(views, requirements) -> list[dict]:
    """The required-device fields, one group per arm of the rig.

    The operator holds a right-hand board and a left-hand board, so the form is
    laid out the way the bench is rather than as one undifferentiated grid. Both
    the arms and the membership come from the template, so a preset with one
    arm renders one group and a template that grows a device grows a field
    inside the right one.
    """
    arms = _arm_names(requirements)
    wanted = [v for v in views if v["slot"] in ("camera", "gripper")]
    groups = [
        {
            "arm": arm,
            "label": f"{arm.capitalize()} arm",
            "id": config_form.dom_id("arm", arm),
            "fields": [v for v in wanted if v.get("arm") == arm],
        }
        for arm in arms
    ]
    groups = [group for group in groups if group["fields"]]
    # Render required devices without an arm assignment too; every declared
    # slot must remain configurable.
    shared = [v for v in wanted if not v.get("arm")]
    if shared:
        groups.append(
            {
                "arm": "",
                "label": "Shared devices",
                "id": config_form.dom_id("arm", "shared"),
                "fields": shared,
            }
        )
    return groups


# Explain missing OAK discovery results with the action needed to obtain them.
SCAN_RUNNING_MESSAGE = "The device scan is still running — wait for it to finish, then start the action again."
NOT_SCANNED_MESSAGE = (
    "Not seen by the device scan, so its picture size cannot be set. Free or "
    "re-seat the camera, press “Refresh devices”, then start the action again."
)


def _oak_sockets(draft, views, requirements):
    """``(view, socket)`` for every OAK stream the form names: the arms' boards
    from the template blocks, the head from the optional rows. Apply checks
    each socket for a scanned sensor before choosing its picture size."""
    by_key = {view.get("name"): view for view in views if view["slot"] == "camera"}
    by_field = {view["field"]: view for view in views}
    for block in requirements.blocks:
        view = by_key.get(block.key)
        if view is None:
            continue
        for stream in block.streams:
            if stream.socket:
                yield view, stream.socket
    for assignment in optional_slots(draft):
        view = by_field.get(
            config_form.optional_field(assignment.logical_name, "device")
        )
        try:
            descriptor = optional_devices.get_optional_type(assignment.type_id)
        except KeyError:
            continue
        if view is None:
            continue
        stream = optional_stream(descriptor.source_type, assignment.logical_name)
        if stream.socket:
            yield view, stream.socket


def _unscanned_oak_fields(draft, views, requirements, found, job) -> tuple[dict, str]:
    """``field -> message`` for every OAK row whose stream size would be a
    guess, and the banner to show above them. Empty when every OAK the form
    names has a scanned module behind each socket it records from.

    This blocks Apply rather than warning. Without the scan the console would
    write the wrist default for every OAK; a head camera with a 4:3 IMX378 would
    then be cropped to 16:10 by the recorder, and the data check would pass the
    episodes because the file matches the config that asked for it. An OAK's
    size is the one config value the console cannot know without the scan, so
    without the scan it does not write one.

    OAK rows only. `unavailable_devices` deliberately lets a device the scan
    never saw through, because configs are prepped before hardware reaches the
    bench and the registry falls back offline; that stays true for phones and
    grippers, whose configs do not depend on what a scan reads. The pairs come
    from `_oak_sockets`, so a row with no socket to scan is never gated.
    """
    scanning = _scan_running(job)
    scanner = DeviceScanner.from_records(found or [])
    out: dict = {}
    for view, socket in _oak_sockets(draft, views, requirements):
        field = view.get("field")
        device = view.get("resolved") or view.get("value") or ""
        if not field or not device or field in out:
            continue
        _, warning = resolution_for(device, socket, scanner)
        if warning is None:
            continue
        out[field] = SCAN_RUNNING_MESSAGE if scanning else NOT_SCANNED_MESSAGE
    if not out:
        return {}, ""
    return out, (
        "Wait for the device scan to finish."
        if scanning
        else "A camera has not been scanned, so its picture size is unknown."
    )


def _diagram_context(request: Request, draft, requirements, views) -> dict:
    """The V4 diagrams, using the latest scan for the page and live swaps."""
    found, _ = device_state.outcome_for(device_state.current_job(request.app))
    return _gripper_diagrams_context(draft, requirements, views, found)


def _gripper_diagrams_context(draft, requirements, views, found) -> dict:
    """V4 assignments and pairing evidence, for the full page and live swaps."""
    preset = _preset_for(draft)
    inventory = None
    diagrams = []
    if preset and preset.gripper_generation == "v4":
        inventory = gripper_pairing.evidence(found)
        # A suffix is resolved against the registry plus the scan, just as
        # Apply resolves it. The dropdown alone only lists connected devices.
        inventory["options"] = {
            kind: _options(kind, preset, found)
            for kind in gripper_pairing.ROLES.values()
        }
        for group in _arm_groups(views, requirements):
            if not group["arm"]:
                continue
            fields = {
                "gripper" if field["slot"] == "gripper" else field.get("role"): field
                for field in group["fields"]
            }
            parts = [
                {
                    "role": role,
                    "label": label,
                    "field": fields[role]["field"],
                    "kind": fields[role]["kind"],
                    "value": fields[role].get("resolved") or fields[role]["value"],
                    "number": number,
                }
                for role, label, number in (
                    ("tracking", "Phone", 1),
                    ("umi", "Top camera · OAK", 2),
                    ("trigger", "Trigger camera · USB", 3),
                    ("gripper", "Gripper controller", 4),
                )
                if role in fields
            ]
            values = {part["role"]: part["value"] for part in parts}
            diagrams.append(
                {
                    "arm": group["arm"],
                    "label": f"{group['arm'].capitalize()} V4 gripper",
                    "parts": parts,
                    "has_phone": "tracking" in fields,
                    **gripper_pairing.check(values, inventory),
                }
            )
    return {"gripper_diagrams": diagrams, "gripper_pairing_evidence": inventory}


def _pairing_errors(draft, requirements, views, found) -> dict:
    context = _gripper_diagrams_context(draft, requirements, views, found)
    return {
        part["field"]: diagram["message"]
        for diagram in context["gripper_diagrams"]
        for part in diagram["parts"]
        if part["role"] in diagram["affected"]
    }


def _rows(draft):
    """One rendering row per optional camera, in the order they were added."""
    rows = []
    for assignment in draft.optional:
        descriptor = optional_devices.get_optional_type(assignment.type_id)
        rows.append(
            {
                "type_id": assignment.type_id,
                "label": descriptor.label,
                "section": descriptor.section,
                # Share section DOM IDs between Add and Remove responses.
                "section_id": _section_id(descriptor.section),
                "kind": descriptor.device_kind,
                "logical_name": assignment.logical_name,
                "device": assignment.device,
                "placement": assignment.placement,
                "device_field": config_form.optional_field(
                    assignment.logical_name, "device"
                ),
                "placement_field": config_form.optional_field(
                    assignment.logical_name, "placement"
                ),
            }
        )
    return rows


def _addable(draft, section):
    """The types this section still accepts, each carrying whether it may be
    added. Rendered disabled rather than hidden, so the operator can see that a
    head already has an iPhone instead of wondering where the option went."""
    return [
        {
            "id": descriptor.id,
            "label": descriptor.label,
            "can_add": optional_devices.can_add(descriptor.id, draft.optional),
        }
        for descriptor in optional_devices.types_for_section(section)
    ]


# Build optional sections from descriptors so rendering follows supported hardware.
def _sections() -> tuple[str, ...]:
    seen = []
    for descriptor in optional_devices.OPTIONAL_DEVICE_TYPES:
        if descriptor.section not in seen:
            seen.append(descriptor.section)
    return tuple(seen)


def _section_id(section: str) -> str:
    return f"{section}-section"


def _section_template(section: str) -> str:
    return f"_partials/{section}_section.html"


def _optional_context(draft, views, errors, preset, confirm_remove: str = "") -> dict:
    """What the head and side sections render from.

    Shared by the whole page and by the htmx swap that replaces one section, so
    a fragment can never disagree with the page it was cut out of.

    `confirm_remove` names the one row that has asked to be confirmed. It lives
    on the row rather than at the top of the page because the answer to Remove
    is the section alone -- a banner rendered above a page that was never sent
    is a click that silently does nothing.
    """
    by_logical = {v["logical_name"]: v for v in views if v["slot"] == "optional"}
    rows = _rows(draft)
    for row in rows:
        row["error"] = errors.get(row["device_field"], "")
        row["error_id"] = config_form.error_id(row["device_field"])
        row["device_view"] = by_logical.get(row["logical_name"])
        row["confirm"] = bool(confirm_remove) and row["logical_name"] == confirm_remove
        row["confirm_message"] = (
            f"Remove {row['device']}? This session will not record it."
        )

    # Build placement choices per section while preserving accepted persisted aliases.
    context = {
        "placements": optional_devices.placement_options,
        "section_error": "",
    }
    for section in _sections():
        context[f"{section}_rows"] = [r for r in rows if r["section"] == section]
        context[f"{section}_types"] = _addable(draft, section) if preset else []
        context[f"{section}_id"] = _section_id(section)
    return context


def _camera_label(name: str) -> str:
    """A camera's config name as words: right_tracking_camera -> the same,
    underscores opened out, first letter up. Enough to read; the exact source
    name still travels as the option value the assembler validates."""
    text = str(name or "").replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else ""


def _reference_context(request: Request, draft) -> dict:
    """The world-frame reference picker, or ``{"reference": None}`` when there is
    nothing to choose between.

    Built from the same assembler that will write the config, so the cameras
    offered here are exactly the ones it can anchor on. An override that names a
    camera the rig no longer records falls back to Automatic in the picker -- the
    operator sees the auto-pick preselected rather than a dead name, and the next
    submit clears the stale override.
    """
    if not draft.preset_id:
        return {"reference": None}
    devices = DeviceSpec(
        grippers=dict(draft.grippers),
        required_cameras=dict(draft.required_cameras),
        optional=tuple(draft.optional),
        reference_override=draft.reference_override,
    )
    try:
        choices = _assembler(request).reference_choices(draft.preset_id, devices)
    except (KeyError, OSError):
        return {"reference": None}
    if not choices.candidates:
        return {"reference": None}

    names = {candidate.name for candidate in choices.candidates}
    override = str(draft.reference_override or "")
    selected = override if override in names else ""

    options = [
        {
            "value": "",
            "label": f"Automatic — {_camera_label(choices.auto)}",
            "selected": selected == "",
        }
    ]
    for candidate in choices.candidates:
        label = _camera_label(candidate.name)
        if candidate.name == choices.auto:
            label += " (automatic default)"
        options.append(
            {
                "value": candidate.name,
                "label": label,
                "selected": candidate.name == selected,
            }
        )
    return {
        "reference": {
            "field": config_form.REFERENCE_FIELD,
            "auto_label": _camera_label(choices.auto),
            "options": options,
        }
    }


# ------------------------------------------------------ the Vive bridge and pucks


def _probe_summary(result) -> dict:
    """What the last probe found, split the way `config_scan.html` splits a scan.

    Two places, because they answer different questions and the operator reads
    them at different moments:

    * `probe_badge` -- the headline, rendered beside the button that asked, so
      the answer appears where the operator pressed.
    * `message` -- the detail under the box: how long the bridge took, or why it
      said nothing. It does not repeat the count; the badge owns that, and two
      copies drift.

    One `message_state` between them, not one each. There is a single verdict
    here -- how the probe went -- and the badge and the sentence are two
    renderings of it; a second field would be a second answer that can disagree
    with the first.

    Both empty until Test connection has been pressed, because this page never
    asks the bridge on its own. Never "error": a probe that failed blocks
    nothing -- a bench is routinely configured before the bridge is up -- so it
    is the amber weight `config_scan.html` gives a scan that could not run, not
    the tinted one reserved for a rejection that stopped the save.
    """
    if result is None:
        return {"message": "", "message_state": "", "probe_badge": ""}
    if not result.ok:
        return {
            "message": result.error or "The bridge did not answer.",
            "message_state": "warn",
            "probe_badge": "No answer",
        }
    count = len(result.keys)
    return {
        "message": f"replied in {result.rtt_ms:.0f} ms",
        # A bridge reply with no trackers confirms connectivity, not recording
        # readiness.
        "message_state": "ok" if count else "warn",
        "probe_badge": (
            f"{count} tracker{'' if count == 1 else 's'} found"
            if count
            else "No trackers"
        ),
    }


def _vive_key_value(draft, source) -> str:
    """The key this arm will record: the operator's when they named one, the
    template's when they left the box empty.

    One rule, one place. The box renders it, `_errors_for` matches the
    assembler's rejections against it, and `_reject_shared_devices` refuses a
    duplicate on exactly this value -- so a row whose key was never typed is
    still the row a "both arms name one puck" message is about.

    The fallback is applied here, in the view, and never written into the
    draft. Seeding the draft would erase the difference between "the operator
    has not chosen" and "the operator typed the template's value", and it is
    that difference `ConfigAssembler._apply_vive` relies on to leave an
    untouched rig writing exactly what it wrote before.
    """
    return draft.vive_keys.get(source.arm) or source.tracker_key


def _vive_row(source, draft, result, published, errors) -> dict:
    """One arm's tracker-key picker."""
    field = config_form.vive_key_field(source.arm)
    value = _vive_key_value(draft, source)
    message, state = "", ""
    # Compare keys only after a probe has returned tracker information.
    if published:
        if value not in published:
            message, state = "not a tracker this bridge publishes", "warn"
        elif value in result.tracked:
            message, state = "tracked", "ok"
        else:
            # Published but not tracking: a puck that is asleep, occluded or
            # unpaired. The name is right, the hardware is not ready.
            message, state = "published, but not tracking right now", "warn"
    return {
        "field": field,
        "list_id": config_form.dom_id("vive-keys", source.arm),
        "message_id": config_form.dom_id("message", field),
        "error_id": config_form.error_id(field),
        "label": f"{_upper_first(source.arm)} tracker",
        "value": value,
        "message": message,
        "message_state": state,
        # A field's assembly error takes precedence over probe status.
        "error": (errors or {}).get(field, ""),
    }


def _vive_context(request: Request, draft, requirements, errors=None) -> dict | None:
    """What `_partials/vive_section.html` renders from, or None for a rig that
    declares no tracker.

    None rather than an empty dict, so the one question "is there a section
    here at all" is answered in one place: the page renders nothing for it and
    the probe route answers 204 rather than an empty body htmx would swap.
    """
    sources = requirements.vive_sources
    if not sources:
        return None

    # The form supports one Vive endpoint per rig. All configured tracker
    # sources are rewritten to that address; templates with multiple bridges
    # are therefore outside this form's model.
    endpoint = draft.vive_endpoint or sources[0].endpoint

    # Store the latest probe with its address on app.state, not in recording
    # config. Discard it when the field changes to avoid validating keys
    # against another bridge.
    probed, result = getattr(request.app.state, "last_vive_probe", (None, None))
    if probed != endpoint:
        result = None
    published = list(result.keys) if result is not None and result.ok else []

    return {
        "field": config_form.VIVE_ENDPOINT_FIELD,
        "message_id": config_form.dom_id("message", config_form.VIVE_ENDPOINT_FIELD),
        "endpoint": config_form.display_vive_endpoint(endpoint),
        "rows": [
            _vive_row(source, draft, result, published, errors) for source in sources
        ],
        # Use the shared device-choice shape for the tracker dropdown.
        "options": [
            {"value": key, "note": "tracked" if key in result.tracked else ""}
            for key in published
        ],
        **_probe_summary(result),
    }


# ------------------------------------------------------------ the task name

# Distinguish blank, registered and new tasks from an unreadable task list.
# A failed lookup cannot establish that a task is new.
TASK_EMPTY = "empty"
TASK_KNOWN = "known"
TASK_NEW = "new"
TASK_UNREADABLE = "unreadable"

_TASK_MESSAGE = {
    TASK_EMPTY: "",
    TASK_NEW: (
        "New task — describe it below. Starting calibration or recording saves it to this station's task "
        "list, so it can be picked next time."
    ),
    TASK_UNREADABLE: (
        "Task list unavailable — recording is disabled until this station can "
        "read its task list. Restart the console, and check the station log if "
        "this persists."
    ),
}


def _task_list():
    """Every task on this station's list, or None when it could not be read.

    None is not `{}`. `task_registry_store` already falls back from
    `config/tasks.json` to the snapshot bundled in the repo, so reaching this
    exception means the station has no list to check against. An empty list is
    a station with no tasks yet, where every name is a new one.
    """
    try:
        tasks = task_registry_store.get_all_tasks()
    except Exception:
        # Keep the form usable when registry loading fails.
        return None
    return dict(tasks) if tasks else {}


def _task_state(name: str, tasks) -> str:
    if not name:
        return TASK_EMPTY
    if tasks is None:
        return TASK_UNREADABLE
    return TASK_KNOWN if name in tasks else TASK_NEW


def _task_options(tasks) -> list[dict]:
    """The previous tasks the Task box offers, each with what it records as."""
    return [
        {"value": name, "label": str((entry or {}).get("short_description") or "")}
        for name, entry in sorted((tasks or {}).items(), key=lambda kv: kv[0].lower())
    ]


def _new_task_errors(draft) -> dict:
    """What a new task still needs before Apply can save it to the list.

    Only the name: the description is optional and is recorded as "Unknown"
    when left blank.
    """
    errors = {}
    # Reject task text with no usable path characters rather than directing
    # unrelated sessions to raw/unnamed.
    if not any(ch.isalnum() for ch in draft.task_name):
        errors["task_name"] = (
            "Use letters or numbers in the task name — it names the session folder."
        )
    return errors


# Allowed in an operator name besides letters: "Mary-Jane O'Brien", "J. Doe".
_NAME_MARKS = frozenset(" -'’.")


def _operator_name_ok(name: str) -> bool:
    """Blank, or a name written in letters: any case, any alphabet, one word or
    several.

    Blank is allowed because the assembler records it as UNKNOWN_OPERATOR.
    Spaces, hyphens, apostrophes and periods belong to names; digits and other
    symbols do not. A combining mark (an accent typed on its own, a Devanagari
    vowel sign) belongs to the letter before it.
    """
    name = name.strip()
    return not name or (
        any(ch.isalpha() for ch in name)
        and all(
            ch.isalpha()
            or ch in _NAME_MARKS
            or unicodedata.category(ch).startswith("M")
            for ch in name
        )
    )


def _registry_source() -> str | None:
    """``bundled`` / ``file:<path>``, or None when the
    store recorded no source. What counts as "no source" is decided once, in
    ``task_registry_store.source_label``, so this and the recorder's banner
    cannot drift apart."""
    try:
        info = task_registry_store.get_registry_info()
    except Exception:
        return None
    return task_registry_store.source_label(info.get("source"))


def _registry_is_stale_copy() -> bool:
    """True when the registry came from an unexpected source. The rule itself
    lives in
    ``task_registry_store.is_stale_copy`` so the recorder's banner and this
    card can never disagree about which sources count as live."""
    return task_registry_store.is_stale_copy(_registry_source())


def _hw_version_for(preset, meta) -> str:
    """The hardware revision this draft would record under.

    Without a preset there is no rig to prefer, so the registry's value is all
    there is to say.
    """
    registry_value = meta.hw_version if meta else ""
    if preset is None:
        return registry_value
    return resolve_hw_version(preset, registry_value)


def _task_view(request: Request, draft, preset) -> dict:
    """Everything the Session card says about the task: which of the four
    states this name is in, and what it will be recorded as."""
    name = draft.task_name
    tasks = _task_list()
    state = _task_state(name, tasks)
    meta = task_metadata(name) if state == TASK_KNOWN else None

    message = _TASK_MESSAGE.get(state, "")
    if state == TASK_KNOWN:
        description = (meta.description if meta else "") or name
        message = f"Recorded as “{description}”."
        if _registry_is_stale_copy():
            message += (
                " The task list came from an unexpected source — the task"
                " details shown may be out of date."
            )

    return {
        "state": state,
        "message": message,
        "description": meta.description if meta else "",
        # What the operator typed for a new task, which Apply saves with it.
        "new_description": draft.task_description,
        "hw_version": _hw_version_for(preset, meta),
    }


def _task_fragment(request: Request, draft):
    """The task's two derived lines alone, for an htmx swap.

    One template, one `_task_view`, so the fragment and the page it was cut out
    of can never disagree -- and neither can the two lines inside it.
    """
    return request.app.state.templates.TemplateResponse(
        request,
        "_partials/task_status.html",
        {
            "task": _task_view(request, draft, _preset_for(draft)),
            # Refresh the new-task description field without carrying whole-form
            # refusals.
            "can_edit": is_owner(request.app.state.rig, tokens.token_from(request)),
            "errors": {},
            "error_ids": {},
        },
    )


@router.post("/config/task", response_class=HTMLResponse)
async def task_status(request: Request):
    """Re-read what the typed task name means, without saving anything.

    The line under the task name (its duration and description) is derived from
    the name, so it has to follow the name as it is typed rather than wait for
    Apply. Otherwise the only way to find out whether a name was even recognised
    would be to press Apply.

    Posted with the whole form, exactly like Add-a-camera, and absorbed the same
    way: a refresh fired while the device boxes are half filled in must carry
    them, not discard them. Only the line comes back, so nothing else on the
    page is re-rendered and no `<input>` the operator is typing into is replaced
    under their cursor.

    Always 200 with the line rendered. htmx does not swap a non-2xx, and this
    is the one control whose answer sits where the operator is looking: a
    refusal here would either blank the line or freeze it with no way back short
    of a reload. A caller who may not edit simply gets the draft as it stands.
    """
    if not request.headers.get("HX-Request"):
        # Return fragments only to HTMX requests; a normal navigation needs the full
        # page.
        return RedirectResponse("/config", status_code=SEE_OTHER)
    rig = request.app.state.rig
    draft = rig.draft
    if is_owner(rig, tokens.token_from(request)) and draft.preset_id:
        await _absorbed(request, draft)
    return _task_fragment(request, draft)


def _warned_devices(warnings) -> list[str]:
    """The camera each resolution warning is ABOUT, and nothing else.

    `resolution_for` writes "oak-FFC-ZZ: not found in the device scan —
    resolution unverified, using [640, 400]" (and, for an unknown module,
    "oak-FFC-ZZ CAM_A: ..."). The device name is the operator's own word -- they
    typed it -- so it stays; the socket and the pixel tuple are the half that
    only means something to whoever wrote the template.
    """
    names = []
    for warning in warnings:
        head = str(warning).split(":", 1)[0].split()
        if head and head[0] not in names:
            names.append(head[0])
    return names


def _page(
    request: Request,
    token,
    *,
    status_code: int = 200,
    errors: dict | None = None,
    warnings=(),
    banner: str = "",
    notice: str = "",
    confirm: dict | None = None,
    confirm_remove: str = "",
) -> HTMLResponse:
    return render_page(
        request,
        "config",
        _page_context(
            request,
            token,
            status_code=status_code,
            errors=errors,
            warnings=warnings,
            banner=banner,
            notice=notice,
            confirm=confirm,
            confirm_remove=confirm_remove,
        ),
        token=token,
        status_code=status_code,
    )


def _page_context(
    request: Request,
    token,
    *,
    status_code: int = 200,
    errors: dict | None = None,
    warnings=(),
    banner: str = "",
    notice: str = "",
    confirm: dict | None = None,
    confirm_remove: str = "",
) -> dict:
    rig = request.app.state.rig
    draft = rig.draft
    errors = dict(errors or {})

    preset = None
    if draft.preset_id:
        try:
            preset = setup_presets.get_preset(draft.preset_id)
        except KeyError:
            preset = None

    folder, folder_error = "", ""
    if preset is not None:
        try:
            folder = session_state.data_dir(draft)
        except FieldError as exc:
            folder_error = exc.message
            errors.setdefault(exc.field, exc.message)

    requirements = _requirements(request, draft) if preset else TemplateRequirements()

    scan = _scan_context(request, token)
    views = _device_views(request, draft, requirements, errors)

    return {
        "page_title": "Configuration",
        "active": "/config",
        "can_edit": is_owner(rig, token),
        "preset": preset,
        "draft": draft,
        "arm_groups": _arm_groups(views, requirements),
        "device_lists": _device_lists(views),
        "sections": _sections(),
        "folder": folder,
        "folder_error": folder_error,
        "iphone_notice": _iphone_notice(draft),
        "task": _task_view(request, draft, preset),
        "task_options": _task_options(_task_list()),
        "errors": errors,
        "error_ids": {f: config_form.error_id(f) for f in errors},
        "has_warnings": bool(warnings),
        "warned_devices": _warned_devices(warnings),
        "banner": banner,
        # A session dated before today. Reported, never auto-corrected: see
        # `session.recording_day_is_stale`.
        "stale_day": session_state.recording_day_is_stale(draft),
        # Show save confirmation once on the Apply redirect.
        "notice": notice,
        "confirm": confirm,
        "form_marker": FORM_MARKER,
        "recording_checks": enforcement.catalog(),
        "recording_checks_marker": enforcement.FORM_MARKER,
        "vive": _vive_context(request, draft, requirements, errors),
        **_optional_context(draft, views, errors, preset, confirm_remove),
        **_reference_context(request, draft),
        **_diagram_context(request, draft, requirements, views),
        **scan,
    }


def _section_response(
    request: Request,
    token,
    section: str,
    *,
    status_code: int = 200,
    error: str = "",
    confirm_remove: str = "",
):
    """One optional section, re-rendered on its own for an htmx swap.

    Only the section changes, so only the section is sent: a full page POST
    would throw away the scroll position and land the operator back at the
    session fields, hunting for the row they had just created.

    Falls back to the whole page for anything this cannot draw -- a section with
    no partial, or a refusal that has no section to blame -- because a full page
    swapped into a section is a worse answer than a full page reload.
    """
    rig = request.app.state.rig
    draft = rig.draft
    requirements = _requirements(request, draft)
    views = _device_views(request, draft, requirements)
    context = {
        "can_edit": is_owner(rig, token),
        # Send newly referenced device lists out of band with the added row.
        "device_lists": _device_lists(views),
        "section_template": _section_template(section),
        **_optional_context(draft, views, {}, _preset_for(draft), confirm_remove),
        # Update reference-camera choices alongside optional head-camera changes.
        **_reference_context(request, draft),
    }
    context["section_error"] = error
    try:
        return request.app.state.templates.TemplateResponse(
            request,
            "_partials/optional_section_fragment.html",
            context,
            status_code=status_code,
        )
    except TemplateNotFound:
        return _page(
            request,
            token,
            status_code=status_code,
            banner=error,
            confirm_remove=confirm_remove,
        )


def _iphone_notice(draft) -> str:
    """Banner text warning about mixed iPhone generations, shown before Apply.

    Empty unless the two tracking cameras name different generations.
    """
    generations = session_state.iphone_generations(draft)
    if not generations:
        return ""
    distinct = sorted(set(generations.values()))
    if len(distinct) > 1:
        return (
            "Mixed iPhone generations: "
            + " + ".join(distinct)
            + " — both tracking cameras must be the same generation."
        )
    return ""


def _mixed_iphone_generations(draft) -> dict:
    """Left/right tracking cameras naming different iPhone generations.

    Blocks Apply.
    """
    generations = session_state.iphone_generations(draft)
    if len(set(generations.values())) <= 1:
        return {}
    return {
        config_form.camera_field(f"{side}_tracking_camera"): (
            f"iPhone {generation} — the other side is a different generation. "
            "Both tracking cameras must be the same iPhone generation."
        )
        for side, generation in generations.items()
    }


def _gripper_choices(preset) -> list[str]:
    """Grippers of this preset's generation, from the device registry.

    Rendered as a datalist rather than a select, because unregistered hardware
    has to be enterable by raw serial.
    """
    if preset is None:
        return []
    try:
        return list_umi_gripper_device_names_for_pipeline(
            pipeline=preset.gripper_generation
        )
    except (KeyError, ValueError):
        return []


async def _absorbed(request: Request, draft):
    """Read the posted form into the draft. Returns (form, requirements).

    The device registry travels with it, so a typed `05` becomes `oak-FFC-05`
    on the way in rather than reaching the config as `05`. Resolution is
    against the registry widened by the last scan, which is what lets an
    attached but unregistered device be selected by its raw id.
    """
    form = await request.form()
    if enforcement.FORM_MARKER in form:
        try:
            enforcement.from_form(form)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    requirements = _requirements(request, draft)
    preset = None
    if draft.preset_id:
        try:
            preset = setup_presets.get_preset(draft.preset_id)
        except KeyError:
            preset = None
    found, _ = device_state.outcome_for(device_state.current_job(request.app))
    config_form.absorb(
        draft,
        form,
        requirements,
        resolve=_resolver(preset, found),
        umi_camera_slots=_umi_camera_slots(requirements),
    )
    return form, requirements


def _owner_or_conflict(request: Request):
    """(token, response). The response is non-None when the caller may not act."""
    rig = request.app.state.rig
    token = tokens.token_from(request)
    try:
        require_owner(rig, token)
    except StaleOwner as exc:
        return token, _page(request, token, status_code=CONFLICT, banner=str(exc))
    # Enforce read-only recording state at every mutating endpoint. Config or
    # session changes during recording would make the page disagree with the
    # folder receiving episodes.
    try:
        recording.require_idle(request.app)
    except RecordingInProgress as exc:
        return token, _page(request, token, status_code=CONFLICT, banner=str(exc))
    if not rig.draft.preset_id:
        return token, _page(
            request,
            token,
            status_code=BAD_REQUEST,
            banner="Choose a setup first.",
        )
    return token, None


@router.get("/config", response_class=HTMLResponse)
async def config_page(request: Request) -> HTMLResponse:
    rig = request.app.state.rig
    token = tokens.token_from(request) or tokens.new_token()
    claim(rig, token)
    if rig.draft.preset_id:
        _reopen(request, rig.draft)
    # Store one-shot warnings on app.state so query strings cannot forge or replay them.
    warnings = getattr(request.app.state, "config_warnings", [])
    request.app.state.config_warnings = []
    # Consume the server-side save confirmation once after a successful Apply.
    notice = getattr(request.app.state, "config_notice", "")
    request.app.state.config_notice = ""
    response = _page(request, token, warnings=warnings, notice=notice)
    tokens.remember(response, token)
    return response


def _previous_data_dir(config_path) -> str:
    """The folder the armed config names, before this Apply rewrites it."""
    return seal_state.armed_data_dir(config_path)


def _seal_if_orphaned(previous: str, current: str) -> bool:
    """Seal the folder this Apply just stopped pointing at.

    An operator who edits the task name moves to a new folder while keeping the
    session uid, so nothing rolls and no roll-driven trigger fires -- the
    commonest way a session ends without anybody deciding to. `dir_change`
    rather than `config_drift` because the comparison is strictly broader.

    True only when this call closed a session: a mistyped task name closes one
    for good, and "Configuration saved." over that says nothing.
    """
    return seal_state.seal_if_left_behind(previous, current, reason="dir_change")


def _roll_if_drifted(request: Request, draft) -> str | None:
    """Move the draft onto a fresh session if its folder holds another config.

    Returns the new folder, or None when nothing was wrong -- which is the
    normal case and must stay a no-op, or every Apply between two episodes would
    scatter one session across a folder per press.

    Only the draft is moved here; Apply re-assembles from it. `session_roll`
    also writes the generated file, which is redundant on this path and load-
    bearing on the other one (`routes/record`, where there is nothing left to
    re-assemble from).
    """
    return session_roll.roll_generated_config(
        _generated_path(request, draft),
        str(
            data_check.output_path(
                request.app.state.settings.state_dir, draft.task_name
            )
        ),
        draft,
    )


@router.post("/config/apply")
async def apply(request: Request):
    """Compatibility endpoint for clients that explicitly save a configuration."""
    refused = await save_configuration(request)
    return (
        refused
        if refused is not None
        else RedirectResponse("/config", status_code=SEE_OTHER)
    )


async def save_for_action(request: Request):
    """Apply the complete browser form before launching downstream work.

    Short API requests retain their existing saved-config semantics. Every UI
    launch button submits the configuration form, including without JavaScript.
    Reuse the same validation, assembly and session-roll path as explicit saves.
    """
    if FORM_MARKER not in await request.form():
        return None
    token, refused = _owner_or_conflict(request)
    if refused is None:
        draft = request.app.state.rig.draft
        await _absorbed(request, draft)
        state = cal.calibration_state(request.app)
        fingerprint = session_state.device_fingerprint(draft)
        active = (
            (cal.STEP_VERIFY, cal.check_running(request.app)),
            (cal.STEP_VIVE, cal.vive_running(request.app)),
            (
                cal.STEP_GRIPPER,
                any(
                    cal.run_state(cal.job_registry(request.app), run) == cal.RUNNING
                    for run in state.runs(cal.STEP_GRIPPER).values()
                ),
            ),
        )
        changing_busy_rig = any(
            running and state.stamp_of(step) not in ("", fingerprint)
            for step, running in active
        )
        if changing_busy_rig:
            refused = _page(
                request,
                token,
                status_code=CONFLICT,
                banner="Wait for calibration to finish or stop the running check before changing the configured devices.",
            )
        else:
            refused = await save_configuration(request)
    if refused is not None and request.headers.get("HX-Request"):
        refused.headers.update(
            {
                "X-Station-Fragment": "1",
                "HX-Retarget": "#configuration",
                "HX-Reselect": "#configuration",
                "HX-Reswap": "outerHTML",
                "HX-Trigger-After-Swap": "configuration-error",
            }
        )
    return refused


async def save_configuration(request: Request):
    """Write both validated configs; return an error response or None."""
    token, refused = _owner_or_conflict(request)
    if refused is not None:
        return refused
    draft = request.app.state.rig.draft
    # Captured before assembly overwrites the file this reads.
    previous_folder = _previous_data_dir(_generated_path(request, draft))
    _, requirements = await _absorbed(request, draft)

    # An unreadable task registry blocks Apply because labels and check
    # overrides cannot be validated. Save new tasks only after every other
    # form validation succeeds.
    task_state = _task_state(draft.task_name, _task_list())
    if task_state == TASK_NEW:
        task_errors = _new_task_errors(draft)
        if task_errors:
            return _page(
                request,
                token,
                status_code=BAD_REQUEST,
                errors=task_errors,
                banner="New task incomplete.",
            )
    if task_state == TASK_UNREADABLE:
        return _page(
            request,
            token,
            status_code=BAD_REQUEST,
            errors={"task_name": _TASK_MESSAGE[TASK_UNREADABLE]},
            banner="Task list unavailable.",
        )

    if not _operator_name_ok(draft.operator_name):
        return _page(
            request,
            token,
            status_code=BAD_REQUEST,
            errors={
                "operator_name": f"Write the operator name in letters, or leave it blank to record it as {UNKNOWN_OPERATOR}."
            },
            banner="Operator name invalid.",
        )

    # Resolve selections before testing availability so ambiguous names get
    # a useful field error rather than a misleading scan failure.
    views = _device_views(request, draft, requirements)
    job = device_state.current_job(request.app)
    found, _ = device_state.outcome_for(job)
    for unresolved, banner in (
        (
            _ambiguous(views),
            "One device name matches more than one device.",
        ),
        (
            _mixed_iphone_generations(draft),
            "The tracking cameras are different iPhone generations.",
        ),
        (
            _pairing_errors(draft, requirements, views, found),
            "Selected devices belong to different grippers. Correct the highlighted selections.",
        ),
        (
            config_form.unavailable_devices(views, found),
            "One device is attached but not available. Free it and rescan.",
        ),
    ):
        if unresolved:
            return _page(
                request,
                token,
                status_code=BAD_REQUEST,
                errors=unresolved,
                banner=banner,
            )

    # Require discovered OAK module information before writing a resolution;
    # a guessed aspect ratio can silently crop recorded images.
    unscanned, banner = _unscanned_oak_fields(draft, views, requirements, found, job)
    if unscanned:
        return _page(
            request,
            token,
            status_code=BAD_REQUEST,
            errors=unscanned,
            banner=banner,
        )

    try:
        folder = session_state.data_dir(draft)
    except FieldError as exc:
        # Surface invalid date/UID path components as field errors.
        return _page(
            request,
            token,
            status_code=BAD_REQUEST,
            errors={exc.field: exc.message},
            banner="This session cannot be saved yet.",
        )

    # Save the task before assembly so its description and check rules resolve
    # through the same registry path as existing tasks.
    if task_state == TASK_NEW:
        try:
            task_registry_store.add_task(
                draft.task_name,
                draft.task_description.strip() or UNKNOWN_TASK_DESCRIPTION,
            )
        except (OSError, ValueError) as exc:
            return _page(
                request,
                token,
                status_code=BAD_REQUEST,
                errors={
                    "task_name": "The new task could not be saved to "
                    f"{task_registry_store.local_registry_path()}: {exc}"
                },
                banner="New task not saved.",
            )

    # Pass the completed discovery result to the assembler for native OAK
    # aspect ratios. Missing results produce explicit fallback warnings.
    assembler = _assembler(request, scanner=DeviceScanner.from_records(found))
    spec = SessionSpec(
        preset_id=draft.preset_id,
        operator_name=draft.operator_name,
        task_name=draft.task_name,
        session_uid=draft.session_uid,
        data_dir=folder,
    )
    devices = DeviceSpec(
        grippers=dict(draft.grippers),
        required_cameras=dict(draft.required_cameras),
        optional=tuple(draft.optional),
        reference_override=draft.reference_override,
        # Pass only explicit Vive overrides; blank settings retain template defaults.
        vive_endpoint=draft.vive_endpoint,
        vive_keys=dict(draft.vive_keys),
        relaxed_checks=tuple(draft.relaxed_checks),
    )
    try:
        assembler.assemble(spec, devices)
    except (ValueError, KeyError, OSError) as exc:
        message = str(exc).strip("'")
        return _page(
            request,
            token,
            status_code=BAD_REQUEST,
            errors=_errors_for(message, draft, requirements),
            banner=message,
        )

    # Generate recording and check configs together, after assembly succeeds,
    # so both use the same task settings and failed forms leave no partial pair.
    try:
        data_check.generate(
            request.app.state.settings.repo_root,
            state_dir=request.app.state.settings.state_dir,
            task_name=draft.task_name,
        )
    except data_check.BaselineMissing as exc:
        # Report missing checkout resources at Apply rather than deferring failure
        # to recording preflight.
        message = str(exc)
        return _page(
            request,
            token,
            status_code=BAD_REQUEST,
            errors=_errors_for(message, draft, requirements),
            banner=message,
        )

    # Check session conflicts only after recording/check configs and calibration
    # references exist. Roll to a new session when acquisition provenance differs
    # from a folder containing episodes.
    rolled = _roll_if_drifted(request, draft)
    if rolled is not None:
        # Reassemble both configs after rollover so the new UID and task bounds
        # reach each generated artifact consistently.
        try:
            # Derive the rolled folder through the same draft-to-path function used by
            # other pages.
            spec = replace(
                spec,
                session_uid=draft.session_uid,
                data_dir=session_state.data_dir(draft),
            )
            assembler.assemble(spec, devices)
            data_check.generate(
                request.app.state.settings.repo_root,
                state_dir=request.app.state.settings.state_dir,
                task_name=draft.task_name,
            )
        except (FieldError, ValueError, KeyError, OSError) as exc:
            # Assembly already passed for this draft; a rewrite failure is a persistence
            # error rather than new device-selection feedback.
            message = str(exc).strip("'")
            return _page(
                request,
                token,
                status_code=BAD_REQUEST,
                errors=_errors_for(message, draft, requirements),
                banner=f"Could not open a new session: {message}",
            )

    # Seal any session folder the console stopped naming, including task/path
    # changes that do not require a new UID.
    sealed = _seal_if_orphaned(previous_folder, session_state.data_dir(draft))

    # Preserve resolution-fallback warnings across redirects so field-of-view
    # cropping cannot go unnoticed.
    request.app.state.config_warnings = list(assembler.last_resolution_warnings)
    # Report saves, rollovers and seals separately. Sealing can occur without
    # a UID change and is irreversible through Apply; rollover need not create
    # a new seal. Name the affected folder so the operator can locate its data.
    if rolled is not None:
        notice = (
            f"The configuration changed, so this is now a new session "
            f"({draft.session_uid}). The episodes already recorded stay where "
            f"they are."
        )
    elif sealed:
        notice = (
            f"Configuration saved. The previous session "
            f"({os.path.basename(str(previous_folder).rstrip('/'))}) is closed "
            f"and will take no more episodes — the episodes already recorded "
            f"stay where they are."
        )
    else:
        notice = "Configuration saved."
    request.app.state.config_notice = notice
    # Invalidate verification and release its watcher when selected hardware
    # changes. Preserve results for unchanged hardware, including task-only edits.
    readiness.reset_verification_for_new_rig(request.app)
    saved = cal.recording_config(request.app)
    roles = cal.verification_roles(saved)
    if roles and set(roles) <= identification.verified_roles(request.app, saved):
        cal.calibration_state(request.app).stamp(
            cal.STEP_VERIFY, session_state.device_fingerprint(draft)
        )
    return None


def _errors_for(message: str, draft, requirements) -> dict:
    """Attach a rejection to every field it is about.

    Matched on the slot name and on the device the operator typed, because the
    two rejections that matter name different things: the completeness check
    names an empty slot (which holds no device to match on), while the sharing
    check names a device rather than a field -- "device 'X' assigned to both 'a'
    and 'b'" has to light up every row holding X.

    Slots come from the template rather than from the draft for exactly that
    reason: a blank field is absent from the draft, so a draft-only candidate
    list can never find the field the operator has to go and fill in.
    """
    candidates = []
    for arm in requirements.gripper_arms:
        field = config_form.gripper_field(arm)
        candidates += [(field, arm), (field, f"{arm}_umi_gripper")]
    for arm, device in draft.grippers.items():
        # Match duplicate-device errors to both numeric-source rows through the
        # shared device serial.
        candidates.append((config_form.gripper_field(arm), device))
    # Map every stream in a physical block to its shared device field so
    # stream-specific assembly errors attach to the correct control.
    for block in requirements.blocks:
        field = config_form.camera_field(block.key)
        candidates += [(field, name) for name in block.stream_names]
        candidates += [
            (field, draft.required_cameras[name])
            for name in block.stream_names
            if name in draft.required_cameras
        ]
    for assignment in draft.optional:
        field = config_form.optional_field(assignment.logical_name, "device")
        candidates += [(field, assignment.logical_name), (field, assignment.device)]
    # Map tracker errors by both source name and effective key, including
    # unchanged template defaults, so duplicate assignments mark both arms.
    for source in requirements.vive_sources:
        field = config_form.vive_key_field(source.arm)
        candidates += [(field, _vive_key_value(draft, source)), (field, source.name)]
    hit = config_form.fields_named_in(message, candidates)
    return {field: message for field in hit}


# ------------------------------------------------------------- device scanning


def _scan_fragment(request: Request, token, *, status_code: int = 200, error: str = ""):
    """The scan strip, carrying the refreshed device options with it.

    See `_partials/config_scan_fragment.html`: the device lists are swapped
    out-of-band with the strip, so the boxes update without discarding what
    was typed in them.
    """
    draft = request.app.state.rig.draft
    context = _scan_context(request, token)
    context["scan_error"] = error
    requirements = _requirements(request, draft)
    views = _device_views(request, draft, requirements)
    context["device_lists"] = _device_lists(views)
    context.update(_diagram_context(request, draft, requirements, views))
    return request.app.state.templates.TemplateResponse(
        request,
        "_partials/config_scan_fragment.html",
        context,
        status_code=status_code,
    )


def _scan_response(request: Request, token, *, status_code=200, error=""):
    """A fragment for htmx, a redirect or a full page for a plain browser.

    The Refresh control is a real submit button in the page's one form, so an
    operator whose vendored htmx never loaded still rescans -- they get the
    whole page back instead of just the card.
    """
    if request.headers.get("HX-Request"):
        return _scan_fragment(request, token, status_code=status_code, error=error)
    if status_code == 200:
        return RedirectResponse("/config", status_code=SEE_OTHER)
    return _page(request, token, status_code=status_code, banner=error)


@router.post("/config/devices/scan")
@serialize_start
async def scan_devices(request: Request):
    """Enumerate the station's hardware, without ever blocking this request.

    The scan opens devices, an OAK exclusively, and no hardware open in this
    repo has a timeout of its own. So it runs as a capped child through the job
    registry and this handler returns as soon as the pid exists. Every scan on
    the console shares that one job, so two requests never hold two exclusive
    connections to one camera.
    """
    rig = request.app.state.rig
    token = tokens.token_from(request)
    try:
        require_owner(rig, token)
    except StaleOwner as exc:
        return _scan_response(request, token, status_code=CONFLICT, error=str(exc))
    if identification.running(request.app):
        return _scan_response(
            request, token, status_code=CONFLICT, error=identification.MESSAGE
        )
    # Do not probe devices while recording owns them exclusively.
    try:
        recording.require_idle(request.app)
    except RecordingInProgress as exc:
        return _scan_response(request, token, status_code=CONFLICT, error=str(exc))

    # Cancel camera verification and wait off the event loop for its workers
    # to release exclusive camera ownership before discovery.
    if cal.check_running(request.app):
        await run_in_threadpool(
            cal.stop_check, request.app, wait=device_state.CHECK_RELEASE_WAIT_S
        )
    # A live preview holds one camera the same way; the scan needs it back.
    if preview.running(request.app):
        await run_in_threadpool(preview.stop, request.app, wait=preview.RELEASE_WAIT_S)

    # Ignore absent form fields on body-less auto-scan requests.
    form = await request.form()
    if FORM_MARKER in form:
        await _absorbed(request, rig.draft)

    _start_scan(request)
    return _scan_response(request, token)


@router.get("/config/devices/status", response_class=HTMLResponse)
async def scan_status(request: Request) -> HTMLResponse:
    """The scan card alone, for the poller. Read-only, so any tab may watch."""
    return _scan_fragment(request, tokens.token_from(request))


@router.post("/config/devices/adopt")
async def adopt_scan(request: Request):
    """Absorb the posted form and re-render the page.

    No control on the page posts here: the scan fragment swaps the device lists
    out-of-band, so there is nothing to adopt. The route stays because
    absorb-then-render is a useful thing to be able to ask for, and several
    tests drive the form through it. It is a plain owner-gated absorb.
    """
    token, refused = _owner_or_conflict(request)
    if refused is not None:
        return refused
    await _absorbed(request, request.app.state.rig.draft)
    return RedirectResponse("/config", status_code=SEE_OTHER)


@router.post("/config/devices/link/{arm}", response_class=HTMLResponse)
async def link_devices(request: Request, arm: str):
    """Re-render one arm's card, after its gripper changed.

    `absorb` already fills a bound gripper's camera. This exists only so the
    operator sees it at the moment they pick, rather than one full render later
    -- Apply's redirect, adding a camera, adopting a scan. The answer is the one
    card, so the scroll position and every other half-typed field on the page
    survive, which is the same reason `/config/optional/add` returns its section
    alone.

    The arm is in the path rather than read off the body. The box posts the
    whole form (`hx-include="closest form"`), so every arm's gripper is in every
    one of these and nothing in the body says which of them changed -- taking
    the first would answer a right-arm pick with the left card, which htmx would
    then swap into `#arm-right`.

    Degrades to nothing: with the vendored script blocked the box carries no
    behaviour at all and the fill still lands on Apply.
    """
    if not request.headers.get("HX-Request"):
        # Return card fragments only for HTMX requests; use a full page otherwise.
        return RedirectResponse("/config", status_code=SEE_OTHER)
    token, refused = _owner_or_conflict(request)
    if refused is not None:
        return refused
    draft = request.app.state.rig.draft
    _, requirements = await _absorbed(request, draft)

    views = _device_views(request, draft, requirements)
    group = next((g for g in _arm_groups(views, requirements) if g["arm"] == arm), None)
    if group is None:
        # Return 204 when the template has no matching arm. A 200 response would
        # swap a whole document into the card.
        return Response(status_code=NO_CONTENT)
    return request.app.state.templates.TemplateResponse(
        request,
        # Refresh hardware diagrams with a changed gripper/camera link.
        "_partials/arm_group_fragment.html",
        {
            "group": group,
            "can_edit": is_owner(request.app.state.rig, token),
            **_diagram_context(request, draft, requirements, views),
        },
    )


@router.post("/config/vive/probe")
async def probe_vive(request: Request):
    """Ask the bridge what it is serving, and re-render just this section.

    This is the only thing on the console that probes. No timer, nothing on
    page load: the bridge serves from a single `zmq.REP` thread shared by every
    station, so a page that asked on its own would make every open browser tab
    a permanent client of it.

    The whole form is absorbed first, so the address tested is the one
    half-typed in the box rather than the one last saved, and the operator's
    other fields survive the press -- the same contract every other button on
    this form has.
    """
    hx = bool(request.headers.get("HX-Request"))
    token, refused = _owner_or_conflict(request)
    if refused is not None:
        return refused
    draft = request.app.state.rig.draft
    _, requirements = await _absorbed(request, draft)

    if requirements.vive_sources:
        endpoint = draft.vive_endpoint or requirements.vive_sources[0].endpoint
        # Probe the bridge off the event loop. Store the result with its address
        # so editing the address invalidates stale tracker information.
        request.app.state.last_vive_probe = (
            endpoint,
            await run_in_threadpool(vive_probe.probe, endpoint),
        )

    if not hx:
        # For non-HTMX posts, render the full page using the result saved on app.state.
        return RedirectResponse("/config", status_code=SEE_OTHER)

    vive = _vive_context(request, draft, requirements)
    if vive is None:
        # Return 204 for rigs without trackers; an empty outerHTML replacement
        # would delete the section.
        return Response(status_code=NO_CONTENT)
    return request.app.state.templates.TemplateResponse(
        request,
        "_partials/vive_section.html",
        {"vive": vive, "can_edit": is_owner(request.app.state.rig, token)},
    )


@router.post("/config/session/uid")
async def regenerate(request: Request):
    """Start a new session: a new uid, stamped today, on the same rig.

    The uid is the last segment of `output/raw/{task}/{session}/`, so minting it
    after episodes exist forks the session across two folders -- hence the
    confirmation, which is a second POST rather than a JavaScript `confirm()`
    so it still guards the operator when the vendored script fails to load.

    What carries over and what does not is decided in
    `session.start_new_session`; the rest -- unarming the file the recorder
    reads, sealing the folder that is left, forgetting the finished recording
    on /record -- is `next_session.start_next_session`. That sequence is
    order-sensitive enough that it lives in one place.
    """
    token, refused = _owner_or_conflict(request)
    if refused is not None:
        return refused
    draft = request.app.state.rig.draft
    form, _ = await _absorbed(request, draft)

    if not form.get("confirm") and session_state.has_episodes(draft):
        return _page(
            request,
            token,
            confirm={
                "action": "/config/session/uid",
                "label": "Start a new session",
                "message": (
                    "This session already holds episodes. They stay where they are."
                ),
            },
        )

    next_session.start_next_session(request.app, draft, reason="next_session")
    return RedirectResponse("/config", status_code=SEE_OTHER)


@router.post("/config/optional/add")
async def add_optional(request: Request):
    """Add one optional camera, swapping only the section it lands in.

    The button is a real submit in the page's one form, aimed with `formaction`,
    so with the vendored script blocked this is a plain POST and a 303 back to
    the page. htmx merely intercepts it, and then the answer is the section
    alone -- which is what keeps the operator's scroll position where they left
    it instead of at the top of a re-rendered page.
    """
    hx = bool(request.headers.get("HX-Request"))
    token, refused = _owner_or_conflict(request)
    if refused is not None:
        return refused
    draft = request.app.state.rig.draft
    form, _ = await _absorbed(request, draft)

    type_id = str(form.get("type_id") or "")
    try:
        descriptor = optional_devices.get_optional_type(type_id)
    except KeyError:
        # No descriptor means no section to swap, so this one is always the
        # whole page -- there is nowhere else to put the message.
        return _page(
            request,
            token,
            status_code=BAD_REQUEST,
            banner=stale_choice("device type"),
        )
    if not optional_devices.can_add(type_id, draft.optional):
        # Enforce descriptor limits server-side; disabled controls alone are not a
        # guard.
        message = (
            f"{descriptor.label} is already configured; remove it before "
            "adding another."
        )
        if hx:
            return _section_response(
                request,
                token,
                descriptor.section,
                status_code=BAD_REQUEST,
                error=message,
            )
        return _page(request, token, status_code=BAD_REQUEST, banner=message)

    draft.optional.append(
        optional_devices.OptionalDeviceAssignment(
            type_id=type_id,
            logical_name=optional_devices.allocate_logical_name(type_id),
        )
    )
    if hx:
        return _section_response(request, token, descriptor.section)
    return RedirectResponse("/config", status_code=SEE_OTHER)


@router.post("/config/optional/remove")
async def remove_optional(request: Request):
    """Drop one optional camera, swapping only the section it sat in.

    The same shape as Add -- a real `formaction` submit that htmx merely
    intercepts -- so removing a camera keeps the operator's scroll position, and
    the answer is the section alone.

    The confirmation cannot be a `<form>` of its own: this page is one form
    already, and HTML forbids nesting. So it is a plain submit button in the
    page's own form, aimed at this endpoint with `?confirm=yes`. The query
    string carries the agreement because the button's one name/value pair is
    spent naming the row, and a hidden `confirm` input would sit in the page's
    form and be posted by Apply too.
    """
    hx = bool(request.headers.get("HX-Request"))
    token, refused = _owner_or_conflict(request)
    if refused is not None:
        return refused
    draft = request.app.state.rig.draft
    form, _ = await _absorbed(request, draft)

    logical_name = str(form.get("logical_name") or "")
    remaining = [a for a in draft.optional if a.logical_name != logical_name]
    if len(remaining) == len(draft.optional):
        # No row means no section to swap, so this one is always the whole page.
        return _page(
            request,
            token,
            status_code=BAD_REQUEST,
            banner=stale_choice("camera"),
        )

    doomed = next(a for a in draft.optional if a.logical_name == logical_name)
    section = _optional_kind_section(doomed.type_id)
    confirmed = bool(form.get("confirm") or request.query_params.get("confirm"))
    if doomed.device and not confirmed:
        # Confirm removal server-side so it also works without JavaScript.
        # Unassigned rows can be removed without confirmation.
        if hx and section:
            return _section_response(
                request, token, section, confirm_remove=logical_name
            )
        return _page(request, token, confirm_remove=logical_name)
    # Reuse the removed camera's index without renaming other rows.
    draft.optional[:] = remaining
    if hx and section:
        return _section_response(request, token, section)
    return RedirectResponse("/config", status_code=SEE_OTHER)


def _optional_kind_section(type_id: str) -> str:
    """Which section a row belongs to, or "" for a descriptor nothing declares.

    Empty rather than raising: a draft carrying a type this build no longer
    knows about must still be removable, and falling back to the whole page is
    a better answer than a 500.
    """
    try:
        return optional_devices.get_optional_type(type_id).section
    except KeyError:
        return ""

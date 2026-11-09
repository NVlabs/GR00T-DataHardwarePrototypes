# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The three translations the configuration page needs, kept out of the route.

A posted form, the in-memory draft, and a generated `config.yaml` are three
spellings of the same rig, and each pair has a trap in it:

* **form -> draft.** Absorbing whatever keys arrive would let a hand-posted
  `camera.nose_camera` into the draft. Every value is taken slot by slot from
  what the *template* declares instead, so the form can only fill fields that
  exist. The recording date and the session uid are deliberately not among
  them: they are derived, not typed, so a post cannot be the source of a date
  `build_data_dir` refuses.
* **config -> draft.** This is what "reopen" means, and it is why `session_uid`
  survives a server restart instead of being re-minted into a second session
  folder.
* **ValueError -> field.** The assembler's messages name devices and cameras.
  Matching them back to the input that caused them is what turns a 500 into an
  error the operator can act on next to the offending row.

Nothing here touches HTTP, so all three are testable without a request.
"""

from __future__ import annotations

import re

from station_ui import enforcement
from station_ui.core.config_assembler import ConfigAssembler, source_device
from station_ui.core.optional_devices import (
    OptionalDeviceAssignment,
    get_optional_type,
    type_for_logical_name,
)
from station_ui.core.session_identity import is_valid_session_uid
from station_ui.devices import OK as RESOLVED
from station_ui.devices import unavailable_reason

# Use the shared tracking-source arm resolver when reading templates and
# restoring saved configs so both interpret source names identically.
from umi.common.utility.tracking import tracking_device_side
from umi.utils import device_registry

__all__ = [
    "CAMERA_PREFIX",
    "FORM_MARKER",
    "GRIPPER_PREFIX",
    "OPTIONAL_PREFIX",
    "REFERENCE_FIELD",
    "VIVE_DEFAULT_PORT",
    "VIVE_ENDPOINT_FIELD",
    "VIVE_PREFIX",
    "absorb",
    "camera_field",
    "display_vive_endpoint",
    "dom_id",
    "error_id",
    "fields_named_in",
    "gripper_field",
    "hydrate",
    "normalize_vive_endpoint",
    "optional_field",
    "unavailable_devices",
    "vive_key_field",
]

CAMERA_PREFIX = "camera."
GRIPPER_PREFIX = "gripper."
OPTIONAL_PREFIX = "optional."
VIVE_PREFIX = "vive."
# All trackers on an endpoint share a client and socket; configure one
# bridge address per rig.
VIVE_ENDPOINT_FIELD = f"{VIVE_PREFIX}endpoint"
# Default bridge port for forms that accept a bare host address.
VIVE_DEFAULT_PORT = 5555
# Only absorb the full config form. A short action request omits fields
# and must not clear their saved values.
FORM_MARKER = "config_form"
# Use a native form select so reference-camera selection works without JavaScript.
REFERENCE_FIELD = "reference_camera"

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def camera_field(name: str) -> str:
    return f"{CAMERA_PREFIX}{name}"


def gripper_field(arm: str) -> str:
    return f"{GRIPPER_PREFIX}{arm}"


def vive_key_field(arm: str) -> str:
    return f"{VIVE_PREFIX}key.{arm}"


_VIVE_ENDPOINT_SUFFIX = f":{VIVE_DEFAULT_PORT}"
_VIVE_ENDPOINT_PREFIX = "tcp://"


def normalize_vive_endpoint(text: str) -> str:
    """What the operator typed, turned into the address ZMQ actually dials.

    The box only ever asks for the bridge's IP: `tcp://` and `:5555` are
    implied, never typed. A value that already names a scheme is left exactly
    as posted, so a config that stores a full address still round-trips
    instead of getting a second scheme glued on.
    """
    text = text.strip()
    if not text or "://" in text:
        return text
    return f"{_VIVE_ENDPOINT_PREFIX}{text}{_VIVE_ENDPOINT_SUFFIX}"


def display_vive_endpoint(endpoint: str) -> str:
    """The bare host the box shows -- the inverse of `normalize_vive_endpoint`,
    so a saved `tcp://host:5555` reads back as what the operator typed rather
    than growing a visible prefix on reload."""
    text = (endpoint or "").strip()
    if text.startswith(_VIVE_ENDPOINT_PREFIX) and text.endswith(_VIVE_ENDPOINT_SUFFIX):
        return text[len(_VIVE_ENDPOINT_PREFIX) : -len(_VIVE_ENDPOINT_SUFFIX)]
    return text


def optional_field(logical_name: str, attribute: str) -> str:
    return f"{OPTIONAL_PREFIX}{logical_name}.{attribute}"


def dom_id(prefix: str, field: str) -> str:
    """A DOM id derived from a field name.

    Field names carry dots (`camera.left_umi_camera`), which are legal in an id
    but awkward everywhere that reads one -- a CSS or querySelector lookup has
    to escape them -- so they are flattened here once, in one place, for every
    id the page builds out of a field name.
    """
    return f"{prefix}-" + re.sub(r"[^A-Za-z0-9_-]+", "-", field)


def error_id(field: str) -> str:
    """A DOM id for a field's error message."""
    return dom_id("error", field)


def _text(form, key: str, fallback: str = "") -> str:
    value = form.get(key)
    return fallback if value is None else str(value).strip()


def _canonical(typed: str, kind: str, resolve) -> str:
    """The canonical device name when *typed* names exactly one device.

    Anything else is kept exactly as typed. `resolve` never fabricates a name,
    and neither does this: an ambiguous or unmatched string stays on screen as
    the operator wrote it, so the field they have to fix is the field they can
    see, rather than a silently different device landing in the config.
    """
    if resolve is None or not typed:
        return typed
    name, status = resolve(kind, typed)
    return name if status == RESOLVED else typed


def _slot(form, field: str, kind: str, resolve):
    """`(present, value)` for one device slot after this submit.

    One control per physical device: an `<input list=...>` over a `<datalist>`
    of everything the scan found, which is "pick a connected device or type it
    by hand" in a single box. The typed text is resolved against the registry
    on the way in. The box has to stay writable because configs get prepped
    before the hardware is attached, and because the device registry may not
    list every device yet.
    """
    if field not in form:
        return False, ""
    return True, _canonical(_text(form, field), kind, resolve)


def absorb(draft, form, requirements, resolve=None, umi_camera_slots=None) -> None:
    """Copy a posted form into *draft*, one declared slot at a time.

    Called by every mutating route, not just Apply: adding a head camera posts
    the whole form, so the operator's half-typed session fields have to survive
    it. A route that only absorbed on Apply would reset the page every time a
    camera was added.

    A key that is absent leaves the draft alone; a key that is present and blank
    clears its slot. The distinction matters because the confirm-regenerate step
    posts a short form, and treating its missing fields as "cleared" would wipe
    the rig the operator just finished configuring.

    `resolve(kind, typed) -> (name | None, status)` is the device registry. It
    is optional only so the translation stays testable without one; every route
    passes it, because a raw string reaching the config is exactly what it
    prevents.

    `umi_camera_slots` is `arm -> block key` for the camera each arm's gripper
    carries (`routes.config._umi_camera_slots`). Omitting it links nothing.
    """
    if enforcement.FORM_MARKER in form:
        draft.relaxed_checks = enforcement.from_form(form)
    if "operator_name" in form:
        draft.operator_name = _text(form, "operator_name")
    if "task_name" in form:
        task_name = _text(form, "task_name")
        # Clear an unposted description when the task name changes so descriptions
        # cannot be assigned to the wrong task.
        if task_name != draft.task_name and "task_description" not in form:
            draft.task_description = ""
        draft.task_name = task_name
    if "task_description" in form:
        draft.task_description = _text(form, "task_description")
    # Blank selects Automatic; an absent field preserves the current override.
    # Validate the camera name in the assembler rather than resolving it as a device ID.
    if REFERENCE_FIELD in form:
        draft.reference_override = _text(form, REFERENCE_FIELD)

    # Derive date and session_uid internally; do not accept them from form input.

    for arm in requirements.gripper_arms:
        present, device = _slot(form, gripper_field(arm), "gripper", resolve)
        if not present:
            continue
        if device:
            draft.grippers[arm] = device
        else:
            draft.grippers.pop(arm, None)

    # Accept Vive fields only for templates with trackers. Tracker keys belong
    # to the bridge and are not UMI device-registry names.
    if requirements.vive_sources:
        if VIVE_ENDPOINT_FIELD in form:
            draft.vive_endpoint = normalize_vive_endpoint(
                _text(form, VIVE_ENDPOINT_FIELD)
            )
        for source in requirements.vive_sources:
            field_name = vive_key_field(source.arm)
            if field_name not in form:
                continue
            key = _text(form, field_name)
            if key:
                draft.vive_keys[source.arm] = key
            else:
                draft.vive_keys.pop(source.arm, None)

    declared = [camera.name for camera in requirements.cameras]
    # Select a physical OAK board once and apply its identity to every stream.
    for block in requirements.blocks:
        present, device = _slot(
            form, camera_field(block.key), block.device_kind, resolve
        )
        if not present:
            continue
        for name in block.stream_names:
            if device:
                draft.required_cameras[name] = device
            else:
                draft.required_cameras.pop(name, None)

    # Fill only empty camera slots from the selected gripper's linked camera.
    # Preserve explicit assignments; this makes repeated full-form posts idempotent.
    for arm, block_key in (umi_camera_slots or {}).items():
        block = next((b for b in requirements.blocks if b.key == block_key), None)
        if block is None:
            continue
        if any(draft.required_cameras.get(name) for name in block.stream_names):
            continue
        camera = device_registry.camera_for_gripper(draft.grippers.get(arm, ""))
        if not camera:
            continue
        for name in block.stream_names:
            draft.required_cameras[name] = camera

    # Remove assignments to slots the current template does not declare.
    for arm in list(draft.grippers):
        if arm not in requirements.gripper_arms:
            del draft.grippers[arm]
    for name in list(draft.required_cameras):
        if name not in declared:
            del draft.required_cameras[name]
    # Prune Vive keys only when tracker requirements are available. Unreadable
    # templates preserve those keys, and switching to a non-Vive preset does not
    # erase tracker assignments needed if the operator switches back.
    if requirements.vive_sources:
        declared_arms = {source.arm for source in requirements.vive_sources}
        for arm in list(draft.vive_keys):
            if arm not in declared_arms:
                del draft.vive_keys[arm]

    draft.optional = [
        OptionalDeviceAssignment(
            type_id=assignment.type_id,
            logical_name=assignment.logical_name,
            device=_optional_device(form, assignment, resolve),
            placement=_present(
                form,
                optional_field(assignment.logical_name, "placement"),
                assignment.placement,
            ),
        )
        for assignment in draft.optional
    ]


def _optional_device(form, assignment, resolve) -> str:
    """One optional camera's device, keeping its current value when unposted.

    An unknown `type_id` degrades to "no kind" rather than raising: a stale
    draft is a listing problem the operator can fix by removing the row, and a
    500 on the way to telling them is not an improvement.
    """
    try:
        kind = get_optional_type(assignment.type_id).device_kind
    except KeyError:
        kind = ""
    present, device = _slot(
        form, optional_field(assignment.logical_name, "device"), kind, resolve
    )
    return device if present else assignment.device


def _present(form, key: str, fallback: str) -> str:
    return _text(form, key) if key in form else fallback


def hydrate(draft, cfg) -> None:
    """Load a previously generated config back into *draft* -- "reopen".

    Only fields the assembler wrote are read back, and each one is validated
    rather than trusted: a hand-edited `session_uid` that is not 12 hex
    characters is ignored in favour of the freshly minted one, because
    `build_data_dir` raises on it and the page would be stuck on a 400 with no
    way back.
    """
    draft.relaxed_checks = enforcement.restored(cfg)
    meta = cfg.get("meta") or {}
    recording = cfg.get("recording") or {}

    draft.operator_name = str(meta.get("operator_name") or "")
    draft.task_name = str(meta.get("task_name") or "")
    draft.task_description = str(meta.get("task_description") or "")
    session_uid = meta.get("session_uid")
    if is_valid_session_uid(session_uid):
        draft.session_uid = session_uid

    date = _date_from(recording.get("data_dir"))
    if date:
        draft.date = date
        if _named_with_a_shift(recording.get("data_dir")):
            draft.legacy_folder = str(recording["data_dir"])

    draft.grippers = {}
    for entry in cfg.get("numeric_sources") or []:
        name = str(entry.get("name") or "")
        device = str(entry.get("uart_device_name") or "").strip()
        if (
            entry.get("type") == "umi_gripper"
            and name.endswith("_umi_gripper")
            and device
        ):
            draft.grippers[name[: -len("_umi_gripper")]] = device

    # The form stores one Vive endpoint per rig; a template with multiple
    # endpoints would restore only the last. Tolerate malformed source entries
    # when reading persisted configs.
    draft.vive_endpoint = ""
    draft.vive_keys = {}
    for entry in cfg.get("numeric_sources") or []:
        if entry.get("type") != "vive_tracker":
            continue
        arm = tracking_device_side(str(entry.get("name") or ""))
        if arm is None:
            continue
        # Validate params as a mapping before reading fields. A malformed source
        # must not abort restoration and silently create a new session UID.
        params = entry.get("params")
        if not isinstance(params, dict):
            params = {}
        endpoint = str(params.get("endpoint") or "").strip()
        if endpoint:
            draft.vive_endpoint = endpoint
        key = str(params.get("tracker_key") or "").strip()
        if key:
            draft.vive_keys[arm] = key

    draft.required_cameras = {}
    optional = []
    for source in cfg.get("video_sources") or []:
        device = source_device(source)
        names = [
            stream.get("name")
            for stream in source.get("streams") or []
            if stream.get("name")
        ]
        if not names and source.get("name"):
            names = [source["name"]]
        for name in names:
            descriptor = type_for_logical_name(name)
            if descriptor is not None:
                optional.append(
                    OptionalDeviceAssignment(
                        type_id=descriptor.id,
                        logical_name=name,
                        device=device,
                        placement=str(source.get("placement") or ""),
                    )
                )
            elif device:
                draft.required_cameras[name] = device
    draft.optional = optional

    # Restore an explicit reference override only when it differs from the
    # automatic choice. Automatic sessions must adapt when cameras are added.
    reference = str(cfg.get("reference_camera") or "")
    if reference:
        auto = ConfigAssembler._auto_reference(cfg)
        if auto and reference != auto:
            draft.reference_override = reference


def _date_from(data_dir) -> str:
    """The `YYYY-MM-DD` prefix of a session folder.

    Positional on purpose: the audit finds a day's sessions by this prefix, so
    if the date is not the first ten characters the folder is already wrong and
    there is nothing here to recover.
    """
    basename = str(data_dir or "").rstrip("/").split("/")[-1]
    head = basename[:10]
    return head if _DATE_RE.match(head) else ""


def _named_with_a_shift(data_dir) -> bool:
    """Whether an older version put a shift label (AM, PM or OOS) after the folder's date.

    The current version names the same session without it, so the restored
    draft keeps the recorded folder (`SessionDraft.legacy_folder`) rather than
    silently renaming a half-recorded one, and the day roll moves on to a new
    session.
    """
    basename = str(data_dir or "").rstrip("/").split("/")[-1]
    return basename[11:].split("_", 1)[0] in ("AM", "PM", "OOS")


def unavailable_devices(views, found) -> dict:
    """`field -> message` for every row naming a device the scan refused.

    The gate that gives the picker's `unavailable` label teeth. Marking an
    option in the markup does not make it unselectable -- a browser submits a
    labelled `<option>` like any other, and a hand-crafted POST never renders
    the label at all -- so this runs on the posted VALUE, next to `_ambiguous`,
    which is the same shape of refusal for the same reason.

    Narrow on purpose. Only a device the scan positively refused is blocked;
    see `devices.unavailable_reason` for why a device the scan did not find
    must still go through, and why blocking it would make the page unusable.

    A device named by two rows lights up both: the operator has to fix the
    device, and a message on one of the two rows holding it points at half the
    problem.
    """
    out = {}
    for view in views:
        field = view.get("field")
        value = str(view.get("value") or "")
        # Match IDs within the device kind; short UVC indices can collide with
        # identifiers of unrelated cameras.
        reason = unavailable_reason(found, value, view.get("kind"))
        if not field or reason is None:
            continue
        out[field] = (
            f"'{value}' is attached to this station but is not available — "
            f"{reason}. Free it and rescan, or choose another device."
        )
    return out


def fields_named_in(message: str, candidates) -> list[str]:
    """Which of *candidates* a rejection message is about.

    The assembler raises about devices and camera names -- "device 'iphone-R'
    assigned to both 'left_umi_camera' and 'right_tracking_camera'" -- and a
    message shown at the top of a long form makes the operator hunt for the row
    it means. Matching on the value the operator typed as well as on the slot
    name is what finds the field when only the device is named.
    """
    hit = []
    for field, token in candidates:
        token = str(token or "").strip()
        if token and re.search(rf"(?<![\w.-]){re.escape(token)}(?![\w-])", message):
            hit.append(field)
    return hit

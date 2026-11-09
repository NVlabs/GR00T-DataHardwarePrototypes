# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble an active recording config from a preset plus device choices.

This merges into a round-trip load of the immutable template rather than
generating a file from scratch: ``pre_episode_checks`` and
``reference_camera`` live only in the templates, and a from-scratch writer
drops them silently. The template is only ever read; the merged result is
written atomically under the station state dir -- ``~/.umi_station/
data_recording/`` by default, never the checkout. See
``station_ui.settings.STATE_DIR_ENV`` for why it lives outside the repo.

Every assignment that lands nowhere raises. A gripper arm or camera name the
template does not declare cannot be applied, and applying it silently would
leave the operator recording from the template's placeholder hardware while
``meta.hardware`` claims otherwise.
"""

from __future__ import annotations

import copy
import functools
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

# Resolve the default state directory when constructing the assembler,
# after environment configuration is available.
from station_ui import settings
from station_ui.core.device_scan import (
    HEAD_STREAM_RESOLUTION,
    resolution_for,
    sensor_module_for,
)
from station_ui.core.optional_devices import (
    OPTIONAL_DEVICE_TYPES,
    OptionalDeviceAssignment,
    get_optional_type,
    matches_logical_name,
    validate_placement,
)
from station_ui.core.session_identity import is_valid_session_uid
from station_ui.core.setup_presets import (
    POSE_TRACKER_IPHONE,
    POSE_TRACKER_NONE,
    POSE_TRACKER_VIVE,
    get_preset,
    template_path,
)
from station_ui.core.task_metadata import task_metadata
from umi.common.utility.gripper_pipeline import GripperPipeline
from umi.common.utility.reference_camera import (
    sync_reference_camera_to_video_sources,
    video_source_names,
)
from umi.common.utility.tracking import tracking_device_side, tracking_devices_in
from umi.utils.device_registry import (
    get_iphone_devices,
    get_oak_devices,
    get_umi_v4_grippers,
)

# Store active configs outside the checkout so repository cleanup cannot
# delete session state.
GENERATED_DIR = ("data_recording",)

# Read optional source defaults from production/optional_devices.yaml.
# Inject selected device IDs and scanned OAK resolution in code. Latency
# is a source-level field, not a device parameter.
_OPTIONAL_DEVICES_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "data_recording"
    / "production"
    / "optional_devices.yaml"
)

# Map each assignable source type to its identity key. oak_device stores
# that key at source level; other camera types use params. Retain the
# oak_device entry so saved multi-stream board assignments can be restored.
_DEVICE_PARAM_KEY = {
    "oak_device": "oak_device_name",
    "oak_camera": "oak_device_name",
    "iphone_camera": "iphone_device_name",
    # UVC selection uses the V4L2 index expected by UVCCamera.
    "uvc_camera": "camera_index",
}
# Use raw-ID keys for unregistered devices; name keys trigger registry resolution.
_IDENTIFIER_SIBLING = {
    "oak_device_name": "oak_device_id",
    "iphone_device_name": "iphone_udid",
}


def _looks_like_oak_id(value) -> bool:
    """depthai ids are long hex; registry names ('oak-FFC-05') are not."""
    text = str(value or "").strip()
    return len(text) >= 12 and all(c in "0123456789abcdefABCDEF" for c in text)


def _looks_like_iphone_udid(value) -> bool:
    """UDIDs are hex + hyphen; 'iphone15p-01' has non-hex letters."""
    text = str(value or "").strip()
    return len(text) >= 20 and all(c in "0123456789abcdefABCDEF-" for c in text)


_IS_RAW_ID = {
    "oak_device_name": _looks_like_oak_id,
    "iphone_device_name": _looks_like_iphone_udid,
}

# Map source types to hardware kinds for device pickers.
_DEVICE_KIND = {
    "oak_device": "oak",
    "oak_camera": "oak",
    "iphone_camera": "iphone",
    "uvc_camera": "uvc",
}
_GRIPPER_SUFFIX = "_umi_gripper"
# Record UNKNOWN_OPERATOR when the optional operator field is blank.
UNKNOWN_OPERATOR = "Unknown"
# Record UNKNOWN_TASK_DESCRIPTION when no task description is provided.
UNKNOWN_TASK_DESCRIPTION = "Unknown"


@dataclass(frozen=True)
class SessionSpec:
    preset_id: str
    operator_name: str
    task_name: str
    session_uid: str
    data_dir: str
    # Blank requests a task-registry lookup; a resolved description avoids repeating it.
    task_description: str = ""
    hw_version: str = ""


@dataclass(frozen=True)
class RequiredCamera:
    """One camera a template declares and therefore demands a device for."""

    name: str
    source_type: str
    device_kind: str


@dataclass(frozen=True)
class ViveSource:
    """One Vive tracker a template declares, and the arm it drives.

    Identification is a type question -- `type: vive_tracker` -- while the arm
    comes from the leading side token of the source name, which is the rule
    `umi/common/utility/tracking.py` already owns and which the data check, the
    replay scripts and the trajectory visualizers all share. `endpoint` and
    `tracker_key` are the template's values: the console offers them as the
    starting point and writes over them only when the operator chooses
    something else.

    `arm` holds the bare side -- "left", the spelling `gripper_arms` beside it
    uses -- and not the "left_arm" that `tracking_device_to_arm` returns. The
    two spellings live one import apart and mixing them fails silently: a
    lookup keyed on one form misses every entry stored under the other, so the
    caller concludes the arm has no tracker instead of raising.
    """

    name: str
    arm: str
    endpoint: str
    tracker_key: str


@dataclass(frozen=True)
class DeviceStream:
    """One recorded stream of a block, as the template declares it.

    ``socket`` is the OAK connector the stream is taken off (``CAM_B``), and is
    empty for every source type that has only one -- an iPhone, a UVC camera.
    It is the half of the pair ``resolution_for`` needs, so a caller wanting a
    stream's real picture size can ask for it without re-reading the template.

    ``resolution`` is what the template asks for, which is not always what gets
    written: ``_assign_device`` overrides an OAK stream with the size its sensor
    module actually wants. Kept anyway, because for every other source type it
    is the only answer there is.
    """

    name: str
    socket: str = ""
    resolution: tuple = ()


@dataclass(frozen=True)
class DeviceBlock:
    """One physical source a template declares, plus every stream it carries.

    An OAK board is one device behind as many streams as it has sockets
    recording, so a form with a field per stream would ask for the same board
    once per stream and let the answers disagree. The block is what the operator
    actually plugs in, which is what a device field should be asking about.

    ``key`` is the first stream name, so a block is addressable in the same
    namespace as the cameras it declares -- no synthetic ids, and a rejection
    naming any of its streams still points at the field that owns it.

    ``streams`` is ``stream_names`` plus the two things the form could not ask
    for without re-opening the template: the socket and the declared size. The
    names stay because every existing caller matches on them, and the pair is
    kept in step by ``__post_init__`` rather than by both call sites remembering
    to fill in the second.
    """

    key: str
    source_type: str
    device_kind: str
    stream_names: tuple[str, ...] = ()
    streams: tuple[DeviceStream, ...] = ()

    def __post_init__(self):
        # Expose streams even for simplified source blocks with unknown socket
        # or resolution metadata.
        if not self.streams and self.stream_names:
            object.__setattr__(
                self,
                "streams",
                tuple(DeviceStream(name) for name in self.stream_names),
            )


@dataclass(frozen=True)
class TemplateRequirements:
    """Every device a preset's template will not record without.

    Read off the template rather than hardcoded per preset: that is the same
    property SETUP_PRESETS bought for the setup page, extended to the form.
    """

    cameras: tuple[RequiredCamera, ...] = ()
    gripper_arms: tuple[str, ...] = ()
    blocks: tuple[DeviceBlock, ...] = ()
    # Keep this field last to preserve positional TemplateRequirements construction.
    vive_sources: tuple[ViveSource, ...] = ()

    def __post_init__(self):
        # Treat each flat camera as a one-stream physical block when explicit block
        # metadata is absent.
        if not self.blocks and self.cameras:
            object.__setattr__(
                self,
                "blocks",
                tuple(
                    DeviceBlock(c.name, c.source_type, c.device_kind, (c.name,))
                    for c in self.cameras
                ),
            )


@dataclass(frozen=True)
class DeviceSpec:
    grippers: Mapping[str, str] = field(default_factory=dict)
    required_cameras: Mapping[str, str] = field(default_factory=dict)
    optional: tuple[OptionalDeviceAssignment, ...] = ()
    # An empty reference selects automatic anchoring. Reject an explicit
    # reference camera that the rig does not record.
    reference_override: str = ""
    # Blank Vive settings use the template's working defaults; they are not
    # unassigned device placeholders.
    vive_endpoint: str = ""
    vive_keys: Mapping[str, str] = field(default_factory=dict)
    relaxed_checks: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReferenceCandidate:
    """One camera the operator may anchor the world frame on."""

    name: str


@dataclass(frozen=True)
class ReferenceChoices:
    """What the reference picker renders from.

    ``auto`` is the camera "Automatic" resolves to -- the same one the assembler
    auto-picks. ``candidates`` are the cameras the operator may name instead:
    the head camera and the template's calibratable tracking cameras.
    """

    auto: str
    candidates: tuple[ReferenceCandidate, ...]


def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    # Match the templates' block style: "  - type: oak_device", not "- type:".
    y.indent(mapping=2, sequence=4, offset=2)
    y.width = 4096
    # Without this, repeated lists such as [640, 400] emit &id001/*id001.
    y.representer.ignore_aliases = lambda data: True
    return y


@functools.lru_cache(maxsize=64)
def _parsed_once(text: str):
    """The parsed document for one exact piece of YAML. Never handed out.

    Keyed on the text, not on the path and a stat. Rewrite a file with content
    of the same length quickly enough that the filesystem gives both writes the
    same mtime, and a stat-keyed cache serves the previous parse. That is not
    hypothetical for a console whose whole job is rewriting config, and "two
    writes cannot land in the same mtime tick" is a claim about the filesystem
    that this code has no business making. Keying on the bytes removes the
    question: identical text parses to an identical document, and different
    text is a different key.

    What it costs is re-reading the file, which is the cheap half -- reading and
    decoding is a fraction of what ruamel then does with it.

    Bounded because the tests open a fresh copy of the config tree per temp
    directory, and an unbounded map would keep every one of them alive for the
    life of the process.
    """
    yaml = YAML()
    # Cache parsing independently of formatting; dump settings belong to the
    # YAML writer used by the caller.
    yaml.preserve_quotes = True
    return yaml.load(text)


def _load_yaml(path: Path):
    """A private, mutable copy of *path*, parsed only when its content is new.

    Read-only callers would otherwise re-parse the same rig template on every
    request, and ruamel parsing dominates their CPU time. Parsing a template
    costs 4-10ms and deep-copying the result 1-3ms, so the copy is 3-5x cheaper
    -- and it is a copy, not the cached object, so a caller that mutates what
    it gets back cannot poison the next reader. That matters here: ruamel nodes
    carry comment tokens, and two callers must never share one node (see
    `_optional_source_specs`).

    Deliberately not used by `assemble`, which mutates the document it loads and
    then dumps it back out with comments preserved. It is the one caller whose
    correctness depends on the exact object graph ruamel built, and it runs once
    per apply rather than once per render, so it has little to gain and the most
    to lose.
    """
    return copy.deepcopy(_parsed_once(path.read_text()))


def _tail_owner(node):
    """``(container, key)`` of the deepest-last entry of *node*.

    ruamel parks the comment that follows a whole block on its last *leaf*, not
    on the block itself: in production/bimanual_v4.yaml the banner that
    introduces ``reference_camera`` hangs off
    ``video_sources[-1].params.camera_resolution``. Anything appended after
    that leaf therefore lands under the wrong heading unless the comment is
    carried along.
    """
    while True:
        if isinstance(node, dict) and node:
            key = list(node.keys())[-1]
        elif isinstance(node, list) and len(node):
            key = len(node) - 1
        else:
            return None
        child = node[key]
        flow = getattr(child, "fa", None)
        if (
            isinstance(child, (dict, list))
            and len(child)
            # A flow-style [640, 400] renders on one line, so its own key is
            # already the right anchor; descending would split it.
            and not (flow is not None and flow.flow_style())
        ):
            node = child
            continue
        return node, key


def _pop_tail_comment(container):
    owner = _tail_owner(container)
    if owner is None:
        return None
    node, key = owner
    comments = getattr(node, "ca", None)
    return comments.items.pop(key, None) if comments is not None else None


def _push_tail_comment(container, carried) -> None:
    if carried is None:
        return
    owner = _tail_owner(container)
    if owner is None:
        return
    node, key = owner
    if getattr(node, "ca", None) is not None:
        node.ca.items[key] = carried


def _set_keys(mapping, pairs) -> None:
    """Set keys, keeping the block's trailing comment at the block's end."""
    carried = _pop_tail_comment(mapping)
    for key, value in pairs:
        mapping[key] = value
    _push_tail_comment(mapping, carried)


def _extend(sequence, items) -> None:
    """Append, keeping the sequence's trailing comment at the sequence's end."""
    if not items:
        return
    carried = _pop_tail_comment(sequence)
    for item in items:
        sequence.append(item)
    _push_tail_comment(sequence, carried)


def _quoted(value):
    """Double-quote a scalar so it matches the style already in the templates."""
    return DoubleQuotedScalarString(str(value))


def _flow(values):
    """Inline ``[640, 400]`` rather than a block list, matching file style."""
    seq = CommentedSeq(list(values))
    seq.fa.set_flow_style()
    return seq


def _resolution_of(value) -> tuple:
    """``[640, 400]`` from a template as ``(640, 400)``, or ``()``.

    Tolerant on purpose: this feeds a panel that describes the rig, so a
    template with a malformed or missing size must leave that one line blank
    rather than stop the page that would have shown the other nine.
    """
    try:
        width, height = (int(part) for part in value)
    except (TypeError, ValueError):
        return ()
    return (width, height)


def _pipeline_value(generation) -> str:
    """A preset's gripper generation in the spelling ``meta.gripper_pipeline``
    is read back with.

    Normalised through :class:`GripperPipeline` so ``V4`` and ``v4`` cannot
    become two configs that compare unequal. An unknown version is passed
    through untouched rather than folded onto the default: ``parse`` falls back
    silently, and a rig quietly recorded as the wrong version is precisely the
    failure this key exists to prevent. ``recording_utils`` already prints a red
    warning for a value it does not recognise, which is the right place to
    notice it.
    """
    if GripperPipeline.is_known(generation):
        return GripperPipeline.parse(generation).value
    return str(generation or "")


def _major(version) -> str:
    """The major token of a hardware label: ``v2.4`` and ``v2`` both give ``2``.

    Tolerant by design -- a label this cannot parse simply fails to match, and
    :func:`resolve_hw_version` then prefers the rig. It must never raise: what
    it reads is a registry entry somebody typed.
    """
    return str(version or "").strip().lstrip("vV").split(".", 1)[0]


def resolve_hw_version(preset, registry_value, session_value="") -> str:
    """Which hardware revision this session actually ran on.

    The rig is primary. The task registry's ``HW_version`` is a per-task label:
    it records the revision a task was registered under, not the bench in the
    room. Many registered tasks carry an older label such as ``v1``, so taking
    the registry value alone would stamp a V3 bench's session ``v1`` --
    contradicting ``gripper_pipeline`` and ``hardware.*.generation`` a few lines
    below it in the same file.

    The registry supplies the minor. A preset knows ``v2``; the registry knows
    ``v2.4``, ``v2.1``, ``v1.5``. When the two agree on the major, the
    registry's is the fuller label and is kept verbatim -- taking the rig's
    would drop information. When they disagree, the rig wins, because the rig
    is the one that recorded.

    ``session_value`` stays an override for a caller that has already resolved
    the pair. No station_ui caller sets it today; every ``SessionSpec`` leaves
    it blank.
    """
    if session_value:
        return session_value
    # Normalize generation labels so V4 and v4 compare as the same hardware.
    rig = _pipeline_value(preset.gripper_generation)
    registry = str(registry_value or "").strip()
    if rig and registry and _major(registry) == _major(rig):
        return registry
    return rig or registry


# Write every pose slot explicitly, including absent arms, so consumers
# can read a stable metadata schema.
_TRACKING_SLOTS = ("head", "left", "right")

# iPhone sources carry ARKit poses, including head-mounted phones; OAK and
# UVC sources do not. Use setup_presets.POSE_TRACKER_* names for both config
# metadata and session-folder identity, including notracker for absence.
_POSE_SOURCE = {
    "vive_tracker": POSE_TRACKER_VIVE,
    "iphone_camera": POSE_TRACKER_IPHONE,
}


def _tracking_slot(name) -> str:
    """The slot a source name leads with, or "" when it names none.

    The leading-token rule `umi/common/utility/tracking.py` owns, widened by
    exactly one member: `tracking_device_side` answers arms only and returns
    None for `head_*`, because a head is not an arm. Here the head is a slot,
    so the token is matched against `_TRACKING_SLOTS` directly rather than
    calling that function and special-casing its None.
    """
    token = str(name or "").split("_", 1)[0].lower()
    return token if token in _TRACKING_SLOTS else ""


def _tracking_sources(cfg, optional) -> CommentedMap:
    """What produced each slot's pose: ``iphone``, ``vive`` or ``notracker``.

    The per-slot half of what `SetupPreset.pose_tracker` states for the rig as a
    whole: that field is one token and reaches the session folder name, this
    block is three and reaches the config and the dataset attrs. They are held
    to the same vocabulary and pinned to each other in
    `tests/gui/test_template_pose_tracker_declaration.py`, so the name a session
    is filed under and the file inside it cannot name two different rigs.

    Derived from what the rig records, never from the preset id. An iPhone rig
    declares `{side}_tracking_camera` and a Vive rig `{side}_tracking_pose`, but
    reading that requires knowing both naming conventions and both source types;
    this block writes the answer down once so consumers do not have to match
    preset ids or source types themselves.

    Numeric sources first, so a slot named by both a puck and a phone reads as
    `vive`: a rig carrying both records the phone for its frames and the puck
    for the pose. No shipped template does this; the order decides it anyway
    rather than leaving it to source order.

    The head arrives through `optional` rather than through *cfg*: `_apply_meta`
    runs before `_apply_optional` appends the head camera, so the source block
    does not exist yet. Same information -- `_apply_optional` emits exactly one
    source per assignment -- and it keeps this key ahead of `session_uid`
    instead of appending a map after the block's last scalar.
    """
    declared = list(cfg.get("numeric_sources") or []) + list(
        cfg.get("video_sources") or []
    )
    # The shared "which groups carry a tracked pose" rule, so a camera that
    # merely leads with a side token can never be read as an arm's pose source.
    tracked = set(tracking_devices_in([str(s.get("name") or "") for s in declared]))

    found = {}

    def claim(name, source_type):
        slot = _tracking_slot(name)
        kind = _POSE_SOURCE.get(source_type)
        if slot and kind:
            found.setdefault(slot, kind)

    for source in declared:
        name = str(source.get("name") or "")
        if name in tracked:
            claim(name, source.get("type"))
    for assignment in optional or ():
        claim(
            assignment.logical_name,
            get_optional_type(assignment.type_id).source_type,
        )

    out = CommentedMap()
    for slot in _TRACKING_SLOTS:
        out[slot] = found.get(slot, POSE_TRACKER_NONE)
    return out


def _params(mapping) -> CommentedMap:
    out = CommentedMap()
    for key, value in mapping.items():
        out[key] = _flow(value) if isinstance(value, list) else value
    return out


def _optional_source_specs():
    """The static optional-device source blocks, from optional_devices.yaml.

    Code-adjacent static data, like the descriptors in optional_devices.py: the
    values are a property of the codebase, not of a deployment, so they are read
    from the module's own tree rather than the caller's repo_root (which selects
    the template and the output path, both of which legitimately vary). Fresh
    per call so two same-type blocks never share a ruamel node -- `_load_yaml`
    keeps that promise by handing back a deep copy, and only the parse behind it
    is shared.
    """
    return _load_yaml(_OPTIONAL_DEVICES_PATH)


def _optional_source_spec(source_type: str):
    specs = _optional_source_specs()
    if source_type not in specs:
        raise ValueError(
            f"no optional-device spec for source type {source_type!r} in "
            f"{_OPTIONAL_DEVICES_PATH.name} (known: {sorted(specs)})"
        )
    return specs[source_type]


def optional_stream(source_type: str, name: str) -> DeviceStream:
    """What one optional camera will record, without assembling anything.

    A page describing the rig has no generated file to read the answer out of --
    the whole point of describing it is that it has not been written yet. So the
    two values ``_optional_block`` injects are answered here instead: the socket
    an OAK camera is taken off, and the static size every other type declares.

    Beside the block builder deliberately. A copy of "an optional OAK is CAM_A"
    in the view layer would be right until the day that block moved, and wrong
    silently after it. An unknown source type degrades to a nameless stream
    rather than raising: a descriptor this file has never seen must cost one
    blank line on a panel, not the page around it.
    """
    try:
        spec = _optional_source_spec(source_type)
    except ValueError:
        return DeviceStream(name)
    if source_type == "oak_camera":
        # Optional-source resolution is discovered from the module on this socket;
        # return the socket so the caller can query resolution_for.
        return DeviceStream(name, "CAM_A")
    return DeviceStream(
        name, "", _resolution_of((spec.get("params") or {}).get("camera_resolution"))
    )


def _set_device_identifier(params, key: str, value) -> None:
    """Write the identifier under the key its value calls for.

    The sibling key is dropped, so one block can never carry both a name and an
    id naming different cameras — recording_utils prefers the id, which would
    make the file disagree with what actually records.
    """
    sibling = _IDENTIFIER_SIBLING.get(key)
    is_raw_id = _IS_RAW_ID.get(key)
    if sibling and is_raw_id and is_raw_id(value):
        key, sibling = sibling, key
    params[key] = _quoted(value)
    if sibling and sibling in params:
        del params[sibling]


def _camera_index(name: str, value) -> int:
    """The V4L2 index the operator typed, as an int.

    The message names the camera because that is what pins the rejection to the
    field the operator has to go and fix -- `_errors_for` in routes/config.py
    matches on the stream names a block carries.
    """
    text = str(value or "").strip()
    # "/dev/video4" is what v4l2-ctl prints and what an operator reads off the
    # bench, so it is accepted as the index it names rather than refused.
    if text.startswith("/dev/video"):
        text = text[len("/dev/video") :]
    if not text.isdigit():
        raise ValueError(
            f"camera {name!r} needs a V4L2 index (a number, or /dev/videoN), "
            f"not {str(value).strip()!r}"
        )
    return int(text)


def _descriptor_name_matches(descriptor, name: str) -> bool:
    """True when *name* is the name this descriptor would allocate."""
    return matches_logical_name(descriptor, name)


def _is_optional_source_name(name) -> bool:
    return any(
        _descriptor_name_matches(descriptor, str(name))
        for descriptor in OPTIONAL_DEVICE_TYPES
    )


def source_device(source) -> str:
    """The device identifier a video-source block names, whatever key it used.

    The inverse of _set_device_identifier, and the only correct way to read a
    generated config back: a block written from a raw hardware id carries
    ``oak_device_id`` where a registry name would have used ``oak_device_name``,
    and a reader that knows only one of the two silently sees an empty device.
    """
    source_type = source.get("type")
    if source_type == "oak_device":
        # oak_device carries the identifier at source level, not in params.
        holders = [source]
    else:
        holders = [source.get("params") or {}]
    key = _DEVICE_PARAM_KEY.get(source_type)
    if key is None:
        return ""
    for holder in holders:
        for candidate in (key, _IDENTIFIER_SIBLING.get(key)):
            if (
                candidate
                and holder.get(candidate) is not None
                and str(holder[candidate]).strip()
            ):
                return str(holder[candidate]).strip()
    return ""


# Resolve registry aliases at comparison time. Capturing the loaded tables
# at import would ignore later registry updates.
_IDENTITY_REGISTRIES = (
    get_oak_devices,
    get_iphone_devices,
    get_umi_v4_grippers,
)


def canonical_device_identity(value) -> str:
    """One physical device, one key -- whichever spelling names it.

    ``right_umi_camera = 'oak-FFC-05'`` and ``right_trigger_camera = '<mxid>'``
    are the same board when ``<mxid>`` is that board's raw depthai id: one is
    the registry name, the other the id depthai reports. Comparing the two
    strings would call that a mismatch and refuse a correct form, so every
    device comparison in this module folds through here first.

    The fold is the registry's own name -> id table, which is the same table
    ``DeviceScanner.by_identifier`` matches a scan record on. A device the
    registry does not know keeps its own spelling (case-folded, because a raw
    hex id is written both ways), so it still compares equal to itself and
    never equal to a different one.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    folded = text.casefold()
    for load in _IDENTITY_REGISTRIES:
        try:
            table = load() or {}
        except Exception:
            # If the registry cannot load, compare device spellings directly rather
            # than failing the form.
            continue
        for name, identifier in table.items():
            if str(name).casefold() == folded:
                return str(identifier).strip().casefold()
            if str(identifier).strip().casefold() == folded:
                return folded
    return folded


def _has_name(stream) -> bool:
    try:
        return bool(stream.get("name"))
    except AttributeError:
        return False


def _deduped(items) -> list:
    """Order-preserving unique. Two streams of one board warn identically."""
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


class ConfigAssembler:
    """Preset + device choices -> one generated recording config."""

    def __init__(self, repo_root, *, state_dir=None, scanner=None):
        self.repo_root = Path(repo_root)
        # Read templates from the checkout and write active configs to the
        # environment-selected state directory.
        self.state_dir = (
            Path(state_dir)
            if state_dir is not None
            else settings.state_dir_from_environment()
        )
        # Pass device-discovery results from the web process's scan worker. The
        # module-global scanner is empty there; omitting results would hide native
        # OAK resolutions and produce false missing-device warnings.
        self.scanner = scanner
        # Retain fallback-resolution warnings: a default aspect ratio can silently
        # crop a camera's field of view.
        self.last_resolution_warnings: list[str] = []

    def output_path(self, session: SessionSpec) -> Path:
        return self.state_dir.joinpath(*GENERATED_DIR, f"{session.preset_id}.yaml")

    def template_requirements(self, preset_id: str) -> TemplateRequirements:
        """Every device the preset's template demands, in template order.

        The form renders from this, so a template that grows a camera grows a
        field with no UI change -- and, because ``assemble`` enforces the same
        list, the field cannot be left empty either.
        """
        preset = get_preset(preset_id)
        template = template_path(preset, self.repo_root)
        return self._requirements(_load_yaml(template))

    def reference_choices(
        self, preset_id: str, devices: DeviceSpec
    ) -> ReferenceChoices:
        """The cameras the operator may anchor the world frame on, and the pick.

        Computed from the template plus the operator's optional cameras without
        a full assemble, so the picker renders on a half-filled form. Eligible
        are the head camera and the template's tracking cameras that the
        ``relative_calibration`` rig lists (always recorded, always calibrated).
        "Automatic" resolves to the same name ``assemble`` would auto-pick --
        computed by the same :meth:`_pick_reference`, off a rate table built
        from the draft rather than from an assembled file, so the label over the
        picker and the anchor in the config cannot disagree.
        """
        preset = get_preset(preset_id)
        template = template_path(preset, self.repo_root)
        cfg = _load_yaml(template)
        base = {camera.name for camera in self._requirements(cfg).cameras}
        params = self._calibration_check(cfg) or {}
        tracking = [c for c in (params.get("cameras") or []) if c in base]
        head = next(
            (
                name
                for name in self._HEAD_REFERENCE_NAMES
                if name in self._rates_from_devices(cfg, devices)
            ),
            "",
        )
        auto = self._pick_reference(self._rates_from_devices(cfg, devices)) or str(
            cfg.get("reference_camera") or ""
        )
        # List reference candidates in priority order, head first.
        candidates = [ReferenceCandidate(head)] if head else []
        seen = {candidate.name for candidate in candidates}
        for name in tracking:
            if name not in seen:
                candidates.append(ReferenceCandidate(name))
                seen.add(name)
        # Include the automatic choice even when it falls back to a wrist camera.
        if auto and auto not in seen:
            candidates.append(ReferenceCandidate(auto))
        return ReferenceChoices(auto=auto, candidates=tuple(candidates))

    @staticmethod
    def _requirements(cfg) -> TemplateRequirements:
        cameras = []
        blocks = []
        for source in cfg.get("video_sources") or []:
            source_type = source.get("type")
            # Read stream names, sockets and sizes together to keep topology metadata
            # consistent.
            declared = [
                DeviceStream(
                    stream["name"],
                    str(stream.get("socket") or ""),
                    _resolution_of(stream.get("resolution")),
                )
                for stream in source.get("streams") or []
                if stream.get("name")
            ]
            if not declared and source.get("name"):
                # A single-stream source carries its size under params instead
                # (`camera_resolution`), and has no socket to name.
                declared = [
                    DeviceStream(
                        source["name"],
                        "",
                        _resolution_of(
                            (source.get("params") or {}).get("camera_resolution")
                        ),
                    )
                ]
            # Regenerate optional blocks from operator selections rather than treating
            # optional template blocks as requests to enable hardware.
            declared = [s for s in declared if not _is_optional_source_name(s.name)]
            if not declared:
                continue
            names = [stream.name for stream in declared]
            kind = _DEVICE_KIND.get(source_type, "")
            for name in names:
                cameras.append(RequiredCamera(name, source_type, kind))
            # One block per source, so the form can ask for the board once
            # instead of once per stream it happens to carry.
            blocks.append(
                DeviceBlock(names[0], source_type, kind, tuple(names), tuple(declared))
            )
        arms = [
            str(entry["name"])[: -len(_GRIPPER_SUFFIX)]
            for entry in cfg.get("numeric_sources") or []
            if entry.get("type") == "umi_gripper"
            and str(entry.get("name") or "").endswith(_GRIPPER_SUFFIX)
        ]
        vive = []
        for entry in cfg.get("numeric_sources") or []:
            if entry.get("type") != "vive_tracker":
                continue
            name = str(entry.get("name") or "")
            # Skip pose sources without an arm token; head trackers do not belong in
            # left/right arm fields.
            arm = tracking_device_side(name)
            if arm is None:
                continue
            params = entry.get("params") or {}
            vive.append(
                ViveSource(
                    name=name,
                    arm=arm,
                    endpoint=str(params.get("endpoint") or ""),
                    tracker_key=str(params.get("tracker_key") or ""),
                )
            )
        return TemplateRequirements(
            cameras=tuple(cameras),
            gripper_arms=tuple(arms),
            blocks=tuple(blocks),
            vive_sources=tuple(vive),
        )

    @classmethod
    def _reject_partial_form(cls, cfg, devices: DeviceSpec) -> None:
        """Refuse a form that leaves a declared device unassigned.

        An unassigned slot is not "left alone": the template's placeholder
        stays, so the file names placeholder hardware as confidently as it
        names real hardware. Worse, a placeholder is invisible to
        _reject_shared_devices, so omitting one camera is how one physical
        board ends up backing two source blocks with nothing complaining.
        """
        requirements = cls._requirements(cfg)
        assigned = {
            name
            for name, device in (devices.required_cameras or {}).items()
            if str(device or "").strip()
        }
        missing = [c.name for c in requirements.cameras if c.name not in assigned]
        armed = {
            arm
            for arm, device in (devices.grippers or {}).items()
            if str(device or "").strip()
        }
        missing += [
            f"{arm}{_GRIPPER_SUFFIX}"
            for arm in requirements.gripper_arms
            if arm not in armed
        ]
        if missing:
            raise ValueError(
                "no device assigned for " + ", ".join(missing) + " — the template's "
                "placeholder hardware would be recorded instead"
            )

    def assemble(self, session: SessionSpec, devices: DeviceSpec) -> Path:
        if not is_valid_session_uid(session.session_uid):
            raise ValueError(f"bad session_uid {session.session_uid!r}")
        preset = get_preset(session.preset_id)
        yaml = _yaml()
        template = template_path(preset, self.repo_root)
        cfg = yaml.load(template.read_text())
        # Clear warnings before each assembly so failures cannot expose results
        # from an earlier config.
        self.last_resolution_warnings = []
        warnings: list[str] = []

        # Reject incomplete selections before applying any configuration changes.
        self._reject_partial_form(cfg, devices)
        self._apply_meta(cfg, preset, session, devices)
        self._apply_recording(cfg, session)
        # Validate required and optional devices together so identity collisions
        # across the two sets cannot pass.
        targets = self._resolve_required_cameras(cfg, devices)
        self._reject_shared_devices(cfg, targets, devices)
        self._apply_required_cameras(cfg, targets, warnings)
        self._apply_grippers(cfg, devices)
        self._apply_vive(cfg, devices)
        self._apply_optional(cfg, devices, warnings)
        # Choose the world-frame reference after adding optional cameras.
        self._apply_reference(cfg, devices.reference_override)
        self._reconcile_reference_camera(cfg)

        from station_ui import enforcement

        policy = enforcement.policy(devices.relaxed_checks)
        if policy:
            cfg["check_enforcement"] = policy
        else:
            cfg.pop("check_enforcement", None)

        out = self.output_path(session)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_dump(yaml, cfg, out)
        # Write task overrides into generated check configs through
        # station_ui.data_check_config. Never modify the tracked baseline in place.
        self.last_resolution_warnings = _deduped(warnings)
        return out

    # ------------------------------------------------------------- meta

    @staticmethod
    def _apply_meta(cfg, preset, session: SessionSpec, devices: DeviceSpec) -> None:
        meta = cfg["meta"]

        hardware = CommentedMap()
        for arm, device in devices.grippers.items():
            entry = CommentedMap()
            entry["device"] = device
            # Take generation from the preset; raw device serials need not encode it.
            entry["generation"] = preset.gripper_generation
            # Record the actual gripper/camera association for auditing device swaps.
            # Omit an unknown association rather than writing an ambiguous empty value.
            camera = str(
                (devices.required_cameras or {}).get(f"{arm}_umi_camera", "") or ""
            ).strip()
            if camera:
                entry["camera"] = camera
            hardware[f"{arm}_umi_gripper"] = entry

        # Derive head-mount metadata from selected devices, not template defaults.
        use_head_mount = any(
            get_optional_type(assignment.type_id).section == "head"
            for assignment in devices.optional or ()
        )

        # Replace template description placeholders with the resolved task text or
        # UNKNOWN_TASK_DESCRIPTION so neither episodes nor LeRobot labels inherit
        # the template's placeholder.
        registry = task_metadata(session.task_name)
        description = (
            session.task_description.strip()
            or registry.description
            or UNKNOWN_TASK_DESCRIPTION
        )
        # Resolve the recording rig's hardware revision separately from the
        # revision associated with a task-registry entry.
        hw_version = resolve_hw_version(preset, registry.hw_version, session.hw_version)

        _set_keys(
            meta,
            [
                ("operator_name", session.operator_name.strip() or UNKNOWN_OPERATOR),
                ("task_name", session.task_name),
                ("task_description", description),
                ("HW_version", hw_version),
                ("training_stage", registry.training_stage),
                ("training_stage_version", registry.training_stage_version),
                ("setup_preset", preset.id),
                ("operation_mode", preset.operation_mode),
                # Use the recording consumer's source key consistently.
                ("gripper_pipeline", _pipeline_value(preset.gripper_generation)),
                ("use_head_mount", use_head_mount),
                # Record the source of each pose slot beside the other rig metadata.
                ("tracking_source", _tracking_sources(cfg, devices.optional)),
                # End the metadata block on a scalar so the next section's YAML comment
                # attaches to the intended key.
                ("hardware", hardware),
                ("session_uid", session.session_uid),
            ],
        )
        # Remove the redundant task-description copy; dataset annotations own the task
        # text.
        meta.pop("language_instruction", None)

    @staticmethod
    def _apply_recording(cfg, session: SessionSpec) -> None:
        recording = cfg.get("recording")
        if recording is None:
            recording = CommentedMap()
            _set_keys(cfg, [("recording", recording)])
        _set_keys(recording, [("data_dir", _quoted(session.data_dir))])

    # --------------------------------------------------------- topology

    @classmethod
    def _resolve_required_cameras(cls, cfg, devices: DeviceSpec) -> dict:
        """Locate the source block behind each required camera name.

        Pure — nothing is written here, so a rejected assignment cannot leave
        half the rig repointed. Returns ``index -> (device, [camera names])``.
        """
        sources = cfg.get("video_sources") or []
        # All streams in an OAK block must resolve to one physical board.
        targets: dict[int, tuple[str, list]] = {}
        for name, device in (devices.required_cameras or {}).items():
            device = str(device or "").strip()
            if not device:
                raise ValueError(f"no device given for camera {name!r}")
            index = cls._locate_camera(sources, name)
            if index is None:
                raise ValueError(
                    f"camera {name!r} is not declared by this template "
                    f"(known: {sorted(cls._declared_camera_names(sources))})"
                )
            claimed, names = targets.setdefault(index, (device, []))
            # Compare canonical identities so registry aliases and raw serials match.
            if canonical_device_identity(claimed) != canonical_device_identity(device):
                raise ValueError(
                    f"{name!r} and {names[0]!r} are streams of one device but "
                    f"were assigned {device!r} and {claimed!r}"
                )
            names.append(name)
        return targets

    @classmethod
    def _reject_shared_devices(cls, cfg, targets, devices: DeviceSpec) -> None:
        """One physical device cannot back two source blocks.

        Checked across the union of required cameras, optional cameras and
        grippers, because every driver fails on the second block and they fail
        in different ways: the OAK device cache hands it the same handle, and
        two ``IphoneCamera``s on one Record3D device leave one of them recording
        nothing at all — silently, which is the worse of the two.

        ``numeric_sources`` are in the same namespace, and a shared gripper is
        worse than any of the above. Two ``UartSensorReader``s on one
        ``/dev/ttyACM*`` do not fail loudly: the bimanual session records, and
        one arm simply carries the other arm's encoder trace. Nothing
        downstream can detect it — the distances are plausible, monotonic and
        correlated with a real gripper, just the wrong one — so the only place
        it can be caught is here, before the file is written.

        A Vive tracker key is the same failure one level out: two arms naming
        one puck record one pose twice, and every pose check — pose_movement,
        pose_jumps, consecutive_movement, the arm-steady pairing — passes on
        both arms because both traces are real. The calibration tracker check
        would stall rather than name it, so this is again the last place it can
        be refused.
        """
        claims = [(device, names[0]) for device, names in targets.values()]
        claims += [
            (str(assignment.device or "").strip(), assignment.logical_name)
            # Report empty selections through the named optional-camera field, not as
            # identity collisions.
            for assignment in devices.optional or ()
            if str(assignment.device or "").strip()
        ]
        claims += [
            # Report numeric-source names, matching the generated recording config.
            (str(device or "").strip(), f"{arm}{_GRIPPER_SUFFIX}")
            for arm, device in (devices.grippers or {}).items()
            if str(device or "").strip()
        ]
        # Detect duplicates by canonical board identity, but report the spelling
        # the operator supplied.
        owner: dict[str, str] = {}
        for device, name in claims:
            first = owner.setdefault(canonical_device_identity(device), name)
            if first != name:
                raise ValueError(
                    f"device {device!r} assigned to both {name!r} and {first!r}"
                )

        # Tracker keys have their own namespace and must not be resolved as device IDs.
        pucks: dict[str, str] = {}
        for source in cls._requirements(cfg).vive_sources:
            # Compare effective tracker keys, including template defaults, so overriding
            # one arm cannot duplicate the other arm's unchanged key.
            key = str((devices.vive_keys or {}).get(source.arm, "") or "").strip()
            key = key or source.tracker_key
            if not key:
                continue
            first = pucks.setdefault(key, source.name)
            if first != source.name:
                raise ValueError(
                    f"tracker key {key!r} assigned to both {source.name!r} "
                    f"and {first!r}"
                )

    def _apply_required_cameras(self, cfg, targets, warnings) -> None:
        """Point each resolved camera at the device the operator picked."""
        sources = cfg.get("video_sources") or []
        for index, (device, names) in targets.items():
            self._assign_device(sources[index], names, device, warnings)

    @staticmethod
    def _declared_camera_names(sources) -> set:
        names = set()
        for source in sources:
            if source.get("name"):
                names.add(source["name"])
            for stream in source.get("streams") or []:
                if stream.get("name"):
                    names.add(stream["name"])
        return names

    @staticmethod
    def _locate_camera(sources, name: str):
        """Index of the source block declaring *name*, at source or stream level."""
        for index, source in enumerate(sources):
            if source.get("name") == name:
                return index
            if any(s.get("name") == name for s in source.get("streams") or []):
                return index
        return None

    def _resolution(self, device, socket, warnings, default=None) -> list:
        """Stream size for this socket, keeping the reason it may be wrong.

        A device missing from the scan (a stale scan, a device plugged in after
        the scan, a headless assemble) falls back to ``default``, the 16:10
        wrist size unless the caller passes the head's. On a 4:3 module the
        wrist size crops, so the reason is collected rather than dropped.
        """
        resolution, warning = resolution_for(
            device, socket, self.scanner, default=default
        )
        if warning:
            warnings.append(warning)
        return resolution

    def _assign_device(self, source, names, device: str, warnings) -> None:
        if source.get("type") == "oak_device":
            # The identifier is a source-level key on oak_device, not a param.
            _set_device_identifier(source, "oak_device_name", device)
            for stream in source.get("streams") or []:
                if stream.get("name") not in names:
                    continue
                # The module on this socket can differ from when the block was
                # written; a 4:3 IMX378 needs 640x480, not 640x400.
                socket = stream.get("socket") or "CAM_A"
                stream["resolution"] = _flow(self._resolution(device, socket, warnings))
                # Record the detected sensor module as descriptive provenance for the
                # chosen
                # resolution. Omit it when discovery cannot identify the module.
                module = sensor_module_for(device, socket, self.scanner)
                if module:
                    stream["sensor_module"] = module
                else:
                    stream.pop("sensor_module", None)
            return
        key = _DEVICE_PARAM_KEY.get(source.get("type"))
        if key is None:
            raise ValueError(
                f"camera {names[0]!r} has type {source.get('type')!r}, which "
                "carries no device identifier"
            )
        params = source.get("params")
        if params is None:
            params = CommentedMap()
            source["params"] = params
        if key == "camera_index":
            # Write UVC indices as integers: VideoCapture interprets a quoted numeric
            # string as a filename. Other device identifiers remain quoted names.
            params[key] = _camera_index(names[0], device)
            return
        _set_device_identifier(params, key, device)
        if key == "oak_device_name":
            params["camera_resolution"] = _flow(
                self._resolution(device, "CAM_A", warnings)
            )

    @staticmethod
    def _apply_grippers(cfg, devices: DeviceSpec) -> None:
        """Set uart_device_name on the numeric sources the template declares."""
        numeric = cfg.get("numeric_sources") or []
        by_name = {n.get("name"): n for n in numeric if n.get("name")}
        for arm, device in (devices.grippers or {}).items():
            source_name = f"{arm}_umi_gripper"
            entry = by_name.get(source_name)
            if entry is None:
                raise ValueError(
                    f"template declares no numeric source {source_name!r} "
                    f"(known: {sorted(by_name)})"
                )
            entry["uart_device_name"] = _quoted(device)

    @classmethod
    def _apply_vive(cls, cfg, devices: DeviceSpec) -> None:
        """Point the template's Vive sources at this bench's bridge and pucks.

        A classmethod, unlike `_apply_grippers` beside it, because it calls
        `_requirements` -- which is a @staticmethod, so a @staticmethod here
        calling `cls._requirements(cfg)` would bind `cfg` to `cls` and raise on
        the first Apply of a Vive rig.

        An empty value leaves the template's own alone. That is what keeps an
        untouched rig writing exactly what it wrote before.
        """
        numeric = cfg.get("numeric_sources") or []
        by_name = {n.get("name"): n for n in numeric if n.get("name")}
        for source in cls._requirements(cfg).vive_sources:
            entry = by_name.get(source.name)
            if entry is None:
                # Report inconsistent hand-edited requirements as a form error, not an
                # assertion failure.
                continue
            params = entry.get("params")
            if params is None:
                # Create a params mapping when the source omits it.
                params = _params({})
                entry["params"] = params
            if devices.vive_endpoint:
                params["endpoint"] = _quoted(devices.vive_endpoint)
            key = (devices.vive_keys or {}).get(source.arm, "")
            if key:
                params["tracker_key"] = _quoted(key)

    # ----------------------------------------------------- optional devices

    def _apply_optional(self, cfg, devices: DeviceSpec, warnings) -> None:
        sources = cfg.get("video_sources")
        if sources is None:
            sources = CommentedSeq()
            cfg["video_sources"] = sources

        # Replace descriptor-generated optional sources so reassembly cannot accumulate
        # stale cameras.
        for index in range(len(sources) - 1, -1, -1):
            if _is_optional_source_name(sources[index].get("name")):
                del sources[index]

        # Validate all assignments before appending any optional sources.
        taken = {s.get("name") for s in sources if s.get("name")}
        blocks = []
        for assignment in devices.optional or ():
            descriptor = get_optional_type(assignment.type_id)
            # Use the descriptor's logical name for both the form row and config source.
            name = assignment.logical_name
            if not _descriptor_name_matches(descriptor, str(name)):
                raise ValueError(
                    f"logical name {name!r} does not match descriptor "
                    f"{descriptor.id!r} ({descriptor.fixed_name})"
                )
            device = str(assignment.device or "").strip()
            if not device:
                # Identify the optional-camera row and explain that it needs a device
                # selection or removal.
                raise ValueError(
                    f"{descriptor.label} is on the page but no device is chosen "
                    f"— pick one, or remove {descriptor.label} "
                    f"({name})"
                )
            # Normalize placement within its section so bottom-only placements cannot
            # be written for head cameras.
            placement = validate_placement(assignment.placement, descriptor.section)
            if name in taken:
                raise ValueError(f"duplicate video source name {name!r}")
            block = self._optional_block(descriptor, name, device, warnings)
            if placement:
                block["placement"] = placement
            blocks.append(block)
            taken.add(name)
        _extend(sources, blocks)

    def _optional_block(
        self, descriptor, name: str, device: str, warnings
    ) -> CommentedMap:
        if descriptor.source_type not in _DEVICE_PARAM_KEY:
            raise ValueError(
                f"descriptor {descriptor.id!r} has unsupported source type "
                f"{descriptor.source_type!r}"
            )
        spec = _optional_source_spec(descriptor.source_type)
        block = CommentedMap()
        block["type"] = spec["type"]
        block["name"] = name
        # Source-level fields, in the order the shipped blocks lay them out.
        # record_fps / stream_fps are absent for oak_camera, so each is copied
        # only when the file declares it.
        for field_name in ("record_fps", "stream_fps", "frame_type", "latency"):
            if field_name in spec:
                block[field_name] = spec[field_name]
        key = _DEVICE_PARAM_KEY[descriptor.source_type]
        identifier = CommentedMap()
        # Inject the operator's device selection rather than storing an identity
        # in reusable source defaults.
        _set_device_identifier(identifier, key, device)
        if descriptor.source_type == "oak_camera":
            # Use the detected OAK module's native aspect ratio. If discovery cannot
            # identify it, use the supported 4:3 head-camera fallback and retain a
            # warning.
            # A 16:10 default would crop an IMX378 image.
            identifier["camera_resolution"] = _flow(
                self._resolution(
                    device, "CAM_A", warnings, default=HEAD_STREAM_RESOLUTION
                )
            )
            # Store descriptive metadata at source level for a single-stream source;
            # recording_utils falls back there when stream metadata is absent.
            module = sensor_module_for(device, "CAM_A", self.scanner)
            if module:
                block["sensor_module"] = module
        for param_key, value in _params(spec.get("params") or {}).items():
            identifier[param_key] = value
        block["params"] = identifier
        return block

    # ------------------------------------------------------ reference camera

    # Head OAK aliases eligible for world-frame anchoring. Rank head iPhones
    # with other tracking cameras rather than moving the anchor merely because
    # a phone is head-mounted.
    _HEAD_REFERENCE_NAMES = ("head_oak_camera", "head_umi_camera")

    @classmethod
    def _reference_rank(cls, name: str) -> tuple:
        """Where *name* sits in the tie-break, lower first.

        Only consulted between cameras that record at the same rate -- rate is
        the primary key, see :meth:`_auto_reference`. The order is
        head -> tracking -> wrist -> trigger, right arm before left:

        * a head OAK is bolted to the rig and aimed down the workspace, so it is
          the most rigidly fixed camera a session can carry;
        * everything below moves with an arm, and is only reached by a rig that
          carries no head camera.

        The head check runs first because `head_umi_camera` would otherwise be
        read as a wrist camera by its suffix. Anything unrecognised sorts last,
        by name, so the pick is total and deterministic rather than dependent on
        `video_sources` order.

        The bottom OAK (`bottom_oak_camera`) is deliberately in that last group.
        It is as fixed as the head, but it looks up from the table, so there is
        no promise it sees the board the other cameras share during the relative
        calibration -- anchoring the world frame on it could stall the check on
        a camera that never sees the marker. It records; it does not anchor.
        """
        if name in cls._HEAD_REFERENCE_NAMES:
            return (0, cls._HEAD_REFERENCE_NAMES.index(name), name)
        for group, suffix in (
            (1, "_tracking_camera"),
            (2, "_umi_camera"),
            (3, "_trigger_camera"),
        ):
            if name.endswith(suffix):
                arm = name[: -len(suffix)]
                return (group, {"right": 0, "left": 1}.get(arm, 2), name)
        return (4, 0, name)

    # Exclude UVC sources from world-frame anchoring: without intrinsics they
    # cannot solve the calibration board, even if their low frame rate would
    # otherwise give them reference priority.
    _UNANCHORABLE_TYPES = frozenset({"uvc_camera"})

    @classmethod
    def _record_rates(cls, cfg) -> dict:
        """``camera name -> the rate frames are written at``, for every camera
        that can anchor the world frame (see ``_UNANCHORABLE_TYPES``).

        ``record_fps``, falling back to ``recording.record_fps`` -- never the
        source-level or ``params`` ``fps``, which is the rate the sensor is
        driven at. An ``oak_device``'s streams have no rate of their own and
        inherit the block's.
        """
        default = (cfg.get("recording") or {}).get("record_fps")
        rates = {}
        for source in cfg.get("video_sources") or []:
            try:
                if source.get("type") in cls._UNANCHORABLE_TYPES:
                    continue
                rate = source.get("record_fps")
                streams = source.get("streams") or []
                name = source.get("name")
            except AttributeError:
                continue
            if rate is None:
                rate = default
            names = [s.get("name") for s in streams if _has_name(s)]
            if not names and name:
                names = [name]
            for camera in names:
                rates[camera] = rate
        return rates

    @classmethod
    def _rates_from_devices(cls, cfg, devices: DeviceSpec) -> dict:
        """The mirror of :meth:`_record_rates`, for a draft that is not assembled.

        The template's own cameras keep their declared rate; each optional camera
        the operator filled in contributes the rate its block in
        `optional_devices.yaml` declares -- the same value `_optional_block` will
        write. One with no device is left out: it is not recorded, so it cannot
        be the anchor.
        """
        rates = cls._record_rates(cfg)
        default = (cfg.get("recording") or {}).get("record_fps")
        for assignment in devices.optional or ():
            try:
                descriptor = get_optional_type(assignment.type_id)
            except KeyError:
                continue
            if not str(assignment.device or "").strip():
                continue
            try:
                spec = _optional_source_spec(descriptor.source_type)
            except ValueError:
                continue
            rate = spec.get("record_fps")
            rates[descriptor.fixed_name] = default if rate is None else rate
        return rates

    @classmethod
    def _pick_reference(cls, rates: dict) -> str:
        """The anchor for a rig whose cameras record at *rates*, or ``""``.

        Lowest rate first. The relative calibration has every camera observe one
        marker board at the same instant, so the slowest camera is what the whole
        convergence is bounded by -- anchoring on it is what stops the world
        frame being defined by a stream the slow camera can only be compared to
        across a resampling gap. Ties break by :meth:`_reference_rank`.

        A camera with no resolvable rate (no ``record_fps`` anywhere, not even a
        `recording.record_fps` default) sorts last rather than first: unknown is
        not the same as slow, and treating it as zero would hand the world frame
        to whichever camera is least described.
        """
        if not rates:
            return ""
        unknown = float("inf")
        return min(
            rates,
            key=lambda name: (
                unknown if rates[name] is None else rates[name],
                cls._reference_rank(name),
            ),
        )

    @classmethod
    def _calibration_check(cls, cfg):
        """The ``camera.relative_calibration`` check's params, or None.

        Every shipped template declares this check, so a None here means a
        hand-authored template without one -- in which case there is no rig to
        anchor and nothing to do.
        """
        for check in cfg.get("pre_episode_checks") or []:
            try:
                if check.get("name") != "camera.relative_calibration":
                    continue
            except AttributeError:
                continue
            return check.get("params")
        return None

    @classmethod
    def _apply_reference(cls, cfg, override: str = "") -> None:
        """Anchor the world frame, honouring an operator override when set.

        Empty override -> the automatic pick (:meth:`_apply_auto_reference`).
        A non-empty override names the camera the operator chose; it must be one
        the rig actually records.
        """
        override = str(override or "").strip()
        if override:
            cls._apply_reference_override(cfg, override)
        else:
            cls._apply_auto_reference(cfg)

    @classmethod
    def _apply_reference_override(cls, cfg, override: str) -> None:
        """Anchor the world frame on the camera the operator named.

        A reference naming a source the rig does not record aborts the session
        at construction (RecorderManager._validate_reference_camera), so an
        override that matches no assembled ``video_sources[].name`` is refused
        here, with a message that names the camera, rather than silently ignored
        or written for the recorder to trip over. The chosen name is written to
        both the top-level ``reference_camera`` and the calibration check's, and
        joined to that check's ``cameras`` list -- a reference has to be
        calibrated. An operator who knows their rig may anchor off a camera the
        auto-pick would not choose, on purpose.
        """
        names = video_source_names(cfg)
        if override not in names:
            raise ValueError(
                f"reference camera {override!r} is not one this rig records "
                f"({sorted(names)}) — choose a camera that is present, or clear "
                f"the override to anchor automatically"
            )
        cfg["reference_camera"] = override
        params = cls._calibration_check(cfg)
        if params is None:
            return
        if "reference_camera" in params:
            params["reference_camera"] = override
        cams = params.get("cameras")
        if cams is not None and override not in set(cams):
            cams.append(override)

    @classmethod
    def _auto_reference(cls, cfg) -> str:
        """The camera anchored on when the operator named none, or ``""``.

        The slowest camera the rig records, ties broken by
        :meth:`_reference_rank`. Computed for every rig, so a template's own
        ``reference_camera`` is a starting value rather than a decision -- the
        anchor is a property of the cameras a session actually ends up with, and
        the template cannot know which optional ones those are.

        ``""`` only when the config declares no camera at all, which leaves the
        template's value alone.

        Shared with ``config_form``'s reopen path, so "is this reference an
        operator's override or just what the assembler would have picked" is
        answered by the code that does the picking.
        """
        return cls._pick_reference(cls._record_rates(cfg))

    @classmethod
    def _apply_auto_reference(cls, cfg) -> None:
        """Anchor the world frame on :meth:`_auto_reference`'s pick.

        The chosen name is written to both the top-level ``reference_camera`` and
        the calibration check's own ``reference_camera`` -- the two places
        RecorderManager reads. Joining it to that check's ``cameras`` list is
        left to ``_reconcile_reference_camera``, which does it for every
        reference (see ``sync_reference_camera_to_video_sources``).
        """
        chosen = cls._auto_reference(cfg)
        if not chosen:
            return
        cfg["reference_camera"] = chosen
        params = cls._calibration_check(cfg)
        if params is not None and "reference_camera" in params:
            params["reference_camera"] = chosen

    # ---------------------------------------------------------- reconcile

    @staticmethod
    def _reconcile_reference_camera(cfg) -> None:
        """Never leave reference_camera naming a source that is not there.

        RecorderManager._validate_reference_camera raises when it does, which
        aborts the session at construction. The reference ``_apply_reference``
        just chose already names a real source, so the sync below leaves it in
        place; only a reference that names an absent source (a head camera the
        operator removed) is repaired, and one that still names nothing raises.
        """
        sync_reference_camera_to_video_sources(cfg)
        reference = cfg.get("reference_camera")
        if not reference:
            return
        names = video_source_names(cfg)
        if reference not in names:
            raise ValueError(
                f"reference_camera={reference!r} matches no video_sources[].name "
                f"(known: {sorted(names)})"
            )

    # ------------------------------------------------------------- write

    @staticmethod
    def _atomic_dump(yaml, cfg, out: Path) -> None:
        fd, tmp = tempfile.mkstemp(dir=str(out.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                yaml.dump(cfg, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, out)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise


def session_identity(cfg: Mapping) -> str:
    """The session an applied config is armed for, or ``""``.

    One reader for the key, because two call sites already ask this question of
    a config on disk and a third spelling of ``cfg["meta"]["session_uid"]``
    would be the one that forgets a level is missing.
    """
    meta = cfg.get("meta") if isinstance(cfg, Mapping) else None
    if not isinstance(meta, Mapping):
        return ""
    return str(meta.get("session_uid") or "")


def restamp_session(path: Path, *, session_uid: str, data_dir: str) -> None:
    """Point an applied config at a different session folder. Nothing else.

    Exactly the two keys that carry session identity are rewritten --
    ``meta.session_uid`` and ``recording.data_dir`` -- through a round-trip
    load, so the comments, the key order and every device resolution the
    operator applied come back out byte-identical.

    A rewrite rather than a re-assemble, and the difference matters on the one
    caller there is (`station_ui.day_roll`). Re-running ``assemble`` would
    re-resolve every OAK against whatever device scan the console happens to
    hold at the time, so a config applied last night with each module's native
    aspect could come back this morning on the 16:10 fallback -- a different
    field of view, silently, for a session nobody re-configured. The rig did
    not change; only the folder did.

    Raises rather than writing a file no recorder should be handed: an invalid
    uid is the one input that reaches ``raw/{task}/{session}/`` as a folder
    name, and ``assemble`` refuses it for the same reason.
    """
    if not is_valid_session_uid(session_uid):
        raise ValueError(f"bad session_uid {session_uid!r}")
    if not str(data_dir or "").strip():
        raise ValueError("data_dir is empty")
    yaml = _yaml()
    cfg = yaml.load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, Mapping):
        raise ValueError(f"{path} is not a YAML mapping")
    meta = cfg.get("meta")
    recording = cfg.get("recording")
    if not isinstance(meta, Mapping) or not isinstance(recording, Mapping):
        # Reject configs without meta and recording blocks rather than producing
        # a partially migrated recording config.
        raise ValueError(f"{path} carries no meta/recording block to restamp")
    meta["session_uid"] = session_uid
    recording["data_dir"] = _quoted(data_dir)
    ConfigAssembler._atomic_dump(yaml, cfg, path)

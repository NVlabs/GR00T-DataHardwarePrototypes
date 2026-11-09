# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The in-progress session configuration: the single source of UI truth.

Every page reads and writes this one draft. It is deliberately plain data plus
three functions -- the derived values (`data_dir`, `has_episodes`) are computed
on demand rather than cached, so a folder preview can never disagree with the
fields it was rendered from.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass, field

from station_ui.core.config_utils import build_data_dir
from station_ui.core.optional_devices import OptionalDeviceAssignment
from station_ui.core.session_identity import (
    is_valid_session_uid,
    new_session_uid,
    session_has_episodes,
)
from station_ui.core.setup_presets import POSE_TRACKER_IPHONE, get_preset
from station_ui.errors import FieldError

__all__ = [
    "OptionalDeviceAssignment",
    "SessionDraft",
    "data_dir",
    "device_fingerprint",
    "has_episodes",
    "iphone_generations",
    "recording_day_is_stale",
    "regenerate_uid",
    "start_new_session",
]


def _today() -> str:
    return dt.date.today().isoformat()


def default_head_optional() -> list[OptionalDeviceAssignment]:
    """The head cameras a fresh draft starts with: none.

    Operators add a head camera when their rig has one. A list rather than a
    constant, because `SessionDraft` mutates it -- a shared default would leak
    one draft's head camera into the next one's.
    """
    return []


@dataclass
class SessionDraft:
    """What the operator has configured so far.

    `session_uid` is minted here and only ever changed by `regenerate_uid`.
    It is the last segment of the session folder
    (`output/raw/{task}/{session}/`), so re-minting it mid-session forks the
    episodes across two folders.
    """

    preset_id: str | None = None
    operator_name: str = ""
    task_name: str = ""
    # Only read for a task the station's list does not carry yet: Apply saves it
    # to that list with this as its description. A listed task keeps its own.
    task_description: str = ""
    session_uid: str = field(default_factory=new_session_uid)
    grippers: dict[str, str] = field(default_factory=dict)
    required_cameras: dict[str, str] = field(default_factory=dict)
    optional: list[OptionalDeviceAssignment] = field(
        default_factory=default_head_optional
    )
    # Reference-camera selection changes the coordinate anchor, not hardware
    # identity; it must not invalidate camera verification.
    reference_override: str = ""
    # Per-launch enforcement, independent of device identity and calibration.
    relaxed_checks: list[str] = field(default_factory=list)
    # Blank Vive fields use template defaults. Endpoint and tracker-key changes
    # select different hardware and must affect the device fingerprint.
    vive_endpoint: str = ""
    vive_keys: dict[str, str] = field(default_factory=dict)
    # Recording date and camera family both sit inside the folder name, so they
    # belong to the draft.
    date: str = field(default_factory=_today)
    camera_type: str = "oak"
    # Preserve restored shift-suffixed session folders until the recording-day
    # rollover creates a new session.
    legacy_folder: str = ""


def device_fingerprint(draft: SessionDraft) -> str:
    """A digest of everything about this draft that decides which hardware opens.

    The calibration step is a claim about the rig: each camera is in the slot
    the config names and each gripper's encoder is zeroed. Change a camera on
    step 2 and that claim is about a rig that no longer exists, so it has to
    expire, or the console keeps a green tick over an unverified board.

    Deliberately not the whole draft, and not the generated file either. The
    task name, the operator, the session uid and the date all reach the config
    and none of them changes what gets opened; expiring a fifteen-minute camera
    check because somebody fixed a typo in their own name is how a gate teaches
    people to route around it.

    Sorted, so two drafts that describe one rig agree whatever order the
    operator filled the boxes in. Truncated because this is compared, never
    inverted.
    """
    parts = [f"preset={draft.preset_id or ''}"]
    parts += [
        f"gripper:{arm}={dev}" for arm, dev in sorted((draft.grippers or {}).items())
    ]
    parts += [
        f"camera:{name}={dev}"
        for name, dev in sorted((draft.required_cameras or {}).items())
    ]
    parts += sorted(
        f"optional:{item.type_id}:{item.logical_name}={item.device}"
        for item in (draft.optional or ())
    )
    parts.append(f"vive_endpoint={draft.vive_endpoint or ''}")
    parts += [
        f"vive:{arm}={key}" for arm, key in sorted((draft.vive_keys or {}).items())
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def recording_day_is_stale(draft: SessionDraft) -> bool:
    """Whether this draft belongs to a day that is no longer today.

    `draft.date` is the first ten characters of the session folder, and that is
    how the audit finds a day's sessions, so recording into a stale one files a
    whole day's recordings under yesterday with nothing anywhere to say so.
    `start_new_session` below exists to prevent exactly that, and this is the
    question that triggers it.

    The date is set at draft creation, by `start_new_session`, and by the
    restore in `config_form.hydrate`, and by nothing else, so a station left
    running past midnight carries yesterday's date until somebody acts
    (`session_store` brings the uid and the date back unchanged on purpose:
    re-minting them mid-session would fork the session across two prefixes).

    A restored session in a folder named with a shift label (`legacy_folder`)
    is stale whatever its date: folders are now named without one.

    This is the question, not the fix. `station_ui.day_roll` is the fix: it asks
    this before every request and calls `start_new_session` itself, so the
    operator is not asked every morning. Note what it does not do: re-date the
    draft in place. That would point the console at a different folder under
    the same session id, which is the one outcome nothing downstream can tell
    apart from a session that moved.

    So the two remaining ways to see a stale draft are a roll that was refused
    (a recorder is alive; `day_roll` says why) and one that failed, and the
    banners keyed on this render for exactly those.
    """
    if draft.legacy_folder:
        return True
    return bool(draft.date) and draft.date != _today()


def regenerate_uid(draft: SessionDraft) -> None:
    """Mint a fresh uid, guaranteed to differ from the current one.

    The loop is not paranoia about uuid4: it makes "the folder changed" a
    property the caller can rely on, so the confirm dialog that guards this
    never fires for a no-op rename.
    """
    while True:
        candidate = new_session_uid()
        if candidate != draft.session_uid:
            draft.session_uid = candidate
            return


def start_new_session(draft: SessionDraft) -> None:
    """Leave the current session and open the next one on the same rig.

    Exactly two things are reset, and the choice is the whole point of the call:

    * the uid, which is what makes it a different session at all; and
    * the recording date, restamped to today: it is the first ten characters
      of the session folder, which is how the audit finds a day's sessions, so
      a stale one files the episodes under the wrong day with nothing on screen
      to say so.

    Everything else the operator configured (the task, their own name, the
    grippers, the cameras) belongs to the rig rather than
    to the session, and follows them into the new one. Re-typing six device
    serials between sessions is the cost that would otherwise be paid every
    time, and it buys nothing: the hardware did not move.
    """
    regenerate_uid(draft)
    draft.date = _today()
    draft.legacy_folder = ""


def _operation_mode(draft: SessionDraft) -> str:
    if not draft.preset_id:
        raise FieldError("preset_id", "choose a setup preset first")
    try:
        return get_preset(draft.preset_id).operation_mode
    except KeyError:
        raise FieldError(
            "preset_id", f"unknown setup preset {draft.preset_id!r}"
        ) from None


def _iphone_generation_of(device: str) -> str | None:
    """'iphone16p-01' -> '16'; 'iphone15p-02' -> '15'; anything else -> None."""
    text = str(device or "").lower()
    if "iphone16p" in text:
        return "16"
    if "iphone15p" in text:
        return "15"
    return None


def iphone_generations(draft: SessionDraft) -> dict[str, str]:
    """Detected iPhone generation per tracking-camera side, e.g. {"left": "16"}.

    Only tracking-camera fields are read; a side with no iPhone is just absent.
    """
    generations = {}
    for side in ("left", "right"):
        device = draft.required_cameras.get(f"{side}_tracking_camera")
        generation = _iphone_generation_of(device)
        if generation is not None:
            generations[side] = generation
    return generations


def _pose_tracker(draft: SessionDraft) -> str:
    """The preset's declared wrist pose source, for the folder name.

    No error path of its own: `_operation_mode` runs first on every caller and
    already raises `FieldError` for a missing or unknown preset, so by the time
    this is asked the row resolves. Returning "" rather than raising on the
    impossible case keeps a naming detail from being the thing that 500s a page.

    An iPhone-tracked rig folds its generation in ("iphone16" not "iphone")
    once both sides agree. Mixed generations fall back to the bare token --
    Apply blocks that case, so this only shows up in the live folder preview.
    """
    try:
        tracker = get_preset(draft.preset_id).pose_tracker
    except KeyError:
        return ""
    if tracker != POSE_TRACKER_IPHONE:
        return tracker
    generations = set(iphone_generations(draft).values())
    if len(generations) == 1:
        return f"iphone{generations.pop()}"
    return tracker


def data_dir(draft: SessionDraft) -> str:
    """The session folder this draft would record into.

    `build_data_dir` raises a bare `ValueError` for a malformed date or uid --
    both of which the operator can produce by typing -- so every raise is
    translated to a `FieldError` naming the field. A route that let the
    `ValueError` through would answer 500 to a half-typed date.
    """
    if draft.legacy_folder:
        return draft.legacy_folder
    mode = _operation_mode(draft)
    if not is_valid_session_uid(draft.session_uid):
        # Rendered under an `<output>` the operator cannot type into, holding a
        # value they did not produce. The format spec would tell them how to fix
        # a field they cannot edit; the button that regenerates it is the fix.
        raise FieldError(
            "session_uid",
            "This session id is invalid — press Start a new session.",
        )
    try:
        return build_data_dir(
            draft.date,
            draft.task_name,
            mode,
            right_gripper_num=draft.grippers.get("right"),
            left_gripper_num=draft.grippers.get("left"),
            camera_type=draft.camera_type,
            pose_tracker=_pose_tracker(draft),
            session_uid=draft.session_uid,
        )
    except ValueError as exc:
        # Report an invalid date as a field error; UID and remaining path components
        # have already been validated or normalized.
        raise FieldError("date", f"{draft.date!r} is not YYYY-MM-DD") from exc


def has_episodes(draft: SessionDraft) -> bool:
    """True once this draft's folder holds a recorded or in-flight episode.

    False when the folder cannot even be computed: this is called while the
    operator is still typing, and a half-filled form means nothing has been
    recorded yet rather than an error.
    """
    try:
        folder = data_dir(draft)
    except FieldError:
        return False
    return session_has_episodes(folder)

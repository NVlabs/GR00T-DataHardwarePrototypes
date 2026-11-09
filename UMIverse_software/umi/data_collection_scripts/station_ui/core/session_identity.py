# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session identity: a collision-proof directory name.

The uid itself -- how it is minted, what shape it has and why it is a trailing
lowercase-hex suffix -- lives in ``umi/common/recording/session_uid.py`` and is
re-exported here, because the recorder mints uids too (``session_slot`` rolls a
session when a launch would otherwise record a second configuration into a
folder that already holds episodes). This module owns what the console knows on
top of that: how a folder name is assembled from the operator's fields, and
what counts as a recorded episode inside one.

Only a malformed ``session_uid`` or ``date`` raises — those two are the identity
of the folder; every other input degrades rather than throwing, because these
functions run on operator-typed GUI fields.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from umi.common.recording.session_uid import (
    SESSION_UID_CHARS,
    SESSION_UID_LEN,
    is_valid_session_uid,
    new_session_uid,
    session_uid_from_dir_name,
)

__all__ = [
    "SESSION_UID_CHARS",
    "SESSION_UID_LEN",
    "build_session_dir_name",
    "count_episodes",
    "is_valid_session_uid",
    "new_session_uid",
    "session_has_episodes",
    "session_uid_from_dir_name",
]

# Validate the fixed-width date prefix here; calendar validity is handled separately.
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# ".pending" is the in-flight episode suffix
# (umi/common/recording/constants.py PENDING_SUFFIX).
_EPISODE_RE = re.compile(r"^episode_\d+(\.pending)?$")


def _text(value) -> str:
    """Coerce an operator-supplied field to a stripped string."""
    return "" if value is None else str(value).strip()


def _gripper_token(device, default: str) -> str:
    """'UMI-V4-0001' -> 'V4_0001'; passes through bare numbers."""
    text = _text(device)
    if not text:
        return default
    if text.upper().startswith("UMI-"):
        from station_ui.core.umi_gripper_devices import (
            gripper_path_segment_from_device_name,
        )

        return gripper_path_segment_from_device_name(text)
    return text


def build_session_dir_name(
    *,
    date,
    task_name,
    mode,
    right_gripper=None,
    left_gripper=None,
    pose_tracker=None,
    session_uid: str,
) -> str:
    """The session folder name, ``YYYY-MM-DD`` first and the uid last.

    ``{date}_{grippers}_{pose_tracker}_{task}_{uid}``, e.g.
    ``2026-08-18_umiV4_0001L_umiV4_0002R_vive_plane_assembly_v5_a1b2c3d4e5f6``.

    The two ends are fixed and everything new goes between them. The date holds
    the first ten characters so a date-range query can match on that prefix
    (for example ``LEFT(session_id, 10)``); the uid trails because
    `session_uid_from_dir_name` anchors on the end of the basename and
    `with_new_session_uid` replaces what it finds there.

    ``pose_tracker`` sits with the grippers rather than after the task, so the
    name reads as when / what hardware / what task / which session. It comes
    from `SetupPreset.pose_tracker` -- a declared field, not a sniff of the
    assembled config -- and is omitted when blank, so a caller that has no
    preset in hand (the CLI path) gets a name without a tracker token.

    Raises ValueError — and only ValueError — when ``session_uid`` is not 12
    lowercase hex characters, or when ``date`` is not a ``YYYY-MM-DD`` string.
    An unsuffixed name can collide with another session in the same task
    folder, and a blank date slot would silently shift the name so its
    first 10 characters are no longer the recording date. Neither is optional
    and neither is silently repaired.
    """
    if not is_valid_session_uid(session_uid):
        raise ValueError(
            f"session_uid must be {SESSION_UID_LEN} lowercase hex characters, "
            f"got {session_uid!r}"
        )
    date_text = _text(date)
    if not _DATE_RE.match(date_text):
        raise ValueError(f"date must be a YYYY-MM-DD string, got {date!r}")
    parts = [date_text]
    if _text(mode) == "bimanual":
        parts.append(f"umi{_gripper_token(left_gripper, '1')}L")
        parts.append(f"umi{_gripper_token(right_gripper, '2')}R")
    else:
        parts.append(f"umi{_gripper_token(right_gripper, '1')}R")
    parts.append(_text(pose_tracker))
    parts.append(_text(task_name))
    parts.append(session_uid)
    # Omit blank segments so partially completed forms retain a parseable name.
    return "_".join(part for part in parts if part)


def _episode_dirs(session_path):
    """Yield each episode directory in ``session_path``, lazily.

    One definition of "what counts as an episode", shared by the flag that
    freezes a session folder's name and the count the audit page shows, so the
    two can never disagree. Lazy so ``session_has_episodes`` still stops at the
    first hit; the ``try`` wraps the loop rather than a materialised listing so
    a directory that becomes unreadable mid-iteration is caught too.
    """
    if not session_path:
        return
    try:
        path = Path(os.path.expanduser(str(session_path)))
        if not path.is_dir():
            return
        for child in path.iterdir():
            if _EPISODE_RE.match(child.name) and child.is_dir():
                yield child
    except (TypeError, ValueError, OSError):
        return


def count_episodes(session_path) -> int:
    """How many recorded episodes the session directory holds.

    Same rule as ``session_has_episodes``. Unreadable or non-path arguments
    count as zero, for the reason given there.
    """
    return sum(1 for _ in _episode_dirs(session_path))


def session_has_episodes(session_path) -> bool:
    """True once the session directory holds at least one recorded episode.

    The folder name is frozen from this point: renaming it would fork the
    episodes across two folders. Only ``episode_<digits>`` and in-progress
    ``episode_<digits>.pending`` directories count — ``logs/`` and
    ``config.yaml`` exist before the first recording, while a ``.pending``
    directory means a recorder is holding open file handles right now, the
    single worst moment to rename.

    The argument is expanded first, because a hand-written ``data_dir`` may
    start with ``~``, and ``Path("~/...")`` is a *relative* directory named
    ``~``.

    A non-path, malformed or unreadable argument reads as "nothing recorded
    here yet" rather than raising: this is called while the operator is still
    typing the folder in, before it exists.

    Stops at the first episode rather than counting them: this runs on every
    keystroke in the GUI's session field, and a full-day folder holds hundreds.
    """
    return any(True for _ in _episode_dirs(session_path))

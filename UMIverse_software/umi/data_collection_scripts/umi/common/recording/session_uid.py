# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The session uid: what makes a session folder name unique.

Without a uid, a session folder name is a pure function of date, gripper
tokens and task, so two sessions that agreed on those three would produce the
same folder raw/{task}/{session}/ and interleave their episodes. A random uid
suffix makes the name unique.

The uid is a suffix and lowercase hex on purpose:

* The audit finds a day's sessions by the folder-name prefix, so the date
  has to stay positional at YYYY-MM-DD.
* Gripper tokens look like umiV4_0001L. Hex contains no 'u', 'm', 'i' or 'L',
  so a uid can never be misread as a gripper token.

Every parser is prefix- or search-based, so older names without a uid keep
working.

This module lives under ``umi/common/recording`` rather than beside the Station
Console's ``station_ui/core/session_identity.py`` because both sides mint uids:
the console mints one per session, and ``session_slot`` mints another when a
launch would otherwise record a second configuration into a folder that
already holds episodes. Two definitions of "12 lowercase hex, anchored at the
end" would let the recorder produce a name the console cannot parse. Pure
stdlib on purpose: a recording process must be able to import this without
dragging in a web framework.
"""

from __future__ import annotations

import re
import uuid

SESSION_UID_LEN = 12
SESSION_UID_CHARS = frozenset("0123456789abcdef")

# Match only the final underscore-delimited 12-hex UID; never truncate
# a longer hexadecimal segment.
_UID_RE = re.compile(rf"_([0-9a-f]{{{SESSION_UID_LEN}}})$")


def new_session_uid() -> str:
    """A fresh 12-character lowercase-hex session uid."""
    return uuid.uuid4().hex[:SESSION_UID_LEN]


def is_valid_session_uid(value) -> bool:
    """True only for exactly 12 lowercase hex characters."""
    return (
        isinstance(value, str)
        and len(value) == SESSION_UID_LEN
        and set(value) <= SESSION_UID_CHARS
    )


def session_uid_from_dir_name(name) -> str | None:
    """Return the trailing uid, or None for an older name without one.

    An older session whose task slug happens to end in 12 hex characters is
    indistinguishable from a uid; that is accepted, since the value is only
    ever used as an identity token and never parsed for meaning.
    """
    basename = ("" if name is None else str(name).strip()).rstrip("/").split("/")[-1]
    match = _UID_RE.search(basename)
    return match.group(1) if match else None


def with_new_session_uid(data_dir: str) -> str:
    """``data_dir`` with a freshly minted uid as its last segment.

    Replaces a trailing uid rather than appending to it, so rolling a session
    twice yields ``…_<uid>`` and not ``…_<uid>_<uid>``: the uid is read by
    anchoring on the end of the name, so only the last one would ever be seen
    and the rest would be indistinguishable from part of the task slug.

    Appends when there is no uid to replace, which is the hand-written
    ``recording.data_dir`` a CLI operator passes: those names have no uid and
    must still come out unique.

    Only the final path segment is touched: the regex is end-anchored, so a
    parent directory that happens to end in 12 hex characters is left alone.
    """
    stem = str(data_dir).rstrip("/")
    uid = new_session_uid()
    rolled, replaced = _UID_RE.subn(f"_{uid}", stem, count=1)
    return rolled if replaced else f"{stem}_{uid}"


# A hand-added run number: operators append `_1` / `_2` when they re-record the
# same task on the same day. Digits only, so it can never swallow `_aligned`.
_RUN_SUFFIX_RE = re.compile(r"^\d+$")


def _strip_suffix(name: str, matches) -> str | None:
    """`name` without its last `_`-segment, if that segment satisfies `matches`."""
    head, sep, tail = name.rpartition("_")
    return head if sep and head and matches(tail) else None


def basename_matches_config_task(basename: str, task_name: str) -> bool:
    """Whether this folder is a raw session for ``task_name``.

    Four layouts:

    * ``…_<task_name>_<session_uid>``: what the Station Console writes. The
      12-hex uid goes last (``build_session_dir_name`` in
      station_ui/core/session_identity.py), so two stations recording the
      same task on the same day cannot collide on one folder name.
    * ``…_<task_name>``: older sessions without a uid.
    * ``…_<task_name>_<n>``: either of the above with a hand-added ``_1`` /
      ``_2``, which operators append when re-recording a task the same day.
    * ``…_<task_name>_<session_uid>_<n>``: written by older versions, which
      appended a counter to a folder that already ended in a uid. This is why
      both suffixes have to be peeled together: matching only one at a time
      would hide these sessions from the health check, even though the audit
      page (which lists on the date prefix alone) shows them.

    Derived dirs reuse the same ``meta.task_name`` but append ``_aligned``,
    ``_quarantine``, etc. Nothing here matches those: only a real uid or a
    run number is ever stripped, and neither of those is a word.

    Defined here rather than in ``session_health_check``, its caller, because
    this module already owns the uid grammar the peeling is made of.
    """
    if not task_name:
        return False
    # Match task names while peeling up to two UID/run suffixes in either order.
    # Test each stage so a genuine task ending in _2 still matches itself.
    stems = [basename]
    stem = basename
    for _ in range(2):
        peeled = _strip_suffix(stem, _RUN_SUFFIX_RE.match) or _strip_suffix(
            stem, is_valid_session_uid
        )
        if peeled is None:
            break
        stems.append(peeled)
        stem = peeled
    return any(candidate.endswith("_" + task_name) for candidate in stems)

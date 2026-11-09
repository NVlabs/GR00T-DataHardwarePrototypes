# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The seal: a session folder that will never receive another episode.

A sealed session is one nobody will add to again. That is a weaker claim than
"nothing in this folder will change" (re-alignment and later tools can still
touch a session after recording ends) and a stronger one than "quiet right
now".

The marker is a file in the session folder. It exists so
`session_slot.slot_is_usable` can refuse a folder the station has already
closed, so a relaunch never records into it.

Deliberately imports nothing from `session_slot`, though `session_slot`
imports this: the dependency has to run one way, and `slot_is_usable` is the
consumer. That is why `episode_count` is a parameter rather than something this
module counts for itself: `session_slot.folder_has_episodes` already owns that
question, and a second definition of "what counts as an episode" is exactly the
drift `session_uid.py` warns about.
"""

from __future__ import annotations

import json
import os
import pathlib
from datetime import datetime, timezone

SEAL_FILENAME = "session_sealed.json"

# Restrict seal reasons to known values so misspelled callers fail at the write site.
REASONS = frozenset(
    {
        "operator",  # Data Audit's "Finish and close session"
        "next_session",  # the operator started the next session
        "day_roll",  # automatic midnight roll
        "config_drift",  # config bytes conflicted in the same folder
        "dir_change",  # folder identity changed under the same uid
        "clear_data",  # local data cleared
        "health_check",  # valid, but nothing in this repo writes it
        "derived",  # valid, but nothing in this repo writes it
        "ttl",  # valid, but nothing in this repo writes it
    }
)


def seal_path(folder) -> pathlib.Path:
    """Where the marker lives for *folder*, whether or not it exists.

    Expanded, because a hand-written ``data_dir`` may start with ``~``, and
    `Path("~/...")` is a *relative* directory named ``~``.
    """
    return pathlib.Path(os.path.expanduser(str(folder))) / SEAL_FILENAME


def folder_is_sealed(folder) -> bool:
    """True once *folder* carries a marker, whatever the marker says.

    An existence check, never a parse. A truncated or half-written marker still
    means somebody sealed this folder, and refusing to honour it because the
    JSON did not load would let a crash during the write re-open a closed
    session, the one outcome this whole mechanism exists to prevent.

    A missing, unreadable or non-path argument reads as "not sealed" rather
    than raising: this is asked of folders the operator is still typing.
    """
    try:
        return seal_path(folder).is_file()
    except (TypeError, ValueError, OSError):
        return False


def read_seal(folder) -> dict | None:
    """The marker's contents, or None if there is none or it does not parse.

    Unlike `folder_is_sealed` this does parse, because its callers want the
    reason and the timestamp. The two disagree on a corrupt marker on purpose:
    the folder is still sealed, but the reason is unknown.
    """
    try:
        return json.loads(seal_path(folder).read_text(encoding="utf-8"))
    except (TypeError, ValueError, OSError):
        return None


def write_seal(
    folder,
    *,
    reason: str,
    episode_count: int,
    session_uid: str = "",
    station_id: str = "",
) -> dict | None:
    """Seal *folder*. The payload written, or None if it was refused.

    Three refusals, and each is a rule the callers would otherwise each have to
    remember:

    * **An empty session is never sealed.** A folder with no episodes is not a
      session anyone recorded; sealing it would mark a session that does not
      exist and whose `zarr.json` will never appear.
    * **An existing seal is never overwritten.** The first trigger to fire is
      the truthful one; a later `ttl` must not claim a session an operator
      already closed by hand. Also what makes every caller idempotent.
    * **A folder that is not there is not sealed.** Writing would create it.

    An unknown *reason* raises instead of refusing: that is a programming
    error at the call site, not a state of the world.
    """
    if reason not in REASONS:
        raise ValueError(
            f"unknown seal reason {reason!r}; expected one of {sorted(REASONS)}"
        )
    if int(episode_count) < 1:
        return None
    path = seal_path(folder)
    if not path.parent.is_dir() or path.is_file():
        return None
    payload = {
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "session_uid": str(session_uid or ""),
        "station_id": str(station_id or ""),
        "episode_count": int(episode_count),
    }
    try:
        # Create the marker exclusively so concurrent consoles cannot both seal
        # the same folder.
        with open(path, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
    except OSError:
        return None
    return payload

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sealing a session from the console.

`session_seal.write_seal` owns the marker and every rule about when a seal is
refused. This module is the half that only the station knows: how many episodes
the folder holds and which station this is.

One function, called by every station trigger, so they all write the same
marker the same way. The marker is local (`session_sealed.json` in the session
folder) and is what stops this station recording into a session it declared
finished.
"""

from __future__ import annotations

import logging
import os
import pathlib
import socket

import yaml

from station_ui.core.session_identity import count_episodes
from umi.common.recording.session_seal import write_seal
from umi.common.recording.session_uid import session_uid_from_dir_name

logger = logging.getLogger(__name__)

__all__ = ["armed_data_dir", "seal_session", "seal_if_left_behind"]


def _station_id() -> str:
    return os.environ.get("UMI_STATION_ID") or socket.gethostname()


def seal_session(folder, *, reason: str, session_uid: str = "") -> bool:
    """Seal *folder*. True when a marker was written by this call.

    False covers every refusal `write_seal` makes (an empty session, one
    already sealed, a folder that is not there), so a caller can use the
    return value to decide whether to tell the operator anything, and calling
    it twice is safe.
    """
    path = pathlib.Path(folder)
    written = write_seal(
        path,
        reason=reason,
        episode_count=count_episodes(path),
        session_uid=session_uid,
        station_id=_station_id(),
    )
    return written is not None


def armed_data_dir(config_path) -> str:
    """The folder the armed config names, before anything rewrites it.

    The draft cannot answer this: it already holds the operator's new fields,
    so it names where the next episode goes. Blank when the answer cannot be
    established: this gates a seal, and a guess would close a live session.
    """
    if config_path is None:
        return ""
    try:
        path = pathlib.Path(config_path)
        if not path.is_file():
            return ""
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return str((cfg.get("recording") or {}).get("data_dir") or "")
    except (OSError, TypeError, ValueError, yaml.YAMLError, AttributeError):
        return ""


def seal_if_left_behind(previous: str, current: str, *, reason: str) -> bool:
    """Seal *previous* when the console has stopped pointing at it.

    The one rule every new-session path obeys. Comparing folders catches the
    move however it happened (a re-minted uid, an edited task name, a config
    conflict), so no caller enumerates the ways, and none tests whether the
    folder is already sealed: `write_seal` refuses to overwrite.

    Both sides are expanded first; one comes from a YAML file an operator may
    have written a `~` into, and a spurious difference would seal a live
    session. True only when this call closed a session.
    """
    if not previous or not current:
        return False
    if os.path.abspath(os.path.expanduser(previous)) == os.path.abspath(
        os.path.expanduser(current)
    ):
        return False
    return seal_session(
        previous,
        reason=reason,
        # From the folder name, not the draft: the draft has already left this
        # folder and its uid may name a different session by now.
        session_uid=session_uid_from_dir_name(previous) or "",
    )

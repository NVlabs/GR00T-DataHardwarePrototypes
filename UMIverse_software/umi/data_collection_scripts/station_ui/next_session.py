# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Leaving one session for the next one: the sequence, in one place.

Four things have to happen in one order, and getting the order wrong is how a
console ends up armed for a folder `session_slot.slot_is_usable` refuses:

1. the draft moves onto a fresh uid, stamped today;
2. the config the recorder actually reads is dropped, because it still names
   the session being left;
3. the folder that was left is sealed, after the draft points elsewhere, or
   the console would be refusing the folder it is still on; and
4. the finished-recording card on /record is forgotten, because the recording
   it reports belongs to the session that just ended.

Two callers need this sequence: `POST /config/session/uid`, and the audit page
(a health check that seals the armed session is, in effect, starting the next
one). It lives here so both run the same sequence rather than two copies kept
in step by hand.

Deliberately takes an `app` rather than a `Request`: the audit page's caller
runs on a job's supervisor thread after the child exits, where there is no
request to be had. Nothing here awaits or touches the network, so it is safe on
either.
"""

from __future__ import annotations

import logging

import yaml

from station_ui import recording
from station_ui import seal as seal_state
from station_ui import session as session_state
from station_ui.core.config_assembler import ConfigAssembler, SessionSpec
from station_ui.errors import FieldError

logger = logging.getLogger(__name__)

__all__ = ["disarm", "generated_path", "start_next_session"]


def generated_path(app, draft):
    """Where this preset's config is written, whether or not it exists yet.

    Asked of the assembler rather than rebuilt here, so no caller can point
    somewhere Apply does not write. `output_path` reads only the preset, and the
    placeholder uid keeps `SessionSpec`'s own invariant intact
    rather than depending on it not looking.
    """
    if not draft.preset_id:
        return None
    settings = app.state.settings
    assembler = ConfigAssembler(settings.repo_root, state_dir=settings.state_dir)
    return assembler.output_path(
        SessionSpec(
            preset_id=draft.preset_id,
            operator_name="",
            task_name="",
            session_uid="0" * 12,
            data_dir="",
        )
    )


def disarm(app, draft, left_behind: str) -> None:
    """Drop the recording config written for the session just left.

    The station records from this file, not from a page (`routes/record.py:
    preflight` reads `recording.data_dir` out of it). Left in place it is armed
    with the previous session's folder, so the new session's first episode
    would land under the old id while the page showed a different one, and
    nothing anywhere would say the two disagreed.

    Removed rather than rewritten: assembly is what Apply does, with the device
    and completeness checks that go with it, and a config written behind the
    operator's back is a config nobody chose. The Configuration step unticks
    itself with the file (`app._shell_context` keys the tick on it existing),
    which is exactly what "apply this session before recording it" looks like.

    Only ever removes a file that carries the uid being left, so a config
    belonging to some other session is never touched.
    """
    path = generated_path(app, draft)
    if path is None or not path.is_file():
        return
    try:
        cfg = yaml.safe_load(path.read_text()) or {}
    except (OSError, ValueError, yaml.YAMLError):
        return
    if str((cfg.get("meta") or {}).get("session_uid") or "") != left_behind:
        return
    try:
        path.unlink()
    except OSError:
        # A read-only tree is a station install problem, not something to fail
        # the operator's click over: the page still shows the new session id.
        logger.warning("could not remove the previous session's config %s", path)


def start_next_session(app, draft, *, reason: str) -> str:
    """Move the console onto a fresh session, sealing the one it leaves.

    Returns the folder that was sealed, or "" when nothing was (an empty
    session, one already sealed, or a draft whose folder cannot even be
    computed). The caller needs that to know whether it has anything to tell the
    operator: closing a session for good without saying so would leave the
    operator unaware it happened.

    *reason* is the trigger written into the seal marker.

    The folder is captured before the draft moves. `session.data_dir` is
    derived from the draft, so a capture taken afterwards names the empty
    folder this call just opened, and the seal would land on a session with
    no episodes, which `write_seal` refuses, silently leaving the real one
    open.
    """
    left_behind = draft.session_uid
    try:
        left_folder = session_state.data_dir(draft)
    except FieldError:
        # A draft whose folder cannot even be computed has recorded nothing.
        left_folder = ""

    session_state.start_new_session(draft)
    disarm(app, draft, left_behind)
    # Disarm the generated config before sealing its session. A disarm failure
    # must not leave the recorder pointing at a sealed folder.
    sealed = ""
    if left_folder and seal_state.seal_session(
        left_folder,
        reason=reason,
        session_uid=left_behind,
    ):
        sealed = left_folder
    # Clear the prior session's recorder status when starting a new session.
    # Keep calibration valid for unchanged hardware.
    recording.forget_last_recording(app)
    return sealed

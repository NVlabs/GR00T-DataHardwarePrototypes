# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Who owns the rig.

This console drives one physical set of grippers and cameras. Two browser tabs
must not both be able to rewrite the config or start a recording, and the
losing tab must be *told* rather than silently overwritten.

The rules: the first claim wins, every later token is read-only, and control
only ever moves through `take_control` or an owner's own `release`.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from station_ui.errors import StaleOwner
from station_ui.session import SessionDraft

logger = logging.getLogger(__name__)

__all__ = [
    "RigState",
    "claim",
    "is_owner",
    "release",
    "require_owner",
    "take_control",
]


@dataclass
class RigState:
    """The rig's owner and the draft that owner is editing.

    One instance per process, held on `app.state`. The lock covers ownership
    transitions only: the draft itself has a single writer by construction,
    because every mutating route calls `require_owner` first.
    """

    owner_token: str | None = None
    draft: SessionDraft = field(default_factory=SessionDraft)

    def __post_init__(self) -> None:
        self._lock = threading.RLock()


def _short(token) -> str:
    """As much of a rig token as a log line is allowed to carry.

    The token is the `station_rig_token` cookie value, so a line that prints it
    whole hands a valid session to anyone who can read the file, and the
    station log is a durable file, not a terminal someone happened to be
    watching. Eight characters still tell two browsers apart, which is the
    whole job here.
    """
    return str(token)[:8]


def _validate(token) -> str:
    """A token has to be a non-blank string to be an identity.

    Blank is rejected rather than accepted-and-ignored: `owner_token` is falsy
    when blank, so a blank claim would leave the rig reading as unowned while
    the tab believed it had control, and the next tab would take it.
    """
    if not isinstance(token, str) or not token.strip():
        raise ValueError(f"a rig token must be a non-blank string, got {token!r}")
    return token


def claim(state: RigState, token) -> bool:
    """Take the rig if it is free. True if `token` owns it afterwards.

    Never steals: an already-owned rig answers False and stays with its owner.
    Re-claiming with the owner's own token is idempotent, because a page reload
    re-posts the same token and must not demote the operator who owns it.
    """
    token = _validate(token)
    with state._lock:
        if state.owner_token is None:
            state.owner_token = token
            logger.info("rig claimed by %s", _short(token))
            return True
        return state.owner_token == token


def take_control(state: RigState, token) -> None:
    """Transfer the rig to `token`, displacing whoever held it.

    This is the *only* way control moves to a tab that did not claim it first,
    and it exists so the transfer is a deliberate operator action behind a
    confirm. The displaced tab is not notified out of band -- it finds out on
    its next mutating call, which raises `StaleOwner`.
    """
    token = _validate(token)
    with state._lock:
        previous = state.owner_token
        state.owner_token = token
        if previous is not None and previous != token:
            logger.warning(
                "rig control taken from %s by %s", _short(previous), _short(token)
            )


def release(state: RigState, token) -> None:
    """Give up the rig. Only the owner may.

    A stale token releasing would be a silent takeover with one extra step:
    free the rig, then claim it. An already-free rig is a no-op, so a tab
    closing twice is harmless.
    """
    with state._lock:
        if state.owner_token is None:
            return
        if state.owner_token != token:
            raise StaleOwner("only the current owner can release the rig")
        logger.info("rig released by %s", _short(token))
        state.owner_token = None


def is_owner(state: RigState, token) -> bool:
    """Whether `token` may mutate. Total -- use it to pick read-only markup."""
    with state._lock:
        return state.owner_token is not None and state.owner_token == token


def require_owner(state: RigState, token) -> None:
    """Gate for every mutating call. Raises `StaleOwner` if `token` is not it.

    An unowned rig raises too: a mutating request must not implicitly become
    the owner, or the first stray POST takes the rig with nobody asking.
    """
    if not is_owner(state, token):
        # Offer Take control only when another tab owns the rig. When unowned,
        # visiting a page claims it; there is no separate claim control.
        raise StaleOwner(
            "Another tab controls this rig. Press Take control to make changes."
            if state.owner_token
            else "Reload the page and try again."
        )

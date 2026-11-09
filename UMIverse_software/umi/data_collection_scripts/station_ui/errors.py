# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The errors this app raises at its own boundary.

Every one of these is an *expected* outcome of an operator action, not a bug:
a second tab posting a form, a half-typed date, a job id from a page that was
open when the server restarted. They exist so a route can answer 409/400/404
with something the operator can act on, instead of letting a bare `ValueError`
or `KeyError` from the config backend surface as a 500.
"""

from __future__ import annotations


def stale_choice(noun: str) -> str:
    """What to say when a posted id names nothing this station has.

    One sentence for every such case on the console (a stale tab, a card that
    was disabled after the page loaded, a hand-typed POST), because the
    operator's move is the same in all of them, and several near-identical
    messages would only teach them to read past all of them.

    The id itself is left out: it is a key in this repository's tables, so it
    tells an operator nothing they can act on.
    """
    return f"That {noun} is no longer available — reload the page."


class StationUIError(Exception):
    """Base class, so a route can catch every expected failure at once."""


class StaleOwner(StationUIError):
    """The caller's token is not the token that currently owns the rig.

    Raised for a tab that never claimed, a tab whose claim was taken over, and
    a tab that is still open after the owner released. All three mean the same
    thing to the operator: this page is read-only until you take control.
    """


class FieldError(StationUIError):
    """A form field is unusable, and which field it is matters.

    `station_ui.core.config_utils.build_data_dir` raises a bare `ValueError` on a
    malformed date or uid. Re-raised as a `FieldError` the route can render the
    message next to the offending input rather than as a stack trace.
    """

    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = message


class RecordingInProgress(StationUIError):
    """A rig mutation refused because this station is recording.

    Steps 1-4 are read-only while a recorder is alive. The camera check and the
    device scan each open every camera the recorder holds exclusively, so they
    report the rig broken and send the operator to re-seat a cable that is fine;
    Apply rewrites the config the recorder already read; and "Start a new
    session" mints a uid while the recorder goes on writing into the old folder,
    so the page shows one session while episodes land in another.
    """


class UnknownJob(StationUIError):
    """No job with that id in this registry.

    Jobs live in process memory, so every id from a page that outlived a server
    restart lands here. That is a 404, not a crash.
    """

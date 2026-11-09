# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which part of shutdown the recorder is in, printed by it and read by the console.

Stopping a recording has two phases with very different meanings.

**Teardown** closes cameras, grippers and the preview window. Seconds, normally.
It is also where a recorder gets stuck when a camera has dropped off the bus;
``stop_streaming`` on a dead device is the case ``station_ui``'s stall timer
exists for.

**Drain** finishes queued alignments. Minutes, legitimately, and *silent*:
the aligner prints per episode, so a perfectly healthy drain looks exactly like
a hang.

The console has to tell them apart. "Queued work is finishing, this is normal"
is right during the drain, but during teardown a long wait is not normal and
the operator needs to know a camera has gone.

Inferring the phase from incidental log lines would couple the console to
wording nobody thinks of as an interface. Instead the recorder says where it is,
in one line per phase, and both sides import the spelling from here.

Plain text on purpose: the recorder colours most of its output, and an ANSI
escape inside the marker would break a substring match on the captured line.
"""

from __future__ import annotations

__all__ = [
    "DRAIN",
    "DONE",
    "PHASES",
    "TAG",
    "TEARDOWN",
    "marker",
    "phase_from_lines",
]

TAG = "[shutdown]"

# Closing hardware: cameras, grippers, the preview window.
TEARDOWN = "teardown"
# Finishing queued alignments. The hardware is already released.
DRAIN = "drain"
# The `finally` ran to the end. Anything still alive is a straggler.
DONE = "done"

PHASES = (TEARDOWN, DRAIN, DONE)

_TEXT = {
    TEARDOWN: "releasing cameras and grippers",
    DRAIN: "finishing queued alignments",
    DONE: "shutdown complete",
}


def marker(phase: str) -> str:
    """The line the recorder prints on entering `phase`."""
    return f"{TAG} {_TEXT[phase]}"


def phase_from_lines(lines) -> str | None:
    """The latest phase named in `lines`, or None if none is.

    Scans backwards and stops at the first hit: phases only move forward, so the
    last marker printed is the current one. None means the recorder is not in
    (or has not reached) a marked phase -- callers fall back to saying nothing
    specific rather than guessing, which is what makes this safe to add to a
    page that also renders recorders started by an older build.
    """
    for line in reversed(list(lines)):
        for phase, text in _TEXT.items():
            if text in line:
                return phase
    return None

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The recorder's SIGTERM handler must disarm before it raises.

Pressing `q` to quit the recording window begins the shutdown drain. A second
SIGTERM arriving after that (the web Stop button) must not land a
KeyboardInterrupt inside the drain and abort it, which would lose alignments
and leave resources held. So SIGTERM is disarmed the moment shutdown begins: a
second signal is ignored and the drain runs to completion (SIGKILL escalation
in station_ui/jobs.py is still the hard stop).
"""

from __future__ import annotations

import signal

import pytest

from umi.common.utility import signal_handling


def test_the_sigterm_handler_disarms_itself_before_raising(monkeypatch):
    installed = []
    # Patch the module's own `signal.signal` so the test process's real SIGTERM
    # disposition is never touched -- the test only asserts what the handler
    # asks for.
    monkeypatch.setattr(
        signal, "signal", lambda sig, handler: installed.append((sig, handler))
    )

    with pytest.raises(KeyboardInterrupt):
        signal_handling._stop_on_sigterm(signal.SIGTERM, None)

    assert (signal.SIGTERM, signal.SIG_IGN) in installed, (
        "the first SIGTERM did not disarm the handler, so a second Stop would "
        "abort the alignment drain"
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SIGTERM handling shared by long-running station scripts."""

import signal


def _stop_on_sigterm(signum, frame):
    """Turn the first SIGTERM into the interrupt the shutdown path handles, then
    get out of the way.

    Disarm before raising: a second SIGTERM must not raise again, or it lands a
    KeyboardInterrupt inside the `finally` that drains queued alignments and aborts
    the very drain this handler exists to protect. The web Stop button pressed
    after a `q`-quit, or an impatient double-press, is exactly that second
    signal. The `finally` disarms too, for the `q`-then-Stop path that never
    entered this handler; SIGKILL escalation in station_ui/jobs.py is still the
    hard stop if the drain truly hangs.
    """
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise KeyboardInterrupt


def install_stop_on_sigterm() -> None:
    """Make ``SIGTERM`` a *safe* stop.

    Otherwise the only ways to end a recording are the cv2 ``q`` key and SIGINT,
    and a web UI can use neither: there is no OpenCV window on the browser's
    machine. Python's default SIGTERM handling terminates without unwinding, so
    ``kill <pid>`` would skip the ``finally`` block at the end of
    ``_run_session`` (the one that calls ``shutdown_background_services()``)
    and silently discard every queued alignment.

    Raising ``KeyboardInterrupt`` routes SIGTERM into the existing
    ``except KeyboardInterrupt`` / ``finally`` pair instead of adding a second
    shutdown path that would drift out of step with it.
    """
    signal.signal(signal.SIGTERM, _stop_on_sigterm)

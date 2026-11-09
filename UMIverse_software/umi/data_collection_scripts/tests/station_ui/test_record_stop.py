# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stop must never be a dead end, and must never claim to have worked.

Stop sends SIGTERM and schedules SIGKILL much later, to protect the drain.
Without a way to shorten that, a wedged recorder would be a sixteen-minute dead
end, and pressing Stop again would only stack timers. These tests pin the way
out.
"""

from __future__ import annotations

import pathlib
import re
import time

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

# Import the shared recording fixtures and _gate_open explicitly. Autouse
# fixtures are not inherited merely by importing another module's helpers.
from test_record_page import (  # noqa: E402,F401
    _arm,
    _gate_open,
    _job_of,
    _start,
    _wait_for_line,
    client,
    launcher,
    ready,
    repo,
)

from station_ui import recording  # noqa: E402
from station_ui.routes import record  # noqa: E402
from umi.common.recording import shutdown_progress  # noqa: E402


def _button(html: str, element_id: str) -> str | None:
    match = re.search(rf"<button[^>]*id=\"{element_id}\"[^>]*>.*?</button>", html, re.S)
    return match.group(0) if match else None


# Use a child that ignores SIGTERM so Force stop tests exercise escalation
# rather than the normal immediate-exit path.
WEDGED = """
    import signal, time
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print("ready", flush=True)
    time.sleep(45)
"""


@pytest.fixture
def stopping(ready, launcher):  # noqa: F811 - imported pytest fixtures
    """A client whose recorder ignored Stop, with Stop already pressed.

    The wait is not politeness. `_start` returns as soon as the pid exists, so a
    Stop sent immediately after it can deliver SIGTERM before the child has run
    `signal.signal(SIG_IGN)` -- the stub then dies of the default disposition
    (exit -15) and the test measures a healthy stop while claiming to measure a
    wedged one.
    """
    launcher.body = WEDGED
    _start(ready)
    _wait_for_line(_job_of(ready), "ready")
    ready.post("/record/stop", follow_redirects=False)
    return ready


def test_force_stop_is_rendered_but_disabled_before_the_countdown(stopping):
    """Rendered, not hidden. A control that silently appears later is one the
    operator never learns is coming -- and an operator who thinks there is no
    way out unplugs USB or power-cycles the rig, which is the thing this whole
    design exists to prevent.

    Structural, not a substring check: a control greyed only by a stylesheet
    would pass `"disabled" in html` while still being clickable.
    """
    button = _button(stopping.get("/record").text, "force-stop")
    assert button is not None, "no Force stop control at all -- the dead end"
    assert "disabled" in button, "Force stop was live immediately"
    assert re.search(r"\(\d+s\)", button), "no countdown in the label"


def test_force_stop_becomes_available_after_force_after_s(stopping):
    state = recording.state_for(stopping.app)
    state.stopping_since -= record.FORCE_AFTER_S + 1

    button = _button(stopping.get("/record").text, "force-stop")

    assert "disabled" not in button, "Force stop never became available"
    assert "(" not in button, "the countdown outlived the countdown"


def test_the_stopping_panel_says_how_long_since_the_child_printed(stopping):
    """The part that makes 15s safe. The aligner prints per episode and its
    own timeout is 300s, so a healthy drain is silent for minutes and looks
    exactly like a hang. A countdown with nothing beside it is a machine for
    killing healthy alignments."""
    assert "since the last output" in stopping.get("/record").text


def test_silence_is_only_called_normal_once_the_drain_has_started(stopping):
    """The reassurance is shown only during the drain.

    "Queued alignments are finishing, this is normal" shown from the instant
    Stop is pressed would cover the whole teardown, minutes before any drain
    begins -- and teardown is exactly where a recorder hangs when a camera has
    dropped off the bus. The console must not reassure the operator then.

    The recorder says which phase it is in
    (`umi.common.recording.shutdown_progress`), so the promise waits for the
    phase it is true of.
    """
    body = stopping.get("/record").text
    assert "can be silent for" not in body, (
        "the drain's reassurance shown before the drain, where a hang means a "
        "camera has gone"
    )

    job = record.workflow_state(stopping.app).jobs.get(
        record.workflow_state(stopping.app).record_job_id
    )
    job._append(shutdown_progress.marker(shutdown_progress.DRAIN))  # noqa: SLF001

    assert "can be silent for" in stopping.get("/record").text


def test_the_panel_is_absent_until_stop_is_pressed(ready, launcher):  # noqa: F811
    launcher.body = WEDGED
    _start(ready)
    _wait_for_line(_job_of(ready), "ready")
    assert _button(ready.get("/record").text, "force-stop") is None


def test_force_stop_is_not_gated(stopping):
    """It is the recording lock's only escape hatch -- a wedged rig blocks the
    whole bench."""
    response = stopping.post("/record/force-stop", follow_redirects=False)
    assert response.status_code in (200, 303), response.text


def test_force_stop_kills_the_recorder(stopping):
    state = recording.state_for(stopping.app)
    job = state.jobs.get(state.record_job_id)

    stopping.post("/record/force-stop", follow_redirects=False)

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not job.finished:
        time.sleep(0.02)
    assert job.finished, "Force stop left the recorder running"


def test_force_stop_clears_the_pid_file(stopping):
    state_dir = pathlib.Path(stopping.app.state.settings.state_dir)
    assert (state_dir / recording.PID_FILE).exists()

    stopping.post("/record/force-stop", follow_redirects=False)

    assert not (state_dir / recording.PID_FILE).exists()


def test_pressing_stop_twice_does_not_restart_the_countdown(stopping):
    """Otherwise mashing Stop pushes Force stop further away every time -- the
    operator's impatience feeding the thing making them impatient."""
    state = recording.state_for(stopping.app)
    first = state.stopping_since

    time.sleep(0.05)
    stopping.post("/record/stop", follow_redirects=False)

    assert state.stopping_since == first, "the countdown restarted"


# ---- The countdown refreshes without a page reload ---------------------------


def _stopping_block(html: str) -> str | None:
    """The self-refreshing stop panel, as rendered."""
    match = re.search(r'<div class="stopping"[^>]*>', html)
    return match.group(0) if match else None


def test_the_stopping_panel_refreshes_itself(stopping):
    """The countdown refreshes itself.

    Every number in this panel is server-rendered, and nothing else re-renders
    them: the SSE stream only appends log lines, and the drain it reports on is
    deliberately silent -- which is the whole reason the age line exists. So the
    panel has to ask for itself.
    """
    block = _stopping_block(stopping.get("/record").text)

    assert block is not None, "no stopping panel to refresh"
    assert 'hx-trigger="every 1s"' in block, f"the countdown cannot advance: {block}"
    assert 'hx-get="/record"' in block
    assert 'hx-select="#stopping"' in block, "the poll would swap the whole page in"


def test_the_poll_is_scoped_to_the_panel_and_never_the_whole_card(stopping):
    """The panel polls itself, not the whole card -- a change that would look
    like a simplification.

    htmx executes <script> in swapped content, and #record-status carries the
    EventSource for the recorder's log -- rendered whenever the job is unfinished,
    which is exactly this window. Polling the card would close and reopen that
    stream once a second, and its `onopen` clears the log as it goes.
    """
    body = stopping.get("/record").text
    card = re.search(r'<div class="section" id="record-status"[^>]*>', body)

    assert card is not None, "the status card is gone"
    assert "hx-trigger" not in card.group(0), (
        f"the card polls itself and will thrash the log stream: {card.group(0)}"
    )
    # And the stream really is live in this window, which is what makes it matter.
    assert "EventSource" in body


def test_the_countdown_actually_advances_between_renders(stopping):
    """The point of the poll, measured rather than assumed: two renders a moment
    apart must not report the same second."""
    first = _remaining(stopping.get("/record").text)
    assert first is not None, "no countdown rendered"
    time.sleep(1.1)
    second = _remaining(stopping.get("/record").text)

    assert second is not None
    assert second < first, f"the countdown is frozen at {first}s"


def _remaining(html: str) -> int | None:
    match = re.search(r"Force stop\s*\((\d+)s\)", html)
    return int(match.group(1)) if match else None

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fold: what the worker saw, turned into what the page shows.

State is derived from the child's output on every render, never stored --
the same rule `station_ui/devices.statuses` follows, and for the same reason:
a row still reading green from a scan five minutes ago is the worst thing
this page could show.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

from station_ui import calibration as cal

ROLES = ["left_umi", "left_trigger", "right_umi"]


def _line(**event):
    return cal.WATCH_PREFIX + json.dumps(event)


def test_a_role_with_no_event_is_waiting():
    folded = cal.fold_events(ROLES, [])
    assert set(folded.values()) == {cal.WAITING}


def test_a_role_that_decoded_its_own_side_is_matched():
    events = cal.parse_events(
        [_line(event="saw", role="left_umi", payload="UMI_VERIFY_LEFT")]
    )
    folded = cal.fold_events(ROLES, events)
    assert folded["left_umi"] == cal.MATCHED
    assert folded["left_trigger"] == cal.WAITING


def test_a_role_that_decoded_another_side_is_mismatched():
    """A cross-wired camera is named as such.

    Seeing the right side's code through the camera the config calls left_umi
    names the actual fault, rather than a timeout that says only "failed".
    """
    events = cal.parse_events(
        [_line(event="saw", role="left_umi", payload="UMI_VERIFY_RIGHT")]
    )
    assert cal.fold_events(ROLES, events)["left_umi"] == cal.MISMATCHED


def test_a_later_decode_corrects_an_earlier_one():
    """Re-aiming the camera is the retry; there is no retry button."""
    events = cal.parse_events(
        [
            _line(event="saw", role="left_umi", payload="UMI_VERIFY_RIGHT"),
            _line(event="saw", role="left_umi", payload="UMI_VERIFY_LEFT"),
        ]
    )
    assert cal.fold_events(ROLES, events)["left_umi"] == cal.MATCHED


def test_a_verified_camera_latches_against_a_later_stray_code():
    """Verification latches. Once a camera has decoded its own code it is
    verified and stays verified -- a later glimpse of another side's code (a
    reflection, or the board moving past it while the operator checks the next
    side) must not un-verify what was already confirmed. Only the un-verified
    states (WAITING / MISMATCHED) still move."""
    events = cal.parse_events(
        [
            _line(event="saw", role="left_umi", payload="UMI_VERIFY_LEFT"),  # verified
            _line(event="saw", role="left_umi", payload="UMI_VERIFY_RIGHT"),  # stray
        ]
    )
    assert cal.fold_events(ROLES, events)["left_umi"] == cal.MATCHED


def test_a_verified_camera_does_not_regress_to_open_failed_either():
    """The latch holds against a late open-failure too: a camera the operator
    already confirmed is not un-verified by a teardown-time driver error."""
    events = cal.parse_events(
        [
            _line(event="saw", role="left_umi", payload="UMI_VERIFY_LEFT"),
            _line(event="open_failed", role="left_umi", error="RuntimeError: closing"),
        ]
    )
    assert cal.fold_events(ROLES, events)["left_umi"] == cal.MATCHED


def test_a_device_that_would_not_open_is_failed():
    events = cal.parse_events(
        [_line(event="open_failed", role="right_umi", error="RuntimeError: locked")]
    )
    assert cal.fold_events(ROLES, events)["right_umi"] == cal.OPEN_FAILED


def test_non_event_output_is_ignored():
    """The child prints driver warnings too, and one of them will contain a
    brace sooner or later."""
    assert cal.parse_events(["[watch] opening OAK-FFC-LEFT...", "{not json}"]) == []


def test_the_current_side_is_the_first_one_still_waiting():
    folded = {
        "left_umi": cal.MATCHED,
        "left_trigger": cal.MATCHED,
        "right_umi": cal.WAITING,
    }
    assert cal.current_side(folded) == "right"


def test_a_side_whose_cameras_all_matched_is_left_behind():
    folded = {"left_umi": cal.MATCHED, "right_umi": cal.MATCHED}
    assert cal.current_side(folded) is None


def test_a_failed_camera_does_not_pin_the_page_to_its_side():
    """A dead device is red and stays red. Blocking the other sides behind it
    would strand an operator who can still check the rest of the rig."""
    folded = {"left_umi": cal.OPEN_FAILED, "right_umi": cal.WAITING}
    assert cal.current_side(folded) == "right"


def test_a_mismatched_camera_keeps_its_side_current():
    """Unlike a dead device, this one is fixable where the operator stands."""
    folded = {"left_umi": cal.MISMATCHED, "right_umi": cal.WAITING}
    assert cal.current_side(folded) == "left"


@pytest.mark.parametrize(
    "role,side",
    [
        ("left_umi", "left"),
        ("left_tracking", "left"),
        ("right_trigger", "right"),
        ("head", "head"),
        ("head_umi", "head"),
    ],
)
def test_role_sides(role, side):
    assert cal.side_of_role(role) == side
    assert cal.payload_for_side(side) == f"UMI_VERIFY_{side.upper()}"


def test_a_recorded_bottom_camera_is_not_checked():
    """The bottom OAK is recorded but not checked: a config that records one
    lists the same roles as one without it, and the walk has no table side to
    end on, so no marker is drawn for it."""
    sources = [
        {"type": "oak_camera", "name": "left_umi_camera"},
        {"type": "oak_camera", "name": "head_oak_camera"},
        {"type": "oak_camera", "name": "bottom_oak_camera"},
    ]
    config = {"meta": {"use_head_mount": True}, "video_sources": sources}

    assert cal.verification_roles(config) == ["left_umi", "head_umi"]
    assert cal.SIDES == ("left", "right", "head")


from fastapi.testclient import TestClient  # noqa: E402
from test_calibration_page import apply_config  # noqa: E402

from station_ui.app import create_app  # noqa: E402


def _client():
    app = create_app()
    client = TestClient(app)
    client.get("/")
    client.post("/setup/preset", data={"preset_id": "bimanual_v4"})
    # The check reads the config Apply generated, never the template -- the
    # template's placeholder serials are on nobody's bench. So an operator
    # reaching this page has applied one, and the fixture has to as well.
    apply_config(app)
    return app, client


@pytest.mark.parametrize("side", ["left", "right", "head"])
def test_each_side_renders_its_own_qr_as_svg(side):
    """One code per side is what makes the check mean anything: a single shared
    code would be decoded by every camera and would pass a left/right swap."""
    _app, client = _client()
    response = client.get(f"/calibration/check/qr/{side}.svg")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert "<svg" in response.text


@pytest.mark.parametrize("side", ["left", "right", "head"])
def test_the_qr_carries_its_own_white_field_rather_than_a_transparent_one(side):
    """A camera has to decode this, not just see it.

    On a transparent background the code would render black on this console's
    #2D2D2D surface, which is hard to see. A decoder needs the light modules to
    be light, so a transparent code over a dark card is one the camera can fail
    to read at all, and reading it off the screen is the entire camera check.

    The white field also makes the quiet zone actually quiet, which is what an
    ArUco detector needs to find the marker's black border as a quad at all.
    Asserted on the SVG rather than on CSS: the image is served as its own
    document and is not styled by the page it is embedded in.

    Written against the paint order rather than the element names, because the
    property is "the light field is behind the dark code" and that has to hold
    whichever marker is drawn -- a QR path over a rect or ArUco rects over a
    rect -- and neither spelling is the point.
    """
    _app, client = _client()
    body = client.get(f"/calibration/check/qr/{side}.svg").text

    assert 'fill="white"' in body, "the code has no light field to decode against"
    assert body.index('fill="white"') < body.index('fill="black"'), (
        "the field is painted over the code"
    )


def test_the_three_sides_render_three_different_codes():
    _app, client = _client()
    bodies = {
        side: client.get(f"/calibration/check/qr/{side}.svg").text
        for side in ("left", "right", "head")
    }
    assert len(set(bodies.values())) == 3


def test_an_unknown_side_is_not_rendered():
    """The side names the payload; anything else would encode a code no camera
    is looking for and read as a silently failing check."""
    _app, client = _client()
    assert client.get("/calibration/check/qr/middle.svg").status_code == 404


def test_marker_rendering_does_not_import_cv2_into_the_web_process():
    """Drawing the marker must not pull cv2 into the web process."""
    import station_ui.routes.calibration as route_module

    source = pathlib.Path(route_module.__file__).read_text()
    assert "import cv2" not in source


# ------------------------------------------------------------------ the panel


def test_starting_the_check_launches_one_watch_job_for_the_whole_rig(monkeypatch):
    """Six cameras, one process, rather than six subprocesses back to back with
    a click between each."""
    launched = []

    def fake_start(registry, argv, **kwargs):
        launched.append(argv)
        return _FakeJob()

    _app, client = _client()
    monkeypatch.setattr("station_ui.routes.calibration.jobs.start", fake_start)
    response = client.post("/calibration/check/start", follow_redirects=False)

    assert response.headers["location"] == "/calibration#step-device_verify"
    assert len(launched) == 1
    assert "--watch" in launched[0]
    assert "--role" not in launched[0]


def test_the_panel_shows_the_current_side_and_its_cameras():
    app, client = _client()
    _seed_check(app, [("left_umi", "UMI_VERIFY_LEFT")])

    body = client.get("/calibration/check/panel").text

    assert "/calibration/check/qr/left.svg" in body
    assert "Left UMI (wrist) camera" in body


def test_the_panel_advances_to_the_next_side_when_a_side_is_clear():
    """No 'Next' button: finishing a side is the signal to move on."""
    app, client = _client()
    _seed_check(
        app,
        [
            ("left_umi", "UMI_VERIFY_LEFT"),
            ("left_trigger", "UMI_VERIFY_LEFT"),
            ("left_tracking", "UMI_VERIFY_LEFT"),
        ],
    )

    body = client.get("/calibration/check/panel").text
    assert "/calibration/check/qr/right.svg" in body


def test_a_cross_wired_camera_is_named_as_such():
    app, client = _client()
    _seed_check(app, [("left_umi", "UMI_VERIFY_RIGHT")])

    body = client.get("/calibration/check/panel").text
    assert "cross-wired" in body.lower() or "wrong side" in body.lower()


def test_a_read_only_browser_cannot_start_the_check():
    """Starting the check opens every camera on the rig."""
    app, _client_owner = _client()
    other = TestClient(app)
    response = other.post("/calibration/check/start", follow_redirects=False)
    assert response.status_code == 409


def test_a_check_that_cannot_even_be_spawned_reads_failed_not_pending(monkeypatch):
    """`uv` missing from PATH, or the script moved.

    A child that never ran prints no events, so the fold sees six waiting
    cameras -- and "waiting" folded up to a step badge reads *Not started*,
    which tells an operator nothing is wrong when the check never happened.
    """
    monkeypatch.setattr(cal, "LAUNCHER", ("/nonexistent/uv", "run"))
    app, client = _client()

    started = client.post("/calibration/check/start", follow_redirects=False)
    assert started.status_code == 303

    # `jobs.start` catches the OSError itself, so the job is already terminal.
    assert cal.step_state(app, cal.STEP_VERIFY) == cal.FAILED
    body = client.get("/calibration").text
    assert "✕ Failed" in body
    assert "could not start" in body, "the operator is not told what went wrong"


def test_stopping_the_check_is_not_a_failure(monkeypatch):
    """An operator who has seen enough presses Stop. That is a decision, not a
    fault, and must not paint the step red."""
    app, client = _client()
    job = _seed_check(app, [("left_umi", "UMI_VERIFY_LEFT")])
    job.state = "cancelled"
    job.finished = True

    assert cal.step_state(app, cal.STEP_VERIFY) == cal.PENDING


def test_a_second_check_cannot_be_started_while_one_is_running():
    """An OAK open is exclusive. A second watcher would fight the first over the
    same board and both would report it unavailable -- which reads on the page
    as a rig that has come unplugged."""
    app, client = _client()
    _seed_check(app, [])

    response = client.post("/calibration/check/start", follow_redirects=False)

    assert response.status_code == 409
    assert len(cal.job_registry(app).all()) == 1, "a second watcher was launched"


def test_stopping_the_check_cancels_the_watcher(monkeypatch):
    """Forgetting the job id is not stopping it: every camera the watcher holds
    open stays claimed until the process itself goes away."""
    app, client = _client()
    job = _seed_check(app, [])
    cancelled = []
    monkeypatch.setattr(
        "station_ui.routes.calibration.jobs.cancel",
        lambda registry, job_id, **kwargs: cancelled.append(job_id),
    )

    response = client.post("/calibration/check/stop", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/calibration#step-device_verify"
    assert cancelled == [job.id]


def test_the_stream_only_ever_serves_this_pages_own_check():
    """The registry also holds gripper and recording jobs. An endpoint under
    /calibration/check/ that would happily stream a recording is an accident
    waiting to be linked to."""
    from station_ui import jobs

    app, client = _client()
    registry = cal.job_registry(app)
    recording = jobs.start(
        registry, [sys.executable, "-c", "print('episode 4')"], cwd="."
    )
    watcher = jobs.start(registry, [sys.executable, "-c", "print('watching')"], cwd=".")
    cal.check_state(app).job_id = watcher.id

    assert client.get(f"/calibration/check/stream/{recording.id}").status_code == 404
    assert client.get("/calibration/check/stream/deadbeef").status_code == 404

    response = client.get(f"/calibration/check/stream/{watcher.id}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: watching" in response.text
    # Every stream ends with this, so the page knows to reload rather than sit
    # on a socket that will never say anything again.
    assert "event: end" in response.text


# ------------------------------------------- the rest of the app, same cameras


@pytest.fixture
def harmless_scan(monkeypatch):
    """The real scan child enumerates and opens every device on this machine.

    Same rule the rest of the suite follows: `devices.scan_argv` is the seam,
    and no test in this file may reach hardware other people are using.
    """
    from station_ui import devices as devices_core

    monkeypatch.setattr(
        devices_core,
        "scan_argv",
        lambda: [sys.executable, "-c", "print('[scan] fake')"],
    )


def test_a_device_scan_started_under_a_running_check_releases_the_cameras(
    harmless_scan, monkeypatch
):
    """The check holds every camera open (an OAK open is exclusive), so a scan
    cannot simply run under it.

    Refusing would be a dead end: the operator who changed a device on
    Configuration and came here to re-test has no Stop button on that page. So
    starting a scan stops the check first and waits for the watcher to release
    the cameras, then proceeds.
    """
    from station_ui import devices as device_state

    app, client = _client()
    check = _seed_check(app, [])

    # Simulate watcher exit when cancellation is requested so release tests
    # can finish without a real camera worker.
    cancelled = []

    def fake_cancel(registry, job_id, **kwargs):
        cancelled.append(job_id)
        registry.get(job_id).finished = True

    monkeypatch.setattr("station_ui.calibration.jobs.cancel", fake_cancel)

    response = client.post("/config/devices/scan", follow_redirects=False)

    assert response.status_code == 303, "the scan was refused instead of releasing"
    assert cancelled == [check.id], "the running check was not stopped"
    assert not cal.check_running(app), "the check is still holding the cameras"
    assert device_state.state_for(app).job_id is not None, "no scan actually started"


def test_a_recording_cannot_start_while_the_check_holds_every_camera():
    """Same collision, worse consequence: the recorder fails to open its own
    cameras and the operator has a dead session to explain."""
    from station_ui.routes import record as record_routes

    app, client = _client()
    _seed_check(app, [])

    response = client.post(
        "/record/start", data={"confirmed": "yes"}, follow_redirects=False
    )

    assert response.status_code == 400
    assert "Stop it there first" in response.text
    assert record_routes.workflow_state(app).record_job_id is None


def test_a_finished_check_holds_nothing_and_blocks_nothing(harmless_scan):
    """The guard is about a live watcher. A check the operator already stopped
    has released every camera, and refusing after it would make the two pages
    permanently unusable together."""
    from station_ui import devices as device_state

    app, client = _client()
    job = _seed_check(app, [])
    job.state = "done"
    job.finished = True

    assert cal.check_running(app) is False

    response = client.post("/config/devices/scan", follow_redirects=False)
    assert response.status_code == 303
    assert device_state.state_for(app).job_id is not None


def test_an_app_that_has_never_run_a_check_is_not_holding_cameras():
    """Asked on every scan and every recording, including the first one."""
    app, _client_owner = _client()
    assert cal.check_running(app) is False


class _FakeJob:
    """A job that never ran, for asserting what would have been launched."""

    id = "fake-job"
    state = "running"
    finished = False
    exit_code = None
    error = None
    argv = ["fake"]
    cwd = "."

    def lines(self):
        return []


def _feed(app, lines):
    """Put a live check job in the registry printing exactly *lines*.

    The sibling of `_seed_check`, which only ever emits `saw` events. This one
    takes the raw output, because what is under test here is the driver's words
    alongside the events rather than the fold over them.
    """
    job = _FakeJob()
    job.lines = lambda: list(lines)
    cal.job_registry(app).add(job)
    cal.check_state(app).job_id = job.id
    return job


def _seed_check(app, seen):
    """Put a live check job in the registry with canned events.

    Goes through the real registry rather than a stub attribute so the fold
    reads exactly what a live job would give it. The job is left running
    because that is the only state in which the panel offers a code to aim at:
    a finished watcher decodes nothing, and a marker drawn over a dead process
    would be an instruction the operator cannot carry out.
    """
    job = _FakeJob()
    job._lines = [
        cal.WATCH_PREFIX
        + json.dumps({"event": "saw", "role": role, "payload": payload})
        for role, payload in seen
    ]
    job.lines = lambda: job._lines
    cal.job_registry(app).add(job)
    cal.check_state(app).job_id = job.id
    return job


# Show watcher diagnostics for successful and failed checks. Per-camera
# status identifies the device; driver output explains the cause.


def test_the_watchers_own_events_are_not_part_of_the_log():
    """One event per decode at 30fps. Left in, they bury the driver messages
    that are the reason anyone opened this."""
    lines = [
        "[verify] opening left_umi_camera...",
        cal.WATCH_PREFIX + '{"role": "left_umi", "event": "saw"}',
        "[verify] left_umi_camera: no such device",
    ]

    assert cal.watcher_log(lines) == [
        "[verify] opening left_umi_camera...",
        "[verify] left_umi_camera: no such device",
    ]


def test_a_camera_that_would_not_open_shows_the_drivers_reason(monkeypatch):
    """The check itself is fine and one camera would not open: still shown."""
    app, client = _client()
    roles = cal.verification_roles(cal.recording_config(app))
    reason = "could not open /dev/video0: Permission denied"
    _feed(
        app,
        [
            f"[verify] {reason}",
            cal.WATCH_PREFIX + json.dumps({"role": roles[0], "event": "open_failed"}),
        ],
    )

    body = client.get("/calibration").text

    assert reason in body, "the operator is told which camera, never why"
    assert "<details" in body
    # Expanded: something is wrong, so the output is the answer.
    assert 'class="job-details" open>' in body


def test_a_clean_check_keeps_its_output_collapsed(monkeypatch):
    app, client = _client()
    roles = cal.verification_roles(cal.recording_config(app))
    _feed(
        app,
        ["[verify] opening every camera..."]
        + [
            cal.WATCH_PREFIX
            + json.dumps(
                {
                    "role": role,
                    "event": "saw",
                    "payload": cal.payload_for_side(cal.side_of_role(role)),
                }
            )
            for role in roles
        ],
    )

    body = client.get("/calibration").text

    assert "opening every camera" in body, "a passing check hides its output"
    assert 'class="job-details" open>' not in body, "expanded on a pass"


def test_the_panel_waits_for_the_cameras_to_open_before_showing_a_qr():
    """Before the watcher's `ready` event -- no events at all -- the panel says
    the cameras are still opening and shows no code to aim at."""
    app, client = _client()
    _feed(app, [])  # a running check that has opened nothing yet

    body = client.get("/calibration").text

    assert "Waiting for the cameras to open" in body
    assert "check__qr" not in body, "a QR appeared before the cameras were open"


def test_the_waiting_message_is_a_highlighted_block_not_grey_body_copy():
    """It sits there for tens of seconds on a bimanual rig -- every OAK is opened
    exclusively -- and as `.muted` grey among grey it would read as the panel
    having nothing to say. It carries the console's warn tint, and the glyph and
    the sentence carry the state as well as the colour does."""
    app, client = _client()
    _feed(app, [])

    body = client.get("/calibration").text

    assert "check__waiting" in body, "the waiting state is not highlighted"
    assert "◐ Waiting for the cameras to open" in body, "no non-colour cue"
    # The explanatory line must not be muted grey.
    assert 'class="check__starting muted"' not in body


def test_the_qr_appears_once_the_cameras_report_open():
    """The watcher's `ready` event means every camera is open; the marker to aim
    at then replaces the waiting message."""
    app, client = _client()
    _feed(app, [cal.WATCH_PREFIX + json.dumps({"event": "ready", "roles": []})])

    body = client.get("/calibration").text

    assert "check__qr" in body, "no QR after the cameras opened"
    assert "Waiting for the cameras to open" not in body


# ---- The check completes once every camera has verified ---------------------


def _all_matched(app):
    """Seed a still-running watcher that has already matched every role."""
    roles = cal.verification_roles(cal.recording_config(app))
    assert roles, "the test rig declares camera-check roles"
    _seed_check(
        app,
        [(role, cal.payload_for_side(cal.side_of_role(role))) for role in roles],
    )
    return roles


def test_a_check_with_every_camera_matched_reads_done_not_running():
    """The watcher keeps its cameras open until it is stopped, so 'running' (the
    job is alive) can outlive 'done' (all matched). `done` wins."""
    app, _client_owner = _client()
    roles = _all_matched(app)

    view = cal.check_view(app, roles, lambda role: role)
    assert view.running is True and view.done is True, "precondition: alive + verified"
    assert cal.step_state(app, cal.STEP_VERIFY) == cal.DONE


def test_the_panel_stops_the_watcher_once_every_camera_is_verified(monkeypatch):
    """The watcher holds the exclusive OAK open until stopped, so leaving it
    running over a verified rig would block Record behind a finished check. The
    panel route -- the live pulse the SSE stream drives -- stops it once done."""
    from station_ui.routes import calibration as cal_routes

    app, client = _client()
    _all_matched(app)

    cancelled = []
    monkeypatch.setattr(
        cal_routes.jobs, "cancel", lambda registry, job_id: cancelled.append(job_id)
    )
    client.get("/calibration/check/panel")
    assert cancelled, "a verified check left its watcher (and cameras) running"


def test_the_panel_does_not_stop_a_watcher_with_work_left(monkeypatch):
    """The auto-stop is only for a finished check: a camera still waiting must
    keep the watcher alive so the operator can aim at it."""
    from station_ui.routes import calibration as cal_routes

    app, client = _client()
    roles = cal.verification_roles(cal.recording_config(app))
    # Match every role but the last -> not done, still work to do.
    _seed_check(
        app,
        [(role, cal.payload_for_side(cal.side_of_role(role))) for role in roles[:-1]],
    )

    cancelled = []
    monkeypatch.setattr(
        cal_routes.jobs, "cancel", lambda registry, job_id: cancelled.append(job_id)
    )
    client.get("/calibration/check/panel")
    assert not cancelled, "the watcher was stopped while a camera was still waiting"


def test_the_verified_panel_updates_the_step_badge_out_of_band(monkeypatch):
    """The step's header badge lives outside #check-panel; the panel fragment
    carries an out-of-band update for it, so it flips to Done the instant every
    camera verifies -- not 1-2s later when the watcher dies and the page reloads."""
    from station_ui.routes import calibration as cal_routes

    app, client = _client()
    _all_matched(app)
    # The fake job has no real process; keep the auto-stop from touching it.
    monkeypatch.setattr(cal_routes.jobs, "cancel", lambda registry, job_id: None)

    body = client.get("/calibration/check/panel").text
    assert 'id="check-badge"' in body, "no out-of-band step badge in the panel"
    assert 'hx-swap-oob="true"' in body
    assert cal.BADGES[cal.DONE][1] in body, "the badge did not flip to Done"


def test_the_current_sides_rows_are_highlighted():
    """The cameras being aimed now carry the side accent (a `check--<side>` panel
    class) and an 'Aiming' tag, so which side is live is clear at the rows."""
    app, client = _client()
    _seed_check(app, [])  # running, nothing matched -> current_side is the first
    roles = cal.verification_roles(cal.recording_config(app))
    view = cal.check_view(app, roles, lambda r: r)
    assert view.current_side is not None
    assert view.current_roles == frozenset(
        r for r in roles if cal.side_of_role(r) == view.current_side
    )

    body = client.get("/calibration/check/panel").text
    assert "device-row--current" in body, "the current side's rows are not marked"
    assert "Aiming" in body
    assert "check--" + view.current_side in body, "the panel carries no side accent"


def test_gui_camera_preview_uses_the_current_check_and_disables_native_windows(
    monkeypatch,
):
    app, client = _client()
    launched = []

    def fake_start(registry, argv, **kwargs):
        launched.append(argv)
        job = _FakeJob()
        registry.add(job)
        return job

    monkeypatch.setattr("station_ui.routes.calibration.jobs.start", fake_start)
    assert (
        client.post("/calibration/check/start", follow_redirects=False).status_code
        == 303
    )
    argv = launched[0]
    state = cal.check_state(app)
    assert "--preview" not in argv
    assert argv[argv.index("--preview-file") + 1] == state.preview_path
    assert str(app.state.settings.state_dir) in state.preview_path
    frame_url = f"/calibration/check/frame/{state.job_id}.jpg"
    assert client.get(frame_url).status_code == 204
    pathlib.Path(state.preview_path).write_bytes(b"fake jpeg")
    response = client.get(frame_url)
    assert response.content == b"fake jpeg"
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "no-store"
    assert client.get("/calibration/check/frame/another-job.jpg").status_code == 404
    cal.job_registry(app).get(state.job_id).finished = True
    assert client.get(frame_url).status_code == 404


def test_gui_embeds_camera_preview_beside_the_current_sides_marker():
    app, client = _client()
    job = _feed(
        app,
        [_line(event="ready", roles=cal.verification_roles(cal.recording_config(app)))],
    )
    body = client.get("/calibration/check/panel").text
    assert (
        f'data-calibration-camera-src="/calibration/check/frame/{job.id}.jpg"' in body
    )
    assert "calibration-camera-layout" in body
    assert "Live preview" in body
    assert "Current action" in body
    assert "preview window opens" not in body


@pytest.mark.parametrize("preset", ["bimanual_v4", "bimanual_v4_vive"])
def test_camera_completion_shows_verification_without_a_continue_prompt(preset):
    app, client = _client()
    if preset != "bimanual_v4":
        client.post("/setup/preset", data={"preset_id": preset})
        apply_config(app)
    roles = cal.verification_roles(cal.recording_config(app))
    job = _feed(
        app,
        [
            _line(
                event="saw",
                role=role,
                payload=cal.payload_for_side(cal.side_of_role(role)),
            )
            for role in roles
        ],
    )
    job.finished = True
    job.state = "done"
    job.exit_code = 0
    body = client.get("/calibration/check/panel").text
    assert "All cameras verified." in body
    assert "Check again" in body
    assert "Camera check complete" not in body
    assert 'href="#step-' not in body

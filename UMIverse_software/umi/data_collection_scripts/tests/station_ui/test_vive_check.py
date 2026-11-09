# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fold: what the child saw, turned into what the panel shows.

Derived on every render, never stored -- the same rule the camera check and the
device scan follow, and for the same reason: a row still reading confirmed from
a check five minutes ago is the worst thing this page could show.

The tail of this file covers the job-level plumbing next door in
`calibration.py`: the command, the bound, and what merely asking whether a check
runs is allowed to touch.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import yaml
from fastapi.testclient import TestClient
from test_calibration_page import apply_config

from station_ui import calibration as cal
from station_ui import jobs
from station_ui import vive_check as vc
from station_ui.app import create_app

READY = {
    "event": "ready",
    "arms": ["left", "right"],
    "keys": {"left": "left_knee", "right": "right_knee"},
    "move_m": 0.05,
    "still_m": 0.01,
    "hold_s": 1.0,
}


def _line(**event):
    return vc.VIVE_PREFIX + json.dumps(event)


def _lines(*events):
    return [_line(**event) for event in events]


def test_before_anything_is_read_there_are_no_rows():
    view = vc.fold([])
    assert view.rows == []
    assert view.prompted is None
    assert not view.done


def test_ready_names_the_arms_and_carries_the_thresholds():
    """The thresholds travel on the wire so the panel scales its bars from the
    child's own numbers, rather than from a second copy of the constants in the
    template that could drift from them."""
    view = vc.fold(vc.parse(_lines(READY)))

    assert [row.arm for row in view.rows] == ["left", "right"]
    assert view.thresholds["move_m"] == 0.05
    assert view.thresholds["still_m"] == 0.01


def test_a_missing_ready_falls_back_to_default_thresholds():
    """The job line buffer trims from the front, so a long run can evict its own
    `ready` line. The thresholds then come from the module defaults."""
    view = vc.fold(vc.parse(_lines({"event": "prompt", "arm": "left"})))
    assert view.thresholds["move_m"] == vc.DEFAULT_MOVE_M


def test_an_unusable_threshold_or_reading_cannot_take_the_console_down():
    """`move_m` is the divisor in `percent_value` and `disp` is compared against
    it, and both arrive from a separate process. A raise here is not a broken
    bar: `fold` -> `vive_view` -> `step_state` -> `calibration_done` ->
    `_shell_context` means a 500 on every page of the console, the same blast
    radius the FieldError guard on the generated config exists for."""
    view = vc.fold(
        vc.parse(
            _lines(
                dict(READY, move_m=0, still_m=None),
                {
                    "event": "sample",
                    "t": 1.0,
                    "arms": {
                        "left": {"disp": "nan", "tracked": True},
                        "right": {"disp": 0.2, "tracked": True},
                    },
                },
            )
        )
    )
    by_arm = {row.arm: row for row in view.rows}

    # Both fell back to the module defaults rather than being believed.
    assert view.thresholds == {
        "move_m": vc.DEFAULT_MOVE_M,
        "still_m": vc.DEFAULT_STILL_M,
    }
    assert by_arm["left"].disp is None and by_arm["left"].state == vc.WAITING
    assert by_arm["right"].percent_value == 100.0


def test_the_latest_sample_wins_for_each_arm():
    view = vc.fold(
        vc.parse(
            _lines(
                READY,
                {
                    "event": "sample",
                    "t": 1.0,
                    "arms": {
                        "left": {"disp": 0.01, "tracked": True},
                        "right": {"disp": 0.0, "tracked": True},
                    },
                },
                {
                    "event": "sample",
                    "t": 2.0,
                    "arms": {
                        "left": {"disp": 0.14, "tracked": True},
                        "right": {"disp": 0.001, "tracked": True},
                    },
                },
            )
        )
    )
    by_arm = {row.arm: row for row in view.rows}

    assert by_arm["left"].disp == 0.14
    assert by_arm["left"].state == vc.MOVING
    assert by_arm["right"].state == vc.STILL


def test_the_prompted_arm_is_the_one_the_operator_must_move():
    view = vc.fold(vc.parse(_lines(READY, {"event": "prompt", "arm": "left"})))
    assert view.prompted == "left"


def test_a_confirmed_arm_latches_against_a_later_sample():
    """The operator moves the right gripper next, and the left tracker will
    show motion again -- a bench is not still. An arm that already proved
    itself must not silently un-confirm."""
    view = vc.fold(
        vc.parse(
            _lines(
                READY,
                {"event": "prompt", "arm": "left"},
                {"event": "confirmed", "arm": "left"},
                {"event": "prompt", "arm": "right"},
                {
                    "event": "sample",
                    "t": 9.0,
                    "arms": {
                        "left": {"disp": 0.20, "tracked": True},
                        "right": {"disp": 0.14, "tracked": True},
                    },
                },
            )
        )
    )
    by_arm = {row.arm: row for row in view.rows}

    assert by_arm["left"].state == vc.CONFIRMED
    assert view.prompted == "right"


def test_every_arm_confirmed_is_done():
    view = vc.fold(
        vc.parse(
            _lines(
                READY,
                {"event": "confirmed", "arm": "left"},
                {"event": "confirmed", "arm": "right"},
            )
        )
    )
    assert view.done


def test_only_narrows_the_fold_to_the_arm_this_run_was_launched_for():
    """A per-arm run's section must not wait on an arm it was never going to
    walk. The child still watches both pucks -- that is what catches a swapped
    pair -- so its `ready` line names both, and without `only` the left arm's
    own section would fold in a right arm that is not its business and never
    call itself done."""
    events = vc.parse(
        _lines(
            READY,
            {"event": "prompt", "arm": "left"},
            {"event": "confirmed", "arm": "left"},
        )
    )

    assert vc.fold(events).done is False
    left = vc.fold(events, only=["left"])
    assert [row.arm for row in left.rows] == ["left"]
    assert left.done is True


def test_only_does_not_hide_a_cross_found_against_another_arm():
    """The narrowing is about whose row and whose verdict, never about what the
    run was allowed to see. A left-only run that catches the right puck moving
    still reports the swap."""
    view = vc.fold(
        vc.parse(
            _lines(READY, {"event": "cross", "prompted": "left", "moved": "right"})
        ),
        only=["left"],
    )

    assert view.cross == {"prompted": "left", "moved": "right"}
    assert view.done is False


def test_a_cross_names_both_keys_and_fails_the_view():
    """Every arm confirmed and a cross seen -- and it is still not done.

    Both halves are load-bearing. Without the confirms, `done` would be False
    because nothing was confirmed, and the assertion would pass with the cross
    playing no part: drop `not view.cross` from the fold and a check that
    proved the keys are swapped would badge green.
    """
    view = vc.fold(
        vc.parse(
            _lines(
                READY,
                {"event": "confirmed", "arm": "left"},
                {"event": "confirmed", "arm": "right"},
                {"event": "cross", "prompted": "left", "moved": "right"},
            )
        )
    )

    assert view.cross == {"prompted": "left", "moved": "right"}
    assert not view.done


def test_an_untracked_arm_says_so():
    view = vc.fold(
        vc.parse(
            _lines(
                READY,
                {
                    "event": "sample",
                    "t": 1.0,
                    "arms": {
                        "left": {"disp": None, "tracked": False},
                        "right": {"disp": 0.0, "tracked": True},
                    },
                },
            )
        )
    )
    by_arm = {row.arm: row for row in view.rows}
    assert by_arm["left"].state == vc.UNTRACKED


def test_a_puck_in_the_dead_space_gets_no_verdict_rather_than_a_wrong_one():
    """3 cm is neither of the two things this panel can say.

    The gap between `still_m` and `move_m` is dead space on purpose -- the same
    range the child's `classify` refuses to rule on, because a lighthouse
    reflection drifting a puck 2 cm is not an operator moving a gripper. Folding
    that to MOVING would assert motion on an arm nobody is touching; folding it
    to STILL would claim a puck that is visibly wandering is parked.
    """
    view = vc.fold(
        vc.parse(
            _lines(
                READY,
                {
                    "event": "sample",
                    "t": 1.0,
                    "arms": {
                        "left": {"disp": 0.03, "tracked": True},
                        "right": {"disp": 0.03, "tracked": True},
                    },
                },
            )
        )
    )

    assert [row.state for row in view.rows] == [vc.WAITING, vc.WAITING]


def test_the_bar_fills_in_proportion_to_the_move_threshold():
    """The panel draws `percent_value`, so it has to be a number computed from
    the child's own `move_m` -- an arm halfway to the threshold is a half-full
    bar, and one past it is full rather than overflowing."""
    view = vc.fold(
        vc.parse(
            _lines(
                READY,
                {
                    "event": "sample",
                    "t": 1.0,
                    "arms": {
                        "left": {"disp": 0.025, "tracked": True},
                        "right": {"disp": 0.20, "tracked": True},
                    },
                },
            )
        )
    )
    by_arm = {row.arm: row for row in view.rows}

    assert by_arm["left"].percent_value == 50.0
    assert by_arm["right"].percent_value == 100.0


def test_a_prefixed_line_that_will_not_parse_is_dropped():
    """A child killed mid-print at 20 Hz leaves a partial line. A panel that
    500s on it is worse than one that ignores it."""
    view = vc.fold(vc.parse([vc.VIVE_PREFIX + "{not json", _line(**READY)]))
    assert [row.arm for row in view.rows] == ["left", "right"]


def test_the_camera_checks_lines_are_never_folded_here():
    from station_ui import calibration as cal

    lines = [cal.WATCH_PREFIX + json.dumps({"event": "saw", "role": "left_umi"})]
    assert vc.parse(lines) == []


def test_the_prefix_matches_the_child_that_prints_it():
    """The literal is declared in both modules on purpose -- `station_ui`
    importing the child would drag a ZMQ/matplotlib shell into the web process,
    the same trade `WATCH_PREFIX` already makes next door. This is what keeps
    the two copies from drifting: change one and this fails.
    """
    from real_scripts.device_verification.verify_vive_assignment import (
        VIVE_PREFIX as child_prefix,
    )

    assert vc.VIVE_PREFIX == child_prefix


def test_the_camera_check_keeps_its_own_prefix_by_default():
    """`parse_events` and `watcher_log` take a `prefix` argument for the Vive
    check; a caller that passes nothing must still get the camera check's."""
    from station_ui import calibration as cal

    camera = cal.WATCH_PREFIX + json.dumps({"event": "saw", "role": "left_umi"})
    vive = _line(**READY)

    assert cal.parse_events([camera, vive]) == [{"event": "saw", "role": "left_umi"}]
    assert cal.watcher_log([camera, vive, "opening cameras"]) == [
        vive,
        "opening cameras",
    ]


# ----------------------------------------------------------- the job plumbing


def test_the_gui_command_names_the_config_and_arm_without_a_native_plot():
    argv = cal.vive_argv("/tmp/generated.yaml", "left")

    assert "--watch" in argv
    assert "--plot" not in argv
    assert argv[argv.index("--config") + 1] == "/tmp/generated.yaml"
    # The arm is what makes this a per-button run: the child confirms it and
    # returns rather than walking on to the other gripper.
    assert argv[argv.index("--arm") + 1] == "left"


def test_the_script_the_command_names_exists():
    script = [
        a for a in cal.vive_argv("x", "left") if a.endswith("verify_vive_assignment.py")
    ]
    assert script and pathlib.Path(script[0]).is_file()


def test_the_check_is_bounded_unlike_the_camera_check():
    """The camera check runs until stopped -- it holds cameras, and the panel
    stops it on success. This one polls a bridge shared by every station, so an
    operator who walks away must not leave it running."""
    assert cal.VIVE_TIMEOUT_S == 300.0
    assert cal.CHECK_TIMEOUT_S is None


def test_asking_whether_a_vive_check_runs_attaches_nothing():
    """`check_running` reads app.state with bare getattr on purpose, so merely
    asking does not attach a job registry and its shutdown hook to an app that
    has never run one. A version that went through `_shared` would attach one."""
    app = create_app()
    assert cal.vive_running(app) is False
    assert getattr(app.state, "jobs", None) is None


def test_the_calibration_registry_is_reaped_on_shutdown_exactly_once():
    """`jobs.start` detaches its children (`start_new_session=True`), so nothing
    else ever collects them: without this hook, a console that never ran a
    device scan -- the only other accessor that attaches one -- would leave the
    camera check's held OAK, or a Vive poller, running after the server exited.

    Once however many times the registry is asked for: the handler list is read
    at shutdown, and a hundred renders must not append a hundred copies.
    """
    app = create_app()
    registry = cal.job_registry(app)
    for _ in range(3):
        cal.job_registry(app)

    assert app.router.on_shutdown.count(registry.shutdown) == 1


def test_both_registry_accessors_share_one_shutdown_marker():
    """The flag lives on `app.state` under the key `devices.registry_for` also
    sets. Both accessors hand out the same `app.state.jobs`, so whichever is
    asked first must be the only one that hooks it. The literal is re-declared
    rather than imported -- importing `devices` drags the device-discovery SDK
    into the web process -- and this is what keeps the two copies from drifting.
    """
    from station_ui import devices

    assert cal._HOOKED == devices._HOOKED


# ------------------------------------------------- what the job adds to the fold


class _FakeJob:
    """A job that never ran, for asserting what the view makes of one."""

    id = "fake-vive-job"
    state = jobs.RUNNING
    finished = False
    exit_code = None
    error = None
    argv = ["fake"]
    cwd = "."

    def __init__(self, output=(), **fields):
        self._lines = list(output)
        for name, value in fields.items():
            setattr(self, name, value)

    def lines(self):
        return list(self._lines)


def _seed_vive(app, job, arm="left"):
    """Put one arm's check job in the registry the way a real start would."""
    cal.job_registry(app).add(job)
    cal.vive_check_state(app).job_ids[arm] = job.id
    return job


def test_a_timed_out_check_is_a_stop_not_a_failure():
    """`jobs` has no TIMED_OUT state -- it finishes a timed-out job as FAILED
    carrying "timed out after 300s", and only `jobs.cancel` produces CANCELLED.
    So copying the camera check's `failed = state == JOB_FAILED` verbatim would
    badge a walked-away check red and tell the operator something broke when
    nothing did. This is the only place the two can be told apart."""
    app = create_app()
    _seed_vive(
        app,
        _FakeJob(state=jobs.FAILED, finished=True, error="timed out after 300s"),
    )

    view = cal.vive_view(app, "left")

    assert view.timed_out is True
    assert view.failed is False


def test_a_check_whose_child_actually_died_is_a_failure():
    """The other half of the distinction above: a child that could not be
    spawned, or exited non-zero, prints no events at all -- so the fold sees
    nothing and the panel would read "not started" over a check that broke."""
    app = create_app()
    _seed_vive(
        app,
        _FakeJob(state=jobs.FAILED, finished=True, exit_code=1, error="exited with"),
    )

    view = cal.vive_view(app, "left")

    assert view.failed is True
    assert view.timed_out is False


def test_crossed_trackers_fail_the_view_even_though_the_child_exited_cleanly():
    """The child returns 0 having proved the keys are swapped. That is the worst
    thing this check can find and the one thing an exit code cannot carry, so
    the view has to fail on the event rather than on the job's fate."""
    app = create_app()
    _seed_vive(
        app,
        _FakeJob(
            output=_lines(
                READY, {"event": "cross", "prompted": "left", "moved": "right"}
            ),
            state=jobs.DONE,
            finished=True,
            exit_code=0,
        ),
    )

    view = cal.vive_view(app, "left")

    assert view.cross == {"prompted": "left", "moved": "right"}
    assert view.failed is True


def test_the_panel_log_keeps_the_bridges_words_and_drops_the_checks_own():
    """One `##station-ui-vive##` line per 20 Hz sample would bury whatever the
    bridge said, which is the only reason anyone opens the log."""
    app = create_app()
    _seed_vive(app, _FakeJob(output=[_line(**READY), "connecting to tcp://…:8888"]))

    assert cal.vive_view(app, "left").log == ["connecting to tcp://…:8888"]


def test_a_rig_change_releases_the_bridge(monkeypatch):
    """Apply resets the calibration when the devices change. Leaving the poller
    running would keep it hitting a bridge the new config may no longer name --
    and that bridge is one REP thread shared with every station on the bench."""
    app = create_app()
    job = _seed_vive(app, _FakeJob())
    cancelled = []
    monkeypatch.setattr(
        cal.jobs, "cancel", lambda registry, job_id: cancelled.append(job_id)
    )

    cal.reset(app)

    assert cancelled == [job.id]
    assert cal.vive_check_state(app).job_ids == {}


def test_the_timeout_verdict_is_tied_to_the_message_jobs_really_writes():
    """A real job through the real wall-clock cap, not a fake carrying the
    string.

    `jobs` has no TIMED_OUT state, so the only thing separating "the operator
    walked away" from "the check broke" is the wording of `job.error` -- a
    message owned by another module. Reword it in `jobs.py` and the match here
    silently stops firing: `timed_out` goes False, `failed` goes True, and an
    operator returning after five minutes finds a red FAILED badge and re-runs a
    check that never failed, on a bridge every station on the bench shares.
    `test_jobs.py` would stay green throughout.

    So the two files have to fail together. This is the same pin the
    `VIVE_PREFIX` and `_HOOKED` duplications carry, for the same reason.
    """
    app = create_app()
    job = jobs.start(
        cal.job_registry(app),
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=".",
        timeout=1.0,
    )
    cal.vive_check_state(app).job_ids["left"] = job.id
    deadline = time.monotonic() + 20
    while not job.finished and time.monotonic() < deadline:
        time.sleep(0.02)
    assert job.finished, "the child never hit its wall-clock cap"

    view = cal.vive_view(app, "left")

    assert view.timed_out is True, f"not read as a timeout: {job.error!r}"
    assert view.failed is False, "a walked-away check was badged as broken"


# ------------------------------------------------- the routes and the panel


def _vive_client():
    """A browser that owns a rig whose preset records Vive trackers.

    The panel reads the config Apply generated, never the template it came from
    -- the same reason `test_camera_check` applies one before touching its
    check panel.
    """
    app = create_app()
    client = TestClient(app)
    client.get("/")
    client.post("/setup/preset", data={"preset_id": "bimanual_v4_vive"})
    app.state.rig.draft.grippers.update({"left": "UMI-V4-0001", "right": "UMI-V4-0002"})
    apply_config(app)
    return app, client


def _seed_events(app, *events, arm="left", **fields):
    """A running check for `arm` that has already printed `events`."""
    job = _FakeJob(output=_lines(*events), **fields)
    job.id = f"fake-vive-{arm}"
    return _seed_vive(app, job, arm)


def _confirm(app, arm):
    """`arm` checked and confirmed, its child exited -- what the operator has
    behind them when they turn to the other gripper."""
    return _seed_events(
        app,
        READY,
        {"event": "prompt", "arm": arm},
        {"event": "confirmed", "arm": arm},
        arm=arm,
        state=jobs.DONE,
        finished=True,
        exit_code=0,
    )


def test_the_first_render_names_the_trackers_instead_of_denying_them():
    """The first thing every operator sees of this feature, on a rig that has
    the step because it records trackers. Built only from the child's events,
    the rows would be empty before any check has run, and the panel's only prose
    would read "No trackers to check." directly above the button offering to
    check them -- which reads as "the console has not found my pucks".

    Seeded from the config instead, like the camera check's rows come from
    `verification_roles`, so both keys are on screen to eyeball before Start.
    That matters: a bridge can serve left_knee one day and left_shoulder the
    next, which is exactly what this step is here to catch.
    """
    _app, client = _vive_client()

    body = client.get("/calibration/vive/panel").text

    assert "left_knee" in body and "right_knee" in body
    assert "No trackers to check" not in body
    # Waiting, not a verdict: nothing has looked at a puck yet.
    assert "Waiting" in body
    assert cal.step_state(_app, cal.STEP_VIVE) == cal.PENDING


def test_a_rig_with_no_trackers_still_says_there_are_none():
    """The empty state is kept for the case it was written for. `vive_view` runs
    on every calibration render, Vive rig or not."""
    app, _client = _client_of("bimanual_v4")

    assert cal.vive_sections(app) == []


def test_the_panel_prompts_for_the_arm_being_checked():
    """The whole operator-facing point of the step: which gripper to pick up,
    and which tracker key that arm is being checked against."""
    app, client = _vive_client()
    _seed_events(app, READY, {"event": "prompt", "arm": "left"})

    body = client.get("/calibration/vive/panel").text

    assert "LEFT" in body
    assert "left_knee" in body


def test_each_arm_gets_its_own_button():
    """The shape of the whole step: a button per gripper, not one for the rig.

    Both arms are on screen from the first render -- their keys are what the
    operator eyeballs against the bridge, which can rename pucks between days --
    but each carries its own Start, so the left arm is checked, confirmed, and
    then the right one is started deliberately.
    """
    _app, client = _vive_client()

    body = client.get("/calibration/vive/panel").text

    assert 'value="left"' in body and 'value="right"' in body
    assert "Check left tracker" in body
    assert "Check right tracker" in body


def test_the_right_arm_waits_for_the_left_to_confirm():
    """Sequential, and said in words beside the dead button. A greyed control is
    a dead end on its own -- an operator cannot infer "one arm at a time, in
    order" from a box that will not click."""
    _app, client = _vive_client()

    body = client.get("/calibration/vive/panel").text

    assert "Confirm the left tracker first." in body


def test_the_right_arms_button_comes_alive_once_the_left_confirms():
    app, client = _vive_client()
    _confirm(app, "left")

    body = client.get("/calibration/vive/panel").text

    assert "Confirm the left tracker first." not in body
    assert "Check left again" in body


def test_starting_the_right_arm_early_is_refused_by_the_route_not_just_the_button():
    """A disabled button is a hint. The rule is the route's -- otherwise a stale
    tab, a replayed POST or a browser with the attribute stripped walks straight
    past the ordering the step exists to impose."""
    _app, client = _vive_client()

    response = client.post(
        "/calibration/vive/start", data={"arm": "right"}, follow_redirects=False
    )

    assert response.status_code == 409
    assert "Confirm the left tracker first" in response.text


def test_a_confirmed_arm_stays_confirmed_after_its_child_exits():
    """The thing per-arm runs have to get right. The left run ends when the left
    arm confirms, so its verdict has to outlive the process that produced it --
    otherwise pressing the right arm's button erases the half of the step the
    operator just finished."""
    app, client = _vive_client()
    _confirm(app, "left")

    body = client.get("/calibration/vive/panel").text
    sections = {s["arm"]: s["view"] for s in cal.vive_sections(app)}

    assert sections["left"].done is True
    assert sections["left"].running is False
    assert sections["right"].done is False
    assert "✓ Confirmed" in body
    # Half done is not done: the step still blocks Record.
    assert cal.step_state(app, cal.STEP_VIVE) == cal.PENDING


def test_the_step_is_done_only_once_both_arms_have_confirmed():
    app, client = _vive_client()
    _confirm(app, "left")
    assert cal.step_state(app, cal.STEP_VIVE) == cal.PENDING

    _confirm(app, "right")
    assert cal.step_state(app, cal.STEP_VIVE) == cal.DONE


def test_the_panel_says_which_keys_look_swapped():
    """A cross is the one failure nothing downstream can catch, so the panel has
    to name both arms -- "the check failed" sends an operator to the wrong end
    of the rig.

    Found by the left arm's own run, which is the point: `--arm left` narrows
    what the child confirms, never what it watches, so the swap is caught by the
    first button the operator presses rather than surviving until the second.
    """
    app, client = _vive_client()
    _seed_events(
        app, READY, {"event": "cross", "prompted": "left", "moved": "right"}, arm="left"
    )

    body = client.get("/calibration/vive/panel").text

    assert "swapped" in body.lower()
    assert "LEFT" in body and "RIGHT" in body
    assert cal.step_state(app, cal.STEP_VIVE) == cal.FAILED


def test_one_arms_cross_fails_the_step_even_beside_a_confirmed_arm():
    """A swap is a property of the rig, not of one arm. The left arm sitting
    confirmed must not outvote the right arm's proof that the keys are crossed:
    half a rig verified is not a rig verified."""
    app, _client = _vive_client()
    _confirm(app, "left")
    _seed_events(
        app,
        READY,
        {"event": "cross", "prompted": "right", "moved": "left"},
        arm="right",
        state=jobs.DONE,
        finished=True,
        exit_code=0,
    )

    assert cal.step_state(app, cal.STEP_VIVE) == cal.FAILED


def test_a_timed_out_check_does_not_read_as_broken():
    """`jobs` finishes a timed-out job as FAILED, so the panel is the last place
    the distinction can still be lost: an operator who walked away must be told
    to press Start again, not that their rig is faulty."""
    app, client = _vive_client()
    _seed_vive(
        app,
        _FakeJob(
            output=_lines(READY),
            state=jobs.FAILED,
            finished=True,
            error="timed out after 300s",
        ),
    )

    body = client.get("/calibration/vive/panel").text

    assert "Timed out" in body
    assert "Failed" not in body


def test_starting_an_arm_launches_a_child_for_that_arm_and_remembers_it(monkeypatch):
    launched = []

    def fake_start(registry, argv, **kwargs):
        launched.append(argv)
        return _FakeJob()

    app, client = _vive_client()
    monkeypatch.setattr("station_ui.routes.calibration.jobs.start", fake_start)
    response = client.post(
        "/calibration/vive/start", data={"arm": "left"}, follow_redirects=False
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/calibration#tracker-heading-left"
    assert len(launched) == 1
    assert "--watch" in launched[0] and "--plot" not in launched[0]
    assert launched[0][launched[0].index("--arm") + 1] == "left"
    # Remembered under the arm, so the right one's later run does not evict it.
    assert cal.vive_check_state(app).job_ids == {"left": "fake-vive-job"}


def test_a_second_start_while_one_runs_is_refused_even_for_the_other_arm():
    """One bridge, one `zmq.REP` thread, shared by every station on the bench --
    two pollers on it is exactly the load this check is careful about, and two
    for different arms is still two."""
    app, client = _vive_client()
    _confirm(app, "left")
    _seed_events(app, READY, arm="right")

    response = client.post(
        "/calibration/vive/start", data={"arm": "right"}, follow_redirects=False
    )

    assert response.status_code == 409
    assert "already running" in response.text


def test_starting_an_arm_this_rig_does_not_record_is_refused():
    """A stale form against a re-applied config. Named rather than ignored: the
    button came from a page that has stopped being true."""
    _app, client = _vive_client()

    response = client.post(
        "/calibration/vive/start", data={"arm": "head"}, follow_redirects=False
    )

    assert response.status_code == 409
    assert "no head tracker" in response.text


def test_stopping_the_check_cancels_the_child(monkeypatch):
    app, client = _vive_client()
    job = _seed_events(app, READY)
    cancelled = []
    monkeypatch.setattr(
        "station_ui.routes.calibration.jobs.cancel",
        lambda registry, job_id, **kwargs: cancelled.append(job_id),
    )

    response = client.post("/calibration/vive/stop", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/calibration#tracker-heading-left"
    assert cancelled == [job.id]


def test_the_panel_stops_an_arms_child_the_moment_that_arm_confirms(monkeypatch):
    """The fold never cancels anything -- the panel route is where "done, so let
    go" happens. Without it a confirmed check keeps hammering the shared bridge
    until the five-minute timeout, and then flips from Done to a timeout.

    It happens per arm: releasing the bridge as soon as the left arm confirms is
    what lets the operator press the right arm's button straight away instead of
    waiting out a poller with nothing left to watch."""
    app, client = _vive_client()
    _seed_events(app, READY, {"event": "confirmed", "arm": "left"}, arm="left")
    cancelled = []
    monkeypatch.setattr(
        "station_ui.routes.calibration.jobs.cancel",
        lambda registry, job_id, **kwargs: cancelled.append(job_id),
    )

    client.get("/calibration/vive/panel")

    assert cancelled == ["fake-vive-left"]


def test_the_stream_only_ever_serves_this_pages_own_vive_check():
    """The registry also holds gripper, camera-check and recording jobs, and an
    endpoint under /calibration/vive/ that would happily stream a recording is
    an accident waiting to be linked to."""
    app, client = _vive_client()
    registry = cal.job_registry(app)
    other = jobs.start(registry, [sys.executable, "-c", "print('audit')"], cwd=".")
    watcher = jobs.start(registry, [sys.executable, "-c", "print('watching')"], cwd=".")
    cal.vive_check_state(app).job_ids["left"] = watcher.id

    assert client.get(f"/calibration/vive/stream/{other.id}").status_code == 404
    assert client.get("/calibration/vive/stream/deadbeef").status_code == 404

    response = client.get(f"/calibration/vive/stream/{watcher.id}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_the_page_only_offers_a_stream_while_a_check_is_alive():
    """An EventSource opened against a finished job connects, replays the whole
    buffer and closes on every reload -- so the URL is offered only while there
    is something to follow, exactly as the camera check's is.

    Read off the render context rather than the page: the URL is what this
    route owns, and asserting it there does not also depend on how the template
    happens to spell the `<script>` it feeds.
    """
    app, client = _vive_client()

    assert (
        next(
            section["context"]
            for section in client.get("/calibration").context["workflow_sections"]
            if section["id"] == "calibration"
        )["vive_stream_url"]
        is None
    )

    job = _seed_events(app, READY)
    context = next(
        section["context"]
        for section in client.get("/calibration").context["workflow_sections"]
        if section["id"] == "calibration"
    )
    assert context["vive_stream_url"] == f"/calibration/vive/stream/{job.id}"
    # Left before right, whatever order the config declares them in: the right
    # arm is gated on the left, so this order is the workflow.
    assert [s["arm"] for s in context["vive_sections"]] == ["left", "right"]

    job.finished = True
    assert (
        next(
            section["context"]
            for section in client.get("/calibration").context["workflow_sections"]
            if section["id"] == "calibration"
        )["vive_stream_url"]
        is None
    )


# --------------------------------------------------- which steps this rig has


def _config_of(preset_id):
    """The template for a preset, parsed. Enough for `steps_for`, which asks
    only whether the rig declares a tracker -- a question the placeholder
    serials in a template cannot get wrong."""
    from station_ui.core import setup_presets as sp

    path = sp.template_path(sp.get_preset(preset_id), cal.repo_root())
    return yaml.safe_load(path.read_text())


def _client_of(preset_id):
    """A browser that owns an applied rig of `preset_id`. `_vive_client` for
    any preset, so the non-Vive half of every rule can be asserted too."""
    app = create_app()
    client = TestClient(app)
    client.get("/")
    client.post("/setup/preset", data={"preset_id": preset_id})
    app.state.rig.draft.grippers.update({"left": "L", "right": "R"})
    apply_config(app)
    return app, client


def test_a_vive_rig_gets_three_steps_and_a_plain_rig_two():
    plain = [step.id for step in cal.steps_for(_config_of("bimanual_v4"))]
    vive = [step.id for step in cal.steps_for(_config_of("bimanual_v4_vive"))]

    assert plain == [cal.STEP_VERIFY, cal.STEP_GRIPPER]
    # Between the camera check and the grippers: both are "each device is in
    # the slot the config names"; gripper calibration is the odd one out.
    assert vive == [cal.STEP_VERIFY, cal.STEP_VIVE, cal.STEP_GRIPPER]


def test_the_catalogue_still_carries_every_step():
    """STEPS stays the full catalogue so CalibrationState keeps a bucket per
    step and reset_verification_for_new_rig still expires a stale Vive stamp
    after the operator switches away from a Vive preset."""
    assert {step.id for step in cal.STEPS} == {
        cal.STEP_VERIFY,
        cal.STEP_VIVE,
        cal.STEP_GRIPPER,
    }


def test_the_vive_steps_units_are_the_arms_the_config_records():
    """A step's units are what it has to clear, and this one clears arms."""
    app, _client = _vive_client()

    assert cal.expected_keys(app, cal.STEP_VIVE) == ("left", "right")


def test_a_rig_with_no_trackers_has_no_vive_units():
    app = create_app()
    client = TestClient(app)
    client.get("/")
    client.post("/setup/preset", data={"preset_id": "bimanual_v4"})
    apply_config(app)

    assert cal.expected_keys(app, cal.STEP_VIVE) == ()


def test_the_calibration_page_renders_for_every_rig():
    """A third id reaching `actions[step.id]` in _step_views raises KeyError
    out of _page, which 500s every GET on this page for every rig -- not just
    Vive ones."""
    for preset in ("bimanual_v4", "bimanual_v4_vive"):
        app = create_app()
        client = TestClient(app)
        client.get("/")
        client.post("/setup/preset", data={"preset_id": preset})
        app.state.rig.draft.grippers.update({"left": "L", "right": "R"})
        apply_config(app)

        assert client.get("/calibration").status_code == 200, preset


def test_calibration_titles_have_no_step_number_prefixes():
    _app, client = _vive_client()

    body = client.get("/calibration").text

    assert '<h2 id="step-device_verify">Camera check</h2>' in body
    assert '<h2 id="step-vive_check">Tracker check</h2>' in body
    assert '<h2 id="step-gripper_cal">Calibrate grippers</h2>' in body


def test_a_plain_rig_never_draws_the_tracker_step():
    _app, client = _client_of("bimanual_v4")

    body = client.get("/calibration").text

    assert "Tracker check" not in body
    assert "Step 3" not in body


def test_record_is_not_blocked_on_a_non_vive_rig_by_the_vive_step(monkeypatch):
    """The whole reason steps_for exists. A third step in the catalogue that
    every rig had to pass would block Record on every station, behind a check
    with no trackers to look at."""
    from station_ui import readiness

    app, _client = _client_of("bimanual_v4")

    monkeypatch.setattr(
        cal,
        "step_state",
        lambda app_, step_id: cal.DONE if step_id != cal.STEP_VIVE else cal.PENDING,
    )

    assert readiness.calibration_done(app) is True


def test_record_waits_for_the_tracker_check_on_a_vive_rig(monkeypatch):
    """The other half: on a rig that does record trackers the step is a gate,
    not decoration."""
    from station_ui import readiness

    app, _client = _vive_client()

    monkeypatch.setattr(
        cal,
        "step_state",
        lambda app_, step_id: cal.DONE if step_id != cal.STEP_VIVE else cal.PENDING,
    )

    assert readiness.calibration_done(app) is False


def test_calibration_done_is_false_rather_than_raising_with_no_config():
    """readiness.done/blocking run from _shell_context on every page, so an
    uncaught FieldError here 500s the whole console."""
    from station_ui import readiness

    app = create_app()

    assert readiness.calibration_done(app) is False


def test_a_timed_out_vive_check_reads_pending_not_failed():
    """A timeout proved nothing, so "not started" is the honest badge. Failed
    would send an operator looking for a rig fault there is not."""
    app, _client = _vive_client()
    job = _seed_events(app, READY, {"event": "prompt", "arm": "left"})
    job.state = jobs.FAILED
    job.finished = True
    job.error = "timed out after 300s"

    assert cal.step_state(app, cal.STEP_VIVE) == cal.PENDING
    assert cal.vive_view(app, "left").timed_out


def test_every_arm_confirmed_makes_the_step_done():
    """Each arm confirmed by its own run -- which is the step's shape, and what
    the roll-up in `step_state` has to add up."""
    app, _client = _vive_client()
    _confirm(app, "left")
    _confirm(app, "right")

    assert cal.step_state(app, cal.STEP_VIVE) == cal.DONE


def test_a_cross_fails_the_step():
    """The child exits 0 having proved the keys are swapped, so the exit code
    alone would badge this rig verified."""
    app, _client = _vive_client()
    _seed_events(app, READY, {"event": "cross", "prompted": "left", "moved": "right"})

    assert cal.step_state(app, cal.STEP_VIVE) == cal.FAILED


def test_a_check_that_has_not_run_reads_pending():
    app, _client = _vive_client()

    assert cal.step_state(app, cal.STEP_VIVE) == cal.PENDING


# ------------------------------------------------------- the live-update script


def test_the_page_streams_the_tracker_check_only_while_one_is_alive():
    """An EventSource opened against a finished job connects, replays the whole
    buffer and closes on every reload -- so the `<script>` is guarded, not just
    the URL it would dial."""
    app, client = _vive_client()

    assert "/calibration/vive/stream/" not in client.get("/calibration").text

    job = _seed_events(app, READY)
    assert "/calibration/vive/stream/" in client.get("/calibration").text

    job.finished = True
    assert "/calibration/vive/stream/" not in client.get("/calibration").text


def test_the_tracker_stream_refetches_its_panel_instead_of_reloading():
    """Two streams that both reload mean whichever child exits first reloads the
    page out from under the other panel mid-check. The camera check's handler
    navigates; this one refetches its own fragment, so both can run at once."""
    app, client = _vive_client()
    _seed_events(app, READY)
    cal.check_state(app).job_id = (
        cal.job_registry(app).add(_FakeJob(output=["opening cameras"])).id
    )

    body = client.get("/calibration").text

    # One of each: the camera check still navigates on `end`, the tracker check
    # swaps only #vive-panel.
    assert body.count("window.location.reload()") == 1
    assert body.count('"/calibration/vive/panel"') == 1
    assert body.count('"#vive-panel"') == 1


def test_the_panel_fragment_carries_no_script_of_its_own():
    """It is swapped in on every event, so a `<script>` inside it would open a
    fresh EventSource several times a second."""
    app, client = _vive_client()
    _seed_events(app, READY)

    assert "<script" not in client.get("/calibration/vive/panel").text


def test_tracker_progress_does_not_restore_the_removed_forward_control(monkeypatch):
    """Live tracker updates keep the removed footer absent while badges update."""
    app, client = _vive_client()
    real = cal.step_state
    # Only the tracker check is left to pass, and it is left real -- the point
    # is that the control follows this check's own verdict.
    monkeypatch.setattr(
        cal,
        "step_state",
        lambda app_, sid: real(app_, sid) if sid == cal.STEP_VIVE else cal.DONE,
    )
    # The panel lets go of the bridge once every arm is confirmed, and these
    # jobs are fakes with nothing to cancel.
    monkeypatch.setattr(
        "station_ui.routes.calibration.jobs.cancel",
        lambda registry, job_id, **kwargs: None,
    )

    _seed_events(app, READY, {"event": "prompt", "arm": "left"})
    before = client.get("/calibration/vive/panel").text

    assert 'id="next-step"' not in before
    assert "Go to Record" not in before
    assert 'id="vive-badge"' in before and 'hx-swap-oob="true"' in before

    _confirm(app, "left")
    _confirm(app, "right")
    after = client.get("/calibration/vive/panel").text

    assert 'id="next-step"' not in after
    assert "Go to Record" not in after
    assert "All trackers confirmed" in after
    assert 'id="vive-badge"' in after and 'hx-swap-oob="true"' in after
    assert "window.location" not in after


def test_the_calibration_page_has_no_forward_control():
    _app, client = _vive_client()

    assert 'id="next-step"' not in client.get("/calibration").text


def test_tracker_samples_are_plotted_inside_the_panel():
    app, client = _vive_client()
    _seed_events(
        app,
        READY,
        {"event": "prompt", "arm": "left"},
        {
            "event": "sample",
            "t": 0,
            "arms": {
                "left": {"disp": 0.01, "tracked": True},
                "right": {"disp": 0.002, "tracked": True},
            },
        },
        {
            "event": "sample",
            "t": 1,
            "arms": {
                "left": {"disp": 0.06, "tracked": True},
                "right": {"disp": 0.003, "tracked": True},
            },
        },
    )
    body = client.get("/calibration/vive/panel").text
    assert 'class="calibration-tracker-chart"' in body
    assert 'class="calibration-chart-trace calibration-chart-trace--primary"' in body
    assert 'class="calibration-chart-trace calibration-chart-trace--secondary"' in body
    assert "Move ≥ 50 mm" in body
    assert "Current action" in body
    assert "plot opens on the station screen" not in client.get("/calibration").text


def test_completed_tracker_check_links_to_the_unchanged_gripper_step():
    app, client = _vive_client()
    _confirm(app, "left")
    _confirm(app, "right")
    body = client.get("/calibration/vive/panel").text
    assert "All trackers confirmed" in body
    assert 'href="#step-gripper_cal"' in body

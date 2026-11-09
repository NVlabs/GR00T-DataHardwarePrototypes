# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The rule that decides whether the right puck moved.

Pure functions, no ZMQ, no matplotlib, no puck -- which is the entire reason
they are separated out of the poll loop. The thresholds are the only thing
standing between "the keys are swapped" and a whole shift of mislabelled pose,
and nothing downstream can detect that after the fact: both arms show plausible
motion either way.
"""

import io
import json
import math
import pathlib
import signal

from real_scripts.device_verification import verify_vive_assignment as vv

TH = (vv.MOVE_M, vv.STILL_M, vv.HOLD_S)

REPO = pathlib.Path(__file__).resolve().parents[2]


def _track(*points):
    """(t, xyz) samples 50 ms apart."""
    return [(i * 0.05, p) for i, p in enumerate(points)]


def test_amplitude_is_the_span_of_the_window_not_the_net_displacement():
    """An operator who moves the gripper and puts it back has moved it. Net
    displacement would score that zero; span scores it correctly."""
    there_and_back = _track((0, 0, 0), (0.2, 0, 0), (0, 0, 0))
    assert math.isclose(vv.amplitude(there_and_back), 0.2, abs_tol=1e-9)


def test_amplitude_of_a_motionless_puck_is_zero():
    assert vv.amplitude(_track((1.0, 2.0, 3.0), (1.0, 2.0, 3.0))) == 0.0


def test_amplitude_combines_all_three_axes():
    diagonal = _track((0, 0, 0), (0.3, 0.4, 0.0))
    assert math.isclose(vv.amplitude(diagonal), 0.5, abs_tol=1e-9)


def test_amplitude_of_an_empty_window_is_zero():
    assert vv.amplitude([]) == 0.0


def test_the_prompted_arm_moving_alone_and_held_confirms():
    verdict = vv.classify(
        {"left": 0.14, "right": 0.001}, prompted="left", held_s=1.0, thresholds=TH
    )
    assert verdict == "confirmed"


def test_the_same_state_held_too_briefly_confirms_nothing():
    """Sustained is what makes the verdict mean something -- a single frame of
    clean separation happens by accident."""
    verdict = vv.classify(
        {"left": 0.14, "right": 0.001}, prompted="left", held_s=0.4, thresholds=TH
    )
    assert verdict is None


def test_the_wrong_arm_moving_alone_is_a_cross():
    """The failure this step exists to catch: the operator moved the left
    gripper and the tracker bound to the right arm is what moved."""
    verdict = vv.classify(
        {"left": 0.001, "right": 0.14}, prompted="left", held_s=1.0, thresholds=TH
    )
    assert verdict == "cross"


def test_a_cross_also_needs_holding():
    verdict = vv.classify(
        {"left": 0.001, "right": 0.14}, prompted="left", held_s=0.4, thresholds=TH
    )
    assert verdict is None


def test_the_wrong_arm_at_exactly_the_move_threshold_still_crosses():
    """MOVE_M is one bar, not two. The prompted arm confirms at exactly MOVE_M,
    so an arm that is crossed has to cross at exactly MOVE_M -- an asymmetry
    here means a puck sitting precisely on the threshold reports nothing at all
    rather than the swap it is showing."""
    verdict = vv.classify(
        {"left": 0.001, "right": vv.MOVE_M}, prompted="left", held_s=1.0, thresholds=TH
    )
    assert verdict == "cross"


def test_both_arms_moving_is_not_a_cross():
    """A two-handed operator, or a bench knock. Requiring mutual exclusivity is
    what keeps an accidental bump from reading as swapped keys."""
    verdict = vv.classify(
        {"left": 0.14, "right": 0.14}, prompted="left", held_s=2.0, thresholds=TH
    )
    assert verdict is None


def test_both_arms_still_is_neither():
    verdict = vv.classify(
        {"left": 0.001, "right": 0.001}, prompted="left", held_s=2.0, thresholds=TH
    )
    assert verdict is None


def test_motion_between_the_two_thresholds_decides_nothing():
    """The gap between STILL_M and MOVE_M is deliberate dead space: a puck
    drifting 2 cm is neither convincingly moving nor convincingly still."""
    verdict = vv.classify(
        {"left": 0.03, "right": 0.001}, prompted="left", held_s=2.0, thresholds=TH
    )
    assert verdict is None


def test_an_untracked_arm_never_confirms():
    """A puck the server does not report as tracked has no position to judge,
    so its amplitude is None and no verdict can be reached."""
    verdict = vv.classify(
        {"left": None, "right": 0.001}, prompted="left", held_s=2.0, thresholds=TH
    )
    assert verdict is None


def test_a_third_arm_still_must_stay_still_for_a_confirm():
    verdict = vv.classify(
        {"left": 0.14, "right": 0.001, "head": 0.14},
        prompted="left",
        held_s=2.0,
        thresholds=TH,
    )
    assert verdict is None


def test_the_thresholds_are_unmistakable_rather_than_sensitive():
    """5 cm to count as moved and 1 cm to count as still, with a 5x gap. Raise
    STILL_M if a parked puck on your bench drifts more than a centimetre."""
    assert vv.MOVE_M == 0.05
    assert vv.STILL_M == 0.01
    assert vv.MOVE_M >= 5 * vv.STILL_M


def test_the_child_reads_both_arms_off_a_vive_config(tmp_path):
    config = tmp_path / "generated.yaml"
    config.write_text(
        (REPO / "config/data_recording/production/bimanual_v4_vive.yaml").read_text()
    )
    sources = vv.plan_sources(str(config))

    assert [s["arm"] for s in sources] == ["left", "right"]
    assert [s["tracker_key"] for s in sources] == ["left_knee", "right_knee"]
    assert {s["endpoint"] for s in sources} == {"tcp://127.0.0.1:5555"}


def test_a_config_with_no_vive_sources_plans_nothing(tmp_path):
    config = tmp_path / "generated.yaml"
    config.write_text(
        (REPO / "config/data_recording/production/bimanual_v4.yaml").read_text()
    )
    assert vv.plan_sources(str(config)) == []


def test_the_plot_is_skipped_with_no_display(monkeypatch):
    """matplotlib never raises headless -- it silently resolves to Agg, so a
    try/except guard would produce a happily-running figure drawing into
    nothing and calling plt.pause() twenty times a second."""
    monkeypatch.delenv("DISPLAY", raising=False)
    assert vv.open_plot(["left", "right"]) is None


def test_the_plot_is_skipped_when_the_backend_resolves_to_agg(monkeypatch):
    """The second guard, and it is not redundant with the DISPLAY one: the only
    interactive backend reachable here is TkAgg, so a station whose python
    lacks a working tkinter falls back to Agg even with DISPLAY set -- and
    would then draw into nothing exactly as the headless case does."""
    import matplotlib.pyplot as plt

    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(plt, "get_backend", lambda: "Agg")
    assert vv.open_plot(["left", "right"]) is None


# Drive the real watch loop to verify that dwell time belongs only to the
# current evidence classification.

TICK = 0.0625  # 1/16 s


class _Clock:
    """Deterministic stand-in for `time` inside `watch`.

    Advances by an exactly-representable TICK rather than the loop's requested
    1/POLL_HZ -- 0.05 is not a binary fraction, and these tests assert on the
    HOLD_S boundary itself, where accumulated drift is the difference between
    "fires on the first qualifying tick" and "fires one tick late".
    """

    def __init__(self):
        self.start = 1000.0
        self.now = self.start

    def monotonic(self):
        return self.now

    def sleep(self, _seconds):
        self.now += TICK


class _FakeBridge:
    """`start`, `stop` and `latest` -- all `watch` asks of a client."""

    def __init__(self, frames, clock):
        self._frames = frames
        self._clock = clock
        self.index = 0
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def latest(self):
        left_x, right_x = self._frames[self.index]
        self.index += 1
        payload = {"left_knee": _entry(left_x), "right_knee": _entry(right_x)}
        return payload, self._clock.monotonic()


def _entry(x):
    """One tracker's payload entry. `None` means the server is not tracking it."""
    if x is None:
        return {"tracked": False}
    return {"tracked": True, "position": {"x": x, "y": 0.0, "z": 0.0}}


SOURCES = [
    {
        "name": "left_tracking_pose",
        "arm": "left",
        "endpoint": "tcp://bench:5555",
        "tracker_key": "left_knee",
    },
    {
        "name": "right_tracking_pose",
        "arm": "right",
        "endpoint": "tcp://bench:5555",
        "tracker_key": "right_knee",
    },
]


def _run(monkeypatch, frames, walk=None):
    """Drive `watch` over a scripted bridge, one frame per tick.

    Returns `(exit_code, events, clock)`, where each event is
    `(clock_time, payload)` -- so a test can assert when a verdict landed and
    not merely that it did.

    `walk` is the arms to walk, defaulting to every source. Passing one arm is
    what the station console does per button; the bridge still reports both
    pucks either way, which is what keeps a cross detectable.
    """
    clock = _Clock()
    monkeypatch.setattr(vv, "time", clock)
    bridge = _FakeBridge(frames, clock)
    events = []

    class _Log(io.StringIO):
        def write(self, text):
            if text.startswith(vv.VIVE_PREFIX):
                body = text[len(vv.VIVE_PREFIX) :]
                events.append((clock.monotonic(), json.loads(body)))
            return super().write(text)

    code = vv.watch(
        SOURCES,
        _Log(),
        should_stop=lambda: bridge.index >= len(frames),
        walk=walk,
        make_client=lambda **_kwargs: bridge,
    )
    return code, events, clock


def _of(events, name):
    return [(t, event) for t, event in events if event["event"] == name]


def test_a_confirm_lands_exactly_one_hold_after_the_shape_appears(monkeypatch):
    """The left puck moves and the right stays parked, so the shape is a
    confirm from the second tick -- the first has a single sample and therefore
    no span. The verdict must arrive one HOLD_S later, and not a tick before."""
    frames = [(0.1 if tick % 2 else 0.0, 0.0) for tick in range(24)]
    _code, events, clock = _run(monkeypatch, frames)

    confirmed = _of(events, "confirmed")
    assert [event["arm"] for _t, event in confirmed] == ["left"]

    # The shape first appears on tick 1, the first tick whose window holds two
    # distinct positions.
    dwell = confirmed[0][0] - (clock.start + TICK)
    assert dwell >= vv.HOLD_S
    assert dwell < vv.HOLD_S + TICK


def test_a_one_tick_dropout_pushes_the_confirm_out_rather_than_letting_it_through(
    monkeypatch,
):
    """HOLD_S means "continuously", so a puck the server drops for a single
    tick restarts the clock. Nine ticks of clean confirm plus a blink must not
    add up to a confirm on the original schedule."""
    frames = [(0.1 if tick % 2 else 0.0, 0.0) for tick in range(32)]
    frames[9] = (None, 0.0)  # the left puck blinks out for exactly one tick
    _code, events, clock = _run(monkeypatch, frames)

    confirmed = _of(events, "confirmed")
    assert [event["arm"] for _t, event in confirmed] == ["left"]

    # Nothing before a full HOLD_S measured from the dropout itself, and
    # strictly later than the unblinked run above would have delivered it.
    assert confirmed[0][0] >= clock.start + 9 * TICK + vv.HOLD_S
    assert confirmed[0][0] > clock.start + 17 * TICK


def test_a_shape_change_does_not_inherit_the_previous_shape_s_dwell(monkeypatch):
    """The misfire this ordering exists to prevent.

    The left puck made one excursion and has been still since; the right puck
    is parked. That reads as a confirm until the excursion ages out of the
    one-second window, which happens on the very tick the confirm's own dwell
    matures -- and on that same tick the right puck catches a reflection.

    Reading the verdict before updating the timer would let the fresh cross
    inherit the confirm's accumulated second and fire instantly, telling an
    operator with a correct config that their tracker keys were swapped.
    """
    # Left: one excursion on tick 0, still afterwards -- so its window spans
    # 0.1 m until tick 0 falls out, and 0 from tick 17 on.
    # Right: parked until tick 17, then moving.
    frames = [(0.1 if tick == 0 else 0.0, 0.0) for tick in range(21)]
    for tick in range(17, 21):
        frames[tick] = (0.0, 0.2)
    _code, events, _clock = _run(monkeypatch, frames)

    assert _of(events, "cross") == []
    assert _of(events, "confirmed") == []

    # Not vacuous: the run really did present swapped-looking evidence -- the
    # prompted arm still, the other one plainly moving -- it just had not held
    # it for HOLD_S yet.
    _t, last_sample = _of(events, "sample")[-1]
    assert last_sample["arms"]["left"]["disp"] == 0.0
    assert last_sample["arms"]["right"]["disp"] >= vv.MOVE_M


def test_a_cross_at_exactly_the_move_threshold_names_the_arm_instead_of_crashing(
    monkeypatch,
):
    """MOVE_M is one bar, and both halves of the loop have to agree on it.

    `classify` calls a cross at `>= MOVE_M`, and the lookup that turns that
    verdict into an arm name must use the same bar: with `> MOVE_M`, an arm
    resting exactly on the bar would produce a cross that matched nothing and
    `next` would raise StopIteration -- killing the check at the instant it had
    its answer, after the operator had held the evidence for a full second.
    """
    # Only meaningful if the run really does sit ON the bar rather than near
    # it: one axis, so `amplitude` reduces to sqrt(v*v), which is exact.
    on_the_bar = vv.amplitude([(0.0, (0.0, 0.0, 0.0)), (1.0, (vv.MOVE_M, 0.0, 0.0))])
    assert on_the_bar == vv.MOVE_M

    # Left is prompted and parked; right -- the wrong arm -- swings exactly
    # MOVE_M, the signature of a swapped pair of tracker keys.
    frames = [(0.0, vv.MOVE_M if tick % 2 else 0.0) for tick in range(20)]
    code, events, _clock = _run(monkeypatch, frames)

    crossed = _of(events, "cross")
    assert [event["moved"] for _t, event in crossed] == ["right"]
    assert [event["prompted"] for _t, event in crossed] == ["left"]
    assert code == 1


# Per-arm walking reduces the confirmation queue, not the set of observed
# trackers needed to detect swapped keys.


def test_walking_one_arm_prompts_and_confirms_only_that_arm(monkeypatch):
    """The left puck moves alone, so the run confirms left and returns. It must
    never go on to prompt for the right: that arm has its own button, and its
    own run, started when the operator is ready for it."""
    frames = [(0.1 if tick % 2 else 0.0, 0.0) for tick in range(24)]
    code, events, _clock = _run(monkeypatch, frames, walk=["left"])

    assert [event["arm"] for _t, event in _of(events, "prompt")] == ["left"]
    assert [event["arm"] for _t, event in _of(events, "confirmed")] == ["left"]
    assert code == 0


def test_walking_one_arm_still_watches_the_other_for_a_cross(monkeypatch):
    """The reason `walk` is not simply a shorter source list.

    Splitting the check per arm would be worthless if a one-arm run only read
    one puck: the operator moves the left gripper, the right tracker is what
    moves, and a run watching only left sees a still puck and waits forever
    rather than reporting the swap. The bridge's reply carries every tracker, so
    a run walking left still compares it against right and fails the same way
    a whole-rig run does.
    """
    frames = [(0.0, 0.1 if tick % 2 else 0.0) for tick in range(20)]
    code, events, _clock = _run(monkeypatch, frames, walk=["left"])

    crossed = _of(events, "cross")
    assert [event["prompted"] for _t, event in crossed] == ["left"]
    assert [event["moved"] for _t, event in crossed] == ["right"]
    assert code == 1


def test_the_ready_line_names_every_puck_watched_not_just_the_walked_one(
    monkeypatch,
):
    """The panel seeds its rows and its bars off `ready`. A one-arm run that
    announced one arm would blank the other's row mid-check -- and the keys on
    that line are what the operator eyeballs against the bridge."""
    frames = [(0.0, 0.0) for _ in range(4)]
    _code, events, _clock = _run(monkeypatch, frames, walk=["right"])

    (_t, ready) = _of(events, "ready")[0]
    assert ready["arms"] == ["left", "right"]
    assert ready["keys"] == {"left": "left_knee", "right": "right_knee"}
    # And the arm actually being asked for is the walked one, not the first
    # source: a `--arm right` run that prompted left would have the operator
    # moving the wrong gripper.
    assert [event["arm"] for _t, event in _of(events, "prompt")] == ["right"]


def test_walking_every_arm_is_what_no_flag_means(monkeypatch):
    """`walk=None` is the by-hand bench invocation: one run that walks the
    whole rig. The console is what passes a single arm."""
    frames = [(0.1 if tick % 2 else 0.0, 0.0) for tick in range(24)]
    _code, events, _clock = _run(monkeypatch, frames, walk=None)

    prompts = [event["arm"] for _t, event in _of(events, "prompt")]
    assert prompts == ["left", "right"]


def test_the_client_is_stopped_even_though_the_loop_was_cut_short(monkeypatch):
    frames = [(0.0, 0.0) for _ in range(4)]
    clock = _Clock()
    monkeypatch.setattr(vv, "time", clock)
    bridge = _FakeBridge(frames, clock)

    vv.watch(
        SOURCES,
        io.StringIO(),
        should_stop=lambda: bridge.index >= len(frames),
        make_client=lambda **_kwargs: bridge,
    )

    assert bridge.started is True
    assert bridge.stopped is True


class _FakePlot:
    """`closed()` -- all `watch` asks of a plot beyond `update`/`close`."""

    def __init__(self, closed_after=None):
        self._tick = 0
        self._closed_after = closed_after

    def update(self, *_args, **_kwargs):
        self._tick += 1

    def close(self):
        pass

    def closed(self):
        return self._closed_after is not None and self._tick >= self._closed_after


def test_closing_the_plot_window_stops_the_loop_like_should_stop_does(monkeypatch):
    """'q' is matplotlib's own close-window binding, not a key this loop reads
    -- unlike the camera check's `cv2.waitKey`, closing the figure raises
    nothing `watch` would see on its own. Without polling `plot.closed()` the
    loop keeps prompting a window nobody can see until the console's 5-minute
    timeout, and the station's Start button for this arm stays swapped for
    Stop the whole time."""
    frames = [(0.0, 0.0) for _ in range(20)]
    clock = _Clock()
    monkeypatch.setattr(vv, "time", clock)
    bridge = _FakeBridge(frames, clock)
    plot = _FakePlot(closed_after=3)

    code = vv.watch(
        SOURCES,
        io.StringIO(),
        should_stop=lambda: False,
        plot=plot,
        make_client=lambda **_kwargs: bridge,
    )

    # Stopped well short of the 20 scripted frames -- the closed window ended
    # the check, not exhausting the bridge's script.
    assert bridge.index < len(frames)
    # Neither arm was confirmed, so this is the same "operator stopped the
    # check" exit `should_stop()` already produces, not the swapped-key failure
    # (1) some other verdict would return.
    assert code == 1


def test_sigterm_is_handled_rather_than_left_to_kill_the_child():
    """`jobs.cancel` is SIGTERM-first and the child runs in its own session, so
    the default disposition would tear the process down without unwinding --
    releasing the ZMQ socket and poll thread only because the image went away.

    The handler is invoked directly rather than by signalling this process: a
    regression here would otherwise kill the pytest run itself, and a test that
    takes down the runner when it fails tells you nothing."""
    previous = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    try:
        should_stop = vv.install_stop_on_sigterm()
        handler = signal.getsignal(signal.SIGTERM)
        assert handler not in (signal.SIG_DFL, signal.SIG_IGN)

        assert should_stop() is False
        handler(signal.SIGTERM, None)
        assert should_stop() is True
    finally:
        signal.signal(signal.SIGTERM, previous[0])
        signal.signal(signal.SIGINT, previous[1])

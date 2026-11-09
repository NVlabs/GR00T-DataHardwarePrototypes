# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`--watch`: one open per physical device, and one event per change.

Watching every camera at once opens each OAK-FFC board once, with both streams
in one pipeline, and reports a camera that decodes another side's payload as
exactly that rather than as an indistinguishable timeout.

The events are deliberately sparse. One line per decoded frame would overrun
the job buffer that holds them, which trims from the front -- so a check left
running would evict the `open_failed` lines emitted at startup and a broken rig
would read back as "Not started".
"""

from __future__ import annotations

import io
import json
import pathlib
import signal
import sys
import time

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from real_scripts.device_verification import (  # noqa: E402
    verify_device_assignment as verify,
)

FFC_CONFIG = {
    "meta": {"use_head_mount": False},
    "video_sources": [
        {
            "type": "oak_device",
            "oak_device_name": "OAK-FFC-LEFT",
            "fps": 30,
            "latency": 0.0875,
            "streams": [
                {
                    "name": "left_umi_camera",
                    "socket": "CAM_A",
                    "resolution": [640, 480],
                },
                {
                    "name": "left_trigger_camera",
                    "socket": "CAM_B",
                    "resolution": [640, 480],
                },
            ],
        },
        {
            "type": "iphone_camera",
            "name": "left_tracking_camera",
            "params": {"iphone_udid": "UDID-LEFT"},
        },
    ],
}


@pytest.fixture
def ffc_config(tmp_path):
    path = tmp_path / "recording.yaml"
    path.write_text(yaml.safe_dump(FFC_CONFIG))
    return path


def test_two_streams_of_one_board_become_a_single_device_open(ffc_config):
    """One open per board.

    left_umi and left_trigger are two streams of one OAK-FFC. An exclusive
    open per role would open that board twice, back to back, and never prove
    the two streams work together the way recording uses them.
    """
    groups = verify.plan_devices(
        ffc_config, ["left_umi", "left_trigger", "left_tracking"]
    )

    oak = [g for g in groups if g.camera_type == "oak_device_stream"]
    assert len(oak) == 1
    assert set(oak[0].roles) == {"left_umi", "left_trigger"}
    assert len(oak[0].streams) == 2
    assert {s["socket"] for s in oak[0].streams} == {"CAM_A", "CAM_B"}

    iphone = [g for g in groups if g.camera_type == "iphone"]
    assert len(iphone) == 1
    assert iphone[0].roles == ("left_tracking",)


def test_a_role_the_config_does_not_declare_is_planned_as_nothing(ffc_config):
    """A skipped camera is reported: a silent skip turns the gate into a no-op."""
    groups = verify.plan_devices(ffc_config, ["right_umi"])
    assert groups == []


def test_watch_names_the_payload_each_camera_decoded():
    """The worker is stateless about sides: it reports what was decoded, and
    the server decides whether that is the right code for that camera."""
    group = verify.DeviceGroup(
        key=("iphone", "UDID-LEFT"),
        camera_type="iphone",
        kwargs={"iphone_udid": "UDID-LEFT"},
        roles=("left_tracking",),
        streams=(),
    )

    class FakeWatcher:
        started = False

        def start(self):
            FakeWatcher.started = True

        def frames(self):
            yield ("left_tracking", "frame-1")

        def stop(self):
            pass

    out = io.StringIO()
    ticks = iter([False, True])
    verify.watch(
        [group],
        out,
        make_watcher=lambda g: FakeWatcher(),
        decode=lambda frame: "UMI_VERIFY_LEFT",
        should_stop=lambda: next(ticks),
    )

    events = _events(out.getvalue())
    assert {"event": "ready", "roles": ["left_tracking"]} in events
    assert {
        "event": "saw",
        "role": "left_tracking",
        "payload": "UMI_VERIFY_LEFT",
    } in events


def test_a_device_that_will_not_open_is_reported_and_does_not_stop_the_others():
    """One dead iPhone must not take the OAK board's check down with it."""
    dead = verify.DeviceGroup(
        key=("iphone", "UDID-DEAD"),
        camera_type="iphone",
        kwargs={},
        roles=("left_tracking",),
        streams=(),
    )
    live = verify.DeviceGroup(
        key=("uvc", 0),
        camera_type="uvc",
        kwargs={"device_id": 0},
        roles=("right_trigger",),
        streams=(),
    )

    class LiveWatcher:
        def start(self):
            pass

        def frames(self):
            yield ("right_trigger", "frame")

        def stop(self):
            pass

    def make(group):
        if group.roles == ("left_tracking",):
            raise RuntimeError("Record3D: device is locked")
        return LiveWatcher()

    out = io.StringIO()
    ticks = iter([False, True])
    verify.watch(
        [dead, live],
        out,
        make_watcher=make,
        decode=lambda frame: "UMI_VERIFY_RIGHT",
        should_stop=lambda: next(ticks),
    )

    events = _events(out.getvalue())
    failures = [e for e in events if e["event"] == "open_failed"]
    assert len(failures) == 1
    assert failures[0]["role"] == "left_tracking"
    assert "locked" in failures[0]["error"]
    assert any(e["event"] == "saw" and e["role"] == "right_trigger" for e in events)


def test_watch_never_imports_a_display(ffc_config, monkeypatch):
    """The marker is shown in the browser. A window here would open on a
    monitor the operator has walked away from."""
    import cv2

    def explode(*args, **kwargs):
        raise AssertionError("watch mode must not draw a window")

    monkeypatch.setattr(cv2, "imshow", explode)
    monkeypatch.setattr(cv2, "namedWindow", explode)

    group = verify.DeviceGroup(
        key=("uvc", 0),
        camera_type="uvc",
        kwargs={"device_id": 0},
        roles=("left_trigger",),
        streams=(),
    )

    class Watcher:
        def start(self):
            pass

        def frames(self):
            yield ("left_trigger", "frame")

        def stop(self):
            pass

    ticks = iter([False, True])
    verify.watch(
        [group],
        io.StringIO(),
        make_watcher=lambda g: Watcher(),
        decode=lambda frame: None,
        should_stop=lambda: next(ticks),
    )


# ------------------------------------------------- one event per CHANGE, not frame


class _ParkedWatcher:
    """A camera aimed at one code, returning a frame on every pass."""

    def __init__(self, role="left_tracking"):
        self.role = role

    def start(self):
        pass

    def frames(self):
        yield (self.role, "frame")

    def stop(self):
        pass


def _watch_for(ticks, **kwargs):
    """Run `watch` for `ticks` passes over one parked camera."""
    group = verify.DeviceGroup(
        key=("iphone", "UDID-LEFT"),
        camera_type="iphone",
        kwargs={"iphone_udid": "UDID-LEFT"},
        roles=("left_tracking",),
        streams=(),
    )
    out = io.StringIO()
    stops = iter([False] * ticks + [True])
    verify.watch(
        [group],
        out,
        make_watcher=lambda g: _ParkedWatcher(),
        should_stop=lambda: next(stops),
        **kwargs,
    )
    return _events(out.getvalue())


def test_a_camera_parked_on_one_code_is_reported_exactly_once():
    """Emitting per decoded frame would fill the job buffer and evict the
    check's own failures.

    `station_ui/jobs.py` keeps 20,000 lines and trims from the front, while
    `open_failed` is emitted once, at startup, so it is the first line evicted.
    At ~30 lines/second a check left running for twenty minutes -- which
    `CHECK_TIMEOUT_S = None` exists to allow -- would lose its own `open_failed`
    lines: a camera that would not open would revert from `Would not open` to
    `Waiting` and the step badge would go from `Failed` back to `Not started`.
    """
    events = _watch_for(30, decode=lambda frame: "UMI_VERIFY_LEFT")

    saw = [e for e in events if e["event"] == "saw"]
    assert len(saw) == 1, f"one code, thirty passes, {len(saw)} events"
    assert saw[0] == {
        "event": "saw",
        "role": "left_tracking",
        "payload": "UMI_VERIFY_LEFT",
    }


def test_re_aiming_a_camera_is_still_reported():
    """Dedup must not cost the only retry this check has: an operator who
    pointed a camera at the wrong code fixes it by pointing it at the right
    one, and that correction has to reach the page."""
    codes = iter(
        ["UMI_VERIFY_RIGHT"] * 5 + ["UMI_VERIFY_LEFT"] * 5 + ["UMI_VERIFY_RIGHT"] * 5
    )
    events = _watch_for(14, decode=lambda frame: next(codes))

    assert [e["payload"] for e in events if e["event"] == "saw"] == [
        "UMI_VERIFY_RIGHT",
        "UMI_VERIFY_LEFT",
        "UMI_VERIFY_RIGHT",
    ]


def test_the_loop_yields_the_cpu_between_passes():
    """An OAK queue get blocks on its own, but a UVC or iPhone camera whose
    get_frame() returns None immediately would spin a core -- on a process
    that by design has no wall-clock cap."""
    assert verify.WATCH_POLL_S > 0
    start = time.monotonic()
    _watch_for(4, decode=lambda frame: None)
    assert time.monotonic() - start >= 3 * verify.WATCH_POLL_S


def test_sigterm_asks_the_loop_to_stop_rather_than_killing_the_interpreter():
    """Stop on the page is `jobs.cancel`, which is SIGTERM-first.

    Python's default disposition terminates without unwinding, so without a
    handler the `finally` that stops every camera would never run and the
    boards would be released only because the process died. With the handler
    it is the cleanup block that releases them.
    """
    previous = signal.getsignal(signal.SIGTERM)
    try:
        should_stop = verify.install_stop_on_sigterm()
        assert should_stop() is False
        signal.raise_signal(signal.SIGTERM)
        assert should_stop() is True
    finally:
        signal.signal(signal.SIGTERM, previous)


# ------------------------------------------------------------- device handles


def test_a_watcher_that_fails_partway_through_start_still_releases_the_board():
    """`_MultiStreamWatcher.start` opens the exclusive DepthAI connection and
    then starts each stream. A stream that raises leaves the board claimed for
    the life of a process that has no timeout, and the cleanup at the end of
    `watch` only walks the watchers that started."""
    stopped = []

    class HalfOpenWatcher:
        def start(self):
            raise RuntimeError("stream 3 would not start")

        def frames(self):
            yield from ()

        def stop(self):
            stopped.append("stopped")

    group = verify.DeviceGroup(
        key=("oak_device_stream", "OAK-FFC-LEFT"),
        camera_type="oak_device_stream",
        kwargs={"oak_device_name": "OAK-FFC-LEFT"},
        roles=("left_umi", "left_trigger"),
        streams=({"name": "left_umi_camera", "socket": "CAM_A"},),
    )
    out = io.StringIO()
    stops = iter([True])
    verify.watch(
        [group],
        out,
        make_watcher=lambda g: HalfOpenWatcher(),
        decode=lambda frame: None,
        should_stop=lambda: next(stops),
    )

    assert stopped == ["stopped"], "the board was never released"
    failed = [e for e in _events(out.getvalue()) if e["event"] == "open_failed"]
    assert {e["role"] for e in failed} == {"left_umi", "left_trigger"}


# ------------------------------------------------------- role i <- stream i


class _FakeFrame:
    def __init__(self, rgb):
        self.rgb = rgb


class _FakeStream:
    def __init__(self, socket):
        self.socket = socket
        self.streaming = False

    def start_streaming(self):
        self.streaming = True

    def get_frame(self):
        return _FakeFrame(f"frame-from-{self.socket}")

    def stop_streaming(self):
        self.streaming = False


class _FakeOAKDevice:
    """Records what it was built with, and hands back streams in config order,
    exactly as `OAKDevice.build_pipeline` / `get_streams` do."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.configs = []
        self.streams = []
        self.started = False
        self.stopped = False
        _FakeOAKDevice.instances.append(self)

    def build_pipeline(self, configs):
        self.configs = list(configs)
        self.streams = [_FakeStream(config["socket"]) for config in self.configs]

    def start(self):
        self.started = True

    def get_streams(self):
        return list(self.streams)

    def stop(self):
        self.stopped = True


@pytest.fixture
def fake_oak(monkeypatch):
    _FakeOAKDevice.instances = []
    monkeypatch.setattr(verify, "OAKDevice", _FakeOAKDevice)
    return _FakeOAKDevice


def _ffc_group():
    return verify.DeviceGroup(
        key=("oak_device_stream", "OAK-FFC-LEFT"),
        camera_type="oak_device_stream",
        kwargs={"oak_device_name": "OAK-FFC-LEFT", "fps": 30, "latency": 0.0875},
        roles=("left_umi", "left_trigger"),
        streams=(
            {"name": "left_umi_camera", "socket": "CAM_A", "resolution": [640, 400]},
            {
                "name": "left_trigger_camera",
                "socket": "CAM_B",
                "resolution": [640, 400],
            },
        ),
    )


def test_stream_i_answers_for_role_i(fake_oak):
    """The headline invariant of the fold: `build_pipeline` appends one stream
    per config in config order, and `plan_devices` builds `roles` and `streams`
    in lockstep. If that ever slipped, every camera on the board would be
    reported under its neighbour's name and the operator would be sent to fix a
    cross-wiring that is not there."""
    watcher = verify._MultiStreamWatcher(_ffc_group())
    watcher.start()

    assert list(watcher.frames()) == [
        ("left_umi", "frame-from-CAM_A"),
        ("left_trigger", "frame-from-CAM_B"),
    ]


def test_the_whole_board_is_opened_once_with_every_stream_in_one_pipeline(fake_oak):
    watcher = verify._MultiStreamWatcher(_ffc_group())
    watcher.start()

    assert len(fake_oak.instances) == 1, "the board was opened more than once"
    device = fake_oak.instances[0]
    assert device.started
    assert [config["socket"] for config in device.configs] == ["CAM_A", "CAM_B"]
    # The device's latency is what a stream records with unless it says otherwise.
    assert [config["latency"] for config in device.configs] == [0.0875, 0.0875]
    assert all(stream.streaming for stream in device.streams)


def test_stopping_releases_every_stream_and_the_board(fake_oak):
    watcher = verify._MultiStreamWatcher(_ffc_group())
    watcher.start()
    device = fake_oak.instances[0]
    watcher.stop()

    assert not any(stream.streaming for stream in device.streams)
    assert device.stopped
    # Idempotent: `watch` stops in a finally, and the failure path stops too.
    watcher.stop()


# --------------------------------------------------------- grouping identity


UNNAMED_TWO_BOARDS = {
    "meta": {"use_head_mount": False},
    "video_sources": [
        {
            "type": "oak_device",
            "fps": 30,
            "streams": [
                {
                    "name": "left_umi_camera",
                    "socket": "CAM_A",
                    "resolution": [640, 400],
                },
            ],
        },
        {
            "type": "oak_device",
            "fps": 30,
            "streams": [
                {
                    "name": "right_umi_camera",
                    "socket": "CAM_A",
                    "resolution": [640, 400],
                },
            ],
        },
    ],
}


def test_two_boards_that_carry_no_identifier_are_not_folded_together(tmp_path):
    """`None` is the absence of an identifier, not a shared one.

    Merging on it builds one pipeline on the first board and serves the second
    board's roles from the first board's sockets. Both boards use CAM_A, so
    nothing raises: the check simply oscillates forever, telling the operator
    two cameras are cross-wired when nothing is.
    """
    path = tmp_path / "unnamed.yaml"
    path.write_text(yaml.safe_dump(UNNAMED_TWO_BOARDS))

    groups = verify.plan_devices(path, ["left_umi", "right_umi"])

    assert len(groups) == 2, f"two unidentified boards became {len(groups)} group(s)"
    assert [group.roles for group in groups] == [("left_umi",), ("right_umi",)]


def test_one_camera_will_not_answer_for_two_roles():
    """`_SingleCameraWatcher.frames` can only ever yield roles[0], so a group of
    two would leave the second role Waiting for the whole check with the page
    pinned to its side and nothing saying why. Refusing reports both as
    open_failed, which is at least a fault the operator can read."""
    group = verify.DeviceGroup(
        key=("iphone", None),
        camera_type="iphone",
        kwargs={},
        roles=("left_tracking", "head"),
        streams=(),
    )

    with pytest.raises(ValueError) as raised:
        verify._SingleCameraWatcher(group)
    assert "one role" in str(raised.value)

    # And end to end: fail closed, naming both roles, not silently half-open.
    out = io.StringIO()
    verify.watch([group], out, decode=lambda frame: None, should_stop=lambda: True)
    failed = [e for e in _events(out.getvalue()) if e["event"] == "open_failed"]
    assert {e["role"] for e in failed} == {"left_tracking", "head"}


# ------------------------------------------------- one role table, two readers

GENERATED_HEAD_CONFIG = {
    "meta": {"use_head_mount": True},
    "video_sources": [
        {
            "type": "oak_device",
            "oak_device_name": "OAK-FFC-LEFT",
            "fps": 30,
            "streams": [
                {
                    "name": "left_umi_camera",
                    "socket": "CAM_A",
                    "resolution": [640, 400],
                },
            ],
        },
        {
            "type": "iphone_camera",
            "name": "head_iphone_camera",
            "params": {"iphone_udid": "UDID-HEAD"},
        },
        {
            "type": "oak_camera",
            "name": "head_oak_camera",
            "params": {"oak_device_name": "OAK-HEAD"},
        },
    ],
}


def test_the_server_lists_exactly_the_roles_the_worker_can_resolve(tmp_path):
    """The server's role list and the worker's source-name table are the same
    fact. If they drifted -- say the worker knew the names `/config` generates
    for a head mount (`head_iphone_camera` / `head_oak_camera`) and the server
    did not -- the child would open and decode the head camera, and the server
    would throw the answer away as belonging to a role it never listed,
    badging the step Done with that camera unverified."""
    from station_ui import calibration as cal

    path = tmp_path / "generated.yaml"
    path.write_text(yaml.safe_dump(GENERATED_HEAD_CONFIG))
    listed = cal.verification_roles(GENERATED_HEAD_CONFIG)

    assert "head" in listed and "head_umi" in listed
    for role in verify.ROLE_LABELS:
        resolvable = verify.resolve_camera_for_role(path, role)[0] is not None
        assert resolvable == (role in listed), (
            f"{role}: the worker "
            f"{'can' if resolvable else 'cannot'} verify it, the server "
            f"{'lists' if role in listed else 'does not list'} it"
        )


def _events(text):
    return [
        json.loads(line[len(verify.WATCH_PREFIX) :])
        for line in text.splitlines()
        if line.startswith(verify.WATCH_PREFIX)
    ]


# Test preview composition without opening windows; cover display calls
# through their failure guards.


def test_the_preview_follows_the_first_unverified_side():
    roles = ["left_umi", "left_trigger", "right_umi", "head"]
    # Nothing decoded yet: the left rig is first.
    assert verify._current_preview_side(roles, set()) == "left"
    # Both left cameras verified: move to the right.
    assert verify._current_preview_side(roles, {"left_umi", "left_trigger"}) == "right"
    # Everything verified: nothing left to aim.
    assert verify._current_preview_side(roles, set(roles)) is None


def test_a_camera_that_has_not_verified_keeps_its_side_current_in_the_preview():
    """A camera pointed at the wrong code is not verified, so its side stays
    current -- that is the side the operator can still fix where they stand."""
    roles = ["left_umi", "right_umi"]
    assert verify._current_preview_side(roles, {"right_umi"}) == "left"


def test_a_verified_side_is_not_reopened_by_a_later_glimpse_of_another_code():
    """The verdict latches, so the head preview opens.

    The cameras stay open and streaming after their side clears, so the
    right-hand rig decodes UMI_VERIFY_HEAD the moment the operator carries the
    code past it on the way to the head mount. Judged on "what did each camera
    last see", right would un-verify itself and this window would snap back to
    RIGHT -- so RIGHT would never close and HEAD would never open. The verdict
    latches, exactly as `station_ui.calibration.fold_events` latches on the
    server.
    """
    group = verify.DeviceGroup(
        key=("oak", "RIGHT-BOARD"),
        camera_type="oak",
        kwargs={},
        roles=("right_umi",),
        streams=(),
    )

    class FakeWatcher:
        def start(self):
            pass

        def frames(self):
            yield ("right_umi", "frame")

        def stop(self):
            pass

    # Its own code first, then the head code as the operator walks past it.
    payloads = iter(
        [verify.payload_for_role("right_umi"), verify.payload_for_role("head")]
    )
    rendered = []

    class Preview:
        def begin(self, roles):
            pass

        def render(self, latest_frame, verified):
            rendered.append(set(verified))

        def close(self):
            pass

    ticks = iter([False, False, True])
    verify.watch(
        [group],
        io.StringIO(),
        make_watcher=lambda g: FakeWatcher(),
        decode=lambda frame: next(payloads, None),
        should_stop=lambda: next(ticks),
        preview=Preview(),
    )

    assert rendered, "the preview was never rendered"
    assert all("right_umi" in state for state in rendered), (
        f"a verified camera was un-latched by a later code: {rendered}"
    )


def test_compose_preview_draws_the_current_side_and_nothing_when_done():
    import numpy as np

    roles = ["left_umi", "right_umi"]
    frames = {
        "left_umi": np.zeros((400, 640, 3), np.uint8),
        "right_umi": np.zeros((400, 640, 3), np.uint8),
    }
    # Left outstanding, a frame in hand -> a canvas at the tile height.
    canvas = verify._compose_preview(frames, set(), roles, lambda r: r)
    assert canvas is not None
    assert canvas.shape[0] == verify.PREVIEW_TILE_H

    # Everything verified -> nothing to draw (the check auto-stops anyway).
    assert verify._compose_preview(frames, set(roles), roles, lambda r: r) is None


def test_compose_preview_shows_a_starting_placeholder_until_a_frame_arrives():
    """The window must not be blank while the cameras are still opening: a side
    that is current but has no frame yet still draws a 'starting' placeholder."""
    roles = ["left_umi", "right_umi"]
    canvas = verify._compose_preview({}, set(), roles, lambda r: r)
    assert canvas is not None, "a starting side drew nothing"
    assert canvas.shape[0] == verify.PREVIEW_TILE_H


def test_the_preview_never_draws_a_qr_code():
    """The browser already serves the code the operator aims. A raster copy here
    would spend a third of the window on an image they are looking at anyway, on
    a monitor that is behind them while they aim it."""
    import numpy as np

    roles = ["left_umi"]
    frames = {"left_umi": np.zeros((480, 640, 3), np.uint8)}
    canvas = verify._compose_preview(frames, set(), roles, lambda r: r)
    # Exactly one tile wide -- the camera's, at its own aspect. A QR column would
    # add a square tile of PREVIEW_TILE_H on top of this.
    assert canvas.shape[1] == round(640 * verify.PREVIEW_TILE_H / 480)
    for helper in ("_qr_tile", "_qr_image"):
        assert not hasattr(verify, helper), f"{helper} is still here"


def test_three_cameras_wrap_to_a_second_row_so_each_one_is_bigger():
    """The tiles are laid out at most two per row, and that -- not the doubled
    tile height on its own -- is what makes each camera bigger.

    Three 4:3 cameras side by side on a 1920px bench monitor are 640px each
    however large the canvas is composed, because the monitor is the limit.
    Wrapping spends the screen's height too: 720x540 per camera rather than
    480x360.
    """
    import numpy as np

    roles = ["left_tracking", "left_umi", "left_trigger"]
    frames = {r: np.zeros((480, 640, 3), np.uint8) for r in roles}
    canvas = verify._compose_preview(frames, set(), roles, lambda r: r)
    # Two rows of tiles, not one.
    assert canvas.shape[0] == verify.PREVIEW_TILE_H * 2, "the tiles did not wrap"

    win_w, win_h = verify._window_size(canvas.shape[1], canvas.shape[0])
    per_camera_w = win_w / verify.PREVIEW_MAX_COLS
    assert per_camera_w > 640, (
        f"each camera is {per_camera_w:.0f}px wide -- no bigger than one row gave"
    )

    # A short last row is padded, never left ragged: the rows must vstack.
    assert canvas.shape[1] == verify.PREVIEW_TILE_H * 4 / 3 * verify.PREVIEW_MAX_COLS


def test_the_preview_window_is_fitted_to_the_monitor_not_to_the_canvas():
    """The tile height is doubled, so a three-camera side composes wider than a
    1080p bench monitor -- and WINDOW_NORMAL scales the image into whatever size
    the window already has, so a bigger canvas alone changes nothing on screen.
    The canvas stays full size (sharp when maximised); the window is bounded,
    aspect preserved."""
    assert verify.PREVIEW_TILE_H == 960, "the tile height is no longer doubled"
    wide = verify._window_size(verify.PREVIEW_TILE_H * 4, verify.PREVIEW_TILE_H)
    assert wide[0] <= verify.PREVIEW_MAX_W and wide[1] <= verify.PREVIEW_MAX_H
    assert abs(wide[0] / wide[1] - 4.0) < 0.01, "the aspect was not preserved"
    # A canvas that already fits is never blown up into a blurry frame.
    assert verify._window_size(640, 480) == (640, 480)
    assert verify._window_size(0, 0) is None


def test_the_preview_window_is_best_effort_and_never_raises(monkeypatch):
    """A station that cannot draw still runs the check: the event stream is the
    real job, and a preview that cannot render must never take it down.

    The window calls are forced to raise rather than left to whatever display
    the harness happens to have. That tests the property directly instead of
    hoping the environment is hostile -- and it is the only way this can run at
    all on a machine with a display, where cv2's GUI backend aborts the process
    at the C level on a bad window call. No `except` catches an abort, and an
    abort here would silently skip every test below it in this file.
    """

    def boom(*_args, **_kwargs):
        raise RuntimeError("no display")

    for name in ("namedWindow", "imshow", "waitKey", "destroyWindow", "resizeWindow"):
        if hasattr(verify.cv2, name):
            monkeypatch.setattr(verify.cv2, name, boom)

    window = verify._PreviewWindow(label_for=lambda r: r)
    window.begin(["left_umi"])
    window.render({}, set())  # must not raise
    window.close()  # must not raise


# ---- Reading the marker: ArUco first, QR fallback, centre-crop rescue --------


def _marker_image(side, *, scale=20, pad=60):
    """The picture the page shows, as a camera would see it: marker on white."""
    import numpy as np

    from station_ui import markers

    grid = markers.marker_matrix(side)
    span = len(grid) + 2
    img = np.full((span * scale, span * scale), 255, np.uint8)
    for y, row in enumerate(grid):
        for x, module in enumerate(row):
            if module == "1":
                top, left = (y + 1) * scale, (x + 1) * scale
                img[top : top + scale, left : left + scale] = 0
    return np.pad(img, pad, constant_values=255)


@pytest.mark.parametrize("side", ["left", "right", "head"])
def test_an_aruco_marker_decodes_to_the_same_payload_the_qr_carried(side):
    """A detected marker id is translated back into `UMI_VERIFY_<SIDE>`, so the
    event stream, the fold in `station_ui.calibration` and every test of them
    do not depend on which kind of marker is shown."""
    assert verify.decode_marker(_marker_image(side)) == f"UMI_VERIFY_{side.upper()}"


def test_a_frame_with_nothing_in_it_decodes_to_nothing():
    """Guard the guard: a detector that answered on blank frames would mark
    every camera verified without anybody aiming anything."""
    import numpy as np

    assert verify.decode_marker(np.full((400, 640), 255, np.uint8)) is None


def test_the_centre_crop_rescues_a_marker_the_edge_would_lose():
    """Barrel distortion is worst at the rim and near zero on the optical axis,
    so the second attempt is an upscaled centre. Simulated here by a marker too
    small to read at full frame but readable once the periphery -- most of the
    pixels and none of the information -- is cropped away."""
    import numpy as np

    frame = np.full((480, 640), 255, np.uint8)
    small = _marker_image("left", scale=6, pad=0)
    top = (480 - small.shape[0]) // 2
    left = (640 - small.shape[1]) // 2
    frame[top : top + small.shape[0], left : left + small.shape[1]] = small

    # Whether the full frame alone would have read it is not the assertion --
    # that it reads at all through the two-candidate path is.
    assert verify.decode_marker(frame) == "UMI_VERIFY_LEFT"


def test_browser_preview_writes_a_jpeg_without_any_window_calls(tmp_path, monkeypatch):
    import numpy as np

    def no_window(*args, **kwargs):
        pytest.fail("Browser calibration opened a native window")

    for name in ("namedWindow", "imshow", "waitKey", "resizeWindow", "destroyWindow"):
        monkeypatch.setattr(verify.cv2, name, no_window)
    path = tmp_path / "preview.jpg"
    preview = verify._PreviewFile(path, lambda role: role)
    preview.begin(["left_umi", "right_umi"])
    frame = np.full((120, 160, 3), 127, dtype=np.uint8)
    preview.render({"left_umi": frame}, set())
    assert path.read_bytes().startswith(b"\xff\xd8")
    image = verify.cv2.imread(str(path))
    assert image is not None and image.shape[1] <= 1280
    assert not (tmp_path / "preview.jpg.part").exists()
    preview.close()


def test_browser_preview_failure_does_not_stop_verification(
    tmp_path, monkeypatch, capsys
):
    preview = verify._PreviewFile(
        tmp_path / "missing" / "preview.jpg", lambda role: role
    )
    preview.begin(["left_umi"])
    preview.render({}, set())
    assert "[preview] unavailable" in capsys.readouterr().err
    preview.render({}, set())  # disabled preview must remain harmless
    preview.close()


@pytest.mark.parametrize(
    "flag,expected", [("--preview", "native"), ("--preview-file", "browser")]
)
def test_direct_cli_still_selects_native_preview_only_when_requested(
    flag, expected, monkeypatch, tmp_path
):
    monkeypatch.setattr(verify, "plan_devices", lambda *args: [object()])
    monkeypatch.setattr(verify, "_PreviewWindow", lambda **kw: "native")
    monkeypatch.setattr(verify, "_PreviewFile", lambda *args: "browser")
    received = []
    monkeypatch.setattr(
        verify, "watch", lambda *args, **kw: received.append(kw["preview"]) or 0
    )
    args = ["verify", "--config", "test.yaml", "--watch", flag]
    if flag == "--preview-file":
        args.append(str(tmp_path / "frame.jpg"))
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as exited:
        verify.main()
    assert exited.value.code == 0
    assert received == [expected]


def test_watch_cli_plans_only_roles_still_needing_verification(monkeypatch):
    planned = []
    monkeypatch.setattr(
        verify, "plan_devices", lambda path, roles: planned.append(roles) or [object()]
    )
    monkeypatch.setattr(verify, "watch", lambda *args, **kwargs: 0)
    monkeypatch.setattr(verify, "install_stop_on_sigterm", lambda: lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify", "--config", "test.yaml", "--watch", "--roles", "head_umi"],
    )
    with pytest.raises(SystemExit) as exited:
        verify.main()
    assert exited.value.code == 0
    assert planned == [["head_umi"]]

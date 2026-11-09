# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vive tracker: transport sharing, tracking validity, and drop behaviour.

Hardware-free by construction -- every test injects a fake client or a fake
socket, so nothing here opens a ZMQ connection to a real bridge.
"""

import json
import time

import numpy as np
import pytest

from umi.common.data.frame import PoseFrame
from umi.common.hardware_sdk.vive.client import ViveTrackerClient, reset_vive_clients
from umi.common.hardware_sdk.vive.tracker import ViveTracker, pose_dict_to_matrix


def _entry(x=0.0, tracked=None):
    entry = {
        "position": {"x": x, "y": 0.0, "z": 0.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }
    if tracked is not None:
        entry["tracked"] = tracked
    return entry


class _FakeClient:
    """Stands in for ViveTrackerClient: hands back (payload, arrival).

    `payload` is either a dict (the server keeps reporting the same thing, i.e.
    a frozen puck) or a callable(i) -> dict (it changes, i.e. a live one), or
    None (the server never answers). Modelled on what the real client does --
    hand back whatever the poller last stored -- rather than draining a queue,
    because a queue would silently make startup consume the samples a test
    meant for get_frame().
    """

    def __init__(self, payload):
        self._payload = payload
        self._i = 0
        self._arrival = 1000.0
        self.started = 0
        self.stopped = 0
        self.poll_failures = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def latest(self):
        if self._payload is None:
            return None
        self._i += 1
        self._arrival += 0.03
        payload = self._payload(self._i) if callable(self._payload) else self._payload
        return (payload, self._arrival)


def _tracker(client, **kw):
    kw.setdefault("startup_timeout_s", 1.0)
    return ViveTracker(
        device_name="left_tracking_pose", tracker_key="k", client=client, **kw
    )


# --------------------------------------------------------------------------
# pose conversion
# --------------------------------------------------------------------------


def test_pose_dict_to_matrix_builds_a_rigid_transform():
    m = pose_dict_to_matrix(_entry(x=0.5))
    assert m.shape == (4, 4)
    assert np.isclose(np.linalg.det(m[:3, :3]), 1.0)
    assert np.allclose(m[:3, 3], [0.5, 0.0, 0.0])


@pytest.mark.parametrize(
    "entry",
    [
        {"position": {"x": 0, "y": 0, "z": 0}},  # no orientation
        {
            "position": {"x": 0, "y": 0, "z": 0},
            "orientation": {"x": 0, "y": 0, "z": 0, "w": 0},  # degenerate quat
        },
        {
            "position": {"x": float("nan"), "y": 0, "z": 0},
            "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
        },
    ],
)
def test_pose_dict_to_matrix_rejects_unusable_entries(entry):
    assert pose_dict_to_matrix(entry) is None


# --------------------------------------------------------------------------
# tracking validity
# --------------------------------------------------------------------------


def test_server_tracked_flag_is_authoritative_and_refuses_to_start():
    client = _FakeClient({"k": _entry(0.5, tracked=False)})
    with pytest.raises(RuntimeError, match="reports it as NOT tracked"):
        _tracker(client).start_streaming()


def test_a_server_without_a_tracked_field_is_refused_outright():
    """Old servers are rejected, not worked around. One that forwards only the
    pose cannot report a puck that has lost tracking, so it would serve a stale
    pose that records as a flat, plausible trajectory."""
    client = _FakeClient({"k": _entry(0.5)})  # no `tracked`
    with pytest.raises(RuntimeError, match="does not publish a 'tracked' field"):
        _tracker(client).start_streaming()


def test_a_moving_puck_starts():
    client = _FakeClient(lambda i: {"k": _entry(i * 0.01, tracked=True)})
    tracker = _tracker(client)
    tracker.start_streaming()
    assert tracker.is_running
    assert client.started == 1


def test_missing_key_names_what_is_available():
    client = _FakeClient({"other": _entry(0.1, tracked=True)})
    with pytest.raises(RuntimeError, match=r"not in the payload.*other"):
        _tracker(client).start_streaming()


def test_empty_payload_is_reported_as_no_trackers_not_as_a_dead_server():
    client = _FakeClient({})
    with pytest.raises(RuntimeError, match="reported no trackers"):
        _tracker(client).start_streaming()


def test_silent_server_is_reported_as_no_response():
    client = _FakeClient(None)
    with pytest.raises(RuntimeError, match="No response"):
        _tracker(client).start_streaming()


def test_require_tracking_at_start_false_starts_but_records_nothing():
    """Never fabricate a pose: an untracked puck yields frames=0, not a frozen
    trajectory that reads downstream as a motionless gripper."""
    client = _FakeClient({"k": _entry(0.5, tracked=False)})
    tracker = _tracker(client, require_tracking_at_start=False)
    tracker.start_streaming()
    assert [tracker.get_frame() for _ in range(10)] == [None] * 10
    assert tracker.dropped_frames == 10


# --------------------------------------------------------------------------
# frames + drops
# --------------------------------------------------------------------------


def test_get_frame_stamps_capture_from_arrival_minus_latency():
    client = _FakeClient(lambda i: {"k": _entry(i * 0.01, tracked=True)})
    tracker = _tracker(client, latency=0.02)
    tracker.start_streaming()
    frame = tracker.get_frame()
    assert isinstance(frame, PoseFrame)
    assert frame.capture_time == pytest.approx(frame.receive_time - 0.02)


def test_losing_tracking_mid_stream_drops_rather_than_repeating_the_last_pose():
    state = {"tracked": True, "x": 0.0}

    def payload(_i):
        state["x"] += 0.01
        x = state["x"] if state["tracked"] else 0.99
        return {"k": _entry(x, tracked=state["tracked"])}

    tracker = _tracker(_FakeClient(payload))
    tracker.start_streaming()
    assert all(tracker.get_frame() is not None for _ in range(5))

    state["tracked"] = False  # puck loses line of sight mid-episode
    assert all(tracker.get_frame() is None for _ in range(5))
    assert tracker.dropped_frames == 5


def test_stale_sample_from_the_client_is_a_drop():
    """max_age_s expiry reaches the tracker as latest() -> None. That must be a
    drop, never a replay of the last pose."""
    tracker = _tracker(_FakeClient(lambda i: {"k": _entry(i * 0.01, tracked=True)}))
    tracker.start_streaming()
    tracker._client = _FakeClient(None)
    assert tracker.get_frame() is None
    assert tracker.dropped_frames == 1


def test_tracker_key_is_required():
    with pytest.raises(ValueError, match="tracker_key is required"):
        ViveTracker(device_name="left_tracking_pose", tracker_key="")


# --------------------------------------------------------------------------
# shared client
# --------------------------------------------------------------------------


class _FakeSocket:
    """Minimal REQ stand-in. `fail` makes recv raise like a timeout."""

    def __init__(self, payload, fail_first=0):
        import zmq

        self._zmq = zmq
        self._payload = payload
        self._fail_first = fail_first
        self.sends = 0
        self.closed = 0

    def send_string(self, _msg):
        self.sends += 1

    def recv_string(self):
        if self._fail_first > 0:
            self._fail_first -= 1
            raise self._zmq.Again()
        return json.dumps(self._payload)

    def close(self):
        self.closed += 1


def test_two_trackers_on_one_endpoint_share_a_client_and_a_socket():
    reset_vive_clients()
    sockets = []

    def factory():
        s = _FakeSocket(
            {"a": _entry(0.1, tracked=True), "b": _entry(0.2, tracked=True)}
        )
        sockets.append(s)
        return s

    try:
        a = ViveTracker(
            device_name="left_tracking_pose",
            tracker_key="a",
            endpoint="tcp://127.0.0.1:1",
            socket_factory=factory,
            startup_timeout_s=1.0,
        )
        b = ViveTracker(
            device_name="right_tracking_pose",
            tracker_key="b",
            endpoint="tcp://127.0.0.1:1",
            socket_factory=factory,
            startup_timeout_s=1.0,
        )
        a.start_streaming()
        b.start_streaming()
        assert a._client is b._client
        assert len(sockets) == 1, "each tracker opened its own socket"
    finally:
        reset_vive_clients()


def test_a_timeout_rebuilds_the_socket_so_the_next_poll_succeeds():
    """zmq.REQ wedges after a missed reply -- the next send raises EFSM. Without
    a rebuild, one dropped packet kills tracking for the whole session."""
    sockets = []

    def factory():
        s = _FakeSocket(
            {"k": _entry(0.1, tracked=True)}, fail_first=1 if not sockets else 0
        )
        sockets.append(s)
        return s

    client = ViveTrackerClient(
        endpoint="tcp://127.0.0.1:1", poll_hz=200, socket_factory=factory
    )
    client.start()
    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and client.latest() is None:
            time.sleep(0.01)
        assert client.latest() is not None, "never recovered from the timeout"
        assert len(sockets) >= 2, "socket was not rebuilt"
        assert client.poll_failures >= 1
    finally:
        client.force_stop()


def test_a_held_sample_expires_so_a_dead_server_never_serves_a_frozen_pose():
    def factory():
        return _FakeSocket({"k": _entry(0.1, tracked=True)})

    client = ViveTrackerClient(
        endpoint="tcp://127.0.0.1:1",
        poll_hz=200,
        max_age_s=0.05,
        socket_factory=factory,
    )
    client.start()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and client.latest() is None:
        time.sleep(0.01)
    assert client.latest() is not None
    client._stop_event.set()  # freeze the poller, leaving the sample in place
    time.sleep(0.3)
    assert client.latest() is None, "stale sample was served past max_age_s"
    client.force_stop()


def test_drops_are_silent_by_default(capsys):
    """Every other source reports drops through BaseRecorder's one path. This
    device must not add a second voice for the same event during a session."""
    tracker = _tracker(
        _FakeClient({"k": _entry(0.5, tracked=False)}), require_tracking_at_start=False
    )
    tracker.start_streaming()
    capsys.readouterr()  # discard startup lines
    for _ in range(10):
        tracker.get_frame()
    assert capsys.readouterr().out == ""
    assert tracker.dropped_frames == 10  # counted, just not narrated


def test_verbose_names_the_cause(capsys):
    tracker = _tracker(
        _FakeClient({"k": _entry(0.5, tracked=False)}),
        require_tracking_at_start=False,
        verbose=True,
    )
    tracker.start_streaming()
    capsys.readouterr()
    for _ in range(10):
        tracker.get_frame()
    out = capsys.readouterr().out
    assert "reported not tracked by the server" in out
    assert out.count("dropped so far") == 1, "cause line should be rate-limited"

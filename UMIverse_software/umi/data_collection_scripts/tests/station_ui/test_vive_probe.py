# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One request to the bridge, and nothing left behind.

The Vive transport is designed around a cached, refcounted client with a
background poll thread (`get_vive_client`), which is right for the recorder and
wrong for a web page: a config page that used it would leave a poll thread
running past the request, against a bridge shared by every station, and the
cache would then hand that same misconfigured client to the next caller.

So this is a throwaway socket, closed before the call returns, and the last
test here is the one that matters most.
"""

import pytest

zmq = pytest.importorskip("zmq")

from station_ui.core.vive_probe import probe  # noqa: E402
from umi.common.hardware_sdk.vive import client as vive_client  # noqa: E402

PAYLOAD = (
    '{"left_knee": {"position": {"x": 0.1, "y": 0.2, "z": 0.3},'
    ' "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}, "tracked": true},'
    ' "right_knee": {"position": {"x": 1.1, "y": 1.2, "z": 1.3},'
    ' "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}, "tracked": false}}'
)


class FakeSocket:
    """Stands in for a REQ socket. Records that it was closed.

    `raises` fires from `connect` as well as from the request, because the two
    failures reach the probe by different routes: a socket handed over by
    `socket_factory` is never connected, while one the probe builds itself can
    die inside `_open` before `probe` has a reference to close.
    """

    def __init__(self, reply=PAYLOAD, raises=None):
        self.reply = reply
        self.raises = raises
        self.sent = []
        self.closed = False

    def setsockopt(self, *_args):
        pass

    def connect(self, _endpoint):
        if self.raises is not None:
            raise self.raises

    def send_string(self, message):
        if self.raises is not None:
            raise self.raises
        self.sent.append(message)

    def recv_string(self):
        if self.raises is not None:
            raise self.raises
        return self.reply

    def close(self):
        self.closed = True


def test_a_healthy_bridge_reports_every_key_and_which_are_tracked():
    socket = FakeSocket()
    result = probe("tcp://fake:5555", socket_factory=lambda: socket)

    assert result.ok
    assert result.keys == ("left_knee", "right_knee")
    assert result.tracked == frozenset({"left_knee"})
    assert result.error == ""
    assert socket.sent == [vive_client.REQUEST_COMMAND]


def test_the_socket_is_always_closed():
    socket = FakeSocket()
    probe("tcp://fake:5555", socket_factory=lambda: socket)
    assert socket.closed, "the probe leaked a socket"


def test_the_socket_is_closed_even_when_the_request_fails():
    socket = FakeSocket(raises=zmq.Again())
    probe("tcp://fake:5555", socket_factory=lambda: socket)
    assert socket.closed, "the probe leaked a socket on the failure path"


def test_a_socket_that_cannot_connect_is_closed_before_the_error_escapes(monkeypatch):
    """The one leak `probe`'s `finally` cannot see.

    `_open` owns the socket until it returns, so a `connect` that raises leaves
    `probe`'s local `socket` at None and the `finally` closes nothing. Under
    CPython the refcount drop eventually reaches `zmq.Socket.__del__`, so the
    socket is reclaimed in practice -- but deterministic close is this module's
    headline guarantee, and delegating it to the garbage collector makes that
    guarantee false the moment anyone retains the traceback.

    This is the only test here that reaches `_open`'s real branch, so the
    context is faked rather than the socket: nothing may open a real one.
    """
    socket = FakeSocket(raises=zmq.ZMQError(zmq.EINVAL))

    class FakeContext:
        def socket(self, _kind):
            return socket

    monkeypatch.setattr(zmq.Context, "instance", classmethod(lambda cls: FakeContext()))

    # A bridge address with its scheme left off -- the real failure this
    # guards, and inert either way: libzmq refuses it at connect, so even a
    # future edit that dropped the fake could not reach the network.
    result = probe("192.168.1.20:5555")

    assert socket.closed, "the probe leaked a socket that failed to connect"
    assert not result.ok


def test_a_timeout_is_reported_not_raised():
    result = probe(
        "tcp://fake:5555", socket_factory=lambda: FakeSocket(raises=zmq.Again())
    )

    assert not result.ok
    assert result.keys == ()
    assert "tcp://fake:5555" in result.error
    assert "did not answer" in result.error


def test_undecodable_json_is_reported_not_raised():
    result = probe(
        "tcp://fake:5555", socket_factory=lambda: FakeSocket(reply="not json")
    )
    assert not result.ok
    assert "tcp://fake:5555" in result.error


@pytest.mark.parametrize(
    ("endpoint", "errno"),
    [
        # An address with its scheme left off, and one with the scheme but no
        # port. libzmq calls both EINVAL, at connect, before a packet moves.
        ("192.168.1.20:5555", zmq.EINVAL),
        ("tcp://192.168.1.20", zmq.EINVAL),
        # A transport libzmq does not implement is a different errno (93), which
        # is why this branch cannot key on EINVAL alone.
        ("wat://192.168.1.20:5555", zmq.EPROTONOSUPPORT),
    ],
)
def test_an_unusable_address_blames_the_address_not_the_bridge(endpoint, errno):
    """The likeliest operator mistake must not send them to check SteamVR.

    The bridge refused nothing here -- it was never dialled. Telling the
    operator it "refused the request" points them at the far end of a link that
    does not exist, when the fix is four characters in the box in front of them.
    """
    result = probe(
        endpoint, socket_factory=lambda: FakeSocket(raises=zmq.ZMQError(errno))
    )

    assert not result.ok
    assert endpoint in result.error
    assert "refused" not in result.error
    assert "tcp://" in result.error, "the operator is not shown the shape to type"


def test_a_genuine_socket_error_still_reads_as_the_bridge_refusing():
    """The address branch must split off the address errnos and no others."""
    result = probe(
        "tcp://fake:5555",
        socket_factory=lambda: FakeSocket(raises=zmq.ZMQError(zmq.EHOSTUNREACH)),
    )

    assert not result.ok
    assert "refused the request" in result.error


def test_a_bridge_serving_nothing_is_reachable_not_broken():
    """A reachable bridge with no pucks paired is `ok` with no keys.

    This is the distinction the page exists to draw, and the code draws it only
    because nothing collapses `ok` into `bool(keys)`. Told "the address is
    wrong", an operator whose SteamVR is up but empty goes hunting the network
    instead of pairing a tracker.
    """
    result = probe("tcp://fake:5555", socket_factory=lambda: FakeSocket(reply="{}"))

    assert result.ok
    assert result.keys == ()
    assert result.error == ""


def test_a_non_object_payload_is_refused():
    result = probe("tcp://fake:5555", socket_factory=lambda: FakeSocket(reply="[1, 2]"))
    assert not result.ok


def test_an_empty_endpoint_is_refused_without_opening_anything():
    opened = []

    def factory():
        opened.append(1)
        return FakeSocket()

    result = probe("", socket_factory=factory)

    assert not result.ok
    assert opened == [], "the probe opened a socket for an empty endpoint"


def test_a_server_that_publishes_no_tracked_flag_reports_the_key_untracked():
    """The recorder refuses a server too old to publish `tracked`. The probe is
    softer -- it is a discovery aid, and telling the operator which keys exist
    is still useful -- but it must not claim a puck is live."""
    result = probe(
        "tcp://fake:5555",
        socket_factory=lambda: FakeSocket(reply='{"left_knee": {"position": {}}}'),
    )

    assert result.ok
    assert result.keys == ("left_knee",)
    assert result.tracked == frozenset()


def test_the_probe_never_touches_the_shared_client_cache():
    """The most important test in this file. get_vive_client mutates a
    module-level dict on every miss and never evicts, so a probe that went
    through it would leave a poll thread hammering the shared bridge after the
    request ended.

    The endpoint is unique to this test on purpose. The cache is keyed by
    endpoint and only grows on a miss, so probing the address every other test
    here uses would make this assertion order-dependent: whichever test ran
    first would take the miss, `before` would already carry the entry, and a
    probe wired through `get_vive_client` would sail past a whole-file run --
    caught only when this test happened to run alone. With the probe
    deliberately routed through `get_vive_client`, the shared address passes
    here and this one fails.
    """
    endpoint = "tcp://cache-canary:5555"
    assert endpoint not in vive_client._CLIENT_CACHE

    before = dict(vive_client._CLIENT_CACHE)
    probe(endpoint, socket_factory=lambda: FakeSocket())
    assert vive_client._CLIENT_CACHE == before

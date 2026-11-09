# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ask the Vive bridge, once, what it is serving.

The configuration form needs two things the operator cannot know by looking:
whether the address they typed answers at all, and what the server currently
calls each puck. The key names a bridge serves can change between sessions
(`left_knee`/`right_knee` one day, `left_shoulder`/`right_shoulder` the next),
so the second question matters.

One request, no state. `umi.common.hardware_sdk.vive.client.get_vive_client`
exists for the recorder and is exactly wrong here: it returns a cached,
refcounted client with a background poll thread out of a module-level dict that
is never evicted. A page that called it would leave that thread polling a bridge
shared by every station long after the request ended, and hand the same client
to whoever asked next. So this opens its own socket, asks once, and closes it.

Nothing here raises. The caller is a route rendering a page, and every failure
-- a refused connection, a timeout, a server speaking something other than JSON
-- is a sentence the operator can act on.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import zmq

from umi.common.hardware_sdk.vive.client import REQUEST_COMMAND

__all__ = ["DEFAULT_PROBE_TIMEOUT_S", "ProbeResult", "probe"]

# A manual bridge probe gets a longer timeout than the 0.25 s recording
# poll because it has no immediate retry loop.
DEFAULT_PROBE_TIMEOUT_S = 1.0

# Treat EINVAL and EPROTONOSUPPORT as endpoint-format errors. An unreachable
# but well-formed endpoint connects asynchronously and instead times out
# with Again while waiting for a reply.
_UNUSABLE_ADDRESS_ERRNOS = frozenset({zmq.EINVAL, zmq.EPROTONOSUPPORT})


@dataclass(frozen=True)
class ProbeResult:
    """What one round trip to the bridge revealed.

    `ok` is about the transport: the server answered with a JSON object. A
    server that answers with an empty object is `ok` with no keys, which is a
    real and different answer from "the address is wrong".
    """

    ok: bool
    keys: tuple[str, ...] = ()
    # The subset reporting `tracked: true`. A key that is published but not
    # tracked is a puck that is asleep, occluded, or unpaired -- worth showing
    # differently from one that is live.
    tracked: frozenset[str] = field(default_factory=frozenset)
    rtt_ms: float = 0.0
    error: str = ""


def _open(endpoint: str, timeout_s: float, socket_factory):
    if socket_factory is not None:
        return socket_factory()
    socket = zmq.Context.instance().socket(zmq.REQ)
    try:
        socket.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
        socket.setsockopt(zmq.SNDTIMEO, int(timeout_s * 1000))
        # So closing never blocks on an undelivered message.
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(endpoint)
    except Exception:
        # Close the socket on setup failure before ownership can pass to the caller.
        socket.close()
        raise
    return socket


def probe(
    endpoint: str,
    timeout_s: float = DEFAULT_PROBE_TIMEOUT_S,
    socket_factory=None,
) -> ProbeResult:
    """One REQ/REP round trip. Never raises.

    `socket_factory` is the injection seam, mirroring `ViveTrackerClient`'s --
    and it is not optional in tests: nothing in `tests/conftest.py` fences
    sockets, so a test without one opens a real connection to whatever address
    the template names.
    """
    endpoint = str(endpoint or "").strip()
    if not endpoint:
        return ProbeResult(ok=False, error="No bridge address to test.")

    socket = None
    started = time.monotonic()
    try:
        socket = _open(endpoint, timeout_s, socket_factory)
        socket.send_string(REQUEST_COMMAND)
        message = socket.recv_string()
    except zmq.Again:
        return ProbeResult(
            ok=False,
            error=f"{endpoint} did not answer within {timeout_s:g}s.",
        )
    except zmq.ZMQError as exc:
        if exc.errno in _UNUSABLE_ADDRESS_ERRNOS:
            return ProbeResult(
                ok=False,
                error=(
                    f"{endpoint} is not an address ZMQ can dial ({exc}). "
                    "A bridge address looks like tcp://192.168.1.20:5555."
                ),
            )
        return ProbeResult(ok=False, error=f"{endpoint} refused the request ({exc}).")
    except Exception as exc:  # noqa: BLE001 - a page must render whatever happened
        return ProbeResult(ok=False, error=f"{endpoint} could not be reached ({exc}).")
    finally:
        # Close the REQ socket on every return path to release its descriptor and
        # connection.
        if socket is not None:
            try:
                socket.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass

    rtt_ms = (time.monotonic() - started) * 1000.0

    try:
        payload = json.loads(message)
    except (ValueError, TypeError):
        return ProbeResult(
            ok=False,
            error=f"{endpoint} answered with something that is not JSON.",
        )
    if not isinstance(payload, dict):
        return ProbeResult(
            ok=False,
            error=(
                f"{endpoint} answered with a {type(payload).__name__}, "
                "not the object of trackers this bridge is meant to serve."
            ),
        )

    keys = tuple(payload)
    # Only tracked: true establishes a live pose; OpenXR validity can remain
    # set after tracking loss. Discovery still lists entries with missing flags,
    # but must report them as untracked.
    tracked = frozenset(
        key
        for key, entry in payload.items()
        if isinstance(entry, dict) and entry.get("tracked") is True
    )
    return ProbeResult(ok=True, keys=keys, tracked=tracked, rtt_ms=rtt_ms)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ZMQ transport for the external Vive tracker server.

The server (``real_scripts/vive_server/vive_server.py``) runs on a separate
tracker host. It speaks REQ/REP only: send the literal string
``"get_vive_data"``, receive a JSON object keyed by tracker name, each value
``{"position": {x,y,z}, "orientation": {x,y,z,w}, "tracked": bool, "t": ...}``::

    {"left_knee":  {"position": {...}, "orientation": {...}, "tracked": true, ...},
     "right_knee": {"position": {...}, "orientation": {...}, "tracked": true, ...}}

Two design points, both deliberate:

**Polling runs on a background thread, not in the caller.** ``ViveTracker`` sits
behind ``base_recorder._stream_thread``, which calls ``get_frame()`` inside a
``RateKeeper`` at ``stream_fps``. If the round trip happened there, a single
socket timeout would stall the recorder's rate budget for two frame periods.
Here a timeout merely ages the held sample by one tick.

**One client per endpoint, shared by every tracker on it.** The server's reply
already carries every tracker, so N ``ViveTracker`` sources cost one round trip
between them, not N. Sharing also means the trackers read the same held sample,
so the arms are stamped from one server reply rather than from N round trips
milliseconds apart. This mirrors ``recording_utils._uart_readers_cache``, which
exists for the same reason (several UMI devices, one UART).

PUB/SUB would be the better transport (native-rate push, lossy instead of
lockstep, free fan-out), but it needs a PUB socket on the server. If the server
ever offers one, only ``_poll_once`` changes; ``latest()`` and every consumer
stay put.
"""

import json
import threading
import time
from typing import Any, Dict, Optional, Tuple

import zmq

from umi.common.utility.print_utils import print_green, print_red, print_yellow
from umi.common.utility.rate_keeper import RateKeeper

REQUEST_COMMAND = "get_vive_data"

# Seconds between repeats of the same warning, per client. Poll failures arrive
# at poll_hz, so unthrottled logging would bury the console.
WARN_INTERVAL_S = 2.0

# Check achieved rate periodically and warn when it cannot serve active consumers.
RATE_CHECK_EVERY = 60
RATE_SHORTFALL_RATIO = 0.8

# Share polling defaults with ViveTracker to keep source and client timing consistent.
DEFAULT_POLL_HZ = 90.0
# Allow for measured bridge round-trip times of approximately 43–55 ms.
DEFAULT_TIMEOUT_S = 0.25
# Set sample age above the achieved poll interval, not 1/poll_hz, so valid
# bridge samples do not expire before readers can use them.
DEFAULT_MAX_AGE_S = 0.15

# endpoint -> client. Module-level so several sources naming the same endpoint
# share one socket and one poll thread.
_CLIENT_CACHE: Dict[str, "ViveTrackerClient"] = {}
_CACHE_LOCK = threading.Lock()


class ViveTrackerClient:
    """Background REQ/REP poller holding the newest payload from one server.

    Args:
        endpoint: ZMQ address, e.g. ``"tcp://192.168.1.20:5555"``.
        poll_hz: Rate the background thread requests at. Must exceed the
            recorder's ``stream_fps`` (see the note in ``latest()``). This is a
            ceiling, not a guarantee: REQ/REP cannot poll faster than 1/RTT, so
            the loop warns if it cannot keep up (see ``_poll_loop``).
        timeout_s: Send/recv timeout for a single round trip. Must comfortably
            exceed the server's real RTT, which is not the network RTT: a live
            bridge was measured at 43-55 ms against a host that pings in 1.4 ms,
            so nearly all of it is server-side. A timeout at or below the RTT is
            pathological: every request times out, resets the socket, and the
            stream never produces a sample.
        max_age_s: A held sample older than this is considered stale and
            ``latest()`` returns None rather than serving it. Must exceed the
            achieved poll period or every sample expires before it is read.
        socket_factory: Injection seam for tests. Called with no arguments and
            must return an object with ``send_string`` / ``recv_string`` /
            ``close``. Defaults to a real ZMQ REQ socket.
    """

    def __init__(
        self,
        endpoint: str,
        poll_hz: float = DEFAULT_POLL_HZ,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_age_s: float = DEFAULT_MAX_AGE_S,
        socket_factory: Optional[Any] = None,
        expected_read_hz: float = 0.0,
    ) -> None:
        self.endpoint = endpoint
        self.poll_hz = float(poll_hz)
        # Fastest rate any source reads this client at (the recorder's
        # stream_fps). 0 disables the starvation warning.
        self.expected_read_hz = float(expected_read_hz)
        self.timeout_s = float(timeout_s)
        self.max_age_s = float(max_age_s)
        self._socket_factory = socket_factory

        self._context: Optional[zmq.Context] = None
        self._socket: Optional[Any] = None

        # Replace the payload/timestamp pair atomically under _sample_lock.
        # Keep it separate from the socket lock so latest() never waits on network
        # I/O or connection recovery.
        self._sample: Optional[Tuple[Dict[str, Any], float]] = None
        self._sample_lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        # Consumers currently holding this client. The last one out stops the
        # thread; anything else would tear the socket out from under a sibling
        # tracker that is still recording.
        self._refcount = 0

        self.poll_failures = 0
        self._last_warn_t = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Acquire the client, starting the poll thread on first use.

        The whole decision runs under the lifecycle lock: two trackers starting
        concurrently must not both observe "no thread yet" and each spawn one.
        """
        with self._lifecycle_lock:
            self._refcount += 1
            if self._thread is not None and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._open_socket()
            self._thread = threading.Thread(
                target=self._poll_loop,
                name=f"ViveTrackerClient({self.endpoint})",
                daemon=True,
            )
            self._thread.start()
            print_green(f"[vive] Polling {self.endpoint} at {self.poll_hz:g} Hz.")

    def stop(self) -> None:
        """Release the client; the last consumer out stops the poll thread."""
        with self._lifecycle_lock:
            self._refcount = max(0, self._refcount - 1)
            if self._refcount > 0:
                return
            self._shutdown()

    def force_stop(self) -> None:
        """Tear down regardless of refcount. For tests and hard restarts."""
        with self._lifecycle_lock:
            self._refcount = 0
            self._shutdown()

    def _shutdown(self) -> None:
        """Stop the poll thread and close the socket. Holds _lifecycle_lock."""
        self._stop_event.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._close_socket()
        with self._sample_lock:
            self._sample = None
        print_green(f"[vive] Stopped polling {self.endpoint}.")

    # ------------------------------------------------------------------
    # Read side
    # ------------------------------------------------------------------

    def latest(self) -> Optional[Tuple[Dict[str, Any], float]]:
        """Newest payload and the monotonic time it arrived, or None.

        Returns None when nothing has arrived yet, or when the held sample is
        older than ``max_age_s``. That staleness guard is not optional: without
        it a dead server would leave the last payload in place forever and every
        tracker would record a frozen pose, indistinguishable from a genuinely
        motionless gripper and the exact failure mode the pose checks cannot
        catch. Expiring instead degrades to dropped samples.

        The timestamp is the sample's arrival time, never the caller's read
        time. Two trackers reading one held sample must agree it is one sample:
        ``base_recorder.get_next_frame`` only accepts frames whose
        ``receive_time`` is strictly greater than the last one it processed, so
        per-reader timestamps would fabricate skew between the arms and defeat
        that dedup.
        """
        with self._sample_lock:
            sample = self._sample
        if sample is None:
            return None
        if (time.monotonic() - sample[1]) > self.max_age_s:
            return None
        return sample

    # ------------------------------------------------------------------
    # Poll thread
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        rate_keeper = RateKeeper(target_rate=self.poll_hz, name="ViveTrackerClient")
        ticks = 0
        while not self._stop_event.is_set():
            with rate_keeper:
                payload = self._poll_once()
                if payload is not None:
                    with self._sample_lock:
                        self._sample = (payload, time.monotonic())
            ticks += 1
            # Compare achieved polling rate with consumer stream rate, not requested
            # poll_hz. The server blocks requests until its next sample, so achieved
            # rate can legitimately be lower than the request target. A consumer that
            # outpaces the server receives repeated timestamps and can falsely report
            # source loss after exhausting its fresh-frame timeout.
            if ticks % RATE_CHECK_EVERY == 0 and self.expected_read_hz:
                achieved = rate_keeper.actual_rate
                if 0 < achieved < (self.expected_read_hz * RATE_SHORTFALL_RATIO):
                    self._warn(
                        f"[vive] {self.endpoint} is producing {achieved:.1f} Hz "
                        f"but sources read it at {self.expected_read_hz:g} Hz -- "
                        "the reader outruns the server and the recorder will "
                        "starve on repeat samples. Lower stream_fps below "
                        f"{achieved:.0f} Hz."
                    )

    def _poll_once(self) -> Optional[Dict[str, Any]]:
        """One round trip. Returns the decoded payload, or None on any failure."""
        if self._socket is None:
            self._open_socket()
            if self._socket is None:
                return None
        try:
            self._socket.send_string(REQUEST_COMMAND)
            message = self._socket.recv_string()
        except zmq.Again:
            # A timed-out REQ receive leaves the socket expecting a reply; sending
            # again raises EFSM. Recreate the socket to recover tracking.
            self._warn(f"[vive] {self.endpoint} timed out; resetting socket.")
            self.poll_failures += 1
            self._reset_socket()
            return None
        except zmq.ZMQError as exc:
            self._warn(f"[vive] {self.endpoint} socket error ({exc}); resetting.")
            self.poll_failures += 1
            self._reset_socket()
            return None

        try:
            payload = json.loads(message)
        except (ValueError, TypeError):
            # The REQ/REP cycle completed, so the socket is healthy and only the
            # content is bad. Do not reset; just drop this sample.
            self._warn(f"[vive] {self.endpoint} sent undecodable JSON; dropping.")
            self.poll_failures += 1
            return None

        if not isinstance(payload, dict):
            self._warn(
                f"[vive] {self.endpoint} sent {type(payload).__name__}, "
                "expected a JSON object; dropping."
            )
            self.poll_failures += 1
            return None
        return payload

    # ------------------------------------------------------------------
    # Socket plumbing
    # ------------------------------------------------------------------

    def _open_socket(self) -> None:
        try:
            if self._socket_factory is not None:
                self._socket = self._socket_factory()
                return
            if self._context is None:
                self._context = zmq.Context.instance()
            socket = self._context.socket(zmq.REQ)
            socket.setsockopt(zmq.RCVTIMEO, int(self.timeout_s * 1000))
            socket.setsockopt(zmq.SNDTIMEO, int(self.timeout_s * 1000))
            # LINGER 0 so a reset never blocks on undelivered messages.
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(self.endpoint)
            self._socket = socket
        except zmq.ZMQError as exc:
            print_red(f"[vive] Cannot open socket to {self.endpoint}: {exc}")
            self._socket = None

    def _close_socket(self) -> None:
        socket, self._socket = self._socket, None
        if socket is None:
            return
        try:
            socket.close()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass

    def _reset_socket(self) -> None:
        self._close_socket()
        self._open_socket()

    def _warn(self, message: str) -> None:
        now = time.monotonic()
        if (now - self._last_warn_t) < WARN_INTERVAL_S:
            return
        self._last_warn_t = now
        print_yellow(message)


def get_vive_client(
    endpoint: str,
    poll_hz: float = DEFAULT_POLL_HZ,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_age_s: float = DEFAULT_MAX_AGE_S,
    socket_factory: Optional[Any] = None,
    expected_read_hz: float = 0.0,
) -> ViveTrackerClient:
    """Get the shared client for ``endpoint``, creating it on first request.

    Every tracker naming the same endpoint gets the same object, so they share
    one socket, one poll thread and one round trip. Tuning arguments are honored
    from the first caller only; a later caller asking for different values gets
    a warning rather than a silent second connection, since the whole point of
    the cache is that one endpoint means one socket.
    """
    with _CACHE_LOCK:
        client = _CLIENT_CACHE.get(endpoint)
        if client is None:
            client = ViveTrackerClient(
                endpoint=endpoint,
                poll_hz=poll_hz,
                timeout_s=timeout_s,
                max_age_s=max_age_s,
                socket_factory=socket_factory,
                expected_read_hz=expected_read_hz,
            )
            _CLIENT_CACHE[endpoint] = client
            return client

    mismatched = [
        name
        for name, requested in (
            ("poll_hz", float(poll_hz)),
            ("timeout_s", float(timeout_s)),
            ("max_age_s", float(max_age_s)),
        )
        if getattr(client, name) != requested
    ]
    # The fastest reader on a shared client is the one that can starve, so keep
    # the maximum rather than first-wins.
    if expected_read_hz > client.expected_read_hz:
        client.expected_read_hz = float(expected_read_hz)
    if mismatched:
        print_yellow(
            f"[vive] {endpoint} already open with different "
            f"{', '.join(mismatched)}; keeping the existing client's values. "
            "Make the settings match across sources on this endpoint."
        )
    return client


def reset_vive_clients() -> None:
    """Stop and forget every cached client. For tests and hard restarts."""
    with _CACHE_LOCK:
        clients = list(_CLIENT_CACHE.values())
        _CLIENT_CACHE.clear()
    for client in clients:
        client.force_stop()

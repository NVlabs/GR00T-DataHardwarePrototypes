# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One Vive tracker as a UMI numeric source.

One ``ViveTracker`` == one physical puck == one recorded zarr group. Several
trackers on the same server share a single ``ViveTrackerClient`` (one socket, one
round trip); each picks its own ``tracker_key`` out of the shared payload.

Per-tracker sources rather than one multi-pose source, deliberately:

* **Failure isolation is structural.** A tracker with no usable sample makes
  ``get_frame()`` return None, which ``base_recorder._is_frame_valid`` drops
  before it reaches a buffer. The other tracker cannot be affected, because it
  is a different source.
* **Per-tracker FPS and delay tracking.** ``numeric_recorder`` gates those
  counters to ``key_idx == 0``, so a split source reports one signal for the
  whole device; separate sources each get their own.
* **The data check keys on the group name.** ``{left,right}_tracking_pose`` puts
  "tracking" in the group name and a ``pose`` array inside it, which is what
  enables pose_movement / pose_jumps / consecutive_movement. See ``PoseFrame``.

Pose is recorded raw in the lighthouse frame; the rigid tracker-to-gripper mount
offset is applied at conversion time, not here.

Requires a server that publishes ``tracked`` per entry. OpenXR leaves
POSITION_VALID set on a puck that has lost tracking and signals the loss by
clearing POSITION_TRACKED, so a server forwarding only the pose hands back a
stale but plausible one that is indistinguishable from a motionless gripper.
A server without the flag is rejected at start_streaming() rather than worked
around: inference from byte-identical repetition can be made to work, but it
needs a second of identical samples to decide, and it reports a *guess* in a
message an operator will read as fact.

Who reports what: returning None is the only signal this device sends upward,
and ``BaseRecorder`` owns what happens next. It counts consecutive drops and
escalates to a ``frame_drop`` warning and then ``source_lost`` on its own
thresholds, for every source alike, so this one stays quiet by default rather
than adding a second voice for the same event. ``verbose=true`` additionally
prints a rate-limited line naming the cause, which the recorder cannot know;
it is off unless someone is diagnosing a specific tracker.
"""

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from umi.common.data.frame import PoseFrame
from umi.common.hardware_sdk.sensor_interface import SensorInterface
from umi.common.hardware_sdk.vive.client import (
    DEFAULT_MAX_AGE_S,
    DEFAULT_POLL_HZ,
    DEFAULT_TIMEOUT_S,
    get_vive_client,
)
from umi.common.utility.print_utils import print_green, print_yellow

# How long start_streaming() waits for a first usable sample.
STARTUP_TIMEOUT_S = 5.0

# Seconds between repeats of the same cause line, when verbose.
WARN_INTERVAL_S = 2.0


def pose_dict_to_matrix(entry: Dict[str, Any]) -> Optional[np.ndarray]:
    """``{"position": {x,y,z}, "orientation": {x,y,z,w}} -> 4x4`` homogeneous.

    None if malformed or the quaternion is degenerate.
    """
    if not isinstance(entry, dict):
        return None
    position = entry.get("position")
    orientation = entry.get("orientation")
    if not isinstance(position, dict) or not isinstance(orientation, dict):
        return None
    try:
        xyz = [float(position[axis]) for axis in ("x", "y", "z")]
        # scipy's quaternion convention is (x, y, z, w).
        quat = [float(orientation[axis]) for axis in ("x", "y", "z", "w")]
    except (KeyError, TypeError, ValueError):
        return None
    if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(quat)):
        return None
    if np.allclose(quat, 0.0):
        return None
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_quat(quat).as_matrix()
    matrix[:3, 3] = xyz
    return matrix


class ViveTracker(SensorInterface):
    """A single Vive tracker, read through the shared per-endpoint client.

    Args:
        device_name: Recorded source name, and therefore the zarr group name.
            Include "tracking" (e.g. ``left_tracking_pose``) or the data check
            will not run pose checks on it.
        tracker_key: The key this puck has in the server's payload (e.g.
            ``left_shoulder``). Decoupled from ``device_name`` so the server's
            naming and UMI's naming can differ without either changing.
        endpoint: ZMQ address of the tracker server.
        latency: Seconds subtracted from arrival to get capture time, the same
            convention every other UMI device uses.
        require_tracking_at_start: Refuse to start on a puck that is present but
            not tracked. Set False to record anyway (the pose will be frozen).
        verbose: Print a rate-limited line naming why each sample was dropped.
            Off by default: the recorder already reports drops for every source
            through one path, and a per-device voice for the same event is noise
            during a session. Turn it on to diagnose one tracker.
    """

    def __init__(
        self,
        device_name: str = "vive_tracking_pose",
        tracker_key: str = "",
        endpoint: str = "tcp://127.0.0.1:5555",
        poll_hz: float = DEFAULT_POLL_HZ,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_age_s: float = DEFAULT_MAX_AGE_S,
        latency: Optional[float] = None,
        startup_timeout_s: float = STARTUP_TIMEOUT_S,
        require_tracking_at_start: bool = True,
        verbose: bool = False,
        expected_read_hz: float = 0.0,
        group_suffix: str = "",
        socket_factory: Optional[Any] = None,
        client: Optional[Any] = None,
    ) -> None:
        super().__init__(
            device_name=device_name, latency=latency, group_suffix=group_suffix
        )
        if not tracker_key:
            raise ValueError(
                f"[{device_name}] tracker_key is required -- it names this puck "
                "in the server's payload (e.g. 'left_shoulder')."
            )
        self.tracker_key = tracker_key
        self.endpoint = endpoint
        self.poll_hz = float(poll_hz)
        self.timeout_s = float(timeout_s)
        self.max_age_s = float(max_age_s)
        self.startup_timeout_s = float(startup_timeout_s)
        self.require_tracking_at_start = bool(require_tracking_at_start)
        self.verbose = bool(verbose)
        self.expected_read_hz = float(expected_read_hz)
        self._socket_factory = socket_factory
        # Resolve the shared client only at start_streaming so construction performs
        # no I/O; tests can inject a client.
        self._client = client

        self.dropped_frames = 0
        self._last_warn_t = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_streaming(self) -> None:
        if self.is_running:
            print_yellow(f"[{self.device_name}] Already streaming.")
            return

        if self._client is None:
            self._client = get_vive_client(
                endpoint=self.endpoint,
                poll_hz=self.poll_hz,
                timeout_s=self.timeout_s,
                max_age_s=self.max_age_s,
                socket_factory=self._socket_factory,
                expected_read_hz=self.expected_read_hz,
            )
        self._client.start()

        self._await_usable_sample()
        self.dropped_frames = 0
        self.is_running = True
        print_green(
            f"[{self.device_name}] Streaming '{self.tracker_key}' from {self.endpoint}."
        )

    def _await_usable_sample(self) -> None:
        """Block until this puck reports a live pose, or raise.

        Each failure names a different subsystem, because they send whoever
        reads the message somewhere different: a dead bridge process, a server
        too old to report tracking state, a server with no pucks, a mistyped
        key, or a puck present but not tracked.
        """
        deadline = time.monotonic() + self.startup_timeout_s
        answered = False
        seen: list = []
        entry = None

        while time.monotonic() < deadline:
            sample = self._client.latest()
            if sample is not None:
                answered = True
                payload, _arrival = sample
                seen = sorted(payload.keys())
                entry = payload.get(self.tracker_key)
                if entry is not None and self._tracked_flag(entry) is True:
                    return
            time.sleep(0.02)

        if not answered:
            raise RuntimeError(
                f"[{self.device_name}] No response from {self.endpoint} within "
                f"{self.startup_timeout_s:g}s. Is the Vive tracker server running? "
                f"(Poll failures: {getattr(self._client, 'poll_failures', '?')}.)"
            )
        if not seen:
            raise RuntimeError(
                f"[{self.device_name}] {self.endpoint} answered but reported no "
                "trackers at all. The server publishes a puck only once it has "
                "been tracked, so nothing has been seen yet."
            )
        if self.tracker_key not in seen:
            raise RuntimeError(
                f"[{self.device_name}] Tracker key {self.tracker_key!r} is not in "
                f"the payload from {self.endpoint}. Available: {seen}."
            )
        if self._tracked_flag(entry) is None:
            raise RuntimeError(
                f"[{self.device_name}] {self.endpoint} does not publish a "
                "'tracked' field. That server cannot report a puck that has LOST "
                "tracking -- OpenXR keeps POSITION_VALID set and clears "
                "POSITION_TRACKED, so it would keep serving a stale pose that "
                "records as a flat, plausible trajectory. Update the server to "
                "test both bits and publish 'tracked' per entry."
            )

        message = (
            f"[{self.device_name}] Tracker {self.tracker_key!r} IS in the payload "
            f"from {self.endpoint}, but the server reports it as NOT tracked. It "
            "is serving a stale last-known pose, which would record as a flat, "
            "plausible-looking trajectory. Check that the puck is powered and "
            "awake, has line of sight to the base stations, and is paired."
        )
        if self.require_tracking_at_start:
            raise RuntimeError(message)
        print_yellow(message + " (require_tracking_at_start=false, continuing.)")

    def stop_streaming(self) -> None:
        if not self.is_running:
            return
        self.is_running = False
        if self._client is not None:
            self._client.stop()
        if self.dropped_frames:
            print_yellow(
                f"[{self.device_name}] Dropped {self.dropped_frames} sample(s) "
                f"for '{self.tracker_key}' this session."
            )
        print_green(f"[{self.device_name}] Streaming stopped.")

    # ------------------------------------------------------------------
    # Read side
    # ------------------------------------------------------------------

    def get_frame(self) -> Optional[PoseFrame]:
        """Latest pose, or None when this tracker has no usable sample.

        None is the whole dropout policy: the recorder drops the sample, the
        zarr gets a real gap that PoseInterpolator bridges at ingest, and
        nothing fabricates a pose. Holding the last known pose instead would
        make a dead puck look like a motionless gripper -- which is exactly what
        the pose checks cannot detect, and exactly what the server does.
        """
        if not self.is_running or self._client is None:
            return None

        sample = self._client.latest()
        if sample is None:
            return self._drop("no fresh sample from the server")

        payload, arrival = sample
        entry = payload.get(self.tracker_key)
        if entry is None:
            return self._drop(f"'{self.tracker_key}' missing from payload")

        tracked, why = self._is_tracked(entry)
        if not tracked:
            return self._drop(why)

        matrix = pose_dict_to_matrix(entry)
        if matrix is None:
            return self._drop(f"'{self.tracker_key}' pose is malformed")

        capture_time = arrival - self.latency if self.latency else arrival
        return PoseFrame(pose=matrix, capture_time=capture_time, receive_time=arrival)

    # ------------------------------------------------------------------
    # Tracking validity
    # ------------------------------------------------------------------

    @staticmethod
    def _tracked_flag(entry: Dict[str, Any]) -> Optional[bool]:
        """The server's verdict, or None if it does not publish one."""
        value = entry.get("tracked")
        return bool(value) if isinstance(value, bool) else None

    def _is_tracked(self, entry: Dict[str, Any]) -> Tuple[bool, str]:
        """Is this pose live? ``(tracked, reason_if_not)``.

        Purely the server's flag. start_streaming() has already refused a server
        that does not publish one, so a missing flag here means the server was
        restarted mid-session with an older build -- worth saying plainly rather
        than silently falling back to a guess.
        """
        flag = self._tracked_flag(entry)
        if flag is True:
            return True, ""
        if flag is False:
            return False, (
                f"'{self.tracker_key}' reported not tracked by the server "
                "(puck asleep, occluded from the base stations, or unpaired)"
            )
        return False, (
            f"'{self.tracker_key}' has no 'tracked' field -- the server was "
            "replaced mid-session with one too old to report tracking state"
        )

    def _drop(self, reason: str) -> None:
        """Count the drop; the recorder reports and escalates it.

        The cause line is verbose-only. BaseRecorder already surfaces a drop for
        every source through one path, on thresholds tuned for recording, and
        printing here as well means two voices for one event -- at this device's
        rate rather than the recorder's.
        """
        self.dropped_frames += 1
        if not self.verbose:
            return None
        now = time.monotonic()
        if (now - self._last_warn_t) >= WARN_INTERVAL_S:
            self._last_warn_t = now
            print_yellow(
                f"[{self.device_name}] {reason} "
                f"({self.dropped_frames} sample(s) dropped so far)."
            )
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Live-print one Vive tracker.")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--tracker-key", default="left_shoulder")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--allow-untracked", action="store_true")
    args = parser.parse_args()

    tracker = ViveTracker(
        device_name="vive_tracking_pose",
        tracker_key=args.tracker_key,
        endpoint=args.endpoint,
        require_tracking_at_start=not args.allow_untracked,
    )
    tracker.start_streaming()
    try:
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            frame = tracker.get_frame()
            if frame is not None:
                print(np.round(frame.pose[:3, 3], 4))
            time.sleep(1 / 30)
    finally:
        tracker.stop_streaming()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identify an exoskeleton's cameras by a displayed ChArUco board.

Every connected camera is observed: an unavailable camera could conceal a
second match. SDKs live only in disposable workers (one subprocess per camera),
and ``CameraSession.close()`` returns only after all of them have exited. The
caller must exclude recorder launch and concurrent pairing for the whole life
of a session (including cancellation).

Two ways to run the workers:

- **one-shot** (``identify_pair`` / ``check_streams``): start, wait until every
  worker is ready, one common scan window, close. Nobody sees frames.
- **preview** (``CameraSession(preview=True)``, the Identify-cameras dialog):
  workers stay open, stream a JPEG thumbnail with the detected markers drawn
  plus a per-side "sees the board" marker count about ten times a second, and run
  a scan window for either side on request without rebooting the cameras. The
  GUI closes the session when the dialog closes, when the page stops polling
  (``PREVIEW_IDLE_SECONDS``) or after ``PREVIEW_MAX_SECONDS``.

Worker protocol (one JSON object per line on the worker's stdout):
``{"ready": true}`` once the camera streams; ``{"preview": <base64 jpeg or
null>, "seq", "frames", "sees": {"left", "right"}, "stalled"}`` heartbeats in
preview mode; ``{"frames", "hits", "error"}`` after a scan window (plus
``"scan": side`` in preview mode) and as the final message of a worker that
failed. Commands on stdin: ``scan`` (one-shot, the side comes from ``--side``),
``scan <side>`` (preview) and ``quit``.
"""

import argparse
import base64
import binascii
import contextlib
import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
# OAK USB 2 boot and re-enumeration can take 10–20 s, longer with several
# cameras. Allow the SDK to finish booting before treating a worker as unresponsive.
STARTUP_SECONDS = 45.0
SCAN_SECONDS = 8.0
HEALTH_SCAN_SECONDS = 2.0
# Grace period for a scan result to arrive after its preview window closes.
SCAN_RESULT_GRACE_SECONDS = 2.0
# OAK close resets the device over USB; resets serialize on a shared bus.
# Allow several seconds for cleanup before killing a worker, which can leave
# the camera unavailable to the recorder.
SHUTDOWN_SECONDS = 15.0
STOP_GRACE_SECONDS = 1.0
MAX_FRAME_GAP_SECONDS = 1.5
MIN_FRAMES = 3
MIN_HITS = 3
MIN_MARKERS = 2
# One protocol line. A 320 px JPEG thumbnail is ~20 KB as base64.
MAX_LINE_BYTES = 262144
PREVIEW_INTERVAL = 0.1  # thumbnails per camera: 10 Hz keeps aiming lag under ~200 ms
PREVIEW_WIDTH = 320
PREVIEW_JPEG_QUALITY = 70
PREVIEW_IDLE_SECONDS = 60.0
PREVIEW_MAX_SECONDS = 900.0
MARKER_OFFSETS = {"left": 40, "right": 80}
# Translate SDK transport-state messages into actionable device errors.
OAK_ERROR_HINTS = (
    (
        "Failed to find device after booting",
        "OAK did not come back on USB after its firmware boot: check the cable, hub "
        "and power (USB 2 hubs are the usual cause), then replug it",
    ),
    (
        "Couldn't open stream",
        "OAK booted but the host could not open its stream: replug the camera "
        "(USB 2 hubs and long cables are the usual cause)",
    ),
    ("X_LINK_DEVICE_ALREADY_IN_USE", "another process owns this OAK"),
)


def _marker_ids(side):
    if side not in MARKER_OFFSETS:
        raise ValueError("side must be 'left' or 'right'")
    return set(range(MARKER_OFFSETS[side], MARKER_OFFSETS[side] + 10))


def board_png(side):
    """A real 5×4 ChArUco board with a white quiet margin; IDs encode the side."""
    import cv2
    import numpy as np

    ids = np.array(sorted(_marker_ids(side)), dtype=np.int32)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard((5, 4), 1.0, 0.75, dictionary, ids)
    image = board.generateImage((1000, 800), marginSize=40)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("could not encode camera identification board")
    return encoded.tobytes()


def _detector():
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    return cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())


def _detect(gray, detector):
    """(set of marker ids, corners) found in one grayscale frame."""
    corners, ids, _ = detector.detectMarkers(gray)
    found = {int(x) for x in ids.ravel()} if ids is not None else set()
    return found, corners


def _sees(found):
    return {side: len(_marker_ids(side) & found) for side in MARKER_OFFSETS}


def _encode_preview(gray, corners):
    import cv2

    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if corners:
        try:
            cv2.aruco.drawDetectedMarkers(bgr, corners)
        except cv2.error:
            pass
    height, width = bgr.shape[:2]
    if width > PREVIEW_WIDTH:
        size = (PREVIEW_WIDTH, max(1, round(height * PREVIEW_WIDTH / width)))
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(
        ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY]
    )
    return base64.b64encode(encoded.tobytes()).decode("ascii") if ok else None


class _Previewer:
    """Worker side: throttled thumbnails + marker counts over the protocol pipe."""

    def __init__(self, emit):
        self.emit = emit
        self.seq = 0
        self.frames = 0
        self.sees = {side: 0 for side in MARKER_OFFSETS}
        self.last_frame_at = time.monotonic()
        self.last_emit = 0.0
        self.pending = None
        self.latest = None

    def due(self):
        return time.monotonic() - self.last_emit >= PREVIEW_INTERVAL

    def seen(self, gray):
        """A fresh frame from the idle loop; kept as the newest thumbnail candidate."""
        self.frames += 1
        self.last_frame_at = time.monotonic()
        self.latest = gray

    def tick(self, detector):
        """Idle loop: when an emit is due, detect on the newest frame and send it.

        The newest frame is encoded even when the current read() returned
        nothing. Sources that return None on most polls (an OAK at 15 fps
        polled every 20 ms) used to have their thumbnail slot taken by the
        heartbeat almost every time, so their tiles never showed an image.
        """
        if not self.due():
            return
        if self.latest is not None:
            gray, self.latest = self.latest, None
            found, corners = _detect(gray, detector)
            self.pending = (gray, corners)
            self.sees = _sees(found)
        self.flush(force=True)

    def frame(self, gray, found, corners):
        """Scan window: every fresh frame is already detected; emit when due."""
        self.frames += 1
        self.last_frame_at = time.monotonic()
        self.latest = None
        self.pending = (gray, corners)
        self.sees = _sees(found)
        self.flush()

    def flush(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_emit < PREVIEW_INTERVAL:
            return
        self.last_emit = now
        message = {
            "preview": None,
            "seq": self.seq,
            "frames": self.frames,
            "sees": dict(self.sees),
            "stalled": round(
                max(0.0, now - self.last_frame_at - MAX_FRAME_GAP_SECONDS), 1
            ),
        }
        if self.pending is not None:
            gray, corners = self.pending
            self.pending = None
            encoded = _encode_preview(gray, corners)
            if encoded is not None:
                self.seq += 1
                message["preview"] = encoded
                message["seq"] = self.seq
        self.emit(message)


def _collect_frames(read, side, duration=None, observer=None):
    duration = SCAN_SECONDS if duration is None else duration
    result = {"frames": 0, "hits": 0, "error": None}
    seen = set()
    detector = None if side == "health" else _detector()
    last_frame_at = time.monotonic()
    deadline = last_frame_at + duration
    try:
        while time.monotonic() < deadline:
            sample = read()
            now = time.monotonic()
            # Check before accepting a returned sample: a native read may
            # have blocked through a gap that concealed another camera match.
            if now - last_frame_at > MAX_FRAME_GAP_SECONDS:
                raise RuntimeError(
                    "camera stopped delivering fresh frames during the scan"
                    if result["frames"]
                    else "camera opened but no fresh frames arrived; check its USB "
                    "connection and streaming mode, then retry"
                )
            if sample is None:
                continue
            token, gray = sample
            if gray is None or token in seen:
                continue
            seen.add(token)
            last_frame_at = now
            result["frames"] += 1
            found, corners = _detect(gray, detector) if detector else (set(), None)
            if side != "health" and len(_marker_ids(side) & found) >= MIN_MARKERS:
                result["hits"] += 1
            if observer is not None:
                observer(gray, found, corners)
        if time.monotonic() - last_frame_at > MAX_FRAME_GAP_SECONDS:
            raise RuntimeError(
                "camera stopped delivering fresh frames before scan completion"
            )
    except InterruptedError:
        raise  # SIGTERM: the worker exits instead of reporting a window.
    except Exception as exc:
        result["error"] = str(exc)[:500] or type(exc).__name__
    return result


@contextlib.contextmanager
def _iphone_frames(device_id):
    import cv2
    from record3d import Record3DStream

    stream = Record3DStream()
    event = threading.Event()
    stopped = threading.Event()
    lock = threading.Lock()
    frame_number = 0
    latest_rgb = None
    frame_error = None
    first_read = True

    def on_frame():
        nonlocal frame_number, latest_rgb, frame_error
        with lock:
            try:
                # Record3D overwrites its native RGB buffer before the next
                # callback. Copy on the producer thread while this callback
                # still owns the frame, so each token identifies that image.
                rgb = stream.get_rgb_frame()
                latest_rgb = None if rgb is None else rgb.copy()
                frame_number += 1
            except Exception as exc:
                frame_error = exc
            event.set()

    def read():
        nonlocal first_read
        if first_read:
            first_read = False
            with lock:
                event.clear()  # Discard the image buffered before the scan barrier.
            return None
        if stopped.is_set():
            raise RuntimeError(
                "iPhone stream stopped; check Record3D and the USB cable"
            )
        if frame_error is not None:
            raise RuntimeError(f"iPhone frame read failed: {frame_error}")
        if not event.wait(0.1):
            return None
        with lock:
            if frame_error is not None:
                raise RuntimeError(f"iPhone frame read failed: {frame_error}")
            event.clear()
            token = frame_number
            rgb = latest_rgb
        if rgb is None:
            return None
        if stream.get_device_type() == 0:  # Record3D TrueDepth is mirrored.
            rgb = cv2.flip(rgb, 1)
        return token, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    try:
        target = next(
            (
                d
                for d in Record3DStream.get_connected_devices()
                if str(d.udid) == device_id
            ),
            None,
        )
        if target is None:
            raise RuntimeError("iPhone is no longer connected")
        stream.on_new_frame = on_frame
        stream.on_stream_stopped = stopped.set
        if stream.connect(target) is False:
            raise RuntimeError(
                "iPhone connection failed; enable USB streaming in Record3D"
            )
        yield read
    finally:
        # IphoneCamera.stop_streaming() does not release Record3D. Explicitly
        # disconnect here; the parent also enforces process exit after cleanup.
        stream.disconnect()


def _explain_oak_error(exc):
    text = str(exc)
    for needle, hint in OAK_ERROR_HINTS:
        if needle in text:
            return RuntimeError(f"{hint} [{text}]")
    return exc


@contextlib.contextmanager
def _oak_frames(device_id):
    import cv2
    import depthai as dai

    device = None
    pipeline = None
    try:
        info = next(
            (
                d
                for d in dai.Device.getAllAvailableDevices()
                if d.getDeviceId() == device_id
            ),
            None,
        )
        if info is None:
            raise RuntimeError("OAK is unavailable or another process owns it")
        try:
            device = dai.Device(info)
            pipeline = dai.Pipeline(device)
            camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            queue = camera.requestOutput((640, 480), fps=15).createOutputQueue(
                maxSize=1, blocking=False
            )
            pipeline.start()
        except RuntimeError as exc:
            raise _explain_oak_error(exc) from exc
        first_read = True

        def read():
            nonlocal first_read
            if not pipeline.isRunning():
                raise RuntimeError("OAK pipeline stopped")
            packet = queue.tryGet()
            if first_read:
                first_read = False
                return None  # The one-slot queue may predate the scan barrier.
            if packet is None:
                time.sleep(0.02)
                return None
            return packet.getSequenceNum(), cv2.cvtColor(
                packet.getCvFrame(), cv2.COLOR_BGR2GRAY
            )

        yield read
    finally:
        try:
            if pipeline is not None:
                pipeline.stop()
        finally:
            if device is not None:
                device.close()


def _worker_argv(device, side):
    argv = [
        sys.executable,
        "-m",
        "gui.dc_ui.core.camera_pairing",
        "--kind",
        device.kind,
        "--device-id",
        device.device_id,
    ]
    if side is not None:
        argv.extend(["--side", side])
    argv.extend(["--parent-pid", str(os.getpid())])
    return argv


def _arm_parent_death(expected_parent):
    if sys.platform != "linux":
        return
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(
            ctypes.get_errno(), "could not arm camera worker parent-death guard"
        )
    # Check for parent death before SDK imports; it can occur between Popen and prctl.
    if os.getppid() != expected_parent:
        raise RuntimeError("camera worker parent has exited")


def _stop_process(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=STOP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
    # This is the release barrier, including after SIGKILL. Never return a
    # successful assignment while a process may still own a camera.
    proc.wait()


def _camera_result(device, error=None):
    return {
        "kind": device.kind,
        "device_id": device.device_id,
        "name": device.name,
        "frames": 0,
        "hits": 0,
        "error": error,
    }


class _Worker:
    """Parent-side handle of one camera worker process and its protocol reader."""

    def __init__(self, device, argv, preview=False):
        self.device = device
        self.preview_mode = preview
        self.lock = threading.RLock()
        self.ready = threading.Event()
        self.settled = threading.Event()  # ready, failed or exited
        self.result_event = threading.Event()
        self.error = None
        self.result = None
        self.scanning = None
        self.closing = False
        self.preview = None
        self.preview_seq = 0
        self.preview_at = None
        self.frames = 0
        self.sees = {side: 0 for side in MARKER_OFFSETS}
        self.stalled = 0.0
        self.proc = subprocess.Popen(
            argv,
            cwd=str(REPO_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.thread = threading.Thread(
            target=self._pump,
            name=f"camera-{device.kind}-{device.device_id}",
            daemon=True,
        )
        self.thread.start()

    # -- protocol reader ------------------------------------------------------

    def _pump(self):
        fd = self.proc.stdout.fileno()
        pending = b""
        try:
            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                pending += chunk
                if len(pending) > MAX_LINE_BYTES:
                    self.fail("camera worker output exceeded its limit")
                    break
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    self._handle(line)
        except OSError:
            pass
        self.proc.wait()
        with self.lock:
            if self.error is None and not self.closing:
                if self.result is None or self.scanning is not None:
                    self.error = (
                        f"camera worker exited with status {self.proc.returncode}"
                    )
                elif self.preview_mode:
                    self.error = (
                        f"camera stream ended (worker exit {self.proc.returncode})"
                    )
            self.settled.set()
            self.result_event.set()

    def _handle(self, line):
        try:
            message = json.loads(line)
        except ValueError:
            return self.fail("camera worker returned an invalid response")
        if not isinstance(message, dict):
            return self.fail("camera worker returned an invalid response")
        if message == {"ready": True}:
            with self.lock:
                self.ready.set()
                self.settled.set()
            return None
        if "preview" in message:
            return self._preview(message)
        if "frames" in message and "hits" in message:
            return self._scan_result(message)
        if "error" in message:
            return self.fail(str(message["error"])[:500] or "camera worker failed")
        return self.fail("camera worker returned an invalid response")

    def _preview(self, message):
        frames, seq, sees = (
            message.get("frames"),
            message.get("seq"),
            message.get("sees"),
        )
        if (
            type(frames) is not int
            or type(seq) is not int
            or not isinstance(sees, dict)
        ):
            return self.fail("camera worker returned an invalid preview")
        jpeg = None
        image = message.get("preview")
        if image is not None:
            try:
                jpeg = base64.b64decode(image, validate=True)
            except (ValueError, TypeError, binascii.Error):
                return self.fail("camera worker returned an invalid preview")
            if not jpeg.startswith(b"\xff\xd8"):
                return self.fail("camera worker returned an invalid preview")
        with self.lock:
            if not self.ready.is_set():
                return self.fail("camera worker streamed before it was ready")
            self.frames = frames
            self.sees = {side: int(sees.get(side) or 0) for side in MARKER_OFFSETS}
            try:
                self.stalled = float(message.get("stalled") or 0.0)
            except (TypeError, ValueError):
                self.stalled = 0.0
            if jpeg is not None:
                self.preview = jpeg
                self.preview_seq = seq
                self.preview_at = time.time()
        return None

    def _scan_result(self, message):
        frames, hits, error = (
            message.get("frames"),
            message.get("hits"),
            message.get("error"),
        )
        if (
            type(frames) is not int
            or type(hits) is not int
            or not 0 <= hits <= frames
            or (error is not None and not isinstance(error, str))
        ):
            return self.fail("camera worker returned invalid frame counts")
        with self.lock:
            if self.scanning is None:
                if error:
                    return self.fail(error[:500])
                return self.fail(
                    "camera worker returned evidence before the scan began"
                )
            self.result = {"frames": frames, "hits": hits, "error": error}
            self.scanning = None
            self.result_event.set()
        return None

    def fail(self, text):
        with self.lock:
            if self.error is None:
                self.error = text
            self.settled.set()
            self.result_event.set()

    # -- commands -------------------------------------------------------------

    def send(self, command):
        try:
            self.proc.stdin.write(f"{command}\n".encode())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            if not self.closing:
                self.fail("camera worker is not accepting commands")

    def begin_scan(self, side, one_shot):
        with self.lock:
            self.result = None
            self.result_event.clear()
            self.scanning = side
        self.send("scan" if one_shot else f"scan {side}")

    def is_live(self):
        with self.lock:
            return self.ready.is_set() and self.error is None

    def stop(self):
        with self.lock:
            self.closing = True
        _stop_process(self.proc)
        self.thread.join(timeout=2.0)
        for pipe in (self.proc.stdin, self.proc.stdout):
            try:
                pipe.close()
            except OSError:
                pass

    def snapshot(self):
        with self.lock:
            if self.error:
                status = "error"
            elif not self.ready.is_set():
                status = "starting"
            elif self.scanning:
                status = "scanning"
            else:
                status = "live"
            return {
                "kind": self.device.kind,
                "device_id": self.device.device_id,
                "name": self.device.name,
                "state": status,
                "error": self.error,
                "frames": self.frames,
                "sees": dict(self.sees),
                "stalled": self.stalled,
                "preview_seq": self.preview_seq,
                "preview_at": self.preview_at,
            }


class CameraSession:
    """The workers of one identification or stream check; owns their lifetime.

    ``devices`` are the cameras to open. ``excluded`` are cameras that were not
    opened (duplicate, in use) as ``_camera_result`` dicts and ``errors`` the
    discovery errors; both travel with the session so a live scan can judge
    completeness the same way a one-shot scan does.
    """

    def __init__(
        self,
        devices,
        *,
        preview=False,
        side=None,
        duration=None,
        excluded=(),
        errors=(),
    ):
        if not preview and side is None:
            raise ValueError("a one-shot camera session needs a side")
        self.devices = list(devices)
        self.preview = preview
        self.side = side
        self.duration = duration
        self.excluded = list(excluded)
        self.errors = list(errors)
        self.workers = []
        self.lock = threading.Lock()
        self._close_lock = threading.Lock()
        self.started_at = None
        self.last_poll = None
        self.scanning = None
        self.closed = False
        self.graceful = {}

    def start(self):
        for device in self.devices:
            argv = list(_worker_argv(device, self.side))
            if self.preview:
                argv.append("--preview")
            if self.duration is not None:
                argv.extend(["--duration", str(self.duration)])
            self.workers.append(_Worker(device, argv, preview=self.preview))
        self.started_at = self.last_poll = time.monotonic()
        return self

    def settled(self):
        return all(worker.settled.is_set() for worker in self.workers)

    def wait_ready(self, cancel, timeout=None):
        """Block until every worker is ready or has failed; False on cancel/timeout."""
        deadline = time.monotonic() + (STARTUP_SECONDS if timeout is None else timeout)
        while time.monotonic() < deadline and not cancel.is_set():
            if self.settled():
                break
            time.sleep(0.02)
        if not cancel.is_set():
            for worker in self.workers:
                if not worker.settled.is_set():
                    worker.fail("camera startup timed out")
        return not cancel.is_set() and all(w.is_live() for w in self.workers)

    def scan(self, side, cancel):
        """One counting window on every live worker -> per-camera result dicts.

        Cameras that failed or never started are reported with their error;
        workers are judged independently, so one bad camera no longer hides
        which others streamed fine.
        """
        with self.lock:
            if self.scanning not in (None, side):
                raise RuntimeError(f"a {self.scanning} scan is already running")
            self.scanning = side
        try:
            live = [] if cancel.is_set() else [w for w in self.workers if w.is_live()]
            for worker in live:
                worker.begin_scan(side, one_shot=not self.preview)
            window = SCAN_SECONDS if self.duration is None else self.duration
            deadline = time.monotonic() + window + SCAN_RESULT_GRACE_SECONDS
            while time.monotonic() < deadline and not cancel.is_set():
                if all(worker.result_event.is_set() for worker in live):
                    break
                time.sleep(0.02)
            results = []
            for worker in self.workers:
                camera = _camera_result(worker.device)
                if worker.error:
                    camera["error"] = worker.error
                elif worker.result is not None and worker in live:
                    camera.update(worker.result)
                elif cancel.is_set():
                    camera["error"] = "cancelled"
                elif worker in live:
                    worker.fail("camera scan timed out")
                    camera["error"] = worker.error
                else:
                    camera["error"] = "camera did not start"
                if camera["frames"] < MIN_FRAMES and not camera["error"]:
                    camera["error"] = (
                        f"only {camera['frames']} fresh frames received; need at least {MIN_FRAMES}"
                    )
                results.append(camera)
            return results
        finally:
            with self.lock:
                self.scanning = None

    def close(self, grace=None):
        """Stop every worker and reap it; returns {(kind, id): exited on its own}.

        ``grace`` is how long workers may take to exit after ``quit`` before
        they are terminated (default SHUTDOWN_SECONDS; a cancelled caller
        passes 0 so the release barrier does not wait for cooperative exits).
        """
        with self._close_lock:
            with self.lock:
                if self.closed:
                    return dict(self.graceful)
                self.closed = True
            grace = SHUTDOWN_SECONDS if grace is None else grace
            for worker in self.workers:
                with worker.lock:
                    worker.closing = True
                # Without grace the workers are terminated right away; a
                # ``quit`` racing that signal would let one exit half-way
                # through its cleanup, so cancellation uses the signal alone.
                if grace > 0:
                    worker.send("quit")
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline and any(
                worker.proc.poll() is None for worker in self.workers
            ):
                time.sleep(0.02)
            for worker in self.workers:
                key = worker.device.kind, worker.device.device_id
                self.graceful[key] = worker.proc.poll() is not None
                worker.stop()
            return dict(self.graceful)

    # -- preview bookkeeping ----------------------------------------------------

    def touch(self):
        self.last_poll = time.monotonic()

    def expired(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            if (
                not self.preview
                or self.closed
                or self.scanning
                or self.started_at is None
            ):
                return False
        return (
            now - self.last_poll > PREVIEW_IDLE_SECONDS
            or now - self.started_at > PREVIEW_MAX_SECONDS
        )

    def preview_jpeg(self, kind, device_id):
        for worker in self.workers:
            if worker.device.kind == kind and worker.device.device_id == device_id:
                with worker.lock:
                    return worker.preview
        return None

    def snapshot(self):
        with self.lock:
            scanning, closed, started = self.scanning, self.closed, self.started_at
        return {
            "preview": self.preview,
            "closed": closed,
            "scanning": scanning,
            "age_s": None if started is None else round(time.monotonic() - started, 1),
            "cameras": [worker.snapshot() for worker in self.workers]
            + [{**camera, "state": "error"} for camera in self.excluded],
            "errors": list(self.errors),
        }


def pairing_candidates(detection):
    """(cameras to open, excluded camera results, discovery errors) for identification."""
    candidates, excluded = [], []
    errors = list(detection.get("errors") or [])
    seen = set()
    for kind in ("iphone", "oak"):
        devices = detection.get(kind) or []
        if not devices:
            errors.append(f"no connected {kind} cameras were discovered")
        for device in devices:
            error = None
            key = kind, device.device_id
            if key in seen:
                error = "duplicate camera in discovery"
            elif device.in_use:
                error = "camera is in use by another process"
            seen.add(key)
            if error:
                excluded.append(_camera_result(device, error))
            else:
                candidates.append(device)
    return candidates, excluded, errors


def _assemble(side, cameras, errors, cancelled):
    """``assignments`` = slot -> camera name (its id; what the page fills in);
    ``identities`` = slot -> {kind, name, device_id}: the raw id each worker
    opened, which the GUI keeps for the launch's device binding."""
    result = {
        "side": side,
        "assignments": {},
        "identities": {},
        "cameras": cameras,
        "errors": list(errors),
        "cancelled": cancelled,
    }
    for camera in cameras:
        if camera["error"]:
            result["errors"].append(
                f"{camera['name'] or camera['device_id']}: {camera['error']}"
            )
    assignments, identities = {}, {}
    for kind in ("iphone", "oak"):
        matches = [
            c
            for c in cameras
            if c["kind"] == kind and not c["error"] and c["hits"] >= MIN_HITS
        ]
        if len(matches) == 1:
            assignments[f"{side}_{kind}"] = matches[0]["name"]
            identities[f"{side}_{kind}"] = {
                "kind": kind,
                "name": matches[0]["name"],
                "device_id": matches[0]["device_id"],
            }
        elif len(matches) > 1:
            names = ", ".join(c["name"] for c in matches)
            result["errors"].append(
                f"multiple {kind} cameras see the board: {names}; point the other cameras away"
            )
        else:
            result["errors"].append(
                f"no {kind} camera saw the board in at least {MIN_HITS} fresh frames"
            )
    if not result["errors"] and not cancelled:
        result["assignments"] = assignments
        result["identities"] = identities
    return result


def _flag_slow_exits(cameras, graceful):
    for camera in cameras:
        if not camera["error"] and not graceful.get(
            (camera["kind"], camera["device_id"]), True
        ):
            camera["error"] = (
                f"camera did not release within {SHUTDOWN_SECONDS:g} s after the check "
                "(shutdown timed out); the probe was killed and the camera may need a "
                "moment to reset, so press Retry Start"
            )
    return cameras


def check_streams(detection, selected, cancel):
    """Verify selected real cameras deliver fresh frames before recorder launch.

    The caller holds its camera-operation reservation over fresh discovery and
    this entire call. Only selected iPhone/OAK slots (including the head when
    one is selected) are opened. No board is needed. All workers exit before
    this returns.
    """
    result = {
        "cameras": [],
        "errors": [],
        "cancelled": cancel.is_set(),
    }
    if result["cancelled"]:
        return result
    candidates = []
    slots = {}
    for slot, name in selected.items():
        kind = slot.rsplit("_", 1)[-1]
        if kind not in ("iphone", "oak"):
            continue
        matches = [
            device for device in detection.get(kind) or [] if device.name == name
        ]
        if len(matches) != 1:
            result["cameras"].append(
                {
                    "kind": kind,
                    "device_id": None,
                    "name": name,
                    "slot": slot,
                    "frames": 0,
                    "hits": 0,
                    "error": (
                        "selected camera was not detected; check its USB connection "
                        "and refresh devices"
                        if not matches
                        else "selected name matches multiple discovered cameras"
                    ),
                }
            )
            continue
        device = matches[0]
        key = kind, device.device_id
        error = None
        if device.in_use:
            error = "camera is in use by another process; stop that process and retry"
        elif key in slots:
            error = f"camera is also selected for {slots[key]}"
        if error:
            result["cameras"].append({**_camera_result(device, error), "slot": slot})
        else:
            slots[key] = slot
            candidates.append(device)
    if candidates and not cancel.is_set():
        session = CameraSession(
            candidates, side="health", duration=HEALTH_SCAN_SECONDS
        ).start()
        try:
            session.wait_ready(cancel)
            scanned = session.scan("health", cancel)
        finally:
            graceful = session.close(grace=0 if cancel.is_set() else None)
        for camera in _flag_slow_exits(scanned, graceful):
            result["cameras"].append(
                {**camera, "slot": slots[camera["kind"], camera["device_id"]]}
            )
    result["cancelled"] = cancel.is_set()
    for camera in result["cameras"]:
        if camera["error"]:
            result["errors"].append(
                f"{camera['slot']} ({camera['name']}): {camera['error']}"
            )
    return result


def identify_pair(detection, side, cancel):
    """Assign one iPhone + one OAK only after complete, unambiguous observation.

    The caller must refresh discovery while holding its camera-operation
    reservation, then pass that current snapshot here. One-shot: workers boot
    (up to STARTUP_SECONDS), share one SCAN_SECONDS window, then exit; at most
    STOP_GRACE_SECONDS termination grace before kill/reap. Cancellation uses the
    same release barrier. No camera data is saved.
    """
    _marker_ids(side)
    candidates, excluded, errors = pairing_candidates(detection)
    if cancel.is_set():
        return _assemble(side, [], errors, True)
    session = CameraSession(candidates, side=side).start()
    try:
        session.wait_ready(cancel)
        scanned = session.scan(side, cancel)
    finally:
        graceful = session.close(grace=0 if cancel.is_set() else None)
    return _assemble(
        side, excluded + _flag_slow_exits(scanned, graceful), errors, cancel.is_set()
    )


def identify_pair_live(session, side, cancel):
    """One scan window on a running preview session; the cameras stay open."""
    _marker_ids(side)
    if not session.settled():
        return _assemble(
            side,
            list(session.excluded),
            session.errors + ["cameras are still starting"],
            False,
        )
    scanned = session.scan(side, cancel)
    return _assemble(
        side,
        list(session.excluded) + scanned,
        session.errors,
        cancel.is_set() or session.closed,
    )


# -- worker process -------------------------------------------------------------


class _Commands:
    """Non-blocking line reader over the worker's stdin pipe."""

    def __init__(self, fd=0):
        self.fd = fd
        self.pending = b""
        self.closed = False

    def poll(self, timeout):
        if b"\n" in self.pending:
            return self._line()
        if self.closed:
            return "quit"
        readable, _, _ = select.select([self.fd], [], [], timeout)
        if not readable:
            return None
        chunk = os.read(self.fd, 4096)
        if not chunk:
            self.closed = True
            return "quit"  # The parent closed the pipe or died.
        self.pending += chunk
        return self._line() if b"\n" in self.pending else None

    def _line(self):
        line, self.pending = self.pending.split(b"\n", 1)
        return line.decode(errors="replace").strip()


def _serve(read, emit, commands, default_side, preview, duration):
    previewer = _Previewer(emit) if preview else None
    detector = _detector() if preview else None
    while True:
        command = commands.poll(0.0 if preview else 0.1)
        if command == "quit":
            return
        if command is not None:
            words = command.split()
            if not words or words[0] != "scan":
                raise RuntimeError(f"unknown camera worker command {command!r}")
            side = words[1] if len(words) > 1 else default_side
            if side not in (*MARKER_OFFSETS, "health"):
                raise RuntimeError(f"invalid scan side {side!r}")
            observer = previewer.frame if previewer is not None else None
            result = _collect_frames(read, side, duration=duration, observer=observer)
            emit({"scan": side, **result} if preview else result)
            if not preview:
                return
            continue
        if previewer is None:
            continue
        sample = read()
        if sample is not None and sample[1] is not None:
            previewer.seen(sample[1])
        previewer.tick(detector)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("iphone", "oak"), required=True)
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--side", choices=("left", "right", "health"))
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    if args.side is None and not args.preview:
        parser.error("--side is required unless --preview is given")
    _arm_parent_death(args.parent_pid)
    # Native SDKs also write directly to fd 1/2. Suppress those logs rather
    # than collecting unbounded output. A duplicate fd carries only our
    # protocol messages back to the parent.
    output_fd = os.dup(sys.stdout.fileno())
    with open(os.devnull, "w") as quiet:
        os.dup2(quiet.fileno(), 1)
        os.dup2(quiet.fileno(), 2)

    def emit(message):
        os.write(output_fd, (json.dumps(message) + "\n").encode())

    def stop(signum, frame):
        raise InterruptedError("cancelled")

    signal.signal(signal.SIGTERM, stop)
    failure = None
    try:
        factory = _iphone_frames if args.kind == "iphone" else _oak_frames
        with factory(args.device_id) as read:
            emit({"ready": True})
            _serve(read, emit, _Commands(), args.side, args.preview, args.duration)
    except Exception as exc:
        failure = str(exc)[:500] or type(exc).__name__
    if failure is not None:
        emit({"frames": 0, "hits": 0, "error": failure})
    os.close(output_fd)


if __name__ == "__main__":
    main()

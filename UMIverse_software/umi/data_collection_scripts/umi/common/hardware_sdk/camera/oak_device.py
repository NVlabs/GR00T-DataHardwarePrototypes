# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-stream OAK camera support (OAK-FFC-3P).

OAKDevice manages a single physical OAK device with a shared DepthAI pipeline.
Each socket is configured as an RGB colour camera. OAKStream wraps individual
output queues and implements the Camera interface so streams are transparent to
VideoRecorder.

Usage (FFC-3P with 3 color cameras)::

    device = OAKDevice(oak_device_name="oak-FFC-01", fps=30)
    device.build_pipeline([
        {"name": "front_cam", "socket": "CAM_A", "resolution": [640, 400]},
        {"name": "left_cam",  "socket": "CAM_B", "resolution": [640, 400]},
        {"name": "right_cam", "socket": "CAM_C", "resolution": [640, 400]},
    ])
    device.start()
    streams = device.get_streams()  # List[OAKStream], each is a Camera
"""

import time
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import depthai as dai
import numpy as np

from umi.common.data.frame import RGBCameraFrame
from umi.common.hardware_sdk.camera.camera import Camera
from umi.common.utility.print_utils import print_green, print_red, print_yellow
from umi.utils.device_manager import get_oak_device_id

_SOCKET_MAP: Dict[str, dai.CameraBoardSocket] = {
    "CAM_A": dai.CameraBoardSocket.CAM_A,
    "CAM_B": dai.CameraBoardSocket.CAM_B,
    "CAM_C": dai.CameraBoardSocket.CAM_C,
}


def _parse_socket(name: str) -> dai.CameraBoardSocket:
    if name in _SOCKET_MAP:
        return _SOCKET_MAP[name]
    if hasattr(dai.CameraBoardSocket, name):
        return getattr(dai.CameraBoardSocket, name)
    raise ValueError(
        f"Unknown socket name: {name!r}. "
        f"Expected one of {list(_SOCKET_MAP.keys())} or a valid dai.CameraBoardSocket attribute."
    )


def eeprom_intrinsics(
    device: Any,
    socket: dai.CameraBoardSocket,
    output_resolution: Optional[Tuple[int, int]],
) -> Optional[np.ndarray]:
    """K from the device's flashed calibration, scaled to the output resolution.

    The middle rung of the intrinsics cascade: per-frame metadata first, this
    next, then FALLBACK_INTRINSICS. Devices with a factory EEPROM block (OAK-D
    and friends) answer here even when their ImgTransformation carries none, so
    a module with no entry in the hardcoded table can still be solved.

    Deliberately `readCalibration()`, never `readCalibrationOrDefault()`: the
    "default" is a fabricated K, and a plausible-but-wrong K is the worst
    outcome available -- fx off by 10% costs ~29 mm of board pose, silently.

    CAVEAT: `getCameraIntrinsics` rescales for a resize, not for a crop. The
    pipeline requests outputs with `ImgResizeMode.CROP`, so when the output
    aspect differs from the sensor mode's, cx/cy carry the crop's error.
    """
    if device is None or output_resolution is None:
        return None
    try:
        calib = device.readCalibration()
        width, height = int(output_resolution[0]), int(output_resolution[1])
        matrix = calib.getCameraIntrinsics(
            socket, resizeWidth=width, resizeHeight=height
        )
    except Exception:
        return None

    K = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    # Same guard as _frame_intrinsics: an uncalibrated block reads as identity.
    if not np.isfinite(K).all() or K[0, 0] <= 1 or K[1, 1] <= 1:
        return None
    return K


def _frame_intrinsics(frame: Any) -> Optional[np.ndarray]:
    """Effective intrinsics carried by a DepthAI output frame."""
    try:
        matrix = np.asarray(
            frame.getTransformation().getIntrinsicMatrix(), dtype=np.float64
        ).reshape(3, 3)
    except Exception:
        return None

    # An uncalibrated ImgTransformation carries an identity matrix.
    if not np.isfinite(matrix).all() or matrix[0, 0] <= 1 or matrix[1, 1] <= 1:
        return None
    return matrix.copy()


class OAKDevice:
    """Manages a single physical OAK device with multiple RGB camera streams.

    Creates a shared ``dai.Pipeline`` and builds one colour camera node per
    socket.  Each stream becomes an ``OAKStream`` that implements the Camera
    interface.
    """

    def __init__(
        self,
        oak_device_name: Optional[str] = None,
        oak_device_id: Optional[str] = None,
        fps: int = 30,
        latency: float = 0.0,
        ae_max_exposure_us: Optional[int] = None,
        allow_cropped_output: bool = True,
    ) -> None:
        self.fps = fps
        self.latency = latency
        # Auto-exposure ceiling for every socket; None keeps the sensor default
        # (about one frame period).
        self.ae_max_exposure_us = ae_max_exposure_us
        # Viewers and calibration allow cropped aspect ratios by default.
        # Recording disables them unless source params explicitly opt in.
        self.allow_cropped_output = bool(allow_cropped_output)
        self.streams: List[Camera] = []

        # Resolve device ID: explicit ID > name lookup > None (first device)
        if oak_device_id is not None:
            self.device_id = oak_device_id
        elif oak_device_name is not None:
            self.device_id = get_oak_device_id(oak_device_name)
        else:
            self.device_id = None

        device_infos = dai.Device.getAllAvailableDevices()
        if len(device_infos) == 0:
            raise RuntimeError("No OAK devices found.")

        print(
            f"Available OAK devices: "
            f"{[(d.name, d.getDeviceId()) for d in device_infos]}"
        )

        if self.device_id is not None:
            device_found = False
            for device_info in device_infos:
                if device_info.getDeviceId() == self.device_id:
                    self._device_info = device_info
                    self.device = dai.Device(device_info)
                    device_found = True
                    break
            if not device_found:
                raise RuntimeError(
                    f"OAK device (ID: {self.device_id}) not found. "
                    f"Available: {[d.getDeviceId() for d in device_infos]}"
                )
        else:
            device_info = device_infos[0]
            self._device_info = device_info
            self.device = dai.Device(device_info, maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS)

        self.device_id = self.device.getDeviceId()
        print_green(f"[OAKDevice] Connected to device ID {self.device_id}")

        self.pipeline: Optional[dai.Pipeline] = None
        self._started = False

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def build_pipeline(self, stream_configs: List[Dict[str, Any]]) -> None:
        """Build the DepthAI pipeline from a list of stream config dicts.

        Each config dict requires:
          ``name``       – stream identifier (e.g. ``"front_cam"``).
          ``socket``     – board socket name (``"CAM_A"``, ``"CAM_B"``, …).
          ``resolution`` – ``[width, height]``.

        Optional per-stream keys:
          ``latency``    – per-stream latency override (seconds).

        All sockets are treated as RGB colour cameras.
        """
        self.pipeline = dai.Pipeline(self.device)
        available_sockets = self.device.getConnectedCameras()
        print(f"[OAKDevice] Available sockets: {[str(s) for s in available_sockets]}")

        # Track camera nodes by socket to avoid duplicates when the same
        # socket is used by multiple stream configs.
        camera_nodes: Dict[dai.CameraBoardSocket, Any] = {}

        # A socket can feed several streams; its sensor mode must cover the
        # largest output requested on it.
        resolutions_by_socket: Dict[dai.CameraBoardSocket, Tuple[int, int]] = {}
        for cfg in stream_configs:
            sock = _parse_socket(cfg["socket"])
            w, h = tuple(cfg["resolution"])
            prev_w, prev_h = resolutions_by_socket.get(sock, (0, 0))
            resolutions_by_socket[sock] = (max(w, prev_w), max(h, prev_h))

        def _get_or_create_camera_node(socket: dai.CameraBoardSocket):
            if socket not in available_sockets:
                raise RuntimeError(
                    f"Socket {socket} not available on device. "
                    f"Available: {[str(s) for s in available_sockets]}"
                )
            if socket not in camera_nodes:
                cam = self.pipeline.create(dai.node.Camera)
                cam.setSensorType(dai.CameraSensorType.COLOR)
                sensor_res = select_sensor_resolution(
                    self.device,
                    socket,
                    resolutions_by_socket[socket],
                    self.fps,
                    sensor_type=dai.CameraSensorType.COLOR,
                    allow_cropped_output=self.allow_cropped_output,
                )
                if sensor_res is not None:
                    cam = cam.build(
                        socket, sensorResolution=sensor_res, sensorFps=self.fps
                    )
                    print(
                        f"  [{socket}] sensor mode {sensor_res[0]}x{sensor_res[1]}@{self.fps}fps"
                    )
                else:
                    cam = cam.build(socket)
                if self.ae_max_exposure_us:
                    cam.initialControl.setAutoExposureLimit(self.ae_max_exposure_us)
                camera_nodes[socket] = cam
            return camera_nodes[socket]

        for cfg in stream_configs:
            name = cfg["name"]
            socket = _parse_socket(cfg["socket"])
            resolution = tuple(cfg["resolution"])

            cam_node = _get_or_create_camera_node(socket)
            output = cam_node.requestOutput(
                resolution,
                fps=self.fps,
                type=dai.ImgFrame.Type.BGR888i,
                resizeMode=dai.ImgResizeMode.CROP,
            )
            queue = output.createOutputQueue(maxSize=3, blocking=False)

            fallback = self._resolve_intrinsics(socket, output_resolution=resolution)
            latency = cfg.get("latency", self.latency)

            stream = OAKStream(
                device_name=name,
                camera_resolution=resolution,
                fps=self.fps,
                latency=latency,
                output_queue=queue,
                fallback_intrinsics=fallback,
                fallback_lens=self._load_fallback_lens(socket),
                oak_device=self,
                socket_name=str(cfg["socket"]),
            )
            self.streams.append(stream)
            w, h = resolution
            print(f"  [{name}] color on {cfg['socket']} @ {w}x{h}@{self.fps}fps")

        if not self.streams:
            raise RuntimeError(
                "[OAKDevice] No streams configured. Provide at least one stream config."
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._started:
            print_yellow("[OAKDevice] Pipeline already started.")
            return
        if self.pipeline is None:
            raise RuntimeError(
                "[OAKDevice] Pipeline not built. Call build_pipeline() first."
            )
        self.pipeline.start()
        time.sleep(0.5)  # allow pipeline to start producing frames
        self._started = True
        for stream in self.streams:
            stream.is_running = True
        print_green(f"[OAKDevice] Pipeline started with {len(self.streams)} stream(s).")

    def stop(self) -> None:
        for stream in self.streams:
            stream.is_running = False
        if self.pipeline is not None:
            try:
                if self.pipeline.isRunning():
                    self.pipeline.stop()
            except Exception:
                pass
        if self.device is not None:
            try:
                self.device.close()
            except Exception:
                pass
            self.device = None
        self._started = False
        print_green("[OAKDevice] Stopped.")

    def get_streams(self) -> List[Camera]:
        return list(self.streams)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sensor_name(self, socket: dai.CameraBoardSocket) -> Optional[str]:
        feature = next(
            (
                feature
                for feature in self.device.getConnectedCameraFeatures()
                if feature.socket == socket
            ),
            None,
        )
        return None if feature is None else feature.sensorName

    def _resolve_intrinsics(
        self,
        socket: dai.CameraBoardSocket,
        output_resolution: Optional[Tuple[int, int]] = None,
    ) -> Optional[np.ndarray]:
        """K for frames whose DepthAI metadata has none: EEPROM, then table."""
        K = eeprom_intrinsics(self.device, socket, output_resolution)
        if K is not None:
            return K
        return fallback_intrinsics(self._sensor_name(socket), output_resolution)

    def _load_fallback_lens(
        self, socket: dai.CameraBoardSocket
    ) -> Tuple[str, Optional[np.ndarray]]:
        """Fallback (model, D) matching `_resolve_intrinsics`. No
        resolution arg -- KB coefficients are resolution-independent."""
        return fallback_lens(self._sensor_name(socket))

    @staticmethod
    def list_connected_devices():
        """Attached OAK devices as ConnectedDevice records."""
        from umi.common.hardware_sdk.device_discovery import list_connected_oak

        return list_connected_oak()


# ======================================================================
# OAKStream — single RGB colour stream
# ======================================================================


class OAKStream(Camera):
    """A single RGB colour stream from an :class:`OAKDevice`.

    Implements the full :class:`Camera` / :class:`SensorInterface` interface
    so it is transparent to :class:`VideoRecorder`.
    """

    def __init__(
        self,
        device_name: str,
        camera_resolution: Tuple[int, int],
        fps: int,
        latency: float,
        output_queue: Any,
        fallback_intrinsics: Optional[np.ndarray],
        oak_device: "OAKDevice",
        fallback_lens: Tuple[str, Optional[np.ndarray]] = ("none", None),
        socket_name: str = "",
    ) -> None:
        super().__init__(device_name, camera_resolution, latency)
        self.fps = fps
        self._output_queue = output_queue
        self._oak_device = oak_device
        self.socket_name = socket_name
        # Detect sensors that enumerate but never deliver frames, such as the
        # phantom CAM_C on a V4 board sharing CAM_B's I2C bus. Each empty queue
        # read blocks about one second, so the count approximates startup silence.
        self._frames_delivered = 0
        self._empty_polls = 0
        self._warned_silent = False
        self._raw_intrinsics = fallback_intrinsics
        self.intrinsic_matrix = fallback_intrinsics
        # Unlike K, never refreshed per frame: DepthAI's ImgTransformation
        # carries no distortion, so this is the only source available.
        self.distortion_model, self.distortion = fallback_lens

    SILENT_SOCKET_WARN_AFTER_POLLS = 5

    def _note_empty_poll(self) -> None:
        self._empty_polls += 1
        if (
            self._warned_silent
            or self._frames_delivered
            or self._empty_polls < self.SILENT_SOCKET_WARN_AFTER_POLLS
        ):
            return
        self._warned_silent = True
        where = f" on {self.socket_name}" if self.socket_name else ""
        print_yellow(
            f"[OAKDevice] [{self.device_name}]{where} has delivered no frames "
            f"since start (~{self._empty_polls} s). The device lists this socket "
            "but nothing is streaming from it: if no module is physically "
            "connected there (the V4 baseboard reports a phantom CAM_C), remove "
            "the stream from the recording config -- a silent socket records an "
            "empty video with no other warning."
        )

    # -- SensorInterface contract --

    def start_streaming(self) -> None:
        # Pipeline is already started by OAKDevice.start()
        self.is_running = True

    def stop_streaming(self) -> None:
        # Pipeline is stopped by stop_all_oak_devices()
        self.is_running = False

    def get_frame(
        self, camera_resolution: Optional[Tuple[int, int]] = None
    ) -> Optional[RGBCameraFrame]:
        if not self.is_running:
            return None
        try:
            frame = _drain_queue_get_latest(self._output_queue)
            if frame is None:
                self._note_empty_poll()
                return None
            self._frames_delivered += 1
            receive_time = time.monotonic()
            capture_time = receive_time - self.latency if self.latency else receive_time
            img = frame.getCvFrame()

            intrinsics = _frame_intrinsics(frame)
            if intrinsics is not None:
                self._raw_intrinsics = intrinsics
            intrinsics = self._raw_intrinsics
            self.intrinsic_matrix = intrinsics

            return self._apply_resolution_override(
                RGBCameraFrame(
                    rgb=img,
                    pose=None,
                    capture_time=capture_time,
                    receive_time=receive_time,
                    intrinsics=intrinsics,
                    distortion=self.distortion,
                ),
                camera_resolution,
            )
        except Exception as e:
            print_red(f"[{self.device_name}] Failed to read frame: {e}")
            return None


# ======================================================================
# Shared helpers
# ======================================================================


# Fallback intrinsics apply only to AR0234 fisheye modules at 640x400.
# OAK-FFC boards may have no usable EEPROM calibration. These values average
# two modules measured with a 14x9 ChArUco board (approximately 0.735 pixel RMS).
# Their principal-point variation was 2.24 pixels, so this fallback can add
# approximately 0.7 degrees of ray error and does not replace per-module
# calibration. Use the matching OpenCV fisheye projection and coefficients.
AR0234_640X400_FISHEYE_K = np.array(
    [
        [198.367, 0.0, 320.001],
        [0.0, 197.878, 196.493],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
AR0234_640X400_FISHEYE_D = np.array([-0.013400, -0.004618, 0.0, 0.0], dtype=np.float64)

# (sensorName, (width, height)) -> K. Keyed on both, so a config change to any
# other resolution falls through to None rather than silently reusing a K that
# does not apply to it.
FALLBACK_INTRINSICS: Dict[Tuple[str, Tuple[int, int]], np.ndarray] = {
    ("AR0234", (640, 400)): AR0234_640X400_FISHEYE_K,
}

# Map sensor names to lens models and dimensionless distortion coefficients.
# Resizing changes K, not D. AR0234 implies fisheye only for the UMI lens
# module; another lens requires a corresponding calibration and model.
# IMX378 uses a rectilinear model with no fitted coefficients.
FALLBACK_LENS: Dict[str, Tuple[str, Optional[np.ndarray]]] = {
    "AR0234": ("fisheye", AR0234_640X400_FISHEYE_D),
    "IMX378": ("pinhole", None),
}


def fallback_intrinsics(
    sensor_name: Optional[str], output_resolution: Optional[Tuple[int, int]]
) -> Optional[np.ndarray]:
    """Hardcoded K for sensors with no on-board calibration. See table above."""
    if sensor_name is None or output_resolution is None:
        return None
    K = FALLBACK_INTRINSICS.get((sensor_name, tuple(output_resolution)))
    return None if K is None else K.copy()


def fallback_lens(sensor_name: Optional[str]) -> Tuple[str, Optional[np.ndarray]]:
    """(projection model, D) for a sensor. See the note on FALLBACK_LENS.

    Unknown sensors get ("none", None), a zero-distortion solve, rather than a
    guess, because guessing the model is the expensive error.
    """
    model, D = FALLBACK_LENS.get(sensor_name, ("none", None))
    return model, (None if D is None else D.copy())


# Prefer a full-field-of-view sensor mode independent of requested FPS,
# provided it supports the target size and rate:
# IMX378: 2024x1520, 4:3, 2x2 binning, up to 85 FPS.
# AR0234: 1920x1200, 16:10, up to 60 FPS on RVC2.
# OV9782/OV9282: 1280x800, 16:10; OV9782 supports up to 129 FPS.
# Otherwise use the generic mode ranking.
PREFERRED_SENSOR_MODES: Dict[str, Tuple[int, int]] = {
    "IMX378": (2024, 1520),
    "AR0234": (1920, 1200),
    "OV9782": (1280, 800),
    "OV9282": (1280, 800),
}


def select_sensor_resolution(
    device,
    socket: dai.CameraBoardSocket,
    output_resolution: Tuple[int, int],
    fps: float,
    sensor_type: Optional[Any] = None,
    allow_cropped_output: bool = False,
) -> Optional[Tuple[int, int]]:
    """Pick the sensor mode that sustains ``fps`` without cropping the FOV.

    depthai's built-in auto-selection picks the smallest sensor mode that
    covers the requested output and **ignores fps entirely**. On IMX378 that
    always lands on 1352x1012 (52fps ceiling), so fps=60 fails even though the
    sensor does 85fps at 2024x1520. Sensor mode tables are also not monotonic
    in resolution, so "smaller is faster" cannot be assumed.

    Selection order:
      1. ``PREFERRED_SENSOR_MODES[sensor]``, if it covers the output and
         sustains ``fps`` — keeps FOV fixed across frame rates
      2. otherwise: modes that cover the output and sustain ``fps``, preferring
         the sensor's native aspect ratio (uncropped FOV), then the smallest

    ``sensor_type`` must match what the Camera node will request: a MONO-only
    mode (e.g. OV9282's 640x400) is unusable by a node set to COLOR, and
    pinning it makes pipeline start fail.

    Returns None if the sensor is unknown, in which case the caller should fall
    back to depthai's auto-selection.
    """
    feature = next(
        (f for f in device.getConnectedCameraFeatures() if f.socket == socket), None
    )
    if feature is None or not feature.configs:
        return None

    out_w, out_h = output_resolution
    native_aspect = feature.width / feature.height

    # Dedupe by size — a mode can appear once per sensor type (COLOR/MONO);
    # keep the highest ceiling advertised for that size.
    by_size: Dict[Tuple[int, int], float] = {}
    for c in feature.configs:
        if sensor_type is not None and c.type != sensor_type:
            continue
        key = (c.width, c.height)
        by_size[key] = max(by_size.get(key, 0.0), c.maxFps)
    if not by_size:
        return None

    covering = [(w, h, f) for (w, h), f in by_size.items() if w >= out_w and h >= out_h]
    usable = [m for m in covering if m[2] >= fps - 0.01]
    if not usable:
        modes = ", ".join(
            f"{w}x{h}@{f:.0f}fps" for w, h, f in sorted(covering, reverse=True)
        )
        raise RuntimeError(
            f"[{feature.sensorName}] {socket}: no sensor mode covering "
            f"{out_w}x{out_h} can sustain {fps}fps. Available: {modes or 'none'}"
        )

    # Reject outputs that crop the sensor's native aspect ratio during recording.
    # An intentional crop requires allow_cropped_output: true in source params
    # so the field-of-view change remains part of session provenance.
    if abs(out_w / out_h - native_aspect) > 0.02:
        message = (
            f"[{feature.sensorName}] {socket}: output {out_w}x{out_h} "
            f"(aspect {out_w / out_h:.2f}) does not match the sensor's native "
            f"{native_aspect:.2f} — the ISP would crop and you would lose FOV. "
            f"Closest uncropped size: {out_w}x{round(out_w / native_aspect)}."
        )
        if not allow_cropped_output:
            raise RuntimeError(
                message
                + " Fix camera_resolution in the config (re-run the device scan "
                "and Apply again), or set allow_cropped_output: true to record "
                "cropped on purpose."
            )
        print_yellow(message + " Recording cropped: allow_cropped_output is set.")

    preferred = PREFERRED_SENSOR_MODES.get(feature.sensorName)
    if preferred is not None and preferred in {(w, h) for w, h, _ in usable}:
        return preferred

    # Native-aspect modes preserve the full FOV; a 16:9 mode on a 4:3 sensor is
    # a vertical crop, and the ISP then crops horizontally to reach the output
    # aspect. Fall back to all usable modes if none match natively.
    native = [m for m in usable if abs(m[0] / m[1] - native_aspect) < 0.02]
    candidates = native or usable
    w, h, _ = min(candidates, key=lambda m: m[0] * m[1])
    return (w, h)


def _drain_queue_get_latest(queue, timeout_ms: int = 1000) -> Optional[Any]:
    """Drain a DepthAI output queue and return the latest message."""
    latest = queue.tryGet()
    if latest is None:
        try:
            latest = queue.get(timeout=timedelta(milliseconds=timeout_ms))
        except Exception:
            return None
    if latest is None:
        return None
    while True:
        frame = queue.tryGet()
        if frame is None:
            break
        latest = frame
    return latest


# ======================================================================
# CLI example
# ======================================================================

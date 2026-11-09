# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from collections import deque
from datetime import timedelta
from typing import Any, Deque, Dict, List, Optional, Tuple

import depthai as dai
import numpy as np

from umi.common.data.frame import RGBCameraFrame
from umi.common.hardware_sdk.camera.camera import Camera
from umi.common.utility.print_utils import print_green, print_red, print_yellow
from umi.utils.device_manager import normalize_device_id


def _oak_identity(expected_id: Optional[str]) -> dict:
    return {"kind": "oak", "name": None, "id": normalize_device_id(expected_id)}


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


class OAKCamera(Camera):
    # Despite the RGB frame-class names, getCvFrame() supplies BGR pixels.
    # Consumers that require RGB must convert explicitly.
    def __init__(
        self,
        device_name: str = "OAK Camera",
        camera_resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
        latency: float = 0.0,
        oak_device_id: Optional[str] = None,
        enable_color: bool = True,
        enable_mono_cameras: bool = False,
    ) -> None:
        """
        Initialize OAK camera.

        Args:
            device_name: Identifier name for this camera instance.
            camera_resolution: Tuple (width, height) for the RGB camera.
            fps: Desired frames per second.
            latency: Artificial latency adjustment (optional).
            oak_device_id: depthai device ID of the camera to open (e.g.
                          "1944301000000000A1"). Without one, the first available
                          OAK is used.
            enable_color: Stream CAM_A, the RGB camera. Must be true: the
                         recorder records only the RGB stream.
            enable_mono_cameras: Must be false; the monochrome cameras are not
                                 streamed.
        """
        if not enable_color or enable_mono_cameras:
            raise ValueError(
                f"[{device_name}] OAKCamera streams only the RGB camera (CAM_A): "
                "set enable_color: true and enable_mono_cameras: false."
            )
        super().__init__(device_name, camera_resolution, latency)
        self.fps = fps

        self.pipeline: Optional[dai.Pipeline] = None
        self.output_queues: Dict[str, Any] = {}

        # None opens the first device; self.device_id becomes the opened
        # device's id below.
        self.device_id = oak_device_id
        self._expected_device_id = self.device_id

        device_infos = dai.Device.getAllAvailableDevices()
        if len(device_infos) == 0:
            raise RuntimeError("No OAK devices found.")

        print(
            f"Available OAK devices: {[(d.name, d.getDeviceId()) for d in device_infos]}"
        )

        if self.device_id is not None:
            device_found = False
            for device_info in device_infos:
                if device_info.getDeviceId() == self.device_id:
                    self.device = dai.Device(device_info)
                    device_found = True
                    break
            if not device_found:
                raise RuntimeError(
                    f"OAK device (ID: {self.device_id}) not found. "
                    f"Available: {[d.getDeviceId() for d in device_infos]}"
                )
        else:
            self.device = dai.Device(
                device_infos[0], maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS
            )

        self.device_id = self.device.getDeviceId()
        print_green(f"[{self.device_name}] Connected to device ID {self.device_id}")
        self.set_intrinsic_matrix()

    def identity(self) -> dict:
        return _oak_identity(self._expected_device_id)

    def observed_identity(self) -> dict:
        return {"id": normalize_device_id(self.device_id)}

    @staticmethod
    def list_connected_devices() -> List[Dict[str, Any]]:
        """
        List all connected OAK devices.

        Returns:
            list: List of dicts with 'name', 'device_id', 'index', 'state' keys

        Example:
            >>> devices = OAKCamera.list_connected_devices()
            >>> for dev in devices:
            ...     print(f"Device {dev['index']}: {dev['device_id']}")
        """
        cameras = []
        device_infos = dai.Device.getAllAvailableDevices()
        print(f"Found {len(device_infos)} OAK device(s):")
        for index, device_info in enumerate(device_infos):
            cam = {
                "name": device_info.name,
                "device_id": device_info.getDeviceId(),
                "index": index,
                "state": str(device_info.state),
            }
            cameras.append(cam)
            print(f"  [{index}] ID: {cam['device_id']}, Name: {cam['name']}")
        return cameras

    def start_streaming(self):
        if self.is_running:
            print_yellow(f"[{self.device_name}] Already streaming.")
            return

        sockets = self.device.getConnectedCameras()
        print(f"Available camera sockets: {[str(s) for s in sockets]}")

        self.pipeline = dai.Pipeline(self.device)
        self.output_queues = {}

        if dai.CameraBoardSocket.CAM_A in sockets:
            cam_rgb = self.pipeline.create(dai.node.Camera)
            cam_rgb = cam_rgb.build(dai.CameraBoardSocket.CAM_A)
            self.output_queues["color"] = cam_rgb.requestOutput(
                self.camera_resolution, fps=self.fps, enableUndistortion=False
            ).createOutputQueue(maxSize=3, blocking=False)
            print(
                f"Enabled CAM_A (RGB) @ {self.camera_resolution[0]}x{self.camera_resolution[1]}@{self.fps}fps"
            )

        if len(self.output_queues) == 0:
            raise RuntimeError(
                f"[{self.device_name}] No RGB camera (CAM_A) on this device. "
                f"Available sockets: {[str(s) for s in sockets]}"
            )

        self.pipeline.start()
        time.sleep(0.5)  # Give pipeline time to start producing frames

        self.is_running = True
        print_green(f"[{self.device_name}] Streaming started.")

    def set_intrinsic_matrix(self, matrix=None):
        """
        Set the camera's intrinsic matrix.
        If matrix is None, loads intrinsics from the OAK device's stored calibration.
        """
        if matrix is not None:
            self.intrinsic_matrix = matrix
            print(f"{self.device_name}: Intrinsic matrix set.")
            return

        if self.device is None:
            print_yellow(
                f"[{self.device_name}] Device not connected. Cannot load intrinsics."
            )
            return

        try:
            calib_data = self.device.readFactoryCalibration()
            M_rgb, width, height = calib_data.getDefaultIntrinsics(
                dai.CameraBoardSocket.CAM_A
            )
            self.intrinsic_matrix = np.array(M_rgb)
            print(
                f"{self.device_name}: Intrinsic matrix loaded from device calibration."
            )
            print(f"  Resolution: {width}x{height}")
            print(f"  Matrix:\n{self.intrinsic_matrix}")
        except Exception as e:
            self.intrinsic_matrix = None
            print_yellow(
                f"[{self.device_name}] Could not load intrinsics from device "
                f"(will use None): {e}"
            )

    def get_frame(
        self, camera_resolution: Optional[Tuple[int, int]] = None
    ) -> Optional[RGBCameraFrame]:
        """
        Get the latest frame from the camera.

        Returns:
            RGBCameraFrame, or None if not available.
        """
        if not self.is_running or self.pipeline is None:
            return None

        if not self.pipeline.isRunning():
            print_red(f"[{self.device_name}] Pipeline stopped unexpectedly.")
            return None

        if not self.device.isPipelineRunning():
            print_red(f"[{self.device_name}] Device disconnected.")
            return None

        try:
            rgb_frame = _drain_queue_get_latest(self.output_queues["color"])
            receive_time = time.monotonic()

            if rgb_frame is None:
                print_yellow(f"[{self.device_name}] No frame available yet.")
                return None

            capture_time = receive_time - self.latency if self.latency else receive_time

            return self._apply_resolution_override(
                RGBCameraFrame(
                    rgb=rgb_frame.getCvFrame(),
                    pose=None,
                    capture_time=capture_time,
                    receive_time=receive_time,
                    intrinsics=self.intrinsic_matrix,
                ),
                camera_resolution,
            )

        except Exception as e:
            print_red(f"[{self.device_name}] Failed to read frame: {e}")
            return None

    def stop_streaming(self):
        if self.pipeline is not None and self.pipeline.isRunning():
            self.pipeline.stop()
        if self.device is not None:
            self.device.close()
            self.device = None

        self.pipeline = None
        self.output_queues = {}
        self.is_running = False
        print_green(f"[{self.device_name}] Streaming stopped.")


# --- Example Usage ---
if __name__ == "__main__":
    import argparse

    import cv2

    parser = argparse.ArgumentParser(
        description="Stream all connected OAK cameras with live FPS display."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[640, 400],
        metavar=("WIDTH", "HEIGHT"),
        help="Capture resolution (default: 640 400)",
    )
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    parser.add_argument(
        "--oak-device-id",
        type=str,
        default=None,
        metavar="ID",
        help="If set, stream only this camera (its depthai device ID). "
        "Otherwise all connected OAK cameras are shown.",
    )
    args = parser.parse_args()

    resolution = tuple(args.resolution)

    print("Scanning for OAK cameras...")
    device_infos = dai.Device.getAllAvailableDevices()
    if not device_infos:
        print("No OAK devices found.")
        raise SystemExit(1)

    if args.oak_device_id is not None:
        device_infos = [
            d for d in device_infos if d.getDeviceId() == args.oak_device_id
        ]
        if not device_infos:
            raise SystemExit(
                f"OAK device {args.oak_device_id} not found. "
                f"Available: {[d.getDeviceId() for d in dai.Device.getAllAvailableDevices()]}"
            )
        print(f"  Filtering to device: {args.oak_device_id}")

    cameras: List[OAKCamera] = []
    for i, dev_info in enumerate(device_infos):
        dev_id = dev_info.getDeviceId()
        print(f"  Connecting to device [{i}]: {dev_id}")
        cam = OAKCamera(
            device_name=f"OAK-{i}",
            camera_resolution=resolution,
            fps=args.fps,
            oak_device_id=dev_id,
        )
        cam.start_streaming()
        cameras.append(cam)

    fps_windows: List[Deque[float]] = [deque(maxlen=30) for _ in cameras]
    print(f"\nStreaming {len(cameras)} OAK camera(s). Press 'q' to quit.\n")

    for i, cam in enumerate(cameras):
        cv2.namedWindow(f"OAK [{i}] {cam.device_id[:12]}", cv2.WINDOW_NORMAL)

    try:
        while True:
            for i, cam in enumerate(cameras):
                frame_data = cam.get_frame()
                if frame_data is None or frame_data.rgb is None:
                    continue

                fps_windows[i].append(time.monotonic())
                window = fps_windows[i]
                measured_fps = (
                    (len(window) - 1) / (window[-1] - window[0])
                    if len(window) > 1
                    else 0.0
                )

                img = frame_data.rgb.copy()
                label = f"{cam.device_name} | {cam.device_id[:8]}...  {measured_fps:.1f} fps"
                cv2.putText(
                    img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                cv2.imshow(f"OAK [{i}] {cam.device_id[:12]}", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        for cam in cameras:
            cam.stop_streaming()
        cv2.destroyAllWindows()

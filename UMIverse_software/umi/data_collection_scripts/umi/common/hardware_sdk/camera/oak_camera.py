# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import depthai as dai
import numpy as np

from umi.common.data.frame import RGBCameraFrame, RGBDCameraFrame
from umi.common.hardware_sdk.camera.camera import Camera
from umi.common.hardware_sdk.camera.oak_device import (
    _drain_queue_get_latest,
    _frame_intrinsics,
    eeprom_intrinsics,
    fallback_intrinsics,
    fallback_lens,
    select_sensor_resolution,
)
from umi.common.utility.print_utils import print_green, print_red, print_yellow
from umi.utils.device_manager import get_oak_device_id


class OAKCamera(Camera):
    # Despite the RGB frame-class names, getCvFrame() supplies BGR pixels.
    # Consumers that require RGB must convert explicitly.

    # Provide lens attributes at class level because intrinsics initialization
    # can return before resolving per-device calibration.
    distortion_model: str = "none"
    distortion: Optional[np.ndarray] = None

    def __init__(
        self,
        device_name: str = "OAK Camera",
        camera_resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
        sensor_resolution: Optional[Tuple[int, int]] = None,
        latency: float = 0.0,
        oak_device_name: Optional[str] = None,
        oak_device_id: Optional[str] = None,
        enable_color: bool = True,
        enable_mono_cameras: bool = False,
        monochrome_resolution: Tuple[int, int] = (640, 480),
        enable_depth: bool = False,
        depth_resolution: Tuple[int, int] = (640, 400),
        debug_timing: bool = False,
        ae_max_exposure_us: Optional[int] = None,
        allow_cropped_output: bool = True,
    ) -> None:
        """
        Initialize OAK camera.

        Args:
            device_name: Identifier name for this camera instance.
            camera_resolution: Tuple (width, height) for the RGB camera.
            fps: Desired frames per second.
            sensor_resolution: Optional (width, height) sensor mode to pin for the
                RGB camera. Leave as None — select_sensor_resolution() picks the
                full-FOV mode that sustains `fps` (IMX378 → 2024x1520, up to
                85fps). Set this only to override that choice; see
                list_sensor_configs() for the modes a device supports.
            latency: Artificial latency adjustment (optional).
            oak_device_name: Device name from OAK_DEVICES map (e.g., "oak_1").
                            If provided, looks up device ID automatically.
            oak_device_id: Raw depthai device ID string (the device's MxId).
                          Takes priority over oak_device_name. Useful when iterating
                          over all connected devices without a name mapping.
            enable_color: Enable CAM_A (RGB camera).
            enable_mono_cameras: Enable CAM_B & CAM_C (Monochrome stereo pair).
            monochrome_resolution: Tuple (width, height) for monochrome cameras.
            enable_depth: Enable stereo depth via CAM_B + CAM_C + StereoDepth node.
                         Depth is aligned to the RGB frame (CAM_A). Requires an
                         OAK-D series device. get_frame() returns RGBDCameraFrame
                         when enabled.
            depth_resolution: Tuple (width, height) for the mono cameras used by
                              StereoDepth. Common values: (640, 400), (1280, 720).
            debug_timing: Print per-step wall-clock timings inside get_frame()
                         every frame so you can pinpoint which queue or decode
                         step is contributing most to depth latency.
            ae_max_exposure_us: Auto-exposure ceiling for CAM_A, in microseconds.
                         None keeps the sensor default (about one frame period).
        """
        super().__init__(device_name, camera_resolution, latency)
        self.oak_device_name = oak_device_name
        self.fps = fps
        self.sensor_resolution = (
            tuple(sensor_resolution) if sensor_resolution is not None else None
        )
        self.enable_color = enable_color
        self.enable_mono_cameras = enable_mono_cameras
        self.monochrome_resolution = monochrome_resolution
        self.enable_depth = enable_depth
        self.depth_resolution = depth_resolution
        self.debug_timing = debug_timing
        self.ae_max_exposure_us = ae_max_exposure_us
        # Allow crop requests for viewers and calibration; recording explicitly
        # requires the native sensor aspect ratio.
        self.allow_cropped_output = bool(allow_cropped_output)
        self._device_info = None  # set in __init__ for platform detection (e.g. RVC4)

        self.pipeline: Optional[dai.Pipeline] = None
        self.output_queues: Dict[str, Any] = {}

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
            f"Available OAK devices: {[(d.name, d.getDeviceId()) for d in device_infos]}"
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
        print_green(f"[{self.device_name}] Connected to device ID {self.device_id}")
        self.set_intrinsic_matrix()

    @staticmethod
    def list_connected_devices():
        """Attached OAK devices as ConnectedDevice records."""
        from umi.common.hardware_sdk.device_discovery import list_connected_oak

        return list_connected_oak()

    def list_sensor_configs(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List the sensor modes each connected camera socket supports, with their
        FPS ceilings. Use this to pick a valid `sensor_resolution` for a target FPS.
        """
        configs: Dict[str, List[Dict[str, Any]]] = {}
        for feat in self.device.getConnectedCameraFeatures():
            entries = []
            for c in feat.configs:
                entries.append(
                    {
                        "width": c.width,
                        "height": c.height,
                        "type": str(c.type),
                        "min_fps": c.minFps,
                        "max_fps": c.maxFps,
                    }
                )
            configs[str(feat.socket)] = entries
            print(f"{feat.socket} ({feat.sensorName}):")
            for e in entries:
                print(
                    f"  {e['width']}x{e['height']} {e['type']} "
                    f"fps[{e['min_fps']:.2f}, {e['max_fps']:.2f}]"
                )
        return configs

    def start_streaming(self):
        if self.is_running:
            print_yellow(f"[{self.device_name}] Already streaming.")
            return

        sockets = self.device.getConnectedCameras()
        print(f"Available camera sockets: {[str(s) for s in sockets]}")

        self.pipeline = dai.Pipeline(self.device)
        self.output_queues = {}

        # Route color through Sync when depth is enabled so both arrive as one
        # synchronized frame group.
        color_output = None
        if self.enable_color and dai.CameraBoardSocket.CAM_A in sockets:
            cam_rgb = self.pipeline.create(dai.node.Camera)
            # Pick the sensor mode explicitly: depthai's auto-selection ignores
            # the requested fps and can land on a mode that caps below it.
            sensor_res = self.sensor_resolution or select_sensor_resolution(
                self.device,
                dai.CameraBoardSocket.CAM_A,
                self.camera_resolution,
                float(self.fps),
                sensor_type=dai.CameraSensorType.COLOR,
                allow_cropped_output=self.allow_cropped_output,
            )
            if sensor_res is not None:
                cam_rgb = cam_rgb.build(
                    dai.CameraBoardSocket.CAM_A,
                    sensorResolution=sensor_res,
                    sensorFps=float(self.fps),
                )
                print(
                    f"CAM_A sensor mode {sensor_res[0]}x{sensor_res[1]}@{self.fps}fps"
                )
            else:
                cam_rgb = cam_rgb.build(dai.CameraBoardSocket.CAM_A)
            if self.ae_max_exposure_us:
                cam_rgb.initialControl.setAutoExposureLimit(self.ae_max_exposure_us)
            # BGR888i: converted on-device, matching OAKDevice. Costs 2x the USB
            # bytes of NV12 but needs no host-side colour conversion.
            color_output = cam_rgb.requestOutput(
                self.camera_resolution,
                type=dai.ImgFrame.Type.BGR888i,
                fps=self.fps,
                enableUndistortion=self.enable_depth,
                resizeMode=dai.ImgResizeMode.CROP,
            )
            if not self.enable_depth:
                self.output_queues["color"] = color_output.createOutputQueue(
                    maxSize=3, blocking=False
                )
            print(
                f"Enabled CAM_A (RGB) @ {self.camera_resolution[0]}x{self.camera_resolution[1]}@{self.fps}fps"
            )

        if self.enable_mono_cameras:
            if dai.CameraBoardSocket.CAM_B in sockets:
                cam_mono_left = self.pipeline.create(dai.node.Camera)
                cam_mono_left = cam_mono_left.build(dai.CameraBoardSocket.CAM_B)
                self.output_queues["mono_left"] = cam_mono_left.requestOutput(
                    self.monochrome_resolution,
                    fps=self.fps,
                ).createOutputQueue(maxSize=3, blocking=False)
                w, h = self.monochrome_resolution
                print(f"Enabled CAM_B (Mono Left) @ {w}x{h}@{self.fps}fps")

            if dai.CameraBoardSocket.CAM_C in sockets:
                cam_mono_right = self.pipeline.create(dai.node.Camera)
                cam_mono_right = cam_mono_right.build(dai.CameraBoardSocket.CAM_C)
                self.output_queues["mono_right"] = cam_mono_right.requestOutput(
                    self.monochrome_resolution,
                    fps=self.fps,
                ).createOutputQueue(maxSize=3, blocking=False)
                w, h = self.monochrome_resolution
                print(f"Enabled CAM_C (Mono Right) @ {w}x{h}@{self.fps}fps")

        if self.enable_depth:
            has_b = dai.CameraBoardSocket.CAM_B in sockets
            has_c = dai.CameraBoardSocket.CAM_C in sockets
            if not (has_b and has_c):
                raise RuntimeError(
                    f"[{self.device_name}] enable_depth=True requires both CAM_B and "
                    f"CAM_C (stereo pair). Available sockets: {[str(s) for s in sockets]}"
                )

            cam_b = self.pipeline.create(dai.node.Camera)
            cam_b = cam_b.build(dai.CameraBoardSocket.CAM_B)
            cam_c = self.pipeline.create(dai.node.Camera)
            cam_c = cam_c.build(dai.CameraBoardSocket.CAM_C)

            left_out = cam_b.requestOutput(self.depth_resolution, fps=self.fps)
            right_out = cam_c.requestOutput(self.depth_resolution, fps=self.fps)

            stereo = self.pipeline.create(dai.node.StereoDepth)
            stereo.setExtendedDisparity(True)
            left_out.link(stereo.left)
            right_out.link(stereo.right)

            if color_output is not None:
                sync = self.pipeline.create(dai.node.Sync)
                sync.setSyncThreshold(timedelta(seconds=1 / (2 * self.fps)))
                color_output.link(sync.inputs["rgb"])
                platform = self.device.getPlatform()
                if platform == dai.Platform.RVC4:
                    align = self.pipeline.create(dai.node.ImageAlign)
                    stereo.depth.link(align.input)
                    color_output.link(align.inputAlignTo)
                    align.outputAligned.link(sync.inputs["depth_aligned"])
                else:
                    stereo.depth.link(sync.inputs["depth_aligned"])
                    color_output.link(stereo.inputAlignTo)
                self.output_queues["synced"] = sync.out.createOutputQueue(
                    maxSize=3, blocking=False
                )
            else:
                self.output_queues["depth"] = stereo.depth.createOutputQueue(
                    maxSize=3, blocking=False
                )

        if len(self.output_queues) == 0:
            raise RuntimeError(
                "No camera output queues were enabled. Check device capabilities."
            )

        self.pipeline.start()
        time.sleep(0.5)  # Give pipeline time to start producing frames

        self.is_running = True
        print_green(f"[{self.device_name}] Streaming started.")

    def set_intrinsic_matrix(self, matrix=None):
        """Set a temporary K, or reset to the sensor-specific fallback."""
        if matrix is not None:
            super().set_intrinsic_matrix(matrix)
            return

        if self.device is None:
            self.intrinsic_matrix = None
            return

        feature = next(
            (
                feature
                for feature in self.device.getConnectedCameraFeatures()
                if feature.socket == dai.CameraBoardSocket.CAM_A
            ),
            None,
        )
        sensor_name = None if feature is None else feature.sensorName
        # Prefer EEPROM calibration; it can support modules absent from the fallback
        # table.
        K = eeprom_intrinsics(
            self.device, dai.CameraBoardSocket.CAM_A, self.camera_resolution
        )
        if K is None:
            K = fallback_intrinsics(sensor_name, self.camera_resolution)
        self.intrinsic_matrix = K
        # Resolution-independent, so it survives a set_intrinsic_matrix() reset.
        self.distortion_model, self.distortion = fallback_lens(sensor_name)

    def _intrinsics_for_frame(self, frame: Any) -> Optional[np.ndarray]:
        """Return output-coordinate K, retaining the last valid fallback."""
        intrinsics = _frame_intrinsics(frame)
        if intrinsics is not None:
            self.intrinsic_matrix = intrinsics
        return self.intrinsic_matrix

    def get_frame(
        self, camera_resolution: Optional[Tuple[int, int]] = None
    ) -> Optional[RGBCameraFrame]:
        """
        Get the latest frame from the camera.

        Returns:
            RGBDCameraFrame (rgb + depth) when enable_depth=True, otherwise
            RGBCameraFrame (rgb only). Returns None if not available.
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
            # ------------------------------------------------------------------
            # Path A: synchronised RGB + depth (Sync node; pixel-, time-, resolution-aligned)
            # ------------------------------------------------------------------
            if "synced" in self.output_queues:
                t0 = time.monotonic()
                msg_grp = _drain_queue_get_latest(self.output_queues["synced"])
                t1 = time.monotonic()

                if msg_grp is None:
                    print_yellow(
                        f"[{self.device_name}] No rgb-depth synced frame available yet."
                    )
                    return None

                receive_time = t1
                capture_time = (
                    receive_time - self.latency if self.latency else receive_time
                )

                rgb_frame = msg_grp["rgb"]
                rgb_image = rgb_frame.getCvFrame()
                depth_frame = msg_grp["depth_aligned"]
                depth_image = np.array(depth_frame.getFrame(), dtype=np.uint16)
                intrinsics = self._intrinsics_for_frame(rgb_frame)
                t2 = time.monotonic()

                if self.debug_timing:
                    print(
                        f"[{self.device_name}] get_frame timing  total={(t2 - t0) * 1000:.1f}ms | "
                        f"sync_queue={(t1 - t0) * 1000:.1f}ms  decode={(t2 - t1) * 1000:.1f}ms"
                    )

                return self._apply_resolution_override(
                    RGBDCameraFrame(
                        rgb=rgb_image,
                        depth=depth_image,
                        pose=None,
                        capture_time=capture_time,
                        receive_time=receive_time,
                        intrinsics=intrinsics,
                        distortion=self.distortion,
                    ),
                    camera_resolution,
                )

            # ------------------------------------------------------------------
            # Path B: colour only (no depth), or depth-only via separate queue
            # ------------------------------------------------------------------
            if "color" not in self.output_queues:
                print_yellow(f"[{self.device_name}] Color camera not enabled.")
                return None

            t0 = time.monotonic()
            rgb_frame = _drain_queue_get_latest(self.output_queues["color"])
            t1 = time.monotonic()

            if rgb_frame is None:
                print_yellow(f"[{self.device_name}] No frame available yet.")
                return None

            receive_time = t1
            capture_time = receive_time - self.latency if self.latency else receive_time

            rgb_image = rgb_frame.getCvFrame()
            intrinsics = self._intrinsics_for_frame(rgb_frame)
            t2 = time.monotonic()

            if self.debug_timing:
                print(
                    f"[{self.device_name}] get_frame timing  total={(t2 - t0) * 1000:.1f}ms | "
                    f"rgb_queue={(t1 - t0) * 1000:.1f}ms  decode={(t2 - t1) * 1000:.1f}ms"
                )

            return self._apply_resolution_override(
                RGBCameraFrame(
                    rgb=rgb_image,
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

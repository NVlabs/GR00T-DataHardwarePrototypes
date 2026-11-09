# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-2.1-only
#
# Adapted material: Record3D demo-main.py, by Marek Simonik.
# Reference: https://github.com/marek-simonik/record3d/blob/79e6f44697150a4f036034af8fec903d11ac6f00/demo-main.py
# The reference identifies matching material; the original copy revision is unknown.
# NVIDIA adaptations include Camera/RGBDCameraFrame integration, UDID selection,
# device identity reporting, capture timestamps, pose conversion, frame backlog
# handling, depth conversion, resolution/calibration handling, and diagnostics.
# 2026-11-05: corrected licensing and documented existing adaptations;
# executable code is unchanged by this notice correction.
# This adapter, including its adaptations, is distributed under LGPL-2.1-only.
# See NOTICE, LICENSES/Record3D/README.md, and
# LICENSES/DexUMI/record3d-79e6f446/LICENSE for attribution and license terms.
# Distributed WITHOUT ANY WARRANTY; see the LGPL for details.

import time
import traceback
from threading import Event
from typing import Optional, Tuple

import cv2
import numpy as np
from record3d import Record3DStream
from scipy.spatial.transform import Rotation as R

from umi.common.data.frame import RGBDCameraFrame
from umi.common.hardware_sdk.camera.camera import Camera
from umi.common.utility.print_utils import print_green, print_red, print_yellow
from umi.utils.device_manager import normalize_device_id


class IphoneCamera(Camera):
    # Record3D always streams RGB at this native (width, height). Any
    # camera_resolution override must preserve the 3:4 aspect ratio; RGB
    # frames are resized to the requested size in get_frame().
    NATIVE_RGB_RESOLUTION = (720, 960)

    def __init__(
        self,
        device_name: str = "IphoneCamera",
        camera_resolution: Tuple[int, int] = NATIVE_RGB_RESOLUTION,
        latency: float = 0.143,
        iphone_udid: Optional[str] = None,
    ) -> None:
        """
        Initialize iPhone camera.

        Args:
            device_name: Identifier name for this camera instance.
            camera_resolution: Tuple (width, height) for the RGB frame. Must
                preserve the native 3:4 aspect ratio (720x960). Defaults to
                the native resolution; otherwise frames are resized.
            latency: Latency adjustment in seconds.
            iphone_udid: UDID of the iPhone to open. Without one, the first
                connected iPhone is used.

        Note: depth is always returned at native LiDAR resolution (192x256)
            for storage efficiency. Consumers that need depth aligned to RGB
            (e.g. ArUco marker pose estimation) should resize at use time.
        """
        if camera_resolution is None:
            camera_resolution = self.NATIVE_RGB_RESOLUTION
        native_w, native_h = self.NATIVE_RGB_RESOLUTION
        req_w, req_h = camera_resolution
        # Integer cross-product keeps the check exact for any valid size.
        if req_w * native_h != req_h * native_w:
            raise ValueError(
                f"[{device_name}] camera_resolution {req_w}x{req_h} does not "
                f"preserve the iPhone's native aspect ratio "
                f"({native_w}x{native_h}, 3:4)."
            )
        super().__init__(device_name, camera_resolution, latency)
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.udid = iphone_udid or None
        # connect_to_device() overwrites self.udid when it falls back to the
        # first phone; keep what was asked for and what was opened apart.
        self._expected_udid = self.udid
        self._observed_udid = None

    @staticmethod
    def list_connected_devices():
        """
        List all connected iPhone devices.

        Returns:
            list: List of connected device objects with .udid and .product_id attributes

        Example:
            >>> devices = IphoneCamera.list_connected_devices()
            >>> for i, dev in enumerate(devices):
            ...     print(f"Device {i}: {dev.udid}")
        """
        devs = Record3DStream.get_connected_devices()
        print(f"Found {len(devs)} iPhone device(s):")
        for i, dev in enumerate(devs):
            print(f"  [{i}] UDID: {dev.udid}, Product ID: {dev.product_id}")
        return devs

    def on_new_frame(self):
        """
        This method is called from non-main thread, therefore cannot be used for presenting UI.
        """
        self.event.set()  # Notify the main thread to stop waiting and process new frame.

    def on_stream_stopped(self):
        print_red(
            f"[{self.device_name}] Stream stopped. Please check if the device is connected."
        )

    def connect_to_device(self):
        print("Searching for devices")
        devs = Record3DStream.get_connected_devices()

        if len(devs) == 0:
            raise RuntimeError("No iPhone devices found.")

        target_dev = None
        if self.udid is not None:
            for dev in devs:
                if dev.udid == self.udid:
                    target_dev = dev
                    print_green(
                        f"[{self.device_name}] Found device (UDID: {self.udid})"
                    )
                    break
            if target_dev is None:
                raise RuntimeError(
                    f"Cannot find device with UDID: {self.udid}. "
                    f"Available devices: {[dev.udid for dev in devs]}"
                )
        else:
            target_dev = devs[0]
            self.udid = target_dev.udid
            print_green(
                f"[{self.device_name}] Connecting to first available device (UDID: {self.udid})"
            )

        self._observed_udid = target_dev.udid
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(target_dev)  # Initiate connection and start capturing

    def identity(self) -> dict:
        return {
            "kind": "iphone",
            "name": None,
            "id": normalize_device_id(self._expected_udid),
        }

    def observed_identity(self) -> dict:
        return {"id": normalize_device_id(self._observed_udid)}

    def get_intrinsic_mat_from_coeffs(self, coeffs):
        return np.array(
            [[coeffs.fx, 0, coeffs.tx], [0, coeffs.fy, coeffs.ty], [0, 0, 1]]
        )

    def get_frame(self, camera_resolution: Optional[Tuple[int, int]] = None):
        """
        Args:
            camera_resolution: One-shot override for this call. Must preserve
                the iPhone's native 3:4 aspect ratio. The returned frame's
                `rgb` and `intrinsics` reflect this size; `self.camera_resolution`
                and `self.intrinsic_matrix` are NOT mutated, so the streaming
                publisher's view of the camera is unchanged. If None, the
                init-time resolution is used.
        """
        if camera_resolution is not None:
            req_w, req_h = camera_resolution
            native_w, native_h = self.NATIVE_RGB_RESOLUTION
            if req_w * native_h != req_h * native_w:
                raise ValueError(
                    f"[{self.device_name}] camera_resolution {req_w}x{req_h} "
                    f"does not preserve the iPhone's native aspect ratio "
                    f"({native_w}x{native_h}, 3:4)."
                )
        if self.is_running:
            try:
                event_result = self.event.wait(
                    timeout=1.0
                )  # Wait for new frame to arrive
                if not event_result:
                    return None

                # Record3D can deliver queued USB frames in a burst after GIL
                # contention.
                # Discard frames arriving less than 5 ms apart (normal 30 Hz frames are
                # approximately 33 ms apart) until the backlog is drained.
                skipped = 0
                self.event.clear()
                while self.event.wait(timeout=0.005):
                    self.event.clear()
                    skipped += 1
                if skipped > 10:
                    print_yellow(
                        f"[{self.device_name}] Drained {skipped} backlogged frame(s)"
                    )

                depth = self.session.get_depth_frame()
                rgb = self.session.get_rgb_frame()
                if rgb is None:
                    return None
                receive_time = time.monotonic()
                if self.latency is not None:
                    capture_time = receive_time - self.latency
                else:
                    capture_time = receive_time

                native_intrinsics = self.get_intrinsic_mat_from_coeffs(
                    self.session.get_intrinsic_mat()
                )
                # Quaternion + world position (qx, qy, qz, qw, tx, ty, tz)
                camera_pose = self.session.get_camera_pose()

                extrinsic_matrix = np.eye(4)
                qx, qy, qz, qw, px, py, pz = (
                    camera_pose.qx,
                    camera_pose.qy,
                    camera_pose.qz,
                    camera_pose.qw,
                    camera_pose.tx,
                    camera_pose.ty,
                    camera_pose.tz,
                )
                # scipy convention is (x, y, z, w)
                extrinsic_matrix[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
                extrinsic_matrix[:3, -1] = [px, py, pz]
                extrinsic_matrix[3, 3] = 1
                if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                    depth = cv2.flip(depth, 1)
                    rgb = cv2.flip(rgb, 1)

                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                # Sanitize depth and convert float32 meters to uint16 millimeters before
                # resizing; nearest-neighbor sampling must preserve measured depths.
                depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                np.clip(depth, 0.0, 65.535, out=depth)
                depth = (depth * 1000).astype(np.uint16)
                # Rescale RGB and intrinsics together. Keep depth at native LiDAR
                # resolution
                # (typically 192x256) unless the caller requests RGB-aligned depth.
                frame = RGBDCameraFrame(
                    rgb=rgb,
                    depth=depth,
                    pose=extrinsic_matrix,
                    capture_time=capture_time,
                    receive_time=receive_time,
                    intrinsics=native_intrinsics,
                )
                target_resolution = (
                    camera_resolution
                    if camera_resolution is not None
                    else self.camera_resolution
                )
                self._apply_resolution_override(frame, target_resolution)
                # RGB and native LiDAR depth have different shapes, so the shared
                # rescaler
                # leaves depth unchanged. Publish intrinsics for the configured stream
                # size;
                # a one-call resolution override must not change that persistent state.
                if camera_resolution is None:
                    self.intrinsic_matrix = frame.intrinsics

            except Exception as e:
                traceback.print_exc()
                print("Error: ", e)
            return frame

    def start_streaming(self):
        self.connect_to_device()
        time.sleep(1)
        self.is_running = True  # Flag to control the processing loop
        if self.get_frame() is None:
            print_red(
                f"[{self.device_name}] Failed to get frame. Please toggle streaming in iPhone."
            )
            return
        print_green(f"[{self.device_name}] Streaming started.")

    def stop_streaming(self):
        self.is_running = False
        print_green(f"[{self.device_name}] Streaming stopped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iphone_udid", type=str, default=None, help="UDID (default: first iPhone)"
    )
    args = parser.parse_args()

    cam_list = IphoneCamera.list_connected_devices()

    cam = IphoneCamera(iphone_udid=args.iphone_udid, camera_resolution=(360, 480))
    cam.start_streaming()
    cam.live_streaming(target_fps=60)
    cam.stop_streaming()

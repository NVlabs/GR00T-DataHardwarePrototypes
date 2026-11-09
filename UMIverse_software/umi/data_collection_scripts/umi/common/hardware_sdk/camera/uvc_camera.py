# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import shutil
import subprocess
import time
from typing import Optional, Tuple

import cv2

from umi.common.data.frame import RGBCameraFrame
from umi.common.hardware_sdk.camera.camera import Camera
from umi.common.utility.print_utils import print_green, print_red, print_yellow


def get_connected_cameras():
    """
    Returns a list of dictionaries describing connected UVC cameras.
    Format: [{'name': 'Camera Name', 'path': '/dev/videoX', 'index': X}, ...]
    """
    cameras = []

    try:
        output = subprocess.check_output(
            ["v4l2-ctl", "--list-devices"], stderr=subprocess.STDOUT
        ).decode("utf-8")
    except FileNotFoundError:
        print("Error: v4l2-ctl not found. Please run: sudo apt install v4l-utils")
        return []
    except subprocess.CalledProcessError as e:
        print(f"Error running v4l2-ctl: {e.output.decode()}")
        return []

    # 2. Parse the output. Devices are separated by double newlines.
    # Output format example:
    # H264 USB Camera: USB Camera (usb-0000:00:14.0-11):
    #         /dev/video4
    #         /dev/video5
    raw_devices = output.strip().split("\n\n")

    for raw_device in raw_devices:
        lines = raw_device.split("\n")
        if not lines:
            continue

        # The first line is the camera name and ID
        # Example: "H264 USB Camera: USB Camera (usb-0000:00:14.0-11):"
        header = lines[0]
        camera_name = header.split("(")[0].strip().rstrip(":")

        # The following lines are the device paths (/dev/videoX)
        device_paths = [
            line.strip() for line in lines[1:] if line.strip().startswith("/dev/video")
        ]

        valid_path = None

        for path in device_paths:
            if is_video_capture_node(path):
                valid_path = path
                break  # Found the main video feed, stop looking at this device

        if valid_path:
            index = int(re.search(r"\d+$", valid_path).group())

            cameras.append({"name": camera_name, "path": valid_path, "index": index})

    return cameras


def is_video_capture_node(path):
    """
    Checks if a specific /dev/videoX node is actually a Video Capture device
    (and not just Metadata or Touch).
    """
    try:
        cmd = ["v4l2-ctl", f"--device={path}", "--all"]
        details = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8")

        # Look for the "Device Caps" section, which lists the capabilities of this
        # node, and check that it includes "Video Capture".
        if "Device Caps" in details and "Video Capture" in details:
            return True

    except Exception:
        return False

    return False


class UVCCamera(Camera):
    def __init__(
        self,
        camera_index: int,
        device_name: str,
        camera_resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
        latency: float = 0.0,
        pixel_format: str = "MJPG",
    ) -> None:
        """
        Args:
            camera_index: The OS index for the camera (usually 0 for built-in, 1 for USB).
            device_name: Identifier name.
            camera_resolution: Tuple (width, height).
            fps: Desired frames per second.
            latency: Artificial latency adjustment (optional).
            pixel_format: 'MJPG' (compressed, faster) or 'YUYV' (uncompressed, clearer but slower).
        """
        super().__init__(device_name, camera_resolution, latency)
        self.camera_index = camera_index
        self.fps = fps
        self.pixel_format = pixel_format.upper()
        self.cap = None

    def list_supported_formats(self):
        """
        Lists supported formats using v4l2-ctl for debugging purposes.
        This does not affect the OpenCV configuration.
        """
        if shutil.which("v4l2-ctl"):
            device_path = f"/dev/video{self.camera_index}"
            print(f"\n--- Supported Formats for {device_path} ---")
            try:
                subprocess.run(
                    ["v4l2-ctl", "-d", device_path, "--list-formats-ext"], check=True
                )
            except subprocess.CalledProcessError as e:
                print_red(f"[{self.device_name}] Failed to list formats: {e}")
            print("-------------------------------------------\n")
        else:
            print_yellow(
                f"[{self.device_name}] v4l2-ctl not found. Cannot list supported formats."
            )

    def start_streaming(self):
        if self.is_running:
            print_yellow(f"[{self.device_name}] Already streaming.")
            return

        print(f"Opening UVC Camera {self.camera_index}...")

        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera with index {self.camera_index}")

        # Set pixel format before resolution so the requested mode fits the USB
        # bandwidth.
        if self.pixel_format == "MJPG":
            # MJPG is compressed, allowing high resolution @ high FPS over USB 2.0
            print("Configuring Mode: MJPG (Compressed)")
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        else:
            # YUYV is uncompressed. High quality, but usually capped at 15fps for HD.
            print("Configuring Mode: YUYV (Uncompressed)")
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))

        width, height = self.camera_resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        print(
            f"Format Requested: {self.pixel_format} | Actual: {fourcc_str}\n"
            f"Req: {width}x{height}@{self.fps}fps | "
            f"Act: {int(actual_w)}x{int(actual_h)}@{actual_fps}fps"
        )

        if int(actual_w) != width or int(actual_h) != height:
            self.cap.release()
            self.cap = None
            raise RuntimeError(
                f"[{self.device_name}] Requested resolution {width}x{height} "
                f"not honored by device (got {int(actual_w)}x{int(actual_h)}). "
                f"Run `list_supported_formats()` to see supported modes."
            )

        print_green(f"[{self.device_name}] Streaming started.")
        self.is_running = True

    def get_frame(
        self, camera_resolution: Optional[Tuple[int, int]] = None
    ) -> Optional[RGBCameraFrame]:
        if not self.is_running or self.cap is None:
            return None

        ret, frame = self.cap.read()

        if not ret:
            print_red(f"[{self.device_name}] Failed to grab frame.")
            return None

        receive_time = time.monotonic()
        if self.latency is not None:
            capture_time = receive_time - self.latency
        else:
            capture_time = receive_time

        return self._apply_resolution_override(
            RGBCameraFrame(
                rgb=frame,  # Keeping as BGR for OpenCV compatibility
                pose=None,
                capture_time=capture_time,
                receive_time=receive_time,
                intrinsics=self.intrinsic_matrix,
            ),
            camera_resolution,
        )

    def stop_streaming(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_running = False
        print_green(f"[{self.device_name}] Streaming stopped.")


# --- Example Usage ---

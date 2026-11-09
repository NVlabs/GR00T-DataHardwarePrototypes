# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Mock Camera for Testing

A mock camera that generates synthetic video frames for testing the recording system.
This simulates camera behavior including streaming, frame generation, and error conditions.
"""

import time
from typing import Optional, Tuple

import numpy as np

from umi.common.data.frame import RGBCameraFrame, RGBDCameraFrame
from umi.common.hardware_sdk.camera.camera import Camera
from umi.common.utility.print_utils import print_green, print_yellow


class MockCamera(Camera):
    """
    Mock camera that generates synthetic RGB (or RGBD) frames for testing.

    Simulates a camera with configurable resolution, latency, and error conditions.
    When enable_depth=True, generates RGBDCameraFrame with synthetic uint16 depth maps.
    """

    def __init__(
        self,
        device_name: str,
        camera_resolution: Tuple[int, int] = (640, 480),
        latency: float = 0.0,
        fps: int = 30,
        frame_pattern: str = "gradient",
        enable_pose: bool = False,
        enable_intrinsics: bool = False,
        intrinsics=None,
        intrinsics_hfov_deg: float = 70.0,
        enable_depth: bool = False,
        depth_min_mm: int = 200,
        depth_max_mm: int = 5000,
        simulate_disconnect: bool = False,
        disconnect_after_frames: int = 100,
        add_large_latency: bool = False,
        large_latency_after_frames: int = 50,
        large_latency_duration: float = 0.5,
    ) -> None:
        """
        Args:
            device_name: Identifier name for the camera.
            camera_resolution: Resolution as (width, height) tuple (default: 640x480).
            latency: Normal latency in seconds (default: 0.0).
            fps: Target frames per second for the camera (default: 30).
            frame_pattern: Type of frame pattern to generate. Options:
                - "gradient": Colored gradient pattern
                - "checkerboard": Black and white checkerboard
                - "noise": Random noise
                - "solid": Solid color (changes over time)
                - "counter": Frame counter display
            enable_pose: If True, attach a smooth, continuously-moving synthetic
                4x4 camera pose to each frame (mimics a tracking camera such as
                the iPhone). Randomized per run. Default False (plain RGB camera
                with no pose, like a fixed wrist cam).
            enable_intrinsics: If True, attach a synthetic 3x3 pinhole intrinsic
                matrix to every frame (like a real calibrated camera). Without
                this, frames carry ``intrinsics=None`` and the recorder writes no
                ``intrinsics`` array — unlike real cameras (OAK/iPhone), which
                always report a calibrated matrix. Default False (preserves the
                bare-RGB behavior).
            intrinsics: Optional explicit 3x3 matrix (anything ``np.asarray``-able
                to shape (3, 3)). Takes precedence over ``enable_intrinsics`` /
                ``intrinsics_hfov_deg`` and is used verbatim (at the native
                ``camera_resolution``); the recorder/aligner scale it if a frame
                is resized.
            intrinsics_hfov_deg: Horizontal FOV (degrees) used to synthesize a
                self-consistent pinhole matrix from ``camera_resolution`` when
                ``enable_intrinsics`` is set and no explicit ``intrinsics`` is
                given (fx=fy from the FOV, principal point at the image center).
                Default 70.0.
            enable_depth: If True, generate RGBDCameraFrame with depth maps (default: False).
            depth_min_mm: Minimum depth value in millimeters (default: 200).
            depth_max_mm: Maximum depth value in millimeters (default: 5000).
            simulate_disconnect: If True, simulate device disconnection (default: False).
            disconnect_after_frames: Number of frames before simulating disconnect (default: 100).
            add_large_latency: If True, add periodic large latency spikes (default: False).
            large_latency_after_frames: Frames before adding large latency (default: 50).
            large_latency_duration: Duration of large latency spike in seconds (default: 0.5).
        """
        super().__init__(
            device_name=device_name,
            camera_resolution=camera_resolution,
            latency=latency,
        )

        self.fps = fps
        self.frame_pattern = frame_pattern
        self.enable_pose = enable_pose

        # Optional synthetic camera intrinsics. Real cameras always report a
        # calibrated 3x3; the mock leaves it None unless asked, so the recorder
        # only writes an `intrinsics` array when this is set.
        if intrinsics is not None:
            self.set_intrinsic_matrix(
                np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
            )
        elif enable_intrinsics:
            self.set_intrinsic_matrix(
                self._synthesize_intrinsics(self.camera_resolution, intrinsics_hfov_deg)
            )
        self.enable_depth = enable_depth
        self.depth_min_mm = depth_min_mm
        self.depth_max_mm = depth_max_mm
        self.simulate_disconnect = simulate_disconnect
        self.disconnect_after_frames = disconnect_after_frames
        self.add_large_latency = add_large_latency
        self.large_latency_after_frames = large_latency_after_frames
        self.large_latency_duration = large_latency_duration

        self.frame_count = 0
        self.start_time = None
        self.disconnected = False

        # Randomized-at-init sinusoid params for a smooth, continuously-moving
        # 6-DoF camera pose (different each run; low frequencies keep the Slerp
        # interpolation well-conditioned). Only used when enable_pose=True.
        if self.enable_pose:
            rng = np.random.default_rng()
            self._pose_trans_center = rng.uniform(-0.1, 0.1, size=3)
            self._pose_trans_amp = rng.uniform(0.05, 0.15, size=3)  # meters
            self._pose_trans_freq = rng.uniform(0.05, 0.25, size=3)  # Hz
            self._pose_trans_phase = rng.uniform(0.0, 2 * np.pi, size=3)
            self._pose_rot_amp = rng.uniform(0.1, 0.4, size=3)  # radians
            self._pose_rot_freq = rng.uniform(0.05, 0.2, size=3)  # Hz
            self._pose_rot_phase = rng.uniform(0.0, 2 * np.pi, size=3)

    def start_streaming(self):
        """Start the mock camera streaming."""
        if self.is_running:
            print(f"[{self.device_name}] Already streaming.")
            return

        self.is_running = True
        self.start_time = time.monotonic()
        self.frame_count = 0
        self.disconnected = False
        print_green(f"[{self.device_name}] Streaming started.")
        print(f"  Resolution: {self.camera_resolution}")
        print(f"  Frame pattern: {self.frame_pattern}")
        print(f"  Latency: {self.latency}s")
        if self.simulate_disconnect:
            print_yellow(
                f"  Will simulate disconnect after {self.disconnect_after_frames} frames"
            )
        if self.add_large_latency:
            print_yellow(
                f"  Will add large latency after {self.large_latency_after_frames} frames"
            )

    def get_frame(
        self, camera_resolution: Optional[Tuple[int, int]] = None
    ) -> Optional[RGBCameraFrame]:
        """
        Get a single frame from the mock camera.

        Returns:
            RGBCameraFrame (or RGBDCameraFrame if depth enabled) with synthetic data,
            or None if not streaming or disconnected.
        """
        if not self.is_running:
            return None

        if (
            self.simulate_disconnect
            and self.frame_count >= self.disconnect_after_frames
        ):
            if not self.disconnected:
                print_yellow(f"[{self.device_name}] Simulating device disconnect!")
                self.disconnected = True
            return None

        receive_time = time.monotonic()

        current_latency = self.latency

        if (
            self.add_large_latency
            and self.frame_count >= self.large_latency_after_frames
        ):
            current_latency += self.large_latency_duration
            if self.frame_count == self.large_latency_after_frames:
                print_yellow(
                    f"[{self.device_name}] Adding large latency spike: {self.large_latency_duration}s"
                )

        capture_time = receive_time - current_latency

        frame_rgb = self._generate_frame()

        # Optional synthetic camera pose (continuous, randomized per run).
        pose = (
            self._generate_pose(self.frame_count / self.fps)
            if self.enable_pose
            else None
        )

        self.frame_count += 1

        if self.enable_depth:
            depth = self._generate_depth_frame()
            return self._apply_resolution_override(
                RGBDCameraFrame(
                    rgb=frame_rgb,
                    depth=depth,
                    pose=pose,
                    capture_time=capture_time,
                    receive_time=receive_time,
                    frame_source=self.device_name,
                    intrinsics=self.intrinsic_matrix,
                ),
                camera_resolution,
            )

        return self._apply_resolution_override(
            RGBCameraFrame(
                rgb=frame_rgb,
                pose=pose,
                capture_time=capture_time,
                receive_time=receive_time,
                frame_source=self.device_name,
                intrinsics=self.intrinsic_matrix,
            ),
            camera_resolution,
        )

    @staticmethod
    def _synthesize_intrinsics(
        resolution: Tuple[int, int], hfov_deg: float
    ) -> np.ndarray:
        """Pinhole 3x3 intrinsics consistent with a frame of `resolution`.

        Square pixels (fx == fy) from the horizontal FOV, principal point at the
        image center. Matches the convention real cameras report (and that
        ``Camera._apply_resolution_override`` scales).
        """
        width, height = resolution
        fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        cx, cy = width / 2.0, height / 2.0
        return np.array(
            [[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    def _generate_frame(self) -> np.ndarray:
        """Generate a synthetic RGB frame based on the frame pattern."""
        width, height = self.camera_resolution

        if self.frame_pattern == "gradient":
            return self._generate_gradient_frame(width, height)
        elif self.frame_pattern == "checkerboard":
            return self._generate_checkerboard_frame(width, height)
        elif self.frame_pattern == "noise":
            return self._generate_noise_frame(width, height)
        elif self.frame_pattern == "solid":
            return self._generate_solid_frame(width, height)
        elif self.frame_pattern == "counter":
            return self._generate_counter_frame(width, height)
        else:
            return self._generate_gradient_frame(width, height)

    def _generate_gradient_frame(self, width: int, height: int) -> np.ndarray:
        """Generate a colored gradient frame that changes over time."""
        t = self.frame_count / self.fps

        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Red channel: horizontal gradient
        frame[:, :, 2] = np.linspace(0, 255, width, dtype=np.uint8)

        # Green channel: vertical gradient
        frame[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1)

        # Blue channel: time-varying
        frame[:, :, 0] = int(127 + 127 * np.sin(2 * np.pi * t / 5))

        return frame

    def _generate_checkerboard_frame(self, width: int, height: int) -> np.ndarray:
        """Generate a checkerboard pattern."""
        square_size = 40
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(0, height, square_size):
            for j in range(0, width, square_size):
                if ((i // square_size) + (j // square_size)) % 2 == 0:
                    frame[i : i + square_size, j : j + square_size] = 255

        return frame

    def _generate_noise_frame(self, width: int, height: int) -> np.ndarray:
        """Generate random noise frame."""
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    def _generate_solid_frame(self, width: int, height: int) -> np.ndarray:
        """Generate solid color frame that changes over time."""
        t = self.frame_count / self.fps

        r = int(127 + 127 * np.sin(2 * np.pi * t / 3))
        g = int(127 + 127 * np.sin(2 * np.pi * t / 5))
        b = int(127 + 127 * np.sin(2 * np.pi * t / 7))

        frame = np.full((height, width, 3), [b, g, r], dtype=np.uint8)
        return frame

    def _generate_counter_frame(self, width: int, height: int) -> np.ndarray:
        """Generate frame with frame counter displayed."""
        import cv2

        frame = np.zeros((height, width, 3), dtype=np.uint8)

        cv2.putText(
            frame,
            f"Frame: {self.frame_count}",
            (width // 4, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 0),
            3,
        )

        return frame

    def _generate_depth_frame(self) -> np.ndarray:
        """Generate a synthetic uint16 depth frame (values in millimeters).

        Produces a smooth depth surface with ~47% zero pixels (invalid/background)
        to mimic real depth sensor characteristics (e.g., iPhone LiDAR).
        """
        width, height = self.camera_resolution
        t = self.frame_count / self.fps

        # Create a smooth depth surface: slanted plane with sinusoidal ripple
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        y_norm = y_coords / height
        x_norm = x_coords / width

        # Base plane with slow drift
        base_depth = self.depth_min_mm + (self.depth_max_mm - self.depth_min_mm) * (
            0.3 * x_norm + 0.3 * y_norm + 0.1 * np.sin(2 * np.pi * t / 10)
        )
        # Add spatial ripple for compressibility variation
        ripple = 200 * np.sin(2 * np.pi * x_norm * 4) * np.cos(2 * np.pi * y_norm * 3)
        depth = base_depth + ripple

        depth = np.clip(depth, 0, 65535).astype(np.uint16)

        # Zero out ~47% of pixels to simulate invalid/background regions
        mask = (x_norm**2 + y_norm**2) > 0.7
        depth[mask] = 0

        return depth

    def _generate_pose(self, t: float) -> np.ndarray:
        """Smooth, continuously-moving 4x4 homogeneous pose (randomized per run).

        Mirrors a real tracking camera's pose stream so alignment can interpolate
        ``{source}/pose``. Translation and rotation each follow randomized
        low-frequency sinusoids, so the pose drifts continuously without jumps.
        """
        trans = self._pose_trans_center + self._pose_trans_amp * np.sin(
            2 * np.pi * self._pose_trans_freq * t + self._pose_trans_phase
        )
        rot = self._pose_rot_amp * np.sin(
            2 * np.pi * self._pose_rot_freq * t + self._pose_rot_phase
        )
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = self._euler_to_rotation_matrix(rot[0], rot[1], rot[2])
        pose[:3, 3] = trans
        return pose

    @staticmethod
    def _euler_to_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
        """XYZ-euler (radians) -> proper 3x3 rotation matrix (numpy-only)."""
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return rot_z @ rot_y @ rot_x

    def stop_streaming(self):
        """Stop the mock camera streaming."""
        self.is_running = False
        self.disconnected = False
        print_green(f"[{self.device_name}] Streaming stopped.")


# --- Example Usage ---
if __name__ == "__main__":
    import cv2

    camera = MockCamera(
        device_name="MockCamera",
        camera_resolution=(640, 480),
        frame_pattern="gradient",
        fps=30,
    )

    camera.start_streaming()

    print("\nDisplaying frames (press 'q' to quit)...")
    while True:
        frame_data = camera.get_frame()
        if frame_data is not None and frame_data.rgb is not None:
            cv2.imshow("Mock Camera", frame_data.rgb)
            if cv2.waitKey(33) & 0xFF == ord("q"):  # ~30 FPS
                break
        time.sleep(1 / camera.fps)

    camera.stop_streaming()
    cv2.destroyAllWindows()

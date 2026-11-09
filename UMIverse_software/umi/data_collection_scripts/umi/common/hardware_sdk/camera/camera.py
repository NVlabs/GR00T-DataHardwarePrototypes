# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from typing import Optional, Tuple

import cv2

from umi.common.hardware_sdk.sensor_interface import SensorInterface
from umi.common.utility.rate_keeper import RateKeeper


class Camera(SensorInterface):
    def __init__(
        self,
        device_name: str,
        camera_resolution: Tuple[int, int],
        latency: float = None,
    ) -> None:
        super().__init__(device_name=device_name, latency=latency)
        self.camera_resolution = camera_resolution
        self.intrinsic_matrix = None

    def set_intrinsic_matrix(self, matrix):
        """
        Utility method to set the camera's intrinsic matrix.
        """
        self.intrinsic_matrix = matrix
        print(f"{self.device_name}: Intrinsic matrix set.")

    def get_calibration_frame(self):
        """Return a frame suitable for marker-based pose calibration.

        Default implementation just delegates to `get_frame()`, i.e. the
        same image the recorder streams. Subclasses with a native sensor
        resolution larger than their streaming resolution (e.g. iPhones
        downsized for bandwidth) should override this to return a
        higher-resolution frame so marker corner detection is accurate.

        Called by the camera-relative-calibration pre-episode check while
        it has exclusive access to this camera (the recorder stream thread
        is paused for the duration of the calibration session).
        """
        return self.get_frame()

    def _apply_resolution_override(
        self,
        frame,
        camera_resolution: Optional[Tuple[int, int]],
    ):
        """Resize `frame.rgb` (and `frame.depth` when shapes match) to
        `camera_resolution` and scale `frame.intrinsics` to match. Mutates
        the frame in place and returns it. No-op when `camera_resolution`
        is None or already equal to the frame's current size.

        Subclass `get_frame()` implementations should call this helper just
        before returning so they can honour the optional `camera_resolution`
        kwarg uniformly. Cameras with a native-resolution capture path
        larger than streaming (e.g. `IphoneCamera`) bake the resize into
        their own pipeline instead and do not need this helper.
        """
        if frame is None or camera_resolution is None:
            return frame
        rgb = getattr(frame, "rgb", None)
        if rgb is None:
            return frame
        cur_h, cur_w = rgb.shape[:2]
        tgt_w, tgt_h = camera_resolution
        if (cur_w, cur_h) == (tgt_w, tgt_h):
            return frame
        sx = tgt_w / cur_w
        sy = tgt_h / cur_h
        frame.rgb = cv2.resize(rgb, (tgt_w, tgt_h))
        depth = getattr(frame, "depth", None)
        if depth is not None and depth.shape[:2] == (cur_h, cur_w):
            frame.depth = cv2.resize(
                depth, (tgt_w, tgt_h), interpolation=cv2.INTER_NEAREST
            )
        intrinsics = getattr(frame, "intrinsics", None)
        if intrinsics is not None:
            K = intrinsics.copy()
            K[0, :] *= sx
            K[1, :] *= sy
            frame.intrinsics = K
        return frame

    def live_streaming(self, target_fps: float = 30.0, verbose: bool = False):
        rate_keeper = RateKeeper(
            target_rate=target_fps, verbose=verbose, name=f"{self.device_name}_stream"
        )

        start_time = time.monotonic()
        frame_count = 0

        while True:
            with rate_keeper:
                frame_data = self.get_frame()
                if frame_data.rgb is not None:
                    current_time = time.monotonic()
                    frame_count += 1
                    fps = frame_count / (current_time - start_time)

                    cv2.putText(
                        frame_data.rgb,
                        f"FPS: {fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                    cv2.imshow("RGB", frame_data.rgb)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        cv2.destroyAllWindows()

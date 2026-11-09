# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from typing import Optional

import numpy as np

from umi.common.data.frame import EncoderFrame
from umi.common.hardware_sdk.sensor_interface import SensorInterface
from umi.common.utility.print_utils import print_green


class MockSensor(SensorInterface):
    """
    Mock sensor that generates synthetic numeric data for testing.

    Simulates a multi-channel sensor (e.g., force sensor, accelerometer, etc.)
    with configurable number of channels and sampling characteristics. It is a
    proper ``SensorInterface`` (like a real ``SensorDevice`` sensor) and
    declares ``frame_class`` so the recorder auto-detects its frame
    type — no per-source ``frame_type`` needs to be set in the recording config.

    Note: it has no UART reader/protocol, so it subclasses ``SensorInterface``
    directly rather than ``SensorDevice`` (which requires a reader).
    """

    # Emitted by get_frame(); read by create_numeric_recorders via
    # getattr(source, "frame_class", ...) to size/type the recorder.
    frame_class = EncoderFrame

    def __init__(
        self,
        device_name: str,
        num_channels: int = 3,
        latency: float = 0.0,
        noise_level: float = 0.1,
        signal_type: str = "sine",
    ) -> None:
        """
        Args:
            device_name: Identifier name for the sensor.
            num_channels: Number of data channels to simulate (default: 3).
            latency: Artificial latency adjustment in seconds (default: 0.0).
            noise_level: Amount of random noise to add to signals (default: 0.1).
            signal_type: Type of signal to generate. Options:
                - "sine": Sine waves with different frequencies per channel
                - "random": Random values
                - "ramp": Linear ramp that resets periodically
                - "step": Step function that changes periodically
        """
        super().__init__(device_name=device_name, latency=latency)
        self.num_channels = num_channels
        self.noise_level = noise_level
        self.signal_type = signal_type

        self.start_time = None

        self.frequencies = [1.0 + i * 0.5 for i in range(num_channels)]  # Hz
        self.amplitudes = [1.0 + i * 0.2 for i in range(num_channels)]
        self.offsets = [i * 0.5 for i in range(num_channels)]

    def start_streaming(self):
        """Start the mock sensor streaming."""
        if self.is_running:
            print(f"[{self.device_name}] Already streaming.")
            return

        self.is_running = True
        self.start_time = time.monotonic()
        print_green(f"[{self.device_name}] Streaming started.")
        print(f"  Channels: {self.num_channels}")
        print(f"  Signal type: {self.signal_type}")
        print(f"  Latency: {self.latency}s")

    def get_frame(self) -> Optional[EncoderFrame]:
        """
        Get a single frame of sensor data.

        Returns:
            EncoderFrame with synthetic sensor values, or None if not streaming.
        """
        if not self.is_running:
            return None

        receive_time = time.monotonic()
        if self.latency is not None:
            capture_time = receive_time - self.latency
        else:
            capture_time = receive_time

        t = receive_time - self.start_time

        if self.signal_type == "sine":
            values = self._generate_sine_wave(t)
        elif self.signal_type == "random":
            values = self._generate_random(t)
        elif self.signal_type == "ramp":
            values = self._generate_ramp(t)
        elif self.signal_type == "step":
            values = self._generate_step(t)
        else:
            values = self._generate_sine_wave(t)

        noise = np.random.randn(self.num_channels) * self.noise_level
        values = values + noise

        return EncoderFrame(
            position=values,
            capture_time=capture_time,
            receive_time=receive_time,
            frame_source=self.device_name,
        )

    def _generate_sine_wave(self, t: float) -> np.ndarray:
        """Generate sine wave values for each channel."""
        values = np.zeros(self.num_channels)
        for i in range(self.num_channels):
            values[i] = (
                self.amplitudes[i] * np.sin(2 * np.pi * self.frequencies[i] * t)
                + self.offsets[i]
            )
        return values

    def _generate_random(self, t: float) -> np.ndarray:
        """Generate random values for each channel."""
        return np.random.randn(self.num_channels)

    def _generate_ramp(self, t: float) -> np.ndarray:
        """Generate ramping values that reset periodically."""
        values = np.zeros(self.num_channels)
        for i in range(self.num_channels):
            # Reset every 5 seconds
            period = 5.0
            t_mod = t % period
            values[i] = (t_mod / period) * self.amplitudes[i] + self.offsets[i]
        return values

    def _generate_step(self, t: float) -> np.ndarray:
        """Generate step function values."""
        values = np.zeros(self.num_channels)
        for i in range(self.num_channels):
            # Step changes every 2 seconds
            step = int(t / 2.0) % 2
            values[i] = step * self.amplitudes[i] + self.offsets[i]
        return values

    def stop_streaming(self):
        """Stop the mock sensor streaming."""
        self.is_running = False
        print(f"[{self.device_name}] Stopped streaming.")


# --- Example Usage ---

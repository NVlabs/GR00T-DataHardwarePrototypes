# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encoder sensor implementation.

Provides encoder position and velocity readings via UMI protocol.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from umi.common.data.frame import EncoderFrame
from umi.common.hardware_sdk.sensor.protocol import SENSOR_TYPE_JOINT_ENCODER
from umi.common.hardware_sdk.sensor.sensor_device import (
    DeviceSample,
    SensorDevice,
    SensorSpec,
)


class EncoderSensor(SensorDevice):
    """Multi-channel encoder sensor device.

    Reads encoder position and velocity data from UMI protocol frames.
    Each encoder channel has its own sensor_id.
    """

    def __init__(
        self,
        device_name: str,
        reader: Any,  # UartSensorReader
        sensor_ids: list[int] | None = None,
        sensors: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize encoder sensor.

        Args:
            device_name: Unique device identifier
            reader: Shared UartSensorReader instance
            sensor_ids: List of sensor IDs to register. If None, defaults to [0, 1, 2, 3, 4, 5].
            sensors: Custom sensor specs. If None, generates defaults based on sensor_ids.
            **kwargs: Additional arguments for SensorDevice
        """
        self.sensor_ids = sensor_ids if sensor_ids is not None else list(range(6))

        super().__init__(
            device_name=device_name,
            reader=reader,
            sensors=sensors,
            **kwargs,
        )

    def _default_sensors(self) -> tuple[SensorSpec, ...]:
        """Generate default encoder sensor specs based on sensor_ids."""
        return tuple(
            SensorSpec(
                sensor_type=SENSOR_TYPE_JOINT_ENCODER,
                sensor_id=sid,
                name=f"encoder_{sid:02d}",
                fmt="<f",  # Single float (position)
            )
            for sid in self.sensor_ids
        )

    def _sample_to_frame(self, sample: DeviceSample) -> EncoderFrame:
        """Convert DeviceSample to EncoderFrame.

        Args:
            sample: Parsed device sample

        Returns:
            EncoderFrame with position data
        """
        receive_time = self._receive_time_of(sample)
        capture_time = None

        positions = []

        for sid in self.sensor_ids:
            sensor_name = f"encoder_{sid:02d}"
            reading = sample.readings.get(sensor_name)

            if reading and isinstance(reading.value, (int, float)):
                positions.append(float(reading.value))
            else:
                # Fill with -np.inf if data missing
                positions.append(-np.inf)

        position_array = np.array(positions, dtype=np.float32)

        return EncoderFrame(
            position=position_array,  # Position data
            capture_time=capture_time,
            receive_time=receive_time,
            frame_source=self.device_name,
        )

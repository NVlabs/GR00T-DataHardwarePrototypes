# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tactile sensor implementation.

Decodes tactile TLVs into generated frame fields.

Data format (80 bytes total):
- Raw channels: 13 x uint32 (52B)
- Group 1: normal_force (4B), tangential_force (4B), tangential_angle (2B)
- Group 2: normal_force (4B), tangential_force (4B), tangential_angle (2B)
- Proximity: proximity_1 (4B), proximity_2 (4B)

For each measurement point the sensor reports a triplet
(normal_force, tangential_force, tangential_angle). The frame exposes only
the decomposed (fx, fy, fz) force components (the raw triplet is decoded to
compute them but is not recorded)::

    fx = tangential_force * cos(angle)
    fy = tangential_force * sin(angle)
    fz = normal_force

"""

from __future__ import annotations

import math
import struct
from typing import Any

from umi.common.data.frame import create_frame_from_sample, flatten_sample

from .protocol import SENSOR_TYPE_TACTILE
from .sensor_device import DeviceSample, SensorDevice, SensorSpec

# Clamp angles above the SDK limit to 0 degrees, including the 0xFFFF
# no-contact sentinel. Zero forces then decompose to (0, 0, 0).
ANGLE_MAX_DEG = 360


class TactileSensor(SensorDevice):
    """Tactile sensor device.

    For each configured sensor id the generated frame exposes the
    decomposed ``(fx, fy, fz)`` force components for the two measurement
    points (``_1`` / ``_2``). Field names are flattened with the sensor name
    as prefix, e.g. ``tactile_0_fx_1``.

    Per measurement point::

        fx_{1,2}, fy_{1,2}, fz_{1,2}

    Example:
        reader = UartSensorReader(port="/dev/ttyUSB0", baudrate=1000000)
        reader.start()

        tactile = TactileSensor(
            device_name="left_tactile",
            reader=reader,
        )
        tactile.start_streaming()
        frame = tactile.get_frame()
        if frame is not None:
            print(f"Force xyz pt 1: "
                  f"({frame.tactile_0_fx_1}, "
                  f"{frame.tactile_0_fy_1}, "
                  f"{frame.tactile_0_fz_1})")
    """

    def __init__(
        self,
        device_name: str,
        reader: Any,  # UartSensorReader
        sensor_ids: list[int] | None = None,
        sensors: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize tactile sensor.

        Args:
            device_name: Unique device identifier
            reader: Shared UartSensorReader instance
            sensor_ids: List of sensor IDs to register. If provided, automatically
                       generates sensor specs for these IDs. Ignored if sensors is provided.
            sensors: Custom sensor specs. If None and sensor_ids is None, uses default spec.
            **kwargs: Additional arguments for SensorDevice
        """
        self.sensor_ids = sensor_ids if sensor_ids is not None else [0]

        super().__init__(
            device_name=device_name,
            reader=reader,
            sensors=sensors,
            **kwargs,
        )

        sample_data = self._decode_tactile_data(bytes(80))
        samples = {spec.name: sample_data for spec in self.sensors}
        self.frame_class = create_frame_from_sample(
            f"{device_name}_Frame", samples, interpolate=True
        )

    def _default_sensors(self) -> tuple[SensorSpec, ...]:
        """Generate default tactile sensor specs based on sensor_ids."""
        return tuple(
            SensorSpec(
                sensor_type=SENSOR_TYPE_TACTILE,
                sensor_id=sid,
                name=f"tactile_{sid}",
                decoder=lambda payload: TactileSensor._decode_tactile_data(payload),
            )
            for sid in self.sensor_ids
        )

    @staticmethod
    def _decompose_xyz(
        normal_force: int, tangential_force: int, tangential_angle: int
    ) -> tuple[float, float, float]:
        """Decompose a (nf, tf, angle) triplet into (fx, fy, fz).

        Matches the vendor SDK ``get_sensor_state_xyz`` formula::

            theta = radians(angle)        # angle clamped to 0 if > 360 deg
            fx = tangential_force * cos(theta)
            fy = tangential_force * sin(theta)
            fz = normal_force

        but keeps full float precision (the vendor SDK truncates fx / fy
        into 16- / 32-bit signed integer slots).

        No-contact frames (where the device sets ``angle = 0xFFFF`` and
        both forces to 0) are handled by the same angle clamp: the angle
        is treated as 0 deg and the zero-magnitude forces give
        ``(0, 0, 0)`` automatically.

        Args:
            normal_force: Raw normal force (uint32 from payload).
            tangential_force: Raw tangential force magnitude (uint32).
            tangential_angle: Tangential force direction in degrees
                (uint16, expected range [0, 360]; 0xFFFF means
                no-contact).

        Returns:
            ``(fx, fy, fz)`` floats.
        """
        # Clamp the SDK no-contact angle sentinel before decomposing force components.
        if tangential_angle > ANGLE_MAX_DEG:
            theta = 0.0
        else:
            theta = math.radians(tangential_angle)

        fx = tangential_force * math.cos(theta)
        fy = tangential_force * math.sin(theta)
        fz = float(normal_force)
        return fx, fy, fz

    @staticmethod
    def _decode_tactile_data(payload: bytes) -> dict:
        """Decode the complete tactile sensor payload.

        Internally reads the raw ``(normal_force, tangential_force,
        tangential_angle)`` triplet for each of the two measurement points and
        returns only the decomposed ``(fx, fy, fz)`` force components. See
        module docstring for the conversion formulas and the sentinel /
        clamping rules.

        Args:
            payload: 80 bytes of tactile data

        Returns:
            Dictionary with decoded tactile measurements. Keys::

                fx_{1,2}, fy_{1,2}, fz_{1,2}  (float, decomposed)
        """
        if len(payload) != 80:
            raise ValueError(f"Expected 80 bytes, got {len(payload)}")

        # Payload layout: 80 bytes, little-endian, no padding.
        # 13 uint32 raw values (52 bytes); two groups of normal/tangential force
        # (uint32) and angle (uint16), 10 bytes each; two uint32 proximity values
        # (8 bytes). Proximity values are decoded but not used.
        unpacked = struct.unpack("<13I 2IH 2IH 2I", payload)
        nf1, tf1, ang1 = unpacked[13:16]
        nf2, tf2, ang2 = unpacked[16:19]

        fx1, fy1, fz1 = TactileSensor._decompose_xyz(nf1, tf1, ang1)
        fx2, fy2, fz2 = TactileSensor._decompose_xyz(nf2, tf2, ang2)

        return {
            # Record only decomposed force components: fz is normal force, while fx/fy
            # resolve tangential force by angle, for each measurement point.
            "fx_1": fx1,
            "fy_1": fy1,
            "fz_1": fz1,
            "fx_2": fx2,
            "fy_2": fy2,
            "fz_2": fz2,
        }

    def _sample_to_frame(self, sample: DeviceSample):
        """Convert DeviceSample to a tactile frame.

        Args:
            sample: Parsed device sample

        Returns:
            Frame with tactile measurements, or None if no data
        """
        receive_time = self._receive_time_of(sample)
        capture_time = None

        kwargs = {
            "capture_time": capture_time,
            "receive_time": receive_time,
            "frame_source": self.device_name,
        }

        has_data = False
        for spec in self.sensors:
            sensor_name = spec.name
            if sensor_name in sample.readings:
                reading = sample.readings[sensor_name]
                kwargs.update(flatten_sample(sensor_name, reading.value))
                has_data = True

        if not has_data:
            return None

        return self.frame_class(**kwargs)

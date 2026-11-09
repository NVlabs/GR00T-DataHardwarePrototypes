# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import struct
import time
from abc import abstractmethod
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Self, Sequence

from umi.common.data.frame import Frame
from umi.common.data.frame_buffer import FrameBuffer
from umi.common.hardware_sdk.sensor_interface import SensorInterface
from umi.common.utility.print_utils import print_green, print_yellow

from .protocol import (
    SENSOR_TYPE_SENSOR_HEALTH,
    Packet,
    SensorHealth,
    parse_sensor_health,
)
from .uart_reader import UartSensorReader

LOG = logging.getLogger(__name__)

__all__ = [
    "DeviceSample",
    "SensorDevice",
    "SensorFaultError",
    "SensorReading",
    "SensorSpec",
]

Decoder = Callable[[bytes], Any]
HealthKey = tuple[int, int]


@dataclass(frozen=True)
class SensorSpec:
    """Specification for a sensor data channel.

    Defines how to decode raw bytes from a UMI sensor segment.
    """

    sensor_type: int  # Protocol sensor type ID
    sensor_id: int  # Instance ID for this type
    name: str  # Human-readable name
    fmt: str = "<f"  # Struct format (default: little-endian float)
    scale: float | Sequence[float] = 1.0  # Scaling factor(s)
    decoder: Decoder | None = None  # Custom decoder function
    config: Mapping[str, Any] | None = None  # Additional configuration

    def decode(self, payload: bytes) -> Any:
        """Decode sensor payload bytes to value(s).

        Args:
            payload: Raw bytes from sensor segment

        Returns:
            Decoded value (scalar or tuple)

        Raises:
            ValueError: If payload size doesn't match format
        """
        if self.decoder is not None:
            return self.decoder(payload)

        expected = struct.calcsize(self.fmt)
        if expected != len(payload):
            raise ValueError(
                f"Sensor {self.name} expects {expected} bytes but received {len(payload)}"
            )

        values = struct.unpack(self.fmt, payload)
        scaled = self._apply_scale(values)

        if len(scaled) == 1:
            return scaled[0]
        return scaled

    def _apply_scale(self, values: Sequence[float]) -> Sequence[float]:
        """Apply scaling factor(s) to decoded values.

        Args:
            values: Raw decoded values

        Returns:
            Scaled values
        """
        if isinstance(self.scale, SequenceABC):
            return tuple(v * s for v, s in zip(values, self.scale))
        else:
            return tuple(v * self.scale for v in values)


@dataclass
class SensorReading:
    """A single sensor reading from a UMI frame."""

    spec: SensorSpec  # Sensor specification
    value: Any  # Decoded value
    raw: bytes  # Original raw bytes


@dataclass
class DeviceSample:
    """A complete sample from a sensor device.

    Contains timestamp and all sensor readings from one UMI frame.
    """

    timestamp_ms: int  # Hardware timestamp (milliseconds)
    source: int  # Source device ID
    readings: dict[str, SensorReading]  # Sensor name -> reading
    health: dict[HealthKey, SensorHealth] = field(default_factory=dict)  # (type, id)
    # Host monotonic time the reader thread received the packet (None when the
    # packet did not come through UartSensorReader).
    receive_time: float | None = None


class SensorFaultError(RuntimeError):
    """Raised when SENSOR_HEALTH reports a fatal state for this device."""

    def __init__(
        self,
        device_name: str,
        sensor_type: int,
        sensor_id: int,
        health: SensorHealth,
    ) -> None:
        self.device_name = device_name
        self.sensor_type = sensor_type
        self.sensor_id = sensor_id
        self.health = health
        super().__init__(
            f"[{device_name}] sensor 0x{sensor_type:02X}#{sensor_id} "
            f"fatal health state: "
            f"state={health.state.name} err={health.last_error_name} "
            f"count={health.error_count}"
        )


class SensorDevice(SensorInterface):
    """Base class for sensor devices communicating via UMI protocol.

    This class handles:
    - Registering with UartSensorReader for filtered packet delivery
    - Decoding sensor segments using SensorSpec
    - Providing get_frame() interface for Recorder

    The UartSensorReader owns the filtering logic and dispatches packets
    directly to each device's queue based on sensor types. This is more
    efficient than having each device run its own filter thread.

    Inherits from SensorInterface for consistent sensor API.
    """

    def __init__(
        self,
        device_name: str,
        reader: UartSensorReader,
        sensors: Sequence[SensorSpec] | None = None,
        latency: float | None = None,
        degraded_warn_period_s: float = 1.0,
    ) -> None:
        """Initialize sensor device.

        Args:
            device_name: Unique device identifier
            reader: Shared UartSensorReader for reading frames
            sensors: Sequence of SensorSpec defining the sensor channels.
                     If None, uses default sensors from _default_sensors().
            latency: Optional latency value for timing compensation
            degraded_warn_period_s: Minimum seconds between degraded warnings.
        """
        self.sensors = sensors if sensors is not None else self._default_sensors()

        if not self.sensors:
            raise ValueError("sensors must not be empty")

        super().__init__(device_name=device_name, latency=latency)

        self._sensor_map: dict[tuple[int, int], SensorSpec] = {
            (spec.sensor_type, spec.sensor_id): spec for spec in self.sensors
        }

        self._reader = reader

        # The reader writes matching packets into this subscription buffer.
        self._packet_buffer: FrameBuffer | None = None

        self._last_good_frame: Frame | None = None
        self._degraded_warn_period_s = degraded_warn_period_s
        self._last_degraded_warn_t: float = 0.0

    def start_streaming(self) -> None:
        """Start receiving data from the device."""
        if self.is_running:
            print_yellow(f"[{self.device_name}] Already streaming.")
            return

        self._packet_buffer = self._reader.register_device(self)
        self.is_running = True
        print_green(f"[{self.device_name}] Streaming started.")

    def stop_streaming(self) -> None:
        """Stop receiving data from the device."""
        if not self.is_running:
            return

        self._reader.unregister_device(self)
        self._packet_buffer = None

        self.is_running = False
        print_green(f"[{self.device_name}] Streaming stopped.")

    def identity(self) -> dict | None:
        """The board behind the shared UART reader (see UartSensorReader)."""
        getter = getattr(self._reader, "identity", None)
        return getter() if callable(getter) else None

    def observed_identity(self) -> dict | None:
        getter = getattr(self._reader, "observed_identity", None)
        return getter() if callable(getter) else None

    def get_frame(self) -> Frame | None:
        """Get the latest frame (required by BaseRecorder).

        Returns:
            Frame object with sensor data or None if no data available
        """
        if not self.is_running:
            return None

        sample = self._read_latest()
        if sample is None:
            return None

        return self._sample_to_frame_with_health(sample)

    @abstractmethod
    def _sample_to_frame(self, sample: DeviceSample) -> Frame | None:
        """Convert DeviceSample to appropriate Frame subclass.

        Args:
            sample: Parsed device sample

        Returns:
            Frame object for this sensor type.
        """
        pass

    def _subscription_keys(self) -> set[tuple[int, int]]:
        """Return packet keys this device should receive from the UART reader."""
        keys = {(spec.sensor_type, spec.sensor_id) for spec in self.sensors}
        keys |= {(SENSOR_TYPE_SENSOR_HEALTH, spec.sensor_id) for spec in self.sensors}
        return keys

    def _sample_to_frame_with_health(self, sample: DeviceSample) -> Frame | None:
        device_keys = {(spec.sensor_type, spec.sensor_id) for spec in self.sensors}
        device_health = {
            key: health for key, health in sample.health.items() if key in device_keys
        }

        for (sensor_type, sensor_id), health in device_health.items():
            if health.is_fatal:
                raise SensorFaultError(self.device_name, sensor_type, sensor_id, health)

        degraded = {
            key: health
            for key, health in device_health.items()
            if health.is_recoverable
        }
        if degraded:
            self._maybe_warn_degraded(degraded)
            if self._last_good_frame is not None:
                # Re-emit the last good values, stamped with the arrival time
                # of the degraded sample that triggered the hold.
                return replace(
                    self._last_good_frame, receive_time=self._receive_time_of(sample)
                )

        frame = self._sample_to_frame(sample)
        if frame is not None and not degraded:
            self._last_good_frame = frame
        return frame

    @staticmethod
    def _receive_time_of(sample: DeviceSample) -> float:
        """Arrival time to stamp on a frame built from ``sample``.

        The reader thread stamps every packet when its bytes arrive; frames
        must carry that time, not the time of the poll that converts them,
        otherwise a stalled reader re-emits stale values as fresh rows.
        ``time.monotonic()`` is only a fallback for samples that did not come
        through ``UartSensorReader``.
        """
        if sample.receive_time is not None:
            return sample.receive_time
        return time.monotonic()

    def _maybe_warn_degraded(self, degraded: Mapping[HealthKey, SensorHealth]) -> None:
        now = time.monotonic()
        if now - self._last_degraded_warn_t < self._degraded_warn_period_s:
            return

        details = ", ".join(
            f"0x{sensor_type:02X}#{sensor_id}:"
            f"err={health.last_error_name},count={health.error_count}"
            for (sensor_type, sensor_id), health in sorted(degraded.items())
        )
        print_yellow(f"[{self.device_name}] SENSOR_HEALTH DEGRADED ({details})")
        self._last_degraded_warn_t = now

    def _default_sensors(self) -> Sequence[SensorSpec]:
        """Return default sensor specifications for this device type.

        Subclasses should override this to provide default sensors.
        Users can still override by passing sensors parameter to __init__.

        Returns:
            Sequence of SensorSpec, or empty sequence if no defaults.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _default_sensors()"
        )

    def _read_latest(self) -> DeviceSample | None:
        """Get the most recent sample.

        Returns:
            Latest device sample or None
        """
        if self._packet_buffer is None:
            return None

        if self._reader.disconnect_reason is not None:
            raise RuntimeError(self._reader.disconnect_reason)

        latest = self._packet_buffer.peek_latest()

        if latest is None:
            return None

        sample = self._packet_to_sample(latest)
        return sample if sample.readings or sample.health else None

    def _packet_to_sample(self, packet: Packet) -> DeviceSample:
        """Convert packet to DeviceSample.

        Args:
            packet: Parsed packet

        Returns:
            DeviceSample with decoded readings
        """
        readings: dict[str, SensorReading] = {}
        health: dict[HealthKey, SensorHealth] = {}

        for segment in packet.segments:
            if segment.sensor_type == SENSOR_TYPE_SENSOR_HEALTH:
                parsed_health = parse_sensor_health(segment.data)
                if parsed_health is None:
                    LOG.warning(
                        "Invalid SENSOR_HEALTH segment for sensor #%d: %d bytes",
                        segment.sensor_id,
                        len(segment.data),
                    )
                    continue
                health[(parsed_health.source_sensor_type, segment.sensor_id)] = (
                    parsed_health
                )
                continue

            spec = self._sensor_map.get((segment.sensor_type, segment.sensor_id))
            if not spec:
                continue

            try:
                value = spec.decode(segment.data)
            except ValueError as exc:
                LOG.warning("Failed to decode sensor %s: %s", spec.name, exc)
                continue

            readings[spec.name] = SensorReading(
                spec=spec, value=value, raw=segment.data
            )

        return DeviceSample(
            timestamp_ms=packet.timestamp_ms,
            source=packet.source,
            readings=readings,
            health=health,
            receive_time=packet.receive_time,
        )

    def __enter__(self) -> Self:
        """Context manager entry."""
        self.start_streaming()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Context manager exit."""
        self.stop_streaming()

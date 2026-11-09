# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UART sensor reader for UMI protocol.

Background task that reads bytes from UART, parses packets, and dispatches
them to registered device buffers based on sensor type filtering.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

try:
    import serial
    import serial.tools.list_ports
except ImportError as exc:
    raise ImportError(
        "UartSensorReader requires pyserial. Install with: pip install pyserial"
    ) from exc

from umi.common.data.frame_buffer import FrameBuffer
from umi.common.hardware_sdk.sensor.protocol import (
    DEVICE_ID_LENGTH,
    SENSOR_TYPE_SENSOR_HEALTH,
    Packet,
    PacketParser,
    parse_sensor_health,
)
from umi.common.utility.print_utils import print_red, print_yellow
from umi.utils.device_manager import (
    exoskeleton_serials,
    get_exoskeleton_serial,
    normalize_device_id,
)

if TYPE_CHECKING:
    from .sensor_device import SensorDevice

LOG = logging.getLogger(__name__)


@dataclass
class DeviceSubscription:
    """Subscription entry for a registered device.

    Tracks the device's buffer and the sensor specs it cares about.
    """

    device: "SensorDevice"
    sensor_keys: Set[tuple[int, int]]  # Set of (sensor_type, sensor_id) tuples
    buffer: FrameBuffer  # Device-specific packet buffer


class UartSensorReader:
    """Background task that reads bytes from UART and dispatches packets to devices.

    This reader maintains a persistent thread that:
    1. Reads raw bytes from serial port
    2. Feeds bytes to PacketParser
    3. Dispatches parsed packets to registered device buffers based on sensor specs

    Devices register themselves with their sensor specs. The reader filters packets
    and only delivers packets to devices that have matching sensor specs in their
    TLV segments. Each device gets a FrameBuffer for efficient latest-frame access.
    """

    PICO_DESCRIPTION_PATTERNS = ["pico", "raspberry pi pico", "rp2040"]

    @staticmethod
    def list_connected_devices(
        filter_pico: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        List all connected serial devices (optionally filtered for Pico boards).

        Args:
            filter_pico: If True, only return devices matching Pico board descriptions.

        Returns:
            list: List of dicts with 'device', 'description', 'serial_number',
                  'hwid', 'index' keys.

        Example:
            >>> devices = UartSensorReader.list_connected_devices()
            >>> for dev in devices:
            ...     print(f"Device {dev['index']}: {dev['device']} - {dev['serial_number']}")
        """
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("No serial ports found.")
            return []

        devices = []
        for index, port_info in enumerate(ports):
            description = port_info.description.lower() if port_info.description else ""
            serial_num = port_info.serial_number if port_info.serial_number else None

            if filter_pico:
                is_pico = any(
                    pattern in description
                    for pattern in UartSensorReader.PICO_DESCRIPTION_PATTERNS
                )
                if not is_pico:
                    continue

            device_info = {
                "device": port_info.device,
                "description": port_info.description,
                "serial_number": serial_num,
                "hwid": port_info.hwid,
                "index": len(devices),  # Index among filtered devices
            }
            devices.append(device_info)

        print(f"Found {len(devices)} device(s):")
        for dev in devices:
            print(f"  [{dev['index']}] {dev['device']}")
            print(f"      Description: {dev['description']}")
            print(f"      Serial: {dev['serial_number']}")
        return devices

    @staticmethod
    def find_port_by_serial(serial_number: str) -> Optional[str]:
        """
        Find serial port path by USB serial number.

        The board firmware exposes two CDC ACM interfaces sharing the same
        USB serial: CDC0 is hijacked by the Pico SDK for stdio/REPL, CDC1
        carries the UMI binary protocol. The OS enumerates them in descriptor
        order, so the lexicographically-last device path corresponds to the
        higher CDC interface number (e.g. ``/dev/cu.usbmodemXXX4`` over
        ``XXX1`` on macOS, ``/dev/ttyACM1`` over ``ttyACM0`` on Linux). When
        more than one port matches the serial, that last one is what we want.

        Args:
            serial_number: The USB serial number to match.

        Returns:
            str: Port path (e.g., '/dev/ttyACM1') if found, None otherwise.
        """
        ports = serial.tools.list_ports.comports()
        matches = [p.device for p in ports if p.serial_number == serial_number]
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]

        chosen = sorted(matches)[-1]
        LOG.info(
            "Multiple serial ports share serial %s: %s; selecting %s "
            "(higher CDC interface, expected to carry the UMI binary protocol).",
            serial_number,
            matches,
            chosen,
        )
        return chosen

    @staticmethod
    def get_serial_for_port(port: str) -> Optional[str]:
        """Get the USB serial number of the device on a given port.

        Args:
            port: Serial port path (e.g., '/dev/ttyACM0').

        Returns:
            USB serial number string if found, None otherwise.
        """
        for port_info in serial.tools.list_ports.comports():
            if port_info.device == port:
                return port_info.serial_number
        return None

    def __init__(
        self,
        port: Optional[str] = None,
        uart_device_name: Optional[str] = None,
        baudrate: int = 115200,
        buffer_size: int = 256,
        read_size: int = 4096,
        idle_sleep: float = 0.002,
        serial_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize UART reader.

        Args:
            port: Serial port path (e.g., '/dev/ttyUSB0'). If not provided,
                  uart_device_name must be specified.
            uart_device_name: Exoskeleton name (e.g., "DEXUMI-Right"). If
                              provided, the board serial its calibration records
                              selects the port automatically.
            baudrate: Communication speed (default: 115200)
            buffer_size: Maximum number of packets to buffer per device
            read_size: Maximum bytes per ``serial.read`` call. The port is
                       opened non-blocking (``timeout=0``), so each call
                       returns whatever is already buffered, up to this size,
                       after one ``select`` + one ``os.read``. Keep it at
                       least a few frames large: reading one byte per call
                       costs two GIL round trips per byte (~1,800 per DexUMI
                       frame) and lets the reader fall behind the board under
                       recorder load, which surfaces as stale, duplicated
                       sensor rows (see
                       real_scripts/latency_measurement/uart_reader_sim/).
            idle_sleep: Sleep time when no data available (seconds)
            serial_kwargs: Additional arguments for serial.Serial()

        Raises:
            ValueError: If neither port nor uart_device_name is provided.
            RuntimeError: If uart_device_name is provided but device not found.
        """
        self.uart_device_name = uart_device_name

        # Expected RP2040 board ID for frame-level device-swap detection on protocol
        # v0x11+. Available when the reader is opened by calibrated device name.
        self._expected_device_id: Optional[bytes] = None

        if uart_device_name is not None:
            device_serial = get_exoskeleton_serial(uart_device_name)
            resolved_port = self.find_port_by_serial(device_serial)
            if resolved_port is None:
                print_red(
                    f"Device '{uart_device_name}' (serial: {device_serial}) not found."
                )
                print("Available devices:")
                self.list_connected_devices(filter_pico=False)
                raise RuntimeError(
                    f"UART device '{uart_device_name}' (serial: {device_serial}) not found. "
                    "Check if the device is connected."
                )
            self.port = resolved_port
            LOG.info(
                "Resolved device '%s' (serial: %s) to port %s",
                uart_device_name,
                device_serial,
                self.port,
            )

            # The USB serial is the RP2040 board ID encoded as hexadecimal.
            # Cache its bytes for constant-time frame identity checks.
            try:
                expected = bytes.fromhex(device_serial)
            except ValueError:
                LOG.warning(
                    "Calibration serial '%s' for '%s' is not valid hex; "
                    "skipping per-frame device_id validation.",
                    device_serial,
                    uart_device_name,
                )
                expected = None
            if expected is not None and len(expected) != DEVICE_ID_LENGTH:
                LOG.warning(
                    "Calibration serial for '%s' decodes to %d bytes "
                    "(expected %d); skipping per-frame device_id validation.",
                    uart_device_name,
                    len(expected),
                    DEVICE_ID_LENGTH,
                )
                expected = None
            self._expected_device_id = expected
        elif port is not None:
            self.port = port
        else:
            raise ValueError("Either 'port' or 'uart_device_name' must be provided.")

        self.baudrate = baudrate
        self._buffer_size = buffer_size
        self._parser = PacketParser()
        self._serial_kwargs = serial_kwargs or {}
        self._read_size = read_size
        self._idle_sleep = idle_sleep

        self._serial: Any | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        # Set by the reader thread if the USB serial link dies mid-session.
        self._disconnect_reason: str | None = None

        self._device_id_mismatch_count: int = 0
        self._legacy_v1_warned: bool = False
        self._sensor_fault_count: int = 0
        # Track the last observed board ID, including mismatches, and count v0x10
        # frames that contain no identity. Exposed through observed_identity().
        self._last_device_id: bytes | None = None
        self._legacy_frame_count: int = 0

        self._subscriptions: dict[str, DeviceSubscription] = {}
        self._subscriptions_lock = threading.Lock()

        # Verify the device on the resolved port is the board the calibration
        # was made on.
        self._verify_device_serial()

    def _verify_device_serial(self) -> None:
        """Verify the USB serial of the device on self.port.

        When uart_device_name is set, checks that the physical device serial
        matches the ``device_serial`` its calibration records.

        When a raw port is given (no uart_device_name), logs which calibration
        records the board on that port, if any. A raw port is how a board is
        opened before its first calibration, so an unknown board only warns.

        Raises:
            AssertionError: If the serial number doesn't match.
        """
        actual_serial = self.get_serial_for_port(self.port)
        if actual_serial is None:
            if self.uart_device_name is not None:
                raise AssertionError(
                    f"[UART DISCONNECTED] Could not read USB serial for "
                    f"{self.port} (device {self.uart_device_name!r}). "
                    "The board may have been unplugged or the port disappeared."
                )
            LOG.warning(
                "Could not read USB serial for port %s — skipping verification",
                self.port,
            )
            return

        if self.uart_device_name is not None:
            expected_serial = get_exoskeleton_serial(self.uart_device_name)
            if actual_serial != expected_serial:
                raise AssertionError(
                    f"[UART] Serial mismatch on {self.port} for device "
                    f"{self.uart_device_name!r}: expected {expected_serial}, "
                    f"got {actual_serial}. Wrong board on this USB port or hot-swap."
                )
            LOG.info(
                "Verified device '%s' serial %s on %s",
                self.uart_device_name,
                actual_serial,
                self.port,
            )
        else:
            matched_names = [
                name
                for name, serial in exoskeleton_serials().items()
                if serial == actual_serial
            ]
            if not matched_names:
                LOG.warning(
                    "Device on %s has serial %s, which no exoskeleton calibration "
                    "records yet",
                    self.port,
                    actual_serial,
                )
                return
            LOG.info(
                "Device on %s has serial %s (calibrated as %s)",
                self.port,
                actual_serial,
                ", ".join(matched_names),
            )

    def start(self) -> None:
        """Start the background reading thread."""
        if self._started:
            return

        self._stop_event.clear()
        self._open_port()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._started = True
        LOG.info(
            f"UART reader ({self.uart_device_name}) started on {self.port} @ {self.baudrate} baud"
        )

    def stop(self) -> None:
        """Stop the background reading thread."""
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        if self._serial and self._serial.is_open:
            self._serial.close()

        self._started = False
        LOG.info("UART reader stopped")

    def register_device(self, device: "SensorDevice") -> FrameBuffer:
        """Register a device to receive packets matching its sensor specs.

        Args:
            device: The SensorDevice to register

        Returns:
            FrameBuffer that will receive filtered packets for this device
        """
        # Extract subscription keys from device. SensorDevice includes both
        # physical sensor TLVs and paired SENSOR_HEALTH meta TLVs by default.
        sensor_keys = device._subscription_keys()

        device_buffer = FrameBuffer(maxlen=self._buffer_size)

        subscription = DeviceSubscription(
            device=device,
            sensor_keys=sensor_keys,
            buffer=device_buffer,
        )

        with self._subscriptions_lock:
            self._subscriptions[device.device_name] = subscription

        return device_buffer

    def unregister_device(self, device: "SensorDevice") -> None:
        """Unregister a device from receiving packets.

        Args:
            device: The SensorDevice to unregister
        """
        with self._subscriptions_lock:
            if device.device_name in self._subscriptions:
                del self._subscriptions[device.device_name]

    def _open_port(self) -> None:
        """Open the serial port."""
        self._serial = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=0,  # Non-blocking reads
            **self._serial_kwargs,
        )

    def _loop(self) -> None:
        """Main reading loop (runs in background thread)."""
        assert self._serial is not None

        while not self._stop_event.is_set():
            try:
                chunk = self._serial.read(self._read_size)
            except serial.SerialException as exc:  # type: ignore[attr-defined]
                label = self.uart_device_name or self.port
                self._disconnect_reason = (
                    f"[UART DISCONNECTED] Serial read failed for {label!r} "
                    f"on {self.port}: {exc!r}. USB unplugged, cable fault, or board "
                    "power loss during recording."
                )
                LOG.error(self._disconnect_reason)
                break

            if chunk:
                # Timestamp each received chunk in the reader thread. All packets
                # completed
                # by that chunk share its arrival time; later polls must not refresh it.
                received = time.monotonic()
                if LOG.isEnabledFor(logging.DEBUG):
                    LOG.debug("RX [%d bytes]: %s", len(chunk), chunk.hex())

                for packet in self._parser.feed(chunk):
                    self._dispatch(replace(packet, receive_time=received))
            else:
                time.sleep(self._idle_sleep)

    @property
    def disconnect_reason(self) -> str | None:
        """Reason recorded when the serial reader exits due to a read failure."""
        return self._disconnect_reason

    @property
    def device_id_mismatch_count(self) -> int:
        """Number of v0x11+ packets dropped because their device_id did not
        match the expected board id.

        Always 0 when the reader was opened by raw port (no expected id) or
        when the firmware is still on protocol v0x10. Useful for recording
        layers to flag a session that experienced a silent device swap.
        """
        return self._device_id_mismatch_count

    @property
    def sensor_fault_count(self) -> int:
        """Number of fatal SENSOR_HEALTH reports observed on this reader."""
        return self._sensor_fault_count

    # Device family assigned by the recording factory.
    device_kind: Optional[str] = None
    # Class defaults support readers constructed without running __init__.
    _last_device_id: bytes | None = None
    _legacy_frame_count: int = 0

    def identity(self) -> dict:
        """Expected board: the serial ``uart_device_name``'s calibration records.

        A reader opened by raw port has no expected board (``id`` None).
        """
        name = self.uart_device_name
        kind = self.device_kind or "dexumi_exoskeleton"
        expected = None
        if name:
            try:
                expected = get_exoskeleton_serial(name)
            except ValueError:
                expected = None
        return {"kind": kind, "name": name, "id": normalize_device_id(expected)}

    def observed_identity(self) -> dict:
        """Board on the port now: its USB serial (re-read on every call) plus
        the in-band ids of the frames received so far."""
        try:
            observed = self.get_serial_for_port(self.port)
        except Exception:  # noqa: BLE001 - port enumeration is best effort here
            observed = None
        last = self._last_device_id
        return {
            "id": normalize_device_id(observed),
            "in_band_id": last.hex().upper() if last else None,
            "foreign_id_frames": int(self._device_id_mismatch_count),
            "legacy_frames": int(self._legacy_frame_count),
        }

    def _dispatch(self, packet: Packet) -> None:
        """Dispatch packet to all registered devices with matching sensor specs.

        If this reader was opened by named device, v0x11+ packets whose
        device_id does not match the expected board id are dropped (with a
        rate-limited red warning). Legacy v0x10 packets carry no device_id
        and fall through unconditionally — they are accepted for backward
        compatibility with v0x10 frames.

        Args:
            packet: Parsed packet to dispatch
        """
        if packet.device_id is None:
            self._legacy_frame_count += 1
        else:
            self._last_device_id = packet.device_id
        if self._expected_device_id is not None:
            if packet.device_id is None:
                # Warn once when v0x10 frames cannot verify device identity, then accept
                # them.
                if not self._legacy_v1_warned:
                    self._legacy_v1_warned = True
                    print_yellow(
                        f"[{self.uart_device_name}] receiving legacy UMI v0x10 frames "
                        f"(no device_id field); silent device-swap detection is "
                        f"disabled until firmware is upgraded to v0x11."
                    )
            elif packet.device_id != self._expected_device_id:
                self._device_id_mismatch_count += 1
                # Report the first mismatch and every 1,000 subsequent dropped frames.
                if (
                    self._device_id_mismatch_count == 1
                    or self._device_id_mismatch_count % 1000 == 0
                ):
                    print_red(
                        f"[{self.uart_device_name}] device_id mismatch! "
                        f"expected {self._expected_device_id.hex().upper()}, "
                        f"got {packet.device_id.hex().upper()} "
                        f"(dropped={self._device_id_mismatch_count})"
                    )
                return

        self._observe_sensor_health(packet)

        packet_sensor_keys = {
            (seg.sensor_type, seg.sensor_id) for seg in packet.segments
        }

        with self._subscriptions_lock:
            for subscription in self._subscriptions.values():
                if packet_sensor_keys & subscription.sensor_keys:
                    subscription.buffer.put(packet)

    def _observe_sensor_health(self, packet: Packet) -> None:
        """Update reader-level health counters for all SENSOR_HEALTH TLVs."""
        for segment in packet.segments:
            if segment.sensor_type != SENSOR_TYPE_SENSOR_HEALTH:
                continue

            health = parse_sensor_health(segment.data)
            if health is not None and health.is_fatal:
                self._sensor_fault_count += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--uart_device_name", type=str, default=None)
    parser.add_argument("--port", type=str, default="/dev/ttyACM1")
    args = parser.parse_args()

    UartSensorReader.list_connected_devices(filter_pico=True)
    reader = UartSensorReader(port=args.port, uart_device_name=args.uart_device_name)
    reader.start()
    print(f"Reader started on {reader.port}")
    time.sleep(1)
    reader.stop()
    print("Reader stopped")

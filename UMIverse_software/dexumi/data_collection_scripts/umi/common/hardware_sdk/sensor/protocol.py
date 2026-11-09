# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UMI protocol packet parsing utilities.

Based on the UMI protocol specification from gaoyichuan/umi-sdk.
Implements the protocol format with sync bytes, CRC16, and TLV segments.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence

LOG = logging.getLogger(__name__)

SYNC_BYTES = b"\xa5\x5a"

# Wire formats, including the two-byte sync word:
# v0x10: 10-byte header, no device identity.
# v0x11: 18-byte header, adds the eight-byte RP2040 board ID (USB serial)
# between source and timestamp_ms.
# v0x12: v0x11 header with SENSOR_HEALTH (0xF0) metadata TLVs.
PROTOCOL_VERSION_V1 = 0x10
PROTOCOL_VERSION_V2 = 0x11
PROTOCOL_VERSION_V3 = 0x12
PROTOCOL_VERSION = PROTOCOL_VERSION_V3  # current canonical version
SUPPORTED_VERSIONS = frozenset(
    {PROTOCOL_VERSION_V1, PROTOCOL_VERSION_V2, PROTOCOL_VERSION_V3}
)

DEVICE_ID_LENGTH = 8

HEADER_LENGTH_V1 = 10  # Sync(2) + Version + Source + Timestamp(4) + Payload Length(2)
HEADER_LENGTH_V2 = HEADER_LENGTH_V1 + DEVICE_ID_LENGTH  # 18 bytes

# Alias for the v0x10 header layout.
HEADER_LENGTH = HEADER_LENGTH_V1

# Per-version field offsets (all measured from the start of the packet,
# i.e. including the 2-byte sync word).
_LAYOUTS = {
    PROTOCOL_VERSION_V1: {
        "header_len": HEADER_LENGTH_V1,
        "src_off": 3,
        "ts_off": 4,
        "ts_end": 8,
        "plen_off": 8,
        "plen_end": 10,
        "payload_off": 10,
        "devid_off": None,
    },
    PROTOCOL_VERSION_V2: {
        "header_len": HEADER_LENGTH_V2,
        "src_off": 3,
        "ts_off": 12,
        "ts_end": 16,
        "plen_off": 16,
        "plen_end": 18,
        "payload_off": 18,
        "devid_off": 4,
    },
    PROTOCOL_VERSION_V3: {
        "header_len": HEADER_LENGTH_V2,
        "src_off": 3,
        "ts_off": 12,
        "ts_end": 16,
        "plen_off": 16,
        "plen_end": 18,
        "payload_off": 18,
        "devid_off": 4,
    },
}

SENSOR_TYPE_JOINT_ENCODER = 0x01
SENSOR_TYPE_TACTILE = 0x02
SENSOR_TYPE_SENSOR_HEALTH = 0xF0


class SensorState(IntEnum):
    """Health state for SENSOR_HEALTH TLVs.

    Unknown wire values are mapped to ``UNKNOWN`` and treated as fatal by
    consumers so they fail safe when firmware gets ahead of the SDK.
    """

    OK = 0
    DEGRADED = 1
    FAULT = 2
    INIT_FAILED = 3
    UNKNOWN = 0xFF


class Tle5012bError(IntEnum):
    """TLE5012B driver error codes reported via SENSOR_HEALTH.last_error."""

    OK = 0
    SYSTEM = 1
    INTERFACE = 2
    INVALID_ANGLE = 3
    ANGLE_SPEED = 4
    CRC = 0xFF


HealthExtraParser = Callable[[bytes], Mapping[str, Any]]
ErrorNameResolver = Callable[[int], str]

_HEALTH_EXTRA_REGISTRY: dict[int, HealthExtraParser] = {}
_ERROR_NAME_REGISTRY: dict[int, ErrorNameResolver] = {}


def register_health_extra_parser(
    source_sensor_type: int, parser: HealthExtraParser
) -> None:
    """Register a parser for a SENSOR_HEALTH device-private extra payload."""
    _HEALTH_EXTRA_REGISTRY[source_sensor_type] = parser


def register_error_name_resolver(
    source_sensor_type: int, resolver: ErrorNameResolver
) -> None:
    """Register a last_error name resolver for a SENSOR_HEALTH source type."""
    _ERROR_NAME_REGISTRY[source_sensor_type] = resolver


@dataclass(frozen=True)
class SensorHealth:
    """Parsed SENSOR_HEALTH (0xF0) TLV body.

    The first 8 bytes are the protocol-level health header. ``extra_raw`` is
    sensor-driver private and can be decoded through a registered parser into
    ``extra_decoded``. The outer TLV's sensor_id is intentionally stored by the
    caller (DeviceSample.health key), because the body describes only the
    source sensor type and diagnostic state.
    """

    source_sensor_type: int
    state: SensorState
    last_error: int
    error_count: int
    extra_raw: bytes
    extra_decoded: Mapping[str, Any]

    @property
    def last_error_name(self) -> str:
        resolver = _ERROR_NAME_REGISTRY.get(self.source_sensor_type)
        if resolver is not None:
            return resolver(self.last_error)
        return f"0x{self.last_error:02X}"

    @property
    def is_healthy(self) -> bool:
        return self.state == SensorState.OK

    @property
    def is_recoverable(self) -> bool:
        return self.state == SensorState.DEGRADED

    @property
    def is_fatal(self) -> bool:
        return self.state in (
            SensorState.FAULT,
            SensorState.INIT_FAILED,
            SensorState.UNKNOWN,
        )


def _parse_tle5012b_extra(payload: bytes) -> Mapping[str, Any]:
    if len(payload) != 4:
        return {}
    safety_word, status_word = struct.unpack("<HH", payload)
    return {
        "safety_word": safety_word,
        "status_word": status_word,
    }


def _tle5012b_error_name(code: int) -> str:
    try:
        return Tle5012bError(code).name
    except ValueError:
        return f"0x{code:02X}"


register_health_extra_parser(SENSOR_TYPE_JOINT_ENCODER, _parse_tle5012b_extra)
register_error_name_resolver(SENSOR_TYPE_JOINT_ENCODER, _tle5012b_error_name)


def parse_sensor_health(payload: bytes) -> SensorHealth | None:
    """Parse a SENSOR_HEALTH (0xF0) TLV body.

    Wire layout:
      [0]    source_sensor_type (u8)
      [1]    state              (u8)
      [2]    last_error         (u8)
      [3]    reserved           (u8)
      [4:8]  error_count        (u32 LE)
      [8:]   source-sensor-specific fixed-size extra payload
    """
    if len(payload) < 8:
        return None

    source_sensor_type, state_value, last_error, _reserved, error_count = struct.unpack(
        "<BBBBI", payload[:8]
    )
    extra_raw = payload[8:]
    parser = _HEALTH_EXTRA_REGISTRY.get(source_sensor_type)
    extra_decoded = parser(extra_raw) if parser is not None else {}

    try:
        state = SensorState(state_value)
    except ValueError:
        state = SensorState.UNKNOWN

    return SensorHealth(
        source_sensor_type=source_sensor_type,
        state=state,
        last_error=last_error,
        error_count=error_count,
        extra_raw=extra_raw,
        extra_decoded=extra_decoded,
    )


# fmt: off
# CRC16 lookup table (from umi-sdk)
CRC16_TABLE = [
    0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50A5, 0x60C6, 0x70E7,
    0x8108, 0x9129, 0xA14A, 0xB16B, 0xC18C, 0xD1AD, 0xE1CE, 0xF1EF,
    0x1231, 0x0210, 0x3273, 0x2252, 0x52B5, 0x4294, 0x72F7, 0x62D6,
    0x9339, 0x8318, 0xB37B, 0xA35A, 0xD3BD, 0xC39C, 0xF3FF, 0xE3DE,
    0x2462, 0x3443, 0x0420, 0x1401, 0x64E6, 0x74C7, 0x44A4, 0x5485,
    0xA56A, 0xB54B, 0x8528, 0x9509, 0xE5EE, 0xF5CF, 0xC5AC, 0xD58D,
    0x3653, 0x2672, 0x1611, 0x0630, 0x76D7, 0x66F6, 0x5695, 0x46B4,
    0xB75B, 0xA77A, 0x9719, 0x8738, 0xF7DF, 0xE7FE, 0xD79D, 0xC7BC,
    0x48C4, 0x58E5, 0x6886, 0x78A7, 0x0840, 0x1861, 0x2802, 0x3823,
    0xC9CC, 0xD9ED, 0xE98E, 0xF9AF, 0x8948, 0x9969, 0xA90A, 0xB92B,
    0x5AF5, 0x4AD4, 0x7AB7, 0x6A96, 0x1A71, 0x0A50, 0x3A33, 0x2A12,
    0xDBFD, 0xCBDC, 0xFBBF, 0xEB9E, 0x9B79, 0x8B58, 0xBB3B, 0xAB1A,
    0x6CA6, 0x7C87, 0x4CE4, 0x5CC5, 0x2C22, 0x3C03, 0x0C60, 0x1C41,
    0xEDAE, 0xFD8F, 0xCDEC, 0xDDCD, 0xAD2A, 0xBD0B, 0x8D68, 0x9D49,
    0x7E97, 0x6EB6, 0x5ED5, 0x4EF4, 0x3E13, 0x2E32, 0x1E51, 0x0E70,
    0xFF9F, 0xEFBE, 0xDFDD, 0xCFFC, 0xBF1B, 0xAF3A, 0x9F59, 0x8F78,
    0x9188, 0x81A9, 0xB1CA, 0xA1EB, 0xD10C, 0xC12D, 0xF14E, 0xE16F,
    0x1080, 0x00A1, 0x30C2, 0x20E3, 0x5004, 0x4025, 0x7046, 0x6067,
    0x83B9, 0x9398, 0xA3FB, 0xB3DA, 0xC33D, 0xD31C, 0xE37F, 0xF35E,
    0x02B1, 0x1290, 0x22F3, 0x32D2, 0x4235, 0x5214, 0x6277, 0x7256,
    0xB5EA, 0xA5CB, 0x95A8, 0x8589, 0xF56E, 0xE54F, 0xD52C, 0xC50D,
    0x34E2, 0x24C3, 0x14A0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405,
    0xA7DB, 0xB7FA, 0x8799, 0x97B8, 0xE75F, 0xF77E, 0xC71D, 0xD73C,
    0x26D3, 0x36F2, 0x0691, 0x16B0, 0x6657, 0x7676, 0x4615, 0x5634,
    0xD94C, 0xC96D, 0xF90E, 0xE92F, 0x99C8, 0x89E9, 0xB98A, 0xA9AB,
    0x5844, 0x4865, 0x7806, 0x6827, 0x18C0, 0x08E1, 0x3882, 0x28A3,
    0xCB7D, 0xDB5C, 0xEB3F, 0xFB1E, 0x8BF9, 0x9BD8, 0xABBB, 0xBB9A,
    0x4A75, 0x5A54, 0x6A37, 0x7A16, 0x0AF1, 0x1AD0, 0x2AB3, 0x3A92,
    0xFD2E, 0xED0F, 0xDD6C, 0xCD4D, 0xBDAA, 0xAD8B, 0x9DE8, 0x8DC9,
    0x7C26, 0x6C07, 0x5C64, 0x4C45, 0x3CA2, 0x2C83, 0x1CE0, 0x0CC1,
    0xEF1F, 0xFF3E, 0xCF5D, 0xDF7C, 0xAF9B, 0xBFBA, 0x8FD9, 0x9FF8,
    0x6E17, 0x7E36, 0x4E55, 0x5E74, 0x2E93, 0x3EB2, 0x0ED1, 0x1EF0,
]
# fmt: on


def crc16(data: bytes) -> int:
    """Calculate CRC16 checksum using UMI protocol table.

    Args:
        data: Bytes to calculate checksum for

    Returns:
        16-bit CRC value
    """
    crc = 0
    for byte in data:
        crc = (crc << 8) ^ CRC16_TABLE[((crc >> 8) ^ byte) & 0xFF]
    return crc & 0xFFFF


@dataclass(frozen=True)
class SensorSegment:
    """Represents a single TLV entry within the payload.

    Format:
        - Sensor Type (1 byte): Sensor category ID
        - Sensor ID (1 byte): Instance index for same type
        - Data Length (2 bytes, little-endian): Length of data
        - Data Bytes (N bytes): Sensor payload
    """

    sensor_type: int
    sensor_id: int
    data: bytes


@dataclass(frozen=True)
class Packet:
    """Represents a fully parsed UMI protocol packet.

    This is a low-level protocol message, distinct from sensor Frame classes.

    Format (v0x10, legacy):
        - Sync (2 bytes): 0xA5 0x5A
        - Version (1 byte): 0x10
        - Source (1 byte): Device type (high nibble) + instance ID (low nibble)
        - Timestamp (4 bytes, little-endian): Millisecond counter
        - Payload Length (2 bytes, little-endian): Length of payload
        - Payload (N bytes): Concatenated TLV segments
        - CRC16 (2 bytes, little-endian): Checksum of Version to end of Payload

    Format (v0x11+):
        - Sync (2 bytes): 0xA5 0x5A
        - Version (1 byte): 0x11 or 0x12
        - Source (1 byte): Device type (high nibble) + instance ID (low nibble)
        - Device ID (8 bytes): RP2040 unique board id (== USB serial hex)
        - Timestamp (4 bytes, little-endian)
        - Payload Length (2 bytes, little-endian)
        - Payload (N bytes): Concatenated TLV segments
        - CRC16 (2 bytes, little-endian): Checksum of Version to end of Payload

    Attributes:
        device_id: 8-byte board id present on v0x11+ packets; ``None`` for
            legacy v0x10 packets.
        receive_time: Host ``time.monotonic()`` when the reader thread pulled
            the bytes that completed this packet off the serial port. Set by
            ``UartSensorReader`` at dispatch; ``None`` for packets built by
            the parser alone. This is the arrival time that sensor frames
            report as ``receive_time`` (not the time of a later poll).
    """

    version: int
    source: int
    timestamp_ms: int
    segments: Sequence[SensorSegment]
    payload: bytes
    device_id: Optional[bytes] = None
    receive_time: Optional[float] = None


class PacketParser:
    """Incrementally consumes raw bytes and yields validated packets."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> List[Packet]:
        """Feed raw bytes and extract complete packets.

        Args:
            chunk: New bytes from serial port

        Returns:
            List of parsed packets (may be empty)
        """
        if not chunk:
            return []

        self._buffer.extend(chunk)
        packets: List[Packet] = []

        while True:
            packet = self._try_extract_packet()
            if packet is None:
                break
            packets.append(packet)

        return packets

    def _try_extract_packet(self) -> Packet | None:
        """Attempt to extract one complete packet from buffer.

        Returns:
            Parsed packet or None if incomplete/invalid
        """
        buf = self._buffer

        sync_index = self._find_sync(buf)
        LOG.debug("Sync index: %d", sync_index)

        if sync_index < 0:
            # No sync word present yet -> keep only tail to avoid unbounded growth
            if len(buf) > len(SYNC_BYTES):
                LOG.debug("Dropping %d bytes before sync", len(buf) - len(SYNC_BYTES))
                del buf[: -len(SYNC_BYTES)]
            return None

        if sync_index:
            LOG.debug("Dropping %d bytes before sync", sync_index)
            del buf[:sync_index]

        # Need at least sync(2) + version(1) to dispatch on protocol version
        if len(buf) < 3:
            return None

        version = buf[2]
        layout = _LAYOUTS.get(version)
        if layout is None:
            # An unknown version may indicate a false sync match inside payload data.
            # Advance one byte and resynchronize.
            LOG.warning(
                "Unsupported protocol version 0x%02X at sync; dropping sync byte",
                version,
            )
            del buf[0]
            return None

        header_len = layout["header_len"]

        if len(buf) < header_len:
            LOG.debug(
                "Not enough data for v0x%02X header yet (%d < %d)",
                version,
                len(buf),
                header_len,
            )
            return None

        payload_len = int.from_bytes(
            buf[layout["plen_off"] : layout["plen_end"]], "little"
        )
        full_len = header_len + payload_len + 2  # +CRC16

        if len(buf) < full_len:
            LOG.debug(
                "Not enough data for full packet yet (%d < %d)", len(buf), full_len
            )
            return None

        packet_bytes = bytes(buf[:full_len])

        # Validate CRC (range is "version .. end of payload", version-independent)
        if not self._crc_valid(packet_bytes):
            LOG.debug("CRC check failed, dropping sync byte and searching again")
            del buf[0]
            return None

        del buf[:full_len]
        source = packet_bytes[layout["src_off"]]
        timestamp_ms = int.from_bytes(
            packet_bytes[layout["ts_off"] : layout["ts_end"]], "little"
        )
        payload = packet_bytes[layout["payload_off"] : -2]
        segments = tuple(self._parse_segments(payload))

        device_id: Optional[bytes] = None
        devid_off = layout["devid_off"]
        if devid_off is not None:
            device_id = bytes(packet_bytes[devid_off : devid_off + DEVICE_ID_LENGTH])

        return Packet(
            version=version,
            source=source,
            timestamp_ms=timestamp_ms,
            segments=segments,
            payload=payload,
            device_id=device_id,
        )

    def _find_sync(self, buf: bytearray) -> int:
        """Find index of sync bytes in buffer.

        Returns:
            Index of sync bytes or -1 if not found
        """
        for idx in range(max(0, len(buf) - len(SYNC_BYTES) + 1)):
            if buf[idx : idx + 2] == SYNC_BYTES:
                return idx
        return -1

    def _crc_valid(self, packet_bytes: bytes) -> bool:
        """Verify CRC16 checksum.

        Args:
            packet_bytes: Complete packet including sync and CRC

        Returns:
            True if CRC is valid
        """
        payload = packet_bytes[2:-2]  # Version to end of Payload
        transmitted_crc = int.from_bytes(packet_bytes[-2:], "little")
        expected_crc = crc16(payload)

        if expected_crc != transmitted_crc:
            LOG.warning(
                "CRC mismatch: expected 0x%04X, got 0x%04X",
                expected_crc,
                transmitted_crc,
            )
            LOG.debug("Packet bytes: %s", packet_bytes.hex())

        return expected_crc == transmitted_crc

    def _parse_segments(self, payload: bytes) -> Iterable[SensorSegment]:
        """Parse TLV segments from payload.

        Args:
            payload: Raw payload bytes

        Yields:
            Parsed sensor segments
        """
        offset = 0
        total = len(payload)

        while offset + 4 <= total:
            sensor_type = payload[offset]
            sensor_id = payload[offset + 1]
            data_len = int.from_bytes(payload[offset + 2 : offset + 4], "little")
            offset += 4

            if offset + data_len > total:
                LOG.warning("Incomplete segment at offset %d", offset - 4)
                break

            data = payload[offset : offset + data_len]
            offset += data_len

            yield SensorSegment(sensor_type=sensor_type, sensor_id=sensor_id, data=data)

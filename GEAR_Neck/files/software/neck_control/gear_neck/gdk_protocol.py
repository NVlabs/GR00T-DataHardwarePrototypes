# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SteadyWin GDK CAN protocol V2.0, MIT variant: pack and unpack, with no bus.

The GIM3505-9 actuators of the GEAR Neck run on GDK drive boards, which speak
it over classic CAN with standard ids at 1 Mbit/s. A drive takes frames on its
ID (register 7) and sends every reply on its CAN_ID (register 14). Replies
carry no command byte, so a reply is known only by its length and by what was
asked:

    request              bytes   reply                                     bytes
    0x01 ping            1       ID, CAN_ID, model, firmware, drive mode    8
    0x02 status          1       mode, torque switch, voltage, faults       8
    register read        3       the register bytes                         n
    register write       7       0x01 done, 0x00 refused                    1
    FF x7 FC torque on   8       MIT state                                  8
    FF x7 FD torque off  8       none
    FF x7 FE set zero    8       none; taken only with torque off
    MIT command          8       MIT state, only while torque is on         8

Register values are little-endian; MIT fields are packed high bits first. The
vendor manual is GEAR_Neck/files/firmware/steadywin-gim-manual-v2.0.pdf.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from typing import NamedTuple

ID_MIN = 1  # the manual allows 0, but the vendor tool searches from 1
ID_MAX = 252
ID_REGISTER = 7
CAN_ID_REGISTER = 14
OPERATING_MODE_REGISTER = 33  # one byte, also byte 0 of the status reply
MIT_MODE = 6
CALIBRATION_REGISTER = 403  # one byte
PRESENT_POSITION_REGISTER = 552
PRESENT_POSITION_LENGTH = 4
# Firmware 3.0.1 counts this many per turn of the MIT position; the manual's
# demo implies 16384, 4x off.
PRESENT_POSITION_COUNTS_PER_TURN = 65536
FRAME_LENGTH = 8  # ping, status and MIT state replies
RANGE_LENGTH = 4  # one MIT range: two int16 ends
WRITE_REPLY_LENGTH = 1

_PING = 0x01
_READ_STATUS = 0x02
_WRITE_LENGTH_MAX = 4
_WRITE_PAD = 0xFF
_WRITE_DONE = 0x01
_TORQUE_ON = 0xFC
_TORQUE_OFF = 0xFD
_SET_ZERO = 0xFE
# Register 403: both encoders connected (0x30) and calibrated (0x0C).
_BOTH_ENCODERS_CALIBRATED = 0x3C
_POSITION_BITS = 16
_FIELD_BITS = 12  # velocity, kp, kd and torque
# Where each MIT range lives: its two int16 ends from this address, in
# 1/divisor units. In MitRanges order.
_RANGE_REGISTERS = (
    (330, 100),  # position, rad
    (334, 100),  # velocity, rad/s
    (338, 500),  # torque, Nm
    (342, 100),  # kp, Nm/rad
    (346, 100),  # kd, Nm*s/rad
)
RANGE_COUNT = len(_RANGE_REGISTERS)
_OPERATING_MODES = {0: "CURRENT", 1: "VELOCITY", 3: "POSITION", MIT_MODE: "MIT"}
_FAULT_FLAGS = (
    (0x01, "VOLTAGE"),
    (0x10, "FOC_TIMEOUT"),
    (0x20, "OVERLOAD"),
    (0x40, "MOTOR_TEMPERATURE"),
    (0x80, "TEMPERATURE_SENSOR"),
)


class Ping(NamedTuple):
    """The ping reply: the drive's ids, model and firmware."""

    id: int
    can_id: int
    model_number: int
    firmware: tuple[int, int, int]
    drive_mode: int


class Status(NamedTuple):
    """The status reply."""

    operating_mode: int  # 6 is MIT
    external_encoder_bits: int  # 0 when no encoder is fitted at the output
    torque_on: bool
    bus_voltage_v: float
    fault: int  # fault_names()


class Span(NamedTuple):
    """One MIT mapping range; each end is set separately, so it need not be symmetric."""

    low: float
    high: float


class MitRanges(NamedTuple):
    """The five MIT field ranges.

    Each drive stores its own in registers 330-349 and they differ between
    drives (kp tops out at 12 on a GDK468, not the manual's 100), so they are
    read from the drive rather than assumed.
    """

    position: Span  # rad
    velocity: Span  # rad/s
    torque: Span  # Nm
    kp: Span  # Nm/rad
    kd: Span  # Nm*s/rad


class MitState(NamedTuple):
    """The reply to an MIT command or to torque on."""

    error: int  # nonzero while faulted; the manual gives the values no meaning
    position_rad: float  # wraps at the ends of the position range
    velocity_rad_s: float
    torque_nm: float
    mos_temperature_c: int
    rotor_temperature_c: int


def pack_ping() -> bytes:
    return bytes([_PING])


def pack_status_read() -> bytes:
    return bytes([_READ_STATUS])


def pack_torque_on() -> bytes:
    """FF x7 FC. The drive replies with an MIT state."""
    return bytes([0xFF] * 7 + [_TORQUE_ON])


def pack_torque_off() -> bytes:
    """FF x7 FD. The drive does not reply."""
    return bytes([0xFF] * 7 + [_TORQUE_OFF])


def pack_set_zero() -> bytes:
    """FF x7 FE: the present position becomes 0. Taken only with torque off; no reply."""
    return bytes([0xFF] * 7 + [_SET_ZERO])


def pack_register_read(address: int, length: int) -> bytes:
    """Three bytes: address, then how many bytes to read. The reply is those bytes."""
    return struct.pack("<HB", address, length)


def pack_register_write(address: int, value: int, length: int = 1) -> bytes:
    """Seven bytes: address, length, then the value padded with 0xFF."""
    if not 1 <= length <= _WRITE_LENGTH_MAX:
        raise ValueError(f"a write carries 1-{_WRITE_LENGTH_MAX} bytes, not {length}")
    padding = bytes([_WRITE_PAD] * (_WRITE_LENGTH_MAX - length))
    return struct.pack("<HB", address, length) + value.to_bytes(length, "little") + padding


def pack_operating_mode_write(mode: int = MIT_MODE) -> bytes:
    return pack_register_write(OPERATING_MODE_REGISTER, mode)


def pack_range_read(index: int) -> bytes:
    """Read MIT range ``index``, in MitRanges order."""
    return pack_register_read(_RANGE_REGISTERS[index][0], RANGE_LENGTH)


def _max_code(bits: int) -> int:
    return (1 << bits) - 1


def _float_to_uint(value: float, span: Span, bits: int) -> int:
    clamped = min(max(value, span.low), span.high)
    # Half away from zero, as the C++ std::lround; round() would round half to even.
    return math.floor((clamped - span.low) * _max_code(bits) / (span.high - span.low) + 0.5)


def _uint_to_float_centered(code: int, span: Span, bits: int) -> float:
    """The drive truncates when it encodes, so the middle of a code is the best
    estimate of what it measured; the top code holds the top of the range alone."""
    top = _max_code(bits)
    centered = code + 0.5 if code < top else float(top)
    return centered * (span.high - span.low) / top + span.low


def pack_mit_command(
    ranges: MitRanges,
    position_rad: float = 0.0,
    velocity_rad_s: float = 0.0,
    kp: float = 0.0,
    kd: float = 0.0,
    torque_nm: float = 0.0,
) -> bytes:
    """torque = kp * (position - measured) + kd * (velocity - measured) + torque_nm.

    Fields are clamped to their ranges.
    """
    position = _float_to_uint(position_rad, ranges.position, _POSITION_BITS)
    velocity = _float_to_uint(velocity_rad_s, ranges.velocity, _FIELD_BITS)
    kp_code = _float_to_uint(kp, ranges.kp, _FIELD_BITS)
    kd_code = _float_to_uint(kd, ranges.kd, _FIELD_BITS)
    torque = _float_to_uint(torque_nm, ranges.torque, _FIELD_BITS)
    return bytes(
        [
            position >> 8,
            position & 0xFF,
            velocity >> 4,
            ((velocity & 0x0F) << 4) | (kp_code >> 8),
            kp_code & 0xFF,
            kd_code >> 4,
            ((kd_code & 0x0F) << 4) | (torque >> 8),
            torque & 0xFF,
        ]
    )


def is_ping_reply(data: bytes, pinged_id: int, reply_id: int) -> bool:
    """Whether a frame from ``reply_id`` answers a ping sent to ``pinged_id``.

    Nothing else a drive sends names both its ids in its first two bytes, so
    this tells a ping reply from a status or an MIT state, which are 8 bytes too.
    """
    return len(data) == FRAME_LENGTH and data[0] == pinged_id and data[1] == reply_id


def unpack_ping(data: bytes) -> Ping:
    drive_id, can_id, model, major, minor, patch, drive_mode = struct.unpack("<BBHBBBB", data)
    return Ping(drive_id, can_id, model, (major, minor, patch), drive_mode)


def unpack_status(data: bytes) -> Status:
    mode, encoder_bits, torque, voltage_centivolts, fault = struct.unpack_from("<BBBHB", data)
    return Status(mode, encoder_bits, torque != 0, voltage_centivolts / 100.0, fault)


def unpack_present_position(data: bytes) -> float:
    """Register 552 in rad."""
    (counts,) = struct.unpack("<i", data)
    return counts * 2.0 * math.pi / PRESENT_POSITION_COUNTS_PER_TURN


def unpack_range(index: int, data: bytes) -> Span | None:
    """MIT range ``index`` from its register pair; None for an empty range.

    Either end order: firmware 3.0.1 stores the bottom first, the manual lists
    the top first. An empty range cannot scale an MIT field.
    """
    divisor = _RANGE_REGISTERS[index][1]
    first, second = struct.unpack_from("<hh", data)
    span = Span(min(first, second) / divisor, max(first, second) / divisor)
    return span if span.low < span.high else None


def mit_ranges(spans: Sequence[Span | None]) -> MitRanges | None:
    """The MitRanges from every range in order; None while any is unknown."""
    if len(spans) != RANGE_COUNT or any(span is None for span in spans):
        return None
    return MitRanges(*spans)


def unpack_mit_state(data: bytes, ranges: MitRanges) -> MitState:
    position = (data[1] << 8) | data[2]
    velocity = (data[3] << 4) | (data[4] >> 4)
    torque = ((data[4] & 0x0F) << 8) | data[5]
    return MitState(
        error=data[0] >> 4,
        position_rad=_uint_to_float_centered(position, ranges.position, _POSITION_BITS),
        velocity_rad_s=_uint_to_float_centered(velocity, ranges.velocity, _FIELD_BITS),
        torque_nm=_uint_to_float_centered(torque, ranges.torque, _FIELD_BITS),
        mos_temperature_c=data[6],
        rotor_temperature_c=data[7],
    )


def keeps_position(calibration: int) -> bool:
    """Whether register 403 lets the drive keep its position through a power cycle.

    Only a drive with both encoders connected and calibrated does, within +-pi.
    """
    return (calibration & _BOTH_ENCODERS_CALIBRATED) == _BOTH_ENCODERS_CALIBRATED


def write_accepted(data: bytes) -> bool:
    return len(data) == WRITE_REPLY_LENGTH and data[0] == _WRITE_DONE


def operating_mode_name(operating_mode: int) -> str:
    return _OPERATING_MODES.get(operating_mode, f"MODE({operating_mode})")


def fault_names(fault: int) -> str:
    if fault == 0:
        return "NONE"
    names = [name for bit, name in _FAULT_FLAGS if fault & bit]
    known_bits = 0
    for bit, _name in _FAULT_FLAGS:
        known_bits |= bit
    if fault & ~known_bits:
        names.append(f"UNKNOWN_BITS(0x{fault & ~known_bits:02X})")
    return "|".join(names)

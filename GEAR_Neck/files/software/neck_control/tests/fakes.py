# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A simulated GDK drive and CAN bus for the tests."""

from __future__ import annotations

import collections
import math
import struct
import threading

import can

from gear_neck import gdk_protocol as proto
from gear_neck.gdk_protocol import MitRanges, Span

MODEL_NUMBER = 31011
FIRMWARE = (3, 3, 0)
BOTH_ENCODERS_CALIBRATED = 0x3C
POSITION_MODE = 3
# The MIT ranges a GDK468 on firmware 3.0.1 reported.
GDK468_RANGES = MitRanges(
    position=Span(-12.5, 12.5),
    velocity=Span(-30.0, 30.0),
    torque=Span(-10.0, 10.0),
    kp=Span(0.0, 12.0),
    kd=Span(0.0, 5.0),
)
_RANGE_DIVISORS = (100, 100, 500, 100, 100)
_RANGE_ADDRESSES = (330, 334, 338, 342, 346)


def decode_field(code: int, span: Span, bits: int) -> float:
    return code * (span.high - span.low) / ((1 << bits) - 1) + span.low


def encode_field(value: float, span: Span, bits: int) -> int:
    """Truncating, as a drive encodes its MIT state."""
    clamped = min(max(value, span.low), span.high)
    return int((clamped - span.low) * ((1 << bits) - 1) / (span.high - span.low))


class FakeDrive:
    """A GDK drive's side of the protocol: ping, status, register reads and writes,
    torque on and off, set zero, and MIT commands. ``position_rad`` is in the drive's
    own direction. An MIT command moves it straight to the commanded position."""

    def __init__(
        self,
        drive_id: int = 1,
        can_id: int = 0,
        *,
        position_rad: float = 0.0,
        torque_on: bool = False,
        operating_mode: int = POSITION_MODE,
        calibration: int = BOTH_ENCODERS_CALIBRATED,
        fault: int = 0,
        ranges: MitRanges = GDK468_RANGES,
        honors_torque_off: bool = True,
        accepts_writes: bool = True,
        answers_write_from_new_can_id: bool = True,
    ) -> None:
        self.id = drive_id
        self.can_id = can_id
        self.position_rad = position_rad
        self.torque_on = torque_on
        self.operating_mode = operating_mode
        self.calibration = calibration
        self.fault = fault
        self.ranges = ranges
        self.honors_torque_off = honors_torque_off
        self.accepts_writes = accepts_writes
        self.answers_write_from_new_can_id = answers_write_from_new_can_id
        self.silent = False  # answers nothing at all, as if unplugged
        self.mit_error = 0  # error nibble of every MIT state
        self.writes: list[tuple[int, int]] = []
        # (position, velocity, kp, kd, torque) of every MIT command, decoded.
        self.mit_commands: list[tuple[float, float, float, float, float]] = []

    def handle(self, arbitration_id: int, data: bytes) -> list[tuple[int, bytes]]:
        """Replies as (arbitration id, data) to a frame on the bus."""
        if arbitration_id != self.id or self.silent:
            return []
        if data == proto.pack_ping():
            ping = struct.pack("<BBHBBBB", self.id, self.can_id, MODEL_NUMBER, *FIRMWARE, 0)
            return [(self.can_id, ping)]
        if data == proto.pack_status_read():
            # 24.00 V, 16-bit output encoder, 30 and 31 C.
            status = bytes(
                [self.operating_mode, 16, int(self.torque_on), 0x60, 0x09, self.fault, 30, 31]
            )
            return [(self.can_id, status)]
        if data == proto.pack_torque_off():
            if self.honors_torque_off:
                self.torque_on = False
            return []
        if data == proto.pack_torque_on():
            if self.operating_mode != proto.MIT_MODE or self.fault:
                return []
            self.torque_on = True
            return [(self.can_id, self.mit_state())]
        if data == proto.pack_set_zero():
            if not self.torque_on:
                self.position_rad = 0.0
            return []
        if len(data) == 3:
            return self._read(data)
        if len(data) == 7:
            return [self._write(data)]
        if len(data) == proto.FRAME_LENGTH:
            return self._mit_command(data)
        return []

    def mit_state(self) -> bytes:
        position = encode_field(self.position_rad, self.ranges.position, 16)
        velocity = encode_field(0.0, self.ranges.velocity, 12)
        torque = encode_field(0.0, self.ranges.torque, 12)
        return bytes(
            [
                (self.mit_error << 4) | (self.id & 0x0F),
                position >> 8,
                position & 0xFF,
                velocity >> 4,
                ((velocity & 0x0F) << 4) | (torque >> 8),
                torque & 0xFF,
                30,
                31,
            ]
        )

    def _read(self, data: bytes) -> list[tuple[int, bytes]]:
        address, length = struct.unpack("<HB", data)
        if address == proto.PRESENT_POSITION_REGISTER:
            counts = round(self.position_rad * proto.PRESENT_POSITION_COUNTS_PER_TURN / math.tau)
            value = struct.pack("<i", counts)
        elif address == proto.CALIBRATION_REGISTER:
            value = bytes([self.calibration])
        elif address in _RANGE_ADDRESSES:
            index = _RANGE_ADDRESSES.index(address)
            span, divisor = self.ranges[index], _RANGE_DIVISORS[index]
            # Bottom first, as firmware 3.0.1 stores it.
            value = struct.pack("<hh", round(span.low * divisor), round(span.high * divisor))
        else:
            return []
        return [(self.can_id, value[:length])]

    def _write(self, data: bytes) -> tuple[int, bytes]:
        address, _length = struct.unpack_from("<HB", data)
        value = data[3]
        old_can_id = self.can_id
        self.writes.append((address, value))
        accepted = self.accepts_writes and not self.torque_on
        if accepted and address == proto.ID_REGISTER:
            self.id = value
        elif accepted and address == proto.CAN_ID_REGISTER:
            self.can_id = value
        elif accepted and address == proto.OPERATING_MODE_REGISTER:
            self.operating_mode = value
        reply_id = self.can_id if self.answers_write_from_new_can_id else old_can_id
        return reply_id, bytes([1 if accepted else 0])

    def _mit_command(self, data: bytes) -> list[tuple[int, bytes]]:
        if not self.torque_on:
            return []
        position = (data[0] << 8) | data[1]
        velocity = (data[2] << 4) | (data[3] >> 4)
        kp = ((data[3] & 0x0F) << 8) | data[4]
        kd = (data[5] << 4) | (data[6] >> 4)
        torque = ((data[6] & 0x0F) << 8) | data[7]
        command = (
            decode_field(position, self.ranges.position, 16),
            decode_field(velocity, self.ranges.velocity, 12),
            decode_field(kp, self.ranges.kp, 12),
            decode_field(kd, self.ranges.kd, 12),
            decode_field(torque, self.ranges.torque, 12),
        )
        self.mit_commands.append(command)
        self.position_rad = command[0]
        return [(self.can_id, self.mit_state())]


class FakeBus:
    """Delivers each sent frame to every drive and queues their replies at once."""

    def __init__(self, *drives: FakeDrive) -> None:
        self.drives = drives
        self.sent: list[tuple[int, bytes]] = []
        self._rx: collections.deque[can.Message] = collections.deque()
        self._lock = threading.Lock()
        self.is_shutdown = False

    def send(self, msg: can.Message) -> None:
        frame = (msg.arbitration_id, bytes(msg.data))
        with self._lock:
            self.sent.append(frame)
            for drive in self.drives:
                for reply_id, data in drive.handle(*frame):
                    reply = can.Message(arbitration_id=reply_id, data=data, is_extended_id=False)
                    self._rx.append(reply)

    def recv(self, timeout: float | None = None) -> can.Message | None:
        with self._lock:
            return self._rx.popleft() if self._rx else None

    def shutdown(self) -> None:
        self.is_shutdown = True

    def sent_data(self) -> list[bytes]:
        with self._lock:
            return [data for _id, data in self.sent]

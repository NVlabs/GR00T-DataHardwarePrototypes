# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import unittest

from fakes import GDK468_RANGES

from gear_neck import gdk_protocol as proto
from gear_neck.gdk_protocol import Span


def torque_code(frame: bytes) -> int:
    return ((frame[6] & 0x0F) << 8) | frame[7]


class RequestTest(unittest.TestCase):
    def test_status_read_is_one_byte(self) -> None:
        self.assertEqual(proto.pack_status_read(), bytes([0x02]))

    def test_register_reads_are_the_little_endian_address_then_the_length(self) -> None:
        self.assertEqual(proto.pack_register_read(552, 4), bytes.fromhex("28 02 04"))
        self.assertEqual(proto.pack_register_read(403, 1), bytes.fromhex("93 01 01"))
        self.assertEqual(proto.pack_range_read(0), bytes.fromhex("4A 01 04"))  # 330
        self.assertEqual(proto.pack_range_read(2), bytes.fromhex("52 01 04"))  # 338
        self.assertEqual(proto.pack_range_read(4), bytes.fromhex("5A 01 04"))  # 346

    def test_register_write_pads_a_one_byte_value_with_ff(self) -> None:
        self.assertEqual(
            proto.pack_register_write(proto.ID_REGISTER, 50), bytes.fromhex("07 00 01 32 FF FF FF")
        )
        self.assertEqual(
            proto.pack_register_write(proto.CAN_ID_REGISTER, 51),
            bytes.fromhex("0E 00 01 33 FF FF FF"),
        )
        self.assertEqual(proto.pack_operating_mode_write(), bytes.fromhex("21 00 01 06 FF FF FF"))

    def test_system_frames_are_seven_ff_then_the_code(self) -> None:
        self.assertEqual(proto.pack_torque_on(), bytes.fromhex("FF FF FF FF FF FF FF FC"))
        self.assertEqual(proto.pack_torque_off(), bytes.fromhex("FF FF FF FF FF FF FF FD"))
        self.assertEqual(proto.pack_set_zero(), bytes.fromhex("FF FF FF FF FF FF FF FE"))


class MitCommandTest(unittest.TestCase):
    def test_packs_range_ends_high_bits_first(self) -> None:
        upper = proto.pack_mit_command(
            GDK468_RANGES, position_rad=12.5, velocity_rad_s=-30.0, kp=12.0, kd=0.0, torque_nm=10.0
        )
        self.assertEqual(upper, bytes.fromhex("FF FF 00 0F FF 00 0F FF"))
        lower = proto.pack_mit_command(
            GDK468_RANGES, position_rad=-12.5, velocity_rad_s=30.0, kp=0.0, kd=5.0, torque_nm=-10.0
        )
        self.assertEqual(lower, bytes.fromhex("00 00 FF F0 00 FF F0 00"))

    def test_rounds_mid_range_to_the_nearest_code(self) -> None:
        # 32767.5 and 2047.5 round up to 0x8000 and 0x800, not to even.
        frame = proto.pack_mit_command(GDK468_RANGES, kp=6.0, kd=2.5)
        self.assertEqual(frame, bytes.fromhex("80 00 80 08 00 80 08 00"))

    def test_maps_each_end_of_an_asymmetric_range(self) -> None:
        ranges = GDK468_RANGES._replace(torque=Span(-4.0, 6.0))
        self.assertEqual(torque_code(proto.pack_mit_command(ranges, torque_nm=-4.0)), 0)
        self.assertEqual(torque_code(proto.pack_mit_command(ranges, torque_nm=6.0)), 0xFFF)
        self.assertEqual(torque_code(proto.pack_mit_command(ranges, torque_nm=1.0)), 0x800)

    def test_clamps_out_of_range_fields(self) -> None:
        clamped = proto.pack_mit_command(
            GDK468_RANGES,
            position_rad=100.0,
            velocity_rad_s=-100.0,
            kp=100.0,
            kd=-1.0,
            torque_nm=50.0,
        )
        range_ends = proto.pack_mit_command(
            GDK468_RANGES, position_rad=12.5, velocity_rad_s=-30.0, kp=12.0, kd=0.0, torque_nm=10.0
        )
        self.assertEqual(clamped, range_ends)


class RangeTest(unittest.TestCase):
    def test_unpacks_a_range_stored_bottom_first(self) -> None:
        # The bytes a GDK468 on firmware 3.0.1 returned for registers 330-333.
        self.assertEqual(proto.unpack_range(0, bytes.fromhex("1E FB E2 04")), Span(-12.5, 12.5))

    def test_unpacks_a_range_stored_top_first_with_its_own_scale(self) -> None:
        # The manual's order: 5000 then -5000, in 0.002 Nm.
        self.assertEqual(proto.unpack_range(2, bytes.fromhex("88 13 78 EC")), Span(-10.0, 10.0))
        self.assertEqual(proto.unpack_range(3, bytes.fromhex("B0 04 00 00")), Span(0.0, 12.0))

    def test_empty_range_is_rejected(self) -> None:
        self.assertIsNone(proto.unpack_range(0, bytes(4)))

    def test_mit_ranges_need_every_range(self) -> None:
        spans = list(GDK468_RANGES)
        self.assertEqual(proto.mit_ranges(spans), GDK468_RANGES)
        self.assertIsNone(proto.mit_ranges([*spans[:4], None]))
        self.assertIsNone(proto.mit_ranges(spans[:4]))


class ReplyTest(unittest.TestCase):
    def test_unpacks_mit_state_fields(self) -> None:
        # Error 3, id 1, position code 40000, velocity 1000, torque 3000, 33 and 41 C.
        state = proto.unpack_mit_state(
            bytes([0x31, 0x9C, 0x40, 0x3E, 0x8B, 0xB8, 33, 41]), GDK468_RANGES
        )
        self.assertEqual(state.error, 3)
        # Each code decodes to its middle, as the drive truncates.
        self.assertAlmostEqual(state.position_rad, 40000.5 * 25.0 / 65535.0 - 12.5)
        self.assertAlmostEqual(state.velocity_rad_s, 1000.5 * 60.0 / 4095.0 - 30.0)
        self.assertAlmostEqual(state.torque_nm, 3000.5 * 20.0 / 4095.0 - 10.0)
        self.assertEqual((state.mos_temperature_c, state.rotor_temperature_c), (33, 41))

    def test_mit_state_top_code_is_the_top_of_the_range(self) -> None:
        state = proto.unpack_mit_state(bytes.fromhex("00 FF FF FF FF FF 00 00"), GDK468_RANGES)
        self.assertEqual(state.error, 0)
        self.assertEqual(
            (state.position_rad, state.velocity_rad_s, state.torque_nm), (12.5, 30.0, 10.0)
        )

    def test_mit_state_round_trips_a_command_within_one_code(self) -> None:
        command = proto.pack_mit_command(
            GDK468_RANGES, position_rad=1.234, velocity_rad_s=-3.21, torque_nm=2.5
        )
        # The state puts position and velocity one byte later, behind the error/id
        # byte, and has no gains: splice the command's torque bits in after the
        # velocity nibble.
        state_data = bytes(
            [
                0x01,
                command[0],
                command[1],
                command[2],
                (command[3] & 0xF0) | (command[6] & 0x0F),
                command[7],
                25,
                25,
            ]
        )
        state = proto.unpack_mit_state(state_data, GDK468_RANGES)
        self.assertAlmostEqual(state.position_rad, 1.234, delta=25.0 / 65535.0)
        self.assertAlmostEqual(state.velocity_rad_s, -3.21, delta=60.0 / 4095.0)
        self.assertAlmostEqual(state.torque_nm, 2.5, delta=20.0 / 4095.0)

    def test_unpacks_status(self) -> None:
        # MIT mode, 14-bit output encoder, torque on, 48.00 V, overload.
        status = proto.unpack_status(bytes([6, 14, 1, 0xC0, 0x12, 0x20, 35, 40]))
        self.assertEqual(status, proto.Status(proto.MIT_MODE, 14, True, 48.0, 0x20))

    def test_ping_reply_names_both_ids(self) -> None:
        data = bytes([50, 51, 0x34, 0x12, 3, 0, 1, 0])
        self.assertTrue(proto.is_ping_reply(data, 50, 51))
        self.assertFalse(proto.is_ping_reply(data, 50, 50))
        self.assertEqual(proto.unpack_ping(data), proto.Ping(50, 51, 0x1234, (3, 0, 1), 0))

    def test_present_position_is_signed_65536_counts_per_turn(self) -> None:
        # A GDK468 read -769 counts with its MIT position at -0.0736 rad.
        position_rad = proto.unpack_present_position(bytes.fromhex("FF FC FF FF"))
        self.assertAlmostEqual(position_rad, -769 * 2.0 * math.pi / 65536)
        self.assertAlmostEqual(position_rad, -0.0736, places=3)

    def test_write_reply_is_done_or_refused(self) -> None:
        self.assertTrue(proto.write_accepted(bytes([0x01])))
        self.assertFalse(proto.write_accepted(bytes([0x00])))

    def test_only_both_encoders_calibrated_keep_position(self) -> None:
        self.assertTrue(proto.keeps_position(60))
        # Both connected but uncalibrated; only the internal calibrated; one encoder.
        for calibration in (48, 52, 0x14):
            with self.subTest(calibration=calibration):
                self.assertFalse(proto.keeps_position(calibration))


class NamesTest(unittest.TestCase):
    def test_names_fault_bits(self) -> None:
        self.assertEqual(proto.fault_names(0x00), "NONE")
        self.assertEqual(proto.fault_names(0x41), "VOLTAGE|MOTOR_TEMPERATURE")
        self.assertEqual(proto.fault_names(0x22), "OVERLOAD|UNKNOWN_BITS(0x02)")

    def test_names_operating_modes(self) -> None:
        self.assertEqual(proto.operating_mode_name(6), "MIT")
        self.assertEqual(proto.operating_mode_name(3), "POSITION")
        self.assertEqual(proto.operating_mode_name(2), "MODE(2)")


if __name__ == "__main__":
    unittest.main()

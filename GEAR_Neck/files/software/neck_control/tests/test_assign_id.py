# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import io
import unittest

from fakes import FIRMWARE, MODEL_NUMBER, FakeBus, FakeDrive

from gear_neck import assign_id
from gear_neck import gdk_protocol as proto
from gear_neck.assign_id import AssignError
from gear_neck.gdk_link import GdkLink
from gear_neck.load_config import FACTORY_IDS, Ids, load_config

SCAN_FRAMES = proto.ID_MAX - proto.ID_MIN + 1
CONFIG = load_config()
NECK_IDS = CONFIG.neck_ids


def link_to(*drives: FakeDrive) -> tuple[GdkLink, FakeBus]:
    bus = FakeBus(*drives)
    return GdkLink(bus), bus


class ConfigTest(unittest.TestCase):
    def test_neck_ids_are_distinct_and_in_range(self) -> None:
        every_id = [value for joint in CONFIG.joints for value in joint.ids]
        self.assertEqual(len(set(every_id)), len(every_id))
        self.assertNotIn(FACTORY_IDS.id, every_id)
        for joint in CONFIG.joints:
            with self.subTest(joint=joint.name):
                assign_id.check_ids(joint.ids)


class ScanTest(unittest.TestCase):
    def test_finds_every_drive_by_pinging_only(self) -> None:
        link, bus = link_to(FakeDrive(1, 0), FakeDrive(50, 51))

        found = assign_id.scan(link)

        self.assertEqual([(ping.id, ping.can_id) for ping in found], [(1, 0), (50, 51)])
        self.assertEqual(found[1].firmware, FIRMWARE)
        self.assertEqual(len(bus.sent), SCAN_FRAMES)
        self.assertTrue(all(data == proto.pack_ping() for _id, data in bus.sent))

    def test_half_assigned_drive_is_still_found(self) -> None:
        link, _bus = link_to(FakeDrive(1, 51))

        self.assertEqual([(ping.id, ping.can_id) for ping in assign_id.scan(link)], [(1, 51)])

    def test_describes_the_joint_status_and_calibration(self) -> None:
        link, _bus = link_to(FakeDrive(50, 51), FakeDrive(52, 53, calibration=0x30))
        yaw, roll = assign_id.scan(link)

        yaw_line = assign_id.describe(link, yaw, NECK_IDS)
        self.assertIn("(yaw)", yaw_line)
        self.assertIn("24.00 V", yaw_line)
        self.assertIn("encoders calibrated", yaw_line)
        self.assertIn(
            "register 403 = 0x30: encoders NOT calibrated",
            assign_id.describe(link, roll, NECK_IDS),
        )


class ChooseDriveTest(unittest.TestCase):
    factory = proto.Ping(1, 0, MODEL_NUMBER, FIRMWARE, 0)
    yaw = proto.Ping(50, 51, MODEL_NUMBER, FIRMWARE, 0)

    def test_takes_the_only_drive(self) -> None:
        self.assertEqual(assign_id.choose_drive([self.factory], Ids(52, 53), None), self.factory)

    def test_needs_from_when_several_answer(self) -> None:
        with self.assertRaisesRegex(AssignError, "--from"):
            assign_id.choose_drive([self.factory, self.yaw], Ids(52, 53), None)
        self.assertEqual(
            assign_id.choose_drive([self.factory, self.yaw], Ids(52, 53), 1), self.factory
        )

    def test_from_must_answer(self) -> None:
        with self.assertRaisesRegex(AssignError, "ID 7"):
            assign_id.choose_drive([self.factory], Ids(52, 53), 7)

    def test_refuses_ids_another_drive_uses(self) -> None:
        for target in (Ids(50, 53), Ids(52, 51), Ids(49, 50)):
            with self.subTest(target=target), self.assertRaisesRegex(AssignError, "already uses"):
                assign_id.choose_drive([self.factory, self.yaw], target, 1)

    def test_moving_a_drive_onto_its_own_ids_is_no_conflict(self) -> None:
        self.assertEqual(assign_id.choose_drive([self.yaw], Ids(51, 52), None), self.yaw)

    def test_refuses_a_shared_or_out_of_range_id(self) -> None:
        for target in (Ids(50, 50), Ids(0, 1), Ids(50, 253)):
            with self.subTest(target=target), self.assertRaises(AssignError):
                assign_id.choose_drive([self.factory], target, None)


class AssignTest(unittest.TestCase):
    def test_moves_a_factory_drive_to_the_yaw_ids(self) -> None:
        drive = FakeDrive(1, 0)
        link, bus = link_to(drive)

        ping = assign_id.assign(link, assign_id.scan(link)[0], NECK_IDS["yaw"])

        self.assertEqual((ping.id, ping.can_id), (50, 51))
        self.assertEqual((drive.id, drive.can_id), (50, 51))
        self.assertEqual(drive.writes, [(proto.CAN_ID_REGISTER, 51), (proto.ID_REGISTER, 50)])
        self.assertEqual(
            bus.sent[SCAN_FRAMES:],
            [
                (1, proto.pack_torque_off()),
                (1, proto.pack_status_read()),
                (1, proto.pack_register_write(proto.CAN_ID_REGISTER, 51)),
                (1, proto.pack_register_write(proto.ID_REGISTER, 50)),
                (50, proto.pack_ping()),
            ],
        )

    def test_switches_off_torque_before_writing(self) -> None:
        drive = FakeDrive(1, 0, torque_on=True)
        link, _bus = link_to(drive)

        assign_id.assign(link, assign_id.scan(link)[0], Ids(52, 53))

        self.assertFalse(drive.torque_on)
        self.assertEqual((drive.id, drive.can_id), (52, 53))

    def test_writes_nothing_while_torque_stays_on(self) -> None:
        drive = FakeDrive(1, 0, torque_on=True, honors_torque_off=False)
        link, _bus = link_to(drive)

        with self.assertRaisesRegex(AssignError, "torque on"):
            assign_id.assign(link, assign_id.scan(link)[0], Ids(52, 53))
        self.assertEqual(drive.writes, [])

    def test_stops_at_a_refused_write(self) -> None:
        drive = FakeDrive(1, 0, accepts_writes=False)
        link, _bus = link_to(drive)

        with self.assertRaisesRegex(AssignError, "refused the CAN_ID write"):
            assign_id.assign(link, assign_id.scan(link)[0], Ids(52, 53))
        self.assertEqual(drive.writes, [(proto.CAN_ID_REGISTER, 53)])

    def test_takes_the_can_id_write_reply_from_the_old_can_id(self) -> None:
        drive = FakeDrive(1, 0, answers_write_from_new_can_id=False)
        link, _bus = link_to(drive)

        ping = assign_id.assign(link, assign_id.scan(link)[0], Ids(54, 55))

        self.assertEqual((ping.id, ping.can_id), (54, 55))


class CliTest(unittest.TestCase):
    def test_joint_selects_the_neck_ids(self) -> None:
        args = assign_id._parse_args(["assign", "--joint", "pitch", "--channel", "can0"])
        self.assertEqual((args.joint, args.channel, args.from_id), ("pitch", "can0", None))
        # Left to the settings file.
        self.assertEqual((args.interface, args.config), (None, None))

    def test_config_selects_another_settings_file(self) -> None:
        args = assign_id._parse_args(["scan", "--config", "other_neck.yaml"])
        self.assertEqual(str(args.config), "other_neck.yaml")

    def test_slcan_interface(self) -> None:
        args = assign_id._parse_args(["scan", "--interface", "slcan"])
        self.assertEqual((args.interface, args.channel), ("slcan", None))

    def test_id_needs_can_id(self) -> None:
        for argv in (["assign", "--id", "60"], ["assign", "--joint", "yaw", "--can-id", "61"]):
            with (
                self.subTest(argv=argv),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                assign_id._parse_args(argv)


if __name__ == "__main__":
    unittest.main()

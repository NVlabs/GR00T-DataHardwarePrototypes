# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from fakes import BOTH_ENCODERS_CALIBRATED, FakeBus, FakeDrive

from gear_neck import gdk_protocol as proto
from gear_neck import zeroing
from gear_neck.gdk_link import GdkLink
from gear_neck.load_config import Joint, load_config
from gear_neck.zeroing import NotZeroable

JOINTS_BY_NAME = load_config().joints_by_name
YAW = JOINTS_BY_NAME["yaw"]
ROLL = JOINTS_BY_NAME["roll"]
PITCH = JOINTS_BY_NAME["pitch"]
POSITION_READ = proto.pack_register_read(proto.PRESENT_POSITION_REGISTER, 4)
CALIBRATION_READ = proto.pack_register_read(proto.CALIBRATION_REGISTER, 1)


def drive_for(joint: Joint, **kwargs: object) -> FakeDrive:
    return FakeDrive(*joint.ids, **kwargs)


def neck_drives() -> tuple[FakeDrive, FakeDrive, FakeDrive]:
    """Drive positions in each drive's own direction."""
    return (
        drive_for(YAW, position_rad=-0.86),
        drive_for(ROLL, position_rad=0.94),
        drive_for(PITCH, position_rad=-2.89),
    )


class PlanZeroTest(unittest.TestCase):
    def test_captures_position_using_reads_only(self) -> None:
        bus = FakeBus(drive_for(PITCH, position_rad=-2.89))

        plan = zeroing.plan_zero(GdkLink(bus), PITCH)

        self.assertAlmostEqual(plan.position_rad, -2.89, places=4)
        self.assertEqual(plan.calibration, BOTH_ENCODERS_CALIBRATED)
        self.assertEqual(
            bus.sent,
            [
                (PITCH.ids.id, proto.pack_status_read()),
                (PITCH.ids.id, CALIBRATION_READ),
                (PITCH.ids.id, POSITION_READ),
            ],
        )

    def test_reports_a_reversed_drive_in_its_joint_direction(self) -> None:
        self.assertTrue(YAW.reversed)
        bus = FakeBus(drive_for(YAW, position_rad=-0.86))

        self.assertAlmostEqual(zeroing.plan_zero(GdkLink(bus), YAW).position_rad, 0.86, places=4)

    def test_drive_with_torque_on_is_not_zeroable(self) -> None:
        bus = FakeBus(drive_for(YAW, torque_on=True))

        with self.assertRaisesRegex(NotZeroable, "torque on"):
            zeroing.plan_zero(GdkLink(bus), YAW)
        self.assertEqual(bus.sent, [(YAW.ids.id, proto.pack_status_read())])

    def test_silent_drive_times_out(self) -> None:
        with self.assertRaises(TimeoutError):
            zeroing.plan_zero(GdkLink(FakeBus()), YAW)


class ApplyZeroTest(unittest.TestCase):
    def test_zeroes_then_returns_the_position_read_back(self) -> None:
        bus = FakeBus(drive_for(ROLL, position_rad=0.94))
        link = GdkLink(bus)
        plan = zeroing.plan_zero(link, ROLL)

        with mock.patch.object(zeroing, "ZERO_SETTLE_S", 0.0):
            position_rad = zeroing.apply_zero(link, plan)

        self.assertEqual(position_rad, 0.0)
        self.assertEqual(
            bus.sent[-2:], [(ROLL.ids.id, proto.pack_set_zero()), (ROLL.ids.id, POSITION_READ)]
        )


class PlanDrivesTest(unittest.TestCase):
    def test_skips_silent_drives_and_drives_with_torque_on(self) -> None:
        yaw, roll, _pitch = neck_drives()
        roll.torque_on = True
        out = io.StringIO()

        with contextlib.redirect_stdout(out):
            plans = zeroing._plan_drives(GdkLink(FakeBus(yaw, roll)), [YAW, ROLL, PITCH])

        self.assertEqual([plan.joint.name for plan in plans], ["yaw"])
        self.assertIn("roll (ID 52): torque on", out.getvalue())
        self.assertIn("pitch (ID 54): no reply from CAN_ID 55", out.getvalue())

    def test_warns_when_the_zero_would_not_survive_power_off(self) -> None:
        yaw = drive_for(YAW, calibration=0x30)
        out = io.StringIO()

        with contextlib.redirect_stdout(out):
            zeroing._print_plans(zeroing._plan_drives(GdkLink(FakeBus(yaw)), [YAW]))

        self.assertIn("register 403 reads 0x30", out.getvalue())


class MainTest(unittest.TestCase):
    def run_main(self, bus: FakeBus, answers: list[str], argv: list[str]) -> int:
        with (
            mock.patch.object(zeroing, "open_bus", return_value=bus),
            mock.patch("builtins.input", side_effect=answers),
            mock.patch.object(zeroing, "ZERO_SETTLE_S", 0.0),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return zeroing.main(argv)

    def test_declining_writes_nothing(self) -> None:
        drives = neck_drives()
        bus = FakeBus(*drives)

        self.assertEqual(self.run_main(bus, ["", "n"], []), 0)

        self.assertNotIn(proto.pack_set_zero(), bus.sent_data())
        self.assertAlmostEqual(drives[2].position_rad, -2.89)
        self.assertTrue(bus.is_shutdown)

    def test_confirming_zeroes_every_neck_drive(self) -> None:
        drives = neck_drives()
        bus = FakeBus(*drives)

        self.assertEqual(self.run_main(bus, ["", "y"], []), 0)

        zeroed = [drive_id for drive_id, data in bus.sent if data == proto.pack_set_zero()]
        self.assertEqual(zeroed, [50, 52, 54])
        self.assertEqual([drive.position_rad for drive in drives], [0.0, 0.0, 0.0])

    def test_joints_limits_the_zero_to_those_drives(self) -> None:
        drives = neck_drives()
        bus = FakeBus(*drives)

        self.assertEqual(self.run_main(bus, ["", "y"], ["--joints", "pitch"]), 0)

        self.assertEqual([drive.position_rad for drive in drives], [-0.86, 0.94, 0.0])

    def test_nothing_to_zero_fails(self) -> None:
        self.assertEqual(self.run_main(FakeBus(), [""], []), 1)


class CliTest(unittest.TestCase):
    def test_joints_default_to_the_whole_neck(self) -> None:
        args = zeroing._parse_args([])
        self.assertEqual(args.joints, ["yaw", "roll", "pitch"])
        self.assertEqual((args.interface, args.channel, args.config), (None, None, None))


if __name__ == "__main__":
    unittest.main()

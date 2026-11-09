# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import dataclasses
import io
import itertools
import math
import time
import unittest
from collections.abc import Callable
from unittest import mock

import can
from fakes import GDK468_RANGES, FakeBus, FakeDrive

from gear_neck import cli
from gear_neck import gdk_protocol as proto
from gear_neck.load_config import load_config
from gear_neck.neck import Angles, Neck, NeckError, clamp_deg, ramp

JOINTS_BY_NAME = load_config().joints_by_name
YAW = JOINTS_BY_NAME["yaw"]
ROLL = JOINTS_BY_NAME["roll"]
PITCH = JOINTS_BY_NAME["pitch"]
KP_STEP = GDK468_RANGES.kp.high / 4095
KD_STEP = GDK468_RANGES.kd.high / 4095


def wait_until(condition: Callable[[], bool], timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.002)
    return condition()


class NeckDrives:
    """The three neck drives on one fake bus; positions in each drive's own direction."""

    def __init__(self, yaw: float = 0.0, roll: float = 0.0, pitch: float = 0.0, **kwargs) -> None:
        self.yaw = FakeDrive(*YAW.ids, position_rad=yaw, **kwargs)
        self.roll = FakeDrive(*ROLL.ids, position_rad=roll, **kwargs)
        self.pitch = FakeDrive(*PITCH.ids, position_rad=pitch, **kwargs)
        self.all = (self.yaw, self.roll, self.pitch)
        self.bus = FakeBus(*self.all)

    def open(self) -> Neck:
        return Neck(bus=self.bus, rate_hz=1000.0, max_speed_deg_s=2000.0)


class HelperTest(unittest.TestCase):
    def test_ramp_steps_toward_the_target_and_lands_on_it(self) -> None:
        self.assertEqual(ramp(0.0, 1.0, 0.25), 0.25)
        self.assertEqual(ramp(0.0, -1.0, 0.25), -0.25)
        self.assertEqual(ramp(0.1, 0.3, 0.25), 0.3)

    def test_clamp_to_the_joint_limits(self) -> None:
        self.assertEqual(clamp_deg(PITCH, 120.0), PITCH.max_deg)
        self.assertEqual(clamp_deg(PITCH, -120.0), PITCH.min_deg)
        self.assertEqual(clamp_deg(PITCH, 10.0), 10.0)


class OpenTest(unittest.TestCase):
    def test_reads_angles_in_the_joint_directions(self) -> None:
        drives = NeckDrives(yaw=-0.5, roll=0.25, pitch=0.75)

        with drives.open() as neck:
            angles = neck.angles()

        # The yaw drive is mounted reversed.
        self.assertAlmostEqual(angles.yaw, math.degrees(0.5), places=2)
        self.assertAlmostEqual(angles.roll, math.degrees(0.25), places=2)
        self.assertAlmostEqual(angles.pitch, math.degrees(0.75), places=2)

    def test_takes_another_settings_file(self) -> None:
        settings = load_config()
        yaw_not_reversed = dataclasses.replace(settings.joints_by_name["yaw"], reversed=False)
        joints = tuple(yaw_not_reversed if j.name == "yaw" else j for j in settings.joints)
        drives = NeckDrives(yaw=-0.5)

        config = dataclasses.replace(settings, joints=joints)
        with Neck(bus=drives.bus, config=config, rate_hz=1000.0) as neck:
            self.assertAlmostEqual(neck.angles().yaw, math.degrees(-0.5), places=2)

    def test_reports_each_drive(self) -> None:
        drives = NeckDrives()
        drives.roll.calibration = 0x30

        with drives.open() as neck:
            info = {drive.joint: drive for drive in neck.drives}

        self.assertEqual(info["yaw"].ids, YAW.ids)
        self.assertEqual(info["yaw"].ranges, GDK468_RANGES)
        self.assertEqual(info["yaw"].bus_voltage_v, 24.0)
        self.assertTrue(info["yaw"].keeps_zero)
        self.assertFalse(info["roll"].keeps_zero)

    def test_switches_off_torque_left_on(self) -> None:
        drives = NeckDrives(torque_on=True, operating_mode=proto.MIT_MODE)

        with drives.open():
            self.assertFalse(any(drive.torque_on for drive in drives.all))

    def test_names_a_missing_drive(self) -> None:
        drives = NeckDrives()
        drives.pitch.silent = True

        with self.assertRaisesRegex(NeckError, "pitch: no reply from ID 54"):
            drives.open()

    def test_follows_the_neck_turned_by_hand(self) -> None:
        drives = NeckDrives()
        with drives.open() as neck:
            drives.pitch.position_rad = 0.5

            self.assertTrue(wait_until(lambda: neck.angles().pitch > 28.0))

    def test_lists_drives_that_stop_answering(self) -> None:
        drives = NeckDrives()
        with drives.open() as neck:
            drives.roll.silent = True

            self.assertTrue(wait_until(lambda: neck.silent_joints() == ["roll"]))


class EnableTest(unittest.TestCase):
    def test_puts_the_drives_in_mit_mode_and_holds_where_they_are(self) -> None:
        drives = NeckDrives(yaw=0.3, roll=-0.2, pitch=0.1)
        with drives.open() as neck:
            neck.enable()

            self.assertTrue(neck.enabled)
            for drive in drives.all:
                self.assertEqual(drive.operating_mode, proto.MIT_MODE)
                self.assertTrue(drive.torque_on)
            first_yaw_command = drives.yaw.mit_commands[0]
            self.assertAlmostEqual(first_yaw_command[0], 0.3, delta=1e-3)
            self.assertAlmostEqual(first_yaw_command[2], YAW.kp, delta=KP_STEP)
            self.assertAlmostEqual(first_yaw_command[3], YAW.kd, delta=KD_STEP)
            self.assertAlmostEqual(neck.target().yaw, math.degrees(-0.3), places=1)

    def test_refuses_a_faulted_drive_and_leaves_torque_off(self) -> None:
        drives = NeckDrives()
        drives.roll.fault = 0x20
        with drives.open() as neck:
            with self.assertRaisesRegex(NeckError, "roll: drive fault OVERLOAD"):
                neck.enable()

            self.assertFalse(neck.enabled)
            self.assertFalse(any(drive.torque_on for drive in drives.all))

    def test_commands_need_torque_on(self) -> None:
        with NeckDrives().open() as neck, self.assertRaisesRegex(NeckError, "enable"):
            neck.set_target(yaw=10.0)


class MoveTest(unittest.TestCase):
    def test_moves_to_the_angles_in_the_joint_directions(self) -> None:
        drives = NeckDrives()
        with drives.open() as neck:
            neck.enable()

            reached = neck.move_to(yaw=20.0, pitch=10.0, roll=-5.0, settle_s=0.0, timeout_s=2.0)

        self.assertAlmostEqual(reached.yaw, 20.0, delta=0.05)
        self.assertAlmostEqual(reached.pitch, 10.0, delta=0.05)
        self.assertAlmostEqual(reached.roll, -5.0, delta=0.05)
        # The reversed yaw drive was sent the negated angle.
        self.assertAlmostEqual(drives.yaw.position_rad, math.radians(-20.0), delta=1e-3)
        self.assertAlmostEqual(drives.pitch.position_rad, math.radians(10.0), delta=1e-3)

    def test_clamps_to_the_limits(self) -> None:
        drives = NeckDrives()
        with drives.open() as neck:
            neck.enable()

            target = neck.set_target(pitch=200.0, roll=-200.0)

        self.assertEqual((target.pitch, target.roll), (PITCH.max_deg, ROLL.min_deg))

    def test_ramps_rather_than_jumps(self) -> None:
        drives = NeckDrives()
        with Neck(bus=drives.bus, rate_hz=1000.0, max_speed_deg_s=100.0) as neck:
            neck.enable()
            neck.move_to(roll=10.0, settle_s=0.0, timeout_s=2.0)

        roll_positions = [command[0] for command in drives.roll.mit_commands]
        steps = [abs(b - a) for a, b in itertools.pairwise(roll_positions)]
        # 100 deg/s at 1 kHz is 0.1 deg a tick, plus one position code.
        self.assertLess(max(steps), math.radians(0.1) + 25.0 / 65535 * 2)
        self.assertGreater(len(roll_positions), 50)

    def test_a_joint_left_out_keeps_its_target(self) -> None:
        drives = NeckDrives(pitch=0.2)
        with drives.open() as neck:
            neck.enable()

            target = neck.set_target(yaw=5.0)

        self.assertEqual(target.yaw, 5.0)
        self.assertAlmostEqual(target.pitch, math.degrees(0.2), places=1)


class FaultTest(unittest.TestCase):
    def test_a_drive_that_stops_answering_switches_torque_off(self) -> None:
        drives = NeckDrives()
        with drives.open() as neck:
            neck.enable()
            drives.pitch.silent = True

            self.assertTrue(wait_until(lambda: not neck.enabled))
            self.assertFalse(drives.yaw.torque_on)
            with self.assertRaisesRegex(NeckError, "pitch: the drive stopped answering"):
                neck.set_target(yaw=0.0)

    def test_a_bus_error_with_torque_on_switches_torque_off(self) -> None:
        drives = NeckDrives()
        with drives.open() as neck:
            neck.enable()
            with mock.patch.object(drives.bus, "send", side_effect=can.CanOperationError("full")):
                self.assertTrue(wait_until(lambda: not neck.enabled))
            with self.assertRaisesRegex(NeckError, "CAN bus error: full"):
                neck.set_target(yaw=0.0)
            # The torque off at the fault failed too; it goes out again once the bus is back.
            self.assertTrue(wait_until(lambda: not any(drive.torque_on for drive in drives.all)))

    def test_a_bus_error_with_torque_off_costs_only_readings(self) -> None:
        drives = NeckDrives()
        with drives.open() as neck:
            with mock.patch.object(drives.bus, "send", side_effect=can.CanOperationError("full")):
                self.assertTrue(wait_until(lambda: len(neck.silent_joints()) == 3))
            self.assertTrue(wait_until(lambda: neck.silent_joints() == []))
            neck.enable()
            self.assertTrue(neck.enabled)

    def test_an_mit_error_switches_torque_off(self) -> None:
        drives = NeckDrives()
        with drives.open() as neck:
            neck.enable()
            drives.roll.mit_error = 2

            self.assertTrue(wait_until(lambda: not neck.enabled))
            self.assertFalse(any(drive.torque_on for drive in drives.all))
            with self.assertRaisesRegex(NeckError, "roll: drive reported MIT error 2"):
                neck.move_to(roll=0.0)


class CloseTest(unittest.TestCase):
    def test_leaving_the_block_switches_torque_off(self) -> None:
        drives = NeckDrives()
        with drives.open() as neck:
            neck.enable()

        self.assertFalse(any(drive.torque_on for drive in drives.all))
        self.assertFalse(neck.enabled)
        with self.assertRaisesRegex(NeckError, "closed"):
            neck.angles()

    def test_disable_goes_limp_and_keeps_reading(self) -> None:
        drives = NeckDrives()
        with drives.open() as neck:
            neck.enable()
            neck.disable()

            self.assertFalse(any(drive.torque_on for drive in drives.all))
            drives.yaw.position_rad = -0.4
            self.assertTrue(wait_until(lambda: neck.angles().yaw > 22.0))


class CliTest(unittest.TestCase):
    def test_move_turns_the_neck_and_homes_it(self) -> None:
        drives = NeckDrives()
        out = io.StringIO()
        with (
            mock.patch.object(cli, "Neck", lambda *args, **kwargs: drives.open()),
            mock.patch("builtins.input", return_value=""),
            contextlib.redirect_stdout(out),
        ):
            self.assertEqual(cli.move_main(["--yaw", "15", "--pitch", "5"]), 0)

        self.assertIn("target  yaw  +15.0  pitch   +5.0", out.getvalue())
        self.assertAlmostEqual(drives.yaw.position_rad, 0.0, delta=1e-3)
        self.assertFalse(any(drive.torque_on for drive in drives.all))

    def test_angles_prints_until_ctrl_c(self) -> None:
        drives = NeckDrives(pitch=0.5)
        out = io.StringIO()
        interrupting_time = mock.Mock(sleep=mock.Mock(side_effect=KeyboardInterrupt))
        with (
            mock.patch.object(cli, "Neck", lambda *args, **kwargs: drives.open()),
            mock.patch.object(cli, "time", interrupting_time),
            contextlib.redirect_stdout(out),
        ):
            self.assertEqual(cli.angles_main([]), 0)

        self.assertIn("pitch  +28.6", out.getvalue())
        self.assertIn("ID 50 / CAN_ID 51", out.getvalue())

    def test_angles_str(self) -> None:
        self.assertEqual(str(Angles(1.0, -2.0, 3.0)), "yaw   +1.0  pitch   -2.0  roll   +3.0 deg")


if __name__ == "__main__":
    unittest.main()

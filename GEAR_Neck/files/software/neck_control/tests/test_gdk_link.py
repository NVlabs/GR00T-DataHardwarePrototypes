# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import errno
import types
import unittest
from unittest import mock

import can
from fakes import FakeBus, FakeDrive

from gear_neck import gdk_link
from gear_neck import gdk_protocol as proto
from gear_neck.gdk_link import BusUnavailable, GdkLink
from gear_neck.load_config import Ids, load_config

POSITION_READ = proto.pack_register_read(proto.PRESENT_POSITION_REGISTER, 4)
CAN = load_config().can


def port(device: str, vid: int | None, pid: int | None) -> types.SimpleNamespace:
    return types.SimpleNamespace(device=device, vid=vid, pid=pid)


class FindAdapterTest(unittest.TestCase):
    def test_finds_the_adapter_by_its_usb_id(self) -> None:
        ports = [port("/dev/ttyACM0", 0x0D28, 0x0204), port("/dev/ttyACM1", *CAN.adapter_usb_id)]
        with mock.patch.object(gdk_link.list_ports, "comports", return_value=ports):
            self.assertEqual(gdk_link.find_adapter_port(CAN.adapter_usb_id), "/dev/ttyACM1")

    def test_none_without_the_adapter(self) -> None:
        ports = [port("/dev/ttyACM0", 0x0D28, 0x0204), port("/dev/ttyS0", None, None)]
        with mock.patch.object(gdk_link.list_ports, "comports", return_value=ports):
            self.assertIsNone(gdk_link.find_adapter_port(CAN.adapter_usb_id))


class OpenBusTest(unittest.TestCase):
    def test_slcan_without_the_adapter_says_what_to_check(self) -> None:
        with (
            mock.patch.object(gdk_link, "find_adapter_port", return_value=None),
            self.assertRaisesRegex(BusUnavailable, "16d0:117e"),
        ):
            gdk_link.open_bus(CAN, "slcan")

    def test_slcan_opens_the_found_port_at_the_neck_bitrate(self) -> None:
        with (
            mock.patch.object(gdk_link, "find_adapter_port", return_value="/dev/ttyACM1"),
            mock.patch.object(gdk_link.can, "Bus") as bus,
        ):
            gdk_link.open_bus(CAN, "slcan")
        bus.assert_called_once_with(interface="slcan", channel="/dev/ttyACM1", bitrate=1_000_000)

    def test_uses_the_interface_and_port_of_the_settings(self) -> None:
        settings = dataclasses.replace(CAN, interface="slcan", slcan_port="/dev/ttyUSB7")
        with (
            mock.patch.object(gdk_link, "find_adapter_port") as find,
            mock.patch.object(gdk_link.can, "Bus") as bus,
        ):
            gdk_link.open_bus(settings)
        bus.assert_called_once_with(interface="slcan", channel="/dev/ttyUSB7", bitrate=1_000_000)
        find.assert_not_called()

    def test_arguments_override_the_settings(self) -> None:
        settings = dataclasses.replace(CAN, interface="slcan", slcan_port="/dev/ttyUSB7")
        with mock.patch.object(gdk_link.can, "Bus") as bus:
            gdk_link.open_bus(settings, "socketcan", "vcan0")
        bus.assert_called_once_with(interface="socketcan", channel="vcan0")

    def test_a_refused_serial_port_points_at_the_dialout_group(self) -> None:
        # python-can wraps pyserial's error, which carries the errno.
        error = can.CanInitializationError("could not open port /dev/ttyACM1")
        error.__cause__ = OSError(errno.EACCES, "Permission denied")
        with (
            mock.patch.object(gdk_link, "find_adapter_port", return_value="/dev/ttyACM1"),
            mock.patch.object(gdk_link.can, "Bus", side_effect=error),
            self.assertRaisesRegex(BusUnavailable, "dialout"),
        ):
            gdk_link.open_bus(CAN, "slcan")

    def test_an_absent_serial_port_points_at_the_adapter(self) -> None:
        error = can.CanInitializationError("could not open port /dev/ttyACM1")
        error.__cause__ = OSError(errno.ENOENT, "No such file or directory")
        with (
            mock.patch.object(gdk_link.can, "Bus", side_effect=error),
            self.assertRaisesRegex(BusUnavailable, "plugged in"),
        ):
            gdk_link.open_bus(CAN, "slcan", "/dev/ttyACM1")

    def test_a_missing_socketcan_interface_points_at_the_setup_script(self) -> None:
        with (
            mock.patch.object(gdk_link.can, "Bus", side_effect=OSError(19, "No such device")),
            self.assertRaisesRegex(BusUnavailable, "can_neck .*setup_can.sh"),
        ):
            gdk_link.open_bus(CAN)

    def test_rejects_an_unknown_interface(self) -> None:
        with self.assertRaises(ValueError):
            gdk_link.open_bus(CAN, "pcan")


class ExchangeTest(unittest.TestCase):
    def test_collects_one_reply_per_drive_and_leaves_out_the_silent(self) -> None:
        yaw, roll = FakeDrive(50, 51, position_rad=0.5), FakeDrive(52, 53)
        roll.silent = True
        bus = FakeBus(yaw, roll)

        replies = GdkLink(bus).exchange(
            [(Ids(50, 51), POSITION_READ), (Ids(52, 53), POSITION_READ)], 4, 0.01
        )

        self.assertEqual(list(replies), [51])
        self.assertAlmostEqual(proto.unpack_present_position(replies[51]), 0.5, places=4)

    def test_skips_frames_of_the_wrong_length_or_kind(self) -> None:
        bus = FakeBus()
        noise = [
            can.Message(arbitration_id=51, data=bytes(8), is_extended_id=False),
            can.Message(arbitration_id=51, data=bytes(4), is_extended_id=True),
            can.Message(arbitration_id=51, is_error_frame=True, is_extended_id=False),
            can.Message(arbitration_id=51, data=bytes([1, 2, 3, 4]), is_extended_id=False),
        ]
        link = GdkLink(bus)
        with mock.patch.object(link, "_drain_rx"):
            bus._rx.extend(noise)
            replies = link.exchange([(Ids(50, 51), POSITION_READ)], 4, 0.01)

        self.assertEqual(replies, {51: bytes([1, 2, 3, 4])})


if __name__ == "__main__":
    unittest.main()

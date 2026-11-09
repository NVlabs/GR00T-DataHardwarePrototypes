# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line tools on top of Neck.

neck_angles                          print yaw, pitch and roll; torque stays off
neck_move --yaw 20 --pitch 10        turn the head there, then home on Enter
"""

from __future__ import annotations

import argparse
import sys
import time

from . import gdk_protocol as proto
from .gdk_link import BusUnavailable, add_bus_arguments
from .load_config import ConfigError, add_config_argument
from .neck import Neck, NeckError


def _open(args: argparse.Namespace) -> Neck | None:
    try:
        return Neck(args.interface, args.channel, config=args.config)
    except (ConfigError, BusUnavailable, NeckError) as exc:
        print(f"error: {exc}")
        return None


def _print_drives(neck: Neck) -> None:
    for info in neck.drives:
        line = (
            f"{info.joint:5s}  ID {info.ids.id} / CAN_ID {info.ids.can_id}  "
            f"{info.bus_voltage_v:.1f} V  mode {proto.operating_mode_name(info.operating_mode)}"
        )
        if info.fault:
            line += f"  FAULT {proto.fault_names(info.fault)}"
        if not info.keeps_zero:
            line += f"  register 403 = 0x{info.calibration:02X}: zero lost at power off"
        print(line)


def angles_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print the GEAR Neck's yaw, pitch and roll in degrees; torque stays off"
    )
    parser.add_argument("--rate", type=float, default=10.0, help="lines per second")
    add_config_argument(parser)
    add_bus_arguments(parser)
    args = parser.parse_args(argv)
    neck = _open(args)
    if neck is None:
        return 1
    with neck:
        _print_drives(neck)
        print("Turn the head by hand to check each joint's direction; Ctrl-C to stop.")
        try:
            while True:
                silent = neck.silent_joints()
                note = f"  no reply from {', '.join(silent)}" if silent else ""
                print(f"{neck.angles()}{note}", flush=True)
                time.sleep(1.0 / args.rate)
        except KeyboardInterrupt:
            print()
    return 0


def move_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Turn the GEAR Neck to the given angles in degrees, wait for Enter, then go back "
            "to zero and switch torque off. A joint left out holds where it is."
        )
    )
    parser.add_argument("--yaw", type=float, help="degrees, positive turns the head right")
    parser.add_argument("--pitch", type=float, help="degrees, positive nods the head down")
    parser.add_argument("--roll", type=float, help="degrees, positive tilts the head right")
    add_config_argument(parser)
    add_bus_arguments(parser)
    args = parser.parse_args(argv)
    neck = _open(args)
    if neck is None:
        return 1
    with neck:
        try:
            print(f"now     {neck.angles()}")
            neck.enable()
            measured = neck.move_to(yaw=args.yaw, pitch=args.pitch, roll=args.roll)
            print(f"target  {neck.target()}")
            print(f"reached {measured}")
            input("Press Enter to go back to zero and switch torque off...")
            print(f"reached {neck.move_to(yaw=0.0, pitch=0.0, roll=0.0)}")
        except NeckError as exc:
            print(f"error: {exc}")
            return 1
        except KeyboardInterrupt:
            print("\ncancelled; switching torque off where the neck is")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(angles_main())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zero the GEAR Neck: make the pose it holds now every joint's 0.

Seat the Homing Fixture on the assembled neck (or hold the neck at its zero
pose) and press Enter. The script reads every drive's present position
(register 552) and shows how far each zero would move, in the joint directions
of the settings file, then asks for confirmation before writing anything. Only on
confirmation does it make the pose the neck holds at that moment each drive's
zero (FF x7 FE), which has no reply, and read the position back to check.
Nothing is written if you decline, so declining just shows the positions.

A drive with both encoders calibrated (register 403 reads 0x3C) keeps its
position through a power cycle, within +-pi of zero: power the drives off and
on, then run this again and decline to check the zero held. A drive with its
torque on takes no zero, so such drives are skipped; no other program may talk
to the drives meanwhile.

    neck_zero
    neck_zero --joints pitch
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass

from . import gdk_protocol as proto
from .gdk_link import BusUnavailable, GdkLink, add_bus_arguments, open_bus
from .load_config import JOINT_NAMES, ConfigError, Joint, add_config_argument, load_config

# The zero has no reply; the position is read back this long after it.
ZERO_SETTLE_S = 0.05
# How far from 0 the position read back may be, as the neck is held by hand.
ZERO_TOLERANCE_RAD = 0.02


class NotZeroable(RuntimeError):
    """The drive answered but cannot be zeroed in its current state."""


@dataclass(frozen=True)
class ZeroPlan:
    """Captured pose of one joint; zeroing moves its position by ``-position_rad``."""

    joint: Joint
    position_rad: float  # in the joint's direction
    calibration: int  # register 403


def read_position(link: GdkLink, joint: Joint) -> float:
    """The joint's present position (register 552) in rad, in the joint's direction."""
    data = link.read_register(
        joint.ids, proto.PRESENT_POSITION_REGISTER, proto.PRESENT_POSITION_LENGTH
    )
    return joint.direction * proto.unpack_present_position(data)


def plan_zero(link: GdkLink, joint: Joint) -> ZeroPlan:
    """Read the drive's status, encoder calibration and present position. No writes."""
    if link.read_status(joint.ids).torque_on:
        raise NotZeroable("torque on, holding a setpoint; stop the program driving the neck")
    calibration = link.read_register(joint.ids, proto.CALIBRATION_REGISTER, 1)[0]
    return ZeroPlan(joint, read_position(link, joint), calibration)


def apply_zero(link: GdkLink, plan: ZeroPlan) -> float:
    """Make the present position the drive's zero; return the position read back."""
    link.send(plan.joint.ids.id, proto.pack_set_zero())
    time.sleep(ZERO_SETTLE_S)
    return read_position(link, plan.joint)


def _plan_drives(link: GdkLink, joints: list[Joint]) -> list[ZeroPlan]:
    """Read a zeroing plan per drive. Reads only; skips drives that cannot be zeroed."""
    plans = []
    for joint in joints:
        try:
            plans.append(plan_zero(link, joint))
        except NotZeroable as exc:
            print(f"{joint.name} (ID {joint.ids.id}): {exc}, skipped")
        except TimeoutError:
            ids = joint.ids
            print(f"{joint.name} (ID {ids.id}): no reply from CAN_ID {ids.can_id}, skipped")
    return plans


def _print_plans(plans: list[ZeroPlan]) -> None:
    print("\nplanned zeroing (the current pose becomes position 0):")
    for plan in plans:
        print(
            f"  {plan.joint.name:5s} (ID {plan.joint.ids.id}): "
            f"position {plan.position_rad:+.4f} rad "
            f"({math.degrees(plan.position_rad):+.1f} deg) -> 0"
        )
        if not proto.keeps_position(plan.calibration):
            print(
                f"        register 403 reads 0x{plan.calibration:02X}, not both encoders "
                "calibrated: this zero is lost at power off"
            )


def _apply_plans(link: GdkLink, plans: list[ZeroPlan]) -> None:
    for plan in plans:
        try:
            position_rad = apply_zero(link, plan)
        except TimeoutError as exc:
            print(f"{plan.joint.name}: zero sent, but {exc}")
            continue
        if abs(position_rad) > ZERO_TOLERANCE_RAD:
            print(
                f"{plan.joint.name}: zero sent, but the position reads {position_rad:+.4f} rad; "
                "did the neck move?"
            )
        else:
            print(f"{plan.joint.name}: zeroed, position now {position_rad:+.4f} rad")
    print("Power the drives off and on, then run this again and decline to check the zero held.")


def _confirm(prompt: str) -> bool:
    return input(prompt).strip().lower() in ("y", "yes")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero the GEAR Neck's drives by capturing the current pose"
    )
    parser.add_argument(
        "--joints",
        nargs="+",
        choices=JOINT_NAMES,
        default=list(JOINT_NAMES),
        help="neck joints to zero",
    )
    add_config_argument(parser)
    add_bus_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = load_config(args.config)
        bus = open_bus(config.can, args.interface, args.channel)
    except (ConfigError, BusUnavailable) as exc:
        print(f"error: {exc}")
        return 1
    link = GdkLink(bus)
    try:
        input("Hold the neck at its ZERO pose (Homing Fixture seated), then press Enter...")
        plans = _plan_drives(link, [config.joints_by_name[name] for name in args.joints])
        if not plans:
            print("no drives available to zero")
            return 1
        _print_plans(plans)
        if not _confirm("\nHold the neck still. Make this pose each joint's zero? [y/N] "):
            print("Aborted; nothing written.")
            return 0
        _apply_plans(link, plans)
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        return 1
    finally:
        bus.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assign the CAN ids of the GEAR Neck's GDK drives.

A GDK drive takes frames on its ID (register 7) and replies on its CAN_ID
(register 14); drives leave the factory on ID 1 and CAN_ID 0. The neck expects
the ids in its settings file, config/neck.yaml: yaw 50/51, roll 52/53, pitch
54/55.

  scan    ping every ID and list the drives that answer, with their voltage,
          faults and encoder calibration (register 403); reads only
  assign  move one drive to a neck joint's ids (--joint) or to --id/--can-id

Connect the drives still on the factory ids one at a time: two drives on one ID
send identical ping replies at the same moment, which merge on the bus into
one, so no scan can tell them apart. assign scans first, refuses ids another
drive uses, and shows the change and asks before writing anything. On
confirmation it switches the drive's torque off (the joint goes limp), writes
CAN_ID then ID, and pings the new ID to check. The manual does not say whether
these writes survive a power cycle: power the drive off and on, then scan
again. No other program may talk to the drives meanwhile.

    neck_assign_id scan
    neck_assign_id assign --joint yaw
    neck_assign_id assign --id 60 --can-id 61 --from 1
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterable, Mapping

from . import gdk_protocol as proto
from .gdk_link import BusUnavailable, GdkLink, add_bus_arguments, open_bus
from .gdk_protocol import Ping
from .load_config import JOINT_NAMES, ConfigError, Ids, add_config_argument, load_config

# Per ID while scanning, so all 252 take about 5 s.
SCAN_REPLY_TIMEOUT_S = 0.02
# Torque off has no reply; the status is read this long after it.
TORQUE_OFF_SETTLE_S = 0.02


class AssignError(RuntimeError):
    """The ids cannot be assigned as asked."""


def scan(link: GdkLink, ids: Iterable[int] = range(proto.ID_MIN, proto.ID_MAX + 1)) -> list[Ping]:
    """Ping each ID and return the replies of the drives that answer. Reads only."""
    found = []
    for drive_id in ids:
        ping = link.ping(drive_id, SCAN_REPLY_TIMEOUT_S)
        if ping is not None:
            found.append(ping)
    return found


def check_ids(target: Ids) -> None:
    for name, value in (("ID", target.id), ("CAN_ID", target.can_id)):
        if not proto.ID_MIN <= value <= proto.ID_MAX:
            raise AssignError(f"{name} {value} is outside {proto.ID_MIN}-{proto.ID_MAX}")
    if target.id == target.can_id:
        raise AssignError(
            "ID and CAN_ID must differ: a reply on its request's id can collide with the "
            "next request on the bus"
        )


def choose_drive(found: list[Ping], target: Ids, from_id: int | None) -> Ping:
    """The drive to move to ``target``, once no other drive is found using its ids."""
    check_ids(target)
    if from_id is not None:
        drive = next((ping for ping in found if ping.id == from_id), None)
        if drive is None:
            raise AssignError(f"no drive answers on ID {from_id}")
    elif len(found) == 1:
        drive = found[0]
    elif not found:
        raise AssignError("no drive answered")
    else:
        ids = ", ".join(str(ping.id) for ping in found)
        raise AssignError(f"drives answered on IDs {ids}; pick one with --from")
    for other in found:
        if other.id == drive.id:
            continue
        taken = sorted({other.id, other.can_id} & set(target))
        if taken:
            raise AssignError(
                f"the drive on ID {other.id} / CAN_ID {other.can_id} already uses {taken}"
            )
    return drive


def assign(link: GdkLink, drive: Ping, target: Ids) -> Ping:
    """Switch torque off, write CAN_ID then ID, and return the drive's ping reply on its new ids."""
    link.send(drive.id, proto.pack_torque_off())
    time.sleep(TORQUE_OFF_SETTLE_S)
    if link.read_status(drive).torque_on:
        raise AssignError(f"the drive on ID {drive.id} kept its torque on; nothing written")
    # Both writes go to the old ID. The manual does not say which CAN_ID answers
    # the CAN_ID write, so either is taken.
    reply_ids = {drive.can_id, target.can_id}
    for name, register, value in (
        ("CAN_ID", proto.CAN_ID_REGISTER, target.can_id),
        ("ID", proto.ID_REGISTER, target.id),
    ):
        data = proto.pack_register_write(register, value)
        reply = link.request(drive.id, data, reply_ids, proto.WRITE_REPLY_LENGTH)
        if not proto.write_accepted(reply):
            raise AssignError(f"the drive on ID {drive.id} refused the {name} write")
    ping = link.ping(target.id)
    if ping is None or (ping.id, ping.can_id) != target:
        raise AssignError(
            f"no ping reply from ID {target.id} on CAN_ID {target.can_id} after the writes; "
            "scan to see where the drive is"
        )
    return ping


def describe(link: GdkLink, ping: Ping, neck_ids: Mapping[str, Ids]) -> str:
    """One line on a drive that answered a ping: ids, firmware, status and calibration."""
    joint = next((name for name, ids in neck_ids.items() if ids == (ping.id, ping.can_id)), None)
    text = (
        f"ID {ping.id:3d}  CAN_ID {ping.can_id:3d}  model {ping.model_number}  "
        f"firmware {'.'.join(str(part) for part in ping.firmware)}"
    )
    if joint is not None:
        text += f"  ({joint})"
    try:
        status = link.read_status(ping)
        calibration = link.read_register(ping, proto.CALIBRATION_REGISTER, 1)[0]
    except TimeoutError:
        return text + "  no status reply"
    text += (
        f"  torque {'ON' if status.torque_on else 'off'}  {status.bus_voltage_v:.2f} V  "
        f"fault {proto.fault_names(status.fault)}"
    )
    if proto.keeps_position(calibration):
        return text + "  encoders calibrated"
    return text + f"  register 403 = 0x{calibration:02X}: encoders NOT calibrated"


def _scan_and_print(link: GdkLink, neck_ids: Mapping[str, Ids]) -> list[Ping]:
    print(f"pinging IDs {proto.ID_MIN}-{proto.ID_MAX}...")
    found = scan(link)
    for ping in found:
        print("  " + describe(link, ping, neck_ids))
    if not found:
        print(
            "  nothing answered: check power, 1 Mbit/s, 120 ohm at both ends, and the "
            "CAN/485 switch on CAN"
        )
    return found


def _confirm(prompt: str) -> bool:
    return input(prompt).strip().lower() in ("y", "yes")


def _run_assign(
    link: GdkLink, target: Ids, from_id: int | None, neck_ids: Mapping[str, Ids]
) -> None:
    found = _scan_and_print(link, neck_ids)
    drive = choose_drive(found, target, from_id)
    if (drive.id, drive.can_id) == target:
        print(f"the drive already has ID {target.id} / CAN_ID {target.can_id}; nothing to do")
        return
    print(
        f"\nplanned: ID {drive.id} / CAN_ID {drive.can_id} -> "
        f"ID {target.id} / CAN_ID {target.can_id}"
    )
    if link.read_status(drive).torque_on:
        print("its torque is ON and is switched off first: the joint goes limp")
    if not _confirm("Write the new ids? [y/N] "):
        print("Aborted; nothing written.")
        return
    ping = assign(link, drive, target)
    print(
        f"done: the drive answers on ID {ping.id} / CAN_ID {ping.can_id}. Power it off and "
        "on, then scan to check it kept them."
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    add_config_argument(common)
    add_bus_arguments(common)
    parser = argparse.ArgumentParser(description="Assign the CAN ids of the GEAR Neck's drives")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("scan", parents=[common], help="list the drives that answer a ping")
    assign_parser = commands.add_parser(
        "assign", parents=[common], help="move one drive to new ids"
    )
    target = assign_parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--joint", choices=JOINT_NAMES, help="assign this neck joint's ids from the settings file"
    )
    target.add_argument("--id", type=int, help="new ID (register 7), with --can-id")
    assign_parser.add_argument("--can-id", type=int, help="new CAN_ID (register 14), with --id")
    assign_parser.add_argument(
        "--from",
        dest="from_id",
        type=int,
        help="current ID of the drive to move; needed when more than one answers",
    )
    args = parser.parse_args(argv)
    if args.command == "assign" and (args.id is None) != (args.can_id is None):
        parser.error("--id and --can-id go together")
    return args


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
        if args.command == "scan":
            _scan_and_print(link, config.neck_ids)
        else:
            target = config.neck_ids[args.joint] if args.joint else Ids(args.id, args.can_id)
            _run_assign(link, target, args.from_id, config.neck_ids)
    except (AssignError, TimeoutError) as exc:
        print(f"error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        return 1
    finally:
        bus.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

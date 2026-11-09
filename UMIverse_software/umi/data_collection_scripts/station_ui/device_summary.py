# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which detected devices are plugged in together: one row per gripper.

A gripper carries its own USB hub. Its controller, its OAK board and its
iPhone all hang off that hub, so the devices sharing a hub with a gripper are
the camera and phone mounted on it -- which is what an operator needs to know
when two look-alike serials have to go in the left and right boxes. The scan
records where each device is plugged in (`metadata["usb_location"]`, see
`umi.common.hardware_sdk.usb_location`); this folds that into rows.

The USB (trigger) camera under each gripper is plugged into the computer, not
the gripper's hub, so it lands in a row of its own: its live preview is what
says which gripper it is under.

Pure: no hardware and no I/O. The scan worker prints it as text and the
configuration page renders the same rows, so the two can never disagree.
"""

from __future__ import annotations

from collections import Counter
from typing import NamedTuple

from station_ui.devices import identifier_for
from umi.common.hardware_sdk.usb_location import hub_of

__all__ = [
    "COLUMNS",
    "EMPTY_CELL",
    "NO_DEVICES",
    "PORT_HEADING",
    "UNAVAILABLE",
    "Cell",
    "empty_message",
    "rows",
    "table_lines",
]

COLUMNS = (
    ("gripper", "Gripper"),
    ("oak", "OAK camera"),
    ("iphone", "iPhone"),
    ("uvc", "USB camera"),
)
PORT_HEADING = "USB port"
EMPTY_CELL = "—"
NO_DEVICES = "No devices detected."
# Distinguish devices outside the table's supported kinds from an empty scan.
NO_TABLE_DEVICES = "No grippers, OAK cameras, iPhones or USB cameras detected."
UNAVAILABLE = "Device information could not be loaded"

_KINDS = {kind for kind, _ in COLUMNS}


class Cell(NamedTuple):
    """One device in a row: what the device box takes, and whether the scan
    could open it. Reads as text the way the terminal prints it."""

    name: str
    available: bool = True

    def __str__(self) -> str:
        return self.name if self.available else f"{self.name} (unavailable)"


def rows(found) -> list[dict]:
    """One row per gripper with what shares its hub, then one per other device.

    A row is `{"port": str | None, "gripper": [Cell], "oak": [Cell], "iphone":
    [Cell], "uvc": [Cell]}`. `port` is where the row's cable is plugged in: the
    gripper's hub, or the device's own location. Each cell's name is exactly
    what the device box takes, so the operator can copy it across.

    A hub carrying two grippers is the bench's hub, not either gripper's own, so
    nothing is grouped there: a pairing on it would be a guess, and a wrong one
    is worse than a row each.
    """
    records = [
        record
        for record in found or []
        if record.get("kind") in _KINDS and identifier_for(record)
    ]
    locations = [_location(record) for record in records]
    grippers_per_hub = Counter(
        hub_of(location)
        for record, location in zip(records, locations)
        if record["kind"] == "gripper" and hub_of(location)
    )
    owned = {hub for hub, count in grippers_per_hub.items() if count == 1}

    by_hub: dict[str, dict] = {}
    out = []
    for record, location in zip(records, locations):
        hub = hub_of(location)
        if hub in owned:
            row = by_hub.get(hub)
            if row is None:
                row = by_hub[hub] = _row(hub)
                out.append(row)
        else:
            row = _row(location)
            out.append(row)
        row[record["kind"]].append(_cell(record))
    return sorted(out, key=_order)


def empty_message(found) -> str:
    """What to say instead of a table with no rows."""
    return NO_DEVICES if not found else NO_TABLE_DEVICES


def table_lines(found) -> list[str]:
    """`rows` as a plain-text table, for the scan's terminal output."""
    summary = rows(found)
    if not summary:
        return [empty_message(found)]
    header = [PORT_HEADING, *(heading for _, heading in COLUMNS)]
    body = [
        [
            row["port"] or EMPTY_CELL,
            *(", ".join(map(str, row[kind])) or EMPTY_CELL for kind, _ in COLUMNS),
        ]
        for row in summary
    ]
    widths = [max(len(line[i]) for line in (header, *body)) for i in range(len(header))]

    def _line(cells):
        return " | ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()

    return [_line(header), "-+-".join("-" * w for w in widths), *map(_line, body)]


def _location(record) -> str | None:
    return str((record.get("metadata") or {}).get("usb_location") or "") or None


def _row(port) -> dict:
    return {"port": port, **{kind: [] for kind, _ in COLUMNS}}


def _cell(record) -> Cell:
    return Cell(identifier_for(record), record.get("state") == "available")


def _order(row):
    """Grippers first, then by where they are plugged in; unknown ports last."""
    port = row["port"]
    numbers = tuple(int(n) for n in port.replace("-", ".").split(".")) if port else ()
    return (not row["gripper"], port is None, numbers)

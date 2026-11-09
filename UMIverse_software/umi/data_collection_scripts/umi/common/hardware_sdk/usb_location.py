# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where a USB device is plugged in, spelled one way for every SDK.

The form is ``"<bus>-<port>.<port>..."`` -- what Linux names a device in sysfs
and pyserial reports as ``location`` (``"1-1.3.4"``). depthai, pyserial and the
OS each spell it differently; these turn each spelling into that one, so an OAK
board, a gripper, an iPhone and a USB camera can be compared by the hub they
hang off.

Bus numbers agree across sources because they all come from the same place:
libusb (depthai) and pyserial read sysfs on Linux, and on macOS all three take
the bus from the top byte of the IOKit ``locationID``.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

__all__ = [
    "by_serial_number",
    "from_pyserial",
    "from_video_node",
    "from_xlink_name",
    "hub_of",
]

_LOCATION = re.compile(r"^\d+-\d+(?:\.\d+)*$")

# `ioreg -p IOUSB -l` prints each device as a `+-o Name@<locationID>` header
# followed by its properties, before any of its children.
_IOREG_HEADER = re.compile(r"\+-o .*@([0-9a-fA-F]{8})\s+<class ")
_IOREG_SERIAL = re.compile(r'"USB Serial Number" = "([^"]*)"')
IOREG_TIMEOUT_S = 5.0


def from_xlink_name(name) -> str | None:
    """depthai's ``DeviceInfo.name`` for a USB board: ``"0.1.3.3"``, the bus then
    each port."""
    parts = str(name or "").split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts):
        return None
    return f"{parts[0]}-{'.'.join(parts[1:])}"


def from_pyserial(location) -> str | None:
    """pyserial's ``ListPortInfo.location``: ``"0-1.3.4"``, which Linux follows
    with the interface (``"1-1.3.4:1.0"``)."""
    text = str(location or "").split(":", 1)[0]
    return text if _LOCATION.match(text) else None


def from_video_node(
    path, root: pathlib.Path = pathlib.Path("/sys/class/video4linux")
) -> str | None:
    """Where the camera behind ``/dev/videoN`` is plugged in, read from sysfs.

    ``<root>/videoN/device`` links to the camera's USB interface
    (``.../usb3/3-7/3-7:1.0``), named the way pyserial spells a location. None
    off Linux and for a camera that is not on USB.
    """
    try:
        interface = (root / pathlib.PurePath(str(path or "")).name / "device").resolve(
            strict=True
        )
    except OSError:
        return None
    return from_pyserial(interface.name)


def hub_of(location) -> str | None:
    """The hub a device hangs off: ``"0-1.3.4"`` -> ``"0-1.3"``.

    None for a device on a root port, which hangs off the computer itself.
    """
    if not location:
        return None
    bus, _, ports = str(location).partition("-")
    head, dot, _ = ports.rpartition(".")
    return f"{bus}-{head}" if dot else None


def by_serial_number() -> dict[str, str]:
    """Upper-case USB serial number -> location, for every device the OS lists.

    For the SDKs that report an identity but not a port (Record3D gives an
    iPhone's UDID and nothing else). Empty wherever the OS cannot be asked:
    a missing location only costs the device its place in a grouping.
    """
    try:
        if sys.platform == "darwin":
            return _from_ioreg()
        if sys.platform.startswith("linux"):
            return _from_sysfs()
    except Exception:  # noqa: BLE001 - a location is a nicety, never a failure
        pass
    return {}


def _from_location_id(location_id: int) -> str | None:
    """IOKit ``locationID`` 0x00131000 -> ``"0-1.3.1"``: the bus, then a nibble
    per port until the first zero."""
    ports = []
    for shift in range(20, -1, -4):
        port = (location_id >> shift) & 0xF
        if port == 0:
            break
        ports.append(str(port))
    if not ports:
        return None
    return f"{location_id >> 24}-{'.'.join(ports)}"


def _from_ioreg() -> dict[str, str]:
    output = subprocess.run(
        ["ioreg", "-p", "IOUSB", "-l", "-w0"],
        capture_output=True,
        text=True,
        timeout=IOREG_TIMEOUT_S,
        check=True,
    ).stdout
    found: dict[str, str] = {}
    location = None
    for line in output.splitlines():
        header = _IOREG_HEADER.search(line)
        if header:
            location = _from_location_id(int(header.group(1), 16))
            continue
        serial = _IOREG_SERIAL.search(line)
        if serial and location:
            found.setdefault(serial.group(1).upper(), location)
    return found


def _from_sysfs(root: pathlib.Path = pathlib.Path("/sys/bus/usb/devices")):
    found: dict[str, str] = {}
    for device in root.iterdir():
        if not _LOCATION.match(device.name):
            continue
        try:
            serial = (device / "serial").read_text().strip()
        except OSError:
            continue
        if serial:
            found.setdefault(serial.upper(), device.name)
    return found

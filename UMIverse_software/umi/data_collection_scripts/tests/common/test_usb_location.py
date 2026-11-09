# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every SDK's spelling of "where is this plugged in" becomes one string.

The device summary groups an OAK board, a gripper and an iPhone by the hub they
share, and it can only do that if depthai, pyserial and the OS agree on how a
port is written. These pin each translation against what the real tools print
(captured from a station with two v3 grippers).
"""

import subprocess

import pytest

from umi.common.hardware_sdk import usb_location as ul

# `ioreg -p IOUSB -l -w0`, trimmed: a gripper hub with its Pico, iPhone and OAK
# (the OAK's serial is the shared bootloader id, so it maps nowhere useful).
IOREG = """\
+-o Root  <class IORegistryEntry, id 0x100000100, retain 31>
  +-o AppleT8142USBXHCI@00000000  <class AppleT8142USBXHCI, id 0x100000a1b>
    +-o USB HUB@00130000  <class IOUSBHostDevice, id 0x10004fc22>
    | {
    |   "USB Product Name" = "USB HUB"
    |   "locationID" = 1245184
    | }
    | +-o Pico@00134000  <class IOUSBHostDevice, id 0x10004fc37>
    | | {
    | |   "USB Serial Number" = "534A3035000000B8"
    | |   "locationID" = 1261568
    | | }
    | +-o iPhone@00131000  <class IOUSBHostDevice, id 0x1000506f9>
    |   {
    |     "kUSBSerialNumberString" = "0000813000000000000000A1"
    |     "USB Serial Number" = "0000813000000000000000A1"
    |   }
    +-o YubiKey OTP+FIDO+CCID@03100000  <class IOUSBHostDevice, id 0x10004f0e1>
      {
        "USB Serial Number" = "abc123"
      }
"""


@pytest.mark.parametrize(
    "name, expected",
    [
        ("0.1.3.3", "0-1.3.3"),  # macOS: bus 0, behind a hub on root port 1
        ("3.2", "3-2"),  # Linux: bus 3, root port 2
        ("", None),
        (None, None),
        ("not-a-path", None),
        ("7", None),  # a bus with no port is not a location
    ],
)
def test_depthai_names_become_locations(name, expected):
    assert ul.from_xlink_name(name) == expected


@pytest.mark.parametrize(
    "location, expected",
    [
        ("0-1.3.4", "0-1.3.4"),  # macOS
        ("1-1.3.4:1.0", "1-1.3.4"),  # Linux appends the interface
        ("", None),
        (None, None),
        ("garbage", None),
    ],
)
def test_pyserial_locations_lose_the_interface(location, expected):
    assert ul.from_pyserial(location) == expected


def _video_node(sysfs, name, device):
    """`<sysfs>/class/video4linux/<name>/device` -> `<sysfs>/devices/<device>`,
    the link the kernel makes for every V4L2 node."""
    target = sysfs / "devices" / device
    target.mkdir(parents=True)
    node = sysfs / "class" / "video4linux" / name
    node.mkdir(parents=True)
    (node / "device").symlink_to(target)


def test_a_usb_cameras_video_node_is_located_by_its_interface(tmp_path):
    # As a V4 station links its two trigger cameras: one on a root port, one
    # behind a hub.
    _video_node(tmp_path, "video0", "pci0000:80/0000:80:14.0/usb3/3-7/3-7:1.0")
    _video_node(tmp_path, "video2", "pci0000:80/0000:80:14.0/usb3/3-1/3-1.2/3-1.2:1.0")
    root = tmp_path / "class" / "video4linux"

    assert ul.from_video_node("/dev/video0", root) == "3-7"
    assert ul.from_video_node("/dev/video2", root) == "3-1.2"


def test_a_video_node_that_is_not_on_usb_or_not_there_has_no_location(tmp_path):
    _video_node(tmp_path, "video0", "platform/tegra-capture-vi/video0")
    root = tmp_path / "class" / "video4linux"

    assert ul.from_video_node("/dev/video0", root) is None
    assert ul.from_video_node("/dev/video9", root) is None
    assert ul.from_video_node("", root) is None
    assert ul.from_video_node(None, root) is None


@pytest.mark.parametrize(
    "location, hub",
    [
        ("0-1.3.4", "0-1.3"),
        ("1-2.1", "1-2"),
        ("0-1", None),  # a root port hangs off the computer, not a hub
        (None, None),
        ("", None),
    ],
)
def test_hub_of(location, hub):
    assert ul.hub_of(location) == hub


@pytest.mark.parametrize(
    "location_id, expected",
    [
        (0x00131000, "0-1.3.1"),
        (0x00134000, "0-1.3.4"),
        (0x01140000, "1-1.4"),
        (0x00000000, None),  # the controller itself
    ],
)
def test_iokit_location_ids(location_id, expected):
    assert ul._from_location_id(location_id) == expected


def test_ioreg_maps_each_serial_to_its_own_header(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a, 0, stdout=IOREG, stderr=""),
    )

    found = ul._from_ioreg()

    assert found["534A3035000000B8"] == "0-1.3.4"
    assert found["0000813000000000000000A1"] == "0-1.3.1"
    assert found["ABC123"] == "3-1", "serials are keyed upper-case"


def test_sysfs_maps_serials_and_skips_interfaces_and_root_hubs(tmp_path):
    for name, serial in (
        ("1-1.3.1", "0000813000000000000000A1\n"),
        ("1-1.3.4", "534A3035000000B8"),
        ("1-1.3.4:1.0", "interface, not a device"),
        ("usb1", "0000:00:14.0"),
        ("1-1.2", None),  # no serial file at all
    ):
        device = tmp_path / name
        device.mkdir()
        if serial is not None:
            (device / "serial").write_text(serial)

    found = ul._from_sysfs(tmp_path)

    assert found == {
        "0000813000000000000000A1": "1-1.3.1",
        "534A3035000000B8": "1-1.3.4",
    }


def test_an_os_that_cannot_be_asked_is_an_empty_answer(monkeypatch):
    def _broken():
        raise FileNotFoundError("ioreg")

    monkeypatch.setattr(ul, "_from_ioreg", _broken)
    monkeypatch.setattr(ul, "_from_sysfs", _broken)
    monkeypatch.setattr(ul.sys, "platform", "darwin")
    assert ul.by_serial_number() == {}
    monkeypatch.setattr(ul.sys, "platform", "linux")
    assert ul.by_serial_number() == {}
    monkeypatch.setattr(ul.sys, "platform", "win32")
    assert ul.by_serial_number() == {}

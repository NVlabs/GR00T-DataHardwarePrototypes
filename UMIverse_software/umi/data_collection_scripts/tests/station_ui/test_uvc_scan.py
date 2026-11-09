# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The device scan enumerates UVC cameras, so a UVC box has a dropdown.

There is no UVC registry, so `_REGISTRY_LOOKUP` has no `uvc` entry and a UVC box
has no pre-scan fallback: its dropdown is built only from scan records. If
`scan_worker._enumerators()` covered only oak / iphone / gripper,
`detected_for(found, "uvc")` would always be empty, and the trigger-camera
dropdown would say "No devices found" on a bench where
`uv run umi/common/hardware_sdk/camera/uvc_camera.py` lists both cameras.

**No test here opens real hardware or shells out to v4l2-ctl.** The enumerator
is imported inside the function under test, so the fake is installed on the
module and picked up at call time.
"""

from __future__ import annotations

import pytest

from station_ui import devices as device_state
from station_ui import scan_worker
from umi.common.hardware_sdk import device_discovery, usb_location

# What `get_connected_cameras` returns for a two-webcam bench: the second
# capture node of each camera is not a capture node, so v4l2-ctl parsing has
# already dropped it and the indices are not contiguous.
FAKE_CAMERAS = [
    {"name": "H264 USB Camera", "path": "/dev/video4", "index": 4},
    {"name": "H264 USB Camera", "path": "/dev/video6", "index": 6},
]
# Where sysfs says each is plugged in: straight into the computer, one root
# port each, as a V4 station's trigger cameras are.
FAKE_LOCATIONS = {"/dev/video4": "3-7", "/dev/video6": "3-8"}


@pytest.fixture
def fake_v4l2(monkeypatch):
    monkeypatch.setattr(
        device_discovery, "_default_enumerate_uvc", lambda: list(FAKE_CAMERAS)
    )
    monkeypatch.setattr(usb_location, "from_video_node", FAKE_LOCATIONS.get)


def test_uvc_cameras_are_enumerated_by_their_index():
    """The identifier is the V4L2 index -- there is no UVC registry to carry a
    name, and the index is what `params.camera_index` stores."""
    found = device_discovery.list_connected_uvc(lambda: list(FAKE_CAMERAS))
    assert [d.device_id for d in found] == ["4", "6"]
    assert all(d.kind == "uvc" for d in found)
    assert all(d.registry_name is None for d in found)
    assert all(d.state == "available" for d in found)


def test_the_name_and_path_survive_so_two_identical_webcams_can_be_told_apart():
    """Both trigger cameras are the same model, so the index and the path are
    the only things that distinguish the two rows in the dropdown."""
    found = device_discovery.list_connected_uvc(lambda: list(FAKE_CAMERAS))
    assert [d.metadata["path"] for d in found] == ["/dev/video4", "/dev/video6"]
    assert found[0].metadata["name"] == "H264 USB Camera"


def test_each_camera_records_where_it_is_plugged_in(fake_v4l2):
    """What puts a trigger camera in the detected-devices table with a USB port,
    like every other device there."""
    found = device_discovery.list_connected_uvc()
    assert [d.metadata["usb_location"] for d in found] == ["3-7", "3-8"]


def test_a_camera_the_os_cannot_locate_keeps_its_old_metadata():
    found = device_discovery.list_connected_uvc(
        lambda: list(FAKE_CAMERAS), locate=lambda path: None
    )
    assert found[0].metadata == {"name": "H264 USB Camera", "path": "/dev/video4"}


def test_a_bench_with_no_v4l2_ctl_is_an_empty_list_not_a_crash():
    """`get_connected_cameras` already returns [] on FileNotFoundError, and the
    scan must survive a station where v4l-utils was never installed."""

    def _boom():
        raise FileNotFoundError("v4l2-ctl")

    assert device_discovery.list_connected_uvc(_boom) == []


def test_the_scan_covers_uvc(fake_v4l2):
    """`scan()` runs every enumerator in `_enumerators()`, and the dropdown is
    built from the records they return, so UVC has to be in that table."""
    found = scan_worker.scan(out=None)
    uvc = [record for record in found if record.get("kind") == "uvc"]
    assert [record["device_id"] for record in uvc] == ["4", "6"]


def test_a_scanned_uvc_camera_is_not_described_as_uncheckable(fake_v4l2):
    """`SCANNED_KINDS` drives the "the scan does not cover uvc cameras — check
    this one by hand" line under a filled box. The scan does cover them, so that
    sentence would be false."""
    assert "uvc" in device_state.SCANNED_KINDS


def test_each_dropdown_row_says_which_physical_camera_it_is(fake_v4l2):
    """Two identical webcams: the model name is the same on both, so the /dev
    path is what the operator matches against the one they just unplugged.

    And not "not in the device registry" -- there is no UVC registry to be
    absent from, so that line reads as a registration the operator has to go
    and do.
    """
    found = scan_worker.scan(out=None)
    notes = [
        device_state.describe_record(record)
        for record in device_state.detected_for(found, "uvc")
    ]
    assert "/dev/video4" in notes[0]
    assert "/dev/video6" in notes[1]
    assert all("device registry" not in note for note in notes)


def test_the_dropdown_offers_every_uvc_camera_on_the_bench(fake_v4l2):
    """The dropdown the operator sees. `detected_for` is what the picker folds
    into its option list."""
    found = scan_worker.scan(out=None)
    offered = [
        device_state.identifier_for(record)
        for record in device_state.detected_for(found, "uvc")
    ]
    assert offered == ["4", "6"]

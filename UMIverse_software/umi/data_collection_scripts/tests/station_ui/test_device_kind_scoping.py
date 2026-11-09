# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device identifiers are matched within their own kind, never across kinds.

The scan returns one flat list for every kind it enumerates, and a device's
identifier is only unique within its kind. A UVC camera's id is its V4L2 index
-- `0`, `4`, `10` -- short enough to collide with anything, so every lookup that
knows which kind of device a row holds has to say so.
"""

from __future__ import annotations

from station_ui import devices as devices_core

# A finished scan of a real bench: a registered OAK and gripper, and an OAK the
# device bank does not know -- the ordinary case, and the reason
# `identifier_for` falls back to the serial.
SCANNED = [
    {
        "kind": "oak",
        "registry_name": "oak-W97-01",
        "device_id": "1944",
        "state": "available",
    },
    {
        "kind": "gripper",
        "registry_name": "UMI-044",
        "device_id": "/dev/ttyACM1",
        "state": "available",
    },
    {
        "kind": "oak",
        "registry_name": None,
        "device_id": "1944301000000000A7",
        "state": "available",
    },
]

UVC_TEN = {
    "kind": "uvc",
    "device_id": "10",
    "registry_name": None,
    "state": "available",
    "metadata": {},
}

WEDGED_UVC = {
    "kind": "uvc",
    "device_id": "10",
    "registry_name": None,
    "state": "unavailable",
    "error": "device busy",
    "metadata": {},
}


def test_the_kind_comes_from_the_enumerator_never_from_the_registry():
    """Each `list_connected_*` stamps its own literal `kind=` on every record it
    builds, so a record's kind is a statement about which SDK produced it. The
    device bank is consulted afterwards and only to attach a friendly name;
    being absent from it changes `registry_name`, never `kind`."""
    unregistered = [r for r in SCANNED if not r["registry_name"]]
    assert unregistered, "the fixture no longer covers the unregistered case"

    for record in unregistered:
        # Identified by serial, because there is no name to use...
        assert devices_core.identifier_for(record) == record["device_id"]
        # ...and still sorted into its own kind's picker.
        assert record in devices_core.detected_for(SCANNED, record["kind"])
        for other in devices_core.SCANNED_KINDS - {record["kind"]}:
            assert record not in devices_core.detected_for(SCANNED, other)


def test_an_identifier_is_only_matched_within_its_own_kind():
    """Unscoped, an OAK slot holding `10` would match the UVC record at index
    10 and report `Connected` for a board the scan never saw."""
    found = [UVC_TEN, *SCANNED]

    assert devices_core.record_for(found, "10", "uvc") is UVC_TEN
    assert devices_core.record_for(found, "10", "oak") is None
    assert devices_core.record_for(found, "10", "iphone") is None


def test_a_declared_device_of_the_wrong_kind_is_not_reported_connected():
    """The same hole, through the surface an operator reads."""
    oak_slot = devices_core.DeclaredDevice("Head OAK", "10", kind="oak")

    (status,) = devices_core.statuses([oak_slot], [UVC_TEN])

    assert status.is_unavailable, "a UVC camera answered for an OAK"


def test_a_device_declared_with_no_kind_still_matches_anything():
    """Not merely a lazy default. A template-required tracking camera is
    declared with `kind=None` (see `declared_devices`), so scoping those would
    match nothing and report every one of them missing."""
    tracking = devices_core.DeclaredDevice("Tracking camera", "oak-W97-01", kind=None)

    (status,) = devices_core.statuses([tracking], SCANNED)

    assert status.is_connected


def test_the_refusal_predicate_is_scoped_to_the_kind():
    assert devices_core.unavailable_reason([WEDGED_UVC], "10", "uvc") == "device busy"
    assert devices_core.unavailable_reason([WEDGED_UVC], "10", "oak") is None
    # Unscoped stays unscoped, for the kind-less tracking-camera rows.
    assert devices_core.unavailable_reason([WEDGED_UVC], "10") == "device busy"


def test_the_selection_gate_passes_the_row_kind_through():
    """Through `unavailable_devices`, not the predicate underneath it. The
    predicate taking a kind proves nothing if the gate never hands one over --
    and it is the gate that decides whether Apply is refused.

    Left unscoped it refuses an OAK row because some UVC camera sharing a short
    id will not open, which is a rejection the operator cannot act on: the
    device named in the message is not the device that is wedged.
    """
    from station_ui import config_form

    views = [
        {"field": "head_oak_camera_device", "value": "10", "kind": "oak"},
        {"field": "trigger_camera_device", "value": "10", "kind": "uvc"},
    ]
    refused = config_form.unavailable_devices(views, [WEDGED_UVC])

    assert "trigger_camera_device" in refused, "the wedged UVC row was not refused"
    assert "head_oak_camera_device" not in refused, (
        "an OAK row was refused because a UVC camera with the same id is wedged"
    )


def test_the_worker_enumerates_exactly_the_kinds_the_console_claims():
    """`SCANNED_KINDS` is what the console reads to decide whether the scan
    looked at a device at all, and the worker is what actually looks. If the
    two drift, the console reports on hardware nothing checked."""
    from station_ui import scan_worker

    enumerated = {kind for kind, _label, _fn in scan_worker._enumerators()}

    assert enumerated == devices_core.SCANNED_KINDS

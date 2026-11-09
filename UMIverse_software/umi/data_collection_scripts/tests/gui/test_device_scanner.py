# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the shared GUI device scanner.

`station_ui.core.device_scan.DeviceScanner` caches "what is plugged in right now" for the
config editor: one scan per kind (OAK, iPhone, gripper), cached results handed to every
widget that needs a device picker. The behaviours that matter are cache
hygiene — a re-scan must replace, never accumulate; a kind whose probe raises
must go empty instead of serving stale devices, and must not take the other
kinds down with it — plus lookup semantics: names come from the registry, never
from raw device ids.

Fully hardware-free: all probe callables are injected fakes.

Run from project root:
    uv run pytest tests/gui/test_device_scanner.py -v
"""

import sys
from pathlib import Path

import pytest

# Project root so that "umi" and "station_ui" resolve.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from station_ui.core.device_scan import SCANNER, DeviceScanner  # noqa: E402
from umi.common.hardware_sdk.device_discovery import (  # noqa: E402
    ConnectedDevice,
)


def _device(kind, device_id, registry_name=None, state="available", **metadata):
    """Build a ConnectedDevice without caring about the metadata payload."""
    return ConnectedDevice(
        kind=kind,
        device_id=device_id,
        registry_name=registry_name,
        state=state,
        metadata=metadata,
    )


class _FakeProbe:
    """Zero-arg probe returning a scripted result per call.

    Each entry is either a list of devices to return or an exception instance
    to raise. The last entry repeats once the script runs out.
    """

    def __init__(self, *results):
        self.results = list(results) or [[]]
        self.calls = 0

    def __call__(self):
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        if isinstance(result, Exception):
            raise result
        return list(result)


_OAK_A = _device("oak", "oak-id-a", "oak-FFC-01", sockets={"CAM_A": "AR0234"})
_OAK_B = _device("oak", "oak-id-b", "oak-FFC-02")
_PHONE_A = _device("iphone", "udid-a", "iphone15p-01")
_PHONE_UNREGISTERED = _device("iphone", "udid-unknown", None)
_GRIPPER_A = _device("gripper", "serial-a", "UMI-001")


def _scanner(oak=(_OAK_A, _OAK_B), iphone=(_PHONE_A,), gripper=(_GRIPPER_A,)):
    return DeviceScanner(
        scan_oak=_FakeProbe(list(oak)),
        scan_iphone=_FakeProbe(list(iphone)),
        scan_gripper=_FakeProbe(list(gripper)),
    )


def test_no_devices_before_first_scan():
    scanner = _scanner()
    assert scanner.devices("oak") == []
    assert scanner.devices("iphone") == []
    assert scanner.devices("gripper") == []


def test_last_scan_time_is_none_before_first_scan():
    assert _scanner().last_scan_time() is None


def test_scan_populates_all_kinds():
    scanner = _scanner()
    scanner.scan()
    assert scanner.devices("oak") == [_OAK_A, _OAK_B]
    assert scanner.devices("iphone") == [_PHONE_A]
    assert scanner.devices("gripper") == [_GRIPPER_A]


def test_scan_records_a_float_timestamp():
    scanner = _scanner()
    scanner.scan()
    assert isinstance(scanner.last_scan_time(), float)


def test_by_identifier_finds_oak_device():
    scanner = _scanner()
    scanner.scan()
    assert scanner.by_identifier("oak-FFC-02") == _OAK_B


def test_by_identifier_searches_across_all_kinds():
    """A name is looked up globally, not per-kind."""
    scanner = _scanner()
    scanner.scan()
    assert scanner.by_identifier("iphone15p-01") == _PHONE_A
    assert scanner.by_identifier("oak-FFC-01") == _OAK_A
    assert scanner.by_identifier("UMI-001") == _GRIPPER_A


def test_by_identifier_returns_none_for_unknown_name():
    scanner = _scanner()
    scanner.scan()
    assert scanner.by_identifier("no-such-device") is None


def test_by_identifier_matches_on_device_id():
    """Ids are first-class identifiers.

    Unregistered hardware has no registry name, so a config stores its id. A
    name-only lookup would return None for those devices and silently give an
    unregistered IMX378 the default resolution instead of its own.
    """
    scanner = _scanner(iphone=(_PHONE_UNREGISTERED,))
    scanner.scan()
    assert scanner.by_identifier("udid-unknown") == _PHONE_UNREGISTERED
    assert scanner.by_identifier("no-such-device") is None


def test_unregistered_device_still_listed_by_kind():
    scanner = _scanner(iphone=(_PHONE_UNREGISTERED,))
    scanner.scan()
    assert scanner.devices("iphone") == [_PHONE_UNREGISTERED]


def test_rescan_replaces_previous_results():
    scanner = DeviceScanner(
        scan_oak=_FakeProbe([_OAK_A, _OAK_B], [_OAK_B]),
        scan_iphone=_FakeProbe([_PHONE_A], []),
        scan_gripper=_FakeProbe([_GRIPPER_A]),
    )
    scanner.scan()
    scanner.scan()
    assert scanner.devices("oak") == [_OAK_B]
    assert scanner.devices("iphone") == []


def test_rescan_drops_names_that_disappeared():
    scanner = DeviceScanner(
        scan_oak=_FakeProbe([_OAK_A, _OAK_B], [_OAK_B]),
        scan_iphone=_FakeProbe([]),
        scan_gripper=_FakeProbe([]),
    )
    scanner.scan()
    scanner.scan()
    assert scanner.by_identifier("oak-FFC-01") is None
    assert scanner.by_identifier("oak-FFC-02") == _OAK_B


def test_scan_does_not_raise_when_a_probe_fails():
    scanner = DeviceScanner(
        scan_oak=_FakeProbe(RuntimeError("depthai exploded")),
        scan_iphone=_FakeProbe([_PHONE_A]),
        scan_gripper=_FakeProbe([_GRIPPER_A]),
    )
    scanner.scan()  # must not propagate


def test_failing_oak_probe_clears_stale_oak_results():
    scanner = DeviceScanner(
        scan_oak=_FakeProbe([_OAK_A], RuntimeError("usb reset")),
        scan_iphone=_FakeProbe([_PHONE_A]),
        scan_gripper=_FakeProbe([_GRIPPER_A]),
    )
    scanner.scan()
    scanner.scan()
    assert scanner.devices("oak") == []


def test_failing_oak_probe_leaves_iphone_scan_intact():
    scanner = DeviceScanner(
        scan_oak=_FakeProbe(RuntimeError("usb reset")),
        scan_iphone=_FakeProbe([_PHONE_A]),
        scan_gripper=_FakeProbe([_GRIPPER_A]),
    )
    scanner.scan()
    assert scanner.devices("iphone") == [_PHONE_A]


def test_failing_iphone_probe_leaves_oak_scan_intact():
    scanner = DeviceScanner(
        scan_oak=_FakeProbe([_OAK_A]),
        scan_iphone=_FakeProbe(RuntimeError("record3d exploded")),
        scan_gripper=_FakeProbe([_GRIPPER_A]),
    )
    scanner.scan()
    assert scanner.devices("oak") == [_OAK_A]
    assert scanner.devices("iphone") == []


def test_failed_kind_is_not_findable_by_identifier():
    scanner = DeviceScanner(
        scan_oak=_FakeProbe([_OAK_A], RuntimeError("usb reset")),
        scan_iphone=_FakeProbe([_PHONE_A]),
        scan_gripper=_FakeProbe([_GRIPPER_A]),
    )
    scanner.scan()
    scanner.scan()
    assert scanner.by_identifier("oak-FFC-01") is None
    assert scanner.by_identifier("iphone15p-01") == _PHONE_A


def test_devices_returns_a_copy():
    scanner = _scanner()
    scanner.scan()
    listed = scanner.devices("oak")
    listed.append(_PHONE_A)
    listed.remove(_OAK_A)
    assert scanner.devices("oak") == [_OAK_A, _OAK_B]


def test_unknown_kind_returns_empty_list():
    scanner = _scanner()
    scanner.scan()
    assert scanner.devices("lidar") == []


def test_unknown_kind_does_not_raise_before_scan():
    assert _scanner().devices("") == []


def test_module_level_scanner_is_a_scanner():
    assert isinstance(SCANNER, DeviceScanner)


def test_module_level_scanner_does_not_scan_at_import():
    """Importing the GUI must never touch hardware."""
    assert SCANNER.last_scan_time() is None


@pytest.mark.parametrize("kind", ["oak", "iphone", "gripper"])
def test_devices_empty_for_each_kind_before_scan(kind):
    assert _scanner().devices(kind) == []


# --- Unregistered devices carry registry_name=None, so an unguarded
# by_identifier() comparison would match them on a falsy argument.


def test_by_identifier_none_does_not_match_unregistered_device():
    scanner = DeviceScanner(
        scan_oak=lambda: [_device("oak", "IDX", None)],
        scan_iphone=lambda: [],
        scan_gripper=lambda: [],
    )
    scanner.scan()
    assert scanner.by_identifier(None) is None


def test_by_identifier_empty_string_does_not_match_unregistered_device():
    scanner = DeviceScanner(
        scan_oak=lambda: [_device("oak", "IDX", None)],
        scan_iphone=lambda: [],
        scan_gripper=lambda: [],
    )
    scanner.scan()
    assert scanner.by_identifier("") is None

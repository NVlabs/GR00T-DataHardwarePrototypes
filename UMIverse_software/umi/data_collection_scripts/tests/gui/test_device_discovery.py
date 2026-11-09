# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for OAK, iPhone, and gripper discovery.

`umi.common.hardware_sdk.device_discovery` reports which supported devices are
physically attached, annotated with their friendly registry names. The GUI
feeds those results into config YAML, so socket names, availability state, and
stable hardware identities must be preserved exactly.

Fully hardware-free: enumerators are injected, registries are monkeypatched,
and depthai / record3d are never imported.

Run from project root:
    uv run pytest tests/gui/test_device_discovery.py -v
"""

import sys
import time
from pathlib import Path

import pytest

# Project root so that "umi" resolves.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from umi.common.hardware_sdk import device_discovery as cd  # noqa: E402

# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeSocket:
    """Mimics depthai's CameraBoardSocket: .name is bare, str() is noisy."""

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"CameraBoardSocket.{self.name}"

    def __repr__(self):
        return str(self)


class FakeFeature:
    def __init__(self, socket_name, sensor_name):
        self.socket = FakeSocket(socket_name)
        self.sensorName = sensor_name


class FakeDeviceInfo:
    def __init__(self, device_id):
        self._device_id = device_id

    def getDeviceId(self):
        return self._device_id


class FakeDevice:
    """Opened OAK handle. Optionally raises when features are queried."""

    def __init__(self, features=None, features_error=None):
        self._features = features or []
        self._features_error = features_error
        self.closed = False

    def getConnectedCameraFeatures(self):
        if self._features_error is not None:
            raise self._features_error
        return self._features

    def close(self):
        self.closed = True


class FakePhone:
    def __init__(self, udid, product_id):
        self.udid = udid
        self.product_id = product_id


def _enumerator(items):
    return lambda: list(items)


def _raiser(exc):
    def _fn(*args, **kwargs):
        raise exc

    return _fn


def _by_id(results):
    return {r.device_id: r for r in results}


@pytest.fixture
def oak_registry(monkeypatch):
    """Install a {friendly_name: device_id} OAK registry."""

    def _install(mapping):
        monkeypatch.setattr(cd, "_oak_registry", lambda: dict(mapping))

    return _install


@pytest.fixture
def iphone_registry(monkeypatch):
    def _install(mapping):
        monkeypatch.setattr(cd, "_iphone_registry", lambda: dict(mapping))

    return _install


# --------------------------------------------------------------------------- #
# list_connected_oak
# --------------------------------------------------------------------------- #


def test_registered_oak_device_reports_name_state_and_sockets(oak_registry):
    oak_registry({"oak-ffc-01": "1844301041"})
    device = FakeDevice(
        features=[
            FakeFeature("CAM_A", "AR0234"),
            FakeFeature("CAM_B", "OV9782"),
        ]
    )

    results = cd.list_connected_oak(
        enumerate_devices=_enumerator([FakeDeviceInfo("1844301041")]),
        open_device=lambda info: device,
    )

    assert len(results) == 1
    result = results[0]
    assert result.kind == "oak"
    assert result.device_id == "1844301041"
    assert result.registry_name == "oak-ffc-01"
    assert result.state == "available"
    assert result.metadata["sockets"] == {"CAM_A": "AR0234", "CAM_B": "OV9782"}


def test_unregistered_oak_device_is_listed_with_none_registry_name(oak_registry):
    oak_registry({"oak-ffc-01": "1844301041"})
    device = FakeDevice(features=[FakeFeature("CAM_A", "AR0234")])

    results = cd.list_connected_oak(
        enumerate_devices=_enumerator([FakeDeviceInfo("9999999999")]),
        open_device=lambda info: device,
    )

    assert len(results) == 1
    assert results[0].device_id == "9999999999"
    assert results[0].registry_name is None
    assert results[0].state == "available"


def test_socket_keys_are_bare_names_not_enum_repr(oak_registry):
    """Socket keys must be "CAM_A", never "CameraBoardSocket.CAM_A".

    Downstream YAML matching keys off the bare socket name; using str(socket)
    produces keys that silently never match.
    """
    oak_registry({})
    device = FakeDevice(
        features=[
            FakeFeature("CAM_A", "AR0234"),
            FakeFeature("CAM_B", "AR0234"),
            FakeFeature("CAM_C", "AR0234"),
        ]
    )

    results = cd.list_connected_oak(
        enumerate_devices=_enumerator([FakeDeviceInfo("1844301041")]),
        open_device=lambda info: device,
    )

    sockets = results[0].metadata["sockets"]
    assert set(sockets) == {"CAM_A", "CAM_B", "CAM_C"}
    for key in sockets:
        assert "CameraBoardSocket" not in key


def test_device_that_fails_to_open_is_kept_as_unavailable(oak_registry):
    oak_registry({"oak-ffc-01": "1844301041"})

    results = cd.list_connected_oak(
        enumerate_devices=_enumerator([FakeDeviceInfo("1844301041")]),
        open_device=_raiser(RuntimeError("device already in use")),
    )

    assert len(results) == 1, "an unopenable device must not be dropped"
    assert results[0].device_id == "1844301041"
    assert results[0].registry_name == "oak-ffc-01"
    assert results[0].state == "unavailable"
    assert results[0].metadata == {"sockets": {}}


def test_open_failure_does_not_hide_healthy_devices(oak_registry):
    oak_registry({})
    good = FakeDevice(features=[FakeFeature("CAM_A", "AR0234")])

    def _open(info):
        if info.getDeviceId() == "bad":
            raise RuntimeError("boom")
        return good

    results = cd.list_connected_oak(
        enumerate_devices=_enumerator([FakeDeviceInfo("bad"), FakeDeviceInfo("good")]),
        open_device=_open,
    )

    states = {r.device_id: r.state for r in results}
    assert states == {"bad": "unavailable", "good": "available"}


def test_close_is_called_when_feature_query_raises(oak_registry):
    """A leaked handle blocks the recorder — close() must run regardless."""
    oak_registry({})
    device = FakeDevice(features_error=RuntimeError("XLink error"))

    results = cd.list_connected_oak(
        enumerate_devices=_enumerator([FakeDeviceInfo("1844301041")]),
        open_device=lambda info: device,
    )

    assert device.closed is True
    assert len(results) == 1
    assert results[0].state == "unavailable"


def test_close_is_called_on_every_opened_device(oak_registry):
    oak_registry({})
    devices = {
        did: FakeDevice(features=[FakeFeature("CAM_A", "AR0234")])
        for did in ("a", "b", "c")
    }

    results = cd.list_connected_oak(
        enumerate_devices=_enumerator([FakeDeviceInfo(d) for d in "abc"]),
        open_device=lambda info: devices[info.getDeviceId()],
    )

    assert len(results) == 3
    assert all(d.closed for d in devices.values())


def test_enumerate_failure_returns_empty_list(oak_registry):
    oak_registry({"oak-ffc-01": "1844301041"})

    results = cd.list_connected_oak(
        enumerate_devices=_raiser(RuntimeError("no depthai runtime")),
        open_device=lambda info: FakeDevice(),
    )

    assert results == []


def test_lazy_enumerate_failure_returns_empty_list(oak_registry):
    oak_registry({})

    def _broken_enumerator():
        yield FakeDeviceInfo("partial")
        raise RuntimeError("USB enumeration failed")

    results = cd.list_connected_oak(
        enumerate_devices=_broken_enumerator,
        open_device=lambda info: pytest.fail(f"unexpected probe of {info}"),
    )

    assert results == []


def test_result_order_matches_enumeration_order(oak_registry):
    """Concurrent probing must not scramble order.

    The first device is the slowest to open, so any implementation that
    appends results as they complete returns them reversed.
    """
    oak_registry({})
    order = ["first", "second", "third"]
    delays = {"first": 0.15, "second": 0.05, "third": 0.0}

    def _open(info):
        time.sleep(delays[info.getDeviceId()])
        return FakeDevice(features=[FakeFeature("CAM_A", "AR0234")])

    results = cd.list_connected_oak(
        enumerate_devices=_enumerator([FakeDeviceInfo(d) for d in order]),
        open_device=_open,
    )

    assert [r.device_id for r in results] == order


def test_order_preserved_when_some_devices_fail(oak_registry):
    """Failures resolve instantly; they must not jump ahead of slow successes."""
    oak_registry({})

    def _open(info):
        if info.getDeviceId() == "broken":
            raise RuntimeError("boom")
        time.sleep(0.1)
        return FakeDevice(features=[FakeFeature("CAM_A", "AR0234")])

    order = ["slow", "broken", "also_slow"]
    results = cd.list_connected_oak(
        enumerate_devices=_enumerator([FakeDeviceInfo(d) for d in order]),
        open_device=_open,
    )

    assert [r.device_id for r in results] == order


def test_results_keep_enumeration_order_whatever_the_probe_order(oak_registry):
    """Order in == order out. Deliberately does not pin serial vs concurrent.

    Whether probing is serial or concurrent is an implementation detail: a
    device that reports partial sockets does so either way. The invariant that
    actually matters is that a device's sockets belong to that device, and that
    positions do not shuffle.
    """
    oak_registry({})
    sensors = {"0": "AR0234", "1": "IMX378", "2": "OV9782"}

    def _open(info):
        # Return slow workers first to expose ordering mistakes.
        time.sleep(0.03 - 0.01 * int(info.getDeviceId()))
        return FakeDevice(features=[FakeFeature("CAM_A", sensors[info.getDeviceId()])])

    results = cd.list_connected_oak(
        enumerate_devices=_enumerator([FakeDeviceInfo(str(i)) for i in range(3)]),
        open_device=_open,
    )

    assert [r.device_id for r in results] == ["0", "1", "2"]
    for r in results:
        assert r.metadata["sockets"] == {"CAM_A": sensors[r.device_id]}, (
            f"{r.device_id} got another device's sensor"
        )


def test_no_connected_oak_devices_returns_empty_list(oak_registry):
    oak_registry({"oak-ffc-01": "1844301041"})

    results = cd.list_connected_oak(
        enumerate_devices=_enumerator([]),
        open_device=lambda info: FakeDevice(),
    )

    assert results == []


# --------------------------------------------------------------------------- #
# list_connected_iphone
# --------------------------------------------------------------------------- #


def test_registered_iphone_reports_name_state_and_product_id(iphone_registry):
    iphone_registry({"iphone15p-01": "00008130-001A2B3C4D5E"})

    results = cd.list_connected_iphone(
        enumerate_devices=_enumerator([FakePhone("00008130-001A2B3C4D5E", 0x12A8)]),
    )

    assert len(results) == 1
    result = results[0]
    assert result.kind == "iphone"
    assert result.device_id == "00008130-001A2B3C4D5E"
    assert result.registry_name == "iphone15p-01"
    assert result.state == "available"
    assert result.metadata["product_id"] == 0x12A8


def test_unregistered_iphones_are_not_filtered_out(iphone_registry):
    iphone_registry({"iphone15p-01": "udid-known"})
    phones = [
        FakePhone("udid-unknown-a", 1),
        FakePhone("udid-known", 2),
        FakePhone("udid-unknown-b", 3),
    ]

    results = cd.list_connected_iphone(enumerate_devices=_enumerator(phones))

    assert [r.device_id for r in results] == [
        "udid-unknown-a",
        "udid-known",
        "udid-unknown-b",
    ]
    named = _by_id(results)
    assert named["udid-known"].registry_name == "iphone15p-01"
    assert named["udid-unknown-a"].registry_name is None
    assert named["udid-unknown-b"].registry_name is None


def test_iphone_enumerate_failure_returns_empty_list(iphone_registry):
    iphone_registry({"iphone15p-01": "udid-known"})

    results = cd.list_connected_iphone(
        enumerate_devices=_raiser(OSError("usbmuxd not running")),
    )

    assert results == []


def test_no_connected_iphones_returns_empty_list(iphone_registry):
    iphone_registry({"iphone15p-01": "udid-known"})

    results = cd.list_connected_iphone(enumerate_devices=_enumerator([]))

    assert results == []


# One Pico USB device exposes two CDC ACM ports with one serial number.
# Deduplicate by serial; port paths are connection endpoints, not identities.


def _port(device, serial_number, index=0, description="Pico"):
    """One entry as produced by UartSensorReader.list_connected_devices()."""
    return {
        "device": device,
        "description": description,
        "serial_number": serial_number,
        "hwid": f"USB VID:PID=2E8A:000A SER={serial_number}",
        "index": index,
    }


def _ports_of(result):
    """Every port path recorded on a result, however metadata spells it."""
    metadata = result.metadata or {}
    found = []
    for key in ("ports", "port", "devices", "device"):
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            found.append(value)
        else:
            found.extend(value)
    return found


@pytest.fixture
def gripper_registry(monkeypatch):
    """Install a {gripper_name: serial_number} registry."""

    def _install(mapping):
        monkeypatch.setattr(cd, "_gripper_registry", lambda: dict(mapping))

    return _install


def test_registered_gripper_reports_name_serial_and_state(gripper_registry):
    gripper_registry({"UMI-006": "E6614C77000000BA"})

    results = cd.list_connected_gripper(
        enumerate_devices=_enumerator(
            [_port("/dev/ttyACM0", "E6614C77000000BA", index=0)]
        ),
    )

    assert len(results) == 1
    result = results[0]
    assert result.kind == "gripper"
    assert result.device_id == "E6614C77000000BA"
    assert result.registry_name == "UMI-006"
    assert result.state == "available"
    assert "/dev/ttyACM0" in _ports_of(result)


def test_two_ports_sharing_one_serial_collapse_to_one_device(gripper_registry):
    """The load-bearing test: one physical gripper, two CDC ports, one entry.

    Without dedupe every gripper shows up twice in the picker and the operator
    cannot tell which of the two rows is the real device.
    """
    gripper_registry({"UMI-006": "E6614C77000000BA"})

    results = cd.list_connected_gripper(
        enumerate_devices=_enumerator(
            [
                _port("/dev/ttyACM0", "E6614C77000000BA", index=0),
                _port("/dev/ttyACM1", "E6614C77000000BA", index=1),
            ]
        ),
    )

    assert len(results) == 1, "two CDC ports of one gripper must collapse"
    assert results[0].device_id == "E6614C77000000BA"
    assert results[0].registry_name == "UMI-006"


def test_both_ports_of_a_deduped_gripper_are_recorded(gripper_registry):
    """Dedupe must not lose the second port — callers still need to open it."""
    gripper_registry({"UMI-006": "E6614C77000000BA"})

    results = cd.list_connected_gripper(
        enumerate_devices=_enumerator(
            [
                _port("/dev/ttyACM0", "E6614C77000000BA", index=0),
                _port("/dev/ttyACM1", "E6614C77000000BA", index=1),
            ]
        ),
    )

    ports = _ports_of(results[0])
    assert "/dev/ttyACM0" in ports
    assert "/dev/ttyACM1" in ports


def test_duplicate_gripper_port_is_recorded_once(gripper_registry):
    gripper_registry({})

    results = cd.list_connected_gripper(
        enumerate_devices=_enumerator(
            [
                _port("/dev/ttyACM0", "serial-a"),
                _port("/dev/ttyACM0", "serial-a"),
            ]
        ),
    )

    assert results[0].metadata["ports"] == ["/dev/ttyACM0"]


def test_two_distinct_grippers_are_not_merged(gripper_registry):
    gripper_registry({"UMI-006": "serial-a", "UMI-007": "serial-b"})

    results = cd.list_connected_gripper(
        enumerate_devices=_enumerator(
            [
                _port("/dev/ttyACM0", "serial-a", index=0),
                _port("/dev/ttyACM1", "serial-a", index=1),
                _port("/dev/ttyACM2", "serial-b", index=2),
                _port("/dev/ttyACM3", "serial-b", index=3),
            ]
        ),
    )

    assert [r.device_id for r in results] == ["serial-a", "serial-b"]
    named = _by_id(results)
    assert named["serial-a"].registry_name == "UMI-006"
    assert named["serial-b"].registry_name == "UMI-007"
    assert "/dev/ttyACM2" in _ports_of(named["serial-b"])
    assert "/dev/ttyACM3" in _ports_of(named["serial-b"])


def test_entry_without_serial_number_is_skipped(gripper_registry):
    """No serial means no identity and no way to dedupe — drop it."""
    gripper_registry({"UMI-006": "E6614C77000000BA"})

    results = cd.list_connected_gripper(
        enumerate_devices=_enumerator(
            [
                _port("/dev/ttyACM0", None, index=0),
                _port("/dev/ttyACM1", "E6614C77000000BA", index=1),
            ]
        ),
    )

    assert [r.device_id for r in results] == ["E6614C77000000BA"]
    assert "/dev/ttyACM0" not in _ports_of(results[0])


def test_all_entries_without_serial_number_return_empty_list(gripper_registry):
    gripper_registry({"UMI-006": "E6614C77000000BA"})

    results = cd.list_connected_gripper(
        enumerate_devices=_enumerator(
            [
                _port("/dev/ttyACM0", None, index=0),
                _port("/dev/ttyACM1", None, index=1),
            ]
        ),
    )

    assert results == []


def test_unregistered_gripper_is_listed_with_none_registry_name(gripper_registry):
    gripper_registry({"UMI-006": "E6614C77000000BA"})

    results = cd.list_connected_gripper(
        enumerate_devices=_enumerator(
            [_port("/dev/ttyACM0", "DEADBEEFDEADBEEF", index=0)]
        ),
    )

    assert len(results) == 1
    assert results[0].kind == "gripper"
    assert results[0].device_id == "DEADBEEFDEADBEEF"
    assert results[0].registry_name is None
    assert results[0].state == "available"


def test_gripper_order_of_first_appearance_is_preserved(gripper_registry):
    """Dedupe keeps the position of a serial's first port, not its last."""
    gripper_registry({})

    results = cd.list_connected_gripper(
        enumerate_devices=_enumerator(
            [
                _port("/dev/ttyACM0", "serial-a", index=0),
                _port("/dev/ttyACM1", "serial-b", index=1),
                _port("/dev/ttyACM2", "serial-a", index=2),
                _port("/dev/ttyACM3", "serial-c", index=3),
                _port("/dev/ttyACM4", "serial-b", index=4),
            ]
        ),
    )

    assert [r.device_id for r in results] == ["serial-a", "serial-b", "serial-c"]


def test_gripper_enumerate_failure_returns_empty_list(gripper_registry):
    gripper_registry({"UMI-006": "E6614C77000000BA"})

    results = cd.list_connected_gripper(
        enumerate_devices=_raiser(OSError("no permission on /dev/ttyACM0")),
    )

    assert results == []


def test_no_connected_grippers_returns_empty_list(gripper_registry):
    gripper_registry({"UMI-006": "E6614C77000000BA"})

    results = cd.list_connected_gripper(enumerate_devices=_enumerator([]))

    assert results == []


# Use USB location to group a gripper, OAK and phone on the same hub.
# Omit location metadata when the operating system cannot provide it.


class FakeProtocol:
    def __init__(self, name):
        self.name = name


class FakeUsbDeviceInfo(FakeDeviceInfo):
    """depthai's DeviceInfo as the real SDK fills it in for a USB board."""

    def __init__(self, device_id, name, protocol="X_LINK_USB_VSC"):
        super().__init__(device_id)
        self.name = name
        self.protocol = FakeProtocol(protocol)


def test_a_usb_oak_records_where_it_is_plugged_in(oak_registry):
    oak_registry({})

    results = cd.list_connected_oak(
        enumerate_devices=_enumerator(
            [FakeUsbDeviceInfo("1944301000000000A9", "0.1.3.3")]
        ),
        open_device=lambda info: FakeDevice(features=[FakeFeature("CAM_A", "AR0234")]),
    )

    assert results[0].metadata["usb_location"] == "0-1.3.3"
    assert results[0].metadata["sockets"] == {"CAM_A": "AR0234"}


def test_a_poe_oak_has_no_usb_location(oak_registry):
    """A network board's name is an IP address, which must not read as a port."""
    oak_registry({})

    results = cd.list_connected_oak(
        enumerate_devices=_enumerator(
            [FakeUsbDeviceInfo("poe", "192.168.1.10", protocol="X_LINK_TCP_IP")]
        ),
        open_device=lambda info: FakeDevice(),
    )

    assert "usb_location" not in results[0].metadata


def test_a_gripper_records_its_usb_location_once(gripper_registry):
    gripper_registry({})
    first = _port("/dev/ttyACM0", "534A3035000000B8", index=0)
    second = _port("/dev/ttyACM1", "534A3035000000B8", index=1)
    first["location"] = "1-1.3.4:1.0"
    second["location"] = "1-1.3.4:1.2"

    results = cd.list_connected_gripper(enumerate_devices=_enumerator([first, second]))

    assert results[0].metadata["usb_location"] == "1-1.3.4"


def test_a_gripper_without_a_location_keeps_its_old_metadata(gripper_registry):
    gripper_registry({})

    results = cd.list_connected_gripper(
        enumerate_devices=_enumerator([_port("/dev/ttyACM0", "serial-a")])
    )

    assert results[0].metadata == {"ports": ["/dev/ttyACM0"]}


def test_an_iphone_is_located_by_its_udid_without_the_dash(iphone_registry):
    """The OS lists an iPhone's USB serial as its UDID with the dash removed."""
    iphone_registry({})

    results = cd.list_connected_iphone(
        enumerate_devices=_enumerator(
            [
                FakePhone("00008130-00000000000000A1", 0x12A8),
                FakePhone("00008130-00000000000000A2", 0x12A8),
            ]
        ),
        usb_locations=lambda: {"0000813000000000000000A1": "0-1.3.1"},
    )

    located = _by_id(results)
    assert located["00008130-00000000000000A1"].metadata["usb_location"] == "0-1.3.1"
    assert "usb_location" not in located["00008130-00000000000000A2"].metadata


# --------------------------------------------------------------------------- #
# ConnectedDevice dataclass
# --------------------------------------------------------------------------- #


def test_connected_device_is_frozen():
    device = cd.ConnectedDevice(
        kind="oak",
        device_id="1844301041",
        registry_name="oak-ffc-01",
        state="available",
        metadata={"sockets": {}},
    )

    with pytest.raises(Exception):
        device.device_id = "mutated"

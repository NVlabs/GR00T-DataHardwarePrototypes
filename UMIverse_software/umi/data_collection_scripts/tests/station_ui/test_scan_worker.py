# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The two device-open bodies in `station_ui/scan_worker.py`, against a fake SDK.

There is deliberately **no OAK test here**: `scan_worker._probers()` has no
`oak` entry, because `list_connected_oak` already performs the exclusive
`dai.Device(info)` open while enumerating and a second open would only be
slower.

**No test in this file may open real hardware.** The gripper's
`UartSensorReader` and the iPhone's `IphoneCamera` are both imported *inside*
the function under test, so the fakes are installed on the module (and on
`sys.modules`) and are picked up at call time.
"""

from __future__ import annotations

import io
import sys
from types import SimpleNamespace

import pytest

from station_ui import scan_worker

RAW_GRIPPER_SERIAL = "1E00210000000000000000BB"
RAW_IPHONE_UDID = "00008140-000C11223344001C"


# ------------------------------------------------------------------- fakes


class _FakeUartSensorReader:
    """Records its kwargs and its start/stop, and nothing else opens a port."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.disconnect_reason = None
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def _install_reader(monkeypatch, factory):
    """Put `factory` behind `UartSensorReader` and hand back the instances.

    `open_gripper` does `from ... import UartSensorReader` inside the function,
    so patching the attribute on the module reaches it at call time.
    """
    from umi.common.hardware_sdk.sensor import uart_reader

    instances = []

    def _make(**kwargs):
        reader = factory(**kwargs)
        instances.append(reader)
        return reader

    monkeypatch.setattr(uart_reader, "UartSensorReader", _make)
    return instances


def _install_iphone(monkeypatch, factory):
    """The same for `open_iphone`, which imports a module that may be absent.

    `setitem` on `sys.modules` rather than `setattr` on the module: record3d is
    an optional dependency, so importing the real one is not something a test
    on any machine may rely on.
    """
    instances = []

    def _make(**kwargs):
        camera = factory(**kwargs)
        camera.kwargs = kwargs
        instances.append(camera)
        return camera

    monkeypatch.setitem(
        sys.modules,
        "umi.common.hardware_sdk.camera.iphone_camera",
        SimpleNamespace(IphoneCamera=_make),
    )
    return instances


# ----------------------------------------------------------------- gripper


@pytest.mark.parametrize("identifier", ["UMI-044", RAW_GRIPPER_SERIAL])
def test_the_gripper_is_opened_by_its_configured_identity(monkeypatch, identifier):
    """Registered name or raw serial, the port is opened by what the record says.

    The baudrate is pinned rather than read from the config: the reader's own
    default of 115200 opens the port and reads noise, which is a *passing*
    probe on a gripper the recording would fail to talk to.
    """
    instances = _install_reader(monkeypatch, _FakeUartSensorReader)

    scan_worker.open_gripper({"registry_name": identifier})
    scan_worker.open_gripper({"device_id": identifier})

    assert len(instances) == 2
    for reader in instances:
        assert reader.kwargs == {
            "uart_device_name": identifier,
            "baudrate": 1_000_000,
        }
        assert reader.started is True
        assert reader.stopped is True


def test_a_gripper_that_disconnects_during_the_probe_is_a_failure(monkeypatch):
    """Enumeration lists a port; only the read loop notices an unplugged cable.

    Without the settle the reason is still None and a dead gripper probes
    clean, which is the whole reason `GRIPPER_SETTLE_S` exists.
    """

    class _Disconnecting(_FakeUartSensorReader):
        def start(self):
            self.started = True
            self.disconnect_reason = "USB cable disconnected"

    _install_reader(monkeypatch, _Disconnecting)

    with pytest.raises(RuntimeError, match="USB cable disconnected"):
        scan_worker.open_gripper({"device_id": RAW_GRIPPER_SERIAL})


def test_a_permission_error_on_the_uart_port_is_a_failure(monkeypatch):
    """A root-only `/dev/ttyACM*` is the commonest way a healthy-looking rig
    fails at record time, and `start()` opens synchronously, which is what
    surfaces it."""

    class _Forbidden(_FakeUartSensorReader):
        def start(self):
            raise PermissionError("permission denied")

    _install_reader(monkeypatch, _Forbidden)

    with pytest.raises(PermissionError, match="permission denied"):
        scan_worker.open_gripper({"registry_name": "UMI-044"})


def test_the_reader_is_stopped_even_when_the_probe_fails(monkeypatch):
    """A probe that leaves the port open makes the next one fail for the wrong
    reason -- and a scan opens every gripper on the station in turn."""

    class _Disconnecting(_FakeUartSensorReader):
        def start(self):
            self.started = True
            self.disconnect_reason = "USB cable disconnected"

    class _Forbidden(_FakeUartSensorReader):
        def start(self):
            raise PermissionError("permission denied")

    disconnected = _install_reader(monkeypatch, _Disconnecting)
    with pytest.raises(RuntimeError):
        scan_worker.open_gripper({"device_id": RAW_GRIPPER_SERIAL})
    assert disconnected[0].stopped is True, "the disconnected port was left open"

    forbidden = _install_reader(monkeypatch, _Forbidden)
    with pytest.raises(PermissionError):
        scan_worker.open_gripper({"registry_name": "UMI-044"})
    assert forbidden[0].stopped is True, "the refused port was left open"


# ------------------------------------------------------------------ iPhone


class _FakeIphoneCamera:
    """A phone with Record3D open and streaming: frames arrive immediately."""

    frames = object()

    def __init__(self, **_kwargs):
        self.started = False
        self.stopped = False
        self.reads = 0

    def start_streaming(self):
        self.started = True

    def get_frame(self):
        self.reads += 1
        return self.frames

    def stop_streaming(self):
        self.stopped = True


class _FakeIphoneWithRecord3DClosed(_FakeIphoneCamera):
    """What the real camera does when the app is shut.

    `start_streaming` prints in red and returns -- it does not raise -- and
    every subsequent read is None. That combination would make a closed phone
    probe as Connected.
    """

    frames = None


def test_the_iphone_is_opened_by_its_raw_udid(monkeypatch):
    """Addressed by UDID, with no registry lookup on the way, so a phone that
    was never registered is probed rather than skipped."""
    from umi.utils import device_manager

    instances = _install_iphone(monkeypatch, _FakeIphoneCamera)
    monkeypatch.setattr(
        device_manager,
        "get_iphone_udid",
        lambda _name: pytest.fail("raw UDID must bypass the registry lookup"),
    )

    scan_worker.open_iphone({"device_id": RAW_IPHONE_UDID})

    assert [camera.kwargs for camera in instances] == [{"iphone_udid": RAW_IPHONE_UDID}]
    assert instances[0].started is True


def test_the_iphone_stream_is_stopped_even_when_the_start_raises(monkeypatch):
    """Record3D claims the phone exclusively, so a probe that starts and then
    walks away leaves the recording unable to open the camera it just proved."""

    class _Refusing(_FakeIphoneCamera):
        def start_streaming(self):
            raise RuntimeError("Record3D app is not open")

    instances = _install_iphone(monkeypatch, _Refusing)

    with pytest.raises(RuntimeError, match="Record3D app is not open"):
        scan_worker.open_iphone({"device_id": RAW_IPHONE_UDID})

    assert instances[0].stopped is True, "the stream was left claimed"


def test_a_phone_with_record3d_closed_is_refused(monkeypatch):
    """A phone without the Record3D app open is not Connected.

    `start_streaming` swallows the failure, so the probe has to read a frame.
    Without that the scan would greenlight a phone the recording cannot open --
    on the step whose entire job is the go/no-go before a shift.
    """
    monkeypatch.setattr(scan_worker, "IPHONE_FRAME_TIMEOUT_S", 0.05)
    instances = _install_iphone(monkeypatch, _FakeIphoneWithRecord3DClosed)

    with pytest.raises(RuntimeError, match="Record3D"):
        scan_worker.open_iphone({"device_id": RAW_IPHONE_UDID})

    assert instances[0].started is True, "the probe never even tried to stream"
    assert instances[0].stopped is True, "the stream was left claimed"


def test_the_refusal_names_the_thing_the_operator_has_to_do(monkeypatch):
    """ "Failed to open" sends somebody to check a cable. The app is the fix."""
    monkeypatch.setattr(scan_worker, "IPHONE_FRAME_TIMEOUT_S", 0.05)
    _install_iphone(monkeypatch, _FakeIphoneWithRecord3DClosed)

    with pytest.raises(RuntimeError) as raised:
        scan_worker.open_iphone({"device_id": RAW_IPHONE_UDID})

    message = str(raised.value)
    assert "Record3D app" in message and "USB stream" in message, message


def test_a_streaming_phone_still_passes(monkeypatch):
    """The probe must not become a wall: a phone that streams is Connected."""
    instances = _install_iphone(monkeypatch, _FakeIphoneCamera)

    scan_worker.open_iphone({"device_id": RAW_IPHONE_UDID})

    assert instances[0].reads >= 1, "the frame was never read"
    assert instances[0].stopped is True


def test_a_slow_first_frame_is_waited_for_rather_than_failed(monkeypatch):
    """The first frame after a stream comes up is not instantaneous, and a
    single read would fail a healthy phone on a slow wake."""

    class _SlowToWake(_FakeIphoneCamera):
        def get_frame(self):
            self.reads += 1
            return None if self.reads < 3 else object()

    instances = _install_iphone(monkeypatch, _SlowToWake)

    scan_worker.open_iphone({"device_id": RAW_IPHONE_UDID})

    assert instances[0].reads == 3
    assert instances[0].stopped is True


def test_the_probe_reports_the_refusal_through_the_state_the_page_renders(
    monkeypatch,
):
    """End of the chain: `probe` downgrades the record, so the scan reports
    Failed and the console reports the phone as unavailable."""
    monkeypatch.setattr(scan_worker, "IPHONE_FRAME_TIMEOUT_S", 0.05)
    _install_iphone(monkeypatch, _FakeIphoneWithRecord3DClosed)
    found = [
        {
            "kind": "iphone",
            "device_id": RAW_IPHONE_UDID,
            "registry_name": "iphone15p-04",
            "state": "available",
            "metadata": {},
        }
    ]

    scan_worker.probe(found, io.StringIO())

    assert found[0]["state"] == "unavailable"
    assert "Record3D" in found[0]["error"]

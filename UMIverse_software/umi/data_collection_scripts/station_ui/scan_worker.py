# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enumerate the station's hardware in a child process, narrating as it goes.

Run as `python -m station_ui.scan_worker`. It exists as a separate process
because every hardware open in this repo is untimed and reading an OAK's
sensor modules needs an *exclusive* connect. In-process that is a request
handler pinned until the operator gives up; as a child of `station_ui.jobs` it
is a pid with a wall-clock cap, a real exit code, and captured output.

Two audiences, one stdout: human progress lines for the operator, and one
final `RESULT_PREFIX` line carrying
the machine-readable result that `station_ui.devices.parse_result` reads. After
it, for a person at a terminal, the same devices grouped by gripper
(`station_ui.device_summary`).

`station_ui.core.device_scan.DeviceScanner` is the same enumeration behind a cache, built
for the config form where several fields ask about one scan. A child process
runs exactly once and then exits, so there is nothing to cache, and calling
the three enumerators directly is what lets this report each kind as it
finishes rather than all three at the end.
"""

from __future__ import annotations

import json
import sys
import time

from umi.common.hardware_sdk.device_discovery import (
    list_connected_gripper,
    list_connected_iphone,
    list_connected_oak,
    list_connected_uvc,
)

__all__ = [
    "RESULT_PREFIX",
    "main",
    "open_gripper",
    "open_iphone",
    "print_summary",
    "scan",
]

# Prefix rather than "the last line": the child may print anything at all after
# the result (a driver warning on teardown is common), and a positional rule
# would then parse a warning as the device list.
RESULT_PREFIX = "##station-ui-devices## "

# Probe at the recording baud rate. The UART reader's 115200 default would
# open the port without decoding the configured gripper protocol.
GRIPPER_BAUDRATE = 1_000_000
# The reader's read loop needs a moment to notice a cable that is already
# unplugged; without it `disconnect_reason` is still None and a dead gripper
# probes clean.
GRIPPER_SETTLE_S = 0.02

# Allow extra time for the phone's first frame after start_streaming's
# initial one-second wait.
IPHONE_FRAME_TIMEOUT_S = 3.0
IPHONE_FRAME_POLL_S = 0.1


def _enumerators():
    """(kind, label, callable), looked up at call time.

    Module globals rather than a table built at import, so a test can replace
    one enumerator and have this function see the replacement.
    """
    return (
        ("oak", "OAK cameras", list_connected_oak),
        ("iphone", "iPhones", list_connected_iphone),
        ("gripper", "UMI grippers", list_connected_gripper),
        ("uvc", "UVC cameras", list_connected_uvc),
    )


def _record(device) -> dict:
    """The device, reduced to what the page can act on.

    `metadata` is carried through because it is the only place the per-socket
    sensor module lives, and that is what decides whether a stream is written
    640x400 or 640x480 -- an operator picking an OAK on /config has to be able
    to see the modules on the board they are about to select. Serialized as a
    plain dict: the enumerators return a Mapping, and json refuses anything else.
    """
    return {
        "kind": device.kind,
        "device_id": device.device_id,
        "registry_name": device.registry_name,
        "state": device.state,
        "metadata": dict(device.metadata or {}),
    }


# ------------------------------------------------------------------- probing


def open_gripper(record) -> None:
    """Open the gripper's UART port and close it again, or raise.

    Enumeration is only a port listing, so it says nothing about whether the
    recording user can *open* the port, and a root-only `/dev/ttyACM*` is the
    single most common way a rig that looks healthy fails at record time.
    `start()` opens synchronously, which is what surfaces the permission
    error; the settle then lets the read loop report an already-dead cable.
    """
    from umi.common.hardware_sdk.sensor.uart_reader import UartSensorReader

    reader = UartSensorReader(
        uart_device_name=record.get("registry_name") or record.get("device_id"),
        baudrate=GRIPPER_BAUDRATE,
    )
    try:
        reader.start()
        time.sleep(GRIPPER_SETTLE_S)
        reason = reader.disconnect_reason
        if reason:
            raise RuntimeError(reason)
    finally:
        reader.stop()


def open_iphone(record) -> None:
    """Stream one frame off the iPhone and stop again, or raise.

    `Record3DStream.get_connected_devices()` lists a phone that is plugged in
    but has the Record3D app closed, or is locked, or is already claimed by
    another process. Only pulling a frame tells those apart. Addressed by UDID
    rather than registry name so an unregistered phone is probed too.

    `start_streaming()` alone is not the test: on failure it prints "Failed to
    get frame. Please toggle streaming in iPhone." in red and returns without
    raising, so a phone with Record3D closed would otherwise be reported as
    `Connected`. The camera driver keeps that non-raising path on purpose: the
    recorder depends on it, and changing it has a much wider blast radius than
    this probe.

    Polled rather than read once: `start_streaming` already sleeps a second, but
    the first frame after a stream comes up is not instantaneous and a single
    miss would fail a healthy phone.
    """
    from umi.common.hardware_sdk.camera.iphone_camera import IphoneCamera

    camera = IphoneCamera(iphone_udid=record.get("device_id"))
    try:
        camera.start_streaming()
        deadline = time.monotonic() + IPHONE_FRAME_TIMEOUT_S
        while camera.get_frame() is None:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "no frame from Record3D — open the Record3D app on the "
                    "phone and start the USB stream"
                )
            time.sleep(IPHONE_FRAME_POLL_S)
    finally:
        try:
            camera.stop_streaming()
        except Exception:  # noqa: BLE001 - the start failure is the real news
            pass


def _probers():
    """kind -> opener, looked up at call time so a test can replace one.

    No entry for `oak`: `list_connected_oak` already performs an exclusive
    `dai.Device(info)` open to read the sensor modules, and opening it a second
    time here would just be slower.

    The other drivers were checked for the same trap `open_iphone` documents
    (a failed open reported as a print and a bare `return`, which a probe reads
    as success). The iPhone is the only one:

      * `UvcCamera.start_streaming` raises RuntimeError on a refused open;
      * `UartSensorReader.start` raises out of `_open_port`, which is what
        surfaces the root-only `/dev/ttyACM*`.

    The ~50 other `print_red(...)` / `return None` pairs in the camera drivers
    are all inside `get_frame`, where None is the deliberate per-frame contract
    the recorder polls on. Raising there would turn a dropped frame into a dead
    session, so they are deliberately left alone.
    """
    return {"gripper": open_gripper, "iphone": open_iphone}


def probe(found, out) -> None:
    """Open each enumerated device in turn, downgrading the ones that refuse.

    Mutates `found` in place: a device that will not open is exactly the
    `unavailable` state `list_connected_oak` already uses for the same
    condition, so the page renders it `Failed` through the path it already has.
    """
    for record in found:
        opener = _probers().get(record.get("kind"))
        if opener is None or record.get("state") != "available":
            continue
        name = record.get("registry_name") or record.get("device_id")
        print(f"[scan]   opening {name}...", file=out, flush=True)
        try:
            opener(record)
        except Exception as exc:  # noqa: BLE001 - the reason is the whole point
            record["state"] = "unavailable"
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[scan]   {name}: {record['error']}", file=out, flush=True)
        else:
            print(f"[scan]   {name}: opened and released", file=out, flush=True)


def scan(out=None, *, probe_devices: bool = False) -> list[dict]:
    """Enumerate every kind, printing progress, and return what was found.

    A kind whose enumerator raises contributes nothing and is *announced*:
    `list_connected_*` already treats a probe failure as an empty list, so
    without this line a station missing `depthai` would show an OAK rig as
    simply absent, with no hint that the SDK is the problem.

    With `probe_devices`, each enumerated iPhone and gripper is then actually
    opened -- see `probe`. Off by default so the enumeration stays usable on
    its own; the page always asks for it.
    """
    out = sys.stdout if out is None else out
    found: list[dict] = []

    for kind, label, enumerate_kind in _enumerators():
        print(f"[scan] {label}...", file=out, flush=True)
        try:
            devices = list(enumerate_kind())
        except Exception as exc:  # noqa: BLE001 - the reason is the useful part
            print(f"[scan]   {label}: enumeration failed: {exc}", file=out, flush=True)
            continue
        if not devices:
            print(f"[scan]   no {label.lower()} found", file=out, flush=True)
        for device in devices:
            name = device.registry_name or device.device_id
            print(f"[scan]   {name} ({device.state})", file=out, flush=True)
            found.append(_record(device))

    if probe_devices:
        print(
            "[scan] opening each device to confirm it is usable...",
            file=out,
            flush=True,
        )
        probe(found, out)

    print(f"[scan] {len(found)} device(s) found", file=out, flush=True)
    print(RESULT_PREFIX + json.dumps({"devices": found}), file=out, flush=True)
    return found


def print_summary(found, out=None) -> None:
    """The devices grouped by gripper, after everything else the scan printed.

    Never raises: the result line is already out, and a table that cannot be
    built must not turn a scan that worked into a failed one.
    """
    out = sys.stdout if out is None else out
    print("", file=out)
    print(
        "[scan] Devices by USB port (a gripper's camera and phone share its row):",
        file=out,
    )
    try:
        # Here, not at the top: `station_ui.devices` imports RESULT_PREFIX from
        # this module.
        from station_ui import device_summary

        lines = device_summary.table_lines(found)
    except Exception as exc:  # noqa: BLE001 - the reason is the useful part
        lines = [f"Device information could not be loaded: {exc}"]
    for line in lines:
        print(line, file=out, flush=True)


PROBE_FLAG = "--probe"


def main(argv=None, out=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        found = scan(out=out, probe_devices=PROBE_FLAG in argv)
    except Exception as exc:  # noqa: BLE001 - a message, not a traceback
        print(
            f"[scan] Device information could not be loaded: {exc}",
            file=sys.stdout if out is None else out,
            flush=True,
        )
        return 1
    print_summary(found, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

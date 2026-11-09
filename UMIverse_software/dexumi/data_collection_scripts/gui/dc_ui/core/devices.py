# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connected iPhones, OAK cameras and DexUMI exoskeleton boards.

Cameras are listed by their own ids (iPhone UDID, OAK device ID); the recording
config names them the same way. An exoskeleton board is matched to the one
calibration in config/dexumi_exoskeleton/ whose ``device_serial`` is its USB
serial; that calibration's file name is the board's name.

Three independent enumerations, so one hang cannot blank the others:

- ``oak``    -- subprocess (``python -m gui.dc_ui.core.devices --json --kind oak``):
  depthai enumeration can hang and cannot be interrupted from a thread, and
  the GUI process must never own a camera handle. ``getAllConnectedDevices``
  is used (not ``getAllAvailableDevices``): the latter hides a camera another
  process holds, which must show as "in use", not vanish.
- ``iphone`` -- separate subprocess: record3d's compiled extension talks to
  usbmuxd and prints to stdout; only the last JSON line is parsed.
- ``exo``    -- in-process: ``serial.tools.list_ports.comports()`` is a sysfs
  read (~250 ms), never opens a port, and cannot hang. One board shows up as
  two CDC ports sharing the USB serial; the list is deduped by serial. The
  board's calibration yaml is read to predict the recorder's launch gates
  (simulation_only, geometry_verified).

Every failure becomes a string in ``errors``; nothing here raises.
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import yaml

from gui.dc_ui.core.composer import EXO_CONFIG_DIR, exo_config_file, repo_path
from gui.dc_ui.core.kinds import (
    DEVICE_KINDS,
    EXO_NAME_RE,
    REPO_ROOT,
    DetectedDevice,
)

ENUM_MODULE = "gui.dc_ui.core.devices"
DEFAULT_TIMEOUT_S = 20
SUBPROCESS_KINDS = ("oak", "iphone")
# Same descriptors UartSensorReader.list_connected_devices filters on.
PICO_DESCRIPTION_PATTERNS = ("pico", "raspberry pi pico", "rp2040")


# Raw device enumeration


def _iphone_list():
    from umi.common.hardware_sdk.camera.iphone_camera import IphoneCamera

    return [str(dev.udid) for dev in IphoneCamera.list_connected_devices()]


def oak_records(device_infos):
    """depthai DeviceInfo objects -> the dict shape OAKCamera.list_connected_devices returns."""
    records = []
    for index, info in enumerate(device_infos or []):
        records.append(
            {
                "name": getattr(info, "name", None),
                "device_id": info.getDeviceId(),
                "index": index,
                "state": str(getattr(info, "state", "")),
            }
        )
    return records


def _oak_list():
    import depthai as dai

    lister = getattr(dai.Device, "getAllConnectedDevices", None) or getattr(
        dai.Device, "getAllAvailableDevices"
    )
    return oak_records(lister())


def pico_ports(ports):
    """pyserial ListPortInfo objects -> Pico-family ports (no printing, no port opened)."""
    records = []
    for port in ports or []:
        description = (getattr(port, "description", None) or "").lower()
        if not any(pattern in description for pattern in PICO_DESCRIPTION_PATTERNS):
            continue
        records.append(
            {
                "device": getattr(port, "device", None),
                "description": getattr(port, "description", None),
                "serial_number": getattr(port, "serial_number", None),
                "hwid": getattr(port, "hwid", None),
            }
        )
    return records


def _pico_list():
    import serial.tools.list_ports

    return pico_ports(serial.tools.list_ports.comports())


_LISTER_NAMES = {"iphone": "_iphone_list", "oak": "_oak_list", "exo": "_pico_list"}


# -- exoskeleton calibrations -------------------------------------------------------


def exo_calibrations():
    """``{calibration name: device_serial}`` of every calibration that records a board."""
    from umi.utils.device_manager import exoskeleton_serials

    return exoskeleton_serials()


def names_by_serial(calibrations):
    """board serial -> sorted calibration names recording it."""
    by_serial = {}
    for name, serial in (calibrations or {}).items():
        by_serial.setdefault(str(serial), []).append(name)
    return {serial: sorted(names) for serial, names in by_serial.items()}


# -- joins -------------------------------------------------------------------------


def exo_side(name):
    match = EXO_NAME_RE.match(name or "")
    return match.group(1) if match else None


def join_iphones(udids):
    devices = []
    for udid in udids or []:
        udid = str(udid or "")
        if not udid:
            continue
        devices.append(
            DetectedDevice(
                kind="iphone",
                device_id=udid,
                name=udid,
                label=label_for("iphone", udid),
            )
        )
    return devices


def _short_state(state):
    text = str(state or "")
    if not text:
        return None
    return text.rsplit(".", 1)[-1].replace("X_LINK_", "")


KIND_LABELS = {"iphone": "iPhone", "oak": "OAK"}


def label_for(kind, device_id, state=None):
    label = f"{KIND_LABELS[kind]} {device_id}"
    if state and "BOOTED" in state and "UNBOOTED" not in state:
        label += " [BOOTED = in use]"
    return label


def join_oaks(raw):
    devices = []
    for record in raw or []:
        device_id = str(record.get("device_id") or record.get("serial") or "")
        if not device_id:
            continue
        state = _short_state(record.get("state"))
        devices.append(
            DetectedDevice(
                kind="oak",
                device_id=device_id,
                name=device_id,
                label=label_for("oak", device_id, state),
                in_use=state == "BOOTED",
            )
        )
    return devices


def read_calibration(rel_path):
    """The calibration yaml as a dict, or None when absent / unreadable."""
    try:
        with open(repo_path(rel_path)) as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None


def _calibration_issue(calib):
    # Mirrors DexUmiExoskeleton.validate_config_file: the recorder refuses both.
    if calib.get("simulation_only") is True:
        return "simulation_only: true"
    if calib.get("geometry_verified") is False:
        return "geometry_verified: false"
    return None


def _exo_label(serial, names):
    if len(names) == 1:
        return f"{names[0]} ({serial})"
    if names:
        return (
            f"BOARD {serial} — recorded by several calibrations "
            f"({', '.join(names)}); keep the one for this board"
        )
    return (
        f"UNCALIBRATED BOARD {serial} — no calibration in {EXO_CONFIG_DIR} records it"
    )


def join_exos(raw, calibrations, read_config=None):
    """Boards (deduped by serial) named by the one calibration recording each serial."""
    if read_config is None:
        read_config = read_calibration
    by_serial = names_by_serial(calibrations)
    devices, seen = [], set()
    for port in raw or []:
        serial = str(port.get("serial_number") or "")
        if not serial or serial in seen:
            continue
        seen.add(serial)
        names = by_serial.get(serial, [])
        name = names[0] if len(names) == 1 else None
        config_path = exo_config_file(name) if name else None
        calib = read_config(config_path) if config_path else None
        devices.append(
            DetectedDevice(
                kind="exo",
                device_id=serial,
                name=name,
                label=_exo_label(serial, names),
                side=exo_side(name),
                config_path=config_path,
                config_issue=_calibration_issue(calib) if calib else None,
            )
        )
    return devices


_JOINERS = {"iphone": join_iphones, "oak": join_oaks}


# -- enumeration -------------------------------------------------------------------


def enumerate_kind(kind):
    """In-process enumeration of one kind -> {"devices": [DetectedDevice], "errors": [str]}."""
    result = {"devices": [], "errors": []}
    if kind not in DEVICE_KINDS:
        result["errors"].append(f"unknown device kind {kind!r}; kinds: {DEVICE_KINDS}")
        return result
    lister = globals()[_LISTER_NAMES[kind]]
    try:
        raw = lister()
    except ImportError as exc:
        result["errors"].append(f"{kind}: SDK not installed ({exc})")
        return result
    except Exception as exc:  # noqa: BLE001 - a hung/odd device must not kill the list
        result["errors"].append(f"{kind}: enumeration failed: {exc}")
        return result
    if kind != "exo":
        result["devices"] = _JOINERS[kind](raw)
        return result
    try:
        calibrations = exo_calibrations()
    except Exception as exc:  # noqa: BLE001 - the page still lists the board serials
        calibrations = {}
        result["errors"].append(f"exo: exoskeleton calibrations unreadable: {exc}")
    result["devices"] = join_exos(raw, calibrations)
    return result


def _to_json(result):
    return {
        "devices": [d.to_dict() for d in result["devices"]],
        "errors": list(result["errors"]),
    }


def enumeration_argv(kind):
    return [
        sys.executable,
        "-m",
        ENUM_MODULE,
        "--json",
        "--kind",
        kind,
        "--parent-pid",
        str(os.getpid()),
    ]


def last_json_line(stdout):
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                return None
    return None


def discover_kind_subprocess(kind, timeout_s=DEFAULT_TIMEOUT_S, argv=None, cwd=None):
    # Run enumeration in a separate process using this checkout and interpreter.
    # Tests may override argv and cwd.
    empty = {"devices": [], "errors": []}
    try:
        proc = subprocess.run(
            list(argv) if argv else enumeration_argv(kind),
            cwd=str(cwd or REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        empty["errors"].append(
            f"{kind}: enumeration timed out after {timeout_s}s (a device may be hung; replug it)"
        )
        return empty
    except OSError as exc:
        empty["errors"].append(f"{kind}: enumeration could not start: {exc}")
        return empty
    payload = last_json_line(proc.stdout)
    if payload is None:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
        empty["errors"].append(
            f"{kind}: enumeration exited {proc.returncode} without JSON: {' | '.join(tail)}"
        )
        return empty
    return {
        "devices": [DetectedDevice.from_dict(d) for d in payload.get("devices") or []],
        "errors": list(payload.get("errors") or []),
    }


def _default_runner(kind, timeout_s):
    if kind in SUBPROCESS_KINDS:
        return discover_kind_subprocess(kind, timeout_s=timeout_s)
    return enumerate_kind(kind)


def discover_devices(timeout_s=DEFAULT_TIMEOUT_S, runner=None):
    """All kinds -> {"iphone": [...], "oak": [...], "exo": [...], "errors": [...], "at": epoch}.

    Every call re-reads the exoskeleton calibrations, so a hand calibrated since
    the last scan shows up by name.
    """
    runner = runner or _default_runner
    with ThreadPoolExecutor(max_workers=len(DEVICE_KINDS)) as pool:
        futures = {kind: pool.submit(runner, kind, timeout_s) for kind in DEVICE_KINDS}
        results = {kind: future.result() for kind, future in futures.items()}
    out = {kind: list(results[kind]["devices"]) for kind in DEVICE_KINDS}
    out["errors"] = [err for kind in DEVICE_KINDS for err in results[kind]["errors"]]
    out["at"] = time.time()
    return out


def payload(result):
    """discover_devices result -> JSON-safe dict for the API."""
    return {
        **{kind: [d.to_dict() for d in result[kind]] for kind in DEVICE_KINDS},
        "errors": list(result["errors"]),
        "at": result["at"],
    }


def from_payload(data):
    """Inverse of payload(): a persisted detection -> DetectedDevice objects (None if unusable)."""
    if not isinstance(data, dict):
        return None
    result = {}
    for kind in DEVICE_KINDS:
        result[kind] = [
            DetectedDevice.from_dict(item)
            for item in data.get(kind) or []
            if isinstance(item, dict)
        ]
    result["errors"] = list(data.get("errors") or [])
    result["at"] = data.get("at")
    return result


# -- CLI ---------------------------------------------------------------------------


def _arg(args, flag):
    if flag in args:
        i = args.index(flag)
        return args[i + 1] if i + 1 < len(args) else None
    return None


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    expected_parent = _arg(args, "--parent-pid")
    if expected_parent is not None:
        # An SDK enumeration can hang below Python. The parent's normal
        # timeout reaps it, and this also releases it if the GUI dies first.
        from gui.dc_ui.core.camera_pairing import _arm_parent_death

        _arm_parent_death(int(expected_parent))
    kind = _arg(args, "--kind")
    if kind is not None and kind not in DEVICE_KINDS:
        print(f"unknown --kind {kind!r}; kinds: {DEVICE_KINDS}", file=sys.stderr)
        return 2
    if kind is not None:
        result = enumerate_kind(kind)
        if "--json" in args:
            print(json.dumps(_to_json(result)))
            return 0
        for device in result["devices"]:
            print(device.label)
        for error in result["errors"]:
            print(f"error: {error}")
        return 0
    # Without --kind, enumerate all device types in the current process.
    result = discover_devices(runner=lambda k, t: enumerate_kind(k))
    if "--json" in args:
        print(json.dumps(payload(result)))
        return 0
    for k in DEVICE_KINDS:
        for device in result[k]:
            print(f"{k}: {device.label}")
    for error in result["errors"]:
        print(f"error: {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One enumeration pass, shared by every device field on the screen.

Scanning is not free: reading an OAK's sensor modules needs an exclusive
connect at roughly 1-2s per camera. So it runs once when the configuration page
opens and again only on Refresh — never per field.
"""

import time
from typing import Callable, Dict, List, Optional

from umi.common.hardware_sdk.device_discovery import (
    ConnectedDevice,
    DeviceKind,
    list_connected_gripper,
    list_connected_iphone,
    list_connected_oak,
)


class DeviceScanner:
    def __init__(
        self,
        scan_oak: Optional[Callable] = None,
        scan_iphone: Optional[Callable] = None,
        scan_gripper: Optional[Callable] = None,
    ):
        self._scans = {
            "oak": scan_oak or list_connected_oak,
            "iphone": scan_iphone or list_connected_iphone,
            "gripper": scan_gripper or list_connected_gripper,
        }
        self._by_kind: Dict[DeviceKind, List[ConnectedDevice]] = {}
        self._last_scan: Optional[float] = None

    def scan(self) -> None:
        """Refresh every kind. A failing enumerator clears that kind rather
        than leaving stale results on screen."""
        for kind, fn in self._scans.items():
            try:
                self._by_kind[kind] = list(fn())
            except Exception:
                self._by_kind[kind] = []
        self._last_scan = time.time()

    def devices(self, kind: DeviceKind) -> List[ConnectedDevice]:
        return list(self._by_kind.get(kind, []))

    def by_identifier(self, value: str) -> Optional[ConnectedDevice]:
        """Look up by registry name or device id.

        Configs store whichever identifies the device: unregistered hardware
        has no name, so it is stored by id. This is the only lookup — a
        name-only variant would return None for every unregistered device and
        give its OAK the wrong resolution.
        """
        if not value:
            return None
        for devices in self._by_kind.values():
            for dev in devices:
                if dev.registry_name == value or dev.device_id == value:
                    return dev
        return None

    def last_scan_time(self) -> Optional[float]:
        return self._last_scan

    @classmethod
    def from_records(cls, records) -> "DeviceScanner":
        """A scanner holding a scan that already happened, somewhere else.

        The station console does not scan in-process: it runs
        `station_ui.scan_worker` as a capped child and keeps the parsed result
        in the job registry, because reading an OAK's sensor modules needs an
        exclusive connect and no hardware open in this repo has a timeout.

        The module-global `SCANNER` is never filled in that process, so
        `resolution_for` -- which reads it -- would find nothing and every OAK
        would fall back to the 16:10 default. This is the bridge: the child's
        records, in the shape the resolver reads.

        `state` is carried through rather than filtered on. A device that
        enumerated but would not open is still a device whose sensor modules
        were read, and its aspect is as knowable as any other -- refusing it a
        resolution here would answer a hardware fault with a silently cropped
        FOV.
        """
        scanner = cls()
        for record in records or ():
            if not isinstance(record, dict):
                continue
            kind = record.get("kind")
            if not kind:
                continue
            scanner._by_kind.setdefault(kind, []).append(
                ConnectedDevice(
                    kind=kind,
                    device_id=str(record.get("device_id") or ""),
                    registry_name=record.get("registry_name") or None,
                    state=record.get("state") or "available",
                    metadata=dict(record.get("metadata") or {}),
                )
            )
        scanner._last_scan = time.time()
        return scanner


SCANNER = DeviceScanner()


# Output width is 640 pixels at the sensor's native aspect ratio so the ISP
# does not crop the field of view. AR0234/OV9782/OV9282 use 640x400; IMX378
# uses 640x480. Sensor mode selection is separate: oak_device selects
# 2024x1520 for IMX378 to retain the full binned array and an 85 Hz ceiling.
SENSOR_STREAM_RESOLUTION: Dict[str, List[int]] = {
    "AR0234": [640, 400],  # 1920x1200 native, 16:10
    "OV9782": [640, 400],  # 1280x800  native, 16:10
    "OV9282": [640, 400],  # 1280x800  native, 16:10
    "IMX378": [640, 480],  # 4056x3040 native, 4:3
}

# Fallback aspect ratio for the 16:10 wrist and trigger modules.
DEFAULT_STREAM_RESOLUTION: List[int] = [640, 400]

# Fallback for the supported 4:3 IMX378 head camera; 16:10 would crop its view.
HEAD_STREAM_RESOLUTION: List[int] = [640, 480]


def sensor_module_for(
    device_name: Optional[str],
    socket: str,
    scanner: Optional[DeviceScanner] = None,
) -> str:
    """The sensor module the scan found on one socket of one OAK, or ``""``.

    The fact ``resolution_for`` derives its answer from, and then throws away:
    640x480 means an IMX378 on that socket, and nothing downstream could say so.
    Callers write it into the generated config beside the size, so a session
    records what it was really shot with rather than only the number.

    ``""`` for every miss -- no scan, an unknown board, a socket the scan did
    not report -- and never a guess. A module is not derivable from a fallback
    resolution, so the honest answer is silence and the caller omits the key.

    No probing: the scanner enumerated modules when the scan ran.
    """
    scanner = scanner if scanner is not None else SCANNER
    device = scanner.by_identifier(device_name) if device_name else None
    if device is None:
        return ""
    # Treat a null sockets entry as empty rather than calling get on None.
    return str(((device.metadata or {}).get("sockets") or {}).get(socket) or "")


def resolution_for(
    device_name: Optional[str],
    socket: str,
    scanner: Optional[DeviceScanner] = None,
    *,
    default: Optional[List[int]] = None,
) -> tuple:
    """Aspect-correct stream resolution for one socket of one OAK device.

    Returns ``(resolution, warning)``. ``warning`` is None on success, else a
    string naming why the default was used — the caller surfaces it rather than
    silently writing a resolution that crops.

    ``default`` is what a miss falls back to: the wrist/trigger size unless the
    caller knows better (the head block passes ``HEAD_STREAM_RESOLUTION``). A
    miss is still a miss -- the warning is returned either way -- but the
    fallback is then right for the head camera too.

    No probing happens here: the scanner enumerated sensor modules when the
    scan ran, before the operator picked anything.
    """
    scanner = scanner if scanner is not None else SCANNER
    fallback = list(default if default is not None else DEFAULT_STREAM_RESOLUTION)

    device = scanner.by_identifier(device_name) if device_name else None
    if device is None:
        return list(fallback), (
            f"{device_name or '(unnamed)'}: not found in the device scan — "
            f"resolution unverified, using {fallback}"
        )

    # Treat a null sockets entry as empty rather than calling get on None.
    sensor = ((device.metadata or {}).get("sockets") or {}).get(socket)
    if not sensor:
        return list(fallback), (
            f"{device_name}: no sensor module reported on {socket} — using {fallback}"
        )

    resolution = SENSOR_STREAM_RESOLUTION.get(sensor)
    if resolution is None:
        return list(fallback), (
            f"{device_name} {socket}: unknown sensor module {sensor!r} — "
            f"using {fallback}, which may crop its FOV"
        )
    # Copy: callers put this straight into a config dict and may mutate it.
    return list(resolution), None

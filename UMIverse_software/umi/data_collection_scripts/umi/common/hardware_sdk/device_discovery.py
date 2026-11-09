# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live discovery of OAK cameras, iPhones, and UMI grippers.

Hardware-specific libraries are imported lazily so callers can use the shared
``ConnectedDevice`` type without installing every device SDK.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping

from umi.common.hardware_sdk import usb_location

DeviceKind = Literal["oak", "iphone", "gripper", "uvc"]
DeviceState = Literal["available", "unavailable"]

__all__ = [
    "ConnectedDevice",
    "DeviceKind",
    "DeviceState",
    "list_connected_gripper",
    "list_connected_iphone",
    "list_connected_oak",
    "list_connected_uvc",
]


@dataclass(frozen=True)
class ConnectedDevice:
    """Normalized identity, state, and provider-specific metadata.

    Metadata contains ``sockets`` for OAK, ``product_id`` for iPhone,
    ``ports`` for grippers, and ``name``/``path`` for UVC. Every kind also
    carries ``usb_location`` (``"1-1.3.4"``, see ``usb_location``) when the SDK
    or the OS can say where the device is plugged in.
    """

    kind: DeviceKind
    device_id: str
    registry_name: str | None
    state: DeviceState
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _oak_registry() -> dict[str, str]:
    from umi.utils.device_registry import get_oak_devices

    return get_oak_devices()


def _iphone_registry() -> dict[str, str]:
    from umi.utils.device_registry import get_iphone_devices

    return get_iphone_devices()


def _gripper_registry() -> dict[str, str]:
    from umi.utils.device_registry import get_umi_v4_grippers

    return get_umi_v4_grippers()


def _registry_name_for(registry: Mapping[str, str], device_id: str) -> str | None:
    for name, ident in registry.items():
        if ident == device_id:
            return name
    return None


def _default_enumerate_oak():
    import depthai as dai

    return dai.Device.getAllAvailableDevices()


def _default_open_oak(info):
    import depthai as dai

    return dai.Device(info)


def _default_enumerate_iphone():
    from record3d import Record3DStream

    return Record3DStream.get_connected_devices()


def _default_enumerate_uvc():
    from umi.common.hardware_sdk.camera.uvc_camera import get_connected_cameras

    return get_connected_cameras()


def _default_enumerate_gripper():
    from umi.common.hardware_sdk.sensor.uart_reader import UartSensorReader

    return UartSensorReader.list_connected_devices()


def _enumerate_or_empty(enumerator: Callable[[], Iterable[Any]]) -> list[Any]:
    """Materialize one hardware enumeration, treating probe failure as empty."""
    try:
        return list(enumerator())
    except Exception:
        return []


def _oak_usb_location(info) -> str | None:
    """Where a board is plugged in; None for a PoE board, whose name is an IP."""
    protocol = getattr(getattr(info, "protocol", None), "name", "")
    if "USB" not in str(protocol):
        return None
    return usb_location.from_xlink_name(getattr(info, "name", None))


def list_connected_oak(
    enumerate_devices: Callable[[], Iterable[Any]] | None = None,
    open_device: Callable[[Any], Any] | None = None,
) -> list[ConnectedDevice]:
    """Attached OAK devices, each with its per-socket sensor module.

    Two tiers: enumeration is cheap, but reading sensor modules needs an
    exclusive connect (~1-2s each). A device that enumerates but will not open
    is reported unavailable rather than dropped, so the UI can say so.
    """
    enumerate_devices = enumerate_devices or _default_enumerate_oak
    open_device = open_device or _default_open_oak

    infos = _enumerate_or_empty(enumerate_devices)
    if not infos:
        return []

    registry = _oak_registry()

    def _probe(info) -> ConnectedDevice:
        device_id = info.getDeviceId()
        sockets: dict[str, str] = {}
        state: DeviceState = "available"
        device = None
        try:
            device = open_device(info)
            for feat in device.getConnectedCameraFeatures():
                sockets[feat.socket.name] = feat.sensorName
        except Exception:
            state = "unavailable"
            sockets = {}
        finally:
            if device is not None:
                try:
                    device.close()
                except Exception:
                    pass
        metadata: dict[str, Any] = {"sockets": sockets}
        location = _oak_usb_location(info)
        if location:
            metadata["usb_location"] = location
        return ConnectedDevice(
            kind="oak",
            device_id=device_id,
            registry_name=_registry_name_for(registry, device_id),
            state=state,
            metadata=metadata,
        )

    # Exclusive opens run concurrently; pool.map preserves enumeration order.
    with ThreadPoolExecutor(max_workers=len(infos)) as pool:
        return list(pool.map(_probe, infos))


def list_connected_gripper(
    enumerate_devices: Callable[[], Iterable[Any]] | None = None,
) -> list[ConnectedDevice]:
    """Attached UMI grippers, identified by their Pico serial number.

    One gripper enumerates as two consecutive ports (ttyACM0 + ttyACM1) sharing
    a serial number, so results are deduped by serial — otherwise every gripper
    would appear twice. Ports with no serial cannot be identified and are
    skipped. Enumeration is a port listing, so this is fast: no device is opened.
    """
    enumerate_devices = enumerate_devices or _default_enumerate_gripper
    entries = _enumerate_or_empty(enumerate_devices)
    if not entries:
        return []

    registry = _gripper_registry()
    by_serial: dict[str, list[str]] = {}
    locations: dict[str, str] = {}
    for entry in entries:
        serial = entry.get("serial_number")
        if not serial:
            continue
        ports = by_serial.setdefault(serial, [])
        port = entry.get("device")
        if port:
            ports.append(port)
        location = usb_location.from_pyserial(entry.get("location"))
        if location:
            locations.setdefault(serial, location)

    def _metadata(serial, ports) -> dict[str, Any]:
        metadata: dict[str, Any] = {"ports": sorted(set(ports))}
        if serial in locations:
            metadata["usb_location"] = locations[serial]
        return metadata

    return [
        ConnectedDevice(
            kind="gripper",
            device_id=serial,
            registry_name=_registry_name_for(registry, serial),
            state="available",
            metadata=_metadata(serial, ports),
        )
        for serial, ports in by_serial.items()
    ]


def list_connected_uvc(
    enumerate_devices: Callable[[], Iterable[Any]] | None = None,
    locate: Callable[[str], str | None] | None = None,
) -> list[ConnectedDevice]:
    """Every UVC camera v4l2-ctl reports, identified by its V4L2 index.

    There is no UVC registry, so ``registry_name`` is always None and the index
    is the identifier, which is what a ``uvc_camera`` source stores in
    ``params.camera_index`` and what ``cv2.VideoCapture`` takes.

    ``state`` is always available: enumeration here is a parse of v4l2-ctl
    output, with no open attempt to distinguish "here" from "here but wedged"
    the way the OAK probe can.

    v4l2-ctl names a camera's port under its PCI controller rather than its USB
    bus, so where each camera is plugged in comes from the OS, by its
    ``/dev/video`` path.
    """
    enumerate_devices = enumerate_devices or _default_enumerate_uvc
    locate = locate or usb_location.from_video_node

    def _metadata(cam) -> dict[str, Any]:
        # Both trigger cameras are usually the same model, so the path is
        # the only thing distinguishing two otherwise identical rows.
        metadata: dict[str, Any] = {
            "name": cam.get("name") or "",
            "path": cam.get("path") or "",
        }
        location = locate(metadata["path"])
        if location:
            metadata["usb_location"] = location
        return metadata

    return [
        ConnectedDevice(
            kind="uvc",
            device_id=str(cam["index"]),
            registry_name=None,
            state="available",
            metadata=_metadata(cam),
        )
        for cam in _enumerate_or_empty(enumerate_devices)
        if cam.get("index") is not None
    ]


def list_connected_iphone(
    enumerate_devices: Callable[[], Iterable[Any]] | None = None,
    usb_locations: Callable[[], Mapping[str, str]] | None = None,
) -> list[ConnectedDevice]:
    """Every phone Record3D reports, registered or not — no filtering.

    Record3D gives a UDID and no port, so where each phone is plugged in comes
    from the OS, matched on the USB serial number (the UDID without its dash).
    """
    enumerate_devices = enumerate_devices or _default_enumerate_iphone
    usb_locations = usb_locations or usb_location.by_serial_number
    devices = _enumerate_or_empty(enumerate_devices)
    if not devices:
        return []

    registry = _iphone_registry()
    locations = usb_locations()

    def _metadata(dev) -> dict[str, Any]:
        metadata: dict[str, Any] = {"product_id": getattr(dev, "product_id", None)}
        location = locations.get(str(dev.udid).replace("-", "").upper())
        if location:
            metadata["usb_location"] = location
        return metadata

    return [
        ConnectedDevice(
            kind="iphone",
            device_id=dev.udid,
            registry_name=_registry_name_for(registry, dev.udid),
            state="available",
            metadata=_metadata(dev),
        )
        for dev in devices
    ]

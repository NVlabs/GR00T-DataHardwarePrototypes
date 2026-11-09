# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request/reply exchanges with GDK drives over a python-can bus."""

from __future__ import annotations

import argparse
import errno
import time
from collections.abc import Callable, Collection, Sequence

import can
from serial.tools import list_ports

from . import gdk_protocol as proto
from .gdk_protocol import Ping, Status
from .load_config import INTERFACES, CanSettings, Ids

DEFAULT_REPLY_TIMEOUT_S = 0.2


class BusUnavailable(RuntimeError):
    """The CAN interface could not be opened."""


def find_adapter_port(usb_id: tuple[int, int]) -> str | None:
    """The serial port of the USB-CAN adapter with this USB id; None if absent."""
    for port in list_ports.comports():
        if (port.vid, port.pid) == usb_id:
            return port.device
    return None


def open_bus(
    settings: CanSettings, interface: str | None = None, channel: str | None = None
) -> can.BusABC:
    """Open the neck's CAN bus; ``interface`` and ``channel`` override ``settings``.

    socketcan: the channel is the network interface, can_neck by default, which
    scripts/setup_can.sh brings up at 1 Mbit/s. slcan: the channel is the
    adapter's serial port, by default the one with the adapter's USB id;
    python-can sets the bitrate itself, with no slcand and no root.
    """
    interface = interface or settings.interface
    if interface not in INTERFACES:
        raise ValueError(f"interface must be one of {INTERFACES}, not {interface!r}")
    if interface == "socketcan":
        channel = channel or settings.socketcan_channel
    else:
        channel = channel or settings.slcan_port or find_adapter_port(settings.adapter_usb_id)
        if channel is None:
            vid, pid = settings.adapter_usb_id
            raise BusUnavailable(
                f"no USB-CAN adapter ({vid:04x}:{pid:04x}) found; plug it in, check it is not "
                "in DFU mode (BOOT switch off), or pass its serial port with --channel"
            )
    try:
        if interface == "slcan":
            return can.Bus(interface="slcan", channel=channel, bitrate=settings.bitrate)
        return can.Bus(interface="socketcan", channel=channel)
    except (OSError, can.CanError) as exc:
        if interface == "socketcan":
            hint = "bring it up with sudo scripts/setup_can.sh, or use --interface slcan"
        elif _permission_denied(exc):
            hint = (
                "no permission for the serial port: add yourself to the dialout group "
                "(sudo usermod -aG dialout $USER), then log out and back in"
            )
        else:
            hint = "check the adapter is plugged in and not in DFU mode (BOOT switch off)"
        raise BusUnavailable(f"cannot open {channel} ({interface}): {exc}; {hint}") from exc


def _permission_denied(exc: BaseException | None) -> bool:
    """Whether opening failed for lack of permission. python-can wraps pyserial's
    error, so the errno can sit anywhere along the exception's chain."""
    while exc is not None:
        if isinstance(exc, PermissionError) or getattr(exc, "errno", None) == errno.EACCES:
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def add_bus_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--interface",
        choices=INTERFACES,
        help="how to reach the CAN adapter (default: can.interface in the settings file)",
    )
    parser.add_argument(
        "--channel",
        help=(
            "socketcan interface or slcan serial port (default: can.socketcan_channel, or "
            "can.slcan_port or the adapter found by its USB id, in the settings file)"
        ),
    )


class GdkLink:
    """Exchange requests and replies with GDK drives on one bus."""

    def __init__(self, bus: can.BusABC, reply_timeout_s: float = DEFAULT_REPLY_TIMEOUT_S) -> None:
        self.bus = bus
        self._reply_timeout_s = reply_timeout_s

    def send(self, drive_id: int, data: bytes) -> None:
        self.bus.send(can.Message(arbitration_id=drive_id, data=data, is_extended_id=False))

    def ping(self, drive_id: int, timeout_s: float | None = None) -> Ping | None:
        """The ping reply of the drive on ``drive_id``, whatever CAN_ID it replies on."""
        self._drain_rx()
        self.send(drive_id, proto.pack_ping())
        msg = self._await(
            lambda m: proto.is_ping_reply(bytes(m.data), drive_id, m.arbitration_id), timeout_s
        )
        return None if msg is None else proto.unpack_ping(bytes(msg.data))

    def request(
        self, drive_id: int, data: bytes, reply_ids: Collection[int], reply_length: int
    ) -> bytes:
        """Send ``data`` to ``drive_id``; return the first reply of ``reply_length`` bytes
        from one of ``reply_ids``."""
        self._drain_rx()
        self.send(drive_id, data)
        msg = self._await(lambda m: m.arbitration_id in reply_ids and len(m.data) == reply_length)
        if msg is None:
            raise TimeoutError(f"no reply from ID {drive_id}")
        return bytes(msg.data)

    def read_status(self, drive: Ids | Ping) -> Status:
        data = self.request(drive.id, proto.pack_status_read(), [drive.can_id], proto.FRAME_LENGTH)
        return proto.unpack_status(data)

    def read_register(self, drive: Ids | Ping, address: int, length: int) -> bytes:
        request = proto.pack_register_read(address, length)
        return self.request(drive.id, request, [drive.can_id], length)

    def exchange(
        self, requests: Sequence[tuple[Ids, bytes]], reply_length: int, timeout_s: float
    ) -> dict[int, bytes]:
        """Send every request at once, then collect one ``reply_length``-byte reply from
        each request's CAN_ID. Replies are keyed by CAN_ID; a drive that does not answer
        within ``timeout_s`` is missing."""
        self._drain_rx()
        for ids, data in requests:
            self.send(ids.id, data)
        expected = {ids.can_id for ids, _data in requests}
        replies: dict[int, bytes] = {}
        deadline = time.monotonic() + timeout_s
        while len(replies) < len(expected):
            remaining_s = deadline - time.monotonic()
            msg = self.bus.recv(timeout=max(0.0, remaining_s))
            if msg is None:  # recv waited out the deadline
                break
            if (
                _is_data_frame(msg)
                and msg.arbitration_id in expected
                and len(msg.data) == reply_length
            ):
                replies.setdefault(msg.arbitration_id, bytes(msg.data))
            if remaining_s <= 0.0:  # past the deadline: stop, even while frames keep coming
                break
        return replies

    def _drain_rx(self) -> None:
        while self.bus.recv(timeout=0.0) is not None:
            pass

    def _await(
        self, matches: Callable[[can.Message], bool], timeout_s: float | None = None
    ) -> can.Message | None:
        deadline = time.monotonic() + (self._reply_timeout_s if timeout_s is None else timeout_s)
        while time.monotonic() < deadline:
            msg = self.bus.recv(timeout=max(0.0, deadline - time.monotonic()))
            if msg is None:  # recv waited out the deadline
                return None
            if _is_data_frame(msg) and matches(msg):
                return msg
        return None


def _is_data_frame(msg: can.Message) -> bool:
    return not (msg.is_extended_id or msg.is_error_frame or msg.is_remote_frame)

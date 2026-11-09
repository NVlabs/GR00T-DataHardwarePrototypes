# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A simple interface to the GEAR Neck: command yaw, pitch and roll, read them back.

    from gear_neck import Neck

    with Neck() as neck:                         # finds the drives; torque stays off
        print(neck.angles())                     # works with torque off
        neck.enable()                            # torque on, holding where it is
        neck.move_to(yaw=20, pitch=10, roll=0)   # degrees; blocks until there
    # leaving the block switches torque off

Angles are in degrees, in the joint directions of the settings file,
config/neck.yaml (yaw right, roll right, pitch down are positive); the file
also holds each joint's limits and whether its drive is mounted reversed. A
background thread runs the control loop at its control_rate_hz. With torque
on, every tick moves each joint's setpoint toward its target by at most
max_speed_deg_s and sends it to the drive as an MIT command with the joint's
kp and kd; the drive's reply gives the measured angle. With torque off, every
tick reads each drive's present position (register 552).

The drives have no command watchdog: left alone they hold their last setpoint.
Torque therefore goes off on close(), on leaving a with block, at interpreter
exit, and as soon as a drive reports a fault or stops answering.
"""

from __future__ import annotations

import atexit
import contextlib
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import can

from . import gdk_protocol as proto
from .gdk_link import GdkLink, open_bus
from .gdk_protocol import MitRanges
from .load_config import Ids, Joint, NeckConfig, resolve_config

# Torque off has no reply; its status is checked this long after.
TORQUE_OFF_SETTLE_S = 0.02
TORQUE_OFF_ATTEMPTS = 3
# While faulted, torque off goes out again at this period.
FAULT_TORQUE_OFF_PERIOD_S = 0.1
# Status reads around arming and disarming.
REPLY_TIMEOUT_S = 0.05


class NeckError(RuntimeError):
    """The neck cannot do what was asked, or a drive faulted."""


class Angles(NamedTuple):
    """Joint angles in degrees."""

    yaw: float
    pitch: float
    roll: float

    def __str__(self) -> str:
        return f"yaw {self.yaw:+6.1f}  pitch {self.pitch:+6.1f}  roll {self.roll:+6.1f} deg"


class DriveInfo(NamedTuple):
    """What a drive reported when the neck opened."""

    joint: str
    ids: Ids
    bus_voltage_v: float
    operating_mode: int
    fault: int  # proto.fault_names()
    calibration: int  # register 403
    ranges: MitRanges

    @property
    def keeps_zero(self) -> bool:
        """Both encoders are calibrated, so the zero survives a power cycle."""
        return proto.keeps_position(self.calibration)


@dataclass
class _Drive:
    """Live state of one joint. Positions are in rad, in the joint's direction."""

    joint: Joint
    info: DriveInfo
    position_rad: float
    velocity_rad_s: float = 0.0
    setpoint_rad: float = 0.0
    target_rad: float = 0.0
    missed: int = 0  # consecutive ticks without a reply


def ramp(setpoint: float, target: float, max_step: float) -> float:
    """Move ``setpoint`` toward ``target`` by at most ``max_step``, landing on it exactly."""
    error = target - setpoint
    if abs(error) <= max_step:
        return target
    return setpoint + math.copysign(max_step, error)


def clamp_deg(joint: Joint, angle_deg: float) -> float:
    return min(max(angle_deg, joint.min_deg), joint.max_deg)


class Neck:
    """The three neck drives, driven from one background control loop.

    Opening reads the settings file (``config``: a path, a loaded NeckConfig,
    or None for config/neck.yaml), finds every drive, reads its MIT ranges and
    position, and switches off torque a previous run may have left on.
    ``interface`` and ``channel`` override the file's CAN settings, and
    ``rate_hz`` and ``max_speed_deg_s`` its control_rate_hz and
    max_speed_deg_s. All methods are thread-safe.
    """

    def __init__(
        self,
        interface: str | None = None,
        channel: str | None = None,
        *,
        config: NeckConfig | str | Path | None = None,
        bus: can.BusABC | None = None,
        rate_hz: float | None = None,
        max_speed_deg_s: float | None = None,
    ) -> None:
        settings = resolve_config(config)
        rate_hz = rate_hz or settings.control_rate_hz
        max_speed_deg_s = max_speed_deg_s or settings.max_speed_deg_s
        self._owns_bus = bus is None
        self._bus = open_bus(settings.can, interface, channel) if bus is None else bus
        self._link = GdkLink(self._bus, reply_timeout_s=REPLY_TIMEOUT_S)
        self._period_s = 1.0 / rate_hz
        # Each tick's replies must land within the tick.
        self._tick_reply_timeout_s = 0.6 * self._period_s
        self._max_step_rad = math.radians(max_speed_deg_s) * self._period_s
        self._feedback_timeout_ticks = max(1, round(settings.feedback_timeout_s * rate_hz))
        self._lock = threading.Lock()
        self._enabled = False
        self._fault: str | None = None
        self._last_fault_torque_off = 0.0
        self._closed = False
        try:
            self._drives = [self._open_drive(joint) for joint in settings.joints]
        except BaseException:
            if self._owns_bus:
                self._bus.shutdown()
            raise
        self._by_name = {drive.joint.name: drive for drive in self._drives}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gear-neck", daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def __enter__(self) -> Neck:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def drives(self) -> tuple[DriveInfo, ...]:
        return tuple(drive.info for drive in self._drives)

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def angles(self) -> Angles:
        """The measured joint angles."""
        with self._lock:
            self._check_open()
            return self._angles(lambda drive: drive.position_rad)

    def target(self) -> Angles:
        """The commanded joint angles, after the limits."""
        with self._lock:
            self._check_open()
            return self._angles(lambda drive: drive.target_rad)

    def silent_joints(self) -> list[str]:
        """Joints whose drive has not answered for the settings' feedback_timeout_s."""
        with self._lock:
            return [
                drive.joint.name
                for drive in self._drives
                if drive.missed >= self._feedback_timeout_ticks
            ]

    def enable(self) -> None:
        """Switch torque on, holding every joint where it is."""
        with self._lock:
            self._check_ok()
            if self._enabled:
                return
            try:
                for drive in self._drives:
                    self._arm(drive)
            except BaseException:
                self._torque_off()
                raise
            self._enabled = True

    def disable(self) -> None:
        """Switch torque off; the neck goes limp."""
        with self._lock:
            self._check_open()
            self._torque_off()

    def set_target(
        self, yaw: float | None = None, pitch: float | None = None, roll: float | None = None
    ) -> Angles:
        """Command joint angles in degrees and return at once; None keeps a joint's target.

        Each angle is clamped to its joint's limits, and the setpoint ramps
        toward it at max_speed_deg_s. Returns the clamped target.
        """
        with self._lock:
            self._check_ok()
            if not self._enabled:
                raise NeckError("torque is off: call enable() first")
            for name, angle_deg in (("yaw", yaw), ("pitch", pitch), ("roll", roll)):
                if angle_deg is not None:
                    drive = self._by_name[name]
                    drive.target_rad = math.radians(clamp_deg(drive.joint, angle_deg))
            return self._angles(lambda drive: drive.target_rad)

    def move_to(
        self,
        yaw: float | None = None,
        pitch: float | None = None,
        roll: float | None = None,
        *,
        settle_s: float = 0.2,
        timeout_s: float | None = None,
    ) -> Angles:
        """Command joint angles in degrees, as set_target(), and wait until the setpoints
        reach them, then ``settle_s`` more for the joints to catch up. Returns the
        measured angles.

        With soft gains a joint can stop short of its target: compare the result
        with target().
        """
        self.set_target(yaw, pitch, roll)
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            with self._lock:
                self._check_ok()
                if not self._enabled:
                    raise NeckError("torque went off during the move")
                if all(drive.setpoint_rad == drive.target_rad for drive in self._drives):
                    break
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("the move did not finish in time")
            time.sleep(self._period_s)
        time.sleep(settle_s)
        return self.angles()

    def close(self) -> None:
        """Switch torque off, stop the control loop and release the bus."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        # The bus may be gone; there is nothing more to do then.
        with self._lock, contextlib.suppress(Exception):
            self._torque_off()
        if self._owns_bus:
            self._bus.shutdown()
        atexit.unregister(self.close)

    # Everything below runs with self._lock held.

    def _check_open(self) -> None:
        if self._closed:
            raise NeckError("the neck is closed")

    def _check_ok(self) -> None:
        self._check_open()
        if self._fault is not None:
            raise NeckError(f"{self._fault}; torque is off. Close the neck and open it again")

    def _angles(self, value: Callable[[_Drive], float]) -> Angles:
        return Angles(*(math.degrees(value(self._by_name[name])) for name in Angles._fields))

    def _open_drive(self, joint: Joint) -> _Drive:
        ids = joint.ids
        try:
            status = self._link.read_status(ids)
            if status.torque_on:  # left live by an earlier run
                self._link.send(ids.id, proto.pack_torque_off())
            spans = [
                proto.unpack_range(
                    index,
                    self._link.request(
                        ids.id, proto.pack_range_read(index), [ids.can_id], proto.RANGE_LENGTH
                    ),
                )
                for index in range(proto.RANGE_COUNT)
            ]
            calibration = self._link.read_register(ids, proto.CALIBRATION_REGISTER, 1)[0]
            position_rad = self._read_position(joint)
        except TimeoutError:
            raise NeckError(
                f"{joint.name}: no reply from ID {ids.id} on CAN_ID {ids.can_id}; check its "
                "power and wiring, and its ids with neck_assign_id scan"
            ) from None
        ranges = proto.mit_ranges(spans)
        if ranges is None:
            raise NeckError(f"{joint.name}: an MIT range (registers 330-349) is empty")
        info = DriveInfo(
            joint.name,
            ids,
            status.bus_voltage_v,
            status.operating_mode,
            status.fault,
            calibration,
            ranges,
        )
        return _Drive(joint, info, position_rad, setpoint_rad=position_rad, target_rad=position_rad)

    def _read_position(self, joint: Joint) -> float:
        data = self._link.read_register(
            joint.ids, proto.PRESENT_POSITION_REGISTER, proto.PRESENT_POSITION_LENGTH
        )
        return joint.direction * proto.unpack_present_position(data)

    def _arm(self, drive: _Drive) -> None:
        """MIT mode, then torque on, then hold the present position."""
        joint, ids = drive.joint, drive.joint.ids
        try:
            status = self._link.read_status(ids)
            if status.fault:
                raise NeckError(
                    f"{joint.name}: drive fault {proto.fault_names(status.fault)}; "
                    "power-cycle the neck"
                )
            if status.operating_mode != proto.MIT_MODE:
                reply = self._link.request(
                    ids.id,
                    proto.pack_operating_mode_write(),
                    [ids.can_id],
                    proto.WRITE_REPLY_LENGTH,
                )
                if not proto.write_accepted(reply):
                    raise NeckError(f"{joint.name}: the drive refused MIT mode")
            drive.position_rad = self._read_position(joint)
            drive.setpoint_rad = drive.target_rad = drive.position_rad
            drive.velocity_rad_s = 0.0
            self._check_mit_state(drive, self._exchange_one(drive, proto.pack_torque_on()))
            # Hold at once rather than leave the drive on whatever setpoint it had.
            self._check_mit_state(drive, self._exchange_one(drive, self._mit_command(drive, 0.0)))
        except TimeoutError:
            raise NeckError(
                f"{joint.name}: no reply from CAN_ID {ids.can_id} while arming"
            ) from None
        drive.missed = 0

    def _exchange_one(self, drive: _Drive, data: bytes) -> bytes:
        ids = drive.joint.ids
        return self._link.request(ids.id, data, [ids.can_id], proto.FRAME_LENGTH)

    def _check_mit_state(self, drive: _Drive, data: bytes) -> None:
        state = proto.unpack_mit_state(data, drive.info.ranges)
        if state.error:
            raise NeckError(f"{drive.joint.name}: drive reported MIT error {state.error}")
        drive.position_rad = drive.joint.direction * state.position_rad
        drive.velocity_rad_s = drive.joint.direction * state.velocity_rad_s

    def _mit_command(self, drive: _Drive, velocity_rad_s: float) -> bytes:
        joint = drive.joint
        return proto.pack_mit_command(
            drive.info.ranges,
            position_rad=joint.direction * drive.setpoint_rad,
            velocity_rad_s=joint.direction * velocity_rad_s,
            kp=joint.kp,
            kd=joint.kd,
        )

    def _torque_off(self) -> None:
        """Send torque off to every drive until each status says it is off."""
        self._enabled = False
        pending = list(self._drives)
        for _attempt in range(TORQUE_OFF_ATTEMPTS):
            for drive in pending:
                self._link.send(drive.joint.ids.id, proto.pack_torque_off())
            time.sleep(TORQUE_OFF_SETTLE_S)
            still_on = []
            for drive in pending:
                try:
                    if self._link.read_status(drive.joint.ids).torque_on:
                        still_on.append(drive)
                except TimeoutError:  # a silent drive: nothing more can be done for it
                    pass
            pending = still_on
            if not pending:
                return

    def _fail(self, message: str) -> None:
        self._fault = message
        with contextlib.suppress(Exception):  # keep the fault; the bus may be what failed
            self._torque_off()

    def _run(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                if not self._closed:
                    try:
                        if self._enabled:
                            self._tick_enabled()
                        else:
                            self._tick_disabled()
                    except Exception as exc:  # a CAN error, such as a full TX queue
                        if self._enabled:
                            self._fail(f"CAN bus error: {exc}")
                        else:  # torque is off: it costs only this tick's reading
                            for drive in self._drives:
                                drive.missed += 1
            next_tick += self._period_s
            delay = next_tick - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:  # overran: do not try to catch up
                next_tick = time.monotonic()

    def _tick_enabled(self) -> None:
        requests = []
        for drive in self._drives:
            previous = drive.setpoint_rad
            drive.setpoint_rad = ramp(previous, drive.target_rad, self._max_step_rad)
            velocity_rad_s = (drive.setpoint_rad - previous) / self._period_s
            requests.append((drive.joint.ids, self._mit_command(drive, velocity_rad_s)))
        replies = self._link.exchange(requests, proto.FRAME_LENGTH, self._tick_reply_timeout_s)
        for drive in self._drives:
            reply = replies.get(drive.joint.ids.can_id)
            if reply is None:
                drive.missed += 1
                if drive.missed >= self._feedback_timeout_ticks:
                    self._fail(f"{drive.joint.name}: the drive stopped answering")
                    return
                continue
            drive.missed = 0
            try:
                self._check_mit_state(drive, reply)
            except NeckError as exc:
                self._fail(str(exc))
                return

    def _tick_disabled(self) -> None:
        now = time.monotonic()
        if (
            self._fault is not None
            and now - self._last_fault_torque_off >= FAULT_TORQUE_OFF_PERIOD_S
        ):
            # The torque off sent at the fault may not have got through (a bus error).
            self._last_fault_torque_off = now
            for drive in self._drives:
                self._link.send(drive.joint.ids.id, proto.pack_torque_off())
        request = proto.pack_register_read(
            proto.PRESENT_POSITION_REGISTER, proto.PRESENT_POSITION_LENGTH
        )
        replies = self._link.exchange(
            [(drive.joint.ids, request) for drive in self._drives],
            proto.PRESENT_POSITION_LENGTH,
            self._tick_reply_timeout_s,
        )
        for drive in self._drives:
            reply = replies.get(drive.joint.ids.can_id)
            if reply is None:
                drive.missed += 1
                continue
            drive.missed = 0
            drive.position_rad = drive.joint.direction * proto.unpack_present_position(reply)
            drive.velocity_rad_s = 0.0

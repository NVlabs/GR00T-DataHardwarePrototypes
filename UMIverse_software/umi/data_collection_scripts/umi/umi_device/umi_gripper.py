# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UMI Gripper device implementation.

Provides angle-to-distance conversion and calibration for UMI gripper with encoder sensor.
Acts as a data source factory, producing a gripper data source for the recorder.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import numpy as np

from umi.common.data.frame import create_frame
from umi.common.hardware_sdk.sensor.encoder_sensor import EncoderSensor
from umi.common.hardware_sdk.sensor_interface import SensorInterface
from umi.common.utility.gripper_pipeline import GripperPipeline
from umi.umi_device.umi_device_interface import UmiDeviceInterface, create_data_source

# V4 defaults derive from a 15 mm pinion and a provisional prototype travel
# band; validate the band when hardware changes. One encoder degree is
# approximately 0.52 mm of jaw opening. Repeated closes must agree to avoid
# a biased zero. Too little open travel indicates a jam or unreliable
# direction signal; too much indicates an incorrect mechanical-stop zero.
# The travel band is approximately 63–92 mm, near the 90 mm data-check cap.
CLOSE_SPREAD_MAX_DEG = 2.0
OPEN_TRAVEL_MIN_DEG, OPEN_TRAVEL_MAX_DEG = GripperPipeline.V4.open_travel_band_deg
DEFAULT_GEAR_RADIUS_MM = GripperPipeline.V4.gear_radius_mm

CHECK_CLOSE_REPEATABILITY = "close_repeatability"
CHECK_OPEN_TRAVEL = "open_travel"
CHECK_OPEN_TRAVEL_HIGH = "open_travel_high"

# Validate in encoder degrees. Display approximate millimeters using this
# gripper's configured gear radius; do not use the display conversion as
# an independent acceptance criterion.
_FAULT_HEADLINES = {
    CHECK_CLOSE_REPEATABILITY: (
        "Close repeatability: the closes spread {measured:.2f}° (~{measured_mm:.1f} mm), "
        "over the {limit:.2f}° (~{limit_mm:.1f} mm) limit"
    ),
    CHECK_OPEN_TRAVEL: (
        "Open travel: only {measured:.2f}° (~{measured_mm:.1f} mm) from closed, "
        "under the {limit:.2f}° (~{limit_mm:.1f} mm) minimum"
    ),
    CHECK_OPEN_TRAVEL_HIGH: (
        "Open travel: {measured:.2f}° (~{measured_mm:.1f} mm) from closed, "
        "over the {limit:.2f}° (~{limit_mm:.1f} mm) maximum "
        "-- the closes were probably not on the stop"
    ),
}


class GripperHardwareFault(Exception):
    """The operator declared the gripper itself faulty and stopped calibrating.

    Raised rather than returned because ``UmiGripper.__init__`` calls
    ``interactive_calibrate`` for its side effect and ignores the return value:
    a ``False`` here would leave a half-calibrated gripper in the caller's hands,
    which is the exact thing the checks exist to prevent.

    ``payload`` is what the fault report is built from -- see
    ``umi.common.utility.gripper_fault_report``.
    """

    def __init__(self, payload: dict) -> None:
        # Derive displayed millimeters from the payload's gear_radius_mm so
        # calibration prompts and validation reports agree.
        to_mm = UmiGripper._delta_angle_to_distance
        radius = payload["gear_radius_mm"]
        super().__init__(
            _FAULT_HEADLINES[payload["check"]].format(
                measured=payload["measured_deg"],
                limit=payload["limit_deg"],
                measured_mm=to_mm(payload["measured_deg"], radius),
                limit_mm=to_mm(payload["limit_deg"], radius),
            )
        )
        self.payload = payload


def _wrapped_offsets(angles: list[float]) -> tuple[float, list[float]]:
    """Angles as signed offsets from the first one, folded into (-180, 180].

    The encoder is absolute 0-360, so anything comparing raw angles breaks at
    the seam: closes at 359.5 and 0.2 are 0.7 apart, not 359.3.
    """
    reference = angles[0]
    return reference, [((a - reference + 180.0) % 360.0) - 180.0 for a in angles]


def _wrapped_spread(angles: list[float]) -> float:
    """How far apart the furthest two angles are, in degrees."""
    _, offsets = _wrapped_offsets(angles)
    return max(offsets) - min(offsets)


def _wrapped_mean(angles: list[float]) -> float:
    """Mean angle in [0, 360), computed through the seam rather than across it.

    A plain ``sum / len`` of 359.5 and 0.2 is 179.85, a zero pointing at the
    opposite side of the encoder. The spread check treats such a set as
    perfectly repeatable, so the mean has to agree with it.
    """
    reference, offsets = _wrapped_offsets(angles)
    return (reference + sum(offsets) / len(offsets)) % 360.0


class UmiGripper(UmiDeviceInterface):
    """UMI gripper with angle-distance conversion and calibration.

    Acts as a data source factory: call get_data_sources() to obtain
    a gripper data source for the recorder.

    Attributes:
        encoder: The underlying EncoderSensor instance
        gear_radius: Pinion pitch radius in mm. Defaults to the V4's 15.0.
        gripper_generation: ``GripperPipeline`` value this gripper was built
            as, or None for an older config without the field.

    Example:
        gripper = UmiGripper(
            device_name="umi_gripper",
            reader=uart_reader,
        )

        # Interactive calibration
        gripper.interactive_calibrate()

        # Get data sources for recorder
        sources = gripper.get_data_sources()

        # Load from config file
        gripper = UmiGripper.from_config("config/gripper.json", reader=uart_reader)
    """

    _config_keys = [
        "device_name",
        "gear_radius",
        "distance_zero_angle",
        "directions",
    ]

    def __init__(
        self,
        device_name: str,
        reader: Any,
        gear_radius: float | None = None,
        distance_zero_angle: float | None = None,
        directions: str | None = None,
        latency: float | None = None,
        gripper_generation: str | None = None,
        open_travel_deg: tuple[float, float] | None = None,
    ) -> None:
        """Initialize UMI gripper.

        Args:
            device_name: Unique device identifier
            reader: Shared UartSensorReader instance
            gear_radius: Pinion pitch radius in mm for distance calculation.
                None takes the V4's 15.0.
            distance_zero_angle: Calibration angle at fully closed position in degrees (default: None)
            directions: Single-char direction string, '+' or '-'. '+' means
                increasing encoder angle corresponds to gripper opening.
                If None, interactive calibration is run to determine it.
            latency: Latency for data source (default: None)
            gripper_generation: Which ``GripperPipeline`` this gripper is
                (``"v4"``). Sets the defaults for ``gear_radius`` and the
                open-travel band the calibration accepts, and is written into
                the saved config so a recording carries the fact.
            open_travel_deg: ``(min, max)`` encoder degrees a full open may
                land in during calibration. None takes the generation's band,
                or the module constants (the same V4 band) without one.
        """
        super().__init__()

        self.device_name = device_name
        self.latency = latency

        self.encoder = EncoderSensor(
            device_name=device_name,
            reader=reader,
            sensor_ids=[0],
        )

        if gripper_generation is not None and not GripperPipeline.is_known(
            gripper_generation
        ):
            raise ValueError(
                f"unknown gripper_generation {gripper_generation!r}; expected one "
                f"of {GripperPipeline.values()}"
            )
        generation = (
            GripperPipeline.parse(gripper_generation)
            if gripper_generation is not None
            else None
        )
        self.gripper_generation = generation.value if generation else None

        if gear_radius is None:
            gear_radius = (
                generation.gear_radius_mm if generation else DEFAULT_GEAR_RADIUS_MM
            )
        self.gear_radius = float(gear_radius)

        if open_travel_deg is None:
            open_travel_deg = (
                generation.open_travel_band_deg
                if generation
                else (OPEN_TRAVEL_MIN_DEG, OPEN_TRAVEL_MAX_DEG)
            )
        self.open_travel_min_deg = float(open_travel_deg[0])
        self.open_travel_max_deg = float(open_travel_deg[1])

        # Record calibrated angle, jaw distance and raw encoder reading. Do not
        # interpolate raw_encoder_angles across its 0/360 wrap; use joint_angles
        # for smooth aligned data.
        self.encoder_frame_class = create_frame(
            f"{device_name}_encoder_Frame",
            {
                "joint_angles": {"interpolate": True},
                "distance": {"interpolate": True},
                "raw_encoder_angles": {"interpolate": False},
            },
        )

        if directions is not None and directions not in ("+", "-"):
            raise ValueError(f"directions must be '+' or '-', got {directions!r}")
        self.directions = directions
        self.direction_sign = (
            1.0 if directions == "+" else (-1.0 if directions == "-" else None)
        )

        self.distance_zero_angle = distance_zero_angle
        if self.distance_zero_angle is None or self.directions is None:
            self.interactive_calibrate()

    def get_config(self) -> dict[str, Any]:
        """The saved calibration, plus the generation when one is known.

        Not in ``_config_keys`` on purpose: those are required, and older
        calibration files have no such key. Written when present so an
        episode's ``gripper_calibration`` attrs say which gear the distances
        were computed for.
        """
        config = super().get_config()
        if self.gripper_generation:
            config["gripper_generation"] = self.gripper_generation
        return config

    # ==================== Data Source Factory ====================

    def get_data_sources(self) -> list[SensorInterface]:
        """Return gripper data source for the recorder.

        Returns:
            List containing a single gripper data source.
        """
        self._validate_required_config()
        return [
            create_data_source(
                sensor=self.encoder,
                device_name=self.device_name,
                latency=self.latency,
                frame_class=self.encoder_frame_class,
                get_frame_fn=self._get_encoder_frame,
            )
        ]

    def _get_encoder_frame(self):
        """Read encoder and return a frame with joint_angles, distance, and
        raw_encoder_angles.

        joint_angles is the calibrated, sign-applied joint angle clipped at the
        closed position (a meaningful gripper opening angle that interpolates
        smoothly across the 0/360 encoder wrap); distance is the opening in mm;
        raw_encoder_angles is the unprocessed encoder reading, kept for
        diagnostics. Distance is unchanged from the raw-angle path:
        angle_to_distance applies the same gear formula to the same delta.
        """
        encoder_frame = self.encoder.get_frame()
        if encoder_frame is None:
            return None
        if encoder_frame.position is None or len(encoder_frame.position) == 0:
            return None

        raw_angle = float(encoder_frame.position[0])
        joint_angle = max(0.0, self.get_signed_angle_offset(raw_angle))
        distance = self.angle_to_distance(joint_angle)

        return self.encoder_frame_class(
            joint_angles=np.array([joint_angle]),
            distance=distance,
            raw_encoder_angles=np.array([raw_angle]),
            capture_time=encoder_frame.capture_time,
            receive_time=encoder_frame.receive_time,
            frame_source=self.device_name,
        )

    # ==================== Distance Conversion ====================

    def get_signed_angle_offset(self, current_angle: float) -> float:
        """Signed angular offset from zero with wrap and direction applied.

        Formula: ((current - zero + 180) % 360 - 180) * direction_sign
        Positive = opening (after direction calibration).

        Args:
            current_angle: Current encoder angle in degrees.

        Returns:
            Signed angular offset in degrees (-180, 180].
        """
        diff = (current_angle - self.distance_zero_angle + 180.0) % 360.0 - 180.0
        return diff * self.direction_sign

    @staticmethod
    def _delta_angle_to_distance(delta_angle, gear_radius):
        """Gear model: distance (mm) = 2 * pi * gear_radius * (delta_angle / 360) * 2.

        Linear in ``delta_angle``, so it applies elementwise to numpy arrays as
        well as to the online scalar path (``angle_to_distance``). The factor of
        2 accounts for both sides of the gripper moving.
        """
        return 2 * math.pi * gear_radius * delta_angle / 360.0 * 2

    def angle_to_distance(self, delta_angle: float) -> float:
        """Convert a signed gripper joint angle to gripper opening distance.

        Args:
            delta_angle: Signed angular offset from the closed position in
                degrees, already clipped to >= 0 by the caller. This is the
                value stored as the gripper joint position; see
                ``get_signed_angle_offset`` for how it is derived from the raw
                encoder angle.

        Returns:
            Opening distance in millimeters
        """
        return self._delta_angle_to_distance(delta_angle, self.gear_radius)

    # ==================== Calibration Methods ====================

    def _capture_angle(self, num_samples: int) -> float | None:
        """Mean of *num_samples* encoder reads, or None if none arrived."""
        samples: list[float] = []
        for _ in range(num_samples):
            frame = self.encoder.get_frame()
            if frame is not None and frame.position is not None:
                samples.append(float(frame.position[0]))
            time.sleep(0.01)
        return sum(samples) / len(samples) if samples else None

    def _retry_or_fault(
        self,
        check: str,
        measured: float,
        limit: float,
        close_angles: list[float],
        open_angle: float | None,
        attempt: int,
    ) -> None:
        """Show the failed check, then return to retry or raise on the operator's word.

        Retries are unbounded and the fault option is offered every time: only
        the person at the bench can tell "I did not squeeze hard enough" from
        "this gripper is broken", and a fixed cap would either strand someone who
        needs one more go or force a fault report out of a healthy gripper.
        """
        print(
            "\n  ✗ "
            + _FAULT_HEADLINES[check].format(
                measured=measured,
                limit=limit,
                measured_mm=self._delta_angle_to_distance(measured, self.gear_radius),
                limit_mm=self._delta_angle_to_distance(limit, self.gear_radius),
            )
        )
        print("     closes: " + ", ".join(f"{a:.2f}°" for a in close_angles))
        if open_angle is not None:
            print(f"     open:   {open_angle:.2f}°")

        # Only explicit f reports hardware failure; other input, including EOF, retries.
        try:
            answer = input(
                "  [Enter] retry from the start   "
                "[f] report a hardware fault and stop: "
            )
        except EOFError:
            answer = ""
        if answer.strip().lower() != "f":
            return

        self.encoder.stop_streaming()
        raise GripperHardwareFault(
            {
                "check": check,
                "measured_deg": measured,
                "limit_deg": limit,
                "close_angles_deg": close_angles,
                "open_angle_deg": open_angle,
                "attempts": attempt,
                "device_name": self.device_name,
                "gear_radius_mm": self.gear_radius,
            }
        )

    def interactive_calibrate(
        self, n: int = 3, auto_save_path: str | None = None
    ) -> bool:
        """Calibrate the gripper zero distance angle interactively.

        Closes are retaken until they agree to within ``CLOSE_SPREAD_MAX_DEG``
        and the open capture lands between ``OPEN_TRAVEL_MIN_DEG`` and
        ``OPEN_TRAVEL_MAX_DEG``. A failure restarts
        the whole sequence rather than the failed half: reusing an earlier zero
        for a fresh open capture pairs measurements taken from different gripper
        positions, which is the class of error being checked for.

        Args:
            n: Number of times to ask user to close gripper (default: 3)
            auto_save_path: Optional path to automatically save config after calibration.
                          If provided, skips the interactive save prompt. If None, checks
                          environment variable UMI_GRIPPER_AUTO_SAVE_PATH, then prompts user.

        Returns:
            True if calibration succeeded, False otherwise

        Raises:
            GripperHardwareFault: the operator answered a failed check by
                reporting the hardware. No config is saved on that path.
        """
        NUM_SAMPLES_PER_CAPTURE = 10

        self.encoder.start_streaming()

        print("Starting gripper calibration...")

        attempt = 0
        while True:
            attempt += 1

            recorded_angles: list[float] = []
            for i in range(n):
                input(f"[{i + 1}/{n}] Close the gripper fully, then press Enter...")
                avg_angle = self._capture_angle(NUM_SAMPLES_PER_CAPTURE)
                if avg_angle is not None:
                    recorded_angles.append(avg_angle)
                    print(f"  Recorded angle: {avg_angle:.2f}")

            if not recorded_angles:
                print("Error: No angles recorded during calibration.")
                self.encoder.stop_streaming()
                return False

            # Require repeated closes to agree before averaging a zero or prompting
            # for open travel.
            spread = _wrapped_spread(recorded_angles)
            if spread > CLOSE_SPREAD_MAX_DEG:
                self._retry_or_fault(
                    CHECK_CLOSE_REPEATABILITY,
                    spread,
                    CLOSE_SPREAD_MAX_DEG,
                    recorded_angles,
                    None,
                    attempt,
                )
                continue

            self.distance_zero_angle = _wrapped_mean(recorded_angles)
            spread_mm = self._delta_angle_to_distance(spread, self.gear_radius)
            limit_mm = self._delta_angle_to_distance(
                CLOSE_SPREAD_MAX_DEG, self.gear_radius
            )
            print(
                f"Closes agree to {spread:.2f}° (~{spread_mm:.1f} mm), "
                f"limit {CLOSE_SPREAD_MAX_DEG:.2f}° (~{limit_mm:.1f} mm)"
            )
            print(f"Calibration complete. Zero angle: {self.distance_zero_angle:.2f}")

            input("Open the gripper fully, then press Enter...")
            open_angle = self._capture_angle(NUM_SAMPLES_PER_CAPTURE)

            if open_angle is None:
                print("Error: No angles recorded during direction calibration.")
                self.encoder.stop_streaming()
                return False

            raw_offset = (open_angle - self.distance_zero_angle + 180.0) % 360.0 - 180.0

            # Report lower and upper travel failures separately to distinguish jams
            # from an incorrectly captured mechanical stop.
            travel = abs(raw_offset)
            if travel < self.open_travel_min_deg or travel > self.open_travel_max_deg:
                too_low = travel < self.open_travel_min_deg
                self._retry_or_fault(
                    CHECK_OPEN_TRAVEL if too_low else CHECK_OPEN_TRAVEL_HIGH,
                    travel,
                    self.open_travel_min_deg if too_low else self.open_travel_max_deg,
                    recorded_angles,
                    open_angle,
                    attempt,
                )
                continue

            self.directions = "+" if raw_offset >= 0 else "-"
            self.direction_sign = 1.0 if self.directions == "+" else -1.0
            travel_mm = self._delta_angle_to_distance(travel, self.gear_radius)
            print(
                f"Open angle: {open_angle:.2f}, raw offset: {raw_offset:+.2f} "
                f"(~{travel_mm:.1f} mm of travel), direction: '{self.directions}'"
            )
            break

        self.encoder.stop_streaming()

        # Step 4: Save config (automatic or interactive)
        # Priority: parameter > environment variable > interactive prompt
        save_path_to_use = auto_save_path or os.environ.get(
            "UMI_GRIPPER_AUTO_SAVE_PATH"
        )

        if save_path_to_use:
            print(f"Auto-saving config to: {save_path_to_use}")
            self.save_config(save_path_to_use)
            print("Config saved successfully")
        else:
            save_path = input("Enter path to save config (or press Enter to skip): ")
            if save_path.strip():
                self.save_config(save_path.strip())

        return True

    # ==================== String Representation ====================

    def __repr__(self) -> str:
        """String representation of gripper."""
        cal_status = (
            "calibrated" if self.distance_zero_angle is not None else "not calibrated"
        )
        return (
            f"UmiGripper(device={self.device_name}, "
            f"gear_radius={self.gear_radius}mm, "
            f"distance_cal={cal_status})"
        )


def main() -> None:
    """Quick smoke test for UMI gripper frame parsing.

    Resolves the named gripper to a USB serial port via
    the UMI device registry, opens the UART, parses
    UMI protocol frames through ``UartSensorReader`` + ``EncoderSensor``,
    and prints decoded ``GripperFrame`` fields for ``--duration`` seconds.

    Calibration is skipped (``distance_zero_angle=0.0``) since this is only
    intended to verify the parsing pipeline, not produce meaningful distances.

    Example:
        uv run python -m umi.umi_device.umi_gripper --device UMI-001
    """
    import argparse

    from umi.common.hardware_sdk.sensor import (
        SENSOR_TYPE_JOINT_ENCODER,
        SensorFaultError,
        SensorHealth,
    )
    from umi.common.hardware_sdk.sensor.uart_reader import UartSensorReader

    def format_health(health: SensorHealth | None) -> str:
        if health is None:
            return "health=n/a"

        extra = ""
        if health.extra_decoded:
            extra = " " + " ".join(
                f"{key}=0x{value:04X}" if isinstance(value, int) else f"{key}={value}"
                for key, value in health.extra_decoded.items()
            )
        return (
            f"health_state={health.state.name} "
            f"last_error={health.last_error_name} "
            f"error_count={health.error_count}{extra}"
        )

    parser = argparse.ArgumentParser(description="UMI gripper frame-parsing smoke test")
    parser.add_argument(
        "-d",
        "--device",
        required=True,
        help="UMI gripper device name (e.g. 'UMI-V4-0001') or its raw "
        "16-hex USB serial.",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=1_000_000,
        help="UART baudrate (default: 1_000_000)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Streaming duration in seconds (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=30.0,
        help="Frame poll rate in Hz (default: 30)",
    )
    parser.add_argument(
        "--gear-radius",
        type=float,
        default=26.0,
        help="Gear radius in mm (default: 26.0)",
    )
    args = parser.parse_args()

    reader = UartSensorReader(
        uart_device_name=args.device,
        baudrate=args.baudrate,
    )
    reader.start()

    source: SensorInterface | None = None
    try:
        gripper = UmiGripper(
            device_name=args.device,
            reader=reader,
            gear_radius=args.gear_radius,
            distance_zero_angle=0.0,
            directions="+",
        )
        source = gripper.get_data_sources()[0]
        source.start_streaming()

        time.sleep(0.2)

        period = 1.0 / args.rate
        deadline = (
            time.monotonic() + args.duration if args.duration is not None else None
        )
        n_frames = 0
        last_receive_time: float | None = None
        while deadline is None or time.monotonic() < deadline:
            try:
                frame = source.get_frame()
            except SensorFaultError as exc:
                print(
                    f"[FAULT] sensor=0x{exc.sensor_type:02X}#{exc.sensor_id} "
                    f"{format_health(exc.health)}"
                )
                break
            if (
                frame is not None
                and frame.joint_angles is not None
                and frame.receive_time != last_receive_time
            ):
                angle = float(frame.joint_angles[0])
                distance = (
                    float(frame.distance)
                    if frame.distance is not None
                    else float("nan")
                )
                print(
                    f"[{n_frames:04d}] angle={angle:8.2f} deg  "
                    f"distance={distance:7.2f} mm  "
                    f"recv={frame.receive_time:.4f}  "
                    f"{format_health(gripper.encoder.latest_health(SENSOR_TYPE_JOINT_ENCODER, 0))}"
                )
                n_frames += 1
                last_receive_time = frame.receive_time
            time.sleep(period)

        duration_text = (
            f"{args.duration:.2f}s" if args.duration is not None else "continuous mode"
        )
        print(
            f"\nParsed {n_frames} unique frames in {duration_text} "
            f"from device '{args.device}' on {reader.port}"
        )
    finally:
        if source is not None:
            source.stop_streaming()
        reader.stop()


if __name__ == "__main__":
    main()

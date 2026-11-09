# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DexUMI Exoskeleton device implementation.

Provides signed angular offset computation with direction handling and
combined encoder + tactile sensor data for the DexUMI exoskeleton.
Acts as a data source factory, producing independent data sources for the recorder.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from umi.common.data.frame import create_frame
from umi.common.hardware_sdk.sensor.encoder_sensor import EncoderSensor
from umi.common.hardware_sdk.sensor.tactile_sensor import TactileSensor
from umi.common.hardware_sdk.sensor_interface import SensorInterface
from umi.umi_device.umi_device_interface import UmiDeviceInterface, create_data_source


class DexUmiExoskeleton(UmiDeviceInterface):
    """DexUMI exoskeleton with encoder offset computation and tactile sensing.

    Acts as a data source factory: call get_data_sources() to obtain
    independent data sources (encoder, tactile) for the recorder.

    Attributes:
        encoder: The underlying EncoderSensor instance
        tactile: The underlying TactileSensor instance (optional)

    Example:
        exo = DexUmiExoskeleton(
            device_name="exo",
            reader=uart_reader,
            encoder_sensor_ids=list(range(16)),
            tactile_sensor_ids=[1, 4, 7, 10, 13],
            zero_position_angle=np.zeros(16),
            directions="++++++++++++++++",
        )
        # Get data sources for recorder
        sources = exo.get_data_sources()
    """

    _config_keys = [
        "device_name",
        "encoder_sensor_ids",
        "tactile_sensor_ids",
        "zero_position_angle",
        "directions",
        "fingertip_poly_map",
        "exo_to_robot_joint_map",
        "num_hand_joints",
        "joint_limits",
    ]

    # Separate calibration provenance and accepted compatibility keys from constructor
    # settings.
    _config_metadata_keys = tuple(
        yaml.safe_load(
            Path(__file__).with_name("dexumi_exoskeleton_metadata.yaml").read_text()
        )
    )

    def __init__(
        self,
        device_name: str,
        reader: Any,
        encoder_sensor_ids: list[int],
        tactile_sensor_ids: list[int] | None = None,
        zero_position_angle: np.ndarray | list | None = None,
        directions: str | None = None,
        fingertip_poly_map: dict[int, list[float]] | None = None,
        exo_to_robot_joint_map: list[int] | None = None,
        num_hand_joints: int | None = None,
        joint_limits: list | None = None,
        latency: float | None = None,
        hand_config: str | Path | None = None,
        simulation_only: bool = False,
        geometry_verified: bool | None = None,
    ) -> None:
        """Initialize DexUMI exoskeleton.

        Args:
            device_name: Unique device identifier
            reader: Shared UartSensorReader instance
            encoder_sensor_ids: List of encoder channel IDs
            tactile_sensor_ids: List of tactile sensor IDs. If None, no
                tactile sensor is created.
            zero_position_angle: Zero reference angle per encoder (degrees).
            directions: Per-encoder direction string, e.g. "-+--+-...".
                Each char is '+' or '-'. '+' means increasing encoder angle
                corresponds to positive joint movement. If None, all '+'.
            fingertip_poly_map: Polynomial mapping from exo joint angles to
                robot joint angles. Keys are joint indices, values are
                coefficient lists (highest degree first) for np.polyval.
            exo_to_robot_joint_map: Per-encoder destination joint index on the
                robot hand, length matching encoder_sensor_ids. When combined
                with `num_hand_joints`, `convert_to_robot_actions` scatters its
                output into a robot-hand-DoF vector; robot joints not listed
                remain at 0.0.
            num_hand_joints: Total robot-hand DoF. If set together with
                `exo_to_robot_joint_map`, enables the scatter step described
                above. Leave None to keep the legacy behavior of returning
                exo-DoF outputs.
            joint_limits: Optional per-joint ``[min, max]`` limits (degrees)
                carried in the device config. Not used by the online path; it
                is consumed by the offline ``convert_to_robot_actions`` /
                ``load_joint_limits`` classmethods (which read it from the
                config file directly). Accepted and round-tripped here so
                ``from_config()`` does not reject the key.
            latency: Latency for data sources
            hand_config: Optional path to the config yaml of the robot hand
                this exoskeleton mirrors (e.g.
                ``umi/control/robot/configs/xhand_right.yaml``). Backs the
                lazy ``hand_kinematics`` / ``hand_tactile`` model used by the
                tactile robot-action conversion; recording never needs it.
            simulation_only: Marks placeholder calibration as safe only for
                mock/offline use. Real-source creation rejects such configs
                before opening a UART.
            geometry_verified: Whether production hand geometry/calibration
                has been measured. ``False`` is rejected for a real source;
                ``None`` keeps legacy configs compatible.
        """
        super().__init__()

        self.device_name = device_name
        self.latency = latency
        self.encoder_sensor_ids = list(encoder_sensor_ids)
        self.tactile_sensor_ids = (
            list(tactile_sensor_ids) if tactile_sensor_ids is not None else None
        )

        self.encoder = EncoderSensor(
            device_name=f"{device_name}_encoder",
            reader=reader,
            sensor_ids=self.encoder_sensor_ids,
        )
        self.tactile = None
        if self.tactile_sensor_ids:
            self.tactile = TactileSensor(
                device_name=f"{device_name}_tactile",
                reader=reader,
                sensor_ids=self.tactile_sensor_ids,
            )

        n_encoders = len(self.encoder_sensor_ids)
        if directions is not None:
            if len(directions) != n_encoders:
                raise ValueError(
                    f"directions length ({len(directions)}) must match "
                    f"encoder_sensor_ids length ({n_encoders})"
                )
            self.direction_signs = np.array(
                [1.0 if c == "+" else -1.0 for c in directions]
            )
        else:
            self.direction_signs = np.ones(n_encoders)
        self.directions = directions or "+" * n_encoders

        if fingertip_poly_map is not None:
            self.fingertip_poly_map = {
                int(k): np.asarray(v, dtype=np.float64)
                for k, v in fingertip_poly_map.items()
            }
        else:
            self.fingertip_poly_map = None

        if exo_to_robot_joint_map is not None:
            if len(exo_to_robot_joint_map) != n_encoders:
                raise ValueError(
                    f"exo_to_robot_joint_map length ({len(exo_to_robot_joint_map)}) "
                    f"must match encoder_sensor_ids length ({n_encoders})"
                )
            self.exo_to_robot_joint_map = np.asarray(
                exo_to_robot_joint_map, dtype=np.int64
            )
        else:
            self.exo_to_robot_joint_map = None

        self.num_hand_joints = (
            int(num_hand_joints) if num_hand_joints is not None else None
        )
        # Keep configured limits in degrees for serialization; cache a
        # (num_joints, 2) radians array for per-frame action clamping. Offline
        # conversion reads limits directly from the config.
        self.joint_limits = joint_limits
        self.joint_limits_rad = (
            np.deg2rad(np.asarray(joint_limits, dtype=np.float64))
            if joint_limits is not None
            else None
        )

        # Encoder frame class (separate from tactile for independent timestamps)
        self.encoder_frame_class = create_frame(
            f"{device_name}_encoder_Frame",
            {
                "joint_angles": {"interpolate": True},
                "raw_encoder_angles": {"interpolate": True},
            },
        )

        if zero_position_angle is not None:
            self.zero_position_angle = np.asarray(zero_position_angle, dtype=np.float64)
        else:
            self.zero_position_angle = None

        # Model of the mirrored robot hand, built lazily on first
        # hand_kinematics / hand_tactile access (see _build_hand_model).
        self._hand_config_path = hand_config
        self.simulation_only = bool(simulation_only)
        self.geometry_verified = geometry_verified
        self._hand_kinematics = None
        self._hand_tactile = None
        self._hand_use_point = "mean"

    # ==================== Data Source Factory ====================

    def get_data_sources(self) -> list[SensorInterface]:
        """Return independent data sources for the recorder.

        Returns:
            List containing encoder data source (always) and tactile data
            source (if tactile sensors are configured).
        """
        self._validate_required_config()
        sources: list[SensorInterface] = [
            create_data_source(
                sensor=self.encoder,
                device_name=f"{self.device_name}_encoder",
                latency=self.latency,
                frame_class=self.encoder_frame_class,
                get_frame_fn=self._get_encoder_frame,
            )
        ]
        if self.tactile is not None:
            sources.append(
                create_data_source(
                    sensor=self.tactile,
                    device_name=f"{self.device_name}_tactile",
                    latency=self.latency,
                    frame_class=self.tactile.frame_class,
                )
            )
        return sources

    def _get_encoder_frame(self):
        """Read encoder and return frame with signed angle offsets."""
        encoder_frame = self.encoder.get_frame()
        if encoder_frame is None:
            return None
        if encoder_frame.position is None or len(encoder_frame.position) == 0:
            return None

        raw_angles = np.array(encoder_frame.position, dtype=np.float64)
        joint_angles = self.get_signed_angle_offset(raw_angles)

        return self.encoder_frame_class(
            joint_angles=joint_angles,
            raw_encoder_angles=raw_angles,
            capture_time=encoder_frame.capture_time,
            receive_time=encoder_frame.receive_time,
            frame_source=f"{self.device_name}_encoder",
        )

    # ==================== Robot Action Conversion (offline) ====================

    # Recorded field fed to convert_to_robot_actions in offline conversion: the
    # exoskeleton's calibrated value is the signed joint angle (degrees).
    action_source_key = "joint_angles_interp"

    @classmethod
    def convert_to_robot_actions(
        cls,
        data: np.ndarray,
        mapping: dict[int, np.ndarray] | None = None,
        joint_map: np.ndarray | None = None,
        num_hand_joints: int | None = None,
        umi_config: str | Path | None = None,
        joint_limits_rad: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Convert recorded joint angles (degrees) to robot joint positions (radians).

        Works without hardware, so it suits offline conversion of recorded data.

        Two call styles are supported:

        - Pre-resolved: pass ``mapping`` / ``joint_map`` / ``num_hand_joints``
          directly.
        - Config path (offline conversion): pass ``umi_config`` (the saved
          exoskeleton device config); this loads ``mapping``, ``joint_map``,
          ``num_hand_joints`` and (when present) ``joint_limits`` from that
          single file. Explicitly-passed values take precedence over
          config-loaded ones.

        When per-joint limits are available (``joint_limits`` in ``umi_config``,
        or pre-resolved ``joint_limits_rad``), the converted action is clamped
        into range so converted commands stay reachable. ``joint_limits``
        is a generic robot-DoF field carried by the device config — this method
        stays hand-agnostic (plain min/max clip, no hand-specific code).

        Args:
            data: Joint angles in degrees. May be 1D `(N_exo,)` (single
                waypoint) or 2D `(T, N_exo)` (loop over time in the caller
                if you prefer per-row processing).
            mapping: Optional polynomial mapping per joint index.
            joint_map: Optional per-encoder destination joint index on the
                robot hand. When given together with `num_hand_joints`, the
                returned array is scattered into a robot-hand-DoF vector;
                robot joints not listed remain at 0.0.
            num_hand_joints: Total robot-hand DoF. Required iff `joint_map`
                is given.
            umi_config: Path to the saved exoskeleton device config YAML; when
                given, supplies `mapping`/`joint_map`/`num_hand_joints` and
                `joint_limits` (each unless passed explicitly).
            joint_limits_rad: Pre-resolved per-joint [min, max] limits in radians,
                shape `(num_hand_joints, 2)`. When provided, the converted action
                is clamped into range.

        Returns:
            Joint positions in radians. Shape matches `data` when no scatter
            is requested; otherwise the last axis becomes `num_hand_joints`.
        """
        # Resolve the calibration config supplied by the offline caller before
        # converting actions.
        if umi_config is not None:
            if mapping is None:
                mapping = cls.load_mapping(umi_config)
            if joint_map is None:
                joint_map = cls.load_joint_map(umi_config)
            if num_hand_joints is None:
                num_hand_joints = cls.load_num_hand_joints(umi_config)
            if joint_limits_rad is None:
                joint_limits_rad = cls.load_joint_limits(umi_config)

        if mapping is not None:
            mapped = data.copy()
            for idx, coeffs in mapping.items():
                if idx < mapped.shape[-1]:
                    mapped[..., idx] = np.polyval(coeffs, data[..., idx])
        else:
            mapped = data
        actions_rad = np.deg2rad(mapped)

        if joint_map is None and num_hand_joints is None:
            result = actions_rad
        else:
            if joint_map is None or num_hand_joints is None:
                raise ValueError(
                    "joint_map and num_hand_joints must be provided together"
                )
            if len(joint_map) != actions_rad.shape[-1]:
                raise ValueError(
                    f"joint_map length ({len(joint_map)}) must match exo width "
                    f"({actions_rad.shape[-1]})"
                )
            if joint_map.size and int(joint_map.max()) >= num_hand_joints:
                raise ValueError(
                    f"joint_map references joint index {int(joint_map.max())} which "
                    f"exceeds num_hand_joints={num_hand_joints}"
                )
            out = np.zeros(
                actions_rad.shape[:-1] + (num_hand_joints,), dtype=np.float64
            )
            out[..., joint_map] = actions_rad
            result = out

        if joint_limits_rad is not None:
            if joint_limits_rad.shape[0] != result.shape[-1]:
                raise ValueError(
                    f"joint_limits length ({joint_limits_rad.shape[0]}) must match "
                    f"action width ({result.shape[-1]})"
                )
            result = np.clip(result, joint_limits_rad[:, 0], joint_limits_rad[:, 1])
        return result

    @classmethod
    def resolve_config_file(cls, uart_device_name: str) -> str | None:
        """Each exoskeleton name maps to exactly one calibration config."""
        from umi.utils.device_manager import get_exoskeleton_config_path

        return get_exoskeleton_config_path(uart_device_name, require_exists=False)

    @classmethod
    def validate_config_file(cls, config_path: str) -> None:
        """Reject placeholder calibrations before any hardware is opened.

        Placeholder calibration/geometry supports the mock recorder and offline
        compliance flow, but it must never drive a real hand.
        """
        if not os.path.isfile(config_path):
            return
        with open(config_path) as f:
            device_config = yaml.safe_load(f) or {}
        if device_config.get("simulation_only", False) or (
            device_config.get("geometry_verified") is False
        ):
            raise ValueError(
                f"{cls.__name__} calibration {config_path!r} is simulation-only; "
                "replace its placeholder calibration/geometry and set "
                "geometry_verified: true before connecting real hardware"
            )

    @staticmethod
    def _load_config_dict(config_path: str | Path | None) -> dict | None:
        """Read a YAML config into a dict, or return None if the path is missing."""
        if config_path is None or not os.path.exists(config_path):
            return None
        with open(config_path) as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def load_mapping(config_path: str | Path | None) -> dict[int, np.ndarray] | None:
        """Load fingertip_poly_map from a config YAML file."""
        config = DexUmiExoskeleton._load_config_dict(config_path)
        if config is None:
            return None
        poly_map = config.get("fingertip_poly_map")
        if poly_map is None:
            return None
        return {int(k): np.asarray(v, dtype=np.float64) for k, v in poly_map.items()}

    @staticmethod
    def load_joint_map(config_path: str | Path | None) -> np.ndarray | None:
        """Load exo_to_robot_joint_map from a config YAML file."""
        config = DexUmiExoskeleton._load_config_dict(config_path)
        if config is None:
            return None
        joint_map = config.get("exo_to_robot_joint_map")
        if joint_map is None:
            return None
        return np.asarray(joint_map, dtype=np.int64)

    @staticmethod
    def load_num_hand_joints(config_path: str | Path | None) -> int | None:
        """Read ``num_hand_joints`` (total robot-hand DoF) from the exoskeleton
        device config YAML.

        Sizes the exo->hand scatter when converting recorded exoskeleton angles
        (e.g. 15 exo DoF) into the connected hand's full DoF (e.g. 16 joints);
        robot joints not driven by any encoder are held at 0.0.
        Returns None if the path is missing or the key is absent.
        """
        config = DexUmiExoskeleton._load_config_dict(config_path)
        if config is None:
            return None
        return config.get("num_hand_joints")

    @staticmethod
    def load_joint_limits(config_path: str | Path | None) -> np.ndarray | None:
        """Load per-joint [min, max] limits from a device config, in radians.

        ``joint_limits`` is a generic robot-DoF field (list of ``[min, max]``
        pairs in degrees) carried by the exoskeleton config so offline
        conversion can clamp the converted action without a separate hand
        config — keeping this device hand-agnostic. Returns a
        ``(num_joints, 2)`` float array in radians, or None if the path is
        missing or has no ``joint_limits``.
        """
        config = DexUmiExoskeleton._load_config_dict(config_path)
        if config is None:
            return None
        raw = config.get("joint_limits")
        if raw is None:
            return None
        return np.deg2rad(np.asarray(raw, dtype=np.float64))

    # ==================== Mirrored Hand Model (offline) ====================

    # The exoskeleton mirrors a specific robot hand (the X Hand) and
    # carries the SAME tactile sensors on its fingertips, so a kinematics +
    # tactile model of that hand — built from the hand's own config yaml —
    # is enough to reason about recorded exoskeleton data: FK for mapping
    # recorded fingertip forces into the hand-base frame, and Jacobians for
    # projecting them onto the actuated joints (tau = B^T J^T f).

    def _build_hand_model(self) -> None:
        if self._hand_kinematics is not None:
            return
        if self._hand_config_path is None:
            raise RuntimeError(
                "no hand_config was provided — construct the exoskeleton "
                "with hand_config=<mirrored hand yaml> (e.g. "
                "umi/control/robot/configs/xhand_right.yaml) to use "
                "hand_kinematics / hand_tactile / compute_joint_torque"
            )
        # Imported here, not at module top: the control stack (Pinocchio)
        # must not load on recording stations, which construct this device
        # but never touch the hand model.
        from umi.control.robot.hand.kinematics import HandKinematics
        from umi.control.robot.hand.tactile import (
            HandTactile,
            load_tactile_sensor_specs,
        )

        config_path = Path(self._hand_config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        kinematics = HandKinematics.from_config(config, config_path.parent)
        specs = load_tactile_sensor_specs(config)
        if specs is None:
            raise ValueError(
                f"hand config {config_path} has no tactile block — the "
                "mirrored-hand model needs the fingertip sensor specs"
            )
        self._hand_tactile = HandTactile(kinematics, specs)
        self._hand_use_point = config.get("tactile_use_point", "mean")
        self._hand_kinematics = kinematics

    @property
    def hand_kinematics(self):
        """``HandKinematics`` of the mirrored robot hand (lazy-built from
        ``hand_config``)."""
        self._build_hand_model()
        return self._hand_kinematics

    @property
    def hand_tactile(self):
        """``HandTactile`` of the mirrored robot hand (lazy-built from
        ``hand_config``); shares the fingertip sensor specs with the live
        hand."""
        self._build_hand_model()
        return self._hand_tactile

    def compute_joint_torque(
        self,
        q_state_rad: np.ndarray,
        tactile_arrays: dict,
        *,
        ema_alpha: float = 0.3,
        deadband_n: float = 0.1,
        components: str | tuple = ("z",),
        tactile_use_point: str | None = None,
    ) -> np.ndarray:
        """Per-frame actuated-joint torques ``tau = B^T J^T f`` from recorded
        fingertip tactile arrays.

        Calibrate the recorded arrays, EMA-smooth and deadband-gate them per
        finger, map the surviving sensor-frame forces into the hand-base frame
        at that frame's pose, and project them onto the actuated joints.
        Stateless — the EMA lives in the call, never on ``self``.

        Args:
            q_state_rad: ``(N, n_actuated)`` kinematically converted joint
                trajectory in radians. Only used to evaluate FK / Jacobians
                per frame; it is never modified.
            tactile_arrays: Recorded tactile channels in the shared
                ``tactile_{sensor_id}_f{x,y,z}_{1,2}[_interp]`` schema, each
                ``(N,)`` raw counts (see
                ``HandTactile.forces_in_sensor_frame_from_arrays``).
            ema_alpha: EMA factor on the calibrated sensor-frame forces
                (1.0 = no smoothing).
            deadband_n: Force-norm deadband in Newtons (below -> no force).
                The exoskeleton's sensors are 3-dim forces, so the full-vector
                norm IS the force norm — no separate force-only deadband.
            components: Sensor-frame force components entering the torque
                math — 'all', a single token 'x' | 'y' | 'z' | 'normal',
                or an iterable of tokens (e.g. ``("y", "z")``); see
                ``HandTactile.forces_in_hand_base``. Default: pad-normal
                only (``("z",)`` — every current hand's normal axis is z).
            tactile_use_point: first | second | mean; None uses the hand
                config's ``tactile_use_point``.

        Returns:
            ``(N, n_actuated)`` joint torques in N*m. Frames where every finger
            sits below the deadband are exactly zero.
        """
        if not 0 < ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")
        if deadband_n < 0:
            raise ValueError("deadband_n must be >= 0")
        from umi.control.robot.hand.tactile import resolve_force_components

        components = resolve_force_components(components)

        kinematics = self.hand_kinematics
        q_state_rad = np.asarray(q_state_rad, dtype=np.float64)
        if q_state_rad.ndim != 2 or q_state_rad.shape[1] != kinematics.n_actuated:
            raise ValueError(
                f"q_state_rad must be (N, {kinematics.n_actuated}), "
                f"got {q_state_rad.shape}"
            )
        num_frames = q_state_rad.shape[0]

        use_point = (
            tactile_use_point if tactile_use_point is not None else self._hand_use_point
        )
        forces = self.hand_tactile.forces_in_sensor_frame_from_arrays(
            tactile_arrays, use_point=use_point
        )
        for finger, force in forces.items():
            if force.shape[0] != num_frames:
                raise ValueError(
                    f"tactile arrays for finger '{finger}' have "
                    f"{force.shape[0]} frames, expected {num_frames}"
                )

        torques = np.zeros((num_frames, kinematics.n_actuated))
        ema = {finger: np.zeros(force.shape[1]) for finger, force in forces.items()}
        for t in range(num_frames):
            f_use = {}
            for finger, force in forces.items():
                ema[finger] = (1 - ema_alpha) * ema[finger] + ema_alpha * force[t]
                in_contact = np.linalg.norm(ema[finger]) >= deadband_n
                f_use[finger] = (
                    ema[finger] if in_contact else np.zeros_like(ema[finger])
                )
            # Nothing above the deadband -> exactly zero torque, and no reason
            # to pay for FK + Jacobians on that frame.
            if not any(np.any(f) for f in f_use.values()):
                continue
            f_base = self.hand_tactile.forces_in_hand_base(
                f_use, q_state_rad[t], components=components
            )
            torques[t] = kinematics.joint_torques_from_fingertip_forces(
                q_state_rad[t], f_base
            )
        return torques

    # ==================== Angle Offset Computation ====================

    def get_signed_angle_offset(self, current_angles: np.ndarray) -> np.ndarray:
        """Compute signed angular offset from zero position with wrapping.

        Uses modular arithmetic to handle 360->0 encoder wrapping.
        Applies per-encoder direction signs.

        Formula: ((current - zero + 180) % 360 - 180) * direction_sign

        Args:
            current_angles: Current encoder readings in degrees

        Returns:
            Signed angular offsets in degrees, shape matching current_angles
        """
        diff = (current_angles - self.zero_position_angle + 180) % 360 - 180
        return diff * self.direction_signs

    # ==================== String Representation ====================

    def __repr__(self) -> str:
        cal_status = (
            "calibrated" if self.zero_position_angle is not None else "not calibrated"
        )
        n_tactile = len(self.tactile_sensor_ids) if self.tactile_sensor_ids else 0
        return (
            f"DexUmiExoskeleton(device={self.device_name}, "
            f"encoders={len(self.encoder_sensor_ids)}, "
            f"tactile={n_tactile}, "
            f"{cal_status})"
        )

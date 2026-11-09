# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tactile sensing for hands: sensor specs, hardware reading converters, and
the hand-agnostic processing pipeline.

The pipeline (owned end-to-end by ``HandTactile``)::

    state frame
      --TactileReadingConverter-->  {finger: RAW sensor-frame vector}
                                    (hardware units; hardware-specific)
      --forces_in_sensor_frame-->   {finger: sensor-frame force, NEWTONS}
                                    (axis_signs, force_unit_scale,
                                     measurement_dim slice, non-finite -> None)
      --forces_in_hand_base----->   {finger: hand-base force/wrench}
                                    (mounting R_fingertip_sensor /
                                     t_fingertip_sensor_m, then FK
                                     R_base_fingertip(q))

Transform naming follows the T_A_B convention: ``T_A_B`` is B's pose in
frame A, so ``R_A_B`` maps B-frame vectors into A-frame coordinates
(``v_A = R_A_B @ v_B``) and ``t_A_B`` is B's origin expressed in frame A.

Unit boundary: raw counts become Newtons in exactly one place —
``HandTactile.forces_in_sensor_frame`` (via each spec's
``force_unit_scale``). Converters extract raw readings and do no
calibration.
"""

import abc
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from umi.control.robot.hand.kinematics import HandKinematics

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

# Single-string selections used by the compliance suite's config/GUI enum.
FORCE_COMPONENT_CHOICES = ("all", "normal", "x", "y")

# Axis tokens accepted inside a ``components`` iterable ('normal' resolves
# to each spec's normal_axis). Moments are kept only by 'all' — any
# explicit token selection zeroes them (opt into the unverified moment
# channels via 'all' alone).
FORCE_AXIS_TOKENS = ("x", "y", "z", "normal")


def resolve_force_components(components):
    """Canonicalize a force-component selection.

    Accepts None (= 'all'), the string 'all', a single axis-token string,
    or an iterable of tokens from ``FORCE_AXIS_TOKENS``. Returns 'all' or
    a deduped tuple of tokens; empty/unknown selections raise."""
    if components is None or components == "all":
        return "all"
    if isinstance(components, str):
        components = (components,)
    tokens = tuple(dict.fromkeys(components))
    bad = sorted(set(tokens) - set(FORCE_AXIS_TOKENS))
    if not tokens or bad:
        raise ValueError(
            f"components must be 'all' or a non-empty selection of "
            f"{FORCE_AXIS_TOKENS}, got {components!r}"
        )
    return tokens


# Old-name -> new-name map for the T_A_B convention rename; the loader
# rejects the old keys loudly instead of silently ignoring them (which would
# strip a finger's mounting rotation).
_RENAMED_TACTILE_KEYS = {
    "rotation_sensor_to_fingertip": "R_fingertip_sensor",
    "translation_sensor_to_fingertip_m": "t_fingertip_sensor_m",
}

# Keys accepted per tactile.fingers.<name> entry (the TactileSensorSpec
# fields), and at the tactile block level (per-sensor defaults + the
# fingers mapping; sensor_id is per-finger only).
_TACTILE_FINGER_KEYS = {
    "sensor_id",
    "measurement_dim",
    "normal_axis",
    "axis_signs",
    "force_unit_scale",
    "moment_unit_scale",
    "R_fingertip_sensor",
    "t_fingertip_sensor_m",
}
_TACTILE_BLOCK_KEYS = (_TACTILE_FINGER_KEYS - {"sensor_id"}) | {"fingers"}


@dataclass
class TactileSensorSpec:
    """One tactile sensor: measurement spec + fingertip mounting.

    The finger the sensor is mounted on is the key of the
    ``{finger: TactileSensorSpec}`` mapping the spec lives in — the spec
    itself carries no finger name, so it cannot disagree with the mapping.

    Args:
        sensor_id: Hardware sensor id.
        measurement_dim: 1 (pad-normal force scalar), 3 (force vector), or
            6 (wrench ``[f; m]``).
        normal_axis: Sensor axis carrying the pad-normal force. A 1-DOF
            reading is embedded on this axis.
        axis_signs: Per-axis ±1 polarity correction applied to raw readings
            so pressing reads POSITIVE on the normal axis. A flipped axis
            flips both its force and its moment component.
        force_unit_scale: Newtons per raw count (e.g. 1e-3 when the hardware
            reports millinewtons). The single place raw counts become SI.
        moment_unit_scale: N*m per raw count for the moment half of a 6-DOF
            wrench (measurement_dim 6 only; 1.0 when the hardware already
            reports N*m).
        R_fingertip_sensor: ``(3, 3)`` sensor orientation in the fingertip
            frame (T_A_B convention): maps (polarity-corrected) sensor-frame
            vectors into the fingertip frame,
            ``v_fingertip = R_fingertip_sensor @ v_sensor``. A pure mounting
            rotation — never use it to fix polarity.
        t_fingertip_sensor_m: ``(3,)`` sensor origin in the fingertip frame;
            only enters the 6-DOF moment transport
            (``m_fingertip = R m + t x (R f)``).
    """

    sensor_id: int
    measurement_dim: int = 3
    normal_axis: str = "z"
    axis_signs: np.ndarray = field(default_factory=lambda: np.ones(3))
    force_unit_scale: float = 1.0
    moment_unit_scale: float = 1.0
    R_fingertip_sensor: np.ndarray = field(default_factory=lambda: np.eye(3))
    t_fingertip_sensor_m: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        if self.measurement_dim not in (1, 3, 6):
            raise ValueError(
                f"measurement_dim must be 1, 3, or 6, got {self.measurement_dim}"
            )
        if self.normal_axis not in _AXIS_INDEX:
            raise ValueError(f"normal_axis must be x|y|z, got {self.normal_axis!r}")
        self.axis_signs = np.asarray(self.axis_signs, dtype=np.float64)
        if self.axis_signs.shape != (3,) or not np.all(np.abs(self.axis_signs) == 1):
            raise ValueError("axis_signs must be three entries of +1/-1")
        self.R_fingertip_sensor = np.asarray(self.R_fingertip_sensor, dtype=np.float64)
        if self.R_fingertip_sensor.shape != (3, 3):
            raise ValueError("R_fingertip_sensor must be a 3x3 matrix")
        self.t_fingertip_sensor_m = np.asarray(
            self.t_fingertip_sensor_m, dtype=np.float64
        )
        if self.t_fingertip_sensor_m.shape != (3,):
            raise ValueError("t_fingertip_sensor_m must be a 3-vector")
        if self.force_unit_scale <= 0:
            raise ValueError("force_unit_scale must be positive (N per raw count)")
        if self.moment_unit_scale <= 0:
            raise ValueError("moment_unit_scale must be positive (N*m per raw count)")

    @property
    def normal_axis_index(self) -> int:
        return _AXIS_INDEX[self.normal_axis]


def _reject_renamed_keys(entry: dict, where: str) -> None:
    for old, new in _RENAMED_TACTILE_KEYS.items():
        if old in entry:
            raise ValueError(
                f"tactile key '{old}' in {where} was renamed '{new}' "
                "(T_A_B convention: R_fingertip_sensor maps sensor-frame "
                "vectors into the fingertip frame) — update the yaml"
            )


def _reject_unknown_keys(entry: dict, allowed: set, where: str) -> None:
    # A silently ignored key (typo, or an old suite-config name like
    # sensor_signs/force_scale/sensor_to_tip) would fall back to defaults —
    # e.g. dropping a station's polarity fix or a finger's mounting rotation.
    unknown = sorted(set(entry) - allowed)
    if unknown:
        raise ValueError(
            f"unknown tactile key(s) {unknown} in {where} (allowed: {sorted(allowed)})"
        )


def load_tactile_sensor_specs(config: dict) -> Optional[dict[str, TactileSensorSpec]]:
    """Parse a hand yaml's ``tactile:`` block into
    ``{finger: TactileSensorSpec}``; None when the config has no ``tactile:``
    block or the block lists no fingers.

    Keys mirror the TactileSensorSpec fields exactly. Block-level keys are
    the defaults shared by every sensor (``measurement_dim``,
    ``force_unit_scale``, ``axis_signs``, ``normal_axis``,
    ``R_fingertip_sensor``, ``t_fingertip_sensor_m``); each
    ``fingers.<name>`` entry contributes its ``sensor_id`` and may override
    any default (frame conventions differ per finger — e.g. the X Hand
    thumb's mounting rotation). Keys outside the spec fields raise at
    either level, so typos and legacy names never fall back to defaults
    silently. Finger names must exist in the
    ``kinematics.fingers`` wiring when a kinematics block is present: the
    fingertip frame is what the mounting rotation maps into.
    """
    tactile_config = config.get("tactile")
    if not tactile_config:
        return None
    _reject_renamed_keys(tactile_config, "the tactile block")
    _reject_unknown_keys(tactile_config, _TACTILE_BLOCK_KEYS, "the tactile block")
    fingers_config = tactile_config.get("fingers")
    if not fingers_config:
        kinematics_fingers = (config.get("kinematics") or {}).get("fingers") or {}
        if any("sensor_id" in (fc or {}) for fc in kinematics_fingers.values()):
            raise ValueError(
                "per-finger tactile keys moved: put sensor_id (and any "
                "R_fingertip_sensor override) under "
                "tactile.fingers.<name>, not kinematics.fingers.<name>"
            )
        return None

    kinematics_fingers = (config.get("kinematics") or {}).get("fingers")
    if kinematics_fingers is not None:
        unknown = sorted(set(fingers_config) - set(kinematics_fingers))
        if unknown:
            raise ValueError(
                f"tactile fingers {unknown} have no kinematics.fingers entry"
            )

    defaults = {key: value for key, value in tactile_config.items() if key != "fingers"}
    specs = {}
    for name, fc in fingers_config.items():
        _reject_renamed_keys(fc or {}, f"tactile.fingers.{name}")
        _reject_unknown_keys(fc or {}, _TACTILE_FINGER_KEYS, f"tactile.fingers.{name}")
        entry = {**defaults, **(fc or {})}
        if entry.get("sensor_id") is None:
            raise ValueError(f"tactile finger '{name}' has no sensor_id")
        specs[name] = TactileSensorSpec(
            sensor_id=int(entry["sensor_id"]),
            measurement_dim=int(entry.get("measurement_dim", 3)),
            normal_axis=entry.get("normal_axis", "z"),
            axis_signs=entry.get("axis_signs", np.ones(3)),
            force_unit_scale=float(entry.get("force_unit_scale", 1.0)),
            moment_unit_scale=float(entry.get("moment_unit_scale", 1.0)),
            R_fingertip_sensor=entry.get("R_fingertip_sensor", np.eye(3).tolist()),
            t_fingertip_sensor_m=entry.get("t_fingertip_sensor_m", np.zeros(3)),
        )

    sensor_ids = [spec.sensor_id for spec in specs.values()]
    if len(set(sensor_ids)) != len(sensor_ids):
        # A duplicate would silently shadow one finger's readings, corrupting
        # recordings and blinding any force cap watching that finger.
        raise ValueError(f"duplicate sensor_ids across fingers: {sorted(sensor_ids)}")
    return specs


class TactileReadingConverter(abc.ABC):
    """Hardware-specific extraction of raw tactile readings from a state.

    Each hand hardware implements one: it knows its state frame's field
    layout (or payload format) and returns per-finger RAW sensor-frame
    vectors in hardware units, exactly as reported. No calibration —
    polarity, unit scaling, and measurement-dim handling are
    ``HandTactile``'s job. Entries may be None or contain NaNs for
    dropped-out sensors; ``HandTactile`` maps non-finite readings to None.
    """

    @abc.abstractmethod
    def raw_readings(self, state) -> dict[str, Optional[np.ndarray]]:
        """``{finger: raw sensor-frame vector | None}`` extracted from
        ``state``, in hardware units."""


class HandTactile:
    """Hand-agnostic tactile pipeline: hardware state -> sensor-frame
    Newtons -> fingertip frame -> hand-base frame (see module docstring).

    Reached as ``hand.tactile`` on a hand with a ``tactile:`` config block
    (sharing the hand's kinematics lock and state provider), or constructed
    standalone for offline/sim math. pinocchio's scratch data is shared
    mutable state: never drive a live hand's ``HandKinematics`` from a
    standalone instance without passing the hand's lock — or FK on a
    ``kinematics.clone()``.

    Args:
        kinematics: The hand's ``HandKinematics``; provides each finger's
            ``R_base_fingertip(q)`` via ``fingertip_pose``.
        sensors: ``{finger: TactileSensorSpec}``; finger names must exist
            in the kinematics description.
        converter: Hardware-specific ``TactileReadingConverter`` backing
            ``forces_in_sensor_frame``. None is fine for geometry-only use
            (``forces_in_hand_base`` on readings obtained elsewhere).
        kinematics_lock: Lock serializing access to the kinematics' shared
            pinocchio scratch data. Pass the owning hand's lock; standalone
            instances default to a private one.
        state_provider: Zero-arg callable returning the latest state frame,
            used by ``forces_in_sensor_frame(state=None)``; a hand wires
            its ``get_state``.
    """

    def __init__(
        self,
        kinematics: "HandKinematics",
        sensors: dict[str, TactileSensorSpec],
        converter: Optional[TactileReadingConverter] = None,
        *,
        kinematics_lock: Optional[threading.Lock] = None,
        state_provider: Optional[Callable[[], Any]] = None,
    ):
        if not sensors:
            raise ValueError(
                "HandTactile requires a non-empty {finger: TactileSensorSpec}"
            )
        unknown = sorted(set(sensors) - set(kinematics.finger_names))
        if unknown:
            raise ValueError(
                f"tactile sensors on unknown fingers {unknown} "
                f"(kinematics fingers: {list(kinematics.finger_names)})"
            )
        self._kinematics = kinematics
        self._sensors = dict(sensors)
        self._converter = converter
        self._lock = (
            kinematics_lock if kinematics_lock is not None else threading.Lock()
        )
        self._state_provider = state_provider

    @property
    def sensors(self) -> dict[str, TactileSensorSpec]:
        """Copy of the ``{finger: TactileSensorSpec}`` mapping."""
        return dict(self._sensors)

    @property
    def fingers(self) -> tuple[str, ...]:
        """Fingers with a configured tactile sensor, in spec order."""
        return tuple(self._sensors)

    def forces_in_sensor_frame(self, state=None) -> dict[str, Optional[np.ndarray]]:
        """Calibrated sensor-frame forces in NEWTONS, keyed by finger.

        Runs the hardware converter on ``state`` (default: the latest state
        from the state provider), then applies each spec's calibration:
        ``axis_signs`` polarity, ``force_unit_scale`` (the single place raw
        counts become SI), and ``measurement_dim`` slicing (1 -> the
        ``(1,)`` pad-normal component). A finger whose raw reading is
        missing or non-finite maps to None so callers can track dropouts.
        """
        if self._converter is None:
            raise RuntimeError(
                "this hand provides no tactile reading converter — raw "
                "state decode is hardware-specific (construct HandTactile "
                "with a TactileReadingConverter)"
            )
        if state is None:
            if self._state_provider is None:
                raise RuntimeError(
                    "no state provider configured — pass the state frame explicitly"
                )
            state = self._state_provider()
        raw_readings = self._converter.raw_readings(state)
        out: dict[str, Optional[np.ndarray]] = {}
        for finger, spec in self._sensors.items():
            raw = raw_readings.get(finger)
            if raw is not None:
                raw = np.asarray(raw, dtype=np.float64)
            if raw is None or not np.all(np.isfinite(raw)):
                out[finger] = None
                continue
            out[finger] = self._calibrate(spec, raw)
        return out

    @staticmethod
    def _calibrate(spec: TactileSensorSpec, raw: np.ndarray) -> np.ndarray:
        """Raw sensor-frame reading(s) -> Newtons: polarity signs, unit
        scale, then ``measurement_dim`` slicing. Accepts a single ``(3,)``
        reading (``(6,)`` wrench for dim-6 specs) or a batch ``(N, 3)`` /
        ``(N, 6)``; the calibration broadcasts."""
        if spec.measurement_dim == 6:
            if raw.shape[-1:] != (6,):
                raise ValueError(
                    f"expected a (..., 6) raw wrench for sensor {spec.sensor_id}, "
                    f"got shape {raw.shape}"
                )
            # A flipped axis flips both its force and its moment component.
            signs = np.concatenate([spec.axis_signs, spec.axis_signs])
            scale = np.concatenate(
                [np.full(3, spec.force_unit_scale), np.full(3, spec.moment_unit_scale)]
            )
            return scale * (signs * raw)
        if raw.shape[-1:] != (3,):
            raise ValueError(
                f"expected a (..., 3) raw reading for sensor {spec.sensor_id}, "
                f"got shape {raw.shape}"
            )
        calibrated = spec.force_unit_scale * (spec.axis_signs * raw)
        if spec.measurement_dim == 1:
            return calibrated[..., [spec.normal_axis_index]]
        return calibrated

    def forces_in_sensor_frame_from_arrays(
        self, arrays, *, use_point: str = "mean"
    ) -> dict[str, np.ndarray]:
        """Calibrated sensor-frame forces in NEWTONS from RECORDED tactile
        arrays, keyed by finger — the offline twin of
        ``forces_in_sensor_frame``.

        ``arrays`` maps recorded field names to per-frame arrays in the
        shared DexUMI-exoskeleton / X Hand tactile schema
        ``tactile_{sensor_id}_f{x,y,z}_{1,2}`` (raw hardware counts; the
        ``_interp``-suffixed names of aligned datasets are accepted too).
        The two recorded measurement points are combined per ``use_point``
        (``first | second | mean``, the same choice as on the live hand),
        then each spec's calibration
        is applied via ``_calibrate``. A frame whose combined reading is
        non-finite calibrates to ZEROS — offline dropout means "no force",
        unlike the live path's per-frame None — so every finger returns a
        dense ``(N, measurement_dim)`` array.
        """
        if use_point not in ("first", "second", "mean"):
            raise ValueError(f"use_point must be first|second|mean, got {use_point!r}")

        def channel(name: str) -> np.ndarray:
            # Aligned datasets store both the raw stream (receive-time
            # timeline) and its time-aligned `_interp` resample under the
            # same base name; prefer `_interp` so passing a whole episode
            # group picks the per-valid-frame timeline.
            for key in (f"{name}_interp", name):
                if key in arrays:
                    return np.asarray(arrays[key], dtype=np.float64).reshape(-1)
            raise KeyError(
                f"missing tactile array '{name}' (or '{name}_interp') — "
                f"available: {sorted(arrays)}"
            )

        out: dict[str, np.ndarray] = {}
        num_frames = None
        for finger, spec in self._sensors.items():
            points = [
                np.stack(
                    [
                        channel(f"tactile_{spec.sensor_id}_f{axis}_{pt}")
                        for axis in "xyz"
                    ],
                    axis=-1,
                )
                for pt in (1, 2)
            ]
            if use_point == "first":
                raw = points[0]
            elif use_point == "second":
                raw = points[1]
            else:
                raw = 0.5 * (points[0] + points[1])
            if num_frames is None:
                num_frames = raw.shape[0]
            elif raw.shape[0] != num_frames:
                raise ValueError(
                    f"tactile arrays for sensor {spec.sensor_id} have "
                    f"{raw.shape[0]} frames, expected {num_frames}"
                )
            calibrated = self._calibrate(spec, raw)
            calibrated[~np.all(np.isfinite(raw), axis=-1)] = 0.0
            out[finger] = calibrated
        return out

    @staticmethod
    def _force_in_fingertip_frame(spec: TactileSensorSpec, reading, components: str):
        """Calibrated sensor-frame reading ->
        ``(f_fingertip (3,), m_fingertip (3,)|None)``.

        Embeds a 1-DOF reading on the spec's normal axis, applies the
        mounting rotation ``R_fingertip_sensor``, and (for 6-DOF wrenches)
        transports the moment to the fingertip origin:
        ``m_fingertip = R m + t x (R f)``. With a non-'all' ``components``
        everything except the selected sensor-frame force axes ('normal' =
        the spec's normal_axis) is zeroed FIRST, including any measured
        moment.
        """
        reading = np.asarray(reading, dtype=np.float64)
        moment_sensor = None
        if spec.measurement_dim == 1:
            flat = reading.reshape(-1)
            if flat.shape != (1,):
                raise ValueError(
                    f"expected a scalar/(1,) reading for sensor {spec.sensor_id}, "
                    f"got shape {reading.shape}"
                )
            force_sensor = np.zeros(3)
            force_sensor[spec.normal_axis_index] = float(flat[0])
        elif spec.measurement_dim == 3:
            if reading.shape != (3,):
                raise ValueError(
                    f"expected a (3,) force for sensor {spec.sensor_id}, "
                    f"got shape {reading.shape}"
                )
            force_sensor = reading
        else:
            if reading.shape != (6,):
                raise ValueError(
                    f"expected a (6,) wrench for sensor {spec.sensor_id}, "
                    f"got shape {reading.shape}"
                )
            force_sensor = reading[:3]
            moment_sensor = reading[3:]
        if components != "all":
            projected = np.zeros(3)
            for token in components:
                idx = (
                    spec.normal_axis_index if token == "normal" else _AXIS_INDEX[token]
                )
                projected[idx] = force_sensor[idx]
            force_sensor = projected
            moment_sensor = None if moment_sensor is None else np.zeros(3)
        R_fingertip_sensor = spec.R_fingertip_sensor
        force_fingertip = R_fingertip_sensor @ force_sensor
        if moment_sensor is None:
            return force_fingertip, None
        moment_fingertip = R_fingertip_sensor @ moment_sensor + np.cross(
            spec.t_fingertip_sensor_m, force_fingertip
        )
        return force_fingertip, moment_fingertip

    def forces_in_hand_base(
        self,
        forces_in_sensor_frame_n: dict[str, Optional[np.ndarray]],
        q_actuated_rad,
        *,
        components=None,
    ) -> dict[str, Optional[np.ndarray]]:
        """Map calibrated sensor-frame readings into the hand-base frame.

        Args:
            forces_in_sensor_frame_n: ``{finger: reading}`` with each
                reading shaped per the finger's spec — scalar/(1,) for
                1-DOF, ``(3,)`` force, or ``(6,)`` wrench — in N / N*m
                (e.g. from ``forces_in_sensor_frame``). None entries
                (sensor dropout) pass through as None.
            q_actuated_rad: ``(num_joints,)`` actuated joints in radians.
                Sample once per control tick and reuse for every
                model-based call so all quantities refer to the same tick.
            components: Sensor-frame force-component selection — 'all'
                (default, ``None`` means the same), a single token
                'x' | 'y' | 'z' | 'normal', or an iterable of tokens
                (e.g. ``("y", "z")``; 'normal' = the spec's normal_axis).
                Anything but 'all' keeps exactly the selected force axes
                and zeroes any measured moment. A per-call policy flag,
                not part of the sensor spec.

        Returns:
            ``{finger: (3,) force}`` (or ``(6,)`` wrench for 6-DOF specs)
            in the hand-base frame — the force the fingertip APPLIES to the
            environment; moments are about the fingertip frame origin.
        """
        components = resolve_force_components(components)
        q_actuated_rad = np.asarray(q_actuated_rad, dtype=np.float64)
        kinematics = self._kinematics
        out: dict[str, Optional[np.ndarray]] = {}
        with self._lock:
            q_full_rad = kinematics.actuated_to_full_configuration(q_actuated_rad)
            for finger, reading in forces_in_sensor_frame_n.items():
                # Validate the key even for None readings — otherwise a
                # typo'd finger name only raises on ticks where its sensor
                # happens to deliver, making the error nondeterministic.
                if finger not in self._sensors:
                    raise KeyError(
                        f"finger '{finger}' has no tactile sensor configured "
                        f"(sensors: {sorted(self._sensors)})"
                    )
                if reading is None:
                    out[finger] = None
                    continue
                force_fingertip, moment_fingertip = self._force_in_fingertip_frame(
                    self._sensors[finger], reading, components
                )
                R_base_fingertip, _ = kinematics.fingertip_pose(q_full_rad, finger)
                force_base = R_base_fingertip @ force_fingertip
                if moment_fingertip is None:
                    out[finger] = force_base
                else:
                    out[finger] = np.concatenate(
                        [force_base, R_base_fingertip @ moment_fingertip]
                    )
        return out

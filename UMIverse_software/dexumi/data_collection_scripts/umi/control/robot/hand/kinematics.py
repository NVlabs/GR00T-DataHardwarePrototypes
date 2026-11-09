# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic hand kinematics: actuated-joint vector <-> full URDF configuration,
fingertip poses / Jacobians, and the quasi-static map from fingertip forces to
actuated-joint torques.

Terminology follows the underactuated-hands literature:

- ``q_actuated_rad`` — the ``(n_actuated,)`` joint vector in HARDWARE order
  (what the hand is commanded with and reports).
- ``q_full_rad`` — the ``(nq,)`` URDF configuration in pinocchio order,
  including passive (coupled) joints.
- ``B`` — the actuation/coupling matrix with ``q_dot_full = B @ q_dot_actuated``.
  For a fully-actuated hand B is a pure selection matrix; each passive joint
  adds a row carrying its coupling derivative. By the principle of virtual
  work, fingertip forces map to actuated-joint torques as
  ``tau_actuated = B^T J^T f``.

Sign convention: ``f`` is the force the fingertip APPLIES to the environment,
expressed in the hand-base (URDF root) frame; a positive pad-normal reading
means pressing.

All pinocchio lookups are BY NAME (joint / frame names from the hand
description), never by index: pinocchio's BFS model order interleaves the
fingers differently from both the URDF file order and the hardware joint
order.
"""

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pinocchio as pin

from umi.common.utility.print_utils import print_yellow


class JointCoupling:
    """A passive URDF joint kinematically driven by one actuated joint.

    The pluggable seam for underactuation: subclasses define the coupling
    map ``q_coupled = value(q_driving)`` and its ``derivative``, which is
    what enters the actuation matrix B.
    """

    def __init__(self, coupled_joint_name: str, driving_joint_name: str):
        self.coupled_joint_name = coupled_joint_name
        self.driving_joint_name = driving_joint_name

    def value(self, q_driving_rad):
        raise NotImplementedError

    def derivative(self, q_driving_rad):
        raise NotImplementedError


class PolynomialJointCoupling(JointCoupling):
    """Polynomial coupling ``q_coupled = polyval(coefficients, q_driving)``.

    Coefficients are highest-degree-first, matching ``np.polyval``. The
    X Hand's dip joints use cubics from the vendor's underactuation table.
    """

    def __init__(
        self,
        coupled_joint_name: str,
        driving_joint_name: str,
        coefficients,
    ):
        super().__init__(coupled_joint_name, driving_joint_name)
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self._derivative_coefficients = np.polyder(self.coefficients)

    def value(self, q_driving_rad):
        """Coupled joint angle = polyval(coefficients, q_driving_rad).

        Works on scalars or arrays.
        """
        return np.polyval(self.coefficients, q_driving_rad)

    def derivative(self, q_driving_rad):
        """d(coupled)/d(driving) — the coupling term that enters the B matrix."""
        return np.polyval(self._derivative_coefficients, q_driving_rad)


@dataclass
class ResolvedJointCoupling:
    """A JointCoupling bound to a specific pinocchio model.

    Where ``JointCoupling`` is the pure coupling law from the hand
    description (joint names + value/derivative), this carries the
    model-resolved binding: the coupled joint's slot in the full
    configuration vector, its driving joint's slot in the actuated vector,
    and the coupled joint's URDF limits — and owns the clip semantics used
    by both ``actuated_to_full_configuration`` and ``coupling_matrix``.
    """

    coupling: JointCoupling
    coupled_joint_configuration_index: int
    driving_joint_actuated_index: int
    lower_limit_rad: float
    upper_limit_rad: float

    def clipped_value(self, q_driving_rad: float) -> float:
        """Coupled joint angle, clipped into its URDF range as a guard."""
        return float(
            np.clip(
                self.coupling.value(q_driving_rad),
                self.lower_limit_rad,
                self.upper_limit_rad,
            )
        )

    def derivative(self, q_driving_rad: float) -> float:
        """Coupling derivative for the B matrix — zero when the coupling
        output is clipped at a URDF limit (the passive joint can't follow
        there)."""
        if (
            self.lower_limit_rad
            <= self.coupling.value(q_driving_rad)
            <= self.upper_limit_rad
        ):
            return float(self.coupling.derivative(q_driving_rad))
        return 0.0


@dataclass
class FingerDescription:
    """One finger's wiring: actuated joints (hardware order) and tip frame.

    ``actuated_joint_indices[i]`` is the index into the actuated joint vector
    of the URDF joint ``actuated_joint_names[i]``. ``coupling`` describes an
    optional passive joint on the chain (None for fully-actuated fingers).
    """

    name: str
    actuated_joint_indices: list[int]
    actuated_joint_names: list[str]
    fingertip_frame: str
    coupling: Optional[JointCoupling] = None


@dataclass
class AuxiliaryJointDescription:
    """An actuated joint that belongs to no finger chain (e.g. a palm joint).

    ``transmit_fingertip_torque`` False zeroes this joint's row of
    ``B^T J^T f``, so its torque is 0 even when a fingertip chain passes
    through it.
    """

    actuated_joint_index: int
    urdf_joint_name: str
    transmit_fingertip_torque: bool = False


@dataclass
class HandKinematicsDescription:
    """Everything needed to build a HandKinematics from a URDF.

    ``fingers`` insertion order is the hand's canonical finger order.
    """

    urdf_path: Path
    fingers: dict[str, FingerDescription]
    auxiliary_joints: list[AuxiliaryJointDescription] = field(default_factory=list)


class HandKinematics:
    """Pinocchio wrapper around a hand URDF with optional underactuation.

    NOT thread-safe: forward-kinematics calls mutate the shared
    ``pin.Data``. Use ``clone()`` for a second thread, or go through
    ``HandBase``, which serializes access with an internal lock.
    """

    def __init__(self, description: HandKinematicsDescription):
        """Args:
        description: URDF path + finger/auxiliary wiring.
        """
        self.description = description
        self.model = pin.buildModelFromUrdf(str(description.urdf_path))
        self.data = self.model.createData()
        self.nq = self.model.nq
        if self.model.nq != self.model.nv:
            raise ValueError(
                f"expected a fixed-base all-revolute model (nq == nv), got "
                f"nq={self.model.nq} nv={self.model.nv}"
            )

        def joint_idx_q(name):
            if not self.model.existJointName(name):
                raise ValueError(f"URDF has no joint named '{name}'")
            return self.model.joints[self.model.getJointId(name)].idx_q

        # Actuated joints: hardware index -> idx_q in the pinocchio
        # configuration vector. The indices must form exactly 0..n-1.
        index_to_urdf_name = {}
        for finger in description.fingers.values():
            if len(finger.actuated_joint_indices) != len(finger.actuated_joint_names):
                raise ValueError(
                    f"finger '{finger.name}': actuated_joint_indices and "
                    "actuated_joint_names must have the same length"
                )
            for idx, name in zip(
                finger.actuated_joint_indices, finger.actuated_joint_names
            ):
                index_to_urdf_name[int(idx)] = name
        for aux in description.auxiliary_joints:
            index_to_urdf_name[int(aux.actuated_joint_index)] = aux.urdf_joint_name

        self.n_actuated = len(index_to_urdf_name)
        if sorted(index_to_urdf_name) != list(range(self.n_actuated)):
            raise ValueError(
                "actuated joint indices must cover exactly 0..n-1 with no "
                f"duplicates; got {sorted(index_to_urdf_name)}"
            )
        self._actuated_idx_q = np.array(
            [joint_idx_q(index_to_urdf_name[i]) for i in range(self.n_actuated)],
            dtype=int,
        )
        if len(set(self._actuated_idx_q.tolist())) != self.n_actuated:
            raise ValueError("two actuated joints map to the same URDF joint")
        urdf_name_to_index = {v: k for k, v in index_to_urdf_name.items()}

        # Passive (coupled) joints, bound to this model's indices/limits.
        self._couplings: list[ResolvedJointCoupling] = []
        actuated_slots = set(self._actuated_idx_q.tolist())
        for finger in description.fingers.values():
            coupling = finger.coupling
            if coupling is None:
                continue
            coupled_idx_q = joint_idx_q(coupling.coupled_joint_name)
            if coupled_idx_q in actuated_slots:
                raise ValueError(
                    f"coupled joint '{coupling.coupled_joint_name}' is also "
                    "an actuated joint"
                )
            if coupling.driving_joint_name not in urdf_name_to_index:
                raise ValueError(
                    f"coupling for '{coupling.coupled_joint_name}' is driven "
                    f"by '{coupling.driving_joint_name}', which is not an "
                    "actuated joint"
                )
            self._couplings.append(
                ResolvedJointCoupling(
                    coupling=coupling,
                    coupled_joint_configuration_index=coupled_idx_q,
                    driving_joint_actuated_index=urdf_name_to_index[
                        coupling.driving_joint_name
                    ],
                    lower_limit_rad=self.model.lowerPositionLimit[coupled_idx_q],
                    upper_limit_rad=self.model.upperPositionLimit[coupled_idx_q],
                )
            )

        covered = actuated_slots | {
            resolved.coupled_joint_configuration_index for resolved in self._couplings
        }
        uncovered = sorted(set(range(self.nq)) - covered)
        if uncovered:
            names = [
                self.model.names[jid]
                for jid in range(1, self.model.njoints)
                if self.model.joints[jid].idx_q in uncovered
            ]
            print_yellow(
                "Warning: URDF joints neither actuated nor coupled "
                f"(held at 0 in FK): {names}"
            )

        self._fingertip_frame_id = {}
        for name, finger in description.fingers.items():
            if not self.model.existFrame(finger.fingertip_frame):
                raise ValueError(f"URDF has no frame named '{finger.fingertip_frame}'")
            self._fingertip_frame_id[name] = self.model.getFrameId(
                finger.fingertip_frame
            )

    @classmethod
    def from_config(
        cls, config: dict, config_dir: Optional[Path] = None
    ) -> "HandKinematics":
        """Build a HandKinematics from a hand config dict's ``kinematics:``
        block. Every hand requires one — a config without it raises.

        Keys mirror the description dataclass fields exactly. Per-finger:
        ``actuated_joint_indices``, ``actuated_joint_names`` (same order),
        ``fingertip_frame``, and an optional ``coupling: {coupled_joint_name,
        driving_joint_name, coefficients}``. ``auxiliary_joints`` lists
        actuated joints outside any finger chain (``{actuated_joint_index,
        urdf_joint_name, transmit_fingertip_torque}``). A relative
        ``urdf_path`` is resolved against ``config_dir``, then the repo root.
        """
        kinematics_config = config.get("kinematics")
        if not kinematics_config:
            raise ValueError(
                "hand config has no `kinematics:` block — a URDF-backed "
                "kinematics description is required for every hand"
            )
        fingers = {}
        for name, fc in kinematics_config["fingers"].items():
            coupling = None
            if fc.get("coupling") is not None:
                cc = fc["coupling"]
                coupling = PolynomialJointCoupling(
                    coupled_joint_name=cc["coupled_joint_name"],
                    driving_joint_name=cc["driving_joint_name"],
                    coefficients=cc["coefficients"],
                )
            fingers[name] = FingerDescription(
                name=name,
                actuated_joint_indices=[int(i) for i in fc["actuated_joint_indices"]],
                actuated_joint_names=list(fc["actuated_joint_names"]),
                fingertip_frame=fc["fingertip_frame"],
                coupling=coupling,
            )
        auxiliary_joints = [
            AuxiliaryJointDescription(
                actuated_joint_index=int(a["actuated_joint_index"]),
                urdf_joint_name=a["urdf_joint_name"],
                transmit_fingertip_torque=bool(
                    a.get("transmit_fingertip_torque", False)
                ),
            )
            for a in kinematics_config.get("auxiliary_joints", [])
        ]
        return cls(
            HandKinematicsDescription(
                urdf_path=_resolve_urdf_path(
                    kinematics_config["urdf_path"], config_dir
                ),
                fingers=fingers,
                auxiliary_joints=auxiliary_joints,
            )
        )

    @property
    def finger_names(self) -> tuple[str, ...]:
        return tuple(self.description.fingers.keys())

    def clone(self) -> "HandKinematics":
        """A lock-free handle for another thread: shares the (read-only)
        model and description, gets its own ``pin.Data`` scratch space."""
        other = copy.copy(self)
        other.data = self.model.createData()
        return other

    # -- configuration mapping ------------------------------------------------

    def actuated_to_full_configuration(self, q_actuated_rad) -> np.ndarray:
        """``(n_actuated,)`` hardware joints -> ``(nq,)`` URDF configuration.

        Coupled joints are filled from their coupling maps and clipped to
        their URDF limits.
        """
        q_actuated_rad = np.asarray(q_actuated_rad, dtype=np.float64)
        q_full_rad = np.zeros(self.nq)
        q_full_rad[self._actuated_idx_q] = q_actuated_rad
        for resolved in self._couplings:
            q_full_rad[resolved.coupled_joint_configuration_index] = (
                resolved.clipped_value(
                    q_actuated_rad[resolved.driving_joint_actuated_index]
                )
            )
        return q_full_rad

    def coupling_matrix(self, q_actuated_rad) -> np.ndarray:
        """Actuation matrix B ``(nq, n_actuated)`` with
        ``q_dot_full = B @ q_dot_actuated``.

        Identity rows for actuated joints (a pure selection matrix for a
        fully-actuated hand); each coupled joint's row carries its coupling
        derivative at the current driving angle in the driving joint's
        column. When the coupling output is outside the coupled joint's URDF
        limits (i.e. ``actuated_to_full_configuration`` clipped it) the
        derivative is zeroed — the passive joint can't follow there.
        """
        q_actuated_rad = np.asarray(q_actuated_rad, dtype=np.float64)
        B = np.zeros((self.nq, self.n_actuated))
        B[self._actuated_idx_q, np.arange(self.n_actuated)] = 1.0
        for resolved in self._couplings:
            B[
                resolved.coupled_joint_configuration_index,
                resolved.driving_joint_actuated_index,
            ] = resolved.derivative(
                q_actuated_rad[resolved.driving_joint_actuated_index]
            )
        return B

    # -- forward kinematics / Jacobians ---------------------------------------

    def _fingertip_frame_id_for(self, finger: str) -> int:
        try:
            return self._fingertip_frame_id[finger]
        except KeyError:
            raise ValueError(
                f"unknown finger '{finger}' (fingers: {list(self.finger_names)})"
            ) from None

    def fingertip_pose(self, q_full_rad, finger):
        """``(R_base_fingertip (3, 3), p_base_fingertip_m (3,))`` — the
        finger's tip frame pose in the hand-base frame (T_A_B convention:
        ``v_base = R_base_fingertip @ v_fingertip``).

        Copies are returned: ``data.oMf`` entries are overwritten in place by
        the next FK call, so views into them would silently mutate.
        """
        frame_id = self._fingertip_frame_id_for(finger)
        pin.framesForwardKinematics(self.model, self.data, np.asarray(q_full_rad))
        M = self.data.oMf[frame_id]
        return np.array(M.rotation), np.array(M.translation)

    def fingertip_pose_jacobian(
        self, q_full_rad, finger, *, include_rotation: bool = False
    ) -> np.ndarray:
        """Fingertip-frame Jacobian, hand-base-frame aligned
        (``LOCAL_WORLD_ALIGNED``).

        Returns the ``(3, nv)`` positional rows by default; with
        ``include_rotation`` the full ``(6, nv)`` frame Jacobian (rows 0-2
        linear, 3-5 angular — pinocchio convention), which is what a 6-DOF
        wrench measurement contracts against.
        """
        J = pin.computeFrameJacobian(
            self.model,
            self.data,
            np.asarray(q_full_rad),
            self._fingertip_frame_id_for(finger),
            pin.LOCAL_WORLD_ALIGNED,
        )
        return J if include_rotation else J[:3, :]

    # -- statics: fingertip forces -> joint torques ----------------------------

    def joint_torques_from_fingertip_forces(
        self, q_actuated_rad, fingertip_forces_in_base_n: dict[str, np.ndarray]
    ) -> np.ndarray:
        """``(n_actuated,)`` actuated-joint torques ``tau = B^T J^T f``.

        ``fingertip_forces_in_base_n`` maps finger name -> hand-base-frame
        ``(3,)`` force or ``(6,)`` wrench ``[f; m]`` in N / N*m. Auxiliary
        joints with ``transmit_fingertip_torque`` False have their row
        zeroed.
        """
        q_full_rad = self.actuated_to_full_configuration(q_actuated_rad)
        B = self.coupling_matrix(q_actuated_rad)
        tau_full = np.zeros(self.nq)
        for finger, f in fingertip_forces_in_base_n.items():
            f = np.asarray(f, dtype=np.float64)
            if not np.any(f):
                continue
            if f.shape == (3,):
                J = self.fingertip_pose_jacobian(q_full_rad, finger)
            elif f.shape == (6,):
                J = self.fingertip_pose_jacobian(
                    q_full_rad, finger, include_rotation=True
                )
            else:
                raise ValueError(
                    f"finger '{finger}': expected a (3,) force or (6,) wrench, "
                    f"got shape {f.shape}"
                )
            tau_full += J.T @ f
        tau_actuated = B.T @ tau_full
        for aux in self.description.auxiliary_joints:
            if not aux.transmit_fingertip_torque:
                tau_actuated[aux.actuated_joint_index] = 0.0
        return tau_actuated


# ---------------------------------------------------------------------------
# Config-loading helpers
# ---------------------------------------------------------------------------

# URDF folders renamed since hand configs were written: the vendor URDFs moved
# from the "Orange" codename folder to xhand/ (the X Hand), and a hand config
# written before the move still names the old folder.
_RENAMED_URDF_DIRS = {"Orange_URDF": "xhand"}


def _resolve_urdf_path(urdf: str, config_dir: Optional[Path]) -> Path:
    """Absolute path as-is; else try the config dir (the more specific,
    caller-supplied location), then the repo root. A path through a renamed
    folder (``_RENAMED_URDF_DIRS``) also resolves at the folder's new name."""
    path = Path(urdf)
    if path.is_absolute():
        return path
    candidates = [path]
    renamed = Path(*(_RENAMED_URDF_DIRS.get(part, part) for part in path.parts))
    if renamed != path:
        candidates.append(renamed)
    repo_root = Path(__file__).resolve().parents[4]
    for candidate in candidates:
        for base in (config_dir, repo_root):
            if base is not None and (base / candidate).exists():
                return base / candidate
    raise FileNotFoundError(
        f"URDF '{urdf}' not found under "
        + (f"{config_dir} or " if config_dir is not None else "")
        + str(repo_root)
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert aligned DexUMI glove data to robot-hand state and action.

Walks every episode in an aligned dataset and, for each DexUMI encoder source
listed in the dataset's ``config.yaml``, reads ``joint_angles_interp``, converts
it via ``device_cls.convert_to_robot_actions(...)`` with the glove calibration
bundled in the session, and writes the robot-hand state, the joint-position
action and (on tactile runs) the joint torques back into the aligned zarr in
one pass.

Output layout per encoder source (D = robot-hand DoF):

    episode_X/{source}_robot_state/
      state_interp              (N, D)   kinematic conversion          [always]
      joints_torque_interp      (N, D)   tau = B^T J^T f, N*m          [tactile]
    episode_X/{source}_robot_action/
      action_interp             (N, D)   raw positions (== state)      [always]
      action_torque_interp      (N, 2D)  [raw positions | torques]     [tactile]

``--tactile`` (the default) computes the force-derived arrays from the recorded
fingertip tactile data (``{side}_dexumi_exoskeleton_tactile``): the per-joint
torque (``compute_joint_torque``) and the joints-force ``(N, 2D)`` stack.
``--no-tactile`` writes only ``state_interp`` and ``action_interp``.

The tactile parameters (deadband, EMA, force components, tactile point,
fallback hand config) live in the ``robot_action`` block of
``config/dexumi/robot_action.yaml``; a CLI flag overrides its key for one run.

Each action array carries ``action_layout`` / ``action_position_dims`` attrs;
shared provenance (``torque_params``, ``tactile_source``, ``tactile``) lives on
the group. Both groups carry a ``write_state`` completion marker so readers can
detect interrupted writes (``umi.common.data.write_state``).

The force math uses a model of the robot hand each glove mirrors (the X Hand):
the ``hand_config`` named in the glove calibration bundled in the session (right
``xhand_right.yaml``, left ``xhand_left.yaml``). A calibration without that key
uses the default model for its side; an unprefixed legacy single-hand source
uses the run-level ``--hand-config`` fallback. A calibration recorded before the
hand configs were renamed names ``orange_hand_<side>.yaml``; it resolves to the
renamed file. Provenance (``torque_params.hand_config_path``) records which
model produced each action.

Only DexUMI encoder sources are converted. In a bimanual dataset, both encoder
sources must exist in every episode. All state/action arrays are computed
before any output is committed, so an absent or malformed side cannot leave a
plausible one-handed result behind.

Usage:
    ./launch.sh convert --aligned-dir <session>_aligned --sequential
    ./launch.sh convert --aligned-dir <session>_aligned --sequential --no-tactile
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click
import numpy as np
import ray
import yaml
import zarr

from umi.common.data import write_state
from umi.umi_device.dexumi_exoskeleton import DexUmiExoskeleton
from umi.umi_device.mock_dexumi_exoskeleton import MockDexUmiExoskeleton
from umi.umi_device.umi_device_interface import UmiDeviceInterface
from umi.utils.ray_utils import ray_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Map recorded source types to their robot-action converters.
DEVICE_CLASS_BY_TYPE: dict[str, type[UmiDeviceInterface]] = {
    "dexumi_exoskeleton": DexUmiExoskeleton,
    # Mock recordings use the same conversion as hardware recordings.
    "mock_dexumi_exoskeleton": MockDexUmiExoskeleton,
}

_BIMANUAL_EMBODIMENTS = {"bimanual_dexumi"}
_SIDES = ("left", "right")

# data_collection_scripts/, put on the Ray workers' PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[2]
# The tactile parameters live in one place; every caller reads this block and a
# CLI flag overrides its key for one run.
DEFAULT_CONFIG = REPO_ROOT / "config" / "dexumi" / "robot_action.yaml"
ROBOT_ACTION_KEYS = (
    "deadband_n",
    "ema_alpha",
    "components",
    "tactile_use_point",
    "hand_config",
)
TACTILE_USE_POINTS = ("first", "second", "mean")
_DEFAULT_HAND_CONFIG = REPO_ROOT / "umi/control/robot/configs/xhand_right.yaml"
# Hand model for a glove calibration without a hand_config key.
_SIDE_HAND_CONFIGS = {
    side: f"umi/control/robot/configs/xhand_{side}.yaml" for side in _SIDES
}
# Hand configs renamed since glove calibrations were recorded: the "Orange"
# codename files became xhand_<side>.yaml (the X Hand). A calibration recorded
# before the rename still names the old file, which resolves to the new one.
_RENAMED_HAND_CONFIGS = {
    f"orange_hand_{side}.yaml": f"umi/control/robot/configs/xhand_{side}.yaml"
    for side in _SIDES
}


def load_robot_action_params(config_path=None) -> dict:
    """Return the validated ``robot_action`` block of the parameter yaml.

    Raises ValueError naming the problem so every caller fails the same way.
    """
    path = Path(config_path or DEFAULT_CONFIG).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        document = yaml.safe_load(path.read_text()) or {}
    except OSError as exc:
        raise ValueError(f"robot-action config unavailable: {path} ({exc})") from exc
    block = document.get("robot_action") if isinstance(document, dict) else None
    if not isinstance(block, dict):
        raise ValueError(
            f"{path} has no robot_action block; the converter's parameters live "
            "there (see config/dexumi/robot_action.yaml)"
        )
    unknown = sorted(set(block) - set(ROBOT_ACTION_KEYS))
    missing = [key for key in ROBOT_ACTION_KEYS if key not in block]
    if unknown or missing:
        raise ValueError(
            f"{path} robot_action: "
            + (f"unknown keys {unknown}" if unknown else "")
            + ("; " if unknown and missing else "")
            + (f"missing keys {missing}" if missing else "")
        )
    return validate_robot_action_params(dict(block), source=str(path))


def validate_robot_action_params(params: dict, source: str = "robot_action") -> dict:
    def number(key, minimum=None, maximum=None, exclusive_min=False):
        value = params.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{source}: {key} must be a number, got {value!r}")
        value = float(value)
        if minimum is not None and (
            value <= minimum if exclusive_min else value < minimum
        ):
            raise ValueError(
                f"{source}: {key} must be {'>' if exclusive_min else '>='} {minimum}"
            )
        if maximum is not None and value > maximum:
            raise ValueError(f"{source}: {key} must be <= {maximum}")
        params[key] = value

    number("deadband_n", 0.0)
    number("ema_alpha", 0.0, 1.0, exclusive_min=True)
    components = params.get("components")
    if components is not None and (
        not isinstance(components, str) or not components.strip()
    ):
        raise ValueError(
            f"{source}: components must be a token list such as 'z' or 'all'"
        )
    if params.get("tactile_use_point") not in TACTILE_USE_POINTS:
        raise ValueError(
            f"{source}: tactile_use_point must be one of {', '.join(TACTILE_USE_POINTS)}"
        )
    hand_config = params.get("hand_config")
    if hand_config is not None and (
        not isinstance(hand_config, str) or not hand_config
    ):
        raise ValueError(f"{source}: hand_config must be a path or null")
    return params


def resolve_robot_action_params(config_path=None, **overrides) -> dict:
    """Return the effective parameters: the yaml block with the given non-None
    CLI overrides applied and the fallback hand config resolved."""
    params = load_robot_action_params(config_path)
    for key, value in overrides.items():
        if key not in ROBOT_ACTION_KEYS:
            raise ValueError(f"unknown robot_action override {key!r}")
        if value is not None:
            params[key] = value
    params = validate_robot_action_params(params, source="robot_action overrides")
    hand_config = params.get("hand_config") or str(_DEFAULT_HAND_CONFIG)
    hand_path = Path(hand_config).expanduser()
    if not hand_path.is_absolute():
        hand_path = REPO_ROOT / hand_path
    params["hand_config"] = str(hand_path)
    return params


@dataclass(frozen=True)
class ConversionPlan:
    """Source, device class and calibration used to convert each episode."""

    source_name: str  # zarr subgroup to read the signal from
    source_type: str
    device_class_name: str
    action_source_key: str  # array within source_name to read
    umi_config_path: Optional[str]
    # Mirrored-hand config for the force math (the calibration's hand_config,
    # or the default for the source's side). None only for an unprefixed
    # legacy source, which may use the run-level fallback.
    hand_config_path: Optional[str] = None
    # Geometry gates combined from the bundled glove and hand-model YAMLs.
    # False means placeholder/unverified; None means neither declares it.
    geometry_verified: Optional[bool] = None
    geometry_simulation_only: bool = False
    # Mock sources and sessions marked meta.simulation_only may use placeholder
    # geometry; production sources may not.
    simulation_run: bool = False


@dataclass(frozen=True)
class _HandGeometry:
    """Resolved mirrored-hand geometry for one source."""

    hand_config_path: str
    geometry_verified: Optional[bool]
    simulation_only: bool


@dataclass(frozen=True)
class _PendingConversion:
    """One source's fully computed output, ready for transactional commit."""

    plan: ConversionPlan
    state: np.ndarray
    action: np.ndarray
    action_attrs: dict
    state_attrs: dict
    # Tactile-derived variants (None on a --no-tactile run).
    torque: Optional[np.ndarray] = None
    action_torque: Optional[np.ndarray] = None


@dataclass(frozen=True)
class _ArraySnapshot:
    data: np.ndarray
    chunks: tuple[int, ...]
    attrs: dict


@dataclass(frozen=True)
class _GroupSnapshot:
    arrays: dict[str, _ArraySnapshot]
    attrs: dict


@dataclass(frozen=True)
class ComplianceRun:
    """Tactile-run settings, picklable so they travel to Ray workers.

    The hand model itself is rebuilt per device inside the worker.
    ``hand_config_path`` is the run-level FALLBACK hand config (--hand-config)
    for an unprefixed legacy source. The torque knobs mirror
    ``DexUmiExoskeleton.compute_joint_torque`` kwargs.
    """

    hand_config_path: str
    ema_alpha: float
    deadband_n: float
    # 'all' or a token tuple (see resolve_force_components).
    components: str | tuple
    tactile_use_point: Optional[str]

    def effective_hand_config(
        self,
        plan: ConversionPlan,
        *,
        allow_legacy_fallback: bool = False,
    ) -> str:
        """Resolve a source's hand model without leaking the right-hand
        fallback into a left or bimanual source."""
        if plan.hand_config_path:
            return plan.hand_config_path
        if allow_legacy_fallback:
            return self.hand_config_path
        raise RuntimeError(
            f"source {plan.source_name!r} has no per-source hand_config; "
            "the global --hand-config fallback is restricted to legacy "
            "single-right-hand data"
        )

    def torque_kwargs(self) -> dict:
        """``compute_joint_torque`` kwargs: the tactile filtering knobs."""
        return {
            "ema_alpha": self.ema_alpha,
            "deadband_n": self.deadband_n,
            "components": self.components,
            "tactile_use_point": self.tactile_use_point,
        }


def _resolve_components_option(components_spec: Optional[str]):
    """Map ``--components`` (comma-separated tokens or 'all', default 'z' =
    pad-normal only) onto the torque ``components``."""
    from umi.control.robot.hand.tactile import resolve_force_components

    if components_spec is None:
        components_spec = "z"
    tokens = [t.strip() for t in components_spec.split(",") if t.strip()]
    return resolve_force_components("all" if tokens == ["all"] else tokens)


def _tactile_group_name(source_name: str) -> str:
    """Sibling tactile group of an encoder source (shared device prefix):
    ``right_dexumi_exoskeleton_encoder`` -> ``right_dexumi_exoskeleton_tactile``."""
    return source_name.removesuffix("_encoder") + "_tactile"


def _source_side(source_name: str) -> Optional[str]:
    """Return the canonical side encoded in a source name, if present."""
    return next(
        (side for side in _SIDES if source_name.startswith(f"{side}_")),
        None,
    )


def _bimanual_plan_map(
    plans: list[ConversionPlan],
) -> Optional[dict[str, ConversionPlan]]:
    """Return a side→plan map only for an exact left/right plan pair."""
    by_side: dict[str, ConversionPlan] = {}
    for plan in plans:
        side = _source_side(plan.source_name)
        if side is None or side in by_side:
            return None
        by_side[side] = plan
    if len(plans) == 2 and set(by_side) == set(_SIDES):
        return by_side
    return None


def _combined_geometry_status(*configs: dict) -> tuple[Optional[bool], bool]:
    """Combine production/simulation gates from glove and hand-model YAMLs."""
    verified_values = [
        config["geometry_verified"]
        for config in configs
        if "geometry_verified" in config
    ]
    if any(value is False for value in verified_values):
        geometry_verified: Optional[bool] = False
    elif any(value is True for value in verified_values):
        geometry_verified = True
    else:
        geometry_verified = None
    simulation_only = any(
        bool(config.get("simulation_only", False)) for config in configs
    )
    return geometry_verified, simulation_only


def _resolve_referenced_config(
    aligned_dir: Path,
    declared_path: str,
    *,
    source_name: str,
    description: str,
) -> Path:
    """Resolve a config bundled in the session first, then its repository path,
    then the current file of a renamed one (``_RENAMED_HAND_CONFIGS``)."""
    bundled = aligned_dir / Path(declared_path).name
    declared = Path(os.path.expanduser(declared_path))
    repo_path = declared if declared.is_absolute() else REPO_ROOT / declared
    if bundled.is_file():
        return bundled
    if repo_path.is_file():
        return repo_path
    renamed = _RENAMED_HAND_CONFIGS.get(Path(declared_path).name)
    if renamed is not None and (REPO_ROOT / renamed).is_file():
        logger.info(
            "Source %s: %s %s was renamed; using %s",
            source_name,
            description,
            Path(declared_path).name,
            renamed,
        )
        return REPO_ROOT / renamed
    raise click.ClickException(
        f"{description} {Path(declared_path).name!r} declared for source "
        f"{source_name!r} is neither bundled in the session at {aligned_dir} "
        f"nor present at {repo_path}."
    )


def build_conversion_plans(aligned_dir: Path) -> list[ConversionPlan]:
    """Build plans only for DexUMI encoder sources in aligned ``config.yaml``.

    Expanded tactile sources, grippers, and all unrelated numeric devices are
    deliberately ignored. If the aligned config declares a bimanual DexUMI
    embodiment, it must yield exactly one left and one right encoder plan.

    Read each device calibration from the session copy beside ``config.yaml``.
    A referenced calibration must exist there so conversion uses the settings
    captured at recording time. The mirrored-hand config it names is read from
    the session when bundled there, otherwise from the repository.
    """
    config_path = aligned_dir / "config.yaml"
    if not config_path.exists():
        raise click.ClickException(f"Aligned config.yaml not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise click.ClickException(
            f"Aligned config.yaml must contain a mapping: {config_path}"
        )

    meta = config.get("meta") or {}
    embodiment = meta.get("embodiment") if isinstance(meta, dict) else None
    explicit_simulation = bool(
        isinstance(meta, dict) and meta.get("simulation_only", False)
    )

    plans: list[ConversionPlan] = []
    seen_sources: set[str] = set()
    for src in config.get("numeric_sources", []) or []:
        if not isinstance(src, dict):
            continue
        src_type = src.get("type")
        src_name = src.get("name")
        if not src_type or not src_name:
            continue

        device_cls = DEVICE_CLASS_BY_TYPE.get(src_type)
        if device_cls is None:
            logger.info(
                "Skipping source %s: type %r not in registry", src_name, src_type
            )
            continue

        if not src_name.endswith("_encoder"):
            logger.info(
                "Skipping source %s: robot-action conversion consumes only "
                "DexUMI *_encoder sources",
                src_name,
            )
            continue
        if src_name in seen_sources:
            raise click.ClickException(
                f"Duplicate DexUMI encoder source {src_name!r} in {config_path}"
            )
        seen_sources.add(src_name)

        action_source_key = device_cls.action_source_key
        if not action_source_key:
            logger.info(
                "Skipping source %s: %s has no action_source_key",
                src_name,
                device_cls.__name__,
            )
            continue

        params = src.get("params") or {}
        if not isinstance(params, dict):
            raise click.ClickException(f"Source {src_name!r} params must be a mapping")
        umi_config_path: Optional[str] = None
        device_config: dict = {}
        cfg_file = params.get("config_file")
        if cfg_file:
            bundled = aligned_dir / Path(cfg_file).name
            if not bundled.is_file():
                raise click.ClickException(
                    f"Device config {bundled.name!r} for source {src_name!r} is "
                    f"not bundled in the session at {aligned_dir}. The recording "
                    "session bundles each device's calibration alongside the "
                    "data; re-run alignment so it travels with the aligned dataset."
                )
            umi_config_path = str(bundled)
            with open(bundled) as f:
                device_config = yaml.safe_load(f) or {}
            if not isinstance(device_config, dict):
                raise click.ClickException(
                    f"Bundled device config {bundled} must contain a mapping"
                )

        # The mirrored-hand config: params.hand_config_file, else the glove
        # calibration's hand_config, else the default model for the source's
        # side. Declared but resolvable nowhere is a provenance gap.
        hand_config_path: Optional[str] = None
        hand_config: dict = {}
        hand_cfg_file = params.get("hand_config_file") or device_config.get(
            "hand_config"
        )
        if not hand_cfg_file and _source_side(src_name) is not None:
            hand_cfg_file = _SIDE_HAND_CONFIGS[_source_side(src_name)]
            logger.info(
                "Source %s: calibration names no hand_config; using %s",
                src_name,
                hand_cfg_file,
            )
        if hand_cfg_file:
            resolved_hand = _resolve_referenced_config(
                aligned_dir,
                str(hand_cfg_file),
                source_name=src_name,
                description="Hand config",
            )
            hand_config_path = str(resolved_hand)
            with open(resolved_hand) as f:
                hand_config = yaml.safe_load(f) or {}
            if not isinstance(hand_config, dict):
                raise click.ClickException(
                    f"Hand config {resolved_hand} must contain a mapping"
                )

        geometry_verified, geometry_simulation_only = _combined_geometry_status(
            device_config,
            hand_config,
        )

        plans.append(
            ConversionPlan(
                source_name=src_name,
                source_type=src_type,
                device_class_name=device_cls.__name__,
                action_source_key=action_source_key,
                umi_config_path=umi_config_path,
                hand_config_path=hand_config_path,
                geometry_verified=geometry_verified,
                geometry_simulation_only=geometry_simulation_only,
                simulation_run=(
                    explicit_simulation or src_type == "mock_dexumi_exoskeleton"
                ),
            )
        )

    if embodiment in _BIMANUAL_EMBODIMENTS and _bimanual_plan_map(plans) is None:
        found = sorted(plan.source_name for plan in plans)
        raise click.ClickException(
            f"{embodiment} requires exactly one left and one right DexUMI "
            f"encoder source; found {found}"
        )

    return plans


def list_episode_indices(aligned_dir: Path) -> list[int]:
    """Return sorted episode indices in the aligned zarr."""
    root = zarr.open(str(aligned_dir), mode="r")
    indices: list[int] = []
    for key in root.group_keys():
        if not key.startswith("episode_"):
            continue
        try:
            indices.append(int(key.split("_", 1)[1]))
        except ValueError:
            continue
    return sorted(indices)


def _write_result_group(
    ep,
    group_name: str,
    arrays: dict[str, np.ndarray],
    plan: ConversionPlan,
    extra_attrs: dict,
    array_attrs: Optional[dict[str, dict]] = None,
) -> None:
    """Replace a result group and write its arrays and provenance.

    Recreating the group removes stale arrays (e.g. a tactile run's
    ``joints_torque_interp`` that a --no-tactile re-run no longer produces),
    attributes and files from interrupted writes. The caller snapshots
    existing data for rollback and marks all output groups as incomplete
    before replacement begins.

    Materialize zero-filled chunks so readers can verify chunk counts (a
    no-contact stretch of ``joints_torque_interp`` is all zeros). Mark the
    group complete with its final provenance after every array is written.
    """
    out_group = ep.create_group(
        group_name,
        overwrite=True,
        attributes={write_state.WRITE_STATE: write_state.WRITING},
    )
    for array_name, data in arrays.items():
        t = data.shape[0]
        inner_dims = data.shape[1:]
        chunk_t = min(t, 256) if t > 0 else 1
        arr = out_group.create_array(
            name=array_name,
            shape=data.shape,
            chunks=(chunk_t, *inner_dims) if inner_dims else (chunk_t,),
            dtype="float64",
            overwrite=True,
            config={"write_empty_chunks": True},
        )
        arr[:] = data
        for key, value in ((array_attrs or {}).get(array_name) or {}).items():
            arr.attrs[key] = value

    # Publish provenance and completion in a single metadata update.
    out_group.update_attributes(
        {
            "source_name": plan.source_name,
            "source_type": plan.source_type,
            "device_class": plan.device_class_name,
            "action_source_key": plan.action_source_key,
            **extra_attrs,
            write_state.WRITE_STATE: write_state.COMPLETE,
        }
    )


def _resolve_hand_geometry(
    plan: ConversionPlan,
    plans: list[ConversionPlan],
    compliance: ComplianceRun,
) -> _HandGeometry:
    """Resolve and production-gate the mirrored hand for one conversion plan."""
    bimanual = _bimanual_plan_map(plans) is not None
    side = _source_side(plan.source_name)
    # A single unqualified ``dexumi_exoskeleton_encoder`` source is the legacy
    # right hand; only an explicitly left-prefixed source is known not to be.
    legacy_single_right = len(plans) == 1 and not bimanual and side != "left"
    hand_config_path = compliance.effective_hand_config(
        plan,
        allow_legacy_fallback=legacy_single_right,
    )

    geometry_verified = plan.geometry_verified
    simulation_only = plan.geometry_simulation_only
    if plan.hand_config_path is None:
        fallback_path = Path(hand_config_path)
        if not fallback_path.is_file():
            raise RuntimeError(
                f"legacy fallback hand config not found: {fallback_path}"
            )
        with open(fallback_path) as f:
            fallback_config = yaml.safe_load(f) or {}
        if not isinstance(fallback_config, dict):
            raise RuntimeError(
                f"legacy fallback hand config must contain a mapping: {fallback_path}"
            )
        fallback_verified, fallback_simulation_only = _combined_geometry_status(
            fallback_config
        )
        if geometry_verified is None:
            geometry_verified = fallback_verified
        simulation_only = simulation_only or fallback_simulation_only

    if not plan.simulation_run:
        if simulation_only or geometry_verified is False:
            raise RuntimeError(
                f"source {plan.source_name!r} uses simulation-only or unverified "
                f"hand geometry {Path(hand_config_path).name!r}; use a mock "
                "source / meta.simulation_only for simulation, or replace the "
                "placeholder and set geometry_verified: true"
            )
        if bimanual and geometry_verified is not True:
            raise RuntimeError(
                f"source {plan.source_name!r} has no explicitly verified "
                "per-source hand geometry; production bimanual "
                "conversion requires geometry_verified: true"
            )

    return _HandGeometry(
        hand_config_path=hand_config_path,
        geometry_verified=geometry_verified,
        simulation_only=simulation_only,
    )


def _snapshot_group(ep, group_name: str) -> Optional[_GroupSnapshot]:
    """Capture an output group so a failed multi-group commit can roll back."""
    if group_name not in ep:
        return None
    group = ep[group_name]
    arrays = {
        name: _ArraySnapshot(
            data=np.asarray(group[name][:]),
            chunks=tuple(group[name].chunks),
            attrs=dict(group[name].attrs),
        )
        for name in group.array_keys()
    }
    return _GroupSnapshot(arrays=arrays, attrs=dict(group.attrs))


def _restore_group(
    ep,
    group_name: str,
    snapshot: Optional[_GroupSnapshot],
) -> None:
    """Restore a snapshot, keeping the group incomplete until restoration ends."""
    write_state.mark_writing(ep, group_name)
    if snapshot is None:
        if group_name in ep:
            del ep[group_name]
        return
    group = ep.create_group(
        group_name,
        overwrite=True,
        attributes={write_state.WRITE_STATE: write_state.WRITING},
    )
    for name, array_snapshot in snapshot.arrays.items():
        data = array_snapshot.data
        array = group.create_array(
            name=name,
            shape=data.shape,
            chunks=array_snapshot.chunks,
            dtype=data.dtype,
            # Preserve zero-filled chunks so chunk-count validation succeeds.
            config={"write_empty_chunks": True},
        )
        array[:] = data
        for key, value in array_snapshot.attrs.items():
            array.attrs[key] = value
    # Preserve the snapshot's marker; unmarked older groups count as complete.
    group.update_attributes(
        {write_state.WRITE_STATE: write_state.COMPLETE, **snapshot.attrs}
    )


def _settle_zarr_io() -> None:
    """Wait for Zarr's pending writes before restoring a failed conversion.

    A failed chunk write can leave sibling writes running. They must finish
    before rollback to avoid overwriting restored data.
    """
    from zarr.core.sync import sync

    async def settle():
        me = asyncio.current_task()
        others = [task for task in asyncio.all_tasks() if task is not me]
        await asyncio.gather(*others, return_exceptions=True)

    sync(settle())


def _commit_pending_conversions(ep, pending: list[_PendingConversion]) -> None:
    """Write all staged groups, rolling back the entire set on any failure."""
    group_names = [
        group_name
        for converted in pending
        for group_name in (
            f"{converted.plan.source_name}_robot_state",
            f"{converted.plan.source_name}_robot_action",
        )
    ]
    snapshots = {
        group_name: _snapshot_group(ep, group_name) for group_name in group_names
    }
    try:
        # Mark every group first so interrupted commits cannot appear complete
        # while mixing outputs from different conversions.
        for group_name in group_names:
            write_state.mark_writing(ep, group_name)
        for converted in pending:
            plan = converted.plan
            dims = int(converted.state.shape[1])
            positions_attrs = {
                "action_layout": ["joint_positions"],
                "action_position_dims": dims,
            }
            state_arrays = {"state_interp": converted.state}
            state_array_attrs: dict[str, dict] = {
                "state_interp": {"content": "kinematic conversion"},
            }
            action_arrays = {"action_interp": converted.action}
            action_array_attrs: dict[str, dict] = {
                "action_interp": {
                    **positions_attrs,
                    "content": "raw positions (== state)",
                },
            }
            if converted.torque is not None:
                state_arrays["joints_torque_interp"] = converted.torque
                state_array_attrs["joints_torque_interp"] = {
                    "content": (
                        "per-joint torque tau = B^T J^T f from the recorded "
                        "fingertip forces, N*m"
                    ),
                }
            if converted.action_torque is not None:
                action_arrays["action_torque_interp"] = converted.action_torque
                action_array_attrs["action_torque_interp"] = {
                    "action_layout": ["joint_positions", "joint_torques"],
                    "action_position_dims": dims,
                    "content": "[raw positions | torques] (joints-force stack)",
                }
            _write_result_group(
                ep,
                f"{plan.source_name}_robot_state",
                state_arrays,
                plan,
                converted.state_attrs,
                array_attrs=state_array_attrs,
            )
            _write_result_group(
                ep,
                f"{plan.source_name}_robot_action",
                action_arrays,
                plan,
                converted.action_attrs,
                array_attrs=action_array_attrs,
            )
    # Interrupts leave incomplete markers: rollback could race Zarr's IO thread.
    except Exception:
        _settle_zarr_io()
        for group_name, snapshot in snapshots.items():
            _restore_group(ep, group_name, snapshot)
        raise


def _hand_model_inputs(ep, plan: ConversionPlan, geometry: _HandGeometry):
    """Build the exoskeleton's hand model and read its recorded fingertip forces.

    Missing tactile data fails the episode: a silent kinematic fallback would
    mislabel the outputs as force-derived.
    """
    if plan.umi_config_path is None:
        raise RuntimeError(
            f"source {plan.source_name!r} has no bundled device config — "
            "cannot build the exoskeleton for the mirrored-hand model"
        )
    tactile_group = _tactile_group_name(plan.source_name)
    if tactile_group not in ep:
        raise RuntimeError(
            f"tactile group {tactile_group!r} not found — the force-derived "
            "outputs need the recorded fingertip forces (use --no-tactile for "
            "a session without tactile recordings)"
        )
    src_grp = ep[tactile_group]
    tactile_arrays = {
        name: np.asarray(src_grp[name][:])
        for name in src_grp.array_keys()
        if name.startswith("tactile_") and name.endswith("_interp")
    }
    device_cls = DEVICE_CLASS_BY_TYPE[plan.source_type]
    device = device_cls.from_config(
        plan.umi_config_path,
        reader=None,
        hand_config=geometry.hand_config_path,
    )
    return device, tactile_arrays


def _compute_torque(
    device,
    tactile_arrays: dict,
    plan: ConversionPlan,
    state: np.ndarray,
    compliance: ComplianceRun,
) -> np.ndarray:
    """Per-joint torque ``tau = B^T J^T f`` for one exoskeleton source."""
    torques = device.compute_joint_torque(
        state, tactile_arrays, **compliance.torque_kwargs()
    )
    logger.info(
        "%s: joint torque — max |tau| %.4f N*m, %d/%d contact frames",
        plan.source_name,
        float(np.abs(torques).max(initial=0.0)),
        int((np.abs(torques) > 1e-9).any(axis=1).sum()),
        len(torques),
    )
    return torques


def _convert_episode(
    aligned_dir: str,
    episode_idx: int,
    plans: list[ConversionPlan],
    compliance: Optional[ComplianceRun] = None,
) -> tuple[int, bool, Optional[str]]:
    """Validate and compute every planned source before committing any output.

    Always writes ``episode_X/{source_name}_robot_state/state_interp``
    (kinematic conversion) and ``episode_X/{source_name}_robot_action/
    action_interp`` (raw positions, == state), overwriting any existing
    entries. A tactile run (``compliance`` set) also writes
    ``joints_torque_interp`` next to the state and the ``action_torque_interp``
    variant next to the action; ``compliance=None`` (--no-tactile) keeps the
    two kinematic arrays only.
    A bimanual episode is all-or-nothing: both encoder inputs and every output
    array must succeed.
    Concurrent tasks for different episodes touch disjoint paths so no
    inter-task locking is needed.
    """
    try:
        root = zarr.open(aligned_dir, mode="a")
        episode_key = f"episode_{episode_idx}"
        if episode_key not in root:
            return episode_idx, False, f"{episode_key} not found"
        ep = root[episode_key]

        if not plans:
            raise RuntimeError("no DexUMI encoder conversion plans")
        bimanual_plans = _bimanual_plan_map(plans)
        if len(plans) > 1 and bimanual_plans is None:
            raise RuntimeError(
                "multiple DexUMI encoder plans must be an exact left/right pair; "
                f"got {sorted(plan.source_name for plan in plans)}"
            )

        missing_inputs: list[str] = []
        for plan in plans:
            if plan.source_name not in ep:
                missing_inputs.append(plan.source_name)
                continue
            src_grp = ep[plan.source_name]
            if plan.action_source_key not in src_grp:
                missing_inputs.append(f"{plan.source_name}/{plan.action_source_key}")
        if missing_inputs:
            prefix = (
                "bimanual episode requires both DexUMI encoder sources"
                if bimanual_plans is not None
                else "DexUMI encoder input is incomplete"
            )
            raise RuntimeError(f"{prefix}; missing {sorted(missing_inputs)}")

        pending: list[_PendingConversion] = []
        for plan in plans:
            src_grp = ep[plan.source_name]
            data = np.asarray(src_grp[plan.action_source_key][:])
            device_cls = DEVICE_CLASS_BY_TYPE[plan.source_type]
            # The device converter applies configured joint limits.
            state = np.asarray(
                device_cls.convert_to_robot_actions(
                    data, umi_config=plan.umi_config_path
                )
            )
            if state.dtype != np.float64:
                state = state.astype(np.float64)
            if state.ndim != 2 or not np.isfinite(state).all():
                raise RuntimeError(
                    f"{plan.source_name!r} conversion must produce a finite "
                    f"2-D state array, got shape={state.shape}"
                )

            tactile = compliance is not None
            state_attrs = {"tactile": tactile}
            action_attrs = {"tactile": tactile}
            # The raw-positions action is the state by definition; the other
            # variants live in their own arrays next to it.
            action = state
            torque = None
            action_torque = None
            if tactile:
                geometry = _resolve_hand_geometry(plan, plans, compliance)
                device, tactile_arrays = _hand_model_inputs(ep, plan, geometry)
                torque = np.asarray(
                    _compute_torque(device, tactile_arrays, plan, state, compliance),
                    dtype=np.float64,
                )
                if torque.shape != state.shape or not np.isfinite(torque).all():
                    raise RuntimeError(
                        f"{plan.source_name!r} joint torque must be finite with "
                        f"shape {state.shape}, got {torque.shape}"
                    )
                # Positions first, torques second: the layout the array attrs
                # advertise and downstream converters slice on.
                action_torque = np.concatenate([state, torque], axis=1)
                torque_params = {
                    **compliance.torque_kwargs(),
                    "hand_config_path": Path(geometry.hand_config_path).name,
                }
                if geometry.geometry_verified is not None:
                    torque_params["hand_geometry_verified"] = geometry.geometry_verified
                action_attrs.update(
                    {
                        "torque_source": (
                            f"{plan.source_name}_robot_state/joints_torque_interp"
                        ),
                        "torque_params": torque_params,
                        "tactile_source": _tactile_group_name(plan.source_name),
                    }
                )
            pending.append(
                _PendingConversion(
                    plan=plan,
                    state=state,
                    action=action,
                    action_attrs=action_attrs,
                    state_attrs=state_attrs,
                    torque=torque,
                    action_torque=action_torque,
                )
            )

        if bimanual_plans is not None:
            frame_counts = {converted.state.shape[0] for converted in pending}
            if len(frame_counts) != 1:
                raise RuntimeError(
                    "bimanual DexUMI encoder outputs have mismatched frame "
                    f"counts: {sorted(frame_counts)}"
                )

        _commit_pending_conversions(ep, pending)
        for converted in pending:
            variants = ["action_interp"]
            if converted.action_torque is not None:
                variants.append("action_torque_interp")
            logger.info(
                "%s/%s_robot_{state,action} state=%s actions=%s tactile=%s",
                episode_key,
                converted.plan.source_name,
                tuple(converted.state.shape),
                variants,
                converted.torque is not None,
            )

        return episode_idx, True, None
    except Exception as e:
        return episode_idx, False, str(e)


def build_compliance_run(plans: list[ConversionPlan], params: dict) -> ComplianceRun:
    """Build the tactile-run settings and fail fast on an unusable hand model.

    Resolves and loads each distinct per-source hand model once (URDF load,
    tactile spec validation) before any episode is touched.
    """
    compliance_run = ComplianceRun(
        hand_config_path=str(Path(params["hand_config"]).resolve()),
        ema_alpha=params["ema_alpha"],
        deadband_n=params["deadband_n"],
        components=_resolve_components_option(params["components"]),
        tactile_use_point=params["tactile_use_point"],
    )
    probed: set[str] = set()
    for exo_plan in plans:
        try:
            geometry = _resolve_hand_geometry(exo_plan, plans, compliance_run)
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from exc
        effective = geometry.hand_config_path
        if effective in probed:
            continue
        probed.add(effective)
        probe = DEVICE_CLASS_BY_TYPE[exo_plan.source_type].from_config(
            exo_plan.umi_config_path,
            reader=None,
            hand_config=effective,
        )
        logger.info(
            "Mirrored hand model: %s (%d joints, verified=%s, simulation_only=%s)",
            Path(effective).name,
            probe.hand_kinematics.n_actuated,
            geometry.geometry_verified,
            geometry.simulation_only,
        )
    logger.info("Tactile run: torque_params=%s", compliance_run.torque_kwargs())
    return compliance_run


def convert_all_episodes(
    aligned_path: Path,
    *,
    tactile: bool = True,
    config_path=None,
) -> list[tuple[int, bool, Optional[str]]]:
    """Convert every episode of an aligned session in this process (no Ray).

    Uses the CLI defaults: tactile unless ``tactile=False``, with the
    parameters of ``config_path`` (default config/dexumi/robot_action.yaml).
    Returns ``(episode index, ok, error)`` per episode; the LeRobot conversion
    runs this first (real_scripts/data_processing/lerobot_conversion/).
    """
    plans = build_conversion_plans(aligned_path)
    if not plans:
        raise click.ClickException(
            f"No processable device sources found in {aligned_path / 'config.yaml'}"
        )
    compliance_run: Optional[ComplianceRun] = None
    if tactile:
        try:
            params = resolve_robot_action_params(config_path)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        compliance_run = build_compliance_run(plans, params)
    return [
        _convert_episode(str(aligned_path), idx, plans, compliance_run)
        for idx in list_episode_indices(aligned_path)
    ]


@ray.remote
def _convert_episode_remote(
    aligned_dir: str,
    episode_idx: int,
    plans: list[ConversionPlan],
    compliance: Optional[ComplianceRun] = None,
) -> tuple[int, bool, Optional[str]]:
    return _convert_episode(aligned_dir, episode_idx, plans, compliance)


@ray.remote
def _workers_have_pinocchio() -> bool:
    """Probe that Ray workers can run the hand-model math before dispatch."""
    import importlib.util

    return importlib.util.find_spec("pinocchio") is not None


@click.command()
@click.option(
    "--aligned-dir",
    "-d",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Path to the aligned dataset directory (zarr root).",
)
@click.option(
    "--episode",
    "-e",
    type=int,
    multiple=True,
    default=None,
    help="Restrict to these episode indices (repeatable). Default: all.",
)
@click.option(
    "--num-workers",
    "-n",
    type=int,
    default=-1,
    help="Number of Ray workers. -1 uses all available CPUs.",
)
@click.option(
    "--sequential",
    "-s",
    is_flag=True,
    help="Disable Ray and use a sequential for-loop (easier for debugging).",
)
@click.option(
    "--tactile/--no-tactile",
    default=True,
    show_default=True,
    help="Compute the force-derived outputs from the recorded fingertip "
    "tactile data: joints_torque_interp (tau = B^T J^T f, N*m) next to the "
    "state, plus the action_torque_interp ([positions | torques]) variant. "
    "--no-tactile writes only the kinematic state_interp + action_interp.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    default=str(DEFAULT_CONFIG),
    show_default=True,
    help="YAML whose robot_action block supplies every tactile parameter "
    "below; each flag overrides its key for this run.",
)
@click.option(
    "--hand-config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="FALLBACK config yaml of the robot hand an unprefixed legacy "
    "single-hand source mirrors. Glove calibrations name their own "
    "hand_config. Default: robot_action.hand_config in --config.",
)
@click.option(
    "--ema-alpha",
    type=float,
    default=None,
    help="EMA factor on the calibrated tactile forces (1.0 = no smoothing). "
    "Default: robot_action.ema_alpha in --config.",
)
@click.option(
    "--deadband-n",
    type=float,
    default=None,
    help="Force-norm deadband in Newtons (below -> treated as no contact). "
    "Default: robot_action.deadband_n in --config.",
)
@click.option(
    "--components",
    "components_spec",
    type=str,
    default=None,
    help="Sensor-frame force components entering the torque math: "
    "comma-separated tokens from x, y, z, normal (e.g. 'y,z'), or 'all' "
    "(full vector incl. 6-D moments). Default: robot_action.components in "
    "--config (z = pad-normal only).",
)
@click.option(
    "--tactile-use-point",
    type=click.Choice(list(TACTILE_USE_POINTS)),
    default=None,
    help="Which recorded tactile measurement point to use. Default: "
    "robot_action.tactile_use_point in --config.",
)
def main(
    aligned_dir: str,
    episode: tuple,
    num_workers: int,
    sequential: bool,
    tactile: bool,
    config_path: str,
    hand_config: Optional[str],
    ema_alpha: Optional[float],
    deadband_n: Optional[float],
    components_spec: Optional[str],
    tactile_use_point: Optional[str],
):
    """Convert the aligned glove signals to robot-hand state and the actions
    variant in place."""
    aligned_path = Path(os.path.expanduser(aligned_dir)).resolve()
    plans = build_conversion_plans(aligned_path)
    if not plans:
        raise click.ClickException(
            f"No processable device sources found in {aligned_path / 'config.yaml'}"
        )
    for plan in plans:
        logger.info(
            "Plan: %s/%s -> %s_robot_{state,action} (%s, %s) umi_config=%s "
            "hand_config=%s",
            plan.source_name,
            plan.action_source_key,
            plan.source_name,
            plan.source_type,
            plan.device_class_name,
            plan.umi_config_path,
            plan.hand_config_path or "(legacy single-right fallback only)",
        )

    compliance_run: Optional[ComplianceRun] = None
    if tactile:
        try:
            params = resolve_robot_action_params(
                config_path,
                hand_config=hand_config,
                ema_alpha=ema_alpha,
                deadband_n=deadband_n,
                components=components_spec,
                tactile_use_point=tactile_use_point,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        logger.info(
            "Robot-action parameters (from %s, CLI flags override): %s",
            config_path,
            params,
        )
        compliance_run = build_compliance_run(plans, params)

    all_episodes = list_episode_indices(aligned_path)
    if episode:
        keep = set(int(i) for i in episode)
        episode_indices = [i for i in all_episodes if i in keep]
        missing = sorted(keep - set(all_episodes))
        if missing:
            logger.warning("Requested episodes not in dataset: %s", missing)
    else:
        episode_indices = all_episodes

    if not episode_indices:
        logger.warning("No episodes to process.")
        return
    logger.info("Episodes to process: %s", episode_indices)

    aligned_abs = str(aligned_path)
    results: list[tuple[int, bool, Optional[str]]] = []

    if sequential:
        for idx in episode_indices:
            results.append(_convert_episode(aligned_abs, idx, plans, compliance_run))
    else:
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        pythonpath = str(REPO_ROOT) + (
            f":{existing_pythonpath}" if existing_pythonpath else ""
        )
        extra_env = {"env_vars": {"PYTHONPATH": pythonpath}}
        with ray_session(num_workers, extra_env=extra_env):
            if compliance_run is not None and not ray.get(
                _workers_have_pinocchio.remote()
            ):
                raise click.ClickException(
                    "Ray workers cannot import pinocchio (the `pin` package of "
                    "the DexUMI `ik` dependency group). Run `uv sync` at the "
                    "repository root, or re-run with --sequential (-s)."
                )
            futures = [
                _convert_episode_remote.remote(aligned_abs, idx, plans, compliance_run)
                for idx in episode_indices
            ]
            results = ray.get(futures)

    success = sum(1 for _, ok, _ in results if ok)
    failed = [(idx, err) for idx, ok, err in results if not ok]
    for idx, ok, err in sorted(results, key=lambda r: r[0]):
        if ok:
            logger.info("Episode %d: OK", idx)
        else:
            logger.error("Episode %d: FAILED - %s", idx, err)
    logger.info("Done. Success: %d/%d episodes", success, len(episode_indices))
    if failed:
        logger.warning("Failed: %s", [idx for idx, _ in failed])
        details = "; ".join(f"episode_{idx}: {err}" for idx, err in failed)
        raise click.ClickException(
            f"Robot-action conversion failed for {len(failed)}/"
            f"{len(episode_indices)} episode(s): {details}"
        )


if __name__ == "__main__":
    main()

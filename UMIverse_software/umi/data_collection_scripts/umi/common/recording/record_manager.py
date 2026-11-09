# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

import yaml
import zarr

from umi.common.recording.base_recorder import PENDING_SUFFIX
from umi.common.recording.episode_utils import (
    QC_PASS_STATUS_KEY,
    effective_episode_count_for_qc,
    qc_pass_status_from_attrs,
)
from umi.common.recording.session_slot import (
    config_yaml_bytes,
    data_check_yaml_bytes,
    device_config_entries,
)
from umi.common.recording.session_uid import (
    SESSION_UID_LEN,
    is_valid_session_uid,
)
from umi.common.utility.code_provenance import (
    code_provenance,
    head_commit,
    repo_root,
)
from umi.common.utility.output_layout import counterpart
from umi.common.utility.print_utils import print_green, print_red, print_yellow
from umi.common.utility.reference_camera import video_source_names
from umi.utils.device_manager import (
    get_iphone_udid,
    get_oak_device_id,
    get_umi_gripper_serial,
)


def _local_timestamp() -> str:
    """Current local wall-clock time as ISO-8601 with the UTC offset attached."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Recorder(Protocol):
    """Protocol defining required recorder interface"""

    @property
    def episode_id(self) -> int: ...
    @property
    def is_recording(self) -> bool: ...
    def start_streaming(self) -> bool: ...
    def stop_streaming(self) -> bool: ...
    def start_recording(self, episode_id: int = None) -> bool: ...
    def stop_recording(self) -> bool: ...
    def reset_episode_recording(self) -> bool: ...
    def save_recordings(self) -> None: ...
    def peek_latest_frame(self, idx: int): ...
    def set_episode_id(self, episode_id: int) -> None: ...
    def clear_recording(self) -> bool: ...
    def get_last_k_frames_from_all(self, k: int, remove: bool = False) -> dict: ...
    def finalize_episode(self) -> bool: ...
    def discard_staging(self) -> bool: ...
    def get_pending_episode_path(self) -> str: ...
    def get_final_episode_path(self) -> str: ...


# Resolve each camera type's name key to its raw identity, accepting
# normalized registry aliases and raw serials.
_CAMERA_SERIAL_RESOLVERS = (
    ("oak_device_name", get_oak_device_id),
    ("iphone_device_name", get_iphone_udid),
)


class RecorderManager:
    def __init__(
        self,
        recorders: List[Recorder],
        record_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        data_check_config_path: Optional[str] = None,
        verbose: bool = False,
        background_aligner: Optional[Any] = None,
    ):
        self.recorders = recorders
        self.record_path = record_path
        self.config = config
        self.data_check_config_path = data_check_config_path
        self.verbose = verbose
        self._task_description: str = ""
        # Snapshot the displayed setup instruction and placement layout onto
        # episode attributes when recording starts.
        self._operator_instruction: str = ""
        self._placement_layout: Optional[dict] = None
        # Record task-registry source, version and lookup results per episode so
        # missing instructions can be traced to the session's registry state.
        self._registry_provenance: Optional[dict] = None
        # Local wall-clock time the current episode started recording, stamped by
        # start_recording() and written onto the episode's zarr attrs at save.
        self._episode_start_local: Optional[str] = None
        self.background_aligner = background_aligner
        # Snapshot the active camera-relative calibration onto each episode.
        self.last_calibration_data: Optional[List[dict]] = None

        self._validate_reference_camera()
        # Validate before best-effort metadata saving so failures cannot be swallowed.
        self._validate_session_uid()

        self._save_config()
        self._save_data_check_config()
        self._save_device_configs()
        self._save_meta_data()

    @property
    def task_description(self) -> str:
        """Get the current task description."""
        return self._task_description

    @task_description.setter
    def task_description(self, value: str) -> None:
        """Set the task description."""
        self._task_description = value
        if self.verbose:
            print(f"Task description set to: {value}")

    @property
    def operator_instruction(self) -> str:
        """The task's scene-setup instruction (from the task registry)."""
        return self._operator_instruction

    @operator_instruction.setter
    def operator_instruction(self, value: str) -> None:
        self._operator_instruction = value

    @property
    def registry_provenance(self) -> Optional[dict]:
        return self._registry_provenance

    @registry_provenance.setter
    def registry_provenance(self, value: Optional[dict]) -> None:
        self._registry_provenance = value

    @property
    def placement_layout(self) -> Optional[dict]:
        """The placement layout currently on screen (None = no grid)."""
        return self._placement_layout

    @placement_layout.setter
    def placement_layout(self, value: Optional[dict]) -> None:
        self._placement_layout = value

    def _save_config(self) -> None:
        """Save the config to the record folder.

        No content lock: the launch-time session-slot resolver already decided the
        folder by comparing ``launch_config.yaml`` (the raw config snapshot), so a
        reused folder is one the resolver deemed the same session. The enriched
        ``config.yaml`` is simply (re)written here for provenance; per-episode
        snapshots (``_save_episode_configs``) preserve the config each episode was
        actually recorded with."""
        if self.config is None or self.record_path is None:
            if self.verbose:
                print("No config or record_path set, skipping config save")
            return

        os.makedirs(self.record_path, exist_ok=True)
        config_path = os.path.join(self.record_path, "config.yaml")

        data = config_yaml_bytes(self.config)
        with open(config_path, "wb") as f:
            f.write(data)

        print(f"Config saved to {config_path}")

    def _save_data_check_config(self) -> None:
        """Save the data check config to the record folder."""
        if self.data_check_config_path is None or self.record_path is None:
            if self.verbose:
                print(
                    "No check_config_path or record_path set, skipping check config save"
                )
            return

        data_check_config_path = os.path.expanduser(self.data_check_config_path)
        if not os.path.exists(data_check_config_path):
            print_yellow(
                f"Warning: Check config file {data_check_config_path} not found"
            )
            return

        os.makedirs(self.record_path, exist_ok=True)
        dest_path = os.path.join(self.record_path, "data_check_config.yaml")

        # Global check config + optional per-task registry overrides, computed by
        # the shared helper so the bytes are consistent across a session.
        data = data_check_yaml_bytes(self.config, self.data_check_config_path)
        # Not write-time locked: the launch-time resolver already decided the
        # folder (comparing launch_config + data-check + device files). A reused
        # folder simply refreshes this provenance copy.
        if data is None:
            return
        with open(dest_path, "wb") as f:
            f.write(data)

        print(f"Data check config saved to {dest_path}")

    def _device_config_files(self) -> list[tuple[str, str]]:
        """Resolve unique config files referenced by any source in the config.

        Scans every source collection — i.e. each top-level list of source
        dicts (``video_sources``, ``numeric_sources``, and any future
        ``*_sources`` list) — for a ``params.config_file``. Today that is the
        UMI gripper calibration; it could equally be a camera or other device
        calibration, and this picks it up without needing to know the
        collection's name. Returns ``(abs_source_path, basename)`` deduped by
        source path, so the config that produced an episode is bundled
        alongside it (provenance + self-contained data once it leaves the
        repo). Sub-sources of one device share a config_file and collapse to
        one entry.
        """
        return device_config_entries(self.config)

    def _save_device_configs(self) -> None:
        """Copy each referenced device config file into the session folder.

        Mirrors ``_save_data_check_config``: the calibration files travel with
        the recording (session root + every episode, see ``_save_episode_configs``)
        so an episode / aligned session is self-describing once it leaves the
        repo.

        Not write-time locked here: the launch-time session-slot resolver
        (``session_file_conflicts``) already decided the folder. If a
        recalibration conflicts with an existing session that has episodes, the
        launcher rolls onto a fresh session first; only then is the session
        copy written or refreshed for the chosen folder.
        """
        if self.record_path is None:
            return
        files = self._device_config_files()
        if not files:
            return
        os.makedirs(self.record_path, exist_ok=True)
        for src_path, basename in files:
            if not os.path.exists(src_path):
                print_yellow(
                    f"Warning: device config file {src_path} not found; not bundled"
                )
                continue
            dst = os.path.join(self.record_path, basename)
            try:
                shutil.copy2(src_path, dst)
                print(f"Device config saved to {dst}")
            except Exception as e:
                print_yellow(f"Warning: Failed to save device config {basename}: {e}")

    def _save_episode_configs(self) -> None:
        """Snapshot the recording and data-check configs into the pending
        episode folder.

        This captures *which config was in effect when this episode was
        recorded*. Useful when the script is restarted mid-dataset with a
        modified config.yaml or data_check_config.yaml — without these
        per-episode snapshots there is no way to tell which episodes used
        which settings.

        Written files:
            episode_{id}.pending/config.yaml            (copied from session copy)
            episode_{id}.pending/data_check_config.yaml (copied from session copy)

        Every snapshot is copied from the **session-level** file (written at setup
        by ``_save_config`` / ``_save_data_check_config``) — never from in-memory
        state or the live source on disk — so each episode captures the config in
        effect when it was recorded, and a mid-session edit to the live inputs
        cannot leak into an already-recorded episode.
        """
        if not self.record_path:
            return

        episode_dir = os.path.join(
            self.record_path, f"episode_{self.episode_id}{PENDING_SUFFIX}"
        )
        if not os.path.exists(episode_dir):
            print_yellow(
                f"Warning: pending episode dir missing, skipping config snapshot: "
                f"{episode_dir}"
            )
            return

        # Snapshot the session config written during setup. A missing session copy
        # is a provenance error and must fail the save.
        if self.config is not None:
            session_config = os.path.join(self.record_path, "config.yaml")
            if not os.path.isfile(session_config):
                raise FileNotFoundError(
                    f"Session config {session_config!r} is missing; cannot "
                    f"snapshot it into {episode_dir!r}. _save_config must run "
                    "(at setup) before episodes are recorded."
                )
            episode_config = os.path.join(episode_dir, "config.yaml")
            shutil.copy2(session_config, episode_config)
            if self.verbose:
                print(f"Saved episode config snapshot to {episode_dir}/config.yaml")

        # data_check_config.yaml — copy the session copy only (never fall back
        # to the global source). If the session copy is absent (e.g. the global
        # data-check config was missing at setup), there is nothing to snapshot.
        if self.data_check_config_path:
            session_check = os.path.join(self.record_path, "data_check_config.yaml")
            if os.path.isfile(session_check):
                dst = os.path.join(episode_dir, "data_check_config.yaml")
                shutil.copy2(session_check, dst)
                if self.verbose:
                    print(f"Saved episode data-check config snapshot to {dst}")

        # Snapshot calibrations from the session copy, never mutable source files.
        # A missing session calibration is a provenance error and must fail the save.
        for _src_path, basename in self._device_config_files():
            session_copy = os.path.join(self.record_path, basename)
            if not os.path.isfile(session_copy):
                raise FileNotFoundError(
                    f"Session device config {session_copy!r} is missing; cannot "
                    f"snapshot it into {episode_dir!r}. _save_device_configs must "
                    "run (at setup) before episodes are recorded."
                )
            dst = os.path.join(episode_dir, basename)
            shutil.copy2(session_copy, dst)
            if self.verbose:
                print(f"Saved episode device config snapshot to {dst}")

    def _validate_reference_camera(self) -> None:
        """Fail loud if `reference_camera` doesn't match a recorded camera name.

        `reference_camera` is optional and only validated when set. The
        camera.relative_calibration pre-episode check is what consumes it, and
        it resolves cameras through `get_source`, which knows an oak_device's
        streams by name. So the names accepted here come from the shared
        `video_source_names`, which reads `streams[].name` too: a rig whose
        cameras are all oak_device streams has no source-level name at all, and
        a source-level check would reject every reference such a config could
        carry.
        """
        if not self.config:
            return
        ref = self.config.get("reference_camera")
        if not ref:
            return
        names = video_source_names(self.config)
        if ref not in names:
            raise ValueError(
                f"reference_camera={ref!r} does not match any video_sources[].name "
                f"(known: {sorted(names)})"
            )

    def _validate_session_uid(self) -> None:
        """Fail loud at startup on a missing or malformed ``meta.session_uid``.

        Without the uid, a session folder name is a pure function of date,
        gripper tokens and task, so two stations that agreed on those would
        produce the same ``raw/{task}/{session}/`` prefix and interleave their
        episodes. The uid is what makes the name unique, so losing it is a
        silent data-corruption bug, not a cosmetic one.

        Two rules, because two kinds of config reach this constructor:

        * A uid that is *present but malformed* is rejected everywhere. It is
          never intentional.
        * A *missing* uid is rejected only for configs the Station Console's
          assembler produced, identified by ``meta.setup_preset``, which
          ``ConfigAssembler._apply_meta`` writes next to ``session_uid``. Those
          are exactly the configs that are supposed to carry one. Hand-written
          configs and older sessions without session identity carry none;
          rejecting them would break them instead of protecting a recording.

        ``setup_preset`` is the discriminator on purpose: it is *intrinsic to
        the artifact* rather than to how the artifact was loaded, so the rule
        cannot be defeated by copying a generated config to a new path, passing
        it via a different flag, or loading it on another station. A
        config-path check or a caller-supplied "this is a recording" flag would
        be extrinsic and would drift away from the thing it is meant to
        describe.
        """
        meta = (self.config or {}).get("meta") or {}
        session_uid = meta.get("session_uid")
        if session_uid is None and "session_uid" not in meta:
            if "setup_preset" not in meta:
                return
            raise ValueError(
                "meta.session_uid is required: without it two stations can "
                "write to the same raw/{task}/{session}/ prefix."
            )
        # Use the shared session_uid definition so minting and validation accept
        # the same identifier shape.
        if not is_valid_session_uid(session_uid):
            raise ValueError(
                f"meta.session_uid must be {SESSION_UID_LEN} lowercase hex "
                f"characters, got {session_uid!r}."
            )

    def _save_meta_data(self) -> None:
        """Save the current meta data (commit ID, operator_name, task_name, HW_version, session_uid) to dataset-level zarr attributes."""
        if not self.record_path:
            return

        # Resolve code provenance relative to this checkout. Missing Git metadata
        # must not prevent recording or discard other session identity.
        root = repo_root()
        commit_id = head_commit(root)

        try:
            store = zarr.open(self.record_path, mode="a")
            if commit_id:
                store.attrs["commit_id"] = commit_id
                if self.verbose:
                    print(f"Saved commit ID to dataset-level zarr: {commit_id}")
            provenance = code_provenance(root)
            if provenance:
                store.attrs["code_provenance"] = provenance
                if self.verbose:
                    print(f"Saved code provenance: {provenance.get('version')}")
            meta = self.config.get("meta", {}) if self.config else {}
            operator_name = meta.get("operator_name", "")
            if operator_name:
                store.attrs["operator_name"] = operator_name
            task_name = meta.get("task_name", "")
            if task_name:
                store.attrs["task_name"] = task_name
            hw_version = meta.get("HW_version", "")
            if hw_version:
                store.attrs["HW_version"] = hw_version
            # Carry the console's per-slot pose-source metadata; expanded runtime
            # sources no longer preserve its original topology. Accept mappings only
            # and leave configs without this metadata unchanged.
            tracking_source = meta.get("tracking_source")
            if isinstance(tracking_source, dict) and tracking_source:
                store.attrs["tracking_source"] = dict(tracking_source)
            # Carry the validated session UID into dataset metadata so identity survives
            # folder renaming.
            session_uid = meta.get("session_uid", "")
            if session_uid:
                store.attrs["session_uid"] = session_uid
            if "discarded_reasons" not in store.attrs:
                store.attrs["discarded_reasons"] = {}
            self._sync_qc_pass_status(store, explicit_num_episodes=None)
        except Exception as e:
            print_yellow(f"Warning: Failed to save commit ID to zarr — {e}")

    def set_episode_id(self, episode_id: int = None) -> None:
        for recorder in self.recorders:
            recorder.set_episode_id(episode_id)

    @property
    def episode_id(self) -> int:
        episode_ids = {recorder.episode_id for recorder in self.recorders}
        assert len(episode_ids) == 1, "All recorders should have the same episode ID"
        return episode_ids.pop()

    @property
    def is_recording(self) -> bool:
        """Check if all recorders are currently recording.

        Returns:
            bool: True if all recorders are recording, False if any are not recording.
        """
        return all(recorder.is_recording for recorder in self.recorders)

    def start_streaming(self) -> bool:
        success = True
        for recorder in self.recorders:
            success &= recorder.start_streaming()
        return success

    def stop_streaming(self) -> bool:
        success = True
        for recorder in self.recorders:
            success &= recorder.stop_streaming()
        return success

    def reset_episode_recording(self) -> bool:
        success = True
        for recorder in self.recorders:
            success &= recorder.reset_episode_recording()
        return success

    def start_recording(self, episode_id: int = None) -> bool:
        # Clear the prior start timestamp before attempting a new episode.
        # A partial start may still be saved and must fall back to its own save time.
        self._episode_start_local = None
        success = True
        for recorder in self.recorders:
            success &= recorder.start_recording(episode_id=episode_id)

        if success:
            self._episode_start_local = _local_timestamp()
            if self.verbose:
                print(f"Started recording episode {self.episode_id}")
        return success

    def stop_recording(self) -> bool:
        success = True
        for recorder in self.recorders:
            success &= recorder.stop_recording()

        return success

    def clear_recording(self) -> bool:
        success = True
        for recorder in self.recorders:
            success &= recorder.clear_recording()
        return success

    def save_recordings(self, pending: bool = False) -> None:
        """Save all recorded data to the pending episode folder, then optionally finalize.

        Individual recorders always write into ``episode_N.pending/``.
        Metadata (task description, calibration, latency, etc.) is saved to
        the same pending group.

        If *pending* is ``True`` the episode stays in its staging folder so
        that an external validation step can run before calling
        ``finalize_episode()`` or ``discard_staging()``.

        If *pending* is ``False`` the episode is finalized immediately
        (renamed to ``episode_N/`` and the episode ID is incremented).

        Args:
            pending: If True, keep the episode in the staging folder only.
                     If False, finalize the episode after saving.
        """
        for recorder in self.recorders:
            recorder.save_recordings()

        self._save_episode_timestamp()
        self._save_task_description()
        self._save_scene_provenance()
        self._save_calibration_data()
        self._save_latency_data()
        self._save_gripper_serial_data()
        # Record gripper and associated camera together for per-episode provenance.
        self._save_camera_serial_data()
        self._save_episode_length_and_duration()
        self._save_episode_configs()

        if not pending:
            self.finalize_episode()

    def _save_episode_timestamp(self) -> None:
        """Stamp when the episode was recorded, in the station's local time.

        Two ISO-8601 attrs with UTC offset (e.g. ``2026-08-04T13:45:12-07:00``):
        ``recorded_at`` (recording start, from ``start_recording``) and
        ``recorded_end_at`` (now, i.e. when the episode was saved). Falls back to
        now for ``recorded_at`` if the manager never saw the start (e.g. a
        recorder-driven save). The start stamp is consumed here so a re-save
        without a fresh start can't inherit it.
        """
        if not self.record_path:
            return

        try:
            store = zarr.open(self.record_path, mode="a")
            episode_key = f"episode_{self.episode_id}{PENDING_SUFFIX}"
            if episode_key not in store:
                return

            now = _local_timestamp()
            started_at = self._episode_start_local or now
            store[episode_key].attrs["recorded_at"] = started_at
            store[episode_key].attrs["recorded_end_at"] = now
            self._episode_start_local = None
            if self.verbose:
                print(f"Saved recorded_at={started_at} to {episode_key}")
        except Exception as e:
            print(f"Warning: Failed to save episode timestamp: {e}")

    def _save_task_description(self) -> None:
        """Save task description to zarr attributes for the pending episode."""
        if not self.record_path or not self._task_description:
            return

        try:
            store = zarr.open(self.record_path, mode="a")
            episode_key = f"episode_{self.episode_id}{PENDING_SUFFIX}"
            if episode_key in store:
                store[episode_key].attrs["task_description"] = self._task_description
                if self.verbose:
                    print(f"Saved task description to {episode_key}")
            store.attrs["task_description"] = self._task_description
        except Exception as e:
            print(f"Warning: Failed to save task description: {e}")

    def _save_scene_provenance(self) -> None:
        """Save the setup instruction and placement layout for the pending
        episode. Absent for tasks without them — absence is meaningful.
        Written at w-press like every other episode attr, so finalized and
        quarantined episodes both carry the layout their scene was staged
        from ('g' is blocked during recording, so this is unambiguous)."""
        if not self.record_path:
            return
        if (
            not self._operator_instruction
            and self._placement_layout is None
            and self._registry_provenance is None
        ):
            return
        try:
            store = zarr.open(self.record_path, mode="a")
            episode_key = f"episode_{self.episode_id}{PENDING_SUFFIX}"
            if episode_key not in store:
                return
            if self._operator_instruction:
                store[episode_key].attrs["operator_instruction"] = (
                    self._operator_instruction
                )
            if self._placement_layout is not None:
                store[episode_key].attrs["placement_layout"] = self._placement_layout
            if self._registry_provenance is not None:
                store[episode_key].attrs["task_registry"] = self._registry_provenance
            if self.verbose:
                print(f"Saved scene provenance to {episode_key}")
        except Exception as e:
            print(f"Warning: Failed to save scene provenance: {e}")

    def _save_calibration_data(self) -> None:
        """Save calibration data of cameras and grippers for the current episode."""

        self._save_camera_calibration_data()

        self._save_gripper_calibration_data()

    def _save_camera_calibration_data(self) -> None:
        """Stamp the cached camera calibration onto the pending episode's zarr.

        Source of truth is `self.last_calibration_data`, populated by the
        `camera.relative_calibration` pre-episode check. If the check is not
        configured (or hasn't run yet), nothing is written.
        """
        if not self.record_path or self.last_calibration_data is None:
            return

        store = zarr.open(self.record_path, mode="a")
        episode_key = f"episode_{self.episode_id}{PENDING_SUFFIX}"
        if episode_key in store:
            store[episode_key].attrs["calibration"] = self.last_calibration_data
            if self.verbose:
                print(f"Saved calibration data to {episode_key}")

    def _save_gripper_calibration_data(self) -> None:
        """Save gripper calibration data to zarr for the pending episode."""
        if not self.record_path or not self.config:
            return

        gripper_calib_data = {}
        for source_cfg in self.config.get("numeric_sources", []):
            if source_cfg.get("type") != "umi_gripper":
                continue

            config_file = source_cfg.get("params", {}).get("config_file")
            if not config_file:
                continue

            config_path = os.path.expanduser(config_file)
            if not os.path.exists(config_path):
                continue

            with open(config_path, "r") as f:
                gripper_calib_data[source_cfg["name"]] = yaml.safe_load(f)

        store = zarr.open(self.record_path, mode="a")
        episode_key = f"episode_{self.episode_id}{PENDING_SUFFIX}"
        if episode_key in store:
            store[episode_key].attrs["gripper_calibration"] = gripper_calib_data

    def _save_latency_data(self) -> None:
        """Save sensor and robot latency data to zarr for the pending episode."""
        if not self.record_path or not self.config:
            return

        latency_data = {}

        video_sources = self.config.get("video_sources", [])
        for source_cfg in video_sources:
            source_name = source_cfg.get("name")
            source_latency = source_cfg.get("latency", 0.0)
            latency_data[source_name] = source_latency

        numeric_sources = self.config.get("numeric_sources", [])
        for source_cfg in numeric_sources:
            source_name = source_cfg.get("name")
            source_latency = source_cfg.get("latency", 0.0)
            latency_data[source_name] = source_latency

        try:
            store = zarr.open(self.record_path, mode="a")
            episode_key = f"episode_{self.episode_id}{PENDING_SUFFIX}"
            if episode_key in store:
                store[episode_key].attrs["latency"] = latency_data
                if self.verbose:
                    print(f"Saved latency data to {episode_key}")
        except Exception as e:
            print(f"Warning: Failed to save latency data: {e}")

    def _save_gripper_serial_data(self) -> None:
        """Save UMI gripper device serial numbers to zarr for the pending episode."""
        if not self.record_path or not self.config:
            return

        gripper_serial_data = {}

        numeric_sources = self.config.get("numeric_sources", [])
        for source_cfg in numeric_sources:
            if source_cfg.get("type") != "umi_gripper":
                continue

            source_name = source_cfg.get("name")
            uart_device_name = source_cfg.get("uart_device_name")

            if not uart_device_name:
                continue

            try:
                device_serial = get_umi_gripper_serial(uart_device_name)
                gripper_serial_data[source_name] = {
                    "device_name": uart_device_name,
                    "serial": device_serial,
                }
            except ValueError as e:
                print_yellow(
                    f"Warning: Could not get serial for {uart_device_name}: {e}"
                )
                continue

        if not gripper_serial_data:
            return

        try:
            store = zarr.open(self.record_path, mode="a")
            episode_key = f"episode_{self.episode_id}{PENDING_SUFFIX}"
            if episode_key in store:
                store[episode_key].attrs["gripper_serial"] = gripper_serial_data
                if self.verbose:
                    print(f"Saved gripper serial data to {episode_key}")
        except Exception as e:
            print(f"Warning: Failed to save gripper serial data: {e}")

    def _save_camera_serial_data(self) -> None:
        """Save which physical camera recorded this episode, into its attrs.

        `meta.hardware.<arm>.camera` already records this in the config, which
        answers the question when the session config is at hand. That is not
        the case during QC: episodes are quarantined and re-examined one at a
        time, often long after the fact, so each episode carries its own
        camera identity next to its gripper identity.

        The stored `device_name` is the raw spelling from the config, not a
        canonical identity, for the same reason `meta.hardware` keeps it:
        `canonical_device_identity` falls back to whatever spelling it was
        given when the registry cannot be loaded, so canonicalising here would
        freeze a value that depends on whether the registry file happened to be
        readable at record time. A permanent record must not have that property.
        The serial is the canonical identity, resolved through the registry's
        own name -> serial table, which is deterministic or absent.

        An unresolvable camera keeps its name and omits the serial, which
        differs from `_save_gripper_serial_data` above (that drops the source
        entirely). Deliberate: absence is ambiguous (no camera? unregistered?
        lookup failed?), while a name with no serial says exactly what is and
        is not known. Writing the name into the serial field is the one option
        ruled out, since nothing downstream could then tell an identity from a
        spelling.
        """
        if not self.record_path or not self.config:
            return

        camera_serial_data = {}

        for source_cfg in self.config.get("video_sources", []):
            params = source_cfg.get("params") or {}

            for param_key, resolve in _CAMERA_SERIAL_RESOLVERS:
                # Single iPhone sources store identity under params; multi-stream
                # oak_device sources store it at source level. Preserve each stream's
                # association with the shared physical board.
                device_name = params.get(param_key) or source_cfg.get(param_key)
                if not device_name:
                    continue

                entry = {"device_name": device_name}
                try:
                    entry["serial"] = str(resolve(device_name))
                except Exception as e:
                    # Keep an unresolvable camera name without inventing a serial.
                    print_yellow(f"Warning: no serial for camera {device_name}: {e}")

                streams = source_cfg.get("streams") or []
                names = [s.get("name") for s in streams if s.get("name")]
                if not names and source_cfg.get("name"):
                    names = [source_cfg["name"]]
                for name in names:
                    # Copy identity metadata per stream so later changes cannot mutate
                    # sibling records.
                    camera_serial_data[name] = dict(entry)
                break

        if not camera_serial_data:
            return

        try:
            store = zarr.open(self.record_path, mode="a")
            episode_key = f"episode_{self.episode_id}{PENDING_SUFFIX}"
            if episode_key in store:
                store[episode_key].attrs["camera_serial"] = camera_serial_data
                if self.verbose:
                    print(f"Saved camera serial data to {episode_key}")
        except Exception as e:
            print(f"Warning: Failed to save camera serial data: {e}")

    def _save_episode_length_and_duration(self) -> None:
        """Save episode length and duration to episode-level zarr attributes."""
        if not self.record_path:
            return

        try:
            store = zarr.open(self.record_path, mode="a")
            episode_key = f"episode_{self.episode_id}{PENDING_SUFFIX}"

            if episode_key not in store:
                return

            episode_data = store[episode_key]

            episode_length = None
            for device_name in episode_data.group_keys():
                device_data = episode_data[device_name]
                for field_name in ["capture_time", "receive_time"]:
                    if field_name in device_data:
                        try:
                            arr = device_data[field_name]
                            if hasattr(arr, "shape") and arr.shape:
                                episode_length = int(arr.shape[0])
                                break
                        except (AttributeError, TypeError):
                            continue
                if episode_length is None:
                    for field_name in device_data.keys():
                        try:
                            arr = device_data[field_name]
                            if hasattr(arr, "shape") and arr.shape:
                                episode_length = int(arr.shape[0])
                                break
                        except (AttributeError, TypeError):
                            continue
                if episode_length is not None:
                    break

            if episode_length is None:
                print_yellow(
                    f"Warning: Could not determine episode length for {episode_key}"
                )
                return

            duration = self._episode_duration(episode_data)

            episode_data.attrs["episode_length"] = episode_length
            episode_data.attrs["episode_duration"] = duration

            if self.verbose:
                print(
                    f"Episode {self.episode_id}: length={episode_length}, duration={duration:.2f}s"
                )
        except Exception as e:
            print(f"Warning: Failed to save episode length/duration: {e}")

    def _save_episode_length_to_dataset(self, episode_id: int) -> None:
        """Save episode length to dataset-level zarr attributes during finalization."""
        if not self.record_path:
            return

        try:
            store = zarr.open(self.record_path, mode="a")
            episode_key = f"episode_{episode_id}"

            if episode_key not in store:
                return

            episode_data = store[episode_key]

            episode_length = episode_data.attrs.get("episode_length")

            if episode_length is None:
                print_yellow(
                    f"Warning: Could not find episode_length in episode-level metadata for {episode_key}"
                )
                return

            episode_lengths = dict(store.attrs.get("episode_lengths", {}))
            episode_lengths[str(episode_id)] = episode_length
            store.attrs["episode_lengths"] = episode_lengths

            if self.verbose:
                print(
                    f"Saved episode length {episode_length} for episode {episode_id} to dataset-level zarr"
                )
        except Exception as e:
            print(f"Warning: Failed to save episode length to dataset-level: {e}")

    def _save_num_episodes(self) -> None:
        """Save the number of episodes to dataset-level zarr attributes."""
        if not self.record_path:
            return

        try:
            store = zarr.open(self.record_path, mode="a")

            num_episodes = 0
            for key in store.keys():
                if key.startswith("episode_") and not key.endswith(PENDING_SUFFIX):
                    num_episodes += 1

            store.attrs["num_episodes"] = num_episodes
            self._sync_qc_pass_status(store, explicit_num_episodes=num_episodes)

            if self.verbose:
                print(f"Saved number of episodes {num_episodes} to dataset-level zarr")
        except Exception as e:
            print(f"Warning: Failed to save number of episodes: {e}")

    def _sync_qc_pass_status(self, store, explicit_num_episodes: int | None) -> None:
        """Align ``qc_pass_status`` length with *explicit_num_episodes* or attrs."""
        attrs = dict(store.attrs)
        if explicit_num_episodes is not None:
            n = int(explicit_num_episodes)
        else:
            n = effective_episode_count_for_qc(attrs, None)
        status = qc_pass_status_from_attrs(attrs, num_episodes=n)
        store.attrs[QC_PASS_STATUS_KEY] = status
        store.attrs.pop("discarded_episode_indices", None)

    @staticmethod
    def _episode_duration(episode_data) -> float:
        """Return the overlapping duration (seconds) across all devices in an episode.

        Computed as ``min(t_end) - max(t_start)`` over every device's
        timestamp range.  Uses ``capture_time`` if present, otherwise
        ``receive_time``.  Returns 0.0 if no usable timestamps are found.
        """
        t_starts = []
        t_ends = []
        for device_name in episode_data.group_keys():
            device = episode_data[device_name]
            ts_field = next(
                (f for f in ("capture_time", "receive_time") if f in device), None
            )
            if ts_field and device[ts_field].shape[0] > 1:
                ts = device[ts_field][:]
                t_starts.append(float(ts[0]))
                t_ends.append(float(ts[-1]))
        if not t_starts:
            return 0.0
        return max(min(t_ends) - max(t_starts), 0.0)

    def get_total_data_hours(self) -> float:
        """Return total recorded data time (hours) by summing stored episode durations."""
        if not self.record_path:
            return 0.0

        try:
            store = zarr.open(self.record_path, mode="r")
            finalized = [
                k
                for k in store.group_keys()
                if k.startswith("episode_") and not k.endswith(PENDING_SUFFIX)
            ]
            total_seconds = sum(
                store[k].attrs.get("episode_duration", 0.0) for k in finalized
            )
            return round(total_seconds / 3600.0, 6)
        except Exception as e:
            print(f"Warning: Failed to compute total data hours: {e}")
            return 0.0

    def _save_total_data_time(self) -> None:
        """Save total recorded data time (hours) to dataset-level zarr attributes."""
        if not self.record_path:
            return

        try:
            total_hours = self.get_total_data_hours()
            store = zarr.open(self.record_path, mode="a")
            store.attrs["total_data_time_hours"] = total_hours

            if self.verbose:
                print(f"Saved total data time {total_hours:.4f}h to dataset-level zarr")
        except Exception as e:
            print(f"Warning: Failed to save total data time: {e}")

    def get_pending_episode_path(self) -> str:
        """Get the path to the current pending episode folder."""
        return os.path.join(
            self.record_path, f"episode_{self.episode_id}{PENDING_SUFFIX}"
        )

    def get_final_episode_path(self) -> str:
        """Get the path to the final episode folder (without pending suffix)."""
        return os.path.join(self.record_path, f"episode_{self.episode_id}")

    def shutdown_background_services(self) -> None:
        """Drain + stop the background aligner.

        Idempotent and a no-op when no aligner is set. The ``.shutdown()`` is
        guarded so a failing teardown in a ``finally`` block can't mask the
        original error or block the rest of cleanup.
        """
        if self.background_aligner is not None:
            try:
                self.background_aligner.shutdown()
            except Exception as e:  # noqa: BLE001
                print_yellow(f"background_aligner.shutdown() failed: {e}")

    def finalize_episode(self, findings: Optional[Dict[str, Any]] = None) -> bool:
        """Finalize a pending episode after validation passes.

        Renames episode_N.pending/ to episode_N/, updates zarr metadata,
        and increments episode ID for all recorders.

        Args:
            findings: Report-only data-check measurements
                (``data_check_findings``), stored as the episode attr
                ``data_check_findings``. None writes nothing.

        Returns:
            bool: True if finalization successful, False otherwise.
        """
        pending_path = self.get_pending_episode_path()
        final_path = self.get_final_episode_path()
        episode_id = self.episode_id

        if not os.path.exists(pending_path):
            print(f"Error: Pending episode not found: {pending_path}")
            return False

        if os.path.exists(final_path):
            print(f"Error: Final episode already exists: {final_path}")
            return False

        # Before the rename, so the finalized episode carries them.
        if findings is not None:
            try:
                store = zarr.open(self.record_path, mode="a")
                episode_key = f"episode_{episode_id}{PENDING_SUFFIX}"
                store[episode_key].attrs["data_check_findings"] = findings
            except Exception as e:  # noqa: BLE001 - findings never block a save
                print_yellow(f"Warning: Could not save data check findings: {e}")

        os.rename(pending_path, final_path)
        print(f"Renamed {pending_path} -> {final_path}")

        self._save_episode_length_to_dataset(episode_id)

        for recorder in self.recorders:
            recorder.finalize_episode()

        self._save_num_episodes()
        self._save_total_data_time()

        if self.background_aligner is not None:
            try:
                self.background_aligner.queue_alignment(episode_idx=episode_id)
            except Exception as e:  # noqa: BLE001
                print(
                    f"Warning: Failed to queue alignment for episode {episode_id}: {e}"
                )

        print(f"Finalized episode {episode_id}")
        return True

    def discard_staging(self) -> bool:
        """Discard the pending episode folder after validation fails.

        Deletes episode_N.pending/ and cleans up zarr metadata.
        Episode ID is not incremented.

        Returns:
            bool: True if discard successful, False otherwise.
        """
        pending_path = self.get_pending_episode_path()
        episode_id = self.episode_id

        if os.path.exists(pending_path):
            shutil.rmtree(pending_path)
            print(f"Deleted pending episode folder: {pending_path}")

        for recorder in self.recorders:
            recorder.discard_staging()

        if os.path.exists(pending_path):
            print(f"Warning: Pending folder still exists after discard: {pending_path}")
            return False

        print(f"Discarded pending episode {episode_id}")
        return True

    def quarantine_staging(
        self,
        issues: Optional[List[str]] = None,
        findings: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Move a failed episode to a separate quarantine dataset for later review.

        Moves episode_N.pending/ from the main dataset into the session's
        quarantine dataset (``output/quarantine/{task}/{session}``, or the sibling
        ``<record_path>_quarantine/`` for a session outside the output folder)
        with a sequentially assigned episode ID.
        The main dataset's episode ID is not incremented, so the next recording
        reuses the same ID.

        Args:
            issues: Optional list of failure reasons to store as zarr metadata.
            findings: Report-only data-check measurements, stored as the attr
                ``data_check_findings``. None writes nothing.

        Returns:
            bool: True if quarantine successful, False otherwise.
        """
        pending_path = self.get_pending_episode_path()

        if not os.path.exists(pending_path):
            print(f"Error: Pending episode not found: {pending_path}")
            return False

        quarantine_root = counterpart(self.record_path, "quarantine")
        os.makedirs(quarantine_root, exist_ok=True)

        for cfg_name in ("config.yaml", "data_check_config.yaml"):
            src = os.path.join(self.record_path, cfg_name)
            dst = os.path.join(quarantine_root, cfg_name)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

        group = zarr.open(quarantine_root, mode="a")
        existing_ids = [
            int(k.split("_")[1]) for k in group.group_keys() if k.startswith("episode_")
        ]
        episode_id = max(existing_ids) + 1 if existing_ids else 0

        dest_path = os.path.join(quarantine_root, f"episode_{episode_id}")

        shutil.move(pending_path, dest_path)
        print(f"Quarantined: {pending_path} -> {dest_path}")

        if issues or findings is not None:
            try:
                q_store = zarr.open(quarantine_root, mode="a")
                q_group = q_store[f"episode_{episode_id}"]
                if issues:
                    q_group.attrs["data_check_issues"] = issues
                    q_group.attrs["data_check_passed"] = False
                    q_group.attrs["original_dataset"] = self.record_path
                if findings is not None:
                    q_group.attrs["data_check_findings"] = findings
            except Exception as e:
                print(f"Warning: Could not save failure metadata: {e}")

        for recorder in self.recorders:
            recorder.discard_staging()

        print(f"Quarantined episode {episode_id} -> {quarantine_root}")
        return True

    def get_latest_frames(self) -> dict:
        frames = {}
        for recorder in self.recorders:
            frames.update(recorder.get_last_k_frames_from_all(k=1))
        return frames

    def get_last_k_frames_from_all(self, k: int) -> dict:
        """Get last k frames from all recorders.

        Args:
            k: Number of frames to retrieve from each source.

        Returns:
            Dict mapping source names to lists of frames.
        """
        frames = {}
        for recorder in self.recorders:
            frames.update(recorder.get_last_k_frames_from_all(k=k))
        return frames

    def _find_source(self, name: str):
        """Locate (recorder, source_idx, source) by device_name. Returns None
        if not found. Used by the by-name pause/resume/push helpers below."""
        for recorder in self.recorders:
            for i, source in enumerate(recorder.data_sources):
                if getattr(source, "device_name", None) == name:
                    return recorder, i, source
        return None

    def get_source(self, name: str):
        """Return the underlying source object (e.g. Camera) by device_name,
        or None if no such source is registered."""
        found = self._find_source(name)
        return found[2] if found else None

    def pause_source_streaming(self, name: str) -> None:
        """Pause the stream thread of one source by device_name. Caller must
        match each pause with a resume."""
        found = self._find_source(name)
        if found is None:
            print_red(
                f"[RecorderManager] pause_source_streaming: no source named {name!r}"
            )
            return
        recorder, idx, _ = found
        recorder.pause_source(idx)
        print_green(f"[RecorderManager] paused stream for {name!r}")

    def resume_source_streaming(self, name: str) -> None:
        """Resume a previously paused source by device_name. Idempotent."""
        found = self._find_source(name)
        if found is None:
            print_red(
                f"[RecorderManager] resume_source_streaming: no source named {name!r}"
            )
            return
        recorder, idx, _ = found
        recorder.resume_source(idx)
        print_green(f"[RecorderManager] resumed stream for {name!r}")

    def push_frame(self, name: str, frame) -> None:
        """Inject a frame into a source's buffer. Used while the stream thread
        is paused so the visualizer keeps seeing fresh imagery (e.g. during the
        calibration session, where the calibration check is the sole producer
        for the affected cameras). Silent no-op if `name` is unknown."""
        found = self._find_source(name)
        if found is not None:
            recorder, idx, _ = found
            recorder.frame_buffers[idx].put(frame)

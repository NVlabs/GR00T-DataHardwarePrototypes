# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Protocol

import zarr

from umi.common.recording.base_recorder import PENDING_SUFFIX
from umi.common.recording.session_slot import (
    config_yaml_bytes,
    device_config_entries,
)
from umi.common.utility.print_utils import print_red, print_yellow
from umi.utils.device_manager import get_exoskeleton_serial

# Numeric source types whose device identity is persisted per episode.
_SERIAL_TRACKED_SOURCE_TYPES = ("dexumi_exoskeleton",)


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
    def get_active_warnings(self) -> dict: ...
    def get_frame_drop_stats(self) -> dict: ...


class RecorderManager:
    def __init__(
        self,
        recorders: List[Recorder],
        record_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        background_aligner: Optional[Any] = None,
    ):
        self.recorders = recorders
        self.record_path = record_path
        self.config = config
        self.verbose = verbose
        self._task_description: str = ""
        self.background_aligner = background_aligner
        # Device binding set after all sources open. Copy it and its launch_id into
        # each episode; None disables binding snapshots for standalone use.
        self.device_binding: Optional[dict] = None

        self._save_config()
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

    def save_config(self) -> None:
        """Rewrite the session config.yaml after a between-episode config edit.

        Later episodes snapshot it (``_save_episode_configs``); earlier
        episodes keep the snapshot they were saved with.
        """
        self._save_config()

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
        # Replace config files atomically so concurrent snapshotting or uploads cannot
        # read partial content.
        tmp_path = config_path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(data)
        os.replace(tmp_path, config_path)

        print(f"Config saved to {config_path}")

    def _device_config_files(self) -> list[tuple[str, str]]:
        """Resolve unique config files referenced by ANY source in the config.

        Scans every source collection — i.e. each top-level list of source
        dicts (``video_sources``, ``numeric_sources``, and any future
        ``*_sources`` list) — for a ``params.config_file``. Today that is the
        UMI gripper / DexUMI exoskeleton calibration; tomorrow it could be a
        camera or other device calibration, and this picks it up without
        needing to know the collection's name. Returns
        ``(abs_source_path, basename)`` deduped by source path — so the config
        that produced an episode is bundled alongside it (provenance +
        self-contained data once it leaves the repo). Sub-sources of one device
        (e.g. encoder + tactile) share a config_file and collapse to one entry.
        """
        return device_config_entries(self.config)

    def _save_device_configs(self) -> None:
        """Copy each referenced device config file into the session folder.

        Like ``_save_config``: the calibration files travel with
        the recording (session root + every episode, see ``_save_episode_configs``)
        so an episode / aligned session is self-describing once it leaves the repo
        (DexUMI convert reads the session-root copy by basename).

        Not write-time locked here: the launch-time session-slot resolver
        (``session_file_conflicts``) already decided the folder. If a
        recalibration conflicts with an existing session that has episodes, the
        launcher bumps to a new ``_N`` folder first; only then do we write /
        refresh the session copy for the chosen folder.
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
        """Snapshot the recording config and calibrations into the pending
        episode folder.

        This captures *which config was in effect when this episode was
        recorded*. Useful when the script is restarted mid-dataset with a
        modified config.yaml — without these per-episode snapshots there is no
        way to tell which episodes used which settings.

        Written files:
            episode_{id}.pending/config.yaml   (copied from session copy)
            episode_{id}.pending/<calibration> (copied from session copy)

        Every snapshot is copied from the **session-level** file (written at setup
        by ``_save_config`` / ``_save_device_configs``) — never from in-memory
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
            shutil.copy2(session_config, os.path.join(episode_dir, "config.yaml"))
            if self.verbose:
                print(f"Saved episode config snapshot to {episode_dir}/config.yaml")

        # Snapshot calibrations from the session copy, never the mutable source
        # files. A missing session calibration is a provenance error and must fail
        # the episode save.
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

    def _save_meta_data(self) -> None:
        """Save the current meta data (commit ID, operator_name, task_name, HW_version) to dataset-level zarr attributes."""
        if not self.record_path:
            return

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            commit_id = result.stdout.strip()
        except subprocess.SubprocessError as e:
            print_yellow(f"Warning: Failed to run git — {e}")
            return

        try:
            store = zarr.open(self.record_path, mode="a")
            store.attrs["commit_id"] = commit_id
            if self.verbose:
                print(f"Saved commit ID to dataset-level zarr: {commit_id}")
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
            if "discarded_reasons" not in store.attrs:
                store.attrs["discarded_reasons"] = {}
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
        success = True
        for recorder in self.recorders:
            success &= recorder.start_recording(episode_id=episode_id)

        if success:
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
        Metadata (task description, latency, etc.) is saved to
        the same pending group.

        If *pending* is ``True`` the episode stays in its staging folder until
        the caller runs ``finalize_episode()`` or ``discard_staging()``.

        If *pending* is ``False`` the episode is finalized immediately
        (renamed to ``episode_N/`` and the episode ID is incremented).

        Args:
            pending: If True, keep the episode in the staging folder only.
                     If False, finalize the episode after saving.
        """
        for recorder in self.recorders:
            recorder.save_recordings()

        self._save_task_description()
        self._save_latency_data()
        self._save_device_serial_data()
        self._save_device_identity_data()
        self._save_episode_length_and_duration()
        self._save_episode_configs()

        if not pending:
            self.finalize_episode()

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

    @staticmethod
    def _get_cached_uart_reader(uart_device_name: str):
        # Local import: recording_utils imports umi.common.recording at module
        # level (record_manager is imported mid-way through that package's
        # __init__), so a top-level import here is circular.
        from umi.common.utility.recording_utils import get_cached_uart_reader

        return get_cached_uart_reader(uart_device_name)

    def _save_device_serial_data(self) -> None:
        """Save device identity for uart-backed numeric sources to the pending episode.

        Per source: the configured device name, its registry serial, the USB
        serial actually observed on the open port, and the count of frames
        that carried a foreign board id — so a wrong/swapped device is provable
        offline, not just blocked live.
        """
        if not self.record_path or not self.config:
            return

        gripper_serial_data = {}

        numeric_sources = self.config.get("numeric_sources", [])
        for source_cfg in numeric_sources:
            if source_cfg.get("type") not in _SERIAL_TRACKED_SOURCE_TYPES:
                continue

            source_name = source_cfg.get("name")
            uart_device_name = source_cfg.get("uart_device_name")

            if not uart_device_name:
                continue

            try:
                device_serial = get_exoskeleton_serial(uart_device_name)
            except ValueError as e:
                print_yellow(
                    f"Warning: Could not get serial for {uart_device_name}: {e}"
                )
                continue

            entry = {
                "device_name": uart_device_name,
                "serial": device_serial,
            }
            reader = self._get_cached_uart_reader(uart_device_name)
            if reader is not None:
                observed = reader.get_serial_for_port(reader.port)
                if observed is not None:
                    entry["observed_serial"] = observed
                entry["device_id_mismatch_count"] = int(reader.device_id_mismatch_count)
            gripper_serial_data[source_name] = entry

        if not gripper_serial_data:
            return

        try:
            store = zarr.open(self.record_path, mode="a")
            episode_key = f"episode_{self.episode_id}{PENDING_SUFFIX}"
            if episode_key in store:
                # gripper_serial is the on-disk name of this identity attribute.
                store[episode_key].attrs["gripper_serial"] = gripper_serial_data
                if self.verbose:
                    print(f"Saved device serial data to {episode_key}")
        except Exception as e:
            print(f"Warning: Failed to save device serial data: {e}")

    def _save_device_identity_data(self) -> None:
        """Record which physical device stood behind each source in this episode.

        ``device_identity`` = ``{source: {kind, name, id, observed_id, ...}}``
        for every source with an identity (observations re-read now); with a
        launch binding also ``device_binding_launch_id`` and a copy of
        ``device_binding.yaml`` in the episode folder. Provenance only: a
        failure is a warning and never blocks saving the episode.
        """
        if not self.record_path:
            return
        from umi.common.recording.device_identity import (
            DEVICE_BINDING_FILE,
            binding_yaml_bytes,
            collect_identities,
        )

        try:
            identities = collect_identities(getattr(self, "recorders", None) or [])
            binding = getattr(self, "device_binding", None)
            if not identities and not binding:
                return
            store = zarr.open(self.record_path, mode="a")
            episode_key = f"episode_{self.episode_id}{PENDING_SUFFIX}"
            if episode_key not in store:
                return
            attrs = {"device_identity": identities}
            if binding:
                attrs["device_binding_launch_id"] = binding.get("launch_id")
            store[episode_key].attrs.update(attrs)
            if binding:
                episode_dir = os.path.join(self.record_path, episode_key)
                with open(os.path.join(episode_dir, DEVICE_BINDING_FILE), "wb") as f:
                    f.write(binding_yaml_bytes(binding))
        except Exception as e:  # noqa: BLE001 - provenance never blocks a save
            print_yellow(f"Warning: Failed to save device identity data: {e}")

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

            episode_ids = self._finalized_episode_ids(store)
            num_episodes = len(episode_ids)

            store.attrs["num_episodes"] = num_episodes

            if self.verbose:
                print(f"Saved number of episodes {num_episodes} to dataset-level zarr")
        except Exception as e:
            print(f"Warning: Failed to save number of episodes: {e}")

    @staticmethod
    def _finalized_episode_ids(store) -> set[int]:
        return {
            int(match.group(1))
            for key in store.group_keys()
            if (match := re.fullmatch(r"episode_(\d+)", key))
        }

    def get_saved_episode_count(self) -> int:
        """Count retained finalized episodes, excluding staging."""
        if not self.record_path:
            return 0
        return len(self._finalized_episode_ids(zarr.open(self.record_path, mode="r")))

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

    def get_total_data_hours(self, *, strict: bool = False) -> float:
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
            if strict:
                raise
            print(f"Warning: Failed to compute total data hours: {e}")
            return 0.0

    def _save_total_data_time(self) -> None:
        """Save total recorded data time (hours) to dataset-level zarr attributes."""
        if not self.record_path:
            return

        try:
            total_hours = round(self.get_total_data_hours(), 6)
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
        """Drain + stop the background aligner if attached.

        Idempotent and no-op when no aligner is set. ``.shutdown()`` is guarded
        so a failing teardown in a ``finally`` block can't mask the original
        error or block the rest of cleanup.
        """
        if self.background_aligner is not None:
            try:
                self.background_aligner.shutdown()
            except Exception as e:  # noqa: BLE001
                print_yellow(f"background_aligner.shutdown() failed: {e}")

    def finalize_episode(self) -> bool:
        """Finalize a saved pending episode.

        Renames episode_N.pending/ to episode_N/, updates zarr metadata,
        and increments episode ID for all recorders.

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

        os.rename(pending_path, final_path)
        print(f"Renamed {pending_path} -> {final_path}")

        # Advance recorder IDs immediately after renaming staging data. Later
        # metadata failures must not leave an abort pointing at finalized data.
        for recorder in self.recorders:
            recorder.finalize_episode()

        try:
            self._save_episode_length_to_dataset(episode_id)

            self._save_num_episodes()
            self._save_total_data_time()
        except Exception as e:  # noqa: BLE001
            # The episode is finalized on disk. Recovery can resume alignment; do not
            # report it as discarded.
            print_red(
                f"Error: episode_{episode_id} was finalized locally but session "
                f"bookkeeping failed; restart to recover: {e}"
            )
            return False

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
        """Discard the pending episode folder.

        Deletes episode_N.pending/ and cleans up zarr metadata.
        Episode ID is NOT incremented.

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

    def get_latest_frames(self) -> dict:
        frames = {}
        for recorder in self.recorders:
            frames.update(recorder.get_last_k_frames_from_all(k=1))
        return frames

    def get_active_warnings(self) -> dict:
        """Every recorder's live warnings: {(source_name, warning_type): message}."""
        warnings = {}
        for recorder in self.recorders:
            warnings.update(recorder.get_active_warnings())
        return warnings

    def get_frame_drop_stats(self) -> dict:
        """Every recorder's missed frames this episode: {source_name: (count, last_time)}."""
        stats = {}
        for recorder in self.recorders:
            stats.update(recorder.get_frame_drop_stats())
        return stats

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

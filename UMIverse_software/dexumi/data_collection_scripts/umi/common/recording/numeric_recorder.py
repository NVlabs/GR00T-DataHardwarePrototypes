# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, Type

import numpy as np

from umi.common.data.frame import NumericFrame
from umi.common.recording.base_recorder import BaseRecorder
from umi.common.utility.rate_keeper import RateKeeper


class NumericRecorder(BaseRecorder):
    def __init__(
        self,
        record_fps: int,
        stream_fps: int,
        record_path: str,
        numeric_sources: List,
        frame_data_class: Type[NumericFrame] = NumericFrame,
        verbose=False,
        frame_queue_size: int = 20,
        info_verbose: bool = True,
        warning_verbose: bool = True,
    ):
        super().__init__(
            record_fps=record_fps,
            stream_fps=stream_fps,
            record_path=record_path,
            data_sources=numeric_sources,
            frame_data_class=frame_data_class,
            verbose=verbose,
            frame_queue_size=frame_queue_size,
            info_verbose=info_verbose,
            warning_verbose=warning_verbose,
        )

    def _recording_thread(self, source_idx: int, key_idx: int, episode_id: int):
        """Thread function to process and store frames from the queue.

        Reads the combined frame from the per-source buffer, splits it,
        selects this thread's sub-key, and accumulates the sub-frame's
        non-None fields into the per-(source, key) accumulator.

        For non-splittable frames (`split_key == ""`), `frame.split()`
        returns ``{"": frame}`` so the sub-frame IS the combined frame -
        equivalent to the legacy behavior.
        """
        source = self.sources[source_idx]
        source_name = source.device_name
        split_key = self._split_keys[key_idx]
        sub_class = self._sub_frame_classes[split_key]
        this_source_episode_frames = self.episode_frames[source_idx][key_idx]
        fps = self.record_fps
        last_timestamp = None

        thread_label = source_name if split_key == "" else f"{source_name}_{split_key}"
        rate_keeper = RateKeeper(
            target_rate=fps,
            verbose=self.verbose,
            name=f"NumericRecorder_{thread_label}",
        )

        while self._recording_event.is_set():
            with rate_keeper:
                frame_data = self.get_next_frame_until_valid(
                    source_idx,
                    last_timestamp,
                    timeout=1 / fps,
                    count_drops=key_idx == 0,
                )
                # Advance FPS and delay tracking only for the first split key: all
                # sub-streams
                # share one source fetch.
                if key_idx == 0:
                    self.frame_delay_detection(
                        source_idx, frame_data, last_timestamp, timeout=1 / fps
                    )
                    self.fps_tracking(source_idx, fps)

                if frame_data is None:
                    continue
                last_timestamp = frame_data.receive_time

                sub = frame_data.split().get(split_key)
                if sub is None:
                    continue
                if not isinstance(sub, sub_class):
                    print(
                        f"Warning: sub-frame for {thread_label!r} is "
                        f"{type(sub).__name__}, expected {sub_class.__name__}"
                    )
                    continue

                for field_name, value in vars(sub).items():
                    if field_name in this_source_episode_frames:
                        if value is not None:
                            this_source_episode_frames[field_name].append(value)
                    else:
                        print(
                            f"Warning: Field {field_name} not found in "
                            f"episode_frames for {thread_label}"
                        )

    def save_recordings(self):
        """Save all recorded numeric data to the pending episode folder.

        Note: Episode ID is NOT incremented here. Call finalize_episode() after
        data validation passes to rename the pending folder and increment the ID.
        """
        from umi.common.recording.base_recorder import PENDING_SUFFIX

        this_episode_id = self._episode_id
        episode_key = f"episode_{this_episode_id}{PENDING_SUFFIX}"
        episode_data = self.record_roots.require_group(episode_key)

        # Fields to skip saving (metadata, not useful time-series data)
        skip_fields = {"frame_source"}

        for i, source in enumerate(self.sources):
            base = source.device_name
            for k_idx, key in enumerate(self._split_keys):
                # Group = {base}[_{split_key}]: non-splittable (key=="") ->
                # {base}; splittable -> {base}_{key}.
                group_name = base if key == "" else f"{base}_{key}"
                print(f"Writing {group_name} data to disk...")
                group = episode_data.require_group(group_name)
                frames_to_save = self.episode_frames[i][k_idx]

                for field_name, value_list in frames_to_save.items():
                    if field_name not in skip_fields:
                        self._save_field(group, field_name, value_list)

        # Advance episode IDs only after finalization, not while saving staging data.
        print(
            f"[{', '.join(s.device_name for s in self.sources)}] Recording saved to staging."
        )

    def _save_field(self, group, field_name: str, value_list: List[Any]):
        """Save one field's per-frame values as a single array."""
        if not value_list:
            print(f"Warning: No data to save for field {field_name}")
            return

        first = value_list[0]

        if isinstance(first, (int, float, np.integer, np.floating)):
            arr = np.array(value_list)
        elif isinstance(first, np.ndarray):
            arr = np.stack(value_list, axis=0)
        elif isinstance(first, list):
            arr = np.array(value_list)
        elif isinstance(first, str):
            max_len = max(len(s) for s in value_list if s)
            arr = np.array(value_list, dtype=f"U{max_len}")
        else:
            print(
                f"Warning: Skipping field {field_name}: unsupported value "
                f"type {type(first).__name__}"
            )
            return

        self._write_array(group, field_name, arr)

    @staticmethod
    def _write_array(group, field_name: str, arr: np.ndarray) -> None:
        """Write `arr` to `group[field_name]`, forcing zarr to materialize chunks
        even when their contents equal the fill_value.

        zarr v3 elides chunks that match the fill_value (default 0.0 for
        floats), which is fine for general storage but breaks the data
        check's `exists` rule: a sensor that legitimately reads all zeros
        for an entire episode (e.g. a tactile pad never engaged) becomes
        indistinguishable from a sensor that wrote nothing — both report
        `nchunks_initialized == 0`. Creating the array up-front with
        `config={"write_empty_chunks": True}` and then assigning the data
        forces every chunk to be physically written.
        """
        # Materializing fill values is harmless for string arrays and required
        # for reliable zero-valued numeric data.
        zarr_arr = group.create_array(
            name=field_name,
            shape=arr.shape,
            dtype=arr.dtype,
            config={"write_empty_chunks": True},
        )
        zarr_arr[...] = arr

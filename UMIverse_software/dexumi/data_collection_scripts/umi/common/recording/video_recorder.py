# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
from dataclasses import fields
from typing import List, Optional, Type

import numpy as np

from umi.common.data.frame import RGBCameraFrame, get_zarr_storage_config
from umi.common.data.zarr_utils import create_zarr_array
from umi.common.hardware_sdk.camera.camera import Camera
from umi.common.recording.base_recorder import BaseRecorder
from umi.common.recording.video_writer_backends import (
    VideoEncoderKind,
    VideoWriterBackend,
    create_video_writer,
)
from umi.common.utility.rate_keeper import RateKeeper


class VideoRecorder(BaseRecorder):
    def __init__(
        self,
        record_fps: int,
        stream_fps: int,
        record_path: str,
        data_sources: List[Camera],
        frame_data_class: Type[RGBCameraFrame] = RGBCameraFrame,
        verbose: bool = False,
        max_workers: int = 32,
        frame_queue_size: int = 20,
        info_verbose: bool = True,
        warning_verbose: bool = True,
        video_encoder: VideoEncoderKind = "ffmpeg_h264",
    ) -> None:
        super().__init__(
            record_fps=record_fps,
            stream_fps=stream_fps,
            record_path=record_path,
            data_sources=data_sources,
            frame_data_class=frame_data_class,
            verbose=verbose,
            frame_queue_size=frame_queue_size,
            info_verbose=info_verbose,
            warning_verbose=warning_verbose,
        )

        self.max_workers = max_workers
        self.video_encoder = video_encoder
        self.video_writers: List[Optional[VideoWriterBackend]] = []

    def _release_video_writers(self) -> None:
        """Close every encoder after its producer stops, including on discard."""
        if self._recording_event.is_set():
            raise RuntimeError("Stop recording before releasing video writers")
        # Base stop has a short join timeout. A producer can still be finishing
        # a bounded queue write; never close or replace its writer concurrently.
        for thread in self.recording_threads:
            thread.join(timeout=10.0)
            if thread.is_alive():
                raise RuntimeError("Video recording thread did not stop")
        first_error = None
        for index, writer in enumerate(self.video_writers):
            try:
                if writer is not None:
                    writer.release()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
            finally:
                self.video_writers[index] = None
        if first_error is not None:
            raise first_error

    def _discard_video_writers(self) -> None:
        # During discard, encoder failures must not interrupt cleanup. Save paths
        # still propagate errors to prevent reporting incomplete data as saved.
        try:
            self._release_video_writers()
        except Exception as exc:  # noqa: BLE001
            self._print_error(
                f"[Recorder] Ignoring video writer error while discarding: {exc}"
            )

    def clear_recording(self) -> bool:
        if self._recording_event.is_set():
            return super().clear_recording()
        self._discard_video_writers()
        return super().clear_recording()

    def discard_staging(self) -> bool:
        if self._recording_event.is_set():
            return super().discard_staging()
        self._discard_video_writers()
        return super().discard_staging()

    def reset_episode_recording(self) -> bool:
        """Reset and prepare for a new episode recording."""
        if self._recording_event.is_set():
            return False
        self._discard_video_writers()
        if not super().reset_episode_recording():
            return False

        pending_path = self.get_pending_episode_path()
        for source in self.sources:
            video_path = os.path.join(
                pending_path,
                f"{source.device_name}.mp4",
            )
            if os.path.exists(video_path):
                os.remove(video_path)

        self.video_writers = [None for _ in self.sources]

        return True

    def _recording_thread(self, source_idx: int, key_idx: int, episode_id: int):
        """Thread function to process and store frames from the queue.

        Camera frames are non-splittable (`splittable_keys() == [""]`), so
        ``key_idx`` is always 0 here. The parameter is part of the abstract
        signature to support per-(source, sub-stream) recording for splittable
        frames (see NumericRecorder).
        """
        from umi.common.recording.base_recorder import PENDING_SUFFIX

        source = self.sources[source_idx]
        source_name = source.device_name  # identity, for logging
        group_name = source.device_name  # recorded group / mp4 file name
        # Cameras use the non-splittable entry at episode_frames[source_idx][0].
        stored_frame = self.episode_frames[source_idx][key_idx]
        video_path = os.path.join(
            self.record_path,
            f"episode_{episode_id}{PENDING_SUFFIX}",
            f"{group_name}.mp4",
        )
        fps = self.record_fps

        first_frame = self.peek_latest_frame(source_idx)
        while first_frame is None and self._recording_event.is_set():
            self._print_warning(
                f"[{source_name}] Waiting for first frame to initialize video writer..."
            )
            time.sleep(0.1)
            first_frame = self.peek_latest_frame(source_idx)
        if not self._recording_event.is_set():
            return

        frame = first_frame.rgb
        height, width = frame.shape[:2]
        print(
            f"[{source_name}] video_path: {video_path} (encoder={self.video_encoder})"
        )
        self.video_writers[source_idx] = create_video_writer(
            self.video_encoder,
            video_path,
            width,
            height,
            float(fps),
        )
        if self.verbose:
            print(f"Initialized video writer for {source_name}")
        video_writer = self.video_writers[source_idx]

        try:
            last_timestamp = first_frame.receive_time
            rate_keeper = RateKeeper(
                target_rate=fps,
                verbose=self.verbose,
                name=f"VideoRecorder_{self.data_sources[source_idx].device_name}",
            )

            while self._recording_event.is_set():
                with rate_keeper:
                    frame_data = self.get_next_frame_until_valid(
                        source_idx, last_timestamp, timeout=1 / fps
                    )
                    if frame_data is None:
                        continue
                    self.frame_delay_detection(
                        source_idx, frame_data, last_timestamp, timeout=1 / fps
                    )
                    self.fps_tracking(source_idx, fps)
                    last_timestamp = frame_data.receive_time

                    for field in fields(self.frame_data_class):
                        value = getattr(frame_data, field.name)
                        if value is not None:
                            if field.name == "rgb":
                                video_writer.write(value)
                                if self.verbose:
                                    print(
                                        f"Written frame to video for camera {source_idx}"
                                    )
                            elif field.name not in ["rgb"]:
                                stored_frame[field.name].append(value)
        except Exception as e:
            self._print_error(f"[{source_name}] Error processing frame: {e}")

    def save_recordings(self):
        """Save all recorded frames to the pending episode folder.

        Note: Episode ID is NOT incremented here. Call finalize_episode() after
        data validation passes to rename the pending folder and increment the ID.
        """
        from umi.common.recording.base_recorder import PENDING_SUFFIX

        this_episode_id = self._episode_id
        episode_key = f"episode_{this_episode_id}{PENDING_SUFFIX}"
        episode_data = self.record_roots.require_group(episode_key)

        print("Releasing video writers...")
        self._release_video_writers()

        for i, source in enumerate(self.sources):
            source_name = source.device_name
            if self.verbose:
                print(f"Writing {source_name} data to disk...")
            episode_camera_data = episode_data.require_group(source_name)
            # Cameras are non-splittable -> single sub-stream at key_idx 0.
            episode_to_save = self.episode_frames[i][0]

            # Skip string metadata fields (zarr V3 has unstable spec for strings)
            skip_fields = {"frame_source"}

            for key, value_list in episode_to_save.items():
                if not value_list or key in skip_fields:
                    if self.verbose and not value_list:
                        print(f"No {key} data to save for {source_name}")
                    continue
                if self.verbose:
                    print(f"Writing {key} data...")
                data = np.array(value_list)
                storage_config = get_zarr_storage_config(self.frame_data_class, key)
                create_zarr_array(
                    episode_camera_data,
                    key,
                    data,
                    storage_config=storage_config,
                    max_workers=self.max_workers,
                )

        # Advance episode IDs only after finalization, not while saving staging data.
        print(
            f"[{', '.join(s.device_name for s in self.sources)}] Recording saved to staging."
        )

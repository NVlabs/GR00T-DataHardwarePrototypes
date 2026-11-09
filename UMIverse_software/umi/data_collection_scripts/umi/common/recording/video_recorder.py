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

    @property
    def video_record_path(self) -> str:
        """Alias for record_path for backward compatibility."""
        return self.record_path

    def reset_episode_recording(self) -> bool:
        """Reset and prepare for a new episode recording."""
        if not super().reset_episode_recording():
            return False

        pending_path = self.get_pending_episode_path()
        for source in self.sources:
            video_path = os.path.join(
                pending_path,
                f"{source.device_name}{self._group_suffix(source)}.mp4",
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
        group_name = (  # recorded group / mp4 file name (device_name + role suffix)
            f"{source.device_name}{self._group_suffix(source)}"
        )
        # Cameras use episode_frames[source_idx][0], the non-splittable entry.
        stored_frame = self.episode_frames[source_idx][key_idx]
        video_path = os.path.join(
            self.video_record_path,
            f"episode_{episode_id}{PENDING_SUFFIX}",
            f"{group_name}.mp4",
        )
        fps = self.record_fps

        first_frame = self.peek_latest_frame(source_idx)
        while first_frame is None:
            self._print_warning(
                f"[{source_name}] Waiting for first frame to initialize video writer..."
            )
            time.sleep(0.1)
            first_frame = self.peek_latest_frame(source_idx)

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
            # After a stall, consume each fresh frame that arrived rather than skip
            # the recording slots.
            rate_keeper = RateKeeper(
                target_rate=fps,
                verbose=self.verbose,
                name=f"VideoRecorder_{self.data_sources[source_idx].device_name}",
                catch_up=True,
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

        Note: Episode ID is not incremented here. Call finalize_episode() after
        data validation passes to rename the pending folder and increment the ID.
        """
        from umi.common.recording.base_recorder import PENDING_SUFFIX

        this_episode_id = self._episode_id
        episode_key = f"episode_{this_episode_id}{PENDING_SUFFIX}"
        episode_data = self.record_roots.require_group(episode_key)

        print("Releasing video writers...")
        for writer in self.video_writers:
            if writer is not None:
                writer.release()

        for i, source in enumerate(self.sources):
            source_name = f"{source.device_name}{self._group_suffix(source)}"
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

        print(
            f"[{', '.join(s.device_name for s in self.sources)}] Recording saved to staging."
        )

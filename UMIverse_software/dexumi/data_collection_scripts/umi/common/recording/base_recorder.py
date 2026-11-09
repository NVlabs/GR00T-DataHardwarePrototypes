# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import fields
from threading import Event, Lock, Thread
from typing import Any, List, Optional, Type

import numpy as np
import zarr

from umi.common.data.frame_buffer import FrameBuffer
from umi.common.recording.constants import PENDING_SUFFIX
from umi.common.utility.print_utils import print_green, print_red, print_yellow
from umi.common.utility.rate_keeper import RateKeeper


class BaseRecorder(ABC):
    """Base class for recorders that handle streaming and recording of data sources."""

    def __init__(
        self,
        record_fps: int,
        stream_fps: int,
        record_path: str,
        data_sources: List,
        frame_data_class: Type,
        verbose: bool = False,
        frame_queue_size: int = 20,
        info_verbose: bool = True,
        warning_verbose: bool = True,
    ):
        device_names = [device.device_name for device in data_sources]
        if np.unique(device_names).size != len(device_names):
            raise ValueError("Device names must be unique.")

        self.record_fps = record_fps
        self.stream_fps = stream_fps
        self.record_path = record_path
        self.data_sources = data_sources
        self.frame_data_class = frame_data_class
        # Derive split keys from the combined frame class once. Each key gets its
        # own recording thread and Zarr group; a non-splittable frame uses [""].
        self._split_keys: List[str] = list(frame_data_class.splittable_keys())
        self._sub_frame_classes = dict(frame_data_class.sub_frame_classes())
        self.verbose = verbose
        self.info_verbose = info_verbose
        self.warning_verbose = warning_verbose

        print(
            f"Verbose: {verbose}, Info verbose: {info_verbose}, Warning verbose: {warning_verbose}"
        )

        self._streaming_event: Event = Event()
        self._recording_event: Event = Event()
        self.frame_buffers: List[FrameBuffer] = [
            FrameBuffer(maxlen=frame_queue_size) for _ in data_sources
        ]

        os.makedirs(record_path, exist_ok=True)
        self.record_roots = zarr.open(record_path, mode="a")

        self.stream_threads: List[Thread] = []
        self.recording_threads: List[Thread] = []

        self._episode_id = self.init_episode_id()
        self.episode_frames: List[dict] = []

        # Frame delay and FPS tracking (per-source)
        self.reset_delay_and_fps_tracking()

        # Recording threads write active warnings; the main loop reads them through
        # get_active_warnings(). Keys are (source_name, warning_type).
        self._active_warnings: dict = {}
        self._warnings_lock = Lock()
        # Per-source missed-frame count and latest monotonic miss time, including
        # isolated drops. Guard with _warnings_lock and reset for each episode.
        self._frame_drop_stats: dict = {}

        print(f"Starting episode {self.episode_id}")

    def reset_delay_and_fps_tracking(self):
        """Reset frame delay and FPS tracking for all sources."""
        self.consecutive_frame_delay_count = [0] * len(self.data_sources)
        self.fps_frame_count = [0] * len(self.data_sources)
        self.fps_start_time = [time.monotonic()] * len(self.data_sources)

    def _print_info(self, message: str) -> None:
        """Print info message (green) if info_verbose is enabled."""
        if self.info_verbose:
            print_green(message)

    def _print_warning(self, message: str) -> None:
        """Print warning message (yellow) if warning_verbose is enabled."""
        if self.warning_verbose or self.info_verbose:
            print_yellow(message)

    def _print_error(self, message: str) -> None:
        """Print error message (red). Always printed regardless of verbosity."""
        print_red(message)

    @property
    def sources(self) -> List:
        """Alias for data_sources for backward compatibility."""
        return self.data_sources

    @property
    def episode_id(self):
        return self._episode_id

    @property
    def is_recording(self) -> bool:
        """Check if currently recording.

        Returns:
            bool: True if recording is in progress, False otherwise.
        """
        return self._recording_event.is_set()

    def init_episode_id(self):
        """Initialize episode ID based on existing zarr groups."""
        keys = list(self.record_roots.group_keys())

        episode_numbers = []
        for key in keys:
            if key.startswith("episode_"):
                try:
                    episode_num = int(key.split("_")[1])
                    episode_numbers.append(episode_num)
                except (IndexError, ValueError):
                    continue

        return max(episode_numbers) + 1 if episode_numbers else 0

    def set_episode_id(self, episode_id: int) -> None:
        """Set the episode ID."""
        self._episode_id = episode_id

    def get_pending_episode_path(self) -> str:
        """Get the path to the current pending episode folder."""
        return os.path.join(
            self.record_path, f"episode_{self.episode_id}{PENDING_SUFFIX}"
        )

    def get_final_episode_path(self) -> str:
        """Get the path to the final episode folder (without pending suffix)."""
        return os.path.join(self.record_path, f"episode_{self.episode_id}")

    def reset_episode_recording(self) -> bool:
        """Reset and prepare for a new episode recording.

        Creates a pending episode folder (episode_N.pending/) for staging.

        Returns:
            bool: True if reset successful, False otherwise.
        """
        if self._recording_event.is_set():
            # Resetting during recording replaces accumulator references held by worker
            # threads and can lose data. Call only while recording is stopped.
            return False

        os.makedirs(self.get_pending_episode_path(), exist_ok=True)

        # Accumulators are indexed by source and split key. Non-splittable sources
        # have one entry in the inner list.
        self.episode_frames: List[List[dict]] = [
            [
                {field.name: [] for field in fields(self._sub_frame_classes[key])}
                for key in self._split_keys
            ]
            for _ in self.sources
        ]

        return True

    def clear_recording(self) -> bool:
        """Delete the current episode's pending folder.

        Returns:
            bool: True if deletion successful, False otherwise.
        """
        if self._recording_event.is_set():
            self._print_warning(
                "[Recorder] Cannot delete while recording is in progress."
            )
            return False

        pending_path = self.get_pending_episode_path()
        final_path = self.get_final_episode_path()

        try:
            if os.path.exists(pending_path):
                shutil.rmtree(pending_path)
                print(f"Deleted pending episode folder: {pending_path}")
                return True
            elif os.path.exists(final_path):
                shutil.rmtree(final_path)
                print(f"Deleted episode folder: {final_path}")
                return True
        except Exception as e:
            self._print_error(f"[Recorder] Error deleting episode: {e}")
        return False

    def finalize_episode(self) -> bool:
        """Finalize the episode by incrementing the episode ID.

        Note: Folder renaming is handled by RecorderManager.

        Returns:
            bool: True if finalization successful, False otherwise.
        """
        if self._recording_event.is_set():
            self._print_warning(
                "[Recorder] Cannot finalize while recording is in progress."
            )
            return False

        if not os.path.exists(self.get_final_episode_path()):
            self._print_error(
                f"[Recorder] Final episode not found: {self.get_final_episode_path()}"
            )
            return False

        self.set_episode_id(self.episode_id + 1)
        return True

    def discard_staging(self) -> bool:
        """Discard staging state without incrementing episode ID.

        Note: Folder deletion is handled by RecorderManager.

        Returns:
            bool: True if discard successful, False otherwise.
        """
        if self._recording_event.is_set():
            self._print_warning(
                "[Recorder] Cannot discard while recording is in progress."
            )
            return False

        return True

    def _set_warning(self, source_name: str, warning_type: str, message: str) -> None:
        """Register an active warning for a data source (thread-safe)."""
        with self._warnings_lock:
            self._active_warnings[(source_name, warning_type)] = message

    def _clear_warning(self, source_name: str, warning_type: str) -> None:
        """Clear an active warning for a data source (thread-safe)."""
        with self._warnings_lock:
            self._active_warnings.pop((source_name, warning_type), None)

    def get_active_warnings(self) -> dict:
        """Return a snapshot of active warnings as {(source_name, warning_type): message}."""
        with self._warnings_lock:
            return dict(self._active_warnings)

    def _count_frame_drop(self, source_name: str) -> None:
        with self._warnings_lock:
            count, _ = self._frame_drop_stats.get(source_name, (0, None))
            self._frame_drop_stats[source_name] = (count + 1, time.monotonic())

    def get_frame_drop_stats(self) -> dict:
        """Snapshot of this episode's missed frames as {source_name: (count, last_time)}.

        Only sources that missed at least one frame are listed; ``last_time``
        is the ``time.monotonic()`` of the latest miss.
        """
        with self._warnings_lock:
            return dict(self._frame_drop_stats)

    def _get_frame_from_source(self, source, source_idx: int) -> Optional[Any]:
        """Get a frame from the specified source.

        Args:
            source: The data source object.
            source_idx: Index of the data source.

        Returns:
            Frame data or None if no frame is available.
        """
        return source.get_frame()

    def _is_frame_valid(self, frame_data: Any) -> bool:
        """Check if the frame data is valid and should be queued.

        A valid frame must be an instance of the configured frame_data_class.
        This guards against e.g. server error dicts ({"error": "..."}) from
        ZMQ clients leaking into the buffer and crashing the recording thread
        on attribute access (see get_frame on RobotClient/EndEffectorClient).

        Args:
            frame_data: The frame data to validate.

        Returns:
            True if frame should be queued, False otherwise.
        """
        if frame_data is None:
            return False
        if not isinstance(frame_data, self.frame_data_class):
            self._print_warning(
                f"[Recorder] Dropping frame of unexpected type "
                f"{type(frame_data).__name__} "
                f"(expected {self.frame_data_class.__name__})"
            )
            return False
        return True

    def _stream_thread(self, source_idx: int):
        """Thread function to continuously get frames from a source.

        Args:
            source_idx: Index of the data source.
        """
        source = self.sources[source_idx]
        source_name = source.device_name
        buffer = self.frame_buffers[source_idx]

        rate_keeper = RateKeeper(
            target_rate=self.stream_fps,
            verbose=self.verbose,
            name=f"StreamThread_{source_name}",
        )

        while self._streaming_event.is_set():
            with rate_keeper:
                try:
                    frame_data = self._get_frame_from_source(source, source_idx)
                    if self._is_frame_valid(frame_data):
                        buffer.put(frame_data)
                        if self.verbose:
                            print(
                                f"Put frame queued from source {source_name}({source_idx})"
                            )
                    else:
                        if self.verbose:
                            print(
                                f"No valid frame data from source {source_name}({source_idx})"
                            )
                except Exception as e:
                    self._print_error(
                        f"[Recorder] Error capturing {source_name}({source_idx}) frame: {e}"
                    )
                    time.sleep(0.1)

    @abstractmethod
    def _recording_thread(self, source_idx: int, key_idx: int, episode_id: int):
        """Thread function to process and store frames from the queue.

        One thread is spawned per (source, split_key). For non-splittable
        frame classes (`_split_keys == [""]`), there is exactly one thread
        per source - equivalent to the legacy 1D layout.

        Args:
            source_idx: Index of the data source.
            key_idx: Index into ``self._split_keys`` selecting which sub-frame
                this thread writes (`self._split_keys[key_idx]`).
            episode_id: ID of the episode being recorded.
        """
        pass

    def peek_latest_frame(self, source_idx: int) -> Optional[Any]:
        """Peek the latest frame without removing it.

        Args:
            source_idx: Index of the data source.

        Returns:
            The latest frame or None if buffer is empty.
        """
        if 0 <= source_idx < len(self.frame_buffers):
            return self.frame_buffers[source_idx].peek_latest()
        return None

    def get_last_k_frames(
        self, source_idx: int, k: int, remove: bool = False
    ) -> Optional[List]:
        """Get the latest k frames with option to remove oldest k.

        Args:
            source_idx: Index of the data source.
            k: Number of frames to get.
            remove: Whether to remove the oldest k frames from buffer (FIFO order).

        Returns:
            List of frames or None if buffer is empty.
        """
        if 0 <= source_idx < len(self.frame_buffers):
            return self.frame_buffers[source_idx].get_last_k(k, remove)
        return None

    def get_last_k_frames_from_all(self, k: int, remove: bool = False) -> dict:
        """Get the latest k frames from all sources.

        Args:
            k: Number of latest frames to retrieve.
            remove: Whether to remove the oldest k frames from buffers.

        Returns:
            dict: Dictionary with source names as keys and lists of frames as values.
        """
        frames_dict = {}
        for i, source in enumerate(self.data_sources):
            source_name = getattr(source, self._get_source_name_attr(), f"source_{i}")
            frames = self.get_last_k_frames(i, k, remove)
            frames_dict[source_name] = frames
        return frames_dict

    def _get_source_name_attr(self) -> str:
        """Get the attribute name used for source naming.

        Returns:
            str: Attribute name ('device_name').
        """
        return "device_name"

    def get_next_frame(
        self,
        source_idx: int,
        last_timestamp: Optional[float] = None,
        timeout: float = 1.0,
    ) -> Optional[Any]:
        """Get the newest frame that has a receive_time larger than the last timestamp.

        Uses efficient blocking instead of polling for zero CPU usage while waiting.

        Args:
            source_idx: Index of the data source.
            last_timestamp: The timestamp of the last frame that was processed.
            timeout: Maximum time to wait for a newer frame in seconds.

        Returns:
            The newest frame with receive_time > last_timestamp, or None if timeout.
        """
        if not 0 <= source_idx < len(self.frame_buffers):
            return None

        return self.frame_buffers[source_idx].wait_for_new_frame(
            last_timestamp=last_timestamp, timeout=timeout
        )

    def _start_source_streaming(self):
        """Start streaming from all data sources."""
        print("Starting source streaming...")
        for source in self.data_sources:
            if hasattr(source, "start_streaming"):
                source.start_streaming()
        time.sleep(0.5)

    def _stop_source_streaming(self):
        """Stop streaming from all data sources."""
        print("Stopping source streaming...")
        for source in self.data_sources:
            source_name = source.device_name
            try:
                if hasattr(source, "stop_streaming"):
                    source.stop_streaming()
            except Exception as e:
                self._print_error(
                    f"[Recorder] Error stopping source {source_name}: {e}"
                )

    def start_streaming(self) -> bool:
        """Start streaming from all sources.

        Returns:
            bool: True if streaming started successfully, False if already streaming.
        """
        if self._streaming_event.is_set():
            self._print_warning("[Recorder] Streaming is already in progress.")
            return False

        self._streaming_event.set()

        self._start_source_streaming()

        time.sleep(0.1)

        print("Starting streaming threads...")
        self.stream_threads = []
        for i in range(len(self.data_sources)):
            thread = Thread(target=self._stream_thread, args=(i,))
            thread.daemon = True
            thread.start()
            self.stream_threads.append(thread)

        return True

    def start_recording(self, episode_id: Optional[int] = None) -> bool:
        """Start recording from all sources.

        Args:
            episode_id: Optional episode ID to use.

        Returns:
            bool: True if recording started successfully, False otherwise.
        """
        if episode_id is not None:
            self.set_episode_id(episode_id)

        if self._recording_event.is_set():
            self._print_warning("[Recorder] Recording is already in progress.")
            return False

        with self._warnings_lock:
            self._frame_drop_stats.clear()
        self._recording_event.set()

        try:
            print("Starting recording threads...")
            self.recording_threads = []
            # One recording thread per source and split key; non-splittable sources use
            # one thread.
            for i in range(len(self.data_sources)):
                for k_idx in range(len(self._split_keys)):
                    thread = Thread(
                        target=self._recording_thread,
                        args=(i, k_idx, self.episode_id),
                    )
                    thread.daemon = True
                    thread.start()
                    self.recording_threads.append(thread)

            self.reset_delay_and_fps_tracking()

            print("Recording started successfully.")
            return True

        except Exception as e:
            self._print_error(f"[Recorder] Failed to start recording: {e}")
            self.stop_recording()
            return False

    def stop_streaming(self) -> bool:
        """Stop streaming from all sources.

        Returns:
            bool: True if streaming stopped successfully, False if not streaming.
        """
        if not self._streaming_event.is_set():
            self._print_warning("[Recorder] Streaming is not in progress.")
            return False

        print("Stopping streaming...")
        self._streaming_event.clear()

        print("Waiting for threads to finish...")
        for thread in self.stream_threads:
            thread.join(timeout=2.0)

        self._stop_source_streaming()

        print("Streaming stopped successfully.")
        return True

    def stop_recording(self) -> bool:
        """Stop recording.

        Returns:
            bool: True if recording stopped successfully, False if not recording.
        """
        if not self._recording_event.is_set():
            self._print_warning("[Recorder] Recording is not in progress.")
            return False

        print("Stopping recording...")
        self._recording_event.clear()

        print("Waiting for threads to finish...")
        for thread in self.recording_threads:
            thread.join(timeout=2.0)

        return True

    @abstractmethod
    def save_recordings(self):
        """Save all recorded data to disk."""
        pass

    # After this many consecutive frame drops the device is considered dead.
    # At ~33ms timeout per attempt this is roughly 10 seconds of silence.
    FATAL_FRAME_DROP_THRESHOLD = 300

    def get_next_frame_until_valid(
        self,
        source_idx: int,
        last_timestamp: Optional[float] = None,
        timeout: float = 1.0,
        count_drops: bool = True,
    ) -> Optional[Any]:
        """Get the next frame until it is valid or the recording is stopped.
        Print warning if frame drops consecutively for more than 20 times.
        If drops exceed FATAL_FRAME_DROP_THRESHOLD, set a source_lost warning
        and return None so the caller can discard the take.

        Every timed-out wait also counts toward get_frame_drop_stats() unless
        ``count_drops`` is False (the extra per-key threads of a split source,
        which wait on the same buffer and would count each miss again)."""
        consecutive_frame_drop_count = 0
        source_name = self.sources[source_idx].device_name
        while True:
            frame_data = self.get_next_frame(
                source_idx, last_timestamp, timeout=timeout
            )
            if frame_data is not None or not self._recording_event.is_set():
                if frame_data is not None:
                    self._clear_warning(source_name, "frame_drop")
                break
            consecutive_frame_drop_count += 1
            if count_drops:
                self._count_frame_drop(source_name)
            if consecutive_frame_drop_count >= self.FATAL_FRAME_DROP_THRESHOLD:
                self._print_error(
                    f"[{source_name}] SOURCE LOST: {consecutive_frame_drop_count} "
                    f"consecutive frame drops (~{consecutive_frame_drop_count * timeout:.1f}s). "
                    f"Device appears disconnected."
                )
                self._set_warning(
                    source_name,
                    "source_lost",
                    f"No data for ~{consecutive_frame_drop_count * timeout:.0f}s",
                )
                return None
            if consecutive_frame_drop_count > 20:
                self._print_error(
                    f"[{source_name}] Consecutive frame drops: {consecutive_frame_drop_count} data not received."
                )
                self._set_warning(
                    source_name,
                    "frame_drop",
                    f"{consecutive_frame_drop_count} consecutive drops",
                )
        return frame_data

    def frame_delay_detection(
        self,
        source_idx: int,
        frame_data: Any,
        last_timestamp: Optional[float] = None,
        timeout: float = 1.0,
    ) -> None:
        """Detect frame delay and print warning if it occurs consecutively for more than 20 times."""
        # Source-loss handling already reports None results; do not feed them to
        # timestamp-based delay tracking.
        if frame_data is None:
            return
        source_name = self.sources[source_idx].device_name
        if last_timestamp is not None:
            time_delta = frame_data.receive_time - last_timestamp
            if time_delta > timeout:
                self.consecutive_frame_delay_count[source_idx] += 1
                count = self.consecutive_frame_delay_count[source_idx]
                if count > 10:
                    self._print_warning(
                        f"[{source_name}] Consecutive frame delays: "
                        f"{count} data not received on time."
                    )
                    self._set_warning(
                        source_name,
                        "frame_delay",
                        f"{count} consecutive delays",
                    )
            else:
                self.consecutive_frame_delay_count[source_idx] = 0
                self._clear_warning(source_name, "frame_delay")

    def fps_tracking(self, source_idx: int, fps) -> None:
        tracking_interval = 100
        tracking_time_window = tracking_interval / fps
        source_name = self.sources[source_idx].device_name
        self.fps_frame_count[source_idx] += 1
        n_frames = self.fps_frame_count[source_idx]
        elapsed_time = time.monotonic() - self.fps_start_time[source_idx]
        if (n_frames % tracking_interval == 0) or (
            elapsed_time >= tracking_time_window
        ):
            avg_fps = n_frames / elapsed_time
            fps_tolerance = fps * 0.1  # 10% tolerance
            if abs(avg_fps - fps) <= fps_tolerance:
                self._print_info(
                    f"[{source_name}] Avg FPS (last {n_frames} frames): {avg_fps:.2f} (target: {fps})"
                )
            else:
                self._print_warning(
                    f"[{source_name}] Avg FPS (last {n_frames} frames): {avg_fps:.2f} (target: {fps})"
                )
            self.fps_start_time[source_idx] = time.monotonic()
            self.fps_frame_count[source_idx] = 0

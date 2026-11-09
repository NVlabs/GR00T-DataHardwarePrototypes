# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DataBuffer: A flexible data interpolation and alignment system.

This module provides a dynamic data buffer that:
1. Uses camera timestamps as the index for interpolation
2. Applies latency corrections to adjust capture timestamps
3. Finds valid overlapping time ranges across all sources
4. Interpolates data from various sources to align with camera frames

The design is driven by the Frame dataclasses defined in frame.py,
allowing dynamic field discovery and appropriate interpolation strategies.
"""

import logging
import os
import shutil
from dataclasses import fields, is_dataclass
from typing import Callable, Dict, List, Optional, Type, Union

import numpy as np
import yaml
import zarr

from umi.common.data.frame import (
    FRAME_CLASS_MAP,
    Frame,
    NumericFrame,
    RGBCameraFrame,
    RGBDCameraFrame,
    get_field_interpolator,
    get_zarr_storage_config,
    is_interpolatable,
    load_frame_definitions,
)
from umi.common.data.interpolator import LinearInterpolator, PoseInterpolator
from umi.common.data.zarr_utils import create_zarr_array

logger = logging.getLogger(__name__)


def find_nearest_timestamp_indices(
    source_times: np.ndarray, target_times: np.ndarray
) -> np.ndarray:
    """Find indices of source timestamps nearest to each target timestamp.

    Args:
        source_times: Array of source timestamps (must be sorted).
        target_times: Array of target timestamps to match (must be sorted).

    Returns:
        Array of indices into source_times, one per target_time.
    """
    m, n = len(source_times), len(target_times)
    indices = np.empty(n, dtype=int)
    j = 0
    for i in range(n):
        while j < m - 1 and abs(source_times[j + 1] - target_times[i]) < abs(
            source_times[j] - target_times[i]
        ):
            j += 1
        indices[i] = j
    return indices


def get_interpolatable_fields(frame_class: Type[Frame]) -> Dict[str, type]:
    """Get fields from a Frame class that should be interpolated.

    Args:
        frame_class: The Frame dataclass type.

    Returns:
        Dict mapping field names to their types.
    """
    if not is_dataclass(frame_class):
        return {}
    return {
        f.name: f.type
        for f in fields(frame_class)
        if is_interpolatable(frame_class, f.name)
    }


def create_interpolator(
    timestamps: np.ndarray,
    values: np.ndarray,
    field_name: str,
    frame_class: Optional[Type[Frame]] = None,
    custom_interpolators: Optional[Dict[str, Callable]] = None,
) -> Optional[Union[LinearInterpolator, PoseInterpolator]]:
    """Create an interpolator for a field.

    Args:
        timestamps: Array of timestamps.
        values: Array of values to interpolate.
        field_name: Name of the field being interpolated.
        frame_class: Optional Frame class to read interpolator metadata from.
        custom_interpolators: Optional dict mapping field names to custom
                            interpolation functions.

    Returns:
        Interpolation object or None if interpolation not possible.
    """
    if custom_interpolators and field_name in custom_interpolators:
        return custom_interpolators[field_name](timestamps, values)

    InterpolatorClass = LinearInterpolator
    if frame_class is not None:
        InterpolatorClass = (
            get_field_interpolator(frame_class, field_name) or LinearInterpolator
        )

    if InterpolatorClass == PoseInterpolator:
        if values.ndim == 3 and values.shape[1:] == (4, 4):
            return PoseInterpolator(timestamps, values)
        else:
            logger.warning(
                f"Pose field {field_name} has unexpected shape {values.shape}, "
                "using linear interpolation"
            )
            InterpolatorClass = LinearInterpolator

    try:
        return InterpolatorClass(timestamps, values)
    except Exception as e:
        logger.error(f"Failed to create interpolator for {field_name}: {e}")
        return None


class DataSourceConfig:
    """Configuration for a data source with latency and interpolation settings."""

    def __init__(
        self,
        name: str,
        frame_class: Type[Frame] = Frame,
        latency: Optional[float] = None,
        is_index_source: bool = False,
        custom_interpolators: Optional[Dict[str, Callable]] = None,
    ):
        """Initialize data source configuration.

        Args:
            name: Name of the data source (must match the recorded source name).
            frame_class: The Frame dataclass type used for this source.
            latency: Latency in seconds to subtract from receive_time to get capture_time.
                    If None, uses the recorded capture_time directly.
            is_index_source: If True, this source's timestamps are used as the
                           interpolation index (typically a camera source).
            custom_interpolators: Dict mapping field names to custom interpolation
                                 functions. Function signature: (timestamps, values) -> interpolator
        """
        self.name = name
        self.frame_class = frame_class
        self.latency = latency
        self.is_index_source = is_index_source
        self.custom_interpolators = custom_interpolators or {}


class DataBuffer:
    """A flexible data buffer for interpolating and aligning multi-source recordings.

    The buffer uses one source (typically a camera) as the index source, and
    interpolates all other sources to align with the index source's timestamps.

    Key concepts:
    1. Index source: The source whose timestamps are used for alignment (usually camera).
    2. Latency adjustment: capture_time = receive_time - latency
    3. Valid time range: Overlapping time range across all sources after latency adjustment.
    4. Interpolation: Data from non-index sources is interpolated to match index timestamps.
    """

    def __init__(
        self,
        data_dir: str,
        target_dir: str,
        index_source_name: str,
        latencies: Optional[Dict[str, float]] = None,
        downsample_rate: int = 1,
        copy_videos: bool = True,
        use_capture_time: bool = False,
        max_workers: int = 4,
        uniform_fps: Optional[float] = None,
    ) -> None:
        """Initialize the DataBuffer.

        Args:
            data_dir: Path to the source data directory (zarr format).
            target_dir: Path to the target directory for interpolated data.
            index_source_name: Name of the source to use as the interpolation index.
            latencies: Dict mapping source names to latency values in seconds.
                      If a source is not in this dict, uses recorded capture_time.
            downsample_rate: Downsample factor for the index source (1 = no downsampling).
            copy_videos: If True, copy video files to target directory.
            use_capture_time: If True, use capture_time directly instead of requiring
                            latency values. Skips latency validation in config.
            uniform_fps: If set, output timestamps are an even 1/uniform_fps grid
                        inside the index source's valid window (phased to the
                        index frames) instead of the index source's own frame
                        times. Every source, the index source included, is then
                        interpolated (numeric) or mapped to its nearest frame
                        (camera) at the grid times. None or 0 = the index
                        source's own frame times.
        """
        # The station, the health check and align_data.py always pass a rate
        # (umi.common.utility.alignment_settings); None or 0 is library-only.
        uniform_fps = uniform_fps or None
        if uniform_fps is not None and uniform_fps < 0:
            raise ValueError(f"uniform_fps must be positive, got {uniform_fps}")
        if uniform_fps is not None and downsample_rate != 1:
            raise ValueError(
                "uniform_fps sets the output rate; it cannot be combined with "
                f"downsample_rate={downsample_rate}"
            )
        self.data_dir = data_dir
        self.target_dir = target_dir
        self.data_buffer = zarr.open(data_dir, mode="a")
        self.target_buffer = zarr.open(target_dir, mode="a")
        self.latencies = latencies or {}
        self.downsample_rate = downsample_rate
        self.copy_videos = copy_videos
        self.index_source_name = index_source_name
        self.use_capture_time = use_capture_time
        self.max_workers = max_workers
        self.uniform_fps = uniform_fps

        self.source_configs = self._load_source_configs()

        if index_source_name not in self.source_configs:
            raise ValueError(
                f"Index source '{index_source_name}' not found in config. "
                f"Available sources: {list(self.source_configs.keys())}"
            )
        # On a uniform grid the index source only bounds the valid window; its
        # data is resampled at the grid times like every other source's.
        self.source_configs[index_source_name].is_index_source = uniform_fps is None

    def _load_source_configs(self) -> Dict[str, DataSourceConfig]:
        """Load source configurations from the saved config.yaml.

        Returns:
            Dict mapping source names to DataSourceConfig objects.
        """
        config_path = os.path.join(self.data_dir, "config.yaml")

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found at {config_path}. "
                "A config.yaml file is required for data alignment."
            )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        load_frame_definitions(config)

        source_configs = {}

        recording_cfg = config.get("recording", {})
        default_video_frame_type = recording_cfg.get(
            "video_frame_type", "RGBCameraFrame"
        )
        default_numeric_frame_type = recording_cfg.get(
            "numeric_frame_type", "NumericFrame"
        )

        for source in config.get("video_sources", []):
            name = source["name"]
            frame_type_name = source.get("frame_type", default_video_frame_type)
            frame_class = FRAME_CLASS_MAP.get(frame_type_name, RGBCameraFrame)

            # Check if latency is defined in config (skip if use_capture_time is True)
            if not self.use_capture_time and "latency" not in source:
                raise ValueError(
                    f"Latency not defined for video source '{name}' in config.yaml. "
                    "Please add a 'latency' field to the source configuration."
                )

            latency = (
                self.latencies.get(name)
                if self.latencies.get(name) is not None
                else source.get("latency")
            )

            source_configs[name] = DataSourceConfig(
                name=name,
                frame_class=frame_class,
                latency=latency,
            )
            logger.info(f"Loaded video source: {name} (frame_type={frame_type_name})")

        for source in config.get("numeric_sources", []):
            name = source["name"]
            frame_type_name = source.get("frame_type", default_numeric_frame_type)
            frame_class = FRAME_CLASS_MAP.get(frame_type_name, NumericFrame)

            # Check if latency is defined in config (skip if use_capture_time is True)
            if not self.use_capture_time and "latency" not in source:
                raise ValueError(
                    f"Latency not defined for numeric source '{name}' in config.yaml. "
                    "Please add a 'latency' field to the source configuration."
                )

            latency = (
                self.latencies.get(name)
                if self.latencies.get(name) is not None
                else source.get("latency")
            )

            source_configs[name] = DataSourceConfig(
                name=name,
                frame_class=frame_class,
                latency=latency,
            )
            logger.info(f"Loaded numeric source: {name} (frame_type={frame_type_name})")

        return source_configs

    def _update_latencies_from_episode(self, episode_data: zarr.Group) -> None:
        """Update source latencies from episode metadata.

        Reads latency values from the episode's 'latency' attribute and updates
        the source_configs accordingly. Command-line latencies (in self.latencies)
        take precedence over episode metadata.

        Args:
            episode_data: Zarr group containing episode data with latency metadata.
        """
        episode_latencies = episode_data.attrs.get("latency", {})
        if not episode_latencies:
            return

        updated_sources = []
        for source_name, latency_value in episode_latencies.items():
            if source_name not in self.source_configs:
                continue

            # Command-line latencies take precedence
            if self.latencies.get(source_name) is not None:
                continue

            self.source_configs[source_name].latency = latency_value
            updated_sources.append(f"{source_name}={latency_value:.4f}s")

        if updated_sources:
            logger.info(
                f"Loaded latencies from episode metadata: {', '.join(updated_sources)}"
            )

    def get_adjusted_capture_time(
        self, source_data: zarr.Group, source_config: DataSourceConfig
    ) -> np.ndarray:
        """Get capture timestamps with latency adjustment applied.

        Args:
            source_data: Zarr group containing the source data.
            source_config: Configuration for this source.

        Returns:
            Array of adjusted capture timestamps.
        """

        timestamp_field = "capture_time" if self.use_capture_time else "receive_time"
        if timestamp_field not in source_data:
            raise ValueError(
                f"{source_config.name}: no {timestamp_field} timestamps were recorded. "
                "This source has no recorded samples to align. Check its recording "
                "connection and record the episode again."
            )
        timestamps = source_data[timestamp_field][:]
        if timestamps.size == 0:
            raise ValueError(
                f"{source_config.name}: {timestamp_field} timestamps are empty; "
                "there are no recorded samples to align."
            )
        if self.use_capture_time:
            logger.info(
                "Using capture_time directly for %s (use_capture_time=True)",
                source_config.name,
            )
            return timestamps
        else:
            logger.info(
                "Applying latency correction of %.3fs to %s",
                source_config.latency,
                source_config.name,
            )
            return timestamps - source_config.latency

    def get_valid_time_range(
        self, episode_data: zarr.Group, capture_time_cache: Dict[str, np.ndarray]
    ) -> tuple[float, float]:
        """Compute the valid overlapping time range across all sources.

        Args:
            episode_data: Zarr group containing episode data.
            capture_time_cache: Dict mapping source names to adjusted capture times.

        Returns:
            Tuple of (min_time, max_time) representing the valid range.
        """
        time_mins = []
        time_maxs = []

        for source_name, config in self.source_configs.items():
            if source_name not in episode_data:
                logger.warning(f"Source {source_name} not found in episode data")
                continue

            capture_time = capture_time_cache[source_name]
            time_mins.append(capture_time.min())
            time_maxs.append(capture_time.max())
            logger.debug(
                f"source_name: {source_name}, min: {capture_time.min()}, max: {capture_time.max()}"
            )

        min_time = max(time_mins)
        max_time = min(time_maxs)

        logger.info(f"Valid time range: [{min_time:.3f}, {max_time:.3f}]")
        return min_time, max_time

    def _copy_attrs(self, source: zarr.Group, target: zarr.Group) -> None:
        """Copy all attributes from source group to target group.

        Args:
            source: Source zarr group.
            target: Target zarr group.
        """
        for key, value in source.attrs.items():
            target.attrs[key] = value
            logger.debug(f"Copied attribute '{key}' to target")

    # Copy recording/check configs and each source's referenced calibration
    # into aligned session and episode output.
    _FIXED_BUNDLED_CONFIGS = ("config.yaml", "data_check_config.yaml")

    def _bundled_config_basenames(self) -> List[str]:
        """Basenames of every config file the recording bundles with the data.

        ``config.yaml`` and ``data_check_config.yaml`` (fixed names) plus each
        distinct device-calibration file referenced by a source's
        ``params.config_file``. Deduped, order preserved — sub-sources of one
        device share a config_file and collapse to one entry.
        Device files are read from the raw config.yaml, which lists them under
        any source list (``video_sources`` / ``numeric_sources`` / any future
        ``*_sources``).
        """
        names: List[str] = list(self._FIXED_BUNDLED_CONFIGS)
        config_path = os.path.join(self.data_dir, "config.yaml")
        if os.path.isfile(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
            for value in config.values():
                if not isinstance(value, list):
                    continue
                for src in value:
                    if not isinstance(src, dict):
                        continue
                    cfg_file = (src.get("params") or {}).get("config_file")
                    if cfg_file:
                        names.append(os.path.basename(str(cfg_file)))

        seen: set[str] = set()
        unique: List[str] = []
        for name in names:
            if name not in seen:
                seen.add(name)
                unique.append(name)
        return unique

    def _copy_bundled_configs(
        self, src_dir: str, dst_dir: str, basenames: List[str]
    ) -> None:
        """Copy each bundled config file present in *src_dir* into *dst_dir*.

        Called at both levels so the aligned dataset mirrors what the recording
        bundled: the session root (src = raw session, dst = aligned session) and
        every episode (src = raw ``episode_X``, dst = aligned ``episode_X``).
        Offline tools can then read the calibration / data-check config straight
        from the aligned dataset without the repo. Files that were never
        bundled (older data, or no data-check config) are simply skipped.
        """
        if not os.path.isdir(dst_dir):
            return
        for basename in basenames:
            src_path = os.path.join(src_dir, basename)
            if not os.path.isfile(src_path):
                continue
            shutil.copy2(src_path, os.path.join(dst_dir, basename))
            logger.info("Bundled %s into %s", basename, dst_dir)

    def align_episode(self, episode_indices: Union[int, List[int]]) -> None:
        """Align data for specified episodes.

        Args:
            episode_indices: Single episode index or list of indices.
        """
        if isinstance(episode_indices, int):
            episode_indices = [episode_indices]

        self._copy_attrs(self.data_buffer, self.target_buffer)
        if self.data_buffer.attrs:
            logger.info(f"Copied {len(self.data_buffer.attrs)} root-level attributes")

        # Keep aligned output self-contained with session config and calibration
        # snapshots.
        bundled_configs = self._bundled_config_basenames()
        self._copy_bundled_configs(self.data_dir, self.target_dir, bundled_configs)

        for episode_index in episode_indices:
            logger.info("\n")
            logger.info(
                "╔══════════════════════════════════════════════════════════════╗"
            )
            logger.info(f"║  Processing episodes: {episode_index!s:<40} ║")
            logger.info(
                "╚══════════════════════════════════════════════════════════════╝"
            )
            episode_key = f"episode_{episode_index}"

            if episode_key not in self.data_buffer:
                logger.warning(f"Episode {episode_index} not found, skipping")
                continue

            episode_data = self.data_buffer[episode_key]

            # Delete existing episode data before rewriting arrays to prevent stale
            # chunks surviving a shape change.
            if episode_key in self.target_buffer:
                logger.info(
                    f"Episode {episode_index}: overwriting existing aligned data"
                )
                del self.target_buffer[episode_key]

            target_episode_data = self.target_buffer.require_group(episode_key)

            self._update_latencies_from_episode(episode_data)

            self._copy_attrs(episode_data, target_episode_data)
            if episode_data.attrs:
                logger.info(
                    f"Episode {episode_index}: Copied {len(episode_data.attrs)} attributes"
                )

            self._align_episode(episode_index, episode_data, target_episode_data)

            # Preserve per-episode config and calibration snapshots in aligned output.
            self._copy_bundled_configs(
                os.path.join(self.data_dir, episode_key),
                os.path.join(self.target_dir, episode_key),
                bundled_configs,
            )

            if self.copy_videos:
                self._copy_videos(episode_index)

    def _interpolate_numerical_fields(
        self,
        source_name: str,
        source_config: DataSourceConfig,
        source_data: zarr.Group,
        source_capture_time: np.ndarray,
        target_source_group: zarr.Group,
        valid_indices: np.ndarray,
        valid_timestamps: np.ndarray,
    ) -> None:
        """Interpolate numerical fields for a single source.

        Args:
            source_name: Name of the source.
            source_config: Configuration for this source.
            source_data: Source data group.
            source_capture_time: Capture timestamps for this source.
            target_source_group: Target group to store interpolated data.
            valid_indices: Boolean mask of valid frames for index source.
            valid_timestamps: Target timestamps to interpolate to.
        """
        interp_fields = get_interpolatable_fields(source_config.frame_class)

        if not interp_fields:
            return

        interpolated_fields = []
        for field_name in interp_fields:
            if field_name not in source_data:
                logger.debug(f"Field {field_name} not found in {source_name}, skipping")
                continue

            values = source_data[field_name][:]

            if source_config.is_index_source:
                interp_values = values[valid_indices]
            else:
                interpolator = create_interpolator(
                    source_capture_time,
                    values,
                    field_name,
                    frame_class=source_config.frame_class,
                    custom_interpolators=source_config.custom_interpolators,
                )

                if interpolator is None:
                    logger.warning(
                        f"Could not create interpolator for {source_name}/{field_name}"
                    )
                    continue

                try:
                    interp_values = interpolator(valid_timestamps)
                except Exception as e:
                    logger.error(
                        f"Interpolation failed for {source_name}/{field_name}: {e}"
                    )
                    continue

            # Materialize zero-valued fields so existence checks can distinguish valid
            # zero readings from missing data.
            interp_key = f"{field_name}_interp"
            create_zarr_array(target_source_group, interp_key, interp_values)
            interpolated_fields.append(field_name)

        if interpolated_fields:
            logger.info(
                f"{source_name}: {len(valid_timestamps)} aligned frames "
                f"(interpolated {len(interpolated_fields)} fields from {len(source_capture_time)} total frames)"
            )

    def _compute_camera_indices(
        self,
        source_name: str,
        source_config: DataSourceConfig,
        source_capture_time: np.ndarray,
        target_source_group: zarr.Group,
        valid_indices: np.ndarray,
        valid_timestamps: np.ndarray,
    ) -> None:
        """Compute and store aligned frame indices for a camera source.

        For the index source, stores indices of frames that fall within the valid time range.
        For non-index sources, stores nearest frame indices matched to the target timestamps.

        Args:
            source_name: Name of the camera source.
            source_config: Configuration for this source.
            source_capture_time: Capture timestamps for this source.
            target_source_group: Target group to store frame indices.
            valid_indices: Boolean mask of valid frames for index source.
            valid_timestamps: Target timestamps to align to.
        """
        if source_config.frame_class not in [RGBCameraFrame, RGBDCameraFrame]:
            return

        if source_config.is_index_source:
            frame_indices = np.where(valid_indices)[0]
            logger.info(
                f"{source_name}: {len(frame_indices)} aligned frames "
                f"(filtered from {len(source_capture_time)} total)"
            )
        else:
            frame_indices = find_nearest_timestamp_indices(
                source_capture_time, valid_timestamps
            )
            logger.info(
                f"{source_name}: {len(frame_indices)} aligned frames "
                f"(matched from {len(source_capture_time)} total)"
            )

        create_zarr_array(target_source_group, "aligned_frame_indices", frame_indices)

    def _align_episode(
        self,
        episode: int,
        episode_data: zarr.Group,
        target_episode_data: zarr.Group,
    ) -> None:
        """Align and process data for a single episode.

        This orchestrates both numerical interpolation and camera frame alignment.

        Args:
            episode: Episode index.
            episode_data: Source episode data.
            target_episode_data: Target episode data group.
        """
        # Cache latency-adjusted timestamps and log each source once.
        capture_time_cache = {}
        for source_name, config in self.source_configs.items():
            if source_name not in episode_data:
                continue
            source_data = episode_data[source_name]
            capture_time = self.get_adjusted_capture_time(source_data, config)
            capture_time_cache[source_name] = capture_time

            if config.latency is not None:
                logger.debug(
                    f"Applying latency correction of {config.latency:.3f}s to {source_name}"
                )

        index_capture_time = capture_time_cache[self.index_source_name]

        min_time, max_time = self.get_valid_time_range(episode_data, capture_time_cache)

        valid_time_mask = (index_capture_time >= min_time) & (
            index_capture_time <= max_time
        )

        downsample_mask = np.zeros_like(valid_time_mask, dtype=bool)
        downsample_mask[:: self.downsample_rate] = True
        valid_indices = valid_time_mask & downsample_mask

        valid_timestamps = index_capture_time[valid_indices]

        # Newest-frame sampling gives uneven timestamps when stream and recording
        # rates differ. Resample onto a uniform grid inside the valid window. Use
        # the circular mean of timestamp phases so grid alignment reflects mean
        # frame age rather than the age of the first sample.
        if self.uniform_fps is not None and len(valid_timestamps) > 0:
            t_first, t_last = valid_timestamps[0], valid_timestamps[-1]
            period = 1.0 / self.uniform_fps
            phase = np.angle(
                np.mean(np.exp(2j * np.pi * (valid_timestamps - t_first) / period))
            )
            t_start = t_first + (phase / (2 * np.pi) * period) % period
            n_grid = int((t_last - t_start) * self.uniform_fps) + 1
            grid = t_start + np.arange(n_grid) * period
            grid = grid[grid <= t_last]
            # A window shorter than the phase offset holds no grid point; keep
            # its first index frame so the episode still has a row.
            valid_timestamps = grid if len(grid) else valid_timestamps[:1]
            target_episode_data.attrs["uniform_fps"] = self.uniform_fps

        logger.info(
            f"Episode {episode}: {len(index_capture_time)} frames -> "
            f"{len(valid_timestamps)} after filtering and downsampling"
        )

        # Materialize valid_indices even when every value is False.
        create_zarr_array(target_episode_data, "valid_indices", valid_indices)
        create_zarr_array(target_episode_data, "valid_timestamps", valid_timestamps)

        for source_name, source_config in self.source_configs.items():
            if source_name not in episode_data:
                logger.warning(f"Source {source_name} not found in episode {episode}")
                continue

            source_data = episode_data[source_name]
            source_capture_time = capture_time_cache[source_name]

            target_source_group = target_episode_data.require_group(source_name)

            self._copy_attrs(source_data, target_source_group)

            for array_name in source_data.array_keys():
                data = source_data[array_name][:]
                storage_config = get_zarr_storage_config(
                    source_config.frame_class, array_name
                )
                create_zarr_array(
                    target_source_group,
                    array_name,
                    data,
                    storage_config=storage_config,
                    max_workers=self.max_workers,
                )
                logger.debug(
                    f"Copied {source_name}/{array_name} with shape {data.shape}"
                )

            self._interpolate_numerical_fields(
                source_name,
                source_config,
                source_data,
                source_capture_time,
                target_source_group,
                valid_indices,
                valid_timestamps,
            )

            self._compute_camera_indices(
                source_name,
                source_config,
                source_capture_time,
                target_source_group,
                valid_indices,
                valid_timestamps,
            )

    def _copy_videos(self, episode_index: int) -> None:
        """Copy video files for an episode to the target directory.

        Args:
            episode_index: Episode index.
        """
        source_episode_dir = os.path.join(self.data_dir, f"episode_{episode_index}")
        target_episode_dir = os.path.join(self.target_dir, f"episode_{episode_index}")

        os.makedirs(target_episode_dir, exist_ok=True)

        if os.path.exists(source_episode_dir):
            for filename in os.listdir(source_episode_dir):
                if filename.endswith((".mp4", ".avi", ".mov")):
                    source_path = os.path.join(source_episode_dir, filename)
                    target_path = os.path.join(target_episode_dir, filename)
                    if os.path.exists(source_path):
                        shutil.copy2(source_path, target_path)
                        logger.info(f"Copied {filename} to target directory")

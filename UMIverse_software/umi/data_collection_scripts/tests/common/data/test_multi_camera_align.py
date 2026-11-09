# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-camera alignment tests for DataBuffer.

DataBuffer aligns sources to the index source's timeline. The time used for
alignment is adjusted_capture_time = receive_time - latency (see DataBuffer
get_adjusted_capture_time). With latency=0 this is receive_time.

These tests set latency=0 for all sources, so:
  - For video: alignment time = receive_time = capture_time + receive_time_offset_s
  - For numeric: receive_time = capture_time (no offset), so alignment time = capture_time

Valid time range = [max(all source mins), min(all source maxs)] (intersection).
Only index-source frames whose adjusted time lies in this range are kept; non-index
sources get nearest-neighbor frame indices to those valid timestamps.
"""

import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest
import yaml
import zarr

from umi.common.data.data_buffer import DataBuffer

# =============================================================================
# Helpers: test data creation (offsets explicit; alignment uses receive_time - latency)
# =============================================================================


def create_test_environment(tmpdir: str) -> Tuple[Path, Path]:
    """Create temporary directories for test data and output."""
    data_dir = Path(tmpdir) / "test_data"
    output_dir = Path(tmpdir) / "test_output"
    data_dir.mkdir()
    output_dir.mkdir()
    return data_dir, output_dir


def create_camera_data(
    episode: zarr.Group,
    name: str,
    capture_times: np.ndarray,
    *,
    receive_time_offset_s: float = 0.0,
    resolution: Tuple[int, int] = (480, 640),
) -> None:
    """Create camera data in a zarr episode group.

    Alignment time (with latency=0) = receive_time = capture_times + receive_time_offset_s.
    """
    cam = episode.create_group(name)
    cam["capture_time"] = capture_times
    cam["receive_time"] = capture_times + receive_time_offset_s
    cam["rgb"] = np.random.randint(
        0, 255, (len(capture_times), resolution[0], resolution[1], 3), dtype=np.uint8
    )


def create_numeric_data(
    episode: zarr.Group, name: str, times: np.ndarray, dim: int = 7
) -> None:
    """Create numeric sensor data in a zarr episode group.

    receive_time is set to `times`, so alignment time (with latency=0) = times.
    """
    sensor = episode.create_group(name)
    sensor["capture_time"] = times
    sensor["receive_time"] = times
    sensor["value"] = np.random.randn(len(times), dim)


def create_config_file(
    data_dir: Path, video_sources: list, numeric_sources: list
) -> None:
    """Create config.yaml. DataBuffer requires latency per source; the tests use 0."""
    video_sources_with_latency = [
        {**vs, "latency": vs.get("latency", 0.0)} for vs in video_sources
    ]
    numeric_sources_with_latency = [
        {**ns, "latency": ns.get("latency", 0.0)} for ns in numeric_sources
    ]
    config = {
        "recording": {
            "video_frame_type": "RGBCameraFrame",
            "numeric_frame_type": "NumericFrame",
        },
        "video_sources": video_sources_with_latency,
        "numeric_sources": numeric_sources_with_latency,
    }
    with open(data_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)


def run_alignment(
    data_dir: Path, output_dir: Path, index_source: str = "main_camera"
) -> zarr.Group:
    """Run DataBuffer alignment and return the output episode."""
    buffer = DataBuffer(
        data_dir=str(data_dir),
        target_dir=str(output_dir),
        index_source_name=index_source,
        downsample_rate=1,
        copy_videos=False,
    )
    buffer.align_episode(0)
    output_store = zarr.open(str(output_dir), mode="r")
    return output_store["episode_0"]


@pytest.mark.parametrize("use_capture_time", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_a_camera_with_no_timestamps_names_the_failed_source(
    tmp_path, use_capture_time, empty
):
    data_dir, output_dir = create_test_environment(str(tmp_path))
    store = zarr.open(str(data_dir), mode="w")
    episode = store.create_group("episode_0")
    source = episode.create_group("left_trigger_camera")
    field = "capture_time" if use_capture_time else "receive_time"
    if empty:
        source[field] = np.array([], dtype=float)
    create_config_file(data_dir, [{"name": "left_trigger_camera"}], [])
    buffer = DataBuffer(
        data_dir=str(data_dir),
        target_dir=str(output_dir),
        index_source_name="left_trigger_camera",
        copy_videos=False,
        use_capture_time=use_capture_time,
    )
    with pytest.raises(ValueError, match=f"left_trigger_camera: .*{field}"):
        buffer.align_episode(0)


# =============================================================================
# Expected indices: computed from the spec only (no reuse of DataBuffer internals)
# =============================================================================


def _nearest_timestamp_indices_from_spec(
    source_times: np.ndarray, target_times: np.ndarray
) -> np.ndarray:
    """Per the alignment spec: for each target time, index of source time that is nearest.

    Implemented independently so this tests DataBuffer's find_nearest behavior, not echo it.
    """
    indices = np.empty(len(target_times), dtype=int)
    for i, t in enumerate(target_times):
        indices[i] = int(np.argmin(np.abs(source_times - t)))
    return indices


def compute_expected_aligned_indices(
    adjusted_times_by_source: Dict[str, np.ndarray],
    index_source_name: str,
) -> Dict[str, np.ndarray]:
    """Compute expected aligned_frame_indices from the alignment spec only.

    Spec (no use of DataBuffer code):
    - Valid time range = [max(mins), min(maxs)] over all sources' adjusted times.
    - Index source: indices i where adjusted_time[i] is in [min_time, max_time].
    - Non-index (video) sources: for each valid timestamp, index of source frame whose
      adjusted time is nearest (by absolute difference).

    This tests the full alignment logic including find_nearest_timestamp_indices.
    """
    min_time = max(adj.min() for adj in adjusted_times_by_source.values())
    max_time = min(adj.max() for adj in adjusted_times_by_source.values())

    index_times = adjusted_times_by_source[index_source_name]
    valid_mask = (index_times >= min_time) & (index_times <= max_time)
    valid_timestamps = index_times[valid_mask]

    expected = {}
    expected[index_source_name] = np.where(valid_mask)[0]

    for name, source_times in adjusted_times_by_source.items():
        if name == index_source_name:
            continue
        expected[name] = _nearest_timestamp_indices_from_spec(
            source_times, valid_timestamps
        )

    return expected


def verify_aligned_indices(
    output_episode: zarr.Group,
    expected_indices: Dict[str, np.ndarray],
    video_source_names: list,
) -> None:
    """Assert aligned_frame_indices match expected for each video source."""
    for name in video_source_names:
        actual = output_episode[name]["aligned_frame_indices"][:]
        expected = expected_indices[name]
        assert np.array_equal(actual, expected), (
            f"{name}: expected {expected}, got {actual}"
        )


# =============================================================================
# Test cases (explicit offsets; expected indices computed from alignment rules)
# =============================================================================


def test_case_1_clipping_when_cameras_not_fully_covered():
    """Frames are clipped when not all cameras cover the full time range."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir, output_dir = create_test_environment(tmpdir)
        store = zarr.open(str(data_dir), mode="w")
        episode = store.create_group("episode_0")

        # Main: capture [0, 0.1, ..., 0.5]. Alignment time = capture + offset.
        main_capture = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        main_offset_s = 0.005
        create_camera_data(
            episode, "main_camera", main_capture, receive_time_offset_s=main_offset_s
        )

        # Wrist: partial coverage. Alignment time = capture + offset.
        wrist_capture = np.array([0.02, 0.12, 0.19, 0.31, 0.43, 0.48])
        wrist_offset_s = 0.008
        create_camera_data(
            episode, "wrist_camera", wrist_capture, receive_time_offset_s=wrist_offset_s
        )

        # Numeric: alignment time = times (receive_time = times).
        create_numeric_data(episode, "robot_state", main_capture)

        create_config_file(
            data_dir,
            [
                {"name": "main_camera", "frame_type": "RGBCameraFrame"},
                {"name": "wrist_camera", "frame_type": "RGBCameraFrame"},
            ],
            [{"name": "robot_state", "frame_type": "NumericFrame"}],
        )

        # Alignment uses receive_time (latency=0): main_capture+0.005, wrist_capture+0.008, main_capture.
        adjusted = {
            "main_camera": main_capture + main_offset_s,
            "wrist_camera": wrist_capture + wrist_offset_s,
            "robot_state": main_capture,
        }
        expected_indices = compute_expected_aligned_indices(adjusted, "main_camera")

        output_episode = run_alignment(data_dir, output_dir)
        verify_aligned_indices(
            output_episode, expected_indices, ["main_camera", "wrist_camera"]
        )


def test_case_2_edge_cases_and_midpoints():
    """Edge cases with timestamps at midpoints between main frames."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir, output_dir = create_test_environment(tmpdir)
        store = zarr.open(str(data_dir), mode="w")
        episode = store.create_group("episode_0")

        main_capture = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
        main_offset_s = 0.003
        create_camera_data(
            episode, "main_camera", main_capture, receive_time_offset_s=main_offset_s
        )

        wrist_capture = np.array([0.05, 0.1000000001, 0.30, 0.51, 0.89])
        wrist_offset_s = 0.006
        create_camera_data(
            episode, "wrist_camera", wrist_capture, receive_time_offset_s=wrist_offset_s
        )

        create_numeric_data(episode, "robot_state", main_capture)

        create_config_file(
            data_dir,
            [
                {"name": "main_camera", "frame_type": "RGBCameraFrame"},
                {"name": "wrist_camera", "frame_type": "RGBCameraFrame"},
            ],
            [{"name": "robot_state", "frame_type": "NumericFrame"}],
        )

        adjusted = {
            "main_camera": main_capture + main_offset_s,
            "wrist_camera": wrist_capture + wrist_offset_s,
            "robot_state": main_capture,
        }
        expected_indices = compute_expected_aligned_indices(adjusted, "main_camera")

        output_episode = run_alignment(data_dir, output_dir)
        verify_aligned_indices(
            output_episode, expected_indices, ["main_camera", "wrist_camera"]
        )


def test_case_3_multiple_frames_to_same_main():
    """Multiple secondary frames can map to the same main frame (nearest-neighbor)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir, output_dir = create_test_environment(tmpdir)
        store = zarr.open(str(data_dir), mode="w")
        episode = store.create_group("episode_0")

        main_capture = np.array([0.0, 0.5, 1.0, 1.5])
        main_offset_s = 0.002
        create_camera_data(
            episode, "main_camera", main_capture, receive_time_offset_s=main_offset_s
        )

        wrist_capture = np.array([0.05, 0.15, 0.48, 0.62, 0.73, 0.98, 1.12, 1.52])
        wrist_offset_s = 0.004
        create_camera_data(
            episode, "wrist_camera", wrist_capture, receive_time_offset_s=wrist_offset_s
        )

        create_numeric_data(episode, "robot_state", main_capture)

        create_config_file(
            data_dir,
            [
                {"name": "main_camera", "frame_type": "RGBCameraFrame"},
                {"name": "wrist_camera", "frame_type": "RGBCameraFrame"},
            ],
            [{"name": "robot_state", "frame_type": "NumericFrame"}],
        )

        adjusted = {
            "main_camera": main_capture + main_offset_s,
            "wrist_camera": wrist_capture + wrist_offset_s,
            "robot_state": main_capture,
        }
        expected_indices = compute_expected_aligned_indices(adjusted, "main_camera")

        output_episode = run_alignment(data_dir, output_dir)
        verify_aligned_indices(
            output_episode, expected_indices, ["main_camera", "wrist_camera"]
        )


def test_case_4_same_frame_to_multiple_main():
    """Cameras with different lengths; one secondary frame can be nearest to multiple main frames."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir, output_dir = create_test_environment(tmpdir)
        store = zarr.open(str(data_dir), mode="w")
        episode = store.create_group("episode_0")

        main_capture = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        main_offset_s = 0.003
        create_camera_data(
            episode, "main_camera", main_capture, receive_time_offset_s=main_offset_s
        )

        wrist_capture = np.array([0.08, 0.24, 0.40, 0.42, 0.55, 0.7])
        wrist_offset_s = 0.004
        create_camera_data(
            episode, "wrist_camera", wrist_capture, receive_time_offset_s=wrist_offset_s
        )

        create_numeric_data(episode, "robot_state", main_capture)

        create_config_file(
            data_dir,
            [
                {"name": "main_camera", "frame_type": "RGBCameraFrame"},
                {"name": "wrist_camera", "frame_type": "RGBCameraFrame"},
            ],
            [{"name": "robot_state", "frame_type": "NumericFrame"}],
        )

        adjusted = {
            "main_camera": main_capture + main_offset_s,
            "wrist_camera": wrist_capture + wrist_offset_s,
            "robot_state": main_capture,
        }
        expected_indices = compute_expected_aligned_indices(adjusted, "main_camera")

        output_episode = run_alignment(data_dir, output_dir)
        verify_aligned_indices(
            output_episode, expected_indices, ["main_camera", "wrist_camera"]
        )


def test_case_5_four_cameras():
    """Four cameras with different frame rates and partial coverage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir, output_dir = create_test_environment(tmpdir)
        store = zarr.open(str(data_dir), mode="w")
        episode = store.create_group("episode_0")

        main_capture = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        main_offset_s = 0.002
        create_camera_data(
            episode, "main_camera", main_capture, receive_time_offset_s=main_offset_s
        )

        wrist_capture = np.array([0.02, 0.12, 0.19, 0.31, 0.43, 0.48])
        wrist_offset_s = 0.003
        create_camera_data(
            episode, "wrist_camera", wrist_capture, receive_time_offset_s=wrist_offset_s
        )

        side_capture = np.array([0.05, 0.22, 0.38, 0.40, 0.52])
        side_offset_s = 0.004
        create_camera_data(
            episode, "side_camera", side_capture, receive_time_offset_s=side_offset_s
        )

        overhead_capture = np.array([0.03, 0.23, 0.42])
        overhead_offset_s = 0.005
        create_camera_data(
            episode,
            "overhead_camera",
            overhead_capture,
            receive_time_offset_s=overhead_offset_s,
        )

        create_numeric_data(episode, "robot_state", main_capture)

        create_config_file(
            data_dir,
            [
                {"name": "main_camera", "frame_type": "RGBCameraFrame"},
                {"name": "wrist_camera", "frame_type": "RGBCameraFrame"},
                {"name": "side_camera", "frame_type": "RGBCameraFrame"},
                {"name": "overhead_camera", "frame_type": "RGBCameraFrame"},
            ],
            [{"name": "robot_state", "frame_type": "NumericFrame"}],
        )

        adjusted = {
            "main_camera": main_capture + main_offset_s,
            "wrist_camera": wrist_capture + wrist_offset_s,
            "side_camera": side_capture + side_offset_s,
            "overhead_camera": overhead_capture + overhead_offset_s,
            "robot_state": main_capture,
        }
        expected_indices = compute_expected_aligned_indices(adjusted, "main_camera")

        output_episode = run_alignment(data_dir, output_dir)
        verify_aligned_indices(
            output_episode,
            expected_indices,
            ["main_camera", "wrist_camera", "side_camera", "overhead_camera"],
        )


# =============================================================================
# Device-config bundling: aligned data must carry its own calibration
# =============================================================================


def test_alignment_bundles_session_and_episode_configs():
    """align_episode mirrors every bundled config into the aligned dataset.

    The recording bundles config.yaml, data_check_config.yaml and each device
    calibration (``params.config_file``) at the session root and inside every
    episode. Alignment must copy all of them to both levels so the aligned
    dataset is self-contained — offline processing can read them without the
    repository. Sub-sources sharing one config_file collapse to a single copy.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir, output_dir = create_test_environment(tmpdir)
        store = zarr.open(str(data_dir), mode="w")
        episode = store.create_group("episode_0")

        times = np.array([0.0, 0.1, 0.2, 0.3])
        create_camera_data(episode, "main_camera", times)
        create_numeric_data(episode, "gripper_encoder", times)
        create_numeric_data(episode, "gripper_force", times)

        # Two numeric sub-sources share one device config_file (repo-relative).
        config = {
            "recording": {
                "video_frame_type": "RGBCameraFrame",
                "numeric_frame_type": "NumericFrame",
            },
            "video_sources": [
                {"name": "main_camera", "frame_type": "RGBCameraFrame", "latency": 0.0}
            ],
            "numeric_sources": [
                {
                    "name": "gripper_encoder",
                    "frame_type": "NumericFrame",
                    "latency": 0.0,
                    "params": {"config_file": "config/umi_gripper/gripper_0002.yaml"},
                },
                {
                    "name": "gripper_force",
                    "frame_type": "NumericFrame",
                    "latency": 0.0,
                    "params": {"config_file": "config/umi_gripper/gripper_0002.yaml"},
                },
            ],
        }
        with open(data_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)

        bundled_names = ("config.yaml", "data_check_config.yaml", "gripper_0002.yaml")
        # Session-root bundles (device config by basename + data-check config).
        (data_dir / "gripper_0002.yaml").write_text("mapping: {0: [1]}\n")
        (data_dir / "data_check_config.yaml").write_text("threshold: 1\n")
        # Per-episode snapshots inside the raw episode dir (config.yaml differs
        # from the session one to prove the episode snapshot is the one copied).
        raw_ep = data_dir / "episode_0"
        (raw_ep / "config.yaml").write_text("episode: snapshot\n")
        (raw_ep / "data_check_config.yaml").write_text("threshold: 1\n")
        (raw_ep / "gripper_0002.yaml").write_text("mapping: {0: [1]}\n")

        run_alignment(data_dir, output_dir, index_source="main_camera")

        # Session level: all three bundled configs present.
        for name in bundled_names:
            assert (output_dir / name).is_file(), f"session-level {name} missing"
        assert (output_dir / "gripper_0002.yaml").read_text() == "mapping: {0: [1]}\n"

        # Episode level: same set mirrored into the aligned episode dir.
        aligned_ep = output_dir / "episode_0"
        for name in bundled_names:
            assert (aligned_ep / name).is_file(), f"episode-level {name} missing"
        # The per-episode config.yaml snapshot is copied, not the session config.
        assert (aligned_ep / "config.yaml").read_text() == "episode: snapshot\n"


def test_alignment_tolerates_unbundled_device_config():
    """A config_file that was never bundled (older data) warns but doesn't fail.

    Offline conversion raises later if it actually needs the missing config; the
    alignment step itself stays best-effort.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir, output_dir = create_test_environment(tmpdir)
        store = zarr.open(str(data_dir), mode="w")
        episode = store.create_group("episode_0")

        times = np.array([0.0, 0.1, 0.2, 0.3])
        create_camera_data(episode, "main_camera", times)
        create_numeric_data(episode, "gripper_encoder", times)

        config = {
            "recording": {
                "video_frame_type": "RGBCameraFrame",
                "numeric_frame_type": "NumericFrame",
            },
            "video_sources": [
                {"name": "main_camera", "frame_type": "RGBCameraFrame", "latency": 0.0}
            ],
            "numeric_sources": [
                {
                    "name": "gripper_encoder",
                    "frame_type": "NumericFrame",
                    "latency": 0.0,
                    "params": {"config_file": "config/umi_gripper/gripper_0002.yaml"},
                }
            ],
        }
        with open(data_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
        # Intentionally do not bundle gripper_0002.yaml at the raw session root.

        # Must not raise; aligned dir simply lacks the (never-bundled) config.
        run_alignment(data_dir, output_dir, index_source="main_camera")
        assert not (output_dir / "gripper_0002.yaml").exists()


# =============================================================================
# Main (for running as script)
# =============================================================================

if __name__ == "__main__":
    test_case_1_clipping_when_cameras_not_fully_covered()
    test_case_2_edge_cases_and_midpoints()
    test_case_3_multiple_frames_to_same_main()
    test_case_4_same_frame_to_multiple_main()
    test_case_5_four_cameras()
    test_alignment_bundles_session_and_episode_configs()
    test_alignment_tolerates_unbundled_device_config()
    print("All test cases passed.")

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for data_check.py orchestrator functions (check_episode, check_dataset).

These tests verify that the orchestration pipeline correctly reads zarr stores,
discovers devices, calls individual checks, and assembles reports. Individual
check logic is tested in test_data_check.py.

Uses configs/test_data_check_recording_config.yaml (recording config) and
configs/test_data_check_config.yaml (check config), both beside this file.

Run with: uv run pytest tests/data_processing/test_data_check_integration.py -v
"""

import copy
import os
from pathlib import Path

import numpy as np
import yaml
import zarr

from real_scripts.data_processing.data_check import (
    check_dataset,
    check_episode,
    get_device_expected_fps,
    load_check_config,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TEST_DIR = Path(__file__).resolve().parent
RECORDING_CONFIG_PATH = str(
    TEST_DIR / "configs" / "test_data_check_recording_config.yaml"
)
CHECK_CONFIG_PATH = str(TEST_DIR / "configs" / "test_data_check_config.yaml")


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _get_device_latencies(config):
    """Build device name -> latency map from recording config."""
    return {
        src["name"]: src.get("latency", 0.0)
        for src in config.get("video_sources", []) + config.get("numeric_sources", [])
        if src.get("name")
    }


RECORDING_CONFIG = _load_yaml(RECORDING_CONFIG_PATH)
CHECK_CONFIG = load_check_config(CHECK_CONFIG_PATH)
DEVICE_FPS = get_device_expected_fps(RECORDING_CONFIG)
DEVICE_LATENCY = _get_device_latencies(RECORDING_CONFIG)
DEVICE_NAMES = list(DEVICE_FPS.keys())


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_timestamps(fps, duration_s, start=0.0):
    """Generate monotonic timestamps at the given FPS."""
    n_frames = int(fps * duration_s)
    return start + np.arange(n_frames) / fps


def _make_tracking_poses(n_frames):
    """Generate 4x4 pose matrices with gradual movement."""
    poses = np.zeros((n_frames, 4, 4))
    for i in range(n_frames):
        poses[i] = np.eye(4)
        poses[i, 0, 3] = i * 0.001
        poses[i, 1, 3] = i * 0.0005
    return poses


def _make_gripper_distance(n_frames):
    """Generate gripper distance values with movement within valid range."""
    base = 60.0
    movement = 20.0 * np.sin(np.linspace(0, 4 * np.pi, n_frames))
    return base + movement


def _build_episode(store, episode_key, duration_s=15.0, defect=None):
    """Build a single synthetic episode in a zarr store.

    Uses per-device FPS from the recording config. Each device gets both
    capture_time and receive_time (receive_time = capture_time + small latency).

    Args:
        store: zarr group (root of the store)
        episode_key: e.g. "episode_0"
        duration_s: episode length in seconds
        defect: optional string describing what to break:
            - "static_gripper": gripper distance is constant
            - "bad_fps": timestamps at wrong FPS (applied to all devices)
    """
    ep = store.create_group(episode_key)

    latencies = {}
    for device_name in DEVICE_NAMES:
        dev = ep.create_group(device_name)
        device_fps = DEVICE_FPS[device_name]
        actual_fps = 20.0 if defect == "bad_fps" else device_fps
        timestamps = _make_timestamps(actual_fps, duration_s)
        latency = DEVICE_LATENCY.get(device_name, 0.0)
        dev.create_array("capture_time", data=timestamps)
        dev.create_array("receive_time", data=timestamps + latency)

        if "tracking" in device_name:
            poses = _make_tracking_poses(len(timestamps))
            dev.create_array("pose", data=poses)

        if "gripper" in device_name:
            if defect == "static_gripper":
                distance = np.full(len(timestamps), 60.0)
            else:
                distance = _make_gripper_distance(len(timestamps))
            dev.create_array("distance", data=distance)

        latency = DEVICE_LATENCY.get(device_name, 0.0)
        latencies[device_name] = latency

    ep.attrs["latency"] = latencies


def _build_dataset(path, num_episodes=2, defects=None):
    """Build a complete synthetic dataset directory.

    Args:
        path: directory path for the dataset
        num_episodes: number of episodes to create
        defects: optional dict mapping episode index to defect string
    """
    os.makedirs(path, exist_ok=True)

    # zarr.open with mode="w" wipes the directory, so write config.yaml after
    store = zarr.open(str(path), mode="w")
    defects = defects or {}

    for i in range(num_episodes):
        _build_episode(store, f"episode_{i}", defect=defects.get(i))

    with open(os.path.join(path, "config.yaml"), "w") as f:
        yaml.dump(RECORDING_CONFIG, f)


# =============================================================================
# check_episode integration tests
# =============================================================================


class TestCheckEpisodeIntegration:
    """Integration tests for check_episode orchestrator."""

    def test_healthy_episode_passes(self, tmp_path):
        """A well-formed episode should pass all enabled checks."""
        store = zarr.open(str(tmp_path / "data"), mode="w")
        _build_episode(store, "episode_0")
        episode_data = store["episode_0"]

        result = check_episode(
            episode_data,
            RECORDING_CONFIG,
            CHECK_CONFIG,
            episode_name="episode_0",
        )

        assert result["passed"], f"Expected pass, got issues: {result['issues']}"
        assert result["issues"] == []

    def test_defective_episode_fails(self, tmp_path):
        """An episode with a defect should fail with issues reported."""
        store = zarr.open(str(tmp_path / "data"), mode="w")
        _build_episode(store, "episode_0", defect="static_gripper")
        episode_data = store["episode_0"]

        result = check_episode(
            episode_data,
            RECORDING_CONFIG,
            CHECK_CONFIG,
            episode_name="episode_0",
        )

        assert not result["passed"]
        assert len(result["issues"]) > 0

    def test_report_structure(self, tmp_path):
        """Returned report should have all expected top-level keys."""
        store = zarr.open(str(tmp_path / "data"), mode="w")
        _build_episode(store, "episode_0")
        episode_data = store["episode_0"]

        result = check_episode(
            episode_data,
            RECORDING_CONFIG,
            CHECK_CONFIG,
            episode_name="episode_0",
        )

        expected_keys = {
            "timestamp_checks",
            "sensor_checks",
            "tracking_checks",
            "device_check",
            "resolution_checks",
            "passed",
            "issues",
            "warnings",
        }
        assert expected_keys.issubset(result.keys())

        ts_keys = {
            "fps",
            "monotonic",
            "frame_intervals",
            "sustained_frame_drop",
            "coverage",
        }
        assert ts_keys.issubset(result["timestamp_checks"].keys())

    def test_a_snapshot_with_the_removed_ik_section_still_checks(self, tmp_path):
        """Session snapshots written before the IK check was removed carry an
        `ik_check` section, some with `enabled: true`. Re-checks read those
        snapshots back, so the section is ignored rather than failing the run."""
        store = zarr.open(str(tmp_path / "data"), mode="w")
        _build_episode(store, "episode_0")
        check_config = copy.deepcopy(CHECK_CONFIG)
        check_config["ik_check"] = {"enabled": True, "max_failure_ratio": 0.05}

        result = check_episode(
            store["episode_0"], RECORDING_CONFIG, check_config, episode_name="episode_0"
        )

        assert result["passed"], result["issues"]
        assert "ik_checks" not in result

    def test_a_snapshot_with_the_removed_policy_checks_still_checks(self, tmp_path):
        """The duration, latency, idle, speed, pace and arm-steady checks were
        removed, and so was the gripper's open/close speed limit. Sessions
        recorded before that carry their sections, enabled, with thresholds no
        episode here could meet. Re-checks ignore them rather than failing."""
        store = zarr.open(str(tmp_path / "data"), mode="w")
        _build_episode(store, "episode_0")
        check_config = copy.deepcopy(CHECK_CONFIG)
        check_config["gripper"]["enabled"] = {"speed": True}
        check_config["gripper"]["close_speed_threshold_mm_per_s"] = 1e-6
        check_config["duration"] = {
            "enabled": True,
            "default": {"min_duration_s": 1000.0, "max_duration_s": 2000.0},
        }
        check_config["latency"] = {"enabled": True, "expected_gripper_latency": 9.9}
        check_config["idle"] = {"enabled": True, "max_idle_duration_s": 1e-6}
        check_config["speed"] = {"enabled": True, "max_linear_speed_ms": 1e-6}
        check_config["pace"] = {"enabled": True, "max_speed_ms": 1e-6}
        check_config["slow_transport"] = {"enabled": True}
        check_config["arm_steady_before_close"] = {
            "enabled": True,
            "steady_threshold_m": 1e-9,
        }

        result = check_episode(
            store["episode_0"], RECORDING_CONFIG, check_config, episode_name="episode_0"
        )

        assert result["passed"], result["issues"]
        assert "speed" not in result["sensor_checks"]["right_umi_gripper"]


# =============================================================================
# check_dataset integration tests
# =============================================================================


class TestGripperCheckSwitches:
    """`gripper.enabled` may be a mapping of per-check switches.

    `enabled: true` still runs every check, so configs snapshotted by older
    stations behave as before. A mapping turns named checks off without taking
    the rest of the section with them -- the shipped baselines use it to drop
    the "did not move" check, because a unimanual task on a two-gripper rig
    leaves the unused gripper flat for the whole episode.
    """

    def _issues(self, tmp_path, enabled):
        store = zarr.open(str(tmp_path / "data"), mode="w")
        # Static distance trips "did not move"; 200mm is outside range_max_mm.
        _build_episode(store, "episode_0", defect="static_gripper")

        check_config = copy.deepcopy(CHECK_CONFIG)
        check_config["gripper"]["enabled"] = enabled

        result = check_episode(
            store["episode_0"],
            RECORDING_CONFIG,
            check_config,
            episode_name="episode_0",
        )
        return result["issues"], result

    def test_enabled_true_runs_every_check(self, tmp_path):
        issues, result = self._issues(tmp_path, enabled=True)
        assert any("did not move" in issue for issue in issues)
        assert "movement" in result["sensor_checks"]["right_umi_gripper"]
        assert "jumps" in result["sensor_checks"]["right_umi_gripper"]
        assert "range" in result["sensor_checks"]["right_umi_gripper"]

    def test_mapping_switches_off_only_the_named_check(self, tmp_path):
        issues, result = self._issues(tmp_path, enabled={"movement": False})
        assert not any("did not move" in issue for issue in issues)
        # Unnamed checks stay on and keep reporting.
        gripper = result["sensor_checks"]["right_umi_gripper"]
        assert "movement" not in gripper
        assert "jumps" in gripper
        assert "range" in gripper

    def test_enabled_false_switches_off_the_whole_section(self, tmp_path):
        _, result = self._issues(tmp_path, enabled=False)
        assert result["sensor_checks"] == {}

    def test_the_baseline_switches_off_the_movement_check_alone(self):
        """The rig parks a gripper for one-handed tasks, so a flat trace is
        expected there. Jumps and range still run."""
        repo_root = TEST_DIR.parent.parent
        path = repo_root / "config" / "data_check" / "data_check_config.yaml"
        gripper = _load_yaml(path)["gripper"]

        assert gripper["enabled"]["movement"] is False
        assert gripper["enabled"]["jumps"] is True
        assert gripper["enabled"]["range"] is True


class TestCheckDatasetIntegration:
    """Integration tests for check_dataset orchestrator."""

    def test_healthy_dataset(self, tmp_path):
        """A dataset with two healthy episodes should produce a passing report."""
        data_dir = str(tmp_path / "dataset")
        _build_dataset(data_dir, num_episodes=2)

        report = check_dataset(
            data_dir,
            check_config_path=CHECK_CONFIG_PATH,
        )

        assert report["summary"]["total_episodes"] == 2
        assert report["summary"]["failed_episodes"] == 0
        assert "config_device_names" in report["dataset_checks"]
        assert "gripper_serial_consistency" in report["dataset_checks"]

    def test_dataset_with_failing_episode(self, tmp_path):
        """A dataset with one defective episode should reflect it in the summary."""
        data_dir = str(tmp_path / "dataset")
        _build_dataset(data_dir, num_episodes=2, defects={1: "static_gripper"})

        report = check_dataset(
            data_dir,
            check_config_path=CHECK_CONFIG_PATH,
        )

        assert report["summary"]["total_episodes"] == 2
        assert report["summary"]["passed_episodes"] >= 1
        assert report["summary"]["failed_episodes"] >= 1
        assert 1 in report["summary"]["failed_episode_indices"]

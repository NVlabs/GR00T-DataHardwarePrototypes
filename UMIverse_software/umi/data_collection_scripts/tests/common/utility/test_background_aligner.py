# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for BackgroundAligner (umi/common/utility/background_aligner.py).

Test groups:
  Group 1 — Status log helpers (_read_log, _write_log, _update_status)
  Group 2 — _validate_aligned_episode
  Group 3 — zarr.json consistency (_write_zarr_json_index_source, resolve_index_source_from_config)
  Group 4 — BackgroundAligner subprocess integration
  Group 5 — _worker_main in-process
  Group 6 — Continuous collect→align simulation (_worker_main in thread)
  Group 7 — Robustness: kill helper + None-worker guard
  Group 8 — Max alignment attempts (retry limit)
  Group 9 — Stuck-worker detection via _last_progress heartbeat
  Group 10 — Fast-exit tight-loop protection
  Group 11 — Re-init recovery of unaligned episodes
  Group 12 — Attempts are charged to the episode in flight, not to the queue

Run with: uv run pytest tests/common/utility/test_background_aligner.py -v
"""

from __future__ import annotations

import json
import multiprocessing
import os
import tempfile
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml
import zarr

from real_scripts.data_processing.align_data import (
    _write_zarr_json_index_source,
    resolve_index_source_from_config,
)
from umi.common.utility.background_aligner import (
    _read_log,
    _update_status,
    _validate_aligned_episode,
    _worker_main,
    _write_log,
)
from umi.common.utility.output_layout import output_root, session_dir

# =============================================================================
# Shared session builders
# =============================================================================


def _write_config(
    session_path: str,
    video_source_names: list[str],
    numeric_source_names: list[str],
) -> None:
    """Write config.yaml required by DataBuffer and resolve_index_source_from_config."""
    config = {
        "recording": {
            "video_frame_type": "RGBCameraFrame",
            "numeric_frame_type": "NumericFrame",
        },
        "video_sources": [
            {"name": n, "frame_type": "RGBCameraFrame", "latency": 0.0}
            for n in video_source_names
        ],
        "numeric_sources": [
            {"name": n, "frame_type": "NumericFrame", "latency": 0.0}
            for n in numeric_source_names
        ],
    }
    with open(os.path.join(session_path, "config.yaml"), "w") as f:
        yaml.dump(config, f)


def _add_episode(session_path: str, episode_idx: int, n_frames: int = 10) -> None:
    """Append a normal (fully overlapping) episode to an existing zarr session."""
    store = zarr.open(session_path, mode="a")
    ep = store.require_group(f"episode_{episode_idx}")
    t = np.linspace(0.0, 1.0, n_frames)

    cam = ep.require_group("right_cam")
    cam["capture_time"] = t
    cam["receive_time"] = t + 0.005  # slight receive delay
    cam["rgb"] = np.zeros((n_frames, 4, 4, 3), dtype=np.uint8)

    robot = ep.require_group("robot_state")
    robot["capture_time"] = t
    robot["receive_time"] = t
    robot["value"] = np.zeros((n_frames, 7))


def _add_nonoverlapping_episode(
    session_path: str, episode_idx: int, n_frames: int = 10
) -> None:
    """Append an episode where camera and robot time ranges do not intersect.

    Camera: 0.0–0.9 s  |  Robot: 2.0–2.9 s  →  valid_timestamps will be empty.
    """
    store = zarr.open(session_path, mode="a")
    ep = store.require_group(f"episode_{episode_idx}")
    t_cam = np.linspace(0.0, 0.9, n_frames)
    t_robot = np.linspace(2.0, 2.9, n_frames)

    cam = ep.require_group("right_cam")
    cam["capture_time"] = t_cam
    cam["receive_time"] = t_cam + 0.005
    cam["rgb"] = np.zeros((n_frames, 4, 4, 3), dtype=np.uint8)

    robot = ep.require_group("robot_state")
    robot["capture_time"] = t_robot
    robot["receive_time"] = t_robot
    robot["value"] = np.zeros((n_frames, 7))


def _make_raw_session(
    tmpdir: str,
    n_episodes: int = 1,
    n_frames: int = 10,
) -> tuple[str, str]:
    """Create a minimal raw zarr session ready for alignment.

    Returns (session_path, aligned_path).
    config.yaml lists right_cam (video) and robot_state (numeric).
    """
    session_path = os.path.join(tmpdir, "session")
    aligned_path = session_path + "_aligned"
    os.makedirs(session_path)
    os.makedirs(aligned_path)
    _write_config(session_path, ["right_cam"], ["robot_state"])
    for i in range(n_episodes):
        _add_episode(session_path, i, n_frames)
    return session_path, aligned_path


# =============================================================================
# Group 1 — Status log helpers
# =============================================================================


def test_read_log_missing_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _read_log(os.path.join(tmpdir, "nonexistent.json"))
        assert result == {}


def test_read_log_corrupt_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "log.json")
        with open(path, "w") as f:
            f.write("{not valid json")
        assert _read_log(path) == {}


def test_write_log_atomic():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "log.json")
        data = {"sess": {"episode_0": {"align_status": "success"}}}
        _write_log(path, data)
        assert os.path.exists(path)
        assert not os.path.exists(path + ".tmp")
        with open(path) as f:
            assert json.load(f) == data


def test_update_status_align_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "log.json")
        _update_status(
            path, "sess", "episode_0", align_status="success", align_error=None
        )
        log = _read_log(path)
        ep = log["sess"]["episode_0"]
        assert ep["align_status"] == "success"
        assert ep["align_error"] is None
        assert "align_timestamp" in ep


def test_update_status_independent_episodes():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "log.json")
        _update_status(
            path, "sess", "episode_0", align_status="success", align_error=None
        )
        _update_status(
            path, "sess", "episode_1", align_status="failure", align_error="err"
        )
        log = _read_log(path)
        assert log["sess"]["episode_0"]["align_status"] == "success"
        assert log["sess"]["episode_1"]["align_status"] == "failure"
        assert log["sess"]["episode_1"]["align_error"] == "err"


def test_update_status_source_field():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "log.json")
        _update_status(
            path,
            "sess",
            "episode_0",
            align_status="success",
            source="health_check",
        )
        log = _read_log(path)
        assert log["sess"]["episode_0"]["source"] == "health_check"


def test_update_status_source_omitted_leaves_no_key():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "log.json")
        _update_status(
            path, "sess", "episode_0", align_status="success", align_error=None
        )
        log = _read_log(path)
        assert "source" not in log["sess"]["episode_0"]


def test_update_status_source_preserved_across_updates():
    """A second _update_status call without source should not erase a previous source."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "log.json")
        _update_status(
            path,
            "sess",
            "episode_0",
            align_status="success",
            source="health_check",
        )
        _update_status(
            path,
            "sess",
            "episode_0",
            align_status="failure",
            align_error="boom",
        )
        log = _read_log(path)
        ep = log["sess"]["episode_0"]
        assert ep["source"] == "health_check"
        assert ep["align_status"] == "failure"


# =============================================================================
# Group 2 — _validate_aligned_episode
# =============================================================================


def _build_valid_aligned_store(tmpdir: str, index_source: str = "right_cam") -> str:
    """Minimal valid aligned zarr store for _validate_aligned_episode."""
    store_path = os.path.join(tmpdir, "aligned")
    store = zarr.open(store_path, mode="w")
    ep = store.require_group("episode_0")
    ep["valid_indices"] = np.array([0, 1, 2])
    ep["valid_timestamps"] = np.array([0.1, 0.2, 0.3])
    src = ep.require_group(index_source)
    src["aligned_frame_indices"] = np.array([0, 1, 2])
    return store_path


def test_validate_valid_episode():
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = _build_valid_aligned_store(tmpdir)
        assert _validate_aligned_episode(store_path, 0, "right_cam") == []


def test_validate_missing_episode_group():
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "aligned")
        zarr.open(store_path, mode="w")  # empty store, no episode_0
        problems = _validate_aligned_episode(store_path, 0, "right_cam")
        assert len(problems) == 1
        assert "episode_0" in problems[0]


def test_validate_missing_valid_indices():
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "aligned")
        store = zarr.open(store_path, mode="w")
        ep = store.require_group("episode_0")
        # valid_timestamps + index source present; valid_indices absent
        ep["valid_timestamps"] = np.array([0.1, 0.2])
        src = ep.require_group("right_cam")
        src["aligned_frame_indices"] = np.array([0, 1])
        problems = _validate_aligned_episode(store_path, 0, "right_cam")
        assert any("valid_indices" in p for p in problems)


def test_validate_missing_valid_timestamps():
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "aligned")
        store = zarr.open(store_path, mode="w")
        ep = store.require_group("episode_0")
        ep["valid_indices"] = np.array([0, 1])
        src = ep.require_group("right_cam")
        src["aligned_frame_indices"] = np.array([0, 1])
        # valid_timestamps deliberately absent
        problems = _validate_aligned_episode(store_path, 0, "right_cam")
        assert any("valid_timestamps" in p for p in problems)


def test_validate_missing_index_source_group():
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "aligned")
        store = zarr.open(store_path, mode="w")
        ep = store.require_group("episode_0")
        ep["valid_indices"] = np.array([0, 1])
        ep["valid_timestamps"] = np.array([0.1, 0.2])
        # index source group deliberately absent
        problems = _validate_aligned_episode(store_path, 0, "right_cam")
        assert any("right_cam" in p for p in problems)


def test_validate_missing_aligned_frame_indices():
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "aligned")
        store = zarr.open(store_path, mode="w")
        ep = store.require_group("episode_0")
        ep["valid_indices"] = np.array([0, 1])
        ep["valid_timestamps"] = np.array([0.1, 0.2])
        ep.require_group("right_cam")  # group present but aligned_frame_indices absent
        problems = _validate_aligned_episode(store_path, 0, "right_cam")
        assert any("aligned_frame_indices" in p for p in problems)


def test_validate_empty_valid_timestamps():
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "aligned")
        store = zarr.open(store_path, mode="w")
        ep = store.require_group("episode_0")
        ep["valid_indices"] = np.array([], dtype=int)
        ep["valid_timestamps"] = np.array([])  # empty — no overlapping window
        src = ep.require_group("right_cam")
        src["aligned_frame_indices"] = np.array([], dtype=int)
        problems = _validate_aligned_episode(store_path, 0, "right_cam")
        assert any("empty" in p for p in problems)


# =============================================================================
# Group 3 — zarr.json consistency
# =============================================================================


def test_zarr_json_written_with_index_source():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_zarr_json_index_source(tmpdir, "right_cam")
        path = os.path.join(tmpdir, "zarr.json")
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["attributes"]["index_source"] == "right_cam"


def test_zarr_json_updated_preserves_existing_attrs():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "zarr.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": {"operator_name": "alice", "task_name": "assembly"},
                },
                f,
            )
        _write_zarr_json_index_source(tmpdir, "right_cam")
        with open(path) as f:
            data = json.load(f)
        assert data["attributes"]["index_source"] == "right_cam"
        assert data["attributes"]["operator_name"] == "alice"
        assert data["attributes"]["task_name"] == "assembly"


def test_resolve_index_source_picks_first_candidate_match():
    """First candidate present in config wins, regardless of config order."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "video_sources": [
                {"name": "right_cam", "frame_type": "RGBCameraFrame"},
                {"name": "left_cam", "frame_type": "RGBCameraFrame"},
            ],
            "numeric_sources": [],
        }
        with open(os.path.join(tmpdir, "config.yaml"), "w") as f:
            yaml.dump(config, f)
        # candidates: left_cam first — left_cam is in config, so it wins
        result = resolve_index_source_from_config(tmpdir, ["left_cam", "right_cam"])
        assert result == "left_cam"


def test_resolve_index_source_raises_if_no_match():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "video_sources": [{"name": "right_cam", "frame_type": "RGBCameraFrame"}],
            "numeric_sources": [],
        }
        with open(os.path.join(tmpdir, "config.yaml"), "w") as f:
            yaml.dump(config, f)
        with pytest.raises(ValueError, match="No index source found"):
            resolve_index_source_from_config(tmpdir, ["wrist_cam"])


# =============================================================================
# Group 4 — BackgroundAligner subprocess integration
# =============================================================================


def _make_background_aligner(
    session_path: str, aligned_path: str, log_dir: str, **kwargs
):
    """Construct a BackgroundAligner with test-safe defaults (local log)."""
    from umi.common.utility.background_aligner import BackgroundAligner

    aligner = BackgroundAligner(
        session_path=session_path,
        station_name="test_station",
        task_name="test_task",
        index_sources="right_cam",
        copy_videos=False,
        aligned_dir=aligned_path,
        log_dir=log_dir,
        **kwargs,
    )
    aligner.start()
    return aligner


def test_aligned_dir_defaults_to_the_sessions_aligned_folder(tmp_path):
    """A session under output/raw/{task}/{session} aligns into
    output/aligned/{task}/{session}; one outside the output folder into its
    `<session>_aligned` sibling."""
    from umi.common.utility.background_aligner import BackgroundAligner

    name = "2026-04-01_umi35L_test_task"
    in_layout = session_dir("raw", "test_task", name)
    outside = str(tmp_path / name)
    expected = {
        in_layout: output_root() / "aligned" / "test_task" / name,
        outside: tmp_path / f"{name}_aligned",
    }
    for session_path, aligned in expected.items():
        aligner = BackgroundAligner(
            session_path=session_path,
            station_name="test-station",
            task_name="test_task",
            log_dir=str(tmp_path / "logs"),
        )
        assert aligner.aligned_session_path == str(aligned)
        assert aligned.is_dir()


def test_aligner_success_saves_valid_zarr():
    with tempfile.TemporaryDirectory() as tmpdir:
        session_path, aligned_path = _make_raw_session(tmpdir, n_episodes=1)
        log_dir = os.path.join(tmpdir, "logs")
        os.makedirs(log_dir)

        aligner = _make_background_aligner(session_path, aligned_path, log_dir)
        aligner.queue_alignment(0)
        aligner.shutdown(timeout=60)

        store = zarr.open(aligned_path, mode="r")
        assert "episode_0" in store
        assert store["episode_0"]["valid_timestamps"].shape[0] > 0
        assert "aligned_frame_indices" in store["episode_0"]["right_cam"]
        # No uniform_fps given: the station default, an even 30 fps grid.
        assert store["episode_0"].attrs["uniform_fps"] == 30


def test_aligner_uniform_fps_reaches_data_buffer_and_validates():
    """uniform_fps travels BackgroundAligner -> worker -> DataBuffer, and the
    output passes _validate_aligned_episode even though the index camera's
    aligned_frame_indices are now nearest-frame picks rather than an arange."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 10 index frames over 1 s, so a 15 Hz grid must repeat frames.
        session_path, aligned_path = _make_raw_session(tmpdir, n_episodes=1)
        log_dir = os.path.join(tmpdir, "logs")
        os.makedirs(log_dir)

        aligner = _make_background_aligner(
            session_path, aligned_path, log_dir, uniform_fps=15
        )
        aligner.queue_alignment(0)
        aligner.shutdown(timeout=60)

        ep_log = aligner.get_status()[os.path.basename(session_path)]["episode_0"]
        assert ep_log["align_status"] == "success", ep_log.get("align_error")
        ep = zarr.open(aligned_path, mode="r")["episode_0"]
        assert ep.attrs["uniform_fps"] == 15
        ts = ep["valid_timestamps"][:]
        np.testing.assert_allclose(np.diff(ts), 1 / 15)
        frames = ep["right_cam"]["aligned_frame_indices"][:]
        assert len(frames) == len(ts)
        assert not np.array_equal(frames, np.arange(len(frames)))


def test_aligner_success_status_log():
    with tempfile.TemporaryDirectory() as tmpdir:
        session_path, aligned_path = _make_raw_session(tmpdir, n_episodes=1)
        log_dir = os.path.join(tmpdir, "logs")
        os.makedirs(log_dir)

        aligner = _make_background_aligner(session_path, aligned_path, log_dir)
        aligner.queue_alignment(0)
        aligner.shutdown(timeout=60)

        log = aligner.get_status()
        session_name = os.path.basename(session_path)
        ep = log[session_name]["episode_0"]
        assert ep["align_status"] == "success"


def test_aligner_invalid_episode_status_log():
    """Non-overlapping sources → align_status=invalid with an error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        session_path, aligned_path = _make_raw_session(tmpdir, n_episodes=0)
        _add_nonoverlapping_episode(session_path, 0)
        log_dir = os.path.join(tmpdir, "logs")
        os.makedirs(log_dir)

        aligner = _make_background_aligner(session_path, aligned_path, log_dir)
        aligner.queue_alignment(0)
        aligner.shutdown(timeout=60)

        log = aligner.get_status()
        session_name = os.path.basename(session_path)
        ep = log[session_name]["episode_0"]
        assert ep["align_status"] == "invalid"
        assert ep["align_error"]  # non-empty


def test_aligner_zarr_json_index_source_written():
    """After alignment, aligned zarr.json must have the index_source attribute.

    _worker_main writes it with _write_zarr_json_index_source once it has
    resolved index_source.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        session_path, aligned_path = _make_raw_session(tmpdir, n_episodes=1)
        log_dir = os.path.join(tmpdir, "logs")
        os.makedirs(log_dir)

        aligner = _make_background_aligner(session_path, aligned_path, log_dir)
        aligner.queue_alignment(0)
        aligner.shutdown(timeout=60)

        zarr_json_path = os.path.join(aligned_path, "zarr.json")
        assert os.path.exists(zarr_json_path), (
            "zarr.json not written — _worker_main must call "
            "_write_zarr_json_index_source after resolving index_source"
        )
        with open(zarr_json_path) as f:
            data = json.load(f)
        assert data["attributes"]["index_source"] == "right_cam"


# =============================================================================
# Group 5 — _worker_main in-process
# =============================================================================


def _run_worker_inprocess(
    session_path: str,
    aligned_path: str,
    log_path: str,
    episode_indices: list[int],
) -> None:
    """Call _worker_main directly (same process)."""
    queue: multiprocessing.Queue = multiprocessing.Queue()
    for idx in episode_indices:
        queue.put(idx)
    queue.put(None)  # sentinel

    _worker_main(
        queue,
        session_path,
        os.path.basename(session_path),
        "test_task",
        aligned_path,
        ["right_cam"],  # index_sources (list)
        False,  # copy_videos
        log_path,
        "test_station_align_test.json",
    )


def test_worker_per_episode_error_isolation():
    """Episode 0 invalid → align_status=invalid; episode 1 normal → success."""
    with tempfile.TemporaryDirectory() as tmpdir:
        session_path, aligned_path = _make_raw_session(tmpdir, n_episodes=0)
        _add_nonoverlapping_episode(session_path, 0)  # will fail validation
        _add_episode(session_path, 1)  # normal

        log_path = os.path.join(tmpdir, "log.json")
        _run_worker_inprocess(session_path, aligned_path, log_path, [0, 1])

        log = _read_log(log_path)
        session_name = os.path.basename(session_path)
        assert log[session_name]["episode_0"]["align_status"] == "invalid"
        assert log[session_name]["episode_1"]["align_status"] == "success"


def test_worker_skips_alignment_when_already_aligned():
    """Pre-aligned episode (align_status=success) → worker skips
    _align_one_episode entirely."""
    with tempfile.TemporaryDirectory() as tmpdir:
        session_path, aligned_path = _make_raw_session(tmpdir, n_episodes=1)
        log_path = os.path.join(tmpdir, "log.json")
        session_name = os.path.basename(session_path)

        # Phase 1: run a real alignment so aligned zarr data exists on disk
        _run_worker_inprocess(session_path, aligned_path, log_path, [0])
        log = _read_log(log_path)
        assert log[session_name]["episode_0"]["align_status"] == "success"

        # Phase 2: re-queue it (as after a restart) — alignment must be skipped
        mock_align = MagicMock(
            side_effect=AssertionError("_align_one_episode should not be called")
        )
        with patch(
            "umi.common.utility.background_aligner._align_one_episode", mock_align
        ):
            _run_worker_inprocess(session_path, aligned_path, log_path, [0])

        mock_align.assert_not_called()
        log = _read_log(log_path)
        assert log[session_name]["episode_0"]["align_status"] == "success"


# =============================================================================
# Group 6 — Continuous collect→align simulation
# =============================================================================


def _run_continuous_simulation(
    session_path: str,
    aligned_path: str,
    log_path: str,
    n_episodes: int,
    bad_episode_idx: int | None,
    timeout: float = 120,
) -> None:
    """Run _worker_main in a thread, feed episodes from the test thread.

    If bad_episode_idx is set, that episode is created as non-overlapping (invalid).
    """
    queue: multiprocessing.Queue = multiprocessing.Queue()
    session_name = os.path.basename(session_path)
    log_filename = "test_continuous_align.json"

    thread = threading.Thread(
        target=_worker_main,
        args=(
            queue,
            session_path,
            session_name,
            "test_task",
            aligned_path,
            ["right_cam"],
            False,  # copy_videos
            log_path,
            log_filename,
        ),
        daemon=True,
    )
    thread.start()

    for i in range(n_episodes):
        if i == bad_episode_idx:
            _add_nonoverlapping_episode(session_path, i)
        else:
            _add_episode(session_path, i)
        queue.put(i)

    queue.put(None)  # sentinel
    thread.join(timeout=timeout)

    assert not thread.is_alive(), "Worker thread did not finish within timeout"


def test_continuous_all_episodes_aligned():
    """5 back-to-back episodes: all aligned, zarr.json present."""
    with tempfile.TemporaryDirectory() as tmpdir:
        session_path, aligned_path = _make_raw_session(tmpdir, n_episodes=0)
        log_path = os.path.join(tmpdir, "log.json")

        _run_continuous_simulation(
            session_path,
            aligned_path,
            log_path,
            n_episodes=5,
            bad_episode_idx=None,
        )

        # All 5 episode groups present and non-empty in aligned zarr
        aligned_store = zarr.open(aligned_path, mode="r")
        for i in range(5):
            ep_key = f"episode_{i}"
            assert ep_key in aligned_store, f"{ep_key} missing from aligned zarr"
            assert aligned_store[ep_key]["valid_timestamps"].shape[0] > 0, (
                f"{ep_key}: valid_timestamps is empty"
            )

        # Status log: all success
        log = _read_log(log_path)
        session_name = os.path.basename(session_path)
        for i in range(5):
            ep = log[session_name][f"episode_{i}"]
            assert ep["align_status"] == "success", f"episode_{i} align failed"

        # zarr.json written with correct index_source
        zarr_json_path = os.path.join(aligned_path, "zarr.json")
        assert os.path.exists(zarr_json_path)
        with open(zarr_json_path) as f:
            data = json.load(f)
        assert data["attributes"]["index_source"] == "right_cam"


def test_continuous_one_bad_episode_does_not_stall():
    """Episode 2 is invalid; episodes 0,1,3,4 complete normally; thread does not stall."""
    with tempfile.TemporaryDirectory() as tmpdir:
        session_path, aligned_path = _make_raw_session(tmpdir, n_episodes=0)
        log_path = os.path.join(tmpdir, "log.json")

        _run_continuous_simulation(
            session_path,
            aligned_path,
            log_path,
            n_episodes=5,
            bad_episode_idx=2,
        )

        log = _read_log(log_path)
        session_name = os.path.basename(session_path)

        # Episode 2 is invalid
        assert log[session_name]["episode_2"]["align_status"] == "invalid"

        # All other episodes succeed
        for i in [0, 1, 3, 4]:
            ep = log[session_name][f"episode_{i}"]
            assert ep["align_status"] == "success", f"episode_{i} align failed"


# =============================================================================
# Group 7 — Robustness: kill helper + None-worker guard
# =============================================================================


def test_kill_worker_terminates_live_process():
    """_kill_worker should terminate a running process within a few seconds."""
    from umi.common.utility.background_aligner import _kill_worker

    p = multiprocessing.Process(target=time.sleep, args=(60,))
    p.start()
    assert p.is_alive()
    _kill_worker(p)
    assert not p.is_alive()


def test_kill_worker_is_noop_for_dead_process():
    """_kill_worker should not raise when the process is already dead."""
    from umi.common.utility.background_aligner import _kill_worker

    p = multiprocessing.Process(target=lambda: None)
    p.start()
    p.join()
    assert not p.is_alive()
    _kill_worker(p)  # must not raise


# =============================================================================
# Group 8 — Max alignment attempts (retry limit)
# =============================================================================


def test_recover_skips_episode_at_max_attempts(tmp_path):
    """An episode with align_attempts >= max_align_attempts is never re-queued."""
    from datetime import datetime

    from umi.common.utility.background_aligner import BackgroundAligner, _write_log

    session_path, _ = _make_raw_session(str(tmp_path), n_episodes=2)
    log_dir = str(tmp_path / "logs")
    os.makedirs(log_dir, exist_ok=True)

    session_name = os.path.basename(session_path)
    log_path = os.path.join(
        log_dir, f"test-station_align_{datetime.now().strftime('%Y-%m-%d')}.json"
    )
    # episode_0: exhausted all 3 attempts, still not success → must be skipped
    # episode_1: 1 attempt, not yet success → should be re-queued
    _write_log(
        log_path,
        {
            session_name: {
                "episode_0": {"align_status": "failure", "align_attempts": 3},
                "episode_1": {"align_status": "failure", "align_attempts": 1},
            }
        },
    )

    aligner = BackgroundAligner(
        session_path=session_path,
        station_name="test-station",
        task_name="test-task",
        index_sources="right_cam",
        max_align_attempts=3,
        log_dir=log_dir,
    )
    aligner.start()
    aligner.shutdown(timeout=60)

    status = aligner.get_status()
    session_status = status.get(session_name, {})
    # episode_0 was permanently skipped — align_status must still be "failure"
    assert session_status["episode_0"]["align_status"] == "failure"
    # episode_1 was re-queued and processed
    assert session_status["episode_1"]["align_status"] in (
        "success",
        "invalid",
        "failure",
    )
    # Retrying episode_1 must increment align_attempts.
    assert session_status["episode_1"]["align_attempts"] >= 2


def test_recover_increments_align_attempts_on_requeue(tmp_path):
    """align_attempts is incremented in the log each time an episode is re-queued."""
    from datetime import datetime

    from umi.common.utility.background_aligner import BackgroundAligner, _write_log

    session_path, _ = _make_raw_session(str(tmp_path), n_episodes=1)
    log_dir = str(tmp_path / "logs")
    os.makedirs(log_dir, exist_ok=True)

    session_name = os.path.basename(session_path)
    log_path = os.path.join(
        log_dir, f"test-station_align_{datetime.now().strftime('%Y-%m-%d')}.json"
    )
    # episode_0 previously attempted once but not yet successful
    _write_log(
        log_path,
        {
            session_name: {
                "episode_0": {"align_status": "failure", "align_attempts": 1},
            }
        },
    )

    aligner = BackgroundAligner(
        session_path=session_path,
        station_name="test-station",
        task_name="test-task",
        index_sources="right_cam",
        max_align_attempts=3,
        log_dir=log_dir,
    )
    aligner.start()
    aligner.shutdown(timeout=60)

    status = aligner.get_status()
    ep = status.get(session_name, {}).get("episode_0", {})
    # Queuing a retry increments align_attempts.
    assert ep.get("align_attempts", 0) >= 2


# =============================================================================
# Group 9 — Stuck-worker detection via _last_progress heartbeat
# =============================================================================


def test_ensure_healthy_restarts_stuck_worker(tmp_path):
    """_ensure_healthy should restart the worker when it is alive but stuck."""
    from umi.common.utility.background_aligner import BackgroundAligner

    session_path, _ = _make_raw_session(str(tmp_path))
    aligner = BackgroundAligner(
        session_path=session_path,
        station_name="test-station",
        task_name="test-task",
        align_timeout=5,
        log_dir=str(tmp_path / "logs"),
    )
    aligner.start()
    original_pid = aligner._worker.pid

    # Simulate a stuck worker: stale heartbeat + non-empty queue
    aligner._last_progress.value = time.time() - 100  # stale
    aligner._queue.put(99)  # something in queue

    aligner._ensure_healthy()

    assert aligner._worker is not None
    assert aligner._worker.pid != original_pid  # new process spawned

    aligner.shutdown()


# =============================================================================
# Group 10 — Fast-exit tight-loop protection
# =============================================================================


def test_fast_exit_backoff_sets_next_restart_allowed(tmp_path):
    """After a fast exit, _next_restart_allowed should be set in the future."""
    from umi.common.utility.background_aligner import (
        BackgroundAligner,
    )

    session_path, _ = _make_raw_session(str(tmp_path))
    aligner = BackgroundAligner(
        session_path=session_path,
        station_name="test-station",
        task_name="test-task",
        log_dir=str(tmp_path / "logs"),
    )
    aligner.start()

    # Simulate an unexpected exit immediately after worker startup.
    aligner._worker_start_time = (
        time.time()
    )  # recent start — well within _FAST_EXIT_THRESHOLD
    aligner._worker.terminate()
    aligner._worker.join()

    aligner._queue.put(0)  # non-empty queue triggers health check
    aligner._ensure_healthy()

    assert aligner._consecutive_fast_exits == 1
    assert (
        aligner._next_restart_allowed >= time.time() + 1
    )  # backoff is min(2**1, 60) = 2s

    aligner.shutdown()


def test_five_fast_exits_permanently_stops_worker(tmp_path):
    """After 5 consecutive fast exits, _worker should be set to None."""
    from umi.common.utility.background_aligner import BackgroundAligner

    session_path, _ = _make_raw_session(str(tmp_path))
    aligner = BackgroundAligner(
        session_path=session_path,
        station_name="test-station",
        task_name="test-task",
        log_dir=str(tmp_path / "logs"),
    )
    aligner.start()

    # Drive consecutive_fast_exits to 4 directly, then trigger one more
    aligner._consecutive_fast_exits = 4
    aligner._next_restart_allowed = 0.0  # no backoff window
    aligner._worker_start_time = time.time()  # recent start
    aligner._worker.terminate()
    aligner._worker.join()
    aligner._queue.put(0)

    aligner._ensure_healthy()

    assert aligner._worker is None
    assert not aligner.is_healthy()

    aligner.shutdown()  # _worker is None — exercises the guard as a safe no-op


# =============================================================================
# Group 11 — Re-init recovery of unaligned episodes
# =============================================================================


def test_reinit_requeues_unaligned_episodes(tmp_path):
    """BackgroundAligner should re-queue episodes not yet successfully aligned."""
    from datetime import datetime

    from umi.common.utility.background_aligner import BackgroundAligner, _write_log

    session_path, _ = _make_raw_session(str(tmp_path), n_episodes=3)
    log_dir = str(tmp_path / "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Simulate: episode_0 fully done, episode_1 alignment failed, episode_2 never started
    session_name = os.path.basename(session_path)
    log_path = os.path.join(
        log_dir, f"test-station_align_{datetime.now().strftime('%Y-%m-%d')}.json"
    )
    _write_log(
        log_path,
        {
            session_name: {
                "episode_0": {"align_status": "success"},
                "episode_1": {"align_status": "failure"},
            }
        },
    )

    aligner = BackgroundAligner(
        session_path=session_path,
        station_name="test-station",
        task_name="test-task",
        index_sources="right_cam",
        log_dir=log_dir,
        align_timeout=60,
    )
    aligner.start()
    aligner.shutdown(timeout=60)

    status = aligner.get_status()
    session_status = status.get(session_name, {})
    # episode_0 unchanged — already success
    assert session_status.get("episode_0", {}).get("align_status") == "success"
    # episode_1 and episode_2 were re-queued and processed
    assert session_status.get("episode_1", {}).get("align_status") in (
        "success",
        "invalid",
    )
    assert session_status.get("episode_2", {}).get("align_status") in (
        "success",
        "invalid",
    )


# =============================================================================
# Group 12 — Attempts are charged to the episode in flight, not to the queue
# =============================================================================


def _align_log_path(log_dir: str) -> str:
    from datetime import datetime

    return os.path.join(
        log_dir, f"test-station_align_{datetime.now().strftime('%Y-%m-%d')}.json"
    )


def test_recover_charges_only_attempted_episodes(tmp_path):
    """A restart charges the episode the worker had started and leaves the one
    still waiting behind it uncharged. Both get aligned."""
    from umi.common.utility.background_aligner import BackgroundAligner, _write_log

    session_path, _ = _make_raw_session(str(tmp_path), n_episodes=2)
    log_dir = str(tmp_path / "logs")
    os.makedirs(log_dir, exist_ok=True)
    session_name = os.path.basename(session_path)
    # Recover both the interrupted episode and queued work from the failed worker.
    _write_log(
        _align_log_path(log_dir),
        {session_name: {"episode_0": {"align_status": "in_progress"}}},
    )

    aligner = BackgroundAligner(
        session_path=session_path,
        station_name="test-station",
        task_name="test-task",
        index_sources="right_cam",
        max_align_attempts=3,
        log_dir=log_dir,
    )
    aligner.start()
    aligner.shutdown(timeout=60)

    status = aligner.get_status()[session_name]
    assert status["episode_0"]["align_attempts"] == 1
    assert status["episode_1"].get("align_attempts", 0) == 0
    assert status["episode_0"]["align_status"] == "success"
    assert status["episode_1"]["align_status"] == "success"


def test_recover_charges_an_interruption_once(tmp_path):
    """Restarts that happen before the worker gets back to an episode do not
    keep charging it, so it is not dropped after the third restart."""
    import queue as _queue

    from umi.common.utility.background_aligner import BackgroundAligner, _write_log

    session_path, _ = _make_raw_session(str(tmp_path), n_episodes=1)
    log_dir = str(tmp_path / "logs")
    os.makedirs(log_dir, exist_ok=True)
    session_name = os.path.basename(session_path)
    _write_log(
        _align_log_path(log_dir),
        {session_name: {"episode_0": {"align_status": "in_progress"}}},
    )

    aligner = BackgroundAligner(
        session_path=session_path,
        station_name="test-station",
        task_name="test-task",
        index_sources="right_cam",
        max_align_attempts=3,
        log_dir=log_dir,
    )
    aligner._queue = _queue.Queue()  # no worker: read the queue back directly
    for _ in range(4):
        aligner._recover_unaligned_episodes()
        assert aligner._queue.get_nowait() == 0

    ep = aligner.get_status()[session_name]["episode_0"]
    assert ep["align_attempts"] == 1
    assert ep["align_status"] == "queued"


def test_shutdown_kill_releases_in_flight_stages(tmp_path, monkeypatch):
    """A kill the shutdown ordered is not the episode's fault: its in-flight
    stage goes back to queued instead of being charged at the next start."""
    from umi.common.utility import background_aligner as ba

    session_path, _ = _make_raw_session(str(tmp_path), n_episodes=2)
    log_dir = str(tmp_path / "logs")
    os.makedirs(log_dir, exist_ok=True)
    session_name = os.path.basename(session_path)
    ba._write_log(
        _align_log_path(log_dir),
        {
            session_name: {
                "episode_0": {"align_status": "success"},
                "episode_1": {"align_status": "in_progress"},
            }
        },
    )

    aligner = ba.BackgroundAligner(
        session_path=session_path,
        station_name="test-station",
        task_name="test-task",
        index_sources="right_cam",
        log_dir=log_dir,
    )

    class _Stuck:
        pid = 1

        def is_alive(self):
            return True

        def join(self, timeout=None):
            return None

    monkeypatch.setattr(ba, "_kill_worker", lambda worker: None)
    aligner._worker = _Stuck()
    aligner.shutdown(timeout=0)

    status = aligner.get_status()[session_name]
    assert status["episode_0"]["align_status"] == "success"
    assert status["episode_1"]["align_status"] == "queued"

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Stress tests for BackgroundAligner.

Test groups:
  Group S1 — Progressive data: episodes added incrementally while aligner runs
  Group S2 — Worker killed mid-alignment: same instance and full restart recovery

Markers:
  @pytest.mark.slow — real alignment (DataBuffer), may take 10-60 s per test.

Run with:
  uv run pytest tests/common/utility/test_background_aligner_stress.py -v -m slow
  uv run pytest tests/common/utility/test_background_aligner_stress.py -v  # all
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import time
from unittest.mock import patch

import numpy as np
import pytest
import yaml
import zarr

import umi.common.data.data_buffer  # noqa: F401 — pre-load before fork so patches survive

# ---------------------------------------------------------------------------
# Shared helpers (self-contained; mirror test_background_aligner.py helpers)
# ---------------------------------------------------------------------------


def _write_config(
    session_path: str,
    video_source_names: list[str],
    numeric_source_names: list[str],
) -> None:
    """Write config.yaml required by DataBuffer / resolve_index_source_from_config."""
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


def _add_episode(session_path: str, episode_idx: int, n_frames: int = 50) -> None:
    """Append a valid (fully overlapping) episode to an existing zarr session.

    Uses n_frames=50 by default (vs 10 in unit tests) so alignment takes
    ~0.5-1 s — long enough to reliably catch kill-mid-alignment scenarios.
    """
    store = zarr.open(session_path, mode="a")
    ep = store.require_group(f"episode_{episode_idx}")
    t = np.linspace(0.0, 1.0, n_frames)

    cam = ep.require_group("right_cam")
    cam["capture_time"] = t
    cam["receive_time"] = t + 0.005
    cam["rgb"] = np.zeros((n_frames, 32, 32, 3), dtype=np.uint8)

    robot = ep.require_group("robot_state")
    robot["capture_time"] = t
    robot["receive_time"] = t
    robot["value"] = np.zeros((n_frames, 7))


def _make_raw_session(
    tmpdir: str,
    n_episodes: int = 0,
) -> tuple[str, str]:
    """Create a minimal raw zarr session ready for alignment.

    Returns (session_path, aligned_path).
    """
    session_path = os.path.join(tmpdir, "session")
    aligned_path = session_path + "_aligned"
    os.makedirs(session_path)
    os.makedirs(aligned_path)
    _write_config(session_path, ["right_cam"], ["robot_state"])
    for i in range(n_episodes):
        _add_episode(session_path, i)
    return session_path, aligned_path


def _assert_all_aligned(
    log: dict,
    session_name: str,
    episode_indices: list[int],
    aligned_session_path: str,
) -> None:
    """Assert every episode in episode_indices is aligned correctly.

    Checks:
    - log[session_name][episode_N]["align_status"] == "success"
    - aligned zarr has episode_N group with non-empty valid_timestamps
    """
    session_status = log.get(session_name, {})
    aligned_store = zarr.open(aligned_session_path, mode="r")
    for i in episode_indices:
        ep_key = f"episode_{i}"
        ep_log = session_status.get(ep_key, {})
        assert ep_log.get("align_status") == "success", (
            f"{ep_key}: align_status={ep_log.get('align_status')!r}, "
            f"error={ep_log.get('align_error')!r}"
        )
        assert ep_key in aligned_store, f"{ep_key} missing from aligned zarr"
        assert aligned_store[ep_key]["valid_timestamps"].shape[0] > 0, (
            f"{ep_key}: valid_timestamps is empty"
        )


def _wait_for_progress(
    last_progress: multiprocessing.Value,
    initial_value: float,
    timeout: float = 60.0,
) -> bool:
    """Block until last_progress.value > initial_value or timeout.

    Returns True if progress was detected, False on timeout.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if last_progress.value > initial_value:
            return True
        time.sleep(0.05)
    return False


# =============================================================================
# Group S1 — Progressive data: public BackgroundAligner
# =============================================================================


@pytest.mark.slow
def test_stress_progressive_data_all_aligned(tmp_path):
    """8 episodes queued one-by-one while the aligner is running; all must be
    aligned (align_status==success) and present in the aligned zarr store.

    Simulates: operator records episodes sequentially during a session.
    """
    from umi.common.utility.background_aligner import BackgroundAligner

    session_path, _ = _make_raw_session(str(tmp_path), n_episodes=0)
    log_dir = str(tmp_path / "logs")

    aligner = BackgroundAligner(
        session_path=session_path,
        station_name="test-station",
        task_name="test-task",
        index_sources="right_cam",
        log_dir=log_dir,
    )
    aligner.start()

    n_episodes = 8
    for i in range(n_episodes):
        _add_episode(session_path, i)
        aligner.queue_alignment(i)
        time.sleep(0.05)  # pace that mimics ~20 Hz data collection

    aligner.shutdown(timeout=120)

    _assert_all_aligned(
        log=aligner.get_status(),
        session_name=aligner.session_name,
        episode_indices=list(range(n_episodes)),
        aligned_session_path=aligner.aligned_session_path,
    )


# =============================================================================
# Group S2 — Worker killed mid-alignment: recovery tests
# =============================================================================


@pytest.mark.slow
def test_stress_worker_killed_mid_alignment_same_instance_recovers(tmp_path):
    """Kill the worker while DataBuffer.align_episode is actively running.

    Mechanism:
    - DataBuffer.align_episode is patched (before fork) to signal a
      multiprocessing.Event then sleep for 30 s.
    - Because multiprocessing uses fork on Linux, the forked worker inherits
      the patched class from sys.modules.
    - The test waits for the Event (= worker is mid-alignment), SIGKILLs it,
      then exits the patch context so the restarted worker uses real DataBuffer.
    - _ensure_healthy / _recover_unaligned_episodes re-queues all incomplete
      episodes; final state must show all episodes aligned.
    """
    from umi.common.data.data_buffer import DataBuffer
    from umi.common.utility.background_aligner import BackgroundAligner

    n_episodes = 5
    session_path, _ = _make_raw_session(str(tmp_path), n_episodes=n_episodes)
    log_dir = str(tmp_path / "logs")

    # Shared event: worker sets this the instant it enters align_episode
    started_aligning = multiprocessing.Event()
    _real_align = DataBuffer.align_episode

    def _slow_align(self, episodes):
        started_aligning.set()  # tell the parent alignment has started
        time.sleep(30)  # hold here long enough for the test to kill the worker
        _real_align(self, episodes)

    with patch.object(DataBuffer, "align_episode", _slow_align):
        aligner = BackgroundAligner(
            session_path=session_path,
            station_name="test-station",
            task_name="test-task",
            index_sources="right_cam",
            log_dir=log_dir,
            align_timeout=120,
        )
        aligner.start()
        # Queue episode 0 — worker picks it up and blocks inside _slow_align
        aligner._queue.put(0)

        assert started_aligning.wait(timeout=30), (
            "Worker never entered align_episode within 30 s"
        )

        # Kill the worker during alignment.
        os.kill(aligner._worker.pid, signal.SIGKILL)
        aligner._worker.join(timeout=5)
        assert not aligner._worker.is_alive(), "Worker did not die after SIGKILL"

    # Remove the mock before restart so recovery uses the real DataBuffer.
    for i in range(n_episodes):
        aligner.queue_alignment(i)

    aligner.shutdown(timeout=120)

    _assert_all_aligned(
        log=aligner.get_status(),
        session_name=aligner.session_name,
        episode_indices=list(range(n_episodes)),
        aligned_session_path=aligner.aligned_session_path,
    )


@pytest.mark.slow
def test_stress_full_restart_new_instance_recovers_pending(tmp_path):
    """Simulate a hard kill of the recording process mid-session:
    1. Instance 1 queues N episodes and starts processing.
    2. After the first episode completes, the worker is killed without shutdown.
    3. A new BackgroundAligner instance (same session path + log dir) is created.
    4. Instance 2 detects incomplete episodes in the log and re-queues them.
    5. All N episodes are eventually aligned.

    Simulates: operator kills record_data.py with Ctrl-C / SIGKILL; next day
    they restart and expect the aligner to pick up where it left off.
    """
    from umi.common.utility.background_aligner import BackgroundAligner

    n_episodes = 6
    session_path, _ = _make_raw_session(str(tmp_path), n_episodes=n_episodes)
    log_dir = str(tmp_path / "logs")

    # ── Phase 1: instance 1 starts working, gets killed ───────────────────────
    aligner1 = BackgroundAligner(
        session_path=session_path,
        station_name="test-station",
        task_name="test-task",
        index_sources="right_cam",
        log_dir=log_dir,
        align_timeout=120,
    )
    aligner1.start()
    for i in range(n_episodes):
        aligner1._queue.put(i)

    # Wait until at least one episode finishes (progress detected)
    initial_progress = aligner1._last_progress.value
    progressed = _wait_for_progress(
        aligner1._last_progress, initial_progress, timeout=60
    )
    assert progressed, "Instance 1 worker made no progress within 60 s"

    # Kill worker without calling shutdown (simulates hard process kill)
    os.kill(aligner1._worker.pid, signal.SIGKILL)
    aligner1._worker.join(timeout=5)

    # ── Phase 2: new instance recovers ────────────────────────────────────────
    aligner2 = BackgroundAligner(
        session_path=session_path,
        station_name="test-station",
        task_name="test-task",
        index_sources="right_cam",
        log_dir=log_dir,
        align_timeout=120,
    )
    aligner2.start()
    # start() calls _recover_unaligned_episodes, which re-queues incomplete episodes
    aligner2.shutdown(timeout=120)

    _assert_all_aligned(
        log=aligner2.get_status(),
        session_name=aligner2.session_name,
        episode_indices=list(range(n_episodes)),
        aligned_session_path=aligner2.aligned_session_path,
    )

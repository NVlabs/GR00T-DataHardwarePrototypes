# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Background episode aligner that runs alongside data collection.

Provides ``BackgroundAligner`` — a daemon process that:

1. Accepts finalized episodes via ``multiprocessing.Queue``
2. Aligns each episode (temporal interpolation via ``DataBuffer``) into the
   session's aligned folder (``output/aligned/{task}/{session}``, see
   ``umi.common.utility.output_layout``)
3. Validates the aligned output
4. Maintains a JSON status log on the station (``~/.umi/alignment_logs``)

Robustness features:

* Per-episode error isolation (one failure never blocks the next)
* Worker health monitoring + automatic restart
* Bounded retries per episode across restarts

Usage::

    aligner = BackgroundAligner(
        session_path="/data/2026-04-01_my_task",
        station_name="station-1",
        task_name="my_task",
        index_sources="right_umi_camera,left_umi_camera",
    )

    # After finalising an episode (in RecorderManager.finalize_episode):
    aligner.queue_alignment(episode_idx=0)

    # On shutdown:
    aligner.shutdown()
"""

from __future__ import annotations

import contextlib
import json
import multiprocessing
import os
import socket
import time
import traceback
from datetime import datetime, timezone
from typing import Optional

from umi.common.utility.alignment_settings import (
    DEFAULT_INDEX_SOURCES,
    DEFAULT_UNIFORM_FPS,
    alignment_settings,
)
from umi.common.utility.output_layout import counterpart
from umi.common.utility.print_utils import print_red

_TAG = "[BackgroundAligner]"
# Charge a recovery attempt only when worker status proves alignment
# started; queued episodes were not attempted.
_ATTEMPTED_ALIGN = frozenset({"in_progress", "failure", "invalid"})


def _was_attempted(entry: dict) -> bool:
    return entry.get("align_status") in _ATTEMPTED_ALIGN


def _mark_requeued(entry: dict) -> None:
    """Put an attempted episode back to ``queued`` so a second interruption
    before the retry starts is not charged again. The worker redoes any
    episode whose alignment is not ``success``."""
    if entry.get("align_status") in _ATTEMPTED_ALIGN:
        entry["align_status"] = "queued"


_FAST_EXIT_THRESHOLD = (
    10  # seconds — worker dying faster than this counts as a fast exit
)


# ---------------------------------------------------------------------------
# Status log helpers
# ---------------------------------------------------------------------------


def _read_log(log_path: str) -> dict:
    """Read the JSON status log; returns ``{}`` on missing file or parse error."""
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _write_log(log_path: str, log_data: dict) -> None:
    """Atomically write *log_data* to *log_path* (write-to-tmp then rename)."""
    tmp = log_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(log_data, fh, indent=2)
    os.replace(tmp, log_path)


def _update_status(
    log_path: str,
    session_name: str,
    episode_name: str,
    *,
    align_status: Optional[str] = None,
    align_error: Optional[str] = None,
    source: Optional[str] = None,
    lock: Optional[multiprocessing.Lock] = None,
) -> None:
    """Merge new fields into an episode's status entry (atomic write)."""
    cm = lock if lock is not None else contextlib.nullcontext()
    with cm:
        log_data = _read_log(log_path)
        entry = log_data.setdefault(session_name, {}).setdefault(episode_name, {})
        ts = datetime.now(timezone.utc).isoformat()
        if align_status is not None:
            entry["align_status"] = align_status
            entry["align_error"] = align_error
            entry["align_timestamp"] = ts
        if source is not None:
            entry["source"] = source
        _write_log(log_path, log_data)


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------


def _kill_worker(worker: multiprocessing.Process) -> None:
    """Terminate a worker process, escalating to SIGKILL if needed."""
    if not worker.is_alive():
        return
    worker.terminate()
    worker.join(timeout=5)
    if worker.is_alive():
        worker.kill()
        worker.join(timeout=2)


# ---------------------------------------------------------------------------
# Single-episode alignment (runs inside the worker process)
# ---------------------------------------------------------------------------


_CORRUPTION_CHECK_MAX_BYTES = 50 * 1024 * 1024  # 50 MB


def _check_zarr_arrays_for_corruption(
    group,
    prefix: str,
) -> list[str]:
    """Read small zarr arrays under *group* to detect stale-chunk corruption.

    Stale chunk files left by a previous alignment with a different shape
    cause a ``ValueError`` when zarr tries to reshape the raw data into the
    metadata shape.  Reading the array is the only reliable way to detect
    this — the metadata alone looks fine.

    Only arrays whose metadata-reported size is below
    ``_CORRUPTION_CHECK_MAX_BYTES`` are read.  Large arrays (e.g. copied
    video frames) are skipped because reading them is too expensive and they
    are written via ``create_zarr_array`` which is less susceptible.
    """
    import numpy as np

    problems: list[str] = []
    for name in group.array_keys():
        arr = group[name]
        path = f"{prefix}/{name}"
        if arr.nbytes > _CORRUPTION_CHECK_MAX_BYTES:
            continue
        try:
            data = np.asarray(arr[:])
        except Exception as exc:
            problems.append(f"{path} corrupted: {exc}")
            continue
        if data.size == 0:
            problems.append(f"{path} is empty")
    for name in group.group_keys():
        problems.extend(
            _check_zarr_arrays_for_corruption(group[name], f"{prefix}/{name}")
        )
    return problems


def _validate_aligned_episode(
    aligned_session_path: str,
    episode_idx: int,
    index_source: str,
) -> list[str]:
    """Check that an aligned episode contains the expected artefacts.

    Returns a list of problems (empty == valid).  This catches the many
    silent-skip paths inside ``DataBuffer`` that log warnings but never
    raise (missing episodes, failed interpolators, absent sources, …).
    """
    import zarr as _zarr

    problems: list[str] = []
    store = _zarr.open(aligned_session_path, mode="r")
    episode_key = f"episode_{episode_idx}"

    if episode_key not in store:
        problems.append(f"{episode_key} group missing from aligned zarr")
        return problems

    ep = store[episode_key]

    # valid_indices and valid_timestamps are always written by _align_episode
    for required in ("valid_indices", "valid_timestamps"):
        if required not in ep:
            problems.append(f"{episode_key}/{required} missing")

    # The index source must have aligned_frame_indices
    if index_source in ep:
        if "aligned_frame_indices" not in ep[index_source]:
            problems.append(
                f"{episode_key}/{index_source}/aligned_frame_indices missing"
            )
    else:
        problems.append(f"{episode_key}/{index_source} group missing")

    # Read small arrays to detect stale-chunk corruption (shape vs data mismatch).
    # Large arrays (video frames) are skipped — too expensive to read.
    problems.extend(_check_zarr_arrays_for_corruption(ep, episode_key))

    return problems


def _align_one_episode(
    raw_session_path: str,
    aligned_session_path: str,
    episode_idx: int,
    index_source: str,
    copy_videos: bool,
    uniform_fps: float,
) -> list[str]:
    """Align a single episode and validate the output.

    Returns a list of validation problems (empty == fully successful).
    Raises on hard failures (e.g. config missing).  Defers heavy imports
    so the main process stays lightweight.
    """
    from umi.common.data.data_buffer import DataBuffer

    os.makedirs(aligned_session_path, exist_ok=True)
    buffer = DataBuffer(
        data_dir=raw_session_path,
        target_dir=aligned_session_path,
        index_source_name=index_source,
        copy_videos=copy_videos,
        uniform_fps=uniform_fps,
    )
    buffer.align_episode([episode_idx])

    return _validate_aligned_episode(aligned_session_path, episode_idx, index_source)


# ---------------------------------------------------------------------------
# Worker process entry-point
# ---------------------------------------------------------------------------


def _worker_main(
    queue: multiprocessing.Queue,
    raw_session_path: str,
    session_name: str,
    task_name: str,
    aligned_session_path: str,
    index_sources: list[str],
    copy_videos: bool,
    log_path: str,
    log_filename: str,
    last_progress: Optional[multiprocessing.Value] = None,
    log_lock: Optional[multiprocessing.Lock] = None,
    uniform_fps: float = DEFAULT_UNIFORM_FPS,
) -> None:
    """Consume alignment tasks until a ``None`` sentinel arrives."""
    # Initialize worker log redirection explicitly for fork, spawn and
    # forkserver; no configured log path leaves output unchanged.
    from umi.common.utility.log_capture import worker_redirect_stdio

    worker_redirect_stdio()

    import logging

    log_dir = os.path.dirname(log_path)
    file_log = os.path.join(log_dir, log_filename.replace(".json", ".log"))
    file_handler = logging.FileHandler(file_log, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )

    for logger_name in (
        "umi.common.data.data_buffer",
        "real_scripts.data_processing.align_data",
    ):
        lgr = logging.getLogger(logger_name)
        lgr.setLevel(logging.DEBUG)
        lgr.addHandler(file_handler)
        lgr.propagate = False  # keep INFO+ off the root (terminal) handler

    from real_scripts.data_processing.align_data import (
        _write_zarr_json_index_source,
        resolve_index_source_from_config,
    )

    try:
        index_source = resolve_index_source_from_config(raw_session_path, index_sources)
        print(f"{_TAG} Using index source: {index_source}")
        _write_zarr_json_index_source(aligned_session_path, index_source)
    except ValueError as exc:
        print(f"{_TAG} FATAL: {exc}")
        return

    while True:
        task = queue.get()
        if task is None:
            break

        episode_idx: int = task
        episode_name = f"episode_{episode_idx}"

        # Skip already aligned episodes recovered from the queue.
        cm = log_lock if log_lock is not None else contextlib.nullcontext()
        with cm:
            cached = _read_log(log_path)
        ep_entry = cached.get(session_name, {}).get(episode_name, {})
        if ep_entry.get("align_status") == "success":
            print(f"{_TAG} {episode_name} already succeeded, skipping")
            if last_progress is not None:
                last_progress.value = time.time()
            continue

        print(f"{_TAG} ── Processing {episode_name} ──")

        try:
            _update_status(
                log_path,
                session_name,
                episode_name,
                align_status="in_progress",
                lock=log_lock,
            )
            try:
                problems = _align_one_episode(
                    raw_session_path=raw_session_path,
                    aligned_session_path=aligned_session_path,
                    episode_idx=episode_idx,
                    index_source=index_source,
                    copy_videos=copy_videos,
                    uniform_fps=uniform_fps,
                )
            except Exception:  # noqa: BLE001
                tb = traceback.format_exc()
                _update_status(
                    log_path,
                    session_name,
                    episode_name,
                    align_status="failure",
                    align_error=tb,
                    lock=log_lock,
                )
                print_red(f"{_TAG} Alignment FAILED for {episode_name}:\n{tb}")
                continue

            if problems:
                detail = "; ".join(problems)
                _update_status(
                    log_path,
                    session_name,
                    episode_name,
                    align_status="invalid",
                    align_error=detail,
                    lock=log_lock,
                )
                print_red(
                    f"{_TAG} Alignment produced INVALID output "
                    f"for {episode_name}: {detail}"
                )
                continue

            _update_status(
                log_path,
                session_name,
                episode_name,
                align_status="success",
                lock=log_lock,
            )
            print(f"{_TAG} Alignment succeeded for {episode_name}")

        finally:
            if last_progress is not None:
                last_progress.value = time.time()


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class BackgroundAligner:
    """Align episodes in a serialised background process.

    Args:
        session_path: Absolute path to the raw session directory (zarr root).
        station_name: Station identifier (used in log filename).
        task_name: Task the session belongs to.
        index_sources: Comma-separated camera names tried in order
            (first match in the session config wins).
        uniform_fps: Rate of the even 1/uniform_fps s output grid.
        copy_videos: Copy mp4 files into aligned output.
        align_timeout: Seconds of no progress before a stuck worker is killed and
            restarted. Defaults to 300.
        max_align_attempts: Maximum number of times a single episode will be re-queued
            for alignment. Episodes that exceed this limit are permanently skipped.
            Defaults to 3.
        aligned_dir: Override for the aligned output directory. Defaults to the
            session's aligned folder (``output/aligned/{task}/{session}``, or
            ``<session_path>_aligned`` for a session outside the output folder).
        log_dir: Local log directory (default ``~/.umi/alignment_logs``).
    """

    def __init__(
        self,
        session_path: str,
        station_name: str,
        task_name: str,
        index_sources: str = ",".join(DEFAULT_INDEX_SOURCES),
        copy_videos: bool = True,
        align_timeout: int = 300,
        max_align_attempts: int = 3,
        aligned_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        uniform_fps: float = DEFAULT_UNIFORM_FPS,
    ):
        self.session_path = os.path.abspath(session_path)
        self.session_name = os.path.basename(self.session_path.rstrip(os.sep))
        self.station_name = station_name
        self.task_name = task_name

        self.aligned_session_path = aligned_dir or counterpart(
            self.session_path, "aligned"
        )
        os.makedirs(self.aligned_session_path, exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d")
        self._log_filename = f"{station_name}_align_{date_str}.json"
        self._log_dir = log_dir or os.path.join(
            os.path.expanduser("~"), ".umi", "alignment_logs"
        )
        os.makedirs(self._log_dir, exist_ok=True)
        self._log_path = os.path.join(self._log_dir, self._log_filename)

        self._align_timeout = align_timeout
        self._max_align_attempts = max_align_attempts
        self._last_progress: multiprocessing.Value = multiprocessing.Value(
            "d", time.time()
        )
        # Progress timestamps and their lock are shared across worker restarts.
        # Reset the timestamp when restarting to avoid an immediate false stall.
        self._log_lock: multiprocessing.Lock = multiprocessing.Lock()
        self._worker_start_time: float = 0.0
        self._consecutive_fast_exits: int = 0
        self._next_restart_allowed: float = 0.0

        # Defer worker creation until start(), after the caller has written session
        # configs.
        self._index_sources = [s.strip() for s in index_sources.split(",")]
        self._queue: multiprocessing.Queue = multiprocessing.Queue()
        self._worker_args = (
            self._queue,
            self.session_path,
            self.session_name,
            task_name,
            self.aligned_session_path,
            self._index_sources,
            copy_videos,
            self._log_path,
            self._log_filename,
            self._last_progress,
            self._log_lock,
            uniform_fps,
        )
        self._worker: Optional[multiprocessing.Process] = None

    def start(self) -> None:
        """Spawn the background worker and recover any unaligned episodes.

        Must be called after all session-level files (config.yaml, zarr
        metadata, etc.) have been written so the worker never reads
        partially-written data.
        """
        self._worker = self._spawn_worker()
        print(
            f"{_TAG} Background worker started "
            f"(pid={self._worker.pid}, session={self.session_name}, "
            f"task={self.task_name}, aligned_dir={self.aligned_session_path})"
        )
        self._recover_unaligned_episodes()

    # ── internal helpers ──────────────────────────────────────────────

    def _recover_unaligned_episodes(self) -> None:
        """Re-queue episodes from the current session not yet aligned.

        An attempt is charged only to an episode the worker had actually
        started (``in_progress``) or that had failed (``failure`` /
        ``invalid``); a restart does not count against the episodes still
        waiting behind it. A charged entry goes back to ``queued`` so the same
        interruption is not charged again at the next restart. Episodes at
        ``_max_align_attempts`` are permanently skipped.
        """
        import zarr as _zarr

        try:
            raw_store = _zarr.open(self.session_path, mode="r")
            raw_indices = sorted(
                int(k.split("_")[1])
                for k in raw_store.keys()
                if k.startswith("episode_")
            )
        except Exception:  # noqa: BLE001
            return  # session doesn't exist yet (first run)

        if not raw_indices:
            return

        with self._log_lock:
            log_data = _read_log(self._log_path)
            session_status = log_data.get(self.session_name, {})

            to_queue = []
            to_skip = []
            changed = False
            for ep_idx in raw_indices:
                ep_name = f"episode_{ep_idx}"
                ep = session_status.get(ep_name, {})
                if ep.get("align_status") == "success":
                    continue
                if _was_attempted(ep):
                    ep = log_data.setdefault(self.session_name, {}).setdefault(
                        ep_name, {}
                    )
                    ep["align_attempts"] = ep.get("align_attempts", 0) + 1
                    changed = True
                if ep.get("align_attempts", 0) >= self._max_align_attempts:
                    to_skip.append(ep_idx)  # keeps its failure status in the log
                else:
                    _mark_requeued(ep)
                    to_queue.append(ep_idx)

            if changed:
                _write_log(self._log_path, log_data)

        if to_skip:
            print(
                f"{_TAG} Skipping {len(to_skip)} episode(s) — "
                f"max attempts ({self._max_align_attempts}) reached: {to_skip}"
            )

        if to_queue:
            print(
                f"{_TAG} Re-queuing {len(to_queue)} episode(s) from prior run: {to_queue}"
            )
            for ep_idx in to_queue:
                self._queue.put(ep_idx)
        elif not to_skip:
            print(f"{_TAG} All {len(raw_indices)} episode(s) already aligned")

    def _spawn_worker(self) -> multiprocessing.Process:
        p = multiprocessing.Process(target=_worker_main, args=self._worker_args)
        p.daemon = True
        p.start()
        self._worker_start_time = time.time()
        return p

    def _ensure_healthy(self) -> None:
        """Restart the worker if it died or stopped making progress."""
        if self._worker is None:
            return  # permanently stopped after too many fast exits

        if time.time() < self._next_restart_allowed:
            return

        worker_alive = self._worker.is_alive()
        queue_stuck = (
            worker_alive
            and self._queue.qsize() > 0
            and time.time() - self._last_progress.value > self._align_timeout
        )

        if worker_alive and not queue_stuck:
            return  # healthy

        if queue_stuck:
            print(
                f"{_TAG} WARNING: worker stuck "
                f"(no progress for {self._align_timeout}s), restarting …"
            )
        else:
            print(
                f"{_TAG} WARNING: worker died "
                f"(exit={self._worker.exitcode}), restarting …"
            )

        worker_lifetime = time.time() - self._worker_start_time
        # Count only unexpected early exits toward fast-exit backoff. A stalled
        # worker ran successfully long enough that its restart resets that count.
        if not queue_stuck and worker_lifetime < _FAST_EXIT_THRESHOLD:
            self._consecutive_fast_exits += 1
        else:
            self._consecutive_fast_exits = 0

        if self._consecutive_fast_exits >= 5:
            print(
                f"{_TAG} FATAL: worker failed to start "
                f"{self._consecutive_fast_exits} times in a row; giving up"
            )
            _kill_worker(self._worker)
            self._worker = None
            return

        if self._consecutive_fast_exits > 0:
            backoff = min(2**self._consecutive_fast_exits, 60)
            self._next_restart_allowed = time.time() + backoff
            print(
                f"{_TAG} Fast exit #{self._consecutive_fast_exits}, "
                f"next restart allowed in {backoff:.0f}s"
            )

        _kill_worker(self._worker)
        self._queue = multiprocessing.Queue()
        self._last_progress.value = time.time()
        self._worker_args = (self._queue, *self._worker_args[1:])
        self._worker = self._spawn_worker()
        self._recover_unaligned_episodes()  # re-queue anything not yet done

    # ── public API ────────────────────────────────────────────────────

    def queue_alignment(self, episode_idx: int) -> None:
        """Enqueue an episode for background alignment.

        Safe to call from the recording loop — only puts an int on the
        queue, so it returns instantly.
        """
        self._ensure_healthy()
        if self._worker is None:
            print_red(
                f"{_TAG} ERROR: worker permanently stopped; dropping episode_{episode_idx}"
            )
            return
        self._queue.put(episode_idx)
        print(f"{_TAG} Queued episode_{episode_idx} for alignment")

    def is_healthy(self) -> bool:
        """Return True if the background worker is alive."""
        return self._worker is not None and self._worker.is_alive()

    def get_status(self) -> dict:
        """Read and return the current status log."""
        with self._log_lock:
            return _read_log(self._log_path)

    def _release_in_flight(self) -> None:
        """After a kill this aligner ordered, the episode in flight is not at
        fault: put its stages back to ``queued`` so recovery does not charge it."""
        with self._log_lock:
            data = _read_log(self._log_path)
            changed = False
            for episodes in data.values():
                for entry in episodes.values():
                    if entry.get("align_status") == "in_progress":
                        entry["align_status"] = "queued"
                        changed = True
            if changed:
                _write_log(self._log_path, data)

    def shutdown(self, timeout: float = 600) -> None:
        """Drain pending work and stop the background worker.

        Args:
            timeout: Seconds to wait for the worker to finish the episodes
                still queued. Defaults to 600 s.
        """
        if self._worker is None or not self._worker.is_alive():
            print(f"{_TAG} Worker already stopped")
            return
        self._queue.put(None)  # sentinel
        self._worker.join(timeout=timeout)
        if self._worker.is_alive():
            print(f"{_TAG} Worker did not stop in time; terminating")
            _kill_worker(self._worker)
            self._release_in_flight()
        print(f"{_TAG} Background worker stopped")


def attach_background_services(recorder_manager, cfg: dict, data_path: str) -> None:
    """Construct + attach the BackgroundAligner to *recorder_manager*.

    Reads the ``alignment`` block and ``meta.task_name`` from *cfg*. No-op when
    ``alignment.enabled`` is false. Aligned episodes land in the session's
    aligned folder (``output/aligned/{task}/{session}``, see
    ``umi.common.utility.output_layout``) unless ``alignment.aligned_dir`` says
    otherwise.

    Must be called **after** the RecorderManager is built: its ``__init__``
    writes config.yaml / data_check_config.yaml / zarr metadata to the session
    dir, and the background worker must never read a partially-written config.

    Args:
        recorder_manager: The ``RecorderManager`` to attach the aligner to.
            ``finalize_episode`` already queues work to it once set.
        cfg: Full configuration dictionary.
        data_path: Absolute session directory.
    """
    alignment_cfg = cfg.get("alignment", {})
    if not alignment_cfg.get("enabled", False):
        return

    task_name = cfg.get("meta", {}).get("task_name", "")
    if not task_name:
        raise ValueError("Task name not set. Please set meta.task_name in config.")

    settings = alignment_settings(cfg)
    recorder_manager.background_aligner = BackgroundAligner(
        session_path=data_path,
        station_name=socket.gethostname(),
        task_name=task_name,
        index_sources=",".join(settings.index_sources),
        uniform_fps=settings.uniform_fps,
        copy_videos=settings.copy_videos,
        align_timeout=alignment_cfg.get("align_timeout", 300),
        aligned_dir=alignment_cfg.get("aligned_dir", None),
    )
    recorder_manager.background_aligner.start()

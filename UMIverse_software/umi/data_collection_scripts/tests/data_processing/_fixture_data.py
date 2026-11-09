# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared S3-backed test data fetcher.

The regression and integration tests need real recorded sessions (raw +
quarantine) to validate the data-check pipeline end-to-end. Those
sessions are tens of MB to several GB each — too big to commit, too important to
skip silently. They live in a private bucket, so in environments without
access the tests that need them are skipped.

This module:
  - Downloads each declared session on demand to `/tmp/umi_test_fixtures/`,
    caching across pytest invocations.
  - Skips the requesting test with a clear warning when gear/S3 is unreachable,
    rather than failing on a missing-file assertion.

Usage from a test:

    from ._fixture_data import S3Unavailable, ensure_session_local, PLANE_V10_RAW

    @pytest.fixture(scope="module")
    def umi_v10_session():
        try:
            return ensure_session_local(PLANE_V10_RAW)
        except S3Unavailable as e:
            pytest.skip(str(e))
"""

from __future__ import annotations

import logging
import subprocess
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path("/tmp/umi_test_fixtures")

# Hard timeouts so a misbehaving network doesn't hang the whole test run.
_LS_TIMEOUT_SEC = 30
_DOWNLOAD_TIMEOUT_SEC = 600

# Parallel slots for per-file gear downloads. Empirically ~6-8 is the sweet
# spot — more than that and `uv run` overhead + S3 throttling dominates.
_PARALLEL_DOWNLOADS = 8


class S3Unavailable(Exception):
    """Raised when gear/S3 cannot be reached or a required key is missing.

    Tests should catch this and `pytest.skip(str(e))` so the suite stays green
    on machines without S3 credentials configured.
    """


@dataclass(frozen=True)
class SessionSpec:
    """Declares a session that lives on S3 and is needed by tests.

    Attributes:
        name: stable local cache subdirectory name.
        s3_prefix: full gear path to the session root (no trailing slash).
        episodes: which episode indices to download.
        extra_files: optional session-level files beyond the standard set
            (config.yaml, data_check_config.yaml, zarr.json). For quarantine
            sessions this typically includes data_check_report.json.
    """

    name: str
    s3_prefix: str
    episodes: tuple[int, ...]
    extra_files: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Known sessions — referenced by tests
# ---------------------------------------------------------------------------

_GRIPPER_UMI = "GrootDatasets/GearUMIData/GripperUMI"

# Bimanual recording fixture: approximately 1.5 GB across three episodes.
PLANE_V10_RAW = SessionSpec(
    name="plane_assembly_v10/2026-04-17_umi15L_umi35R",
    s3_prefix=(
        f"{_GRIPPER_UMI}/raw/plane_assembly_v10/"
        "2026-04-17_umi15L_umi35R_plane_assembly_v10"
    ),
    episodes=(0, 1, 2),
)

# UMI bimanual: raw (~1.5GB, 3 clean episodes) and the matching quarantine
# session (~3.5GB, 7 failing episodes). Together they exercise both the pass
# path and the failure-detection path of the data check.
PLANE_V11_RAW = SessionSpec(
    name="plane_assembly_v11/raw_2026-04-29_umi37L_umi34R",
    s3_prefix=(
        f"{_GRIPPER_UMI}/raw/plane_assembly_v11/"
        "2026-04-29_umi37L_umi34R_plane_assembly_v11"
    ),
    episodes=(0, 1, 2),
)
PLANE_V11_QUARANTINE = SessionSpec(
    name="plane_assembly_v11/quarantine_2026-04-29_umi37L_umi34R",
    s3_prefix=(
        f"{_GRIPPER_UMI}/raw-quarantine/plane_assembly_v11/"
        "2026-04-29_umi37L_umi34R_plane_assembly_v11"
    ),
    episodes=(0, 1, 2, 3, 4, 5, 6),
)

# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def _run_gear(args: list[str], timeout: int) -> subprocess.CompletedProcess:
    """Invoke `gear` via `uv run`. Raises S3Unavailable on failure."""
    cmd = ["uv", "run", "gear", *args]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as e:
        raise S3Unavailable(f"`uv` is not on PATH; cannot reach gear/S3 ({e})") from e
    except subprocess.TimeoutExpired as e:
        raise S3Unavailable(
            f"gear command timed out after {timeout}s: {' '.join(cmd)}"
        ) from e
    if res.returncode != 0:
        raise S3Unavailable(
            f"gear returned {res.returncode} for {' '.join(args)}: "
            f"{(res.stderr or res.stdout)[:300]}"
        )
    return res


def _list_remote_files(prefix: str) -> list[str]:
    """Return the file basenames (not directories) at one level under `prefix`."""
    res = _run_gear(["data", "ls", prefix.rstrip("/") + "/"], timeout=_LS_TIMEOUT_SEC)
    files = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("PRE"):
            continue
        # Lines look like: "2026-04-17 23:36:21    111894052 head_tracking_camera.mp4"
        parts = line.rsplit(maxsplit=1)
        if len(parts) == 2 and not parts[1].endswith("/"):
            files.append(parts[1])
    return files


def _download_one(remote: str, local_dir: Path, skip_if_exists: bool = True) -> None:
    """Download a single file at `remote` into `local_dir/` (using its basename).

    With `skip_if_exists`, an existing file is left alone — makes partial-cache
    populations resumable across interrupted runs.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    basename = remote.rsplit("/", 1)[-1]
    if skip_if_exists and (local_dir / basename).is_file():
        return
    _run_gear(
        ["data", "download", remote, str(local_dir) + "/"],
        timeout=_DOWNLOAD_TIMEOUT_SEC,
    )


def _download_recursive(remote_prefix: str, local_dir: Path) -> None:
    """Download every file under `remote_prefix` recursively into `local_dir`.

    gear's `**/*` glob skips depth-1 files (files directly under the prefix),
    so callers must also download those separately. This function only handles
    the deep tree (zarr chunks inside camera/sensor subdirectories). gear
    skips already-downloaded files internally, making this safely re-runnable.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    _run_gear(
        ["data", "download", remote_prefix.rstrip("/") + "/**/*", str(local_dir) + "/"],
        timeout=_DOWNLOAD_TIMEOUT_SEC,
    )


def _download_many_parallel(tasks: list[tuple[str, Path]]) -> None:
    """Download `[(remote, local_dir), …]` concurrently with bounded parallelism.

    Bubbles the first S3Unavailable encountered.
    """
    if not tasks:
        return
    errors: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=_PARALLEL_DOWNLOADS) as pool:
        futures = [pool.submit(_download_one, r, d) for r, d in tasks]
        for fut in as_completed(futures):
            try:
                fut.result()
            except BaseException as e:  # noqa: BLE001 — re-raise outside
                errors.append(e)
    if errors:
        # If any download failed, surface the first error — S3Unavailable for
        # gear/network issues, otherwise let the caller see what broke.
        raise errors[0]


_SESSION_LEVEL_FILES_DEFAULT = ("config.yaml", "data_check_config.yaml", "zarr.json")


def ensure_session_local(spec: SessionSpec) -> Path:
    """Return a local Path to the cached session, downloading from S3 if needed.

    On subsequent calls with the same spec, returns the cached path without
    network calls (cache hit is detected by presence of the session-level
    config.yaml and every requested episode's zarr.json).

    Raises:
        S3Unavailable: gear/S3 isn't reachable or a required key is missing.
    """
    local = CACHE_DIR / spec.name

    if _cache_complete(local, spec):
        logger.debug("fixture cache hit: %s", local)
        return local

    warnings.warn(
        f"Fetching session {spec.name!r} from S3 (~one-off setup; cached at {local})",
        stacklevel=2,
    )

    # Session-level files. `extra_files` (e.g. data_check_report.json) may
    # legitimately be absent for some sessions — don't fail the whole fetch
    # on a missing extra.
    for fname in _SESSION_LEVEL_FILES_DEFAULT:
        _download_one(f"{spec.s3_prefix}/{fname}", local)
    for fname in spec.extra_files:
        try:
            _download_one(f"{spec.s3_prefix}/{fname}", local)
        except S3Unavailable:
            logger.warning("optional %r not on S3 for %s; skipping", fname, spec.name)

    # Fetch independent episode files concurrently. Keep recursive cache
    # downloads sequential to avoid races in their shared destination.
    for ep in spec.episodes:
        ep_dir = local / f"episode_{ep}"
        ep_prefix = f"{spec.s3_prefix}/episode_{ep}"

        # Depth-1 files (mp4s + zarr.json) — discovered dynamically so camera
        # names are not hardcoded per session.
        depth1 = _list_remote_files(ep_prefix)
        _download_many_parallel([(f"{ep_prefix}/{fname}", ep_dir) for fname in depth1])

        # Depth-2+ files (zarr chunks inside per-device subdirs) via the
        # recursive glob.
        _download_recursive(ep_prefix, ep_dir)

    if not _cache_complete(local, spec):
        raise S3Unavailable(
            f"Download of {spec.name!r} appears incomplete (expected episodes "
            f"{spec.episodes}). Cache: {local}"
        )
    return local


def _cache_complete(local: Path, spec: SessionSpec) -> bool:
    """Cheap structural check that the cache has what tests need."""
    if not (local / "config.yaml").is_file():
        return False
    for ep in spec.episodes:
        ep_zarr = local / f"episode_{ep}" / "zarr.json"
        if not ep_zarr.is_file():
            return False
    return True


def session_path_or_skip(spec: SessionSpec) -> Path:
    """Convenience wrapper: returns the cached path, or pytest.skips on S3 failure.

    Use directly in `@pytest.fixture` bodies.
    """
    import pytest

    try:
        return ensure_session_local(spec)
    except S3Unavailable as e:
        pytest.skip(f"S3 fixture {spec.name!r} unavailable: {e}")


# Normalize report-only gap fields, including accepted baseline aliases.
# Continue comparing passed, num_violations and violation_frames so
# format additions cannot hide verdict changes.
FRAME_INTERVAL_REPORT_ONLY_KEYS = {
    "missing_s",
    "missing_pct",
    "longest_gap_s",
    "gap_min_frames",
    "gap_fail_max_missing_pct",
    "gap_fail_max_gap_s",
    "gap_fail_max_frames",
    "gap_fail_max_interp_error_mm",
    "est_interp_error_mm",
    "max_est_interp_error_mm",
    "interp_error_checked",
    # Baseline alias for the same report field.
    "max_missing_frames",
}

# Normalize report-only speed thresholds and their baseline aliases;
# continue comparing has_jumps, num_jumps and frame_index.
JUMP_REPORT_ONLY_KEYS = {
    "jump_threshold",
    "jump_threshold_m",
    "jump_threshold_mm_per_s",
    "jump_threshold_m_per_s",
    "magnitude_mm_per_s",
    "magnitude_m_per_s",
}

JUMP_ISSUE_MARKERS = ("distance jumps detected", "pose jumps detected")

# The tracker-excursion scan postdates both baselines and is report-only: it
# adds a result per tracking device but no issue (the gating flag ships off),
# so `passed` and `issues` are still compared in full.
EXCURSION_REPORT_ONLY_KEYS = {"excursions"}

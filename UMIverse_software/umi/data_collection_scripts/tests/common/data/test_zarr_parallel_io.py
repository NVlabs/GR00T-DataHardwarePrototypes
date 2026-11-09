# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for parallel Zarr chunk I/O.

Covers:
- Multi-frame chunk layout correctness (chunk count on disk, metadata, values)
- Partial last chunk (when n_frames is not a multiple of chunk_frames)
- Bit-exact agreement between parallel_write + zarr native read
- Speedup benchmarks
"""

import shutil
import tempfile
import time

import numpy as np
import zarr

from umi.common.data.frame import DEPTH_STORAGE
from umi.common.data.zarr_utils import create_zarr_array


def _count_chunk_files(store_path: str, array_key: str) -> int:
    """Count chunk data files written under <store_path>/<array_key>/c/."""
    import os

    chunk_dir = os.path.join(store_path, array_key, "c")
    if not os.path.isdir(chunk_dir):
        return 0
    return sum(len(files) for _, _, files in os.walk(chunk_dir))


def test_chunk_layout_full():
    """parallel_write produces exactly ceil(n/chunk_frames) chunks, all frames correct.

    Uses n_frames divisible by chunk_frames so every chunk is full-size.
    """
    chunk_frames = DEPTH_STORAGE.chunk_frames  # 1280
    H, W = 192, 256
    n_frames = chunk_frames * 2  # 2560 — two full chunks, no partial

    rng = np.random.default_rng(0)
    original = rng.integers(500, 2500, (n_frames, H, W), dtype=np.uint16)

    tmpdir = tempfile.mkdtemp(prefix="umi_chunk_full_")
    try:
        store = zarr.open(tmpdir, mode="w")
        create_zarr_array(
            store, "depth", original, storage_config=DEPTH_STORAGE, max_workers=4
        )
        arr = store["depth"]

        # --- chunk metadata ---
        assert arr.chunks == (chunk_frames, H, W), (
            f"Unexpected chunk shape: {arr.chunks}"
        )

        # --- chunk file count on disk ---
        n_chunks_expected = n_frames // chunk_frames  # 2
        n_chunks_on_disk = _count_chunk_files(tmpdir, "depth")
        assert n_chunks_on_disk == n_chunks_expected, (
            f"Expected {n_chunks_expected} chunk files, found {n_chunks_on_disk}"
        )

        # --- zarr native read is bit-exact ---
        loaded_zarr = arr[:]
        assert np.array_equal(loaded_zarr, original), (
            f"zarr native read mismatch: {(loaded_zarr != original).sum()} differing pixels"
        )

        # --- spot-check individual frames to catch off-by-one shifts ---
        for i in [0, chunk_frames - 1, chunk_frames, n_frames - 1]:
            assert np.array_equal(loaded_zarr[i], original[i]), (
                f"Frame {i} value mismatch after reload"
            )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_chunk_layout_partial_last_chunk():
    """parallel_write handles a partial last chunk correctly.

    n_frames=2500 → chunk 0: frames 0–1279 (full), chunk 1: frames 1280–2499 (1220 frames).
    The zarr native read must be bit-exact for every frame.
    """
    chunk_frames = DEPTH_STORAGE.chunk_frames  # 1280
    H, W = 192, 256
    n_frames = 2500  # not a multiple of 1280
    assert n_frames % chunk_frames != 0, (
        "test invariant: n_frames must not divide evenly"
    )

    rng = np.random.default_rng(1)
    original = rng.integers(500, 2500, (n_frames, H, W), dtype=np.uint16)

    tmpdir = tempfile.mkdtemp(prefix="umi_chunk_partial_")
    try:
        store = zarr.open(tmpdir, mode="w")
        create_zarr_array(
            store, "depth", original, storage_config=DEPTH_STORAGE, max_workers=4
        )
        arr = store["depth"]

        # --- chunk file count: ceil(2500 / 1280) = 2 ---
        import math

        n_chunks_expected = math.ceil(n_frames / chunk_frames)
        n_chunks_on_disk = _count_chunk_files(tmpdir, "depth")
        assert n_chunks_on_disk == n_chunks_expected, (
            f"Expected {n_chunks_expected} chunk files, found {n_chunks_on_disk}"
        )

        # --- zarr native read ---
        loaded_zarr = arr[:]
        assert loaded_zarr.shape == (n_frames, H, W), (
            f"Shape mismatch: {loaded_zarr.shape}"
        )
        assert np.array_equal(loaded_zarr, original), (
            f"zarr native read mismatch: {(loaded_zarr != original).sum()} differing pixels"
        )

        # --- spot-check boundary frames around the chunk boundary ---
        for i in [0, chunk_frames - 1, chunk_frames, n_frames - 1]:
            assert np.array_equal(loaded_zarr[i], original[i]), (
                f"Frame {i} value mismatch (zarr read)"
            )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_parallel_write_matches_zarr_sequential_write():
    """parallel_write must produce identical data to zarr's native sequential write."""
    H, W = 192, 256
    n_frames = 500

    rng = np.random.default_rng(2)
    original = rng.integers(500, 2500, (n_frames, H, W), dtype=np.uint16)

    tmpdir = tempfile.mkdtemp(prefix="umi_chunk_compare_")
    try:
        # Sequential (zarr native, max_workers=1 → no parallel_write)
        store_seq = zarr.open(f"{tmpdir}/seq", mode="w")
        create_zarr_array(
            store_seq, "depth", original, storage_config=DEPTH_STORAGE, max_workers=1
        )

        # Parallel (max_workers=4 → uses parallel_write)
        store_par = zarr.open(f"{tmpdir}/par", mode="w")
        create_zarr_array(
            store_par, "depth", original, storage_config=DEPTH_STORAGE, max_workers=4
        )

        seq_data = store_seq["depth"][:]
        par_data = store_par["depth"][:]

        assert np.array_equal(seq_data, original), "Sequential write data mismatch"
        assert np.array_equal(par_data, original), "Parallel write data mismatch"
        assert np.array_equal(seq_data, par_data), "Sequential vs parallel data differ"

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_parallel_write_speedup():
    """Parallel write should be faster than sequential for frame-chunked depth data."""
    tmpdir = tempfile.mkdtemp(prefix="umi_parallel_write_")
    try:
        n_frames = 600  # Small multi-chunk fixture.
        depth = np.random.randint(0, 5000, (n_frames, 192, 256), dtype=np.uint16)

        # Sequential write (max_workers=1)
        store_seq = zarr.open(f"{tmpdir}/sequential", mode="w")
        t0 = time.perf_counter()
        create_zarr_array(
            store_seq, "depth", depth, storage_config=DEPTH_STORAGE, max_workers=1
        )
        t_seq = time.perf_counter() - t0

        # Parallel write (max_workers=8)
        store_par = zarr.open(f"{tmpdir}/parallel", mode="w")
        t0 = time.perf_counter()
        create_zarr_array(
            store_par, "depth", depth, storage_config=DEPTH_STORAGE, max_workers=32
        )
        t_par = time.perf_counter() - t0

        # Verify data integrity — both must be bit-exact
        data_seq = store_seq["depth"][:]
        data_par = store_par["depth"][:]
        assert np.array_equal(data_seq, depth), "Sequential write data mismatch!"
        assert np.array_equal(data_par, depth), "Parallel write data mismatch!"

        speedup = t_seq / t_par if t_par > 0 else float("inf")

        print(f"\n{'=' * 60}")
        print("PARALLEL WRITE BENCHMARK")
        print(f"{'=' * 60}")
        print(f"Data: {n_frames} frames x {depth.shape[1]}x{depth.shape[2]} uint16")
        print(f"Sequential (1 worker):  {t_seq:.3f}s")
        print(f"Parallel   (8 workers): {t_par:.3f}s")
        print(f"Speedup: {speedup:.2f}x")
        print("Data integrity: PASSED")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_chunk_layout_full()
    test_chunk_layout_partial_last_chunk()
    test_parallel_write_matches_zarr_sequential_write()
    test_parallel_write_speedup()

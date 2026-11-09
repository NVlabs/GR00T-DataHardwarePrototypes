# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Zarr array utilities with parallel chunk writes.

Provides helpers to create Zarr arrays with optional compression and
frame-level chunking, driven by ``ZarrStorageConfig`` metadata on dataclass
fields.

**Why not just use ``array[i] = data[i]`` in a ThreadPoolExecutor?**

zarr v3's synchronous API routes every operation through a single shared
``asyncio`` event loop (see ``zarr.core.sync``).  Multiple threads calling
``array[i] = data[i]`` all submit coroutines to that one loop via
``asyncio.run_coroutine_threadsafe`` and block in a FIFO queue — resulting in
**zero parallelism** plus thread-management overhead.

To get true multi-threaded I/O we bypass zarr's sync layer entirely:

* **Compress** each chunk with ``numcodecs.blosc.Blosc`` directly.
  When ``numcodecs.blosc.use_threads`` is ``False`` (set by zarr at import),
  the C-level ``blosc_compress_ctx`` path is used, which is fully thread-safe.
* **Write** chunk files through plain ``open()`` calls, which also release
  the GIL.

This gives ~3-6x speedup on frame-chunked depth arrays (480x640 uint16).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
from numcodecs.blosc import Blosc
from zarr.storage._local import LocalStore

if TYPE_CHECKING:
    import zarr

    from umi.common.data.frame import ZarrStorageConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_blosc_codec(array: zarr.Array) -> Blosc | None:
    """Build a ``numcodecs.Blosc`` codec matching the array's Blosc settings.

    Scans ``array.metadata.codecs`` for a Blosc codec and mirrors its
    ``cname``, ``clevel``, ``shuffle``, and ``blocksize`` into a standalone
    ``numcodecs.Blosc`` instance suitable for direct encode/decode calls.

    Returns ``None`` when no Blosc codec is found.
    """
    shuffle_map = {
        "noshuffle": Blosc.NOSHUFFLE,
        "shuffle": Blosc.SHUFFLE,
        "bitshuffle": Blosc.BITSHUFFLE,
    }
    for codec in array.metadata.codecs:
        if "blosc" not in type(codec).__name__.lower():
            continue
        cname = codec.cname.value if hasattr(codec.cname, "value") else str(codec.cname)
        shuffle_val = (
            codec.shuffle.value
            if hasattr(codec.shuffle, "value")
            else str(codec.shuffle)
        )
        return Blosc(
            cname=cname,
            clevel=codec.clevel,
            shuffle=shuffle_map.get(shuffle_val, Blosc.BITSHUFFLE),
            blocksize=getattr(codec, "blocksize", 0),
        )
    return None


def _chunk_path(array_dir: str, chunk_idx: int, n_spatial: int) -> str:
    """Return the filesystem path for a chunk file.

    Zarr v3 default layout stores chunk ``(i, 0, 0)`` of a 3-D array at
    ``<array_dir>/c/<i>/0/0``.
    """
    parts = [array_dir, "c", str(chunk_idx)] + ["0"] * n_spatial
    return os.path.join(*parts)


# ---------------------------------------------------------------------------
# Parallel write / read
# ---------------------------------------------------------------------------


def parallel_write(array: zarr.Array, data: np.ndarray, max_workers: int) -> None:
    """Compress and write all chunks in parallel.

    Respects the array's actual chunk shape (e.g. chunk_frames=1280 writes
    multi-frame chunks, not one file per frame).  Compression runs in
    ``max_workers`` threads via ``numcodecs.Blosc.encode`` (GIL released
    inside the C blosc library).

    Falls back to ``array[:] = data`` when the store is not local or the
    array does not use a Blosc codec.
    """
    store = array.store_path.store
    if not isinstance(store, LocalStore):
        array[:] = data
        return

    codec = _extract_blosc_codec(array)
    if codec is None:
        array[:] = data
        return

    array_dir = os.path.join(str(store.root), array.store_path.path)
    n_spatial = data.ndim - 1
    chunk_frames = array.chunks[0]
    n_chunks = int(np.ceil(data.shape[0] / chunk_frames))

    def _write_chunk(chunk_idx: int) -> None:
        start = chunk_idx * chunk_frames
        end = min(start + chunk_frames, data.shape[0])
        chunk_data = data[start:end]
        # Zarr v3 requires every chunk to be full chunk shape — pad the last partial chunk
        if chunk_data.shape[0] < chunk_frames:
            pad = np.zeros(
                (chunk_frames - chunk_data.shape[0], *data.shape[1:]), dtype=data.dtype
            )
            chunk_data = np.concatenate([chunk_data, pad], axis=0)
        compressed = codec.encode(np.ascontiguousarray(chunk_data))
        path = _chunk_path(array_dir, chunk_idx, n_spatial)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(compressed)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_write_chunk, range(n_chunks)))


# ---------------------------------------------------------------------------
# Public: high-level array creation
# ---------------------------------------------------------------------------


def create_zarr_array(
    group,
    key: str,
    data: np.ndarray,
    storage_config: "ZarrStorageConfig | None" = None,
    max_workers: int | None = None,
) -> None:
    """Create a Zarr array in *group* with optional compression and chunking.

    Args:
        group: Zarr group to create the array in.
        key: Name for the array within the group.
        data: Numpy array to store.
        storage_config: Optional ``ZarrStorageConfig`` (from frame field
            metadata) controlling compressors and chunk-by-frame layout.
        max_workers: Thread count for parallel chunk writes.  When ``> 1``
            **and** ``chunk_frames > 0`` is set, chunks are compressed
            and written in parallel threads via :func:`parallel_write`.
            ``None`` or ``1`` uses zarr's default single-call write.
    """
    # Materialize fill-valued chunks so valid all-zero readings remain
    # distinguishable from missing data and are included in chunk-based uploads.
    kwargs: dict = {
        "overwrite": True,
        "config": {"write_empty_chunks": True},
    }

    chunks = None
    if storage_config is not None:
        if storage_config.compressors is not None:
            kwargs["compressors"] = storage_config.compressors
        if storage_config.chunk_frames > 0 and data.ndim >= 2:
            chunks = (storage_config.chunk_frames, *data.shape[1:])
            kwargs["chunks"] = chunks

    use_parallel = (
        storage_config is not None
        and max_workers is not None
        and max_workers > 1
        and chunks is not None
        and data.shape[0] > 1
        and data.shape[0] >= storage_config.chunk_frames
    )

    if use_parallel:
        arr = group.create_array(key, shape=data.shape, dtype=data.dtype, **kwargs)
        parallel_write(arr, data, max_workers)
    else:
        try:
            group.create_array(key, data=data, **kwargs)
        except TypeError:
            # Zarr 3.0 requires shape and dtype before data assignment; it does not
            # accept the data keyword in create_array.
            arr = group.create_array(key, shape=data.shape, dtype=data.dtype, **kwargs)
            arr[...] = data

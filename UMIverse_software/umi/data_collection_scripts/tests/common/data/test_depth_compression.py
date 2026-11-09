# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test depth compression via frame metadata-driven Zarr storage.

Verifies that:
1. ZarrStorageConfig on RGBDCameraFrame.depth drives Blosc compression
2. Data is lossless (bit-exact round-trip)
3. End-to-end MockCamera + VideoRecorder stores depth with correct codecs
"""

import os
import shutil
import sys
import tempfile
import time
from unittest.mock import MagicMock

import numpy as np
import zarr

# Mock optional hardware dependencies that aren't installed in dev
for _mod_name in ["record3d", "dexcontrol", "rtde_control", "rtde_receive"]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

from umi.common.data.frame import (  # noqa: E402
    DEPTH_STORAGE,
    RGBDCameraFrame,
    get_zarr_storage_config,
)
from umi.common.data.zarr_utils import create_zarr_array  # noqa: E402
from umi.common.hardware_sdk.camera.mock_camera import MockCamera  # noqa: E402
from umi.common.recording.video_recorder import VideoRecorder  # noqa: E402


def test_depth_compression_with_mock_camera():
    """End-to-end test: MockCamera with depth -> VideoRecorder -> Zarr with compression."""
    record_path = tempfile.mkdtemp(prefix="umi_depth_test_")

    try:
        # Create mock camera with depth enabled
        camera = MockCamera(
            device_name="test_rgbd_cam",
            camera_resolution=(256, 192),
            fps=30,
            frame_pattern="gradient",
            enable_depth=True,
            depth_min_mm=200,
            depth_max_mm=5000,
        )

        # Create VideoRecorder with RGBDCameraFrame
        recorder = VideoRecorder(
            record_fps=30,
            stream_fps=30,
            record_path=record_path,
            data_sources=[camera],
            frame_data_class=RGBDCameraFrame,
            verbose=False,
            info_verbose=False,
            warning_verbose=False,
        )

        recorder.start_streaming()
        time.sleep(0.5)

        # Record ~15 frames (enough to verify depth storage; reduced for faster test)
        num_frames_target = 15
        recorder.reset_episode_recording()
        recorder.start_recording()
        time.sleep(num_frames_target / 30 + 0.2)
        recorder.stop_recording()
        recorder.save_recordings()

        recorder.stop_streaming()

        # Open the zarr store and verify depth data
        store = zarr.open(record_path, mode="r")
        episode_key = [k for k in store.group_keys() if k.startswith("episode_")][0]
        episode = store[episode_key]
        cam_group = episode["test_rgbd_cam"]

        # Check depth array exists
        assert "depth" in list(cam_group.array_keys()), (
            "Depth array not found in Zarr store"
        )

        depth_array = cam_group["depth"]
        depth_data = depth_array[:]

        print(f"\n{'=' * 60}")
        print("DEPTH COMPRESSION TEST RESULTS")
        print(f"{'=' * 60}")
        print(f"Depth array shape: {depth_data.shape}")
        print(f"Depth dtype: {depth_data.dtype}")
        print(f"Depth value range: [{depth_data.min()}, {depth_data.max()}]")
        print(f"Zero pixels: {(depth_data == 0).sum() / depth_data.size * 100:.1f}%")
        print(f"Number of frames: {depth_data.shape[0]}")

        # Verify dtype is uint16
        assert depth_data.dtype == np.uint16, f"Expected uint16, got {depth_data.dtype}"

        # Verify shape is (N, H, W)
        assert depth_data.ndim == 3, f"Expected 3D array, got {depth_data.ndim}D"
        assert depth_data.shape[1] == 192, (
            f"Expected height 192, got {depth_data.shape[1]}"
        )
        assert depth_data.shape[2] == 256, (
            f"Expected width 256, got {depth_data.shape[2]}"
        )

        # Check compression metadata
        metadata = depth_array.metadata
        print("\nZarr array metadata:")
        print(f"  Chunks: {depth_array.chunks}")
        print(f"  Codecs: {metadata.codecs}")

        # Verify Blosc codec is used
        codec_names = [type(c).__name__ for c in metadata.codecs]
        has_blosc = any("blosc" in name.lower() for name in codec_names)
        print(f"  Has Blosc codec: {has_blosc}")
        assert has_blosc, f"Expected Blosc codec, got codecs: {codec_names}"

        # Verify chunk size matches DEPTH_STORAGE
        from umi.common.data.frame import DEPTH_STORAGE as _DS

        if _DS.chunk_frames > 0:
            assert depth_array.chunks[0] == _DS.chunk_frames, (
                f"Expected chunk_frames={_DS.chunk_frames}, got {depth_array.chunks[0]}"
            )

        # Measure storage sizes
        raw_size_bytes = depth_data.nbytes
        cam_zarr_path = os.path.join(record_path, episode_key, "test_rgbd_cam", "depth")
        compressed_size = 0
        for dirpath, _, filenames in os.walk(cam_zarr_path):
            for f in filenames:
                compressed_size += os.path.getsize(os.path.join(dirpath, f))

        compression_ratio = (
            raw_size_bytes / compressed_size if compressed_size > 0 else 0
        )

        print(f"\n{'=' * 60}")
        print("STORAGE COMPARISON")
        print(f"{'=' * 60}")
        print(
            f"Raw uncompressed size:  {raw_size_bytes / 1024:.1f} KB ({raw_size_bytes:,} bytes)"
        )
        print(
            f"Compressed size on disk: {compressed_size / 1024:.1f} KB ({compressed_size:,} bytes)"
        )
        print(f"Compression ratio:      {compression_ratio:.2f}x")
        print(f"Space saved:            {(1 - 1 / compression_ratio) * 100:.1f}%")

        assert compression_ratio > 1.0, "Compression should reduce size"

        # --- Data integrity: re-open the store from disk and verify ---
        depth_first_load = depth_data.copy()

        store_reloaded = zarr.open(record_path, mode="r")
        depth_reloaded = store_reloaded[episode_key]["test_rgbd_cam"]["depth"][:]

        assert np.array_equal(depth_first_load, depth_reloaded), (
            "Data integrity failure: reloaded depth differs from first load!\n"
            f"  Max abs diff: {np.max(np.abs(depth_first_load.astype(np.int32) - depth_reloaded.astype(np.int32)))}\n"
            f"  Mismatched pixels: {(depth_first_load != depth_reloaded).sum()}"
        )
        print("\nData integrity (reload from disk): PASSED - bit-exact match")

        print("\nAll assertions passed!")

    finally:
        shutil.rmtree(record_path, ignore_errors=True)


def test_create_zarr_array_with_storage_config():
    """Unit test: verify create_zarr_array applies ZarrStorageConfig correctly."""
    tmpdir = tempfile.mkdtemp(prefix="umi_zarr_test_")
    try:
        store = zarr.open(tmpdir, mode="w")

        # Verify frame metadata lookup works
        config = get_zarr_storage_config(RGBDCameraFrame, "depth")
        assert config is DEPTH_STORAGE, f"Expected DEPTH_STORAGE, got {config}"

        # Non-storage fields return None
        assert get_zarr_storage_config(RGBDCameraFrame, "rgb") is None
        assert get_zarr_storage_config(RGBDCameraFrame, "pose") is None

        # Create a synthetic depth array
        depth = np.random.randint(0, 5000, (10, 192, 256), dtype=np.uint16)

        # Write with storage config from frame metadata
        create_zarr_array(store, "depth", depth, storage_config=config)

        # Verify it was stored with Blosc compression
        arr = store["depth"]
        codec_names = [type(c).__name__ for c in arr.metadata.codecs]
        has_blosc = any("blosc" in name.lower() for name in codec_names)
        assert has_blosc, f"Expected Blosc for depth, got {codec_names}"
        assert arr.chunks[0] == DEPTH_STORAGE.chunk_frames, (
            f"Expected chunk_frames={DEPTH_STORAGE.chunk_frames}, got {arr.chunks[0]}"
        )

        # Verify lossless round-trip
        assert np.array_equal(arr[:], depth), "Data mismatch after compression!"

        # Write without storage config — should use Zarr defaults
        pose = np.random.randn(10, 4, 4).astype(np.float32)
        create_zarr_array(store, "pose", pose)
        assert np.array_equal(store["pose"][:], pose), "Pose data mismatch!"

        print("create_zarr_array unit test passed!")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_depth_data_integrity_round_trip():
    """Verify compressed depth data survives a full write-close-reopen cycle.

    Writes known depth patterns to Zarr with DEPTH_STORAGE compression,
    closes the store, re-opens from disk, and checks bit-exact equality.
    Tests multiple patterns: zeros, max values, gradients, and random noise.
    """
    tmpdir = tempfile.mkdtemp(prefix="umi_depth_integrity_")
    try:
        H, W = 192, 256
        patterns = {
            "zeros": np.zeros((5, H, W), dtype=np.uint16),
            "max_val": np.full((5, H, W), 65535, dtype=np.uint16),
            "gradient": np.tile(np.linspace(0, 65535, W, dtype=np.uint16), (5, H, 1)),
            "random": np.random.randint(0, 65536, (10, H, W), dtype=np.uint16),
            "sparse": _make_sparse_depth(10, H, W, fill_ratio=0.3),
        }

        # Write all patterns with compression
        store = zarr.open(tmpdir, mode="w")
        for name, data in patterns.items():
            create_zarr_array(store, name, data, storage_config=DEPTH_STORAGE)

        # Force close by dropping references
        del store

        # Re-open from disk and verify each pattern
        store = zarr.open(tmpdir, mode="r")
        print(f"\n{'=' * 60}")
        print("DEPTH DATA INTEGRITY ROUND-TRIP TEST")
        print(f"{'=' * 60}")

        for name, original in patterns.items():
            loaded = store[name][:]
            match = np.array_equal(original, loaded)
            print(
                f"  {name:12s}: shape={loaded.shape}, dtype={loaded.dtype}, match={match}"
            )
            assert match, (
                f"Data integrity failure for pattern '{name}'!\n"
                f"  Max abs diff: {np.max(np.abs(original.astype(np.int32) - loaded.astype(np.int32)))}\n"
                f"  Mismatched pixels: {(original != loaded).sum()} / {original.size}"
            )

        print("\nAll round-trip integrity checks passed!")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _make_sparse_depth(n_frames, h, w, fill_ratio=0.3):
    """Create sparse depth maps (mostly zeros with some valid pixels)."""
    data = np.zeros((n_frames, h, w), dtype=np.uint16)
    rng = np.random.default_rng(42)
    mask = rng.random((n_frames, h, w)) < fill_ratio
    data[mask] = rng.integers(200, 5000, size=mask.sum(), dtype=np.uint16)
    return data


if __name__ == "__main__":
    test_create_zarr_array_with_storage_config()
    test_depth_data_integrity_round_trip()
    test_depth_compression_with_mock_camera()

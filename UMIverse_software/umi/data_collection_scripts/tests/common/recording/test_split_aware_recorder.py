# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the split-aware NumericRecorder + BaseRecorder.

Covers backwards-compat (non-splittable frames go through the legacy 1D
layout) and the splittable path (per-(source, key) recording threads
and `episode/{source}_{key}/` zarr layout).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import numpy as np
import zarr

from umi.common.data.frame import (
    Frame,
    NumericFrame,
    register_frame,
)
from umi.common.hardware_sdk.sensor_interface import SensorInterface
from umi.common.recording.numeric_recorder import NumericRecorder

# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


@register_frame
@dataclass
class _SingleFieldFrame(NumericFrame):
    """Non-splittable single-stream frame for backwards-compat tests."""

    value: Optional[np.ndarray] = field(default=None, metadata={"interpolate": True})


@register_frame
@dataclass
class _SplitSubA(NumericFrame):
    """Sub-stream A: scalar wrapped as (1,) array."""

    a_value: Optional[np.ndarray] = field(default=None, metadata={"interpolate": True})


@register_frame
@dataclass
class _SplitSubB(NumericFrame):
    """Sub-stream B: vector."""

    b_vector: Optional[np.ndarray] = field(default=None, metadata={"interpolate": True})


@register_frame
@dataclass
class _SplitCombinedFrame(NumericFrame):
    """Splittable combined frame with two sub-streams."""

    a_value: Optional[np.ndarray] = field(default=None, metadata={"interpolate": True})
    b_vector: Optional[np.ndarray] = field(default=None, metadata={"interpolate": True})

    def split(self) -> Dict[str, Frame]:
        return {
            "a": _SplitSubA(
                a_value=self.a_value,
                capture_time=self.capture_time,
                receive_time=self.receive_time,
                frame_source=self.frame_source,
            ),
            "b": _SplitSubB(
                b_vector=self.b_vector,
                capture_time=self.capture_time,
                receive_time=self.receive_time,
                frame_source=self.frame_source,
            ),
        }

    @classmethod
    def splittable_keys(cls) -> List[str]:
        return ["a", "b"]

    @classmethod
    def sub_frame_classes(cls) -> Dict[str, Type[Frame]]:
        return {"a": _SplitSubA, "b": _SplitSubB}


class _FakeSensor(SensorInterface):
    """Minimal SensorInterface emitting deterministic frames at a configurable rate."""

    def __init__(self, device_name: str, frame_factory):
        super().__init__(device_name=device_name, latency=0.0)
        self._factory = frame_factory
        self._counter = 0
        self.is_running = False

    def start_streaming(self) -> None:
        self.is_running = True

    def stop_streaming(self) -> None:
        self.is_running = False

    def get_frame(self):
        if not self.is_running:
            return None
        i = self._counter
        self._counter += 1
        return self._factory(i)


def _single_factory(i: int) -> _SingleFieldFrame:
    return _SingleFieldFrame(
        value=np.array([float(i)], dtype=np.float32),
        receive_time=time.monotonic(),
        frame_source="single",
    )


def _split_factory(i: int) -> _SplitCombinedFrame:
    return _SplitCombinedFrame(
        a_value=np.array([float(i)], dtype=np.float32),
        b_vector=np.array([float(i), -float(i)], dtype=np.float32),
        receive_time=time.monotonic(),
        frame_source="split",
    )


# Non-splittable frames retain one sub-stream per source.


def _record_briefly(recorder: NumericRecorder, duration: float = 0.3) -> None:
    recorder.start_streaming()
    time.sleep(0.1)  # let frames accumulate
    recorder.reset_episode_recording()
    recorder.start_recording()
    time.sleep(duration)
    recorder.stop_recording()
    recorder.stop_streaming()
    recorder.save_recordings()


def test_non_splittable_writes_to_episode_source_group(tmp_path):
    """Backwards-compat: with a non-splittable frame, zarr layout is
    `episode/{source}/{field}` -- no trailing _key suffix. The 2D internal
    layout collapses to length-1 inner lists."""
    rec_path = str(tmp_path / "rec")
    sensor = _FakeSensor("singlesrc", _single_factory)
    rec = NumericRecorder(
        record_fps=60,
        stream_fps=60,
        record_path=rec_path,
        numeric_sources=[sensor],
        frame_data_class=_SingleFieldFrame,
        verbose=False,
        info_verbose=False,
        warning_verbose=False,
    )

    # Non-splittable accumulators still use the source/key shape with one inner entry.
    assert rec._split_keys == [""]
    assert len(rec.frame_buffers) == 1
    rec.reset_episode_recording()
    assert len(rec.episode_frames) == 1
    assert len(rec.episode_frames[0]) == 1

    _record_briefly(rec)

    # Disk layout: episode_0/singlesrc/value (not singlesrc_/value)
    root = zarr.open(rec_path, mode="r")
    groups = list(root.group_keys())
    # Recording is saved to a *.pending folder; any group will do.
    episode_groups = [g for g in groups if g.startswith("episode_")]
    assert len(episode_groups) == 1
    episode = root[episode_groups[0]]
    src_subgroups = list(episode.group_keys())
    assert src_subgroups == ["singlesrc"], (
        f"Expected episode/singlesrc/ but got {src_subgroups}"
    )
    arrays = list(episode["singlesrc"].array_keys())
    assert "value" in arrays, f"Expected 'value' field in singlesrc, got {arrays}"


def test_non_splittable_recording_threads_count_matches_sources(tmp_path):
    """One recording thread per source for non-splittable frames."""
    rec_path = str(tmp_path / "rec")
    sensors = [_FakeSensor(f"s{i}", _single_factory) for i in range(2)]
    rec = NumericRecorder(
        record_fps=60,
        stream_fps=60,
        record_path=rec_path,
        numeric_sources=sensors,
        frame_data_class=_SingleFieldFrame,
        verbose=False,
        info_verbose=False,
        warning_verbose=False,
    )
    rec.start_streaming()
    time.sleep(0.05)
    rec.reset_episode_recording()
    rec.start_recording()
    # 2 sources x 1 split key = 2 record threads
    assert len(rec.recording_threads) == 2
    rec.stop_recording()
    rec.stop_streaming()


# ---------------------------------------------------------------------------
# Split-aware: per-(source, key) buffers, threads, and zarr groups
# ---------------------------------------------------------------------------


def test_splittable_writes_two_zarr_groups(tmp_path):
    """Splittable frame -> episode/{source}_a/ and episode/{source}_b/."""
    rec_path = str(tmp_path / "rec")
    sensor = _FakeSensor("splsrc", _split_factory)
    rec = NumericRecorder(
        record_fps=60,
        stream_fps=60,
        record_path=rec_path,
        numeric_sources=[sensor],
        frame_data_class=_SplitCombinedFrame,
        verbose=False,
        info_verbose=False,
        warning_verbose=False,
    )

    assert rec._split_keys == ["a", "b"]
    # Inner episode_frames has length 2 per source
    rec.reset_episode_recording()
    assert len(rec.episode_frames[0]) == 2
    # Each inner dict has fields from the corresponding sub-class only.
    assert "a_value" in rec.episode_frames[0][0]
    assert "b_vector" not in rec.episode_frames[0][0]
    assert "b_vector" in rec.episode_frames[0][1]
    assert "a_value" not in rec.episode_frames[0][1]

    _record_briefly(rec, duration=0.5)

    root = zarr.open(rec_path, mode="r")
    episode_groups = [g for g in root.group_keys() if g.startswith("episode_")]
    assert len(episode_groups) == 1
    episode = root[episode_groups[0]]
    src_subgroups = set(episode.group_keys())
    assert src_subgroups == {"splsrc_a", "splsrc_b"}, (
        f"Expected splsrc_a/ + splsrc_b/, got {src_subgroups}"
    )
    a_fields = set(episode["splsrc_a"].array_keys())
    b_fields = set(episode["splsrc_b"].array_keys())
    assert "a_value" in a_fields
    assert "b_vector" in b_fields
    # Bound split-stream frame skew relative to the number recorded so scheduling
    # variation is tolerated without hiding divergent sub-streams.
    a_count = episode["splsrc_a"]["a_value"].shape[0]
    b_count = episode["splsrc_b"]["b_vector"].shape[0]
    assert a_count > 0 and b_count > 0
    allowed_skew = max(2, int(0.1 * max(a_count, b_count)))
    assert abs(a_count - b_count) <= allowed_skew, (
        f"splsrc_a wrote {a_count} frames, splsrc_b wrote {b_count}"
    )


def test_splittable_recording_threads_count_is_sources_times_keys(tmp_path):
    """Each (source, sub-key) gets its own recording thread."""
    rec_path = str(tmp_path / "rec")
    sensors = [_FakeSensor(f"s{i}", _split_factory) for i in range(2)]
    rec = NumericRecorder(
        record_fps=60,
        stream_fps=60,
        record_path=rec_path,
        numeric_sources=sensors,
        frame_data_class=_SplitCombinedFrame,
        verbose=False,
        info_verbose=False,
        warning_verbose=False,
    )
    rec.start_streaming()
    time.sleep(0.05)
    rec.reset_episode_recording()
    rec.start_recording()
    # 2 sources x 2 split keys = 4 record threads
    assert len(rec.recording_threads) == 4
    rec.stop_recording()
    rec.stop_streaming()


def test_splittable_multi_source_each_pair_lands_in_own_group(tmp_path):
    """Two splittable sources produce 4 disk groups: s0_a, s0_b, s1_a, s1_b."""
    rec_path = str(tmp_path / "rec")
    sensors = [_FakeSensor(f"s{i}", _split_factory) for i in range(2)]
    rec = NumericRecorder(
        record_fps=60,
        stream_fps=60,
        record_path=rec_path,
        numeric_sources=sensors,
        frame_data_class=_SplitCombinedFrame,
        verbose=False,
        info_verbose=False,
        warning_verbose=False,
    )
    _record_briefly(rec, duration=0.3)

    root = zarr.open(rec_path, mode="r")
    episode_groups = [g for g in root.group_keys() if g.startswith("episode_")]
    episode = root[episode_groups[0]]
    src_subgroups = set(episode.group_keys())
    assert src_subgroups == {"s0_a", "s0_b", "s1_a", "s1_b"}


def test_splittable_stream_thread_unchanged_one_per_source(tmp_path):
    """Stream threads are still 1:1 with sources -- the per-(source, key)
    fan-out happens in the recording thread, not the stream thread."""
    rec_path = str(tmp_path / "rec")
    sensors = [_FakeSensor(f"s{i}", _split_factory) for i in range(3)]
    rec = NumericRecorder(
        record_fps=60,
        stream_fps=60,
        record_path=rec_path,
        numeric_sources=sensors,
        frame_data_class=_SplitCombinedFrame,
        verbose=False,
        info_verbose=False,
        warning_verbose=False,
    )
    rec.start_streaming()
    time.sleep(0.05)
    # 3 sources -> 3 stream threads regardless of split keys
    assert len(rec.stream_threads) == 3
    # frame_buffers is still 1D per source
    assert len(rec.frame_buffers) == 3
    rec.stop_streaming()


def test_peek_latest_frame_returns_combined_frame_unchanged(tmp_path):
    """Public peek API still returns the combined frame -- splitting is a
    recording-thread concern, not a buffering one."""
    rec_path = str(tmp_path / "rec")
    sensor = _FakeSensor("s0", _split_factory)
    rec = NumericRecorder(
        record_fps=60,
        stream_fps=60,
        record_path=rec_path,
        numeric_sources=[sensor],
        frame_data_class=_SplitCombinedFrame,
        verbose=False,
        info_verbose=False,
        warning_verbose=False,
    )
    rec.start_streaming()
    time.sleep(0.1)
    frame = rec.peek_latest_frame(0)
    assert frame is not None
    assert isinstance(frame, _SplitCombinedFrame)  # not a sub-frame
    assert frame.a_value is not None
    assert frame.b_vector is not None
    rec.stop_streaming()

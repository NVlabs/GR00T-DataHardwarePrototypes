# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RecorderManager stamps each episode with its local recording timestamp."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import zarr

from umi.common.recording import record_manager
from umi.common.recording.constants import PENDING_SUFFIX
from umi.common.recording.record_manager import RecorderManager


def _manager(record_path, episode_id=0, started_at=None):
    # Bypass the heavy constructor; _save_episode_timestamp only uses these.
    # `episode_id` is a property derived from `recorders`, so fake one recorder.
    rm = RecorderManager.__new__(RecorderManager)
    rm.record_path = record_path
    rm.recorders = [SimpleNamespace(episode_id=episode_id)]
    rm._episode_start_local = started_at
    rm.verbose = False
    return rm


def _make_pending_episode(tmp_path, episode_id=0):
    record_path = tmp_path / "session"
    store = zarr.open(str(record_path), mode="a")
    store.create_group(f"episode_{episode_id}{PENDING_SUFFIX}")
    return str(record_path)


def test_writes_start_and_end_timestamps(tmp_path):
    record_path = _make_pending_episode(tmp_path)

    rm = _manager(record_path, started_at="2026-08-04T13:45:12-07:00")
    rm._save_episode_timestamp()

    attrs = zarr.open(record_path, mode="r")[f"episode_0{PENDING_SUFFIX}"].attrs
    assert attrs["recorded_at"] == "2026-08-04T13:45:12-07:00"
    # The end timestamp uses local time with an explicit UTC offset.
    end = datetime.fromisoformat(attrs["recorded_end_at"])
    assert end.utcoffset() is not None


def test_falls_back_to_now_when_start_unknown(tmp_path):
    record_path = _make_pending_episode(tmp_path)

    rm = _manager(record_path, started_at=None)
    rm._save_episode_timestamp()

    attrs = zarr.open(record_path, mode="r")[f"episode_0{PENDING_SUFFIX}"].attrs
    assert datetime.fromisoformat(attrs["recorded_at"]).utcoffset() is not None


def test_second_episode_does_not_inherit_first_episode_start(tmp_path, monkeypatch):
    # A failed second start must not reuse the first episode's recorded_at.
    # Use distinct fake timestamps to avoid second-resolution collisions.
    stamps = iter(["T0-start", "T0-save", "T1-save"])
    monkeypatch.setattr(record_manager, "_local_timestamp", lambda: next(stamps))

    record_path = _make_pending_episode(tmp_path, episode_id=0)
    rm = _manager(record_path, episode_id=0)

    rm.recorders = [SimpleNamespace(episode_id=0, start_recording=lambda **_: True)]
    assert rm.start_recording() is True
    rm._save_episode_timestamp()

    _make_pending_episode(tmp_path, episode_id=1)
    rm.recorders = [SimpleNamespace(episode_id=1, start_recording=lambda **_: False)]
    assert rm.start_recording() is False
    rm._save_episode_timestamp()

    store = zarr.open(record_path, mode="r")
    ep0 = store[f"episode_0{PENDING_SUFFIX}"].attrs
    ep1 = store[f"episode_1{PENDING_SUFFIX}"].attrs
    assert (ep0["recorded_at"], ep0["recorded_end_at"]) == ("T0-start", "T0-save")
    # Episode 1 fell back to its own save time -- no carry-over from episode 0.
    assert (ep1["recorded_at"], ep1["recorded_end_at"]) == ("T1-save", "T1-save")


def test_noop_when_pending_episode_absent(tmp_path):
    record_path = str(tmp_path / "session")
    zarr.open(record_path, mode="a")

    rm = _manager(record_path, episode_id=3)
    rm._save_episode_timestamp()  # must not raise

    assert f"episode_3{PENDING_SUFFIX}" not in zarr.open(record_path, mode="r")

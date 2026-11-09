# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the seal marker in ``umi.common.recording.session_seal``.

A sealed session folder is one nothing will add an episode to again. The
marker is read by an existence check rather than a parse, so a crash midway
through the write still closes the session -- re-opening it is the one outcome
the whole mechanism exists to prevent. ``read_seal`` does parse, because its
callers want the reason; the two disagreeing on a corrupt marker is deliberate.

``write_seal`` refuses rather than raises for the three states of the world it
will not seal (empty session, already sealed, missing folder) and raises only
for an unknown reason, which is a call-site bug.
"""

import json

import pytest

from umi.common.recording import session_seal


def test_unsealed_folder_reads_as_unsealed(tmp_path):
    assert session_seal.folder_is_sealed(tmp_path) is False
    assert session_seal.read_seal(tmp_path) is None


def test_write_seal_creates_a_readable_marker(tmp_path):
    payload = session_seal.write_seal(
        tmp_path,
        reason="day_roll",
        episode_count=3,
        session_uid="a1b2c3d4e5f6",
        station_id="gear-umi-07",
    )
    assert payload is not None
    assert session_seal.folder_is_sealed(tmp_path) is True

    on_disk = json.loads((tmp_path / session_seal.SEAL_FILENAME).read_text())
    assert on_disk["reason"] == "day_roll"
    assert on_disk["episode_count"] == 3
    assert on_disk["session_uid"] == "a1b2c3d4e5f6"
    assert on_disk["station_id"] == "gear-umi-07"
    assert on_disk["sealed_at"].endswith("+00:00")


def test_write_seal_refuses_an_empty_session(tmp_path):
    assert session_seal.write_seal(tmp_path, reason="day_roll", episode_count=0) is None
    assert session_seal.folder_is_sealed(tmp_path) is False


def test_write_seal_never_overwrites_an_existing_seal(tmp_path):
    first = session_seal.write_seal(tmp_path, reason="operator", episode_count=1)
    second = session_seal.write_seal(tmp_path, reason="ttl", episode_count=1)
    assert second is None
    assert session_seal.read_seal(tmp_path)["reason"] == first["reason"] == "operator"


def test_write_seal_rejects_an_unknown_reason(tmp_path):
    with pytest.raises(ValueError):
        session_seal.write_seal(tmp_path, reason="because", episode_count=1)


def test_a_corrupt_marker_still_counts_as_sealed(tmp_path):
    (tmp_path / session_seal.SEAL_FILENAME).write_text("{not json")
    assert session_seal.folder_is_sealed(tmp_path) is True
    assert session_seal.read_seal(tmp_path) is None


def test_a_missing_folder_is_not_sealed_and_cannot_be_sealed(tmp_path):
    missing = tmp_path / "nope"
    assert session_seal.folder_is_sealed(missing) is False
    assert session_seal.write_seal(missing, reason="ttl", episode_count=2) is None


def test_folder_is_sealed_expands_a_tilde_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "sess").mkdir()
    session_seal.write_seal(tmp_path / "sess", reason="operator", episode_count=1)
    assert session_seal.folder_is_sealed("~/sess") is True


def test_health_check_is_a_reason_a_caller_may_declare(tmp_path):
    """The end-of-shift audit sweep is its own trigger, not a bulk `operator`.

    Reusing `operator` would make one deliberate button press on one session
    indistinguishable from a sweep that closed a whole day, and the reason
    table exists precisely so a marker says which trigger fired.
    """
    assert "health_check" in session_seal.REASONS
    payload = session_seal.write_seal(tmp_path, reason="health_check", episode_count=1)
    assert payload is not None and payload["reason"] == "health_check"

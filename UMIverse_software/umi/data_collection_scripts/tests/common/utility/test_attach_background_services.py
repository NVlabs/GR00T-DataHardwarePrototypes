# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared background-service wiring.

Covers ``attach_background_services`` (construct + attach + start the
BackgroundAligner per config) and
``RecorderManager.shutdown_background_services``. No hardware, no worker
processes: the service class is monkeypatched with a recording dummy so the
tests check the wiring (enable/disable gating, task-name validation,
construction kwargs, start/shutdown) in isolation.
"""

import types

import pytest

import umi.common.utility.background_aligner as ba_mod
from umi.common.recording.record_manager import RecorderManager
from umi.common.utility.background_aligner import attach_background_services


class _DummyAligner:
    """Records its construction kwargs and start()/shutdown() calls."""

    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.shut = False
        type(self).instances.append(self)

    def start(self):
        self.started = True

    def shutdown(self):
        self.shut = True


@pytest.fixture
def patched_services(monkeypatch):
    _DummyAligner.instances = []
    # The helper looks the class up as a module global at call time, so
    # patching the module attribute is sufficient.
    monkeypatch.setattr(ba_mod, "BackgroundAligner", _DummyAligner)
    return _DummyAligner


def _manager():
    # The helper only sets the one attribute; avoid building real recorders.
    return types.SimpleNamespace(background_aligner=None)


def test_noop_when_alignment_disabled(patched_services):
    mgr = _manager()
    cfg = {"alignment": {"enabled": False}, "meta": {"task_name": "smoke"}}
    attach_background_services(mgr, cfg, "/tmp/sess")
    assert mgr.background_aligner is None
    # Disabled => never constructed.
    assert _DummyAligner.instances == []


def test_raises_when_enabled_without_task_name(patched_services):
    mgr = _manager()
    cfg = {"alignment": {"enabled": True}, "meta": {}}
    with pytest.raises(ValueError, match="Task name not set"):
        attach_background_services(mgr, cfg, "/tmp/sess")


def test_attaches_aligner_with_overrides(patched_services):
    DummyAligner = patched_services
    mgr = _manager()
    cfg = {
        "alignment": {
            "enabled": True,
            "index_sources": "right_umi_camera",
            "copy_videos": False,
        },
        "meta": {"task_name": "smoke"},
    }
    attach_background_services(mgr, cfg, "/tmp/sess")

    al = mgr.background_aligner
    assert al is DummyAligner.instances[0]
    assert al.started is True
    assert al.kwargs["session_path"] == "/tmp/sess"
    assert al.kwargs["task_name"] == "smoke"
    assert al.kwargs["index_sources"] == "right_umi_camera"
    assert al.kwargs["copy_videos"] is False
    # No override: the aligner resolves the session's own aligned folder.
    assert al.kwargs["aligned_dir"] is None


@pytest.mark.parametrize("alignment, expected", [({"uniform_fps": 15}, 15), ({}, 30)])
def test_aligner_gets_the_configs_uniform_fps(patched_services, alignment, expected):
    """The config's alignment.uniform_fps sets the grid; without it the station
    still aligns onto the even 30 fps grid."""
    DummyAligner = patched_services
    mgr = _manager()
    cfg = {
        "alignment": {
            "enabled": True,
            "index_sources": "right_umi_camera",
            **alignment,
        },
        "meta": {"task_name": "smoke"},
    }
    attach_background_services(mgr, cfg, "/tmp/sess")

    assert DummyAligner.instances[0].kwargs["uniform_fps"] == expected


def test_shutdown_drains_the_aligner(patched_services):
    al = _DummyAligner()
    mgr = types.SimpleNamespace(background_aligner=al)
    # Call the unbound method with a stand-in self — it only touches the one
    # attribute, so no full RecorderManager needs to be built.
    RecorderManager.shutdown_background_services(mgr)
    assert al.shut is True


def test_a_failing_drain_does_not_break_shutdown(patched_services):
    """This runs inside `record_data.py`'s `finally`. An exception here would
    mask whatever brought the recorder down."""

    class _Exploding(_DummyAligner):
        instances: list = []

        def shutdown(self):
            raise RuntimeError("worker wedged")

    mgr = types.SimpleNamespace(background_aligner=_Exploding())

    RecorderManager.shutdown_background_services(mgr)  # must not raise


def test_shutdown_noop_when_none():
    mgr = types.SimpleNamespace(background_aligner=None)
    # Must not raise.
    RecorderManager.shutdown_background_services(mgr)

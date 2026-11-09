# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The CALIBRATION stage, one frame at a time, in record_data.

`_tick_calibration_stage` holds the stage while the board is being solved and
after it converges -- the DONE overlay stays up until the operator presses
`s` -- and on a timeout shakes the window, resets the session and drops back
to IDLE. Each outcome is reported once in the terminal.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from real_scripts.data_collection.pre_episode_checks.camera_calibration import (
    CameraRelativeCalibrationCheck,
)
from real_scripts.data_collection.record_data import (
    RecordingStage,
    _tick_calibration_stage,
)
from umi.common.recording.pre_episode_checks import PreEpisodeContext


def _check(loop_timeout_sec=30.0):
    return CameraRelativeCalibrationCheck(
        reference_camera="ref",
        cameras=["ref", "other"],
        run_every_n_episodes=10,
        marker={"mode": "aruco", "marker_size": 0.1875},
        align_stable_required=5,
        loop_timeout_sec=loop_timeout_sec,
    )


def _ctx():
    return PreEpisodeContext(
        latest_frames={},
        cfg={},
        recorder_manager=SimpleNamespace(episode_id=0),
    )


class _Visualizer:
    def __init__(self):
        self.shakes = 0

    def trigger_shake(self, **kwargs):
        self.shakes += 1


def test_a_calibration_still_converging_stays_in_the_stage(capsys):
    chk = _check()
    chk.start_session()
    chk.tick = lambda ctx: None
    viz = _Visualizer()

    stage = _tick_calibration_stage(chk, _ctx(), viz)

    assert stage is RecordingStage.CALIBRATION
    assert viz.shakes == 0
    out = capsys.readouterr().out
    assert "Converged" not in out and "Timed out" not in out


def test_convergence_holds_the_stage():
    chk = _check()
    chk.start_session()
    chk.tick = lambda ctx: setattr(chk, "_converged", True)
    viz = _Visualizer()

    stage = _tick_calibration_stage(chk, _ctx(), viz)

    # The operator still has to press `s`; the DONE overlay keeps rendering
    # until they do, so the stage must not advance on its own.
    assert stage is RecordingStage.CALIBRATION
    assert viz.shakes == 0


def test_convergence_is_reported_once_not_on_every_later_frame(capsys):
    chk = _check()
    chk.start_session()
    chk.tick = lambda ctx: setattr(chk, "_converged", True)
    viz = _Visualizer()

    for _ in range(3):
        _tick_calibration_stage(chk, _ctx(), viz)

    assert capsys.readouterr().out.count("Converged") == 1


def test_timeout_returns_to_idle_and_resets_the_session():
    chk = _check(loop_timeout_sec=30.0)
    chk.start_session()
    chk._session_start = time.monotonic() - 999
    chk.tick = lambda ctx: None
    viz = _Visualizer()

    stage = _tick_calibration_stage(chk, _ctx(), viz)

    assert stage is RecordingStage.IDLE
    assert viz.shakes == 1
    assert chk._session_start is None, "the timed-out session should be reset"


def test_a_converged_session_does_not_also_time_out(capsys):
    # Convergence latches `_converged`, which is what keeps a slow calibration
    # that finished on its last legs from announcing both outcomes.
    chk = _check(loop_timeout_sec=30.0)
    chk.start_session()
    chk._session_start = time.monotonic() - 999
    chk.tick = lambda ctx: setattr(chk, "_converged", True)
    viz = _Visualizer()

    stage = _tick_calibration_stage(chk, _ctx(), viz)

    assert stage is RecordingStage.CALIBRATION
    out = capsys.readouterr().out
    assert "Converged" in out and "Timed out" not in out

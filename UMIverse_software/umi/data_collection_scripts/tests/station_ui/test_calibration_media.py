# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedded charts show the worker's measurements without changing verdicts."""

from station_ui.calibration_media import tracker_chart


def sample(t, value, tracked=True):
    return {
        "event": "sample",
        "t": t,
        "arms": {"left": {"disp": value, "tracked": tracked}},
    }


def test_trace_history_is_bounded_and_uses_worker_thresholds():
    chart = tracker_chart(
        [
            {"event": "ready", "move_m": 0.08, "still_m": 0.02},
            *[sample(t, 0.04) for t in range(200)],
        ]
    )
    assert chart["start"] == 50
    assert chart["end"] == 199
    assert chart["move_mm"] == 80
    assert chart["still_mm"] == 20
    assert len(chart["traces"][0]["paths"][0].split()) == 150


def test_tracking_loss_makes_a_gap_instead_of_inventing_zero_movement():
    chart = tracker_chart([sample(0, 0.01), sample(1, 0.03, False), sample(2, 0.05)])
    assert len(chart["traces"][0]["paths"]) == 2


def test_invalid_samples_cannot_inject_svg_or_nonfinite_coordinates():
    assert tracker_chart([sample("<script>", 0.01)]) is None
    chart = tracker_chart([sample(0, float("nan")), sample(1, "bad"), sample(2, 0.02)])
    assert len(chart["traces"][0]["paths"]) == 1
    assert "nan" not in chart["traces"][0]["paths"][0]

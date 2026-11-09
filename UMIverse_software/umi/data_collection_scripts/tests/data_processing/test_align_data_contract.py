# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for align_data: settings from config.yaml, even output grid,
output location, quarantine refusal, and the station's background aligner.

Written from the spec only; the fixtures build tiny raw sessions whose index
camera has uneven frame times (a 45 Hz stream that keeps 2 of every 3 frames).
"""

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
import zarr
from click.testing import CliRunner

BASE_T = 1000.0


def _index_times(n=90):
    k = np.arange(n)
    return BASE_T + k[k % 3 != 2] / 45.0


def _make_session(session_dir, cameras=("right_umi_camera",), alignment=None, n_eps=1):
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True)
    store = zarr.open(str(session_dir), mode="w")
    for ep in range(n_eps):
        g = store.create_group(f"episode_{ep}")
        for i, cam in enumerate(cameras):
            t = _index_times() + 0.001 * i
            c = g.create_group(cam)
            c["capture_time"] = t
            c["receive_time"] = t
            c["rgb"] = np.zeros((len(t), 4, 4, 3), dtype=np.uint8)
        t = BASE_T - 0.1 + np.arange(500) / 200.0
        s = g.create_group("right_gripper")
        s["capture_time"] = t
        s["receive_time"] = t
        s["value"] = np.random.randn(len(t), 3)
    config = {
        "recording": {
            "video_frame_type": "RGBCameraFrame",
            "numeric_frame_type": "NumericFrame",
        },
        "video_sources": [
            {"name": c, "frame_type": "RGBCameraFrame", "latency": 0.0} for c in cameras
        ],
        "numeric_sources": [
            {"name": "right_gripper", "frame_type": "NumericFrame", "latency": 0.0}
        ],
    }
    if alignment is not None:
        config["alignment"] = alignment
    (session_dir / "config.yaml").write_text(yaml.safe_dump(config))
    return session_dir


def _run(*args):
    from real_scripts.data_processing.align_data import main

    return CliRunner().invoke(main, [str(a) for a in args], catch_exceptions=True)


def _raw(tmp_path, **kw):
    return _make_session(tmp_path / "raw" / "task_a" / "sess_1", **kw)


def _aligned(tmp_path):
    return tmp_path / "aligned" / "task_a" / "sess_1"


def _report(out_dir):
    return json.loads((Path(out_dir) / "data_align_report.json").read_text())


def _assert_even(out_dir, fps, ep=0):
    episode = zarr.open(str(out_dir), mode="r")[f"episode_{ep}"]
    ts = episode["valid_timestamps"][:]
    assert len(ts) > 10, f"too few aligned rows: {len(ts)}"
    np.testing.assert_allclose(np.diff(ts), 1.0 / fps, atol=1e-6)
    assert float(episode.attrs["uniform_fps"]) == pytest.approx(fps)


# ---- 1. CLI surface -------------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        ["--index-sources", "right_umi_camera"],
        ["-i", "right_umi_camera"],
        ["--downsample", "2"],
        ["--uniform-fps", "30"],
        ["--no-copy-videos"],
    ],
)
def test_removed_options_are_usage_errors(tmp_path, extra):
    raw = _raw(tmp_path, alignment={"copy_videos": False})
    result = _run("-d", raw, "-s", *extra)
    assert result.exit_code == 2, result.output
    assert not _aligned(tmp_path).exists()


def test_missing_data_dir_is_usage_error():
    result = _run("-s")
    assert result.exit_code == 2, result.output


def test_short_and_long_options_accepted(tmp_path):
    raw = _raw(tmp_path, alignment={"copy_videos": False}, n_eps=2)
    out = tmp_path / "custom_out"
    result = _run(
        "--data-dir", raw, "--output-dir", out, "--episodes", 1, "-n", 1, "--sequential"
    )
    assert result.exit_code == 0, result.output
    _assert_even(out, 30.0, ep=1)


# ---- 2/3. settings from config.yaml, even grid ----------------------------


@pytest.mark.parametrize(
    "alignment,expected_fps",
    [
        ({}, 30.0),
        ({"uniform_fps": None}, 30.0),
        ({"uniform_fps": 0}, 30.0),
        ({"uniform_fps": 20}, 20.0),
        ({"uniform_fps": 15.0}, 15.0),
    ],
)
def test_uniform_fps_from_config(tmp_path, alignment, expected_fps):
    raw = _raw(tmp_path, alignment={**alignment, "copy_videos": False})
    result = _run("-d", raw, "-s")
    assert result.exit_code == 0, result.output
    _assert_even(_aligned(tmp_path), expected_fps)
    assert float(
        _report(_aligned(tmp_path))["summary"]["uniform_fps"]
    ) == pytest.approx(expected_fps)


def test_negative_uniform_fps_does_not_succeed_silently(tmp_path):
    raw = _raw(tmp_path, alignment={"uniform_fps": -10, "copy_videos": False})
    result = _run("-d", raw, "-s")
    ep_aligned = (_aligned(tmp_path) / "episode_0" / "valid_timestamps").exists()
    assert result.exit_code != 0 or not ep_aligned, result.output
    assert not ep_aligned, "an episode was aligned with a negative uniform_fps"


@pytest.mark.parametrize(
    "cameras,alignment,expected",
    [
        (("right_umi_camera", "left_umi_camera"), {}, "right_umi_camera"),
        (("left_umi_camera",), {}, "left_umi_camera"),
        (
            ("right_umi_camera", "left_umi_camera"),
            {"index_sources": "missing_cam, left_umi_camera,right_umi_camera"},
            "left_umi_camera",
        ),
    ],
)
def test_index_source_is_first_recorded_listed(tmp_path, cameras, alignment, expected):
    raw = _raw(tmp_path, cameras=cameras, alignment={**alignment, "copy_videos": False})
    result = _run("-d", raw, "-s")
    assert result.exit_code == 0, result.output
    assert _report(_aligned(tmp_path))["summary"]["index_source"] == expected
    _assert_even(_aligned(tmp_path), 30.0)


# ---- 4. output location ---------------------------------------------------


def test_default_output_is_aligned_counterpart(tmp_path):
    raw = _raw(tmp_path, alignment={"copy_videos": False})
    result = _run("-d", raw, "-s")
    assert result.exit_code == 0, result.output
    out = _aligned(tmp_path)
    assert (out / "data_align_report.json").is_file()
    assert (out / "episode_0" / "valid_timestamps").exists()
    summary = _report(out)["summary"]
    assert {"index_source", "uniform_fps"} <= set(summary)


# ---- 5. quarantine refusal ------------------------------------------------


def _tree_digest(root):
    root = Path(root)
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_quarantine_session_is_refused(tmp_path):
    q = _make_session(
        tmp_path / "quarantine" / "task_a" / "sess_1", alignment={"copy_videos": False}
    )
    # A real aligned output for the same task/session already exists.
    raw = _raw(tmp_path, alignment={"copy_videos": False})
    assert _run("-d", raw, "-s").exit_code == 0
    before = _tree_digest(tmp_path / "aligned")
    assert before

    result = _run("-d", q, "-s")
    assert result.exit_code != 0, result.output
    assert "quarantine" in (result.output + str(result.exception or "")).lower()
    assert _tree_digest(tmp_path / "aligned") == before


def test_quarantine_session_creates_nothing(tmp_path):
    q = _make_session(
        tmp_path / "quarantine" / "task_a" / "sess_1", alignment={"copy_videos": False}
    )
    result = _run("-d", q, "-s")
    assert result.exit_code != 0, result.output
    assert not (tmp_path / "aligned").exists()


# ---- 6. shared settings module --------------------------------------------


def test_alignment_settings_module(tmp_path):
    from umi.common.utility import alignment_settings as m

    assert isinstance(m.DEFAULT_INDEX_SOURCES, tuple)
    assert m.DEFAULT_INDEX_SOURCES == ("right_umi_camera", "left_umi_camera")
    assert m.DEFAULT_UNIFORM_FPS == 30.0

    d = m.alignment_settings({})
    assert isinstance(d, m.AlignmentSettings)
    assert (d.index_sources, d.uniform_fps, d.copy_videos) == (
        m.DEFAULT_INDEX_SOURCES,
        30.0,
        True,
    )
    for fps in (None, 0):
        assert (
            m.alignment_settings({"alignment": {"uniform_fps": fps}}).uniform_fps
            == 30.0
        )

    s = m.alignment_settings(
        {"alignment": {"index_sources": "a,b", "uniform_fps": 20, "copy_videos": False}}
    )
    assert (s.index_sources, s.uniform_fps, s.copy_videos) == (("a", "b"), 20.0, False)
    with pytest.raises(ValueError):
        m.alignment_settings({"alignment": {"uniform_fps": -1}})

    assert m.session_alignment_settings(tmp_path) == d  # no config.yaml
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"alignment": {"uniform_fps": 10}})
    )
    assert m.session_alignment_settings(tmp_path).uniform_fps == 10.0


# ---- 7. station path ------------------------------------------------------


@pytest.mark.parametrize("cfg_fps,expected", [(None, 30.0), (24, 24.0)])
def test_attach_background_services_passes_uniform_fps(
    tmp_path, monkeypatch, cfg_fps, expected
):
    from unittest import mock

    from umi.common.utility import background_aligner as ba

    made = []

    class DummyAligner:
        def __init__(self, *args, **kwargs):
            made.append(kwargs)

        def start(self):
            pass

    monkeypatch.setattr(ba, "BackgroundAligner", DummyAligner)
    alignment = {
        "enabled": True,
        "index_sources": "right_umi_camera",
        "copy_videos": False,
    }
    if cfg_fps is not None:
        alignment["uniform_fps"] = cfg_fps
    cfg = {"alignment": alignment, "meta": {"task_name": "task_a"}}
    ba.attach_background_services(
        mock.MagicMock(), cfg, str(tmp_path / "raw" / "task_a" / "s")
    )
    assert len(made) == 1, made
    assert "downsample" not in made[0], made[0]
    assert float(made[0]["uniform_fps"]) == pytest.approx(expected)


def test_background_aligner_rejects_downsample_keyword():
    from umi.common.utility.background_aligner import BackgroundAligner

    params = inspect.signature(BackgroundAligner.__init__).parameters
    assert "downsample" not in params
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()), (
        "BackgroundAligner.__init__ takes **kwargs, so downsample= would not raise TypeError"
    )

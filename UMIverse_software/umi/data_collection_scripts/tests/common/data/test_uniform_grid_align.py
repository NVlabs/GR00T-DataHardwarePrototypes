# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Uniform-grid alignment tests for DataBuffer (``uniform_fps``).

The recorder grabs the newest frame at each 30 Hz record tick from a 45 Hz
stream, so every recorded source keeps 2 of every 3 stream frames and its
frame times alternate ~22/~44 ms. Legacy alignment uses the index camera's own
(uneven) frame times as the output timeline; ``uniform_fps`` replaces it with an
even grid over the same window and resamples every source at the grid times.

Expected values are computed from the spec here (np.interp, argmin), not by
calling DataBuffer internals.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml
import zarr
from click.testing import CliRunner
from scipy.spatial.transform import Rotation

from real_scripts.data_processing import align_data
from umi.common.data.data_buffer import DataBuffer

STREAM_DT = 1 / 45
T0 = 1_700_000_000.0  # epoch-sized times, like real receive_time


def _decimated_times(n_stream: int, phase: float, drop_mod: int) -> np.ndarray:
    """45 Hz stream times keeping 2 of every 3 frames (the recorder pattern)."""
    k = np.arange(n_stream)
    return T0 + phase + k[k % 3 != drop_mod] * STREAM_DT


def _poses(times: np.ndarray) -> np.ndarray:
    t = times - T0
    poses = np.tile(np.eye(4), (len(t), 1, 1))
    poses[:, :3, :3] = Rotation.from_euler("z", 0.5 * t[:, None]).as_matrix()
    poses[:, 0, 3] = np.sin(2.0 * t)
    poses[:, 1, 3] = 0.1 * t**2
    return poses


def _make_episode(data_dir: Path) -> dict:
    """Index camera, a second camera, a pose tracker and a gripper; latency 0."""
    times = {
        "main_camera": _decimated_times(140, 0.000, 2),
        "wrist_camera": _decimated_times(140, 0.007, 0),
        "tracker": _decimated_times(140, 0.003, 1),
        "gripper": _decimated_times(140, 0.011, 2),
    }
    episode = zarr.open(str(data_dir), mode="w").create_group("episode_0")
    for name in ("main_camera", "wrist_camera"):
        g = episode.create_group(name)
        g["capture_time"] = times[name]
        g["receive_time"] = times[name]
    g = episode.create_group("tracker")
    g["capture_time"] = times["tracker"]
    g["receive_time"] = times["tracker"]
    g["pose"] = _poses(times["tracker"])
    g = episode.create_group("gripper")
    g["receive_time"] = times["gripper"]
    g["distance"] = 50.0 + 30.0 * np.sin(3.0 * (times["gripper"] - T0))

    config = {
        "recording": {
            "video_frame_type": "RGBCameraFrame",
            "numeric_frame_type": "GripperFrame",
        },
        "video_sources": [
            {"name": n, "latency": 0.0}
            for n in ("main_camera", "wrist_camera", "tracker")
        ],
        "numeric_sources": [{"name": "gripper", "latency": 0.0}],
    }
    with open(data_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    return times


def _align(data_dir: Path, out_dir: Path, **kwargs) -> zarr.Group:
    DataBuffer(
        data_dir=str(data_dir),
        target_dir=str(out_dir),
        index_source_name="main_camera",
        copy_videos=False,
        **kwargs,
    ).align_episode(0)
    return zarr.open(str(out_dir), mode="r")["episode_0"]


def _arrays(group: zarr.Group, prefix: str = "") -> dict:
    out = {}
    for name, arr in group.arrays():
        out[prefix + name] = arr[:]
    for name, sub in group.groups():
        out.update(_arrays(sub, f"{prefix}{name}/"))
    return out


def _window(times: dict) -> tuple:
    return max(t.min() for t in times.values()), min(t.max() for t in times.values())


@pytest.fixture
def episode_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "raw"
        data_dir.mkdir()
        times = _make_episode(data_dir)
        yield data_dir, Path(tmpdir), times


def test_uniform_fps_none_is_legacy(episode_dirs):
    data_dir, tmp, times = episode_dirs
    default = _arrays(_align(data_dir, tmp / "default"))
    explicit = _arrays(_align(data_dir, tmp / "explicit", uniform_fps=None))

    assert default.keys() == explicit.keys()
    for key in default:
        assert default[key].dtype == explicit[key].dtype, key
        assert np.array_equal(default[key], explicit[key]), key

    # With resampling disabled, rows retain the index camera's uneven frame times.
    lo, hi = _window(times)
    mask = (times["main_camera"] >= lo) & (times["main_camera"] <= hi)
    assert np.array_equal(default["valid_timestamps"], times["main_camera"][mask])
    assert np.array_equal(
        default["main_camera/aligned_frame_indices"], np.where(mask)[0]
    )
    gaps_ms = np.round(np.diff(default["valid_timestamps"]) * 1000)
    assert set(gaps_ms) == {22.0, 44.0}


def test_uniform_grid_is_even_and_inside_window(episode_dirs):
    data_dir, tmp, times = episode_dirs
    legacy = _align(data_dir, tmp / "legacy")
    out = _align(data_dir, tmp / "uniform", uniform_fps=30)
    legacy_ts = legacy["valid_timestamps"][:]
    ts = out["valid_timestamps"][:]

    np.testing.assert_allclose(np.diff(ts), 1 / 30, atol=1e-6)
    assert legacy_ts[0] <= ts[0] < legacy_ts[0] + 1 / 30
    assert legacy_ts[-1] - 1 / 30 < ts[-1] <= legacy_ts[-1]
    assert out.attrs["uniform_fps"] == 30
    assert "uniform_fps" not in legacy.attrs


def test_uniform_grid_keeps_output_format(episode_dirs):
    data_dir, tmp, _ = episode_dirs
    legacy = _arrays(_align(data_dir, tmp / "legacy"))
    uniform = _arrays(_align(data_dir, tmp / "uniform", uniform_fps=30))

    assert legacy.keys() == uniform.keys()
    for key in legacy:
        assert legacy[key].dtype == uniform[key].dtype, key
        assert legacy[key].shape[1:] == uniform[key].shape[1:], key
    # valid_indices stays the index camera's in-window mask.
    assert np.array_equal(legacy["valid_indices"], uniform["valid_indices"])


def test_uniform_grid_cameras_use_nearest_frame(episode_dirs):
    data_dir, tmp, times = episode_dirs
    out = _arrays(_align(data_dir, tmp / "uniform", uniform_fps=30))
    ts = out["valid_timestamps"]

    for cam in ("main_camera", "wrist_camera", "tracker"):
        expected = np.array([np.argmin(np.abs(times[cam] - t)) for t in ts])
        assert np.array_equal(out[f"{cam}/aligned_frame_indices"], expected), cam

    # The index camera's 2-of-3 frames alternate half a stream period apart
    # relative to the record ticks; a grid phased between them leaves every
    # frame a quarter stream period (~5.6 ms) off its row: no repeats, no skips.
    idx = out["main_camera/aligned_frame_indices"]
    assert np.all(np.diff(idx) == 1)
    offset_ms = np.abs(times["main_camera"][idx] - ts) * 1000
    # (an odd frame count tilts the circular mean slightly toward one side)
    np.testing.assert_allclose(offset_ms, 250 / 45, atol=0.25)


def test_uniform_grid_numeric_sources_interpolated(episode_dirs):
    data_dir, tmp, times = episode_dirs
    out = _arrays(_align(data_dir, tmp / "uniform", uniform_fps=30))
    ts = out["valid_timestamps"]

    distance = 50.0 + 30.0 * np.sin(3.0 * (times["gripper"] - T0))
    np.testing.assert_allclose(
        out["gripper/distance_interp"], np.interp(ts, times["gripper"], distance)
    )

    pose = out["tracker/pose_interp"]
    assert pose.shape == (len(ts), 4, 4)
    raw = _poses(times["tracker"])
    for axis in range(3):
        np.testing.assert_allclose(
            pose[:, axis, 3], np.interp(ts, times["tracker"], raw[:, axis, 3])
        )


def _align_cli(data_dir: Path, out_dir: Path) -> zarr.Group:
    result = CliRunner().invoke(
        align_data.main, ["-d", str(data_dir), "-o", str(out_dir), "-s"]
    )
    assert result.exit_code == 0, result.output
    return zarr.open(str(out_dir), mode="r")["episode_0"]


def _set_config_alignment(data_dir: Path, **alignment) -> None:
    path = data_dir / "config.yaml"
    config = yaml.safe_load(path.read_text())
    config["alignment"] = {
        "index_sources": "main_camera",
        "copy_videos": False,
        **alignment,
    }
    path.write_text(yaml.dump(config))


def _assert_same(expected: dict, actual: dict) -> None:
    assert expected.keys() == actual.keys()
    for key in expected:
        assert expected[key].dtype == actual[key].dtype, key
        assert np.array_equal(expected[key], actual[key]), key


def test_align_data_defaults_to_the_even_30fps_grid(episode_dirs):
    """A session config without alignment.uniform_fps still gets even rows."""
    data_dir, tmp, _ = episode_dirs
    _set_config_alignment(data_dir)
    expected = _arrays(_align(data_dir, tmp / "direct", uniform_fps=30))
    group = _align_cli(data_dir, tmp / "cli")

    assert group.attrs["uniform_fps"] == 30
    np.testing.assert_allclose(np.diff(group["valid_timestamps"][:]), 1 / 30, atol=1e-6)
    _assert_same(expected, _arrays(group))


def test_align_data_takes_uniform_fps_from_the_session_config(episode_dirs):
    data_dir, tmp, _ = episode_dirs
    _set_config_alignment(data_dir, uniform_fps=15)
    expected = _arrays(_align(data_dir, tmp / "direct", uniform_fps=15))
    group = _align_cli(data_dir, tmp / "cli")

    assert group.attrs["uniform_fps"] == 15
    _assert_same(expected, _arrays(group))


@pytest.mark.parametrize(
    "quarantine, aligned",
    [
        ("quarantine/task/session", "aligned"),  # inside the output folder
        ("session_quarantine", "session_aligned"),  # a session outside it
    ],
)
def test_align_data_refuses_a_quarantine_session(tmp_path, quarantine, aligned):
    """Quarantine episode numbers restart at 0, and a quarantine folder maps to
    the same aligned folder as its raw session: aligning it would overwrite the
    passed episodes."""
    session = tmp_path / quarantine
    session.mkdir(parents=True)
    _make_episode(session)

    result = CliRunner().invoke(align_data.main, ["-d", str(session), "-s"])

    assert result.exit_code != 0
    assert "quarantine" in result.output
    assert not (tmp_path / aligned).exists()


def test_uniform_fps_zero_is_legacy_and_negative_is_rejected(episode_dirs):
    """0 means legacy on every path (station, health check, CLI), not a
    division by zero; a negative rate is a config typo, not 0-row episodes."""
    data_dir, tmp, _ = episode_dirs
    legacy = _arrays(_align(data_dir, tmp / "legacy"))
    group = _align(data_dir, tmp / "zero", uniform_fps=0)
    zero = _arrays(group)

    assert "uniform_fps" not in group.attrs
    assert legacy.keys() == zero.keys()
    for key in legacy:
        assert np.array_equal(legacy[key], zero[key]), key
    with pytest.raises(ValueError, match="positive"):
        _align(data_dir, tmp / "negative", uniform_fps=-30)


def test_window_shorter_than_the_phase_offset_keeps_one_row(tmp_path):
    """Two index frames 22 ms apart: the phase-fitted grid start (~28 ms in)
    lies past the window, which must not leave a 0-row episode."""
    data_dir = tmp_path / "raw"
    episode = zarr.open(str(data_dir), mode="w").create_group("episode_0")
    g = episode.create_group("main_camera")
    g["capture_time"] = g["receive_time"] = T0 + np.array([0.0, STREAM_DT])
    g = episode.create_group("gripper")
    g["receive_time"] = T0 + np.linspace(-0.1, 0.1, 21)
    g["distance"] = np.linspace(0.0, 20.0, 21)
    config = {
        "recording": {
            "video_frame_type": "RGBCameraFrame",
            "numeric_frame_type": "GripperFrame",
        },
        "video_sources": [{"name": "main_camera", "latency": 0.0}],
        "numeric_sources": [{"name": "gripper", "latency": 0.0}],
    }
    (data_dir / "config.yaml").write_text(yaml.dump(config))

    out = _align(data_dir, tmp_path / "uniform", uniform_fps=30)

    assert out["valid_timestamps"][:].tolist() == [T0]
    assert out["main_camera"]["aligned_frame_indices"][:].tolist() == [0]
    np.testing.assert_allclose(out["gripper"]["distance_interp"][:], [10.0])


def test_uniform_fps_rejects_downsample(episode_dirs):
    data_dir, tmp, _ = episode_dirs
    with pytest.raises(ValueError, match="uniform_fps"):
        DataBuffer(
            data_dir=str(data_dir),
            target_dir=str(tmp / "out"),
            index_source_name="main_camera",
            downsample_rate=2,
            uniform_fps=30,
        )

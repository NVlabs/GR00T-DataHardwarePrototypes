# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video confirmation of tracker-excursion candidates.

A rigidly mounted camera moves with the hand. When the tracker reports an
80 mm out-and-back that the hand never made, the camera image does not move
with it: image motion per mm of tracker step at the excursion edges collapses
relative to the surrounding context (gain ratio < 0.20 -> confirmed). When the
image follows the tracker (ratio >= 0.40) the motion was real -> rejected.

Episodes here are laid out the way the station writes them and the way
`check_motion_alignment` reads them: ``{data_dir}/{episode_name}/`` is the
episode zarr group, each camera has a group with per-frame timestamps, and its
video sits beside it as ``{camera}.mp4``. Videos are 160x120, 3 s, 30 fps.

Run with: uv run pytest tests/data_processing/test_excursion_confirm.py -v
"""

import copy

import numpy as np
import pytest

from real_scripts.data_processing.data_check.excursion_confirm import (
    confirm_excursions,
)
from real_scripts.data_processing.data_check.tracking import (
    check_tracking_excursions,
)

cv2 = pytest.importorskip("cv2")
zarr = pytest.importorskip("zarr")

FPS = 30.0
N = 90
W, H = 160, 120
HAND_SPEED_M_PER_S = 0.10  # ~3.3 mm/frame of real hand motion
GLITCH_START, GLITCH_LEN, GLITCH_M = 45, 8, 0.080
PX_PER_MM = 0.3  # image shift per mm of camera motion: 1 px/frame, 24 px glitch

EPISODE = "episode_0"
TRACKER = "left_tracking_camera"
WRIST = "left_umi_camera"
FINAL_STATES = {"confirmed", "rejected", "unverified"}


def _timestamps():
    # Epoch-like clock: the confirmer must not assume episode-relative time.
    return 1_760_000_000.0 + np.arange(N) / FPS


def _hand_positions(ts):
    """Where the hand (and so the rigid camera) really is: steady x drift."""
    t = ts - ts[0]
    return np.stack(
        [HAND_SPEED_M_PER_S * t, np.zeros_like(t), np.full_like(t, 0.3)], axis=1
    )


def _tracker_positions(ts):
    """What the tracker reports: the hand plus an 80 mm out-and-back on y."""
    positions = _hand_positions(ts)
    positions[GLITCH_START : GLITCH_START + GLITCH_LEN, 1] += GLITCH_M
    return positions


def _poses(positions):
    poses = np.tile(np.eye(4), (len(positions), 1, 1))
    poses[:, :3, 3] = positions
    return poses


def _write_video(path, camera_positions_m):
    """Textured plane translating with the camera: shift = PX_PER_MM * mm."""
    rng = np.random.default_rng(7)
    big = rng.integers(0, 255, size=(H * 3, W * 3), dtype=np.uint8)
    big = cv2.GaussianBlur(big, (7, 7), 0)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter cannot open mp4 output")
    for x_m, y_m, _ in camera_positions_m:
        dx = -PX_PER_MM * x_m * 1000.0
        dy = -PX_PER_MM * y_m * 1000.0
        m = np.float32([[1, 0, dx - W // 2], [0, 1, dy - H // 2]])
        frame = cv2.warpAffine(big, m, (W, H), flags=cv2.INTER_LINEAR)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()


def _build(tmp_path, video_follows_tracker, cameras=(WRIST,)):
    """Episode on disk; returns (episode_group, data_dir, events, timestamps).

    video_follows_tracker=False: the camera sees only the real hand motion,
    i.e. the tracker glitch has no image counterpart (a true excursion).
    True: the image moves with every tracker step, glitch included.
    """
    ts = _timestamps()
    tracker = _tracker_positions(ts)
    episode_dir = tmp_path / EPISODE
    root = zarr.open_group(str(episode_dir), mode="w")
    root.attrs["latency"] = {TRACKER: 0.0, WRIST: 0.0}

    group = root.create_group(TRACKER)
    group.create_array("pose", data=_poses(tracker))
    group.create_array("capture_time", data=ts)
    group.create_array("receive_time", data=ts)
    if WRIST in cameras:
        cam = root.create_group(WRIST)
        cam.create_array("capture_time", data=ts)
        cam.create_array("receive_time", data=ts)

    seen = tracker if video_follows_tracker else _hand_positions(ts)
    for camera in cameras:
        _write_video(episode_dir / f"{camera}.mp4", seen)

    detected = check_tracking_excursions(_poses(tracker), ts)
    events = [e for e in detected["events"] if e["state"] == "candidate"]
    assert len(events) == 1, f"precondition: one candidate, got {detected}"
    return zarr.open_group(str(episode_dir), mode="r"), str(tmp_path), events, ts


def _confirm(episode, data_dir, events, ts, params=None):
    return confirm_excursions(
        episode, data_dir, EPISODE, TRACKER, events, ts, params=params
    )


def _assert_confirm_block(event):
    confirm = event["confirm"]
    assert confirm["method"] == "video_gain_v1"
    assert confirm["score"] is None or type(confirm["score"]) is float
    assert isinstance(confirm["cameras"], dict)


# --------------------------------------------------------------------------
# verdicts
# --------------------------------------------------------------------------


class TestVerdict:
    def test_image_does_not_follow_the_glitch_confirmed(self, tmp_path):
        episode, data_dir, events, ts = _build(tmp_path, video_follows_tracker=False)

        out = _confirm(episode, data_dir, events, ts)

        assert len(out) == 1
        assert out[0]["state"] == "confirmed"
        _assert_confirm_block(out[0])
        assert WRIST in out[0]["confirm"]["cameras"]
        assert out[0]["confirm"]["score"] is not None
        assert out[0]["confirm"]["score"] < 0.20

    def test_image_follows_the_tracker_rejected(self, tmp_path):
        episode, data_dir, events, ts = _build(tmp_path, video_follows_tracker=True)

        out = _confirm(episode, data_dir, events, ts)

        assert out[0]["state"] == "rejected"
        assert out[0]["reason"] == "video"
        _assert_confirm_block(out[0])
        assert out[0]["confirm"]["score"] is not None
        assert out[0]["confirm"]["score"] >= 0.40

    def test_both_witness_cameras_are_used(self, tmp_path):
        episode, data_dir, events, ts = _build(
            tmp_path, video_follows_tracker=False, cameras=(TRACKER, WRIST)
        )

        out = _confirm(episode, data_dir, events, ts)

        assert out[0]["state"] == "confirmed"
        assert {TRACKER, WRIST} <= set(out[0]["confirm"]["cameras"])

    def test_event_fields_are_kept(self, tmp_path):
        episode, data_dir, events, ts = _build(tmp_path, video_follows_tracker=False)

        out = _confirm(episode, data_dir, events, ts)

        for key, value in events[0].items():
            if key not in ("state", "reason", "confirm"):
                assert out[0][key] == value, key

    def test_thresholds_are_params(self, tmp_path):
        """A confirm_max nobody can get under turns a confirmation into
        unverified -- the thresholds come from params, not constants."""
        episode, data_dir, events, ts = _build(tmp_path, video_follows_tracker=False)

        out = _confirm(
            episode, data_dir, events, ts, params={"confirm_max": 0.0, "reject_min": 10}
        )

        assert out[0]["state"] == "unverified"


class TestUnverified:
    def test_missing_video_is_unverified(self, tmp_path):
        episode, data_dir, events, ts = _build(
            tmp_path, video_follows_tracker=False, cameras=()
        )

        out = _confirm(episode, data_dir, events, ts)

        assert out[0]["state"] == "unverified"
        _assert_confirm_block(out[0])
        assert out[0]["confirm"]["score"] is None

    def test_wrong_data_dir_is_unverified(self, tmp_path):
        episode, _, events, ts = _build(tmp_path, video_follows_tracker=False)

        out = _confirm(episode, str(tmp_path / "nowhere"), events, ts)

        assert out[0]["state"] == "unverified"

    def test_unreadable_video_is_unverified(self, tmp_path):
        episode, data_dir, events, ts = _build(tmp_path, video_follows_tracker=False)
        (tmp_path / EPISODE / f"{WRIST}.mp4").write_bytes(b"not a video")

        out = _confirm(episode, data_dir, events, ts)

        assert out[0]["state"] == "unverified"

    def test_cap_on_decoded_candidates(self, tmp_path):
        """Past max_events per device the rest are unverified, not decoded."""
        episode, data_dir, events, ts = _build(tmp_path, video_follows_tracker=False)
        many = [copy.deepcopy(events[0]) for _ in range(3)]

        out = _confirm(episode, data_dir, many, ts, params={"max_events": 1})

        states = [e["state"] for e in out]
        assert len(out) == 3
        assert states.count("confirmed") == 1
        assert states.count("unverified") == 2


# --------------------------------------------------------------------------
# pass-through, purity, robustness
# --------------------------------------------------------------------------


class TestPassThrough:
    def test_rejected_events_pass_through_unchanged(self, tmp_path):
        episode, data_dir, events, ts = _build(tmp_path, video_follows_tracker=False)
        rejected = dict(events[0], state="rejected", reason="rotation")
        expected = copy.deepcopy(rejected)

        out = _confirm(episode, data_dir, [rejected], ts)

        assert out == [expected]

    def test_order_and_length_preserved(self, tmp_path):
        episode, data_dir, events, ts = _build(tmp_path, video_follows_tracker=False)
        rejected = dict(events[0], state="rejected", reason="rotation")

        out = _confirm(episode, data_dir, [rejected, events[0]], ts)

        assert len(out) == 2
        assert out[0] == rejected
        assert out[1]["state"] in FINAL_STATES

    def test_empty_event_list(self, tmp_path):
        episode, data_dir, _, ts = _build(tmp_path, video_follows_tracker=False)

        assert _confirm(episode, data_dir, [], ts) == []


class TestPurity:
    def test_inputs_not_mutated(self, tmp_path):
        episode, data_dir, events, ts = _build(tmp_path, video_follows_tracker=False)
        events_before = copy.deepcopy(events)
        ts_before = ts.copy()

        out = _confirm(episode, data_dir, events, ts)

        assert events == events_before
        np.testing.assert_array_equal(ts, ts_before)
        assert out is not events
        assert out[0] is not events[0]


class TestNeverRaises:
    @pytest.fixture
    def candidate(self, tmp_path):
        _, _, events, _ = _build(tmp_path, video_follows_tracker=False)
        return events[0]

    def test_no_episode_group(self, candidate, tmp_path):
        out = confirm_excursions(
            None, str(tmp_path), EPISODE, TRACKER, [candidate], None
        )
        assert [e["state"] for e in out] == ["unverified"]

    def test_device_without_a_side(self, candidate, tmp_path):
        episode = zarr.open_group(str(tmp_path / EPISODE), mode="r")
        out = confirm_excursions(
            episode,
            str(tmp_path),
            EPISODE,
            "head_tracking_camera",
            [candidate],
            _timestamps(),
        )
        assert [e["state"] for e in out] == ["unverified"]

    @pytest.mark.parametrize(
        "event",
        [
            {"state": "candidate"},
            {"state": "candidate", "i0": 10_000, "i1": 10_005},
            {"state": "candidate", "t_out_s": float("nan"), "t_back_s": None},
        ],
    )
    def test_malformed_candidate_is_unverified(self, tmp_path, event):
        episode, data_dir, _, ts = _build(tmp_path, video_follows_tracker=False)

        out = _confirm(episode, data_dir, [event], ts)

        assert len(out) == 1
        assert out[0]["state"] == "unverified"

    @pytest.mark.parametrize("timestamps", [None, np.zeros(0), np.full(N, np.nan)])
    def test_bad_timestamps(self, candidate, tmp_path, timestamps):
        episode = zarr.open_group(str(tmp_path / EPISODE), mode="r")
        out = confirm_excursions(
            episode, str(tmp_path), EPISODE, TRACKER, [candidate], timestamps
        )
        assert len(out) == 1
        assert out[0]["state"] in FINAL_STATES

    def test_garbage_params(self, candidate, tmp_path):
        episode = zarr.open_group(str(tmp_path / EPISODE), mode="r")
        out = confirm_excursions(
            episode,
            str(tmp_path),
            EPISODE,
            TRACKER,
            [candidate],
            _timestamps(),
            params={"confirm_max": "x", "pad_s": -1, "max_events": None},
        )
        assert len(out) == 1
        assert out[0]["state"] in FINAL_STATES

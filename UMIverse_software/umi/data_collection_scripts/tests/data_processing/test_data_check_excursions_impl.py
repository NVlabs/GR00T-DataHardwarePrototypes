# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Implementation-side tests for the tracker-excursion check.

The behavioural suite is test_data_check_excursions*.py / test_excursion_confirm.py;
these pin details of this implementation: the edge-rotation measure, the dispatch
wiring and its config keys, and an end-to-end video confirmation on a synthetic
camera that either does or does not see the tracker's jump.

Run with: uv run pytest tests/data_processing/test_data_check_excursions_impl.py -v
"""

import json
import time

import numpy as np
import pytest
import zarr

from real_scripts.data_processing.data_check import (
    check_episode,
    check_tracking_excursions,
    data_check_findings,
)
from real_scripts.data_processing.data_check.excursion_confirm import (
    confirm_excursions,
)

FS = 30.0


def _clock(seconds, fs=FS):
    return np.arange(0.0, seconds, 1.0 / fs)


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _poses(xyz, yaw=None):
    out = np.tile(np.eye(4), (len(xyz), 1, 1))
    out[:, :3, 3] = xyz
    if yaw is not None:
        out[:, :3, :3] = np.stack([_rot_z(a) for a in yaw])
    return out


def _drift(t, speed=0.05):
    return np.column_stack([speed * t, np.zeros_like(t), np.zeros_like(t)])


def _pulse(t, at=2.0, hold_s=0.27, amp_m=0.08):
    sel = (t >= at) & (t < at + hold_s)
    return sel, amp_m


# ------------------------------------------------------------------ detector
def test_a_box_pulse_is_a_rotation_quiet_candidate():
    t = _clock(5)
    xyz = _drift(t)
    sel, amp = _pulse(t)
    xyz[sel, 2] -= amp
    got = check_tracking_excursions(_poses(xyz), t)
    assert got["num_events"] == got["num_candidates"] == 1
    ev = got["events"][0]
    assert ev["shape"] == "out_and_back" and ev["state"] == "candidate"
    assert ev["reason"] is None and ev["rotation_quiet"] is True
    assert t[ev["i0"]] == pytest.approx(2.0, abs=1.0 / FS)
    assert ev["t_out_s"] == t[ev["i0"]] and ev["t_back_s"] == t[ev["i1"]]
    assert ev["edge_rotation_deg_per_m_out"] == 0.0
    json.dumps(got)  # plain types only: lands in zarr.json


def test_a_pulse_that_rotates_with_the_jump_is_rejected():
    t = _clock(5)
    xyz = _drift(t)
    sel, amp = _pulse(t)
    xyz[sel, 2] -= amp
    # 8 degrees with an 80 mm step = 100 deg/m: what a real wrist motion does.
    yaw = np.where(sel, np.radians(8.0), 0.0)
    got = check_tracking_excursions(_poses(xyz, yaw), t)
    ev = got["events"][0]
    assert ev["state"] == "rejected" and ev["reason"] == "rotation"
    assert ev["edge_rotation_deg_per_m_out"] > 50.0
    assert got["num_candidates"] == 0


def test_edge_rotation_threshold_is_configurable():
    t = _clock(5)
    xyz = _drift(t)
    sel, amp = _pulse(t)
    xyz[sel, 2] -= amp
    yaw = np.where(sel, np.radians(8.0), 0.0)
    got = check_tracking_excursions(_poses(xyz, yaw), t, edge_rotation_deg_per_m=1000.0)
    assert got["events"][0]["state"] == "candidate"


def test_a_step_reports_no_back_edge():
    t = _clock(5)
    xyz = _drift(t)
    xyz[t >= 2.0, 0] += 0.12
    got = check_tracking_excursions(_poses(xyz), t)
    ev = got["events"][0]
    assert ev["shape"] == "step"
    assert ev["edge_rotation_deg_per_m_back"] is None


@pytest.mark.parametrize(
    "poses, ts",
    [
        (np.zeros((0, 4, 4)), np.zeros(0)),
        (np.tile(np.eye(4), (2, 1, 1)), np.array([0.0, 0.1])),
        (np.zeros((10, 3)), np.arange(10.0)),
        (np.tile(np.eye(4), (10, 1, 1)), None),
    ],
)
def test_degenerate_input_measures_nothing(poses, ts):
    got = check_tracking_excursions(poses, ts)
    assert got["events"] == [] and got["num_events"] == 0


def test_non_finite_pose_measures_nothing():
    t = _clock(5)
    xyz = _drift(t)
    xyz[40, 0] = np.nan
    got = check_tracking_excursions(_poses(xyz), t)
    assert got["events"] == [] and got["num_samples"] == len(t)


def test_a_repeated_timestamp_does_not_nominate():
    t = _clock(5)
    t[60] = t[59]
    got = check_tracking_excursions(_poses(_drift(t)), t)
    assert got["num_nominated"] == 0 and got["events"] == []


def test_events_are_capped():
    t = _clock(12)
    xyz = _drift(t)
    for at in np.arange(1.0, 11.0, 1.0):
        xyz[(t >= at) & (t < at + 0.2), 2] -= 0.08
    got = check_tracking_excursions(_poses(xyz), t, max_events=3)
    assert got["num_events"] == 3


def test_pose_and_clock_of_different_lengths_are_cut_to_the_shorter():
    t = _clock(5)
    xyz = _drift(t)
    sel, amp = _pulse(t)
    xyz[sel, 2] -= amp
    got = check_tracking_excursions(_poses(xyz)[:-2], t)
    assert got["num_samples"] == len(t) - 2
    assert got["num_events"] == 1


# ------------------------------------------------------------------ findings
def test_findings_keep_scanned_clean_devices_and_skip_unscanned_ones():
    result = {
        "tracking_checks": {
            "left_tracking_camera": {"excursions": {"events": []}},
            "right_tracking_camera": {"pose_jumps": {}},
        }
    }
    assert data_check_findings(result) == {
        "v": 2,
        "tracking_excursions": {"left_tracking_camera": {"events": []}},
    }
    assert data_check_findings({"tracking_checks": {}}) is None
    assert data_check_findings({"passed": True, "issues": []}) is None


# ------------------------------------------------------------ synthetic video
PX_PER_M = 500.0  # image shift per metre of camera motion


def _texture(h=180, w=900, seed=0):
    import cv2

    rng = np.random.default_rng(seed)
    tex = rng.integers(0, 255, size=(h, w), dtype=np.uint8)
    tex = cv2.GaussianBlur(tex, (0, 0), 2.0)
    return cv2.normalize(tex, None, 0, 255, cv2.NORM_MINMAX)


def _write_video(path, shifts_px, h=180, w=240):
    """A camera looking at a fixed texture, panned by `shifts_px` per frame."""
    import cv2

    tex = _texture(h=h)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FS, (w, h))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter cannot open mp4 output")
    for s in shifts_px:
        x0 = 100 + int(round(s))
        frame = tex[:, x0 : x0 + w]
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()


def _episode(tmp_path, *, camera_sees_jump, devices=("left_tracking_camera",)):
    """Raw episode: iPhones with their own video, a box excursion at 2 s.
    The camera follows the slow drift; it follows the excursion only when the
    hand really moved."""
    t = _clock(5)
    true_xyz = _drift(t, speed=0.05)
    xyz = true_xyz.copy()
    sel, amp = _pulse(t)
    xyz[sel, 0] += amp
    seen = xyz if camera_sees_jump else true_xyz
    root = zarr.open(str(tmp_path / "episode_0"), mode="w")
    for name in devices:
        dev = root.create_group(name)
        dev.create_array("pose", data=_poses(xyz))
        dev.create_array("capture_time", data=t)
        _write_video(tmp_path / "episode_0" / f"{name}.mp4", seen[:, 0] * PX_PER_M)
    return zarr.open(str(tmp_path / "episode_0"), mode="r"), t


@pytest.mark.parametrize(
    "camera_sees_jump, state", [(False, "confirmed"), (True, "rejected")]
)
def test_video_confirms_a_glitch_and_rejects_a_real_jump(
    tmp_path, camera_sees_jump, state
):
    pytest.importorskip("cv2")
    episode, t = _episode(tmp_path, camera_sees_jump=camera_sees_jump)
    det = check_tracking_excursions(
        np.asarray(episode["left_tracking_camera"]["pose"][:]), t
    )
    assert det["num_candidates"] == 1
    events = confirm_excursions(
        episode, str(tmp_path), "episode_0", "left_tracking_camera", det["events"], t
    )
    assert det["events"][0]["state"] == "candidate"  # input untouched
    ev = events[0]
    assert ev["state"] == state, ev["confirm"]
    assert ev["confirm"]["method"] == "video_gain_v1"
    assert set(ev["confirm"]["cameras"]) == {"left_tracking_camera"}
    if state == "rejected":
        assert ev["reason"] == "video" and ev["confirm"]["score"] >= 0.40
    else:
        assert ev["confirm"]["score"] < 0.20
    json.dumps(events)


def test_no_witness_video_leaves_the_event_unverified(tmp_path):
    episode, t = _episode(tmp_path, camera_sees_jump=False)
    det = check_tracking_excursions(
        np.asarray(episode["left_tracking_camera"]["pose"][:]), t
    )
    (tmp_path / "episode_0" / "left_tracking_camera.mp4").unlink()
    events = confirm_excursions(
        episode, str(tmp_path), "episode_0", "left_tracking_camera", det["events"], t
    )
    assert events[0]["state"] == "unverified"
    assert events[0]["confirm"]["score"] is None


def test_malformed_params_fall_back_to_defaults(tmp_path):
    pytest.importorskip("cv2")
    episode, t = _episode(tmp_path, camera_sees_jump=False)
    det = check_tracking_excursions(
        np.asarray(episode["left_tracking_camera"]["pose"][:]), t
    )
    events = confirm_excursions(
        episode,
        str(tmp_path),
        "episode_0",
        "left_tracking_camera",
        det["events"],
        t,
        params={"confirm_max": "x", "pad_s": -1, "max_events": None},
    )
    assert events[0]["state"] == "confirmed"


def test_candidates_over_the_budget_stay_unverified(tmp_path):
    episode, t = _episode(tmp_path, camera_sees_jump=False)
    det = check_tracking_excursions(
        np.asarray(episode["left_tracking_camera"]["pose"][:]), t
    )
    events = confirm_excursions(
        episode,
        str(tmp_path),
        "episode_0",
        "left_tracking_camera",
        det["events"] * 2,
        t,
        params={"max_events": 1},
    )
    assert [e["state"] for e in events] == ["confirmed", "unverified"]
    assert "budget" in events[1]["confirm"]["skip_reason"]


# ------------------------------------------------------------------ dispatch
def _check_config(**tracking):
    return {"tracking": {"enabled": True, **tracking}}


def test_a_confirmed_excursion_warns_and_never_fails_the_episode(tmp_path):
    pytest.importorskip("cv2")
    episode, _ = _episode(tmp_path, camera_sees_jump=False)
    result = check_episode(
        episode,
        config={},
        check_config=_check_config(),
        episode_name="episode_0",
        data_dir=str(tmp_path),
    )
    exc = result["tracking_checks"]["left_tracking_camera"]["excursions"]
    assert [e["state"] for e in exc["events"]] == ["confirmed"]
    assert not [i for i in result["issues"] if "tracker excursions" in i]
    assert result["warnings"] == [
        "left_tracking_camera: confirmed tracker excursions (1 >= 20mm)"
    ]
    findings = data_check_findings(result)
    assert findings["tracking_excursions"]["left_tracking_camera"] is exc
    assert findings["warnings"] == result["warnings"]


def test_findings_round_trip_through_zarr_attrs(tmp_path):
    """The station writes the findings into zarr.json: they must survive it."""
    pytest.importorskip("cv2")
    episode, _ = _episode(tmp_path, camera_sees_jump=False)
    result = check_episode(
        episode, {}, _check_config(), episode_name="episode_0", data_dir=str(tmp_path)
    )
    findings = data_check_findings(result)
    group = zarr.open_group(str(tmp_path / "attrs_probe"), mode="w")
    group.attrs["data_check_findings"] = findings
    reread = zarr.open_group(str(tmp_path / "attrs_probe"), mode="r")
    assert reread.attrs["data_check_findings"] == findings


def test_the_flag_off_leaves_pass_fail_and_issues_unchanged(tmp_path):
    episode, _ = _episode(tmp_path, camera_sees_jump=False)
    on = check_episode(
        episode, {}, _check_config(), episode_name="episode_0", data_dir=str(tmp_path)
    )
    off = check_episode(
        episode,
        {},
        _check_config(enabled={"excursions": False}),
        episode_name="episode_0",
        data_dir=str(tmp_path),
    )
    assert "excursions" not in off["tracking_checks"]["left_tracking_camera"]
    assert (on["passed"], on["issues"]) == (off["passed"], off["issues"])


def test_config_keys_reach_the_detector_and_confirmation(tmp_path):
    episode, _ = _episode(tmp_path, camera_sees_jump=False)
    result = check_episode(
        episode,
        {},
        _check_config(
            excursion_min_amplitude_mm=500.0, excursion_dv_threshold_m_per_s=0.5
        ),
        episode_name="episode_0",
        data_dir=str(tmp_path),
    )
    assert (
        result["tracking_checks"]["left_tracking_camera"]["excursions"]["events"] == []
    )

    result = check_episode(
        episode,
        {},
        _check_config(enabled={"excursion_confirm": False}),
        episode_name="episode_0",
        data_dir=str(tmp_path),
    )
    events = result["tracking_checks"]["left_tracking_camera"]["excursions"]["events"]
    assert [e["state"] for e in events] == ["candidate"]


def test_no_data_dir_skips_the_video(tmp_path):
    episode, _ = _episode(tmp_path, camera_sees_jump=False)
    result = check_episode(episode, {}, _check_config(), episode_name="episode_0")
    events = result["tracking_checks"]["left_tracking_camera"]["excursions"]["events"]
    assert [e["state"] for e in events] == ["candidate"]


def test_a_failing_measurement_never_fails_the_episode(tmp_path, monkeypatch):
    from real_scripts.data_processing.data_check import dispatch

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(dispatch, "check_tracking_excursions", boom)
    episode, _ = _episode(tmp_path, camera_sees_jump=False)
    result = check_episode(episode, {}, _check_config(), episode_name="episode_0")
    exc = result["tracking_checks"]["left_tracking_camera"]["excursions"]
    assert exc["events"] == [] and "boom" in exc["error"]
    assert not [i for i in result["issues"] if "excursion" in i]


# ------------------------------------------------------------- time budget
def test_a_spent_deadline_leaves_candidates_unverified(tmp_path):
    episode, t = _episode(tmp_path, camera_sees_jump=False)
    det = check_tracking_excursions(
        np.asarray(episode["left_tracking_camera"]["pose"][:]), t
    )
    events = confirm_excursions(
        episode,
        str(tmp_path),
        "episode_0",
        "left_tracking_camera",
        det["events"],
        t,
        deadline=time.monotonic() - 1.0,
    )
    assert events[0]["state"] == "unverified"
    assert events[0]["confirm"]["skip_reason"] == "over the time budget"


def test_a_zero_budget_confirms_nothing_and_counts_agree(tmp_path):
    episode, _ = _episode(tmp_path, camera_sees_jump=False)
    result = check_episode(
        episode,
        {},
        _check_config(excursion_confirm_budget_s=0.0),
        episode_name="episode_0",
        data_dir=str(tmp_path),
    )
    exc = result["tracking_checks"]["left_tracking_camera"]["excursions"]
    assert [e["state"] for e in exc["events"]] == ["unverified"]
    assert (exc["num_candidates"], exc["num_unverified"], exc["num_confirmed"]) == (
        0,
        1,
        0,
    )


def test_counts_follow_the_confirmed_states(tmp_path):
    pytest.importorskip("cv2")
    episode, _ = _episode(tmp_path, camera_sees_jump=False)
    result = check_episode(
        episode, {}, _check_config(), episode_name="episode_0", data_dir=str(tmp_path)
    )
    exc = result["tracking_checks"]["left_tracking_camera"]["excursions"]
    assert exc["num_events"] == 1
    assert exc["num_confirmed"] == 1
    assert exc["num_candidates"] == exc["num_unverified"] == exc["num_rejected"] == 0


def test_detector_only_output_has_every_count():
    t = _clock(5)
    xyz = _drift(t)
    sel, amp = _pulse(t)
    xyz[sel, 2] -= amp
    got = check_tracking_excursions(_poses(xyz), t)
    assert (got["num_candidates"], got["num_confirmed"]) == (1, 0)
    assert (got["num_unverified"], got["num_rejected"]) == (0, 0)


def test_arms_are_confirmed_before_the_head(tmp_path, monkeypatch):
    """The budget goes to the arms first, and it is one budget for the episode:
    whoever runs first can spend it all, and the next device gets nothing.

    The fake spends the budget by fiat rather than by sleeping to a real
    deadline. Deciding `confirmed` vs `unverified` from the wall clock would
    make the result depend on how long the numpy and cv2 work before the first
    confirm call takes, which varies on a loaded CI runner.

    Nothing about the behaviour under test is time-dependent -- the ordering
    comes from `sorted(..., key=lambda d: tracking_device_to_arm(d) is None)`
    and the sharing from a single `confirm_deadline` passed to every call -- so
    neither is this test. That the budget is genuinely shared is asserted
    directly.
    """
    from real_scripts.data_processing.data_check import excursion_confirm

    order = []
    deadlines = []

    def spend_the_budget(*args, deadline=None, **kwargs):
        device, events = args[3], args[4]
        order.append(device)
        deadlines.append(deadline)
        # The first device to run consumes the whole episode budget; every
        # later one finds nothing left. No clock, so no race.
        state = "confirmed" if len(order) == 1 else "unverified"
        return [dict(e, state=state) for e in events]

    monkeypatch.setattr(excursion_confirm, "confirm_excursions", spend_the_budget)
    episode, _ = _episode(
        tmp_path,
        camera_sees_jump=False,
        devices=("head_tracking_camera", "left_tracking_camera"),
    )
    result = check_episode(
        episode,
        {},
        _check_config(excursion_confirm_budget_s=0.2),
        episode_name="episode_0",
        data_dir=str(tmp_path),
    )
    assert order == ["left_tracking_camera", "head_tracking_camera"]
    # One budget per episode, shared by every device -- the same deadline object
    # reaches both calls.
    assert len(deadlines) == 2 and deadlines[0] == deadlines[1]
    checks = result["tracking_checks"]
    assert checks["left_tracking_camera"]["excursions"]["num_confirmed"] == 1
    assert checks["head_tracking_camera"]["excursions"]["num_unverified"] == 1

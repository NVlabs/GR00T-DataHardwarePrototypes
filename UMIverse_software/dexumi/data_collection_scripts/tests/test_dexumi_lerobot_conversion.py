# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LeRobot conversion of an aligned DexUMI session (robot actions included)."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import av
import numpy as np
import pandas as pd
import yaml
import zarr

from real_scripts.data_processing.lerobot_conversion import (
    dexumi_lerobot_conversion as lerobot,
)

TEMPLATE_DIR = (
    Path(__file__).resolve().parents[1] / "config/dexumi_exoskeleton/templates"
)
SESSION = "2026-11-05_dexL_dexR_hw1_task"
FRAMES = 6
JOINTS = 16
FPS = 45
TACTILE_SENSORS = (33, 34, 35, 36, 37)


def _write_video(path: Path, frames: int) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("h264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(frames):
            image = np.full((48, 64, 3), i * 20, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())


def write_aligned_session(
    root: Path, embodiment="bimanual_dexumi", robot_actions=False, tactile=True
) -> Path:
    """An aligned session as align_data writes it: config.yaml, the bundled glove
    calibrations, poses, wrist videos, encoder angles and (unless ``tactile`` is
    False) fingertip tactile forces, with a 2 N index-finger press. ``robot_actions``
    also writes the robot-action groups the conversion's first step produces."""
    session = root / f"{SESSION}_aligned"
    session.mkdir()
    sources = []
    for side in ("left", "right"):
        calibration = f"DEXUMI-{side.title()}.template.yaml"
        shutil.copy(TEMPLATE_DIR / calibration, session / calibration)
        sources.append(
            {
                "type": "dexumi_exoskeleton",
                "name": f"{side}_dexumi_exoskeleton_encoder",
                "params": {"config_file": calibration},
            }
        )
    (session / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "meta": {"embodiment": embodiment},
                "numeric_sources": sources,
                "video_sources": [],
            }
        )
    )
    group = zarr.open_group(str(session), mode="a")
    episode = group.create_group("episode_0")
    episode.attrs["task_description"] = "pick up the cube"
    episode.create_array(
        "valid_timestamps", data=np.arange(FRAMES, dtype=np.float64) / FPS
    )
    angles = np.linspace(-5.0, 5.0, FRAMES * 15).reshape(FRAMES, 15)
    for side in ("left", "right"):
        pose = np.tile(np.eye(4), (FRAMES, 1, 1))
        pose[:, 0, 3] = np.arange(FRAMES) * 0.01
        episode.create_group(f"{side}_tracking_camera").create_array(
            "pose_interp", data=pose
        )
        episode.create_group(f"{side}_umi_camera").create_array(
            "aligned_frame_indices", data=np.arange(FRAMES)
        )
        _write_video(session / "episode_0" / f"{side}_umi_camera.mp4", FRAMES)
        source = f"{side}_dexumi_exoskeleton_encoder"
        episode.create_group(source).create_array("joint_angles_interp", data=angles)
        if tactile:
            pads = episode.create_group(f"{side}_dexumi_exoskeleton_tactile")
            for sensor in TACTILE_SENSORS:
                for axis in ("fx", "fy", "fz"):
                    for point in (1, 2):
                        force = np.zeros(FRAMES)  # millinewtons
                        if sensor == 34 and axis == "fz":
                            force[2:5] = 2000.0
                        # Alignment materializes all-zero (no-contact) chunks.
                        pads.create_array(
                            f"tactile_{sensor}_{axis}_{point}_interp",
                            data=force,
                            config={"write_empty_chunks": True},
                        )
        if robot_actions:
            joints = np.linspace(0, 1, FRAMES * JOINTS).reshape(FRAMES, JOINTS)
            for kind, array in (("state", "state_interp"), ("action", "action_interp")):
                out = episode.create_group(f"{source}_robot_{kind}")
                out.attrs["write_state"] = "complete"
                out.attrs["tactile"] = False
                out.create_array(array, data=joints)
    return session


def _parquet(output: Path) -> pd.DataFrame:
    return pd.read_parquet(output / "data/chunk-000/episode_000000.parquet")


def _meta(output: Path, name: str) -> dict:
    return json.loads((output / "meta" / name).read_text())


class DexumiLerobotConversionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.output = self.root / f"{SESSION}_lerobot"

    def tearDown(self):
        self._tmp.cleanup()

    def test_one_command_writes_robot_actions_and_the_dataset(self):
        session = write_aligned_session(self.root)
        self.assertEqual(lerobot.main(["-d", str(session), "--no-video-variants"]), 0)
        episode = zarr.open_group(str(session), mode="r")["episode_0"]
        frame = pd.read_parquet(self.output / "data/chunk-000/episode_000000.parquet")
        self.assertEqual(len(frame), FRAMES)
        np.testing.assert_allclose(
            np.stack(frame["right_tracker.eef"])[:, 0], np.arange(FRAMES) * 0.01
        )
        for side in ("left", "right"):
            source = f"{side}_dexumi_exoskeleton_encoder"
            np.testing.assert_array_equal(
                np.stack(frame[f"{side}_hand.joint_positions"]),
                np.asarray(episode[f"{source}_robot_state/state_interp"]),
            )
            np.testing.assert_array_equal(
                np.stack(frame[f"action.{side}_hand.joint_positions"]),
                np.asarray(episode[f"{source}_robot_action/action_interp"]),
            )
        modality = json.loads((self.output / "meta/modality.json").read_text())
        self.assertEqual(
            modality["action"]["left_hand.joint_positions"]["original_key"],
            "action.left_hand.joint_positions",
        )
        self.assertEqual(modality["state"]["left_hand.joint_positions"]["end"], JOINTS)
        self.assertEqual(modality["action"]["left_tracker.eef"]["end"], 9)
        info = json.loads((self.output / "meta/info.json").read_text())
        self.assertEqual(info["fps"], float(FPS))
        self.assertEqual(info["total_videos"], 2)
        for side in ("left", "right"):
            video = (
                self.output / f"videos/chunk-000/{side}_umi_camera/episode_000000.mp4"
            )
            with av.open(str(video)) as container:
                self.assertEqual(sum(1 for _ in container.decode(video=0)), FRAMES)
        # The robot-action step is tactile by default, like ./launch.sh convert,
        # and the dataset exports its torques and the fingertip forces.
        modality = _meta(self.output, "modality.json")
        relative = _meta(self.output, "relative_stats.json")
        for side in ("left", "right"):
            source = f"{side}_dexumi_exoskeleton_encoder"
            state = episode[f"{source}_robot_state"]
            action = episode[f"{source}_robot_action"]
            self.assertTrue(state.attrs["tactile"])
            self.assertGreater(np.abs(state["joints_torque_interp"][:]).max(), 0)
            self.assertEqual(
                set(action.array_keys()), {"action_interp", "action_torque_interp"}
            )
            np.testing.assert_array_equal(
                np.stack(frame[f"action.{side}_hand.joint_torques"]),
                state["joints_torque_interp"][:],
            )
            entry = modality["action"][f"{side}_hand.joint_torques"]
            self.assertEqual(entry["original_key"], f"action.{side}_hand.joint_torques")
            self.assertEqual(entry["end"], JOINTS)
            self.assertNotIn(f"{side}_hand.joint_torques", modality["state"])
            self.assertIn(f"{side}_hand.joint_positions", relative)
            self.assertNotIn(f"{side}_hand.joint_torques", relative)
            # 30 values: sensors 33..37, each fx, fy, fz at point 1 then 2, in N.
            tactile = np.stack(frame[f"{side}_hand.tactile"])
            self.assertEqual(tactile.shape, (FRAMES, 30))
            self.assertEqual(modality["state"][f"{side}_hand.tactile"]["end"], 30)
            self.assertNotIn(f"{side}_hand.tactile", modality["action"])
            expected = np.zeros((FRAMES, 30))
            expected[2:5, [8, 11]] = 2.0  # sensor 34 (second) fz at both points
            np.testing.assert_allclose(tactile, expected)
        # The stacked [positions | torques] array is not exported, and no
        # fix-offset action exists anywhere in the dataset.
        self.assertFalse([c for c in frame.columns if "action_torque" in c])
        self.assertFalse([c for c in frame.columns if "fix_offset" in c])
        self.assertFalse([k for k in modality["action"] if "fix_offset" in k])
        self.assertFalse([k for k in relative if "fix_offset" in k])
        self.assertFalse([k for k in info["features"] if "fix_offset" in k])
        self.assertNotIn("hand_joint_positions_fix_offset", info["dexumi_geometry"])

    def test_no_tactile_converts_a_session_without_tactile_recordings(self):
        session = write_aligned_session(self.root, tactile=False)
        args = ["-d", str(session), "--no-video-variants"]
        self.assertEqual(lerobot.main(args), 1)
        self.assertEqual(lerobot.main([*args, "--no-tactile"]), 0)
        episode = zarr.open_group(str(session), mode="r")["episode_0"]
        for side in ("left", "right"):
            source = f"{side}_dexumi_exoskeleton_encoder"
            state = episode[f"{source}_robot_state"]
            self.assertFalse(state.attrs["tactile"])
            self.assertEqual(list(state.array_keys()), ["state_interp"])
            self.assertEqual(
                list(episode[f"{source}_robot_action"].array_keys()), ["action_interp"]
            )
        columns = set(_parquet(self.output).columns)
        self.assertFalse({c for c in columns if "fix_offset" in c or "torques" in c})
        self.assertFalse({c for c in columns if c.endswith(".tactile")})
        modality = _meta(self.output, "modality.json")
        self.assertEqual(
            sorted(modality["action"]),
            sorted(
                f"{side}_{key}"
                for side in ("left", "right")
                for key in ("tracker.eef", "hand.joint_positions")
            ),
        )

    def test_no_tactile_still_exports_the_recorded_fingertip_forces(self):
        session = write_aligned_session(self.root)
        args = ["-d", str(session), "--no-video-variants", "--no-tactile"]
        self.assertEqual(lerobot.main(args), 0)
        frame = _parquet(self.output)
        self.assertEqual(np.stack(frame["left_hand.tactile"]).shape, (FRAMES, 30))
        self.assertNotIn("action.left_hand.joint_torques", frame.columns)
        modality = _meta(self.output, "modality.json")
        self.assertIn("right_hand.tactile", modality["state"])
        self.assertNotIn("right_hand.joint_torques", modality["action"])

    def test_dataset_step_refuses_leftover_tactile_arrays(self):
        session = write_aligned_session(self.root, robot_actions=True)
        state = zarr.open_group(str(session), mode="a")[
            "episode_0/left_dexumi_exoskeleton_encoder_robot_state"
        ]
        state.create_array("joints_torque_interp", data=np.zeros((FRAMES, JOINTS)))
        with self.assertRaisesRegex(ValueError, "left from a tactile conversion"):
            lerobot.convert_session(session, self.root / "out", None)

    def test_dataset_step_refuses_tactile_recorded_for_one_hand_only(self):
        session = write_aligned_session(self.root, robot_actions=True)
        del zarr.open_group(str(session), mode="a")[
            "episode_0/right_dexumi_exoskeleton_tactile"
        ]
        with self.assertRaisesRegex(ValueError, "Only some episodes or hands"):
            lerobot.convert_session(session, self.root / "out", None)

    def test_existing_dataset_is_replaced_only_on_request(self):
        session = write_aligned_session(self.root)
        args = ["-d", str(session), "--no-video-variants"]
        self.assertEqual(lerobot.main(args), 0)
        (self.output / "marker").write_text("old")
        self.assertEqual(lerobot.main(args), 1)
        self.assertTrue((self.output / "marker").exists())
        self.assertEqual(lerobot.main([*args, "--replace"]), 0)
        self.assertFalse((self.output / "marker").exists())
        self.assertTrue((self.output / "meta/info.json").is_file())

    def test_dataset_step_refuses_before_the_robot_actions(self):
        session = write_aligned_session(self.root)
        with self.assertRaisesRegex(ValueError, "Robot actions are not ready"):
            lerobot.convert_session(session, self.root / "out", None)
        self.assertFalse((self.root / "out").exists())

    def test_failed_rebuild_keeps_the_previous_dataset(self):
        session = write_aligned_session(self.root)
        (self.output / "meta").mkdir(parents=True)
        (self.output / "meta/info.json").write_text("{}")
        with self.assertRaises(ValueError):
            lerobot.rebuild_dataset(session, self.output)
        self.assertEqual((self.output / "meta/info.json").read_text(), "{}")
        self.assertEqual(
            sorted(p.name for p in self.root.iterdir()),
            sorted([session.name, self.output.name]),
        )

    def test_refuses_other_embodiments(self):
        session = write_aligned_session(
            self.root, embodiment="gripper_umi", robot_actions=True
        )
        with self.assertRaisesRegex(ValueError, "bimanual DexUMI"):
            lerobot.convert_session(session, self.root / "out", None)


if __name__ == "__main__":
    unittest.main()

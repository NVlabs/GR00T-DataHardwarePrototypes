# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot-action conversion of an aligned bimanual DexUMI session: kinematic
(--no-tactile) and tactile (the default) outputs."""

import shutil
import tempfile
import unittest
from pathlib import Path

import click
import numpy as np
import yaml
import zarr
from click.testing import CliRunner

from real_scripts.dexumi_pipeline import convert_to_robot_action as convert
from umi.common.data import write_state

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = REPO_ROOT / "config/dexumi_exoskeleton/templates"
HAND_CONFIGS = REPO_ROOT / "umi/control/robot/configs"
FRAMES = 12
JOINTS = 16
SENSORS = (33, 34, 35, 36, 37)  # thumb..pinky
# Distinct presses per hand so cross-hand mix-ups are detectable: the right
# hand presses index (34) from frame 4 on, the left presses mid (35) on 2..7.
PRESSES = {"right": (34, slice(4, None)), "left": (35, slice(2, 8))}
STATE_TACTILE = {"state_interp", "joints_torque_interp"}
ACTION_TACTILE = {"action_interp", "action_torque_interp"}


def _aligned_session(root: Path, sides=("left", "right"), tactile=True) -> None:
    """An aligned session as align_data writes it: config.yaml, the bundled
    glove calibrations, and each episode's ``valid_timestamps`` and encoder
    ``joint_angles_interp`` plus (with ``tactile``) the fingertip forces."""
    sources = []
    for side in ("left", "right"):
        calibration = f"DEXUMI-{side.title()}.template.yaml"
        shutil.copy(TEMPLATE_DIR / calibration, root / calibration)
        for stream in ("encoder", "tactile"):
            sources.append(
                {
                    "type": "dexumi_exoskeleton",
                    "name": f"{side}_dexumi_exoskeleton_{stream}",
                    "params": {
                        "config_file": f"config/dexumi_exoskeleton/templates/{calibration}"
                    },
                }
            )
    config = {"meta": {"embodiment": "bimanual_dexumi"}, "numeric_sources": sources}
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    episode = zarr.open_group(str(root), mode="a").create_group("episode_0")
    episode.create_array("valid_timestamps", data=np.arange(FRAMES) / 45.0)
    angles = np.linspace(-5.0, 5.0, FRAMES * 15).reshape(FRAMES, 15)
    for side in sides:
        group = episode.create_group(f"{side}_dexumi_exoskeleton_encoder")
        group.create_array("joint_angles_interp", data=angles)
        if not tactile:
            continue
        pads = episode.create_group(f"{side}_dexumi_exoskeleton_tactile")
        press_sensor, press_frames = PRESSES[side]
        for sensor in SENSORS:
            for axis in ("fx", "fy", "fz"):
                for point in (1, 2):
                    force = np.zeros(FRAMES)  # millinewtons
                    if sensor == press_sensor and axis == "fz":
                        force[press_frames] = 2000.0  # 2 N pad-normal press
                    pads.create_array(
                        f"tactile_{sensor}_{axis}_{point}_interp", data=force
                    )


def _compliance_run() -> convert.ComplianceRun:
    params = convert.resolve_robot_action_params()
    return convert.ComplianceRun(
        hand_config_path=params["hand_config"],
        ema_alpha=params["ema_alpha"],
        deadband_n=params["deadband_n"],
        components=convert._resolve_components_option(params["components"]),
        tactile_use_point=params["tactile_use_point"],
    )


class _SessionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def episode(self):
        return zarr.open_group(str(self.root), mode="r")["episode_0"]

    def robot_groups(self):
        return sorted(k for k in self.episode().group_keys() if "_robot_" in k)


class ConvertToRobotActionTest(_SessionTest):
    def test_bimanual_session_gets_state_and_action_per_hand(self):
        _aligned_session(self.root)
        plans = convert.build_conversion_plans(self.root)
        self.assertEqual(
            sorted(plan.source_name for plan in plans),
            ["left_dexumi_exoskeleton_encoder", "right_dexumi_exoskeleton_encoder"],
        )
        self.assertEqual(
            convert._convert_episode(str(self.root), 0, plans), (0, True, None)
        )
        episode = self.episode()
        for side in ("left", "right"):
            state = episode[f"{side}_dexumi_exoskeleton_encoder_robot_state"]
            action = episode[f"{side}_dexumi_exoskeleton_encoder_robot_action"]
            self.assertEqual(state["state_interp"].shape, (FRAMES, JOINTS))
            np.testing.assert_array_equal(
                action["action_interp"][:], state["state_interp"][:]
            )
            self.assertEqual(
                action["action_interp"].attrs["action_position_dims"], JOINTS
            )
            self.assertFalse(state.attrs["tactile"])
            for group in (state, action):
                self.assertTrue(write_state.complete(group))
                self.assertEqual(len(list(group.array_keys())), 1)

    def test_missing_hand_writes_nothing(self):
        _aligned_session(self.root, sides=("right",))
        plans = convert.build_conversion_plans(self.root)
        idx, ok, error = convert._convert_episode(str(self.root), 0, plans)
        self.assertFalse(ok)
        self.assertIn("left_dexumi_exoskeleton_encoder", error)
        self.assertEqual(self.robot_groups(), [])


class TactileConversionTest(_SessionTest):
    def test_tactile_run_writes_every_variant_per_hand(self):
        _aligned_session(self.root)
        plans = convert.build_conversion_plans(self.root)
        run = _compliance_run()
        self.assertEqual(
            convert._convert_episode(str(self.root), 0, plans, run), (0, True, None)
        )
        episode = self.episode()
        for side, (_, press) in PRESSES.items():
            source = f"{side}_dexumi_exoskeleton_encoder"
            state = episode[f"{source}_robot_state"]
            action = episode[f"{source}_robot_action"]
            self.assertEqual(set(state.array_keys()), STATE_TACTILE)
            self.assertEqual(set(action.array_keys()), ACTION_TACTILE)
            q = state["state_interp"][:]
            torque = state["joints_torque_interp"][:]
            np.testing.assert_array_equal(action["action_interp"][:], q)
            # Positions first, torques second.
            np.testing.assert_array_equal(
                action["action_torque_interp"][:], np.concatenate([q, torque], 1)
            )
            self.assertEqual(
                action["action_torque_interp"].attrs["action_layout"],
                ["joint_positions", "joint_torques"],
            )
            # Torque follows this hand's own press only.
            np.testing.assert_array_equal(torque[: press.start], 0.0)
            self.assertGreater(np.abs(torque[press]).max(), 0.0)
            self.assertTrue(state.attrs["tactile"])
            attrs = dict(action.attrs)
            self.assertEqual(
                attrs["torque_params"]["hand_config_path"], f"xhand_{side}.yaml"
            )
            self.assertTrue(attrs["torque_params"]["hand_geometry_verified"])
            self.assertEqual(attrs["torque_params"]["components"], ["z"])
            self.assertEqual(
                attrs["tactile_source"], f"{side}_dexumi_exoskeleton_tactile"
            )
            stored = attrs["torque_params"]
            self.assertEqual(
                {key: stored[key] for key in run.torque_kwargs()},
                {**run.torque_kwargs(), "components": ["z"]},
            )
            self.assertNotIn("baked_offset_params", attrs)
            for group in (state, action):
                self.assertTrue(write_state.complete(group))
        # The hands press different fingers, so their torques differ.
        self.assertFalse(
            np.array_equal(
                episode["left_dexumi_exoskeleton_encoder_robot_state"][
                    "joints_torque_interp"
                ][:],
                episode["right_dexumi_exoskeleton_encoder_robot_state"][
                    "joints_torque_interp"
                ][:],
            )
        )

    def test_torque_matches_the_device_computation(self):
        _aligned_session(self.root)
        plans = convert.build_conversion_plans(self.root)
        run = _compliance_run()
        convert._convert_episode(str(self.root), 0, plans, run)
        episode = self.episode()
        plan = next(p for p in plans if p.source_name.startswith("right_"))
        pads = episode["right_dexumi_exoskeleton_tactile"]
        device = convert.DexUmiExoskeleton.from_config(
            plan.umi_config_path,
            reader=None,
            hand_config=str(HAND_CONFIGS / "xhand_right.yaml"),
        )
        q = episode["right_dexumi_exoskeleton_encoder_robot_state/state_interp"][:]
        expected = device.compute_joint_torque(
            q,
            {name: pads[name][:] for name in pads.array_keys()},
            **run.torque_kwargs(),
        )
        np.testing.assert_array_equal(
            episode[
                "right_dexumi_exoskeleton_encoder_robot_state/joints_torque_interp"
            ][:],
            expected,
        )

    def test_no_tactile_rerun_drops_the_tactile_arrays(self):
        _aligned_session(self.root)
        plans = convert.build_conversion_plans(self.root)
        convert._convert_episode(str(self.root), 0, plans, _compliance_run())
        self.assertEqual(
            convert._convert_episode(str(self.root), 0, plans, None), (0, True, None)
        )
        source = "right_dexumi_exoskeleton_encoder"
        state = self.episode()[f"{source}_robot_state"]
        action = self.episode()[f"{source}_robot_action"]
        self.assertEqual(list(state.array_keys()), ["state_interp"])
        self.assertEqual(list(action.array_keys()), ["action_interp"])
        self.assertNotIn("torque_params", action.attrs)

    def test_missing_tactile_group_fails_without_output(self):
        _aligned_session(self.root, tactile=False)
        plans = convert.build_conversion_plans(self.root)
        idx, ok, error = convert._convert_episode(
            str(self.root), 0, plans, _compliance_run()
        )
        self.assertFalse(ok)
        self.assertIn("_dexumi_exoskeleton_tactile", error)
        self.assertIn("--no-tactile", error)
        self.assertEqual(self.robot_groups(), [])

    def test_calibration_without_hand_config_uses_its_side_model(self):
        _aligned_session(self.root)
        calibration = self.root / "DEXUMI-Left.template.yaml"
        values = yaml.safe_load(calibration.read_text())
        del values["hand_config"]
        calibration.write_text(yaml.safe_dump(values))
        plans = {p.source_name: p for p in convert.build_conversion_plans(self.root)}
        self.assertEqual(
            plans["left_dexumi_exoskeleton_encoder"].hand_config_path,
            str(HAND_CONFIGS / "xhand_left.yaml"),
        )
        # A hand model bundled in the session wins over the repository copy.
        bundled = self.root / "xhand_left.yaml"
        shutil.copy(HAND_CONFIGS / "xhand_left.yaml", bundled)
        plans = {p.source_name: p for p in convert.build_conversion_plans(self.root)}
        self.assertEqual(
            plans["left_dexumi_exoskeleton_encoder"].hand_config_path, str(bundled)
        )

    def test_unresolvable_hand_config_is_refused(self):
        _aligned_session(self.root)
        calibration = self.root / "DEXUMI-Right.template.yaml"
        values = yaml.safe_load(calibration.read_text())
        values["hand_config"] = "umi/control/robot/configs/no_such_hand.yaml"
        calibration.write_text(yaml.safe_dump(values))
        with self.assertRaisesRegex(click.ClickException, "no_such_hand.yaml"):
            convert.build_conversion_plans(self.root)

    def test_calibration_naming_the_old_orange_hand_configs_still_converts(self):
        # The hand configs and URDF folder were renamed from the "Orange"
        # codename; glove calibrations recorded before name the old files.
        _aligned_session(self.root)
        for side in ("left", "right"):
            calibration = self.root / f"DEXUMI-{side.title()}.template.yaml"
            values = yaml.safe_load(calibration.read_text())
            values["hand_config"] = f"umi/control/robot/configs/orange_hand_{side}.yaml"
            calibration.write_text(yaml.safe_dump(values))
        plans = {p.source_name: p for p in convert.build_conversion_plans(self.root)}
        for side in ("left", "right"):
            self.assertEqual(
                plans[f"{side}_dexumi_exoskeleton_encoder"].hand_config_path,
                str(HAND_CONFIGS / f"xhand_{side}.yaml"),
            )
        # An old hand config bundled in the session still names the old URDF
        # folder; the URDF resolves at its new location.
        hand = yaml.safe_load((HAND_CONFIGS / "xhand_left.yaml").read_text())
        urdf = hand["kinematics"]["urdf_path"].replace("/xhand/", "/Orange_URDF/", 1)
        self.assertIn("/Orange_URDF/", urdf)
        hand["kinematics"]["urdf_path"] = urdf
        (self.root / "orange_hand_left.yaml").write_text(yaml.safe_dump(hand))
        plans = convert.build_conversion_plans(self.root)
        self.assertEqual(
            convert._convert_episode(str(self.root), 0, plans, _compliance_run()),
            (0, True, None),
        )
        attrs = self.episode()["left_dexumi_exoskeleton_encoder_robot_action"].attrs
        self.assertEqual(
            attrs["torque_params"]["hand_config_path"], "orange_hand_left.yaml"
        )

    def test_cli_is_tactile_by_default_and_kinematic_with_no_tactile(self):
        _aligned_session(self.root)
        source = "left_dexumi_exoskeleton_encoder"
        result = CliRunner().invoke(convert.main, ["-d", str(self.root), "-s"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            set(self.episode()[f"{source}_robot_action"].array_keys()), ACTION_TACTILE
        )
        result = CliRunner().invoke(
            convert.main, ["-d", str(self.root), "-s", "--no-tactile"]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            list(self.episode()[f"{source}_robot_action"].array_keys()),
            ["action_interp"],
        )

    def test_cli_flag_overrides_the_config_file(self):
        _aligned_session(self.root)
        result = CliRunner().invoke(
            convert.main, ["-d", str(self.root), "-s", "--ema-alpha", "0.5"]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        attrs = self.episode()["right_dexumi_exoskeleton_encoder_robot_action"].attrs
        self.assertEqual(attrs["torque_params"]["ema_alpha"], 0.5)

    def test_parameter_file_must_have_every_key(self):
        params = self.root / "params.yaml"
        params.write_text(yaml.safe_dump({"robot_action": {"deadband_n": 0.1}}))
        with self.assertRaisesRegex(ValueError, "missing keys"):
            convert.load_robot_action_params(params)
        defaults = convert.load_robot_action_params()
        self.assertEqual(set(defaults), set(convert.ROBOT_ACTION_KEYS))
        # The removed fix-offset parameters are refused, not silently ignored.
        block = yaml.safe_load(convert.DEFAULT_CONFIG.read_text())["robot_action"]
        params.write_text(yaml.safe_dump({"robot_action": {**block, "stiffness": 0.5}}))
        with self.assertRaisesRegex(ValueError, r"unknown keys \['stiffness'\]"):
            convert.load_robot_action_params(params)


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recalibrating a glove that is already calibrated, with a stand-in for calibrate_exo.py.

Run from data_collection_scripts with ``.venv/bin/python -m unittest discover -s tests``.
"""

import os
import shutil
import sys
import tempfile
import time
import unittest

import yaml

from gui.dc_ui.core import calibration as cal
from gui.dc_ui.core import devices

# Same two prompts as the tool route of calibrate_exo.py; saves only on "y".
STUB = """
import sys, yaml
path = sys.argv[1]
input("Press Enter when ready to record... ")
cfg = yaml.safe_load(open(path))
cfg["zero_position_angle"] = [9.0] * 15
if input("Save to %s? [y/N] " % path).strip().lower() == "y":
    yaml.safe_dump(cfg, open(path, "w"))
"""

OLD = {
    "device_serial": "AAAA",
    "calibration_method": "calibration_tool_v4",
    "zero_position_angle": [1.0] * 15,
}


class RecalibrationTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.stub = os.path.join(self.root, "stub.py")
        with open(self.stub, "w") as f:
            f.write(STUB)
        self.path = os.path.join(self.root, cal.config_rel("Left"))
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, "w") as f:
            yaml.safe_dump(OLD, f)
        self._saved = (cal.repo_path, devices.repo_path)
        cal.repo_path = devices.repo_path = lambda rel: os.path.join(self.root, rel)

    def tearDown(self):
        cal.repo_path, devices.repo_path = self._saved
        shutil.rmtree(self.root)

    def runner(self, boards):
        return cal.CalibrationRunner(
            argv_for=lambda side: [sys.executable, self.stub, self.path],
            boards=lambda: boards,
        )

    def wait(self, runner, state):
        end = time.time() + 10
        while time.time() < end and runner.snapshot()["state"] != state:
            time.sleep(0.02)
        self.assertEqual(runner.snapshot()["state"], state)

    def zero(self):
        with open(self.path) as f:
            return yaml.safe_load(f)["zero_position_angle"][0]

    def run_to_review(self, runner):
        runner.start("Left", recalibrate=True)
        self.wait(runner, "waiting")
        runner.confirm()
        self.wait(runner, "review")

    def test_saves_with_another_glove_plugged_in(self):
        runner = self.runner(["AAAA", "BBBB"])
        self.run_to_review(runner)
        runner.save(True)
        self.wait(runner, "done")
        self.assertEqual(self.zero(), 9.0)
        self.assertEqual(cal.glove_status("Left")["serial"], "AAAA")
        self.assertTrue(os.path.exists(f"{self.path}.bak"))

    def test_discarding_keeps_the_current_calibration(self):
        runner = self.runner(["AAAA"])
        self.run_to_review(runner)
        runner.save(False)
        self.wait(runner, "cancelled")
        self.assertEqual(self.zero(), 1.0)
        self.assertFalse(os.path.exists(f"{self.path}.bak"))

    def test_cancelling_keeps_the_current_calibration(self):
        runner = self.runner(["AAAA"])
        runner.start("Left", recalibrate=True)
        self.wait(runner, "waiting")
        runner.cancel()
        self.wait(runner, "cancelled")
        self.assertEqual(self.zero(), 1.0)

    def test_the_glove_must_be_plugged_in(self):
        with self.assertRaises(cal.CalibrationRefused):
            self.runner(["BBBB"]).start("Left", recalibrate=True)

    def test_a_side_without_a_calibration_cannot_be_redone(self):
        with self.assertRaises(cal.CalibrationRefused):
            self.runner(["AAAA"]).start("Right", recalibrate=True)

    def test_a_recorded_board_is_not_a_new_glove(self):
        with self.assertRaises(cal.CalibrationRefused):
            self.runner(["AAAA"]).start("Left")


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 05 of the DC GUI: list sessions, convert one, never while recording."""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import zarr
from fastapi.testclient import TestClient

from gui.dc_ui import server
from gui.dc_ui.core import lerobot
from gui.dc_ui.core.composer import MOCK_BASE_CONFIG
from gui.dc_ui.core.kinds import LaunchSpec
from gui.dc_ui.core.session import DcSession

HEADERS = {"Origin": "http://127.0.0.1:8791", "Sec-Fetch-Site": "same-origin"}


def _episodes(folder: Path, names, aligned=False):
    group = zarr.open_group(str(folder), mode="a")
    for name in names:
        episode = group.create_group(name)
        if aligned:
            episode.create_array("valid_timestamps", data=np.zeros(3))


class LerobotStepTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.data = root / "output"
        self.home = root / "home"
        # Cover converted, conversion-ready and partially aligned sessions.
        for name, recorded, aligned in (
            ("2026-11-01_done", ["episode_0"], ["episode_0"]),
            (
                "2026-11-02_ready",
                ["episode_0", "episode_1"],
                ["episode_0", "episode_1"],
            ),
            ("2026-11-03_partial", ["episode_0", "episode_1"], ["episode_0"]),
        ):
            _episodes(self.data / name, recorded)
            _episodes(self.data / f"{name}_aligned", aligned, aligned=True)
        # Rejected recordings must not appear in the session list.
        _episodes(self.data / "2026-11-01_done_quarantine", ["episode_0"])
        meta = self.data / "2026-11-01_done_lerobot" / "meta"
        meta.mkdir(parents=True)
        (meta / "episodes.jsonl").write_text(
            json.dumps({"original_episode_index": "episode_0"}) + "\n"
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_session_rows(self):
        rows = {row["name"]: row for row in lerobot.session_rows(self.data)}
        self.assertEqual(
            sorted(rows), ["2026-11-01_done", "2026-11-02_ready", "2026-11-03_partial"]
        )
        self.assertTrue(rows["2026-11-01_done"]["up_to_date"])
        self.assertIsNone(rows["2026-11-01_done"]["blocker"])
        self.assertIsNone(rows["2026-11-02_ready"]["converted"])
        self.assertIsNone(rows["2026-11-02_ready"]["blocker"])
        self.assertIn("episode_1", rows["2026-11-03_partial"]["blocker"])
        spec = lerobot.lerobot_spec(rows["2026-11-02_ready"]["aligned_path"])
        self.assertEqual(spec.role, lerobot.LEROBOT_ROLE)
        self.assertEqual(spec.argv[-3:-1][0], "--input-dataset")
        self.assertEqual(spec.argv[-1], "--replace")

    def client(self):
        app = server.create_app(
            DcSession.load(str(self.home)),
            server.make_supervisor(str(self.home)),
            data_root=str(self.data),
            base_config=MOCK_BASE_CONFIG,
        )
        return app, TestClient(app, base_url="http://127.0.0.1:8791")

    def test_conversion_routes(self):
        app, client = self.client()
        listing = client.get("/api/lerobot").json()
        self.assertEqual(len(listing["sessions"]), 3)
        start = "/api/lerobot/start"
        self.assertEqual(
            client.post(start, json={"session": "nope"}, headers=HEADERS).status_code,
            404,
        )
        partial = client.post(
            start, json={"session": "2026-11-03_partial"}, headers=HEADERS
        )
        self.assertEqual(partial.status_code, 409)
        self.assertIn("not aligned", partial.json()["detail"])
        # Use a short process to exercise route checks and supervisor state.
        quick = LaunchSpec(
            role=lerobot.LEROBOT_ROLE,
            argv=[
                sys.executable,
                "-c",
                "import time; print('converting'); time.sleep(2)",
            ],
        )
        with mock.patch.object(lerobot, "lerobot_spec", return_value=quick):
            ok = client.post(
                start, json={"session": "2026-11-02_ready"}, headers=HEADERS
            )
            self.assertEqual(ok.status_code, 200, ok.text)
            self.assertTrue(client.get("/api/procs").json()["converting"])
            again = client.post(
                start, json={"session": "2026-11-02_ready"}, headers=HEADERS
            )
            self.assertEqual(again.status_code, 409)
        recorder = client.post(
            "/api/procs/recorder/start",
            json={"task_name": "mock_task"},
            headers=HEADERS,
        )
        self.assertEqual(recorder.status_code, 409)
        self.assertIn("LeRobot conversion is running", recorder.json()["detail"])
        deadline = time.monotonic() + 15
        while client.get("/api/procs").json()["converting"]:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.1)
        status = client.get("/api/procs").json()["procs"][lerobot.LEROBOT_ROLE]
        self.assertEqual((status["state"], status["exit_code"]), ("exited", 0))
        app.state.ui.supervisor.stop_all()

    def test_refused_while_the_recorder_runs(self):
        app, client = self.client()
        with mock.patch.object(type(app.state.ui), "recorder_live", return_value=True):
            response = client.post(
                "/api/lerobot/start",
                json={"session": "2026-11-02_ready"},
                headers=HEADERS,
            )
        self.assertEqual(response.status_code, 409)
        self.assertIn("recorder is running", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()

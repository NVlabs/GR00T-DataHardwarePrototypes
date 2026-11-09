# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The live camera preview behind the ids in the detected-devices table.

Clicking an OAK, iPhone or USB camera id on /config opens a pop-up with that
camera's live picture, so an operator can tell look-alike devices apart by what
they see -- and a trigger camera, plugged into the computer rather than its
gripper, by which gripper it shows. A preview holds its camera exclusively, so
most of what is pinned here is about it never colliding with the other things
that open cameras, and never being left open: one at a time, never under a
recording, a scan or the camera check, released by a scan, the camera check and
a recording, and released when the pop-up stops reading the stream.

**No test here opens a camera.** The worker is replaced by a child that writes
stand-in pictures to the file the server streams; the worker's own picture
code is tested on arrays.
"""

import asyncio
import pathlib
import shutil
import sys
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import test_unavailable_device_gate as gate  # noqa: E402
import yaml
from fastapi.testclient import TestClient
from test_device_summary import BENCH, V4_STATION, record  # noqa: E402

from station_ui import calibration as cal  # noqa: E402
from station_ui import devices as devices_core  # noqa: E402
from station_ui import jobs, preview, preview_worker, recording  # noqa: E402
from station_ui.app import create_app  # noqa: E402
from station_ui.core import device_detector  # noqa: E402
from station_ui.routes import config as config_routes  # noqa: E402
from station_ui.routes import preview as preview_routes  # noqa: E402
from station_ui.settings import Settings  # noqa: E402
from umi.common.hardware_sdk.camera import oak_device, uvc_camera  # noqa: E402
from umi.common.tasks import task_registry_store  # noqa: E402

OAK = "1944301000000000A9"
PHONE_NAME = "iphone15p-28"
PHONE_UDID = "00008130-00000000000000A2"

# Talks the worker's contract -- a picture over --out until SIGTERM -- and
# touches no hardware. Each picture differs, so the stream has news to send.
FAKE_WORKER = r"""
import os, signal, sys, time
out = next(a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out="))
stopping = []
signal.signal(signal.SIGTERM, lambda *_: stopping.append(1))
print("[preview] streaming", flush=True)
n = 0
while not stopping:
    n += 1
    with open(out + ".part", "wb") as f:
        f.write(b"\xff\xd8frame%d\xff\xd9" % n)
    os.replace(out + ".part", out)
    time.sleep(0.05)
"""

BROKEN_WORKER = r"""
print("[preview] error: could not open the board: device busy", flush=True)
raise SystemExit(1)
"""


def worker(source):
    return lambda kind, device_id, frame_path: [
        sys.executable,
        "-c",
        source,
        f"--kind={kind}",
        f"--device={device_id}",
        f"--out={frame_path}",
    ]


def wait_for(predicate, what):
    end = time.monotonic() + gate.DEADLINE_S
    while not predicate() and time.monotonic() < end:
        time.sleep(0.02)
    assert predicate(), f"never happened: {what}"


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(task_registry_store, "get_all_tasks", lambda: dict(gate.TASKS))
    monkeypatch.setattr(
        device_detector, "get_available_oak_cameras", lambda: [gate.OAK_R, gate.OAK_L]
    )
    monkeypatch.setattr(
        device_detector, "get_available_iphones", lambda: [gate.IPHONE_R, gate.IPHONE_L]
    )
    monkeypatch.setattr(
        config_routes,
        "list_umi_gripper_device_names_for_pipeline",
        lambda pipeline: ["UMI-V4-0001", "UMI-V4-0002"],
    )
    monkeypatch.setattr(preview, "preview_argv", worker(FAKE_WORKER))


@pytest.fixture
def client(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    shutil.copytree(
        gate.REPO / "config" / "data_recording",
        root / "config" / "data_recording",
        ignore=shutil.ignore_patterns("generated"),
    )
    shutil.copytree(gate.REPO / "config" / "data_check", root / "config" / "data_check")
    built = TestClient(create_app(Settings(repo_root=root)))
    built.get("/")
    built.post("/setup/preset", data={"preset_id": "bimanual_v4"})
    gate.run_scan(built, monkeypatch, BENCH)
    yield built
    # Every test leaves the station with no preview holding a camera.
    preview.stop(built.app, wait=gate.DEADLINE_S)


def open_preview(client, kind="oak", device=OAK):
    return client.post("/config/preview/start", data={"kind": kind, "device": device})


def started(client, kind="oak", device=OAK):
    response = open_preview(client, kind, device)
    assert response.status_code == 200, response.text
    return devices_core.registry_for(client.app).get(response.json()["job_id"])


# ----------------------------------------------------------------- starting


def test_clicking_a_scanned_camera_starts_its_preview(client):
    response = open_preview(client)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["stream_url"] == f"/config/preview/stream/{body['job_id']}"
    job = devices_core.registry_for(client.app).get(body["job_id"])
    assert f"--device={OAK}" in job.argv
    assert "--kind=oak" in job.argv
    assert preview.running(client.app)


def test_clicking_a_usb_camera_starts_its_preview(client, monkeypatch):
    """Opened by its V4L2 index, which is all a USB camera has for an id."""
    gate.run_scan(client, monkeypatch, V4_STATION)

    job = started(client, "uvc", "2")

    assert "--kind=uvc" in job.argv
    assert "--device=2" in job.argv


def test_a_registered_phone_is_opened_by_its_raw_udid(client):
    """The table shows the registry name; the camera is opened by the UDID, so
    an unregistered phone and a registered one take the same path."""
    job = started(client, "iphone", PHONE_NAME)

    assert f"--device={PHONE_UDID}" in job.argv


def test_a_device_the_scan_did_not_find_is_refused(client):
    response = open_preview(client, device="1944301099999999")

    assert response.status_code == 400
    assert response.json()["message"] == preview_routes.NOT_SCANNED
    assert not preview.running(client.app)


def test_a_gripper_has_no_preview(client):
    response = open_preview(client, "gripper", "534A3035000000B8")

    assert response.status_code == 400
    assert not preview.running(client.app)


def test_a_camera_the_scan_could_not_open_is_refused(client, monkeypatch):
    gate.run_scan(
        client, monkeypatch, [record("oak", OAK, "0-1.3.3", state="unavailable")]
    )

    response = open_preview(client)

    assert response.status_code == 409
    assert "could not open" in response.json()["message"]


def test_a_read_only_tab_cannot_open_a_camera(client):
    other_tab = TestClient(client.app)

    response = other_tab.post(
        "/config/preview/start", data={"kind": "oak", "device": OAK}
    )

    assert response.status_code == 409
    assert not preview.running(client.app)


def test_no_preview_while_devices_are_refreshing(client, monkeypatch):
    monkeypatch.setattr(
        devices_core,
        "scan_argv",
        lambda: [sys.executable, "-c", "import time\ntime.sleep(60)"],
    )
    client.post("/config/devices/scan")
    scan = devices_core.current_job(client.app)
    try:
        response = open_preview(client)
    finally:
        jobs.cancel(devices_core.registry_for(client.app), scan.id)
        wait_for(lambda: scan.finished, "the scan ended")

    assert response.status_code == 409
    assert response.json()["message"] == preview_routes.REFRESHING


def test_no_preview_while_the_camera_check_holds_the_cameras(client, monkeypatch):
    monkeypatch.setattr(cal, "check_running", lambda app: True)

    response = open_preview(client)

    assert response.status_code == 409
    assert response.json()["message"] == devices_core.CHECK_HOLDS_THE_CAMERAS


def test_no_preview_while_recording(client, monkeypatch):
    monkeypatch.setattr(recording, "in_progress", lambda app: True)

    response = open_preview(client)

    assert response.status_code == 409
    assert response.json()["message"] == recording.LOCK_MESSAGE


def test_opening_a_second_preview_stops_the_first(client):
    first = started(client)
    second = started(client, "iphone", PHONE_NAME)

    wait_for(lambda: first.finished, "the first preview let go of its camera")
    assert first.state == jobs.CANCELLED
    assert not second.finished


# ---------------------------------------------------------------- streaming


def test_the_stream_sends_the_workers_pictures_and_closing_it_releases_the_camera(
    client,
):
    job = started(client)
    path = pathlib.Path(preview.state_for(client.app).frame_path)
    wait_for(path.is_file, "the worker wrote a picture")

    async def first_part():
        parts = preview_routes.frames(client.app, job, path)
        part = await parts.__anext__()
        # What the server does when the pop-up stops reading.
        await parts.aclose()
        return part

    part = asyncio.run(first_part())

    assert part.startswith(b"--preview-frame\r\nContent-Type: image/jpeg\r\n")
    assert b"\xff\xd8frame" in part
    wait_for(lambda: job.finished, "the camera was released")
    assert job.state == jobs.CANCELLED


def test_a_pop_up_closed_before_the_first_picture_still_releases_the_camera(client):
    """Under uvicorn a dropped connection only shows when a send fails, and an
    OAK takes seconds to send its first picture -- so the stream asks."""
    job = started(client)

    async def disconnected():
        return True

    async def drain():
        parts = preview_routes.frames(
            client.app, job, pathlib.Path("/nonexistent"), is_disconnected=disconnected
        )
        return [part async for part in parts]

    assert asyncio.run(drain()) == []
    wait_for(lambda: job.finished, "the camera was released")


def test_an_old_pop_ups_stream_url_does_not_reattach(client):
    first = started(client)
    started(client, "iphone", PHONE_NAME)

    response = client.get(f"/config/preview/stream/{first.id}")

    assert response.status_code == 404


def test_the_status_says_when_the_picture_is_live(client):
    job = started(client)
    wait_for(
        lambda: client.get(f"/config/preview/status/{job.id}").json()["streaming"],
        "the status reported a live picture",
    )

    assert client.get(f"/config/preview/status/{job.id}").json() == {
        "running": True,
        "streaming": True,
        "message": "",
    }


def test_the_status_says_why_a_camera_would_not_open(client, monkeypatch):
    monkeypatch.setattr(preview, "preview_argv", worker(BROKEN_WORKER))
    job = started(client)
    wait_for(lambda: job.finished, "the broken worker exited")

    status = client.get(f"/config/preview/status/{job.id}").json()

    assert status["running"] is False
    assert status["message"] == "Could not open the board: device busy."


def test_a_preview_left_open_closes_itself_and_says_so():
    timed_out = SimpleNamespace(
        lines=lambda: ["[preview] streaming"],
        error=f"timed out after {preview.TIMEOUT_S:g}s",
        state=jobs.FAILED,
    )

    assert "closed after 2 minutes" in preview._ended(timed_out)


# ---------------------------------------------------------------- releasing


def test_the_stop_route_releases_the_camera(client):
    job = started(client)

    response = client.post(f"/config/preview/stop/{job.id}")

    assert response.status_code == 204
    wait_for(lambda: job.finished, "the camera was released")


def test_refreshing_devices_releases_the_preview_first(client, monkeypatch):
    """The scan opens every OAK exclusively; a preview left holding one would
    make the scan report a healthy board unavailable."""
    job = started(client)

    gate.run_scan(client, monkeypatch, BENCH)

    assert job.finished
    status = client.get(f"/config/preview/status/{job.id}").json()
    assert "stopped so the camera could be used elsewhere" in status["message"]


def test_starting_a_recording_releases_the_preview_first(client):
    job = started(client)

    client.post("/record/start")

    assert job.finished


def test_starting_the_camera_check_releases_the_preview_first(client, monkeypatch):
    monkeypatch.setattr(cal, "recording_config", lambda app: {})
    monkeypatch.setattr(
        cal,
        "check_argv",
        lambda path, preview_path=None: [
            sys.executable,
            "-c",
            "import time\ntime.sleep(60)",
        ],
    )
    job = started(client)
    try:
        client.post("/calibration/check/start")
        assert job.finished
    finally:
        cal.stop_check(client.app, wait=gate.DEADLINE_S)


# --------------------------------------------------------------------- page


def test_camera_and_phone_ids_open_a_preview_and_grippers_do_not(client):
    page = client.get("/config").text

    assert f'data-preview-kind="oak" data-preview-device="{OAK}"' in page
    assert f'data-preview-kind="iphone" data-preview-device="{PHONE_NAME}"' in page
    assert 'data-preview-device="534A3035000000B8"' not in page
    assert 'id="device-preview"' in page
    assert "/static/device_preview.js" in page


def test_usb_camera_ids_open_a_preview_too(client, monkeypatch):
    gate.run_scan(client, monkeypatch, V4_STATION)

    page = client.get("/config").text

    assert 'data-preview-kind="uvc" data-preview-device="0"' in page
    assert 'data-preview-kind="uvc" data-preview-device="2"' in page


def test_a_read_only_tab_sees_plain_ids(client):
    page = TestClient(client.app).get("/config").text

    assert "data-preview-device" not in page
    assert OAK in page


# ------------------------------------------------------------- the worker


def test_an_oak_board_is_shown_as_one_tile_per_sensor_side_by_side():
    frame = np.zeros((400, 640, 3), np.uint8)

    canvas = preview_worker.compose(
        [("CAM_A AR0234", frame), ("CAM_B AR0234", frame), ("CAM_C AR0234", None)],
        height=200,
    )

    assert canvas.shape[0] == 200
    # Three 16:10 tiles at 200px high, two gaps between them.
    assert canvas.shape[1] == 3 * 320 + 2 * preview_worker._GAP


def test_a_silent_sensor_and_a_mono_one_still_get_their_tiles():
    """A v3 board's unused CAM_C never sends a picture, and says so; a mono
    sensor's grey frame is drawn as-is beside the colour ones."""
    colour = np.zeros((400, 640, 3), np.uint8)
    grey = np.zeros((400, 640), np.uint8)

    canvas = preview_worker.compose(
        [
            ("CAM_A", colour),
            ("CAM_B OV9282", grey),
            ("CAM_C", None, preview_worker.NO_PICTURE),
        ],
        height=200,
    )

    assert canvas.shape == (200, 3 * 320 + 2 * preview_worker._GAP, 3)


def test_a_wide_board_is_scaled_to_fit_the_pop_up():
    frame = np.zeros((400, 640, 3), np.uint8)

    canvas = preview_worker.compose([("", frame)] * 4, height=720)

    assert canvas.shape[1] == preview_worker.MAX_WIDTH


def test_nothing_to_show_is_no_picture():
    assert preview_worker.compose([]) is None


def test_a_picture_is_replaced_whole(tmp_path):
    out = tmp_path / "frame.jpg"
    image = np.full((60, 80, 3), 200, np.uint8)

    assert preview_worker.write_jpeg(image, str(out))

    assert cv2.imdecode(np.frombuffer(out.read_bytes(), np.uint8), 1).shape == (
        60,
        80,
        3,
    )
    assert list(tmp_path.iterdir()) == [out], "a partial file was left behind"


def test_the_worker_writes_until_told_to_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(preview_worker, "FRAME_INTERVAL_S", 0.0)
    source = SimpleNamespace(tiles=lambda: [("CAM_A", np.zeros((40, 64, 3), np.uint8))])
    calls = []

    def should_stop():
        calls.append(1)
        return len(calls) > 3

    preview_worker.run(source, str(tmp_path / "frame.jpg"), 40, should_stop)

    assert (tmp_path / "frame.jpg").is_file()
    assert len(calls) == 4


def test_a_camera_that_will_not_open_is_reported_and_exits_non_zero(
    tmp_path, monkeypatch, capsys
):
    def _busy(kind, device):
        raise RuntimeError("X_LINK_DEVICE_ALREADY_IN_USE")

    monkeypatch.setattr(preview_worker, "_open", _busy)

    code = preview_worker.main(
        ["--kind=oak", f"--device={OAK}", f"--out={tmp_path / 'frame.jpg'}"]
    )

    assert code == 1
    assert (
        f"[preview] error: could not open {OAK}: X_LINK_DEVICE_ALREADY_IN_USE"
        in capsys.readouterr().out
    )


def test_the_worker_closes_the_camera_and_tidies_up_on_the_way_out(
    tmp_path, monkeypatch
):
    out = tmp_path / "frame.jpg"
    closed = []
    source = SimpleNamespace(
        tiles=lambda: [("CAM_A", np.zeros((40, 64, 3), np.uint8))],
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(preview_worker, "_open", lambda kind, device: source)

    def _stop_after_one(source, out, height, should_stop):
        preview_worker.write_jpeg(np.zeros((40, 64, 3), np.uint8), out)

    monkeypatch.setattr(preview_worker, "run", _stop_after_one)

    assert preview_worker.main(["--kind=oak", f"--device={OAK}", f"--out={out}"]) == 0
    assert closed == [True]
    assert not out.exists()


def test_the_worker_never_runs_on_an_unknown_kind():
    with pytest.raises(SystemExit):
        preview_worker.main(["--kind=gripper", "--device=x", "--out=/dev/null"])


class FakeOakDevice:
    """`OAKDevice` as the preview drives it: a board reporting *sensors*, and a
    picture per read on every stream it is asked to build. No device."""

    def __init__(self, sensors, **kwargs):
        features = [
            SimpleNamespace(socket=SimpleNamespace(name=socket), sensorName=name)
            for socket, name in sensors.items()
        ]
        self.device = SimpleNamespace(getConnectedCameraFeatures=lambda: features)
        self.configs = []

    def build_pipeline(self, configs):
        self.configs = configs

    def start(self):
        pass

    def get_streams(self):
        return [SimpleNamespace(get_frame=self._frame) for _ in self.configs]

    def _frame(self):
        time.sleep(0.01)
        return SimpleNamespace(rgb=np.zeros((400, 640, 3), np.uint8))

    def stop(self):
        pass


@pytest.mark.parametrize(
    "sensors, tiles",
    [
        # The V4 gripper baseboard: one AR0234, reported on CAM_B and CAM_C.
        ({"CAM_B": "AR0234", "CAM_C": "AR0234"}, ["CAM_B AR0234"]),
        # A head OAK-D: its CAM_C is the real right mono camera.
        (
            {"CAM_A": "IMX378", "CAM_B": "OV9282", "CAM_C": "OV9282"},
            ["CAM_A IMX378", "CAM_B OV9282", "CAM_C OV9282"],
        ),
        # Two different modules on CAM_B and CAM_C are two real cameras.
        ({"CAM_B": "IMX378", "CAM_C": "AR0234"}, ["CAM_B IMX378", "CAM_C AR0234"]),
    ],
)
def test_only_the_v4_boards_phantom_cam_c_is_left_out(monkeypatch, sensors, tiles):
    """Never built, not merely hidden: a node on the phantom sends no picture,
    and it can blow the real sensor's exposure out to white."""
    built = []

    def _build(**kwargs):
        built.append(FakeOakDevice(sensors, **kwargs))
        return built[-1]

    monkeypatch.setattr(oak_device, "OAKDevice", _build)

    source = preview_worker._open("oak", OAK)
    try:
        labels = [label for label, *_ in source.tiles()]
    finally:
        source.close()

    assert [config["socket"] for config in built[0].configs] == [
        tile.split()[0] for tile in tiles
    ]
    assert labels == tiles


# --------------------------------------------------- the worker, USB cameras


class FakeUvcCamera:
    """`UVCCamera` as the worker drives it: a picture per read, no device."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.released_by = []

    def start_streaming(self):
        pass

    def get_frame(self):
        time.sleep(0.01)
        return SimpleNamespace(rgb=np.zeros((720, 1280, 3), np.uint8))

    def stop_streaming(self):
        self.released_by.append(threading.current_thread())


@pytest.fixture
def uvc_cameras(monkeypatch):
    """Every `UVCCamera` the worker builds, in order, each one `cls`."""
    built = []

    def _install(cls=FakeUvcCamera):
        def _build(**kwargs):
            built.append(cls(**kwargs))
            return built[-1]

        monkeypatch.setattr(uvc_camera, "UVCCamera", _build)
        return built

    return _install


def test_a_usb_camera_previews_in_the_mode_it_is_recorded_in():
    """A picture in the pop-up then also says the recording can open the camera."""
    production = gate.REPO / "config" / "data_recording" / "production"
    modes = {
        (tuple(source["params"]["camera_resolution"]), source["params"]["fps"])
        for template in production.glob("*.yaml")
        for source in yaml.safe_load(template.read_text()).get("video_sources", [])
        if source["type"] == "uvc_camera"
    }

    assert modes == {(preview_worker.UVC_RESOLUTION, preview_worker.UVC_FPS)}


def test_a_usb_camera_shows_its_picture_under_its_video_node(uvc_cameras):
    built = uvc_cameras()

    source = preview_worker._open("uvc", "4")
    try:
        wait_for(lambda: source.tiles()[0][1] is not None, "the first picture")
        (label, frame), *others = source.tiles()
    finally:
        source.close()

    (camera,) = built
    assert camera.kwargs["camera_index"] == 4, "cv2 opens an int index"
    assert label == "/dev/video4"
    assert frame.shape == (720, 1280, 3)
    assert others == []
    # Released once, by the thread that reads it, so never under a read.
    assert len(camera.released_by) == 1
    assert camera.released_by[0] is not threading.current_thread()


def test_a_usb_camera_that_sends_nothing_says_so(uvc_cameras, monkeypatch):
    class Silent(FakeUvcCamera):
        def get_frame(self):
            return None

    uvc_cameras(Silent)
    monkeypatch.setattr(preview_worker, "NO_PICTURE_AFTER_S", 0.0)

    source = preview_worker._open("uvc", "4")
    try:
        tiles = source.tiles()
    finally:
        source.close()

    assert tiles == [("/dev/video4", None, preview_worker.NO_PICTURE)]


def test_a_usb_camera_that_will_not_open_is_reported(uvc_cameras, tmp_path, capsys):
    class Refused(FakeUvcCamera):
        def start_streaming(self):
            raise RuntimeError("Failed to open camera with index 4")

    uvc_cameras(Refused)

    code = preview_worker.main(
        ["--kind=uvc", "--device=4", f"--out={tmp_path / 'frame.jpg'}"]
    )

    assert code == 1
    assert (
        "[preview] error: could not open 4: Failed to open camera with index 4"
        in capsys.readouterr().out
    )

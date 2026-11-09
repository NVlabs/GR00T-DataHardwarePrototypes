# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The sensor module behind each OAK stream, recorded in the config.

`resolution_for` reads the per-socket module out of the scan to pick an
aspect-correct size, and keeps only the number it derived. Without the module
written down, a session records at 640x480 with nothing anywhere saying the
reason was an IMX378 on CAM_B, and anyone reading the config back (or the
LeRobot conversion, or whoever is diagnosing a bad calibration months later)
would have to re-derive it from a bench that has since been re-cabled.

Writing it down costs one key. It is descriptive, like `placement` beside it:
nothing acts on it, and `OAKDevice.build_pipeline` reads
`name`/`socket`/`resolution`/`latency` by key, so an extra one is inert.

OAK only, deliberately. It is the one source type whose optics vary per socket
on one board, and the only one the scan reports a module for.
"""

import pathlib
import shutil

import pytest
from fastapi.testclient import TestClient

yaml = pytest.importorskip("yaml")

import test_unavailable_device_gate as gate  # noqa: E402

from station_ui import settings  # noqa: E402
from station_ui.app import create_app  # noqa: E402
from station_ui.core import device_detector  # noqa: E402
from station_ui.core.device_scan import (  # noqa: E402
    DeviceScanner,
    sensor_module_for,
)
from station_ui.routes import config as config_routes  # noqa: E402
from station_ui.settings import Settings  # noqa: E402
from umi.common.tasks import task_registry_store  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]

OAK_L = gate.OAK_L
OAK_R = gate.OAK_R
IMX378 = "IMX378"
AR0234 = "AR0234"


def oak_record(name, device_id, sockets):
    record = gate.found_record("oak", device_id, name)
    record["metadata"] = {"sockets": dict(sockets)}
    return record


def scan_with_an_imx378():
    """The left board carrying a 4:3 module on CAM_B, the socket its wrist
    stream records, and a 16:10 one on CAM_C, so a lookup that ignored the
    socket is caught; the right board is 16:10 throughout."""
    return [
        oak_record(OAK_L, gate.OAK_L_ID, {"CAM_B": IMX378, "CAM_C": AR0234}),
        oak_record(OAK_R, gate.OAK_R_ID, {"CAM_B": AR0234, "CAM_C": AR0234}),
        gate.found_record("iphone", "00008130-000000000001", gate.IPHONE_R),
        gate.found_record("iphone", "00008130-000000000002", gate.IPHONE_L),
    ]


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(task_registry_store, "get_all_tasks", lambda: dict(gate.TASKS))
    monkeypatch.setattr(
        device_detector, "get_available_oak_cameras", lambda: [OAK_R, OAK_L]
    )
    monkeypatch.setattr(
        device_detector, "get_available_iphones", lambda: [gate.IPHONE_R, gate.IPHONE_L]
    )
    monkeypatch.setattr(
        config_routes,
        "list_umi_gripper_device_names_for_pipeline",
        lambda **kw: ["UMI-V4-0001", "UMI-V4-0002"],
    )


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    shutil.copytree(
        REPO / "config" / "data_recording",
        root / "config" / "data_recording",
        ignore=shutil.ignore_patterns("generated"),
    )
    shutil.copytree(REPO / "config" / "data_check", root / "config" / "data_check")
    return root


@pytest.fixture
def client(repo):
    built = TestClient(create_app(Settings(repo_root=repo)))
    built.get("/")
    built.post("/setup/preset", data={"preset_id": "bimanual_v4"})
    # Most of these tests are about the two arms; the head OAK row a fresh
    # session starts with would put a device no scan reported into every config.
    for assignment in list(built.app.state.rig.draft.optional):
        built.post(
            "/config/optional/remove?confirm=yes",
            data={"config_form": "1", "logical_name": assignment.logical_name},
        )
    return built


def generated():
    return settings.state_dir_from_environment() / "data_recording" / "bimanual_v4.yaml"


def streams_by_name(cfg):
    out = {}
    for source in cfg.get("video_sources") or []:
        for stream in source.get("streams") or []:
            if stream.get("name"):
                out[stream["name"]] = stream
    return out


def sources_by_name(cfg):
    return {
        source.get("name"): source
        for source in (cfg.get("video_sources") or [])
        if source.get("name")
    }


def apply_form(**overrides):
    data = dict(gate.handcrafted())
    data.pop("optional.head_oak_camera.device", None)
    data.update(overrides)
    return data


# ----------------------------------------------------------------- the lookup


def test_the_module_on_a_socket_is_read_from_the_scan():
    scanner = DeviceScanner.from_records(scan_with_an_imx378())

    assert sensor_module_for(OAK_L, "CAM_B", scanner) == IMX378
    assert sensor_module_for(OAK_L, "CAM_C", scanner) == AR0234
    assert sensor_module_for(OAK_R, "CAM_B", scanner) == AR0234


def test_an_unknown_socket_or_device_answers_with_nothing():
    """Empty, never a guess. The whole value of writing this down is that it
    says what was really on the socket, so an invented answer is worse than no
    answer -- and the caller omits the key rather than writing a null."""
    scanner = DeviceScanner.from_records(scan_with_an_imx378())

    assert sensor_module_for(OAK_L, "CAM_A", scanner) == ""
    assert sensor_module_for("oak-FFC-99", "CAM_B", scanner) == ""
    assert sensor_module_for("", "CAM_B", scanner) == ""
    assert sensor_module_for(OAK_L, "CAM_B", DeviceScanner()) == ""


# ------------------------------------------------------------- through Apply


def test_apply_records_the_module_behind_each_oak_stream(client, monkeypatch):
    """The pair this exists for: the size and the reason it is that size."""
    gate.run_scan(client, monkeypatch, scan_with_an_imx378())

    response = client.post("/config/apply", data=apply_form())
    assert response.status_code == 200, response.text

    streams = streams_by_name(yaml.safe_load(generated().read_text()))
    assert streams["left_umi_camera"]["sensor_module"] == IMX378
    assert list(streams["left_umi_camera"]["resolution"]) == [640, 480]
    assert streams["right_umi_camera"]["sensor_module"] == AR0234


def test_an_unscanned_board_leaves_the_key_out_rather_than_guessing(client):
    """No scan has run, so nothing knows what is on CAM_B.

    The resolution still falls back to the 16:10 default and still says so
    through the warning banner -- but a module is not derivable from a default,
    and writing one would be a fabrication in a file that outlives the bench.
    """
    response = client.post("/config/apply", data=apply_form())
    assert response.status_code == 200, response.text

    streams = streams_by_name(yaml.safe_load(generated().read_text()))
    for name in ("left_umi_camera", "right_umi_camera"):
        assert "sensor_module" not in streams[name], streams[name]


def test_only_oak_streams_carry_it(client, monkeypatch):
    """OAK only. An iPhone or a USB camera reports no per-socket module, and
    inventing a key for it would make the absence on OAK look like a bug
    later."""
    gate.run_scan(client, monkeypatch, scan_with_an_imx378())

    client.post("/config/apply", data=apply_form())

    cfg = yaml.safe_load(generated().read_text())
    for name, source in sources_by_name(cfg).items():
        assert "sensor_module" not in source, f"{name} is not an OAK: {source}"


def test_a_head_oak_records_its_module_too(repo, monkeypatch):
    """The optional OAK is a single-source block with no `streams` list, so its
    module sits at source level -- which is where `recording_utils` already
    looks for `placement` on exactly these shapes."""
    client = TestClient(create_app(Settings(repo_root=repo)))
    client.get("/")
    client.post("/setup/preset", data={"preset_id": "bimanual_v4"})
    client.post(
        "/config/optional/add", data={"config_form": "1", "type_id": "head_oak"}
    )
    head = oak_record("oak-head", "oak-head-mxid", {"CAM_A": IMX378})
    gate.run_scan(client, monkeypatch, scan_with_an_imx378() + [head])

    response = client.post(
        "/config/apply",
        data=dict(
            gate.handcrafted(), **{"optional.head_oak_camera.device": "oak-head"}
        ),
    )
    assert response.status_code == 200, response.text

    cfg = yaml.safe_load(generated().read_text())
    assert sources_by_name(cfg)["head_oak_camera"]["sensor_module"] == IMX378


def test_the_module_reaches_the_session_config(client, monkeypatch):
    """It is only worth writing if it survives into the recorded session, which
    is the artifact anyone reading a dataset back actually has."""
    from umi.common.utility.recording_utils import _VIDEO_STREAM_META_KEYS

    gate.run_scan(client, monkeypatch, scan_with_an_imx378())
    client.post("/config/apply", data=apply_form())

    assert "sensor_module" in _VIDEO_STREAM_META_KEYS

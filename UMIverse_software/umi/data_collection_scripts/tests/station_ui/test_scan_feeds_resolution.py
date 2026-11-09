# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The scan the operator ran is the scan the assembler resolves against.

`resolution_for` consults the module-global
`station_ui.core.device_scan.SCANNER` by default, and nothing in the running
server calls `SCANNER.scan()` -- the console enumerates hardware in a capped
child (`station_ui.scan_worker`) and keeps the parsed result in the job
registry. Resolving against the global would return None for every device,
every OAK stream would fall back to the 16:10 default, and the page would tell
the operator to press Refresh devices -- advice that cannot work, because
refreshing fills the other scan.

Two consequences, and this file asserts both are avoided: the operator being
told a falsehood, and a 4:3 IMX378 silently recording at 640x400 with 17% of
its FOV cropped.

No test here opens hardware: every scan is the same `python -c` child the rest
of the station suite uses.
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
    DEFAULT_STREAM_RESOLUTION,
    DeviceScanner,
    resolution_for,
)
from station_ui.routes import config as config_routes  # noqa: E402
from station_ui.settings import Settings  # noqa: E402
from umi.common.tasks import task_registry_store  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
# This file is about Apply resolving against the scan, so the real gate runs.
pytestmark = pytest.mark.oak_gate

# The two wrist boards of the bimanual_v4 template, and what is on their
# sockets. CAM_B is the socket the template's streams name.
OAK_L = gate.OAK_L
OAK_R = gate.OAK_R

# 4:3, and the whole point: asking it for 16:10 makes the ISP centre-crop.
IMX378 = "IMX378"
# 16:10, which the default already matches -- the control case.
AR0234 = "AR0234"


def oak_record(name, device_id, sockets):
    record = gate.found_record("oak", device_id, name)
    record["metadata"] = {"sockets": dict(sockets)}
    return record


def _scan_with_an_imx378():
    """The left board carrying a 4:3 module on CAM_B and a 16:10 one on CAM_C,
    the right board 16:10 throughout."""
    return [
        oak_record(OAK_L, gate.OAK_L_ID, {"CAM_B": IMX378, "CAM_C": AR0234}),
        oak_record(OAK_R, gate.OAK_R_ID, {"CAM_B": AR0234, "CAM_C": AR0234}),
        gate.found_record("iphone", "00008130-000000000001", gate.IPHONE_R),
        gate.found_record("iphone", "00008130-000000000002", gate.IPHONE_L),
    ]


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    """Deterministic task and device registries.

    Patched at the module each caller reads through, never at the function it
    imported -- `_REGISTRY_LOOKUP` stores attribute names for exactly this
    reason, and patching the wrong one passes while changing nothing.
    """
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
        lambda pipeline: ["UMI-V4-0001", "UMI-V4-0002"],
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
    # Use arm cameras only so these tests isolate scanned module resolution
    # without introducing an unscanned optional head camera.
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


def apply_form(**overrides):
    data = dict(gate.handcrafted())
    # Omit the removed head row's field from the submitted form.
    data.pop("optional.head_oak_camera.device", None)
    data.update(overrides)
    return data


# ------------------------------------------------------- the resolver itself


def test_a_scanner_built_from_scan_records_can_find_a_device_by_name():
    """`from_records` is the bridge between the child's JSON and the resolver."""
    scanner = DeviceScanner.from_records(_scan_with_an_imx378())

    assert scanner.by_identifier(OAK_L) is not None
    # By raw id too: unregistered hardware is stored by id, and a name-only
    # lookup would give an unregistered IMX378 the wrong resolution.
    assert scanner.by_identifier(gate.OAK_L_ID) is not None
    assert scanner.by_identifier("oak-FFC-99") is None


def test_the_bridge_carries_the_sensor_modules_the_aspect_is_read_from():
    scanner = DeviceScanner.from_records(_scan_with_an_imx378())

    assert resolution_for(OAK_L, "CAM_B", scanner) == ([640, 480], None)
    assert resolution_for(OAK_L, "CAM_C", scanner) == ([640, 400], None)


def test_an_empty_scanner_is_what_produced_the_phantom_warning():
    """At the unit level: a scanner nothing has filled answers the default plus
    a warning for every OAK.
    """
    resolution, warning = resolution_for(OAK_L, "CAM_B", DeviceScanner())

    assert resolution == list(DEFAULT_STREAM_RESOLUTION)
    assert "not found in the device scan" in warning


# ------------------------------------------------------------ through Apply


def test_apply_resolves_each_stream_against_the_scan_that_ran(client, monkeypatch):
    """End to end: refresh devices, apply, and the 4:3 module keeps its FOV."""
    gate.run_scan(client, monkeypatch, _scan_with_an_imx378())

    response = client.post("/config/apply", data=apply_form())
    assert response.status_code == 200, response.text

    streams = streams_by_name(yaml.safe_load(generated().read_text()))
    # The socket carrying the IMX378, and only it, gets the 4:3 size.
    assert list(streams["left_umi_camera"]["resolution"]) == [640, 480]
    assert list(streams["right_umi_camera"]["resolution"]) == [640, 400]


def test_a_refreshed_page_no_longer_claims_the_cameras_were_not_checked(
    client, monkeypatch
):
    """Refreshing devices clears the "not scanned" warning."""
    gate.run_scan(client, monkeypatch, _scan_with_an_imx378())

    body = client.post("/config/apply", data=apply_form()).text

    assert "have not been checked" not in body, body[:600]
    assert "not scanned" not in body


def test_without_a_scan_apply_is_refused_rather_than_guessing(client):
    """The warning is not deleted -- it became a refusal.

    A config assembled before anything was scanned genuinely does not know the
    modules, and a silently cropped FOV is what this exists to prevent. So
    nothing is written.
    """
    response = client.post("/config/apply", data=apply_form())

    assert response.status_code == 400, response.text[:400]
    assert "has not been scanned" in response.text
    assert not generated().exists()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A device the scan positively refused may not be recorded from.

The picker lists an unavailable device on purpose -- "gone" and "here but
wedged" need different actions from the operator, and a vanished row says
neither. But listing it is not offering it: a browser will happily submit an
option whose label reads `unavailable`, and a hand-crafted POST never sees the
label at all. So the refusal lives on the server, on the value, and the only
thing the markup can do is explain it.

The line this suite defends is not "connected or nothing". Three states, not
two:

* the scan enumerated the device and could not open it -> refused
* the scan never mentioned the device -> accepted, saved exactly as typed
* no scan has run -> accepted

The middle one is the reason the gate is narrow. Configs are prepped before the
hardware reaches the bench, and the device registry may not list every device
yet, so a connected-only rule would empty the screen.

**No test in this file opens hardware.** Every scan is a `python -c` child
named by `devices.scan_argv()`.
"""

import json
import pathlib
import shutil
import sys
import time

import pytest
from fastapi.testclient import TestClient

yaml = pytest.importorskip("yaml")


from station_ui import devices as devices_core  # noqa: E402
from station_ui import scan_worker, settings  # noqa: E402
from station_ui.app import create_app  # noqa: E402
from station_ui.core import device_detector  # noqa: E402
from station_ui.routes import config as config_routes  # noqa: E402
from station_ui.settings import Settings  # noqa: E402
from umi.common.tasks import task_registry_store  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
DEADLINE_S = 30.0
# This file tests the gate itself, so it opts out of the conftest's stand-down.
pytestmark = pytest.mark.oak_gate

BAD_REQUEST = 400

OAK_R = "oak-FFC-01"
OAK_L = "oak-FFC-02"
OAK_R_ID = "1944301000000001"
OAK_L_ID = "1944301000000002"
IPHONE_R = "iphone15p-01"
IPHONE_L = "iphone15p-02"
# A device no scan below reports and no registry knows. This is the free-text
# escape hatch, and it must keep working.
NEVER_SEEN = "oak-FFC-99"

TASKS = {
    "plane_assembly_v9": {
        "name": "plane_assembly_v9",
        "short_description": "Assemble the plane",
        "mode": "bimanual",
        "HW_version": "v2.1",
        "training_group": "plane_assembly_v9",
    },
}


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    """Deterministic task and device registries.

    Patched at the module the route reads through, not at the function it
    imported: `_REGISTRY_LOOKUP` stores attribute names for exactly this
    reason, and patching the wrong one passes while changing nothing.
    """
    monkeypatch.setattr(task_registry_store, "get_all_tasks", lambda: dict(TASKS))
    monkeypatch.setattr(
        device_detector, "get_available_oak_cameras", lambda: [OAK_R, OAK_L]
    )
    monkeypatch.setattr(
        device_detector, "get_available_iphones", lambda: [IPHONE_R, IPHONE_L]
    )
    monkeypatch.setattr(
        config_routes,
        "list_umi_gripper_device_names_for_pipeline",
        lambda pipeline: ["UMI-V4-0001", "UMI-V4-0002"],
    )


# --------------------------------------------------------------------- fakes


def found_record(kind, device_id, registry_name=None, state="available", error=None):
    return {
        "kind": kind,
        "device_id": device_id,
        "registry_name": registry_name,
        "state": state,
        "error": error,
        "metadata": {},
    }


def oak_found(device_id, registry_name, sockets=None, **kw):
    """An OAK the scan opened and read: a V4 wrist board (a 16:10 AR0234 on
    CAM_B, which the baseboard also reports on CAM_C) unless told otherwise. An
    OAK record with no modules is what the scan produces when it could not read
    the board, and Apply refuses it."""
    record = found_record("oak", device_id, registry_name, **kw)
    record["metadata"] = {
        "sockets": dict(sockets or {"CAM_B": "AR0234", "CAM_C": "AR0234"})
    }
    return record


HEAD_FOUND = oak_found("oak-head-mxid", "oak-head", {"CAM_A": "IMX378"})


def fake_scan_argv(found=()):
    """A child that talks the scan's wire format and touches no hardware."""
    payload = scan_worker.RESULT_PREFIX + json.dumps({"devices": list(found)})
    source = (
        "import sys\n"
        "print('[scan] starting', flush=True)\n"
        f"print({payload!r}, flush=True)\n"
        "sys.exit(0)\n"
    )
    return [sys.executable, "-c", source]


def run_scan(client, monkeypatch, found):
    """Run one fake scan to completion, so `/config` has a result to fold."""
    monkeypatch.setattr(devices_core, "scan_argv", lambda: fake_scan_argv(found))
    client.post("/config/devices/scan")
    job = devices_core.current_job(client.app)
    assert job is not None, "no scan was started"
    end = time.monotonic() + DEADLINE_S
    while not job.finished and time.monotonic() < end:
        time.sleep(0.02)
    assert job.finished, f"the fake scan never ended: {job.state}"
    return job


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def repo(tmp_path):
    """A repo root of its own: assembly writes generated/ and repins data_check."""
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
    # A fresh session has no head camera; these tests are about the head OAK.
    built.post("/config/optional/add", data={"config_form": "1", "type_id": "head_oak"})
    return built


def generated(repo, preset="bimanual_v4"):
    # Use the isolated station state directory for generated files, not repo.
    return settings.state_dir_from_environment() / "data_recording" / f"{preset}.yaml"


# One box per physical BLOCK, keyed by the first stream of the source it fills.
# The trigger cameras are USB modules, named by their V4L2 index.
CAMERAS = {
    "camera.right_umi_camera": OAK_R,
    "camera.left_umi_camera": OAK_L,
    "camera.right_trigger_camera": "2",
    "camera.left_trigger_camera": "4",
    "camera.right_tracking_camera": IPHONE_R,
    "camera.left_tracking_camera": IPHONE_L,
}


def handcrafted(**overrides) -> dict:
    """A complete apply POST built from nothing the page rendered.

    Deliberately not derived from the HTML: an operator who can reach the form
    can also reach `curl`, and a gate that only exists in the markup is no gate.
    """
    data = {
        # The marker a full configuration form carries.
        "config_form": "1",
        "operator_name": "Test Operator",
        "task_name": "plane_assembly_v9",
        "gripper.left": "UMI-V4-0001",
        "gripper.right": "UMI-V4-0002",
        # Include the fixture's head-camera selection even when the row was not
        # added by this request; omitted fields do not remove draft assignments.
        "optional.head_oak_camera.device": "oak-head",
    }
    data.update(CAMERAS)
    data.update(overrides)
    return data


# ------------------------------------------------------------------- the gate


def test_a_device_the_scan_could_not_open_is_refused(client, repo, monkeypatch):
    """Without the gate, an operator who picks a wedged camera learns at record
    time, when the session is already the thing being lost."""
    run_scan(
        client,
        monkeypatch,
        [
            found_record(
                "oak",
                OAK_R_ID,
                OAK_R,
                state="unavailable",
                error="X_LINK_DEVICE_ALREADY_IN_USE",
            ),
            found_record("oak", OAK_L_ID, OAK_L),
        ],
    )

    response = client.post("/config/apply", data=handcrafted())

    assert response.status_code == BAD_REQUEST, (
        "an unavailable device was accepted; the operator finds out at record time"
    )
    assert not generated(repo).exists(), "a refused apply still wrote the config"


def test_the_refusal_names_the_device_and_why(client, repo, monkeypatch):
    """A banner the operator cannot act on sends them to press Apply again."""
    run_scan(
        client,
        monkeypatch,
        [
            found_record(
                "oak",
                OAK_R_ID,
                OAK_R,
                state="unavailable",
                error="X_LINK_DEVICE_ALREADY_IN_USE",
            ),
            found_record("oak", OAK_L_ID, OAK_L),
        ],
    )

    response = client.post("/config/apply", data=handcrafted())

    assert response.status_code == BAD_REQUEST
    assert OAK_R in response.text
    assert "X_LINK_DEVICE_ALREADY_IN_USE" in response.text, response.text[:2000]


def test_an_unavailable_device_with_no_reason_is_still_refused(
    client, repo, monkeypatch
):
    """The worker records *why* when it knows. When it does not, the refusal
    still has to happen -- silence about the cause is not consent."""
    run_scan(
        client,
        monkeypatch,
        [found_record("oak", OAK_R_ID, OAK_R, state="unavailable")],
    )

    response = client.post("/config/apply", data=handcrafted())

    assert response.status_code == BAD_REQUEST
    assert not generated(repo).exists()


def test_an_unavailable_gripper_is_refused_too(client, repo, monkeypatch):
    """The gate is on the value, not on the camera rows. A gripper that will
    not open blocks recording outright."""
    run_scan(
        client,
        monkeypatch,
        [
            found_record(
                "gripper",
                "/dev/ttyACM1",
                "UMI-V4-0001",
                state="unavailable",
                error="permission denied on /dev/ttyACM1",
            )
        ],
    )

    response = client.post("/config/apply", data=handcrafted())

    assert response.status_code == BAD_REQUEST
    assert not generated(repo).exists()


# ------------------------------------------------- what must NOT be refused


def test_a_device_the_scan_never_saw_is_still_saved_exactly_as_typed(
    client, repo, monkeypatch
):
    """Free-text entry is the escape hatch, and it is load-bearing -- for every
    kind whose config does not depend on the scan. A hand-typed phone the scan
    never mentioned is saved exactly as typed.

    Not the same answer as "the scan refused it": a config gets prepped before
    the hardware is on the bench, and the device registry may not list every
    device yet.
    """
    run_scan(
        client,
        monkeypatch,
        [oak_found(OAK_R_ID, OAK_R), oak_found(OAK_L_ID, OAK_L), HEAD_FOUND],
    )

    response = client.post(
        "/config/apply",
        data=handcrafted(**{"camera.right_tracking_camera": "iphone15p-99"}),
    )

    assert response.status_code == 200, response.text
    config = yaml.safe_load(generated(repo).read_text())
    names = {
        source.get("iphone_device_name")
        or (source.get("params") or {}).get("iphone_device_name")
        for source in config["video_sources"]
    }
    assert "iphone15p-99" in names, (
        f"a hand-typed device never reached the config: {names}"
    )


def test_an_oak_the_scan_never_saw_is_refused_because_its_size_would_be_a_guess(
    client, repo, monkeypatch
):
    """The one exception to the escape hatch. An OAK's recorded resolution
    comes from its sensor modules, so a resolution written for an OAK no scan
    has reported would be a guess -- and a wrong guess records a cropped
    image."""
    run_scan(client, monkeypatch, [oak_found(OAK_L_ID, OAK_L), HEAD_FOUND])

    response = client.post(
        "/config/apply",
        data=handcrafted(**{"camera.right_umi_camera": NEVER_SEEN}),
    )

    assert response.status_code == BAD_REQUEST, response.text[:400]
    assert "Not seen by the device scan" in response.text
    assert not generated(repo).exists()


def test_a_connected_device_is_accepted(client, repo, monkeypatch):
    """The gate must not be a wall: everything the scan opened still applies."""
    run_scan(
        client,
        monkeypatch,
        [
            oak_found(OAK_R_ID, OAK_R),
            oak_found(OAK_L_ID, OAK_L),
            HEAD_FOUND,
            found_record("iphone", "00008130-000000000001", IPHONE_R),
            found_record("iphone", "00008130-000000000002", IPHONE_L),
        ],
    )

    response = client.post("/config/apply", data=handcrafted())

    assert response.status_code == 200, response.text
    assert generated(repo).exists()


def test_before_a_scan_has_run_only_the_oak_rows_are_refused(client, repo):
    """A page that has never scanned makes no claim about phones or grippers,
    so it may not refuse them. It makes no claim about an OAK's sensor either
    -- which is why it may not write one."""
    response = client.post("/config/apply", data=handcrafted())

    assert response.status_code == BAD_REQUEST, response.text[:400]
    assert not generated(repo).exists()
    body = response.text
    assert "has not been scanned" in body
    # Every OAK box is named -- two wrist boards and the head -- and no phone,
    # USB trigger camera or gripper box is.
    assert body.count("Not seen by the device scan") == 3, body.count(
        "Not seen by the device scan"
    )


def test_an_oak_the_scan_saw_but_could_not_read_is_refused(client, repo, monkeypatch):
    """Enumerated, state available, no modules: the scan did not get into the
    board, so the size would be a guess exactly as if the board were absent."""
    unread = found_record("oak", "oak-head-mxid", "oak-head")  # metadata: {}
    run_scan(
        client,
        monkeypatch,
        [oak_found(OAK_R_ID, OAK_R), oak_found(OAK_L_ID, OAK_L), unread],
    )

    response = client.post("/config/apply", data=handcrafted())

    assert response.status_code == BAD_REQUEST, response.text[:400]
    assert "Not seen by the device scan" in response.text
    assert not generated(repo).exists()


def test_kinds_the_scan_did_not_report_are_never_refused(client, repo, monkeypatch):
    """The scan here reports only the OAK boards. The phones and grippers it
    did not report are not refused: absence of evidence is not a refusal."""
    run_scan(
        client,
        monkeypatch,
        [
            oak_found(OAK_R_ID, OAK_R),
            oak_found(OAK_L_ID, OAK_L),
            HEAD_FOUND,
        ],
    )

    response = client.post("/config/apply", data=handcrafted())

    assert response.status_code == 200, response.text
    assert generated(repo).exists()


# ------------------------------------------------------ the predicate itself


def test_the_predicate_separates_absent_from_refused():
    """Unit-level, because the whole gate turns on this one distinction."""
    found = [
        found_record("oak", OAK_R_ID, OAK_R, state="unavailable", error="in use"),
        found_record("oak", OAK_L_ID, OAK_L),
    ]

    assert devices_core.unavailable_reason(found, OAK_R) == "in use"
    # By raw id as well as by registry name: an unregistered device is stored
    # by id, and a name-only lookup silently refuses nothing.
    assert devices_core.unavailable_reason(found, OAK_R_ID) == "in use"
    assert devices_core.unavailable_reason(found, OAK_L) is None
    assert devices_core.unavailable_reason(found, NEVER_SEEN) is None
    assert devices_core.unavailable_reason(None, OAK_R) is None
    assert devices_core.unavailable_reason(found, "") is None


def test_an_unavailable_device_with_no_error_still_reports_a_reason():
    found = [found_record("oak", OAK_R_ID, OAK_R, state="unavailable")]
    reason = devices_core.unavailable_reason(found, OAK_R)
    assert reason and reason.strip()


def test_a_scanned_imx378_head_camera_is_written_at_640x480(client, repo, monkeypatch):
    """The head camera's resolution comes from its CAM_A sensor module -- a 4:3
    IMX378 is 640x480, a 16:10 AR0234 is 640x400 -- and `oak_camera` streams
    CAM_A, so an IMX378 head OAK must not be written as [640, 400]. Driven
    through the real scan + apply, so a break in the wiring between the scan
    record and the assembler is caught, not just the pure function."""
    head = {
        "kind": "oak",
        "device_id": "oak-head-mxid",
        "registry_name": "oak-head",
        "state": "available",
        "metadata": {"sockets": {"CAM_A": "IMX378"}},
    }
    run_scan(
        client,
        monkeypatch,
        [
            oak_found(OAK_R_ID, OAK_R),
            oak_found(OAK_L_ID, OAK_L),
            head,
        ],
    )
    response = client.post("/config/apply", data=handcrafted())
    assert response.status_code == 200, response.text[:500]

    config = yaml.safe_load(generated(repo).read_text())
    head_src = next(
        s for s in config["video_sources"] if s.get("name") == "head_oak_camera"
    )
    assert head_src["params"]["camera_resolution"] == [640, 480], head_src["params"]

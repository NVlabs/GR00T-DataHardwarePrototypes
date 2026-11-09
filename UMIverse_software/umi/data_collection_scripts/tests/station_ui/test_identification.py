# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hardware-isolated identification: real HTTP/draft flow, synthetic observations."""

import json
import time
import uuid
from html.parser import HTMLParser
from types import SimpleNamespace

import pytest
import test_unavailable_device_gate as gate
from fastapi.testclient import TestClient
from test_gripper_diagram import bench

from station_ui import identification, jobs, readiness
from station_ui.app import create_app
from station_ui.identify_worker import PREFIX
from station_ui.rig import take_control
from station_ui.settings import Settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    app = create_app(Settings())
    client = TestClient(app)
    client.get("/")
    client.post("/setup/preset", data={"preset_id": "bimanual_v4"})
    gate.run_scan(client, monkeypatch, bench())
    originals = {"start": jobs.start, "cancel": jobs.cancel}
    fake = {}

    def start(registry, argv, **kwargs):
        if kwargs.get("kind") != "identification":
            return originals["start"](registry, argv, **kwargs)
        device = next(
            value.split("=", 1)[1] for value in argv if value.startswith("--device=")
        )
        side = (
            "left"
            if device in ("1944301000000000A9", "00008130-00000000000000A1", "0")
            else "right"
        )
        job = SimpleNamespace(
            id=uuid.uuid4().hex,
            finished=False,
            error="",
            state="running",
            kind="identification",
        )
        job.lines = lambda: (
            [PREFIX + json.dumps({"ready": True})]
            + [
                PREFIX
                + json.dumps({"seq": seq, "at": time.monotonic(), "sides": [side]})
                for seq in range(1, 4)
            ]
        )
        fake[job.id] = job
        return job

    def cancel(registry, job_id, **kwargs):
        if job_id in fake:
            fake[job_id].finished = True
        else:
            originals["cancel"](registry, job_id, **kwargs)

    monkeypatch.setattr(jobs, "start", start)
    monkeypatch.setattr(jobs, "cancel", cancel)
    monkeypatch.setattr(identification, "SCAN_SECONDS", 0.01)
    yield client
    state = identification.current(app)
    if state:
        state.stop(app)


def begin(client):
    response = client.post("/config/identify/start")
    assert response.status_code == 200, response.text
    return response.json()["id"]


def test_guided_scans_fill_phone_oak_and_trigger_without_writing_a_config(client):
    session = begin(client)
    assert client.get(f"/config/identify/{session}/status").json()["completed"] == []
    for side in ("left", "right"):
        response = client.post(f"/config/identify/{session}/scan", data={"side": side})
        assert response.status_code == 200, response.text
        assert response.json()["completed"] == (
            ["left"] if side == "left" else ["left", "right"]
        )
        assert len(response.json()["assignments"]) == 4
        assert response.json()["message"] == ""
        draft = client.app.state.rig.draft
        assert response.json()["assignments"][f"gripper.{side}"] == draft.grippers[side]
        assert draft.grippers[side] == (
            "UMI-V4-0001" if side == "left" else "UMI-V4-0002"
        )
        assert draft.required_cameras[f"{side}_umi_camera"]
        assert draft.required_cameras[f"{side}_trigger_camera"] == (
            "0" if side == "left" else "2"
        )
        assert draft.required_cameras[f"{side}_tracking_camera"]
    assert not readiness.generated_config(client.app).exists()
    assert (
        len(client.get(f"/config/identify/{session}/status").json()["completed"]) == 2
    )
    response = client.post(f"/config/identify/{session}/stop")
    assert response.json()["released"]
    assert not identification.running(client.app)


def test_right_identification_is_locked_until_left_succeeds(client):
    session = begin(client)
    draft = client.app.state.rig.draft
    before = dict(draft.required_cameras)
    response = client.post(f"/config/identify/{session}/scan", data={"side": "right"})
    assert response.status_code == 409
    assert "Identify the left cameras" in response.text
    assert draft.required_cameras == before
    assert identification.current(client.app).completed == set()


def test_identified_cameras_correct_swapped_gripper_selections(client):
    draft = client.app.state.rig.draft
    draft.grippers = {"left": "UMI-V4-0002", "right": "UMI-V4-0001"}
    session = begin(client)
    response = client.post(f"/config/identify/{session}/scan", data={"side": "left"})
    assert response.status_code == 200, response.text
    assert response.json()["assignments"]["gripper.left"] == "UMI-V4-0001"
    assert response.json()["assignments"]["gripper.right"] == ""
    assert draft.grippers == {"left": "UMI-V4-0001"}
    response = client.post(f"/config/identify/{session}/scan", data={"side": "right"})
    assert response.status_code == 200, response.text
    assert draft.grippers == {"left": "UMI-V4-0001", "right": "UMI-V4-0002"}


def test_unregistered_gripper_auto_fills_by_serial_id(client, monkeypatch):
    found = bench()
    for record in found:
        if record["kind"] == "gripper":
            record["registry_name"] = None
    gate.run_scan(client, monkeypatch, found)
    session = begin(client)
    response = client.post(f"/config/identify/{session}/scan", data={"side": "left"})
    assert response.status_code == 200, response.text
    assert response.json()["assignments"]["gripper.left"] == "534A3035000000B8"
    assert client.app.state.rig.draft.grippers["left"] == "534A3035000000B8"


def test_missing_gripper_evidence_keeps_manual_selection_and_camera_success(
    client, monkeypatch
):
    found = bench()
    for record in found:
        record["metadata"].pop("usb_location", None)
    gate.run_scan(client, monkeypatch, found)
    client.app.state.rig.draft.grippers["left"] = "UMI-V4-0002"
    session = begin(client)
    response = client.post(f"/config/identify/{session}/scan", data={"side": "left"})
    assert response.status_code == 200, response.text
    assert response.json()["completed"] == ["left"]
    assert "gripper.left" not in response.json()["assignments"]
    assert client.app.state.rig.draft.grippers["left"] == "UMI-V4-0002"
    assert "Select the gripper manually" in response.json()["message"]


def test_a_completed_side_can_be_scanned_again_without_losing_the_other_side(client):
    session = begin(client)
    for side in ("left", "right", "left", "right"):
        response = client.post(f"/config/identify/{session}/scan", data={"side": side})
        assert response.status_code == 200, response.text
        assert side in response.json()["completed"]
    assert response.json()["completed"] == ["left", "right"]
    assert client.get(f"/config/identify/{session}/status").json()["completed"] == [
        "left",
        "right",
    ]


def test_failed_rescan_preserves_previously_identified_sides(client, monkeypatch):
    session = begin(client)
    for side in ("left", "right"):
        assert (
            client.post(
                f"/config/identify/{session}/scan", data={"side": side}
            ).status_code
            == 200
        )
    state = identification.current(client.app)
    before = dict(client.app.state.rig.draft.required_cameras)
    monkeypatch.setattr(
        state, "scan", lambda *args: ({}, ["No camera saw the marker."])
    )
    response = client.post(f"/config/identify/{session}/scan", data={"side": "left"})
    assert response.status_code == 422
    assert client.app.state.rig.draft.required_cameras == before
    assert client.get(f"/config/identify/{session}/status").json()["completed"] == [
        "left",
        "right",
    ]


def test_ambiguous_observations_leave_all_camera_selections_unchanged(client):
    session = begin(client)
    state = identification.current(client.app)
    oak = next(camera for camera in state.cameras if camera.kind == "oak")
    for camera in state.cameras:
        if camera.kind == "oak":
            camera.job.lines = oak.job.lines
    before = dict(client.app.state.rig.draft.required_cameras)
    response = client.post(f"/config/identify/{session}/scan", data={"side": "left"})
    assert response.status_code == 422
    assert "Multiple oak" in response.text
    assert client.app.state.rig.draft.required_cameras == before


def test_stale_frames_cannot_assign_a_camera(client):
    session = begin(client)
    for camera in identification.current(client.app).cameras:
        rows = camera.job.lines()
        camera.job.lines = lambda rows=rows: rows
    response = client.post(f"/config/identify/{session}/scan", data={"side": "left"})
    assert response.status_code == 422
    assert not response.json()["assignments"]


def test_camera_operations_refuse_while_identification_owns_the_cameras(client):
    begin(client)
    for path, data in [
        ("/config/devices/scan", {}),
        ("/config/preview/start", {"kind": "oak", "device": "oak-FFC-54"}),
        ("/calibration/check/start", {}),
    ]:
        response = client.post(path, data=data)
        assert response.status_code == 409, (path, response.text)
        assert "identification" in response.text.lower()
    response = client.post("/record/start")
    assert response.status_code == 400
    assert "identification" in response.text.lower()


def test_owner_change_during_scan_cannot_commit_results(client, monkeypatch):
    session = begin(client)
    state = identification.current(client.app)
    original_scan = state.scan

    def displaced(*args):
        result = original_scan(*args)
        take_control(client.app.state.rig, "another-browser")
        return result

    monkeypatch.setattr(state, "scan", displaced)
    before = dict(client.app.state.rig.draft.required_cameras)
    response = client.post(f"/config/identify/{session}/scan", data={"side": "left"})
    assert response.status_code == 409
    assert client.app.state.rig.draft.required_cameras == before
    # A displaced owner can still release its own cameras.
    assert client.post(f"/config/identify/{session}/stop").status_code == 200


def test_changed_setup_cannot_receive_old_identification_results(client):
    session = begin(client)
    client.post("/setup/preset", data={"preset_id": "bimanual_v4_vive"})
    response = client.post(f"/config/identify/{session}/scan", data={"side": "left"})
    assert response.status_code == 409
    assert not client.app.state.rig.draft.required_cameras


def test_vive_setup_identifies_oak_and_trigger_without_a_phone_slot(client):
    client.post("/setup/preset", data={"preset_id": "bimanual_v4_vive"})
    session = begin(client)
    response = client.post(f"/config/identify/{session}/scan", data={"side": "left"})
    assert response.status_code == 200, response.text
    assert len(response.json()["assignments"]) == 3
    assert response.json()["assignments"]["gripper.left"] == "UMI-V4-0001"
    assert "left_tracking_camera" not in client.app.state.rig.draft.required_cameras


class Structure(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.duplicates, self.sections = set(), [], []
        self.h1 = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "h1":
            self.h1 += 1
        if "id" in attrs:
            if attrs["id"] in self.ids:
                self.duplicates.append(attrs["id"])
            self.ids.add(attrs["id"])
        if tag == "section" and "workflow-section" in attrs.get("class", "").split():
            self.sections.append(attrs["id"])


@pytest.mark.parametrize("path", ["/", "/config", "/calibration", "/record", "/audit"])
def test_legacy_destinations_render_all_five_sections_in_one_document(client, path):
    response = client.get(path)
    assert response.status_code == 200
    page = Structure()
    page.feed(response.text)
    assert page.sections == ["setup", "configuration", "calibration", "record", "audit"]
    assert page.h1 == 1
    assert not page.duplicates
    assert "gripper-diagram__drawing" in response.text
    assert 'viewBox="90 0 260 280"' in response.text
    assert "camera-identify-dialog" in response.text
    assert "GEAR UMI / Data collection /" not in response.text


def test_stop_cancels_an_inflight_scan_without_assigning(client, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    session = begin(client)
    state = identification.current(client.app)
    monkeypatch.setattr(identification, "SCAN_SECONDS", 0.5)
    before = dict(client.app.state.rig.draft.required_cameras)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(
            client.post, f"/config/identify/{session}/scan", data={"side": "left"}
        )
        deadline = time.monotonic() + 2
        while not state.scanning and time.monotonic() < deadline:
            time.sleep(0.01)
        assert state.scanning
        assert client.post(f"/config/identify/{session}/stop").status_code == 200
        assert pending.result(timeout=2).status_code == 409
    assert client.app.state.rig.draft.required_cameras == before


def test_one_camera_cannot_fill_both_grippers(client):
    session = begin(client)
    assert (
        client.post(
            f"/config/identify/{session}/scan", data={"side": "left"}
        ).status_code
        == 200
    )
    state = identification.current(client.app)
    for camera in state.cameras:
        side = (
            "right"
            if camera.device_id
            in ("1944301000000000A9", "00008130-00000000000000A1", "0")
            else "left"
        )
        camera.job.lines = lambda side=side: (
            [PREFIX + json.dumps({"ready": True})]
            + [
                PREFIX
                + json.dumps({"seq": seq, "at": time.monotonic(), "sides": [side]})
                for seq in range(1, 4)
            ]
        )
    response = client.post(f"/config/identify/{session}/scan", data={"side": "right"})
    assert response.status_code == 200
    assert response.json()["completed"] == ["right"]
    assert "Rescan left" in response.json()["message"]
    assert response.json()["assignments"]["camera.left_umi_camera"] == ""
    assert not client.app.state.rig.draft.required_cameras.get("left_umi_camera")


def test_a_nonresponding_candidate_cannot_conceal_another_match(client):
    session = begin(client)
    state = identification.current(client.app)
    camera = next(
        camera
        for camera in state.cameras
        if camera.kind == "oak" and camera.device_id == "1944301000000000A8"
    )
    camera.job.lines = lambda: [PREFIX + json.dumps({"ready": True})]
    before = dict(client.app.state.rig.draft.required_cameras)
    response = client.post(f"/config/identify/{session}/scan", data={"side": "left"})
    assert response.status_code == 422
    assert "fresh frames" in response.text
    assert client.app.state.rig.draft.required_cameras == before


def test_nonowners_and_cross_origin_pages_cannot_start_identification(client):
    intruder = TestClient(client.app)
    assert intruder.post("/config/identify/start").status_code == 409
    assert (
        client.post(
            "/config/identify/start", headers={"Origin": "http://untrusted.invalid"}
        ).status_code
        == 403
    )
    assert not identification.running(client.app)


def test_marker_worker_counts_new_frames_and_preserves_its_source_cleanup(
    tmp_path, monkeypatch, capsys
):
    import signal

    import cv2
    import numpy as np

    from station_ui import identify_worker, preview_worker

    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, 180
    )
    frame = cv2.cvtColor(np.pad(marker, 30, constant_values=255), cv2.COLOR_GRAY2BGR)
    handlers, closed = {}, []

    class Source:
        count = 0

        def tiles(self):
            self.count += 1
            if self.count == 5:
                handlers[signal.SIGTERM](None, None)
            # Four repeated reads of the SAME cached frame count only once.
            return [("camera", frame)]

        def close(self):
            closed.append(True)

    monkeypatch.setattr(preview_worker, "_open", lambda *_: Source())
    monkeypatch.setattr(
        identify_worker,
        "signal",
        SimpleNamespace(
            SIGTERM=signal.SIGTERM,
            SIGINT=signal.SIGINT,
            signal=lambda sig, handler: handlers.update({sig: handler}),
        ),
    )
    monkeypatch.setattr(
        identify_worker,
        "time",
        SimpleNamespace(monotonic=time.monotonic, sleep=lambda _: None),
    )
    assert (
        identify_worker.main(
            ["--kind=uvc", "--device=0", "--out=" + str(tmp_path / "frame.jpg")]
        )
        == 0
    )
    rows = [
        json.loads(line[len(PREFIX) :])
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(PREFIX)
    ]
    assert [row["sides"] for row in rows if "sides" in row] == [["left"]]
    assert closed == [True]
    assert (tmp_path / "frame.jpg").is_file()


def test_read_only_combined_page_has_one_takeover_and_unique_ids(client):
    page = Structure()
    response = TestClient(client.app).get("/")
    page.feed(response.text)
    assert not page.duplicates
    assert response.text.count('action="/take-control"') == 1


def test_camera_frames_are_no_store_snapshots_and_stop_after_release(client):
    session = begin(client)
    camera = identification.current(client.app).cameras[0]
    url = f"/config/identify/{session}/frame/{camera.job.id}"
    assert client.get(url).status_code == 204
    camera.frame.write_bytes(b"test jpeg snapshot")
    response = client.get(url)
    assert response.content == b"test jpeg snapshot"
    assert int(response.headers["content-length"]) == len(response.content)
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["content-type"] == "image/jpeg"
    assert (
        client.get(f"/config/identify/{session}/frame/another-job").status_code == 404
    )
    client.post(f"/config/identify/{session}/stop")
    assert client.get(url).status_code == 404


def identify_both(client):
    session = begin(client)
    for side in ("left", "right"):
        response = client.post(f"/config/identify/{session}/scan", data={"side": side})
        assert response.status_code == 200, response.text
    assert client.post(f"/config/identify/{session}/stop").json()["released"]


def action_payload(client, monkeypatch):
    import test_config_page as forms

    from umi.common.tasks import task_registry_store

    monkeypatch.setattr(task_registry_store, "get_all_tasks", lambda: dict(gate.TASKS))
    client.app.state.rig.draft.optional.clear()
    form = next(f for f in forms._page(client).forms if f["action"] == "/config/apply")
    return {
        **forms._submission(form),
        "task_name": "plane_assembly_v9",
        "operator_name": "Test Operator",
        "gripper.left": "UMI-V4-0001",
        "gripper.right": "UMI-V4-0002",
    }


def test_refresh_releases_identification_before_rescan_and_reopens_new_inventory(
    client, monkeypatch
):
    from station_ui import devices

    session = begin(client)
    state = identification.current(client.app)
    assert (
        client.post(
            f"/config/identify/{session}/scan", data={"side": "left"}
        ).status_code
        == 200
    )
    assert identification.verified_roles(client.app)
    extra = gate.found_record("uvc", "8")
    monkeypatch.setattr(
        devices, "scan_argv", lambda: gate.fake_scan_argv([*bench(), extra])
    )
    old_job = devices.current_job(client.app)
    response = client.post("/config/identify/refresh")
    assert response.status_code == 200, response.text
    assert state.released() and all(camera.job.finished for camera in state.cameras)
    assert not identification.verified_roles(client.app), (
        "a rescan must expire old marker evidence"
    )
    assert response.json()["scan_id"] != old_job.id
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        inventory = client.get("/config/identify/inventory").json()
        if not inventory["running"]:
            break
        time.sleep(0.02)
    assert inventory["count"] == len(bench()) + 1 and not inventory["error"]
    replacement = begin(client)
    assert replacement != session
    cameras = client.get(f"/config/identify/{replacement}/status").json()["cameras"]
    assert any(camera["device_id"] == "8" for camera in cameras)
    assert 'id="identify-refresh"' in client.get("/config").text


def test_refresh_cannot_probe_until_workers_have_released(client, monkeypatch):
    from station_ui import devices

    begin(client)
    previous = devices.current_job(client.app)
    monkeypatch.setattr(jobs, "cancel", lambda *a, **k: None)
    monkeypatch.setattr(identification, "RELEASE_SECONDS", -3)
    response = client.post("/config/identify/refresh")
    assert response.status_code == 409
    assert devices.current_job(client.app) is previous
    assert "still releasing" in response.text
    for camera in identification.current(client.app).cameras:
        camera.job.finished = True


def test_identification_satisfies_camera_check_and_assignment_edits_expire_only_their_evidence(
    client, monkeypatch
):
    import yaml

    from station_ui import calibration as cal

    identify_both(client)
    data = action_payload(client, monkeypatch)
    response = client.post("/config/apply", data=data)
    assert response.status_code == 200, response.text
    config = cal.recording_config(client.app)
    assert identification.verified_roles(client.app, config) == set(
        cal.verification_roles(config)
    )
    assert cal.step_state(client.app, cal.STEP_VERIFY) == cal.DONE
    from html_sections import section_html

    body = section_html(client.get("/calibration").text, "calibration")
    assert 'id="step-device_verify"' not in body
    assert 'form="configuration-form"' in body
    # A manual change cannot borrow a proof for the camera it replaced, even if
    # the draft is subsequently made to agree with the edited saved config.
    source = next(
        s for s in config["video_sources"] if s.get("name") == "left_trigger_camera"
    )
    source["params"]["camera_index"] = 8
    client.app.state.rig.draft.required_cameras["left_trigger_camera"] = "8"
    cal.recording_config_path(client.app).write_text(yaml.safe_dump(config))
    assert "left_trigger" not in identification.verified_roles(client.app, config)
    assert cal.step_state(client.app, cal.STEP_VERIFY) == cal.PENDING
    assert 'id="step-device_verify"' in section_html(
        client.get("/calibration").text, "calibration"
    )


def test_optional_head_check_opens_only_roles_not_already_identified(
    client, monkeypatch
):
    import sys

    from station_ui import calibration as cal

    identify_both(client)
    data = action_payload(client, monkeypatch)
    client.post("/config/optional/add", data={"type_id": "head_oak"})
    data["optional.head_oak_camera.device"] = "oak-HEAD"
    data["optional.head_oak_camera.placement"] = "head"
    monkeypatch.setattr(
        cal, "LAUNCHER", (sys.executable, "-c", "import sys; print(sys.argv)")
    )
    response = client.post(
        "/calibration/check/start", data=data, follow_redirects=False
    )
    assert response.status_code == 303, response.text
    config = cal.recording_config(client.app)
    pending = set(cal.verification_roles(config)) - identification.verified_roles(
        client.app, config
    )
    assert pending == {"head_umi"}
    job = cal.job_registry(client.app).get(cal.check_state(client.app).job_id)
    index = job.argv.index("--roles")
    assert job.argv[index + 1 : index + 2] == ["head_umi"]
    assert cal.step_state(client.app, cal.STEP_VERIFY) != cal.DONE
    cal.stop_check(client.app, wait=2)
    client.post("/config/session/uid", data=data, follow_redirects=False)
    assert not readiness.generated_config(client.app).exists()
    assert "head_umi" in cal.expected_keys(client.app, cal.STEP_VERIFY)
    assert cal.step_state(client.app, cal.STEP_VERIFY) != cal.DONE


@pytest.mark.parametrize("stale_previous_rig", [False, True])
@pytest.mark.parametrize("preview_task_edit", [False, True])
@pytest.mark.parametrize("new_session", [False, True])
def test_calibration_and_recording_save_current_fields_without_apply(
    client, monkeypatch, stale_previous_rig, preview_task_edit, new_session
):
    import sys

    from station_ui import calibration as cal
    from station_ui.routes import record

    identify_both(client)
    data = action_payload(client, monkeypatch)
    assert not readiness.generated_config(client.app).exists()
    if stale_previous_rig:
        # Hardware resets preserve stale stamps for display but must not repeatedly
        # clear new verification runs during automatic saves.
        state = cal.calibration_state(client.app)
        for step in cal.STEPS:
            state.stamp(step.id, "previous-rig")
        client.app.state.applied_fingerprint = "previous-rig"

    # Child jobs are real, but these commands cannot open hardware.
    def simulated_gripper(*args, status_path, **kwargs):
        return [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; p=Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True); p.write_text('0')",
            status_path,
        ]

    monkeypatch.setattr(cal, "gripper_argv", simulated_gripper)
    for side in ("left", "right"):
        response = client.post(
            "/calibration/calibrate-gripper",
            data={**data, "side": side},
            follow_redirects=False,
        )
        assert response.status_code == 303, response.text
        assert "Configuration saved." in client.get("/calibration").text
        assert not client.app.state.config_notice
    deadline = time.monotonic() + 5
    while (
        cal.step_state(client.app, cal.STEP_GRIPPER) != cal.DONE
        and time.monotonic() < deadline
    ):
        time.sleep(0.02)
    assert readiness.calibration_done(client.app)
    assert cal.recording_config(client.app)["meta"]["operator_name"] == "Test Operator"
    if new_session:
        previous_uid = client.app.state.rig.draft.session_uid
        response = client.post("/config/session/uid", data=data, follow_redirects=False)
        assert response.status_code == 303
        assert client.app.state.rig.draft.session_uid != previous_uid
        assert not readiness.generated_config(client.app).exists()
        # Starting a new session must preserve successful hardware verification
        # and offer saving the new config.
        assert cal.step_state(client.app, cal.STEP_GRIPPER) == cal.DONE
        assert cal.step_state(client.app, cal.STEP_VERIFY) == cal.DONE
        assert readiness.calibration_done(client.app)
        import test_config_page as forms

        body = client.get("/record").text
        page = forms._Page()
        page.feed(body)
        _, start = forms._find_button(page, "/record/start")
        assert start is not None and "disabled" not in start
        assert "start calibration to save it automatically" not in body
        assert "Enter Task Name Here" not in body
        # No completed calibration may be accepted for changed hardware, even
        # while the next session has no generated config yet.
        draft = client.app.state.rig.draft
        left = draft.grippers["left"]
        draft.grippers["left"] = "UMI-V4-OTHER"
        assert not readiness.calibration_done(client.app)
        page = forms._Page()
        page.feed(client.get("/record").text)
        _, start = forms._find_button(page, "/record/start")
        assert start is not None and "disabled" in start
        draft.grippers["left"] = left
    # Task and operator edits after calibration leave the hardware and every
    # completed run intact, whether the task preview posts first or Start is
    # the first action to submit the edit.
    import test_config_page as forms

    from umi.common.tasks import task_registry_store

    monkeypatch.setattr(task_registry_store, "get_all_tasks", lambda: dict(forms.TASKS))
    previous_runs = cal.calibration_state(client.app).runs(cal.STEP_GRIPPER)
    updated = {
        **data,
        "task_name": "a_30_second_task",
        "operator_name": "Updated Operator",
    }
    if preview_task_edit:
        response = client.post(
            "/config/task", data=updated, headers={"HX-Request": "true"}
        )
        assert response.status_code == 200
    assert readiness.calibration_done(client.app)
    assert cal.calibration_state(client.app).runs(cal.STEP_GRIPPER) == previous_runs
    # Preserve real preflight, including assignment and readiness validation;
    # recorder hardware is a simulated boundary.
    monkeypatch.setattr(
        record,
        "build_argv",
        lambda *a, **k: [sys.executable, "-c", "print('recorded')"],
    )
    response = client.post(
        "/record/start",
        data=updated,
        headers={"HX-Request": "true"},
        follow_redirects=False,
    )
    assert response.status_code == 200, response.text
    assert "Configuration saved." in response.text
    # The live response includes the refreshed summary, not just the buttons.
    assert "This session" in response.text
    assert "a_30_second_task" in response.text
    assert (
        'id="session-folder"' in response.text
        and 'hx-swap-oob="outerHTML"' in response.text
    )
    assert not client.app.state.config_notice
    assert (
        cal.recording_config(client.app)["meta"]["operator_name"] == "Updated Operator"
    )
    assert cal.recording_config(client.app)["meta"]["task_name"] == "a_30_second_task"
    assert readiness.calibration_done(client.app)
    assert cal.calibration_state(client.app).runs(cal.STEP_GRIPPER) == previous_runs
    assert record.workflow_state(client.app).record_job_id


def test_auto_apply_validation_errors_return_to_configuration_without_launching(
    client, monkeypatch
):
    identify_both(client)
    data = action_payload(client, monkeypatch)
    from station_ui import calibration as cal

    before = len(cal.job_registry(client.app).all())
    response = client.post(
        "/record/start",
        data={**data, "operator_name": "invalid123"},
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 400
    assert response.headers["HX-Retarget"] == "#configuration"
    assert response.headers["HX-Reselect"] == "#configuration"
    assert 'aria-invalid="true"' in response.text
    assert not readiness.generated_config(client.app).exists()
    from station_ui.routes import record

    assert record.workflow_state(client.app).record_job_id is None
    assert len(cal.job_registry(client.app).all()) == before


@pytest.mark.parametrize("kind", ["camera", "gripper"])
def test_auto_save_does_not_forget_a_running_calibration_when_devices_change(
    client, monkeypatch, kind
):
    import sys

    from station_ui import calibration as cal
    from station_ui import session as session_state

    identify_both(client)
    data = action_payload(client, monkeypatch)
    assert client.post("/config/apply", data=data).status_code == 200
    path = cal.recording_config_path(client.app)
    before = path.read_bytes()
    registry = cal.job_registry(client.app)
    job = jobs.start(
        registry,
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=client.app.state.settings.repo_root,
        timeout=35,
    )
    state = cal.calibration_state(client.app)
    fingerprint = session_state.device_fingerprint(client.app.state.rig.draft)
    if kind == "camera":
        cal.check_state(client.app).job_id = job.id
        state.stamp(cal.STEP_VERIFY, fingerprint)
    else:
        state.record(cal.STEP_GRIPPER, cal.Run(key="left", job_id=job.id), fingerprint)
    try:
        response = client.post(
            "/calibration/calibrate-gripper",
            data={**data, "side": "right", "gripper.left": "UMI-V4-9999"},
        )
        assert response.status_code == 409
        assert "stop the running check" in response.text
        assert path.read_bytes() == before
        assert not job.finished
        if kind == "camera":
            assert cal.check_state(client.app).job_id == job.id
        else:
            assert state.run(cal.STEP_GRIPPER, "left").job_id == job.id
    finally:
        jobs.cancel(registry, job.id)

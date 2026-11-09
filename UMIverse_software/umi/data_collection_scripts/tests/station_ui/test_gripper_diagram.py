# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""V4 diagrams, USB pairing evidence, and refusing a wrong pairing before capture.

All inventory here is synthetic; no test enumerates or opens real hardware.
"""

import json
import pathlib
import shutil
import subprocess
from html.parser import HTMLParser

import pytest
import yaml
from fastapi.testclient import TestClient

from station_ui import devices, gripper_pairing, readiness
from station_ui.app import create_app
from station_ui.core import device_detector, umi_gripper_devices
from station_ui.routes import config as config_routes
from station_ui.routes import record as record_routes
from station_ui.settings import Settings
from umi.common.tasks import task_registry_store
from umi.utils import device_registry

REPO = pathlib.Path(__file__).resolve().parents[2]
GRIPPERS = {"UMI-V4-0001": "534A3035000000B8", "UMI-V4-0002": "534A3035000000B7"}
OAKS = {"oak-FFC-54": "1944301000000000A9", "oak-FFC-57": "1944301000000000A8"}
PHONES = {
    "iphone15p-21": "00008130-00000000000000A1",
    "iphone15p-53": "00008130-00000000000000A2",
}


def bench():
    found = []
    for kind, entries, port in (
        ("gripper", GRIPPERS, 4),
        ("oak", OAKS, 3),
        ("iphone", PHONES, 1),
    ):
        for index, (name, serial) in enumerate(entries.items(), 3):
            found.append(
                {
                    "kind": kind,
                    "registry_name": name,
                    "device_id": serial,
                    "state": "available",
                    "error": None,
                    "metadata": {
                        "usb_location": f"0-1.{index}.{port}",
                        "sockets": {"CAM_B": "AR0234"},
                    },
                }
            )
    for index in (0, 2):
        found.append(
            {
                "kind": "uvc",
                "device_id": str(index),
                "registry_name": None,
                "state": "available",
                "error": None,
                "metadata": {},
            }
        )
    return found


def selected(**overrides):
    return {
        "gripper": "UMI-V4-0001",
        "umi": "oak-FFC-54",
        "tracking": "iphone15p-21",
        **overrides,
    }


@pytest.mark.parametrize(
    "values,expected",
    [
        (selected(), "matched"),
        (selected(tracking="iphone15p-53"), "mismatch"),
        (selected(umi="oak-FFC-57"), "mismatch"),
        (selected(gripper="UMI-V4-0002"), "mismatch"),
        (selected(tracking=""), "incomplete"),
        (selected(tracking="unknown-phone"), "unverified"),
    ],
)
def test_pairing_verdicts(values, expected):
    result = gripper_pairing.check(values, gripper_pairing.evidence(bench()))
    assert result["state"] == expected
    if expected == "mismatch":
        assert "Select the phone and top camera" in result["message"]


def test_phone_and_top_camera_are_both_highlighted():
    result = gripper_pairing.check(
        selected(tracking="iphone15p-53"), gripper_pairing.evidence(bench())
    )
    assert {"umi", "tracking"} <= set(result["affected"])
    assert "UMI-V4-0001" in result["message"] and "UMI-V4-0002" in result["message"]


def test_raw_serials_and_case_resolve_to_the_same_pairing():
    values = dict(
        zip(
            ("gripper", "umi", "tracking"),
            (
                next(iter(entries.values())).lower()
                for entries in (GRIPPERS, OAKS, PHONES)
            ),
        )
    )
    assert (
        gripper_pairing.check(values, gripper_pairing.evidence(bench()))["state"]
        == "matched"
    )


@pytest.mark.parametrize(
    "cameras",
    [
        {"oak": "oak-FFC-54", "iphone": "iphone15p-21", "uvc": "0"},
        {"oak": "1944301000000000a9", "iphone": "00008130-00000000000000a1"},
        {"oak": "oak-FFC-54", "uvc": "2"},
    ],
)
def test_identified_hub_cameras_determine_the_gripper(cameras):
    assert gripper_pairing.gripper_for_cameras(cameras, bench()) == ("UMI-V4-0001", "")


@pytest.mark.parametrize(
    "condition",
    [
        "no_locations",
        "shared_hub",
        "extra_oak",
        "missing_phone_location",
        "missing_gripper",
        "unavailable_gripper",
        "different_grippers",
        "trigger_only",
    ],
)
def test_uncertain_camera_ownership_never_guesses_a_gripper(condition):
    found = bench()
    cameras = {"oak": "oak-FFC-54", "iphone": "iphone15p-21"}
    if condition == "no_locations":
        for record in found:
            record["metadata"].pop("usb_location", None)
    elif condition == "shared_hub":
        for record in found:
            if record["kind"] in ("gripper", "oak", "iphone"):
                record["metadata"]["usb_location"] = "0-1.3.4"
    elif condition == "extra_oak":
        oak = next(
            record for record in found if record["registry_name"] == "oak-FFC-54"
        )
        found.append({**oak, "device_id": "other-oak", "registry_name": "other-oak"})
    elif condition == "missing_phone_location":
        next(record for record in found if record["registry_name"] == "iphone15p-21")[
            "metadata"
        ].pop("usb_location")
    elif condition == "missing_gripper":
        found = [record for record in found if record["kind"] != "gripper"]
    elif condition == "unavailable_gripper":
        next(record for record in found if record["registry_name"] == "UMI-V4-0001")[
            "state"
        ] = "unavailable"
    elif condition == "different_grippers":
        cameras["iphone"] = "iphone15p-53"
    elif condition == "trigger_only":
        cameras = {"uvc": "0"}
    gripper, warning = gripper_pairing.gripper_for_cameras(cameras, found)
    assert gripper is None
    assert "manually" in warning


@pytest.mark.parametrize(
    "condition",
    ["no_scan", "no_location", "shared_hub", "multiple_cameras", "malformed_location"],
)
def test_missing_or_ambiguous_usb_evidence_never_claims_a_match(condition):
    found = bench()
    if condition == "no_scan":
        found = None
    elif condition == "no_location":
        for record in found:
            record["metadata"].pop("usb_location", None)
    elif condition == "shared_hub":
        for index, record in enumerate(found):
            record["metadata"]["usb_location"] = f"0-1.{index + 1}"
    elif condition == "multiple_cameras":
        found.append(
            {**found[2], "device_id": "other-oak", "registry_name": "other-oak"}
        )
    else:
        found[2]["metadata"]["usb_location"] = "not-a-port"
    assert (
        gripper_pairing.check(selected(), gripper_pairing.evidence(found))["state"]
        == "unverified"
    )


@pytest.fixture
def console(tmp_path, monkeypatch):
    monkeypatch.setattr(
        device_registry,
        "_REGISTRY",
        {
            "version": 1,
            "umi_v4_grippers": GRIPPERS,
            "oak_devices": OAKS,
            "iphone_devices": PHONES,
        },
    )
    monkeypatch.setattr(umi_gripper_devices, "UMI_V4_GRIPPERS", GRIPPERS)
    monkeypatch.setattr(
        config_routes,
        "list_umi_gripper_device_names_for_pipeline",
        lambda **kw: list(GRIPPERS),
    )
    monkeypatch.setattr(
        device_detector, "get_available_oak_cameras", lambda: list(OAKS)
    )
    monkeypatch.setattr(device_detector, "get_available_iphones", lambda: list(PHONES))
    monkeypatch.setattr(
        task_registry_store,
        "get_all_tasks",
        lambda: {
            "pairing_test": {
                "name": "pairing_test",
                "short_description": "Check gripper assignments",
                "mode": "bimanual",
                "HW_version": "v4",
            },
        },
    )
    repo = tmp_path / "repo"
    shutil.copytree(
        REPO / "config" / "data_recording",
        repo / "config" / "data_recording",
        ignore=shutil.ignore_patterns("generated"),
    )
    shutil.copytree(REPO / "config" / "data_check", repo / "config" / "data_check")
    inventory = {"found": bench()}
    monkeypatch.setattr(devices, "outcome_for", lambda job: (inventory["found"], None))
    client = TestClient(create_app(Settings(repo_root=repo)))
    client.get("/")
    client.post("/setup/preset", data={"preset_id": "bimanual_v4"})
    for assignment in list(client.app.state.rig.draft.optional):
        client.post(
            "/config/optional/remove?confirm=yes",
            data={"config_form": "1", "logical_name": assignment.logical_name},
        )
    return client, inventory


def form(**overrides):
    return {
        "config_form": "1",
        "operator_name": "Jane Doe",
        "task_name": "pairing_test",
        "gripper.left": "UMI-V4-0001",
        "gripper.right": "UMI-V4-0002",
        "camera.left_umi_camera": "oak-FFC-54",
        "camera.right_umi_camera": "oak-FFC-57",
        "camera.left_tracking_camera": "iphone15p-21",
        "camera.right_tracking_camera": "iphone15p-53",
        "camera.left_trigger_camera": "0",
        "camera.right_trigger_camera": "2",
        **overrides,
    }


class Diagrams(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.figures = []
        self.parts = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "data-gripper-diagram" in attrs:
            self.figures.append(attrs)
        if "data-gripper-selection" in attrs:
            self.parts.append(attrs)


def test_diagrams_show_current_devices_and_link_to_existing_selectors(console):
    client, _ = console
    assert client.post("/config/apply", data=form()).status_code == 200
    html = client.get("/config").text
    diagrams = Diagrams(html)
    assert [figure["data-gripper-diagram"] for figure in diagrams.figures] == [
        "left",
        "right",
    ]
    assert all(figure["data-pairing-state"] == "matched" for figure in diagrams.figures)
    assert len(diagrams.parts) == 8
    for part in diagrams.parts:
        assert part["href"] == "#" + part["data-field"]
        assert f'id="{part["data-field"]}"' in html
    assert "Top camera · OAK" in html and "Trigger camera · USB" in html


def test_mismatched_phone_is_refused_without_writing_a_config(console):
    client, _ = console
    response = client.post(
        "/config/apply",
        data=form(
            **{
                "camera.left_tracking_camera": "iphone15p-53",
                "camera.right_tracking_camera": "iphone15p-21",
            }
        ),
    )
    assert response.status_code == 400
    assert "Pairing mismatch." in response.text
    assert not readiness.generated_config(client.app).exists()
    parts = Diagrams(response.text).parts
    for role in ("umi", "tracking"):
        assert all(
            "is-mismatch" in part["class"]
            for part in parts
            if part["data-gripper-selection"] == role
        )


def test_correcting_a_mismatch_allows_apply(console):
    client, _ = console
    assert (
        client.post(
            "/config/apply",
            data=form(**{"camera.left_tracking_camera": "iphone15p-53"}),
        ).status_code
        == 400
    )
    assert client.post("/config/apply", data=form()).status_code == 200
    assert readiness.generated_config(client.app).is_file()
    assert all(
        figure["data-pairing-state"] == "matched"
        for figure in Diagrams(client.get("/config").text).figures
    )


def test_scan_and_gripper_changes_refresh_diagrams_without_replacing_other_fields(
    console,
):
    client, _ = console
    fragment = client.get("/config/devices/status", headers={"HX-Request": "true"}).text
    assert 'id="gripper-diagrams" hx-swap-oob="true"' in fragment
    assert "data-gripper-evidence" in fragment
    # The figures sit beside the selectors. A scan must target each figure by
    # id without replacing an input that the operator is still editing.
    figures = Diagrams(fragment).figures
    assert {figure["id"] for figure in figures} == {
        "gripper-diagram-left",
        "gripper-diagram-right",
    }
    assert all(figure["hx-swap-oob"] == "true" for figure in figures)
    assert "<input" not in fragment
    fragment = client.post(
        "/config/devices/link/left", data=form(), headers={"HX-Request": "true"}
    ).text
    assert 'id="gripper-diagrams" hx-swap-oob="true"' in fragment
    assert all(
        figure["data-pairing-state"] == "matched"
        for figure in Diagrams(fragment).figures
    )


def test_unknown_pairing_is_shown_as_unverified(console):
    client, inventory = console
    inventory["found"] = None
    response = client.post("/config/apply", data=form())
    assert response.status_code == 200
    assert all(
        figure["data-pairing-state"] == "unverified"
        for figure in Diagrams(response.text).figures
    )
    assert "Pairing not verified" in response.text


def test_recording_rechecks_saved_devices_after_a_rescan(console, monkeypatch):
    client, inventory = console
    assert client.post("/config/apply", data=form()).status_code == 200
    # Cabling changes after Apply: the saved left phone now rides on the right.
    inventory["found"][4]["metadata"]["usb_location"] = "0-1.4.1"
    inventory["found"][5]["metadata"]["usb_location"] = "0-1.3.1"
    with pytest.raises(
        record_routes.PreflightError,
        match="Correct the device selections on Configuration",
    ):
        record_routes.preflight(
            client.app.state.rig.draft,
            client.app.state.settings.repo_root,
            app=client.app,
        )
    monkeypatch.setattr(
        record_routes.jobs, "start", lambda *a, **kw: pytest.fail("recorder launched")
    )
    response = client.post("/record/start")
    assert response.status_code == 400
    assert "Correct the device selections on Configuration" in response.text


def test_raw_identifiers_in_a_saved_config_are_checked(console):
    client, _ = console
    assert client.post("/config/apply", data=form()).status_code == 200
    config = yaml.safe_load(readiness.generated_config(client.app).read_text())
    for camera in config["video_sources"]:
        if camera.get("name") == "left_tracking_camera":
            camera["params"]["iphone_udid"] = PHONES["iphone15p-53"]
            camera["params"].pop("iphone_device_name", None)
    assert gripper_pairing.config_mismatches(config, bench())


def test_vive_rig_shows_cameras_without_requesting_a_phone(console):
    client, _ = console
    client.post("/setup/preset", data={"preset_id": "bimanual_v4_vive"})
    html = client.get("/config").text
    assert len(Diagrams(html).figures) == 2
    assert all(
        part["data-gripper-selection"] != "tracking" for part in Diagrams(html).parts
    )
    assert "Vive tracking · no phone" in html


def test_browser_and_server_pairing_verdicts_agree():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node unavailable")
    samples = [
        selected(),
        selected(tracking="iphone15p-53"),
        selected(umi="oak-FFC-57"),
        selected(gripper="UMI-V4-0002"),
        selected(tracking=""),
        selected(tracking="unknown"),
        {"gripper": "UMI-V4-0001", "umi": "oak-FFC-54"},
    ]
    for found in (bench(), None, []):
        inventory = gripper_pairing.evidence(found)
        result = subprocess.run(
            [
                node,
                "-e",
                "const {check}=require(process.argv[1]); const fs=require('fs'); "
                "const {samples,inventory}=JSON.parse(fs.readFileSync(0,'utf8')); "
                "process.stdout.write(JSON.stringify(samples.map(s=>check(s,inventory))));",
                str(REPO / "station_ui" / "static" / "gripper_diagram.js"),
            ],
            input=json.dumps({"samples": samples, "inventory": inventory}),
            text=True,
            capture_output=True,
            check=True,
        )
        assert json.loads(result.stdout) == [
            gripper_pairing.check(sample, inventory) for sample in samples
        ]

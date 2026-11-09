# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration uses one shared form, submitted by downstream actions."""

import pathlib
import shutil

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("yaml")

import test_unavailable_device_gate as gate  # noqa: E402

from station_ui.app import create_app  # noqa: E402
from station_ui.core import device_detector  # noqa: E402
from station_ui.routes import config as config_routes  # noqa: E402
from station_ui.settings import Settings  # noqa: E402
from umi.common.tasks import task_registry_store  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]

OAK_L = gate.OAK_L
OAK_R = gate.OAK_R
IPHONE_L = gate.IPHONE_L
IPHONE_R = gate.IPHONE_R


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(task_registry_store, "get_all_tasks", lambda: dict(gate.TASKS))
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


def _client(repo, *, drop_head=True) -> TestClient:
    built = TestClient(create_app(Settings(repo_root=repo)))
    built.get("/")
    built.post("/setup/preset", data={"preset_id": "bimanual_v4"})
    if drop_head:
        # A fresh session starts with a head OAK row. Most of these tests are
        # about the two arms, and an empty head row would put a fourth group on
        # every one of them.
        for assignment in list(built.app.state.rig.draft.optional):
            built.post(
                "/config/optional/remove?confirm=yes",
                data={"config_form": "1", "logical_name": assignment.logical_name},
            )
    return built


@pytest.fixture
def client(repo):
    return _client(repo)


def test_configuration_uses_the_full_form_without_camera_topology(client):
    body = client.get("/config").text
    assert "Camera topology" not in body
    assert 'id="topology-panel"' not in body
    assert "config-layout" not in body
    assert "page--wide" not in body
    assert "<h2>Devices</h2>" not in body


def test_configuration_keeps_one_shared_post_form(client):
    body = client.get("/config").text
    form = body[body.index('<form method="post" action="/config/apply"') :]
    form = form[: form.index("</form>")]
    assert 'id="configuration-form"' in form
    assert "apply-bar" not in form
    assert "Apply configuration" not in form
    assert "Configuration is validated and saved automatically" not in form


def test_calibration_buttons_submit_configuration_without_javascript(client):
    from html.parser import HTMLParser

    class Buttons(HTMLParser):
        def __init__(self):
            super().__init__()
            self.buttons = []

        def handle_starttag(self, tag, attrs):
            if tag == "button":
                self.buttons.append(dict(attrs))

    page = Buttons()
    page.feed(client.get("/config").text)
    calibrate = [
        button
        for button in page.buttons
        if button.get("formaction") == "/calibration/calibrate-gripper"
    ]
    assert {button["value"] for button in calibrate} == {"left", "right"}
    assert all(
        button["form"] == "configuration-form"
        and button["name"] == "side"
        and button["type"] == "submit"
        for button in calibrate
    )


def test_configuration_has_no_forward_button_or_dangling_reason(client):
    from html_sections import section_html

    body = section_html(client.get("/config").text, "configuration")
    assert "Go to Calibration" not in body
    assert 'aria-describedby="next-step-why"' not in body


def test_calibration_has_no_forward_control_or_completion_message(client):
    from html_sections import section_html

    body = section_html(client.get("/calibration").text, "calibration")

    assert 'id="next-step"' not in body
    assert "Go to Record" not in body
    assert "Calibration has not passed" not in body

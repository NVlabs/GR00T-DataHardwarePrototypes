# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recording-check choices survive real forms, saves, actions and restarts."""

import pytest
import test_config_page as forms
import yaml
from fastapi.testclient import TestClient

from station_ui import enforcement, recording, session
from station_ui.app import create_app
from station_ui.settings import Settings
from umi.common.tasks import task_registry_store


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(task_registry_store, "get_all_tasks", lambda: dict(forms.TASKS))
    client = forms._client(forms._repo(tmp_path))
    forms._apply(client, head=False)
    return client


def payload(client, relaxed=()):
    form = next(
        form for form in forms._page(client).forms if form["action"] == "/config/apply"
    )
    data = forms._submission(form)
    for check in enforcement.KEYS:
        data.pop(enforcement.FORM_FIELD + check, None)
        if check not in relaxed:
            data[enforcement.FORM_FIELD + check] = "1"
    return data


def saved(client):
    from station_ui import readiness

    return yaml.safe_load(readiness.generated_config(client.app).read_text())


def test_all_checks_are_enforced_by_default(client):
    assert "check_enforcement" not in saved(client)
    controls = [
        control
        for control in forms._controls(forms._page(client)).values()
        if (control.get("name") or "").startswith(enforcement.FORM_FIELD)
    ]
    assert len(controls) == len(enforcement.KEYS)
    assert all("checked" in control for control in controls)


def test_choices_save_and_restore_without_changing_the_rig_or_session(client):
    relaxed = {"pre:camera.relative_calibration", "data:timestamp"}
    before = session.device_fingerprint(client.app.state.rig.draft)
    uid = client.app.state.rig.draft.session_uid
    response = client.post("/config/apply", data=payload(client, relaxed))
    assert response.status_code == 200
    assert saved(client)["check_enforcement"] == {
        "pre": {"camera.relative_calibration": False},
        "data": {"timestamp": False},
    }
    assert session.device_fingerprint(client.app.state.rig.draft) == before
    assert client.app.state.rig.draft.session_uid == uid
    settings = client.app.state.settings
    restored = TestClient(
        create_app(Settings(repo_root=settings.repo_root, state_dir=settings.state_dir))
    )
    restored.get("/")
    assert set(restored.app.state.rig.draft.relaxed_checks) == relaxed
    assert restored.app.state.rig.draft.session_uid == uid
    controls = forms._controls(forms._page(restored))
    assert "checked" not in controls[enforcement.FORM_FIELD + "data:timestamp"]
    assert "checked" in controls[enforcement.FORM_FIELD + "data:gripper"]


def test_all_warn_only_and_back_to_defaults(client):
    assert (
        client.post("/config/apply", data=payload(client, enforcement.KEYS)).status_code
        == 200
    )
    assert set(enforcement.restored(saved(client))) == enforcement.KEYS
    assert client.post("/config/apply", data=payload(client)).status_code == 200
    assert "check_enforcement" not in saved(client)


def test_record_start_applies_current_checkboxes_before_readiness_gate(
    client, monkeypatch
):
    from station_ui.routes import record

    monkeypatch.setattr(record, "preflight", lambda *args, **kwargs: ["would-record"])
    response = client.post("/record/start", data=payload(client, {"data:tracking"}))
    assert response.status_code == 400  # The real calibration gate remains.
    assert saved(client)["check_enforcement"] == {"data": {"tracking": False}}
    assert record.workflow_state(client.app).record_job_id is None


def test_short_posts_do_not_clear_checkbox_choices(client):
    client.post("/config/apply", data=payload(client, {"data:gripper"}))
    client.post("/config/task", data={"task_name": "plane_assembly_v9"})
    assert client.app.state.rig.draft.relaxed_checks == ["data:gripper"]


def test_unknown_check_is_rejected_without_changing_saved_config(client):
    before = saved(client)
    data = payload(client)
    data[enforcement.FORM_FIELD + "data:not_a_check"] = "1"
    assert client.post("/config/apply", data=data).status_code == 400
    assert saved(client) == before


def test_read_only_and_recording_locks_disable_checkboxes_and_reject_changes(
    client, monkeypatch
):
    onlooker = TestClient(client.app)
    controls = forms._controls(forms._parse(onlooker.get("/config").text))
    assert all(
        "disabled" in control
        for name, control in controls.items()
        if name.startswith(enforcement.FORM_FIELD)
    )
    assert (
        onlooker.post(
            "/config/apply", data=payload(client, enforcement.KEYS)
        ).status_code
        == 409
    )
    monkeypatch.setattr(recording, "in_progress", lambda app: True)
    controls = forms._controls(forms._page(client))
    assert all(
        "disabled" in control
        for name, control in controls.items()
        if name.startswith(enforcement.FORM_FIELD)
    )
    assert (
        client.post("/config/apply", data=payload(client, enforcement.KEYS)).status_code
        == 409
    )
    assert "check_enforcement" not in saved(client)


def test_warn_only_titles_name_what_the_saved_config_relaxes():
    config = {
        "check_enforcement": {
            "pre": {"camera.relative_calibration": False},
            "data": {"timestamp": False, "gripper": True},
        }
    }
    assert enforcement.warn_only_titles(config) == [
        "Camera calibration",
        "Frame timing",
    ]
    assert enforcement.warn_only_titles({}) == []


def test_record_section_lists_steps_to_redo_and_warn_only_checks(client):
    client.post(
        "/config/apply", data=payload(client, {"pre:camera.relative_calibration"})
    )
    html = client.get("/record").text
    assert "Before you start" in html
    assert "Camera check" in html and "has not been run yet" in html
    assert "Warn only: Camera calibration" in html


def test_start_refusal_refreshes_the_calibration_section(client):
    response = client.post(
        "/record/start", data=payload(client), headers={"HX-Request": "true"}
    )
    assert response.headers["HX-Trigger-After-Swap"] == "workflow-refresh"
    assert "Before you start" in response.text

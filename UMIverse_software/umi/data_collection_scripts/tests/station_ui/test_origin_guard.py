# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A mutating request another origin sent is refused before any route runs.

Cookies ignore ports, so a jupyterlab notebook on another localhost port is a
standing same-site origin and SameSite does not stop its form.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from station_ui.app import create_app

FORBIDDEN = 403
SEE_OTHER = 303


def _client():
    client = TestClient(create_app())
    client.get("/")  # claims the rig, exactly as a browser does
    return client


def test_a_cross_origin_post_is_refused():
    client = _client()

    response = client.post(
        "/setup/preset",
        data={"preset_id": "bimanual_v4"},
        headers={"Origin": "http://localhost:3000"},
        follow_redirects=False,
    )

    assert response.status_code == FORBIDDEN
    assert "did not come from this station's console" in response.text
    assert not client.app.state.rig.draft.preset_id, "the refused post still landed"


def test_a_cross_origin_referer_is_refused_too():
    response = _client().post(
        "/setup/preset",
        data={"preset_id": "bimanual_v4"},
        headers={"Referer": "http://localhost:3000/notebook"},
        follow_redirects=False,
    )

    assert response.status_code == FORBIDDEN


def test_a_post_from_this_console_goes_through():
    client = _client()

    response = client.post(
        "/setup/preset",
        data={"preset_id": "bimanual_v4"},
        headers={"Origin": "http://testserver"},
        follow_redirects=False,
    )

    assert response.status_code == SEE_OTHER
    assert client.app.state.rig.draft.preset_id == "bimanual_v4"


def test_a_cross_origin_get_is_left_alone():
    """Reads change nothing, and a link from anywhere must still open the page."""
    response = _client().get("/", headers={"Origin": "http://localhost:3000"})

    assert response.status_code == 200

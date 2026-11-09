# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Steps 1-4 are read-only while a recorder is alive.

`preflight` refuses to start a recording while a camera check holds the
cameras. The reverse has to be refused too, and all four reverses are real:

* The camera check opens every camera the recorder holds exclusively, so it
  reports the rig broken and the operator re-seats a cable that is fine --
  exactly the failure `CHECK_HOLDS_THE_CAMERAS` prevents, running the other way.
* A device scan does the same through `scan_worker --probe`.
* "Start a new session" mints a new uid and removes the config while the running
  recorder goes on writing into the old folder, so the page shows session B
  while episodes land in session A and nothing says they disagree.
* Apply rewrites the config the recorder already read.
"""

from __future__ import annotations

import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

# Same directory, no `__init__.py`, so pytest's prepend import mode has already
# put `tests/station_ui` on sys.path.
import test_config_page as config_page  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from station_ui import recording  # noqa: E402
from station_ui.app import create_app  # noqa: E402
from station_ui.settings import Settings  # noqa: E402
from umi.common.tasks import task_registry_store  # noqa: E402

# Every route that mutates the rig or opens hardware. Cancels and stops are
# deliberately absent -- see `test_release_actions_are_never_locked`.
LOCKED = [
    ("/setup/preset", {"preset_id": "bimanual_v4"}),
    ("/config/apply", {"config_form": "1"}),
    ("/config/session/uid", {"config_form": "1", "confirm": "yes"}),
    ("/config/devices/scan", {}),
    ("/config/devices/adopt", {"config_form": "1"}),
    ("/config/optional/add", {"config_form": "1", "type_id": "head_oak"}),
    ("/calibration/check/start", {}),
    ("/calibration/calibrate-gripper", {"side": "left"}),
]


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(
        task_registry_store, "get_all_tasks", lambda: dict(config_page.TASKS)
    )


@pytest.fixture
def station(tmp_path):
    repo = config_page._repo(tmp_path)
    client = TestClient(create_app(Settings(repo_root=repo)))
    client.get("/")
    client.post("/setup/preset", data={"preset_id": "bimanual_v4"})
    config_page._apply(client, head=False)
    return client


@pytest.fixture
def busy(station, monkeypatch):
    """A station whose recorder is alive.

    Patched at `in_progress`, which is the one predicate every guard reads, so
    this cannot pass by locking on some second notion of "recording".
    """
    monkeypatch.setattr(recording, "in_progress", lambda app: True)
    return station


@pytest.mark.parametrize("path,data", LOCKED)
def test_every_mutating_route_is_refused_while_recording(busy, path, data):
    """Driven through the real routes. The disabled control is an affordance;
    a bookmark, a replay or a second tab still POSTs here."""
    response = busy.post(path, data=data, follow_redirects=False)
    assert response.status_code == 409, f"{path} was not locked"


@pytest.mark.parametrize("path,data", LOCKED)
def test_the_same_routes_are_allowed_once_the_recorder_is_gone(station, path, data):
    """The other half. A lock that never releases is a worse dead end than the
    one this design removes."""
    response = station.post(path, data=data, follow_redirects=False)
    assert response.status_code != 409, f"{path} stayed locked with no recorder"


@pytest.mark.parametrize("path", ["/calibration/check/stop", "/take-control"])
def test_release_actions_are_never_locked(busy, path):
    """A cancel, a stop and taking control all release something or let the
    operator reach Stop. Locking a release strands the very thing holding the
    hardware -- and locking take-control would leave a recording nobody may
    stop."""
    response = busy.post(path, follow_redirects=False)
    assert response.status_code != 409, f"{path} was locked, stranding the operator"


@pytest.mark.parametrize("page", ["/", "/config", "/calibration"])
def test_the_pages_are_still_readable_while_locked(busy, page):
    """Read-only, not inaccessible. The operator needs to see the config they
    are recording with -- that is diagnostic, and it is when they most want it."""
    assert busy.get(page).status_code == 200


def test_a_locked_page_says_why_and_where_to_go(busy):
    """Otherwise the ownership-conflict and recording-lock cases render
    identically and the operator hunts for the wrong remedy."""
    body = busy.get("/config").text
    assert "A recording is in progress" in body
    assert 'href="#record"' in body, "no route to the page that can stop it"


def test_the_banner_is_absent_with_no_recording(station):
    assert "A recording is in progress" not in station.get("/config").text


def test_the_record_page_is_not_banner_locked(busy):
    """It is where the operator goes to end the recording, so its own controls
    must not read as disabled by the thing they are there to stop."""
    body = busy.get("/record").text
    assert "A recording is in progress, so this page is read-only" not in body

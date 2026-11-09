# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The picker offers what is plugged in, and fills itself in when the scan ends.

After a scan, only the attached devices are offered. Listing the whole device
registry would bury the boards on this bench among serials from other benches,
and choosing one of those is a config that cannot record, found out at record
time.

The lists fill in when the scan ends, without a second button. The auto-scan
fires on load, so the boxes render before any scan result exists, and
re-rendering a box would discard what the operator typed during the 15-30s
scan -- the window they fill the session fields in. So the lists live apart
from the boxes: values stay in the `<input>`s and are never re-rendered,
options live in one node and are replaced wholesale.

What must not regress: the box is a text input, and a name the list does not
carry is still accepted. Configs get prepped before hardware reaches the bench,
and the device registry may not list every device yet.
"""

import pathlib
import re
import shutil
import sys
import time
from html.parser import HTMLParser

import pytest
from fastapi.testclient import TestClient

yaml = pytest.importorskip("yaml")

import test_unavailable_device_gate as gate  # noqa: E402

from station_ui import devices as devices_core  # noqa: E402
from station_ui import jobs  # noqa: E402
from station_ui.app import create_app  # noqa: E402
from station_ui.core import device_detector  # noqa: E402
from station_ui.routes import config as config_routes  # noqa: E402
from station_ui.settings import Settings  # noqa: E402
from umi.common.tasks import task_registry_store  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]

# Registry names that are not attached in any scan below. The registry knows
# them; this bench does not have them.
ELSEWHERE = ("oak-FFC-77", "oak-FFC-88")

CAMERA_FIELD = "camera.left_umi_camera"


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(task_registry_store, "get_all_tasks", lambda: dict(gate.TASKS))
    # The registry, which is deliberately wider than what the scan finds.
    monkeypatch.setattr(
        device_detector,
        "get_available_oak_cameras",
        lambda: [gate.OAK_R, gate.OAK_L, *ELSEWHERE],
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
    return built


def a_scan():
    return [
        gate.found_record("oak", gate.OAK_L_ID, gate.OAK_L),
        gate.found_record("oak", gate.OAK_R_ID, gate.OAK_R),
        gate.found_record("iphone", "00008130-000000000001", gate.IPHONE_R),
        gate.found_record("iphone", "00008130-000000000002", gate.IPHONE_L),
    ]


# ------------------------------------------------------------------- parsing


class _Lists(HTMLParser):
    """`datalist id -> [option value]`, and the `list=` each input points at."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.lists: dict[str, list[str]] = {}
        self.inputs: dict[str, str] = {}
        self._current: str | None = None

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        if tag == "datalist":
            self._current = attr.get("id") or ""
            self.lists.setdefault(self._current, [])
        elif tag == "option" and self._current is not None:
            self.lists[self._current].append(attr.get("value") or "")
        elif tag == "input" and attr.get("list"):
            self.inputs[attr.get("name") or ""] = attr["list"]

    def handle_endtag(self, tag):
        if tag == "datalist":
            self._current = None


def parsed(html: str) -> _Lists:
    out = _Lists()
    out.feed(html)
    return out


def options_for(html: str, field: str) -> list[str]:
    """Everything the box named *field* offers, resolved through its `list=`."""
    page = parsed(html)
    assert field in page.inputs, f"no box named {field}"
    list_id = page.inputs[field]
    assert list_id in page.lists, f"{field} points at a datalist that is not rendered"
    return page.lists[list_id]


# -------------------------------------------------------- what is offered


def test_before_any_scan_the_registry_is_offered(client):
    """Prepping a config before the hardware arrives has to stay possible."""
    offered = options_for(client.get("/config").text, CAMERA_FIELD)

    assert gate.OAK_L in offered
    for name in ELSEWHERE:
        assert name in offered, (
            "nothing else is known yet, so the fleet is all there is"
        )


def test_after_a_scan_only_connected_devices_are_offered(client, monkeypatch):
    """No list full of other benches' serials."""
    gate.run_scan(client, monkeypatch, a_scan())

    offered = options_for(client.get("/config").text, CAMERA_FIELD)

    assert set(offered) == {gate.OAK_L, gate.OAK_R}
    for name in ELSEWHERE:
        assert name not in offered, f"{name} is not attached to this station"


def test_a_device_the_scan_never_saw_is_still_accepted(client, monkeypatch):
    """Offering less must not mean accepting less: the box is a text input.

    The registry falls back to its bundled snapshot on a bad-network day, so a
    picker that were also a gate would empty the screen.
    """
    gate.run_scan(client, monkeypatch, a_scan())

    response = client.post(
        "/config/apply", data=gate.handcrafted(**{CAMERA_FIELD: gate.NEVER_SEEN})
    )

    assert response.status_code == 200, response.text
    cfg = yaml.safe_load(gate.generated(None).read_text())
    names = [
        source.get("oak_device_name")
        for source in cfg["video_sources"]
        if source.get("type") == "oak_device"
    ]
    assert gate.NEVER_SEEN in names


# ----------------------------------------------- filling in without a click


def test_the_scan_fragment_carries_the_refreshed_options(client, monkeypatch):
    """What the poller gets back has to include the lists, or nothing updates."""
    gate.run_scan(client, monkeypatch, a_scan())

    fragment = client.get("/config/devices/status").text

    assert 'id="device-choices"' in fragment
    assert "hx-swap-oob" in fragment, "the lists would land nowhere without this"
    assert gate.OAK_L in fragment and gate.OAK_R in fragment


def test_the_scan_fragment_never_re_renders_a_device_box(client, monkeypatch):
    """Why the boxes themselves are never swapped.

    An operator fills in the session fields while the scan runs. A fragment that
    carried `<input>` elements would swap their typed values away.
    """
    gate.run_scan(client, monkeypatch, a_scan())

    fragment = client.get("/config/devices/status").text

    assert not re.search(r"<input\b(?![^>]*type=[\"']hidden)", fragment), fragment[:500]


def test_there_is_no_second_button_to_adopt_the_results(client, monkeypatch):
    """No "Use the scan results" button: the lists fill in on their own."""
    gate.run_scan(client, monkeypatch, a_scan())
    body = client.get("/config").text

    assert "Use the scan results" not in body
    assert "/config/devices/adopt" not in body


# ------------------------------------------------- what is offered mid-refresh


def _scanning(client, monkeypatch, seconds=60.0):
    """Start a scan that will not finish, and leave it running.

    60s rather than a couple: every assertion below has to be answered while the
    scan is unambiguously in flight, or the test reads a finished job and passes
    for the wrong reason.
    """
    monkeypatch.setattr(
        devices_core,
        "scan_argv",
        lambda: [sys.executable, "-c", f"import time\ntime.sleep({seconds!r})"],
    )
    client.post("/config/devices/scan")
    job = devices_core.current_job(client.app)
    assert job is not None and not job.finished, "no scan is running"
    return job


def _stop(client, job):
    """Cancel the scan and wait until it has really ended.

    The same deadline and the same assertion as `gate.run_scan`: a CI runner
    sharing one CPU across xdist workers can take longer than a few seconds
    to reap the child, and a helper that gave up silently would leave the next
    assertion reading a scan still "in flight" -- an empty picker, reported as
    the registry having gone missing.
    """
    jobs.cancel(devices_core.registry_for(client.app), job.id)
    end = time.monotonic() + gate.DEADLINE_S
    while not job.finished and time.monotonic() < end:
        time.sleep(0.02)
    assert job.finished, f"the cancelled scan never ended: {job.state}"


def test_a_refresh_in_flight_offers_nothing(client, monkeypatch):
    """While a scan runs the list is empty, not the whole registry.

    `found` is None for the whole 15-30s a scan takes. Treating that like "no
    scan has ever run" would fall back to the entire registry, which
    `test_after_a_scan_only_connected_devices_are_offered` prevents for the
    finished case. Worse mid-scan: the list is replaced the moment the scan
    lands, so anything picked from it would come from a list already being
    discarded.
    """
    job = _scanning(client, monkeypatch)
    try:
        offered = options_for(client.get("/config").text, CAMERA_FIELD)
    finally:
        _stop(client, job)

    assert offered == [], f"the fleet is offered while the scan runs: {offered}"


def test_a_refresh_in_flight_says_which_kind_of_empty_it_is(client, monkeypatch):
    """An empty list has two meanings and they need different words. Without the
    marker `combobox.js` reads, the dropdown would tell an operator to "press
    Refresh devices" while the refresh they pressed was still running."""
    job = _scanning(client, monkeypatch)
    try:
        page = client.get("/config").text
    finally:
        _stop(client, job)

    list_id = parsed(page).inputs[CAMERA_FIELD]
    assert f'<datalist id="{list_id}" data-scanning="true">' in page

    script = (REPO / "station_ui" / "static" / "combobox.js").read_text()
    assert "data-scanning" in script, "nothing reads the marker"
    assert "Looking for devices" in script


def test_the_options_arrive_when_the_scan_lands(client, monkeypatch):
    """Emptying the list mid-scan is only acceptable because it fills itself in
    without a click -- the poller carries the lists back out-of-band."""
    gate.run_scan(client, monkeypatch, a_scan())

    page = client.get("/config").text

    assert set(options_for(page, CAMERA_FIELD)) == {gate.OAK_L, gate.OAK_R}
    assert "data-scanning" not in page


def test_a_cancelled_scan_offers_the_registry_again(client, monkeypatch):
    """Cancelled is not "in flight": nothing more is coming, and the operator
    now has to type a name from somewhere. This is the prep-before-hardware
    branch, and it has to come back the moment the scan stops."""
    job = _scanning(client, monkeypatch)
    _stop(client, job)

    offered = options_for(client.get("/config").text, CAMERA_FIELD)

    for name in ELSEWHERE:
        assert name in offered, "a cancelled scan left the operator nothing to type"


def test_every_box_points_at_a_list_that_is_really_rendered(client, monkeypatch):
    """`list=` matches by id, so a box whose list moved out is a silent dead end."""
    gate.run_scan(client, monkeypatch, a_scan())
    page = parsed(client.get("/config").text)

    assert page.inputs, "no device boxes rendered at all"
    for field, list_id in page.inputs.items():
        assert list_id in page.lists, f"{field} points at a missing datalist"

    # And exactly one element carries each id: two `<datalist id=x>` in one
    # document is a page where the browser picks which one wins.
    ids = re.findall(r'<datalist id="([^"]+)"', client.get("/config").text)
    assert len(ids) == len(set(ids)), ids

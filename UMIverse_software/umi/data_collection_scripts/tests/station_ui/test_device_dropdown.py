# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ▾ beside a device box shows the full connected list, never a filtered one.

A `<datalist>` is a filter, not a dropdown -- the browser narrows the options to
what has been typed, and no attribute turns it off. So when reopening a config
puts a device that is not plugged in into the box, both connected boards would
match nothing: an empty dropdown with two cameras sitting on the bench.

So a real dropdown sits beside the box, reading the same `<datalist>`. These
tests cover the half that is markup. What `combobox.js` does with it -- open on
click, ignore the box's contents, write the pick back with a `change` event so
the gripper's linked camera still fills in -- is JavaScript this suite does not
execute, so the contract asserted here is the one the script depends on:

  * the button exists, and only where the page is editable
  * it ships `hidden`, so a blocked script leaves no dead control
  * it points at the same list the input does, which is what makes the
    out-of-band swap after a scan refresh the dropdown too

What must not regress: the box is still a text input with its datalist, so
typing still filters and a name the list does not carry is still accepted.
"""

import pathlib
import shutil
from html.parser import HTMLParser

import pytest
from fastapi.testclient import TestClient

yaml = pytest.importorskip("yaml")

import test_unavailable_device_gate as gate  # noqa: E402

from station_ui.app import create_app  # noqa: E402
from station_ui.core import device_detector  # noqa: E402
from station_ui.routes import config as config_routes  # noqa: E402
from station_ui.settings import Settings  # noqa: E402
from umi.common.tasks import task_registry_store  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]

CAMERA_FIELD = "camera.left_umi_camera"


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(task_registry_store, "get_all_tasks", lambda: dict(gate.TASKS))
    monkeypatch.setattr(
        device_detector, "get_available_oak_cameras", lambda: [gate.OAK_R, gate.OAK_L]
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


class _Combos(HTMLParser):
    """Every `.combo`, as `{input name: {"list": id, "toggle": attrs|None}}`."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.combos: dict[str, dict] = {}
        self._open: dict | None = None

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        classes = (attr.get("class") or "").split()
        if tag == "div" and "combo" in classes:
            self._open = {"list": "", "toggle": None, "name": ""}
        elif self._open is not None and tag == "input":
            self._open["name"] = attr.get("name") or ""
            self._open["list"] = attr.get("list") or ""
        elif self._open is not None and tag == "button" and "combo__toggle" in classes:
            self._open["toggle"] = attr

    def handle_endtag(self, tag):
        if tag == "div" and self._open is not None and self._open["name"]:
            self.combos[self._open["name"]] = self._open
            self._open = None


def combos(html: str) -> dict:
    parser = _Combos()
    parser.feed(html)
    return parser.combos


def test_every_device_box_carries_a_dropdown_button(client):
    page = client.get("/config").text
    found = combos(page)

    assert CAMERA_FIELD in found, "the camera box is not in a .combo at all"
    for name, combo in found.items():
        assert combo["toggle"] is not None, f"{name} has no dropdown"


def test_the_button_ships_hidden_so_a_blocked_script_leaves_no_dead_control(client):
    """Every other enhancement on this console degrades to the plain control.

    `combobox.js` reveals these on load. Shipped visible, a station with the
    vendored script blocked would show a ▾ that does nothing when pressed --
    worse than no button, because it claims a capability.
    """
    combo = combos(client.get("/config").text)[CAMERA_FIELD]

    assert "hidden" in combo["toggle"]


def test_the_dropdown_reads_the_same_list_the_box_filters(client):
    """One source of options, and it is the one the scan already refreshes.

    `_partials/device_choices.html` swaps every `<datalist>` out-of-band when a
    scan finishes. Because the script reads the options from there at click
    time rather than from a copy rendered beside the button, a dropdown opened
    after that swap is current without the script knowing the swap happened.
    """
    combo = combos(client.get("/config").text)[CAMERA_FIELD]

    assert combo["list"], "the box points at no datalist"
    assert combo["toggle"]["aria-controls"] == combo["list"]


def test_the_box_is_still_a_text_input_that_takes_an_unlisted_name(client):
    """The dropdown is an addition, not a replacement.

    Configs get prepped before hardware reaches the bench, and `device_registry`
    falls back to its bundled snapshot when it cannot load. A `<select>` here
    would make both unworkable.
    """
    page = client.get("/config").text
    assert f'name="{CAMERA_FIELD}"' in page
    assert f'list="{combos(page)[CAMERA_FIELD]["list"]}"' in page

    client.post(
        "/config/apply",
        data={
            "config_form": "1",
            "task_name": "assembly",
            "operator_name": "ry",
            CAMERA_FIELD: "oak-FFC-99",
        },
    )
    assert "oak-FFC-99" in client.get("/config").text


def test_a_read_only_tab_gets_no_dropdown(client, repo):
    """A second browser on the same app does not own the rig.

    Its boxes are `readonly`, so a ▾ there would be a control that opens a list
    and then silently fails to write the pick anywhere.
    """
    app = create_app(Settings(repo_root=repo))
    owner = TestClient(app)
    owner.get("/")
    owner.post("/setup/preset", data={"preset_id": "bimanual_v4"})

    watcher = TestClient(app).get("/config").text

    assert 'name="' + CAMERA_FIELD + '"' in watcher, "the page did not render"
    assert "readonly" in watcher, "the second tab was handed an editable page"
    found = combos(watcher)
    assert found, "no device boxes were parsed, so this asserts nothing"
    for name, combo in found.items():
        assert combo["toggle"] is None, f"{name} offers a dropdown it cannot use"

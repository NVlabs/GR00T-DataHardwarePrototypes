# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One pick per arm: choosing a bound gripper fills in its camera.

The bank knows which OAK board is bolted to which gripper, so the form does not
ask twice. Everything unbound -- every gripper the pairing list does not name --
renders as a plain box.
"""

import pathlib
import re

import pytest

from station_ui import config_form
from station_ui.core.config_assembler import (
    DeviceBlock,
    RequiredCamera,
    TemplateRequirements,
)
from station_ui.routes import config as config_routes

# Match template requirements: a RequiredCamera for each stream and a
# DeviceBlock for each physical source. absorb uses the camera list to
# decide which assignments are declared.
_STREAMS = (
    ("left_umi_camera", "oak_device", "oak"),
    ("right_umi_camera", "oak_device", "oak"),
    ("left_trigger_camera", "uvc_camera", "uvc"),
    ("right_trigger_camera", "uvc_camera", "uvc"),
    ("left_tracking_camera", "iphone_camera", "iphone"),
    ("right_tracking_camera", "iphone_camera", "iphone"),
)

REQUIREMENTS = TemplateRequirements(
    cameras=tuple(RequiredCamera(*stream) for stream in _STREAMS),
    gripper_arms=("left", "right"),
    # One block per source, and every source of the V4 template carries one
    # stream: the wrist OAK records CAM_B only.
    blocks=tuple(
        DeviceBlock(name, source_type, kind, (name,))
        for name, source_type, kind in _STREAMS
    ),
)

LINKS = {"UMI-V4-0008": "oak-FFC-08"}


class Draft:
    def __init__(self, grippers=None, cameras=None):
        self.grippers = dict(grippers or {})
        self.required_cameras = dict(cameras or {})
        self.optional = []


@pytest.fixture(autouse=True)
def bank(monkeypatch):
    import umi.utils.device_registry as reg

    monkeypatch.setattr(reg, "get_gripper_cameras", lambda: dict(LINKS))
    monkeypatch.setattr(reg, "name_for_identifier", lambda v: str(v).strip() or None)
    yield


def slots():
    return config_routes._umi_camera_slots(REQUIREMENTS)


def test_each_arms_umi_camera_block_is_found_from_the_template_alone():
    """Derived from the stream names via _block_role, so nothing here has to
    know that the board is an OAK."""
    assert slots() == {
        "left": "left_umi_camera",
        "right": "right_umi_camera",
    }


def test_a_bound_gripper_fills_an_empty_camera_slot():
    draft = Draft()
    form = {"gripper.left": "UMI-V4-0008", "camera.left_umi_camera": ""}
    config_form.absorb(draft, form, REQUIREMENTS, umi_camera_slots=slots())
    assert draft.required_cameras["left_umi_camera"] == "oak-FFC-08"
    # The USB trigger camera under the same gripper is not the bank's to fill.
    assert "left_trigger_camera" not in draft.required_cameras


def test_a_camera_the_operator_typed_is_never_overwritten():
    """Re-pairing is legitimate: a board gets swapped on repair. The override
    is a fact about this session and must survive every re-absorb."""
    draft = Draft()
    form = {
        "gripper.left": "UMI-V4-0008",
        "camera.left_umi_camera": "oak-FFC-14",
    }
    config_form.absorb(draft, form, REQUIREMENTS, umi_camera_slots=slots())
    assert draft.required_cameras["left_umi_camera"] == "oak-FFC-14"


def test_the_fill_is_idempotent_across_repeated_submits():
    draft = Draft()
    form = {"gripper.left": "UMI-V4-0008", "camera.left_umi_camera": ""}
    for _ in range(3):
        config_form.absorb(draft, form, REQUIREMENTS, umi_camera_slots=slots())
    assert draft.required_cameras["left_umi_camera"] == "oak-FFC-08"


def test_an_unbound_gripper_leaves_the_slot_empty():
    """Every gripper the pairing list does not name. The row stays a plain
    box."""
    draft = Draft()
    form = {"gripper.left": "UMI-V4-0044", "camera.left_umi_camera": ""}
    config_form.absorb(draft, form, REQUIREMENTS, umi_camera_slots=slots())
    assert "left_umi_camera" not in draft.required_cameras


def test_the_other_arm_is_untouched():
    draft = Draft()
    form = {
        "gripper.left": "UMI-V4-0008",
        "camera.left_umi_camera": "",
        "gripper.right": "UMI-V4-0009",
        "camera.right_umi_camera": "",
    }
    config_form.absorb(draft, form, REQUIREMENTS, umi_camera_slots=slots())
    assert draft.required_cameras["left_umi_camera"] == "oak-FFC-08"
    assert "right_umi_camera" not in draft.required_cameras


def test_absorb_without_the_slots_argument_links_nothing():
    """Every existing caller and every existing test passes three arguments."""
    draft = Draft()
    form = {"gripper.left": "UMI-V4-0008", "camera.left_umi_camera": ""}
    config_form.absorb(draft, form, REQUIREMENTS)
    assert "left_umi_camera" not in draft.required_cameras


def test_the_tracking_camera_is_not_a_gripper_camera():
    """Only the UMI-camera block is derived. The iPhone stays a free pick."""
    assert "left_tracking_camera" not in slots().values()


# ------------------------------------------------------------------ the page

# Exercise both route-to-absorb wiring and the linked-camera label through the app.

_BOARD = "oak-FFC-08"
_TYPED = "oak-FFC-14"

_TASKS = {
    "plane_assembly_v9": {
        "name": "plane_assembly_v9",
        "short_description": "Assemble the plane",
        "mode": "bimanual",
        "HW_version": "v2.1",
        "training_group": "plane_assembly_v9",
    },
}

_FLEET = {
    "version": 0,
    "updated_at": "1970-01-01T00:00:00Z",
    "oak_devices": {_BOARD: "AAAA0000AAAA0008", _TYPED: "AAAA0000AAAA0014"},
    "iphone_devices": {},
    "umi_v4_grippers": {"UMI-V4-0008": "BBBB0000BBBB0008"},
    "gripper_cameras": dict(LINKS),
    "device_metadata": {},
}


def _console(tmp_path, monkeypatch):
    """A station console on a repo of its own, with a registry that has one pair."""
    import shutil

    from fastapi.testclient import TestClient

    from station_ui.app import create_app
    from station_ui.settings import Settings
    from umi.common.tasks import task_registry_store
    from umi.utils import device_registry

    monkeypatch.setattr(task_registry_store, "get_all_tasks", lambda: dict(_TASKS))
    monkeypatch.setattr(device_registry, "_REGISTRY", dict(_FLEET))

    repo = tmp_path / "repo"
    (repo / "config").mkdir(parents=True)
    root = pathlib.Path(__file__).resolve().parents[2]
    shutil.copytree(
        root / "config" / "data_recording",
        repo / "config" / "data_recording",
        ignore=shutil.ignore_patterns("generated"),
    )
    shutil.copytree(root / "config" / "data_check", repo / "config" / "data_check")

    client = TestClient(create_app(Settings(repo_root=repo)))
    client.get("/")
    client.post("/setup/preset", data={"preset_id": "bimanual_v4"})
    return client


def _refresh(client, **fields) -> str:
    """Post the whole form the way every mutating control does, then re-read the
    page. `/config/task` is the lightest of them and absorbs identically."""
    client.post("/config/task", data=fields, headers={"HX-Request": "true"})
    page = client.get("/config")
    assert page.status_code == 200, page.status_code
    return page.text


def _box(html: str, field: str) -> str:
    """The `<input>` tag for one device field, as written."""
    match = re.search(
        rf'<input[^>]*\bname="{re.escape(field)}"[^>]*>', html
    ) or re.search(rf'<input[^>]*\bid="{re.escape(field)}"[^>]*>', html)
    assert match, f"the page renders no {field} box"
    return match.group(0)


def _value_of(html: str, field: str) -> str:
    value = re.search(r'\bvalue="([^"]*)"', _box(html, field))
    return value.group(1) if value else ""


def _described_by(html: str, field: str) -> list[str]:
    """The ids one box points its `aria-describedby` at, in reading order."""
    described = re.search(r'\baria-describedby="([^"]*)"', _box(html, field))
    return described.group(1).split() if described else []


def _text_at(html: str, element_id: str) -> str:
    """What a screen reader would read out for one id, or None if the page
    renders no element with it -- which is the failure this catches: a
    describedby naming an absent id is read as nothing at all."""
    match = re.search(
        rf'<(\w+)[^>]*\bid="{re.escape(element_id)}"[^>]*>(.*?)</\1>', html, re.S
    )
    if match is None:
        return None
    return " ".join(re.sub(r"<[^>]*>", " ", match.group(2)).split())


def test_the_page_fills_the_box_and_says_where_the_camera_came_from(
    tmp_path, monkeypatch
):
    client = _console(tmp_path, monkeypatch)
    html = _refresh(
        client,
        **{
            "gripper.left": "UMI-V4-0008",
            "camera.left_umi_camera": "",
        },
    )
    assert _value_of(html, "camera.left_umi_camera") == _BOARD
    assert f"linked to {_BOARD}" in html, "the row does not say the box was filled in"


def test_the_page_says_so_when_the_operator_re_paired_the_board(tmp_path, monkeypatch):
    """A board swapped on repair is legitimate, but it must be visible rather
    than silent -- the config records a pair the bank disagrees with."""
    client = _console(tmp_path, monkeypatch)
    html = _refresh(
        client,
        **{
            "gripper.left": "UMI-V4-0008",
            "camera.left_umi_camera": _TYPED,
        },
    )
    assert _value_of(html, "camera.left_umi_camera") == _TYPED
    assert "not the linked camera" in html
    assert f"linked to {_BOARD}" not in html


def test_an_unbound_gripper_leaves_the_page_exactly_as_it_was(tmp_path, monkeypatch):
    """A registry with no pairing list binds nothing."""
    client = _console(tmp_path, monkeypatch)
    html = _refresh(
        client,
        **{
            "gripper.right": "UMI-V4-0009",
            "camera.right_umi_camera": "",
        },
    )
    assert _value_of(html, "camera.right_umi_camera") == ""
    assert "linked to" not in html
    assert "not the linked camera" not in html


# ------------------------------------------------- what a screen reader hears

# Follow aria-describedby to verify that pairing warnings are associated
# with their fields for screen-reader users.

_FIELD = "camera.left_umi_camera"


def test_the_filled_in_box_says_out_loud_where_its_camera_came_from(
    tmp_path, monkeypatch
):
    client = _console(tmp_path, monkeypatch)
    html = _refresh(
        client, **{"gripper.left": "UMI-V4-0008", "camera.left_umi_camera": ""}
    )
    spoken = " ".join(_text_at(html, ref) or "" for ref in _described_by(html, _FIELD))
    assert f"linked to {_BOARD}" in spoken, (
        f"the chip is not announced; the box points at {_described_by(html, _FIELD)}"
    )


def test_the_override_warning_is_announced_and_not_merely_shown(tmp_path, monkeypatch):
    client = _console(tmp_path, monkeypatch)
    html = _refresh(
        client, **{"gripper.left": "UMI-V4-0008", "camera.left_umi_camera": _TYPED}
    )
    spoken = " ".join(_text_at(html, ref) or "" for ref in _described_by(html, _FIELD))
    assert "not the linked camera" in spoken, (
        f"the warning is not announced; the box points at {_described_by(html, _FIELD)}"
    )


@pytest.mark.parametrize("typed", ["", _TYPED])
def test_the_chip_joins_the_boxs_other_description_rather_than_replacing_it(
    tmp_path, monkeypatch, typed
):
    """`aria-describedby` takes a list. The camera named here is registered but
    was never scanned, so the box already carries a message -- and adding the
    chip must not cost the operator that sentence."""
    client = _console(tmp_path, monkeypatch)
    html = _refresh(
        client, **{"gripper.left": "UMI-V4-0008", "camera.left_umi_camera": typed}
    )
    refs = _described_by(html, _FIELD)
    assert len(refs) > 1, f"the chip replaced the box's own message: {refs}"
    assert any(ref.startswith("link-") for ref in refs), refs
    assert any(ref.startswith(("message-", "error-")) for ref in refs), refs


def test_every_id_a_device_box_points_at_is_really_on_the_page(tmp_path, monkeypatch):
    """A describedby naming an absent element is read as nothing, so it is
    worse than saying nothing: the page claims to describe the field and then
    does not. Checked for every device box, in both link states."""
    client = _console(tmp_path, monkeypatch)
    for typed in ("", _TYPED):
        html = _refresh(
            client,
            **{"gripper.left": "UMI-V4-0008", "camera.left_umi_camera": typed},
        )
        boxes = re.findall(r'name="((?:camera|gripper)\.[^"]+)"', html)
        assert boxes, "the page renders no device boxes"
        for field in sorted(set(boxes)):
            for ref in _described_by(html, field):
                assert _text_at(html, ref) is not None, (
                    f"{field} points at id={ref!r}, which nothing on the page has"
                )


# ------------------------------------------------ the card, at the moment of
# ------------------------------------------------ the pick

# Verify that changing a gripper immediately swaps in its linked camera
# rather than waiting for a full-page render.


def _post_link(client, arm, **fields):
    """What the gripper box does on change once it is wired.

    The arm is in the path. The box posts the whole form, so every arm's
    gripper is in the body of every one of these and the body cannot say which
    of them the operator touched.
    """
    return client.post(
        f"/config/devices/link/{arm}", data=fields, headers={"HX-Request": "true"}
    )


def test_choosing_a_bound_gripper_swaps_the_arm_card_with_the_camera_filled(
    tmp_path, monkeypatch
):
    """The point of the link, at the moment the operator picks -- not one full
    render later."""
    client = _console(tmp_path, monkeypatch)
    response = _post_link(client, "left", **{"gripper.left": "UMI-V4-0008"})

    assert response.status_code == 200, response.status_code
    body = response.text
    # One arm card, not the whole page.
    assert "<html" not in body.lower()
    assert 'id="' + config_form.dom_id("arm", "left") + '"' in body
    assert _value_of(body, "camera.left_umi_camera") == _BOARD
    assert "linked" in body
    # A replacement card must preserve editable controls.
    assert "readonly" not in _box(body, "camera.left_umi_camera")
    assert "hx-post" in _box(body, "gripper.left"), (
        "the swapped-in card cannot swap again"
    )


def test_the_swap_carries_only_the_arm_that_changed(tmp_path, monkeypatch):
    client = _console(tmp_path, monkeypatch)
    body = _post_link(client, "left", **{"gripper.left": "UMI-V4-0008"}).text
    assert config_form.dom_id("arm", "right") not in body


def test_the_card_that_comes_back_is_the_arm_that_was_asked_for(tmp_path, monkeypatch):
    """`hx-include="closest form"` means both grippers are in every one of these
    posts, so an arm read off the body would be whichever card the page renders
    first -- and a right-arm pick would answer with the left card, which htmx
    then swaps into `#arm-right`: two cards claiming one id, and the right arm's
    inputs gone from the form until a reload."""
    client = _console(tmp_path, monkeypatch)
    body = _post_link(
        client,
        "right",
        **{"gripper.left": "UMI-V4-0008", "gripper.right": "UMI-V4-0009"},
    ).text
    assert 'id="' + config_form.dom_id("arm", "right") + '"' in body
    assert config_form.dom_id("arm", "left") not in body


def test_an_arm_this_rig_does_not_have_swaps_nothing(tmp_path, monkeypatch):
    """204, because htmx swaps nothing on one. A whole page answered at 200
    would be swapped into the card -- a <html> document inside a <div>."""
    client = _console(tmp_path, monkeypatch)
    response = _post_link(client, "middle", **{"gripper.left": "UMI-V4-0008"})
    assert response.status_code == 204, response.status_code
    assert not response.text


def test_a_read_only_tab_cannot_link(tmp_path, monkeypatch):
    """Same gate every other mutating route uses. A second tab watching the
    console must not be able to rewrite the rig."""
    from fastapi.testclient import TestClient

    client = _console(tmp_path, monkeypatch)
    other = TestClient(client.app)
    other.get("/config")
    response = other.post(
        "/config/devices/link/left",
        data={"gripper.left": "UMI-V4-0008"},
        headers={"HX-Request": "true"},
    )
    assert response.status_code in (403, 409), response.status_code


def test_a_post_without_the_script_gets_the_page_not_a_bare_card(tmp_path, monkeypatch):
    """Nothing posts here without the vendored script -- the box is a plain
    input, not a submit -- so a request with no HX header is a stray or a
    replayed one, and a bare card would render as the whole document."""
    client = _console(tmp_path, monkeypatch)
    response = client.post(
        "/config/devices/link/left",
        data={"gripper.left": "UMI-V4-0008"},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.status_code
    assert response.headers["location"] == "/config"


def test_the_gripper_box_is_wired_and_the_camera_box_is_not(tmp_path, monkeypatch):
    """Only the field that drives the fill posts on change. Wiring the camera
    box too would make an operator's own typing race the swap that overwrites
    it."""
    client = _console(tmp_path, monkeypatch)
    page = client.get("/config").text

    gripper = _box(page, "gripper.left")
    camera = _box(page, "camera.left_umi_camera")

    assert 'hx-post="/config/devices/link/left"' in gripper
    # The card's id and the swap target come from one generator, so the id the
    # page renders and the id the swap aims at cannot drift apart.
    assert 'hx-target="#' + config_form.dom_id("arm", "left") + '"' in gripper
    # Include unsaved camera fields when changing the gripper so linked-camera
    # fill cannot overwrite a manual assignment. Swap on change, not keystroke.
    assert 'hx-include="closest form"' in gripper
    assert 'hx-swap="outerHTML"' in gripper
    assert 'hx-trigger="change"' in gripper
    assert "hx-post" not in camera


def test_a_read_only_tab_is_not_handed_a_control_it_may_not_use(tmp_path, monkeypatch):
    """The box is `readonly` for a watcher, and a readonly input fires no
    `change` -- but the route is the rule and the markup is the hint, so the
    hint should not claim otherwise either."""
    from fastapi.testclient import TestClient

    client = _console(tmp_path, monkeypatch)
    other = TestClient(client.app)
    page = other.get("/config").text
    assert "hx-post" not in _box(page, "gripper.left")

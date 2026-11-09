# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The configuration page, and the failures it exists to prevent.

This page is the production caller of `new_session_uid`,
`session_has_episodes`, `SETUP_PRESETS` and `ConfigAssembler`. This suite holds
that wiring in place, and each block below names the failure it stops.

Nothing here asserts on source text. A test that greps a template can pass while
the server does the opposite, so every claim is made by driving the app: post
the form a browser would post, then read the config that was written or the
state the next GET reports.
"""

import datetime as dt
import os
import pathlib
import re
import shutil
from html.parser import HTMLParser

import pytest
import zarr
from fastapi.testclient import TestClient

yaml = pytest.importorskip("yaml")


from station_ui import day_roll, settings  # noqa: E402
from station_ui import devices as devices_core  # noqa: E402
from station_ui.app import create_app  # noqa: E402
from station_ui.config_form import absorb  # noqa: E402
from station_ui.core.config_assembler import (  # noqa: E402
    UNKNOWN_OPERATOR,
    ConfigAssembler,
    DeviceSpec,
    RequiredCamera,
    SessionSpec,
    TemplateRequirements,
)
from station_ui.core.session_identity import is_valid_session_uid  # noqa: E402
from station_ui.session import SessionDraft  # noqa: E402
from station_ui.settings import Settings  # noqa: E402
from umi.common.recording import session_seal  # noqa: E402
from umi.common.recording.record_manager import RecorderManager  # noqa: E402
from umi.common.tasks import task_registry_store  # noqa: E402
from umi.common.utility.output_layout import output_root  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]

# Two tasks with deliberately different data-check overrides, neither of which
# matches the shipped baseline (gripper.range_max_mm is 90.0 there), so a test
# can tell whose override reached a generated file.
TASKS = {
    "plane_assembly_v9": {
        "name": "plane_assembly_v9",
        "short_description": "Assemble the plane",
        "mode": "bimanual",
        "HW_version": "v2.1",
        "training_group": "plane_assembly_v9",
        "data_check_overrides": {"gripper": {"range_max_mm": 88.0}},
    },
    "a_30_second_task": {
        "name": "a_30_second_task",
        "short_description": "Do the short thing",
        "mode": "bimanual",
        "HW_version": "v3",
        "training_group": "a_30_second_task",
        "data_check_overrides": {"gripper": {"range_max_mm": 85.0}},
    },
}


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    """A deterministic task registry.

    Patched at the store, so the real `_LazyTaskRegistry` / `_spec_from_entry`
    code still runs -- the alternative, stubbing `get_task`, would let a lookup
    that never reaches the registry pass.
    """
    monkeypatch.setattr(task_registry_store, "get_all_tasks", lambda: dict(TASKS))


def _repo(tmp_path) -> pathlib.Path:
    """A repo root carrying its own copy of the files assembly reads.

    A test that used the real repo root could end up modifying tracked files.
    """
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    # `generated/` is deliberately not copied: inheriting a stale one would make
    # "a rejected apply wrote nothing" pass while it wrote a file.
    shutil.copytree(
        REPO / "config" / "data_recording",
        root / "config" / "data_recording",
        ignore=shutil.ignore_patterns("generated"),
    )
    shutil.copytree(REPO / "config" / "data_check", root / "config" / "data_check")
    assert not _generated(root).exists()
    return root


def _client(repo, *, preset="bimanual_v4", head_oak=True) -> TestClient:
    """A browser that owns the rig and has chosen a preset, like a real one.

    A fresh session has no head camera, so by default the rig is given a head
    OAK the way an operator would: through the Add button.
    """
    client = TestClient(create_app(Settings(repo_root=repo)))
    client.get("/")
    if preset:
        client.post("/setup/preset", data={"preset_id": preset})
        if head_oak:
            client.post(
                "/config/optional/add",
                data={"config_form": "1", "type_id": "head_oak"},
            )
    return client


# --------------------------------------------------------------------- parsing


class _Page(HTMLParser):
    """Forms and their controls, exactly as a browser would submit them.

    Deliberately structural rather than string matching: a control that only
    works from JavaScript, or a checkbox rendered `checked` when the saved value
    is false, are both invisible to a substring assertion.
    """

    _CONTROLS = ("input", "select", "textarea", "button")
    # Elements with no end tag, so the id stack must not wait for one.
    _VOID = frozenset(
        {
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
        }
    )

    def __init__(self):
        super().__init__()
        self.forms: list[dict] = []
        self.inline_handlers: list[str] = []
        self.text_by_id: dict[str, str] = {}
        self.attrs_by_id: dict[str, dict] = {}
        self.body = ""
        self._form: dict | None = None
        self._select: dict | None = None
        # The button whose text is being read, until its end tag: text after a
        # button belongs to whatever follows it, not to the button.
        self._button: dict | None = None
        # One entry per open element, id or None, so text is only credited to
        # the elements that actually contain it.
        self._open: list[str | None] = []

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        self.inline_handlers += [k for k in attr if k.startswith("on")]
        if attr.get("id"):
            self.attrs_by_id[attr["id"]] = attr
            self.text_by_id.setdefault(attr["id"], "")
        if tag not in self._VOID:
            self._open.append(attr.get("id"))
        if tag == "form":
            self._form = {
                "action": attr.get("action"),
                "method": (attr.get("method") or "get").lower(),
                "controls": [],
            }
            self.forms.append(self._form)
            return
        if self._form is None:
            return
        if tag == "option":
            if self._select is not None:
                self._select["options"].append(attr)
            return
        if tag not in self._CONTROLS:
            return
        control = {
            **attr,
            "tag": tag,
            "kind": (attr.get("type") or ("submit" if tag == "button" else "text")),
            "classes": (attr.get("class") or "").split(),
            "text": "",
            "options": [],
        }
        self._form["controls"].append(control)
        if tag == "select":
            self._select = control
        elif tag == "button":
            self._button = control

    def handle_endtag(self, tag):
        if tag == "form":
            self._form = None
        elif tag == "select":
            self._select = None
        elif tag == "button":
            self._button = None
        if tag not in self._VOID and self._open:
            self._open.pop()

    def handle_data(self, data):
        self.body += data
        for element_id in self._open:
            if element_id is not None:
                self.text_by_id[element_id] += data
        if self._button is not None:
            self._button["text"] += data


def _parse(html: str) -> _Page:
    from html_sections import section_html

    html = section_html(html, "configuration")
    page = _Page()
    page.feed(html)
    return page


def _controls(page: _Page) -> dict[str, dict]:
    """Every named control on the page, by name. Names are unique here."""
    return {
        control["name"]: control
        for form in page.forms
        for control in form["controls"]
        if control.get("name") and control["kind"] not in ("submit", "button")
    }


def _submission(form: dict) -> dict[str, str]:
    """What a browser would send for this form, with no button pressed.

    Unchecked checkboxes are omitted and a select submits its selected option --
    the two details that make "press Save" change a value if the page rendered
    it wrong.
    """
    data: dict[str, str] = {}
    for control in form["controls"]:
        name = control.get("name")
        if not name or "disabled" in control:
            continue
        if control["tag"] == "select":
            chosen = [o for o in control["options"] if "selected" in o]
            options = chosen or control["options"]
            if options:
                data[name] = options[0].get("value", "")
            continue
        if control["kind"] in ("submit", "button"):
            continue
        if control["kind"] in ("checkbox", "radio"):
            if "checked" in control:
                data[name] = control.get("value", "on")
            continue
        data[name] = control.get("value", "")
    return data


def _buttons(page: _Page) -> list[tuple[dict, dict]]:
    """(form, submit button) for every submit control on the page."""
    return [
        (form, control)
        for form in page.forms
        for control in form["controls"]
        if control["kind"] == "submit"
    ]


def _find_button(page: _Page, action: str, value: str | None = None):
    for form, button in _buttons(page):
        if (button.get("formaction") or form["action"]) != action:
            continue
        if value is not None and button.get("value") != value:
            continue
        return form, button
    return None, None


def _press(
    client,
    page: _Page,
    action: str,
    value: str | None = None,
    overrides: dict | None = None,
    **kw,
):
    """Submit the page the way a browser does when that button is pressed.

    The payload is rebuilt from what the page actually rendered, so a field the
    template forgot to emit cannot be smuggled in by the test.
    """
    form, button = _find_button(page, action, value)
    assert form is not None, f"no submit button posting to {action} ({value})"
    assert form["method"] == "post", f"{action} is reached by {form['method']}"
    data = _submission(form)
    if button.get("name"):
        data[button["name"]] = button.get("value", "")
    data.update(overrides or {})
    return client.post(action, data=data, **kw)


def _page(client) -> _Page:
    response = client.get("/config")
    assert response.status_code == 200, response.status_code
    return _parse(response.text)


def _describes(page: _Page, field: str) -> str:
    """The text a screen reader reads out for this field's error.

    Following aria-describedby is what makes "field-adjacent" a property of the
    page rather than of where the string happens to sit in the markup.
    """
    control = _controls(page).get(field)
    assert control is not None, f"no control named {field}"
    return " ".join(
        page.text_by_id.get(ref, "")
        for ref in (control.get("aria-describedby") or "").split()
    )


def _describes_shown(page: _Page, element_id: str) -> str:
    """The same, for a value that is displayed rather than typed.

    The date and the uid are `<output>`s, so they are not controls -- but they
    still have to carry their own error, or a rejection about the session
    folder appears with nothing to say which half of it is wrong.
    """
    element = page.attrs_by_id.get(element_id)
    assert element is not None, f"nothing on the page has id={element_id!r}"
    return " ".join(
        page.text_by_id.get(ref, "")
        for ref in (element.get("aria-describedby") or "").split()
    )


def _generated(repo, preset="bimanual_v4") -> pathlib.Path:
    # Use the isolated station state directory for generated files, not repo.
    return settings.state_dir_from_environment() / "data_recording" / f"{preset}.yaml"


def _load(path) -> dict:
    return yaml.safe_load(pathlib.Path(path).read_text())


# ------------------------------------------------------------- direct assembly


def _session(**kw) -> SessionSpec:
    base = dict(
        preset_id="bimanual_v4",
        operator_name="Test Operator",
        task_name="plane_assembly_v9",
        session_uid="91e98c89fccb",
        data_dir="/data/raw/plane_assembly_v9/2026-08-04_x_91e98c89fccb",
    )
    base.update(kw)
    return SessionSpec(**base)


# Not the template's own UMI-V4-0001 / UMI-V4-0002: those are its placeholders,
# and a fixture that named them could never show a placeholder surviving.
_GRIPPERS = {"left": "UMI-V4-0003", "right": "UMI-V4-0004"}

# One synthetic device per source block -- no two blocks may name the same
# device. The wrist OAKs are one-stream boards and the trigger cameras are USB,
# named by their V4L2 index (never the template's placeholder 0).
_CAMERAS = {
    "right_umi_camera": "oak-FFC-A",
    "left_umi_camera": "oak-FFC-B",
    "right_trigger_camera": "2",
    "left_trigger_camera": "4",
    "right_tracking_camera": "iphone-R",
    "left_tracking_camera": "iphone-L",
}

# Build one form field per physical device and one config assignment per
# stream; V4 single-stream blocks make these maps coincide.
_CAMERA_FIELDS = dict(_CAMERAS)


def _devices(**kw) -> DeviceSpec:
    base = dict(grippers=dict(_GRIPPERS), required_cameras=dict(_CAMERAS), optional=())
    base.update(kw)
    return DeviceSpec(**base)


# ------------------------------------------------------- the task description


def test_the_assembled_config_never_carries_the_template_placeholder(tmp_path):
    """`task_description: Description` ships in the recording templates. It is
    truthy, so record_data.py skips its `input()` fallback and every
    episode's zarr attrs -- and every LeRobot annotation.human.task_description
    -- would become the literal string "Description"."""
    repo = _repo(tmp_path)
    out = ConfigAssembler(repo).assemble(
        _session(task_name="plane_assembly_v9"), _devices()
    )
    meta = _load(out)["meta"]
    assert meta["task_description"] != "Description"
    assert meta["task_description"] == "Assemble the plane"
    # The rig, not the registry: this fixture task is registered `v2.1` and the
    # bench is a V4. The majors disagree, so what recorded wins -- see
    # config_assembler.resolve_hw_version.
    assert meta["HW_version"] == "v4"


def test_an_unknown_task_is_described_as_unknown_not_placeholder(tmp_path):
    """The description is optional: an undescribed task is labelled "Unknown",
    never the template's "Description" placeholder."""
    repo = _repo(tmp_path)
    out = ConfigAssembler(repo).assemble(
        _session(task_name="not_a_real_task"), _devices()
    )
    meta = _load(out)["meta"]
    assert meta["task_description"] == "Unknown"
    # HW_version is unaffected: the description is a fact about the
    # task, which is unknown here, while the hardware revision is a fact about
    # the bench, which is known whether or not the task is registered.
    assert meta["HW_version"] == "v4"


def test_applying_the_form_writes_the_registry_description(tmp_path):
    """The half that matters: the page is the writer, so the lookup has to
    happen on the path an operator actually takes."""
    repo = _repo(tmp_path)
    client = _client(repo)
    _apply(client, task_name="plane_assembly_v9")
    meta = _load(_generated(repo))["meta"]
    assert meta["task_description"] == "Assemble the plane"
    assert meta["HW_version"] == "v4"


def test_the_task_card_leaves_out_the_hardware_version(tmp_path):
    """The station records only V4 grippers, so the card does not show the
    hardware revision; the file still records it."""
    repo = _repo(tmp_path)
    client = _client(repo)
    # The card only describes a known task, so the form is filled first -- which
    # also puts the written file beside it.
    _apply(client, task_name="plane_assembly_v9")

    page = client.get("/config").text
    assert "Recorded as “Assemble the plane”." in page
    assert "(HW " not in page
    assert _load(_generated(repo))["meta"]["HW_version"] == "v4"


# Task check overrides must reach the generated config without modifying
# the tracked baseline.


_BASELINES = ("data_check_config.yaml",)


def _range_max_default(repo) -> float:
    data = _load(repo / "config" / "data_check" / "data_check_config.yaml")
    return data["gripper"]["range_max_mm"]


def _baseline_bytes(repo) -> dict:
    return {
        name: (repo / "config" / "data_check" / name).read_bytes()
        for name in _BASELINES
    }


def _assert_baselines_untouched(repo, before) -> None:
    for name, blob in _baseline_bytes(repo).items():
        assert blob == before[name], (
            f"config/data_check/{name} is version-controlled and was rewritten"
        )


def test_the_shipped_baseline_is_not_either_task_s_override(tmp_path):
    """Guard the guard. If the fixture repo already carried a task's override,
    every override test below would pass vacuously."""
    assert _range_max_default(_repo(tmp_path)) == 90.0


def test_assemble_leaves_the_tracked_data_check_baselines_untouched(tmp_path):
    """Byte-identical, not merely equivalent.

    Stronger than "the other thresholds survived": a round-trip rewrite that
    reproduced every value but moved a comment or requoted a scalar still shows
    up in `git status` on the station, which is the whole point.
    """
    repo = _repo(tmp_path)
    before = _baseline_bytes(repo)
    # A task whose override differs from the file's, so a rewrite would be visible.
    assert _range_max_default(repo) != 85.0

    ConfigAssembler(repo).assemble(_session(task_name="a_30_second_task"), _devices())

    _assert_baselines_untouched(repo, before)


def test_applying_a_short_task_leaves_the_tracked_baselines_untouched(tmp_path):
    """The operator's own path: pressing Apply must not modify a tracked file in
    the checkout."""
    repo = _repo(tmp_path)
    before = _baseline_bytes(repo)
    client = _client(repo)

    _apply(client, task_name="a_30_second_task")

    _assert_baselines_untouched(repo, before)


def test_a_task_s_overrides_land_in_the_generated_file_instead(tmp_path):
    """The half that must not be lost with the write: 85 mm appears in no
    baseline in the tree, so finding it here means the registry's overrides
    still reach the file the recorder is handed."""
    repo = _repo(tmp_path)
    client = _client(repo)
    _apply(client, task_name="a_30_second_task")

    generated = settings.state_dir_from_environment() / "data_check"
    gripper = _load(generated / "a_30_second_task.yaml")["gripper"]
    assert gripper["range_max_mm"] == 85.0


def test_the_generated_file_keeps_the_rest_of_the_data_check_config(tmp_path):
    """The baseline carries every other threshold, and the generated file is
    what the checker reads -- so a generator that emitted the override alone
    would break data_check far worse than a dirty checkout."""
    repo = _repo(tmp_path)
    before = _load(repo / "config" / "data_check" / "data_check_config.yaml")
    client = _client(repo)
    _apply(client, task_name="a_30_second_task")

    after = _load(
        settings.state_dir_from_environment() / "data_check" / "a_30_second_task.yaml"
    )
    assert set(after) == set(before)
    assert after["tracking"] == before["tracking"]
    assert after["gripper"] == {**before["gripper"], "range_max_mm": 85.0}


# ------------------------------------------------ what the task name resolves to


def _task_says(client) -> str:
    """What the page tells the operator about the task name they typed."""
    return _describes(_page(client), "task_name")


def _typed(client, task_name):
    """Put a task name in the box the way an operator does, and reload."""
    page = _page(client)
    form = [f for f in page.forms if f["action"] == "/config/apply"][0]
    data = _submission(form)
    data["task_name"] = task_name
    client.post("/config/devices/adopt", data=data)
    return client


def test_a_registered_task_says_what_the_recording_will_be_labelled(tmp_path):
    """State (a). The description is what every episode's zarr attrs and every
    LeRobot annotation.human.task_description carry, so the operator has to be
    able to see that the name they typed picked the task they meant."""
    said = _task_says(_typed(_client(_repo(tmp_path)), "plane_assembly_v9"))
    assert "Assemble the plane" in said, said


def test_a_name_the_list_does_not_carry_is_offered_as_a_new_task(tmp_path):
    """State (b). Any task can be recorded: a name nobody registered is a new
    one that Apply saves, so this is an instruction -- describe it -- not a
    rejection."""
    page = _page(_typed(_client(_repo(tmp_path)), "not_a_real_task"))
    said = _describes(page, "task_name")
    assert "new task" in said.lower(), said
    assert "invalid" not in said.lower(), said
    assert "task_description" in _controls(page), "a new task asks for no description"


def test_a_task_list_that_will_not_load_is_not_reported_as_a_bad_name(
    tmp_path, monkeypatch
):
    """State (c), and the reason the four states are worth keeping apart.

    When the registry cannot be read the page knows nothing about the name --
    and calling it a new task would have the operator re-describe a task that
    is already on the list. The hint says what actually blocks them (no task
    list) and the remedy, not a judgement of the name.
    """
    client = _typed(_client(_repo(tmp_path)), "plane_assembly_v9")

    def unreachable():
        raise OSError("the task list cannot be read")

    monkeypatch.setattr(task_registry_store, "get_all_tasks", unreachable)
    said = _task_says(client)
    assert "new task" not in said.lower(), said
    assert "task list" in said.lower(), said
    assert "recording is disabled" in said.lower(), said

    # An empty list is not that outage: it is a station with no tasks yet, where
    # every name is a new one and recording is not disabled.
    monkeypatch.setattr(task_registry_store, "get_all_tasks", dict)
    said = _task_says(client)
    assert "new task" in said.lower(), said
    assert "recording is disabled" not in said.lower(), said


def test_an_empty_task_name_says_nothing_rather_than_judging_it(tmp_path):
    """State (d). Nothing has been typed, so nothing is right or wrong yet --
    and this line says exactly that by saying nothing.

    A prompt to type into the labelled, focused, empty box would tell the
    operator nothing they cannot see. The three other states each carry a fact
    the box does not; this one carries none.
    """
    said = _task_says(_client(_repo(tmp_path)))
    assert said.strip() == "", said
    assert "new task" not in said.lower(), said
    assert "task list" not in said.lower(), said


def test_the_four_task_states_never_read_the_same(tmp_path, monkeypatch):
    """No two of the four task states may share a message."""
    repo = _repo(tmp_path)
    said = {
        "empty": _task_says(_client(repo)),
        "known": _task_says(_typed(_client(repo), "plane_assembly_v9")),
        "new": _task_says(_typed(_client(repo), "not_a_real_task")),
    }
    client = _typed(_client(repo), "plane_assembly_v9")

    def unreachable():
        raise OSError("the task list cannot be read")

    monkeypatch.setattr(task_registry_store, "get_all_tasks", unreachable)
    said["unreachable"] = _task_says(client)

    assert len(set(said.values())) == 4, said


def _kind_of(identifier: str) -> str:
    """Which enumerator would have produced a record for *identifier*.

    An identifier is only unique within a kind -- a UVC index is short enough to
    collide with anything -- so each record carries the kind of the enumerator
    that would produce it. A fake gripper labelled `oak` would not match the
    gripper row it stands in for.
    """
    if identifier in set(_GRIPPERS.values()):
        return "gripper"
    if "iphone" in identifier:
        return "iphone"
    if identifier.isdigit():
        return "uvc"
    return "oak"


def _found(*identifiers, state=devices_core.AVAILABLE):
    """A scan result naming each identifier, in the shape the worker prints."""
    return [
        {
            "kind": _kind_of(name),
            "device_id": name,
            "registry_name": name,
            "state": state,
            "metadata": {},
        }
        for name in identifiers
    ]


_DECLARED = tuple(_CAMERA_FIELDS.values()) + tuple(_GRIPPERS.values())


# ---------------------------------------------------------- the partial form


def test_assemble_rejects_a_device_form_with_an_unassigned_camera(tmp_path):
    repo = _repo(tmp_path)
    with pytest.raises(ValueError, match="left_umi_camera"):
        ConfigAssembler(repo).assemble(_session(), DeviceSpec(grippers=dict(_GRIPPERS)))


def test_placeholder_serials_from_the_template_are_never_accepted(tmp_path):
    """An empty `DeviceSpec()` must not assemble: it would emit the template's
    placeholder devices as if they were real."""
    repo = _repo(tmp_path)
    with pytest.raises(ValueError):
        ConfigAssembler(repo).assemble(_session(), DeviceSpec())


def test_no_placeholder_from_the_template_survives_a_complete_form(tmp_path):
    repo = _repo(tmp_path)
    out = ConfigAssembler(repo).assemble(_session(), _devices())
    text = out.read_text()
    for placeholder in (
        "oak-left-wrist",
        "oak-right-wrist",
        "iphone-right-tracking",
        "iphone-left-tracking",
        "UMI-V4-0001",
        "UMI-V4-0002",
    ):
        assert placeholder not in text, placeholder
    # The USB trigger cameras' placeholder is an index, 0 on both arms -- too
    # common a string to search for, so it is read back from the sources.
    indices = {
        source["name"]: source["params"]["camera_index"]
        for source in _load(out)["video_sources"]
        if source.get("type") == "uvc_camera"
    }
    assert indices == {"left_trigger_camera": 4, "right_trigger_camera": 2}, indices


def test_an_omitted_camera_can_no_longer_let_one_board_back_two_blocks(tmp_path):
    """Omitting a camera would leave that block on its template placeholder,
    which _reject_shared_devices never sees -- so one physical board could end
    up behind two source blocks with nothing complaining."""
    repo = _repo(tmp_path)
    partial = dict(_CAMERAS)
    del partial["left_umi_camera"]
    # oak-left-wrist is the template's placeholder for the left wrist board.
    partial["right_umi_camera"] = "oak-left-wrist"
    with pytest.raises(ValueError, match="left_umi_camera"):
        ConfigAssembler(repo).assemble(_session(), _devices(required_cameras=partial))


def test_an_unassigned_gripper_is_rejected_too(tmp_path):
    repo = _repo(tmp_path)
    with pytest.raises(ValueError, match="left"):
        ConfigAssembler(repo).assemble(
            _session(), _devices(grippers={"right": _GRIPPERS["right"]})
        )


def test_one_gripper_serial_cannot_back_both_arms(tmp_path):
    """A UART gripper is worse than a camera here, not better.

    `_reject_shared_devices` covers `numeric_sources` as well as camera source
    blocks; otherwise the same serial could be assigned to both arms. Two
    `UartSensorReader`s on one `/dev/ttyACM*` do not fail loudly -- one arm
    simply records the other arm's encoder, and no data check downstream can
    tell.
    """
    repo = _repo(tmp_path)
    with pytest.raises(ValueError, match="UMI-V4-SAME"):
        ConfigAssembler(repo).assemble(
            _session(),
            _devices(grippers={"left": "UMI-V4-SAME", "right": "UMI-V4-SAME"}),
        )


def test_the_same_gripper_on_both_arms_is_refused_next_to_the_field(tmp_path):
    repo = _repo(tmp_path)
    client = _client(repo)
    response = _apply(
        client,
        grippers={"left": "UMI-V4-SAME", "right": "UMI-V4-SAME"},
        expect=400,
    )
    assert not _generated(repo).exists(), "a config with one board on two arms"
    for arm in ("left", "right"):
        described = _describes(_parse(response.text), f"gripper.{arm}")
        assert "UMI-V4-SAME" in described, (arm, described)


def test_apply_still_accepts_an_unregistered_serial(tmp_path):
    """A well-formed Pico board id the device bank has never heard of is saved
    -- with a warning, never a refusal: a real gripper must not be blocked for
    being newer than the registry."""
    repo = _repo(tmp_path)
    client = _client(repo)
    unknown_serial = "E6614C31000000B9"

    _apply(client, head=False, grippers={"left": unknown_serial}, expect=200)

    assert client.app.state.rig.draft.grippers["left"] == unknown_serial


def test_a_gripper_and_a_camera_may_not_name_one_device_either(tmp_path):
    """The claims are one namespace, so the check cannot be defeated by
    spreading a device across the two halves of the form."""
    repo = _repo(tmp_path)
    with pytest.raises(ValueError, match="oak-FFC-A"):
        ConfigAssembler(repo).assemble(
            _session(),
            _devices(grippers={"left": "oak-FFC-A", "right": _GRIPPERS["right"]}),
        )


def test_distinct_gripper_serials_are_still_accepted(tmp_path):
    """Guard the guard: a check that rejected every pair would pass the three
    tests above and break every real session."""
    repo = _repo(tmp_path)
    out = ConfigAssembler(repo).assemble(_session(), _devices())
    numeric = {n["name"]: n for n in _load(out)["numeric_sources"]}
    assert numeric["left_umi_gripper"]["uart_device_name"] == _GRIPPERS["left"]
    assert numeric["right_umi_gripper"]["uart_device_name"] == _GRIPPERS["right"]


def test_the_page_refuses_to_apply_until_every_camera_is_assigned(tmp_path):
    repo = _repo(tmp_path)
    client = _client(repo)
    response = _apply(client, cameras={"left_trigger_camera": ""}, expect=400)
    assert not _generated(repo).exists(), "a partial form wrote a config"
    # The message names the missing camera beside the box that fills it, or the
    # operator is told a camera is missing with no row to go and fix.
    described = _describes(_parse(response.text), "camera.left_trigger_camera")
    assert "left_trigger_camera" in described, described


# -------------------------------------------------------- session identity


def _folder(page: _Page) -> str:
    text = page.text_by_id.get("session-folder")
    assert text is not None, "the page renders no session folder preview"
    return " ".join(text.split())


def _shown(page: _Page, element_id: str) -> str:
    """The text the page displays under an id.

    Used for the two session fields that are not inputs. Reading the rendered
    text rather than a control's `value` is the whole point: it stays true
    whether the value is shown in an `<output>`, a `<span>` or anything else,
    and it is false the moment the page stops showing it at all.
    """
    text = page.text_by_id.get(element_id)
    assert text is not None, f"the page shows nothing under id={element_id!r}"
    return " ".join(text.split())


def _editable(page: _Page, name: str) -> bool:
    """Whether a browser would let the operator type this value.

    A control that submits but cannot be edited (`readonly`) still fails this,
    because the point is typing, not posting.
    """
    control = _controls(page).get(name)
    if control is None:
        return False
    return (
        control["tag"] in ("input", "select", "textarea") and "readonly" not in control
    )


def test_the_recording_date_and_uid_are_shown_but_cannot_be_typed(tmp_path):
    """Both are derived -- today's date, and a uid minted once -- so neither is
    an input. They stay on screen because both sit inside the session folder
    name, and a folder whose parts cannot be read is a folder nobody can check.

    Asserted three ways, because any one alone has a hole: no editable control,
    the values still visible (the thing that must not be lost), and the folder
    still built from exactly those two values.
    """
    client = _client(_repo(tmp_path))
    page = _page(client)

    for name in ("date", "session_uid"):
        assert not _editable(page, name), f"{name} is still typeable"

    uid = _shown(page, "session_uid")
    date = _shown(page, "date")
    assert is_valid_session_uid(uid), uid
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", date), date
    assert date == dt.date.today().isoformat(), date

    session = _folder(page).rstrip("/").split("/")[-1]
    assert session.startswith(date) and session.endswith(uid), session


def test_a_posted_date_or_uid_cannot_reach_the_draft(tmp_path):
    """The other half of "not an input": the server does not read them.

    A page that merely hid the inputs would still absorb a hand-posted
    `date=04/08/2026`, which is precisely the value `build_data_dir` refuses --
    so removing the control without removing the read would move the bug rather
    than fix it.
    """
    repo = _repo(tmp_path)
    client = _client(repo)
    before = _shown(_page(client), "session_uid")

    _apply(client, overrides={"date": "04/08/2026", "session_uid": "nope"})

    page = _page(client)
    assert _shown(page, "date") == dt.date.today().isoformat()
    assert _shown(page, "session_uid") == before
    assert _load(_generated(repo))["meta"]["session_uid"] == before


def test_a_fresh_config_page_has_a_valid_uid_and_the_folder_ends_with_it(tmp_path):
    client = _client(_repo(tmp_path))
    page = _page(client)
    uid = _shown(page, "session_uid")
    assert is_valid_session_uid(uid), uid
    assert _folder(page).endswith(uid), _folder(page)


def test_the_folder_preview_keeps_the_date_at_offset_zero(tmp_path):
    """The audit finds a day's sessions by the first ten characters of the
    folder, so the uid has to be a suffix and the date has to stay first."""
    page = _page(_client(_repo(tmp_path)))
    session = _folder(page).rstrip("/").split("/")[-1]
    assert re.match(r"^\d{4}-\d{2}-\d{2}_", session), session


def test_reopening_reuses_the_stored_uid_rather_than_minting_a_new_one(tmp_path):
    """The uid is the last segment of raw/{task}/{session}/, so re-minting it
    mid-session forks the episodes across two folders."""
    repo = _repo(tmp_path)
    _apply(_client(repo), task_name="plane_assembly_v9")
    saved = _load(_generated(repo))["meta"]["session_uid"]

    # A new app is a restarted server: nothing is carried over in memory.
    reopened = _page(_client(repo))
    assert _shown(reopened, "session_uid") == saved
    assert _folder(reopened).endswith(saved)


def test_regenerate_mints_a_new_uid_and_the_folder_preview_follows(tmp_path):
    """The Regenerate button is the operator's only handle on the uid, since the
    field is not typeable, so it has to keep working."""
    client = _client(_repo(tmp_path))
    before = _shown(_page(client), "session_uid")
    _press(client, _page(client), "/config/session/uid")

    page = _page(client)
    after = _shown(page, "session_uid")
    assert after != before
    assert is_valid_session_uid(after)
    assert _folder(page).endswith(after)


def test_regenerate_is_refused_with_a_confirm_once_episodes_exist(
    monkeypatch, tmp_path
):
    """Renaming a folder that already holds episodes forks them across two
    folders, so the second uid is only minted after an explicit confirmation."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    client = _client(_repo(tmp_path))
    page = _page(client)
    before = _shown(page, "session_uid")
    recorded = pathlib.Path(os.path.expanduser(_folder(page))) / "episode_0"
    recorded.mkdir(parents=True)

    refused = _press(client, page, "/config/session/uid")
    assert _shown(_page(client), "session_uid") == before

    # Refusing without an escape hatch would brick the page, so the response
    # carries the confirmed form and pressing it goes through.
    confirmed = _press(client, _parse(refused.text), "/config/session/uid")
    assert confirmed.status_code in (200, 303)
    assert _shown(_page(client), "session_uid") != before


def test_starting_the_next_session_seals_the_one_left_behind(monkeypatch, tmp_path):
    """The operator pressing this button is the clearest statement anywhere in
    the system that a session is finished, so the press seals it rather than
    leaving the folder to simply stop growing. Sealed, `session_slot` also
    refuses to reopen it offline.

    Asserted on the old folder, which is the ordering the handler has to get
    right -- `data_dir` is derived from the draft, so a capture taken after
    `start_new_session` would name the empty folder this press just opened.
    """
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    client = _client(_repo(tmp_path))
    page = _page(client)
    left = _shown(page, "session_uid")
    folder = pathlib.Path(os.path.expanduser(_folder(page)))
    (folder / "episode_0").mkdir(parents=True)

    refused = _press(client, page, "/config/session/uid")
    _press(client, _parse(refused.text), "/config/session/uid")

    marker = session_seal.read_seal(folder)
    assert marker is not None, "the session the operator left was not sealed"
    assert marker["reason"] == "next_session"
    assert marker["session_uid"] == left
    assert marker["episode_count"] == 1
    # Assert exactly one seal under the output root, on the departed folder.
    # Checking a nonexistent new folder alone could pass without any seal.
    markers = sorted(
        path.parent.name for path in output_root().rglob(session_seal.SEAL_FILENAME)
    )
    assert markers == [folder.name], markers


def test_editing_the_task_name_seals_the_folder_it_leaves(monkeypatch, tmp_path):
    """The commonest way a session ends without anybody deciding to.

    `build_data_dir` composes the folder from the task name as well as the uid,
    and only the uid is re-minted by a roll -- so editing the task moves the
    console to a new folder while keeping the same session id. Nothing rolled,
    so none of the roll-driven triggers fire, and without a seal the old folder
    would simply stop growing with no record of why.

    Driven through the real Apply: the helper unit tests in
    `test_seal_triggers.py` would pass just as well against a handler that
    never called them.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    client = _client(_repo(tmp_path))
    _apply(client, task_name="plane_assembly_v9")
    left = pathlib.Path(os.path.expanduser(_folder(_page(client))))
    (left / "episode_0").mkdir(parents=True)

    _apply(client, task_name="a_30_second_task")

    moved = pathlib.Path(os.path.expanduser(_folder(_page(client))))
    assert moved != left, "the task name did not move the session folder"
    marker = session_seal.read_seal(left)
    assert marker is not None, "the orphaned folder was not sealed"
    assert marker["reason"] == "dir_change"
    assert marker["episode_count"] == 1
    # Search the whole output root for the seal because changing task names
    # moves the destination between task directories.
    markers = sorted(
        path.parent.name for path in output_root().rglob(session_seal.SEAL_FILENAME)
    )
    assert markers == [left.name], markers


def test_an_apply_that_leaves_the_folder_alone_seals_nothing(monkeypatch, tmp_path):
    """The normal press, between two episodes of one session. A seal here would
    close a session that is still being recorded, and `slot_is_usable` would
    then refuse the very folder the console is still armed for."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    client = _client(_repo(tmp_path))
    _apply(client)
    folder = pathlib.Path(os.path.expanduser(_folder(_page(client))))
    (folder / "episode_0").mkdir(parents=True)

    _apply(client)

    markers = [
        path.parent.name for path in folder.parent.rglob(session_seal.SEAL_FILENAME)
    ]
    assert markers == [], markers


def test_the_operator_is_told_which_session_the_apply_closed(monkeypatch, tmp_path):
    """A seal is permanent, and a mistyped task name is enough to cause one.

    `slot_is_usable` refuses a sealed folder outright, and `write_seal` never
    overwrites a marker -- so correcting the name and pressing Apply again does
    not reopen the session. "Configuration saved." over that is the console
    ending somebody's shift's session without saying so.

    Read off the response the Apply redirect lands on, not a later GET: the
    notice is shown once and cleared (`config_page`), so a second request would
    assert against a blank banner and pass on any implementation.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    client = _client(_repo(tmp_path))
    _apply(client, task_name="plane_assembly_v9")
    left = pathlib.Path(os.path.expanduser(_folder(_page(client))))
    (left / "episode_0").mkdir(parents=True)

    body = _apply(client, task_name="a_30_second_task").text

    assert left.name in body, "the notice does not name the session it closed"
    assert "closed" in body
    # The one thing they cannot infer from "closed", and the reason the
    # new-session confirmation says it too: their episodes are not going anywhere.
    assert "stay where they are" in body


def test_an_apply_that_seals_nothing_still_just_says_it_saved(monkeypatch, tmp_path):
    """The page is kept short, so the closure sentence is only shown when there
    is a closure."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    client = _client(_repo(tmp_path))
    _apply(client)

    body = _apply(client).text

    assert "Configuration saved." in body
    assert "closed" not in body


# Assert concise text within its owning page block, not merely anywhere in the document.


# Every sentence on this page that narrated the page instead of labelling a
# value. Field-level help that tells the operator what to type is deliberately
# not in this list.
_LECTURES = (
    "One box per camera or gripper you plug in",
    "The recording date and the session ID are filled in",
    "first in the folder name",
    "always last in the folder name",
    "one box per device you plug in",
)


def test_the_page_labels_its_values_instead_of_explaining_them(tmp_path):
    page = _page(_client(_repo(tmp_path)))
    body = " ".join(page.body.split())
    for lecture in _LECTURES:
        assert lecture not in body, lecture


def test_dropping_the_prose_keeps_the_two_values_it_narrated(tmp_path):
    """The date and the session id stay on the page; only the sentences about
    them are dropped."""
    page = _page(_client(_repo(tmp_path)))
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", _shown(page, "date")), _shown(page, "date")
    assert is_valid_session_uid(_shown(page, "session_uid"))


def test_the_save_location_is_labelled_short_and_still_shows_the_path(tmp_path):
    page = _page(_client(_repo(tmp_path)))
    assert "Where this session's episodes are saved" not in " ".join(page.body.split())
    assert " ".join(page.text_by_id.get("folder-label", "").split()) == "Session path"
    assert _folder(page).endswith(_shown(page, "session_uid"))


def test_the_new_session_control_says_what_it_does(tmp_path):
    page = _page(_client(_repo(tmp_path)))
    _, button = _find_button(page, "/config/session/uid")
    assert button is not None, "the page offers no way to start a new session"
    assert " ".join(button["text"].split()) == "Start a new session"


def test_a_new_session_is_stamped_today_not_the_day_the_page_was_opened(
    tmp_path, monkeypatch
):
    """The recording date must not follow the operator into the new session. A
    station is left open overnight, and the audit finds a day's sessions by the
    first ten characters of the session folder -- so a session started at
    00:20 under yesterday's date is filed against the wrong day.

    `day_roll` is stubbed out because it handles that overnight case before
    any handler runs (`test_day_roll.py`), which would make this pass without
    the button being pressed -- and would stop the page ever showing a stale
    date to set the test up with. What is asserted here is the button: pressed
    on a session from another day, it stamps today.
    """
    monkeypatch.setattr(day_roll, "roll_if_new_day", lambda app: None)
    client = _client(_repo(tmp_path))
    client.app.state.rig.draft.date = "2020-01-01"
    assert _shown(_page(client), "date") == "2020-01-01"

    _press(client, _page(client), "/config/session/uid")

    page = _page(client)
    today = dt.date.today().isoformat()
    assert _shown(page, "date") == today
    assert _folder(page).rstrip("/").split("/")[-1].startswith(today)


def test_starting_a_new_session_disarms_the_config_the_old_one_wrote(tmp_path):
    """The recorder records from the file on disk, not from this page. Left
    behind, it sends the new session's first episode into the old session's
    folder while the page shows a different id -- so the new session has to be
    applied before it can be recorded."""
    repo = _repo(tmp_path)
    client = _client(repo)
    _apply(client)
    stale = _load(_generated(repo))["meta"]["session_uid"]

    _press(client, _page(client), "/config/session/uid")
    assert not _generated(repo).exists(), "the old session's config is still armed"

    _apply(client)
    written = _load(_generated(repo))["meta"]["session_uid"]
    assert written != stale
    assert written == _shown(_page(client), "session_uid")


def test_starting_a_new_session_keeps_the_rig_and_the_operator(tmp_path):
    """The other half of the decision: task, operator and every device belong to
    the rig, so re-typing them for each session would be the actual cost."""
    repo = _repo(tmp_path)
    client = _client(repo)
    _apply(client)
    _press(client, _page(client), "/config/session/uid")
    _rig_is_intact(_page(client))


def test_a_non_iso_date_is_a_field_adjacent_400_not_a_500(tmp_path):
    """`build_data_dir` still raises on a non-ISO date, so this path still has
    to answer 400.

    Nothing an operator can do produces one -- the field is not typed and
    `absorb` does not read it -- so the bad value is put on the draft directly,
    which is the only remaining way in (a long-running server, or a later
    caller that sets it). Deleting the guard because "it cannot happen" would
    turn that into a 500 on a page with no way back.
    """
    repo = _repo(tmp_path)
    client = _client(repo)
    client.app.state.rig.draft.date = "04/08/2026"

    response = _apply(client, expect=400)
    assert "04/08/2026" in _describes_shown(_parse(response.text), "date")
    assert not _generated(repo).exists()


def test_a_malformed_uid_is_a_field_adjacent_400_not_a_500(tmp_path):
    """Same guard, the other field. `session_identity.new_session_uid` cannot
    mint a bad one, but `hydrate` reads a uid out of a config on disk and a
    hand-edited file is a real thing on a station."""
    repo = _repo(tmp_path)
    client = _client(repo)
    client.app.state.rig.draft.session_uid = "nope"

    response = _apply(client, expect=400)
    # Session-path errors should identify the editable control, not instruct
    # the operator to edit a derived output field.
    said = _describes_shown(_parse(response.text), "session_uid")
    assert said, "a malformed uid says nothing beside the field"
    assert "invalid" in said.lower(), said
    assert "Start a new session" in said, f"no way out is offered: {said!r}"
    assert not _generated(repo).exists()


# --------------------------------------------------- the recorder's uid guard


def test_the_applied_config_arms_the_recorder_uid_guard(tmp_path):
    """RecorderManager._validate_session_uid only rejects a missing uid for
    configs it can identify as console-assembled, via meta.setup_preset, so the
    assembler has to write that key or the guard never fires on a real file."""
    repo = _repo(tmp_path)
    _apply(_client(repo))
    cfg = _load(_generated(repo))
    assert cfg["meta"]["setup_preset"] == "bimanual_v4"

    class _Recorder:
        pass

    passes = _Recorder()
    passes.config = cfg
    RecorderManager._validate_session_uid(passes)

    stripped = _Recorder()
    stripped.config = {
        **cfg,
        "meta": {k: v for k, v in cfg["meta"].items() if k != "session_uid"},
    }
    with pytest.raises(ValueError, match="session_uid"):
        RecorderManager._validate_session_uid(stripped)


# ---------------------------------------------------------- optional devices


def _rows(page: _Page, section: str) -> list[str]:
    """Logical names of the device rows rendered in a section, in order."""
    return [
        name.split(".")[1]
        for name in _controls(page)
        if name.startswith("optional.")
        and name.endswith(".device")
        and page.attrs_by_id.get(f"row-{name.split('.')[1]}", {}).get("data-section")
        == section
    ]


def _add(client, page, type_id):
    return _press(client, page, "/config/optional/add", type_id)


def _remove(client, page, logical_name):
    return _press(client, page, "/config/optional/remove", logical_name)


# The head is an Add/Remove list, the same one the bottom camera is: one idiom
# for both optional sections.


def _head_rows(page: _Page) -> set[str]:
    """The head cameras currently on the page, by logical name."""
    return {
        name.split(".")[1]
        for name in _controls(page)
        if name.startswith("optional.head_") and name.endswith(".device")
    }


def _add_buttons(page: _Page, action: str) -> dict[str, dict]:
    """Every Add button on the page, keyed by the type it adds."""
    return {
        control.get("value", ""): control
        for form in page.forms
        for control in form["controls"]
        if control.get("formaction") == action and control.get("name") == "type_id"
    }


def test_the_head_offers_add_buttons_like_the_bottom_camera(tmp_path):
    """Same idiom, same route, same shape as `bottom_section.html`."""
    page = _page(_client(_repo(tmp_path)))
    buttons = _add_buttons(page, "/config/optional/add")

    assert "head_oak" in buttons, "no Add for the head OAK"
    assert "head_iphone" in buttons, "no Add for the head iPhone"


def test_the_head_oak_add_button_comes_before_the_iphone_one(tmp_path):
    """The OAK leads the head list.

    Declared first, the iPhone would lead every head list and read as the
    default even when the OAK was the one present.
    """
    body = _client(_repo(tmp_path)).get("/config").text

    assert body.index('value="head_oak"') < body.index('value="head_iphone"')


def test_a_fresh_session_has_no_head_camera(tmp_path):
    """A fresh session has no head camera row: the operator adds one if the rig
    has it, and a phone nobody connected must not appear by default."""
    page = _page(_client(_repo(tmp_path), head_oak=False))

    assert _head_rows(page) == set()


def test_a_head_camera_already_added_cannot_be_added_twice(tmp_path):
    """`max_count=1` per head type, shown as a disabled button rather than a
    missing one -- so the operator can see the head already has an OAK instead
    of wondering where the option went."""
    page = _page(_client(_repo(tmp_path)))
    buttons = _add_buttons(page, "/config/optional/add")

    assert "disabled" in buttons["head_oak"]
    assert "disabled" not in buttons["head_iphone"]


def test_adding_and_removing_a_head_camera_moves_its_assignment(tmp_path):
    client = _client(_repo(tmp_path))
    draft = client.app.state.rig.draft

    client.post(
        "/config/optional/add", data={"config_form": "1", "type_id": "head_iphone"}
    )
    assert {a.type_id for a in draft.optional} == {"head_oak", "head_iphone"}

    client.post(
        "/config/optional/remove?confirm=yes",
        data={"config_form": "1", "logical_name": "head_iphone_camera"},
    )
    assert {a.type_id for a in draft.optional} == {"head_oak"}


def test_the_bottom_camera_is_offered_like_the_head_and_starts_absent(tmp_path):
    """A static OAK on the table with a view from below: its own section, same
    Add/Remove idiom. Unlike the head OAK it is the exception rather than what
    every bench carries, so a fresh session starts without it."""
    page = _page(_client(_repo(tmp_path)))
    buttons = _add_buttons(page, "/config/optional/add")

    assert "bottom_oak" in buttons, "no Add for the bottom OAK"
    assert "disabled" not in buttons["bottom_oak"]
    assert _rows(page, "bottom") == []


def test_adding_and_removing_a_bottom_camera_moves_its_assignment(tmp_path):
    client = _client(_repo(tmp_path))
    draft = client.app.state.rig.draft

    _add(client, _page(client), "bottom_oak")
    assert _rows(_page(client), "bottom") == ["bottom_oak_camera"]
    assert {a.type_id for a in draft.optional} == {"head_oak", "bottom_oak"}
    # One per rig: the button stays, disabled, so the operator sees why.
    _, add = _find_button(_page(client), "/config/optional/add", "bottom_oak")
    assert "disabled" in add

    _remove(client, _page(client), "bottom_oak_camera")
    assert _rows(_page(client), "bottom") == []
    assert {a.type_id for a in draft.optional} == {"head_oak"}


def test_an_added_bottom_camera_is_written_and_reopens(tmp_path):
    """Apply writes it as one more `oak_camera` block under its fixed name, with
    its placement; the page then reads it back from the generated file."""
    repo = _repo(tmp_path)
    client = _client(repo)
    _add(client, _page(client), "bottom_oak")

    _apply(client, overrides={"optional.bottom_oak_camera.placement": "table"})

    sources = {
        s["name"]: s for s in _load(_generated(repo))["video_sources"] if s.get("name")
    }
    assert sources["bottom_oak_camera"]["type"] == "oak_camera"
    assert sources["bottom_oak_camera"]["placement"] == "table"
    assert _rows(_page(client), "bottom") == ["bottom_oak_camera"]


def test_a_head_may_carry_an_oak_and_an_iphone_at_once(tmp_path):
    """One iPhone and one OAK together is a rig that exists; the cap is one
    each. Collapsing the head to a single choice would delete that."""
    client = _client(_repo(tmp_path))
    client.post(
        "/config/optional/add", data={"config_form": "1", "type_id": "head_iphone"}
    )

    _apply(
        client,
        overrides={"optional.head_iphone_camera.device": "iphone15p-04"},
    )

    assert sorted(a.logical_name for a in client.app.state.rig.draft.optional) == [
        "head_iphone_camera",
        "head_oak_camera",
    ]


def test_a_reopened_head_camera_comes_back_as_a_row_naming_its_device(tmp_path):
    """A reopened head row names its device and can be removed.

    `_reopen` hydrates the last generated config, so one carrying a head iPhone
    puts a row on screen -- that is what reopen is for -- but it names the
    device it was saved with, and Remove is one click. A row appearing with
    somebody else's phone already in it and no way to tell why would be a
    mystery; the name and the Remove button answer both.
    """
    repo = _repo(tmp_path)
    client = _client(repo)
    client.post(
        "/config/optional/add", data={"config_form": "1", "type_id": "head_iphone"}
    )
    _apply(client, overrides={"optional.head_iphone_camera.device": "iphone16p-ry"})

    reopened = _page(_client(repo))

    assert "head_iphone_camera" in _head_rows(reopened)
    assert (
        _controls(reopened)["optional.head_iphone_camera.device"]["value"]
        == "iphone16p-ry"
    )


def test_a_head_camera_with_no_device_is_refused_by_name(tmp_path):
    """The OAK row is present by default, so this is the first thing a rig with
    no head camera meets. It has to name the row and both ways out of it: fill
    the box, or take the row off the page."""
    client = _client(_repo(tmp_path))
    page = _page(client)
    data = _submission([f for f in page.forms if f["action"] == "/config/apply"][0])
    data.update(
        {
            "task_name": "plane_assembly_v9",
            "operator_name": "Test Operator",
            **{f"gripper.{a}": d for a, d in _GRIPPERS.items()},
            **{f"camera.{n}": d for n, d in _CAMERA_FIELDS.items()},
        }
    )
    data["optional.head_oak_camera.device"] = ""

    body = client.post("/config/apply", data=data).text

    assert "Head OAK" in body
    assert "no device is chosen" in body
    assert "remove Head OAK" in body


def test_removing_a_camera_that_names_hardware_asks_first(tmp_path):
    """The destructive-action confirm, and it is a server round-trip rather than
    a JavaScript confirm() so it still guards with the script blocked."""
    client = _client(_repo(tmp_path))
    _press(
        client,
        _page(client),
        "/config/optional/add",
        "bottom_oak",
        overrides={"optional.head_oak_camera.device": "oak-HEAD"},
    )

    asked = _remove(client, _page(client), "head_oak_camera")
    assert _rows(_page(client), "head") == ["head_oak_camera"]

    # Pressing Remove a second time is not agreement: it asks again. Only the
    # control that says so removes the camera.
    _remove(client, _parse(asked.text), "head_oak_camera")
    assert _rows(_page(client), "head") == ["head_oak_camera"]

    _press(
        client,
        _parse(asked.text),
        "/config/optional/remove?confirm=yes",
        "head_oak_camera",
    )
    assert _rows(_page(client), "head") == []
    assert _rows(_page(client), "bottom") == ["bottom_oak_camera"]


def _confirm_payload(page: _Page, action: str, value: str | None = None) -> dict:
    """Only the confirm control's own two fields.

    This is what a link, a retry or a hand-rolled client sends. Building the
    payload from what the server rendered -- rather than from the full page
    form -- is the whole point: a fixture that posts every field can never reach
    the code that has to tell an absent key from a blank one.
    """
    form, button = _find_button(page, action, value)
    assert form is not None, f"no confirm control posting to {action}"
    payload = _submission(form)
    if button.get("name"):
        payload[button["name"]] = button.get("value", "")
    return payload


def _rig_is_intact(page: _Page) -> None:
    """Every device and session field the operator had configured is still set."""
    controls = _controls(page)
    for name, device in _CAMERA_FIELDS.items():
        assert controls[camera_field := f"camera.{name}"]["value"] == device, (
            f"{camera_field} was cleared by a short post"
        )
    for arm, device in _GRIPPERS.items():
        assert controls[f"gripper.{arm}"]["value"] == device, arm
    assert controls["task_name"]["value"] == "plane_assembly_v9"
    assert controls["operator_name"]["value"] == "Test Operator"


def test_confirming_a_removal_must_not_wipe_the_rest_of_the_rig(tmp_path):
    """A short post leaves untouched slots alone; only a blank one clears.

    The confirmation sits inside the page's one form (nested forms are
    illegal), so a browser sends the whole form with it. The endpoint still has
    to be safe against the two fields on their own -- that is what a link, a
    retry or a hand-rolled client sends. If `absorb` stopped distinguishing an
    absent key from a present-and-blank one, this post would silently empty
    every camera and gripper assignment on the page, one request after agreeing
    to remove a single camera.
    """
    repo = _repo(tmp_path)
    client = _client(repo)
    _apply(client)
    # Two optional cameras, so the post has a surviving optional row to damage
    # as well as the required slots -- the optional rows are absorbed through a
    # different branch and would otherwise go unwatched.
    _add(client, _page(client), "bottom_oak")
    _apply(
        client,
        overrides={
            "optional.head_oak_camera.device": "oak-HEAD",
            "optional.head_oak_camera.placement": "neck",
            "optional.bottom_oak_camera.device": "oak-BOTTOM",
            "optional.bottom_oak_camera.placement": "table",
        },
    )

    # Asked first, so this is the same second request the confirm control makes.
    _remove(client, _page(client), "head_oak_camera")
    client.post(
        "/config/optional/remove?confirm=yes", data={"logical_name": "head_oak_camera"}
    )

    page = _page(client)
    assert _rows(page, "head") == [], "the wrong rows survived"
    assert _rows(page, "bottom") == ["bottom_oak_camera"], "the wrong rows survived"
    _rig_is_intact(page)
    controls = _controls(page)
    assert controls["optional.bottom_oak_camera.device"]["value"] == "oak-BOTTOM"
    assert [
        o.get("value")
        for o in controls["optional.bottom_oak_camera.placement"]["options"]
        if "selected" in o
    ] == ["table"], "the surviving camera's placement was cleared"


def test_confirming_a_uid_regeneration_must_not_wipe_the_rig(monkeypatch, tmp_path):
    """The same short form, the other destructive confirm."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    repo = _repo(tmp_path)
    client = _client(repo)
    _apply(client)
    page = _page(client)
    before = _shown(page, "session_uid")
    pathlib.Path(os.path.expanduser(_folder(page)) + "/episode_0").mkdir(parents=True)

    asked = _press(client, _page(client), "/config/session/uid")
    payload = _confirm_payload(_parse(asked.text), "/config/session/uid")
    assert set(payload) == {"confirm"}, payload

    client.post("/config/session/uid", data=payload)

    page = _page(client)
    assert _shown(page, "session_uid") != before, "the uid was not minted"
    _rig_is_intact(page)


def test_a_blank_field_still_clears_its_slot(tmp_path):
    """The other half of the rule, so "absent leaves it alone" cannot be
    implemented by ignoring the field altogether."""
    repo = _repo(tmp_path)
    client = _client(repo)
    _apply(client)
    _apply(client, cameras={"left_trigger_camera": ""}, expect=400)
    controls = _controls(_page(client))
    assert controls["camera.left_trigger_camera"]["value"] == ""


def test_absorb_drops_a_slot_the_new_template_no_longer_declares():
    """A preset switch changes the template. A slot the new one does not
    declare would make `assemble` raise "camera X is not declared by this
    template" against a field the page no longer renders -- a 400 the operator
    has no way to clear."""
    draft = SessionDraft(preset_id="bimanual_v4")
    draft.required_cameras["dropped_camera"] = "oak-OLD"
    draft.grippers["middle"] = "UMI-OLD"
    requirements = TemplateRequirements(
        cameras=(RequiredCamera("kept_camera", "oak_device", "oak"),),
        gripper_arms=("right",),
    )

    absorb(
        draft,
        {"camera.kept_camera": "oak-NEW", "gripper.right": "UMI-NEW"},
        requirements,
    )

    assert draft.required_cameras == {"kept_camera": "oak-NEW"}
    assert draft.grippers == {"right": "UMI-NEW"}


def test_removing_a_row_that_names_no_hardware_does_not_ask(tmp_path):
    """A row added by mistake has nothing to lose, and a confirm on it trains
    the operator to click through the one that matters."""
    client = _client(_repo(tmp_path))
    _add(client, _page(client), "bottom_oak")
    _remove(client, _page(client), "bottom_oak_camera")
    assert _rows(_page(client), "bottom") == []


def test_the_page_offers_only_the_head_and_bottom_cameras(tmp_path):
    """Every optional camera an operator can add, read off the rendered Add
    buttons rather than the descriptor table."""
    client = _client(_repo(tmp_path))
    offered = {
        button.get("value")
        for form, button in _buttons(_page(client))
        if (button.get("formaction") or form["action"]) == "/config/optional/add"
    }
    assert offered == {"head_oak", "head_iphone", "bottom_oak"}


def test_placement_round_trips_and_never_changes_a_logical_name(tmp_path):
    repo = _repo(tmp_path)
    client = _client(repo)
    _add(client, _page(client), "bottom_oak")

    _apply(
        client,
        overrides={
            "optional.head_oak_camera.device": "oak-HEAD",
            "optional.head_oak_camera.placement": "neck",
            "optional.bottom_oak_camera.device": "oak-BOTTOM",
            "optional.bottom_oak_camera.placement": "table",
        },
    )

    sources = {
        s["name"]: s for s in _load(_generated(repo))["video_sources"] if s.get("name")
    }
    assert sources["head_oak_camera"]["placement"] == "neck"
    assert sources["bottom_oak_camera"]["placement"] == "table"

    page = _page(client)
    assert _rows(page, "head") == ["head_oak_camera"]
    assert _rows(page, "bottom") == ["bottom_oak_camera"]
    placement = _controls(page)["optional.head_oak_camera.placement"]
    assert [o.get("value") for o in placement["options"] if "selected" in o] == ["neck"]


# ------------------------------------------ the name an optional camera carries


def test_the_page_never_shows_an_optional_camera_s_config_name(tmp_path):
    """An optional camera is keyed on its config name (`bottom_oak_camera`).
    That name carries underscores and is engineer-facing detail; the page does
    not show it, nor a serves line."""
    client = _client(_repo(tmp_path))
    _add(client, _page(client), "bottom_oak")
    _apply(
        client,
        overrides={
            "optional.bottom_oak_camera.device": "oak-BOTTOM",
            "optional.bottom_oak_camera.placement": "table",
        },
    )
    body = _page(client).body

    assert "bottom_oak_camera" not in body
    assert "serves" not in body


def test_an_added_camera_keeps_the_values_already_typed_into_the_form(tmp_path):
    """Adding a row is a POST of the whole form, so the operator's half-filled
    session fields have to survive it -- otherwise adding a camera silently
    resets the page."""
    client = _client(_repo(tmp_path))
    page = _page(client)
    _press(
        client,
        page,
        "/config/optional/add",
        "bottom_oak",
        overrides={"operator_name": "Test Operator", "task_name": "a_30_second_task"},
    )
    reloaded = _controls(_page(client))
    assert reloaded["operator_name"]["value"] == "Test Operator"
    assert reloaded["task_name"]["value"] == "a_30_second_task"


# ------------------------------------------------------- warnings and errors


def test_apply_surfaces_the_resolution_warnings(tmp_path):
    """resolution_for explains why a stream got the 16:10 default instead of its
    module's native size. Swallowed, a 4:3 IMX378 records at 640x400 and loses
    17% of its FOV with nothing on screen to say so.

    The operator keeps the part they can act on -- which camera, and what to do
    -- because a warning they cannot read is a warning they cannot act on.
    """
    repo = _repo(tmp_path)
    client = _client(repo)
    response = _apply(
        client,
        cameras={"left_umi_camera": "oak-FFC-ZZ"},
    )
    body = _parse(response.text).body
    assert "oak-FFC-ZZ" in body
    assert "Refresh devices" in body, body[:400]
    # Match the resolver warning's structure rather than incidental numbers
    # that may also occur in device IDs.
    assert "resolution unverified" not in body
    assert "[640, 400]" not in body


def test_a_duplicate_device_is_shown_next_to_the_field_not_as_a_500(tmp_path):
    """One physical device behind two source blocks fails at record time in
    three different ways, one of them silently."""
    repo = _repo(tmp_path)
    client = _client(repo)
    response = _apply(
        client,
        cameras={"left_trigger_camera": "iphone-R"},
        expect=400,
    )
    described = _describes(_parse(response.text), "camera.left_trigger_camera")
    assert "iphone-R" in described, described
    assert not _generated(repo).exists()


def test_an_optional_camera_with_no_device_is_rejected_at_its_own_row(tmp_path):
    """Adding a row and never filling it is the commonest half-finished form."""
    repo = _repo(tmp_path)
    client = _client(repo)
    _add(client, _page(client), "bottom_oak")
    response = _apply(
        client, overrides={"optional.bottom_oak_camera.device": ""}, expect=400
    )
    described = _describes(_parse(response.text), "optional.bottom_oak_camera.device")
    assert "bottom_oak_camera" in described, described
    assert not _generated(repo).exists()


def test_a_rejected_apply_keeps_the_previously_generated_config(tmp_path):
    repo = _repo(tmp_path)
    client = _client(repo)
    _apply(client)
    good = _generated(repo).read_bytes()
    _apply(client, cameras={"left_trigger_camera": "iphone-R"}, expect=400)
    assert _generated(repo).read_bytes() == good


def test_the_generated_config_never_reaches_an_operator_s_browser(tmp_path):
    """Not hidden -- absent. A `display:none` block is still in view-source and
    still ships every path, serial and resolution in the file to a page whose
    reader is a data collector, so there is nothing to reveal with a stylesheet
    off or a devtools inspector open.
    """
    repo = _repo(tmp_path)
    client = _client(repo)
    _apply(client)
    written = _generated(repo).read_text()

    body = _client(repo).get("/config").text
    assert written not in body
    for engineer_facing in (
        "Generated config",
        "schema_version",
        "video_sources",
        "bimanual_v4.yaml",
        "config/data_recording",
    ):
        assert engineer_facing not in body, engineer_facing


# ------------------------------------------------- progressive enhancement


def test_every_mutating_control_is_a_real_form_a_browser_can_post(tmp_path):
    """With the vendored htmx blocked, the page still has to work. Replayed from
    what was rendered, never from hardcoded values."""
    repo = _repo(tmp_path)
    client = _client(repo)
    page = _page(client)
    submits = _buttons(page)
    assert submits, "the page renders no submit controls at all"
    for form, button in submits:
        assert form["method"] == "post", f"{form['action']} is not a POST"
        action = button.get("formaction") or form["action"]
        assert action, "a submit control with no target is not a real control"
        assert action.startswith("/"), action

    # Not merely well-formed: actually postable, with no script involved.
    assert _add(client, _page(client), "bottom_oak").status_code in (200, 303)
    assert _remove(client, _page(client), "bottom_oak_camera").status_code in (
        200,
        303,
    )
    assert _press(client, _page(client), "/config/session/uid").status_code in (
        200,
        303,
    )
    assert _apply(client).status_code == 200


def test_no_control_on_the_page_depends_on_an_inline_event_handler(tmp_path):
    """No control may depend on `onclick`: with JavaScript off it would
    configure nothing."""
    page = _page(_client(_repo(tmp_path)))
    assert page.inline_handlers == [], sorted(set(page.inline_handlers))


# ---------------------------------------------------------- guards and rights


def test_the_page_explains_itself_when_no_preset_has_been_chosen(tmp_path):
    """Tab gating survives as a route guard. A guarded route explains itself
    rather than 404ing, and offers no half-usable form to fill in."""
    client = TestClient(create_app(Settings(repo_root=_repo(tmp_path))))
    response = client.get("/config")
    assert response.status_code == 200, "a guarded route must not 404"
    page = _parse(response.text)
    # "preset" is the codebase's word for it; the operator's is "setup", which
    # is what the Setup tab and its cards are called.
    assert "no setup chosen" in page.body.lower(), page.body[:400]
    assert 'href="/"' in response.text, "no way back to Setup"
    assert not [f for f in page.forms if f["action"] == "/config/apply"]
    assert not _controls(page), "a form with no preset behind it is offered"

    # And the guard is not merely cosmetic.
    assert client.post("/config/apply", data={}).status_code == 400


def test_a_read_only_tab_cannot_apply(tmp_path):
    repo = _repo(tmp_path)
    owner = _client(repo)
    onlooker = TestClient(owner.app)
    onlooker.get("/config")
    response = onlooker.post(
        "/config/apply",
        data=_submission(
            [f for f in _page(owner).forms if f["action"] == "/config/apply"][0]
        ),
    )
    assert response.status_code == 409
    assert not _generated(repo).exists()


def test_a_read_only_tab_is_shown_the_form_but_cannot_submit_it(tmp_path):
    repo = _repo(tmp_path)
    owner = _client(repo)
    onlooker = TestClient(owner.app)
    onlooker.get("/config")
    page = _parse(onlooker.get("/config").text)
    assert _controls(page), "a read-only tab is shown nothing at all"
    config_buttons = [
        button
        for form, button in _buttons(page)
        if (button.get("formaction") or form["action"] or "").startswith("/config")
    ]
    assert config_buttons, "no configuration controls on the page at all"
    assert all("disabled" in button for button in config_buttons), (
        "a read-only tab is offered live controls"
    )
    # The take-control button is the one thing it must still be able to press.
    whole_page = _Page()
    whole_page.feed(onlooker.get("/config").text)
    assert any(form["action"] == "/take-control" for form in whole_page.forms)


# ------------------------------------------------------------------ the driver


def _apply(
    client,
    *,
    task_name="plane_assembly_v9",
    operator_name="Test Operator",
    cameras=None,
    grippers=None,
    overrides=None,
    head=True,
    expect=200,
):
    """Fill the rendered form the way an operator would, then press Apply.

    Everything starts from the markup the page produced, so a field the template
    never rendered cannot be posted, and a control the page disabled is not sent.
    """
    page = _page(client)
    form = [f for f in page.forms if f["action"] == "/config/apply"]
    assert form, "the page renders no apply form"
    data = _submission(form[0])
    # Remove a default head-camera row through the public route. A row with
    # assigned hardware requires confirm=yes.
    if not head:
        for assignment in list(client.app.state.rig.draft.optional):
            if assignment.type_id.startswith("head_"):
                client.post(
                    "/config/optional/remove?confirm=yes",
                    data={"config_form": "1", "logical_name": assignment.logical_name},
                )
        page = _page(client)
        form = [f for f in page.forms if f["action"] == "/config/apply"]
        data = _submission(form[0])
    # An optional camera the operator added but has not pointed at hardware yet
    # is filled in here, each with its own device -- that is what an operator
    # does, and leaving it blank has its own test below.
    for name, value in list(data.items()):
        if name.startswith("optional.") and name.endswith(".device") and not value:
            data[name] = "dev-" + name.split(".")[1]
    data["task_name"] = task_name
    data["operator_name"] = operator_name
    for arm, device in {**_GRIPPERS, **(grippers or {})}.items():
        data[f"gripper.{arm}"] = device
    # Keyed by block, which is what the form renders: one box per board.
    for name, device in {**_CAMERA_FIELDS, **(cameras or {})}.items():
        data[f"camera.{name}"] = device
    data.update(overrides or {})
    response = client.post("/config/apply", data=data)
    assert response.status_code == expect, (
        f"apply answered {response.status_code}: {response.text[:400]}"
    )
    return response


def test_a_new_task_without_a_description_is_recorded_as_unknown(tmp_path):
    """The description is optional: a new task nobody described applies, is
    saved to the list, and labels every episode "Unknown"."""
    repo = _repo(tmp_path)
    client = _client(repo)

    _apply(client, task_name="not_a_real_task")

    meta = _load(_generated(repo))["meta"]
    assert meta["task_description"] == "Unknown"
    assert task_registry_store.local_registry_path().exists()


def test_a_known_task_still_applies(tmp_path):
    """A task already on the list asks for no description."""
    repo = _repo(tmp_path)
    _apply(_client(repo), task_name="plane_assembly_v9")  # expect=200 default
    assert _generated(repo).exists()


def test_an_unreadable_task_list_blocks_apply_with_a_field_error(tmp_path, monkeypatch):
    """No task list at all blocks Apply: with no list the recorder would stamp
    unvalidated labels and default check rules on every episode. The error
    carries the remedy, not a judgement of the name -- and never calls a task
    that may well be on the list a new one."""
    repo = _repo(tmp_path)
    client = _client(repo)

    def unreachable():
        raise OSError("the task list cannot be read")

    monkeypatch.setattr(task_registry_store, "get_all_tasks", unreachable)
    response = _apply(client, task_name="plane_assembly_v9", expect=400)

    assert not _generated(repo).exists(), "a refused apply wrote a config"
    described = _describes(_parse(response.text), "task_name")
    assert "task list" in described.lower(), described
    assert "recording is disabled" in described.lower(), described
    assert "new task" not in described.lower(), described


@pytest.mark.parametrize(
    "name",
    [
        "Madonna",
        "john doe",
        "Jane doe",
        "Jane Jane",
        "José Núñez",
        "Mary-Jane O'Brien",
        "J. R. Doe",
        "प्रिया",  # Devanagari vowel signs are combining marks, not letters
    ],
)
def test_any_name_written_in_letters_applies_as_typed(tmp_path, name):
    """One word or several, in any case and any alphabet: a first and last name
    is not required. Hyphens, apostrophes and periods belong to names."""
    repo = _repo(tmp_path)
    _apply(_client(repo), operator_name=name)

    assert _load(_generated(repo))["meta"]["operator_name"] == name


@pytest.mark.parametrize("bad", ["Jane2", "jane_doe", "R2-D2", "<b>Jane</b>", "-"])
def test_an_operator_name_with_digits_or_symbols_blocks_apply(tmp_path, bad):
    repo = _repo(tmp_path)
    response = _apply(_client(repo), operator_name=bad, expect=400)

    assert not _generated(repo).exists(), "a refused apply wrote a config"
    assert _describes(_parse(response.text), "operator_name"), (
        "no error beside the operator box"
    )


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_operator_name_is_recorded_as_unknown(tmp_path, blank):
    """The name is optional. A blank one reaches the data as "Unknown" -- in the
    session's config.yaml and the zarr attrs the recorder writes from the
    applied config -- rather than as nothing."""
    repo = _repo(tmp_path)
    _apply(_client(repo), operator_name=blank)

    meta = _load(_generated(repo))["meta"]
    assert meta["operator_name"] == "Unknown"

    session = tmp_path / "session"
    config = {"meta": meta, "video_sources": [], "numeric_sources": []}
    RecorderManager(recorders=[], record_path=str(session), config=config)
    assert _load(session / "config.yaml")["meta"]["operator_name"] == "Unknown"
    assert zarr.open(str(session), mode="r").attrs["operator_name"] == "Unknown"


def test_the_operator_box_says_it_is_optional(tmp_path):
    box = _page(_client(_repo(tmp_path))).attrs_by_id["operator_name"]

    assert "required" not in box
    assert UNKNOWN_OPERATOR in box["placeholder"]


def test_a_full_operator_name_applies(tmp_path):
    """Two distinct, capitalised words is a full name and is accepted."""
    repo = _repo(tmp_path)
    _apply(_client(repo), operator_name="Jane Doe")  # expect=200 default
    assert _generated(repo).exists()


# ------------------------------- the task name answers for itself while typing

# The line under the task box is derived from the task name, so it refreshes
# while the operator types; otherwise the only way to find out whether the name
# had been recognised would be to press Apply and read the page that came back.


def _refresh(client, task_name, *, page=None, overrides=None, htmx=True):
    """Type a task name and let the box refresh itself, as htmx does.

    The payload is the whole form, rebuilt from what the page rendered, because
    that is what `hx-include="closest form"` posts -- a test that sent the task
    name alone would pass just as happily against an endpoint that discarded
    everything else the operator had typed.
    """
    page = page or _page(client)
    form = [f for f in page.forms if f["action"] == "/config/apply"]
    assert form, "the page renders no apply form"
    data = _submission(form[0])
    data["task_name"] = task_name
    data.update(overrides or {})
    return client.post(
        "/config/task",
        data=data,
        headers={"HX-Request": "true"} if htmx else {},
        follow_redirects=False,
    )


def _task_box(page: _Page) -> dict:
    control = _controls(page).get("task_name")
    assert control is not None, "the page renders no task name box"
    return control


def test_changing_the_task_name_refreshes_the_description(tmp_path):
    """The line refreshes from one request, with no save."""
    client = _client(_repo(tmp_path))
    response = _refresh(client, "plane_assembly_v9")

    assert response.status_code == 200, response.status_code
    fragment = _parse(response.text)
    assert "Assemble the plane" in _shown(fragment, "task-hint"), response.text


def test_the_refresh_answers_with_a_fragment_and_writes_no_config(tmp_path):
    """Without a save is the whole request: the recording config must not be
    written by a keystroke, and a whole page swapped into the line would nest a
    second <html> inside the form."""
    repo = _repo(tmp_path)
    client = _client(repo)
    body = _refresh(client, "plane_assembly_v9").text

    assert "<!DOCTYPE" not in body and "<html" not in body, body[:200]
    assert not _generated(repo).exists(), "a refresh wrote the recording config"


def test_the_line_is_one_swap_that_keeps_its_target(tmp_path):
    """The line comes from one `_task_view`, in one node, swapped whole."""
    client = _client(_repo(tmp_path))
    page = _page(client)
    box = _task_box(page)

    target = box.get("hx-target", "")
    assert target.startswith("#"), box
    assert target[1:] in page.attrs_by_id, (
        f"hx-target {target} names an id the page does not declare"
    )
    # outerHTML, so the node that arrives has to carry the id the next swap
    # aims at -- an innerHTML swap would work once and then lose its target.
    assert "outerHTML" in box.get("hx-swap", ""), box

    fragment = _parse(_refresh(client, "a_30_second_task").text)
    root = target[1:]
    assert root in fragment.attrs_by_id, sorted(fragment.attrs_by_id)
    shown = _shown(fragment, root)
    assert "Do the short thing" in shown, shown


def test_the_line_follows_the_name_that_was_last_typed(tmp_path):
    client = _client(_repo(tmp_path))
    first = _parse(_refresh(client, "plane_assembly_v9").text)
    second = _parse(_refresh(client, "a_30_second_task").text)

    assert "Assemble the plane" in _shown(first, "task-hint")
    assert "Do the short thing" in _shown(second, "task-hint")


def test_a_refresh_keeps_every_other_field_the_operator_had_filled_in(tmp_path):
    """A refresh must not blank the device boxes -- or the operator name typed a
    second earlier -- which would be worse than a stale line. That is why the
    box posts the whole form and the endpoint absorbs it exactly as
    Add-a-camera does."""
    client = _client(_repo(tmp_path))
    typed = {
        "operator_name": "Test Operator",
        **{f"gripper.{arm}": device for arm, device in _GRIPPERS.items()},
        **{f"camera.{name}": device for name, device in _CAMERA_FIELDS.items()},
    }
    _refresh(client, "plane_assembly_v9", overrides=typed)

    still_there = _controls(_page(client))
    for name, value in typed.items():
        assert still_there[name].get("value") == value, (
            f"the refresh discarded {name}: {still_there[name].get('value')!r}"
        )
    assert still_there["task_name"].get("value") == "plane_assembly_v9"


def test_a_task_list_that_will_not_load_refreshes_to_unreadable_not_unknown(
    tmp_path, monkeypatch
):
    """State (c) has to survive the live path too. An offline station telling
    the operator mid-keystroke that their task name is wrong is exactly the
    message `_task_state` keeps a fourth state to avoid."""
    client = _client(_repo(tmp_path))

    def unreachable():
        raise OSError("the task list cannot be read")

    monkeypatch.setattr(task_registry_store, "get_all_tasks", unreachable)
    response = _refresh(client, "plane_assembly_v9")

    assert response.status_code == 200, response.status_code
    said = _shown(_parse(response.text), "task-hint")
    assert "new task" not in said.lower(), said
    assert "task list" in said.lower(), said
    assert "recording is disabled" in said.lower(), said


def test_a_refresh_that_cannot_read_the_registry_still_answers(tmp_path, monkeypatch):
    """It must not go blank either: htmx swaps whatever comes back, so an empty
    or errored answer is a line the operator can never get back without a
    reload."""
    client = _client(_repo(tmp_path))

    def unreachable():
        raise OSError("the task list cannot be read")

    monkeypatch.setattr(task_registry_store, "get_all_tasks", unreachable)
    monkeypatch.setattr(
        task_registry_store, "get_registry_info", lambda: (_ for _ in ()).throw(OSError)
    )
    fragment = _parse(_refresh(client, "plane_assembly_v9").text)

    assert _shown(fragment, "task-hint").strip(), (
        "the confirmation line came back blank"
    )


def test_the_task_box_is_debounced_and_cannot_queue_behind_a_slow_registry(tmp_path):
    """A request per keystroke hammers the task registry; `change` alone means
    nothing happens until focus leaves the box. So:
    debounced keyup for while they type, `change` for paste and blur, and a sync
    strategy so the newest name wins rather than queueing behind an older one."""
    box = _task_box(_page(_client(_repo(tmp_path))))
    trigger = box.get("hx-trigger", "")

    assert "keyup changed delay:" in trigger, trigger
    assert "change" in trigger.replace("changed", ""), trigger
    assert box.get("hx-sync", ""), (
        "nothing stops a slow refresh from being overtaken by a newer one"
    )


def test_the_task_box_posts_the_whole_form_like_every_other_control(tmp_path):
    box = _task_box(_page(_client(_repo(tmp_path))))
    assert box.get("hx-post") == "/config/task", box
    assert box.get("hx-include") == "closest form", box


def test_the_task_name_still_works_as_a_plain_form_field_with_the_script_off(
    tmp_path,
):
    """Progressive enhancement. The refresh is an enhancement: with htmx
    blocked the box is still a named input in the page's one form, and Apply
    still produces the line."""
    repo = _repo(tmp_path)
    client = _client(repo)
    page = _page(client)
    box = _task_box(page)

    assert box.get("name") == "task_name" and box.get("tag") == "input", box
    assert not box.get("hx-swap-oob"), box
    form = [f for f in page.forms if f["action"] == "/config/apply"]
    assert form and form[0]["method"] == "post", "the box sits in no plain POST form"

    _apply(client, task_name="a_30_second_task")
    after = _page(client)
    assert "Do the short thing" in _shown(after, "task-hint")
    assert _load(_generated(repo))["meta"]["task_description"] == "Do the short thing"


def test_a_refresh_posted_without_the_script_lands_back_on_the_page(tmp_path):
    """Nothing posts here without htmx, so a stray one is answered with the
    page rather than a bare fragment a browser would render as the whole
    document."""
    response = _refresh(_client(_repo(tmp_path)), "plane_assembly_v9", htmx=False)
    assert response.status_code == 303, response.status_code
    assert response.headers["location"] == "/config"


def test_a_refresh_does_not_name_where_the_task_list_came_from(tmp_path):
    """The swapped partial says what the task is, never which registry file or
    cache the task list was read from."""
    repo = _repo(tmp_path)
    operator = _parse(_refresh(_client(repo), "plane_assembly_v9").text)

    assert "task list" not in _shown(operator, "task-hint").lower(), _shown(
        operator, "task-hint"
    )


# ------------------------------------ the world-frame reference camera picker


def _reference(page: _Page) -> dict | None:
    """The reference-camera `<select>`, or None when the page renders none."""
    return _controls(page).get("reference_camera")


def _selected(control: dict) -> list[str]:
    return [o.get("value", "") for o in control["options"] if "selected" in o]


def test_the_reference_picker_offers_the_head_camera(tmp_path):
    """The head camera a fresh session starts with is a reference option, and
    the picker defaults to Automatic -- which the assembler resolves to the
    auto-pick, so an operator who ignores it gets the assembler's choice."""
    repo = _repo(tmp_path)
    client = _client(repo)
    _apply(client)
    picker = _reference(_page(client))
    assert picker is not None, "no reference-camera picker rendered"
    values = [o.get("value", "") for o in picker["options"]]
    assert "head_oak_camera" in values, values
    assert _selected(picker) == [""], "the picker did not default to Automatic"


def test_ignoring_the_picker_anchors_on_the_auto_pick(tmp_path):
    """The head OAK outranks every arm camera at the same rate; leaving the
    picker on Automatic must write head_oak_camera, exactly as the assembler
    would on its own."""
    repo = _repo(tmp_path)
    client = _client(repo)
    _apply(client)
    assert _load(_generated(repo))["reference_camera"] == "head_oak_camera"


def test_choosing_a_reference_camera_changes_the_config_and_survives_reopen(tmp_path):
    """The operator overrides the auto-pick onto a tracking camera; it reaches
    the written config, and reopening the page shows it selected and keeps it on
    the next apply -- proof the choice round-trips through the draft."""
    repo = _repo(tmp_path)
    client = _client(repo)
    _apply(client)
    _apply(client, overrides={"reference_camera": "right_tracking_camera"})
    assert _load(_generated(repo))["reference_camera"] == "right_tracking_camera"

    # Reopen: the picker shows the override, and re-applying without touching it
    # keeps it -- the value came back through the draft, not the posted field.
    assert _selected(_reference(_page(client))) == ["right_tracking_camera"]
    _apply(client)
    assert _load(_generated(repo))["reference_camera"] == "right_tracking_camera"


def test_the_reference_applies_through_the_plain_form_with_htmx_blocked(tmp_path):
    """Progressive enhancement: the picker is a plain <select> in the page's one
    POST form, so Apply carries it with the vendored htmx blocked -- which is
    exactly what the TestClient is, a browser that runs no script."""
    repo = _repo(tmp_path)
    client = _client(repo)
    _apply(client)
    apply_form = next(f for f in _page(client).forms if f["action"] == "/config/apply")
    assert apply_form["method"] == "post"
    assert "reference_camera" in {c.get("name") for c in apply_form["controls"]}
    # Drive that plain form with the operator's pick; no HX header, no script.
    _apply(client, overrides={"reference_camera": "right_tracking_camera"})
    assert _load(_generated(repo))["reference_camera"] == "right_tracking_camera"


def test_the_picker_offers_the_tracking_cameras(tmp_path):
    """Not a misleading empty dropdown: the picker offers the calibratable
    tracking cameras with Automatic preselected."""
    repo = _repo(tmp_path)
    client = _client(repo)
    _apply(client)
    picker = _reference(_page(client))
    assert picker is not None, "no reference-camera picker rendered"
    values = [o.get("value", "") for o in picker["options"]]
    assert "left_tracking_camera" in values, values
    assert "right_tracking_camera" in values, values
    assert _selected(picker) == [""], "the picker did not default to Automatic"


def test_adding_a_head_camera_switches_the_reference_to_it(tmp_path):
    """With no head camera the world frame is a tracking camera. Adding a head
    OAK makes it the automatic reference -- written to the config, and visible
    in the picker, which stays on Automatic but now names the head camera.
    Driven through the real add + apply flow, not the assembler alone, so it
    pins that the GUI reflects the switch and not merely that assembly would.
    """
    repo = _repo(tmp_path)
    client = _client(repo)

    _apply(client, head=False)
    assert _load(_generated(repo))["reference_camera"] == "right_tracking_camera"

    _add(client, _page(client), "head_oak")
    _apply(client, overrides={"optional.head_oak_camera.device": "oak-HEAD"})

    # The written world frame switched off the tracking camera onto the head...
    assert _load(_generated(repo))["reference_camera"] == "head_oak_camera"

    # Automatic selection must reflect the newly available head reference.
    after = _page(client)
    assert _selected(_reference(after)) == [""], "the picker left Automatic"
    assert "Head oak camera" in after.text_by_id["reference-section"]


def test_a_bottom_camera_does_not_take_the_world_frame_off_the_head(tmp_path):
    """What a real station does: a fresh session starts with a head OAK row, so
    the DEFAULT rig anchors on the head -- and adding the bottom OAK, which looks
    up from the table, must not take the frame off it."""
    repo = _repo(tmp_path)
    client = _client(repo)

    _apply(client)
    assert _load(_generated(repo))["reference_camera"] == "head_oak_camera"

    _add(client, _page(client), "bottom_oak")
    _apply(client, overrides={"optional.bottom_oak_camera.device": "oak-BOTTOM"})

    cfg = _load(_generated(repo))
    assert cfg["reference_camera"] == "head_oak_camera"
    # The bottom camera is recorded all the same -- it just does not anchor.
    names = {s.get("name") for s in cfg["video_sources"] if s.get("name")}
    assert "bottom_oak_camera" in names


# ------- save confirmation + unregistered-device warning ----------------------


def test_a_successful_apply_confirms_the_save_and_the_confirmation_is_one_shot(
    tmp_path,
):
    """A successful Apply redirects to a page identical to the one before it,
    so it lands on a one-shot "Configuration saved." banner that a reload does
    not repeat."""
    client = _client(_repo(tmp_path))
    assert "Configuration saved." not in client.get("/config").text

    applied = _apply(client, head=False)  # follows the 303 redirect back to /config
    assert "Configuration saved." in applied.text

    # One-shot: it lives for one request on app.state, so a reload is silent.
    assert "Configuration saved." not in client.get("/config").text


def test_an_attached_device_the_registry_does_not_know_is_flagged_by_serial():
    """An attached device the scan widened into the options (`_options`) resolves
    cleanly to its raw serial with an OK status, so without a separate check
    nothing on screen would say it carries no registered name. The soft warning
    fires for exactly that case; a registered device, attached or not, stays
    quiet."""
    from station_ui import devices as devices_core
    from station_ui.routes import config as cfg

    found = _found("oak-UNREG", "oak-REG")
    options = ["oak-REG", "oak-UNREG"]

    def resolve(kind, typed):
        return devices_core.resolve_device(typed, options)

    def registered(kind, name):
        return name == "oak-REG"

    unreg = cfg._device_view(
        field="camera.right_trigger_camera",
        label="Right camera",
        kind="oak",
        value="oak-UNREG",
        preset=None,
        found=found,
        scanning=False,
        errors={},
        resolve=resolve,
        registered=registered,
    )
    assert unreg["message_state"] == "warn"
    assert "not a registered" in unreg["message"]
    assert "oak-UNREG" in unreg["message"]

    reg = cfg._device_view(
        field="camera.left_trigger_camera",
        label="Left camera",
        kind="oak",
        value="oak-REG",
        preset=None,
        found=found,
        scanning=False,
        errors={},
        resolve=resolve,
        registered=registered,
    )
    assert "not a registered" not in (reg["message"] or "")


def test_an_unavailable_registry_does_not_flag_every_device(monkeypatch):
    """When the device registry cannot load, `_registry_names` comes back empty.
    The page must not then claim every attached device is unregistered -- that
    would flood the page on a bad-network day. `_registered_check` treats an
    empty registry as "cannot say", so the warning stays off; a loaded registry
    distinguishes normally."""
    from station_ui.core import device_detector
    from station_ui.routes import config as cfg

    monkeypatch.setattr(device_detector, "get_available_oak_cameras", lambda: [])
    empty = cfg._registered_check(preset=None)
    assert empty("oak", "anything-at-all") is True  # cannot say -> no warning

    monkeypatch.setattr(
        device_detector, "get_available_oak_cameras", lambda: ["oak-REG"]
    )
    loaded = cfg._registered_check(preset=None)
    assert loaded("oak", "oak-REG") is True
    assert loaded("oak", "oak-UNREG") is False


def test_the_task_card_never_sends_the_operator_to_a_lead_for_setup_text(tmp_path):
    """Setup text is optional: a task added from this page has none, and tasks
    are not registered by hand. The fixture's tasks carry none."""
    repo = _repo(tmp_path)
    client = _client(repo)
    _apply(client, task_name="plane_assembly_v9")
    page = client.get("/config").text
    assert "No setup instruction" not in page
    assert "ask a lead" not in page.lower()


def test_the_task_card_says_when_the_registry_came_from_an_unexpected_source(
    tmp_path, monkeypatch
):
    """Without naming the source (see
    test_a_refresh_does_not_name_where_the_task_list_came_from), but that it
    may be stale is what explains a missing instruction to the operator."""
    stale = "came from an unexpected source"
    info = {"source": "cache", "task_count": 3}
    monkeypatch.setattr(task_registry_store, "get_registry_info", lambda: info)
    repo = _repo(tmp_path)
    client = _client(repo)
    _apply(client, task_name="plane_assembly_v9")
    page = client.get("/config").text
    assert stale in page
    assert "cache" not in _shown(_parse(page), "task-hint").lower()

    info["source"] = "bundled"
    page = client.get("/config").text
    assert stale not in page

    info["source"] = "file:/tmp/tasks.json"  # the local task list is live, not stale
    page = client.get("/config").text
    assert stale not in page

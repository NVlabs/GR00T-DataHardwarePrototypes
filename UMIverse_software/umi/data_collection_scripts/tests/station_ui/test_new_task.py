# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Any task can be recorded, with no registration step.

The Task box offers every task on this station's list and takes a new name in
the same box. A new name may bring a description, which is optional and is
recorded as "Unknown" when left blank -- and Apply saves both to the station's own task file, so the next session offers it.

These run against the real task store, unlike `test_config_page.py`, which pins
the list. conftest points `UMI_TASK_REGISTRY_FILE` at a file that does not exist
yet, so the list starts as the bundled one, as on a fresh station. The shipped
bundled list is empty; here it holds one task, `KNOWN`, standing for a task
recorded before.
"""

import json
from html.parser import HTMLParser

import pytest

pytest.importorskip("yaml")

import test_config_page as cp  # noqa: E402

from umi.common.tasks import task_registry_store  # noqa: E402

NEW = "Cup on plate"
DESCRIPTION = "Put the cup on the plate"
KNOWN = "stack_cups"
KNOWN_DESCRIPTION = "Stack the cups"


@pytest.fixture(autouse=True)
def _one_bundled_task(monkeypatch, tmp_path_factory):
    bundled = tmp_path_factory.mktemp("bundled") / "tasks_fallback.json"
    task = {"name": KNOWN, "short_description": KNOWN_DESCRIPTION}
    bundled.write_text(json.dumps({"version": 1, "tasks": {KNOWN: task}}))
    monkeypatch.setattr(task_registry_store, "FALLBACK_FILE", bundled)
    task_registry_store.reload()


class _TaskBox(HTMLParser):
    """The Task box's wrapper classes, the options of the list it offers, and
    every input by name -- a refresh fragment has no `<form>`, which is all
    `test_config_page._controls` reads."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.combo_classes: list[str] = []
        self.options: dict[str, str] = {}
        self.inputs: dict[str, dict] = {}
        self._combo: list[str] | None = None
        self._in_list = False

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        classes = (attr.get("class") or "").split()
        if tag == "input" and attr.get("name"):
            self.inputs[attr["name"]] = attr
        if tag == "div" and "combo" in classes:
            self._combo = classes
        elif tag == "input" and attr.get("name") == "task_name" and self._combo:
            self.combo_classes = self._combo
        elif tag == "datalist":
            self._in_list = attr.get("id") == "task-options"
        elif tag == "option" and self._in_list:
            self.options[attr.get("value") or ""] = attr.get("label") or ""

    def handle_endtag(self, tag):
        if tag == "datalist":
            self._in_list = False


def _task_box(html: str) -> _TaskBox:
    from html_sections import section_html

    html = section_html(html, "configuration")
    parser = _TaskBox()
    parser.feed(html)
    return parser


def _saved() -> dict:
    return json.loads(task_registry_store.local_registry_path().read_text())


def test_the_task_box_is_a_searchable_dropdown_of_previous_tasks(tmp_path):
    html = cp._client(cp._repo(tmp_path)).get("/config").text
    box = cp._controls(cp._parse(html))["task_name"]
    found = _task_box(html)

    assert box.get("placeholder") == "Search previous tasks or enter a new task…"
    assert "combo--search" in found.combo_classes
    assert box.get("list") == "task-options"
    # Every task on the station's list, labelled with what it records as.
    assert found.options[KNOWN] == KNOWN_DESCRIPTION
    assert set(found.options) == set(task_registry_store.get_all_tasks())


def test_a_new_name_asks_for_a_description_and_keeps_what_was_typed(tmp_path):
    """The box refreshes the line under it as the operator types, and that
    refresh must not wipe a description they have already started."""
    client = cp._client(cp._repo(tmp_path))

    html = cp._refresh(client, NEW, overrides={"task_description": DESCRIPTION}).text
    assert "new task" in cp._shown(cp._parse(html), "task-hint").lower()
    assert _task_box(html).inputs["task_description"].get("value") == DESCRIPTION

    # A task already on the list asks for nothing more.
    html = cp._refresh(client, KNOWN).text
    assert KNOWN_DESCRIPTION in cp._shown(cp._parse(html), "task-hint")
    assert "task_description" not in _task_box(html).inputs


def test_a_new_task_is_saved_with_its_description_and_offered_next_time(tmp_path):
    repo = cp._repo(tmp_path)
    cp._apply(
        cp._client(repo),
        task_name=NEW,
        overrides={"task_description": DESCRIPTION},
    )

    # Saved to this station's own list, beside the tasks it already offered.
    tasks = _saved()["tasks"]
    assert tasks[NEW] == {"name": NEW, "short_description": DESCRIPTION}
    assert KNOWN in tasks
    # And it is what every episode of the session is labelled with.
    meta = cp._load(cp._generated(repo))["meta"]
    assert (meta["task_name"], meta["task_description"]) == (NEW, DESCRIPTION)

    # The next session reads it back from the file, offers it in the box, and
    # treats it as a task it already knows.
    task_registry_store.reload()
    client = cp._client(repo)
    assert _task_box(client.get("/config").text).options[NEW] == DESCRIPTION
    html = cp._refresh(client, NEW).text
    assert DESCRIPTION in cp._shown(cp._parse(html), "task-hint")
    assert "task_description" not in _task_box(html).inputs


def test_a_description_is_not_carried_over_to_a_different_new_task(tmp_path):
    """A description typed for one new task is not saved under another: type a
    sentence for one new task, pick a listed task (which hides the description
    box), then type another new name and apply."""
    repo = cp._repo(tmp_path)
    client = cp._client(repo)
    cp._refresh(client, NEW, overrides={"task_description": DESCRIPTION})
    cp._refresh(client, KNOWN)

    cp._apply(client, task_name="Fold the towel")

    # Nothing typed for this task, so it is "Unknown" -- not the other
    # task's sentence.
    assert _saved()["tasks"]["Fold the towel"]["short_description"] == "Unknown"


def test_a_task_already_on_the_list_is_not_written_again(tmp_path):
    cp._apply(cp._client(cp._repo(tmp_path)), task_name=KNOWN)
    assert not task_registry_store.local_registry_path().exists()


def test_a_name_with_no_letters_or_digits_is_refused(tmp_path):
    """It names the session folder, and every such name slugs to `unnamed`."""
    repo = cp._repo(tmp_path)
    response = cp._apply(
        cp._client(repo),
        task_name="???",
        overrides={"task_description": DESCRIPTION},
        expect=400,
    )

    described = cp._describes(cp._parse(response.text), "task_name")
    assert "letters or numbers" in described.lower(), described
    assert not task_registry_store.local_registry_path().exists()
    assert not cp._generated(repo).exists()


def test_a_task_file_that_cannot_be_read_is_never_overwritten(tmp_path):
    """The loader falls back to the bundled list when the station's file is
    broken. Saving a new task then would replace the station's whole list
    with that fallback, so Apply refuses and says where the file is."""
    path = task_registry_store.local_registry_path()
    path.write_text("{ not json")
    task_registry_store.reload()
    repo = cp._repo(tmp_path)

    response = cp._apply(
        cp._client(repo),
        task_name=NEW,
        overrides={"task_description": DESCRIPTION},
        expect=400,
    )

    described = cp._describes(cp._parse(response.text), "task_name")
    assert "could not be saved" in described.lower(), described
    assert str(path) in described
    assert path.read_text() == "{ not json"
    assert not cp._generated(repo).exists()

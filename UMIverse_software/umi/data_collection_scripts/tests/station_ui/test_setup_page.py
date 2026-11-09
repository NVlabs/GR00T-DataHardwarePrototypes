# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The setup page: four data rows in, four cards out.

The point of `SETUP_PRESETS` is that a supported setup is a row of data, not a
branch. This suite exists to keep it that way: the cards are built from the
tuple at request time, so a fifth generation costs one row and zero template
edits. `test_a_fifth_preset_renders_with_no_template_change` is the gate --
everything else guards ownership, the no-JavaScript path, and accessibility.
"""

import pathlib
import re
from html.parser import HTMLParser

from fastapi.testclient import TestClient
from jinja2 import Environment, nodes

from station_ui.app import create_app
from station_ui.core import setup_presets as sp

REPO = pathlib.Path(__file__).resolve().parents[2]
TEMPLATES = REPO / "station_ui" / "templates"


# --------------------------------------------------------------------- parsing


class _Page(HTMLParser):
    """Forms, their fields, their buttons, the links, the spans inside a card,
    and any inline event handler.

    Deliberately structural: asserting on rendered markup as a string would let
    a card that is only clickable from JavaScript pass a test named "renders a
    form".
    """

    def __init__(self):
        super().__init__()
        self.forms: list[dict] = []
        self.links: list[dict] = []
        self.inline_handlers: list[str] = []
        self._form: dict | None = None
        self._button: dict | None = None
        self._link: dict | None = None
        # A stack, because a card's spans nest: the selection state wraps an
        # icon and a word inside the title span.
        self._spans: list[dict] = []

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        self.inline_handlers += [k for k in attr if k.startswith("on")]
        if tag == "a":
            self._link = {
                **attr,
                "classes": (attr.get("class") or "").split(),
                "text": "",
            }
            self.links.append(self._link)
            return
        if tag == "form":
            self._form = {
                "action": attr.get("action"),
                "method": (attr.get("method") or "get").lower(),
                "fields": {},
                "buttons": [],
            }
            self.forms.append(self._form)
            return
        if self._form is None:
            return
        if tag in ("input", "select", "textarea") and attr.get("name"):
            self._form["fields"][attr["name"]] = attr.get("value", "")
        elif tag == "button":
            self._button = {
                **attr,
                "classes": (attr.get("class") or "").split(),
                "spans": [],
                "text": "",
            }
            self._form["buttons"].append(self._button)
        elif tag == "span" and self._button is not None:
            span = {
                **attr,
                "classes": (attr.get("class") or "").split(),
                "text": "",
            }
            self._button["spans"].append(span)
            self._spans.append(span)

    def handle_endtag(self, tag):
        if tag == "a":
            self._link = None
        elif tag == "form":
            self._form = None
        elif tag == "button":
            self._button = None
            self._spans = []
        elif tag == "span" and self._spans:
            self._spans.pop()

    def handle_data(self, data):
        if self._button is not None:
            self._button["text"] += data
        if self._link is not None:
            self._link["text"] += data
        for span in self._spans:
            span["text"] += data


def _parse(html: str) -> _Page:
    from html_sections import section_html

    html = section_html(html, "setup")
    page = _Page()
    page.feed(html)
    return page


def _cards(html: str) -> list[tuple[dict, dict]]:
    """(form, button) for every preset card, in document order."""
    page = _parse(html)
    return [
        (form, button)
        for form in page.forms
        for button in form["buttons"]
        if "preset-card" in button["classes"]
    ]


def _text(button: dict) -> str:
    return " ".join(button["text"].split())


def _spans(button: dict, classname: str) -> list[dict]:
    return [s for s in button["spans"] if classname in s["classes"]]


def _owning_client(app=None) -> tuple[object, TestClient]:
    """An app plus a client that has loaded `/` once, so it owns the rig --
    exactly what a browser does before it can click anything."""
    app = app or create_app()
    client = TestClient(app)
    client.get("/")
    return app, client


# ----------------------------------------------------------------- the cards


def test_the_page_lists_every_preset_once_in_document_order():
    _, client = _owning_client()
    labels = [_text(button) for _, button in _cards(client.get("/").text)]
    assert len(labels) == len(sp.SETUP_PRESETS)
    for rendered, preset in zip(labels, sp.SETUP_PRESETS):
        assert preset.label in rendered


def test_a_fifth_preset_renders_with_no_template_change(monkeypatch):
    """The gate: adding a generation must cost one data row and zero UI edits.

    Patching `sp` -- the module object the route itself imported -- is
    load-bearing, and the assertion on the card count is what makes it real.
    `station_ui/routes/setup.py` reads `setup_presets.SETUP_PRESETS` through
    the module per request; if it ever bound the names at import time instead,
    the table would be frozen into the route and the page would keep rendering
    four cards while a weaker assertion here still passed.
    """
    extra = sp.SetupPreset(
        id="bimanual_v9",
        label="Bimanual V9",
        operation_mode="bimanual",
        gripper_generation="v9",
        template_name="bimanual_v4.yaml",
        pose_tracker="iphone",
    )
    monkeypatch.setattr(sp, "SETUP_PRESETS", sp.SETUP_PRESETS + (extra,))
    monkeypatch.setattr(sp, "_BY_ID", {p.id: p for p in sp.SETUP_PRESETS})

    _, client = _owning_client()
    body = client.get("/").text

    # Counted against the patched table, not against a literal: the shipped rows
    # are data and a new one lands here without touching this test -- which is
    # the property the whole test is about.
    expected = len(sp.SETUP_PRESETS)
    assert body.count('class="preset-card"') == expected
    assert len(_cards(body)) == expected
    extra_card = _cards(body)[-1]
    assert "Bimanual V9" in _text(extra_card[1])
    # Rendered from the row, not from a lookup table keyed by a known id.
    assert "V9" in _text(extra_card[1])
    assert extra_card[0]["fields"]["preset_id"] == "bimanual_v9"

    # And it is selectable, not merely drawn.
    resp = client.post(
        extra_card[0]["action"], data=extra_card[0]["fields"], follow_redirects=False
    )
    assert resp.status_code == 303


def test_no_template_hardcodes_a_preset_id_label_generation_or_file():
    """Structural, and stronger than grepping for 'v1'/'v2'/'v3': a template
    could branch on `{% if preset.gripper_generation == something %}` with no
    literal generation in it and pass that grep. If none of the four
    per-row strings appears literally anywhere in the templates, the cards
    cannot have been special-cased."""
    varying = set()
    for preset in sp.SETUP_PRESETS:
        varying |= {
            preset.id,
            preset.label,
            preset.gripper_generation,
            preset.template_name,
        }
    for path in TEMPLATES.rglob("*.html"):
        text = path.read_text()
        for literal in sorted(varying):
            assert not re.search(rf"\b{re.escape(literal)}\b", text), (
                f"{path.name} hardcodes {literal!r} instead of rendering the row"
            )


def test_no_template_branches_a_preset_against_a_string_literal():
    """The other half: `{% if preset.id == 'bimanual_v4' %}` is the anti-pattern
    even when the literal is spelled some other way. Comparing a preset field
    to another *variable* (`preset.id == selected`) is fine -- that is still
    data-driven and a fifth row goes through it unchanged."""
    fields = {"id", "label", "gripper_generation", "template_name"}
    env = Environment()
    for path in TEMPLATES.rglob("*.html"):
        for branch in env.parse(path.read_text()).find_all(nodes.If):
            touched = {node.attr for node in branch.test.find_all(nodes.Getattr)}
            if not touched & fields:
                continue
            literals = [
                node.value
                for node in branch.test.find_all(nodes.Const)
                if isinstance(node.value, str)
            ]
            assert not literals, (
                f"{path.name} branches on a preset field against {literals}"
            )


# ---------------------------------------------------------------- the layout


def test_every_card_states_its_arm_mode_as_a_pill():
    """The arm mode the choice fixes, on the card, rendered from the row. The
    gripper generation is left out: the station records only V4 grippers."""
    _, client = _owning_client()
    cards = _cards(client.get("/").text)
    assert len(cards) == len(sp.SETUP_PRESETS)

    for preset, (_, button) in zip(sp.SETUP_PRESETS, cards):
        pills = [" ".join(s["text"].split()) for s in _spans(button, "pill")]
        assert pills == [preset.operation_mode], f"{preset.label} renders {pills}"
        assert preset.gripper_generation.upper() not in _text(button)


def test_no_template_prints_the_gripper_generation():
    """The Setup cards, the Configuration page's Session heading and the Record
    page's Setup row each printed it. Structural, so a new print anywhere in
    the templates fails here too."""
    env = Environment()
    for path in TEMPLATES.rglob("*.html"):
        for output in env.parse(path.read_text()).find_all(nodes.Output):
            printed = {node.attr for node in output.find_all(nodes.Getattr)}
            assert "gripper_generation" not in printed, (
                f"{path.name} prints the gripper generation"
            )


def test_the_radio_affordance_is_decoration_and_never_the_announced_state():
    """A set of four cards where exactly one may be chosen should look like the
    control it is. It must not be what a screen reader is given, though: that
    stays aria-pressed plus the word, which the radio cannot carry."""
    _, client = _owning_client()
    client.post("/setup/preset", data={"preset_id": sp.SETUP_PRESETS[-1].id})
    cards = _cards(client.get("/").text)

    for _, button in cards:
        radios = _spans(button, "preset-card__radio")
        assert len(radios) == 1, f"{_text(button)} has {len(radios)} radios"
        assert radios[0].get("aria-hidden") == "true", (
            "the radio is exposed to assistive tech, where it says nothing"
        )
    pressed = [b for _, b in cards if b.get("aria-pressed") == "true"]
    assert len(pressed) == 1 and "Selected" in _text(pressed[0])


def test_every_card_is_a_real_form_a_browser_can_post_without_javascript():
    """Replayed from the parsed markup, never from hardcoded values: the point
    is that what the page actually renders is postable on its own."""
    app, client = _owning_client()
    cards = _cards(client.get("/").text)
    assert cards, "the page renders no preset cards at all"

    for form, button in cards:
        assert form["method"] == "post", f"card form uses {form['method']}"
        assert form["action"], "a card form with no action is not a real target"
        assert button.get("type") == "submit", "the card is not a submit control"
        assert form["fields"].get("preset_id"), "the form carries no preset_id"

    form, _ = cards[-1]
    resp = client.post(form["action"], data=form["fields"], follow_redirects=False)
    assert resp.status_code == 303
    assert app.state.rig.draft.preset_id == sp.SETUP_PRESETS[-1].id


def test_no_control_on_the_page_depends_on_an_inline_event_handler():
    """No card may pick a preset with `onclick`: with JavaScript off it would
    select nothing."""
    _, client = _owning_client()
    page = _parse(client.get("/").text)
    assert page.inline_handlers == [], (
        f"inline handlers on the page: {sorted(set(page.inline_handlers))}"
    )


# ------------------------------------------------------------------ selection


def test_choosing_a_preset_records_it_and_stays_on_setup():
    """Picking a card stays on Setup, so an operator who picked the wrong card
    can see what they did before moving on. Choosing and moving on are two
    acts, and only the confirmation bar's one primary action is the second one.

    Still a redirect, not a rendered POST: a reload of the page this lands on
    must not re-post the selection (see SEE_OTHER in routes/setup.py).
    """
    app, client = _owning_client()
    resp = client.post(
        "/setup/preset", data={"preset_id": "bimanual_v4"}, follow_redirects=False
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/", resp.headers["location"]
    assert app.state.rig.draft.preset_id == "bimanual_v4"
    assert "preset-card" not in resp.text, (
        "the POST rendered the page itself, so a reload re-posts the selection"
    )

    # And the choice visibly took, on a page that still offers every card: the
    # point of staying is being able to change it.
    html = client.get(resp.headers["location"]).text
    pressed = [b for _, b in _cards(html) if b.get("aria-pressed") == "true"]
    assert len(pressed) == 1 and "Bimanual (iPhone)" in _text(pressed[0]), (
        "the setup page does not come back on the choice that was just made"
    )
    assert len(_cards(html)) == len(sp.SETUP_PRESETS), "the other setups are gone"


def test_the_selected_card_says_so_in_text_not_only_in_colour():
    """Brand green and healthy green are the same hue on this console, so no
    state is ever expressed by colour alone."""
    _, client = _owning_client()
    client.post("/setup/preset", data={"preset_id": "bimanual_v4"})

    from html_sections import section_html

    html = client.get("/").text
    setup = section_html(html, "setup")
    assert "confirm-bar" not in setup
    assert "Go to Configuration" not in setup
    cards = _cards(html)
    pressed = [b for _, b in cards if b.get("aria-pressed") == "true"]
    assert len(pressed) == 1, f"{len(pressed)} cards report themselves selected"
    assert "Bimanual (iPhone)" in _text(pressed[0])
    assert "Selected" in _text(pressed[0]), "selection is colour-only"
    assert all(
        b.get("aria-pressed") == "false" for _, b in cards if b is not pressed[0]
    )


# `follow_redirects=False` on the two selection tests below keeps each one about
# the POST it names and nothing else: the route answers 303 back to Setup, and
# following it would fold a full page render into an assertion about the draft.


def test_changing_the_preset_drops_devices_chosen_for_the_previous_one():
    """Devices chosen for one setup must never survive into another: the next
    template need not declare the same cameras."""
    app, client = _owning_client()
    client.post(
        "/setup/preset", data={"preset_id": "bimanual_v4"}, follow_redirects=False
    )
    draft = app.state.rig.draft
    draft.grippers["right"] = "UMI-V4-0002"
    draft.required_cameras["left_umi_camera"] = "oak-W97-51"

    client.post(
        "/setup/preset", data={"preset_id": "bimanual_v4_vive"}, follow_redirects=False
    )
    assert draft.preset_id == "bimanual_v4_vive"
    assert draft.grippers == {}
    assert draft.required_cameras == {}


def test_reselecting_the_same_preset_keeps_the_devices_already_chosen():
    """A reload re-posts the same card. Wiping the operator's device choices
    on a no-op re-selection would lose work for nothing."""
    app, client = _owning_client()
    client.post(
        "/setup/preset", data={"preset_id": "bimanual_v4"}, follow_redirects=False
    )
    draft = app.state.rig.draft
    draft.grippers["right"] = "UMI-V4-0002"

    client.post(
        "/setup/preset", data={"preset_id": "bimanual_v4"}, follow_redirects=False
    )
    assert draft.grippers == {"right": "UMI-V4-0002"}


def test_an_unknown_preset_id_is_a_400_not_a_500():
    """A stale tab or a hand-typed POST is an operator-visible refusal, and the
    refusal has to carry the way out. The id itself is left out: it names a row
    in this repository's tables, so an operator can do nothing with it.
    """
    app, client = _owning_client()
    client.post("/setup/preset", data={"preset_id": "bimanual_v4"})

    resp = client.post(
        "/setup/preset", data={"preset_id": "bimanual_v9"}, follow_redirects=False
    )
    assert resp.status_code == 400, f"unknown id answered {resp.status_code}"
    assert "no longer available" in resp.text, "the refusal does not say what happened"
    assert "reload" in resp.text.lower(), "the refusal offers no way out"
    assert "bimanual_v9" not in resp.text, "the operator is shown an internal id"
    assert app.state.rig.draft.preset_id == "bimanual_v4", "the draft was corrupted"


# ---------------------------------------------------------- card metadata


def _card(html: str, preset_id: str) -> tuple[dict, dict]:
    """(form, button) for one preset's card."""
    for form, button in _cards(html):
        if form["fields"].get("preset_id") == preset_id:
            return form, button
    raise AssertionError(f"no card for {preset_id}")


# ------------------------------------------------------ the template filename

# A yaml filename is a fact about this repository. The operator running a
# station is a data collector who was handed a rig, not a checkout, so the file
# is not on the page.


def test_no_template_filename_reaches_an_operator_anywhere_on_the_page():
    """Every file any preset can resolve to -- not just the one the current
    choice picks -- so the check cannot be passed by moving the name to another
    corner of the page."""
    files = {sp.template_for(preset) for preset in sp.SETUP_PRESETS}
    assert files, "no templates to look for; this test would pass vacuously"

    _, client = _owning_client()
    for preset in sp.SETUP_PRESETS:
        client.post("/setup/preset", data={"preset_id": preset.id})
        html = client.get("/").text
        for name in sorted(files):
            assert name not in html, f"{preset.label} shows the operator {name!r}"
        assert ".yaml" not in html, "a filename is on the operator's page"


def test_the_operator_card_carries_no_rig_build_detail():
    """The operator's question on this page is "which rig is this".

    Other facts about a preset -- the resolved template filename, for one --
    are things an operator standing at one station cannot act on. The line is
    absent from their page, not merely styled away: server-side rendering, so
    the stylesheet is not the gate.
    """
    _, operator = _owning_client()
    operator_html = operator.get("/").text

    for preset in sp.SETUP_PRESETS:
        _, button = _card(operator_html, preset.id)
        assert not _spans(button, "preset-card__meta"), (
            f"{preset.label} still shows the operator a metadata line: "
            f"{_text(button)!r}"
        )


# ------------------------------------------------------------------ ownership


def test_a_browser_that_never_loaded_the_page_cannot_select():
    """A mutating request must not implicitly become the owner, or the first
    stray POST takes the rig with nobody asking."""
    app = create_app()
    resp = TestClient(app).post(
        "/setup/preset", data={"preset_id": "bimanual_v4"}, follow_redirects=False
    )
    assert resp.status_code == 409
    assert app.state.rig.draft.preset_id is None


def test_a_second_read_only_tab_cannot_change_the_selection():
    app, owner = _owning_client()
    owner.post("/setup/preset", data={"preset_id": "bimanual_v4"})

    onlooker = TestClient(app)
    page = onlooker.get("/").text
    # `disabled` is a boolean attribute, so HTMLParser gives it the value None
    # -- membership is the only honest check here.
    assert all("disabled" in button for _, button in _cards(page)), (
        "a read-only tab is offered live cards"
    )

    resp = onlooker.post(
        "/setup/preset", data={"preset_id": "bimanual_v4_vive"}, follow_redirects=False
    )
    assert resp.status_code == 409
    assert app.state.rig.draft.preset_id == "bimanual_v4"


def test_the_read_only_tab_is_offered_an_explicit_way_to_take_over():
    """409 with no affordance bricks the tab. Control moves only through a
    deliberate action -- never silently."""
    app, owner = _owning_client()
    owner.post("/setup/preset", data={"preset_id": "bimanual_v4"})

    onlooker = TestClient(app)
    page = _Page()
    page.feed(onlooker.get("/").text)
    takeovers = [f for f in page.forms if f["action"] == "/take-control"]
    assert takeovers and takeovers[0]["method"] == "post"

    onlooker.post("/take-control", follow_redirects=False)
    assert (
        onlooker.post(
            "/setup/preset",
            data={"preset_id": "bimanual_v4_vive"},
            follow_redirects=False,
        ).status_code
        == 303
    )
    assert app.state.rig.draft.preset_id == "bimanual_v4_vive"

    # And the displaced tab finds out on its next mutating call.
    assert (
        owner.post(
            "/setup/preset", data={"preset_id": "bimanual_v4"}, follow_redirects=False
        ).status_code
        == 409
    )
    assert app.state.rig.draft.preset_id == "bimanual_v4_vive"


def test_reloading_the_page_never_demotes_the_owner():
    app, owner = _owning_client()
    for _ in range(3):
        assert all(
            "disabled" not in button for _, button in _cards(owner.get("/").text)
        )
    assert (
        owner.post(
            "/setup/preset", data={"preset_id": "bimanual_v4"}, follow_redirects=False
        ).status_code
        == 303
    )
    assert app.state.rig.draft.preset_id == "bimanual_v4"


# --------------------------------------------------------------- accessibility


def test_every_card_is_a_focusable_control_with_an_accessible_name():
    """A div with an onclick is not a control: it is not tab-reachable, has no
    role, and announces nothing."""
    _, client = _owning_client()
    cards = _cards(client.get("/").text)
    assert cards, "no cards at all -- this test would otherwise pass vacuously"
    for _, button in cards:
        assert "tabindex" not in button, "a card is removed from the tab order"
        assert _text(button) or button.get("aria-label"), "a card has no name"

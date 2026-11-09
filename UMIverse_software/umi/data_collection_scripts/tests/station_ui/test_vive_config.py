# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Vive endpoint and tracker keys, from the form to the file and back.

Two mistakes this section guards against, both silent until a whole shift has
been recorded: a station left pointed at the template's default bridge instead
of its own, and a bridge that renamed its pucks between days, so
`left_tracking_pose` records the right arm's puck. Nothing downstream can detect
the second -- both arms show plausible motion -- so it has to be caught at the
bench.
"""

import pathlib
import shutil
from html.parser import HTMLParser

import pytest

yaml = pytest.importorskip("yaml")

from fastapi.testclient import TestClient  # noqa: E402

from station_ui import config_form  # noqa: E402
from station_ui import settings as station_settings  # noqa: E402
from station_ui.app import create_app  # noqa: E402
from station_ui.config_form import (  # noqa: E402
    VIVE_ENDPOINT_FIELD,
    absorb,
    hydrate,
    vive_key_field,
)
from station_ui.core import vive_probe  # noqa: E402
from station_ui.core.config_assembler import (  # noqa: E402
    TemplateRequirements,
    ViveSource,
)
from station_ui.session import SessionDraft, device_fingerprint  # noqa: E402
from station_ui.settings import Settings  # noqa: E402
from umi.common.tasks import task_registry_store  # noqa: E402

VIVE_REQUIREMENTS = TemplateRequirements(
    gripper_arms=("left", "right"),
    vive_sources=(
        ViveSource("left_tracking_pose", "left", "tcp://t:5555", "left_knee"),
        ViveSource("right_tracking_pose", "right", "tcp://t:5555", "right_knee"),
    ),
)


def test_a_fresh_draft_defers_to_the_template():
    """ "" is "the template's value stands", not "blank"."""
    draft = SessionDraft(preset_id="bimanual_v4_vive")
    assert draft.vive_endpoint == ""
    assert draft.vive_keys == {}


def test_the_endpoint_is_part_of_which_hardware_opens():
    """Repointing the bridge opens a different server, so the camera check and
    the gripper calibration recorded against the old one no longer describe
    this rig. That is exactly the test `device_fingerprint` applies -- and why
    `reference_override`, which opens nothing different, is excluded from it."""
    a = SessionDraft(preset_id="bimanual_v4_vive")
    b = SessionDraft(preset_id="bimanual_v4_vive", session_uid=a.session_uid)
    b.vive_endpoint = "tcp://10.20.30.40:5555"

    assert device_fingerprint(a) != device_fingerprint(b)


def test_swapping_a_tracker_key_changes_the_fingerprint():
    a = SessionDraft(preset_id="bimanual_v4_vive")
    a.vive_keys = {"left": "left_knee", "right": "right_knee"}
    b = SessionDraft(preset_id="bimanual_v4_vive", session_uid=a.session_uid)
    b.vive_keys = {"left": "left_shoulder", "right": "right_knee"}

    assert device_fingerprint(a) != device_fingerprint(b)


def test_the_fingerprint_does_not_depend_on_key_insertion_order():
    a = SessionDraft(preset_id="bimanual_v4_vive")
    a.vive_keys = {"left": "left_knee", "right": "right_knee"}
    b = SessionDraft(preset_id="bimanual_v4_vive", session_uid=a.session_uid)
    b.vive_keys = {"right": "right_knee", "left": "left_knee"}

    assert device_fingerprint(a) == device_fingerprint(b)


def test_a_posted_endpoint_and_keys_reach_the_draft():
    draft = SessionDraft(preset_id="bimanual_v4_vive")
    absorb(
        draft,
        {
            VIVE_ENDPOINT_FIELD: "tcp://10.20.30.40:5555",
            vive_key_field("left"): "left_shoulder",
            vive_key_field("right"): "right_shoulder",
        },
        VIVE_REQUIREMENTS,
    )

    assert draft.vive_endpoint == "tcp://10.20.30.40:5555"
    assert draft.vive_keys == {"left": "left_shoulder", "right": "right_shoulder"}


def test_an_absent_field_leaves_the_draft_alone():
    """The confirm-regenerate step posts a short form. Treating its missing
    fields as cleared would wipe the rig the operator just finished
    configuring -- the same distinction every other slot draws."""
    draft = SessionDraft(preset_id="bimanual_v4_vive")
    draft.vive_endpoint = "tcp://kept:5555"
    draft.vive_keys = {"left": "left_knee"}

    absorb(draft, {"confirm": "yes"}, VIVE_REQUIREMENTS)

    assert draft.vive_endpoint == "tcp://kept:5555"
    assert draft.vive_keys == {"left": "left_knee"}


def test_a_blank_key_clears_it_back_to_the_template_default():
    draft = SessionDraft(preset_id="bimanual_v4_vive")
    draft.vive_keys = {"left": "left_shoulder"}

    absorb(draft, {vive_key_field("left"): ""}, VIVE_REQUIREMENTS)

    assert draft.vive_keys == {}


def test_whitespace_around_a_typed_key_is_stripped():
    draft = SessionDraft(preset_id="bimanual_v4_vive")
    absorb(draft, {vive_key_field("left"): "  left_shoulder  "}, VIVE_REQUIREMENTS)
    assert draft.vive_keys == {"left": "left_shoulder"}


def test_a_blank_endpoint_returns_to_the_templates_bridge():
    """ "" is the fallback, so emptying the box is how an operator says "record
    against whatever this template ships with". A box half-cleared to " " has to
    mean the same thing: `_apply_vive` writes any truthy value straight into
    `params["endpoint"]`, so one surviving space would replace the template's
    working address with one no recorder can connect to."""
    draft = SessionDraft(preset_id="bimanual_v4_vive")
    draft.vive_endpoint = "tcp://10.20.30.40:5555"

    absorb(draft, {VIVE_ENDPOINT_FIELD: ""}, VIVE_REQUIREMENTS)
    assert draft.vive_endpoint == ""

    draft.vive_endpoint = "tcp://10.20.30.40:5555"
    absorb(draft, {VIVE_ENDPOINT_FIELD: "   "}, VIVE_REQUIREMENTS)
    assert draft.vive_endpoint == ""


def test_whitespace_around_a_typed_endpoint_is_stripped():
    """The endpoint is the one field a station actually retypes, and an
    address pasted out of a chat message carries a trailing newline. It is
    truthy, so it reaches the written config, and the recorder is then handed a
    bridge that differs from the right one by a character nobody can see."""
    draft = SessionDraft(preset_id="bimanual_v4_vive")
    absorb(
        draft, {VIVE_ENDPOINT_FIELD: "  tcp://10.20.30.40:5555\n"}, VIVE_REQUIREMENTS
    )
    assert draft.vive_endpoint == "tcp://10.20.30.40:5555"


def test_an_arm_the_new_template_no_longer_declares_is_dropped():
    """A preset switch changes the template. A key for an arm the new
    one does not declare would be written nowhere and shown nowhere."""
    draft = SessionDraft(preset_id="bimanual_v4_vive")
    draft.vive_keys = {"left": "left_knee", "middle": "stale"}

    absorb(draft, {vive_key_field("left"): "left_knee"}, VIVE_REQUIREMENTS)

    assert draft.vive_keys == {"left": "left_knee"}


def test_a_hand_posted_vive_field_is_ignored_on_a_non_vive_rig():
    """Every value is taken slot by slot from what the template declares, so a
    hand-crafted POST cannot fill a field the rig does not have."""
    draft = SessionDraft(preset_id="bimanual_v4")
    absorb(
        draft,
        {
            VIVE_ENDPOINT_FIELD: "tcp://attacker:1",
            vive_key_field("left"): "whatever",
        },
        TemplateRequirements(gripper_arms=("left", "right")),
    )

    assert draft.vive_endpoint == ""
    assert draft.vive_keys == {}


# ------------------------------------------------------ the page an operator sees

REPO = pathlib.Path(__file__).resolve().parents[2]

# Assert the Vive defaults visible to the operator, independently of
# reading the template used to generate them.
DEFAULT_BRIDGE = "tcp://127.0.0.1:5555"

# Enough of a rig for Apply to write a file. Everything but the Vive fields, so
# each test adds only the part it is about.
APPLY = {
    "config_form": "1",
    "task_name": "plane_assembly_v9",
    "operator_name": "Test Operator",
    "gripper.left": "UMI-V4-0001",
    "gripper.right": "UMI-V4-0002",
    "camera.left_umi_camera": "oak-FFC-B",
    "camera.right_umi_camera": "oak-FFC-A",
    "camera.left_trigger_camera": "4",
    "camera.right_trigger_camera": "2",
    "optional.head_oak_camera.device": "oak-W97-51",
}

# What htmx puts on every request it makes. The probe answers a fragment to
# these and a redirect to everything else, so which of the two a test asserts on
# is decided here.
HX = {"HX-Request": "true"}


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    """A task list that does not depend on this machine's registry file.

    Patched at the store so the real registry code still runs. Apply refuses an
    unknown task name, and a station that cannot read the list must not be told
    its name is wrong -- so a test asserting on what Apply wrote needs the name
    it posts to be one the registry positively knows.
    """
    monkeypatch.setattr(
        task_registry_store,
        "get_all_tasks",
        lambda: {
            "plane_assembly_v9": {
                "name": "plane_assembly_v9",
                "short_description": "Assemble the plane",
                "mode": "bimanual",
                "HW_version": "v3",
                "training_group": "plane_assembly_v9",
            }
        },
    )


@pytest.fixture(autouse=True)
def no_live_bridge(monkeypatch):
    """Nothing in this suite may open a socket to a real bridge.

    There is no socket guard in `tests/conftest.py`, and the address the
    template names may be a live server that stations are recording against. A
    test that wants a probe replaces this with `_probing(...)`.
    """
    monkeypatch.setattr(
        vive_probe,
        "probe",
        lambda endpoint, **kw: pytest.fail(f"probed {endpoint} without being asked"),
    )


def _probing(monkeypatch, **result):
    """Answer the next probe with *result*, and record what it was asked."""
    asked = []

    def fake(endpoint, **kw):
        asked.append(endpoint)
        return vive_probe.ProbeResult(**result)

    monkeypatch.setattr(vive_probe, "probe", fake)
    return asked


def _repo(tmp_path):
    """A repo root of its own, so assembly does not write into the checkout."""
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    # `generated/` deliberately not copied -- see tests/station_ui/test_config_page.py.
    shutil.copytree(
        REPO / "config" / "data_recording",
        root / "config" / "data_recording",
        ignore=shutil.ignore_patterns("generated"),
    )
    shutil.copytree(REPO / "config" / "data_check", root / "config" / "data_check")
    return root


def _client(repo, preset="bimanual_v4_vive"):
    client = TestClient(create_app(Settings(repo_root=repo)))
    client.get("/")
    client.post("/setup/preset", data={"preset_id": preset})
    return client


class _Controls(HTMLParser):
    """The text boxes, datalist options and per-element text of a rendered page.

    Structural rather than substring matching, for the reason the suite next
    door states: `"left_knee" in body` passes on a page that only mentions the
    name in a comment, and what has to be true is that the box holds it -- and
    that a warning about the left puck is attached to the left row.
    """

    # No end tag, so the id stack must not wait for one.
    _VOID = frozenset({"br", "hr", "img", "input", "link", "meta", "option", "source"})

    def __init__(self):
        super().__init__()
        self.values: dict[str, str] = {}
        self.options: dict[str, list[str]] = {}
        self.text: dict[str, str] = {}
        self._list: str | None = None
        # One entry per open element, id or None, so text is credited only to
        # the elements that really contain it.
        self._open: list[str | None] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "input" and attrs.get("name"):
            self.values[attrs["name"]] = attrs.get("value") or ""
        elif tag == "datalist":
            self._list = attrs.get("id") or ""
            self.options.setdefault(self._list, [])
        elif tag == "option" and self._list is not None:
            self.options[self._list].append(attrs.get("value") or "")
        if tag not in self._VOID:
            self._open.append(attrs.get("id"))

    def handle_endtag(self, tag):
        if tag == "datalist":
            self._list = None
        if tag not in self._VOID and self._open:
            self._open.pop()

    def handle_data(self, data):
        for element in self._open:
            if element:
                self.text[element] = self.text.get(element, "") + data


def _controls(html: str) -> _Controls:
    parser = _Controls()
    parser.feed(html)
    return parser


def _offered_keys(html: str) -> set[str]:
    """Every tracker name the key pickers offer, whichever list they read."""
    parsed = _controls(html)
    return {
        value
        for list_id, options in parsed.options.items()
        if list_id.startswith("vive-keys")
        for value in options
    }


def _as_submitted(client):
    """`APPLY` plus the Vive fields exactly as the page renders them.

    Read off the page rather than hardcoded, because a browser posts back what
    is in the boxes and the round trip is the thing under test: what the form
    submits has to be what the file holds has to be what `hydrate` reads back.
    """
    boxes = _controls(client.get("/config").text).values
    return {
        **APPLY,
        **{
            name: boxes[name]
            for name in (
                VIVE_ENDPOINT_FIELD,
                vive_key_field("left"),
                vive_key_field("right"),
            )
        },
    }


def _written(preset="bimanual_v4_vive"):
    path = (
        station_settings.state_dir_from_environment()
        / "data_recording"
        / f"{preset}.yaml"
    )
    return yaml.safe_load(path.read_text())


def _vive_params(written) -> dict:
    return {
        entry["name"]: entry["params"]
        for entry in written["numeric_sources"]
        if entry.get("type") == "vive_tracker"
    }


def test_a_vive_rig_is_offered_a_server_box_and_a_key_box_per_arm(tmp_path):
    boxes = _controls(_client(_repo(tmp_path)).get("/config").text).values

    assert VIVE_ENDPOINT_FIELD in boxes
    assert vive_key_field("left") in boxes
    assert vive_key_field("right") in boxes


def test_a_non_vive_rig_is_offered_none_of_them(tmp_path):
    body = _client(_repo(tmp_path), preset="bimanual_v4").get("/config").text

    assert VIVE_ENDPOINT_FIELD not in _controls(body).values
    assert "vive-section" not in body


def test_the_boxes_start_at_the_templates_values(tmp_path):
    """A station using the template's bridge should need zero clicks, so the
    boxes show what the rig would record with if the operator touched nothing.
    The endpoint box shows the bare host -- the default port it implies is not
    repeated back."""
    boxes = _controls(_client(_repo(tmp_path)).get("/config").text).values

    assert boxes[VIVE_ENDPOINT_FIELD] == "127.0.0.1"
    assert boxes[vive_key_field("left")] == "left_knee"
    assert boxes[vive_key_field("right")] == "right_knee"


def test_showing_the_templates_values_does_not_claim_the_operator_chose_them(tmp_path):
    """The fallback is in the view. Writing it into the draft would erase the
    difference between "not chosen" and "typed the template's value", and that
    difference is what leaves an untouched rig recording exactly as before."""
    client = _client(_repo(tmp_path))
    client.get("/config")

    draft = client.app.state.rig.draft
    assert draft.vive_endpoint == ""
    assert draft.vive_keys == {}


def test_the_page_never_probes_the_bridge_on_its_own(tmp_path):
    """The bridge serves from one zmq.REP thread shared by every station. A page
    that probed on load, or on a timer, would make every open browser tab a
    permanent client of it -- so the probe fires on the button and nowhere else.

    `no_live_bridge` fails the test if anything reaches `probe`, which is what
    makes this an assertion rather than a comment."""
    client = _client(_repo(tmp_path))

    client.get("/config")
    client.post("/config/apply", data={**APPLY, VIVE_ENDPOINT_FIELD: DEFAULT_BRIDGE})
    client.get("/config")


def test_pressing_test_connection_shows_the_keys_the_bridge_publishes(
    tmp_path, monkeypatch
):
    asked = _probing(
        monkeypatch,
        ok=True,
        keys=("left_shoulder", "right_shoulder"),
        tracked=frozenset({"left_shoulder"}),
        rtt_ms=48.0,
    )
    client = _client(_repo(tmp_path))
    response = client.post(
        "/config/vive/probe",
        data={"config_form": "1", VIVE_ENDPOINT_FIELD: "tcp://10.20.30.40:5555"},
        headers=HX,
    )

    assert response.status_code == 200
    # The address in the box, not the template's -- the operator is testing what
    # they just typed.
    assert asked == ["tcp://10.20.30.40:5555"]
    assert _offered_keys(response.text) == {"left_shoulder", "right_shoulder"}


def test_what_the_bridge_serves_lands_beside_the_button_that_asked(
    tmp_path, monkeypatch
):
    """The headline goes next to Test connection, the way the device scan's
    count sits next to Refresh devices.

    As a sentence under the Server box, several fields below the control that
    produced it, the answer would appear somewhere other than where the operator
    pressed the button.
    """
    _probing(
        monkeypatch,
        ok=True,
        keys=("left_shoulder", "right_shoulder"),
        tracked=frozenset({"left_shoulder"}),
        rtt_ms=48.0,
    )
    client = _client(_repo(tmp_path))

    body = client.post(
        "/config/vive/probe",
        data={"config_form": "1", VIVE_ENDPOINT_FIELD: DEFAULT_BRIDGE},
        headers=HX,
    ).text

    header = body.split('<div class="card')[0]
    assert "2 trackers found" in header
    # The timing stays as the detail line; the count does not appear twice.
    assert "48 ms" in body
    assert body.count("2 trackers found") == 1


def test_a_bridge_serving_no_pucks_is_not_reported_as_a_clean_answer(
    tmp_path, monkeypatch
):
    """A live server with nothing paired to it answers fine and publishes
    nothing, which is a problem to fix before recording -- not the green tick
    that a bare "connected" would read as."""
    _probing(monkeypatch, ok=True, keys=(), rtt_ms=9.0)
    client = _client(_repo(tmp_path))

    header = client.post(
        "/config/vive/probe",
        data={"config_form": "1", VIVE_ENDPOINT_FIELD: DEFAULT_BRIDGE},
        headers=HX,
    ).text.split('<div class="card')[0]

    assert "No trackers" in header
    assert "badge--warn" in header


def test_a_bridge_that_never_answered_says_so_beside_the_button_too(
    tmp_path, monkeypatch
):
    _probing(
        monkeypatch, ok=False, error="tcp://nowhere:5555 did not answer within 1s."
    )
    client = _client(_repo(tmp_path))

    header = client.post(
        "/config/vive/probe",
        data={"config_form": "1", VIVE_ENDPOINT_FIELD: DEFAULT_BRIDGE},
        headers=HX,
    ).text.split('<div class="card')[0]

    assert "No answer" in header


def test_nothing_is_claimed_beside_the_button_before_it_is_pressed(tmp_path):
    """This page never asks the bridge on its own, so before the first press
    there is no answer to report -- and an empty badge would read as one."""
    client = _client(_repo(tmp_path))

    header = client.get("/config").text.split('<div class="card')[0]

    assert "trackers found" not in header
    assert "No answer" not in header


def test_the_button_is_dressed_like_the_other_control_that_asks_the_hardware(
    tmp_path,
):
    """`.btn--accent` -- the cyan outline Refresh devices wears. Both are "go and
    ask the hardware", neither is the page's primary action (Apply is), and a
    plain `.btn` would look like an ordinary link among the fields."""
    client = _client(_repo(tmp_path))

    body = client.get("/config").text
    button = body.split('formaction="/config/vive/probe"')[0].rsplit("<button", 1)[1]

    assert "btn--accent" in button


def test_the_button_greys_out_for_the_life_of_the_probe(tmp_path):
    """The probe blocks on a socket for up to a second, and a slow or distant
    link is exactly when it will. Without an affordance the press looks like
    nothing happened, and the operator presses again -- onto a bridge that is
    one `zmq.REP` thread shared by every station on the bench.

    `hx-disabled-elt` is the whole mechanism: htmx sets `disabled` for the life
    of the request and the console's existing `.btn:disabled` greys the text.
    No new CSS, deliberately -- `base.html` loads `/static/app.css` with no
    cache-busting query and `StaticFiles` sends no `Cache-Control`, so a newly
    added rule reaches a browser that has loaded the page before only after a
    hard refresh. An affordance that depends on new CSS is an affordance that is
    not there.
    """
    client = _client(_repo(tmp_path))

    body = client.get("/config").text

    assert 'hx-disabled-elt="this"' in body


def test_the_button_carries_one_label_not_two(tmp_path):
    """A swapped-in second label needs a stylesheet rule to hide the first, and
    that rule cannot be relied on to arrive (see above) -- without it the button
    reads "Test connection Testing…", both spans rendering as plain inline
    text."""
    client = _client(_repo(tmp_path))

    body = client.get("/config").text

    assert "Testing…" not in body
    assert "btn__busy" not in body
    assert body.count("Test connection") == 1


def test_a_key_the_bridge_does_not_publish_is_flagged_but_never_blocks_apply(
    tmp_path, monkeypatch
):
    """Apply has to stay possible offline -- a bench configured before the
    bridge is up is normal -- so a name the probe did not see is a warning
    beside the box, and the hard gate is the calibration tracker check."""
    _probing(monkeypatch, ok=True, keys=("left_shoulder",), rtt_ms=12.0)
    client = _client(_repo(tmp_path))
    fragment = _controls(
        client.post(
            "/config/vive/probe",
            data={
                "config_form": "1",
                VIVE_ENDPOINT_FIELD: DEFAULT_BRIDGE,
                vive_key_field("left"): "left_knee",
                vive_key_field("right"): "right_knee",
            },
            headers=HX,
        ).text
    )

    # On the row that names it, not somewhere on the page: a message the
    # operator has to match back to a box by eye is half an answer.
    assert "publishes" in fragment.text["message-vive-key-left"]
    # And the name they typed is left exactly as they typed it.
    assert fragment.values[vive_key_field("left")] == "left_knee"

    client.post(
        "/config/apply",
        data={
            **APPLY,
            VIVE_ENDPOINT_FIELD: DEFAULT_BRIDGE,
            vive_key_field("left"): "left_knee",
        },
    )
    assert _vive_params(_written())["left_tracking_pose"]["tracker_key"] == "left_knee"


def test_a_bridge_that_does_not_answer_says_so_without_blocking_apply(
    tmp_path, monkeypatch
):
    _probing(
        monkeypatch, ok=False, error="tcp://nowhere:5555 did not answer within 1s."
    )
    client = _client(_repo(tmp_path))
    response = client.post(
        "/config/vive/probe",
        data={"config_form": "1", VIVE_ENDPOINT_FIELD: "tcp://nowhere:5555"},
        headers=HX,
    )

    assert response.status_code == 200
    assert "did not answer" in response.text
    # And the section is still there to try again with.
    assert VIVE_ENDPOINT_FIELD in _controls(response.text).values


def test_the_probe_fragment_never_swaps_the_section_away(tmp_path, monkeypatch):
    """An outerHTML swap of an empty body deletes #vive-section until a full
    reload. `_requirements` returns a bare TemplateRequirements() on an
    unreadable template, so this is reachable in production, not hypothetical."""
    _probing(monkeypatch, ok=False)
    client = _client(_repo(tmp_path), preset="bimanual_v4")
    response = client.post(
        "/config/vive/probe",
        data={"config_form": "1", VIVE_ENDPOINT_FIELD: "tcp://x:1"},
        headers=HX,
    )

    # Nothing to render for a non-Vive rig: answer 204, which htmx leaves alone,
    # rather than 200 with an empty body, which htmx swaps in.
    assert response.status_code == 204


def test_the_button_still_works_with_the_vendored_script_blocked(tmp_path, monkeypatch):
    """It is a real submit aimed with `formaction`, so a browser with htmx
    blocked posts the whole form here. A bare <section> answered at 200 would be
    rendered as the entire document; the answer is the page, like every other
    button on this form."""
    _probing(monkeypatch, ok=True, keys=("left_shoulder",), rtt_ms=7.0)
    client = _client(_repo(tmp_path))
    response = client.post(
        "/config/vive/probe",
        data={"config_form": "1", VIVE_ENDPOINT_FIELD: "tcp://10.20.30.40:5555"},
    )

    assert response.status_code == 200
    assert response.url.path == "/config"
    assert "left_shoulder" in _offered_keys(response.text)


def test_a_probe_of_one_bridge_is_not_shown_against_another(tmp_path, monkeypatch):
    """The box stays editable after the press. Pucks from the bridge that was
    in it are worse than no list at all -- they would flag the operator's keys
    against a server they have already navigated away from."""
    _probing(monkeypatch, ok=True, keys=("left_shoulder",), rtt_ms=9.0)
    client = _client(_repo(tmp_path))
    client.post(
        "/config/vive/probe",
        data={"config_form": "1", VIVE_ENDPOINT_FIELD: "tcp://10.20.30.40:5555"},
        headers=HX,
    )

    # The operator retypes the address and the page renders again -- pressing
    # Apply, say -- without pressing Test connection a second time.
    client.post("/config/apply", data={**APPLY, VIVE_ENDPOINT_FIELD: DEFAULT_BRIDGE})
    body = client.get("/config").text

    assert _offered_keys(body) == set()
    assert not _controls(body).text.get("message-vive-key-left")


def test_the_probe_does_not_discard_a_half_filled_form(tmp_path, monkeypatch):
    _probing(monkeypatch, ok=True)
    client = _client(_repo(tmp_path))
    client.post(
        "/config/vive/probe",
        data={
            "config_form": "1",
            VIVE_ENDPOINT_FIELD: "tcp://10.20.30.40:5555",
            "task_name": "plane_assembly_v9",
            "operator_name": "Test Operator",
        },
        headers=HX,
    )

    draft = client.app.state.rig.draft
    assert draft.task_name == "plane_assembly_v9"
    assert draft.operator_name == "Test Operator"
    assert draft.vive_endpoint == "tcp://10.20.30.40:5555"


def test_apply_writes_the_operators_endpoint_and_keys(tmp_path):
    """The whole point, end to end: what the operator typed is what the recorder
    opens."""
    client = _client(_repo(tmp_path))
    client.post(
        "/config/apply",
        data={
            **APPLY,
            VIVE_ENDPOINT_FIELD: "tcp://10.20.30.40:5555",
            vive_key_field("left"): "left_shoulder",
            vive_key_field("right"): "right_shoulder",
        },
    )

    params = _vive_params(_written())
    assert params["left_tracking_pose"]["endpoint"] == "tcp://10.20.30.40:5555"
    assert params["left_tracking_pose"]["tracker_key"] == "left_shoulder"
    assert params["right_tracking_pose"]["tracker_key"] == "right_shoulder"


def test_two_arms_naming_one_puck_are_refused_on_the_rows_that_name_it(tmp_path):
    """Two arms on one tracker record one pose twice, and every pose check
    passes on both arms because both traces are real -- so `assemble` refuses
    it. A refusal the operator only sees as a banner leaves them hunting for
    which of the two boxes to change."""
    client = _client(_repo(tmp_path))
    response = client.post(
        "/config/apply",
        data={
            **APPLY,
            vive_key_field("left"): "left_knee",
            vive_key_field("right"): "left_knee",
        },
    )

    assert response.status_code == 400
    page = _controls(response.text)
    assert "left_knee" in page.text[config_form.error_id(vive_key_field("left"))]
    assert "left_knee" in page.text[config_form.error_id(vive_key_field("right"))]


def test_the_untouched_row_is_named_too_when_only_one_arm_was_retyped(tmp_path):
    """The likelier half of the mistake: the operator retypes one arm and
    collides with the key the other arm was already holding from the template.
    Matching on what was typed would leave the untouched row -- which is half
    the answer to "which two rows" -- with no message at all."""
    client = _client(_repo(tmp_path))
    response = client.post(
        "/config/apply", data={**APPLY, vive_key_field("left"): "right_knee"}
    )

    assert response.status_code == 400
    page = _controls(response.text)
    assert page.text.get(config_form.error_id(vive_key_field("left")))
    assert page.text.get(config_form.error_id(vive_key_field("right")))


def test_reopening_an_applied_session_does_not_expire_its_checks(tmp_path):
    """Why the boxes render the resolved value rather than blank.

    A console restart reloads the draft from the generated config
    (`config_form.hydrate`). `device_fingerprint` expires the device test and
    the gripper calibration when the rig changes, so the round trip has to be
    exact: what the page submitted is what the file holds is what hydrate reads
    back. A box that rendered blank would post a cleared field, and the
    reopened draft would then disagree with the applied one over a rig nobody
    touched -- costing a fifteen-minute camera check for nothing."""
    client = _client(_repo(tmp_path))
    client.post("/config/apply", data=_as_submitted(client))
    applied = client.app.state.rig.draft

    restored = SessionDraft(
        preset_id=applied.preset_id, session_uid=applied.session_uid
    )
    hydrate(restored, _written())

    assert device_fingerprint(restored) == device_fingerprint(applied)


def test_an_untouched_rig_still_records_exactly_what_the_template_said(tmp_path):
    """The other half of the fallback rule. A station that never opens this
    section must write the template's own bridge and pucks, unchanged."""
    client = _client(_repo(tmp_path))
    client.post("/config/apply", data=APPLY)

    params = _vive_params(_written())
    assert params["left_tracking_pose"]["endpoint"] == DEFAULT_BRIDGE
    assert params["left_tracking_pose"]["tracker_key"] == "left_knee"
    assert params["right_tracking_pose"]["tracker_key"] == "right_knee"

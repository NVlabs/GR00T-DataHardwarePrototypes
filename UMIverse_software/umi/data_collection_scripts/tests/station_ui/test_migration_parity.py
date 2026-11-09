# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device picking, name resolution and the alignment block on /config.

Each block names the behaviour it protects. Nothing here asserts on source
text. Detected devices are read out of the `<datalist>` a browser would offer,
an ambiguous name is judged by what the page says next to the field, and the
alignment keys are read back out of the file that was written -- not out of
the assembler that wrote it.

The picker is one control: an `<input list=...>` over a datalist. Choosing a
detected device and typing one by hand are therefore the same act on the same
field -- which is what these tests do -- and both halves are asserted: the
offered list has to carry what the scan found, and the box has to stay
writable.

**No test in this file opens hardware.** Every scan is a fake child named by
`devices.scan_argv()`, the single command every page that scans spawns; the
device registry is replaced at `station_ui.core.device_detector`.
"""

import json
import pathlib
import shutil
import sys
import time
from html.parser import HTMLParser

import pytest
from fastapi.testclient import TestClient

yaml = pytest.importorskip("yaml")


from station_ui import devices as devices_core  # noqa: E402
from station_ui import jobs as job_control  # noqa: E402
from station_ui import scan_worker, settings  # noqa: E402
from station_ui.app import create_app  # noqa: E402
from station_ui.core import device_detector, setup_presets  # noqa: E402
from station_ui.routes import config as config_routes  # noqa: E402
from station_ui.settings import Settings  # noqa: E402
from umi.common.tasks import task_registry_store  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]

FAKE_SCAN_S = 5.0
# Bound request latency relative to the simulated scan duration to
# distinguish asynchronous launch from blocking without depending on host speed.
RESPONSE_BUDGET_S = FAKE_SCAN_S / 5
# Allow cold template compilation while still requiring page rendering
# to finish well before the simulated scan.
PAGE_BUDGET_S = FAKE_SCAN_S / 2
DEADLINE_S = 30.0

# Use two registry names with the same numeric suffix to test ambiguous
# digit matching; a longer suffix still uniquely identifies one device.
OAK_A = "oak-W97-01"
OAK_B = "oak-FFC-01"
OAK_A_ID = "1944301000000001"
OAK_B_ID = "1944301000000002"
IPHONE = "iphone15p-04"
IPHONE_ID = "00008130-000000000001"

TASKS = {
    "plane_assembly_v9": {
        "name": "plane_assembly_v9",
        "short_description": "Assemble the plane",
        "mode": "bimanual",
        "HW_version": "v2.1",
        "training_group": "plane_assembly_v9",
    },
}


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    """A deterministic task registry and a deterministic device registry.

    Patched at the module the route reads through, not at the function it
    imported, so the substitution is actually in force -- `_REGISTRY_LOOKUP`
    stores attribute names for exactly this reason.
    """
    monkeypatch.setattr(task_registry_store, "get_all_tasks", lambda: dict(TASKS))
    monkeypatch.setattr(
        device_detector, "get_available_oak_cameras", lambda: [OAK_A, OAK_B]
    )
    monkeypatch.setattr(device_detector, "get_available_iphones", lambda: [IPHONE])
    monkeypatch.setattr(
        config_routes,
        "list_umi_gripper_device_names_for_pipeline",
        lambda pipeline: ["UMI-V4-0001", "UMI-V4-0002"],
    )


# --------------------------------------------------------------------- fakes


def found_record(kind, device_id, registry_name=None, state="available", **metadata):
    record = {
        "kind": kind,
        "device_id": device_id,
        "registry_name": registry_name,
        "state": state,
        "metadata": dict(metadata),
    }
    return record


def fake_scan_argv(*, sleep=0.0, found=(), exit_code=0):
    """A child that behaves like the scan, with no hardware anywhere near it.

    The result line is built from `scan_worker.RESULT_PREFIX`, so a change to
    the wire format between the worker and the page breaks these tests rather
    than quietly producing a page with empty pickers.
    """
    payload = scan_worker.RESULT_PREFIX + json.dumps({"devices": list(found)})
    source = (
        "import sys, time\n"
        "print('[scan] starting', flush=True)\n"
        f"time.sleep({float(sleep)!r})\n"
        f"print({payload!r}, flush=True)\n"
        f"sys.exit({int(exit_code)!r})\n"
    )
    return [sys.executable, "-c", source]


def use_argv(monkeypatch, argv):
    """Replace the scan child for every page at once.

    Every page that scans calls `devices.scan_argv()` rather than declaring its
    own, precisely so a single patch cannot leave one page spawning the real one.
    """
    monkeypatch.setattr(devices_core, "scan_argv", lambda: list(argv))


def wait_for_scan(app, *, deadline=DEADLINE_S):
    job = devices_core.current_job(app)
    assert job is not None, "no scan was started"
    end = time.monotonic() + deadline
    while not job.finished and time.monotonic() < end:
        time.sleep(0.02)
    assert job.finished, f"the fake scan never ended: {job.state}"
    return job


# ------------------------------------------------------------------- fixtures


def _repo(tmp_path) -> pathlib.Path:
    """A repo root of its own, so assembly never touches the checkout."""
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    shutil.copytree(
        REPO / "config" / "data_recording",
        root / "config" / "data_recording",
        ignore=shutil.ignore_patterns("generated"),
    )
    shutil.copytree(REPO / "config" / "data_check", root / "config" / "data_check")
    return root


def _client(repo, *, preset="bimanual_v4"):
    client = TestClient(create_app(Settings(repo_root=repo)))
    client.get("/")
    client.post("/setup/preset", data={"preset_id": preset})
    return client


@pytest.fixture
def repo(tmp_path):
    return _repo(tmp_path)


@pytest.fixture
def client(repo):
    return _client(repo)


# --------------------------------------------------------------------- parsing


class _Page(HTMLParser):
    """Forms and controls as a browser would submit them, plus selects.

    Structural rather than string matching: a `<select>` that renders but is
    never submitted, and an option a browser would not offer, are both
    invisible to a substring assertion.
    """

    _CONTROLS = ("input", "select", "textarea", "button")
    _VOID = frozenset({"input", "br", "hr", "img", "link", "meta", "source"})
    # Never rendered as text. A script's JSON payload is data for the page's own
    # code -- the gripper diagrams embed every connected device there -- and
    # reading it as body text would make the page appear to say it.
    _NOT_TEXT = frozenset({"script", "style"})

    def __init__(self):
        super().__init__()
        self.forms = []
        self.datalists = {}
        self.attrs_by_id = {}
        self.body = ""
        self._form = None
        self._select = None
        self._datalist = None
        self._not_text = 0

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        if attr.get("id"):
            self.attrs_by_id[attr["id"]] = attr
        if tag in self._NOT_TEXT:
            self._not_text += 1
            return
        if tag == "datalist":
            self._datalist = self.datalists.setdefault(attr.get("id", ""), [])
            return
        if tag == "option" and self._datalist is not None:
            # The whole option, not just its value: the label is what a browser
            # shows beside a choice, so "the picker says this one is
            # unavailable" is a claim about the label.
            self._datalist.append(attr)
            return
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
            "text": "",
            "options": [],
        }
        self._form["controls"].append(control)
        if tag == "select":
            self._select = control

    def handle_endtag(self, tag):
        if tag in self._NOT_TEXT:
            self._not_text -= 1
        elif tag == "form":
            self._form = None
        elif tag == "select":
            self._select = None
        elif tag == "datalist":
            self._datalist = None

    def handle_data(self, data):
        if self._not_text:
            return
        self.body += data
        if self._select is not None and self._select["options"]:
            self._select["options"][-1].setdefault("text", "")
            self._select["options"][-1]["text"] += data
        elif self._form and self._form["controls"]:
            last = self._form["controls"][-1]
            if last["tag"] == "button":
                last["text"] += data


def parse(html: str) -> _Page:
    page = _Page()
    page.feed(html)
    return page


def get_page(client) -> _Page:
    response = client.get("/config")
    assert response.status_code == 200, response.text
    return parse(response.text)


def controls(page) -> dict:
    return {
        control["name"]: control
        for form in page.forms
        for control in form["controls"]
        if control.get("name") and control["kind"] not in ("submit", "button")
    }


def submission(form) -> dict:
    """What a browser sends for this form with no button pressed."""
    data = {}
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


def apply_form(page):
    forms = [f for f in page.forms if f["action"] == "/config/apply"]
    assert forms, "the page renders no apply form"
    return forms[0]


def offered(page, field):
    """Every device the page offers for *field*, as a browser would list them.

    Followed through the input's own `list` attribute rather than guessed from
    the field name: a datalist the input does not point at offers nothing at
    all, and a test that looked it up by id would not notice.
    """
    control = controls(page).get(field)
    assert control is not None, f"no control named {field}"
    list_id = control.get("list")
    assert list_id, f"{field} points at no datalist"
    assert list_id in page.datalists, f"{field} points at a datalist that is not there"
    return page.datalists[list_id]


def offered_values(page, field):
    return {
        option.get("value") for option in offered(page, field) if option.get("value")
    }


def offered_text(page, field):
    """Everything the picker says about its options, labels included."""
    return " ".join(
        f"{option.get('value', '')} {option.get('label', '')}"
        for option in offered(page, field)
    )


def buttons_for(page, action):
    """Every submit control aimed at *action*, form and all."""
    return [
        (form, control)
        for form in page.forms
        for control in form["controls"]
        if control["kind"] == "submit"
        and (control.get("formaction") or form["action"]) == action
    ]


GRIPPERS = {"left": "UMI-V4-0001", "right": "UMI-V4-0002"}
# Keyed by block -- one box per physical device, named after the first stream of
# the source it fills. Every V4 block carries one stream: the wrist OAK, the USB
# trigger camera (named by its V4L2 index) and the iPhone.
CAMERAS = {
    "right_umi_camera": "oak-FFC-A",
    "left_umi_camera": "oak-FFC-B",
    "right_trigger_camera": "2",
    "left_trigger_camera": "4",
    "right_tracking_camera": "iphone-R",
    "left_tracking_camera": "iphone-L",
}
RIGHT_OAK = "camera.right_umi_camera"
LEFT_OAK = "camera.left_umi_camera"


def filled(page, *, overrides=None, head=False):
    """The apply form, filled the way an operator fills it.

    `head=False` by default: a fresh session starts with a head OAK row already
    on the page (`session.default_head_optional`), and these fixtures are rigs
    that have no head camera. The head is an Add/Remove list, so dropping the
    key does nothing -- the row lives in the draft, not in the post -- and the
    row has to be filled in or the Apply is refused by name. Filling it is the
    cheaper of the two here: it keeps this helper a pure function of the form,
    with no side-effecting POST to a second route.
    """
    data = submission(apply_form(page))
    if not head:
        for name in [k for k in data if k.startswith("optional.")]:
            if name.endswith(".device"):
                data[name] = "dev-" + name.split(".")[1]
            else:
                data.pop(name, None)
    data["task_name"] = "plane_assembly_v9"
    data["operator_name"] = "Test Operator"
    for arm, device in GRIPPERS.items():
        data[f"gripper.{arm}"] = device
    for name, device in CAMERAS.items():
        data[f"camera.{name}"] = device
    data.update(overrides or {})
    return data


def generated(repo, preset="bimanual_v4"):
    # Use the isolated station state directory for generated files, not repo.
    return settings.state_dir_from_environment() / "data_recording" / f"{preset}.yaml"


def load(path):
    return yaml.safe_load(path.read_text())


# Offer connected discovery results while retaining manual identifier entry.


def test_a_device_field_offers_the_detected_devices_and_still_takes_typed_text(
    client, repo, monkeypatch
):
    """Both halves, in one test, because either alone fails the operator.

    A picker with no text box cannot author a config before the hardware
    arrives -- which is when configs get prepped. A text box with no picker
    makes an operator hand-type a serial they can read off a screen. One control
    does both: an `<input list=...>` offers the scan and accepts anything typed.
    """
    use_argv(
        monkeypatch,
        fake_scan_argv(
            found=[
                found_record("oak", OAK_A_ID, OAK_A, sockets={"CAM_A": "IMX378"}),
                found_record("oak", OAK_B_ID, OAK_B),
            ]
        ),
    )
    client.post("/config/devices/scan")
    wait_for_scan(client.app)

    page = get_page(client)
    values = offered_values(page, RIGHT_OAK)
    assert values == {OAK_A, OAK_B}, (
        f"the scan found two OAKs and the picker offers {values}"
    )

    # ...and the box itself is still writable, and what is typed still gets in.
    typed = controls(page)[RIGHT_OAK]
    assert typed["tag"] == "input" and "readonly" not in typed
    assert typed.get("type") in (None, "text"), typed

    response = client.post(
        "/config/apply",
        data=filled(page, overrides={RIGHT_OAK: "oak-FFC-A"}),
    )
    assert response.status_code == 200, response.text
    config = load(generated(repo))
    written = {
        source.get("oak_device_name")
        for source in config["video_sources"]
        if source.get("type") == "oak_device"
    }
    assert "oak-FFC-A" in written, (
        f"a hand-typed device never reached the config: {written}"
    )


def test_choosing_a_detected_device_from_the_picker_assigns_it(
    client, repo, monkeypatch
):
    """The picker is not decoration: what it submits is what gets recorded.

    Choosing from a datalist puts the option's value in the box, so this posts
    exactly what a browser posts after a click -- and one box covers every
    stream of the board, which is why there is a single assignment here.
    """
    use_argv(
        monkeypatch,
        fake_scan_argv(found=[found_record("oak", OAK_A_ID, OAK_A)]),
    )
    client.post("/config/devices/scan")
    wait_for_scan(client.app)

    page = get_page(client)
    assert OAK_A in offered_values(page, RIGHT_OAK), "the scan result is not offered"
    data = filled(page)
    data[RIGHT_OAK] = OAK_A
    response = client.post("/config/apply", data=data)
    assert response.status_code == 200, response.text

    config = load(generated(repo))
    names = [
        source.get("oak_device_name")
        for source in config["video_sources"]
        if source.get("type") == "oak_device"
    ]
    assert OAK_A in names, f"the picked device is not in the config: {names}"


def test_an_attached_but_unregistered_device_is_selectable_by_its_raw_id(
    client, repo, monkeypatch
):
    """`device_registry` may be the small bundled snapshot, so a
    connected-and-registered rule would empty the screen on a fresh install.
    Unregistered devices stay selectable by id; the option list has to be
    widened to match, or the operator's own choice comes back "not a known
    device"."""
    use_argv(
        monkeypatch,
        fake_scan_argv(found=[found_record("oak", "1944301000000099", None)]),
    )
    client.post("/config/devices/scan")
    wait_for_scan(client.app)

    page = get_page(client)
    assert "1944301000000099" in offered_values(page, LEFT_OAK)

    data = filled(page)
    data[LEFT_OAK] = "1944301000000099"
    assert client.post("/config/apply", data=data).status_code == 200

    config = load(generated(repo))
    ids = [
        source.get("oak_device_id")
        for source in config["video_sources"]
        if source.get("type") == "oak_device"
    ]
    assert "1944301000000099" in ids, (
        f"a raw id was not written under the id key: {ids}"
    )


def test_an_optional_camera_row_gets_the_same_picker(client, monkeypatch):
    """This is about the abstraction, not one row: every role gets scan +
    resolve + status, or the next role added quietly goes back to a text box."""
    use_argv(
        monkeypatch,
        fake_scan_argv(found=[found_record("iphone", IPHONE_ID, IPHONE)]),
    )
    client.post("/config/devices/scan")
    wait_for_scan(client.app)

    # Added, the same way a side camera is: the head is an Add/Remove list.
    client.post(
        "/config/optional/add",
        data={"config_form": "1", "type_id": "head_iphone"},
        headers={"HX-Request": "true"},
    )

    page = get_page(client)
    field = "optional.head_iphone_camera.device"
    assert field in controls(page), "the head iPhone row rendered no device box"
    assert IPHONE in offered_values(page, field)


def test_a_device_the_scan_could_not_open_is_listed_rather_than_dropped(
    client, monkeypatch
):
    """ "Gone" and "here but wedged" need different actions from the operator,
    and a device that vanishes from the list tells them neither."""
    use_argv(
        monkeypatch,
        fake_scan_argv(
            found=[found_record("oak", OAK_A_ID, OAK_A, state="unavailable")]
        ),
    )
    client.post("/config/devices/scan")
    wait_for_scan(client.app)

    page = get_page(client)
    listed = [o for o in offered(page, RIGHT_OAK) if o.get("value") == OAK_A]
    assert listed, "an unavailable device was dropped from the picker"
    assert "unavailable" in listed[0].get("label", "").lower(), (
        "the picker does not say the device would not open"
    )


def test_the_field_says_when_a_configured_device_is_not_on_this_bench(
    client, monkeypatch
):
    """A field goes green only for a device on this bench, not for any name in
    the device registry."""
    use_argv(monkeypatch, fake_scan_argv(found=[found_record("oak", OAK_A_ID, OAK_A)]))
    client.post("/config/devices/scan")
    wait_for_scan(client.app)

    page = get_page(client)
    data = filled(page, overrides={LEFT_OAK: OAK_B})
    client.post("/config/apply", data=data)

    page = get_page(client)
    message = page.attrs_by_id.get("message-camera-left_umi_camera")
    assert message is not None, "the field carries no status strip at all"
    said = parse(client.get("/config").text).body
    assert "not connected" in said


def test_the_picker_shows_the_sensor_module_on_each_socket(client, monkeypatch):
    """The module decides whether a stream is written 640x400 or 640x480, and a
    4:3 IMX378 given 16:10 loses 17% of its FOV. An operator picking the board
    has to be able to see what is on it."""
    use_argv(
        monkeypatch,
        fake_scan_argv(
            found=[
                found_record(
                    "oak",
                    OAK_A_ID,
                    OAK_A,
                    sockets={"CAM_B": "IMX378", "CAM_C": "OV9282"},
                )
            ]
        ),
    )
    client.post("/config/devices/scan")
    wait_for_scan(client.app)

    text = offered_text(get_page(client), RIGHT_OAK)
    assert "CAM_B: IMX378" in text, f"no per-socket module in the picker: {text}"


# ======================================================== resolving typed names
# Typed text resolves to exactly one canonical name, and a name is never
# fabricated: a raw string is not passed straight through.


def test_a_typed_token_is_echoed_back_as_the_canonical_device_name(client, repo):
    """Operators type `01`, not `oak-W97-01`. The field resolves that and stores
    the canonical name; a raw `01` in the config would be a session recorded
    from nothing."""
    page = get_page(client)
    data = filled(page, overrides={LEFT_OAK: "W97-01"})
    assert client.post("/config/apply", data=data).status_code == 200

    assert controls(get_page(client))[LEFT_OAK]["value"] == OAK_A
    config = load(generated(repo))
    names = [
        source.get("oak_device_name")
        for source in config["video_sources"]
        if source.get("type") == "oak_device"
    ]
    assert OAK_A in names, f"the typed token was not canonicalized: {names}"


def test_an_ambiguous_typed_value_is_reported_and_never_guessed(client, repo):
    """`1` matches oak-W97-01 and oak-FFC-11 by trailing number. The page
    refuses to pick one and says so, rather than writing `1` into the config.

    Both halves are asserted: the page says "ambiguous" next to the row, and
    the file on disk does not exist -- because guessing quietly is exactly the
    failure, and refusing quietly would be nearly as bad.
    """
    page = get_page(client)
    data = filled(page, overrides={LEFT_OAK: "1"})
    response = client.post("/config/apply", data=data)

    assert response.status_code == 400, "an ambiguous device sailed through"
    assert not generated(repo).exists(), "an ambiguous device reached a written config"

    body = parse(response.text).body
    assert "matches more than one" in body, (
        f"the page never says the value is ambiguous: {body[:400]}"
    )
    assert OAK_A not in body.replace(OAK_B, ""), (
        "the page invented a canonical name for an ambiguous token"
    )


def test_an_ambiguous_value_is_attached_to_the_field_it_is_about(client):
    """A message at the top of a long form makes the operator hunt for the row."""
    page = get_page(client)
    data = filled(page, overrides={LEFT_OAK: "1"})
    response = client.post("/config/apply", data=data)

    parsed = parse(response.text)
    field = parsed.attrs_by_id.get(LEFT_OAK)
    assert field is not None, "the field is not rendered at all"
    assert field.get("aria-invalid") == "true", "the offending row is not marked"
    assert field.get("aria-describedby") == "error-camera-left_umi_camera"


def test_an_unmatched_name_is_a_message_beside_the_field_not_a_500(client, repo):
    """`none` is not blocked, deliberately: it is unambiguous -- exactly what
    was typed -- and the device registry may not list every device yet.
    Blocking would brick config authoring until it did. It is said out loud
    instead."""
    page = get_page(client)
    response = client.post("/config/apply", data=filled(page))
    assert response.status_code == 200, response.text

    body = parse(client.get("/config").text).body
    assert "is not a known OAK camera" in body, (
        f"an unregistered name is accepted with nothing said about it: {body[:400]}"
    )
    assert load(generated(repo)) is not None


def test_resolution_never_fabricates_a_name(client):
    """The unit underneath, in isolation: a returned name is always an entry of
    the option list, never a repair of what was typed."""
    options = [OAK_A, OAK_B]
    assert devices_core.resolve_device("W97-01", options) == (OAK_A, "ok")
    assert devices_core.resolve_device("1", options) == (None, "ambiguous")
    assert devices_core.resolve_device("nope", options) == (None, "none")
    assert devices_core.resolve_device("  ", options) == (None, "empty")
    # A suffix with letters in it is how the ambiguity above is resolved.
    assert devices_core.resolve_device("FFC-01", options) == (OAK_B, "ok")
    # Digits match the trailing number as an integer, not as text: `1` is `01`.
    assert devices_core.resolve_device("01", options) == (None, "ambiguous")


# ========================================================== the alignment block
# The assembler merges into the template, so every preset's template has to
# carry an `alignment:` block, and Apply has to keep it.


@pytest.mark.parametrize("preset", [p.id for p in setup_presets.SETUP_PRESETS])
def test_every_applied_config_carries_an_alignment_block(repo, preset):
    """A config with no `alignment:` block neither aligns nor says so.

    `index_sources` has to name a stream the config records; an index source
    the config does not contain would make the aligner find nothing.
    """
    client = _client(repo, preset=preset)
    page = get_page(client)
    assert client.post("/config/apply", data=filled(page)).status_code == 200

    config = load(generated(repo, preset))
    assert config["alignment"]["enabled"] is True
    assert "upload_aligned" not in config["alignment"]
    names = {
        stream.get("name")
        for source in config["video_sources"]
        for stream in (source.get("streams") or [{"name": source.get("name")}])
    }
    assert set(config["alignment"]["index_sources"].split(",")) & names, (
        "the alignment block names an index source the config does not contain"
    )


# ================================================== scanning must never block


def test_starting_a_scan_returns_while_the_scan_is_still_running(client, monkeypatch):
    """Every hardware open in this repo is untimed, and reading an OAK's sensor
    modules needs an exclusive connect. Inline in a handler that is an HTTP
    worker pinned until the operator gives up."""
    use_argv(monkeypatch, fake_scan_argv(sleep=FAKE_SCAN_S))

    started = time.monotonic()
    # Do not follow the 303: that would fold the /config render into the time
    # of the POST, which is the route's response and nothing else.
    response = client.post("/config/devices/scan", follow_redirects=False)
    elapsed = time.monotonic() - started

    assert response.status_code in (200, 303), response.text
    assert elapsed < RESPONSE_BUDGET_S, (
        f"the route waited {elapsed:.2f}s for a {FAKE_SCAN_S}s scan"
    )
    job = devices_core.current_job(client.app)
    assert job is not None and not job.finished, (
        "the scan was already over, so this proves nothing"
    )

    # ...and the page still renders while it runs, which is the other half.
    page_started = time.monotonic()
    page = get_page(client)
    assert time.monotonic() - page_started < PAGE_BUDGET_S
    assert "Refreshing" in page.body


def test_refresh_re_runs_a_finished_scan(client, monkeypatch):
    """Refresh is not optional: devices get attached after the page is open,
    and one held by the camera viewer stays unavailable until that releases."""
    use_argv(monkeypatch, fake_scan_argv(found=[found_record("oak", OAK_A_ID, OAK_A)]))
    client.post("/config/devices/scan")
    first = wait_for_scan(client.app)

    use_argv(
        monkeypatch,
        fake_scan_argv(
            found=[
                found_record("oak", OAK_A_ID, OAK_A),
                found_record("oak", OAK_B_ID, OAK_B),
            ]
        ),
    )
    client.post("/config/devices/scan")
    wait_for_scan(client.app)
    assert devices_core.current_job(client.app) is not first, (
        "Refresh re-used the old scan"
    )

    assert offered_values(get_page(client), RIGHT_OAK) == {OAK_A, OAK_B}


def test_a_read_only_tab_cannot_start_a_scan(client, monkeypatch):
    """A scan holds an exclusive OAK connection. A tab that does not own the
    rig -- one that may be looking at a station mid-recording -- must not be
    able to take one."""
    use_argv(monkeypatch, fake_scan_argv())
    onlooker = TestClient(client.app)  # a second tab; `client` owns the rig
    onlooker.get("/config")

    response = onlooker.post("/config/devices/scan")
    assert response.status_code == 409, response.status_code
    assert devices_core.current_job(client.app) is None, "a read-only tab scanned"


# ======================================================= the no-JavaScript path


def test_every_device_control_works_with_the_vendored_script_blocked(client):
    """Stations run on restricted lab networks and the script does fail to
    load. Refresh and device-edit controls remain real submit buttons in a real
    `<form method="post">`, so a blocked script costs the live status strip and
    nothing else."""
    page = get_page(client)
    form = apply_form(page)
    assert form["method"] == "post"

    for action in ("/config/devices/scan",):
        found = buttons_for(page, action)
        assert found, f"no plain submit button aimed at {action}"
        assert all(f["method"] == "post" for f, _ in found)

    # And the server honours the plain post: 303 back to the page, no HX header
    # anywhere in the exchange.
    response = client.post("/config/devices/scan", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/config"


def test_the_page_auto_scans_on_first_load_and_stops_once_a_scan_exists(
    client, monkeypatch
):
    """The auto-scan is a POST the page issues after it has rendered, never a
    scan spawned from the GET itself: a GET that opens an exclusive OAK
    connection would fire on every reload, every read-only tab and every second
    browser, including during a recording.

    So what is asserted here is the trigger the browser is handed, and -- more
    importantly -- that it is gone once a scan exists, because a trigger that
    re-armed itself would be an unstoppable loop holding a camera open.
    """
    section = get_page(client).attrs_by_id["config-scan"]
    assert section.get("hx-trigger") == "load"
    assert section.get("hx-post") == "/config/devices/scan"
    # No form values ride along: an automatic request must not mutate the draft.
    assert section.get("hx-params") == "none"

    use_argv(monkeypatch, fake_scan_argv())
    client.post("/config/devices/scan")
    wait_for_scan(client.app)

    section = get_page(client).attrs_by_id["config-scan"]
    assert "hx-post" not in section, "the auto-scan would fire again"


def test_the_running_card_polls_and_the_finished_card_does_not(client, monkeypatch):
    """Self-limiting by construction: the swapped-in card carries the trigger
    only while the scan runs, so nothing is left polling a dead page."""
    use_argv(monkeypatch, fake_scan_argv(sleep=FAKE_SCAN_S))
    client.post("/config/devices/scan")

    running = parse(client.get("/config/devices/status").text)
    section = running.attrs_by_id["config-scan"]
    assert section.get("hx-get") == "/config/devices/status"
    assert "load" in section.get("hx-trigger", "")

    # Ended through the registry rather than by reaching into the job: a
    # cancelled scan is a terminal state the card has to stop polling on just as
    # much as a clean finish.
    job_control.cancel(
        devices_core.registry_for(client.app), devices_core.current_job(client.app).id
    )
    wait_for_scan(client.app)

    finished = parse(client.get("/config/devices/status").text)
    assert "hx-get" not in finished.attrs_by_id["config-scan"], (
        "the finished card keeps polling"
    )


def test_adopting_a_scan_keeps_what_was_typed_while_it_ran(client, monkeypatch):
    """The operator fills in the session fields during the 15-30s a scan takes.
    A plain reload to pick up the results would throw all of it away."""
    use_argv(monkeypatch, fake_scan_argv(found=[found_record("oak", OAK_A_ID, OAK_A)]))
    client.post("/config/devices/scan")
    wait_for_scan(client.app)

    page = get_page(client)
    data = submission(apply_form(page))
    data["task_name"] = "plane_assembly_v9"
    data["operator_name"] = "Test Operator"
    data[LEFT_OAK] = "oak-FFC-B"
    assert client.post("/config/devices/adopt", data=data).status_code in (200, 303)

    page = get_page(client)
    named = controls(page)
    assert named["operator_name"]["value"] == "Test Operator"
    assert named[LEFT_OAK]["value"] == "oak-FFC-B"
    assert OAK_A in offered_values(page, RIGHT_OAK)

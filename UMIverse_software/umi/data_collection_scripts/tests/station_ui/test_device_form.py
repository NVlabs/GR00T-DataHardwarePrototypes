# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The required-device form: one box per physical device, grouped by arm.

A field per stream with two controls each would ask the operator the same
question twice for every OAK-FFC board. This suite holds the
one-box-per-device shape, and the device-identity check underneath it.

Everything here is driven: the page is rendered by the app and read as a
browser would read it, and the assembler is called with the devices a form
would produce. Nothing asserts on source text, and **nothing opens hardware** --
the device scan is never started, so every field falls back to the registry.
"""

import pathlib
import shutil
from html.parser import HTMLParser

import pytest
from fastapi.testclient import TestClient

yaml = pytest.importorskip("yaml")


from station_ui import settings  # noqa: E402
from station_ui.app import create_app  # noqa: E402
from station_ui.core.config_assembler import (  # noqa: E402
    ConfigAssembler,
    DeviceSpec,
    SessionSpec,
    canonical_device_identity,
)
from station_ui.core.setup_presets import get_preset, template_path  # noqa: E402
from station_ui.settings import Settings  # noqa: E402
from umi.common.tasks import task_registry_store  # noqa: E402
from umi.utils import device_registry  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]

TASKS = {
    "plane_assembly_v9": {
        "name": "plane_assembly_v9",
        "short_description": "Assemble the plane",
        "mode": "bimanual",
        "HW_version": "v2.1",
        "training_group": "plane_assembly_v9",
    },
}

# One OAK board under both spellings a config can carry: the registry name and
# the raw id depthai reports. Two streams of one board assigned these two
# spellings must not be reported as disagreeing about which board they are.
BOARD_NAME = "oak-DW97-001"
BOARD_ID = "1944301000000000AC"
OTHER_NAME = "oak-DW97-002"
OTHER_ID = "1944301000000000AD"

FAKE_REGISTRY = {
    "version": 0,
    "updated_at": "1970-01-01T00:00:00Z",
    "oak_devices": {BOARD_NAME: BOARD_ID, OTHER_NAME: OTHER_ID},
    "iphone_devices": {"iphone15p-53": "00008130-000000000053"},
    "umi_v4_grippers": {"UMI-V4-0001": "AAAA0000AAAA0001"},
    "device_metadata": {},
}


@pytest.fixture(autouse=True)
def registries(monkeypatch):
    """A deterministic task registry and device registry.

    The device registry is a lazily-loaded singleton -- left alone, these tests
    would compare against whatever registry file the station has, which is a
    test that passes or fails on the machine.
    """
    monkeypatch.setattr(task_registry_store, "get_all_tasks", lambda: dict(TASKS))
    monkeypatch.setattr(device_registry, "_REGISTRY", dict(FAKE_REGISTRY))


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


# The preset every test here renders unless it says otherwise. Named, because
# one test has to reach for the template THIS preset resolves to.
_DEFAULT_PRESET = "bimanual_v4"


def _client(repo, *, preset=_DEFAULT_PRESET) -> TestClient:
    client = TestClient(create_app(Settings(repo_root=repo)))
    client.get("/")
    client.post("/setup/preset", data={"preset_id": preset})
    # A fresh session has no head camera; add the OAK the way an operator would.
    client.post(
        "/config/optional/add", data={"config_form": "1", "type_id": "head_oak"}
    )
    return client


# --------------------------------------------------------------------- parsing


class _Groups(HTMLParser):
    """The form controls nested inside each element that carries an id.

    Nesting-aware rather than regex-sliced: "how many boxes are in the Left
    section" is a containment question, and a regex for the closing tag stops at
    the first nested <div> instead of at the section's own.
    """

    _CONTROLS = ("input", "select", "textarea")
    _VOID = frozenset({"input", "br", "hr", "img", "link", "meta", "source"})

    def __init__(self):
        super().__init__()
        self.controls_by_id: dict[str, list[dict]] = {}
        self.attrs_by_id: dict[str, dict] = {}
        self.labels: list[str] = []
        # field name -> the label's own words, with anything nested inside it
        # left out. The plain label and the technical detail beside it are two
        # different claims, so a test must be able to read them apart.
        self.label_words: dict[str, str] = {}
        self.ids: list[str] = []
        self._open: list[str | None] = []
        self._label: str | None = None
        self._label_own: str = ""
        self._label_for: str | None = None
        self._label_depth = 0

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        element_id = attr.get("id")
        if element_id:
            self.attrs_by_id[element_id] = attr
            self.controls_by_id.setdefault(element_id, [])
            self.ids.append(element_id)
        if tag == "label":
            self._label = ""
            self._label_own = ""
            self._label_for = attr.get("for")
            self._label_depth = 0
        elif self._label is not None and tag not in self._VOID:
            self._label_depth += 1
        if tag in self._CONTROLS and attr.get("name"):
            control = {**attr, "tag": tag}
            for open_id in self._open:
                if open_id is not None:
                    self.controls_by_id[open_id].append(control)
        if tag not in self._VOID:
            self._open.append(element_id)

    def handle_endtag(self, tag):
        if tag == "label" and self._label is not None:
            self.labels.append(" ".join(self._label.split()))
            if self._label_for:
                self.label_words[self._label_for] = " ".join(self._label_own.split())
            self._label = None
        elif self._label is not None and tag not in self._VOID:
            self._label_depth -= 1
        if tag not in self._VOID and self._open:
            self._open.pop()

    def handle_data(self, data):
        if self._label is not None:
            self._label += data
            if self._label_depth == 0:
                self._label_own += data


def _groups(html: str) -> _Groups:
    parser = _Groups()
    parser.feed(html)
    return parser


def _page(client) -> _Groups:
    response = client.get("/config")
    assert response.status_code == 200, response.status_code
    return _groups(response.text)


def _fields_in(page: _Groups, arm: str) -> list[dict]:
    """The device controls inside one arm's group, in document order."""
    return [
        control
        for control in page.controls_by_id.get(f"arm-{arm}", [])
        if str(control.get("name", "")).startswith(("camera.", "gripper."))
    ]


def _generated(repo, preset=_DEFAULT_PRESET) -> pathlib.Path:
    # Use the isolated station state directory for generated files, not repo.
    return settings.state_dir_from_environment() / "data_recording" / f"{preset}.yaml"


def _device_labels(page: _Groups) -> dict[str, str]:
    """The words above each device box, keyed by the field they name.

    Device boxes only: `placement` is a note about a camera already named by
    the card it sits in, so it is not one of the boxes that has to be
    self-identifying.
    """
    return {
        name: text
        for name, text in page.label_words.items()
        if name.startswith(("camera.", "gripper.")) or name.endswith(".device")
    }


# ------------------------------------------------------------- plain language

# Keep socket, sensor-module and source-key details out of operator labels.
# Retain recognizable hardware types such as iPhone and OAK.
_JARGON = ("_", "OAK-FFC", "AR0234", "IMX378", "CAM_A", "CAM_B", "serves", "board")


def _every_optional_camera(client):
    """One row of every optional type, so the labels are read at their worst."""
    for type_id in ("head_iphone", "head_oak", "bottom_oak"):
        _add(client, type_id, htmx=False)


def test_every_device_box_reserves_a_line_under_itself(tmp_path):
    """The two arm cards must be the same height whatever is in the boxes.

    An empty box says nothing -- `_device_view` suppresses the message for
    `EMPTY` -- while a filled one says "connected", so without a floor one
    line's difference per box would stop the columns lining up.

    The floor is in CSS rather than in words put back: every box carries the
    node, and CSS gives it a minimum height. So this asserts the node is always
    there, including under a box with nothing in it.
    """
    client = _client(_repo(tmp_path))
    body = client.get("/config").text
    boxes = _device_labels(_page(client))

    assert boxes, "no device boxes were parsed, so this asserts nothing"
    assert body.count('class="field__foot"') >= len(boxes), (
        "a device box renders no footer node, so its card is a line shorter"
    )


def test_every_device_box_is_labelled_in_the_words_an_operator_uses(tmp_path):
    """Labels name what the operator holds, not the wiring.

    `Right OAK-FFC board — serves right_trigger_camera + right_umi_camera` is
    three things they do not have: a board model, a stream identifier, and a
    wiring diagram. On a V4 card each box is named for the part of the gripper
    it sits on, beside the diagram that numbers it, and the card's own title
    names the arm.
    """
    page = _page(_client(_repo(tmp_path)))
    assert _device_labels(page) == {
        "gripper.left": "Gripper",
        "camera.left_tracking_camera": "Phone",
        "camera.left_umi_camera": "Top camera",
        "camera.left_trigger_camera": "Trigger camera",
        "gripper.right": "Gripper",
        "camera.right_tracking_camera": "Phone",
        "camera.right_umi_camera": "Top camera",
        "camera.right_trigger_camera": "Trigger camera",
        # The head OAK row starts switched on, so its box is on the page from
        # the first load -- see `session.default_head_optional`.
        "optional.head_oak_camera.device": "Head camera (OAK)",
    }


@pytest.mark.parametrize("preset", ["bimanual_v4", "bimanual_v4_vive"])
def test_no_device_label_names_a_board_a_socket_or_a_stream(tmp_path, preset):
    client = _client(_repo(tmp_path), preset=preset)
    _every_optional_camera(client)
    for field, label in _device_labels(_page(client)).items():
        for jargon in _JARGON:
            assert jargon not in label, f"{field} is labelled {label!r}"
        assert label[:1].isupper(), f"{field} is labelled {label!r}"


@pytest.mark.parametrize("preset", ["bimanual_v4", "bimanual_v4_vive"])
def test_two_device_boxes_never_collapse_onto_one_plain_label(tmp_path, preset):
    """Simplifying is not worth an ambiguous form.

    A head may carry an iPhone and an OAK, so "Head camera" is not enough on
    its own. Every box still has to name exactly one device, or the operator
    fills in the wrong one and the page cannot say which.

    A box inside an arm's card is read with the card's title ("Left gripper"),
    so its words only have to be unique within that card. An optional camera
    sits in no arm card, so its words have to be unique on the whole page --
    including against every box in the cards.
    """
    client = _client(_repo(tmp_path), preset=preset)
    _every_optional_camera(client)

    page = _page(client)
    labels = _device_labels(page)
    in_arms = set()
    for arm in ("left", "right"):
        words = [labels[field["name"]] for field in _fields_in(page, arm)]
        assert len(set(words)) == len(words), (arm, words)
        in_arms.update(words)
    optional = [text for name, text in labels.items() if name.startswith("optional.")]
    assert len(set(optional)) == len(optional), sorted(optional)
    assert not set(optional) & in_arms, (
        f"an optional camera reads the same as a box in an arm card: "
        f"{sorted(set(optional) & in_arms)}"
    )


def test_a_bimanual_rig_renders_four_device_boxes_per_arm(tmp_path):
    """One box per physical device.

    bimanual_v4 declares four physical devices per arm: one OAK board for the
    wrist camera, one USB trigger camera, one iPhone, one gripper. A select and
    a text box for every stream would ask each question twice.
    """
    page = _page(_client(_repo(tmp_path)))

    for arm in ("right", "left"):
        fields = _fields_in(page, arm)
        assert len(fields) == 4, (
            f"the {arm} section renders {[f['name'] for f in fields]}"
        )
        # One control per device, and it is a text input over a datalist: a
        # `<select>` beside it is the second control this test exists to stop.
        for field in fields:
            assert field["tag"] == "input", f"{field['name']} is a <{field['tag']}>"
            assert field.get("type") in (None, "text"), field
            assert field.get("list"), f"{field['name']} offers no datalist"

    named = {f["name"] for arm in ("right", "left") for f in _fields_in(page, arm)}
    assert named == {
        "gripper.right",
        "camera.right_tracking_camera",
        "camera.right_umi_camera",
        "camera.right_trigger_camera",
        "gripper.left",
        "camera.left_tracking_camera",
        "camera.left_umi_camera",
        "camera.left_trigger_camera",
    }, named


@pytest.mark.parametrize(
    "preset,ordered_words",
    [
        ("bimanual_v4", ["Gripper", "Phone", "Top camera", "Trigger camera"]),
        ("bimanual_v4_vive", ["Gripper", "Top camera", "Trigger camera"]),
    ],
)
def test_each_arm_asks_for_the_gripper_first_then_its_cameras(
    tmp_path, preset, ordered_words
):
    """The gripper first, then the cameras in the order the card's diagram
    numbers them.

    They fill this form standing at the bench with the gripper in their hand,
    so the gripper is the first box, not the one after the cameras. The order
    comes from what each device is rather than from the order the template
    happens to list its source blocks in.
    """
    page = _page(_client(_repo(tmp_path), preset=preset))
    labels = _device_labels(page)

    for arm in ("left", "right"):
        ordered = [labels[field["name"]] for field in _fields_in(page, arm)]
        assert ordered == ordered_words, (arm, ordered)

    # ...and the columns still read left-to-right the way the rig is laid out.
    assert page.ids.index("arm-left") < page.ids.index("arm-right"), page.ids


def test_no_label_reads_out_the_streams_its_box_drives(tmp_path):
    """An operator is told which camera to plug in, not which streams a box
    points at, so the mapping is absent from the page rather than merely hidden
    in it.
    """
    repo = _repo(tmp_path)
    # Read off the labels: the stream is the form key too, so it is in the
    # markup either way -- what must not survive is it being read out to the
    # operator as part of the box's name.
    for text in _groups(_client(repo).get("/config").text).labels:
        assert "_" not in text, text


def _two_stream_board(repo) -> None:
    """Give the right wrist OAK a second stream: one board feeding two cameras.

    No shipped template declares such a block -- the V4 baseboard records only
    CAM_B -- but the form and the assembler still treat an `oak_device` block as
    one device behind every stream it carries, and this is what exercises that.
    Edited through the preset the tests render, so the file changed is the one
    the page reads.
    """
    template = template_path(get_preset(_DEFAULT_PRESET), repo)
    cfg = yaml.safe_load(template.read_text())
    board = next(
        source
        for source in cfg["video_sources"]
        if source.get("type") == "oak_device"
        and [stream["name"] for stream in source["streams"]] == ["right_umi_camera"]
    )
    board["streams"].append(
        {"name": "right_wide_camera", "socket": "CAM_A", "resolution": [640, 400]}
    )
    template.write_text(yaml.safe_dump(cfg, sort_keys=False))


def test_one_box_stands_for_a_whole_board_and_is_named_after_its_camera(tmp_path):
    """One box points two cameras, and it is named after the one the operator
    aims at the task -- not after the board both of them hang off."""
    repo = _repo(tmp_path)
    _two_stream_board(repo)
    page = _page(_client(repo))
    labels = _device_labels(page)

    assert labels["camera.right_umi_camera"] == "Top camera"
    # ...and there is no second box for the stream it also drives.
    assert "camera.right_wide_camera" not in labels, sorted(labels)


def test_a_source_block_added_to_the_template_grows_a_field_with_no_code_change(
    tmp_path,
):
    """The property that makes the grouping maintainable.

    A new preset, or an extra board on an existing one, must cost a template
    edit and nothing else. This adds a third OAK block to the fixture template
    and expects a fifth box in the Right section -- if the sections were built
    from a list of known camera names, it would still render four.
    """
    repo = _repo(tmp_path)
    # Resolved through the preset the client below renders, not by naming a
    # file, so the template edited is the one the page renders.
    template = template_path(get_preset(_DEFAULT_PRESET), repo)
    cfg = yaml.safe_load(template.read_text())
    cfg["video_sources"].append(
        {
            "type": "oak_device",
            "oak_device_name": "oak-FFC-99",
            "fps": 60,
            "latency": 0.0875,
            "streams": [
                {
                    "name": "right_extra_camera",
                    "socket": "CAM_A",
                    "resolution": [640, 400],
                },
                {
                    "name": "right_spare_camera",
                    "socket": "CAM_B",
                    "resolution": [640, 400],
                },
            ],
        }
    )
    template.write_text(yaml.safe_dump(cfg, sort_keys=False))

    page = _page(_client(repo))
    fields = _fields_in(page, "right")
    assert len(fields) == 5, [f["name"] for f in fields]
    assert "camera.right_extra_camera" in {f["name"] for f in fields}
    # And it lands on the right arm, from the stream names alone.
    assert len(_fields_in(page, "left")) == 4

    # ...labelled from the template's own words, with no table of stream names
    # here to keep in step with it.
    assert _device_labels(page)["camera.right_extra_camera"] == "Right extra camera"


def test_one_box_assigns_every_stream_of_its_block(tmp_path):
    """The point of the box, asserted on the file.

    Both streams of the board have to come out pointing at what was typed once,
    or the operator has filled the form in and the second camera still carries
    the template's placeholder.
    """
    repo = _repo(tmp_path)
    _two_stream_board(repo)
    client = _client(repo)
    response = client.post(
        "/config/apply",
        data={
            "config_form": "1",
            "task_name": "plane_assembly_v9",
            "operator_name": "Test Operator",
            "camera.right_umi_camera": BOARD_NAME,
            "camera.left_umi_camera": OTHER_NAME,
            "camera.right_trigger_camera": "0",
            "camera.left_trigger_camera": "2",
            "camera.right_tracking_camera": "iphone-R",
            "camera.left_tracking_camera": "iphone-L",
            "gripper.right": "UMI-V4-0001",
            "gripper.left": "UMI-V4-0002",
            # A fresh session starts with a head OAK row. It lives in the draft
            # rather than in this post, so omitting the key does not drop it --
            # an empty box is refused by name, which is a different test.
            "optional.head_oak_camera.device": "oak-head",
        },
    )
    assert response.status_code == 200, response.text[:400]

    cfg = yaml.safe_load(_generated(repo).read_text())
    board = next(
        source
        for source in cfg["video_sources"]
        if source.get("type") == "oak_device"
        and {s["name"] for s in source["streams"]}
        == {"right_umi_camera", "right_wide_camera"}
    )
    assert board.get("oak_device_name") == BOARD_NAME, board


# ------------------------------------------------------------- device identity


def test_a_registry_name_and_its_raw_id_are_one_device(tmp_path):
    """The false positive, at the unit that decides it."""
    assert canonical_device_identity(BOARD_NAME) == canonical_device_identity(BOARD_ID)
    # Case is not identity either: the same id is written both ways.
    assert canonical_device_identity(BOARD_ID.lower()) == canonical_device_identity(
        BOARD_ID
    )
    # ...and two different boards stay different, or the fold would pass
    # everything and this file would prove nothing.
    assert canonical_device_identity(BOARD_NAME) != canonical_device_identity(
        OTHER_NAME
    )
    assert canonical_device_identity(BOARD_NAME) != canonical_device_identity(OTHER_ID)


def _session(**kw) -> SessionSpec:
    base = dict(
        preset_id=_DEFAULT_PRESET,
        operator_name="Test Operator",
        task_name="plane_assembly_v9",
        session_uid="91e98c89fccb",
        data_dir="~/Dev/UMI/local_data/2026-08-04_x_91e98c89fccb",
    )
    base.update(kw)
    return SessionSpec(**base)


def _cameras(**overrides) -> dict:
    cameras = {
        "right_umi_camera": BOARD_NAME,
        "left_umi_camera": OTHER_NAME,
        "right_trigger_camera": "0",
        "left_trigger_camera": "2",
        "right_tracking_camera": "iphone-R",
        "left_tracking_camera": "iphone-L",
    }
    cameras.update(overrides)
    return cameras


def _devices(**kw) -> DeviceSpec:
    base = dict(
        grippers={"left": "UMI-V4-0001", "right": "UMI-V4-0002"},
        required_cameras=_cameras(),
        optional=(),
    )
    base.update(kw)
    return DeviceSpec(**base)


def test_a_name_and_the_matching_raw_id_on_one_block_do_not_raise(tmp_path):
    """A registry name and the raw id it maps to are the same board, so two
    streams assigned the two spellings must not be reported as "streams of one
    device assigned X and Y"."""
    repo = _repo(tmp_path)
    _two_stream_board(repo)
    out = ConfigAssembler(repo).assemble(
        _session(),
        _devices(required_cameras=_cameras(right_wide_camera=BOARD_ID)),
    )
    cfg = yaml.safe_load(pathlib.Path(out).read_text())
    board = next(
        source
        for source in cfg["video_sources"]
        if source.get("type") == "oak_device"
        and {s["name"] for s in source["streams"]}
        == {"right_umi_camera", "right_wide_camera"}
    )
    written = board.get("oak_device_name") or board.get("oak_device_id")
    assert canonical_device_identity(written) == canonical_device_identity(BOARD_ID)


def test_two_different_boards_on_one_block_still_raise(tmp_path):
    """Guard the guard. A fold that made every pair equal would pass the test
    above and let a form that really does disagree through."""
    repo = _repo(tmp_path)
    _two_stream_board(repo)
    with pytest.raises(ValueError, match="streams of one device"):
        ConfigAssembler(repo).assemble(
            _session(),
            _devices(required_cameras=_cameras(right_wide_camera=OTHER_ID)),
        )


def test_two_spellings_of_one_board_across_two_blocks_are_still_a_duplicate(tmp_path):
    """The other side of the same fold, and the reason it is applied in
    `_reject_shared_devices` too: one physical OAK behind two source blocks
    hands the second block the same handle out of the device cache, and neither
    block records what it thinks it does."""
    repo = _repo(tmp_path)
    shared = _cameras(left_umi_camera=BOARD_ID)
    with pytest.raises(ValueError, match="assigned to both"):
        ConfigAssembler(repo).assemble(_session(), _devices(required_cameras=shared))


def test_an_unregistered_device_still_compares_only_to_itself(tmp_path):
    """Stations author configs against hardware the registry has never heard of
    -- and the registry itself may be the bundled fallback. Unknown names must
    not all fold together."""
    assert canonical_device_identity("oak-FFC-A") != canonical_device_identity(
        "oak-FFC-B"
    )
    repo = _repo(tmp_path)
    ConfigAssembler(repo).assemble(
        _session(),
        _devices(required_cameras=_cameras(right_umi_camera="oak-FFC-A")),
    )
    assert _generated(repo).exists()


# -------------------------------------------------- adding a camera in place


def _add(client, type_id, *, htmx: bool):
    headers = {"HX-Request": "true"} if htmx else {}
    return client.post(
        "/config/optional/add",
        data={"config_form": "1", "type_id": type_id},
        headers=headers,
        follow_redirects=False,
    )


def test_adding_a_camera_answers_with_the_section_not_the_whole_page(tmp_path):
    """A full page POST throws away the scroll position, so the operator lands
    back at the session fields hunting for the row they just created. Only the
    section changes, so only the section is sent -- htmx swaps that one node and
    nothing else on the page moves.
    """
    client = _client(_repo(tmp_path))
    response = _add(client, "bottom_oak", htmx=True)

    assert response.status_code == 200, response.status_code
    body = response.text
    # A fragment, not a page: a full document swapped into a <section> would
    # nest a second <html> inside the form.
    assert "<!DOCTYPE" not in body and "<html" not in body, body[:200]
    assert "operator_name" not in body, "the whole page came back"

    fragment = _groups(body)
    assert fragment.ids[0] == "bottom-section", fragment.ids[:3]
    assert "row-bottom_oak_camera" in fragment.attrs_by_id, sorted(fragment.attrs_by_id)


def _add_buttons(html: str) -> list[dict]:
    """Every Add control, with its attributes, from the rendered page."""
    found = []

    class _Buttons(HTMLParser):
        def handle_starttag(self, tag, attrs):
            attr = dict(attrs)
            if tag == "button" and attr.get("formaction") == "/config/optional/add":
                found.append(attr)

    _Buttons().feed(html)
    return found


def test_every_add_button_is_a_real_submit_that_htmx_only_intercepts(tmp_path):
    """Both halves of the progressive-enhancement rule, on the same control:
    a real `formaction` submit for a browser with the vendored script blocked,
    and an htmx target that names a section the page actually declares."""
    client = _client(_repo(tmp_path))
    html = client.get("/config").text
    page = _groups(html)
    buttons = _add_buttons(html)
    assert buttons, "the page renders no Add controls"

    for button in buttons:
        assert button.get("type") == "submit", button
        assert button.get("name") == "type_id" and button.get("value"), button
        target = button.get("hx-target", "")
        assert target.startswith("#"), button
        assert target[1:] in page.attrs_by_id, (
            f"hx-target {target} names an id the page does not declare"
        )
        # Without this htmx scrolls the swapped section into view, which is the
        # jump the swap exists to avoid.
        assert "show:none" in button.get("hx-swap", ""), button


def test_adding_a_camera_works_as_a_plain_post_with_javascript_off(tmp_path):
    """The no-JS path, driven with no HX header at all: a 303 back to the page,
    and the row really there when the browser follows it."""
    client = _client(_repo(tmp_path))
    response = _add(client, "bottom_oak", htmx=False)

    assert response.status_code == 303, response.status_code
    assert response.headers["location"] == "/config"

    page = _page(client)
    assert "row-bottom_oak_camera" in page.attrs_by_id, sorted(page.attrs_by_id)


def test_a_refused_add_reports_itself_inside_the_swapped_section(tmp_path):
    """A refusal has to arrive where the swap lands. A banner rendered at the
    top of a page that was never sent is a click that silently does nothing."""
    client = _client(_repo(tmp_path))

    # A fresh session already has the head OAK, and `max_count=1` caps it there,
    # so adding a second is the refusal to test -- and it has to arrive in the
    # section the swap replaces, not at the top of a page that was never sent.
    refused = _add(client, "head_oak", htmx=True)
    assert refused.status_code == 400, refused.status_code
    assert "already configured" in refused.text, refused.text[:400]
    assert _groups(refused.text).ids[0] == "head-section"


def _remove(client, logical_name, *, htmx: bool, confirm: bool = False):
    """Press Remove exactly as the rendered control does.

    The confirmation travels in the query string, not in the body: the button's
    one name/value pair is already spent saying which row to remove, and a
    hidden `confirm` input would sit inside the page's one form and be posted by
    Apply as well.
    """
    headers = {"HX-Request": "true"} if htmx else {}
    return client.post(
        "/config/optional/remove" + ("?confirm=yes" if confirm else ""),
        data={"config_form": "1", "logical_name": logical_name},
        headers=headers,
        follow_redirects=False,
    )


def _assign(client, field, value):
    """Type a device into a box and let the server absorb it, as a browser does."""
    return client.post(
        "/config/devices/adopt",
        data={"config_form": "1", field: value},
        follow_redirects=False,
    )


def _remove_buttons(html: str) -> list[dict]:
    found = []

    class _Buttons(HTMLParser):
        def handle_starttag(self, tag, attrs):
            attr = dict(attrs)
            action = attr.get("formaction", "")
            if tag == "button" and action.startswith("/config/optional/remove"):
                found.append(attr)

    _Buttons().feed(html)
    return found


def test_removing_a_camera_answers_with_the_section_not_the_whole_page(tmp_path):
    """Removing a camera keeps the operator's scroll position, the same as
    adding one: only the section changes, so only the section is sent."""
    client = _client(_repo(tmp_path))
    # The head OAK a fresh session starts with, plus a head iPhone.
    _add(client, "head_iphone", htmx=False)

    response = _remove(client, "head_iphone_camera", htmx=True)

    assert response.status_code == 200, response.status_code
    body = response.text
    assert "<!DOCTYPE" not in body and "<html" not in body, body[:200]
    assert "operator_name" not in body, "the whole page came back"

    fragment = _groups(body)
    assert fragment.ids[0] == "head-section", fragment.ids[:3]
    assert "row-head_iphone_camera" not in fragment.attrs_by_id
    assert "row-head_oak_camera" in fragment.attrs_by_id, sorted(fragment.attrs_by_id)


def test_every_remove_button_is_a_real_submit_that_htmx_only_intercepts(tmp_path):
    client = _client(_repo(tmp_path))
    _add(client, "bottom_oak", htmx=False)

    html = client.get("/config").text
    page = _groups(html)
    # Two: the bottom OAK just added, and the head OAK a fresh session starts
    # with. Both sections share one Add/Remove idiom, so every row carries
    # Remove like any other.
    buttons = _remove_buttons(html)
    assert len(buttons) == 2, buttons

    for button in buttons:
        assert button.get("type") == "submit", button
        assert button.get("name") == "logical_name" and button.get("value"), button
        target = button.get("hx-target", "")
        assert target[1:] in page.attrs_by_id, (
            f"hx-target {target} names an id the page does not declare"
        )
        # Without this htmx scrolls the swapped section into view, which is the
        # jump the swap exists to avoid.
        assert "show:none" in button.get("hx-swap", ""), button


def test_a_removal_that_names_hardware_asks_inside_the_section_it_swaps(tmp_path):
    """The confirm has to arrive where the swap lands, and it cannot be a form.

    The page is one form and HTML forbids nesting, so the confirmation cannot be
    a form of its own inside the row. It is a plain submit button in the page's
    own form instead, aimed at the same endpoint with `?confirm=yes` -- which
    still works with the vendored script blocked, and still refuses to drop a
    camera nobody agreed to drop.
    """
    client = _client(_repo(tmp_path))
    _add(client, "bottom_oak", htmx=False)
    _assign(client, "optional.bottom_oak_camera.device", OTHER_NAME)

    asked = _remove(client, "bottom_oak_camera", htmx=True)
    assert asked.status_code == 200, asked.status_code
    assert "<form" not in asked.text, "a nested form was swapped into the page's form"
    fragment = _groups(asked.text)
    assert fragment.ids[0] == "bottom-section", fragment.ids[:3]
    assert "row-bottom_oak_camera" in fragment.attrs_by_id, (
        "the camera was dropped anyway"
    )
    assert OTHER_NAME in asked.text, "the confirmation does not say what is at stake"

    confirm = [
        button
        for button in _remove_buttons(asked.text)
        if button.get("formaction", "").endswith("?confirm=yes")
    ]
    assert len(confirm) == 1, _remove_buttons(asked.text)
    assert confirm[0].get("value") == "bottom_oak_camera", confirm[0]

    gone = _remove(client, "bottom_oak_camera", htmx=True, confirm=True)
    assert gone.status_code == 200, gone.status_code
    assert "row-bottom_oak_camera" not in _groups(gone.text).attrs_by_id


def test_removing_a_camera_works_as_a_plain_post_with_javascript_off(tmp_path):
    """The no-JS path, with no HX header at all: a 303 back to the page, the
    confirmation still asked, and the row really gone when it is agreed to."""
    client = _client(_repo(tmp_path))
    _add(client, "bottom_oak", htmx=False)
    _assign(client, "optional.bottom_oak_camera.device", OTHER_NAME)

    asked = _remove(client, "bottom_oak_camera", htmx=False)
    assert asked.status_code == 200, asked.status_code
    assert "row-bottom_oak_camera" in _groups(asked.text).attrs_by_id
    assert [
        button
        for button in _remove_buttons(asked.text)
        if button.get("formaction", "").endswith("?confirm=yes")
    ], "the no-script page offers no way to confirm"

    gone = _remove(client, "bottom_oak_camera", htmx=False, confirm=True)
    assert gone.status_code == 303, gone.status_code
    assert gone.headers["location"] == "/config"
    assert "row-bottom_oak_camera" not in _page(client).attrs_by_id


def test_the_placement_control_sits_inside_its_camera_card(tmp_path):
    """`placement` must render inside the row it belongs to, or an operator
    cannot tell which camera it is setting. Containment, not pixels: the select
    has to be a descendant of the card that names the camera."""
    client = _client(_repo(tmp_path))
    _add(client, "bottom_oak", htmx=False)

    page = _page(client)
    inside = {
        control.get("name") for control in page.controls_by_id["row-bottom_oak_camera"]
    }
    assert "optional.bottom_oak_camera.placement" in inside, inside
    assert "optional.bottom_oak_camera.device" in inside, inside

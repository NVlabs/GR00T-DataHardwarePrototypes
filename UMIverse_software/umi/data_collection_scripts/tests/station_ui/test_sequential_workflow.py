# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The four numbered steps are sequential, and a rig change expires the checks.

Three separate properties, and they are separate on purpose:

* **Forward only.** A tab past the first outstanding step is not a link: there
  is nothing useful to show for a later step of a rig with no config, and the
  console does not answer questions about a session that does not exist.
* **Expiry.** Steps 3 and 4 are claims about hardware. Swap a camera and both
  describe a rig that is gone -- so they untick, and the banner says why rather
  than leaving the operator to notice.
* **A control, not just a rule.** Each page carries the forward button, disabled
  with the reason while its own step is outstanding.

What must not expire is a change that opens no different hardware: a task name,
an operator. A gate that fires on those teaches people to route around it.

**No test here opens hardware.** The device test is faked at `outcome_for` and
the calibration at `step_state`, exactly as the readiness suite does.
"""

import pathlib
import re
from html.parser import HTMLParser

import pytest

yaml = pytest.importorskip("yaml")

import test_config_page as config_page  # noqa: E402

from station_ui import calibration as cal  # noqa: E402
from station_ui import readiness, session  # noqa: E402
from umi.common.tasks import task_registry_store  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]

DECLARED = tuple(config_page._CAMERA_FIELDS.values()) + tuple(
    config_page._GRIPPERS.values()
)

STEPS = ("/", "/config", "/calibration")


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(
        task_registry_store, "get_all_tasks", lambda: dict(config_page.TASKS)
    )


@pytest.fixture
def client(tmp_path):
    return config_page._client(config_page._repo(tmp_path))


def pass_calibration(monkeypatch):
    monkeypatch.setattr(cal, "step_state", lambda app, step_id: cal.DONE)


# ------------------------------------------------------------------- parsing


class _Nav(HTMLParser):
    """Which workflow entries are live links and which are disabled spans."""

    def __init__(self):
        super().__init__()
        self.links: set[str] = set()
        self.disabled: list[str] = []
        self._in_nav = False
        self._span = False

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        classes = (attr.get("class") or "").split()
        if tag == "nav":
            self._in_nav = True
        elif self._in_nav and tag == "a" and "nav__link" in classes:
            self.links.add(attr.get("href", ""))
        elif self._in_nav and tag == "span" and "nav__link--disabled" in classes:
            self._span = True
            self.disabled.append("")

    def handle_endtag(self, tag):
        if tag == "nav":
            self._in_nav = False
        elif tag == "span" and self._span:
            self._span = False

    def handle_data(self, data):
        if self._span and data.strip():
            self.disabled[-1] += data.strip() + " "


def nav(client) -> _Nav:
    parsed = _Nav()
    parsed.feed(client.get("/").text)
    return parsed


def next_control(html: str, section="configuration") -> dict | None:
    from html_sections import section_html

    html = section_html(html, section)
    """The forward control on a page: `(enabled, label, reason)`.

    Two containers, because Configuration puts the same control in its sticky
    action bar beside Apply while every other page gives it a section of its
    own. The control itself is the same button either way, so this looks for
    the button rather than for one particular box around it -- otherwise these tests would report
    "no forward control on Configuration" when what changed was the wrapper.
    """
    match = re.search(
        r'<div class="card row next-step">(.*?)</div>', html, re.S
    ) or re.search(r'<div class="apply-bar">(.*?)</div>', html, re.S)
    if match is None:
        return None
    body = match.group(1)
    link = re.search(r'<a class="btn[^"]*" href="([^"]+)">\s*([^<]+?)\s*</a>', body)
    if link:
        return {"enabled": True, "href": link.group(1), "label": link.group(2)}
    reason = re.search(r'id="next-step-why">\s*([^<]+?)\s*</span>', body)
    label = re.search(r"aria-describedby=\"next-step-why\">([^<]+)</button>", body)
    return {
        "enabled": False,
        "label": (label.group(1).strip() if label else ""),
        "reason": (reason.group(1) if reason else ""),
    }


# ------------------------------------------------------------- forward only


def test_a_fresh_station_can_scroll_to_every_section(tmp_path):
    """Nothing is chosen yet, so Configuration onwards are questions about a
    rig that does not exist."""
    # `preset=None`: nothing chosen at all, which is what a station looks like
    # at the start of a shift.
    client = config_page._client(config_page._repo(tmp_path), preset=None)

    assert nav(client).links == {
        "#setup",
        "#configuration",
        "#calibration",
        "#record",
        "#audit",
    }
    assert readiness.blocking(client.app).href == "/"


def test_each_step_unlocks_exactly_the_next_one(client, monkeypatch):
    """Walked forward one step at a time, so this cannot pass by the gate being
    constant in either direction."""
    opened = []
    for arrange in (
        lambda: None,
        lambda: config_page._apply(client, head=False),
        lambda: pass_calibration(monkeypatch),
    ):
        arrange()
        opened.append(sorted(readiness.reachable(client.app) & set(STEPS)))

    assert opened == [
        ["/", "/config"],
        ["/", "/calibration", "/config"],
        ["/", "/calibration", "/config"],
    ]


def test_record_and_data_audit_are_never_gated(client):
    """Record refuses at the point of action, which is stricter than greying a
    tab; Data Audit is about episodes recorded in the past."""
    links = nav(client).links

    assert "#record" in links and "#audit" in links


# ------------------------------------------------------------ the control


def test_configuration_has_no_separate_apply_or_forward_button(client):
    control = next_control(client.get("/config").text)

    assert control is None
    assert (
        "Configuration is validated and saved automatically"
        not in client.get("/config").text
    )


def test_saving_configuration_does_not_restore_a_forward_button(client):
    config_page._apply(client, head=False)

    control = next_control(client.get("/config").text)

    assert control is None


def test_completed_calibration_has_no_forward_button(client, monkeypatch):
    config_page._apply(client, head=False)
    pass_calibration(monkeypatch)

    control = next_control(client.get("/calibration").text, "calibration")

    assert control is None
    assert readiness.blocking(client.app) is None


# --------------------------------------------------------------- expiry


def test_the_operator_is_told_why_the_step_unticked(client, monkeypatch):
    """A step that clears itself with no explanation reads as lost work."""
    config_page._apply(client, head=False)
    state = cal.calibration_state(client.app)
    for step in cal.STEPS:
        state.stamp(step.id, session.device_fingerprint(client.app.state.rig.draft))
    config_page._apply(client, head=False, grippers={"left": "UMI-V4-0009"})

    blocker = readiness.blocking(client.app)

    assert blocker is not None
    assert blocker.href == "/calibration"


def test_changing_a_camera_expires_the_calibration_too(client, monkeypatch):
    """Both checks are claims about the same hardware."""
    config_page._apply(client, head=False)
    state = cal.calibration_state(client.app)
    for step in cal.STEPS:
        state.stamp(step.id, session.device_fingerprint(client.app.state.rig.draft))

    assert not cal.stale(client.app, cal.STEP_VERIFY)

    config_page._apply(client, head=False, cameras={"right_umi_camera": "oak-FFC-88"})

    assert cal.stale(client.app, cal.STEP_VERIFY)
    assert cal.stale(client.app, cal.STEP_GRIPPER)


@pytest.mark.parametrize(
    "change",
    [
        {"task_name": "plane_assembly_v9"},
        {"operator_name": "Somebody Else"},
    ],
    ids=["task", "operator"],
)
def test_a_change_that_opens_no_different_hardware_expires_nothing(
    client, monkeypatch, change
):
    """The other half of the rule, and the one that keeps the gate credible.

    Expiring a fifteen-minute camera check because somebody fixed a typo in
    their own name is how a gate teaches people to route around it.
    """
    config_page._apply(client, head=False)
    state = cal.calibration_state(client.app)
    for step in cal.STEPS:
        state.stamp(step.id, session.device_fingerprint(client.app.state.rig.draft))
    before = readiness.done(client.app)

    config_page._apply(client, head=False, **change)

    assert readiness.done(client.app) == before
    assert not cal.stale(client.app, cal.STEP_VERIFY)


def test_the_fingerprint_ignores_the_order_the_boxes_were_filled_in(tmp_path):
    """Two drafts describing one rig have to agree, or the gate fires on a
    reordered dict."""
    a = session.SessionDraft(preset_id="bimanual_v4")
    b = session.SessionDraft(preset_id="bimanual_v4")
    a.grippers = {"left": "UMI-V4-0001", "right": "UMI-V4-0002"}
    b.grippers = {"right": "UMI-V4-0002", "left": "UMI-V4-0001"}
    a.required_cameras = {"x": "oak-1", "y": "oak-2"}
    b.required_cameras = {"y": "oak-2", "x": "oak-1"}

    assert session.device_fingerprint(a) == session.device_fingerprint(b)


def test_adding_a_head_camera_expires_the_checks(client, monkeypatch):
    """An optional camera is hardware too: adding one puts a device on the rig
    that the calibration has never looked at."""
    config_page._apply(client, head=False)
    state = cal.calibration_state(client.app)
    for step in cal.STEPS:
        state.stamp(step.id, session.device_fingerprint(client.app.state.rig.draft))
    assert not cal.stale(client.app, cal.STEP_VERIFY)

    # Added through the same route the Add button aims at -- the head is an
    # Add/Remove list, so there is no toggle to flip in the apply body.
    client.post(
        "/config/optional/add", data={"config_form": "1", "type_id": "head_oak"}
    )
    config_page._apply(
        client, overrides={"optional.head_oak_camera.device": "oak-FFC-33"}
    )

    assert cal.stale(client.app, cal.STEP_VERIFY)

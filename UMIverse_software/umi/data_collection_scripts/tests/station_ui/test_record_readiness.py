# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recording must not be offered before the steps that make it safe have passed.

`preflight()` refuses at the point of action, but that is after the operator
has committed: on a rig they may already have told a colleague they are
starting. The console should not offer Start in the first place, so this suite
is about the affordance.

Two rules run through all of it.

* **The button is a hint, the check is the rule.** Every test that asserts the
  control is unavailable also asserts the POST is still refused, because a
  bookmarked or replayed one never renders a button at all.
* **One notion of ready.** The gate reads the same `station_ui.readiness` the
  nav ticks render from. A second answer to "is this station ready" would let
  the tabs and the button disagree, and the operator would have no way to tell
  which one was lying.
"""

from __future__ import annotations

import pathlib
import sys
import time
from html.parser import HTMLParser

import pytest

# Reuse browser Apply helpers and the fake scan worker so tests cannot
# open cameras belonging to another process.
import test_config_page as config_page

REPO = pathlib.Path(__file__).resolve().parents[2]

from station_ui import calibration as cal  # noqa: E402
from station_ui import devices as devices_core  # noqa: E402
from station_ui import jobs, readiness, session  # noqa: E402
from station_ui.routes import record  # noqa: E402
from umi.common.tasks import task_registry_store  # noqa: E402

# Every device the applied draft declares. `/config` folds a scan's result
# over these to show which are attached; nothing gates on it.
DECLARED = tuple(config_page._CAMERA_FIELDS.values()) + tuple(
    config_page._GRIPPERS.values()
)


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    """The config page's deterministic task table. No test may reach a remote registry."""
    monkeypatch.setattr(
        task_registry_store, "get_all_tasks", lambda: dict(config_page.TASKS)
    )


# --------------------------------------------------------------------- parsing


class _Buttons(HTMLParser):
    """Every `<button>` inside the form that posts to a given action.

    Structural: a control that is only greyed by a stylesheet, or one whose
    `disabled` lives on a wrapper, both pass a substring check for the word.
    """

    def __init__(self, action):
        super().__init__()
        self.action = action
        self.buttons: list[dict] = []
        self.forms: list[dict] = []
        self._inside = False
        self._button: dict | None = None

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        if tag == "form":
            self.forms.append(attr)
            self._inside = attr.get("action") == self.action
        elif tag == "button" and self._inside:
            self._button = {**attr, "text": ""}
            self.buttons.append(self._button)

    def handle_endtag(self, tag):
        if tag == "form":
            self._inside = False
        elif tag == "button":
            self._button = None

    def handle_data(self, data):
        if self._button is not None:
            self._button["text"] += data


def _start_button(html: str) -> dict | None:
    parser = _Buttons("/record/start")
    parser.feed(html)
    live = [b for b in parser.buttons if "Start" in b["text"]]
    return live[0] if live else None


def _has_start_form(html: str) -> bool:
    parser = _Buttons("/record/start")
    parser.feed(html)
    return any(form.get("action") == "/record/start" for form in parser.forms)


def _blocked(html: str) -> bool:
    """Whether the console is refusing to offer the action."""
    button = _start_button(html)
    return button is None or "disabled" in button


# -------------------------------------------------------------------- fixtures


def _station(tmp_path, *, applied=True):
    """A browser that owns the rig, with as many steps done as asked for."""
    repo = config_page._repo(tmp_path)
    client = config_page._client(repo)
    if applied:
        config_page._apply(client, head=False)
    client.get("/record")  # claims the rig and sets the cookie
    return client


def _scan_found_every_device(client, monkeypatch, *, devices=DECLARED):
    """A finished scan that found every declared device, on this rig.

    Patched at `outcome_for`, exactly as `test_config_page.py` does for the nav
    ticks, so no test in this file can open a camera or a gripper that belongs
    to somebody's running session.

    There is no stamp: the device test is not a workflow step, so this only
    makes the scan's result available to `/config`.
    """
    monkeypatch.setattr(
        devices_core, "outcome_for", lambda job: (config_page._found(*devices), None)
    )


def _pass_the_calibration(monkeypatch):
    """Both calibration steps green.

    Patched at `step_state`, which is what `readiness.calibration_done` reads,
    rather than by driving the real children: those open every camera on the rig
    and block on four keyboard prompts, and no test here may do either.
    """
    monkeypatch.setattr(cal, "step_state", lambda app, step_id: cal.DONE)


def _post_start(client):
    return client.post(
        "/record/start", data={"confirmed": "yes"}, follow_redirects=False
    )


# =============================================================================
# 1. The control is unavailable until the earlier steps pass
# =============================================================================


def test_start_is_unavailable_while_the_configuration_has_not_been_applied(tmp_path):
    """Step 2 is the file on disk, not "the form looks filled in"."""
    client = _station(tmp_path, applied=False)

    assert _blocked(client.get("/record").text)
    assert _post_start(client).status_code == 400


def test_start_is_available_once_every_earlier_step_has_passed(tmp_path, monkeypatch):
    """The control that gives every assertion above its teeth: with all four
    steps genuinely done the button comes back, so none of them is passing
    because the page never renders a live Start at all.
    """
    client = _station(tmp_path)
    _scan_found_every_device(client, monkeypatch)
    _pass_the_calibration(monkeypatch)

    body = client.get("/record").text
    button = _start_button(body)

    assert button is not None and "disabled" not in button
    assert set(readiness.done(client.app)) >= {
        "/",
        "/config",
        "/calibration",
    }


# =============================================================================
# 2. One answer, shared with the nav ticks
# =============================================================================


def test_the_gate_and_the_nav_ticks_are_one_answer(tmp_path, monkeypatch):
    """A second notion of "ready" would let the tabs and the button disagree,
    and nothing on screen would say which was lying. Walked across three
    states, so this cannot pass by both being constant."""
    client = _station(tmp_path, applied=False)
    every = {"/", "/config", "/calibration"}
    seen = []
    for arrange in (
        lambda: None,
        lambda: config_page._apply(client, head=False),
        lambda: _scan_found_every_device(client, monkeypatch),
        lambda: _pass_the_calibration(monkeypatch),
    ):
        arrange()
        done = set(readiness.done(client.app))
        offered = not _blocked(client.get("/record").text)
        seen.append(offered)
        assert offered == (done >= every), done
    assert seen == [False, False, False, True], seen


def test_a_device_the_scan_could_not_open_leaves_start_unavailable(
    tmp_path, monkeypatch
):
    """Enumerated is not verified. A gripper on a root-only /dev/ttyACM1 is
    exactly the healthy-looking rig that fails at record time."""
    client = _station(tmp_path)
    wedged = config_page._found(DECLARED[0], state="unavailable") + config_page._found(
        *DECLARED[1:]
    )
    monkeypatch.setattr(devices_core, "outcome_for", lambda job: (wedged, None))

    assert _blocked(client.get("/record").text)


# =============================================================================
# 3. Saying which one, in the operator's words
# =============================================================================


def test_the_page_names_the_outstanding_step_and_how_to_fix_it(tmp_path):
    """ "Prerequisites not met" sends an operator hunting. The step has a name
    on the tab bar and a page that fixes it, and both belong on screen."""
    client = _station(tmp_path, applied=False)

    body = client.get("/record").text

    # Start saves Configuration itself; a fresh rig still requires calibration.
    assert "Calibration has not passed" in body
    assert 'href="#calibration"' in body
    assert "start calibration to save it automatically" not in body


def test_the_outstanding_step_is_the_first_one_not_the_last(tmp_path):
    """A station with nothing chosen is sent to Setup, not to a Configuration
    page that cannot be filled in until Setup has picked its template.

    The page's own words for this case are `_context`'s `guard`, which carries
    a Go to Setup button; what is asserted here is that both agree about which
    step is outstanding.
    """
    client = _station(tmp_path, applied=False)
    client.app.state.rig.draft.preset_id = None

    body = client.get("/record").text

    assert readiness.blocking(client.app) == readiness.SETUP
    assert "Setup" in body
    assert 'action="/"' in body or 'href="/"' in body
    assert not _has_start_form(body), "no preset, and Start was still offered"


def test_the_disabled_start_is_marked_not_merely_greyed(tmp_path):
    """The same property
    `test_design_system.py::test_a_disabled_nav_item_is_marked_not_merely_greyed`
    holds for the nav: state is never colour alone."""
    client = _station(tmp_path)

    button = _start_button(client.get("/record").text)

    assert button is not None and "disabled" in button
    assert button.get("aria-disabled") == "true", f"greyed but unmarked: {button}"
    # And the reason is attached to the control, not only shown somewhere on
    # the page -- the status card is swapped on its own by htmx.
    described = button.get("aria-describedby")
    assert described and f'id="{described}"' in client.get("/record").text


# =============================================================================
# 4. The server refuses, whatever the page rendered
# =============================================================================


def test_the_server_refuses_a_start_when_a_step_is_outstanding(tmp_path, monkeypatch):
    """The server refuses, whatever the page rendered.

    `preflight` validates the config, not the workflow, so without this gate a
    valid config on a rig whose earlier steps never passed would go straight
    through to `jobs.start`, and the disabled Start button would guard nothing a
    replayed or non-JS POST could not step around. The gate must refuse at the
    point of action.

    Isolated from config validity on purpose: `preflight` is stubbed to a launch
    that would happily succeed, and the recorder launcher is rigged to fail the
    test if it is ever reached. With the gate the request is refused before
    either runs; without it, `jobs.start` fires and the assertion trips.
    """
    client = _station(tmp_path)
    assert readiness.blocking(client.app) is not None, "a step must be outstanding"

    monkeypatch.setattr(record, "preflight", lambda *a, **k: ["would-launch"])

    def _must_not_start(*a, **k):
        raise AssertionError("a recording was started with a step outstanding")

    monkeypatch.setattr(record.jobs, "start", _must_not_start)

    assert _post_start(client).status_code == 400


# Waive discovery gating only for device kinds with no enumerator.


def test_the_gate_waives_only_what_the_scan_cannot_look_for():
    """The predicate the gate turns on: every kind the scan enumerates must
    connect, and only a kind nothing knows how to look for is waived -- so the
    waiver cannot quietly pass a camera the scan could have checked."""
    make = devices_core.DeclaredDevice

    assert not devices_core.is_unscanned_kind(make("OAK", "o", kind="oak"))
    assert not devices_core.is_unscanned_kind(make("iPhone", "i", kind="iphone"))
    assert not devices_core.is_unscanned_kind(make("Gripper", "g", kind="gripper"))
    assert not devices_core.is_unscanned_kind(make("Trigger", "0", kind="uvc"))
    assert not devices_core.is_unscanned_kind(make("Required cam", "c", kind=None))
    # The other side of the same predicate, so the waiver is proved to still
    # exist rather than merely proved to be off.
    assert devices_core.is_unscanned_kind(make("Lidar", "l", kind="lidar"))


def test_changing_a_device_and_back_re_matches_the_calibration_stamp(tmp_path):
    """The same self-heal for calibration: `calibration.stale` compares the
    stamp to the current fingerprint, so restoring the rig clears the staleness
    the change introduced -- no re-calibration for a rig that is identical again."""
    client = _station(tmp_path)
    draft = client.app.state.rig.draft
    cal.calibration_state(client.app).stamp(
        cal.STEP_VERIFY, session.device_fingerprint(draft)
    )
    assert not cal.stale(client.app, cal.STEP_VERIFY)

    original = dict(draft.grippers)
    draft.grippers["left"] = "UMI-V4-SWAPPED"
    assert cal.stale(client.app, cal.STEP_VERIFY), "a genuine change should go stale"

    draft.grippers.clear()
    draft.grippers.update(original)
    assert not cal.stale(client.app, cal.STEP_VERIFY), "change-and-back stayed stale"


def test_applying_a_changed_rig_resets_the_calibration(tmp_path, monkeypatch):
    """A config change that alters the rig gives a clean state -- the recorded
    calibration runs go, so nothing from the old rig carries onto the new one.

    What survives is as load-bearing as what goes: the calibration stamp stays,
    so the operator is told the rig changed rather than just that a step is not
    done (see the expiry assertion below)."""
    client = _station(tmp_path)
    _scan_found_every_device(client, monkeypatch)
    old_fingerprint = session.device_fingerprint(client.app.state.rig.draft)
    cal_state = cal.calibration_state(client.app)
    cal_state.stamp(cal.STEP_VERIFY, old_fingerprint)
    cal_state.record(cal.STEP_GRIPPER, cal.Run(key="left", job_id="j"), old_fingerprint)

    client.app.state.rig.draft.grippers["left"] = "UMI-V4-NEW"
    reset = readiness.reset_verification_for_new_rig(client.app)

    assert reset is True
    # Cleared: the old rig's runs must not carry over.
    assert not cal_state.runs(cal.STEP_GRIPPER), "the old rig's runs survived"
    # Kept: the calibration stamp, so the step reads expired rather than merely
    # unfinished (`cal.stale`).
    assert cal.stale(client.app, cal.STEP_VERIFY)
    assert readiness.blocking(client.app) is readiness.CALIBRATION


def test_applying_an_identical_rig_keeps_the_calibration(tmp_path, monkeypatch):
    """Only a genuine device change resets: re-applying the same rig (a task-name
    typo fix, say) must not throw away a passed calibration."""
    client = _station(tmp_path)
    _scan_found_every_device(client, monkeypatch)
    fp = session.device_fingerprint(client.app.state.rig.draft)
    cal.calibration_state(client.app).stamp(cal.STEP_VERIFY, fp)

    reset = readiness.reset_verification_for_new_rig(client.app)

    assert reset is False, "an unchanged rig reset verification anyway"
    assert cal.calibration_state(client.app).stamp_of(cal.STEP_VERIFY) == fp


# ------------------------------------------- the inventory scan is not the test


def _finished_recording(client):
    """A recording that has run and ended, as `WorkflowState` remembers one."""
    state = record.workflow_state(client.app)
    job = jobs.start(state.jobs, [sys.executable, "-c", "pass"], cwd=REPO, timeout=30)
    end = time.monotonic() + 10
    while not job.finished and time.monotonic() < end:
        time.sleep(0.02)
    assert job.finished, "the fake recording never ended"
    state.record_job_id = job.id
    state.steps_done.add(record.RECORD_STEP)
    return job


def test_applying_a_changed_rig_clears_the_recording_status(tmp_path, monkeypatch):
    """The third page behind Configuration. Everything else on /record is derived
    per render -- the folder, the task, the pre-flight guard all come back out of
    the config file -- but the status badge, the log under it and the Record tick
    live in `WorkflowState`, so a rig change has to clear it, or a green
    "Stopped" would sit over the previous rig's recording, with its log."""
    client = _station(tmp_path)
    _finished_recording(client)
    assert record._current_job(record.workflow_state(client.app)) is not None

    client.app.state.rig.draft.grippers["left"] = "UMI-V4-NEW"
    readiness.reset_verification_for_new_rig(client.app)

    state = record.workflow_state(client.app)
    assert state.record_job_id is None, "the old rig's recording is still reported"
    assert record.RECORD_STEP not in state.steps_done, "the Record step stayed ticked"


def test_re_applying_the_same_rig_keeps_the_recording_status(tmp_path, monkeypatch):
    """The other half. An operator who fixed a typo in their own name and pressed
    Apply again has not changed the rig, and must not watch the console forget
    the session they just recorded."""
    client = _station(tmp_path)
    job = _finished_recording(client)

    readiness.reset_verification_for_new_rig(client.app)

    assert record.workflow_state(client.app).record_job_id == job.id


def test_a_live_recording_is_never_forgotten(tmp_path, monkeypatch):
    """`jobs.start` detaches the child, so dropping the job id of a running
    recorder does not stop it -- it goes on writing episodes and holding every
    camera, with the Stop button (which needs that id) gone from the page."""
    client = _station(tmp_path)
    state = record.workflow_state(client.app)
    job = jobs.start(
        state.jobs,
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=REPO,
        timeout=60,
    )
    state.record_job_id = job.id
    state.steps_done.add(record.RECORD_STEP)
    try:
        client.app.state.rig.draft.grippers["left"] = "UMI-V4-NEW"
        readiness.reset_verification_for_new_rig(client.app)
        assert state.record_job_id == job.id, "a running recorder was orphaned"
        assert record.RECORD_STEP in state.steps_done
    finally:
        jobs.cancel(state.jobs, job.id)


def test_starting_a_new_session_clears_the_recording_status(tmp_path, monkeypatch):
    """A new session is a different recording, so the status card must not still
    report the one belonging to the session just left. The rig has not changed,
    so the device test and the calibration deliberately stay -- that is the whole
    point of `session.start_new_session`."""
    client = _station(tmp_path)
    _scan_found_every_device(client, monkeypatch)
    _finished_recording(client)

    client.post("/config/session/uid", data={"config_form": "1", "confirm": "yes"})

    state = record.workflow_state(client.app)
    assert state.record_job_id is None, "the previous session's recording is reported"
    assert record.RECORD_STEP not in state.steps_done

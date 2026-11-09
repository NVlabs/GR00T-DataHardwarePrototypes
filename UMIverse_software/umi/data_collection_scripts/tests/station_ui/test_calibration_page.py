# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`/calibration`: the browser orchestrates, the station runs the windows.

Both calibration steps are operator-attended native OpenCV or terminal windows
that cannot move into a browser, so this page never tries. What it guarantees:

* No command this page builds contains `sudo`: a web server has no TTY to
  answer a password prompt. The udev rule at the repo root grants device
  access instead.
* The child's exit code is the only thing that decides success; nothing on the
  page lets an operator attest it.
* A non-zero exit renders the captured output, so a failure leaves a trace.
* Step completion is server state, so a restart mid-calibration does not show
  every step pending.

The subprocess boundary is the mock point: `calibration.LAUNCHER` and
`calibration.TERMINAL_ARGV` are swapped for a stub runner, so real jobs with
real pids and real exit codes run, and no hardware is opened.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
from html.parser import HTMLParser

import pytest
from fastapi.testclient import TestClient

from station_ui import calibration as cal
from station_ui.app import create_app
from station_ui.errors import FieldError
from station_ui.routes.calibration import _role_label

REPO = pathlib.Path(__file__).resolve().parents[2]

# The stub the LAUNCHER is replaced with. It stands in for `uv run <script>`:
# it echoes the argv so a test can assert what was really passed, and honours
# UMI_FAKE_RC so a test can choose the exit code.
STUB = r"""
import os, sys
argv = sys.argv[1:]
print("STUB ARGV: " + " ".join(argv))
rc = int(os.environ.get("UMI_FAKE_RC", "0"))
if rc:
    print("stub failed on purpose: gripper did not respond", file=sys.stderr)
sys.stdout.flush()
sys.exit(rc)
"""


@pytest.fixture
def stub_launcher(monkeypatch, tmp_path):
    """Run real child processes, but never a real script and never hardware."""
    stub = tmp_path / "stub_runner.py"
    stub.write_text(STUB)
    monkeypatch.setattr(cal, "LAUNCHER", (sys.executable, str(stub)))
    # No terminal emulator in a test: the wrapper shell runs directly, so the
    # child's exit code also arrives as the job's own.
    monkeypatch.setattr(cal, "TERMINAL_ARGV", ())
    monkeypatch.setattr(cal, "TEMP_DIR", str(tmp_path / "umi_temp"))
    return stub


def apply_config(app, devices=None) -> pathlib.Path:
    """Write the config Apply would have written, which is what this page reads.

    `/calibration` reads the generated config, never the template it came from
    -- the templates carry placeholder device names, and a check pointed at
    those opens devices that are on nobody's bench. So the fixture has to have
    applied one, exactly as an operator reaching step 4 has.

    Built from the template rather than by driving `/config/apply`, so a test
    about calibration does not also have to satisfy the configuration form; the
    shape is the template's, which is what `verification_roles` reads.
    """
    from station_ui import readiness
    from station_ui.core import setup_presets as sp

    draft = app.state.rig.draft
    template = sp.template_path(sp.get_preset(draft.preset_id), cal.repo_root())
    text = template.read_text()
    for placeholder, real in (devices or {}).items():
        text = text.replace(placeholder, real)

    path = readiness.generated_config(app)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _client(app=None, apply=True) -> tuple[object, TestClient]:
    """A browser that owns the rig and has chosen a bimanual V4 setup, which is
    what an operator reaching /calibration has always already done."""
    app = app or create_app()
    client = TestClient(app)
    client.get("/")
    client.post("/setup/preset", data={"preset_id": "bimanual_v4"})
    app.state.rig.draft.grippers.update({"left": "UMI-V4-0001", "right": "UMI-V4-0002"})
    if apply:
        apply_config(app)
    return app, client


# ---------------------------------------------------------------- html parsing


class _Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.forms: list[dict] = []
        self.inline_handlers: list[str] = []
        self.badges: list[dict] = []
        self._form: dict | None = None
        self._button: dict | None = None
        self._badge: dict | None = None

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        self.inline_handlers += [k for k in attr if k.startswith("on")]
        classes = (attr.get("class") or "").split()
        if "badge" in classes:
            self._badge = {**attr, "classes": classes, "text": ""}
            self.badges.append(self._badge)
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
            self._button = {**attr, "classes": classes, "text": ""}
            self._form["buttons"].append(self._button)

    def handle_endtag(self, tag):
        if tag == "form":
            self._form = None
        elif tag == "button":
            self._button = None
        elif tag == "span" and self._badge is not None:
            self._badge = None

    def handle_data(self, data):
        if self._button is not None:
            self._button["text"] += data
        if self._badge is not None:
            self._badge["text"] += data


def _parse(html: str) -> _Page:
    page = _Page()
    page.feed(html)
    return page


def _forms_to(html: str, action: str) -> list[dict]:
    return [f for f in _parse(html).forms if f["action"] == action]


def _text(node: dict) -> str:
    return " ".join(node["text"].split())


def _settled(app, timeout: float = 20.0) -> None:
    """Wait for every job this app has started to reach a terminal state."""
    registry = cal.job_registry(app)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        jobs = registry.all()
        if jobs and all(job.finished for job in jobs):
            return
        time.sleep(0.02)
    raise AssertionError(f"jobs never settled: {[j.state for j in registry.all()]}")


def _run_step(client, app, action: str, data: dict) -> object:
    resp = client.post(action, data=data, follow_redirects=False)
    assert resp.status_code == 303, f"{action} answered {resp.status_code}"
    step = cal.STEP_VERIFY if action == "/calibration/check/start" else cal.STEP_GRIPPER
    assert resp.headers["location"] == f"/calibration#step-{step}"
    _settled(app)
    return resp


def _first(html: str, action: str) -> dict:
    forms = _forms_to(html, action)
    assert forms, f"the page offers no form posting to {action}"
    return forms[0]


def _run_every_unit(client, app, action: str) -> None:
    """Run every control a step offers. A step is done when every camera or
    gripper it covers has passed, not when the first one has."""
    forms = _forms_to(client.get("/calibration").text, action)
    assert forms, f"the page offers no form posting to {action}"
    for form in forms:
        _run_step(client, app, action, form["fields"])


# --------------------------------------------------------- a step is a job


def test_each_step_starts_a_real_child_process_and_records_its_job_id(
    stub_launcher,
):
    """Not a detached terminal: a pid, a captured log, and an exit code.

    A job's outcome can always be read back, rather than being lost with a
    discarded process handle.
    """
    app, client = _client()
    page = client.get("/calibration").text

    for action in (
        "/calibration/check/start",
        "/calibration/calibrate-gripper",
    ):
        form = _first(page, action)
        _run_step(client, app, action, form["fields"])

    jobs = cal.job_registry(app).all()
    assert len(jobs) == 2, f"{len(jobs)} jobs for two steps"
    for job in jobs:
        assert job.exit_code == 0
        assert job.lines(), f"{job.label}: nothing captured from the child"

    # The check is one child for the whole rig, so it is remembered by job id
    # rather than as per-unit runs; the gripper still records one run per side.
    assert cal.check_state(app).job_id, "the check job was not remembered"
    assert cal.calibration_state(app).runs(cal.STEP_GRIPPER), "gripper recorded no run"


def test_each_step_is_a_separate_post_no_step_shares_an_endpoint(stub_launcher):
    app, client = _client()
    page = client.get("/calibration").text
    actions = {f["action"] for f in _parse(page).forms if f["method"] == "post"}
    assert {
        "/calibration/check/start",
        "/calibration/calibrate-gripper",
    } <= actions


# ------------------------------------------------------------- the exit code


def test_a_zero_exit_is_the_only_thing_that_marks_a_step_done(stub_launcher):
    app, client = _client()
    _run_every_unit(client, app, "/calibration/calibrate-gripper")

    body = client.get("/calibration").text
    assert "✓" in body
    assert cal.step_state(app, cal.STEP_GRIPPER) == cal.DONE


def test_a_non_zero_exit_renders_failed_with_the_captured_output(
    stub_launcher, monkeypatch
):
    """A failing child must be impossible to mistake for a pass, and its output
    comes back with it.

    `tee` brings the calibration's own output back (`calibration.gripper_argv`):
    the terminal's stdout is empty by construction, because `gnome-terminal` is
    a client to a session server and says nothing about its child. On a failure
    that output is the answer, so it is expanded. The verdict still leads: the
    badge and the reason are above it, and the disclosure is collapsed on a
    pass, which the test below pins.
    """
    monkeypatch.setenv("UMI_FAKE_RC", "3")
    app, client = _client()
    form = _first(client.get("/calibration").text, "/calibration/calibrate-gripper")
    _run_step(client, app, "/calibration/calibrate-gripper", form["fields"])

    assert cal.step_state(app, cal.STEP_GRIPPER) == cal.FAILED
    body = client.get("/calibration").text
    assert "✕ Failed" in body
    assert "✓ Done" not in body
    assert "gripper did not respond" in body, "the operator is left with a code alone"
    assert "<details" in body, "the diagnostic is inlined rather than disclosed"
    assert "/log" in body, "a failed step offers no way to the output at all"


def test_a_passing_calibration_keeps_its_output_collapsed(stub_launcher):
    """The verdict is the card. On a pass nobody needs the narration in front of
    them -- but it is one click away, which is the whole point of a disclosure."""
    app, client = _client()
    form = _first(client.get("/calibration").text, "/calibration/calibrate-gripper")
    _run_step(client, app, "/calibration/calibrate-gripper", form["fields"])

    body = client.get("/calibration").text

    assert "✓ Done" in body
    assert "<details" in body, "a passing step hides its output entirely"
    assert '<details class="job-details" open>' not in body, "expanded on a pass"


def test_the_calibrations_own_output_survives_the_terminal(stub_launcher):
    """The step's "Full log" carries the calibration's output.

    The job captures `gnome-terminal`'s stdout, and a terminal emulator says
    nothing about its child -- so without this a failed calibration would leave
    the browser holding an exit code and nothing else. The wrapper writes the
    status to a file for exactly this reason; the output goes the same way.
    """
    app, client = _client()
    form = _first(client.get("/calibration").text, "/calibration/calibrate-gripper")
    _run_step(client, app, "/calibration/calibrate-gripper", form["fields"])

    run = cal.calibration_state(app).run(cal.STEP_GRIPPER, form["fields"]["side"])
    captured = cal.gripper_log(run)

    assert captured, "the calibration's output was not captured at all"
    assert any("STUB ARGV" in line for line in captured)


def _write_raw(path, text) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def test_a_redrawn_progress_line_does_not_bury_the_log(stub_launcher, tmp_path):
    """The recording pass redraws a status line with a carriage return at 30Hz.
    Kept whole, fifteen seconds of it is one line carrying ~450 redraws, and the
    lines that matter scroll off behind it."""
    log = tmp_path / "cal.log"
    # newline="" on the write as well: without it the fixture's carriage returns
    # are translated on the way to disk and the test passes against a file that
    # never had the problem in it.
    _write_raw(
        log,
        "Starting gripper calibration...\n"
        "   [ 1.0s] Distance:  10.00 mm\r   [ 2.0s] Distance:  20.00 mm"
        "\r   [ 3.0s] Distance:  30.00 mm\n"
        "Total frames recorded: 450\n",
    )
    run = cal.Run(key="left", job_id="x", log_path=str(log))

    assert cal.gripper_log(run) == [
        "Starting gripper calibration...",
        "   [ 3.0s] Distance:  30.00 mm",
        "Total frames recorded: 450",
    ]


def test_a_calibration_that_wrote_no_log_says_nothing(stub_launcher, tmp_path):
    """Every way there is nothing to show -- no file, an unreadable one, a child
    that died first -- means the same thing to the page."""
    assert cal.gripper_log(cal.Run(key="left", job_id="x")) == []
    assert (
        cal.gripper_log(
            cal.Run(key="left", job_id="x", log_path=str(tmp_path / "nope.log"))
        )
        == []
    )


def _report_fault(app, **overrides):
    """Write what the calibration writes when the operator reports the hardware."""
    run = next(iter(cal.calibration_state(app).runs(cal.STEP_GRIPPER).values()))
    fault = {
        "check": "close_repeatability",
        "measured_deg": 7.31,
        "limit_deg": 2.0,
        "attempts": 2,
        "local_path": "/home/umi/.umi_temp/gripper_faults/2026-08-06_UMI-035.json",
    }
    fault.update(overrides)
    pathlib.Path(run.summary_path).write_text(json.dumps({"hardware_fault": fault}))
    return run


def test_a_reported_hardware_fault_replaces_the_movement_numbers(
    stub_launcher, monkeypatch
):
    """The recording pass never runs on this path, so there are no numbers.

    Drawn as a banner rather than the usual muted line: "this gripper is out of
    service" has to reach the next person at the bench without them opening the
    log.
    """
    monkeypatch.setenv("UMI_FAKE_RC", "3")
    app, client = _client()
    action = "/calibration/calibrate-gripper"
    form = _first(client.get("/calibration").text, action)
    _run_step(client, app, action, form["fields"])
    _report_fault(app)

    body = client.get("/calibration").text

    assert "Hardware fault reported" in body
    assert "7.31" in body and "2.00" in body, "the numbers behind the verdict are gone"
    assert "after 2 attempts" in body
    assert "no config was written" in body
    # The station holds the only copy; say where it is.
    assert "Report saved to" in body
    assert "gripper_faults/2026-08-06_UMI-035.json" in body
    # The movement/jump line has nothing to say here and must not claim zeroes.
    assert "did not move" not in body
    assert cal.step_state(app, cal.STEP_GRIPPER) == cal.FAILED


def test_a_gripper_exiting_two_is_a_plain_failure_not_a_cancellation(
    stub_launcher, monkeypatch
):
    """Exit code 2 is an ordinary failure here.

    This view serves only the gripper step, which the operator cannot cancel.
    A gripper that exits 2 has failed for whatever reason the log gives, and
    labelling it an operator decision would tell somebody to stop investigating.
    """
    monkeypatch.setenv("UMI_FAKE_RC", "2")
    app, client = _client()
    action = "/calibration/calibrate-gripper"
    form = _first(client.get("/calibration").text, action)
    _run_step(client, app, action, form["fields"])

    run = next(iter(cal.calibration_state(app).runs(cal.STEP_GRIPPER).values()))
    registry = cal.job_registry(app)
    view = cal.run_view(registry, run.key, run.label, run)

    assert view.state == cal.FAILED
    assert cal.child_exit_code(registry, run) == 2
    assert not hasattr(view, "cancelled"), "exit 2 still carries a cancelled flag"
    assert not hasattr(cal, "EXIT_USER_CANCELLED")

    assert cal.step_state(app, cal.STEP_GRIPPER) == cal.FAILED
    body = client.get("/calibration").text
    assert "✕ Failed" in body
    # The exit code is a number an operator cannot interpret, beside a badge
    # that already says Failed. What the page owes them here is the verdict and
    # a way to the output, both of which are above.
    assert "Exit status" not in body, "the operator is shown a raw exit code"


def test_no_control_on_the_page_can_declare_a_step_successful(stub_launcher):
    """No control on this page may mark a calibration done: if such a form
    existed, an operator could mark a failed calibration done."""
    app, client = _client()
    page = _parse(client.get("/calibration").text)
    for form in page.forms:
        for button in form["buttons"]:
            label = _text(button).lower()
            assert "success" not in label, f"attestation button: {label!r}"
            assert "mark done" not in label, f"attestation button: {label!r}"

    # And there is no such endpoint hiding behind a hand-written POST either.
    for action in (
        "/calibration/done",
        "/calibration/complete",
        "/calibration/success",
    ):
        assert client.post(action, follow_redirects=False).status_code == 404


@pytest.mark.parametrize(
    "damage,expected_code",
    [
        # The wrapper never got there: segfault, OOM kill, terminal torn down.
        ("missing", 3),
        # Killed between opening the file and finishing the write.
        ("truncated", 3),
        # The file exists but says nothing a status can be read out of.
        ("garbage", None),
    ],
)
def test_a_child_that_never_recorded_its_status_is_not_a_pass(
    stub_launcher, monkeypatch, damage, expected_code
):
    """The recorded status is written by the wrapper *after* the calibration
    returns, so a child that dies hard never produces one. `child_exit_code`
    then falls back to the job's own exit code -- and that fallback is the one
    branch where being wrong would report a hardware fault as a success, which
    is exactly what an operator-attested success button would allow. The
    primary path is covered by every other test in this file; this one exists
    only to hold the fallback."""
    monkeypatch.setenv("UMI_FAKE_RC", "3")
    app, client = _client()
    action = "/calibration/calibrate-gripper"
    form = _first(client.get("/calibration").text, action)
    _run_step(client, app, action, form["fields"])

    run = next(iter(cal.calibration_state(app).runs(cal.STEP_GRIPPER).values()))
    recorded = pathlib.Path(run.status_path)
    assert recorded.is_file(), "the wrapper recorded no status to damage"
    if damage == "missing":
        recorded.unlink()
    elif damage == "truncated":
        recorded.write_text("")
    else:
        recorded.write_text("Terminated")

    registry = cal.job_registry(app)
    assert cal.child_exit_code(registry, run) == expected_code
    assert cal.run_state(registry, run) == cal.FAILED
    assert cal.step_state(app, cal.STEP_GRIPPER) == cal.FAILED

    body = client.get("/calibration").text
    assert "✕ Failed" in body
    assert "✓ Done" not in body, "a child that died hard rendered as a pass"


def test_a_gripper_command_that_cannot_even_be_spawned_is_a_failure_not_a_pass(
    stub_launcher, monkeypatch
):
    """`uv` missing from PATH, or a script that has been moved. `jobs.start`
    turns that into a failed job with no exit code at all, and a run with no
    recorded status has nothing else to fall back to.

    The camera check reaches the same conclusion by a different route -- it has
    no `Run` and no status file at all -- which is held by
    `test_camera_check.py::test_a_check_that_cannot_even_be_spawned_reads_failed_not_pending`.
    """
    # A headless station with no gnome-terminal installed. Nothing runs, so the
    # wrapper never records a status and the fallback to the job's own outcome
    # is the only thing standing between this and a step that reads "pending".
    monkeypatch.setattr(cal, "TERMINAL_ARGV", ("/nonexistent/gnome-terminal", "--"))
    app, client = _client()
    action = "/calibration/calibrate-gripper"
    form = _first(client.get("/calibration").text, action)
    _run_step(client, app, action, form["fields"])

    run = next(iter(cal.calibration_state(app).runs(cal.STEP_GRIPPER).values()))
    registry = cal.job_registry(app)
    assert cal.child_exit_code(registry, run) is None
    assert cal.run_state(registry, run) == cal.FAILED
    assert cal.step_state(app, cal.STEP_GRIPPER) == cal.FAILED

    body = client.get("/calibration").text
    assert "✕ Failed" in body
    assert "could not start" in body, "the operator is not told what went wrong"


def test_a_failed_step_can_be_re_run_and_a_later_pass_replaces_the_failure(
    stub_launcher, monkeypatch
):
    app, client = _client()
    action = "/calibration/calibrate-gripper"
    form = _first(client.get("/calibration").text, action)

    monkeypatch.setenv("UMI_FAKE_RC", "3")
    _run_step(client, app, action, form["fields"])
    assert cal.step_state(app, cal.STEP_GRIPPER) == cal.FAILED

    monkeypatch.setenv("UMI_FAKE_RC", "0")
    _run_every_unit(client, app, action)
    assert cal.step_state(app, cal.STEP_GRIPPER) == cal.DONE
    assert "✕ Failed" not in client.get("/calibration").text


# ------------------------------------------------------------- persistence


def test_step_state_survives_a_reload_and_is_not_held_in_the_page(stub_launcher):
    """Step completion lives on the server, so a reload -- and a second browser
    -- see the same thing, and a restart mid-calibration does not show every
    step pending."""
    app, client = _client()
    _run_every_unit(client, app, "/calibration/calibrate-gripper")

    first = client.get("/calibration").text
    second = client.get("/calibration").text
    assert first.count("✓ Done") == second.count("✓ Done") >= 1

    # A different browser, no cookie: read-only, but it sees the same progress.
    other = TestClient(app)
    assert "✓ Done" in other.get("/calibration").text


# ----------------------------------------------- concise terminal guidance


def test_the_page_uses_the_requested_brief_terminal_instruction(stub_launcher):
    """Detailed close/open prompts belong to the terminal calibration flow."""
    _, client = _client()
    body = client.get("/calibration").text.lower()

    assert "follow the steps in the terminal" in body
    assert "close the gripper fully and press enter" not in body
    assert "carry this browser to the rig" not in body
    # There is no per-role OpenCV window, so no q-to-cancel key either.
    assert "press q" not in body


def test_the_page_does_not_pretend_to_stream_camera_frames(stub_launcher):
    _, client = _client()
    body = client.get("/calibration").text
    for forbidden in ("<video", "<canvas", "mjpeg", "/stream.mjpg"):
        assert forbidden not in body.lower()


# --------------------------------------------------------------- no sudo


def test_no_command_this_page_builds_asks_for_sudo(stub_launcher):
    """The udev rule replaced the chmod. A web server has no TTY to answer a
    password prompt on, so a single `sudo` anywhere here hangs a job until its
    wall-clock cap fires."""
    app, client = _client()
    page = client.get("/calibration").text
    for action in (
        "/calibration/check/start",
        "/calibration/calibrate-gripper",
    ):
        _run_step(client, app, action, _first(page, action)["fields"])

    for job in cal.job_registry(app).all():
        flat = " ".join(job.argv)
        assert "sudo" not in flat, f"{job.label} runs sudo: {flat}"
        assert "chmod" not in flat, f"{job.label} chmods a device: {flat}"
        for line in job.lines():
            assert "sudo" not in line


# ------------------------------------------------------------------ ownership


@pytest.mark.parametrize(
    "action,data",
    [
        ("/calibration/check/start", {}),
        ("/calibration/check/stop", {}),
        ("/calibration/calibrate-gripper", {"side": "right"}),
    ],
)
def test_a_browser_that_does_not_own_the_rig_cannot_start_hardware(
    stub_launcher, action, data
):
    app, owner = _client()
    onlooker = TestClient(app)
    onlooker.get("/calibration")

    resp = onlooker.post(action, data=data, follow_redirects=False)
    assert resp.status_code == 409
    assert cal.job_registry(app).all() == [], "a read-only tab started hardware"


def test_a_read_only_tab_is_shown_the_progress_but_not_the_buttons(stub_launcher):
    app, _owner = _client()
    onlooker = TestClient(app)
    page = _parse(onlooker.get("/calibration").text)
    live = [
        b
        for f in page.forms
        for b in f["buttons"]
        if f["action"] and f["action"].startswith("/calibration/")
    ]
    assert live, "no step controls rendered at all -- the assertion is vacuous"
    for button in live:
        assert "disabled" in button, f"a read-only tab is offered {_text(button)!r}"


# ---------------------------------------------------------------- guard rails


def test_without_a_setup_preset_the_page_explains_instead_of_offering_hardware():
    app = create_app()
    client = TestClient(app)
    client.get("/")  # claims the rig, but chooses nothing

    body = client.get("/calibration").text
    assert "setup" in body.lower()
    assert not _forms_to(body, "/calibration/check/start")

    resp = client.post("/calibration/check/start", follow_redirects=False)
    assert resp.status_code == 409


def test_an_unknown_gripper_side_is_a_400_not_a_500(stub_launcher):
    """The check takes no unit at all -- one scan covers the rig -- so the
    gripper side is the last operator-supplied key this page dereferences."""
    app, client = _client()
    resp = client.post(
        "/calibration/calibrate-gripper",
        data={"side": "middle"},
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert cal.job_registry(app).all() == []


def test_an_unknown_job_id_is_a_404_not_a_crash(stub_launcher):
    _, client = _client()
    assert client.get("/calibration/jobs/deadbeef/log").status_code == 404


# ------------------------------------------------------ argv actually built


def test_the_check_command_watches_the_whole_rig_and_names_the_config(stub_launcher):
    """One child for every camera. A `--role` per invocation would mean six
    subprocesses and six trips to the station monitor."""
    app, client = _client()
    form = _first(client.get("/calibration").text, "/calibration/check/start")
    _run_step(client, app, "/calibration/check/start", form["fields"])

    argv = cal.job_registry(app).all()[0].argv
    assert "--watch" in argv
    assert "--role" not in argv
    config = pathlib.Path(argv[argv.index("--config") + 1])
    assert config.is_file(), f"the config passed to the child does not exist: {config}"


# Production templates contain placeholder serials; assembly must reject
# them for an active recording config.
TEMPLATE_PLACEHOLDERS = (
    "oak-left-wrist",
    "oak-right-wrist",
    "iphone-left-tracking",
    "iphone-right-tracking",
)
REAL_HARDWARE = {
    "oak-left-wrist": "oak-FFC-42",
    "oak-right-wrist": "oak-FFC-43",
    "iphone-left-tracking": "iphone15p-77",
    "iphone-right-tracking": "iphone15p-78",
}


def test_the_check_reads_the_applied_config_not_the_template(stub_launcher):
    """The check reads the generated config, not the template.

    The template names placeholder hardware that is attached to no station, so
    a check run against it would report every role `open_failed` within about a
    second and badge the step Failed while the real cameras sat untouched.
    """
    app, client = _client(apply=False)
    applied = apply_config(app, devices=REAL_HARDWARE)

    form = _first(client.get("/calibration").text, "/calibration/check/start")
    _run_step(client, app, "/calibration/check/start", form["fields"])

    argv = cal.job_registry(app).all()[0].argv
    handed = pathlib.Path(argv[argv.index("--config") + 1])

    assert handed == applied, f"the check read {handed}, not the applied config"
    text = handed.read_text()
    for placeholder in TEMPLATE_PLACEHOLDERS:
        assert placeholder not in text, (
            f"the check would open {placeholder}, which is a template placeholder"
        )
    assert "oak-FFC-42" in text and "iphone15p-77" in text


def test_the_check_and_the_recorder_are_handed_the_same_file(stub_launcher):
    """One definition of "the config this station records from".

    Two derivations could let the check prove a different rig from the one that
    records -- while the operator is told the cameras passed.
    """
    from station_ui import readiness

    app, _client_ = _client()

    assert cal.recording_config_path(app) == readiness.generated_config(app)


def test_calibration_controls_exist_before_the_first_action_saves_configuration(
    stub_launcher,
):
    """Render the controls before saving, without opening placeholder devices.

    The launch action will validate and save the complete configuration form.
    Reading a recording config still requires that validated file to exist.
    """
    app, client = _client(apply=False)

    body = client.get("/calibration").text

    assert len(_forms_to(body, "/calibration/calibrate-gripper")) == 2
    with pytest.raises(FieldError):
        cal.recording_config(app)


def test_the_page_lists_every_configured_camera_behind_one_start_control(stub_launcher):
    """Six rows, one button. Six buttons would mean clicking here and walking
    there six times over."""
    app, client = _client()
    roles = cal.verification_roles(cal.recording_config(app))
    assert len(roles) == 6, f"bimanual V4 should have six roles, got {roles}"

    body = client.get("/calibration").text
    for role in roles:
        assert _role_label(role) in body, f"{role} is not named on the page"
    assert len(_forms_to(body, "/calibration/check/start")) == 1


def test_the_gripper_command_carries_the_chosen_gripper_and_its_save_path(
    stub_launcher,
):
    app, client = _client()
    form = _first(client.get("/calibration").text, "/calibration/calibrate-gripper")
    side = form["fields"]["side"]
    _run_step(client, app, "/calibration/calibrate-gripper", form["fields"])

    job = cal.job_registry(app).all()[0]
    printed = "\n".join(job.lines())
    assert app.state.rig.draft.grippers[side] in printed
    assert "UMI_GRIPPER_AUTO_SAVE_PATH" in " ".join(job.argv)
    assert f"{side}_umi_gripper_config.yaml" in " ".join(job.argv)


# ------------------------------------------------- the gripper step's results


def _gripper_argv(app) -> list[str]:
    return cal.job_registry(app).all()[0].argv


def _start_gripper(client, app):
    form = _first(client.get("/calibration").text, "/calibration/calibrate-gripper")
    _run_step(client, app, "/calibration/calibrate-gripper", form["fields"])
    return form["fields"]["side"]


def test_the_recording_pass_is_bounded_so_the_step_ends_by_itself(stub_launcher):
    """The recording pass stops on its own.

    After its prompts the example would otherwise run `while True` at 30Hz and
    print "Press Ctrl+C to stop recording" into a terminal nobody was told to
    interrupt, so the step would run until the 1800s job timeout. `--duration`
    is what makes it stop.
    """
    app, client = _client()
    _start_gripper(client, app)

    argv = " ".join(_gripper_argv(app))
    assert "--duration" in argv, argv
    assert str(cal.GRIPPER_RECORD_S) in argv
    # And the cap is not sized for a pass that never ends.
    assert cal.GRIPPER_TIMEOUT_S <= 900.0


def test_the_plot_is_saved_rather_than_suppressed(stub_launcher):
    """The calibration plot is saved, neither shown nor thrown away.

    An interactive window would block process exit and strand the job, and
    suppressing the plot would lose it; saving it to a directory does neither.
    """
    app, client = _client()
    _start_gripper(client, app)

    argv = _gripper_argv(app)
    joined = " ".join(argv)
    assert "--no-visualize" not in joined
    assert "--plot-dir" in joined
    # Headless: a station terminal without a DISPLAY must not fail the whole
    # calibration on a windowing import.
    assert "MPLBACKEND=Agg" in joined


def test_a_finished_calibration_shows_its_plot_and_its_numbers(stub_launcher):
    """The operator's payoff: a verdict they can read and a trace they can see."""
    app, client = _client()
    side = _start_gripper(client, app)

    run = cal.calibration_state(app).run(cal.STEP_GRIPPER, side)
    # The stub never writes them, so stand in for the real child here: this is
    # about what the page does with the artefacts, not about matplotlib.
    (pathlib.Path(run.plot_dir) / cal.GRIPPER_PLOT_NAME).write_bytes(b"\x89PNG\r\n")
    pathlib.Path(run.summary_path).write_text(
        '{"moved": true, "range_mm": 84.5, "min_mm": 0.4, "max_mm": 84.9,'
        ' "num_jumps": 0, "has_jumps": false, "in_range": true}'
    )

    body = client.get("/calibration").text
    assert f"/calibration/gripper/{side}/plot.png" in body
    assert "84.5" in body and "0 jumps" in body

    served = client.get(f"/calibration/gripper/{side}/plot.png")
    assert served.status_code == 200
    assert served.headers["content-type"] == "image/png"


def test_no_plot_is_offered_before_one_has_been_written(stub_launcher):
    """An `<img>` pointing at a 404 renders broken, which reads as a failure."""
    app, client = _client()
    side = _start_gripper(client, app)

    assert (
        f"/calibration/gripper/{side}/plot.png" not in client.get("/calibration").text
    )
    assert client.get(f"/calibration/gripper/{side}/plot.png").status_code == 404


def test_a_side_that_never_ran_has_no_plot_to_serve(stub_launcher):
    app, client = _client()

    assert client.get("/calibration/gripper/left/plot.png").status_code == 404


def test_the_console_looks_for_the_filename_the_script_really_writes():
    """The one thing that couples these two modules, pinned.

    `station_ui.calibration` cannot import the example -- it pulls in click,
    numpy, matplotlib and the whole data_check package at import time, none of
    which belongs in the web process -- so the filename is duplicated. A rename
    on either side would leave the step passing with no plot and nothing said.
    """
    import os

    from examples.umi_gripper_example import plot_path

    assert cal.GRIPPER_PLOT_NAME == os.path.basename(plot_path("/anywhere"))


def test_re_running_a_side_never_shows_the_previous_attempts_plot(stub_launcher):
    """Latest run wins, and the artefacts are per run -- so a retry cannot leave
    the old picture beside the new verdict."""
    app, client = _client()
    side = _start_gripper(client, app)
    first = cal.calibration_state(app).run(cal.STEP_GRIPPER, side)
    (pathlib.Path(first.plot_dir) / cal.GRIPPER_PLOT_NAME).write_bytes(b"old")

    _start_gripper(client, app)
    second = cal.calibration_state(app).run(cal.STEP_GRIPPER, side)

    assert second.plot_dir != first.plot_dir
    assert cal.gripper_plot_file(second) is None, "the retry inherited the old plot"


def test_the_gripper_step_reports_the_child_status_not_the_terminals(
    stub_launcher, monkeypatch
):
    """Under a real `gnome-terminal --wait` the wrapper's exit code is the
    terminal's, not the calibration's, so the child's status is recorded to a
    file and read back. Simulated here by leaving the recorded status behind
    while the job itself reports success."""
    app, client = _client()
    _run_every_unit(client, app, "/calibration/calibrate-gripper")
    assert cal.step_state(app, cal.STEP_GRIPPER) == cal.DONE

    run = next(iter(cal.calibration_state(app).runs(cal.STEP_GRIPPER).values()))
    assert run.status_path, "no status file, so a terminal would hide the exit code"
    pathlib.Path(run.status_path).write_text("7")
    assert cal.step_state(app, cal.STEP_GRIPPER) == cal.FAILED
    assert "✕ Failed" in client.get("/calibration").text


def test_no_terminal_fallback_chain_is_built_by_hand(stub_launcher, monkeypatch):
    """Whatever this page launches has to survive a path or a device name
    containing a quote or a space."""
    app, client = _client()
    app.state.rig.draft.grippers.clear()
    app.state.rig.draft.grippers["right"] = 'UMI "V4" 0002; touch /tmp/pwned'
    form = _first(client.get("/calibration").text, "/calibration/calibrate-gripper")
    _run_step(client, app, "/calibration/calibrate-gripper", form["fields"])

    job = cal.job_registry(app).all()[0]
    assert job.exit_code == 0, "a quoted device name broke the command"
    assert 'UMI "V4" 0002; touch /tmp/pwned' in "\n".join(job.lines()), (
        "the device name did not arrive at the child intact"
    )
    assert not pathlib.Path("/tmp/pwned").exists()


# ---------------------------------------------- progressive enhancement / a11y


def test_every_control_is_a_real_form_that_posts_without_javascript(stub_launcher):
    _, client = _client()
    page = _parse(client.get("/calibration").text)
    steps = [f for f in page.forms if (f["action"] or "").startswith("/calibration/")]
    assert steps, "no step forms at all"
    for form in steps:
        assert form["method"] == "post"
        assert form["buttons"], f"{form['action']} has no submit control"
        for button in form["buttons"]:
            assert button.get("type") == "submit"


def test_no_control_depends_on_an_inline_event_handler(stub_launcher):
    _, client = _client()
    page = _parse(client.get("/calibration").text)
    assert page.inline_handlers == [], sorted(set(page.inline_handlers))


def test_every_status_is_a_glyph_and_a_word_never_a_bare_colour(stub_launcher):
    """Brand green and healthy green are the same hue here, so a coloured pill
    with no label is unreadable to half the room."""
    app, client = _client()
    form = _first(client.get("/calibration").text, "/calibration/check/start")
    _run_step(client, app, "/calibration/check/start", form["fields"])

    from html_sections import section_html

    badges = _parse(section_html(client.get("/calibration").text, "calibration")).badges
    assert badges, "the page renders no status at all"
    for badge in badges:
        text = _text(badge)
        assert text, f"a status badge carries only classes {badge['classes']}"
        assert any(ch.isalpha() for ch in text), f"badge {text!r} has no word"
        assert any(not ch.isascii() or ch in "*!?" for ch in text) or "•" in text, (
            f"badge {text!r} has no glyph"
        )


def test_disabled_controls_say_why_and_are_not_merely_faded(stub_launcher):
    """`opacity` as a disabled treatment composites to 2.15:1. A disabled
    control has to be disabled to the browser, not just dimmer."""
    app, _owner = _client()
    onlooker = TestClient(app)
    body = onlooker.get("/calibration").text
    page = _parse(body)
    disabled = [b for f in page.forms for b in f["buttons"] if "disabled" in b]
    assert disabled, "nothing is disabled for a read-only tab"
    assert "opacity" not in body, "an inline opacity is fading a control"
    assert "read-only" in body.lower() or "take control" in body.lower()


def test_the_page_is_named_in_the_navigation_but_gated_until_its_turn(
    stub_launcher,
):
    """The workflow is sequential: a step past the first outstanding one is
    listed and disabled, never a live link. Reaching Calibration means Setup and
    Configuration have both passed -- it is step 3, so `apply=False` is what
    leaves a step outstanding ahead of it."""
    _app, client = _client(apply=False)
    body = client.get("/").text

    assert "Calibration" in body, "the step vanished from the workflow entirely"
    assert 'href="/calibration"' not in body, (
        "Calibration is openable with Configuration still outstanding"
    )


# ------------------------------------------------ latency is not a step here


def test_the_page_offers_the_steps_this_rig_needs_and_never_mentions_latency(
    stub_launcher,
):
    """Latency is a per-hardware-revision number with a good template default.

    Measuring it per session would send the operator on a third walk to the
    station for a value that does not change. The recording templates carry it;
    the web UI does not offer to re-measure it.

    The step list is per-rig (a Vive rig gains a tracker check), so this
    asserts the catalogue rather than a fixed pair -- and that no step carries a
    stored number, since the page renders the position.
    """
    _app, client = _client()
    body = client.get("/calibration").text

    assert [step.id for step in cal.STEPS] == [
        cal.STEP_VERIFY,
        cal.STEP_VIVE,
        cal.STEP_GRIPPER,
    ]
    assert not hasattr(cal.STEPS[0], "number")
    assert not hasattr(cal, "STEP_LATENCY")

    # Inspect visible calibration text rather than searching raw HTML, where
    # rendered script paths may contain the same words.
    for step in cal.STEPS:
        for copy in (step.title, step.summary, step.watch):
            assert "latency" not in copy.lower(), f"step {step.id} offers a latency"
    # This client is a plain V4 (iPhone) rig, so only two of the three are drawn.
    assert "Step 3" not in body

    # And no control on the page would start one.
    for form in _parse(body).forms:
        assert "latency" not in (form["action"] or "").lower()
        for button in form["buttons"]:
            assert "latency" not in _text(button).lower()


def test_the_removed_latency_routes_are_gone(stub_launcher):
    """A bookmarked POST must 404, not silently start a measurement."""
    _app, client = _client()
    for path in ("/calibration/measure-latency", "/calibration/apply-latency"):
        assert client.post(path, data={"target": "anything"}).status_code == 404


def test_the_draft_no_longer_carries_measured_latency():
    """The draft carries no measured latency: nothing in station_ui reads one."""
    from station_ui.session import SessionDraft

    assert not hasattr(SessionDraft(), "camera_latency")


# ---- Gripper-terminal lifecycle ----------------------------------------------


class _StubJob:
    def __init__(self, *, finished, exit_code):
        self.finished = finished
        self.exit_code = exit_code


class _StubRegistry:
    """The one method `run_state` / `child_exit_code` call on a registry."""

    def __init__(self, job):
        self._job = job

    def get(self, job_id):
        return self._job


def test_a_recorded_status_wins_over_a_terminal_left_open(tmp_path):
    """With a terminal profile that holds the window open, the gripper job
    (`gnome-terminal --wait`) stays alive after the calibration has already
    exited. The status the wrapper recorded is authoritative regardless of the
    terminal's lifecycle, so a finished -- even crashed -- calibration does not
    read as 'running'."""
    status = tmp_path / "rc"
    run = cal.Run(key="left", job_id="j", status_path=str(status))
    still_open = _StubRegistry(_StubJob(finished=False, exit_code=None))

    status.write_text("0")
    assert cal.run_state(still_open, run) == cal.DONE

    status.write_text("5")
    assert cal.run_state(still_open, run) == cal.FAILED, (
        "a gripper that exited with an error still showed running"
    )

    # No status recorded yet and the terminal still open -> genuinely running.
    status.unlink()
    assert cal.run_state(still_open, run) == cal.RUNNING


def test_a_terminal_closed_before_a_status_was_recorded_is_not_a_pass(tmp_path):
    """Closing the gripper terminal early leaves the status file unwritten, and
    `gnome-terminal --wait` exits 0 on a clean window-close, so falling back to
    that 0 would mark a calibration nobody finished as done. A job that failed
    loudly still keeps its failure."""
    status = tmp_path / "rc"
    status.write_text("")  # the wrapper never recorded a status
    run = cal.Run(key="left", job_id="j", status_path=str(status))

    closed_clean = _StubRegistry(_StubJob(finished=True, exit_code=0))
    assert cal.child_exit_code(closed_clean, run) is None, (
        "a clean terminal close was read as a calibration pass"
    )
    assert cal.run_state(closed_clean, run) == cal.FAILED

    loud = _StubRegistry(_StubJob(finished=True, exit_code=3))
    assert cal.child_exit_code(loud, run) == 3, "a loud failure lost its code"
    assert cal.run_state(loud, run) == cal.FAILED


def test_the_launcher_runs_scripts_with_the_venv_interpreter_not_uv_run():
    """`uv run` re-resolves the whole project on every launch, which an offline
    station cannot do (it dies with 'Network is unreachable' fetching the
    `spnav` git dependency). The server already runs inside the venv, so its own
    interpreter runs the scripts directly -- no re-resolve, no network, and no
    dependency on `uv` being on the launcher's PATH."""
    import sys as _sys

    assert cal.LAUNCHER == (_sys.executable,)
    assert "uv" not in cal.LAUNCHER, "uv run re-resolves and breaks offline"


# ---- The gripper step updates without a manual reload ------------------------


class _RunningJob:
    """A gripper job whose terminal is still open (not finished)."""

    state = "running"
    finished = False
    exit_code = None
    error = None
    argv = ["fake"]
    cwd = "."

    def __init__(self, job_id="grip-run"):
        self.id = job_id

    def lines(self):
        return []


def test_a_fresh_calibration_page_does_not_poll_itself(stub_launcher):
    """No gripper running -> no self-poll; the check refreshes over SSE instead."""
    app, client = _client()
    body = client.get("/calibration").text
    assert 'data-gripper-running="false"' in body
    assert 'hx-trigger="every 1.5s"' not in body, "polled with nothing running"


def test_a_running_gripper_makes_the_page_poll_its_own_state(stub_launcher):
    """The gripper step has no SSE stream (its child is a terminal the console
    cannot read), so the page polls itself while a calibration runs; otherwise
    it would sit on 'Calibrating' until the operator reloaded."""
    app, client = _client()
    job = _RunningJob()
    cal.job_registry(app).add(job)
    side = sorted(app.state.rig.draft.grippers)[0]
    cal.calibration_state(app).record(
        cal.STEP_GRIPPER, cal.Run(key=side, job_id=job.id)
    )
    assert cal.step_state(app, cal.STEP_GRIPPER) == cal.RUNNING

    body = client.get("/calibration").text
    assert 'data-gripper-running="true"' in body
    assert 'hx-trigger="every 1.5s"' in body, "no self-poll while a gripper runs"


# ---- No way past calibration that is not running it ------------------------


def test_calibration_cannot_be_skipped(stub_launcher):
    """No skip control on the page, and no endpoint behind one."""
    _, client = _client()
    assert "/calibration/skip" not in client.get("/calibration").text
    response = client.post(
        "/calibration/skip", data={"confirm": "yes"}, follow_redirects=False
    )
    assert response.status_code in (404, 405)


def test_the_calibration_child_is_told_the_setups_gripper_generation(stub_launcher):
    """The preset's generation has to travel with the launch: the child writes
    it into the calibration file, and the recorder refuses a calibration whose
    generation disagrees with the template's `params.gripper_generation`."""
    app, client = _client()
    _start_gripper(client, app)

    generation = cal.gripper_generation_for(app.state.rig.draft)
    assert generation, "the test rig has no preset chosen"
    joined = " ".join(_gripper_argv(app))
    assert f"--gripper-generation {generation}" in joined, joined


def test_no_generation_is_passed_when_no_preset_resolves():
    class _Draft:
        preset_id = "bimanual_v9"

    assert cal.gripper_generation_for(_Draft()) == ""
    assert "--gripper-generation" not in " ".join(
        cal.gripper_argv(
            "right",
            "UMI-V4-0001",
            status_path="/tmp/s",
            plot_dir="/tmp/p",
            summary_path="/tmp/j",
            log_path="/tmp/l",
        )
    )

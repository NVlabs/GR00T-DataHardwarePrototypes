# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`/record`: start a recording, watch it, and stop it without losing data.

The load-bearing suite here is the first one: a recording has to stop from
outside its own process without losing data. Without a SIGTERM handler,
`SIGTERM` would terminate `record_data.py` without unwinding and skip the
`finally` block that drains queued alignments. "The process ended" is
therefore a worthless assertion -- `SIGKILL` satisfies it while losing a shift
of data. Every stop test below asserts the *shutdown path ran*: the drain moved
the staged episodes and `"Recording stopped"` was printed.

`test_a_sigterm_without_the_handler_loses_the_drain` is the control that gives
the rest their teeth: it is the same child with the handler left out, and it
must fail every assertion the positive test makes.

Nothing here runs `uv run record_data.py`. The `launcher` fixture keeps the
arguments the page built and swaps only the interpreter and script, so the
stand-in child really receives the argv under test without a camera or a
gripper being opened anywhere.
"""

import inspect
import json
import os
import pathlib
import signal
import subprocess
import sys
import textwrap
import time
from html.parser import HTMLParser

import pytest
import yaml
from fastapi.testclient import TestClient

REPO = pathlib.Path(__file__).resolve().parents[2]

from station_ui import data_check_config as data_check  # noqa: E402
from station_ui import data_check_config as dcc  # noqa: E402
from station_ui import jobs, recording  # noqa: E402
from station_ui.app import create_app  # noqa: E402
from station_ui.core import setup_presets as sp  # noqa: E402
from station_ui.routes import record  # noqa: E402
from station_ui.settings import Settings, state_dir_from_environment  # noqa: E402

# Use a generous state-transition deadline for cold child interpreters.
# This verifies eventual draining, not recording startup performance.
_DEADLINE = 180.0

PRESET = sp.get_preset("bimanual_v4")

# The task the armed session records. It names the generated data-check config
# as well as the recording config, so it lives on the draft rather than only in
# the recording config's `meta`.
TASK = "plane_assembly_v9"


# ---------------------------------------------------------------- child bodies


def _py(source, *args):
    return [sys.executable, "-c", textwrap.dedent(source), *args]


def _wait_until_finished(job, timeout=_DEADLINE):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if job.finished:
            return
        time.sleep(0.02)
    raise AssertionError(f"job {job.id} still {job.state} after {timeout}s")


def _wait_for_line(job, needle, timeout=_DEADLINE):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(needle in line for line in job.lines()):
            return
        time.sleep(0.02)
    raise AssertionError(f"never saw {needle!r}; got {job.lines()!r}")


# Model the real parent/forked-aligner structure. A single-process stand-in
# cannot reproduce group signals interrupting the alignment worker. Use
# the real shutdown-handler installer in the child.
_RECORD_SHAPED = """
    import multiprocessing, pathlib, sys, time

    sys.path.insert(0, {repo!r})
    from real_scripts.data_collection.record_data import install_stop_on_sigterm

    staging = pathlib.Path({staging!r})
    drained = pathlib.Path({drained!r})
    WORK_S = {work_s!r}


    def drain_worker(queue):
        # Deliberately installs NOTHING. This worker is exactly as defenceless
        # as the aligner's: under fork it inherits the parent's SIGTERM
        # handler, and under spawn it gets the default disposition. Either way a
        # group signal ends it mid-drain.
        #
        # Nothing moves until the sentinel arrives, so "what is still queued
        # when Stop is pressed" is deterministic rather than a race with the
        # test's own timing. That queue IS the thing the drain exists to save.
        pending = []
        while True:
            episode = queue.get()
            if episode is not None:
                pending.append(episode)
                continue
            for staged in pending:
                time.sleep(WORK_S)
                (staging / staged).rename(drained / staged)
                print("[aligner] aligned " + staged, flush=True)
            print("[aligner] sentinel received; worker exiting", flush=True)
            return


    if __name__ == "__main__":
        {install}
        queue = multiprocessing.Queue()
        worker = multiprocessing.Process(target=drain_worker, args=(queue,))
        worker.daemon = True
        worker.start()
        for episode in sorted(p.name for p in staging.iterdir()):
            queue.put(episode)

        try:
            print("ready", flush=True)
            while True:
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("Interrupted by user", flush=True)
        finally:
            # The shape of BackgroundAligner.shutdown(): the early-return on a
            # dead worker is what turns a killed worker into a SILENT no-op
            # that still exits 0.
            if worker is None or not worker.is_alive():
                print("[aligner] Worker already stopped", flush=True)
            else:
                queue.put(None)
                worker.join(timeout=120)
            print("Recording stopped", flush=True)
"""


def _record_shaped_source(tmp_path, *, install_handler=True, work_s=0.0, episodes=3):
    staging = tmp_path / "staging"
    drained = tmp_path / "drained"
    staging.mkdir()
    drained.mkdir()
    for index in range(episodes):
        (staging / f"episode_{index}").write_text("frames")
    source = _RECORD_SHAPED.format(
        repo=str(REPO),
        staging=str(staging),
        drained=str(drained),
        work_s=work_s,
        install=(
            "install_stop_on_sigterm()"
            if install_handler
            else "pass  # control: no handler installed"
        ),
    )
    return source, staging, drained


def _record_shaped_child(tmp_path, **kwargs):
    source, staging, drained = _record_shaped_source(tmp_path, **kwargs)
    return _py(source), staging, drained


def _staged(directory):
    return sorted(p.name for p in directory.iterdir())


# ----------------------------------------------------------------- HTML parsing


class _Page(HTMLParser):
    """Forms, their fields and attributes, plus any inline event handler.

    Structural rather than textual: a "Start" control that only works from
    JavaScript would pass a substring check for the word Start.
    """

    def __init__(self):
        super().__init__()
        self.forms: list[dict] = []
        self.inline_handlers: list[str] = []
        self.badges: list[dict] = []
        self.svg_depth = 0
        self._form: dict | None = None
        self._badge: dict | None = None

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        self.inline_handlers += [k for k in attr if k.startswith("on")]
        classes = (attr.get("class") or "").split()
        if tag == "form":
            self._form = {**attr, "fields": {}, "text": ""}
            self.forms.append(self._form)
        elif self._form is not None and tag in ("input", "select", "textarea"):
            if attr.get("name"):
                self._form["fields"][attr["name"]] = attr
        if "badge" in classes:
            self._badge = {**attr, "text": "", "icons": 0}
            self.badges.append(self._badge)
        if tag == "svg":
            self.svg_depth += 1
            if self._badge is not None:
                self._badge["icons"] += 1

    def handle_endtag(self, tag):
        if tag == "form":
            self._form = None
        elif tag == "svg":
            self.svg_depth = max(0, self.svg_depth - 1)
        elif tag == "span" and self._badge is not None:
            self._badge = None

    def handle_data(self, data):
        if self._badge is not None and self.svg_depth == 0:
            self._badge["text"] += data
        if self._form is not None:
            self._form["text"] += data


def _parse(html: str) -> _Page:
    page = _Page()
    page.feed(html)
    return page


def _form(html, action):
    forms = [f for f in _parse(html).forms if f.get("action") == action]
    assert forms, f"no <form action={action!r}> in the page"
    return forms[0]


# --------------------------------------------------------------------- fixtures


@pytest.fixture
def repo(tmp_path):
    """A repo-shaped tree holding the read-only data-check baseline.

    The baseline, not the file the recorder is handed: that one is generated
    from it, per task, and `_arm` below writes it the same way Apply on
    `/config` does. It carries its own name so a test can tell the generated
    file was built from it.
    """
    checks = tmp_path / "config" / "data_check"
    checks.mkdir(parents=True)
    name = "data_check_config.yaml"
    (checks / name).write_text(yaml.safe_dump({"checks": [], "baseline": name}))
    return tmp_path


def _generated_recording_config(preset=PRESET):
    """The file `/config` Apply writes -- in the station state dir, not the repo
    template. The record page reads this (see record.recording_config_path); a
    test that wrote the repo template would pass while production read a stale
    placeholder config."""
    return state_dir_from_environment() / "data_recording" / f"{preset.id}.yaml"


def _write_recording_config(repo, preset=PRESET, **meta):
    path = _generated_recording_config(preset)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = {
        "task_name": TASK,
        "task_description": "Assemble the plane wing bracket",
    }
    fields.update(meta)
    path.write_text(
        yaml.safe_dump(
            {
                "meta": {k: v for k, v in fields.items() if v is not None},
                "recording": {"data_dir": "~/data/session"},
            }
        )
    )
    return path


def _arm(client, repo, *, preset=PRESET, task_name=TASK):
    """Everything an Apply on `/config` leaves behind, minus the form.

    Both artifacts, because the recorder is handed both: the recording config
    and the data-check config generated beside it. `dcc.generate` is the real
    one, so what the generated file is built from is still decided by
    production code rather than restated here.
    """
    _write_recording_config(repo, preset=preset, task_name=task_name)
    dcc.generate(repo, task_name=task_name)
    draft = client.app.state.rig.draft
    draft.preset_id = preset.id
    draft.task_name = task_name
    client.get("/record")  # claims the rig and sets the cookie
    return client


class _Launcher:
    """Keeps the argv the page built; runs something harmless instead.

    Only the `<interpreter> <script>` prefix is replaced -- every argument the
    page appended is passed to the stand-in verbatim, so a test can assert on
    what the child actually received rather than on what the route meant to send.
    """

    ECHO = """
        import json, sys, time
        print("ARGV " + json.dumps(sys.argv[1:]), flush=True)
        print("ready", flush=True)
        time.sleep(45)
    """

    def __init__(self):
        self.calls: list[dict] = []
        self.body = self.ECHO

    def stand_in(self, argv):
        # The page launches `[sys.executable, record_data.py, <args...>]` -- run
        # the harmless stub with the same args (drop the interpreter + script).
        return _py(self.body, *argv[2:])

    @property
    def argv(self):
        assert self.calls, "nothing was launched"
        return self.calls[-1]["argv"]

    def child_argv(self, job):
        _wait_for_line(job, "ARGV ")
        line = next(ln for ln in job.lines() if ln.startswith("ARGV "))
        return json.loads(line.removeprefix("ARGV "))


@pytest.fixture
def launcher(monkeypatch):
    spy = _Launcher()
    real_start = jobs.start

    def fake_start(registry, argv, **kwargs):
        spy.calls.append({"argv": list(argv), "kwargs": kwargs})
        return real_start(registry, spy.stand_in(list(argv)), **kwargs)

    monkeypatch.setattr(record.jobs, "start", fake_start)
    return spy


@pytest.fixture
def client(repo):
    app = create_app(Settings(repo_root=repo))
    with TestClient(app) as test_client:
        test_client.app = app
        yield test_client
    # Keep test shutdown waits short without rearming cancellation. Replacing
    # an existing grace period could kill a real recorder during its data drain;
    # the child's original escalation timer still collects it.
    record.workflow_state(app).jobs.shutdown(timeout=1.0)


@pytest.fixture(autouse=True)
def _gate_open(monkeypatch):
    """Treat every armed rig in this file as having passed steps 1..4.

    `POST /record/start` re-checks `readiness.blocking`, so a replayed or
    bookmarked request cannot slip past the disabled Start button. This file's
    subject is the argv the recorder is handed, stopping, and draining, not the
    workflow gate itself (`test_record_readiness.py` owns that and drives it
    through the real steps). Passing the device test and calibration for real
    opens hardware no test here may touch, so the gate is declared open. The
    no-preset test still refuses at `preflight`, and no test here asserts a
    blocked/disabled Start.
    """
    monkeypatch.setattr(record.readiness, "blocking", lambda app, **kwargs: None)


@pytest.fixture
def ready(client, repo):
    """A client that owns the rig with a preset chosen and both configs on disk."""
    return _arm(client, repo)


def _start(client, **data):
    payload = {"confirmed": "yes"}
    payload.update(data)
    return client.post("/record/start", data=payload, follow_redirects=False)


def _job_of(client):
    state = record.workflow_state(client.app)
    assert state.record_job_id, "no recording job was registered"
    return state.jobs.get(state.record_job_id)


# =============================================================================
# 1. SIGTERM is a safe stop -- the shutdown path, not just the process end
# =============================================================================


def test_a_sigterm_runs_the_whole_shutdown_path_and_drains_the_staging(tmp_path):
    """SIGTERM runs the whole shutdown path.

    Without the handler, Python would terminate without unwinding: the
    `finally` in `record_data.py` would never run and every queued alignment
    would be lost. Asserting the process ended would pass under `SIGKILL` too,
    so the assertions are the *effects* of the `finally` block -- the staged
    episodes moved and the final line printed.
    """
    registry = jobs.JobRegistry()
    argv, staging, drained = _record_shaped_child(tmp_path)
    job = jobs.start(registry, argv, cwd=tmp_path, timeout=None, signal_group=False)
    try:
        _wait_for_line(job, "ready")
        assert _staged(staging) == ["episode_0", "episode_1", "episode_2"]

        jobs.cancel(registry, job.id)
        _wait_until_finished(job)

        assert "Recording stopped" in job.lines(), job.lines()
        assert "Interrupted by user" in job.lines(), job.lines()
        assert _staged(staging) == [], "the staging directory was not drained"
        assert _staged(drained) == ["episode_0", "episode_1", "episode_2"], (
            "the drain did not move the queued episodes"
        )
        assert job.exit_code == 0, "the child died instead of unwinding"
        assert job.exit_code != -int(signal.SIGKILL)
    finally:
        registry.shutdown()


def test_a_sigterm_without_the_handler_loses_the_drain(tmp_path):
    """The control. Same child, handler omitted -- and every assertion the
    test above makes has to fail here, or that test proves nothing."""
    registry = jobs.JobRegistry()
    argv, staging, drained = _record_shaped_child(tmp_path, install_handler=False)
    job = jobs.start(registry, argv, cwd=tmp_path, timeout=None, signal_group=False)
    try:
        _wait_for_line(job, "ready")
        jobs.cancel(registry, job.id)
        _wait_until_finished(job)

        assert "Recording stopped" not in job.lines()
        assert job.exit_code == -int(signal.SIGTERM) == -15
        assert _staged(staging) == ["episode_0", "episode_1", "episode_2"], (
            "an unhandled SIGTERM drained?"
        )
        assert _staged(drained) == []
    finally:
        registry.shutdown()


def test_signalling_the_process_group_kills_the_worker_that_does_the_drain(tmp_path):
    """The second control, and the reason the recording job sets
    `signal_group=False`.

    `killpg` reaches the forked alignment workers, which inherit the
    recorder's pgid *and* its new SIGTERM handler. Each dies mid-drain, and
    `BackgroundAligner.shutdown()`'s early-return on a dead worker then turns the
    drain into a no-op -- while the recorder still prints "Recording stopped"
    and exits 0. Every assertion the positive test makes about *output* holds
    here; only the assertions about the staging directory can tell them apart.
    """
    registry = jobs.JobRegistry()
    argv, staging, drained = _record_shaped_child(tmp_path)
    job = jobs.start(registry, argv, cwd=tmp_path, timeout=None, signal_group=True)
    try:
        _wait_for_line(job, "ready")
        jobs.cancel(registry, job.id)
        _wait_until_finished(job)

        # Looks like a clean shutdown from the outside...
        assert "Recording stopped" in job.lines()
        assert job.exit_code == 0
        # ...and drained nothing.
        assert _staged(drained) == [], "the group signal spared the worker?"
        assert _staged(staging) == ["episode_0", "episode_1", "episode_2"]
        # The exact dual of the positive test: the worker never lived to
        # process the sentinel, so the parent's `finally` asked a corpse to
        # drain and got a clean-looking exit for it.
        assert not any("sentinel received" in line for line in job.lines())
        assert any("KeyboardInterrupt" in line for line in job.lines()), job.lines()
    finally:
        registry.shutdown()


def test_the_installed_handler_turns_a_sigterm_into_a_keyboard_interrupt():
    """`except KeyboardInterrupt` in `record_data.py` and the `finally` under it
    are the existing, working shutdown. The handler reuses them instead of
    adding a second path that would drift out of step."""
    from real_scripts.data_collection.record_data import install_stop_on_sigterm

    previous = signal.getsignal(signal.SIGTERM)
    try:
        install_stop_on_sigterm()
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler), f"SIGTERM is still {handler!r}"
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGTERM, None)
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_main_installs_the_stop_before_it_starts_a_session(tmp_path, monkeypatch):
    """Installed by `main()`, not merely available to import. A helper nobody
    calls leaves `kill <pid>` exactly as unsafe as it was."""
    from real_scripts.data_collection import record_data

    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"recording": {"data_dir": str(tmp_path)}}))

    seen = {}

    def fake_run_session(*_args, **_kwargs):
        seen["handler"] = signal.getsignal(signal.SIGTERM)

    monkeypatch.setattr(record_data, "_run_session", fake_run_session)

    # Trap os._exit when calling main in-process so it cannot terminate pytest
    # before the test result is reported.
    class _MainExited(Exception):
        pass

    exited = {}

    def fake_exit(status):
        exited["status"] = status
        raise _MainExited()

    monkeypatch.setattr(os, "_exit", fake_exit)

    previous = signal.getsignal(signal.SIGTERM)
    try:
        with pytest.raises(_MainExited):
            record_data.main.callback(
                config=str(config),
                data_dir=None,
                verbose=False,
                check_config=None,
                ignore_data_check=False,
                ignore_pre_episode_check=False,
                allow_missing_devices=False,
            )
    finally:
        signal.signal(signal.SIGTERM, previous)

    handler = seen.get("handler")
    assert callable(handler), f"main() left SIGTERM as {handler!r}"
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGTERM, None)


# =============================================================================
# 2. Launching
# =============================================================================


def test_start_launches_record_data_with_the_argv_the_tkinter_path_built(
    ready, launcher, repo
):
    """The argv is exec'd directly rather than pasted into a `bash -c` string,
    so it must not be `shlex.quote`d: quoting would become part of the
    filename."""
    response = _start(ready)
    assert response.status_code == 303

    assert launcher.argv == [
        sys.executable,
        "real_scripts/data_collection/record_data.py",
        "--config",
        # Launch the absolute generated config path in station state, never the
        # placeholder production template.
        str(_generated_recording_config(PRESET)),
        "--check-config",
        # Use the absolute generated check-config path outside the checkout;
        # the version-controlled baseline is read-only.
        str(data_check.output_path(None, TASK)),
    ]
    assert launcher.calls[-1]["kwargs"]["cwd"] == repo


def test_the_child_receives_the_config_path_unquoted(ready, launcher, tmp_path):
    """From the child's side: `shlex.quote` on a path with a space would
    arrive as a literal `'...'` that `click.Path(exists=True)` rejects."""
    _start(ready)
    received = launcher.child_argv(_job_of(ready))
    config = received[received.index("--config") + 1]
    assert "'" not in config and '"' not in config
    assert config.endswith(".yaml")


def test_a_space_in_the_repo_path_survives_the_launch(client, tmp_path, launcher):
    spaced = tmp_path / "my station repo"
    (spaced / "config" / "data_check").mkdir(parents=True)
    (spaced / "config" / "data_check" / "data_check_config.yaml").write_text("{}\n")

    app = create_app(Settings(repo_root=spaced))
    with TestClient(app) as spaced_client:
        spaced_client.app = app
        _arm(spaced_client, spaced)
        _start(spaced_client)
        job = record.workflow_state(app).jobs.get(
            record.workflow_state(app).record_job_id
        )
        received = launcher.child_argv(job)
        record.workflow_state(app).jobs.shutdown()

    assert received[received.index("--config") + 1] == str(
        _generated_recording_config(PRESET)
    )


def test_the_argv_is_never_shell_quoted():
    """The two places a path becomes an argument, given a value that
    `shlex.quote` would actually change.

    The current template names contain nothing special, so quoting them is
    invisible until the day a path has a space in it -- at which point
    `click.Path(exists=True)` rejects a filename with literal quote marks
    around it. Asserted here rather than end to end for that reason.
    """
    argv = record.build_argv(
        "config/my rig/config.yaml",
        "config/data_check/data_check_config.yaml",
    )
    assert argv[argv.index("--config") + 1] == "config/my rig/config.yaml"
    assert not any("'" in part or '"' in part for part in argv)


def test_the_recording_job_has_no_wall_clock_cap(ready, launcher):
    """A session lasts a shift. The default 600s cap would SIGTERM it mid-run
    -- gracefully, but the operator never asked for it to end."""
    _start(ready)
    assert launcher.calls[-1]["kwargs"]["timeout"] is None
    assert _job_of(ready).timeout is None


def test_the_childs_output_is_captured_rather_than_discarded(ready, launcher):
    """Capturing stdout is the whole reason a web status page is possible: the
    job has a pid, an exit code and output."""
    _start(ready)
    _wait_for_line(_job_of(ready), "ready")
    assert any("ready" in line for line in _job_of(ready).lines())


def test_the_page_marks_the_record_step_done_and_never_data_audit(ready, launcher):
    """The Record step is marked done by name on launch, not by a panel index:
    an index such as "the last panel" would tick Data Audit instead and never
    complete Record."""
    state = record.workflow_state(ready.app)
    assert state.steps_done == set()

    _start(ready)

    assert state.steps_done == {record.RECORD_STEP}
    assert record.RECORD_STEP == "record"
    assert "data_audit" not in state.steps_done


def test_a_failed_preflight_marks_nothing_done(ready, launcher, repo):
    _write_recording_config(repo, task_description="Description")
    response = _start(ready)
    assert response.status_code == 400
    assert record.workflow_state(ready.app).steps_done == set()
    assert not launcher.calls


# =============================================================================
# 3. Pre-flight refusals
# =============================================================================


def test_start_refuses_the_template_placeholder_task_description(ready, launcher, repo):
    """`task_description: Description` ships in the templates. It is truthy, so
    `record_data.py` skips its `input()` fallback and every episode -- and
    every LeRobot `annotation.human.task_description` -- would be labelled
    "Description"."""
    _write_recording_config(repo, task_description="Description")
    response = _start(ready)
    assert response.status_code == 400
    assert not launcher.calls, "launched with the placeholder label"
    assert "Description" in response.text


def test_start_refuses_an_empty_task_description(ready, launcher, repo):
    """Empty is the other half: the child reaches
    `input("Enter task description: ")` in `record_data.py` with stdin closed
    and dies of EOFError before a single frame is recorded."""
    _write_recording_config(repo, task_description=None)
    response = _start(ready)
    assert response.status_code == 400
    assert not launcher.calls


def test_start_requires_a_task_name(ready, launcher, repo):
    """Always, not only when something downstream needs it: the task name is
    the folder every take is filed under (`output/raw/{task}/{session}/`)."""
    _write_recording_config(repo, task_name="")

    response = _start(ready)
    assert response.status_code == 400
    # Names the missing field and the page that fixes it, not the YAML key it
    # is read from -- `meta.task_name` is not a thing the operator has seen.
    assert "task name" in response.text.lower(), response.text
    assert "Configuration" in response.text, "the refusal names no way out"
    assert "meta.task_name" not in response.text
    assert not launcher.calls


def test_a_missing_config_file_is_a_refusal_not_a_500(ready, launcher, repo):
    _generated_recording_config(PRESET).unlink()
    response = _start(ready)
    assert response.status_code == 400
    assert not launcher.calls


def test_an_unparsable_config_is_a_refusal_not_a_500(ready, launcher, repo):
    _generated_recording_config(PRESET).write_text("meta: [unclosed\n")
    response = _start(ready)
    assert response.status_code == 400
    assert not launcher.calls


def test_start_with_nothing_outstanding_launches_without_a_checkbox(ready, launcher):
    """With every step done there is nothing to confirm, so a bare post starts
    the recording -- the `readiness.blocking` gate, not a checkbox, is what
    actually decides."""
    response = ready.post("/record/start", data={}, follow_redirects=False)
    assert response.status_code in (200, 303)
    assert launcher.calls, "a ready station did not start recording"
    assert record.workflow_state(ready.app).record_job_id is not None


def test_a_second_start_while_one_is_running_is_refused(ready, launcher):
    assert _start(ready).status_code == 303
    first = record.workflow_state(ready.app).record_job_id

    response = _start(ready)
    assert response.status_code == 409
    assert len(launcher.calls) == 1
    assert record.workflow_state(ready.app).record_job_id == first


def test_a_page_with_no_preset_offers_no_way_to_start(client, launcher):
    body = client.get("/record").text
    assert "/record/start" not in [f.get("action") for f in _parse(body).forms]
    assert (
        client.post(
            "/record/start", data={"confirmed": "yes"}, follow_redirects=False
        ).status_code
        == 400
    )
    assert not launcher.calls


# =============================================================================
# 4. Stopping
# =============================================================================


def test_stop_sends_sigterm_and_only_sigterm(ready, launcher, monkeypatch):
    """The signal number, explicitly, and where it was delivered.

    `SIGKILL` would skip the drain, and a test that only checked the process
    ended could not tell the two apart. The group assertion is the second half:
    a `killpg` here would reach the forked alignment workers and defeat the drain
    just as surely as a hard kill.
    """
    sent = []
    to_group = []
    real_killpg = jobs.os.killpg
    real_send = subprocess.Popen.send_signal

    def killpg_spy(pgid, sig):
        to_group.append(sig)
        return real_killpg(pgid, sig)

    def send_spy(self, sig):
        sent.append(sig)
        return real_send(self, sig)

    _start(ready)
    _wait_for_line(_job_of(ready), "ready")

    monkeypatch.setattr(jobs.os, "killpg", killpg_spy)
    monkeypatch.setattr(subprocess.Popen, "send_signal", send_spy)
    response = ready.post("/record/stop", follow_redirects=False)
    assert response.status_code == 303

    assert to_group == [], "a graceful stop must not signal the process group"

    assert sent == [signal.SIGTERM]
    assert int(sent[0]) == 15
    assert signal.SIGKILL not in sent
    assert not any(int(s) == 9 for s in sent)


def test_stop_lets_the_separate_drain_process_finish(ready, launcher, tmp_path):
    """End to end through the route.

    The drain runs in a **forked worker**, exactly as `BackgroundAligner`
    does: a single-process drain could not exhibit a bug caused by a second
    process in the group.
    """
    source, staging, drained = _record_shaped_source(tmp_path, work_s=0.05)
    launcher.body = source

    _start(ready)
    job = _job_of(ready)
    _wait_for_line(job, "ready")
    ready.post("/record/stop", follow_redirects=False)
    _wait_until_finished(job)

    assert "Recording stopped" in job.lines()
    assert _staged(staging) == [], f"staging not drained: {job.lines()}"
    assert _staged(drained) == ["episode_0", "episode_1", "episode_2"]
    assert any("sentinel received" in line for line in job.lines()), (
        "the worker was killed instead of being asked to finish"
    )
    assert not any("Worker already stopped" in line for line in job.lines())
    assert job.exit_code == 0
    assert job.state == jobs.CANCELLED


def test_the_recording_job_is_given_at_least_the_real_drain_budget(ready, launcher):
    """`jobs.DEFAULT_TERM_GRACE_S` is 10s; the drain it is supposed to allow is
    `BackgroundAligner.shutdown(timeout=600)`, run by
    `record_manager.shutdown_background_services()`.

    The budget is read out of the production signature rather than restated,
    so raising it and not the grace fails here.
    """
    from umi.common.utility.background_aligner import BackgroundAligner

    aligner_budget = (
        inspect.signature(BackgroundAligner.shutdown).parameters["timeout"].default
    )

    _start(ready)
    job = _job_of(ready)
    assert job.term_grace >= aligner_budget, (
        f"grace {job.term_grace}s is shorter than the "
        f"{aligner_budget}s drain it exists to allow"
    )
    assert job.term_grace > jobs.DEFAULT_TERM_GRACE_S


@pytest.mark.slow
def test_a_drain_slower_than_the_default_grace_still_completes(
    ready, launcher, tmp_path
):
    """Behaviourally: the escalation timer must not fire mid-drain.

    Takes longer than `jobs.DEFAULT_TERM_GRACE_S` on purpose -- that is the
    whole point. With the default grace the job would SIGKILL the recorder
    partway through and lose the rest of the drain.
    """
    per_episode = (jobs.DEFAULT_TERM_GRACE_S / 3) + 1.0
    source, staging, drained = _record_shaped_source(tmp_path, work_s=per_episode)
    launcher.body = source

    _start(ready)
    job = _job_of(ready)
    _wait_for_line(job, "ready")
    started = time.monotonic()
    ready.post("/record/stop", follow_redirects=False)
    _wait_until_finished(job, timeout=120)
    elapsed = time.monotonic() - started

    assert elapsed > jobs.DEFAULT_TERM_GRACE_S, (
        f"the drain finished in {elapsed:.1f}s, inside the default grace -- "
        "this test no longer exercises the escalation timer"
    )
    assert _staged(drained) == ["episode_0", "episode_1", "episode_2"]
    assert _staged(staging) == []
    assert job.exit_code == 0
    assert job.exit_code != -int(signal.SIGKILL)
    assert not any("SIGKILL" in line for line in job.lines()), job.lines()


def test_escalation_still_reaches_the_workers_a_narrow_sigterm_spared(ready, launcher):
    """Sparing the process group is for the *graceful* stop only.

    SIGTERM goes to the recorder alone so its drain workers survive long enough
    to drain. SIGKILL is the opposite promise -- nothing survives it -- and a
    recorder that ignored SIGTERM has, by definition, not shut its workers down.
    Narrowing escalation too would leak them: still holding the staging
    directory, still aligning, with nothing left supervising them.
    """
    launcher.body = """
        import signal, subprocess, sys, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        worker = subprocess.Popen([sys.executable, "-c",
            "import signal, time\\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\\n"
            "while True: time.sleep(0.05)"])
        print(worker.pid, flush=True)
        print("ready", flush=True)
        while True:
            time.sleep(0.05)
    """
    _start(ready)
    job = _job_of(ready)
    _wait_for_line(job, "ready")
    worker_pid = int(next(line for line in job.lines() if line.strip().isdigit()))

    jobs.cancel(record.workflow_state(ready.app).jobs, job.id, grace=1.0)
    _wait_until_finished(job, timeout=30)
    assert job.exit_code == -int(signal.SIGKILL)

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            os.kill(worker_pid, 0)
        except OSError:
            return
        time.sleep(0.05)
    os.kill(worker_pid, signal.SIGKILL)
    raise AssertionError(f"worker {worker_pid} outlived the escalation")


def test_a_hard_killed_recording_never_claims_the_drain_ran(ready, launcher):
    """`_escalate` does not clear `job._cancelled`, so a recorder that ignored
    SIGTERM and was SIGKILLed still finishes as CANCELLED. A statically-assigned
    label would show a calm green "alignments drained" for the one case where
    they provably did not."""
    launcher.body = """
        import signal, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print("ready", flush=True)
        while True:
            time.sleep(0.05)
    """
    _start(ready)
    job = _job_of(ready)
    _wait_for_line(job, "ready")

    # Escalate quickly rather than after the (correctly enormous) real grace.
    jobs.cancel(record.workflow_state(ready.app).jobs, job.id, grace=1.0)
    _wait_until_finished(job, timeout=30)

    assert job.exit_code == -int(signal.SIGKILL)
    assert job.state == jobs.CANCELLED, "the precondition for this bug is gone"

    status = record.status_of(job)
    assert "drained" not in status["label"].lower(), status
    assert status["tone"] == "error"
    assert "did not" in status["detail"].lower() or "not drain" in (
        status["detail"].lower()
    )

    badges = _parse(ready.get("/record").text).badges
    assert badges and all("drained" not in b["text"].lower() for b in badges), badges


def test_stop_with_nothing_running_is_not_an_error(ready):
    assert ready.post("/record/stop", follow_redirects=False).status_code == 303


def test_only_the_rig_owner_can_start_or_stop(ready, launcher):
    stale = TestClient(ready.app)
    stale.cookies.set("station_rig_token", "some-other-browser")

    assert (
        stale.post(
            "/record/start", data={"confirmed": "yes"}, follow_redirects=False
        ).status_code
        == 409
    )
    assert not launcher.calls

    _start(ready)
    assert stale.post("/record/stop", follow_redirects=False).status_code == 409
    assert not _job_of(ready).finished


# =============================================================================
# 5. Streaming
# =============================================================================


def test_the_stream_is_server_sent_events_over_a_plain_streaming_response(
    ready, launcher
):
    """Hand-rolled: `sse-starlette` exists only as a transitive `mcp`
    dependency and `uv export --group data_collection` -- what a station
    installs -- carries neither it nor `websockets`."""
    launcher.body = """
        import time
        print("first line", flush=True)
        time.sleep(0.4)
    """
    _start(ready)
    job_id = record.workflow_state(ready.app).record_job_id

    started = time.monotonic()
    frames = []
    with ready.stream("GET", f"/record/stream/{job_id}") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers.get("cache-control") == "no-cache"
        for line in response.iter_lines():
            frames.append(line)
            if line.startswith("data: first line"):
                assert time.monotonic() - started < 3, "buffered instead of streamed"
            if line.startswith("event: end"):
                break

    assert any(line == "data: first line" for line in frames), frames
    assert any(line.startswith("event: end") for line in frames), frames


def test_the_stream_of_an_unknown_job_is_a_404(ready):
    assert ready.get("/record/stream/nope").status_code == 404


# Imported and streamed with both SSE/WebSocket packages made unimportable.
# Blocking them in this interpreter is not enough -- the modules are already
# loaded -- so it runs in a child with a meta-path blocker installed first.
_STATION_INSTALL = """
    import sys

    class _NotOnAStation:
        FORBIDDEN = {"sse_starlette", "websockets", "wsproto"}

        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in self.FORBIDDEN:
                raise ImportError(fullname + " is not in the data_collection group")
            return None

    sys.meta_path.insert(0, _NotOnAStation())
    sys.path.insert(0, __REPO__)

    from fastapi.testclient import TestClient

    from station_ui import jobs
    from station_ui.app import create_app
    from station_ui.routes import record
    from station_ui.settings import Settings

    app = create_app(Settings(repo_root=__REPO__))
    state = record.workflow_state(app)
    job = jobs.start(state.jobs, [sys.executable, "-c", "print('hi')"], cwd=__REPO__)
    with TestClient(app) as client:
        response = client.get("/record/stream/" + job.id)
    assert response.status_code == 200, response.status_code
    assert response.headers["content-type"].startswith("text/event-stream"), (
        response.headers["content-type"]
    )
    assert "data: hi" in response.text, response.text
    print("OK")
"""


def test_the_stream_needs_neither_sse_starlette_nor_websockets(repo):
    """`sse_starlette` imports fine here -- as a transitive `mcp` dependency.
    A station installs `uv export --frozen --group data_collection`
    (`uv sync --group data_collection`), which carries neither it nor
    `websockets`, so the page has to hold up with both unimportable."""
    import subprocess

    child = subprocess.run(
        _py(_STATION_INSTALL.replace("__REPO__", repr(str(REPO)))),
        capture_output=True,
        text=True,
        timeout=_DEADLINE,
        cwd=REPO,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert "OK" in child.stdout, child.stdout + child.stderr


# =============================================================================
# 6. The page itself -- no JavaScript, and status that is never colour alone
# =============================================================================


def test_every_control_is_a_real_form_that_posts_without_javascript(ready, launcher):
    body = ready.get("/record").text
    page = _parse(body)
    assert not page.inline_handlers, page.inline_handlers

    start = _form(body, "/record/start")
    assert start.get("method", "").lower() == "post"
    # Nothing on the start form is a required field the operator has to tick
    # before the post.
    assert "confirmed" not in start["fields"], "the viewer checkbox is back"

    _start(ready)
    stop = _form(ready.get("/record").text, "/record/stop")
    assert stop.get("method", "").lower() == "post"


def test_the_pre_record_check_is_an_htmx_confirm_on_the_start_form(ready):
    """A single "are you sure" before the console starts hardware stays -- as an
    `hx-confirm` on the start form. With JavaScript off it is skipped and
    `readiness.blocking` on the server is the gate that decides."""
    start = _form(ready.get("/record").text, "/record/start")
    assert start.get("hx-post") == "/record/start"
    assert (start.get("hx-confirm") or "").strip(), "no confirmation text"


def test_htmx_gets_a_fragment_and_a_plain_browser_gets_a_redirect(ready, launcher):
    fragment = ready.post(
        "/record/start",
        data={"confirmed": "yes"},
        headers={"HX-Request": "true"},
        follow_redirects=False,
    )
    assert fragment.status_code == 200
    assert "<html" not in fragment.text.lower()
    assert "record-status" in fragment.text


def test_the_status_is_an_icon_and_a_word_never_colour_alone(ready, launcher):
    """Brand green and healthy green are the same hue on this console, so a
    bare coloured pill is unreadable to an operator who cannot separate them."""
    from html_sections import section_html

    idle = _parse(section_html(ready.get("/record").text, "record"))
    assert idle.badges, "no status badge on the page"
    for badge in idle.badges:
        assert badge["text"].strip(), f"badge with no label: {badge}"
        assert badge["icons"] >= 1, f"badge with no icon: {badge}"

    _start(ready)
    running = _parse(section_html(ready.get("/record").text, "record"))
    labels = [b["text"].strip() for b in running.badges]
    assert labels != [b["text"].strip() for b in idle.badges], (
        "the badge says the same thing whether or not a recording is running"
    )
    for badge in running.badges:
        assert badge["text"].strip() and badge["icons"] >= 1


def test_a_plain_reload_shows_the_captured_output(ready, launcher):
    """The no-JavaScript status path: no SSE, no polling, just what the child
    has printed so far, rendered by the server."""
    _start(ready)
    _wait_for_line(_job_of(ready), "ready")
    assert "ready" in ready.get("/record").text


def test_a_read_only_tab_gets_the_take_control_affordance(ready, launcher):
    stale = TestClient(ready.app)
    stale.cookies.set("station_rig_token", "another-browser")
    body = stale.get("/record").text
    assert "/take-control" in [f.get("action") for f in _parse(body).forms]
    assert "/record/start" not in [f.get("action") for f in _parse(body).forms]


def test_the_record_nav_entry_is_enabled(client):
    from station_ui.app import NAV_ITEMS

    entry = next(item for item in NAV_ITEMS if item["href"] == "/record")
    assert entry["enabled"] is True
    assert client.get("/record").status_code == 200


def test_build_argv_launches_with_the_venv_interpreter_not_uv_run():
    """`uv run` re-resolves the whole project on launch and, on an offline
    station, dies fetching the `spnav` git dependency ('Network is
    unreachable') before the recorder starts. The server already runs in the
    venv, so its own interpreter runs `record_data.py` directly -- no
    re-resolve, no network."""
    argv = record.build_argv("cfg.yaml", "check.yaml")
    assert argv[0] == sys.executable
    assert argv[1] == record.RECORD_SCRIPT
    assert "uv" not in argv, "uv run re-resolves and breaks offline"


def test_the_record_page_shows_the_generated_task_not_the_template_placeholder(ready):
    """/record reads the file Apply generated, not the repo template, which
    carries placeholders ('Enter Task Name Here' / 'Description') and
    placeholder hardware the recorder cannot open -- so the real task shows,
    and the recorder is handed this rig's devices."""
    body = ready.get("/record").text
    assert "Enter Task Name Here" not in body, "record page read the repo template"
    assert TASK in body, "the operator's task from the generated config is missing"


def test_a_refused_start_shows_the_reason_on_the_page(ready, launcher, repo):
    """A refused Start has to say why. The Start button is `hx-post` targeting
    #record-status, so the refusal only swaps that fragment -- a banner in the
    outer page is never sent. The reason must be in the swapped fragment.
    Posted with the HX-Request header, which is the path the browser actually
    takes."""
    _write_recording_config(repo, task_description="Description")  # the placeholder
    response = ready.post(
        "/record/start",
        data={"confirmed": "yes"},
        headers={"HX-Request": "true"},
        follow_redirects=False,
    )
    assert response.status_code == 400
    assert not launcher.calls
    assert 'id="record-status"' in response.text, "not the swapped fragment"
    assert "Recording did not start." in response.text, "the refusal is invisible"
    assert "placeholder" in response.text.lower(), "the specific reason is missing"


# ---- a recorder that outlived the console ------------------------------------


def test_start_is_refused_while_any_process_still_holds_the_rig(ready, monkeypatch):
    """Not merely "the leader is alive". An orphaned worker with the leader
    gone still holds an exclusive OAK, and a second recorder launched into that
    fails to open a camera with a message that blames the cable.

    Driven through the real route: the disabled button is only an affordance,
    and a bookmark or a replay still POSTs here.
    """
    monkeypatch.setattr(recording, "in_progress", lambda app: True)

    response = _start(ready)

    assert response.status_code == 409
    assert "already running" in response.text.lower()


def test_starting_a_recording_writes_the_pid_file(ready, launcher):
    """The console has to be able to find this process after a restart -- in
    memory it would die with the web process while the detached child records
    on."""
    _start(ready)

    state_dir = pathlib.Path(ready.app.state.settings.state_dir)
    written = state_dir / recording.PID_FILE
    assert written.exists(), "no pid file after Start"
    assert json.loads(written.read_text())["pgid"] == _job_of(ready).pgid


def test_a_drifted_start_seals_the_session_it_rolls_off(ready, launcher, tmp_path):
    """Start rolls off a drifted folder; the folder it leaves must be sealed.

    Apply does this by comparing the folder it was armed for against the one it
    wrote, and Start has to do the same, or the old session would stay
    unsealed.
    """
    from umi.common.recording import session_seal
    from umi.common.recording.session_slot import save_launch_config

    # Arm at a folder under tmp_path, make it look recorded-into, then drift.
    generated = _generated_recording_config(PRESET)
    cfg = yaml.safe_load(generated.read_text())
    folder = tmp_path / "2026-08-28_umiV3_0001R_vive_task_a1b2c3d4e5f6"
    folder.mkdir()
    cfg["recording"]["data_dir"] = str(folder)
    generated.write_text(yaml.safe_dump(cfg))
    save_launch_config(str(folder), cfg)
    (folder / "episode_0").mkdir()

    cfg["recording"]["record_fps"] = int(cfg["recording"].get("record_fps", 30)) + 15
    generated.write_text(yaml.safe_dump(cfg))

    ready.post("/record/start", data={"confirmed": "yes"}, follow_redirects=False)

    assert session_seal.folder_is_sealed(folder) is True
    assert session_seal.read_seal(folder)["reason"] == "config_drift"

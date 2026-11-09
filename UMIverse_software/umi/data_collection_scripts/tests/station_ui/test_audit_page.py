# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The /audit page: each session's alignment and warnings, the alignment check,
and what it said -- including when what it said was a traceback.

The load-bearing tests here are the ones that force a failure:

* `test_a_raising_script_surfaces_as_a_failed_job_carrying_its_traceback`
  -- the plumbing: a child that raises must reach the operator as failed with
  the traceback, not as a job that quietly succeeded.
* `test_the_real_health_check_script_fails_the_job_when_a_session_raises`
  -- the same thing end to end through the real script.

Nothing here touches the network: the script is either a stub, or the real
script pointed at a session directory it rejects before it checks anything.
Sessions live where the station records them, `output/raw/{task}/{session}`
under the per-test `UMI_OUTPUT_ROOT` the root conftest sets.
"""

import contextlib
import http.client
import json
import pathlib
import socket
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser

import pytest

# Reuse the rendered-form helper from this test directory so Apply
# requests match browser payloads.
import test_config_page as config_page
import uvicorn
import yaml
from fastapi.testclient import TestClient

from station_ui import audit, jobs, readiness
from station_ui import session as session_state
from station_ui.app import create_app
from station_ui.routes import audit as audit_routes
from umi.common.recording import session_seal
from umi.common.tasks import task_registry_store
from umi.common.utility.output_layout import counterpart

REPO = pathlib.Path(__file__).resolve().parents[2]
_DEADLINE = 60.0
TODAY = "2026-08-04"


# --------------------------------------------------------------------- parsing


class _Page(HTMLParser):
    """Elements by class, their text, and every form on the page.

    Structural on purpose: asserting on the raw HTML string would let a session
    row that renders one session's episode count against another session's name
    pass a test named "reports the episode count".
    """

    _VOID = {"input", "br", "img", "meta", "link", "hr", "source"}

    def __init__(self):
        super().__init__()
        self.forms: list[dict] = []
        self.inline_handlers: list[str] = []
        self.scripts: list[str] = []
        self.by_class: dict[str, list[str]] = {}
        self.links: list[str] = []
        self._form: dict | None = None
        self._open: list[tuple[str, list[str]]] = []
        self._collecting: list[list] = []

    # -- element tracking
    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        self.inline_handlers += [k for k in attr if k.startswith("on")]
        if tag == "script" and attr.get("src"):
            self.scripts.append(attr["src"])
        if tag == "a" and attr.get("href"):
            self.links.append(attr["href"])
        if tag == "form":
            self._form = {
                "action": attr.get("action"),
                "method": (attr.get("method") or "get").lower(),
                "fields": {},
                "buttons": [],
            }
            self.forms.append(self._form)
        elif self._form is not None:
            if tag in ("input", "select", "textarea") and attr.get("name"):
                self._form["fields"][attr["name"]] = attr.get("value", "")
            elif tag == "button":
                self._form["buttons"].append(attr)
        if tag not in self._VOID:
            classes = (attr.get("class") or "").split()
            self._open.append((tag, classes))
            for cls in classes:
                self.by_class.setdefault(cls, []).append("")
                self._collecting.append([cls, len(self.by_class[cls]) - 1, []])

    def handle_endtag(self, tag):
        if tag == "form":
            self._form = None
        if tag in self._VOID:
            return
        while self._open and self._open[-1][0] != tag:
            self._close_one()
        if self._open:
            self._close_one()

    def _close_one(self):
        _, classes = self._open.pop()
        for _ in classes:
            cls, index, parts = self._collecting.pop()
            self.by_class[cls][index] = " ".join("".join(parts).split())

    def handle_data(self, data):
        for entry in self._collecting:
            entry[2].append(data)

    def close(self):
        super().close()
        while self._open:
            self._close_one()


def parse(html: str) -> _Page:
    page = _Page()
    page.feed(html)
    page.close()
    return page


@contextlib.contextmanager
def real_server(app):
    """The app on a real loopback socket, for the one thing TestClient cannot do.

    Bound to an ephemeral port on 127.0.0.1 and shut down on the way out, so a
    test never leaves a listener behind.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [sock]}, daemon=True
    )
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started, "uvicorn never came up"
    try:
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        with contextlib.suppress(OSError):
            sock.close()


def rows_by_session(html: str) -> dict[str, str]:
    """Session name -> the visible text of that session's row."""
    page = parse(html)
    out = {}
    for text in page.by_class.get("session-row", []):
        name = text.split(" ", 1)[0]
        out[name] = text
    return out


# -------------------------------------------------------------------- fixtures


@dataclass
class Station:
    root: pathlib.Path
    tmp: pathlib.Path
    client: TestClient
    app: object
    monkeypatch: object
    scripts: dict = field(default_factory=dict)

    def get(self, url="/audit", **kw):
        return self.client.get(url, **kw)

    def session(
        self,
        name,
        *,
        task="taskA",
        episodes=0,
        pending=0,
        quarantine=False,
        aligned=(),
        warnings=None,
    ):
        """A session at `output/raw/{task}/{name}`, as the station records it.

        `quarantine` adds its quarantine folder; `aligned` names the episode
        indices that have an aligned copy under `output/aligned/...`; and
        `warnings` maps an episode index to the warnings the data check saved
        in that episode's `zarr.json`.
        """
        path = self.root / task / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "logs").mkdir(exist_ok=True)
        (path / "config.yaml").write_text(f"meta:\n  task_name: {task}\n")
        for idx in range(episodes):
            (path / f"episode_{idx}").mkdir(exist_ok=True)
        for idx in range(episodes, episodes + pending):
            (path / f"episode_{idx}.pending").mkdir(exist_ok=True)
        if quarantine:
            qdir = pathlib.Path(counterpart(str(path), "quarantine"))
            (qdir / "episode_0").mkdir(parents=True, exist_ok=True)
        aligned_dir = pathlib.Path(counterpart(str(path), "aligned"))
        for idx in aligned:
            _mark_aligned_episode(aligned_dir, idx)
        for idx, messages in (warnings or {}).items():
            findings = {"v": 2, "warnings": list(messages)}
            (path / f"episode_{idx}" / "zarr.json").write_text(
                json.dumps(
                    {
                        "zarr_format": 3,
                        "node_type": "group",
                        "attributes": {"data_check_findings": findings},
                    }
                )
            )
        return path

    def stub_script(self, source):
        """Point the health check at a temp Python file."""
        path = self.tmp / "health.py"
        path.write_text(textwrap.dedent(source))
        self.monkeypatch.setattr(audit, "HEALTH_CHECK", str(path))
        return path

    def start(self, **form):
        return self.client.post(
            "/audit/health-check", data=form, follow_redirects=False
        )

    def registry(self):
        return getattr(self.app.state, "jobs", None)

    def job_of(self, response):
        location = response.headers["location"]
        job_id = location.rsplit("/", 1)[-1]
        return self.registry().get(job_id)

    def wait(self, job, timeout=_DEADLINE):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if job.finished:
                return job
            time.sleep(0.02)
        raise AssertionError(f"job {job.id} still {job.state}: {job.lines()!r}")


def _station(tmp_path, monkeypatch):
    root = audit.recordings_root()
    root.mkdir(parents=True, exist_ok=True)

    app = create_app()
    client = TestClient(app)
    client.get("/audit")  # claim the rig, exactly as a browser landing here does
    return Station(
        root=root, tmp=tmp_path, client=client, app=app, monkeypatch=monkeypatch
    )


@pytest.fixture
def station(tmp_path, monkeypatch):
    """An app whose recordings root (`output/raw` under the per-test
    `UMI_OUTPUT_ROOT`) is temporary."""
    built = _station(tmp_path, monkeypatch)
    yield built
    _shutdown(built)


def _shutdown(built):
    registry = getattr(built.app.state, "jobs", None)
    if registry is not None:
        registry.shutdown()


# ------------------------------------------------------------------ the listing


def test_the_session_list_reports_the_episode_count_from_disk(station):
    station.session("2026-08-04_umi1R_taskA_aaaaaaaaaaaa", episodes=2, pending=1)
    station.session("2026-08-04_umi1R_taskB_bbbbbbbbbbbb", episodes=0)

    rows = rows_by_session(station.get().text)

    assert "3 episodes" in rows["2026-08-04_umi1R_taskA_aaaaaaaaaaaa"], rows
    assert "0 episodes" in rows["2026-08-04_umi1R_taskB_bbbbbbbbbbbb"], rows


def test_only_episode_directories_are_counted(station):
    """`logs/` and `config.yaml` exist before the first recording; counting them
    would report a session as recorded before anything was."""
    empty = station.session("2026-08-04_umi1R_taskA_aaaaaaaaaaaa")
    (empty / "episode_notanumber").mkdir()
    (empty / "episode_9").write_text("a file, not an episode directory")

    rows = rows_by_session(station.get().text)

    assert "0 episodes" in rows["2026-08-04_umi1R_taskA_aaaaaaaaaaaa"]


def test_a_fully_aligned_session_says_so(station):
    station.session(
        "2026-08-04_umi1R_taskA_aaaaaaaaaaaa", episodes=2, aligned=(0, 1), pending=1
    )

    row = rows_by_session(station.get().text)["2026-08-04_umi1R_taskA_aaaaaaaaaaaa"]

    # The pending episode is still being written or checked, so it has no
    # aligned copy to look for yet and does not count against the session.
    assert "All aligned" in row, row


def test_missing_aligned_copies_are_counted_and_offered_a_re_align(station):
    name = "2026-08-04_umi1R_taskA_aaaaaaaaaaaa"
    folder = station.session(name, episodes=3, aligned=(0,))

    html = station.get().text

    assert "1 of 3 aligned" in rows_by_session(html)[name]
    realign = [
        form
        for form in parse(html).forms
        if form["action"] == "/audit/health-check"
        and form["fields"].get("session_folder") == str(folder)
        and "dry_run" not in form["fields"]
    ]
    assert len(realign) == 1, "no Re-align offered for a session with gaps"


def test_a_fully_aligned_session_is_offered_no_re_align(station):
    folder = station.session(
        "2026-08-04_umi1R_taskA_aaaaaaaaaaaa", episodes=1, aligned=(0,)
    )

    forms = parse(station.get().text).forms

    assert not [
        form
        for form in forms
        if form["action"] == "/audit/health-check"
        and form["fields"].get("session_folder") == str(folder)
        and "dry_run" not in form["fields"]
    ]


def test_the_aligned_dir_the_session_config_names_is_the_one_counted(station):
    """The health check honours `alignment.aligned_dir`; the listing must agree
    with it, or a session aligned somewhere else reads as all gaps."""
    name = "2026-08-04_umi1R_taskA_aaaaaaaaaaaa"
    folder = station.session(name, episodes=1)
    elsewhere = station.tmp / "aligned_elsewhere"
    _mark_aligned_episode(elsewhere, 0)
    (folder / "config.yaml").write_text(
        f"meta:\n  task_name: taskA\nalignment:\n  aligned_dir: {elsewhere}\n"
    )

    assert "All aligned" in rows_by_session(station.get().text)[name]


def test_warnings_saved_on_episodes_are_listed_on_their_session(station):
    """The data check's warnings never fail an episode, so this listing is
    where a pattern -- the same camera offset on every episode -- shows."""
    offset = (
        "Motion: left_umi_camera vs left_tracking_camera: camera-pose "
        "misalignment 80ms (expected 0ms on aligned timeline, residual 80ms > 60ms)"
    )
    name = "2026-08-04_umi1R_taskA_aaaaaaaaaaaa"
    station.session(name, episodes=2, warnings={0: [offset], 1: [offset]})
    station.session("2026-08-04_umi1R_taskB_bbbbbbbbbbbb", episodes=1)

    rows = rows_by_session(station.get().text)

    assert "2 warnings" in rows[name], rows[name]
    assert "episode_0" in rows[name] and "episode_1" in rows[name]
    assert "camera-pose misalignment 80ms" in rows[name]
    assert "warning" not in rows["2026-08-04_umi1R_taskB_bbbbbbbbbbbb"]


def test_an_unreadable_episode_attribute_file_shows_no_warnings(station):
    name = "2026-08-04_umi1R_taskA_aaaaaaaaaaaa"
    folder = station.session(name, episodes=1)
    (folder / "episode_0" / "zarr.json").write_text("{not json")

    response = station.get()

    assert response.status_code == 200
    assert "warning" not in rows_by_session(response.text)[name]


def test_derived_folders_are_not_listed_as_sessions(station):
    name = "2026-08-04_umi1R_taskA_aaaaaaaaaaaa"
    folder = station.session(name, episodes=1, quarantine=True)
    assert pathlib.Path(counterpart(str(folder), "quarantine")).is_dir()
    # Derived siblings of sessions without UIDs.
    (folder.parent / f"{name}_quarantine").mkdir()
    (folder.parent / f"{name}_aligned").mkdir()
    (folder.parent / "not-a-session").mkdir()
    (folder.parent / "stray.txt").write_text("x")

    rows = rows_by_session(station.get().text)

    assert set(rows) == {name}, rows


def test_a_missing_recordings_root_renders_instead_of_failing(station):
    """A station whose output folder has not been created yet must still render.

    Without the path: an operator does not navigate to an absolute recordings
    directory from a browser. They get "No sessions yet".
    """
    missing = station.tmp / "nowhere" / "output"
    station.monkeypatch.setenv("UMI_OUTPUT_ROOT", str(missing))

    response = station.get()
    assert response.status_code == 200
    assert "Traceback" not in response.text
    assert "No sessions yet" in response.text
    assert str(missing / "raw") not in response.text


def test_every_status_carries_words_not_only_a_colour(station):
    """Brand green and 'healthy' green are the same hue, so a bare coloured
    pill is not a status."""
    station.session("2026-08-04_umi1R_taskA_aaaaaaaaaaaa", episodes=1)

    page = parse(station.get().text)

    badges = page.by_class.get("badge", [])
    assert badges, "the page renders no status badges at all"
    for text in badges:
        assert text.strip(), "a badge renders as colour with no text"


# --------------------------------------------------------------------- jobs


def test_terminal_colour_codes_do_not_reach_the_browser(station):
    """The script prints through `umi.common.utility.print_utils.Colors`,
    which emits raw ANSI unconditionally -- there is no NO_COLOR or isatty
    guard. Passed through, the operator reads `[91mERROR processing X:[0m`."""
    station.stub_script(
        r"""
        print("\x1b[91mERROR processing 2026-08-04_umi1R_taskA\x1b[0m")
        """,
    )

    response = station.start(date=TODAY)
    job = station.wait(station.job_of(response))

    page = station.client.get(response.headers["location"]).text
    frames = station.client.get(f"/audit/jobs/{job.id}/stream").text
    for rendered in (page, frames):
        assert "ERROR processing 2026-08-04_umi1R_taskA" in rendered
        assert "\x1b" not in rendered
        assert "[91m" not in rendered


def test_starting_the_health_check_redirects_to_a_job_page_showing_its_output(station):
    station.stub_script("print('AUDIT-MARKER')")

    response = station.start(date=TODAY)
    assert response.status_code == 303

    job = station.wait(station.job_of(response))
    page = station.client.get(response.headers["location"])

    assert page.status_code == 200
    assert "AUDIT-MARKER" in page.text
    assert job.state == jobs.DONE


def test_the_check_options_reach_the_script_as_command_line_flags(station):
    station.stub_script("import sys\nprint(' '.join(sys.argv[1:]))")

    response = station.start(
        date="2026-07-01", task_name="plane_assembly_v4", dry_run="on"
    )
    job = station.wait(station.job_of(response))

    printed = "\n".join(job.lines())
    assert "--date 2026-07-01" in printed
    assert "--task-name plane_assembly_v4" in printed
    assert "--dry-run" in printed
    assert job.label.startswith("Alignment check")


def test_re_align_missing_runs_the_script_without_dry_run(station):
    station.stub_script("import sys\nprint(' '.join(sys.argv[1:]))")

    response = station.start(date="2026-07-01")
    job = station.wait(station.job_of(response))

    assert "--dry-run" not in "\n".join(job.lines())
    assert job.label.startswith("Re-align missing")


def test_a_successful_re_align_seals_nothing(station):
    """Sealing a session is not a side effect of the audit: a re-align that
    succeeds leaves every session open to more episodes."""
    folder = station.session(TODAY + "_umi1R_taskA_aaaaaaaaaaaa", episodes=1)
    station.stub_script("print('ok')")

    job = station.wait(station.job_of(station.start(date=TODAY)))

    assert job.state == jobs.DONE
    assert not (folder / "session_sealed.json").exists()


def test_a_raising_script_surfaces_as_a_failed_job_carrying_its_traceback(station):
    """A script that blows up must not read as a run that worked."""
    station.stub_script(
        """
        print("starting the audit")
        raise RuntimeError("zarr.json unreadable")
        """,
    )

    response = station.start(date=TODAY)
    job = station.wait(station.job_of(response))

    assert job.state == jobs.FAILED
    assert job.exit_code not in (0, None)

    page = station.client.get(response.headers["location"]).text
    assert "Failed" in page
    assert "Traceback" in page
    assert "RuntimeError: zarr.json unreadable" in page

    frames = station.client.get(f"/audit/jobs/{job.id}/stream").text
    assert "RuntimeError: zarr.json unreadable" in frames
    end = json.loads(frames.rsplit("data: ", 1)[-1].strip())
    assert end["state"] == jobs.FAILED


def test_the_real_health_check_script_fails_the_job_when_a_session_raises(station):
    """End to end against the real script, no stubs.

    A traceback must not come with exit 0, or a UI wrapping the script would
    see a successful run. The session here has no config.yaml, which the
    script rejects before it checks or aligns anything.
    """
    session = station.session("2026-08-04_umi1R_taskA_aaaaaaaaaaaa", episodes=1)
    (session / "config.yaml").unlink()

    response = station.client.post(
        "/audit/health-check",
        data={"session_folder": str(session), "dry_run": "on"},
        follow_redirects=False,
    )
    assert response.status_code == 303

    job = station.wait(station.job_of(response))
    printed = "\n".join(job.lines())

    assert job.state == jobs.FAILED, printed
    assert "Traceback" in printed
    assert "config.yaml not found" in printed
    assert str(session) in printed

    page = station.client.get(response.headers["location"]).text
    assert "config.yaml not found" in page


def test_a_job_from_another_page_is_invisible_and_untouchable_here(station):
    """The job registry is shared by every page, and `/record` puts
    `record_data.py` in it.

    Unfiltered, `/audit` would list a live recording under "Recent audits" and
    `POST /audit/jobs/<recording>/cancel` would SIGTERM it -- from a button that
    reads "Stop this audit". `record_data.py` drains queued alignments in a
    `finally` block, so an operator who thinks they are cancelling a data check
    would be ending a session instead.
    """
    from station_ui.routes.audit import registry_of

    recording = jobs.start(
        registry_of(station.app),
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=".",
        label="Recording — plane_assembly_v10",
        kind="record",
    )
    try:
        listing = station.get().text
        assert recording.id not in listing
        assert "Recording — plane_assembly_v10" not in listing

        assert station.client.get(f"/audit/jobs/{recording.id}").status_code == 404
        assert (
            station.client.get(f"/audit/jobs/{recording.id}/stream").status_code == 404
        )
        stop = station.client.post(
            f"/audit/jobs/{recording.id}/cancel", follow_redirects=False
        )
        assert stop.status_code == 404

        # A delivered SIGTERM ends this child in milliseconds.
        time.sleep(0.5)
        assert recording.state == jobs.RUNNING, "a foreign job was signalled"
    finally:
        jobs.cancel(registry_of(station.app), recording.id)


def test_the_stream_is_server_sent_events_that_replay_the_whole_log(station):
    station.stub_script("print('one')\nprint('two')")

    response = station.start(date=TODAY)
    job = station.wait(station.job_of(response))

    stream = station.client.get(f"/audit/jobs/{job.id}/stream")

    assert stream.headers["content-type"].startswith("text/event-stream")
    payloads = [
        line.removeprefix("data: ")
        for line in stream.text.splitlines()
        if line.startswith("data: ")
    ]
    assert "one" in payloads and "two" in payloads
    assert "event: end" in stream.text


def test_the_stream_delivers_output_while_the_job_is_still_running(station):
    """`session_health_check` runs for minutes. Output that arrives only at exit
    is indistinguishable from a hang.

    Over a real socket, because Starlette's TestClient drains a
    StreamingResponse before it returns the first line -- measured: a generator
    that sleeps 3s between frames hands over frame one at t=3.04s -- so it
    cannot tell a live stream from a buffered one.
    """
    station.stub_script(
        """
        import time
        print("EARLY-LINE", flush=True)
        time.sleep(30)
        """,
    )

    response = station.start(date=TODAY)
    job = station.job_of(response)
    seen = ""
    still_running = False
    with real_server(station.app) as port:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
        try:
            conn.request("GET", f"/audit/jobs/{job.id}/stream")
            stream = conn.getresponse()
            assert stream.status == 200
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                line = stream.readline().decode("utf-8", "replace")
                if "EARLY-LINE" in line:
                    seen = line
                    still_running = not job.finished
                    break
        finally:
            conn.close()
            # Stop the child inside the server's lifetime: the stream ends with
            # the job, and uvicorn will not shut down under an open response.
            jobs.cancel(station.registry(), job.id)
            station.wait(job)

    assert seen.startswith("data: "), f"not an SSE frame: {seen!r}"
    assert still_running, "the child had already exited; nothing was proven"


def test_progress_bars_reach_the_browser_instead_of_only_a_terminal(station):
    """`session_health_check.py` drives tqdm, which writes to stderr with
    carriage returns; both have to reach the page."""
    station.stub_script(
        """
        from tqdm import tqdm
        for _ in tqdm(range(4), desc="Aligning"):
            pass
        """,
    )

    response = station.start(date=TODAY)
    job = station.wait(station.job_of(response))

    frames = station.client.get(f"/audit/jobs/{job.id}/stream").text
    assert "Aligning" in frames
    assert "100%" in frames


def test_a_running_audit_can_be_stopped_from_the_page(station):
    station.stub_script("import time\ntime.sleep(30)")

    response = station.start(date=TODAY)
    job = station.job_of(response)

    stop = station.client.post(f"/audit/jobs/{job.id}/cancel", follow_redirects=False)
    assert stop.status_code == 303
    station.wait(job)

    assert job.state == jobs.CANCELLED
    assert "Cancelled" in station.client.get(f"/audit/jobs/{job.id}").text


def test_an_unknown_job_id_is_a_404_not_a_crash(station):
    assert station.client.get("/audit/jobs/deadbeef").status_code == 404
    assert station.client.get("/audit/jobs/deadbeef/stream").status_code == 404


def test_an_audit_cannot_outlive_the_page_without_a_cap(station):
    """Nothing here may hang forever: every audit job carries a wall-clock cap."""
    station.stub_script("print('x')")

    response = station.start(date=TODAY)
    job = station.wait(station.job_of(response))

    assert job.timeout is not None and job.timeout > 0


# ------------------------------------------------- ownership and no-JS paths


def test_a_second_browser_is_refused_and_told_how_to_take_over(station):
    other = TestClient(station.app)

    response = other.post(
        "/audit/health-check", data={"date": TODAY}, follow_redirects=False
    )

    assert response.status_code == 409
    page = parse(response.text)
    assert any(form["action"] == "/take-control" for form in page.forms), (
        "the refusal offers no way out"
    )
    assert station.registry() is None or not station.registry().all()


def test_every_action_on_the_page_is_a_real_form(station):
    """If htmx fails to load on a restricted station network, the page must
    still post."""
    station.session("2026-08-04_umi1R_taskA_aaaaaaaaaaaa", episodes=1)

    page = parse(station.get().text)

    posts = [form for form in page.forms if form["method"] == "post"]
    assert len(posts) >= 2, (
        f"expected the per-session Check and Re-align forms: {posts}"
    )
    for form in page.forms:
        assert form["action"], f"a form has no action: {form}"
    assert not page.inline_handlers, f"inline handlers: {page.inline_handlers}"


def test_the_job_page_renders_its_log_server_side(station):
    """With JavaScript disabled there is no EventSource, so the page itself has
    to carry the output and a plain link to fetch more of it."""
    station.stub_script("print('SERVER-RENDERED')")

    response = station.start(date=TODAY)
    job = station.wait(station.job_of(response))
    page = station.client.get(f"/audit/jobs/{job.id}")

    assert "SERVER-RENDERED" in page.text
    parsed = parse(page.text)
    assert not parsed.inline_handlers
    assert any(href == f"/audit/jobs/{job.id}" for href in parsed.links), (
        "no plain refresh link for a browser without JavaScript"
    )


def test_looking_is_the_default_and_writing_is_a_separate_button(station):
    """The per-session Check posts `dry_run`; Re-align writes aligned copies."""
    folder = station.session("2026-08-04_umi1R_taskA_aaaaaaaaaaaa", episodes=1)

    page = parse(station.get().text)

    assert not [
        form
        for form in page.forms
        if form["action"] == "/audit/health-check"
        and not form["fields"].get("session_folder")
    ]
    checks = [
        form
        for form in page.forms
        if form["fields"].get("session_folder") == str(folder)
        and form["fields"].get("dry_run") == "on"
    ]
    assert len(checks) == 1, "no report-only Check on the session row"


def test_the_audit_page_is_reachable_from_the_nav(station):
    page = parse(station.client.get("/").text)
    assert "#audit" in page.links


# ----------------------------------------------------------- argv contract


def test_the_launcher_runs_the_script_with_the_interpreter_serving_the_app():
    """`uv run` would re-resolve the environment per audit and needs `uv` on the
    PATH of whatever started the server; the server's own interpreter already
    has every dependency the script imports."""
    argv = audit.health_check_argv(date=TODAY)
    assert argv[0] == sys.executable
    assert argv[1].endswith("session_health_check.py")


def test_a_session_folder_run_does_not_also_pass_a_date():
    """tyro ignores --date when --session-folder is given; sending both would
    tell the operator a date filter applied when it did not."""
    argv = audit.health_check_argv(date=TODAY, session_folder="/data/session")
    assert "--session-folder" in argv
    assert "--date" not in argv


def test_the_script_accepts_the_flags_this_page_sends():
    """A renamed tyro field would otherwise fail only on a real station."""
    argv = audit.health_check_argv(
        date=TODAY, end_date=TODAY, task_name="taskA", dry_run=True
    )
    result = subprocess.run(
        [argv[0], argv[1], "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    for flag in [part for part in argv if part.startswith("--")]:
        assert flag in result.stdout, f"{flag} is not a flag of {argv[1]}"


# ------------------------------------------------------------- the listing


def _make_session(
    root, name="2026-08-26_umiV4_0001R_vive_task_a1b2c3d4e5f6", episodes=1
):
    folder = root / name
    folder.mkdir(parents=True)
    for index in range(episodes):
        (folder / f"episode_{index}").mkdir()
    return folder


def test_the_active_session_is_marked_active(tmp_path):
    from station_ui import audit

    folder = _make_session(tmp_path)
    (session,) = audit.list_sessions([tmp_path], active=str(folder))
    assert session.active is True


def test_the_active_session_lists_first_regardless_of_name_order(tmp_path):
    """The active row is the one the operator came to check. Name order
    alone can bury it: a re-recorded sibling whose name sorts later would
    otherwise sit on top."""
    from station_ui import audit

    active = _make_session(tmp_path, name="2026-08-26_umiV4_0001R_vive_task_aaaa")
    _make_session(tmp_path, name="2026-08-26_umiV4_0002R_vive_task_zzzz")
    _make_session(tmp_path, name="2026-08-27_umiV4_0003R_vive_task_bbbb")

    sessions = audit.list_sessions([tmp_path], active=str(active))
    assert sessions[0].active is True
    assert sessions[0].name == active.name
    # The rest keep newest-name-first order.
    rest = [session.name for session in sessions[1:]]
    assert rest == sorted(rest, reverse=True)


# ---------------------------------------------------- the session being recorded


def _armed_session(station, *, episodes=1):
    """On disk, the folder the console's draft is currently armed for.

    Derived from the draft rather than named literally, because that is the
    same question `_active_folder` asks -- a hand-written name would assert
    against a string this station may not agree with. A preset is chosen
    first because `session.data_dir` refuses a draft without one, and a
    console that cannot say where it would record has no active session.
    """
    draft = station.app.state.rig.draft
    draft.preset_id = "bimanual_v4"
    draft.task_name = "taskA"
    name = pathlib.Path(session_state.data_dir(draft)).name
    return station.session(name, episodes=episodes)


def _realign_forms(html, folder):
    return [
        form
        for form in parse(html).forms
        if form["action"] == "/audit/health-check"
        and form["fields"].get("session_folder") == str(folder)
        and "dry_run" not in form["fields"]
    ]


def test_the_row_being_recorded_reads_recording_and_offers_no_re_align(station):
    """Its episodes may still be in the background alignment queue, so a gap
    there is not yet a gap, and a re-align would race the aligner."""
    folder = _armed_session(station)

    html = station.get().text

    assert "Recording" in rows_by_session(html)[folder.name]
    assert not _realign_forms(html, folder)


# Derive the active recording folder from the generated config, not an
# unsaved draft that may already describe another session.


@pytest.fixture
def task_registry(monkeypatch):
    """The config page's deterministic task table, as `test_config_page` does.

    Only the Apply-driven test needs it, so it is requested rather than autouse:
    the other tests in this file never reach the registry, and an
    autouse patch would hide that.
    """
    monkeypatch.setattr(
        task_registry_store, "get_all_tasks", lambda: dict(config_page.TASKS)
    )


def _armed_data_dir(client) -> pathlib.Path:
    """The folder the applied config names, expanded.

    Apply writes `output/raw/{task}/{session}` under the per-test
    `UMI_OUTPUT_ROOT`, which is the tree the audit page walks.
    """
    path = readiness.generated_config(client.app)
    assert path is not None and path.is_file(), "Apply wrote no config"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = str((cfg.get("recording") or {}).get("data_dir") or "")
    assert raw, "the applied config names no data_dir"
    return pathlib.Path(raw).expanduser()


def test_the_armed_row_keeps_its_badge_after_the_draft_moves_off_it(
    tmp_path, task_registry
):
    """An operator who applies, records, and then starts typing the next task
    name without pressing Apply has moved the draft and not the file. The row
    the recorder is still writing into keeps its badge."""
    client = config_page._client(config_page._repo(tmp_path))
    config_page._apply(client)
    armed = _armed_data_dir(client)
    armed.mkdir(parents=True, exist_ok=True)
    (armed / "episode_0").mkdir(exist_ok=True)
    assert armed.parent.parent == audit.recordings_root()
    client.app.state.rig.draft.task_name = "the_next_task"

    html = client.get("/audit").text

    assert "Recording" in rows_by_session(html)[armed.name]
    assert not _realign_forms(html, armed)


def test_an_unreadable_armed_config_falls_back_to_the_draft(station):
    """`data_dir` raises `FieldError` for an invalid uid or a malformed date --
    both of which an operator can type -- so an unreadable armed file has to
    degrade towards the draft rather than towards no active session at all."""
    folder = _armed_session(station)
    path = readiness.generated_config(station.app)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("recording: [not a mapping\n")  # unparseable YAML

    assert "Recording" in rows_by_session(station.get().text)[folder.name]


# Replace conversion with a script that reports its arguments; these
# tests validate launch wiring without converting data.


def _convert_forms(html, folder):
    return [
        form
        for form in parse(html).forms
        if form["action"] == "/audit/lerobot"
        and form["fields"].get("session_folder") == str(folder)
    ]


def _aligned_root(folder) -> pathlib.Path:
    """Give the session's aligned copy the Zarr root the converter reads."""
    aligned = pathlib.Path(counterpart(str(folder), "aligned"))
    aligned.mkdir(parents=True, exist_ok=True)
    _write_alignment_root(aligned)
    return aligned


def _write_alignment_root(aligned):
    aligned.mkdir(parents=True, exist_ok=True)
    (aligned / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {"index_source": "right_umi_camera"},
            }
        )
    )


def _mark_aligned_episode(aligned, idx):
    """Metadata needed by the lightweight listing, without loading Zarr."""
    _write_alignment_root(aligned)
    episode = aligned / f"episode_{idx}"
    episode.mkdir(parents=True, exist_ok=True)
    (episode / "zarr.json").write_text('{"zarr_format": 3, "node_type": "group"}')
    for name in (
        "valid_indices",
        "valid_timestamps",
        "right_umi_camera/aligned_frame_indices",
    ):
        array = episode / name
        array.mkdir(parents=True, exist_ok=True)
        (array / "zarr.json").write_text(
            json.dumps({"node_type": "array", "shape": [3]})
        )


@pytest.mark.parametrize(
    "defect",
    [
        "metadata_only",
        "empty_timestamps",
        "corrupt_indices",
        "missing_video",
        "partial_video",
    ],
)
def test_incomplete_alignment_cannot_be_counted_or_converted(station, defect):
    name = "2026-08-04_umi1R_taskA_aaaaaaaaaaaa"
    folder = station.session(name, episodes=1)
    aligned = _aligned_root(folder)
    episode = aligned / "episode_0"
    if defect == "metadata_only":
        episode.mkdir()
        (episode / "zarr.json").write_text('{"node_type": "group", "attributes": {}}')
    else:
        _mark_aligned_episode(aligned, 0)
        if defect == "empty_timestamps":
            (episode / "valid_timestamps/zarr.json").write_text(
                '{"node_type": "array", "shape": [0]}'
            )
        elif defect == "corrupt_indices":
            (episode / "right_umi_camera/aligned_frame_indices/zarr.json").write_text(
                "unreadable"
            )
        else:
            (folder / "episode_0/left_umi_camera.mp4").write_bytes(b"recorded-video")
            if defect == "partial_video":
                (episode / "left_umi_camera.mp4").write_bytes(b"part")

    body = station.get().text
    assert "0 of 1 aligned" in rows_by_session(body)[name]
    assert not _convert_forms(body, folder)
    assert any(
        form["action"] == "/audit/health-check" and "dry_run" not in form["fields"]
        for form in parse(body).forms
    )

    # A replayed request must be refused before launching the converter too.
    _stub_converter(station)
    response = station.client.post(
        "/audit/lerobot", data={"session_folder": str(folder)}, follow_redirects=False
    )
    assert response.status_code == 409
    assert "0 of 1 episodes have completed alignment" in response.text
    assert "Run Check and Re-align" in response.text
    registry = station.registry()
    assert registry is None or not registry.all()


def test_one_complete_episode_does_not_make_an_incomplete_session_convertible(station):
    name = "2026-08-04_umi1R_taskA_aaaaaaaaaaaa"
    folder = station.session(name, episodes=2, aligned=(1,))
    body = station.get().text
    assert "1 of 2 aligned" in rows_by_session(body)[name]
    assert not _convert_forms(body, folder)
    response = station.client.post(
        "/audit/lerobot", data={"session_folder": str(folder)}, follow_redirects=False
    )
    assert response.status_code == 409


@pytest.mark.parametrize("copy_videos", [False, True])
def test_completed_alignment_honors_the_video_copy_setting(station, copy_videos):
    folder = station.session(
        "2026-08-04_umi1R_taskA_aaaaaaaaaaaa", episodes=1, aligned=(0,)
    )
    (folder / "episode_0/left_umi_camera.mp4").write_bytes(b"recorded-video")
    if copy_videos:
        (audit.aligned_folder(folder) / "episode_0/left_umi_camera.mp4").write_bytes(
            b"recorded-video"
        )
    else:
        (folder / "config.yaml").write_text(
            "meta:\n  task_name: taskA\nalignment:\n  copy_videos: false\n"
        )
    assert audit.Session(folder).aligned == 1


def _stub_converter(station):
    """A converter that prints its arguments, on a station with its modules."""
    path = station.tmp / "convert.py"
    path.write_text("import sys\nprint('ARGS', ' '.join(sys.argv[1:]))\n")
    station.monkeypatch.setattr(audit, "LEROBOT_CONVERTER", str(path))
    station.monkeypatch.setattr(audit, "missing_lerobot_modules", lambda: [])
    return path


def test_an_aligned_session_is_offered_a_lerobot_conversion(station):
    folder = station.session(
        "2026-08-04_umi1R_taskA_aaaaaaaaaaaa", episodes=1, aligned=(0,)
    )

    assert len(_convert_forms(station.get().text, folder)) == 1


def test_a_session_with_nothing_aligned_is_offered_no_conversion(station):
    folder = station.session("2026-08-04_umi1R_taskA_aaaaaaaaaaaa", episodes=2)

    assert not _convert_forms(station.get().text, folder)


def _armed_and_aligned(station):
    folder = _armed_session(station)
    _aligned_root(folder)
    _mark_aligned_episode(pathlib.Path(counterpart(str(folder), "aligned")), 0)
    return folder


def _close_forms(html, folder):
    return [
        form
        for form in parse(html).forms
        if form["action"] == "/audit/session/close"
        and form["fields"].get("session_folder") == str(folder)
    ]


def _close(station, folder, **extra):
    return station.client.post(
        "/audit/session/close",
        data={"session_folder": str(folder), **extra},
        follow_redirects=False,
    )


def _not_recording(station):
    station.monkeypatch.setattr(
        audit_routes.recording, "in_progress", lambda app: False
    )


def test_the_current_session_is_closed_before_it_is_converted(station):
    """Finish recording, close the session, convert: until it is closed more
    episodes may still arrive, so Convert is shown unavailable, with why."""
    folder = _armed_and_aligned(station)
    _not_recording(station)

    html = station.get().text

    assert len(_close_forms(html, folder)) == 1
    assert not _convert_forms(html, folder)
    row = rows_by_session(html)[folder.name]
    assert "Convert to LeRobot" in row
    assert "Finish and close this session to convert it" in row

    _stub_converter(station)
    response = station.client.post(
        "/audit/lerobot", data={"session_folder": str(folder)}, follow_redirects=False
    )
    assert response.status_code == 409
    assert "Finish and close it" in response.text


def test_while_recording_neither_close_nor_convert_can_be_pressed(station):
    folder = _armed_and_aligned(station)
    station.monkeypatch.setattr(audit_routes.recording, "in_progress", lambda app: True)

    html = station.get().text

    assert not _close_forms(html, folder)
    assert not _convert_forms(html, folder)
    row = rows_by_session(html)[folder.name]
    assert "Finish and close session" in row
    assert "A recording is running into this session" in row
    assert 'aria-disabled="true"' in html

    response = _close(station, folder, confirm="yes")
    assert response.status_code == 409
    assert "still being recorded" in response.text
    assert not session_seal.folder_is_sealed(folder)


def test_closing_asks_for_confirmation_first(station):
    folder = _armed_and_aligned(station)
    _not_recording(station)

    response = _close(station, folder)

    assert response.status_code == 200
    confirm = [
        form
        for form in parse(response.text).forms
        if form["action"] == "/audit/session/close"
        and form["fields"].get("confirm") == "yes"
    ]
    assert len(confirm) == 1
    assert confirm[0]["fields"]["session_folder"] == str(folder)
    assert not session_seal.folder_is_sealed(folder)


def test_closing_seals_the_session_and_then_offers_its_conversion(station):
    folder = _armed_and_aligned(station)
    _not_recording(station)
    uid = station.app.state.rig.draft.session_uid

    response = _close(station, folder, confirm="yes")

    assert response.status_code == 303
    assert session_seal.read_seal(folder)["reason"] == "operator"
    # The console has moved on: a new session id, so the closed folder is no
    # longer the current one and can be converted.
    assert station.app.state.rig.draft.session_uid != uid
    page = station.client.get(response.headers["location"])
    row = rows_by_session(page.text)[folder.name]
    assert "Closed" in row
    assert "Recording" not in row
    assert f"Closed {folder.name}" in page.text
    assert len(_convert_forms(page.text, folder)) == 1


def test_only_the_current_session_can_be_closed(station):
    _armed_and_aligned(station)
    _not_recording(station)
    other = station.session(
        "2026-08-03_umi1R_taskA_bbbbbbbbbbbb", episodes=1, aligned=(0,)
    )

    response = _close(station, other, confirm="yes")

    assert response.status_code == 409
    assert "current session; it is already closed" in response.text
    assert not session_seal.folder_is_sealed(other)


def test_closing_with_unapplied_configuration_changes_is_refused(station):
    """The sequence seals the draft's folder; with the draft moved off the armed
    one it would seal the wrong folder and leave the armed one open."""
    folder = _armed_and_aligned(station)
    _not_recording(station)
    station.monkeypatch.setattr(audit_routes, "_active_folder", lambda app: str(folder))
    station.app.state.rig.draft.task_name = "taskB"

    response = _close(station, folder, confirm="yes")

    assert response.status_code == 409
    assert "not applied" in response.text
    assert not session_seal.folder_is_sealed(folder)


def test_a_converted_session_shows_its_badge_instead_of_the_button(station):
    name = "2026-08-04_umi1R_taskA_aaaaaaaaaaaa"
    folder = station.session(name, episodes=1, aligned=(0,))
    meta = (
        audit.output_root()
        / audit.LEROBOT_DIR
        / "taskA"
        / f"gripper_umi_trigger_camera.{name}"
        / "meta"
    )
    meta.mkdir(parents=True)
    (meta / "info.json").write_text("{}")

    html = station.get().text

    assert "LeRobot format" in rows_by_session(html)[name]
    assert not _convert_forms(html, folder)


def test_an_interrupted_conversion_does_not_count_as_converted(station):
    name = "2026-08-04_umi1R_taskA_aaaaaaaaaaaa"
    folder = station.session(name, episodes=1, aligned=(0,))
    (
        audit.output_root()
        / audit.LEROBOT_DIR
        / "taskA"
        / f"gripper_umi_trigger_camera.{name}"
        / "data"
    ).mkdir(parents=True)

    html = station.get().text

    assert "LeRobot format" not in rows_by_session(html)[name]
    assert len(_convert_forms(html, folder)) == 1


def test_converting_runs_the_converter_on_the_aligned_copy_into_lerobot_format(
    station,
):
    folder = station.session(
        "2026-08-04_umi1R_taskA_aaaaaaaaaaaa", episodes=1, aligned=(0,)
    )
    aligned = _aligned_root(folder)
    _stub_converter(station)

    response = station.client.post(
        "/audit/lerobot", data={"session_folder": str(folder)}, follow_redirects=False
    )
    assert response.status_code == 303
    job = station.wait(station.job_of(response))

    printed = "\n".join(job.lines())
    assert job.state == jobs.DONE, printed
    assert f"--output-root {audit.output_root()}" in printed
    assert f"--input-dataset {aligned}" in printed
    # The station's own interpreter runs it, as it runs the health check.
    assert audit.lerobot_argv(aligned)[0] == sys.executable
    assert job.label == f"LeRobot conversion — {folder.name}"


def test_converting_a_session_with_no_aligned_copy_is_refused(station):
    folder = station.session("2026-08-04_umi1R_taskA_aaaaaaaaaaaa", episodes=1)
    _stub_converter(station)

    response = station.client.post(
        "/audit/lerobot", data={"session_folder": str(folder)}, follow_redirects=False
    )

    assert response.status_code == 409
    assert "no aligned copy" in response.text
    registry = station.registry()
    assert registry is None or not registry.all(), "a job was started anyway"


def test_a_station_without_the_converter_modules_is_told_to_rerun_setup(station):
    folder = station.session(
        "2026-08-04_umi1R_taskA_aaaaaaaaaaaa", episodes=1, aligned=(0,)
    )
    _aligned_root(folder)
    station.monkeypatch.setattr(
        audit, "missing_lerobot_modules", lambda: ["torchcodec"]
    )

    response = station.client.post(
        "/audit/lerobot", data={"session_folder": str(folder)}, follow_redirects=False
    )

    assert response.status_code == 409
    assert "torchcodec" in response.text
    assert "setup_station.sh" in response.text


def test_the_converter_writes_where_the_audit_page_looks():
    """The page's badge and the converter's output agree on the folder."""
    source = (REPO / audit.LEROBOT_CONVERTER).read_text()
    assert f'LOCAL_DATASETS_DIR = "{audit.LEROBOT_DIR}"' in source


def test_the_station_installs_what_the_converter_declares():
    """The converter's inline `# /// script` block is what a standalone `uv run`
    installs; the `data_collection` group is what a station installs. A package
    added to one and not the other works in one place and fails in the other."""
    import re
    import tomllib

    def name(requirement):
        return re.split(r"[<>=!~\[; ]", requirement, maxsplit=1)[0].lower()

    source = (REPO / audit.LEROBOT_CONVERTER).read_text()
    block = re.search(r"^# /// script\n(.*?)^# ///$", source, re.S | re.M).group(1)
    script = tomllib.loads(re.sub(r"^# ?", "", block, flags=re.M))
    project = tomllib.loads((REPO / "pyproject.toml").read_text())
    station = {
        name(req)
        for req in project["project"]["dependencies"]
        + project["dependency-groups"]["data_collection"]
    }

    assert {name(req) for req in script["dependencies"]} <= station

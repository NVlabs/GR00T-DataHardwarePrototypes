# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""While stopping, the page must describe the processes, not the job object.

The Record page renders from what the recording is actually doing -- its
process group -- rather than from what the console remembers about it. Three
cases depend on that:

* **An adopted recorder must not look idle.** After a console restart there is
  no job object -- `recording.adopt` reattaches only the pgid -- so a page that
  checked `{% if job and not job.finished %}` would show a "Not recording" badge
  and a live Start button over a recorder that is still running and still
  holding every camera, with no Stop button on it. `/record/start` refuses the
  POST anyway (its guard reads the process group), so this is not unsafe, but
  the page would describe nothing the operator can see.

* **The reassurance belongs to one phase.** "Queued alignments are finishing,
  this is normal" is true only during the drain. Before the drain begins, and
  throughout the teardown, a hang means a camera has dropped off the bus and is
  not normal.

* **A kernel-blocked group is reported early.** A process in uninterruptible
  sleep takes no signal, so the console reads each member's `/proc` state and
  reports the block before the Force stop countdown ends, instead of letting
  the operator wait it out and press a destructive button that cannot work.

All three are answered from `recording.survivors()` -- the same process group
walk the recording lock uses.
"""

from __future__ import annotations

import pathlib
import signal

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

# Same directory, no `__init__.py`, so pytest's prepend import mode has already
# put `tests/station_ui` on sys.path.
import test_config_page as config_page  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from station_ui import process_group, recording  # noqa: E402
from station_ui.app import create_app  # noqa: E402
from station_ui.routes import record as record_route  # noqa: E402
from station_ui.settings import Settings  # noqa: E402
from umi.common.recording import shutdown_progress  # noqa: E402
from umi.common.tasks import task_registry_store  # noqa: E402

LEADER = 4242


def _proc(pid=LEADER, state="S", cmdline="record_data.py"):
    return process_group.ProcInfo(pid=pid, cmdline=cmdline, state=state)


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(
        task_registry_store, "get_all_tasks", lambda: dict(config_page.TASKS)
    )


@pytest.fixture
def station(tmp_path):
    repo = config_page._repo(tmp_path)
    client = TestClient(create_app(Settings(repo_root=repo)))
    client.get("/")
    client.post("/setup/preset", data={"preset_id": "bimanual_v4"})
    config_page._apply(client, head=False)
    return client


@pytest.fixture
def adopted(station, monkeypatch):
    """A station holding a recorder it did not start.

    Exactly the shape `recording.adopt` leaves behind: a pgid and a pid, no job
    object, and no captured output. Patched at `process_group.survivors` so the
    page and the lock read the same fake group -- nothing here asserts against
    a second notion of "recording".
    """
    state = recording.state_for(station.app)
    state.pid = state.pgid = LEADER
    state.adopted = True
    state.record_job_id = None
    monkeypatch.setattr(process_group, "survivors", lambda pgid: [_proc()])
    monkeypatch.setattr(process_group, "alive", lambda pid: True)
    return station


# --------------------------------------------------------- the phase markers


def test_a_marker_round_trips_through_the_log():
    lines = ["opening cameras", shutdown_progress.marker(shutdown_progress.TEARDOWN)]
    assert shutdown_progress.phase_from_lines(lines) == shutdown_progress.TEARDOWN


def test_the_latest_marker_wins():
    """Phases only move forward, so the last one printed is the current one."""
    lines = [
        shutdown_progress.marker(shutdown_progress.TEARDOWN),
        "Discarded pending episode 3",
        shutdown_progress.marker(shutdown_progress.DRAIN),
        "[aligner] episode_2 aligned",
    ]
    assert shutdown_progress.phase_from_lines(lines) == shutdown_progress.DRAIN


def test_an_unmarked_log_reports_no_phase():
    """None means "say nothing specific", which is what makes this safe to read
    from a recorder started by a build that predates the markers."""
    assert shutdown_progress.phase_from_lines(["Recording stopped"]) is None
    assert shutdown_progress.phase_from_lines([]) is None


def test_the_markers_carry_no_ansi_escapes():
    """The recorder colours most of its output. An escape inside the marker
    would break the substring match on the captured line."""
    for phase in shutdown_progress.PHASES:
        assert "\x1b" not in shutdown_progress.marker(phase)


def test_the_recorder_marks_teardown_before_the_drain():
    """Order in the source is the contract: a drain marker printed early would
    show the "this is normal" copy over the teardown."""
    import inspect

    from real_scripts.data_collection import record_data

    source = inspect.getsource(record_data._run_session)
    shutdown = source[source.rindex("\n    finally:") :]
    assert (
        shutdown.index("shutdown_progress.TEARDOWN")
        < shutdown.index("shutdown_progress.DRAIN")
        < shutdown.index("shutdown_progress.DONE")
    )
    # The drain marker has to precede the call it describes, or the page
    # promises "alignments are finishing" only once they already have.
    assert shutdown.index("shutdown_progress.DRAIN") < shutdown.index(
        "shutdown_background_services()"
    )


# ------------------------------------------------------ the kernel-block clock


def test_a_single_blocked_sample_is_not_enough(station):
    """Ordinary disk and network I/O parks a process in `D` constantly. One
    sample is a false-positive machine; only sustained blocking is a signal."""
    age = recording.kernel_block_age(station.app, [_proc(state="D")])
    assert age is not None
    assert age < recording.KERNEL_BLOCK_S


def test_any_process_leaving_D_resets_the_clock(station):
    """The group is making progress, whatever the rest of it is doing."""
    recording.kernel_block_age(station.app, [_proc(state="D")])
    assert recording.state_for(station.app).blocked_since is not None

    assert recording.kernel_block_age(station.app, [_proc(state="S")]) is None
    assert recording.state_for(station.app).blocked_since is None


def test_an_empty_group_is_not_blocked(station):
    assert recording.kernel_block_age(station.app, []) is None


def test_one_blocked_process_among_healthy_ones_still_counts(station):
    """SIGKILL cannot empty a group with a `D` member in it, so Force stop
    cannot succeed -- which is the thing the operator needs told."""
    procs = [_proc(pid=1, state="S"), _proc(pid=2, state="D")]
    assert recording.kernel_block_age(station.app, procs) is not None


# -------------------------------------------------------- the adopted recorder


def test_an_adopted_recording_is_not_reported_as_idle(adopted):
    body = adopted.get("/record").text
    assert "Not recording" not in body
    assert "Adopted after a console restart" in body


def _button(html, action):
    """The button inside the form posting to `action`."""
    form = html.split(f'action="{action}"', 1)[1].split("</form>")[0]
    return form[form.index("<button") :]


def test_an_adopted_recording_offers_a_live_stop_and_a_dead_start(adopted):
    """Both are rendered, and that is the point: as one control in one
    position, the card could hand back a live Start under the finger that had
    just pressed Stop."""
    body = adopted.get("/record").text

    assert "disabled" not in _button(body, "/record/stop")
    assert "disabled" in _button(body, "/record/start"), (
        "a live Start over a running recorder: the POST is refused, but the "
        "page is inviting an action that cannot work"
    )
    assert "A recording is in progress" in body


def _actions_row(html):
    """The fixed lever row, as rendered."""
    row = html[html.index('class="row record-actions"') :]
    return row[: row.index("</div>", row.index("</form>"))]


def test_the_levers_share_one_row_in_a_fixed_order(adopted):
    """Start, then Stop, then Force kill -- left to right.

    Stacked vertically they could be pushed around by anything rendered between
    them, so pressing Stop once could hand back a live Start under the same
    finger. Side by side in one row, each keeps its slot in every state, and
    everything that grows and shrinks is outside the row.
    """
    row = _actions_row(adopted.get("/record").text)
    order = [row.index(f'action="/record/{a}"') for a in ("start", "stop", "pkill")]
    assert order == sorted(order), "the levers are not in Start, Stop, Force kill order"


def test_nothing_pressable_renders_between_the_levers(adopted):
    """The row's whole value is that its slots are fixed. A control that only
    sometimes appears inside it would shift the ones after it."""
    row = _actions_row(adopted.get("/record").text)
    assert row.count("<form") == 3, "an extra form inside the lever row"
    assert row.count("<button") == 3, "an extra button inside the lever row"


def test_the_contextual_blocks_stay_outside_the_row(adopted, monkeypatch):
    """They come and go, so inside the row they would move a lever."""
    for body in (_stopping_body(adopted), _straggler_body(adopted)):
        row = _actions_row(body)
        assert 'action="/record/force-stop"' not in row


def test_stopping_an_adopted_recording_signals_the_leader_alone(adopted, monkeypatch):
    """There is no job object, so `jobs.cancel` cannot do it. The leader-only
    rule still holds: the drain runs in forked workers that inherit
    `_stop_on_sigterm`, and a group signal kills the very processes draining."""
    sent = []
    monkeypatch.setattr(recording.os, "kill", lambda pid, sig: sent.append((pid, sig)))

    assert adopted.post("/record/stop").status_code == 200

    assert sent == [(LEADER, signal.SIGTERM)]


def test_a_dead_adopted_leader_is_not_signalled(station, monkeypatch):
    """Its pid may have been recycled onto somebody else's process."""
    state = recording.state_for(station.app)
    state.pid = state.pgid = LEADER
    monkeypatch.setattr(process_group, "alive", lambda pid: False)
    sent = []
    monkeypatch.setattr(recording.os, "kill", lambda pid, sig: sent.append(pid))

    assert recording.request_stop(station.app) is False
    assert sent == []


def test_an_adopted_recording_says_its_log_is_gone(adopted):
    """`adopt` leaves the buffer empty deliberately. An empty box reads as "the
    recorder printed nothing", which is a different and wrong claim."""
    assert "Not available for this recording" in adopted.get("/record").text


# ------------------------------------------------------------- the stopping copy


def _stopping(client, *, lines, blocked=False):
    """Put the station into `stopping`, with `lines` as the recorder's output."""
    state = recording.state_for(client.app)
    state.stopping_since = 0.0
    state.pid = state.pgid = LEADER
    state.blocked_since = None
    proc = _proc(state="D" if blocked else "S")
    monkeypatch_target = [proc]
    if blocked:
        # Already blocked for longer than the threshold when the page renders.
        state.blocked_since = -recording.KERNEL_BLOCK_S * 10
    job = _FakeJob(lines)
    state.jobs._jobs[job.id] = job  # noqa: SLF001 - the registry has no seam
    state.record_job_id = job.id
    return monkeypatch_target


class _FakeJob:
    """The narrow slice of `jobs.Job` that `_context` touches."""

    id = "fake"
    state = "running"
    finished = False
    exit_code = None
    error = ""
    pgid = LEADER
    last_output_at = 0.0

    def __init__(self, lines):
        self._lines = list(lines)

    def lines(self):
        return list(self._lines)


@pytest.fixture
def stopping(station, monkeypatch):
    def _go(*, phase=None, blocked=False):
        lines = ["Interrupted by user"]
        if phase:
            lines.append(shutdown_progress.marker(phase))
        procs = _stopping(station, lines=lines, blocked=blocked)
        monkeypatch.setattr(process_group, "survivors", lambda pgid: procs)
        return station.get("/record").text

    return _go


def test_teardown_does_not_promise_that_silence_is_normal(stopping):
    body = stopping(phase=shutdown_progress.TEARDOWN)
    assert "Closing cameras and grippers" in body
    assert "this is normal" not in body, (
        "the drain's reassurance shown during teardown, where a hang means a "
        "camera has dropped"
    )


def test_teardown_names_the_likely_cause_of_a_hang(stopping):
    assert "a camera has probably disconnected" in stopping(
        phase=shutdown_progress.TEARDOWN
    )


def test_the_drain_is_where_silence_is_promised_to_be_normal(stopping):
    body = stopping(phase=shutdown_progress.DRAIN)
    assert "this is normal" in body
    assert "Cameras and grippers are released" in body


def test_an_unreported_phase_claims_nothing_specific(stopping):
    body = stopping(phase=None)
    assert "Waiting for the recorder to finish shutting down" in body
    assert "this is normal" not in body
    assert "Closing cameras" not in body


def test_a_kernel_blocked_group_outranks_every_phase(stopping):
    """Reported before the countdown ends, because Force stop is already known
    not to work: a process in uninterruptible sleep takes no signal at all."""
    body = stopping(phase=shutdown_progress.DRAIN, blocked=True)
    assert "The recorder is blocked on a device" in body
    assert "this station needs a reboot" in body.lower()
    assert "this is normal" not in body


def test_a_healthy_drain_is_never_called_blocked(stopping):
    assert "blocked on a device" not in stopping(phase=shutdown_progress.DRAIN)


def test_the_episode_keys_are_hidden_once_stopping(stopping):
    """`s`/`w`/`a` do nothing after SIGTERM. Telling somebody to press `w`
    while their episode is being torn down is worse than saying nothing."""
    assert "in the recorder window" not in stopping(phase=shutdown_progress.TEARDOWN)


# ------------------------------------------ Stop and Start share one position


def test_a_fresh_console_offers_start_immediately(station):
    """The hold arms on an observed transition, never on a page load. A station
    opened after a recording ended never saw it alive and must not be told to
    wait for something it did not watch happen."""
    body = station.get("/record").text
    assert "Start recording</button>" in body.replace("\n", "").replace("  ", "")
    assert "The previous recording has ended" not in body


def test_pressing_stop_after_the_recorder_exited_does_not_hand_back_a_live_start(
    station, monkeypatch
):
    """The double-press race, end to end.

    The operator's page still shows Stop -- the recorder quit on the cv2 `q`
    key and the reload has not landed. They press it. htmx swaps in this card,
    whose first control sits exactly where Stop was. Without the hold their
    second press (there is almost always a second: the first appeared to do
    nothing) would start a recorder and open a new cv2 window.
    """
    state = recording.state_for(station.app)
    state.pid = state.pgid = LEADER

    # The console renders once while the recorder is alive -- this is the page
    # the operator is looking at, with Stop on it.
    alive = [_proc()]
    monkeypatch.setattr(process_group, "survivors", lambda pgid: alive)
    assert 'action="/record/stop"' in station.get("/record").text

    # `q` is pressed at the bench: the group empties. The browser does not know.
    alive.clear()

    body = station.post("/record/stop").text
    assert "The previous recording has ended" in body
    assert 'action="/record/start"' in body, "the form is still there..."
    assert (
        "disabled" in body.split('action="/record/start"', 1)[1].split("</form>")[0]
    ), (
        "...but its button must not be pressable by the second half of a "
        "double-press aimed at Stop"
    )


def test_the_hold_expires_on_its_own(station, monkeypatch):
    """A button that stays disabled until a manual reload is a worse dead end
    than the race it was added to close."""
    state = recording.state_for(station.app)
    state.pid = state.pgid = LEADER
    monkeypatch.setattr(process_group, "survivors", lambda pgid: [])
    state.was_live = True

    quiet = recording.note_liveness(station.app, live=False)
    assert quiet is not None
    # Pretend the cool-down has elapsed.
    state.went_quiet_at -= record_route.START_COOLDOWN_S * 2

    body = station.get("/record").text
    assert "The previous recording has ended" not in body
    assert "Start recording" in body


def test_the_held_button_is_refreshed_from_outside_every_form(station, monkeypatch):
    """The countdown needs a poller, and where the poller lives matters.

    htmx resolves `hx-confirm` with `getClosestAttributeValue` -- it walks up
    the DOM. A poller inside the Start form would therefore inherit that form's
    "Start recording?" and raise a browser confirm once a second; answering No
    aborts the refresh, freezing the countdown with the button permanently
    disabled.
    """
    state = recording.state_for(station.app)
    state.pid = state.pgid = LEADER
    state.was_live = True
    monkeypatch.setattr(process_group, "survivors", lambda pgid: [])

    held = station.get("/record").text
    assert 'id="record-watch"' in held, "nothing advances the countdown"

    watch = held.index('id="record-watch"')
    start_form = held.index('action="/record/start"')
    assert watch < start_form, "the poller is inside the confirmed Start form"
    # No hx-trigger anywhere inside the Start form: any request originating
    # there inherits its hx-confirm.
    form = held[start_form:].split("</form>")[0]
    assert "hx-trigger" not in form and "hx-get" not in form


def test_a_recording_that_starts_again_clears_the_hold(station, monkeypatch):
    """Liveness returning resets the clock, so a restarted recorder is not
    followed by a stale cool-down when it later stops."""
    state = recording.state_for(station.app)
    state.pid = state.pgid = LEADER
    state.was_live = True
    recording.note_liveness(station.app, live=False)
    assert state.went_quiet_at is not None

    assert recording.note_liveness(station.app, live=True) is None
    assert state.went_quiet_at is None
    assert state.was_live is True


# ------------------------------- the page has to correct itself without a stream


def _finished_job(state, lines=()):
    job = _FakeJob(lines)
    job.finished = True
    job.state = "done"
    job.exit_code = 0
    state.jobs._jobs[job.id] = job  # noqa: SLF001 - the registry has no seam
    state.record_job_id = job.id
    return job


def test_a_group_outliving_its_job_keeps_polling(station, monkeypatch):
    """A group that outlives its job keeps the page polling.

    The SSE stream is keyed on the job; these controls are keyed on the process
    group, and the two do not end together -- `jobs._finish` reaps stragglers on
    a 2s/3s timer and a forked worker routinely outlives the leader. After a
    cv2 `q`: the job finishes, the stream sends `end`, the page reloads, the
    group is still non-empty so Stop renders, and the stream is gone because it
    only exists while the job is unfinished. Without a poller nothing would ever
    re-render the page again.
    """
    state = recording.state_for(station.app)
    state.pid = state.pgid = LEADER
    _finished_job(state)
    monkeypatch.setattr(process_group, "survivors", lambda pgid: [_proc()])

    body = station.get("/record").text

    assert 'id="record-watch"' in body, (
        "the recorder exited with its group still alive and nothing on the "
        "page would ever refresh it again"
    )
    watch = body.split('id="record-watch"')[1].split(">")[0]
    assert "hx-trigger" in watch and "/record" in watch


def test_a_live_job_is_left_to_its_stream(station, monkeypatch):
    """The poller must not swap the card while the EventSource is in it: htmx
    executes <script> in swapped content, so polling would close and reopen the
    stream, and `onopen` clears the log as it goes."""
    state = recording.state_for(station.app)
    state.pid = state.pgid = LEADER
    job = _FakeJob([])
    state.jobs._jobs[job.id] = job  # noqa: SLF001
    state.record_job_id = job.id
    monkeypatch.setattr(process_group, "survivors", lambda pgid: [_proc()])

    body = station.get("/record").text

    assert 'id="record-watch"' not in body
    assert "EventSource" in body, "the stream is what refreshes this state"


def test_stragglers_are_named_rather_than_left_inexplicable(station, monkeypatch):
    """A Stop button over a finished recording cannot explain itself -- there
    is nothing left to signal."""
    state = recording.state_for(station.app)
    state.pid = state.pgid = LEADER
    _finished_job(state)
    monkeypatch.setattr(
        process_group, "survivors", lambda pgid: [_proc(pid=91), _proc(pid=92)]
    )

    body = station.get("/record").text

    assert "The recorder has exited" in body
    assert "91, 92" in body


def test_stragglers_get_a_force_stop_without_waiting(station, monkeypatch):
    """The staged Force stop renders on `stopping_for`, which a `q` quit never
    sets -- so without this there is no lever here at all. And no countdown:
    the recorder has already exited, so there is no drain left to protect."""
    state = recording.state_for(station.app)
    state.pid = state.pgid = LEADER
    _finished_job(state)
    monkeypatch.setattr(process_group, "survivors", lambda pgid: [_proc()])

    body = station.get("/record").text
    form = body.split('action="/record/force-stop"')[1].split("</form>")[0]

    assert "Force stop leftovers" in form
    assert "disabled" not in form


def test_an_empty_group_stops_polling_and_offers_start(station, monkeypatch):
    """Self-terminating: the poller exists only while there is something to
    watch for, and what it is watching for is this."""
    state = recording.state_for(station.app)
    state.pid = state.pgid = LEADER
    _finished_job(state)
    monkeypatch.setattr(process_group, "survivors", lambda pgid: [])

    body = station.get("/record").text

    assert 'id="record-watch"' not in body
    assert "The recorder has exited" not in body
    assert 'action="/record/start"' in body


def _stopping_body(client):
    """The card mid-stop, where the staged Force stop renders."""
    recording.state_for(client.app).stopping_since = 0.0
    body = client.get("/record").text
    recording.state_for(client.app).stopping_since = None
    return body


def _straggler_body(client):
    """The card with a finished job over a live group, where the leftover
    Force stop renders."""
    state = recording.state_for(client.app)
    job = _FakeJob([])
    job.finished, job.state, job.exit_code = True, "done", 0
    state.jobs._jobs[job.id] = job  # noqa: SLF001
    previous, state.record_job_id = state.record_job_id, job.id
    body = client.get("/record").text
    state.record_job_id = previous
    return body


def test_force_kill_is_pushed_away_from_the_two_that_matter(adopted):
    """It is a recovery lever, not a third peer. `.record-actions__last` puts it
    at the far edge, so a press aimed at Stop cannot drift onto a SIGKILL."""
    row = _actions_row(adopted.get("/record").text)
    at = row.index('action="/record/pkill"')
    assert "record-actions__last" in row[max(0, at - 200) : at]

    css = (REPO / "station_ui" / "static" / "app.css").read_text()
    assert ".record-actions__last" in css
    assert "margin-left: auto" in css

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A 500 has to arrive somewhere an operator is standing.

Every mutating control on the console is an htmx swap, and htmx does not swap a
non-2xx -- so without a handler an operator could press **Apply
configuration**, the handler would raise, and the UI would do nothing at all.
The traceback would go to the launching terminal's stderr and nowhere else,
which on a station started from a closed terminal is nowhere.

Three jobs, and each has tests here: capture (a file that outlives the
terminal), arrive (the operator learns it broke, where they are standing), read
(the engineer gets the traceback).

`TestClient(app, raise_server_exceptions=False)` throughout, and it is not
optional: `ServerErrorMiddleware` re-raises after the console's response has
been sent, so the default client surfaces the original exception instead of
the 500 the browser would actually receive.
"""

from __future__ import annotations

import pathlib
import re
import threading

import pytest
from starlette.testclient import TestClient

from station_ui import crash, logging_setup, readiness
from station_ui.app import create_app
from station_ui.routes import config as config_routes
from station_ui.settings import Settings

BOOM = "synthetic failure, raised by a test"
ERROR_ID = re.compile(r"^[0-9a-f]{12}$")


def _app(**kwargs):
    return create_app(Settings(**kwargs))


def _client(app):
    return TestClient(app, raise_server_exceptions=False)


def _boom_route(app, path="/test-only/boom"):
    """A route that raises, registered on the app under test.

    Registered rather than monkeypatched wherever the action is not what is
    under test, so the test does not depend on the internals of whichever
    shipped handler happened to be convenient.
    """

    @app.post(path)
    async def _boom():  # pragma: no cover - the body never returns
        raise RuntimeError(BOOM)

    @app.get(path)
    async def _boom_get():  # pragma: no cover - the body never returns
        raise RuntimeError(BOOM)

    return path


def test_a_raising_route_answers_a_page_an_operator_can_act_on():
    """Not `Internal Server Error`. That names nothing, offers nothing, and is
    the string the operator would have to read out over the phone."""
    app = _app()
    path = _boom_route(app)

    response = _client(app).post(path)

    assert response.status_code == 500
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert "Internal Server Error" not in body
    assert "failed" in body
    assert "this station needs an engineer" in body
    # The way out is a link, not a script.
    assert 'href="/log"' in body


def test_a_raising_htmx_post_answers_a_fragment_not_a_whole_document():
    """htmx swaps the response into the page; a whole `<html>` document swapped
    into a banner slot is worse than the silence it replaces."""
    app = _app()
    path = _boom_route(app)

    response = _client(app).post(path, headers={"HX-Request": "true"})

    assert response.status_code == 500
    body = response.text
    assert "<html" not in body.lower()
    assert "<!doctype" not in body.lower()
    assert "banner" in body


def test_every_crash_carries_an_error_id_in_the_body_and_the_header():
    """The header is for curl and CI, the body is for the operator reading it
    aloud, and they have to be the same twelve characters."""
    app = _app()
    path = _boom_route(app)

    response = _client(app).post(path)

    error_id = response.headers["X-Station-Error-Id"]
    assert ERROR_ID.match(error_id), error_id
    assert error_id in response.text


def test_the_traceback_reaches_the_station_log_file():
    """Guards the verified mechanic the whole design hangs on: under uvicorn
    the `uvicorn.error` logger has `propagate: False`, so a root file handler
    never sees the ASGI traceback. The exception handler has to log it itself.
    """
    app = _app()
    path = _boom_route(app)

    error_id = _client(app).post(path).headers["X-Station-Error-Id"]

    text = pathlib.Path(app.state.log_file).read_text()
    assert f"[{error_id}]" in text, "the log line does not carry the error id"
    assert "RuntimeError" in text
    assert BOOM in text
    assert "Traceback (most recent call last)" in text


def test_the_crash_is_listed_on_the_log_page_and_the_id_links_to_it():
    app = _app()
    path = _boom_route(app)
    client = _client(app)

    crashed = client.post(path)
    error_id = crashed.headers["X-Station-Error-Id"]

    page = client.get("/log")
    assert page.status_code == 200
    assert error_id in page.text
    assert f'id="error-{error_id}"' in page.text
    # The id in the crash response is a link to its own entry, because nobody
    # on a station transcribes twelve hex characters into a URL bar.
    assert f"/log?error={error_id}" in crashed.text


def test_a_crash_in_the_shell_context_still_answers_a_page(monkeypatch):
    """The handler must not be able to crash.

    `_shell_context` runs for every TemplateResponse, including the crash page's
    own -- so if it is what raised, rendering the crash page re-raises inside
    the handler and `ServerErrorMiddleware` has no second chance. The fallback
    is hand-built HTML: no template, no context processor, no stylesheet.

    The stub goes on `readiness`, which is where `_shell_context` reads the
    numbered steps from. Patching an alias re-exported by `app` would leave
    this passing against a function nothing calls.
    """
    app = _app()
    client = _client(app)
    client.get("/")  # claim the rig, so the preset below is accepted
    client.post("/setup/preset", data={"preset_id": "bimanual_v4"})
    assert app.state.rig.draft.preset_id, "the draft never took the preset"

    def _raise(_app):
        raise RuntimeError(BOOM)

    monkeypatch.setattr(readiness, "generated_config", _raise)

    response = client.get("/")

    assert response.status_code == 500
    assert "X-Station-Error-Id" in response.headers
    assert "Internal Server Error" not in response.text
    assert "this station needs an engineer" in response.text


def test_the_appbar_counts_errors_only_when_there_are_any():
    """The move that matters most: the console still tells you it broke when
    you are three pages away from where it broke."""
    app = _app()
    path = _boom_route(app)
    client = _client(app)

    assert "appbar__errors" not in client.get("/").text

    client.post(path)
    healthy_page = client.get("/").text
    assert "appbar__errors" in healthy_page
    assert "1 error" in healthy_page
    assert "1 errors" not in healthy_page

    client.post(path)
    assert "2 errors" in client.get("/").text


def test_the_action_name_is_the_operator_s_word_not_the_route(monkeypatch):
    """`POST /config/apply` reads **Apply configuration** -- the words on the
    button the operator pressed, not the path they never saw."""
    app = _app()

    def _raise(_request):
        raise RuntimeError(BOOM)

    monkeypatch.setattr(config_routes, "_owner_or_conflict", _raise)

    # The htmx fragment is the banner, so this asserts against the operator's
    # sentence alone -- the route does appear further down the full page, in the
    # collapsed detail addressed to the engineer, where it belongs.
    banner = _client(app).post("/config/apply", headers={"HX-Request": "true"}).text

    assert "Apply configuration failed" in banner
    assert "/config/apply" not in banner, "the operator's sentence quotes the route"


def test_every_mutating_route_has_a_word_in_ACTIONS():
    """Drift is a test failure, not a copy regression.

    A route added without a word here still renders -- it falls back to the page
    label -- but "Configuration failed" where "Apply configuration failed" was
    meant is the kind of wrong that nobody notices until an operator reads it
    out.
    """
    app = _app()
    missing = sorted(
        f"{method} {route.path}"
        for route in app.routes
        for method in (getattr(route, "methods", None) or ())
        if method == "POST" and f"{method} {route.path}" not in crash.ACTIONS
    )
    assert not missing, "mutating routes with no operator word:\n  " + "\n  ".join(
        missing
    )


def test_the_downloadable_report_names_the_station_and_the_action(monkeypatch):
    """An operator pastes this into a bug report, exactly as they do with
    `/calibration/jobs/{id}/log`."""
    monkeypatch.setenv("UMI_STATION_ID", "gear-umi-42")
    app = _app()
    path = _boom_route(app)
    client = _client(app)

    error_id = client.post(path).headers["X-Station-Error-Id"]
    report = client.get(f"/log/errors/{error_id}.txt")

    assert report.status_code == 200
    assert "text/plain" in report.headers["content-type"]
    body = report.text
    assert f"UMI station error {error_id}" in body
    assert "gear-umi-42" in body
    assert "RuntimeError" in body
    assert "Traceback (most recent call last)" in body


def test_an_unknown_error_id_is_404_with_a_sentence_that_explains_why():
    """`no error` on its own reads as "you typed it wrong". The reason it is
    gone is usually that the console restarted."""
    response = _client(_app()).get("/log/errors/a3f19c227b04.txt")

    assert response.status_code == 404
    assert "a3f19c227b04" in response.text
    assert "until the console restarts" in response.text


def test_a_crash_in_a_job_supervisor_thread_is_recorded_not_swallowed(fresh_logging):
    """`jobs._supervise` catches nothing and runs in a daemon thread.

    An exception there goes to stderr only: the `Job` never reaches a terminal
    state, no `event: end` is emitted, and every watching EventSource hangs on
    keepalives while the page still says "Recording".

    `configure` is re-run inside the test on purpose -- pytest's own
    `threadexception` plugin swaps `threading.excepthook` for the duration of
    each test, so asserting the hook without reinstalling it would assert
    pytest's hook, not the console's.
    """
    logging_setup.configure(Settings())
    assert threading.excepthook is logging_setup.thread_excepthook

    def _boom():
        raise RuntimeError(BOOM)

    worker = threading.Thread(target=_boom, name="job-abc123def456", daemon=True)
    worker.start()
    worker.join(timeout=5.0)

    app = _app()
    records = crash.recent(app)
    assert records, "a thread died and the console recorded nothing"
    assert records[0].source == "thread"
    assert records[0].exc_type == "RuntimeError"
    assert BOOM in records[0].traceback

    assert records[0].id in _client(app).get("/log").text


def test_the_crash_banner_needs_no_javascript():
    """With htmx and crash.js blocked, a plain form POST that raises is a
    normal navigation, so the whole crash page has to be the answer."""
    app = _app()
    path = _boom_route(app)

    # No HX-Request header: this is what a <form method="post"> sends.
    response = _client(app).post(path)

    assert response.status_code == 500
    assert "<html" in response.text.lower()
    assert "this station needs an engineer" in response.text


@pytest.fixture
def fresh_logging():
    """Detach the process-wide handler around a test that reconfigures it."""
    logging_setup.reset()
    yield
    logging_setup.reset()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The station log: a file that outlives the terminal, and what `/log` shows.

Modules log through `logging.getLogger(__name__)`. Without a file handler,
every record they emit reaches only uvicorn's default stderr, so `jobs.py`'s
"cancelling", `rig.py`'s "claimed" and "released", and every traceback would be
lost with the terminal window.

Three sections on `/log`, all shown to everyone:
  1. the crash list
  2. the traceback, in a <details>
  3. `station.log` itself -- but never the path it lives at
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
import re

import pytest
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.testclient import TestClient

from station_ui import log_tail, logging_setup, rig
from station_ui.app import create_app
from station_ui.routes import log as log_routes
from station_ui.settings import Settings

REPO = pathlib.Path(__file__).resolve().parents[2]
STATIC = REPO / "station_ui" / "static"
TEMPLATES = REPO / "station_ui" / "templates"

# The heading of section 3, and the only string these tests use to decide
# whether an operator was shown the log file at all.
LOG_FILE_SECTION = "Station log"


def _app(**kwargs):
    return create_app(Settings(**kwargs))


def _client(app):
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def fresh_logging():
    """Detach the process-wide handler around a test that reconfigures it."""
    logging_setup.reset()
    yield
    logging_setup.reset()


# ------------------------------------------------------------------- capture


def test_logging_is_configured_exactly_once_per_process():
    """The suite builds hundreds of apps. Without the named-handler check, one
    root logger accumulates hundreds of file handlers -- hundreds of open
    descriptors, and every record written hundreds of times."""
    apps = [_app() for _ in range(50)]

    handlers = [
        handler
        for handler in logging.getLogger().handlers
        if getattr(handler, "name", None) == logging_setup.HANDLER_NAME
    ]

    assert len(handlers) == 1, f"{len(handlers)} station log handlers are attached"
    assert len({str(app.state.log_file) for app in apps}) == 1


def test_the_suite_never_writes_into_a_real_home_directory(station_log_dir):
    """A test that appends to `~/.umi/station_ui/station.log` is a bug: it
    fills a developer's real log with synthetic operator names and tracebacks,
    and the rotation it triggers deletes their real ones."""
    app = _app()

    resolved = pathlib.Path(app.state.log_file).resolve()

    assert resolved.is_relative_to(pathlib.Path(station_log_dir).resolve()), resolved
    assert not resolved.is_relative_to(pathlib.Path.home() / ".umi"), resolved


def test_info_records_survive_now_that_a_handler_exists():
    """`rig.py`'s claim line must reach the file. Breadcrumbs like it,
    immediately before a traceback, are usually the whole diagnosis."""
    app = _app()

    rig.claim(rig.RigState(), "0123456789abcdef0123456789abcdef")

    assert "rig claimed" in pathlib.Path(app.state.log_file).read_text()


def test_a_rig_token_never_reaches_the_log_file_in_full():
    """The claim line interpolates the live `station_rig_token` cookie value.

    Printing it to a terminal that scrolls away is one thing; a durable file
    holding a valid session cookie would be a security hole of its own.
    """
    token = "deadbeef" + "0" * 24
    app = _app()

    rig.claim(rig.RigState(), token)
    rig.take_control(rig.RigState(), token)
    state = rig.RigState()
    rig.claim(state, token)
    rig.release(state, token)

    text = pathlib.Path(app.state.log_file).read_text()
    assert "deadbeef" in text, "the log does not identify the browser at all"
    assert token not in text, "the whole rig token is in the log file"


def test_the_log_file_rotates_and_keeps_five_older_files(tmp_path, fresh_logging):
    """Size-based, because a station idles for a week and then emits fifty
    thousand lines in one session."""
    assert logging_setup.MAX_BYTES == 2_000_000
    assert logging_setup.BACKUP_COUNT == 5

    # The policy above is asserted; the behaviour is exercised at a size a test
    # can reach without writing twelve megabytes.
    logging_setup.MAX_BYTES = 2_000
    try:
        path, problem = logging_setup.configure(Settings(log_dir=tmp_path))
        assert problem == ""
        logger = logging.getLogger("station_ui.rotation-probe")
        for index in range(400):
            logger.warning("line %d %s", index, "x" * 60)
    finally:
        logging_setup.MAX_BYTES = 2_000_000

    kept = sorted(p.name for p in tmp_path.iterdir())
    assert path.name in kept
    for index in range(1, 6):
        assert f"{path.name}.{index}" in kept, kept
    assert f"{path.name}.6" not in kept, "more than five older files are kept"


def test_an_unwritable_log_dir_does_not_stop_the_app_booting(tmp_path, fresh_logging):
    """A station that cannot write a log must still record data. And it has to
    say so, because what survives a restart has changed."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")

    app = create_app(Settings(log_dir=blocker / "station_ui"))

    assert app.state.log_file is None
    assert app.state.log_file_error

    page = _client(app).get("/log")
    assert page.status_code == 200
    assert "Errors are not being written to a file." in page.text
    assert "until the console restarts" in page.text


# --------------------------------------------------------------- who sees what


def test_an_operator_sees_the_log_itself_but_never_where_it_lives():
    """What is left out is the path, not the log.

    The path is the one thing here an operator cannot act on, and a filesystem
    layout is not a diagnostic.
    """
    app = _app()

    @app.post("/test-only/boom")
    async def _boom():  # pragma: no cover - never returns
        raise RuntimeError("synthetic failure, raised by a test")

    logging.getLogger("station_ui.operator-probe").warning("STATION-LOG-MARKER")

    client = _client(app)
    client.post("/test-only/boom")
    body = client.get("/log").text

    assert "Traceback" in body
    assert "RuntimeError" in body
    # The log itself, rendered server-side for every viewer.
    assert LOG_FILE_SECTION in body
    assert "STATION-LOG-MARKER" in body, "the operator's page renders no tail"
    # The path is absent from the response bytes, not hidden with CSS.
    assert str(app.state.log_file) not in body


def test_the_log_file_can_be_read_downloaded_and_streamed():
    """All three of section 3: the tail on the page, the whole file as a
    download, and a stream that opens.

    The stream is driven through the endpoint rather than `TestClient`, and not
    for convenience: `TestClient` runs the ASGI app to completion and buffers
    the body before it returns a response, so a tail with no last line hangs it
    forever. What the frames themselves do is
    `test_the_tail_stream_emits_no_end_event_and_survives_a_rotation`.
    """
    app = _app()
    logging.getLogger("station_ui.debug-probe").warning("STATION-LOG-MARKER")
    client = _client(app)

    page = client.get("/log").text
    assert LOG_FILE_SECTION in page
    assert "STATION-LOG-MARKER" in page, "the page renders no tail server-side"

    whole = client.get("/log/file.txt")
    assert whole.status_code == 200
    assert "STATION-LOG-MARKER" in whole.text

    response = asyncio.run(log_routes.log_stream(_scope_request(app)))
    assert isinstance(response, StreamingResponse)
    assert response.media_type == "text/event-stream"
    # Resolved eagerly: an unopenable file is a 404 before `200 OK` is on the
    # wire, the same shape as every other stream on this console.
    assert response.headers["cache-control"] == "no-store"
    asyncio.run(response.body_iterator.aclose())


def _scope_request(app) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/log/file/stream",
            "query_string": b"",
            "headers": [],
            "app": app,
        }
    )


def test_the_log_file_routes_serve_an_operator_rather_than_refusing():
    """The download and the stream follow the page.

    A page that renders a log but cannot refresh or download it is half a
    diagnostic. Both are the same bytes the page already shows.
    """
    assert _client(_app()).get("/log/file.txt").status_code == 200

    # The stream is driven through the endpoint, never `TestClient`: the client
    # runs the app to completion and buffers the body, so a tail with no last
    # line would hang it forever.
    response = asyncio.run(log_routes.log_stream(_scope_request(_app())))
    assert isinstance(response, StreamingResponse)
    assert response.media_type == "text/event-stream"
    asyncio.run(response.body_iterator.aclose())


# ----------------------------------------------------------------- the stream


def test_the_tail_stream_emits_no_end_event_and_survives_a_rotation(tmp_path):
    """A log file has no last line, so there is no `end` event to send -- the
    single divergence from `jobs._frames`, and the reason the tail gets its own
    script instead of reusing audit.js, which reloads the page on `end`."""
    path = tmp_path / "station.log"
    path.write_text("one\ntwo\n")

    stream = log_tail.frames(path, lines=10, keepalive=5.0, poll=0.01)
    try:
        assert next(stream) == "data: one\n\n"
        assert next(stream) == "data: two\n\n"

        # Exactly what RotatingFileHandler does: rename, then start a new file.
        path.rename(tmp_path / "station.log.1")
        path.write_text("three\n")

        rotated = next(stream)
        assert "[log] file rotated" in rotated
        assert next(stream) == "data: three\n\n"
        assert "event: end" not in rotated
    finally:
        stream.close()


def test_the_log_page_carries_a_refresh_link_and_no_polling_attribute():
    """The crash list deliberately has no stream: crashes arrive by push -- the
    banner appears where the operator is standing and the appbar count follows
    them -- so a self-repolling list would be a second, weaker channel that has
    to be watched."""
    body = _client(_app()).get("/log").text

    assert re.search(r'<a[^>]*href="/log"[^>]*>\s*Refresh', body), (
        "the log page has no plain Refresh link"
    )
    assert "hx-trigger" not in body
    assert "every 2s" not in body


# ----------------------------------------------------------------- base.html


def test_the_shell_still_reports_the_recorded_htmx_sha():
    """base.html carries a crash slot, an error count and a script tag. The
    htmx digest comment sits between two of them and must survive intact."""
    import hashlib

    digest = hashlib.sha256((STATIC / "htmx.min.js").read_bytes()).hexdigest()

    assert digest in (TEMPLATES / "base.html").read_text()


def test_the_appbar_identity_no_longer_labels_its_own_values(monkeypatch):
    """The identity strip shows the station id bare: a `station:` label spends
    a lowercase word on a value that identifies itself."""
    monkeypatch.setenv("UMI_STATION_ID", "gear-umi-04")
    app = _app()
    body = TestClient(app).get("/").text
    strip = re.search(
        r'<span class="appbar__identity">(.*?)</span>\s*<span', body, re.S
    )

    assert strip, "no identity strip in the app chrome"
    assert "station:" not in strip.group(1)
    # The value itself stays -- several stations share a bench.
    assert "gear-umi-04" in strip.group(1)


def test_the_shell_has_no_placeholder_content_block():
    """Every page overrides `content` and every nav item is enabled, so the
    default could only appear if a future page forgot the block -- in which
    case a build-status message is the worst thing to show an operator."""
    source = (TEMPLATES / "base.html").read_text()

    assert "Web shell is running" not in source
    assert "not wired up yet" not in source
    assert "Not available yet" not in source, "an unreachable tooltip is still copy"

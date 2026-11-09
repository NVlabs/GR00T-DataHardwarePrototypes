# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import pathlib
import subprocess

from fastapi.testclient import TestClient

from station_ui.app import create_app

REPO = pathlib.Path(__file__).resolve().parents[2]
STATIC = REPO / "station_ui" / "static"
TEMPLATES = REPO / "station_ui" / "templates"


def test_app_boots_and_serves_a_page():
    client = TestClient(create_app())
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_htmx_is_vendored_and_no_template_reaches_the_network():
    vendored = STATIC / "htmx.min.js"
    assert vendored.is_file() and vendored.stat().st_size > 10_000
    for tpl in TEMPLATES.rglob("*.html"):
        text = tpl.read_text()
        for needle in (
            "cdn.",
            "unpkg",
            "//fonts.",
            "googleapis",
            "http://",
            "https://",
        ):
            assert needle not in text, f"{tpl.name} reaches the network via {needle!r}"


def test_the_recorded_sha_matches_the_vendored_file():
    """base.html records htmx's digest so an operator can verify it from
    view-source. A bump that forgets the comment leaves the digest certifying a
    file that is not there, which is worse than recording no digest at all."""
    digest = hashlib.sha256((STATIC / "htmx.min.js").read_bytes()).hexdigest()
    assert digest in (TEMPLATES / "base.html").read_text(), (
        f"base.html does not record the vendored file's real sha256 {digest}"
    )


def test_the_static_mount_serves_the_vendored_assets():
    """The traversal test only asserts a non-200, so a mount pointing at the
    wrong directory would pass it while shipping an unstyled, script-less page."""
    client = TestClient(create_app())
    for name in ("app.css", "htmx.min.js"):
        resp = client.get(f"/static/{name}")
        assert resp.status_code == 200, f"/static/{name} -> {resp.status_code}"
        assert resp.content == (STATIC / name).read_bytes()


def test_the_details_persistence_script_is_loaded_and_served():
    """The job-log <details> panels stay open across polled swaps only if this
    enhancement loads. A wrong path fails silently in the browser -- so the boot
    suite checks it is both referenced by the shell and actually served."""
    client = TestClient(create_app())
    assert "/static/details_state.js" in client.get("/").text, (
        "base.html does not load the details-state enhancement"
    )
    resp = client.get("/static/details_state.js")
    assert resp.status_code == 200
    assert resp.content == (STATIC / "details_state.js").read_bytes()


def test_static_files_cannot_escape_the_static_dir():
    """Sent over a raw socket, not httpx: httpx normalises `../..` client-side,
    so the obvious version of this test passes against a vulnerable handler.
    Verified against a deliberately vulnerable hand-rolled handler, which
    answers this exact byte sequence with `HTTP/1.1 200 OK` and pyproject.toml.
    The status-line assertion is the load-bearing one — FileResponse streams, so
    the first recv() can be headers only and `[project]` may not have arrived."""
    import socket
    import threading
    import time

    import uvicorn

    # Use port 0 to avoid collisions between parallel tests or unrelated processes.
    config = uvicorn.Config(create_app(), host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        assert server.started, "test server did not start"
        port = server.servers[0].sockets[0].getsockname()[1]
        raw = b"GET /static/../../pyproject.toml HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"
        sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        sock.sendall(raw)
        response = sock.recv(4096)
        sock.close()
        assert b"200 OK" not in response.split(b"\r\n")[0], (
            f"path traversal served a file: {response[:200]!r}"
        )
        assert b"[project]" not in response
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_web_deps_are_declared_in_project_dependencies_not_only_a_group():
    """Parse the TOML — do not string-split. Declaring jinja2 in the `dev`
    group satisfies a naive substring check while breaking station installs."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    declared = " ".join(data["project"]["dependencies"]).lower()
    for package in (
        "jinja2",
        "python-multipart",
        "fastapi",
        "uvicorn",
        "ruamel.yaml",
    ):
        assert package in declared, f"{package} missing from [project.dependencies]"


def test_the_station_install_group_carries_the_web_deps():
    """`--only-group ... --no-dev` is load-bearing. Plain
    `uv export --group data_collection` also pulls the default `dev` group
    (which declares fastapi and uvicorn) plus transitive jinja2 via
    jupyterlab/nbconvert — so 4 of 5 packages would appear present even if the
    data_collection group declared none of them."""
    out = subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--only-group",
            "data_collection",
            "--no-dev",
            "--no-hashes",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    ).stdout.lower()
    # `uv export` emits PEP 503 normalised names, so ruamel.yaml appears as
    # ruamel-yaml. The declaration test above checks the declared spelling; this
    # one has to check the exported spelling, and they genuinely differ.
    for package in (
        "jinja2",
        "python-multipart",
        "fastapi",
        "uvicorn",
        "ruamel-yaml",
    ):
        assert f"\n{package}==" in out, f"{package} absent from the station install"


def test_default_bind_is_loopback_and_non_loopback_warns(capsys, monkeypatch):
    """Behavioural, not a source grep. Asserting `"127.0.0.1" in __main__.py`
    would be vacuous: the literal also appears in help text and comments, so
    `DEFAULT_HOST = "0.0.0.0"` or a hardcoded `default="0.0.0.0"` would keep the
    suite green while `main([])` bound every interface. This app writes configs
    and starts hardware with no authentication."""
    import station_ui.__main__ as main_mod

    calls = []
    monkeypatch.setattr(main_mod.uvicorn, "run", lambda app, **kw: calls.append(kw))
    # --open is on by default; keep the boot test from spawning a real browser
    # a second later.
    monkeypatch.setattr(main_mod.webbrowser, "open_new_tab", lambda *a, **k: None)

    main_mod.main([])
    assert calls[-1]["host"] == "127.0.0.1"
    assert calls[-1]["port"] == 8770
    capsys.readouterr()

    main_mod.main(["--host", "0.0.0.0"])
    assert calls[-1]["host"] == "0.0.0.0"
    assert "WARNING" in capsys.readouterr().out


def test_the_browser_opens_by_default_and_no_open_suppresses_it(monkeypatch):
    """Launching the console opens the page in a browser by default; `--no-open`
    is the escape for a headless or remote start."""
    import station_ui.__main__ as main_mod

    monkeypatch.setattr(main_mod.uvicorn, "run", lambda app, **kw: None)

    timers = []

    class _FakeTimer:
        def __init__(self, delay, fn, args=()):
            timers.append((delay, fn, args))

        def start(self):
            pass

    monkeypatch.setattr(main_mod.threading, "Timer", _FakeTimer)

    main_mod.main([])
    assert timers, "the browser did not open by default"

    timers.clear()
    main_mod.main(["--no-open"])
    assert not timers, "--no-open still scheduled a browser open"


def test_no_cors_headers_are_emitted():
    """Behavioural, not a source grep: `origins = ["*"]` then
    `allow_origins=origins` defeats a grep while really answering
    `access-control-allow-origin: *`, and a grep of app.py cannot see CORS
    added from any other module."""
    resp = TestClient(create_app()).get("/", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in {k.lower() for k in resp.headers}
    assert "access-control-allow-credentials" not in {k.lower() for k in resp.headers}

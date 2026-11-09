# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A refused Start has to reach the operator, not just the server log.

`_respond` renders the reason into the record-status fragment for an htmx
request, and `preflight` has a sentence for every refusal -- but that only
helps if the browser puts that fragment on the page. htmx 2.0.10 ships

    responseHandling: [{code:"204",swap:false},
                       {code:"[23]..",swap:true},
                       {code:"[45]..",swap:false,error:true}]

so a 4xx is not swapped at all: the server would render a red "Recording did
not start" banner and the browser would drop it, leaving a page that does not
visibly react. This file pins the fix: the server marks the refusal as a
fragment, and `static/crash.js` swaps it.

Two kinds of test here, and the split is deliberate. What the server sends is
asserted end to end. What the browser does with it is JavaScript this suite
does not execute, so what is asserted is the contract the script depends on --
the same convention `test_device_dropdown.py` uses for `combobox.js`.
"""

from __future__ import annotations

import pathlib
import re

import pytest

yaml = pytest.importorskip("yaml")

REPO = pathlib.Path(__file__).resolve().parents[2]

import test_record_page as record_page  # noqa: E402

from station_ui.routes import record as record_route  # noqa: E402

STATIC = REPO / "station_ui" / "static"

# Re-exported so this file's tests can use test_record_page's fixtures.
repo = record_page.repo
client = record_page.client
launcher = record_page.launcher
ready = record_page.ready
_gate_open = record_page._gate_open


def _start(client, **data):
    """Start the way the browser does: as an htmx request."""
    payload = {"confirmed": "yes"}
    payload.update(data)
    return client.post(
        "/record/start",
        data=payload,
        headers={"HX-Request": "true"},
        follow_redirects=False,
    )


# ---------------------------------------------------- what the server sends


def test_a_refused_start_is_a_fragment_the_client_is_told_it_may_swap(
    ready, launcher, repo
):
    """The header is the fix. Without it htmx has no way to tell this apart
    from the whole-page 4xx some /config routes answer with, and swapping a
    document into the status card is worse than the silence."""
    record_page._write_recording_config(repo, task_name="")

    response = _start(ready)

    assert response.status_code == 400
    assert response.headers.get(record_route.FRAGMENT_HEADER) == "1"
    assert not launcher.calls


def test_the_refusal_body_really_is_a_fragment_and_not_a_whole_page(ready, repo):
    """The header would be a lie otherwise, and a lie the client acts on: it
    swaps this straight into `#record-status`."""
    record_page._write_recording_config(repo, task_name="")

    body = _start(ready).text

    assert "<!DOCTYPE" not in body.upper()
    assert "<html" not in body.lower()
    assert body.lstrip().startswith('<div class="section" id="record-status"')


def test_the_reason_is_in_the_body_the_client_swaps(ready, repo):
    """A swappable fragment that does not say why is the same dead end."""
    record_page._write_recording_config(repo, task_name="")

    body = _start(ready).text

    assert "Recording did not start" in body
    assert "Set a task name on Configuration" in body


def test_a_successful_start_still_redirects_a_plain_browser(ready, launcher):
    """The header must not change the non-htmx path: with the vendored script
    blocked, Start is an ordinary form post and a 303 back to /record."""
    response = ready.post(
        "/record/start", data={"confirmed": "yes"}, follow_redirects=False
    )

    assert response.status_code == 303
    assert launcher.calls


# --------------------------------------------------- what the browser does


def test_the_vendored_htmx_still_refuses_to_swap_a_4xx_on_its_own():
    """The fact the whole fix rests on. If an htmx upgrade ever changes this
    default, this fails here -- rather than the handler quietly becoming dead
    code that nobody removes."""
    htmx = (STATIC / "htmx.min.js").read_text()

    # Not a non-greedy `\[(.*?)\]`: the array's own entries contain `]`
    # (`{code:"[23].."`), so that stops in the middle of the first one.
    assert "responseHandling:" in htmx, "htmx no longer declares responseHandling"
    assert '{code:"[45]..",swap:false' in htmx, (
        "the vendored htmx now swaps 4xx by itself — crash.js may be dead code"
    )


def test_crash_js_swaps_a_refusal_the_server_marked_as_a_fragment():
    """`htmx:beforeSwap` with `shouldSwap` is the documented htmx escape hatch,
    and the only one that runs before the response is discarded."""
    script = (STATIC / "crash.js").read_text()

    assert "htmx:beforeSwap" in script
    assert "shouldSwap" in script
    assert record_route.FRAGMENT_HEADER in script, (
        "the client does not read the header the server sends"
    )


def test_crash_js_does_not_swap_a_4xx_that_carries_no_header():
    """The hazard the narrow 500 handler was protecting against: some /config
    routes answer 4xx with a whole document, and htmx would put it inside a
    card. The header, not the status code, is the gate."""
    script = (STATIC / "crash.js").read_text()

    assert re.search(
        r"if\s*\(!\s*xhr\.getResponseHeader\(\"X-Station-Fragment\"\)\)\s*return",
        script,
    ), script


def test_a_500_still_goes_to_the_crash_slot_rather_than_the_card():
    """A crash page is a whole document; the two paths must stay separate."""
    script = (STATIC / "crash.js").read_text()

    assert "htmx:responseError" in script
    assert "xhr.status !== 500" in script
    assert "crash-slot" in script

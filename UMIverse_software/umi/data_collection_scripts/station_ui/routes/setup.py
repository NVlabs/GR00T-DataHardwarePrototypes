# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`/` -- pick the setup preset.

The page is a rendering of `SETUP_PRESETS` and nothing else. That is the whole
point of the preset table: each supported setup is one immutable row rather
than a branch spread across camera and config code. So this module reads the
tuple at request time and the template renders whatever rows it is given --
adding a fifth generation costs a row and no code, which
`tests/station_ui/test_setup_page.py` holds to by rendering a fifth one.

`/take-control` lives here because `/` is the first page an operator lands on
and a read-only tab has to be able to escape from somewhere.
"""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from station_ui import recording, tokens
from station_ui.core import setup_presets
from station_ui.errors import RecordingInProgress, StaleOwner, stale_choice
from station_ui.rig import claim, is_owner, require_owner, take_control

# Read the preset table through its module for each request.
from station_ui.workflow import render_page

router = APIRouter()


# See Other, not 302: the browser must follow a POST with a GET, or a reload of
# the page it lands on would re-post the selection.
SEE_OTHER = 303
# Return 409 for ownership conflicts; credentials are valid, but another
# browser currently controls the rig.
CONFLICT = 409
BAD_REQUEST = 400


def _page(
    request: Request, token: str | None, *, status_code: int = 200, error: str = ""
) -> HTMLResponse:
    return render_page(
        request,
        "setup",
        _page_context(request, token, status_code=status_code, error=error),
        token=token,
        status_code=status_code,
    )


def _page_context(
    request: Request, token: str | None, *, status_code: int = 200, error: str = ""
) -> dict:
    rig = request.app.state.rig
    return {
        "page_title": "Setup",
        "active": "/",
        "presets": setup_presets.SETUP_PRESETS,
        "selected": rig.draft.preset_id,
        "can_edit": is_owner(rig, token),
        "error": error,
    }


@router.get("/", response_class=HTMLResponse)
async def setup_page(request: Request) -> HTMLResponse:
    """Render the cards, and claim the rig if it is going spare.

    Claiming on the first *load* rather than on the first *change* is what makes
    the read-only banner truthful: a second browser is told it cannot edit
    before it clicks, instead of finding out by losing a click.
    """
    rig = request.app.state.rig
    token = tokens.token_from(request) or tokens.new_token()
    claim(rig, token)
    # Shown once, on the redirect that dropped the selection. Kept on app.state
    # rather than in the URL for the same reason `config_warnings` is: a notice
    # in a query string can be forged or replayed by sharing the link.
    notice = getattr(request.app.state, "setup_notice", "")
    request.app.state.setup_notice = ""
    response = _page(request, token, error=notice)
    tokens.remember(response, token)
    return response


@router.post("/setup/preset")
async def choose_preset(request: Request, preset_id: str = Form(...)):
    rig = request.app.state.rig
    token = tokens.token_from(request)
    try:
        require_owner(rig, token)
    except StaleOwner as exc:
        return _page(request, token, status_code=CONFLICT, error=str(exc))

    # Prevent preset changes while recording. Keep Take control available so
    # an operator can acquire ownership and stop the recorder.
    try:
        recording.require_idle(request.app)
    except RecordingInProgress as exc:
        return _page(request, token, status_code=CONFLICT, error=str(exc))

    try:
        preset = setup_presets.get_preset(preset_id)
    except KeyError:
        # A hand-typed or stale id is an operator-visible 400, never the 500 a
        # bare KeyError out of a route would produce.
        return _page(
            request,
            token,
            status_code=BAD_REQUEST,
            error=stale_choice("setup"),
        )

    if rig.draft.preset_id != preset.id:
        # A changed preset invalidates device slots and gripper-generation choices.
        # Selecting the same preset is idempotent and preserves the draft.
        rig.draft.preset_id = preset.id
        rig.draft.grippers.clear()
        rig.draft.required_cameras.clear()
    # Redirect back to Setup to separate selection from confirmation and
    # prevent reload from reposting the selection.
    return RedirectResponse("/", status_code=SEE_OTHER)


@router.post("/take-control")
async def take_the_rig(request: Request):
    """Displace the current owner, deliberately. The only way control moves to
    a browser that did not claim it first."""
    rig = request.app.state.rig
    token = tokens.token_from(request) or tokens.new_token()
    take_control(rig, token)
    response = RedirectResponse("/", status_code=SEE_OTHER)
    tokens.remember(response, token)
    return response

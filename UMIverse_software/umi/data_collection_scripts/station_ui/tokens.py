# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The rig token, and the cookie that carries it.

`station_ui/rig.py` decides who may change the rig; this module is the only
place that knows how a browser proves which token it holds. Kept out of
`rig.py` so the ownership rules stay testable without an HTTP request, and out
of the routes so `/config` and the rest do not each invent their own cookie.

Deliberately a *session* cookie with no expiry: it lasts as long as the browser
is open, and a browser that has been restarted has to take control explicitly
rather than silently inheriting a rig it may no longer be sitting in front of.
A cookie is per-browser, not per-tab, so two tabs of the same browser share one
token and are both the owner. That is the right granularity, because those two
tabs are one operator.
"""

from __future__ import annotations

import uuid

__all__ = ["COOKIE", "new_token", "remember", "token_from"]

COOKIE = "station_rig_token"


def new_token() -> str:
    return uuid.uuid4().hex


def token_from(request) -> str | None:
    """The token this browser presents, or None if it has never loaded a page.

    Empty string maps to None: `rig.claim` rejects a blank token outright, and
    a route must be able to tell "no cookie" from "a cookie worth checking".
    """
    return request.cookies.get(COOKIE) or None


def remember(response, token: str) -> None:
    """Send the token back. `httponly` because no script needs to read it, and
    a token readable from JS is one an injected script can use to take the rig.
    """
    response.set_cookie(COOKIE, token, httponly=True, samesite="lax", path="/")

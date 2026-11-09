# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Refuse a mutating request that another origin sent.

This app writes configs and starts hardware, and is reached only from the
station's own browser. SameSite cannot keep another page from posting to it:
cookies ignore ports, so `http://localhost:3000` is same-site with
`http://localhost:8770`, and any other local web app (a Jupyter server, say)
is a same-site origin on another localhost port.
"""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import PlainTextResponse

__all__ = ["CROSS_ORIGIN", "OriginGuard", "same_origin"]

CROSS_ORIGIN = "This request did not come from this station's console."


class OriginGuard:
    """Answer 403 to a POST another origin sent; pass everything else through.

    A bare ASGI callable, deliberately not `BaseHTTPMiddleware`: that class
    relays every response through an anyio memory object stream, and this app's
    riskiest surface is four live SSE logs. It does one thing and touches
    nothing else.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope)
            if request.method not in ("GET", "HEAD", "OPTIONS") and not same_origin(
                request
            ):
                response = PlainTextResponse(CROSS_ORIGIN, status_code=403)
                return await response(scope, receive, send)
        await self.app(scope, receive, send)


def same_origin(request) -> bool:
    """Whether a request came from a page this server rendered.

    `Origin` is the only header that carries the port. Absent Origin and
    Referer means a non-browser client (curl, TestClient), which has no ambient
    cookie to abuse.
    """
    expected = f"{request.url.scheme}://{request.url.netloc}"
    origin = request.headers.get("origin")
    if origin:
        return origin == expected
    referer = request.headers.get("referer")
    if referer:
        return referer == expected or referer.startswith(expected + "/")
    return True

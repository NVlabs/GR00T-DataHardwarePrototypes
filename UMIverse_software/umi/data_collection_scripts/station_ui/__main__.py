# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the station web UI: `python -m station_ui`."""

from __future__ import annotations

import argparse
import threading
import webbrowser

import uvicorn

from station_ui.app import create_app
from station_ui.settings import DEFAULT_HOST, DEFAULT_PORT, Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m station_ui",
        description="Serve the UMI station data-collection UI on this station.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="interface to bind (default: 127.0.0.1, i.e. this station only)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP port")
    parser.add_argument(
        "--open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="open the console in a browser once the server is up "
        "(default: on; pass --no-open to suppress it)",
    )
    args = parser.parse_args(argv)

    settings = Settings(host=args.host, port=args.port)
    if not settings.is_loopback:
        # This server can change configs and start hardware without authentication.
        # Bind to loopback unless access from another machine is required.
        print(
            f"WARNING: binding {settings.host}:{settings.port} exposes config writes "
            "and hardware control to the network. There is no authentication. "
            "Bind the loopback default unless you know the network is trusted."
        )

    url = f"http://{settings.host}:{settings.port}/"
    print(f"UMI Station: {url}")
    if args.open:
        # Delay browser opening until the server can bind; use a daemon thread so
        # this delay does not hold up shutdown.
        timer = threading.Timer(1.0, webbrowser.open_new_tab, args=(url,))
        timer.daemon = True
        timer.start()

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        # Bound graceful shutdown because recorder SSE streams may remain open
        # indefinitely.
        timeout_graceful_shutdown=5,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

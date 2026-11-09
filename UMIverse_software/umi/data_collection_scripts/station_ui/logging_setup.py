# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Give the console a log file, once per process.

Without this, every module's `logging.getLogger(__name__)` records reach only
uvicorn's default stderr, which on a station launched from a terminal that has
since been closed is nowhere at all: `jobs.py` cancelling a child, `rig.py`
claiming and releasing the rig, and every traceback would be lost with the
window.

Two verified mechanics this hangs on:

* Under `uvicorn.run`, the `uvicorn` logger has `propagate: False`, so a root
  handler receives `station_ui.*` records but not `uvicorn.error`'s ASGI
  traceback. Attaching this handler is therefore not sufficient on its own:
  `app._unhandled` logs the traceback itself, and `test_the_traceback_reaches_
  the_station_log_file` is what keeps that true.
* uvicorn's own `dictConfig` runs with `disable_existing_loggers: False`, so a
  handler attached inside `create_app` (which runs first) survives it.

Called from `create_app`, not `__main__.py`: `__main__` is one of two entry
points and `TestClient` never touches it, so putting it in the factory is the
only way a test can pin the same crash path uvicorn takes.
"""

from __future__ import annotations

import datetime
import logging
import logging.handlers
import os
import pathlib
import socket
import threading

LOG_DIR_ENV = "UMI_STATION_LOG_DIR"
LOG_LEVEL_ENV = "UMI_STATION_LOG_LEVEL"

# Name the handler so repeated app construction reuses it instead of
# adding duplicate file descriptors and repeated writes.
HANDLER_NAME = "station_ui.file"

FORMAT = "%(asctime)s %(levelname)-8s %(name)-24s %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S%z"

FILENAME = "station.log"
# Rotate at 12 MB; size-based rotation handles long idle periods followed
# by large recording logs.
MAX_BYTES = 2_000_000
BACKUP_COUNT = 5

# Scope the in-memory crash summary to this process start; it does not
# claim error-free history across server restarts.
STARTED_AT = datetime.datetime.now().astimezone()

logger = logging.getLogger(__name__)


def started_at() -> datetime.datetime:
    return STARTED_AT


def handler() -> logging.Handler | None:
    """The station's file handler, if this process already attached one."""
    for candidate in logging.getLogger().handlers:
        if getattr(candidate, "name", None) == HANDLER_NAME:
            return candidate
    return None


def configure(settings) -> tuple[pathlib.Path | None, str]:
    """Attach the file handler. Idempotent, and never fatal.

    Returns `(path, problem)`. `problem` is a sentence for the operator when
    the log could not be opened -- a station that cannot write a log must still
    record data, so the failure is reported, never raised.
    """
    existing = handler()
    if existing is not None:
        return pathlib.Path(getattr(existing, "baseFilename", "")), ""

    directory = pathlib.Path(settings.log_dir).expanduser()
    path = directory / FILENAME
    try:
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            path,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
            # Append, so a restart continues the station's history instead of
            # deleting the evidence of why it restarted.
            mode="a",
        )
    except OSError as exc:
        return None, f"{path} could not be opened: {exc}"

    file_handler.name = HANDLER_NAME
    file_handler.setFormatter(logging.Formatter(FORMAT, datefmt=DATEFMT))

    root = logging.getLogger()
    root.addHandler(file_handler)
    root.setLevel(settings.log_level)

    # Capture SDK warnings in the persistent log during recording.
    logging.captureWarnings(True)

    _install_thread_hook()

    # Use the session-log restart marker format for consistent log boundaries.
    logger.info("=" * 72)
    logger.info(
        ">>> station console start — pid %s on %s, bind %s:%s",
        os.getpid(),
        socket.gethostname(),
        settings.host,
        settings.port,
    )
    return path, ""


def reset() -> None:
    """Detach and close the handler. For tests that reconfigure it.

    Not called in production: the handler is meant to live as long as the
    process. It exists because a test that wants a different directory, or an
    unwritable one, has no other way past the once-per-process check.
    """
    existing = handler()
    if existing is not None:
        logging.getLogger().removeHandler(existing)
        existing.close()
    if threading.excepthook is thread_excepthook:
        threading.excepthook = threading.__excepthook__


def thread_excepthook(args) -> None:
    """Record an exception that killed a thread, instead of losing it.

    `jobs._supervise` catches nothing and runs in a daemon thread. Without this,
    an exception there reaches stderr and nowhere else: the `Job` never reaches
    a terminal state, no `event: end` is ever emitted, and every watching
    EventSource sits on keepalives while the page still says "Recording". This
    is what turns that silence into a line on `/log`.
    """
    if args.exc_value is None:
        return
    # Imported here rather than at module scope: `crash` reads the page labels
    # out of `station_ui.app`, which imports this module.
    from station_ui import crash

    name = args.thread.name if args.thread is not None else ""
    entry = crash.record(None, args.exc_value, thread_name=name)
    logging.getLogger("station_ui.crash").error(
        "[%s] thread %s died",
        entry.id,
        name or "?",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


def _install_thread_hook() -> None:
    threading.excepthook = thread_excepthook

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`tail -f` for `station.log`, in the SSE vocabulary `jobs` already speaks.

Deliberately the same frame shape and the same keepalive as `jobs.data_frame`
rather than a second hand-rolled one, so this console has one server-sent-event
dialect with two producers (`jobs.KEEPALIVE_FRAME`, `jobs.KEEPALIVE_S` and
`jobs.data_frame` are shared).

One divergence, and it is the reason this file exists at all: **there is no
`event: end`.** A job ends; a log file does not have a last line. That is also
why the tail gets its own small script instead of reusing `audit.js`, which
reloads the page when it sees `end`.

The generator is synchronous, so Starlette runs it in a threadpool exactly as
it already does for `jobs._frames`, and `time.sleep` here is safe. Bytes, not
text: `TextIOWrapper.tell()` returns an opaque cookie rather than an offset,
and the rotation check below compares a cursor against `st_size`.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

from station_ui import jobs

# Limit initial log history so tracebacks are available without rendering
# an unbounded backlog.
TAIL_LINES = 200
KEEPALIVE_S = jobs.KEEPALIVE_S
POLL_S = 0.5

# `[log] …` mirrors `jobs.py`'s `[job] …`: a line the server inserted, never a
# line the log itself contained.
ROTATED = "[log] file rotated"


def tail(path, lines: int = TAIL_LINES) -> list[str]:
    """The last `lines` lines of the file, for the server-rendered view.

    Whole-file read: the handler's rotation caps one file at 2 MB, so seeking
    backwards would be cleverness bought with a bug surface.
    """
    with open(path, "rb") as handle:
        return handle.read().decode("utf-8", "replace").splitlines()[-lines:]


def frames(
    path, *, lines: int = TAIL_LINES, keepalive=None, poll=None
) -> Iterator[str]:
    """SSE frames: the tail, then everything appended after it.

    Not itself a generator function, for the reason `jobs.stream` is not one
    either: a generator would defer the `open` to the first `next()`, by which
    point `StreamingResponse` has already put `200 OK` on the wire and an
    unreadable file could no longer be answered with a 404.
    """
    handle = open(path, "rb")
    keepalive = KEEPALIVE_S if keepalive is None else keepalive
    poll = POLL_S if poll is None else poll
    return _frames(handle, os.fspath(path), lines, keepalive, poll)


def _frames(handle, path: str, lines: int, keepalive: float, poll: float):
    try:
        history = handle.read().decode("utf-8", "replace").splitlines()
        for line in history[-lines:]:
            yield jobs.data_frame(line)

        inode = _inode(handle)
        quiet_since = time.monotonic()
        while True:
            line = _whole_line(handle)
            if line is not None:
                quiet_since = time.monotonic()
                yield jobs.data_frame(line)
                continue

            reopened = _reopen_if_rotated(handle, path, inode)
            if reopened is not None:
                handle.close()
                handle = reopened
                inode = _inode(handle)
                quiet_since = time.monotonic()
                yield jobs.data_frame(ROTATED)
                continue

            if time.monotonic() - quiet_since >= keepalive:
                quiet_since = time.monotonic()
                yield jobs.KEEPALIVE_FRAME
                continue

            time.sleep(poll)
    finally:
        handle.close()


def _whole_line(handle) -> str | None:
    """One complete line, or None if there is not one yet.

    A reader can arrive between a writer's `write` and its newline. Rewinding
    over a partial line costs one seek and keeps a traceback's last frame from
    being delivered split across two SSE events.
    """
    start = handle.tell()
    raw = handle.readline()
    if not raw.endswith(b"\n"):
        handle.seek(start)
        return None
    return raw.decode("utf-8", "replace").rstrip("\n")


def _inode(handle) -> int:
    try:
        return os.fstat(handle.fileno()).st_ino
    except OSError:
        return -1


def _reopen_if_rotated(handle, path: str, inode: int):
    """A fresh handle when the file under the reader was replaced or truncated.

    `RotatingFileHandler` renames `station.log` to `station.log.1` and opens a
    new one, so the descriptor a reader holds stays valid and stays empty
    forever. Two signals, because either alone misses a case: a new inode is
    the rename, a size below the cursor is a truncation in place.
    """
    try:
        stat = os.stat(path)
    except OSError:
        return None
    if stat.st_ino == inode and stat.st_size >= handle.tell():
        return None
    try:
        return open(path, "rb")
    except OSError:
        return None

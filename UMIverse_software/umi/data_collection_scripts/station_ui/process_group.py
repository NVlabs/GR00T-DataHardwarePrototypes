# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which processes are still alive in a process group.

`jobs.start` spawns with `start_new_session=True`, so a child is its own
process-group leader and `pgid == pid`. Every descendant inherits that pgid,
including the `multiprocessing` workers `BackgroundAligner` forks, so "is anything left of that job?" is exact set membership rather than a
guess.

Deliberately not `ps aux` piped through grep. A text match on a command line
also catches another station's recorder, a `tail -f` of the log, an editor with
the file open, and the grep itself; killing on one is how something unrelated
gets killed. Measured against a leader that forks one child, the pgid walk finds
the two real members and a text match finds three. It also costs a subprocess
per poll and a column format that varies by platform. `/proc` is one read, no
shell, and exact.

Linux-only, which the station is. Every read is guarded: `/proc` entries vanish
between `iterdir` and `open` as a matter of course, and that is not an error.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

__all__ = ["ProcInfo", "alive", "start_time", "survivors"]

PROC = pathlib.Path("/proc")

# Ignore zombies when waiting for group exit: they own no devices or
# file descriptors and cannot be killed again.
ZOMBIE = "Z"


@dataclass(frozen=True)
class ProcInfo:
    """One live process, with as much as the operator needs to identify it."""

    pid: int
    cmdline: str
    state: str


def _stat_fields(pid: int) -> list[str] | None:
    """`/proc/<pid>/stat` past the comm field, or None if it is gone.

    Split after the last `)`: `comm` is the executable name in parentheses and
    may itself contain spaces and parentheses, so a plain `.split()` mis-indexes
    every field after it for a process called `(my prog)`.

    The returned list is 0-based from `state`, so the documented field N is at
    index N-3: state 3 -> 0, pgrp 5 -> 2, starttime 22 -> 19.
    """
    try:
        raw = (PROC / str(pid) / "stat").read_text()
    except (OSError, ValueError):
        return None
    try:
        return raw[raw.rindex(")") + 2 :].split()
    except ValueError:
        return None


def _cmdline(pid: int) -> str:
    """The child's argv, NUL-separated on disk, for a human to recognise it by.

    Empty for a kernel thread and for a zombie, which is one more reason not to
    identify processes by this string.
    """
    try:
        raw = (PROC / str(pid) / "cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()


def alive(pid: int) -> bool:
    """Whether `pid` names a process that exists and has not been reaped."""
    fields = _stat_fields(pid)
    return fields is not None and fields[0] != ZOMBIE


def start_time(pid: int) -> int | None:
    """Field 22 of `/proc/<pid>/stat`: when this process started, in clock ticks.

    The guard against pid reuse. A pid on its own is not an identity (Linux
    recycles them), so a console that signals "the pid in the file" after a
    reboot, or after a busy enough week, can kill an unrelated process. Paired
    with the pid it is an identity.
    """
    fields = _stat_fields(pid)
    if fields is None:
        return None
    try:
        return int(fields[19])
    except (IndexError, ValueError):
        return None


def survivors(pgid: int) -> list[ProcInfo]:
    """Every live process in `pgid`, excluding zombies and this process.

    Excluding this process is not hygiene. A test asks about a group the web
    server may itself belong to, and a console that reported itself as a
    recorder to be killed would SIGKILL the server.
    """
    if not pgid or pgid <= 0:
        return []
    found = []
    me = os.getpid()
    try:
        entries = list(PROC.iterdir())
    except OSError:
        return []
    for entry in entries:
        name = entry.name
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == me:
            continue
        fields = _stat_fields(pid)
        if fields is None or fields[0] == ZOMBIE:
            continue
        try:
            if int(fields[2]) != pgid:
                continue
        except (IndexError, ValueError):
            continue
        found.append(ProcInfo(pid=pid, cmdline=_cmdline(pid), state=fields[0]))
    return found

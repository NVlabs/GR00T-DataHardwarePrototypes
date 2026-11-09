# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exclusive ownership of a GUI's recorder state, independent of its port."""

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path


class RuntimeHomeInUseError(RuntimeError):
    pass


@contextmanager
def claim_runtime_home(home):
    root = Path(home).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Keep one stable inode: unlinking after unlock lets another server lock a
    # replacement while a concurrent opener still holds the previous file.
    # Python's non-inheritable descriptor also keeps recorder/probe children
    # from retaining ownership when their GUI dies.
    with (root / "server.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeHomeInUseError(
                f"DC GUI runtime home {root} is already in use. Open the running "
                "GUI or stop it before restarting; independent GUIs need distinct "
                "--home folders. Nothing was adopted or stopped."
            ) from exc
        lock.seek(0)
        lock.truncate()
        lock.write(f"{os.getpid()}\n")
        lock.flush()
        try:
            yield str(root)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which code produced a recording, in a form a human can read.

Lives here rather than inside ``record_manager`` because nothing about it needs
the recorder: it only reads a git checkout. ``record_manager`` imports the
functions and calls them.

``commit_id`` alone pins the code but cannot say whether the tree was modified
or which branch it came from. A station may record from a checkout with local
edits, so the data has to carry that caveat itself.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from umi.common.utility.print_utils import print_yellow

_REPO_ROOT = Path(__file__).resolve().parents[3]

_TIMEOUT_S = 30


def repo_root() -> str:
    """The UMI checkout this code was imported from.

    Derived from ``__file__``, deliberately not from the process working
    directory: recording is launched several ways, and only the ``gumi``
    launcher happens to ``cd`` into the checkout first.
    """
    return str(_REPO_ROOT)


def _git(root: str, *args: str) -> str:
    # Raises only for a missing git or a hang; a failing command is just "".
    proc = subprocess.run(
        ["git", "-C", root, *args],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def code_provenance(root: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The provenance blob to stamp onto a session, or None if unavailable.

    Read straight from the checkout's git state:

    - ``version``  -- ``git describe --tags --always`` (e.g. ``v1.0-3-gb4377b2``,
      or the short sha when there are no tags), with a ``-dirty`` suffix when
      tracked files were modified
    - ``commit``   -- full HEAD sha
    - ``branch``   -- current branch, or ``"detached HEAD"``
    - ``upstream`` -- the branch it tracks (e.g. ``origin/main``), or ``"(none)"``
    - ``dirty``    -- whether tracked files differ from ``commit``

    Never raises. Provenance is metadata; failing to collect it must not abort a
    recording that is otherwise fine.
    """
    root = root or repo_root()
    try:
        commit = _git(root, "rev-parse", "HEAD")
        if not commit:
            return None
        branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
        upstream = _git(
            root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"
        )
        dirty = bool(_git(root, "status", "--porcelain", "--untracked-files=no"))
        version = _git(root, "describe", "--tags", "--always") or commit[:8]
    except (subprocess.SubprocessError, OSError) as e:
        print_yellow(f"Warning: could not collect code provenance — {e}")
        return None
    return {
        "version": f"{version}-dirty" if dirty else version,
        "commit": commit,
        "branch": "detached HEAD" if branch == "HEAD" else branch,
        "upstream": upstream or "(none)",
        "dirty": dirty,
    }


def head_commit(root: Optional[str] = None) -> str:
    """Full SHA of the checkout's HEAD, or "" if git is unavailable.

    Scoped to the checkout via ``-C``. Returns "" rather than raising so callers
    can carry on writing the rest of a session's metadata.
    """
    root = root or repo_root()
    try:
        proc = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=_TIMEOUT_S,
        )
        return proc.stdout.strip()
    except (subprocess.SubprocessError, OSError) as e:
        print_yellow(f"Warning: Failed to run git — {e}")
        return ""

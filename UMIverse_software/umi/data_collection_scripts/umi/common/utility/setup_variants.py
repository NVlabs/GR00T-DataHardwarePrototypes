# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-episode setup variants: ``{a|b|...}`` groups in ``operator_instruction``.

Groups count like an odometer over the episode id (the first steps every
episode, each later one when the previous wraps), from a per-session offset.
"""

import hashlib
import re

_GROUP_RE = re.compile(r"\{([^{}|]+(?:\|[^{}|]+)+)\}")


def has_variants(text: str) -> bool:
    return bool(_GROUP_RE.search(text or ""))


def _offset(session_uid: str, group_index: int, n: int) -> int:
    digest = hashlib.sha256(f"{session_uid}:setup:{group_index}".encode()).hexdigest()
    return int(digest[:8], 16) % n


def resolve(text: str, session_uid: str, episode_id: int) -> str:
    """``text`` with each group replaced by this episode's option."""
    episode_id = int(episode_id)
    state = {"k": 0, "stride": 1}

    def pick(m):
        options = [o.strip() for o in m.group(1).split("|")]
        n, k = len(options), state["k"]
        choice = (_offset(session_uid, k, n) + episode_id // state["stride"]) % n
        state["k"] += 1
        state["stride"] *= n
        return options[choice]

    return _GROUP_RE.sub(pick, text or "")


def strip(text: str) -> str:
    """``text`` without its groups (and the space before each)."""
    return re.sub(r"\s*" + _GROUP_RE.pattern, "", text or "")

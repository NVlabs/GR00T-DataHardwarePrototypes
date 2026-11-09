# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Episode-level dataset utilities.

Standalone helpers for reading and writing episode metadata in zarr datasets.
These are independent of the RecorderManager and can be used by any pipeline
stage (ingestion, conversion, visualization, etc.).
"""

from __future__ import annotations

from typing import Any

QC_PASS_STATUS_KEY = "qc_pass_status"


def _coerce_qc_cell(v: Any) -> int | None:
    """Normalize one qc_pass_status entry: ``None`` = pending, ``1`` pass, ``0`` fail."""
    if v is None:
        return None
    if hasattr(v, "item"):
        try:
            v = v.item()
        except Exception:
            pass
    if isinstance(v, bool):
        return None
    if isinstance(v, float):
        if v != int(v):
            return None
        v = int(v)
    if isinstance(v, int):
        if v == 1:
            return 1
        if v == 0:
            return 0
    return None


def _legacy_discarded_set(attrs: dict) -> set[int]:
    out: set[int] = set()
    raw = attrs.get("discarded_episode_indices", []) or []
    for x in raw:
        try:
            out.add(int(x))
        except (TypeError, ValueError):
            pass
    return out


def effective_episode_count_for_qc(
    attrs: dict, explicit_num_episodes: int | None
) -> int:
    """Upper bound on episode index count for qc_pass_status padding.

    Uses ``num_episodes``, ``qc_pass_status`` length, legacy discarded indices,
    and ``episode_lengths`` keys when present.
    """
    if explicit_num_episodes is not None:
        return max(0, int(explicit_num_episodes))

    n = int(attrs.get("num_episodes", 0))

    raw = attrs.get(QC_PASS_STATUS_KEY)
    if isinstance(raw, (list, tuple)):
        n = max(n, len(raw))

    legacy = _legacy_discarded_set(attrs)
    if legacy:
        n = max(n, max(legacy) + 1)

    ep_lens = attrs.get("episode_lengths", {})
    if isinstance(ep_lens, dict) and ep_lens:
        try:
            keys = [int(k) for k in ep_lens.keys()]
            n = max(n, max(keys) + 1)
        except (TypeError, ValueError):
            pass

    return n


def qc_pass_status_from_attrs(
    attrs: dict, num_episodes: int | None = None
) -> list[int | None]:
    """Build ``qc_pass_status`` (one entry per episode index) from zarr attributes.

    Values: ``None`` = not QC'd yet, ``1`` = pass, ``0`` = fail.

    Legacy ``discarded_episode_indices`` is treated as fail (``0``) where
    ``qc_pass_status`` has no explicit pass/fail yet for that index.
    """
    legacy = _legacy_discarded_set(attrs)
    n = effective_episode_count_for_qc(attrs, num_episodes)

    raw = attrs.get(QC_PASS_STATUS_KEY)
    if isinstance(raw, (list, tuple)):
        status = [_coerce_qc_cell(x) for x in raw]
    else:
        status = [None] * n

    if len(status) < n:
        status = status + [None] * (n - len(status))
    elif len(status) > n:
        status = status[:n]

    for i in legacy:
        if 0 <= i < len(status) and status[i] is None:
            status[i] = 0

    return status

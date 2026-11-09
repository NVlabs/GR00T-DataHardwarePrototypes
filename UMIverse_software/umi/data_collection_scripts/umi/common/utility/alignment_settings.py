# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How a session is aligned, read from the ``alignment`` block of its config.yaml.

The station's background aligner, ``align_data.py`` and the session health check
all read these settings here, so a re-alignment reproduces what the station
wrote. Every episode is aligned onto an even 1/``uniform_fps`` s grid; there is
no mode that keeps the index camera's own (uneven) frame times.
"""

import os
from dataclasses import dataclass

import yaml

# Cameras tried in order; the first one the session recorded sets the time window.
DEFAULT_INDEX_SOURCES = ("right_umi_camera", "left_umi_camera")
# The UMI cameras record at 30 Hz, so aligned rows are 1/30 s apart.
DEFAULT_UNIFORM_FPS = 30.0


@dataclass(frozen=True)
class AlignmentSettings:
    index_sources: tuple[str, ...]
    uniform_fps: float
    copy_videos: bool


def alignment_settings(config: dict) -> AlignmentSettings:
    """Settings from a loaded session config; absent keys take the defaults."""
    block = (config or {}).get("alignment") or {}
    sources = block.get("index_sources") or ",".join(DEFAULT_INDEX_SOURCES)
    index_sources = tuple(s.strip() for s in str(sources).split(",") if s.strip())
    uniform_fps = float(block.get("uniform_fps") or DEFAULT_UNIFORM_FPS)
    if uniform_fps <= 0:
        raise ValueError(f"alignment.uniform_fps must be positive, got {uniform_fps}")
    return AlignmentSettings(
        index_sources=index_sources,
        uniform_fps=uniform_fps,
        copy_videos=bool(block.get("copy_videos", True)),
    )


def session_alignment_settings(session_dir: str) -> AlignmentSettings:
    """Settings from ``<session_dir>/config.yaml`` (defaults if it is missing)."""
    path = os.path.join(session_dir, "config.yaml")
    if not os.path.exists(path):
        return alignment_settings({})
    with open(path, "r", encoding="utf-8") as f:
        return alignment_settings(yaml.safe_load(f) or {})

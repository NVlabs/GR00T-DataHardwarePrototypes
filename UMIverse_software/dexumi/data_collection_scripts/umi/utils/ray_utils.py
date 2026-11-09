# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import logging
import os
from typing import List, Optional

import ray

logger = logging.getLogger(__name__)


def load_ray_ignore_patterns(ignore_file: Optional[str] = None) -> List[str]:
    """Load Ray ignore patterns from a text file.

    Args:
        ignore_file: Path to ignore file. If None, uses default path.

    Returns:
        List of exclude patterns (empty lines and comments are filtered out)
    """
    if ignore_file is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(this_dir))
        ignore_file = os.path.join(repo_root, "umi", "utils", ".rayignore.txt")

    if not os.path.exists(ignore_file):
        logger.warning(
            "Ray ignore file not found: %s, using empty excludes", ignore_file
        )
        return []

    patterns = []
    with open(ignore_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)

    logger.debug("Loaded %d Ray ignore patterns from: %s", len(patterns), ignore_file)
    return patterns


@contextlib.contextmanager
def ray_session(num_workers: int, extra_env: Optional[dict] = None):
    """Context manager that initializes Ray if not already running and shuts it down on exit.

    Args:
        num_workers: Number of Ray worker CPUs. Pass -1 (or any value <= 0) to let
            Ray use all available CPUs.
        extra_env: Optional extra keys to merge into the Ray ``runtime_env`` dict
            (e.g. ``{"env_vars": {"PYTHONPATH": "..."}}``) before calling
            ``ray.init()``.  Keys from ``extra_env`` take precedence over the
            defaults built here.
    """
    initialized_ray = False
    if not ray.is_initialized():
        ignore_patterns = load_ray_ignore_patterns()
        runtime_env: dict = {"excludes": ignore_patterns}
        if extra_env:
            runtime_env.update(extra_env)
        if num_workers > 0:
            ray.init(num_cpus=num_workers, runtime_env=runtime_env)
        else:
            ray.init(runtime_env=runtime_env)
        initialized_ray = True

    try:
        yield
    finally:
        if initialized_ray:
            ray.shutdown()

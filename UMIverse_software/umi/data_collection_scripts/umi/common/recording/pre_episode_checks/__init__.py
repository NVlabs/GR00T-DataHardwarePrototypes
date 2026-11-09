# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-episode checks: pluggable safety checks run before each new episode.

Public API:
    - PreEpisodeCheck protocol
    - PreEpisodeContext dataclass
    - register decorator
    - discover_checks(search_dir) -> auto-import every *.py in the folder
    - build_checks(cfg, search_dirs=...) -> list[PreEpisodeCheck]
    - run_pre_episode_checks(checks, ctx) -> (ok, failures)

This package provides the registry/runner only. Concrete check files live
in a dedicated folder next to the script that uses them (e.g.
`real_scripts/data_collection/pre_episode_checks/<setup>.py`); passing
that folder via `search_dirs` makes them auto-register — no per-file
import needed.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

REGISTRY: dict[str, type["PreEpisodeCheck"]] = {}
_DISCOVERED_DIRS: set[Path] = set()

# Ignore recognized retired check names in persisted configs; reject every
# other unknown name so misspelled safety checks cannot silently disappear.
RETIRED_CHECKS = frozenset({"scene.change_props"})


@dataclass
class PreEpisodeContext:
    """State checks may inspect."""

    latest_frames: dict[str, list]  # source_name -> [latest_frame, ...]
    cfg: dict  # full app config (YAML)
    recorder_manager: Any  # for advanced checks needing recorder state
    # Order of video source names matching visualizer.combine_frames(video_frames)
    video_source_order: list[str] | None = None


@runtime_checkable
class PreEpisodeCheck(Protocol):
    name: str

    def run(self, ctx: PreEpisodeContext) -> tuple[bool, str]:
        """Return (ok, message). Message is human-readable."""
        ...


def register(name: str):
    """Decorator: register a check class under a name."""

    def deco(cls):
        cls.name = name
        REGISTRY[name] = cls
        return cls

    return deco


def discover_checks(search_dir: str | Path) -> list[str]:
    """Auto-import every `*.py` (excluding dunder files) in `search_dir`.

    Each imported module's @register decorators run as a side effect,
    populating REGISTRY. Idempotent: each directory is only scanned once
    per process. Intended for use with a dedicated checks folder, e.g.
    `real_scripts/data_collection/pre_episode_checks/`.
    """
    search_dir = Path(search_dir).resolve()
    if search_dir in _DISCOVERED_DIRS or not search_dir.is_dir():
        return []
    _DISCOVERED_DIRS.add(search_dir)

    imported: list[str] = []
    for path in sorted(search_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        mod_name = f"_pre_episode_checks_auto.{search_dir.name}.{path.stem}"
        if mod_name in sys.modules:
            imported.append(path.stem)
            continue
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        imported.append(path.stem)
    return imported


def build_checks(
    checks_cfg: list[dict] | None,
    search_dirs: Iterable[str | Path] = (),
    relaxed: Iterable[str] = (),
) -> list[PreEpisodeCheck]:
    """Instantiate checks from a YAML list of {name, params, enabled, enforce}.

    Every `*.py` in `search_dirs` is imported first (see `discover_checks`),
    so callers don't need to import each check module by hand.
    Checks are enabled and enforced by default. ``relaxed`` names checks that
    still run but report failures as warnings rather than blocking an episode.
    """
    relaxed = set(relaxed)
    for d in search_dirs:
        discover_checks(d)
    if not checks_cfg:
        return []
    instances: list[PreEpisodeCheck] = []
    for entry in checks_cfg:
        name = entry["name"]
        if name in RETIRED_CHECKS:
            continue
        if name not in REGISTRY:
            raise KeyError(
                f"unknown pre_episode_check '{name}'. available: {sorted(REGISTRY)}"
            )
        enabled = entry.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(
                f"pre_episode_check '{name}': enabled must be true or false"
            )
        if not enabled:
            continue
        enforce = entry.get("enforce", True)
        if not isinstance(enforce, bool):
            raise ValueError(
                f"pre_episode_check '{name}': enforce must be true or false"
            )
        params = entry.get("params") or {}
        instance = REGISTRY[name](**params)
        # ``name`` identifies the implementation and remains stable for
        # dispatch. ``check_id`` identifies one configured instance, allowing
        # side-specific checks of the same type to produce unambiguous logs
        # and failures.
        instance.check_id = str(entry.get("id") or name)
        instance.enforce = enforce and name not in relaxed
        instances.append(instance)
    return instances


def run_pre_episode_checks(
    checks: list[PreEpisodeCheck],
    ctx: PreEpisodeContext,
    warnings: list[tuple[str, str]] | None = None,
) -> tuple[bool, list[tuple[str, str]]]:
    """Run every check; return whether all enforced checks passed and failures.

    Warn-only failures are appended to ``warnings`` when supplied.
    """
    failures: list[tuple[str, str]] = []
    for check in checks:
        ok, msg = check.run(ctx)
        if not ok:
            entry = (getattr(check, "check_id", check.name), msg)
            if getattr(check, "enforce", True):
                failures.append(entry)
            elif warnings is not None:
                warnings.append(entry)
    return (not failures, failures)

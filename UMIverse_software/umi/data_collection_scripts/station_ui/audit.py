# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Audit page: what is recorded on disk, and how the health check is run.

The Data Audit page answers two questions about each recorded session: does
every episode have an aligned copy, and did the data check leave any warnings
on it. Both are answered from the files on disk.

The page has to render immediately, so this module only reads directory
entries and a few small JSON and YAML files. It never imports
`real_scripts.data_processing`, which loads the data-check code and takes
about five seconds.

Sessions live in the output folder (`umi.common.utility.output_layout`):
`output/raw/{task}/{session}`, with their aligned copies under
`output/aligned/…` and their LeRobot conversions under
`output/lerobot_format/{task}/{embodiment}.{session}`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import sys

import yaml

from station_ui.core import session_identity
from umi.common.recording.session_seal import folder_is_sealed
from umi.common.utility.output_layout import counterpart, output_root

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

HEALTH_CHECK = "real_scripts/data_processing/session_health_check.py"

LEROBOT_CONVERTER = (
    "real_scripts/data_processing/lerobot_conversion/umi_lerobot_conversion.py"
)
# Where the converter writes its datasets, under the output root.
LEROBOT_DIR = "lerobot_format"

# Use the server interpreter and its installed dependencies. Re-running
# uv resolution would require network access and a launcher PATH containing uv.
LAUNCHER = (sys.executable,)

# Allow full-session re-alignment to exceed the ordinary job timeout,
# while retaining a finite limit for stalled processing.
TIMEOUT_S = 24 * 60 * 60.0

# Session folders are named `YYYY-MM-DD_…`. Outside the output layout, sibling
# folders ending in `_aligned`, `_quarantine` or `_reports` share the same stem;
# they hold derived data, not sessions.
_SESSION_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_DERIVED_SUFFIXES = ("_quarantine", "_aligned", "_reports")

# A finalized episode. `.pending` folders are still being written or checked,
# so they have no aligned copy to look for yet.
_EPISODE_RE = re.compile(r"^episode_(\d+)$")

# Strip ANSI color sequences before displaying subprocess text in the browser.
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def plain(text: str) -> str:
    """The same line without terminal colour codes."""
    return _ANSI.sub("", text)


class Session:
    """One local session directory, as the audit page describes it."""

    def __init__(self, path: pathlib.Path, *, active: bool = False):
        self.path = path
        self.name = path.name
        self.episodes = session_identity.count_episodes(path)
        self.active = active
        finalized = _finalized_episodes(path)
        self.finalized = len(finalized)
        aligned_dir = aligned_folder(path)
        self.aligned_dir = aligned_dir
        root = _zarr_metadata(aligned_dir, "group")
        attrs = root.get("attributes")
        index_source = attrs.get("index_source") if isinstance(attrs, dict) else None
        copy_videos = _alignment_config(path).get("copy_videos", True)
        self.alignment_problems = {
            name: problems
            for name in finalized
            if (
                problems := _alignment_problems(
                    aligned_dir / name,
                    index_source,
                    path / name if copy_videos else None,
                )
            )
        }
        self.aligned = self.finalized - len(self.alignment_problems)
        # (episode name, warning) for every warning the record-time data check
        # saved on a finalized episode, in episode order.
        self.warnings = [
            (name, warning)
            for name in finalized
            for warning in _episode_warnings(path / name)
        ]
        self.lerobot = lerobot_datasets(aligned_dir)
        # Sealed: finished, so no more episodes can be recorded into it.
        self.closed = folder_is_sealed(path)

    @property
    def missing_alignment(self) -> int:
        return self.finalized - self.aligned


def _zarr_metadata(path: pathlib.Path, node_type: str) -> dict:
    """Read only metadata; listing sessions must not load sensor arrays."""
    try:
        meta = json.loads((path / "zarr.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) and meta.get("node_type") == node_type else {}


def _alignment_problems(
    episode: pathlib.Path, index_source: str | None, raw_episode: pathlib.Path | None
) -> list[str]:
    """Detect unfinished alignment without importing Zarr or reading chunks.

    The aligner creates the episode folder and copies its attributes before it
    processes any streams. Those alone cannot prove that alignment succeeded.
    The full Check action also reads arrays to detect corrupt chunks.
    """
    if not _zarr_metadata(episode, "group"):
        return [f"{episode.name}: aligned episode is missing or unreadable."]
    if not isinstance(index_source, str) or not index_source:
        return [f"{episode.name}: alignment index source is missing."]
    problems = []
    for name in (
        "valid_indices",
        "valid_timestamps",
        f"{index_source}/aligned_frame_indices",
    ):
        shape = _zarr_metadata(episode / name, "array").get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 1
            or not isinstance(shape[0], int)
            or shape[0] <= 0
        ):
            problems.append(
                f"{episode.name}/{name}: alignment data is missing or empty."
            )
    # Videos are copied after the arrays. Compare sizes to catch interrupted
    # copies without decoding or reading the video contents.
    if raw_episode is not None:
        for video in raw_episode.glob("*.mp4"):
            try:
                size = video.stat().st_size
                if size > 0 and (episode / video.name).stat().st_size == size:
                    continue
            except OSError:
                pass
            problems.append(
                f"{episode.name}/{video.name}: aligned video is missing or incomplete."
            )
    return problems


def _finalized_episodes(path: pathlib.Path) -> list[str]:
    """`episode_N` folder names in `path`, in episode order."""
    try:
        found = [
            (int(match.group(1)), entry.name)
            for entry in path.iterdir()
            if entry.is_dir() and (match := _EPISODE_RE.match(entry.name))
        ]
    except OSError:
        return []
    return [name for _, name in sorted(found)]


def aligned_folder(path: pathlib.Path) -> pathlib.Path:
    """Where this session's aligned copy lives, by the health check's rule.

    `alignment.aligned_dir` from the session's `config.yaml` when it names one,
    else the output layout's `aligned` counterpart.
    """
    explicit = _alignment_config(path).get("aligned_dir")
    if explicit:
        return pathlib.Path(os.path.expanduser(str(explicit)))
    return pathlib.Path(counterpart(str(path), "aligned"))


def _alignment_config(path: pathlib.Path) -> dict:
    try:
        cfg = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8")) or {}
        alignment = cfg.get("alignment") or {}
    except (OSError, ValueError, AttributeError, yaml.YAMLError, RecursionError):
        return {}
    return alignment if isinstance(alignment, dict) else {}


def lerobot_datasets(aligned_dir: pathlib.Path) -> list[pathlib.Path]:
    """The finished LeRobot conversions of the session aligned at `aligned_dir`.

    The converter names each `{task}/{embodiment}.{session}`, taking task and
    session from the aligned folder's parent and name. One that has no
    `meta/info.json` was interrupted and is not counted.
    """
    task_dir = output_root() / LEROBOT_DIR / aligned_dir.parent.name
    try:
        return sorted(
            info.parent.parent
            for info in task_dir.glob(f"*.{aligned_dir.name}/meta/info.json")
        )
    except OSError:
        return []


def _episode_warnings(episode: pathlib.Path) -> list[str]:
    """The warnings the record-time data check saved on one episode.

    Read directly from the episode's `zarr.json` attributes rather than through
    `zarr`, to keep the page fast (see the module docstring). An episode
    recorded without data-check warnings, or whose attributes cannot be read,
    has none to show.
    """
    try:
        attrs = json.loads((episode / "zarr.json").read_text(encoding="utf-8"))
        findings = (attrs.get("attributes") or {}).get("data_check_findings") or {}
        warnings = findings.get("warnings") or []
    except (OSError, ValueError, AttributeError):
        return []
    return [str(warning) for warning in warnings if isinstance(warning, str)]


def recordings_root() -> pathlib.Path:
    """Where raw sessions are recorded: ``output/raw``, as in the script."""
    return output_root() / "raw"


def scan_roots(primary: pathlib.Path) -> list[pathlib.Path]:
    """The dirs the script scans: the root and each task folder in it.

    Sessions are recorded to `output/raw/{task}/{session}`, so the folders one
    level down hold them (`session_health_check.recordings_scan_roots`).
    """
    root = pathlib.Path(os.path.abspath(os.path.expanduser(str(primary))))
    if not root.is_dir():
        return []
    roots = [root]
    try:
        roots += sorted(entry for entry in root.iterdir() if entry.is_dir())
    except OSError:
        pass
    return roots


def list_sessions(roots, active: str = "") -> list[Session]:
    """Every session directory under `roots`: the active one first, then
    newest name first.

    The active session always comes first: it is the one the operator came to
    check, and sorting by name alone could put it below re-recorded or oddly
    named sessions that sort later.

    `active` is the session folder the console is currently set up for. It is
    compared by folder name only: the page scans several roots and the config
    holds one absolute path, so full paths would not match whenever the same
    folder is spelled differently.
    """
    active_name = pathlib.Path(str(active).rstrip("/")).name if active else ""
    found: dict[str, Session] = {}
    for root in roots:
        try:
            entries = sorted(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_dir() or not _SESSION_RE.match(entry.name):
                continue
            if entry.name.endswith(_DERIVED_SUFFIXES):
                continue
            found.setdefault(
                entry.name, Session(entry, active=entry.name == active_name)
            )
    return sorted(
        found.values(),
        key=lambda session: (session.active, session.name),
        reverse=True,
    )


# ----------------------------------------------------------------------- argv


def _scope_flags(session_folder: str, date: str, end_date: str = "") -> list[str]:
    # The script ignores --date when --session-folder is set, so only the folder
    # is sent; otherwise the page would show a date the run did not use.
    if session_folder:
        return ["--session-folder", session_folder]
    flags = ["--date", date] if date else []
    if end_date:
        flags += ["--end-date", end_date]
    return flags


def health_check_argv(
    *,
    date: str = "",
    end_date: str = "",
    task_name: str = "",
    dry_run: bool = False,
    session_folder: str = "",
) -> list[str]:
    flags = _scope_flags(session_folder, date, end_date)
    if task_name:
        flags += ["--task-name", task_name]
    if dry_run:
        flags.append("--dry-run")
    return [*LAUNCHER, str(REPO_ROOT / HEALTH_CHECK), *flags]


# Converter dependencies are installed by the data_collection group.
# Rerun station setup if an existing environment is missing them.
LEROBOT_MODULES = ("av", "pyarrow", "torch", "torchcodec")


def missing_lerobot_modules() -> list[str]:
    """The converter's modules this interpreter cannot import, by find_spec only.

    Importing torch to check would add seconds to a button press.
    """
    return [name for name in LEROBOT_MODULES if importlib.util.find_spec(name) is None]


def lerobot_argv(aligned_dir: pathlib.Path) -> list[str]:
    """Convert one aligned session into `output/lerobot_format`.

    Run with the server's interpreter, like the health check. The output root is
    passed explicitly so the converter writes where this page looks, whatever
    its own default resolves to.
    """
    return [
        *LAUNCHER,
        str(REPO_ROOT / LEROBOT_CONVERTER),
        "--output-root",
        str(output_root()),
        "--input-dataset",
        str(aligned_dir),
    ]


def script_env() -> dict:
    """Make the repository importable for the child, however it was installed.

    The script imports `real_scripts.data_processing`, which only resolves
    through an editable install. A station that installed a wheel, or a
    worktree whose venv points elsewhere, would otherwise fail with a
    misleading ImportError.
    """
    existing = os.environ.get("PYTHONPATH")
    parts = [str(REPO_ROOT)] + ([existing] if existing else [])
    return {"PYTHONPATH": os.pathsep.join(parts)}

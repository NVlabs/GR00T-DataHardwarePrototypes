# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session status and launch settings for the GUI's conversion step.

List recorded, aligned and converted episode counts for each session. The
supervisor runs conversion under the ``lerobot`` role and replaces the output
dataset only after conversion succeeds. The server requires all recorded
episodes to be aligned and blocks conversion during recording to avoid CPU
contention from video encoding.
"""

import json
import os
import re
from pathlib import Path

from gui.dc_ui.core.kinds import REPO_ROOT, LaunchSpec

LEROBOT_ROLE = "lerobot"
CONVERT_SCRIPT = (
    "real_scripts/data_processing/lerobot_conversion/dexumi_lerobot_conversion.py"
)
ALIGNED_SUFFIX = "_aligned"
LEROBOT_SUFFIX = "_lerobot"
# Exclude rejected recordings retained by earlier versions.
QUARANTINE_SUFFIX = "_quarantine"
_EPISODE_RE = re.compile(r"^episode_\d+$")


def _episodes(folder: Path) -> set[str]:
    """``episode_N`` groups of a Zarr session folder (finalized ones only)."""
    if not folder.is_dir():
        return set()
    return {
        path.name
        for path in folder.iterdir()
        if _EPISODE_RE.match(path.name) and (path / "zarr.json").is_file()
    }


def _aligned_episodes(aligned: Path) -> set[str]:
    """Aligned episodes: the aligner writes ``valid_timestamps`` last."""
    return {
        name
        for name in _episodes(aligned)
        if (aligned / name / "valid_timestamps" / "zarr.json").is_file()
    }


def _converted_episodes(dataset: Path) -> set[str] | None:
    """Source episodes of a LeRobot dataset, or None when there is none."""
    try:
        rows = (dataset / "meta" / "episodes.jsonl").read_text().splitlines()
        return {json.loads(row)["original_episode_index"] for row in rows if row}
    except (OSError, ValueError, KeyError):
        return None


def _session_names(root: Path) -> list[str]:
    names = set()
    for path in root.iterdir() if root.is_dir() else []:
        if not path.is_dir() or path.name.endswith((LEROBOT_SUFFIX, QUARANTINE_SUFFIX)):
            continue
        if not (path / "zarr.json").is_file():
            continue
        names.add(path.name.removesuffix(ALIGNED_SUFFIX))
    return sorted(names)


def session_rows(data_root) -> list[dict]:
    """One row per session under ``data_root``, most recently changed first."""
    root = Path(os.path.expanduser(str(data_root)))
    rows = []
    for name in _session_names(root):
        raw, aligned = root / name, root / (name + ALIGNED_SUFFIX)
        dataset = root / (name + LEROBOT_SUFFIX)
        recorded = _episodes(raw)
        aligned_eps = _aligned_episodes(aligned)
        converted = _converted_episodes(dataset)
        if not recorded and not aligned_eps:
            continue
        missing = sorted(recorded - aligned_eps, key=lambda n: int(n.split("_")[1]))
        if not aligned_eps:
            blocker = "not aligned yet"
        elif missing:
            blocker = f"{len(missing)} episode(s) not aligned: {', '.join(missing[:5])}"
        else:
            blocker = None
        latest = max(
            (p.stat().st_mtime for p in (raw, aligned, dataset) if p.exists()),
            default=0.0,
        )
        rows.append(
            {
                "name": name,
                "aligned_path": str(aligned),
                "dataset_path": str(dataset),
                "recorded": len(recorded),
                "aligned": len(aligned_eps),
                "converted": None if converted is None else len(converted),
                "up_to_date": converted is not None and converted == aligned_eps,
                "blocker": blocker,
                "modified": latest,
            }
        )
    rows.sort(key=lambda row: row["modified"], reverse=True)
    return rows


def lerobot_spec(aligned_path) -> LaunchSpec:
    return LaunchSpec(
        role=LEROBOT_ROLE,
        argv=[
            "uv",
            "run",
            CONVERT_SCRIPT,
            "--input-dataset",
            os.path.abspath(os.path.expanduser(str(aligned_path))),
            "--replace",
        ],
        cwd=str(REPO_ROOT),
        env={"PYTHONUNBUFFERED": "1", "PYTHONPATH": str(REPO_ROOT)},
        # Conversion completes on exit; it has no separate readiness event.
        ready_regexes=[],
        fail_regexes=[r"^FAILED "],
        done_regex=None,
    )

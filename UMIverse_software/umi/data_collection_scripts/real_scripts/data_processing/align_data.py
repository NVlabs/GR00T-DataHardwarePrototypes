# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Align a recorded session.

Resamples every source of each episode onto an even 1/uniform_fps s grid inside
the index camera's valid window: poses and gripper values are interpolated,
cameras take the nearest frame. The index cameras, the grid rate and video
copying come from the session's config.yaml (``alignment`` block, see
``umi.common.utility.alignment_settings``), so this re-runs exactly what the
station's background aligner did.

Usage:
    uv run real_scripts/data_processing/align_data.py --data-dir output/raw/<task>/<session>
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import click
import ray
import yaml
import zarr

from umi.common.data.data_buffer import DataBuffer
from umi.common.utility.alignment_settings import session_alignment_settings
from umi.common.utility.output_layout import counterpart
from umi.utils.ray_utils import ray_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Type alias for per-episode results: (episode_idx, success, error_message)
EpisodeResult = Tuple[int, bool, Optional[str]]


def _config_source_names(data_dir: str) -> set:
    """Return set of source names defined in config.yaml."""
    config_path = os.path.join(data_dir, "config.yaml")
    if not os.path.exists(config_path):
        return set()
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    names = set()
    for s in config.get("video_sources", []):
        names.add(s["name"])
    for s in config.get("numeric_sources", []):
        names.add(s["name"])
    return names


def resolve_index_source_from_config(data_dir: str, candidates: List[str]) -> str:
    """Return the first candidate that exists in the dataset config. Raises if none."""
    config_sources = _config_source_names(data_dir)
    for name in candidates:
        if name in config_sources:
            return name
    raise ValueError(
        f"No index source found in config. "
        f"Candidates: {candidates}. In config: {sorted(config_sources)}."
    )


def _write_zarr_json_index_source(output_dir: str, index_source: str) -> None:
    """Write or update dataset-level zarr.json with index_source in attributes."""
    path = os.path.join(output_dir, "zarr.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("attributes", {})["index_source"] = index_source
    else:
        data = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {"index_source": index_source},
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_alignment_report(
    output_dir: str,
    data_dir: str,
    index_source: str,
    uniform_fps: float,
    episode_results: List[EpisodeResult],
) -> str:
    """Write a structured alignment report to the output directory.

    The report follows the layout of ``data_check_report.json``, so a tool that
    reads one report type can read the other.

    Args:
        output_dir: Aligned output directory.
        data_dir: Raw data directory that was aligned.
        index_source: Name of the camera that set the time window.
        uniform_fps: Rate of the even output grid.
        episode_results: Per-episode ``(idx, success, error_msg)`` tuples.

    Returns:
        Absolute path to the written ``data_align_report.json``.
    """
    episodes: Dict[str, dict] = {}
    success_count = 0
    failed_count = 0
    for idx, success, error_msg in sorted(episode_results, key=lambda r: r[0]):
        key = f"episode_{idx}"
        if success:
            episodes[key] = {"status": "success"}
            success_count += 1
        else:
            episodes[key] = {"status": "failed", "error": error_msg or "Unknown error"}
            failed_count += 1

    report = {
        "summary": {
            "total_episodes": len(episode_results),
            "aligned_episodes": success_count,
            "failed_episodes": failed_count,
            "index_source": index_source,
            "uniform_fps": uniform_fps,
        },
        "episodes": episodes,
        "metadata": {
            "data_dir": os.path.abspath(data_dir),
            "output_dir": os.path.abspath(output_dir),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    report_path = os.path.join(output_dir, "data_align_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Alignment report written to: %s", report_path)
    return report_path


def _align_episode(
    data_dir: str,
    output_dir: str,
    episode_idx: int,
    index_source: str,
    copy_videos: bool,
    uniform_fps: float,
) -> EpisodeResult:
    """Align a single episode; never raises."""
    try:
        buffer = DataBuffer(
            data_dir=data_dir,
            target_dir=output_dir,
            index_source_name=index_source,
            copy_videos=copy_videos,
            uniform_fps=uniform_fps,
        )
        buffer.align_episode([episode_idx])
        return episode_idx, True, None
    except Exception as e:
        return episode_idx, False, str(e)


_align_episode_remote = ray.remote(_align_episode)


@click.command()
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Raw session folder, e.g. output/raw/<task>/<session>.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=None,
    help="Path to output directory. Defaults to the session's aligned folder: "
    "output/aligned/{task}/{session} for a session under output/raw, "
    "else <data-dir>_aligned",
)
@click.option(
    "--episodes",
    "-e",
    type=int,
    multiple=True,
    default=None,
    help="Episode indices to process. If not specified, processes all episodes.",
)
@click.option(
    "--num-workers",
    "-n",
    type=int,
    default=-1,
    help="Number of Ray workers for parallel processing. -1 uses all available CPUs.",
)
@click.option(
    "--sequential",
    "-s",
    is_flag=True,
    help="Disable Ray and use sequential for loop (easier for debugging).",
)
def main(
    data_dir: str,
    output_dir: Optional[str],
    episodes: tuple,
    num_workers: int,
    sequential: bool,
):
    """Align a recorded session onto an even time grid."""
    data_dir = os.path.abspath(os.path.expanduser(data_dir))

    # Quarantined episodes failed the data check, and their numbering restarts
    # at 0, so aligning them would overwrite the passed episodes in aligned/.
    # A quarantine folder is its own quarantine counterpart, in either layout.
    if counterpart(data_dir, "quarantine") == os.path.realpath(data_dir):
        raise click.UsageError(
            f"{data_dir} is a quarantine session: its episodes failed the data "
            "check. Align the session under output/raw instead."
        )

    if output_dir is None:
        output_dir = counterpart(data_dir, "aligned")
    else:
        output_dir = os.path.abspath(os.path.expanduser(output_dir))

    os.makedirs(output_dir, exist_ok=True)

    settings = session_alignment_settings(data_dir)
    index_source = resolve_index_source_from_config(
        data_dir, list(settings.index_sources)
    )
    _write_zarr_json_index_source(output_dir, index_source)

    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Index source: {index_source}")
    logger.info(f"Uniform grid: {settings.uniform_fps} fps")

    if episodes:
        episode_indices = list(episodes)
    else:
        data_store = zarr.open(data_dir, mode="r")
        episode_indices = []
        for key in data_store.group_keys():
            if key.startswith("episode_"):
                try:
                    idx = int(key.split("_")[1])
                    episode_indices.append(idx)
                except ValueError:
                    continue
        episode_indices.sort()

    if not episode_indices:
        logger.warning("No episodes found to process")
    else:
        logger.info(f"Episodes to process: {episode_indices}")

    # ------------------------------------------------------------------
    # Run alignment (Ray parallel or sequential)
    # ------------------------------------------------------------------
    args = (index_source, settings.copy_videos, settings.uniform_fps)
    if sequential:
        episode_results: List[EpisodeResult] = [
            _align_episode(data_dir, output_dir, idx, *args) for idx in episode_indices
        ]
    else:
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        pythonpath = project_root + (
            f":{existing_pythonpath}" if existing_pythonpath else ""
        )
        extra_env = {"env_vars": {"PYTHONPATH": pythonpath}}

        with ray_session(num_workers, extra_env=extra_env):
            futures = [
                _align_episode_remote.remote(data_dir, output_dir, idx, *args)
                for idx in episode_indices
            ]
            episode_results = ray.get(futures)

    # ------------------------------------------------------------------
    # Summarise results
    # ------------------------------------------------------------------
    success_count = sum(1 for _, ok, _ in episode_results if ok)
    failed_episodes = [idx for idx, ok, _ in episode_results if not ok]

    for idx, ok, err in sorted(episode_results, key=lambda r: r[0]):
        if ok:
            logger.info("Episode %d: aligned successfully", idx)
        else:
            logger.error("Episode %d: failed - %s", idx, err)

    # ------------------------------------------------------------------
    # Write alignment report
    # ------------------------------------------------------------------
    write_alignment_report(
        output_dir=output_dir,
        data_dir=data_dir,
        index_source=index_source,
        uniform_fps=settings.uniform_fps,
        episode_results=episode_results,
    )

    logger.info("Alignment complete!")
    logger.info("Output saved to: %s", output_dir)
    logger.info("Success: %d/%d episodes", success_count, len(episode_indices))
    if failed_episodes:
        logger.warning("Failed episodes: %s", failed_episodes)


if __name__ == "__main__":
    main()

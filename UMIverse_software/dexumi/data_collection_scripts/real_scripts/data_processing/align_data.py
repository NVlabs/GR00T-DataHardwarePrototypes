# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data Interpolation Script

Interpolates and aligns multi-source recordings using DataBuffer.
Uses camera timestamps as the index and interpolates other sources to match.

Usage:
    uv run real_scripts/data_processing/align_data.py --data-dir /path/to/recordings --output-dir /path/to/output
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
    """Write or update dataset-level zarr.json with index_source in attributes.

    The alignment worker can be SIGKILLed at any point, so the root metadata is
    written atomically (temp file + fsync + rename) and an unreadable file left
    by an earlier kill is recreated: the aligned root is derived data whose
    attributes are re-copied from the raw root on every alignment.
    """
    path = os.path.join(output_dir, "zarr.json")
    data = None
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = None
        if not isinstance(data, dict):
            logger.warning(f"Unreadable aligned root metadata {path}; recreating it")
            data = None
    if data is None:
        data = {"zarr_format": 3, "node_type": "group", "attributes": {}}
    if not isinstance(data.get("attributes"), dict):
        data["attributes"] = {}
    data["attributes"]["index_source"] = index_source

    temporary = path + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def write_alignment_report(
    output_dir: str,
    data_dir: str,
    index_source: str,
    downsample: int,
    episode_results: List[EpisodeResult],
) -> str:
    """Write a structured alignment report to the output directory.

    Args:
        output_dir: Aligned output directory.
        data_dir: Raw data directory that was aligned.
        index_source: Name of the camera/source used as interpolation index.
        downsample: Down-sample rate applied.
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
            "downsample_rate": downsample,
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


@ray.remote
def _align_episode_remote(
    data_dir: str,
    output_dir: str,
    episode_idx: int,
    index_source: str,
    latencies: Optional[Dict[str, float]],
    downsample: int,
    copy_videos: bool,
) -> Tuple[int, bool, Optional[str]]:
    """Ray remote function to align a single episode."""
    try:
        buffer = DataBuffer(
            data_dir=data_dir,
            target_dir=output_dir,
            index_source_name=index_source,
            latencies=latencies,
            downsample_rate=downsample,
            copy_videos=copy_videos,
        )
        buffer.align_episode([episode_idx])
        return episode_idx, True, None
    except Exception as e:
        return episode_idx, False, str(e)


@click.command()
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(exists=True),
    required=True,
    help="Path to the recorded session directory",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=None,
    help="Path to output directory. Defaults to <data-dir>_interpolated",
)
@click.option(
    "--index-sources",
    "-i",
    type=str,
    default="right_umi_camera,left_umi_camera",
    help="Source(s) to use as interpolation index, comma-separated. "
    "First source present in the dataset config is used. Episodes missing it will fail.",
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
    "--downsample",
    type=int,
    default=1,
    help="Downsample rate for the index source (1 = no downsampling)",
)
@click.option(
    "--no-copy-videos",
    is_flag=True,
    help="Don't copy video files to output directory",
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
    index_sources: str,
    episodes: tuple,
    downsample: int,
    no_copy_videos: bool,
    num_workers: int,
    sequential: bool,
):
    """Interpolate and align multi-source recordings."""
    candidates = [s.strip() for s in index_sources.split(",") if s.strip()]
    if not candidates:
        raise click.UsageError("--index-source must specify at least one source.")

    data_dir = os.path.expanduser(data_dir)

    if output_dir is None:
        output_dir = data_dir + "_interpolated"
    else:
        output_dir = os.path.expanduser(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Resolve index source once per dataset: first candidate present in config
    index_source = resolve_index_source_from_config(data_dir, candidates)
    _write_zarr_json_index_source(output_dir, index_source)

    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Index source: {index_source}")
    logger.info(f"Downsample rate: {downsample}")

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
    use_ray = not sequential
    episode_results: List[EpisodeResult] = []

    if use_ray:
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        pythonpath = project_root + (
            f":{existing_pythonpath}" if existing_pythonpath else ""
        )
        extra_env = {"env_vars": {"PYTHONPATH": pythonpath}}

        abs_data_dir = os.path.abspath(data_dir)
        abs_output_dir = os.path.abspath(output_dir)

        with ray_session(num_workers, extra_env=extra_env):
            futures = [
                _align_episode_remote.remote(
                    abs_data_dir,
                    abs_output_dir,
                    idx,
                    index_source,
                    None,  # latencies
                    downsample,
                    not no_copy_videos,
                )
                for idx in episode_indices
            ]
            episode_results = ray.get(futures)
    else:
        for idx in episode_indices:
            try:
                buffer = DataBuffer(
                    data_dir=data_dir,
                    target_dir=output_dir,
                    index_source_name=index_source,
                    latencies=None,
                    downsample_rate=downsample,
                    copy_videos=not no_copy_videos,
                )
                buffer.align_episode([idx])
                episode_results.append((idx, True, None))
            except Exception as e:
                episode_results.append((idx, False, str(e)))

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
        downsample=downsample,
        episode_results=episode_results,
    )

    logger.info("Alignment complete!")
    logger.info("Output saved to: %s", output_dir)
    logger.info("Success: %d/%d episodes", success_count, len(episode_indices))
    if failed_episodes:
        logger.warning("Failed episodes: %s", failed_episodes)


if __name__ == "__main__":
    main()

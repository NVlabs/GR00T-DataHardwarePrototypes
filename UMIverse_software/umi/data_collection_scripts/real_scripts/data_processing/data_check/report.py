# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Console pretty-printer for check_dataset reports, and the findings the
station persists on each episode."""

from typing import Any, Dict, Optional

__all__ = ["data_check_findings"]

FINDINGS_VERSION = 2


def data_check_findings(episode_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The report-only measurements worth keeping with an episode.

    Written by the station into the episode's zarr attrs (`data_check_findings`)
    so they travel with the episode whether it passes or is quarantined: the
    check's warnings, and the excursion scan. A tracking device whose excursion
    scan found nothing is still included, so "scanned, clean" differs from "not
    scanned". None when there is neither.
    """
    episode_result = episode_result or {}
    findings: Dict[str, Any] = {}
    warnings = list(episode_result.get("warnings") or [])
    if warnings:
        findings["warnings"] = warnings
    excursions = {
        device: checks["excursions"]
        for device, checks in episode_result.get("tracking_checks", {}).items()
        if isinstance(checks, dict) and "excursions" in checks
    }
    if excursions:
        findings["tracking_excursions"] = excursions
    if not findings:
        return None
    return {"v": FINDINGS_VERSION, **findings}

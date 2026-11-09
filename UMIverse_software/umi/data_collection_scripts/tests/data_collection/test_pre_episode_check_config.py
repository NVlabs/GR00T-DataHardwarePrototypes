# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Building the pre-episode checks a recording config names.

Every production template's list must build, since an unknown name stops the
recorder at startup. A retired check is the exception: the Station Console's
generated config still names it until the next Apply, so it is skipped.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from umi.common.recording.pre_episode_checks import build_checks

REPO = Path(__file__).resolve().parents[2]
CHECKS_DIR = REPO / "real_scripts" / "data_collection" / "pre_episode_checks"
TEMPLATES = sorted((REPO / "config" / "data_recording" / "production").glob("*.yaml"))


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.name)
def test_every_production_template_builds_its_checks(template):
    entries = (yaml.safe_load(template.read_text()) or {}).get(
        "pre_episode_checks"
    ) or []

    checks = build_checks(entries, search_dirs=[CHECKS_DIR])

    assert [c.name for c in checks] == [e["name"] for e in entries]


def test_a_config_generated_before_the_props_prompt_was_removed_still_builds():
    checks = build_checks(
        [
            {"name": "umi_device.uart_connected"},
            {"name": "scene.change_props", "params": {"every_n_episodes": 15}},
        ],
        search_dirs=[CHECKS_DIR],
    )

    assert [c.name for c in checks] == ["umi_device.uart_connected"]


def test_an_unknown_check_still_stops_the_build():
    with pytest.raises(KeyError, match="unknown pre_episode_check"):
        build_checks([{"name": "scene.no_such_check"}], search_dirs=[CHECKS_DIR])

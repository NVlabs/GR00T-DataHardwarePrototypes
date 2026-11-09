# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`SetupPreset.pose_tracker` must match what the template actually records.

Two places say what a rig is, and only one of them produces the config.

Here the preset field is the one that is not load-bearing for recording -- the
template decides which sources exist -- but it is load-bearing for the session
folder name, which becomes a path component under ``output/raw/{task}/{session}/``
and is the thing anyone later filters on to ask "which sessions used Vive". A
row that says `iphone` against a template full of `vive_tracker` sources would
mislabel every session it records, permanently and silently: nothing downstream
re-derives the tracker from the config, because by then the config is gone and
the folder name is the record.

So they are pinned to each other. Add a preset, retarget a template, or swap a
rig's pose source without updating the row, and this fails here rather than in a
month's worth of folder names.
"""

import pathlib

import pytest
import yaml

from station_ui.core import config_assembler
from station_ui.core.setup_presets import (
    POSE_TRACKER_IPHONE,
    POSE_TRACKER_NONE,
    POSE_TRACKER_VIVE,
    POSE_TRACKERS,
    SETUP_PRESETS,
    template_path,
)

REPO = pathlib.Path(__file__).resolve().parents[2]

# Every preset the console can resolve, with the tracker that preset
# advertises. Derived from the rows rather than a filename list, so a new
# preset or a retargeted template is covered the day it lands.
PRESET_TEMPLATES = sorted(
    {
        (preset.id, str(template_path(preset, REPO)), preset.pose_tracker)
        for preset in SETUP_PRESETS
    }
)


def _declared_tracker(path: str) -> str:
    """What the template's sources say the wrist pose comes from.

    Both source lists, because the answer lives in different ones: an iPhone is
    a `video_sources` camera and a Vive puck is a `numeric_sources` pose. A
    check that looked in only one would read "no tracker" for half the rigs.
    """
    cfg = yaml.safe_load(pathlib.Path(path).read_text()) or {}
    kinds = set()
    for key in ("video_sources", "numeric_sources"):
        for source in cfg.get(key) or []:
            if not isinstance(source, dict):
                continue
            declared = str(source.get("type", "")).lower()
            if "iphone" in declared or "record3d" in declared:
                kinds.add(POSE_TRACKER_IPHONE)
            if "vive" in declared:
                kinds.add(POSE_TRACKER_VIVE)
    if not kinds:
        return POSE_TRACKER_NONE
    if len(kinds) > 1:
        pytest.fail(f"{path} declares more than one pose source: {sorted(kinds)}")
    return kinds.pop()


@pytest.mark.parametrize(
    ("preset_id", "path", "declared"),
    PRESET_TEMPLATES,
    ids=[f"{pid}:{pathlib.Path(p).name}" for pid, p, _t in PRESET_TEMPLATES],
)
def test_preset_pose_tracker_matches_its_template(preset_id, path, declared):
    assert _declared_tracker(path) == declared, (
        f"preset {preset_id!r} declares pose_tracker={declared!r} but "
        f"{pathlib.Path(path).name} records {_declared_tracker(path)!r}"
    )


def test_every_preset_declares_a_known_token():
    """The field is undefaulted, so this cannot catch a missing value -- the
    dataclass does that at import. It catches a typo'd one, which would reach a
    folder name intact and split a rig's sessions across two spellings."""
    for preset in SETUP_PRESETS:
        assert preset.pose_tracker in POSE_TRACKERS, (
            f"{preset.id} declares unknown pose_tracker {preset.pose_tracker!r}"
        )


# ------------------------------------------- the folder name and the config


@pytest.mark.parametrize(
    ("preset_id", "path", "declared"),
    PRESET_TEMPLATES,
    ids=[f"{pid}:{pathlib.Path(p).name}" for pid, p, _t in PRESET_TEMPLATES],
)
def test_meta_tracking_source_agrees_with_the_name_the_session_is_filed_under(
    preset_id, path, declared
):
    """One session, two records of its tracker: `raw/{task}/{session}/` and
    `meta.tracking_source` inside the config. This pins the second to the first.

    They are produced by different mechanisms on purpose -- the folder name
    takes the declared `SetupPreset.pose_tracker` because it is minted from a
    draft before any config exists, while the config block is derived from the
    sources being assembled -- and two mechanisms are exactly what drifts. The
    failure is silent and permanent: nothing re-derives a tracker from a folder
    name, and a config whose block disagreed with the prefix it sits under would
    make "which sessions used Vive" answerable two ways.

    Arms only. The head is not part of the folder name (`pose_tracker` is one
    token for the rig) and comes from the operator's optional-device choice
    rather than from the template, so there is nothing on the name side to pin
    it to.
    """
    cfg = yaml.safe_load(pathlib.Path(path).read_text()) or {}
    block = config_assembler._tracking_sources(cfg, ())

    for side in ("left", "right"):
        if block[side] == POSE_TRACKER_NONE and declared != POSE_TRACKER_NONE:
            # A unimanual rig records one arm; the other legitimately has no
            # tracker while the row still names the one it does record.
            continue
        assert block[side] == declared, (
            f"{preset_id}: the session folder would say {declared!r} while the "
            f"config says {side}={block[side]!r}"
        )

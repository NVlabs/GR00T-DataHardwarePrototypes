# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The preview layout the operator gets is the one in the template.

Stations record from the *generated* config, not the template, so a template
`visualization:` block that the assembler dropped would silently have no effect
on any station -- the configs would look right in the repo and every station
would still render the plain grid. The assembler merges into a round-trip load
and never names `visualization`, so the block passes through; this pins that,
because a future from-scratch writer would break it without touching any file
these tests otherwise read.
"""

import pathlib
import shutil

import pytest

yaml = pytest.importorskip("yaml")


from station_ui.core.config_assembler import (  # noqa: E402
    ConfigAssembler,
    DeviceSpec,
    SessionSpec,
)
from station_ui.core.setup_presets import get_preset, template_path  # noqa: E402
from umi.common.tasks import task_registry_store  # noqa: E402
from umi.utils import device_registry  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]

PRESET = "bimanual_v4"
LEFT_BOARD, RIGHT_BOARD = "oak-DW97-001", "oak-DW97-002"

TASKS = {
    "plane_assembly_v9": {
        "name": "plane_assembly_v9",
        "short_description": "Assemble the plane",
        "mode": "bimanual",
        "HW_version": "v2.1",
        "training_group": "plane_assembly_v9",
    }
}

FAKE_REGISTRY = {
    "version": 0,
    "updated_at": "1970-01-01T00:00:00Z",
    "oak_devices": {
        LEFT_BOARD: "1944301000000000AC",
        RIGHT_BOARD: "1944301000000000AD",
    },
    "iphone_devices": {
        "iphone-L": "00008130-000000000052",
        "iphone-R": "00008130-000000000053",
    },
    "umi_v4_grippers": {
        "UMI-V4-0001": "AAAA0000AAAA0001",
        "UMI-V4-0002": "AAAA0000AAAA0002",
    },
    "device_metadata": {},
}


@pytest.fixture(autouse=True)
def registries(monkeypatch):
    """Deterministic registries: the device registry is a file-backed singleton."""
    monkeypatch.setattr(task_registry_store, "get_all_tasks", lambda: dict(TASKS))
    monkeypatch.setattr(device_registry, "_REGISTRY", dict(FAKE_REGISTRY))


def test_the_template_visualization_block_reaches_the_generated_config(tmp_path):
    repo = tmp_path / "repo"
    (repo / "config").mkdir(parents=True)
    shutil.copytree(
        REPO / "config" / "data_recording",
        repo / "config" / "data_recording",
        ignore=shutil.ignore_patterns("generated"),
    )
    shutil.copytree(REPO / "config" / "data_check", repo / "config" / "data_check")

    out = ConfigAssembler(repo).assemble(
        SessionSpec(
            preset_id=PRESET,
            operator_name="Test Operator",
            task_name="plane_assembly_v9",
            session_uid="91e98c89fccb",
            data_dir=str(tmp_path / "2026-08-25_x_91e98c89fccb"),
        ),
        DeviceSpec(
            grippers={"left": "UMI-V4-0001", "right": "UMI-V4-0002"},
            required_cameras={
                "left_umi_camera": LEFT_BOARD,
                "right_umi_camera": RIGHT_BOARD,
                # USB trigger cameras, named by their V4L2 index.
                "left_trigger_camera": "4",
                "right_trigger_camera": "2",
                "left_tracking_camera": "iphone-L",
                "right_tracking_camera": "iphone-R",
            },
            optional=(),
        ),
    )

    template = yaml.safe_load(template_path(get_preset(PRESET), REPO).read_text())
    generated = yaml.safe_load(pathlib.Path(out).read_text())

    assert generated["visualization"] == template["visualization"]
    # Named explicitly: these four are what the side-panel layout is made of,
    # so an equality assertion that both sides had dropped them would pass.
    for key in ("window_fraction", "filler_position", "side_panel_width", "grid_fill"):
        assert key in generated["visualization"], key

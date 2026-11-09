# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain dataclasses shared by gui/dc_ui/core, server.py and the page (``to_dict()`` = wire format)."""

import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RECORDER_ROLE = "recorder"
DEVICE_KINDS = ("iphone", "oak", "exo")
# Exoskeleton names encode the calibrated hand: DEXUMI-Left or DEXUMI-Right.
# Capture group 1 selects the side.
EXO_NAME_RE = re.compile(r"^DEXUMI-(Left|Right)$")
# Form slot -> recording-config source name. Sources are looked up by name,
# never by list index, so a reordered base yaml cannot silently swap devices.
IPHONE_SLOTS = {
    "right_iphone": "right_tracking_camera",
    "left_iphone": "left_tracking_camera",
}
OAK_SLOTS = {
    "right_oak": "right_umi_camera",
    "left_oak": "left_umi_camera",
    "head_oak": "head_umi_camera",
}
EXO_SLOTS = {
    "right_exo": "right_dexumi_exoskeleton",
    "left_exo": "left_dexumi_exoskeleton",
}
# An empty optional slot removes its source from the generated config.
# The head camera is optional; bimanual processing requires both wrist videos
# and both tracking poses.
OPTIONAL_SLOTS = ("head_oak",)
DEVICE_SLOTS = {**IPHONE_SLOTS, **OAK_SLOTS, **EXO_SLOTS}
SLOT_KIND = {
    **{slot: "iphone" for slot in IPHONE_SLOTS},
    **{slot: "oak" for slot in OAK_SLOTS},
    **{slot: "exo" for slot in EXO_SLOTS},
}
SLOT_SIDE = {"right_exo": "Right", "left_exo": "Left"}
# Camera slots the ChArUco identification assigns (they may also be picked by
# hand). A launch is bound to the scan only when all of them come from it.
IDENTIFIED_SLOTS = ("left_iphone", "left_oak", "right_iphone", "right_oak")
# Session text a launch may leave empty; the generated config records this then
# (the recorder would otherwise prompt on stdin for a missing description).
OPTIONAL_TEXT_FIELDS = ("operator_name", "task_description")
UNKNOWN_TEXT = "Unknown"
# Recordings land in the checkout's git-ignored output/ folder, each session's
# _aligned copy beside it.
DEFAULT_DATA_ROOT = str(REPO_ROOT / "output")


class _Dictable:
    def to_dict(self):
        return dataclasses.asdict(self)


@dataclass
class LaunchSpec(_Dictable):
    role: str
    argv: list
    cwd: str = ""
    env: dict = field(default_factory=dict)
    ready_regexes: list = field(default_factory=list)
    fail_regexes: list = field(default_factory=list)
    done_regex: str | None = None

    @classmethod
    def from_dict(cls, data):
        # Ignore unrecognized fields when reading persisted GUI process state.
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass
class DetectedDevice(_Dictable):
    kind: str
    device_id: str  # iPhone UDID, OAK device ID, exoskeleton board USB serial
    # Camera slots store device IDs. Exoskeleton slots store the calibration name
    # for the connected board, or None if the match is ambiguous.
    name: str | None = None
    label: str = ""
    side: str | None = None  # exo only: Right | Left | None
    config_path: str | None = None  # exo only: config/dexumi_exoskeleton/<name>.yaml
    config_issue: str | None = (
        None  # exo only: why the recorder would refuse the calibration
    )
    in_use: bool | None = None  # oak only: XLink BOOTED = another process owns it

    @classmethod
    def from_dict(cls, data):
        # Ignore unrecognized fields in persisted device-discovery records.
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass
class DcForm(_Dictable):
    task_name: str = ""
    operator_name: str = ""
    task_description: str = ""
    right_iphone: str = ""
    left_iphone: str = ""
    right_oak: str = ""
    left_oak: str = ""
    head_oak: str = ""
    right_exo: str = ""
    left_exo: str = ""

    @classmethod
    def from_dict(cls, data):
        data = dict(data or {})
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - names)
        if unknown:
            raise ValueError(f"unknown form fields: {unknown}")
        clean = {
            key: "" if value is None else str(value).strip()
            for key, value in data.items()
        }
        return cls(**clean)

    def device(self, slot):
        return getattr(self, slot)


@dataclass
class Finding(_Dictable):
    level: str  # error | warn | info
    code: str
    message: str

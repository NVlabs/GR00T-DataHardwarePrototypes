# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base bimanual yaml + the operator's choices -> the generated recording config.

Only these keys change: meta.{operator_name,task_name,task_description} (an
empty operator or description becomes ``Unknown``), recording.data_dir, each camera's id (params.iphone_udid / params.oak_device_id),
and each exoskeleton's uart_device_name + params.config_file (kept explicit so the launch gate in
recording_utils still cross-checks calibration against the device). An empty
optional slot (the head OAK) removes its source and sets meta.use_head_mount to
false, so the recorder's ``require_all_sources`` gate only counts cameras the
rig actually has. Nothing is added: provenance lives in the ``#`` header only,
because session_slot.session_file_conflicts compares the whole config and an
extra key would make every relaunch into an existing folder look like a conflict.
For the same reason the scanned device ids travel in a separate file next to
the config (device_binding.py), passed as ``--device-binding``.
"""

import copy
import datetime
import itertools
import os
import re

import yaml

from gui.dc_ui.core.device_binding import SUFFIX as BINDING_SUFFIX
from gui.dc_ui.core.kinds import (
    DEVICE_SLOTS,
    EXO_SLOTS,
    IPHONE_SLOTS,
    OAK_SLOTS,
    OPTIONAL_SLOTS,
    OPTIONAL_TEXT_FIELDS,
    REPO_ROOT,
    SLOT_KIND,
    UNKNOWN_TEXT,
)

BASE_CONFIG = "config/data_recording/dexumi/bimanual_dexumi_config.yaml"
# Zero-hardware twin of the base (mock cameras + mock exoskeletons): pass it as
# --base-config to exercise the launch path without devices.
MOCK_BASE_CONFIG = "config/data_recording/dexumi/mock_bimanual_dexumi_config.yaml"
EXO_CONFIG_DIR = "config/dexumi_exoskeleton"
# A slot exists only when the base declares a source of that name with the
# real hardware type; mock sources (or an absent source) need no device.
HARDWARE_TYPES = {
    "iphone": "iphone_camera",
    "oak": "oak_camera",
    "exo": "dexumi_exoskeleton",
}
SLOT_SECTION = {
    **{s: "video_sources" for s in IPHONE_SLOTS},
    **{s: "video_sources" for s in OAK_SLOTS},
    **{s: "numeric_sources" for s in EXO_SLOTS},
}

_write_counter = itertools.count(1)


def repo_path(relative):
    relative = os.path.expanduser(os.fspath(relative))
    return relative if os.path.isabs(relative) else os.path.join(REPO_ROOT, relative)


def load_base(base_config=BASE_CONFIG):
    with open(repo_path(base_config)) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{base_config}: base config is not a mapping")
    return cfg


def is_mock_config(cfg):
    sources = [*(cfg.get("video_sources") or []), *(cfg.get("numeric_sources") or [])]
    return bool(sources) and all(
        isinstance(source, dict) and str(source.get("type", "")).startswith("mock_")
        for source in sources
    )


def exo_config_file(device_name):
    return f"{EXO_CONFIG_DIR}/{device_name}.yaml"


def _source(cfg, section, name):
    for source in cfg.get(section) or []:
        if isinstance(source, dict) and source.get("name") == name:
            return source
    return None


def active_slots(cfg):
    """slot -> the source dict it fills, for every slot the config declares as real hardware."""
    active = {}
    for slot, name in DEVICE_SLOTS.items():
        source = _source(cfg, SLOT_SECTION[slot], name)
        if source is not None and source.get("type") == HARDWARE_TYPES[SLOT_KIND[slot]]:
            active[slot] = source
    return active


def exo_device_names(form, base):
    """uart_device_name of every real exoskeleton source after the operator's choices, in base order.

    Mirrors compose_config: a source behind a form slot takes the selected board,
    any other dexumi_exoskeleton source keeps the base's own name. Feeds
    session_dir.rig_token, so the folder's dex{X} names the boards actually recorded.
    """
    slot_by_source = {name: slot for slot, name in EXO_SLOTS.items()}
    names = []
    for source in base.get("numeric_sources") or []:
        if not isinstance(source, dict) or source.get("type") != HARDWARE_TYPES["exo"]:
            continue
        slot = slot_by_source.get(source.get("name"))
        name = form.device(slot) if slot else source.get("uart_device_name")
        if name:
            names.append(str(name))
    return names


def exo_calibration_files(cfg):
    """Repo-resolved ``params.config_file`` of every selected real exoskeleton source, in config order.

    These are the per-board calibrations session_dir.next_session compares
    with the copies the recorder bundled into the previous hw{N} folder.
    """
    files = []
    for source in cfg.get("numeric_sources") or []:
        if not isinstance(source, dict) or source.get("type") != HARDWARE_TYPES["exo"]:
            continue
        config_file = (source.get("params") or {}).get("config_file")
        if source.get("uart_device_name") and config_file:
            files.append(repo_path(config_file))
    return files


def compose_config(form, base, data_dir):
    cfg = copy.deepcopy(base)
    cfg.setdefault("meta", {})
    cfg["meta"]["task_name"] = form.task_name
    # Operator and description are optional: an empty one is recorded as Unknown.
    for field in OPTIONAL_TEXT_FIELDS:
        cfg["meta"][field] = getattr(form, field).strip() or UNKNOWN_TEXT
    cfg.setdefault("recording", {})["data_dir"] = data_dir
    for slot, source in active_slots(cfg).items():
        device = form.device(slot)
        if slot in OPTIONAL_SLOTS and not device:
            _drop_optional_source(cfg, slot, source)
            continue
        if slot in IPHONE_SLOTS:
            source.setdefault("params", {})["iphone_udid"] = device
        elif slot in OAK_SLOTS:
            source.setdefault("params", {})["oak_device_id"] = device
        else:
            source["uart_device_name"] = device
            source.setdefault("params", {})["config_file"] = exo_config_file(device)
    return cfg


def _drop_optional_source(cfg, slot, source):
    """Record without this optional device: no source, and the head flag off."""
    cfg[SLOT_SECTION[slot]].remove(source)
    if slot == "head_oak":
        cfg["meta"]["use_head_mount"] = False


def optional_slots_dropped(cfg, base):
    """Optional slots the base declares as hardware that the composed config records without."""
    return [
        slot
        for slot in OPTIONAL_SLOTS
        if slot in active_slots(base) and slot not in active_slots(cfg)
    ]


def _safe(value):
    return re.sub(r"[^\w.\-]+", "_", str(value))


# Keep only recent generated configs. The recorder copies the active config
# into its session, so pruning older generated files does not affect recordings.
KEEP_GENERATED = 50


def _prune_generated(out_dir, keep):
    """Keep the newest ``keep`` configs; a config's device binding goes with it."""
    names = os.listdir(out_dir)
    configs = sorted(
        n for n in names if n.endswith(".yaml") and not n.endswith(BINDING_SUFFIX)
    )
    stale = configs[: max(0, len(configs) - keep)]
    kept = set(configs) - set(stale)
    stale += [
        n
        for n in names
        if n.endswith(BINDING_SUFFIX)
        and n[: -len(BINDING_SUFFIX)] + ".yaml" not in kept
    ]
    for name in stale:
        try:
            os.unlink(os.path.join(out_dir, name))
        except OSError:
            pass


def write_generated_config(cfg, home, stem, base_config=BASE_CONFIG):
    out_dir = os.path.join(os.path.abspath(os.path.expanduser(home)), "generated")
    os.makedirs(out_dir, exist_ok=True)
    now = datetime.datetime.now()
    # microseconds + a per-process counter: two writes in one second never collide
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(out_dir, f"{stamp}_{next(_write_counter)}_{_safe(stem)}.yaml")
    with open(path, "w") as f:
        f.write(
            f"# Generated by dc_ui (DexUMI data collection GUI) at {now.isoformat(timespec='seconds')} "
            f"from {base_config} -- regenerate from the GUI instead of editing.\n"
            "# Shipped configs under config/data_recording/ are never modified; this file is the "
            "--config of record_data.py.\n"
        )
        yaml.safe_dump(cfg, f, sort_keys=False)
    _prune_generated(out_dir, KEEP_GENERATED)
    return path

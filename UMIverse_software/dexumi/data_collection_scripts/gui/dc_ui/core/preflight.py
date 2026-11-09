# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch gate: findings re-derived from the COMPOSED config plus the latest detection payload.

``error`` = record_data.py would fail at launch or record into the wrong place
(launch refused); ``warn`` = runs but probably not what the operator wants;
``info`` = context. The rules read the composed config, not the form, so what
is judged is exactly what will run.
"""

import os
import re
import time

import yaml

from gui.dc_ui.core.composer import (
    BASE_CONFIG,
    SLOT_SECTION,
    active_slots,
    is_mock_config,
    load_base,
    optional_slots_dropped,
    repo_path,
)
from gui.dc_ui.core.kinds import (
    DEVICE_SLOTS,
    IPHONE_SLOTS,
    OAK_SLOTS,
    SLOT_KIND,
    SLOT_SIDE,
    Finding,
)
from umi.common.recording.session_slot import folder_has_episode_history

LEVEL_ORDER = {"error": 0, "warn": 1, "info": 2}
TASK_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
# Leave room for the date, hardware index and generated-config timestamp in
# the filesystem's 255-byte filename component limit.
MAX_TASK_NAME_LENGTH = 128
STALE_DETECTION_S = 60
SLOT_LABELS = {
    "right_iphone": "right iPhone",
    "left_iphone": "left iPhone",
    "right_oak": "right wrist OAK",
    "left_oak": "left wrist OAK",
    "head_oak": "head OAK",
    "right_exo": "right exoskeleton",
    "left_exo": "left exoskeleton",
}
# Only task name is required; empty operator and description fields become Unknown.
FIELD_LABELS = {"task_name": "task name"}
# A reusable failed-launch folder may contain only recorder metadata:
# launch/config snapshots, device calibrations, device binding and root
# Zarr metadata. Any other content may be recording data and must block reuse.
LAUNCH_ARTIFACT_NAMES = frozenset(
    {"zarr.json", ".zattrs", ".zgroup", "device_binding.yaml"}
)
LAUNCH_ARTIFACT_SUFFIXES = (".yaml", ".yml", ".log")
# Session folders are named <YYYY-MM-DD>_...: the recording date comes from it.
SESSION_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_")
# The sibling folder the recorder derives from a session folder (the aligned
# copy): no task or folder name may end in it.
RESERVED_SUFFIX = "_aligned"


def _err(code, message):
    return Finding("error", code, message)


def _warn(code, message):
    return Finding("warn", code, message)


def _info(code, message):
    return Finding("info", code, message)


def has_blockers(findings):
    return any(f.level == "error" for f in findings)


def selected_devices(cfg):
    """slot -> device as the composed config names it: camera id or exoskeleton name (active hardware slots only)."""
    chosen = {}
    for slot, source in active_slots(cfg).items():
        if slot in IPHONE_SLOTS:
            value = (source.get("params") or {}).get("iphone_udid")
        elif slot in OAK_SLOTS:
            value = (source.get("params") or {}).get("oak_device_id")
        else:
            value = source.get("uart_device_name")
        chosen[slot] = (value or "").strip()
    return chosen


def _text_findings(cfg):
    meta = cfg.get("meta") or {}
    findings = []
    for field, label in FIELD_LABELS.items():
        if not str(meta.get(field) or "").strip():
            findings.append(_err("missing_field", f"{label} is required"))
    task = str(meta.get("task_name") or "").strip()
    if len(task) > MAX_TASK_NAME_LENGTH:
        findings.append(
            _err(
                "task_name_too_long",
                f"task name must be at most {MAX_TASK_NAME_LENGTH} characters so session and config filenames fit",
            )
        )
    if task and not TASK_NAME_RE.match(task):
        findings.append(
            _err(
                "task_name_invalid",
                f"task name {task!r} must be letters, digits, '_', '-' or '.' (it names "
                "the session folder)",
            )
        )
    if task.endswith(RESERVED_SUFFIX):
        findings.append(
            _err(
                "task_name_reserved",
                f"task name {task!r} may not end in {RESERVED_SUFFIX}: the recorder "
                f"keeps each session's aligned copy in <folder>{RESERVED_SUFFIX}, so "
                "this task's folder would be another task's",
            )
        )
    if task and is_mock_config(cfg) and "mock" not in task.lower():
        findings.append(
            _err(
                "mock_task_name",
                "Mock collection requires 'mock' in the task name so its session "
                "folder and recorded data are identifiable as simulation.",
            )
        )
    return findings


def _source_findings(cfg):
    meta = cfg.get("meta") or {}
    if meta.get("embodiment") != "bimanual_dexumi":
        return []
    findings = []
    for slot, name in DEVICE_SLOTS.items():
        if slot == "head_oak" and not meta.get("use_head_mount"):
            continue
        sources = cfg.get(SLOT_SECTION[slot]) or []
        if not any(
            isinstance(source, dict) and source.get("name") == name
            for source in sources
        ):
            findings.append(
                _err(
                    "source_missing",
                    f"Bimanual DexUMI recording requires source {name!r}; "
                    "restore it in the base recording config.",
                )
            )
    return findings


def _device_findings(chosen, detection):
    findings = []
    by_kind = {
        kind: {d.name: d for d in detection.get(kind) or [] if d.name}
        for kind in ("iphone", "oak", "exo")
    }
    for slot in chosen:
        name = chosen[slot]
        label = SLOT_LABELS[slot]
        if not name:
            message = (
                f"no calibrated {SLOT_SIDE[slot].lower()} glove connected: plug it in, "
                "or calibrate it in Step 01"
                if slot in SLOT_SIDE
                else f"{label} not selected"
            )
            findings.append(_err("device_missing", message))
            continue
        device = by_kind[SLOT_KIND[slot]].get(name)
        if device is None:
            findings.append(
                _err(
                    "device_not_connected",
                    f"{label} {name!r} is not among the detected devices; plug it in and "
                    "press Refresh, or pick another",
                )
            )
            continue
        if slot in SLOT_SIDE and device.side != SLOT_SIDE[slot]:
            findings.append(
                _err(
                    "exo_side_mismatch",
                    f"{label} slot holds {name!r} (side {device.side or 'none'}); it needs a "
                    f"DEXUMI-{SLOT_SIDE[slot]} board",
                )
            )
        if slot in SLOT_SIDE and device.config_issue:
            findings.append(
                _err(
                    "exo_config_invalid",
                    f"{name}: calibration {device.config_path} has {device.config_issue}; "
                    "the recorder refuses it",
                )
            )
        if SLOT_KIND[slot] == "oak" and device.in_use:
            findings.append(
                _warn(
                    "oak_in_use",
                    f"{label} {name} reports BOOTED: another process owns it; the recorder "
                    "will fail to open it unless that process exits first",
                )
            )
    seen = {}
    for slot in chosen:
        name = chosen[slot]
        if not name:
            continue
        key = (SLOT_KIND[slot], name)
        if key in seen:
            findings.append(
                _err(
                    "device_duplicate",
                    f"{name!r} is selected for both {SLOT_LABELS[seen[key]]} and "
                    f"{SLOT_LABELS[slot]}",
                )
            )
        else:
            seen[key] = slot
    return findings


def _detection_findings(detection, now):
    if not detection:
        return [
            _err("detection_missing", "no device detection yet; press Refresh devices")
        ]
    findings = []
    errors = detection.get("errors") or []
    if errors:
        findings.append(
            _warn(
                "detection_errors", "device detection reported: " + " | ".join(errors)
            )
        )
    at = detection.get("at")
    if at is not None and (now or time.time()) - float(at) > STALE_DETECTION_S:
        findings.append(
            _info(
                "detection_stale",
                f"device list is {int((now or time.time()) - float(at))} s old; Refresh "
                "before starting if anything was replugged",
            )
        )
    return findings


def _file_findings(base_config):
    if not os.path.isfile(repo_path(base_config)):
        return [
            _err(
                "base_config_missing", f"base recording config not found: {base_config}"
            )
        ]
    return []


def _optional_slot_findings(cfg, base_config, detection):
    """An empty optional slot is a choice, not a blocker; say what it means.

    A spare available OAK that no slot uses is the one case worth a warning:
    on a three-OAK rig it is almost always the head camera left unselected.
    """
    if not os.path.isfile(repo_path(base_config)):
        return []
    try:
        base = load_base(base_config)
    except (OSError, ValueError, yaml.YAMLError):
        return []
    findings = []
    chosen = set(selected_devices(cfg).values())
    for slot in optional_slots_dropped(cfg, base):
        label = SLOT_LABELS[slot]
        findings.append(
            _info(
                "optional_device_off",
                f"no {label} selected: recording without it (meta.use_head_mount: false, "
                f"source {DEVICE_SLOTS[slot]!r} removed)",
            )
        )
        spare = [
            d.name
            for d in (detection or {}).get(SLOT_KIND[slot]) or []
            if d.name and not d.in_use and d.name not in chosen
        ]
        if spare:
            findings.append(
                _warn(
                    "optional_device_spare",
                    f"{SLOT_KIND[slot].upper()} {', '.join(spare)} is connected but not "
                    f"selected; pick it as the {label} or unplug it",
                )
            )
    return findings


def _data_dir_findings(cfg):
    data_dir = (cfg.get("recording") or {}).get("data_dir")
    if not data_dir:
        return []
    if "\x00" in str(data_dir):
        return [
            _err("data_dir_invalid", "recording directory contains a null character")
        ]
    ancestor = os.path.abspath(os.path.expanduser(str(data_dir)))
    # Validate without creating the folder: abandoned previews must not claim
    # session names or leave directories behind. Existing files (including
    # broken symlinks) must never be treated as future parent directories.
    while not os.path.lexists(ancestor):
        ancestor = os.path.dirname(ancestor)
    if not os.path.isdir(ancestor):
        return [
            _err(
                "data_dir_not_directory",
                f"recording directory cannot be created: {ancestor} is not a directory",
            )
        ]
    if not os.access(ancestor, os.W_OK | os.X_OK):
        return [
            _err(
                "data_dir_not_writable",
                f"recording directory cannot be written: {ancestor} needs write and search permissions",
            )
        ]
    return []


def folder_data_entries(folder):
    """Entries of an existing folder that are data, not the recorder's launch snapshot."""
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    return sorted(
        name
        for name in names
        if name not in LAUNCH_ARTIFACT_NAMES
        and not (
            name.lower().endswith(LAUNCH_ARTIFACT_SUFFIXES)
            and os.path.isfile(os.path.join(folder, name))
        )
    )


def _data_dir_contents_findings(cfg, resume=False):
    """Never record into a folder that already holds data, unless it continues hw{N}.

    Check the automatically selected folder: episodes (also pending ones), an
    episode history in the zarr attributes, or any entry beyond
    the launch snapshot block launch unless selected for continuation.
    ``resume`` is the server's
    computed default continuing hw{N} (same boards and calibration, and the
    recorder's own slot rule accepts the folder), which appends by design.
    """
    data_dir = (cfg.get("recording") or {}).get("data_dir")
    if not data_dir or "\x00" in str(data_dir):
        return []
    typed = str(data_dir).rstrip("/")
    expanded = os.path.expanduser(typed)
    findings = []
    if os.path.isdir(expanded) and not resume:
        entries = folder_data_entries(expanded)
        if entries or folder_has_episode_history(expanded):
            sample = ", ".join(entries[:4]) + (", …" if len(entries) > 4 else "")
            findings.append(
                _err(
                    "data_dir_has_data",
                    f"recording folder {typed} already holds data "
                    f"({sample or 'an episode history'}) but was not selected for "
                    "continuation; regenerate the automatic recording folder",
                )
            )
    base = os.path.basename(typed)
    task = str((cfg.get("meta") or {}).get("task_name") or "").strip()
    if base.endswith(RESERVED_SUFFIX) and not task.endswith(RESERVED_SUFFIX):
        findings.append(
            _err(
                "data_dir_reserved",
                f"folder name {base!r} ends in {RESERVED_SUFFIX}, the suffix of "
                "another session's aligned copy; pick another name",
            )
        )
    if base and not SESSION_DATE_RE.match(base):
        findings.append(
            _warn(
                "data_dir_name",
                f"folder name {base!r} does not start with the recording date "
                "(YYYY-MM-DD_); sessions are dated by that prefix",
            )
        )
    return findings


def preflight(
    cfg,
    detection,
    base_config=BASE_CONFIG,
    now=None,
    resume=False,
):
    findings = []
    findings += _text_findings(cfg)
    findings += _source_findings(cfg)
    chosen = selected_devices(cfg)
    if chosen:  # a mock base declares no hardware slot: devices are irrelevant
        findings += _detection_findings(detection, now)
        if detection:
            findings += _device_findings(chosen, detection)
    findings += _file_findings(base_config)
    findings += _optional_slot_findings(cfg, base_config, detection)
    findings += _data_dir_findings(cfg)
    findings += _data_dir_contents_findings(cfg, resume)
    data_dir = (cfg.get("recording") or {}).get("data_dir")
    if data_dir:
        findings.append(_info("session_dir", f"will record to {data_dir}"))
    findings.sort(key=lambda f: LEVEL_ORDER.get(f.level, 9))
    return findings

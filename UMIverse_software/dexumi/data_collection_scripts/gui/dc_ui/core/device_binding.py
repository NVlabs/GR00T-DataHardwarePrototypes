# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``--device-binding`` yaml the GUI hands record_data.py.

It names, per recording source, the device the GUI saw behind it::

    schema: 1
    gui_run: 20261009T101500-3f2a9c1e
    scanned:
      left_tracking_camera: {kind: iphone, name: "00008130-...", id: "00008130-...", scanned_at: ...}
      left_dexumi_exoskeleton: {kind: dexumi_exoskeleton, name: DEXUMI-Left, id: "0123456789ABCDEF"}

Cameras come from the Identify-cameras scan (``scanned_at`` is the time of
that scan, also when it covers several launches of one GUI run); the
exoskeleton boards and the head OAK come from discovery (no ``scanned_at``). Keys are the
recording source names of the config before the recorder expands a board into
``<name>_encoder`` / ``<name>_tactile``. record_data.py compares each id with
the one it opens and exits before its ready banner when they differ, then
records the outcome in the session's own ``device_binding.yaml``.

A launch gets a binding only when every selected wrist/tracking camera comes
from a scan whose ids still match discovery. A mock base (no camera slot) or a
camera picked by hand launches without the flag, as before: the recorder then
records ``bound_by: config``.
"""

import datetime
import os
import uuid

import yaml

from gui.dc_ui.core.kinds import DEVICE_SLOTS, IDENTIFIED_SLOTS, SLOT_KIND

SCHEMA = 1
SUFFIX = ".device_binding.yaml"
BINDING_KIND = {"iphone": "iphone", "oak": "oak", "exo": "dexumi_exoskeleton"}


def id_key(value):
    """Device ids compare after str().strip().upper(); None or blank is unknown."""
    if value is None:
        return None
    return str(value).strip().upper() or None


def same_id(a, b):
    return id_key(a) is not None and id_key(a) == id_key(b)


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def new_gui_run():
    """One id per GUI process; every binding it writes carries it."""
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def detected_device(detection, kind, name):
    """The one discovered device of ``kind`` named ``name`` (a camera's id, an exoskeleton's calibration), else None."""
    if not name:
        return None
    matches = [d for d in (detection or {}).get(kind) or [] if d.name == name]
    return matches[0] if len(matches) == 1 else None


def scanned_sources(chosen, identified, detection):
    """``scanned`` for the binding (source name -> entry), or None when this launch has no complete scan.

    ``chosen`` is slot -> selected device name (preflight.selected_devices).
    Every selected wrist/tracking camera must carry an identification with the
    same name and an id that discovery still reports for that name; otherwise
    the cameras were picked by hand (or the scan went stale) and the launch
    gets no binding.
    """
    cameras = [slot for slot in IDENTIFIED_SLOTS if chosen.get(slot)]
    if not cameras:
        return None
    scanned = {}
    for slot in cameras:
        name = chosen[slot]
        kind = SLOT_KIND[slot]
        record = (identified or {}).get(slot)
        device = detected_device(detection, kind, name)
        if (
            not isinstance(record, dict)
            or record.get("name") != name
            or device is None
            or not same_id(record.get("device_id"), device.device_id)
        ):
            return None
        entry = {"kind": BINDING_KIND[kind], "name": name, "id": str(device.device_id)}
        if record.get("scanned_at"):
            entry["scanned_at"] = record["scanned_at"]
        scanned[DEVICE_SLOTS[slot]] = entry
    for slot, name in chosen.items():
        if slot in IDENTIFIED_SLOTS or not name:
            continue
        device = detected_device(detection, SLOT_KIND[slot], name)
        if device is None or id_key(device.device_id) is None:
            continue  # preflight already blocks a selected device that is not connected
        scanned[DEVICE_SLOTS[slot]] = {
            "kind": BINDING_KIND[SLOT_KIND[slot]],
            "name": name,
            "id": str(device.device_id),
        }
    return scanned


def binding_path(config_path):
    root, _ = os.path.splitext(config_path)
    return root + SUFFIX


def write_binding(config_path, scanned, gui_run):
    """Write the binding next to the generated config; returns its path."""
    path = binding_path(config_path)
    with open(path, "w") as f:
        f.write(
            "# Device binding written by dc_ui for record_data.py --device-binding; "
            f"pairs with {os.path.basename(config_path)}.\n"
            "# Cameras: the Identify-cameras scan of this GUI run; exoskeletons and "
            "the head OAK: device discovery.\n"
        )
        yaml.safe_dump(
            {"schema": SCHEMA, "gui_run": str(gui_run), "scanned": scanned},
            f,
            sort_keys=False,
        )
    return path

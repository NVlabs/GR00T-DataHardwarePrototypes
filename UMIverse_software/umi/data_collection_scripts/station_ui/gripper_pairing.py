# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pairing evidence from the existing USB scan, shared by Configuration and Record.

Only a hub with one gripper and one device of a kind establishes ownership.
Missing locations, shared bench hubs and multiple cameras remain unverified.
"""

from __future__ import annotations

import logging

from station_ui import device_summary
from station_ui.devices import identifier_for

ROLES = {"gripper": "gripper", "umi": "oak", "tracking": "iphone"}
LABELS = {"gripper": "Gripper", "umi": "Top camera", "tracking": "Phone"}
logger = logging.getLogger(__name__)


def evidence(found) -> dict:
    """Kind-scoped aliases and ownership; safe to embed as JSON in the page."""
    aliases = {kind: {} for kind in ROLES.values()}
    owners = {kind: {} for kind in ROLES.values()}
    for record in found or []:
        kind = record.get("kind")
        if kind not in aliases:
            continue
        name = identifier_for(record)
        if not name:
            continue
        for value in (name, record.get("device_id")):
            if value:
                aliases[kind][str(value).strip().lower()] = name
    try:
        rows = device_summary.rows(found)
    except Exception:
        logger.exception("USB pairing evidence could not be read")
        rows = []
    for row in rows:
        if not row["port"] or len(row["gripper"]) != 1:
            continue
        owner = row["gripper"][0].name
        owners["gripper"][owner] = owner
        for kind in ("oak", "iphone"):
            if len(row[kind]) == 1:
                owners[kind][row[kind][0].name] = owner
    return {"aliases": aliases, "owners": owners}


def gripper_for_cameras(cameras: dict, found) -> tuple[str | None, str]:
    """Infer one connected gripper from all identified hub-mounted cameras.

    Trigger cameras connect directly to the station and cannot prove ownership.
    Missing or contradictory USB evidence leaves the gripper selection manual.
    """
    inventory = evidence(found)
    owners = []
    for kind in ("oak", "iphone"):
        if kind not in cameras:
            continue
        name = str(cameras[kind]).strip()
        name = inventory["aliases"][kind].get(name.lower(), name)
        owner = inventory["owners"][kind].get(name)
        if not owner:
            return (
                None,
                "Gripper could not be determined from the cameras' USB connections. Select the gripper manually.",
            )
        owners.append(owner)
    if not owners:
        return (
            None,
            "No gripper-mounted camera was identified. Select the gripper manually.",
        )
    if len(set(owners)) != 1:
        return (
            None,
            "The identified cameras belong to different grippers. Review their assignments and select the gripper manually.",
        )
    connected = {
        identifier_for(record)
        for record in found or []
        if record.get("kind") == "gripper" and record.get("state") == "available"
    }
    if owners[0] not in connected:
        return (
            None,
            "The matching gripper is unavailable. Check its connection and select it manually.",
        )
    return owners[0], ""


def check(values: dict, inventory: dict) -> dict:
    """Return affected roles and an operator-facing verdict for one gripper.

    The browser uses this same evidence and comparison on every input event.
    Unknown devices never count as a verified match or a confirmed mismatch.
    """
    devices, owners = {}, {}
    for role, kind in ROLES.items():
        value = str(values.get(role) or "").strip()
        devices[role] = inventory["aliases"][kind].get(value.lower(), value)
        owners[role] = inventory["owners"][kind].get(devices[role])

    affected, messages = set(), []
    for first, second in (
        ("umi", "tracking"),
        ("gripper", "umi"),
        ("gripper", "tracking"),
    ):
        if owners[first] and owners[second] and owners[first] != owners[second]:
            affected.update((first, second))
            messages.append(
                f"{LABELS[first]} {devices[first]} belongs to {owners[first]}; "
                f"{LABELS[second].lower()} {devices[second]} belongs to {owners[second]}."
            )
    if affected:
        return {
            "state": "mismatch",
            "affected": sorted(affected),
            "message": "Pairing mismatch. "
            + messages[0]
            + (
                " Select the phone and top camera from the same gripper row in Detected devices."
                if "tracking" in values
                else " Select the top camera from the selected gripper's row in Detected devices."
            ),
        }
    required = [role for role in ROLES if role in values]
    if any(not str(values[role] or "").strip() for role in required):
        state, message = (
            "incomplete",
            "Choose the devices below to check their pairing.",
        )
    elif required and all(owners[role] for role in required):
        state, message = (
            "matched",
            "Pairing verified for this gripper's top camera"
            + (" and phone." if "tracking" in values else "."),
        )
    else:
        state, message = (
            "unverified",
            (
                "Pairing not verified. Connect the devices through the gripper's USB hub, "
                "then Refresh devices."
            ),
        )
    return {"state": state, "affected": [], "message": message}


def config_mismatches(config: dict, found) -> list[str]:
    """Re-check the saved V4 devices before starting recording, after any rescan."""
    v4_sources = [
        source
        for source in config.get("numeric_sources") or []
        if (source.get("params") or {}).get("gripper_generation") == "v4"
    ]
    if not v4_sources:
        return []
    inventory = evidence(found)
    messages = []
    for source in v4_sources:
        name = str(source.get("name") or "")
        if not name.endswith("_umi_gripper"):
            continue
        arm = name.removesuffix("_umi_gripper")
        values = {"gripper": source.get("uart_device_name", "")}
        for camera in config.get("video_sources") or []:
            if camera.get("type") == "oak_device" and any(
                stream.get("name") == f"{arm}_umi_camera"
                for stream in camera.get("streams") or []
            ):
                values["umi"] = camera.get("oak_device_name") or camera.get(
                    "oak_device_id", ""
                )
            elif (
                camera.get("name") == f"{arm}_tracking_camera"
                and camera.get("type") == "iphone_camera"
            ):
                params = camera.get("params") or {}
                values["tracking"] = params.get("iphone_device_name") or params.get(
                    "iphone_udid", ""
                )
        result = check(values, inventory)
        if result["affected"]:
            messages.append(f"{arm.capitalize()} gripper: {result['message']}")
    return messages

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep ``reference_camera`` aligned with configured ``video_sources``.

A config can name a ``reference_camera`` that the rig no longer records -- a
head camera the operator removed in the console, say. That camera is gone from
``video_sources`` but ``reference_camera`` and ``camera.relative_calibration``
can be left pointing at it, causing a validation failure at record time.

This module is the single source of truth for reconciling that mismatch. It
is applied when the console assembles a config.
"""

from __future__ import annotations

from typing import Iterable, Optional, Set

# Fallback reference priority: head, tracking, wrist, then trigger; right
# before left, accepting persisted camera aliases. Without source rates
# this cannot reproduce ConfigAssembler's rate-first selection, so assembled
# configs remain authoritative. Wrist fallbacks support Vive rigs with no
# tracking video and avoid choosing an uncalibratable UVC source by name.
REFERENCE_CAMERA_PREFERENCE = (
    "head_oak_camera",
    "head_umi_camera",
    "head_tracking_camera",
    "right_tracking_camera",
    "left_tracking_camera",
    "right_umi_camera",
    "left_umi_camera",
    "right_trigger_camera",
    "left_trigger_camera",
)


def video_source_names(config: Optional[dict]) -> Set[str]:
    """Every camera name *config* records.

    Source-level ``name`` and ``streams[].name``. An OAK-FFC board is one
    ``oak_device`` source carrying two cameras under ``streams``, with no name of
    its own, and those stream names are what everything downstream calls a
    camera by: the mp4s, the zarr groups, ``RecorderManager.get_source`` and the
    ``camera.relative_calibration`` check's ``cameras`` list are all keyed on
    them.

    Reading only the source level is not enough: a rig whose cameras are all
    oak_device streams would appear to record nothing, so a valid
    ``reference_camera`` would be rejected as naming an absent source and the
    session would abort at RecorderManager construction.
    """
    if not config:
        return set()
    names: Set[str] = set()
    for source in config.get("video_sources") or []:
        try:
            name = source.get("name")
            streams = source.get("streams") or []
        except AttributeError:
            continue
        if name:
            names.add(name)
        for stream in streams:
            try:
                stream_name = stream.get("name")
            except AttributeError:
                continue
            if stream_name:
                names.add(stream_name)
    return names


def pick_reference_camera(source_names: Iterable[str]) -> Optional[str]:
    """Choose a reference camera from *source_names* using preference order."""
    names = set(source_names)
    if not names:
        return None
    for candidate in REFERENCE_CAMERA_PREFERENCE:
        if candidate in names:
            return candidate
    return sorted(names)[0]


def _replace_seq(seq, values: list):
    """Mutate ruamel CommentedSeq in place; return new list for plain lists."""
    if seq is not None and hasattr(seq, "clear") and hasattr(seq, "extend"):
        seq.clear()
        seq.extend(values)
    else:
        return values


def sync_reference_camera_to_video_sources(config: Optional[dict]) -> bool:
    """Align ``reference_camera`` and calibration cameras with ``video_sources``.

    When the Station Console removes a camera (e.g. head mount toggled off),
    this corrects ``reference_camera`` and the ``camera.relative_calibration``
    camera list so they only reference cameras that actually exist in
    ``video_sources``.

    Mutates ruamel ``CommentedMap``/``CommentedSeq`` in place when possible so
    round-trips preserve the YAML layout. Works on plain dicts/lists too.

    Returns True if anything was changed.
    """
    if not config:
        return False

    source_names = video_source_names(config)
    if not source_names:
        return False

    changed = False
    ref = config.get("reference_camera")

    if ref and ref not in source_names:
        new_ref = pick_reference_camera(source_names)
        if new_ref and config.get("reference_camera") != new_ref:
            config["reference_camera"] = new_ref
            changed = True
        ref = new_ref
    elif not ref:
        ref = pick_reference_camera(source_names)

    for check in config.get("pre_episode_checks") or []:
        try:
            if check.get("name") != "camera.relative_calibration":
                continue
            params = check.get("params")
        except AttributeError:
            continue
        if not params:
            continue

        if ref and params.get("reference_camera") != ref:
            params["reference_camera"] = ref
            changed = True

        cams = params.get("cameras")
        if cams is None:
            continue
        filtered = [c for c in cams if c in source_names]
        if ref and ref not in filtered:
            filtered.insert(0, ref)
        if list(cams or []) != filtered:
            replacement = _replace_seq(cams, filtered)
            if replacement is not None:
                params["cameras"] = replacement
            changed = True

    return changed

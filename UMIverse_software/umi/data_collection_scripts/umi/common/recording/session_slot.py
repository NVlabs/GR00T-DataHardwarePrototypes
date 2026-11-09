# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session-slot resolution for data collection.

When a session folder already holds episodes recorded under a different
config / data-check / device calibration, roll the session: mint a fresh uid
onto the session folder (``recording.data_dir``) rather than mixing two
provenances under one session id or raising a write-time lockout.
``meta.task_name`` is left untouched so the task folder stays the
operator's real task.

The rule is deliberately blunt, and does not care what drifted:

    A session folder is reusable iff it holds no episodes, or its stored
    acquisition content matches what this launch would write. Otherwise, roll it.

``check_enforcement`` is a per-launch policy and does not change acquisition,
so changing only that selection reuses the session.

A re-applied device swap, a mid-session gripper recalibration, a data-check
override and a restart into an edited config all take the same path. Branching
on the cause would only give the operator four ways to end up in the same
place.

Rolling replaces the folder's trailing uid rather than appending a counter:
the uid is read by anchoring on the end of the folder name (``session_uid.py``),
so ``…_<uid>_1`` would parse as having no uid at all. ``meta.session_uid`` is
rewritten together with ``recording.data_dir`` so both name the same session.
"""

from __future__ import annotations

import glob
import os
import re
from copy import deepcopy
from typing import Optional

import yaml

from umi.common.recording.session_seal import folder_is_sealed
from umi.common.recording.session_uid import (
    session_uid_from_dir_name,
    with_new_session_uid,
)
from umi.common.utility.print_utils import print_yellow

# Snapshot config before runtime expansion for session conflict detection;
# the enriched recording config is not the acquisition reference.
LAUNCH_CONFIG_NAME = "launch_config.yaml"


def _acquisition_config(config: Optional[dict]) -> dict:
    """Enforcement is a per-launch choice, independent of data acquisition."""
    result = deepcopy(config or {})
    result.pop("check_enforcement", None)
    return result


def _same_acquisition_config(stored: bytes, config: Optional[dict]) -> bool:
    try:
        parsed = yaml.safe_load(stored)
    except yaml.YAMLError:
        return False
    return isinstance(parsed, dict) and (
        _acquisition_config(parsed) == _acquisition_config(config)
    )


def _deep_merge_dict(base: dict, overlay: dict) -> dict:
    """Merge ``overlay`` into a copy of ``base`` (nested dicts merged recursively)."""
    merged = dict(base)
    for k, v in overlay.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge_dict(merged[k], v)
        else:
            merged[k] = v
    return merged


def config_yaml_bytes(config: Optional[dict]) -> bytes:
    """config.yaml bytes exactly as ``RecorderManager._save_config`` writes them."""
    return yaml.dump(config or {}, default_flow_style=False, sort_keys=False).encode()


def data_check_yaml_bytes(
    config: Optional[dict], data_check_config_path: Optional[str]
) -> Optional[bytes]:
    """data_check_config.yaml bytes, resolved default -> task, or None."""
    if not data_check_config_path:
        return None
    path = os.path.expanduser(data_check_config_path)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data_check_config = yaml.safe_load(f) or {}
    task_name = ""
    if config:
        task_name = str((config.get("meta") or {}).get("task_name") or "").strip()
    if task_name:
        from umi.common.tasks.task_registry import get_task

        spec = get_task(task_name)
        if spec is not None and spec.data_check_overrides:
            data_check_config = _deep_merge_dict(
                data_check_config, spec.data_check_overrides
            )
    return yaml.dump(
        data_check_config, default_flow_style=False, sort_keys=False
    ).encode()


def device_config_entries(config: Optional[dict]) -> list[tuple[str, str]]:
    """``(abs_source_path, basename)`` for every device config referenced by any
    source in ``config`` (gripper ``params.config_file``, etc.)."""
    if not config:
        return []
    out: list[tuple[str, str]] = []
    seen_paths: set[str] = set()
    seen_basenames: dict[str, str] = {}

    def add_path(raw_path: str) -> None:
        path = os.path.abspath(os.path.expanduser(raw_path))
        if path in seen_paths:
            return
        seen_paths.add(path)
        basename = os.path.basename(path)
        if seen_basenames.get(basename, path) != path:
            print_yellow(
                f"Warning: source configs share basename {basename!r} "
                f"({seen_basenames[basename]} vs {path}); bundling only the first"
            )
            return
        seen_basenames[basename] = path
        out.append((path, basename))

    for value in config.values():
        if not isinstance(value, list):
            continue
        for src in value:
            if not isinstance(src, dict):
                continue
            cfg_file = (src.get("params") or {}).get("config_file")
            if not cfg_file:
                continue
            add_path(str(cfg_file))
    return out


def save_launch_config(record_path: Optional[str], config: Optional[dict]) -> None:
    """Write the raw (as-loaded) config as the session's config reference.

    Call once the session folder is resolved and before the config is expanded,
    so the snapshot reflects the config file rather than the enriched runtime
    config.
    """
    if not record_path:
        return
    os.makedirs(record_path, exist_ok=True)
    with open(os.path.join(record_path, LAUNCH_CONFIG_NAME), "wb") as f:
        f.write(config_yaml_bytes(config))


def session_file_conflicts(
    record_path: Optional[str],
    config: Optional[dict],
    data_check_config_path: Optional[str] = None,
) -> list[str]:
    """Session content files already in ``record_path`` that differ from what
    this launch would use, else ``[]``.

    Compares:

    - ``launch_config.yaml`` — raw config-file snapshot, excluding enforcement
    - ``data_check_config.yaml`` — data-check config
    - each device-calibration file — stored session copy vs live source

    A gripper recalibration (content change) is a conflict. ``[]`` means the
    folder holds the same config + calibration (safe to resume). A conflicting
    folder that already holds episodes is rolled onto a fresh session (see
    ``resolve_session_slot``).
    """
    if not record_path or not os.path.isdir(record_path):
        return []
    conflicts: list[str] = []

    launch = os.path.join(record_path, LAUNCH_CONFIG_NAME)
    if os.path.isfile(launch):
        with open(launch, "rb") as f:
            stored = f.read()
            if stored != config_yaml_bytes(config) and not _same_acquisition_config(
                stored, config
            ):
                conflicts.append(LAUNCH_CONFIG_NAME)

    dc_incoming = data_check_yaml_bytes(config, data_check_config_path)
    if dc_incoming is not None:
        dc = os.path.join(record_path, "data_check_config.yaml")
        if os.path.isfile(dc):
            with open(dc, "rb") as f:
                if f.read() != dc_incoming:
                    conflicts.append("data_check_config.yaml")

    for src_path, basename in device_config_entries(config):
        stored = os.path.join(record_path, basename)
        if not os.path.isfile(stored):
            continue
        # Missing live source counts as drift: don't silently reuse a folder
        # whose only remaining calibration is a stale session copy.
        if not os.path.exists(src_path):
            conflicts.append(basename)
            continue
        with open(stored, "rb") as sf, open(src_path, "rb") as cf:
            if sf.read() != cf.read():
                conflicts.append(basename)

    return conflicts


def _reads_back_as_text(value: str) -> bool:
    """Would YAML parse this unquoted scalar as the string it is?

    Asked of YAML rather than guessed, because the set of scalars it reinterprets
    is longer than anyone remembers: digits, ``1e5``, ``yes``, ``~``, ``null``,
    ``0x1f``, sexagesimals in some versions.

    This is not hypothetical here. ``session_uid`` is ``uuid4().hex[:12]``, so
    about one in 29,000 is all digits -- and that one gets written unquoted,
    read back as an int, and every ``meta.get("session_uid")`` downstream then
    holds a number where the rest of the system has a string.
    """
    try:
        return isinstance(yaml.safe_load(value), str)
    except yaml.YAMLError:
        # Unparseable on its own means it certainly needs quoting.
        return False


def _replace_yaml_value(text: str, key: str, value: str) -> str:
    """Replace the value of the first ``<indent>key: <old>`` line surgically."""
    pattern = re.compile(
        rf"^(?P<indent>[ \t]*){re.escape(key)}:[ \t]*(?P<val>.*)$", re.M
    )

    def _repl(m: "re.Match[str]") -> str:
        old = m.group("val").strip()
        was_quoted = len(old) >= 2 and old[0] == old[-1] and old[0] in ("'", '"')
        # Preserve existing quoting and quote new values whenever parsing them
        # unquoted would change their type.
        newval = f'"{value}"' if was_quoted or not _reads_back_as_text(value) else value
        return f"{m.group('indent')}{key}: {newval}"

    new_text, n = pattern.subn(_repl, text, count=1)
    if n == 0:
        raise ValueError(f"key {key!r} not found in yaml text; refusing silent no-op")
    return new_text


def roll_session(cfg: dict) -> str:
    """Move ``cfg`` onto a freshly minted session, and return the new folder.

    Both halves of the session's identity move together: the folder's trailing
    uid and ``meta.session_uid``. The quarantine, aligned and report folders are
    derived from the folder name (``output_layout.counterpart``), so if the two
    disagreed the config could not say where its own episodes went.

    ``meta.session_uid`` is only written when the config already carries one. A
    hand-written CLI config has no session uid to keep in step, and inventing
    one here would hand ``RecorderManager`` a field it reads as a console-minted
    session.
    """
    rec = cfg.setdefault("recording", {})
    rolled = with_new_session_uid(str(rec.get("data_dir", "./data")).rstrip("/"))
    rec["data_dir"] = rolled
    meta = cfg.get("meta")
    if isinstance(meta, dict) and meta.get("session_uid"):
        meta["session_uid"] = session_uid_from_dir_name(rolled)
    return rolled


def persist_session(config_path: str, cfg: dict) -> None:
    """Best-effort write of the rolled session back into the working config.

    Never blocks recording: a config the station cannot rewrite is an install
    problem, and the episodes are already going to the right folder either way
    (the in-memory ``cfg`` is what this launch records from). The cost of
    failing is that the next launch rolls again, not that this one is lost.
    """
    meta = cfg.get("meta") or {}
    updates = [("data_dir", str((cfg.get("recording") or {}).get("data_dir", "")))]
    if meta.get("session_uid"):
        updates.append(("session_uid", str(meta["session_uid"])))
    try:
        with open(config_path) as f:
            text = f.read()
        for key, value in updates:
            try:
                text = _replace_yaml_value(text, key, value)
            except ValueError:
                # Do not add keys that a hand-written config does not already contain.
                print_yellow(f"Warning: no {key!r} key in {config_path} to update")
        tmp = f"{config_path}.tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, config_path)
    except OSError as e:
        print_yellow(f"Warning: could not persist rolled session to {config_path}: {e}")


def folder_has_episodes(folder: str) -> bool:
    """True if the session folder already holds a recorded (or in-progress) episode."""
    return bool(glob.glob(os.path.join(os.path.expanduser(folder), "episode_*")))


def slot_is_usable(folder: str, cfg: dict, check_config: Optional[str]) -> bool:
    """Usable if content matches, or the folder has no episodes yet.

    A seal outranks both. The rule above is about provenance (do the episodes
    in here come from the rig that is about to record?), and a sealed folder
    fails a different test entirely: somebody has already declared this session
    finished, and anything reading the marker takes that at its word. Recording
    into it would grow a session that is declared frozen.

    So the seal is checked first, and it is checked even for an empty folder.
    An empty sealed folder is not a fresh slot: the seal means the episodes
    were cleared (`clear_data`), and reusing the name would put new episodes
    under a session id already declared closed.

    Returning False is the whole remedy: `resolve_session_slot` and the two
    console call sites already respond to it by minting a fresh uid, which is
    the session-id refresh an operator sees.
    """
    if folder_is_sealed(folder):
        return False
    if not session_file_conflicts(os.path.expanduser(folder), cfg, check_config):
        return True
    return not folder_has_episodes(folder)


def resolve_session_slot(
    cfg: dict, check_config: Optional[str], config_path: str
) -> str:
    """Resolve the session folder for this launch.

    Rolls ``cfg`` onto a fresh session (a newly minted uid on the folder and
    on ``meta.session_uid``) when the current folder already holds episodes
    recorded under a different config/calibration. ``meta.task_name`` is left
    untouched.

    One roll, never a search: a uid does not collide, so there is nothing to
    walk and no attempt cap to fail against. The rolled folder is by
    construction one no session has used.

    Under the Station Console this is a backstop that normally never fires:
    ``station_ui.routes.config.apply`` rolls on Apply and
    ``station_ui.routes.record.start_recording`` rolls before launch, both so
    the page and the disk agree on the session id. It fires for
    ``record_data.py`` run from a shell, where there is no console to keep in
    step.
    """
    rec = cfg.setdefault("recording", {})
    # rstrip trailing slashes so the uid lands on the basename, not after a
    # slash (e.g. ".../cup/" -> ".../cup_<uid>", not ".../cup/_<uid>").
    raw_data_dir = str(rec.get("data_dir", "./data")).rstrip("/")

    if slot_is_usable(raw_data_dir, cfg, check_config):
        return os.path.expanduser(raw_data_dir)

    rolled = roll_session(cfg)
    persist_session(config_path, cfg)
    return os.path.expanduser(rolled)

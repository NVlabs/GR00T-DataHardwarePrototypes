# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session-slot resolution for data collection.

When a session folder already holds episodes recorded under a different
config / device calibration, bump an ``_N`` counter on the
session FOLDER (``recording.data_dir``) instead of mixing provenance or
raising a write-time lockout. ``meta.task_name`` is left untouched so the
PDX task prefix stays the operator's real task.
"""

from __future__ import annotations

import glob
import json
import os
import re
from copy import deepcopy
from typing import Optional

import yaml

from umi.common.utility.print_utils import print_yellow

# Cap on how far the ``_N`` counter walks before failing loudly.
_SESSION_SLOT_CAP = 50

# Snapshot the launch config before source expansion. Session conflict
# detection compares this snapshot, not the enriched recording config.
LAUNCH_CONFIG_NAME = "launch_config.yaml"


def config_yaml_bytes(config: Optional[dict]) -> bytes:
    """config.yaml bytes exactly as ``RecorderManager._save_config`` writes them."""
    return yaml.dump(config or {}, default_flow_style=False, sort_keys=False).encode()


# Non-acquisition options and accepted compatibility keys may change on
# resume without creating a new session.
SYNC_POLICY_KEYS: dict[str, tuple[str, ...]] = {
    "meta": ("data_portal_manifest",),
    "recording": (
        "auto_uploader",
        "sync_deletions_on_startup",
        "sync_deletions_max_age_hours",
        "sync_retry_delay",
        "upload_timeout",
    ),
    "alignment": (
        "enabled",
        "upload_aligned",
        "max_upload_retries",
        "align_timeout",
    ),
}


def _acquisition_config(config: Optional[dict]) -> dict:
    """Compare acquisition settings independently of sync policy and operator.

    Operator attribution remains in the saved config and each episode's snapshot;
    switching operators alone does not allocate a different session folder.
    """
    result = deepcopy(config or {})
    meta = result.get("meta")
    if isinstance(meta, dict):
        meta.pop("operator_name", None)
    for section, keys in SYNC_POLICY_KEYS.items():
        values = result.get(section)
        if isinstance(values, dict):
            for key in keys:
                values.pop(key, None)
            if not values:
                result.pop(section)
    return result


# Rewrite the device binding on every launch. It records the opened devices
# but does not participate in acquisition-config conflict checks.
DEVICE_BINDING_NAME = "device_binding.yaml"

# Prefix calibration filenames that collide with recorder-owned config names
# so bundling cannot overwrite session metadata.
RECORDER_FILE_NAMES = frozenset(
    {
        "config.yaml",
        LAUNCH_CONFIG_NAME,
        DEVICE_BINDING_NAME,
        "zarr.json",
    }
)


def device_config_entries(
    config: Optional[dict], base_dir: Optional[str] = None
) -> list[tuple[str, str]]:
    """``(abs_source_path, basename)`` for every device config referenced by any
    source in ``config`` (gripper / exoskeleton ``params.config_file``, etc.)
    and by the active ``robots[robot_type]`` entry (an eval's arm server YAML).

    Relative paths resolve against ``base_dir`` (default: the current
    directory, which is the repo root for ``record_data.py``)."""
    if not config:
        return []
    out: list[tuple[str, str]] = []
    seen_paths: set[str] = set()
    seen_basenames: dict[str, str] = {}

    def add_path(raw_path: str) -> None:
        path = os.path.abspath(
            os.path.join(base_dir or "", os.path.expanduser(raw_path))
        )
        if path in seen_paths:
            return
        seen_paths.add(path)
        basename = os.path.basename(path)
        if basename in RECORDER_FILE_NAMES:
            basename = f"{os.path.basename(os.path.dirname(path))}_{basename}"
        if seen_basenames.get(basename, path) != path:
            print_yellow(
                f"Warning: source configs share basename {basename!r} "
                f"({seen_basenames[basename]} vs {path}); bundling only the first"
            )
            return
        seen_basenames[basename] = path
        out.append((path, basename))

    sources = [
        src for value in config.values() if isinstance(value, list) for src in value
    ]
    robots = config.get("robots")
    if isinstance(robots, dict):
        sources.append(robots.get(config.get("robot_type")))
    for src in sources:
        if not isinstance(src, dict):
            continue
        cfg_file = (src.get("params") or {}).get("config_file")
        if not cfg_file:
            continue
        add_path(str(cfg_file))

    return out


def save_launch_config(record_path: Optional[str], config: Optional[dict]) -> None:
    """Write the raw (as-loaded) config as the session's config reference.

    Call once the session folder is resolved and BEFORE the config is expanded,
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
    base_dir: Optional[str] = None,
) -> list[str]:
    """Session content files already in ``record_path`` that DIFFER from what this
    launch would use, else ``[]``.

    Compares:

    - ``launch_config.yaml`` — raw config-file snapshot
    - each device-calibration file — stored session copy vs live source

    A gripper / exoskeleton recalibration (content change) is a conflict. ``[]``
    means the folder holds the same config + calibration (safe to resume). The
    launcher skips a conflicting folder that already holds episodes in favor of
    a fresh ``_N``. ``base_dir`` resolves relative device-config paths (see
    ``device_config_entries``).
    """
    if not record_path or not os.path.isdir(record_path):
        return []
    conflicts: list[str] = []

    launch = os.path.join(record_path, LAUNCH_CONFIG_NAME)
    if os.path.isfile(launch):
        with open(launch, "rb") as f:
            if _acquisition_config(yaml.safe_load(f)) != _acquisition_config(config):
                conflicts.append(LAUNCH_CONFIG_NAME)

    for src_path, basename in device_config_entries(config, base_dir):
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


_MISSING = object()


def _by_name(items) -> Optional[dict]:
    """``{name: item}`` for a list of uniquely named dicts (the ``*_sources``), else None."""
    if not isinstance(items, list):
        return None
    named = {item.get("name"): item for item in items if isinstance(item, dict)}
    if None in named or len(named) != len(items):
        return None
    return named


def _changed_paths(old, new, path: str = "") -> list[str]:
    if old == new:
        return []
    if isinstance(old, dict) and isinstance(new, dict):
        children = {f".{key}": (old, new, key) for key in set(old) | set(new)}
    else:
        old_named, new_named = _by_name(old), _by_name(new)
        if old_named is None or new_named is None:
            return [path]
        old, new = old_named, new_named
        children = {f"[{name}]": (old, new, name) for name in set(old) | set(new)}
    changed = []
    for suffix, (a, b, key) in sorted(children.items(), key=lambda item: item[0]):
        child = f"{path}{suffix}".lstrip(".")
        a, b = a.get(key, _MISSING), b.get(key, _MISSING)
        if a is _MISSING or b is _MISSING:
            changed.append(child)
        else:
            changed += _changed_paths(a, b, child)
    return changed or [path]  # e.g. the same sources in another order


def launch_config_changes(
    record_path: Optional[str], config: Optional[dict]
) -> list[str]:
    """Config paths where this launch differs from ``record_path``'s launch_config.yaml.

    The comparison session_file_conflicts makes (sync-policy and operator ignored),
    itemized so a launcher can say what sends it to a new ``_N`` folder, e.g.
    ``meta.task_description`` or ``video_sources[head_umi_camera]``. ``[]`` when the
    snapshot is absent or equal.
    """
    if not record_path:
        return []
    launch = os.path.join(os.path.expanduser(record_path), LAUNCH_CONFIG_NAME)
    if not os.path.isfile(launch):
        return []
    with open(launch, "rb") as f:
        stored = yaml.safe_load(f)
    return _changed_paths(_acquisition_config(stored), _acquisition_config(config))


def folder_stem_and_counter(data_dir: str, task_name: Optional[str]) -> tuple[str, int]:
    """Split the session folder into its stem and a trailing ``_<int>`` counter.

    Anchoring on ``task_name`` keeps the counter off the task portion, so a real
    task whose name ends in ``_N`` is never mistaken for a counter.
    """
    tn = str(task_name) if task_name else None
    if tn and data_dir.endswith(tn):
        return data_dir, 0
    m = re.match(r"^(?P<stem>.+)_(?P<n>\d+)$", data_dir)
    if m and (tn is None or m.group("stem").endswith(tn)):
        return m.group("stem"), int(m.group("n"))
    return data_dir, 0


def with_counter(name: str, n: int) -> str:
    return name if n <= 0 else f"{name}_{n}"


def _replace_yaml_value(text: str, key: str, value: str) -> str:
    """Replace the value of the first ``<indent>key: <old>`` line surgically."""
    pattern = re.compile(
        rf"^(?P<indent>[ \t]*){re.escape(key)}:[ \t]*(?P<val>.*)$", re.M
    )

    def _repl(m: "re.Match[str]") -> str:
        old = m.group("val").strip()
        quoted = len(old) >= 2 and old[0] == old[-1] and old[0] in ("'", '"')
        newval = f'"{value}"' if quoted else value
        return f"{m.group('indent')}{key}: {newval}"

    new_text, n = pattern.subn(_repl, text, count=1)
    if n == 0:
        raise ValueError(f"key {key!r} not found in yaml text; refusing silent no-op")
    return new_text


def _persist_session_names(
    config_path: str, task_name: Optional[str], data_dir: str
) -> None:
    """Best-effort write of bumped ``data_dir`` (and optional task_name) into the
    working config. Never blocks recording."""
    try:
        with open(config_path) as f:
            text = f.read()
        if task_name is not None:
            text = _replace_yaml_value(text, "task_name", task_name)
        text = _replace_yaml_value(text, "data_dir", data_dir)
        tmp = f"{config_path}.tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, config_path)
    except Exception as e:  # noqa: BLE001 — persistence is best-effort
        print_yellow(
            f"Warning: could not persist session counter to {config_path}: {e}"
        )


def folder_has_episodes(folder: str) -> bool:
    """True if the session folder already holds a recorded (or in-progress) episode."""
    return bool(glob.glob(os.path.join(os.path.expanduser(folder), "episode_*")))


def folder_has_episode_history(folder: str) -> bool:
    """An emptied historical session still owns its acquisition configuration."""
    if folder_has_episodes(folder):
        return True
    for name in ("zarr.json", ".zattrs"):
        path = os.path.join(os.path.expanduser(folder), name)
        if not os.path.exists(path):
            continue
        try:
            with open(path) as stream:
                metadata = json.load(stream)
            attrs = metadata.get("attributes", {}) if name == "zarr.json" else metadata
            if any(attrs.get(key) for key in ("num_episodes", "episode_lengths")):
                return True
        except (OSError, ValueError, TypeError, AttributeError):
            # Unreadable history cannot authorize overwriting old provenance.
            return True
    return False


def slot_is_usable(
    folder: str,
    cfg: dict,
    base_dir: Optional[str] = None,
) -> bool:
    """Usable if content matches, or the folder has never held an episode."""
    if not session_file_conflicts(os.path.expanduser(folder), cfg, base_dir):
        return True
    return not folder_has_episode_history(folder)


def find_session_slot(cfg: dict, base_dir: Optional[str] = None) -> Optional[str]:
    """The folder ``resolve_session_slot`` records into (as written, ``~`` kept).

    ``None`` when no slot within the cap is free. Works on a copy of ``cfg`` and
    writes nothing, so a launcher can show the folder before the recorder starts.
    """
    cfg = deepcopy(cfg)
    rec = cfg.setdefault("recording", {})
    meta = cfg.setdefault("meta", {})
    # rstrip trailing slashes so the counter appends to the basename, not after a
    # slash (e.g. ".../cup/" -> ".../cup_1", not ".../cup/_1").
    raw_data_dir = str(rec.get("data_dir", "./data")).rstrip("/")
    if slot_is_usable(raw_data_dir, cfg, base_dir):
        return raw_data_dir
    stem, n_dir = folder_stem_and_counter(raw_data_dir, meta.get("task_name"))
    for n in range(n_dir + 1, n_dir + 1 + _SESSION_SLOT_CAP):
        cand_dir = with_counter(stem, n)
        rec["data_dir"] = cand_dir
        if slot_is_usable(cand_dir, cfg, base_dir):
            return cand_dir
    return None


def resolve_session_slot(cfg: dict, config_path: str) -> str:
    """Resolve the session folder for this launch.

    Bumps an ``_N`` counter on the FOLDER when the current folder already holds
    episodes recorded under a different config/calibration. ``meta.task_name``
    is left untouched.
    """
    rec = cfg.setdefault("recording", {})
    cfg.setdefault("meta", {})
    raw_data_dir = str(rec.get("data_dir", "./data")).rstrip("/")
    slot = find_session_slot(cfg)
    if slot == raw_data_dir:
        return os.path.expanduser(raw_data_dir)
    if slot is not None:
        rec["data_dir"] = slot
        _persist_session_names(config_path, None, slot)
        return os.path.expanduser(slot)

    rec["data_dir"] = raw_data_dir
    raise SystemExit(
        f"Could not find a free session slot after {_SESSION_SLOT_CAP} attempts "
        f"from {raw_data_dir!r}. A config field may be changing every launch, or "
        "record into a fresh task."
    )

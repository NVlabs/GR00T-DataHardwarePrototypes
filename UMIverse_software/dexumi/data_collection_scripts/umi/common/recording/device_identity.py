# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which physical device stood behind each recorded source.

Every source may report two identities (``SensorInterface.identity()`` /
``observed_identity()``): the device its config expects, and the device its
SDK actually opened. This module turns them into

- ``device_binding.yaml`` at the session root: the launch's binding of each
  recorded source to a device (written once every device is open, overwritten
  on every launch; never part of the session-slot comparison);
- the per-episode ``device_identity`` attr (identities re-read at save time);
- the launch rule: when the DC GUI hands over its Identify scan
  (``record_data.py --device-binding``), a scanned device that differs from the
  device actually opened refuses the launch.

A source without an identity (``identity()`` is None) is unbound and skipped
everywhere, so rigs and sources without identity support behave as before.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Iterable, Optional

import yaml

from umi.common.recording.session_slot import DEVICE_BINDING_NAME
from umi.utils.device_manager import normalize_device_id

DEVICE_BINDING_FILE = DEVICE_BINDING_NAME
DEVICE_BINDING_SCHEMA = 1
BOUND_BY_SCAN = "identify_scan"
BOUND_BY_CONFIG = "config"

# Remove mock-only identity parameters before passing arguments to source constructors.
MOCK_IDENTITY_PARAMS = (
    "mock_device_id",
    "mock_observed_device_id",
    "mock_swap_after_reads",
)

_SLOTS = ("left", "right", "head")


def slot_for_source(name: Optional[str]) -> Optional[str]:
    """left | right | head from the source-name prefix, else None."""
    prefix = str(name or "").split("_", 1)[0].lower()
    return prefix if prefix in _SLOTS else None


def _report(source, attr: str) -> Optional[dict]:
    """``source.<attr>()``; a missing, raising or malformed report counts as None.

    The attribute lookup is guarded too: a source whose ``identity`` is a
    property that raises is one unbound source, not a failed episode.
    """
    try:
        fn = getattr(source, attr, None)
        if not callable(fn):
            return None
        value = fn()
    except Exception:  # noqa: BLE001 - identity is provenance, never a crash
        return None
    return value if isinstance(value, dict) else None


def recorded_sources(recorders: Iterable) -> list[tuple[str, object]]:
    """``(recorded group name, source)`` of every source of ``recorders``.

    The group name is how the source appears in an episode zarr
    (``device_name``). Recorders without ``data_sources`` (test doubles)
    contribute nothing.
    """
    out = []
    for recorder in recorders or ():
        for source in getattr(recorder, "data_sources", None) or ():
            name = getattr(source, "device_name", None)
            if not name:
                continue
            out.append((name, source))
    return out


def _clean(value):
    """JSON/YAML-safe scalar for an identity field."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def source_identity(source) -> Optional[dict]:
    """``{kind, name, id, observed_id, ...extras}`` of one source, or None.

    The observed report is re-read on every call.
    """
    expected = _report(source, "identity")
    if expected is None:
        return None
    entry: dict = dict(
        kind=_clean(expected.get("kind")),
        name=_clean(expected.get("name")),
        id=normalize_device_id(expected.get("id")),
    )
    observed = _report(source, "observed_identity") or {}
    entry["observed_id"] = normalize_device_id(observed.get("id"))
    for key, value in observed.items():
        if key != "id" and key not in entry:
            entry[key] = _clean(value)
    return entry


def collect_identities(recorders: Iterable) -> dict:
    """``{recorded name: identity entry}`` for every source with an identity."""
    out = {}
    for name, source in recorded_sources(recorders):
        try:
            entry = source_identity(source)
        except Exception:  # noqa: BLE001 - one bad source never drops the others
            entry = None
        if entry is not None:
            out[name] = entry
    return out


# ---------------------------------------------------------------------------
# DC GUI scan input (record_data.py --device-binding)
# ---------------------------------------------------------------------------


def load_scan_binding(path: str) -> dict:
    """Read and validate the DC GUI's ``--device-binding`` yaml.

    ``{schema: 1, gui_run: str, scanned: {<config source name>: {kind, name,
    id, scanned_at?}}}``. Raises ValueError on anything else so a malformed
    hand-over fails the launch instead of binding nothing silently.
    """
    with open(os.path.expanduser(path)) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: device binding must be a mapping")
    if data.get("schema") != DEVICE_BINDING_SCHEMA:
        raise ValueError(
            f"{path}: unsupported device binding schema {data.get('schema')!r} "
            f"(expected {DEVICE_BINDING_SCHEMA})"
        )
    scanned = data.get("scanned") or {}
    if not isinstance(scanned, dict) or not all(
        isinstance(entry, dict) for entry in scanned.values()
    ):
        raise ValueError(f"{path}: 'scanned' must map source names to mappings")
    gui_run = data.get("gui_run")
    return {
        "schema": DEVICE_BINDING_SCHEMA,
        "gui_run": None if gui_run is None else str(gui_run),
        "scanned": {str(name): dict(entry) for name, entry in scanned.items()},
    }


def build_device_binding(
    identities: dict,
    config_sources: Optional[dict] = None,
    scan: Optional[dict] = None,
    *,
    launch_id: Optional[str] = None,
    bound_at: Optional[str] = None,
) -> tuple[dict, list[str], list[str]]:
    """The launch's binding plus ``(mismatches, notes)``.

    ``identities`` = ``collect_identities(...)``; ``config_sources`` maps each
    recorded name to the recording-config source it was built from (scan keys
    are config names, e.g. ``left_dexumi_exoskeleton`` for its ``_encoder`` and
    ``_tactile`` streams). A scanned id is compared with the observed id only
    when both exist and the scan's kind (if any) is the source's kind; only a
    definite difference is a mismatch (it refuses the launch). Anything that
    could not be verified is a note and leaves the binding ``bound_by: config``.
    """
    config_sources = config_sources or {}
    scanned = (scan or {}).get("scanned") or {}
    sources: dict = {}
    mismatches: list[str] = []
    notes: list[str] = []
    used: set[str] = set()
    for name, ident in identities.items():
        entry = {
            "slot": slot_for_source(name),
            "kind": ident.get("kind"),
            "name": ident.get("name"),
            "id": ident.get("id"),
            "observed_id": ident.get("observed_id"),
        }
        key = name if name in scanned else config_sources.get(name)
        scan_entry = scanned.get(key) if key is not None else None
        if scan_entry is not None:
            used.add(key)
            scanned_id = normalize_device_id(scan_entry.get("id"))
            scan_kind = scan_entry.get("kind")
            observed = entry["observed_id"]
            if scanned_id is None:
                notes.append(f"{name}: the scan of {key!r} carries no device id")
            elif scan_kind and scan_kind != entry["kind"]:
                notes.append(
                    f"{name}: scanned as a {scan_kind} device but recorded from a "
                    f"{entry['kind']} source; not compared"
                )
            else:
                entry["scanned_id"] = scanned_id
                if observed is None:
                    notes.append(
                        f"{name}: the opened device did not report an id; "
                        f"scan ({scanned_id}) not verified"
                    )
                elif observed != scanned_id:
                    mismatches.append(
                        f"{name}: the device behind this slot changed since the "
                        f"Identify scan (scanned {scanned_id}, opened {observed})"
                    )
        sources[name] = entry
    for key in sorted(set(scanned) - used):
        notes.append(
            f"scanned source {key!r} is not a recorded source with an identity"
        )
    matched = bool(scanned) and not mismatches and not notes
    binding = {
        "schema": DEVICE_BINDING_SCHEMA,
        "launch_id": launch_id or uuid.uuid4().hex,
        "bound_at": bound_at
        or datetime.now().astimezone().isoformat(timespec="seconds"),
        "bound_by": BOUND_BY_SCAN if scan is not None and matched else BOUND_BY_CONFIG,
        "gui_run": (scan or {}).get("gui_run"),
        "sources": sources,
    }
    return binding, mismatches, notes


def binding_yaml_bytes(binding: dict) -> bytes:
    return yaml.safe_dump(binding, default_flow_style=False, sort_keys=False).encode()


def write_device_binding(directory: str, binding: dict) -> str:
    """Write ``device_binding.yaml`` into ``directory`` atomically; return its path."""
    path = os.path.join(directory, DEVICE_BINDING_FILE)
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(binding_yaml_bytes(binding))
    os.replace(tmp, path)
    return path


def config_source_names(recorders: Iterable) -> dict:
    """``{recorded name: recording-config source name}`` where the factory set it."""
    return {
        name: source.config_source_name
        for name, source in recorded_sources(recorders)
        if getattr(source, "config_source_name", None)
    }


# ---------------------------------------------------------------------------
# Mock sources: synthetic identity (and a swap hook for tests)
# ---------------------------------------------------------------------------


def split_mock_identity_params(source_config: dict) -> tuple[dict, dict]:
    """``(source_config without the mock identity params, those params)``.

    The returned config is a shallow copy whose ``params`` dropped the
    ``MOCK_IDENTITY_PARAMS``; the caller's dict is untouched, so the saved
    config keeps them as provenance.
    """
    params = dict(source_config.get("params") or {})
    knobs = {key: params.pop(key) for key in MOCK_IDENTITY_PARAMS if key in params}
    if not knobs:
        return source_config, {}
    stripped = dict(source_config)
    stripped["params"] = params
    return stripped, knobs


class MockIdentity:
    """Identity of a mock source: ``MOCK-<SOURCE NAME>`` unless configured.

    ``mock_observed_device_id`` (TEST HOOK) makes the source report another
    device than it is bound to, simulating a swapped device. With
    ``mock_swap_after_reads: N`` the first N observed reads of a stream still
    report the configured id, so a launch binds normally and a later read
    (an episode's saved ``device_identity``) shows the swap.
    """

    def __init__(self, source_name: str, knobs: Optional[dict] = None) -> None:
        knobs = knobs or {}
        self.name = source_name
        self.id = normalize_device_id(knobs.get("mock_device_id")) or (
            f"MOCK-{str(source_name).upper()}"
        )
        self.swapped_id = normalize_device_id(knobs.get("mock_observed_device_id"))
        swap_after = knobs.get("mock_swap_after_reads")
        self.swap_after = None if swap_after is None else int(swap_after)

    def identity(self) -> dict:
        return {"kind": "mock", "name": self.name, "id": self.id}

    def observed_identity_at(self, read_number: int) -> dict:
        swapped = self.swapped_id is not None and (
            self.swap_after is None or read_number > self.swap_after
        )
        return {"id": self.swapped_id if swapped else self.id}


def attach_mock_identity(sources: Iterable, source_name: str, knobs: dict) -> None:
    """Give every stream of one mock source entry the entry's identity.

    A mock exoskeleton's encoder and tactile streams are one device, so they
    share one ``MockIdentity``; each stream counts its own observed reads, so
    with ``mock_swap_after_reads`` they flip on the same read. Bound as instance
    attributes so it applies whatever the stream class.
    """
    mock = MockIdentity(source_name, knobs)

    def observed_reader():
        reads = 0

        def read() -> dict:
            nonlocal reads
            reads += 1
            return mock.observed_identity_at(reads)

        return read

    for source in sources:
        try:
            source.identity = mock.identity
            source.observed_identity = observed_reader()
        except AttributeError:
            continue

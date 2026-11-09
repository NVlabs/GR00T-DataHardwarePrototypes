# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DcSession: the operator's last form and plan, persisted to <home>/session.json.

A restarted GUI shows the last inputs and plan again.
"""

import dataclasses
import json
import math
import os
import tempfile
import time

from gui.dc_ui.core.kinds import DcForm, Finding, LaunchSpec

SESSION_FILE = "session.json"
_FIELDS = ("form", "data_dir", "config_path", "spec", "findings", "detection")


def json_safe(value):
    """Keep blocked YAML configurations representable in strict JSON.

    The original YAML and preflight findings retain invalid numeric values;
    JSON-facing previews use null for NaN/infinity rather than failing to load.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def _construct(cls, data):
    if not isinstance(data, dict):
        return None
    allowed = {f.name for f in dataclasses.fields(cls)}
    if set(data) - allowed:
        return None
    try:
        return cls(**data)
    except TypeError:
        return None


class DcSession:
    def __init__(self, home):
        self.home = str(home)
        self.form = None
        self.data_dir = None
        self.config_path = None
        self.spec = None
        self.findings = []
        self.detection = None
        self.saved_at = None

    @property
    def path(self):
        return os.path.join(self.home, SESSION_FILE)

    def update(self, **fields):
        for key, value in fields.items():
            if key not in _FIELDS:
                raise KeyError(f"unknown session field {key!r}")
            setattr(self, key, value)
        return self

    def blocked(self):
        return any(f.level == "error" for f in self.findings or [])

    def snapshot(self):
        return json_safe(
            {
                "form": self.form.to_dict() if self.form else None,
                "data_dir": self.data_dir,
                "config_path": self.config_path,
                "spec": self.spec.to_dict() if self.spec else None,
                "findings": [f.to_dict() for f in self.findings or []],
                "blocked": self.blocked(),
                "detection": self.detection,
                "saved_at": self.saved_at,
            }
        )

    def save(self):
        self.saved_at = time.time()
        os.makedirs(self.home, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".session-", suffix=".tmp", dir=self.home)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.snapshot(), f, indent=2)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return self.path

    @classmethod
    def load(cls, home):
        session = cls(home)
        try:
            with open(session.path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return session
        if not isinstance(data, dict):
            return session
        session.form = _construct(DcForm, data.get("form"))
        session.data_dir = data.get("data_dir")
        session.config_path = data.get("config_path")
        session.spec = _construct(LaunchSpec, data.get("spec"))
        session.findings = [
            f
            for f in (_construct(Finding, item) for item in data.get("findings") or [])
            if f is not None
        ]
        detection = data.get("detection")
        session.detection = detection if isinstance(detection, dict) else None
        session.saved_at = data.get("saved_at")
        return session

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One safe YAML loader, using libyaml when the build has it.

`yaml.safe_load` always uses PyYAML's pure-Python SafeLoader, even on a build
that shipped the libyaml bindings: the C loader is opt-in, and the opt-in is
passing `Loader=` yourself. On one rig template that is 4.03ms against 0.46ms,
about nine times the cost for the same result.

That matters because the console re-reads config on every render, so parsing
time adds up across a busy page.

`CSafeLoader` accepts the same YAML 1.1 safe subset and builds the same plain
dicts and lists, so this is a swap of implementation and not of behaviour. The
fallback keeps a build without the C extension working, just more slowly.

For round-trip editing (reading a file, changing it, writing it back with its
comments) this is the wrong tool: use ruamel, as `config_assembler` does.
"""

from __future__ import annotations

import yaml

try:  # pragma: no cover - depends on how PyYAML was built, not on our code
    from yaml import CSafeLoader as SafeLoader
except ImportError:  # pragma: no cover
    from yaml import SafeLoader  # type: ignore[assignment]

__all__ = ["SafeLoader", "safe_load"]


def safe_load(text: str):
    """`yaml.safe_load`, on the fastest loader this build offers."""
    return yaml.load(text, Loader=SafeLoader)

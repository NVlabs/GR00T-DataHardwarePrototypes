# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rolling the console onto a fresh session when the config drifted.

`umi.common.recording.session_slot` states the rule and owns the remedy; this
module is how the console applies it, and it exists for one reason: the draft
is the page's source of truth for the session id, and `record_data.py` rolling
the config on its own leaves the two disagreeing. `routes/config._reopen` is
gated on an untouched draft (deliberately, so a reopen cannot discard work in
progress), so it will never pick a rolled uid back up mid-session. The console
has to do the rolling itself, or the page shows session A while the episodes
land in session B.

Two call sites, and they are not redundant:

* **Apply** (`routes/config`): catches everything the operator changed on the
  Configuration page itself.
* **Start** (`routes/record`): catches drift that appeared after the last
  Apply. Calibration is step 4 and Configuration is step 2, so re-zeroing a
  gripper changes a bundled device YAML with no Apply in between. Without this
  one, the most common drift of all reaches the recorder unrolled.

Apply rolls the draft and lets the assembler write the new folder out. Start
rolls the file, because by then the config has been assembled and a session
roll is only ever a two-field edit (`recording.data_dir` and
`meta.session_uid`): nothing else in a generated config depends on the uid,
so re-running the assembler at the point of launch would buy nothing and could
fail on a gate the operator already passed.
"""

from __future__ import annotations

import logging

import yaml

from station_ui import fast_yaml
from umi.common.recording.session_slot import (
    persist_session,
    roll_session,
    slot_is_usable,
)

logger = logging.getLogger(__name__)

__all__ = ["config_drifted", "roll_generated_config"]


def config_drifted(config_path, check_config: str, data_dir: str) -> bool:
    """Whether recording into ``data_dir`` would mix two configurations.

    `session_slot.slot_is_usable` is the question (it states the rule, expands
    the folder and compares all three artefacts), so it is called rather than
    re-derived. It is asked of the file the recorder will actually read rather
    than of the draft: the draft cannot produce the config bytes the comparison
    is made of, and a second opinion assembled here is exactly the kind of
    near-copy that drifts from what the recorder does.

    False whenever the answer cannot be established (an unreadable or absent
    config, a folder that does not exist yet). This gates an automatic rename of
    the operator's session folder, and rolling on a question nobody could answer
    would fork a session over a transient read error.
    """
    cfg = _load(config_path)
    if cfg is None or not data_dir:
        return False
    try:
        return not slot_is_usable(data_dir, cfg, check_config)
    except OSError:
        return False


def roll_generated_config(config_path, check_config: str, draft) -> str | None:
    """Roll the generated config onto a fresh session. The new folder, or None.

    None means nothing was wrong: the folder is empty, or it already holds this
    exact configuration, which is the normal per-episode case and must stay a
    no-op.

    `draft.session_uid` is moved in step with the file; otherwise the page and
    the recorder would name different sessions.
    """
    data_dir = str((_load(config_path) or {}).get("recording", {}).get("data_dir", ""))
    if not data_dir or not config_drifted(config_path, check_config, data_dir):
        return None

    cfg = _load(config_path)
    if cfg is None:
        return None
    # Give the roll a uid to move: a generated config always carries one, but a
    # hand-edited file may not, and the draft's is the console's own answer.
    cfg.setdefault("meta", {}).setdefault("session_uid", draft.session_uid)
    rolled = roll_session(cfg)
    persist_session(str(config_path), cfg)
    draft.session_uid = (cfg.get("meta") or {}).get("session_uid", draft.session_uid)
    logger.info("config drifted; rolled session onto %s", rolled)
    return rolled


def _load(config_path) -> dict | None:
    """The generated config as a mapping, or None if it cannot be read.

    None is the answer to every failure here -- absent, unreadable, hand-edited
    into invalid YAML, or not a mapping at all -- because every caller treats it
    as "do not roll", and that is the safe direction.
    """
    if config_path is None:
        return None
    try:
        loaded = fast_yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError):
        return None
    return loaded if isinstance(loaded, dict) else None

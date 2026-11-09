# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rebuild the session draft from the artefact the station records with.

The draft lives in `app.state.rig`, so without this a console restart would
discard Setup and Configuration while the file those two steps produce is
still sitting in the station's state dir.

**Nothing new is persisted.** The generated config is already the source of
truth for what was applied: Apply writes it, the Configuration tick in
`app._shell_context` keys on it existing, `_disarm` deletes it when the
operator leaves a session, and it is the artefact the recorder is handed. A
second state file beside it could disagree about which config this rig records
with, and two answers to that question would be worse than not restoring at
all.

The file does not carry the preset id, but its path does:
`ConfigAssembler.output_path` writes `data_recording/{preset_id}.yaml` under
the state dir. So the one
field `config_form.hydrate` cannot recover is read off the path, and
everything else comes out of the file through the same `hydrate` that `/config`
already uses for "reopen". Restoring is therefore exactly the reopen the
operator would have got by visiting `/config`, done one page earlier.

Three consequences worth stating, because each was a choice:

* **This is a read.** Two consoles on two ports share one checkout, and a
  restore that wrote its own copy of the draft would have them racing to
  write it. They both read one artefact instead, so they agree without
  coordinating and there is nothing to race over.
* **The uid and the recording date come back unchanged.** Both are baked into
  `recording.data_dir` in the file, which is the folder the recorder writes
  into. Minting a fresh uid would fork a half-recorded session across two
  `output/raw/{task}/{session}/` folders, and re-dating it would file the same
  episodes under two days, with the console showing one folder and the
  recorder using another. `start_new_session` is the operator's deliberate way
  to move on, and `_disarm` removes the config along with it. A console reopened
  the next morning therefore restores yesterday's session, and `day_roll` moves
  it on to today before the first page renders; restoring it stale first is
  what lets that roll carry the whole rig forward instead of starting blank.
* **Nothing about the hardware is restored.** Calibration results are claims
  about what is plugged in right now and live only in this process, so a
  restored draft cannot resurrect a tick for a rig that may have been unplugged
  between the two processes.
"""

from __future__ import annotations

import logging
import pathlib

import yaml

from station_ui import config_form
from station_ui.core.config_assembler import GENERATED_DIR
from station_ui.core.setup_presets import preset_ids
from station_ui.session import SessionDraft

logger = logging.getLogger(__name__)

__all__ = ["restore"]


def _candidates(state_dir) -> list[tuple[float, str, str]]:
    """Every applied config this version can still make sense of, newest first.

    The preset half of the path is checked against what this version declares,
    and that check is not defensive padding: an unknown preset puts an id
    nothing declares into the draft, and every page that resolves a preset then
    renders its "that setup is gone" path, on a console the operator never told
    to choose it. The honest answer is that there is no session to restore,
    which is the empty draft the caller falls back to.
    """
    root = pathlib.Path(state_dir).joinpath(*GENERATED_DIR)
    known = frozenset(preset_ids())
    found = []
    try:
        for path in sorted(root.glob("*.yaml")):
            if path.stem not in known:
                continue
            found.append((path.stat().st_mtime, str(path), path.stem))
    except OSError:
        # No generated directory at all is the common case -- a fresh checkout
        # -- and an unreadable one is a station install problem. Neither is
        # worth failing the boot over.
        return []
    # Newest first, and the path breaks a tie so two files written in the same
    # instant restore the same session on every process that reads them.
    return sorted(found, key=lambda item: (item[0], item[1]), reverse=True)


def restore(state_dir) -> SessionDraft | None:
    """The session this station is set up to record, or None.

    None rather than an empty draft: the caller decides what "no session"
    looks like, currently a plain `SessionDraft()`.

    Only the newest candidate is considered. Falling back to an older one
    would quietly restore a different setup (a preset the operator used last
    week), and a console showing the wrong rig is worse than a console showing
    none. It also bounds the one race there is: a second server booting in the
    microseconds while this one's Apply is mid-write reads a half-written file,
    fails here, and starts empty, as if there were nothing to restore.
    """
    for _mtime, name, preset_id in _candidates(state_dir)[:1]:
        path = pathlib.Path(name)
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, yaml.YAMLError) as exc:
            logger.warning("could not read the applied config %s: %s", path, exc)
            return None
        if not isinstance(cfg, dict):
            logger.warning("the applied config %s is not a YAML mapping", path)
            return None
        draft = SessionDraft(preset_id=preset_id)
        try:
            config_form.hydrate(draft, cfg)
        except Exception:  # noqa: BLE001 - a boot path must not be able to fail
            # Hydration tolerates persisted schema differences. If restoration fails,
            # start with an editable empty draft rather than preventing server startup.
            logger.warning(
                "the applied config %s could not be read into a session",
                path,
                exc_info=True,
            )
            return None
        logger.info(
            "restored the applied session %s (%s)", draft.session_uid, preset_id
        )
        return draft
    return None

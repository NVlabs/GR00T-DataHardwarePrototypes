# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Move an idle session onto today, without asking the operator to.

`draft.date` is the first ten characters of the session folder, and that is
how the audit finds a day's sessions. It is set at draft creation and by
`session.start_new_session`, so a station left running past midnight (or a
console reopened the next morning, since `session_store.restore` deliberately
brings the date back unchanged) carries yesterday's date until somebody acts.
A console reopened on a session whose folder still carries a shift label after
the date is moved on the same way, whatever the date
(`SessionDraft.legacy_folder`).

The console moves such a session onto today by itself, and shows a banner
saying it happened. Three things make that safe to do without asking, and
each is a condition rather than a hope:

* **Never while a recorder is alive.** A running recorder writes into the
  folder named in the file it was handed; rolling under it would leave the
  console showing one folder and the recorder using another, the exact split
  `session_store` refuses to create. A live recorder keeps the stale-day
  warning instead, which is the one case where the operator genuinely has to
  decide.
* **The config is re-stamped, not deleted.** The generated config is this
  station's only persistence (`session_store` reads the draft back out of it),
  so deleting it (what the manual button does through `next_session.disarm`)
  would mean a console restarted after an overnight roll came up with Setup
  and Configuration blank. Only the two keys that name the session move; see
  `config_assembler.restamp_session` for why it is not a re-assemble.
* **Nothing about the rig is touched.** The calibration is a claim about
  hardware that did not move overnight, and `device_fingerprint` excludes the
  date and the uid precisely so a new session does not expire it. The manual
  button works the same way.

Episodes recorded before the roll stay exactly where they are, under the day
they were recorded on. That is the whole point: each day's episodes file
under the day that actually produced them.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

import yaml as plain_yaml
from starlette.requests import Request

from station_ui import recording
from station_ui import seal as seal_state
from station_ui import session as session_state
from station_ui.core.config_assembler import (
    ConfigAssembler,
    SessionSpec,
    restamp_session,
    session_identity,
)
from station_ui.errors import FieldError

logger = logging.getLogger(__name__)

__all__ = ["DayRoll", "Rolled", "notice", "roll_if_new_day"]


@dataclass(frozen=True)
class Rolled:
    """What the operator is told after the console rolled the session itself."""

    from_date: str
    to_date: str
    left_behind: str
    had_episodes: bool


def _generated_path(app, draft):
    """Where this draft's applied config lives, whether or not it exists.

    The same question `routes/config._generated_path` asks, asked through the
    same `output_path` -- but from a module the routes import, rather than the
    other way round. `output_path` reads only the preset, so the placeholder
    uid keeps SessionSpec's own invariant intact without this
    depending on it not looking.
    """
    if not draft.preset_id:
        return None
    settings = app.state.settings
    assembler = ConfigAssembler(settings.repo_root, state_dir=settings.state_dir)
    return assembler.output_path(
        SessionSpec(
            preset_id=draft.preset_id,
            operator_name="",
            task_name="",
            session_uid="0" * 12,
            data_dir="",
        )
    )


def _rearm(app, draft, left_behind: str) -> None:
    """Point the applied config at the session that just started.

    The station records from this file, not from the page (`routes/record.py:
    preflight` reads `recording.data_dir` out of it). Left carrying the old
    folder, the new session's first episode would land under yesterday's id
    while every page showed a different one, and nothing anywhere would say the
    two disagreed.

    Only ever touches a file that carries the uid being left, exactly as
    `next_session.disarm` does: a config another console applied for some other
    session is
    not this roll's to move.

    Every failure ends the same way: the file is removed. A config that could
    not be re-stamped is a config still armed for yesterday, and the
    Configuration step unticking itself (`app._shell_context` keys the tick on
    the file existing) is what "apply this session before recording it" looks
    like. That costs the operator an Apply; being handed the wrong folder costs
    a day's recordings.
    """
    path = _generated_path(app, draft)
    if path is None or not path.is_file():
        return
    try:
        cfg = plain_yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, plain_yaml.YAMLError):
        return
    if session_identity(cfg) != left_behind:
        return
    try:
        restamp_session(
            path,
            session_uid=draft.session_uid,
            data_dir=session_state.data_dir(draft),
        )
    except (OSError, ValueError, KeyError, FieldError, plain_yaml.YAMLError):
        logger.warning(
            "could not move %s onto today; removing it so the station cannot "
            "record into yesterday's folder",
            path,
            exc_info=True,
        )
        try:
            path.unlink()
        except OSError:
            logger.warning("could not remove the stale config %s", path)


def _is_a_day(value: str) -> bool:
    """Whether this is a date at all, as opposed to a broken one.

    The roll's job is yesterday -> today. A draft carrying `04/08/2026` is not a
    session that aged, it is a session whose date was written wrong (by a
    hand-edited config on disk, which `config_form.hydrate` reads back
    verbatim), and rolling it would silently repair the symptom, mint a fresh
    uid over it, and swallow the field-adjacent 400 that Apply answers
    (`session.data_dir`). Two different faults; only one of them is a new day.
    """
    try:
        dt.date.fromisoformat(value)
    except (TypeError, ValueError):
        return False
    return True


def roll_if_new_day(app) -> Rolled | None:
    """Start today's session if the draft still names another day.

    Returns what happened, so the caller can tell the operator; None when there
    was nothing to do, which is every call but one per day.
    """
    draft = app.state.rig.draft
    if not session_state.recording_day_is_stale(draft):
        return None
    if not _is_a_day(draft.date):
        return None
    if recording.in_progress(app):
        # The rig is genuinely busy. The stale-day banner on /config and /record
        # is still rendered in this case, and it is the right answer: stopping
        # the recorder is a decision only the operator can make.
        return None

    from_date = draft.date
    left_behind = draft.session_uid
    # Capture the existing folder and episode state before rollover replaces
    # the draft's session identity.
    had_episodes = session_state.has_episodes(draft)
    try:
        left_folder = session_state.data_dir(draft)
    except FieldError:
        # A draft whose folder cannot even be computed has recorded nothing.
        left_folder = ""

    session_state.start_new_session(draft)
    # Move the draft before sealing the old session so the active config never
    # points at a sealed folder.
    if left_folder and had_episodes:
        seal_state.seal_session(
            left_folder,
            reason="day_roll",
            session_uid=left_behind,
        )
    _rearm(app, draft, left_behind)
    # Clear the completed recorder status when starting a new recording day;
    # yesterday's job must not mark the new session as recorded.
    recording.forget_last_recording(app)

    rolled = Rolled(
        from_date=from_date,
        to_date=draft.date,
        left_behind=left_behind,
        had_episodes=had_episodes,
    )
    # Record automatic rollover in the durable station log as well as the one-shot
    # banner.
    logger.info(
        "new day: session %s (%s) left for %s (%s)",
        left_behind,
        from_date,
        draft.session_uid,
        draft.date,
    )
    app.state.day_rolled = rolled
    return rolled


def notice(request: Request) -> Rolled | None:
    """The roll to tell this browser about, consumed as it is read.

    Consumed on a real page load only. Every poll on this console fetches a
    whole page and picks a fragment out of it client-side (`hx-select`), so a
    notice cleared on any render at all would be swallowed by the one-second
    poll on /record and never reach a human. `HX-Request` is what separates the
    two, and htmx sets it on every request it makes.
    """
    rolled = getattr(request.app.state, "day_rolled", None)
    if rolled is None or request.headers.get("HX-Request"):
        return rolled
    request.app.state.day_rolled = None
    return rolled


class DayRoll:
    """Roll the session before the handler that would render the old one.

    A bare ASGI callable for the reason `origin_guard.OriginGuard` is one: this app
    serves four live SSE streams, and `BaseHTTPMiddleware` relays every response
    through an anyio memory stream.

    Here rather than in a context processor because order is the whole point.
    Context processors run when the template renders, which is after the route
    handler has already computed `stale_day` off the draft, so the page would
    warn about a session that had been rolled a microsecond earlier. Middleware
    runs first, and every page and every POST goes through it.

    `roll_if_new_day` performs no awaits, so under a single-threaded event loop
    two concurrent requests cannot both roll the same draft.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and not scope.get("path", "").startswith("/static"):
            try:
                roll_if_new_day(scope["app"])
            except Exception:  # noqa: BLE001
                # Leave the console usable if automatic rollover fails; the operator can
                # still use the stale-day banner action.
                logger.exception("the automatic new-day session roll failed")
        await self.app(scope, receive, send)

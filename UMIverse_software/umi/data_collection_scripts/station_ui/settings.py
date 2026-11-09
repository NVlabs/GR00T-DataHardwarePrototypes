# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime settings for the station web UI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
# station_ui/ -> the repository root, which is where config/data_recording/ and
# config/data_check/ live.
REPO_ROOT = PACKAGE_ROOT.parent

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8770

# Hosts that keep the server reachable only from the station itself. Anything
# else exposes config writes and hardware starts to the local network.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# Keep persistent logs under the user state directory, separate from
# configuration, temporary files and the checkout. UMI_STATION_LOG_DIR overrides it.
LOG_DIR_ENV = "UMI_STATION_LOG_DIR"
LOG_LEVEL_ENV = "UMI_STATION_LOG_LEVEL"
DEFAULT_LOG_DIR = "~/.umi/station_ui"
DEFAULT_LOG_LEVEL = "INFO"

# Keep active config and session identity outside the checkout so repository
# cleanup cannot delete them. The default is shared across checkouts for one
# station user; UMI_STATION_STATE_DIR isolates independent instances.
STATE_DIR_ENV = "UMI_STATION_STATE_DIR"
DEFAULT_STATE_DIR = "~/.umi_station"
# An allow-list, not `getattr(logging, name)`: a typo must fall back to INFO
# rather than raise at boot or silently resolve to some other module attribute.
LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


def log_dir_from_environment(env=None) -> Path:
    """Which directory `station.log` lives in.

    Read once, when `Settings` is constructed: the handler attaches once per
    process, so a value re-read later could only describe a file nothing is
    writing to.
    """
    source = os.environ if env is None else env
    raw = str(source.get(LOG_DIR_ENV, "")).strip() or DEFAULT_LOG_DIR
    return Path(raw).expanduser()


def log_level_from_environment(env=None) -> str:
    """How much of the console's own commentary is kept.

    INFO by default, because INFO keeps the breadcrumbs: `jobs.py` cancelling a
    child, `rig.py` claiming and releasing the rig. Those lines above a
    traceback are usually the whole diagnosis. Anything not in the allow-list
    falls back to INFO.
    """
    source = os.environ if env is None else env
    named = str(source.get(LOG_LEVEL_ENV, "")).strip().upper()
    return named if named in LOG_LEVELS else DEFAULT_LOG_LEVEL


def state_dir_from_environment(env=None) -> Path:
    """Which directory holds this bench's generated configs.

    Read once at construction, like the two above. A relative value is kept
    relative: a developer who points this at `./state` means the directory
    beside them, and silently absolutising it against the repo would put the
    tree back in the checkout.
    """
    source = os.environ if env is None else env
    raw = str(source.get(STATE_DIR_ENV, "")).strip() or DEFAULT_STATE_DIR
    return Path(raw).expanduser()


@dataclass(frozen=True)
class Settings:
    """Which interface the app binds, where its assets live, and which tree it
    drives.

    `static_dir` and `templates_dir` are package-relative, so the app still
    works installed as a wheel. `repo_root` is different: it is the working
    directory the launched jobs run in, and the tree whose
    `config/data_recording/` templates and `config/data_check/` baselines are
    read. It points at a checkout, never at the package.

    Nothing is written under it. Everything this console generates lands under
    `state_dir` instead; see STATE_DIR_ENV for why.
    """

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    repo_root: Path = REPO_ROOT
    log_dir: Path = field(default_factory=log_dir_from_environment)
    log_level: str = field(default_factory=log_level_from_environment)
    # Runtime state, deliberately outside `repo_root`: see STATE_DIR_ENV. The
    # templates and baselines this generates from stay under `repo_root`, which
    # is what keeps "what UMI ships" separable from "what this bench chose".
    state_dir: Path = field(default_factory=state_dir_from_environment)

    @property
    def static_dir(self) -> Path:
        return PACKAGE_ROOT / "static"

    @property
    def templates_dir(self) -> Path:
        return PACKAGE_ROOT / "templates"

    @property
    def is_loopback(self) -> bool:
        return self.host in LOOPBACK_HOSTS

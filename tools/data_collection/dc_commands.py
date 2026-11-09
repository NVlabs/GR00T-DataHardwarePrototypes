# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The `umi_dc` and `dexumi_dc` commands.

`uv sync` at the repository root installs both into the repository's .venv.
Each starts one data-collection GUI with that environment's Python, from the
pipeline's own folder, and the GUI opens itself in the browser. Arguments are
passed on to the GUI:

    umi_dc                UMI station console at http://127.0.0.1:8770/
    dexumi_dc             DexUMI data-collection GUI at http://127.0.0.1:8791/
    umi_dc --no-open      either one, without opening a browser
    dexumi_dc --help      the GUI's own options
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# uv installs workspace members in editable mode, so this file runs from
# tools/data_collection/ in the checkout and the pipelines are found from it.
REPO_ROOT = Path(__file__).resolve().parents[2]
UMI_DIR = REPO_ROOT / "UMIverse_software" / "umi" / "data_collection_scripts"
DEXUMI_DIR = REPO_ROOT / "UMIverse_software" / "dexumi" / "data_collection_scripts"

# station_ui's default port (station_ui/settings.py DEFAULT_PORT).
UMI_DEFAULT_PORT = "8770"
# How long a console on the port gets to shut down before it is killed.
STOP_GRACE_S = 5.0


def _exec_gui(workdir: Path, module: str, env: dict[str, str]) -> None:
    """Replace this process with `python -m <module> <arguments>` run in `workdir`.

    `exec`, not a child process, so Ctrl-C and SIGTERM go straight to the GUI,
    which then stops the recorders it started.
    """
    if not workdir.is_dir():
        sys.exit(
            f"{workdir} not found. Run this command from the repository's .venv "
            "(`uv sync` at the repository root installs it)."
        )
    os.chdir(workdir)
    os.execve(sys.executable, [sys.executable, "-m", module, *sys.argv[1:]], env)


def _port_argument(args: list[str], default: str) -> str:
    """The port the console will bind: the value of `--port`, or the default."""
    for i, arg in enumerate(args):
        if arg == "--port" and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith("--port="):
            return arg.removeprefix("--port=")
    return default


def _port_in_use(port: str) -> bool:
    return subprocess.run(["fuser", f"{port}/tcp"], capture_output=True).returncode == 0


def _stop_previous_console(port: str) -> None:
    """Stop a console still serving on `port`, so this one can bind it.

    Running the command again is how an operator restarts a console left
    running in another terminal. It gets SIGTERM and a few seconds to shut
    down cleanly, then SIGKILL. Without `fuser` (Ubuntu package psmisc) nothing
    is stopped, and a second console reports the port as in use instead.
    """
    if not port.isdigit() or shutil.which("fuser") is None or not _port_in_use(port):
        return
    print(f"Port {port} in use: stopping the previous console...", file=sys.stderr)
    subprocess.run(["fuser", "-k", "-TERM", f"{port}/tcp"], capture_output=True)
    deadline = time.monotonic() + STOP_GRACE_S
    while time.monotonic() < deadline:
        time.sleep(0.2)
        if not _port_in_use(port):
            return
    subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
    time.sleep(1)


def _asks_for_help(args: list[str]) -> bool:
    """True for `-h` / `--help`: the GUI only prints its options and exits."""
    return any(arg in ("-h", "--help") for arg in args)


def umi_dc() -> None:
    """Start the UMI station console and open it in the browser."""
    args = sys.argv[1:]
    # `umi_dc --help` must not stop a console that is already running.
    if not _asks_for_help(args):
        _stop_previous_console(_port_argument(args, UMI_DEFAULT_PORT))
    _exec_gui(UMI_DIR, "station_ui", dict(os.environ))


def dexumi_dc() -> None:
    """Start the DexUMI data-collection GUI and open it in the browser."""
    env = dict(os.environ)
    # The same environment launch.sh sets. DexUMI's packages (umi, gui,
    # real_scripts) are not installed into the shared .venv, where UMI's
    # packages of the same names are, so the GUI and every recorder it starts
    # import them from this folder through PYTHONPATH.
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(DEXUMI_DIR), env.get("PYTHONPATH")])
    )
    env["PYTHONUNBUFFERED"] = "1"
    _exec_gui(DEXUMI_DIR, "gui.dc_ui.server", env)

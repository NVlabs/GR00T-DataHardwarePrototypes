# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The recorder LaunchSpec: the same command an operator types, plus readiness markers.

    uv run real_scripts/data_collection/record_data.py --config <generated> \
        [--device-binding <generated>.device_binding.yaml]

``--device-binding`` is passed only when the cameras come from an Identify
cameras scan (device_binding.py); without it the recorder behaves as before.

record_data.py has no SIGINT handler of its own: KeyboardInterrupt inside the
session loop ends with ``Recording stopped`` (the done regex, also reached by
``q``), while a SIGINT during startup exits with a traceback (state ``failed``).
Keyboard control goes through the cv2 window, so stdin=DEVNULL is fine.
"""

import os
import shlex

from gui.dc_ui.core.kinds import RECORDER_ROLE, REPO_ROOT, LaunchSpec

RECORD_SCRIPT = "real_scripts/data_collection/record_data.py"
# Session log lines may start with [HH:MM:SS]; usage errors emitted before
# logging starts have no prefix. Accept both when matching status markers.
_STAMP = r"^(?:\[\d\d:\d\d:\d\d\] )?"
RECORDER_READY_REGEX = r"DATA RECORDING INSTRUCTIONS"
# The shutdown message can precede a crash traceback; only exit code 0 confirms success.
RECORDER_DONE_REGEX = None
# KeyboardInterrupt during startup is expected on Stop, so a traceback alone
# is not a failure marker. Report device-binding refusals with the changed slot.
RECORDER_BINDING_REFUSED_REGEX = r"\[device binding\] Launch refused"
RECORDER_FAIL_REGEXES = [
    r"Failed to start streaming",
    _STAMP + r"Error: ",
    RECORDER_BINDING_REFUSED_REGEX,
]
# record_data.py's DEVICE_BINDING_REFUSED_EXIT_CODE: the launch was refused
# because a device differs from the --device-binding scan (the scan is stale).
RECORDER_BINDING_REFUSED_EXIT = 3


def recorder_spec(config_path, device_binding=None):
    argv = [
        "uv",
        "run",
        RECORD_SCRIPT,
        "--config",
        os.path.abspath(os.path.expanduser(config_path)),
    ]
    if device_binding:
        argv += [
            "--device-binding",
            os.path.abspath(os.path.expanduser(device_binding)),
        ]
    return LaunchSpec(
        role=RECORDER_ROLE,
        argv=argv,
        cwd=str(REPO_ROOT),
        # Use this checkout on PYTHONPATH: UMI and DexUMI share package names, and
        # inherited paths can select the wrong implementation or shadow dependencies.
        env={"PYTHONUNBUFFERED": "1", "PYTHONPATH": str(REPO_ROOT)},
        ready_regexes=[RECORDER_READY_REGEX],
        fail_regexes=list(RECORDER_FAIL_REGEXES),
        done_regex=RECORDER_DONE_REGEX,
    )


def shell_line(spec):
    env = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in (spec.env or {}).items())
    argv = " ".join(shlex.quote(str(a)) for a in spec.argv)
    command = f"{env} {argv}" if env else argv
    return f"cd {shlex.quote(spec.cwd)} && {command}"

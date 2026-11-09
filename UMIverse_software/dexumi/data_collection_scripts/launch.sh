#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run from any working directory. Hardware recording requires the station's
# graphical session, FFmpeg on PATH, and calibrated devices selected in the GUI.
set -euo pipefail
WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKFLOW_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$WORKFLOW_ROOT${PYTHONPATH:+:$PYTHONPATH}"

COMMAND="${1:-gui}"
if [[ $# -gt 0 ]]; then
    shift
fi
case "$COMMAND" in
    gui) MODULE=gui.dc_ui.server ;;
    record) MODULE=real_scripts.data_collection.record_data ;;
    align) MODULE=real_scripts.data_processing.align_data ;;
    convert) MODULE=real_scripts.dexumi_pipeline.convert_to_robot_action ;;
    lerobot) MODULE=real_scripts.data_processing.lerobot_conversion.dexumi_lerobot_conversion ;;
    calibrate)
        exec uv run --locked --group data_collection python examples/hand/exoskeleton/calibrate_exo.py "$@"
        ;;
    -h|--help|help)
        printf '%s\n' \
            'Usage: launch.sh [gui|record|align|convert|lerobot|calibrate] [arguments]' \
            'Default: gui. Use COMMAND --help for command options.' \
            'Dependencies: Python 3.12, uv, FFmpeg (ffmpeg/ffplay), a graphical session for recording.' \
            'Install: uv sync at the repository root (one environment with the UMI software; adds the dexumi_dc command).' \
            'Example: launch.sh record --config config/data_recording/dexumi/mock_bimanual_dexumi_config.yaml'
        exit 0
        ;;
    *) printf 'Unknown command: %s\n' "$COMMAND" >&2; exit 2 ;;
esac
exec uv run --locked --group dc_ui python -m "$MODULE" "$@"

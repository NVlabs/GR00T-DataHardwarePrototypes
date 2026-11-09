#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# DexUMI data-collection GUI — start script. Run from a shell on the station's X
# session so DISPLAY reaches record_data.py's cv2 window (s / w / q keys live there).
# Usage: ./gui/dc_ui/start.sh [--host 127.0.0.1] [--port 8791] [--home DIR] [--data-root DIR]
#                         [--base-config YAML] [--allow-remote]
#                         [--scan-every-launch]  (one camera scan per recorder launch; default: per GUI run)
#                         [--no-open]  (do not open the GUI in a browser)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
exec "$REPO_ROOT/launch.sh" gui "$@"

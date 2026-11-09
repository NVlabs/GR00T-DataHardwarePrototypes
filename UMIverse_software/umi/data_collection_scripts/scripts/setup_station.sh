#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# Set up the shared repository-root UMI/DexUMI environment.
# Run from any directory: ./scripts/setup_station.sh





set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UMI_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# The repository root, which holds the uv workspace, its uv.lock and its .venv.
REPO_ROOT="$(cd "$UMI_ROOT/../../.." && pwd)"

print_status() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }

echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}         UMI Station Setup (uv)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "This script will:"
echo "  1. Check system requirements (Python 3.12+, gnome-terminal, dialout)"
echo "  2. Ensure uv is available"
echo "  3. Sync the repository's .venv (uv sync at the repository root): the UMI and DexUMI"
echo "     dependencies, including the LeRobot converter's packages (CPU-only torch)"
echo "  4. Grant gripper serial access: dialout + udev rule (asks first; the only sudo)"
echo "  5. Put the umi_dc and dexumi_dc commands on PATH and print how to run them"
echo ""
echo "UMI root:        $UMI_ROOT"
echo "Repository root: $REPO_ROOT"
echo ""

if [ "$EUID" -eq 0 ]; then
    print_error "Do not run this script with sudo!"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1: System requirements
# ---------------------------------------------------------------------------
echo -e "${BLUE}Step 1: Checking system requirements${NC}"
echo ""

# Python 3.12+
PYTHON_CMD=""
for py in python3.13 python3.12 python3; do
    if command -v "$py" &>/dev/null; then
        VERSION=$("$py" --version 2>&1 | awk '{print $2}')
        MAJOR=$(echo "$VERSION" | cut -d. -f1)
        MINOR=$(echo "$VERSION" | cut -d. -f2)
        if [ "$MAJOR" -eq 3 ] && [ "${MINOR:-0}" -ge 12 ]; then
            PYTHON_CMD=$py
            print_status "Found Python $VERSION"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    print_error "Python 3.12 or higher is required."
    echo "  sudo apt install python3.12"
    echo "  (If 'uv sync' fails building a dependency, try: sudo apt install python3.12-dev build-essential)"
    exit 1
fi

# Gripper calibration opens a station terminal; recording needs a graphical
# preview. Install gnome-terminal for those workflows.


if ! command -v gnome-terminal &>/dev/null; then
    print_error "gnome-terminal is not installed (calibration and latency jobs run in one)."
    echo "  sudo apt install gnome-terminal"
    exit 1
fi
print_status "gnome-terminal is available"

# Serial access to the gripper. Only reported here; step 4 offers to fix it,
# together with the udev rule that goes with it.
if id -nG | tr ' ' '\n' | grep -qx dialout; then
    print_status "serial access: in dialout"
else
    print_warning "Gripper serial access: $USER is not in dialout (offered in step 4)"
fi
echo ""

# ---------------------------------------------------------------------------
# Step 2: uv
# ---------------------------------------------------------------------------
echo -e "${BLUE}Step 2: Checking uv${NC}"
echo ""
if ! command -v uv &>/dev/null; then
    print_error "uv is required. Install from https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi
print_status "uv is available"
echo ""

# ---------------------------------------------------------------------------
# Step 3: Sync station dependencies
# ---------------------------------------------------------------------------
echo -e "${BLUE}Step 3: Syncing station dependencies (uv sync at the repository root)${NC}"
echo ""
cd "$REPO_ROOT"
# Sync at repository root to install both pipelines and every required
# group into one environment, including umi_dc and dexumi_dc. Syncing a
# member alone can remove packages required by the other pipeline.





if ! uv sync; then
    print_error "uv sync failed. Check errors above."
    exit 1
fi
print_status "Station dependencies synced"
# Check that torchcodec actually loaded system FFmpeg libraries; a
# successful Python import alone does not establish decoder availability.



if .venv/bin/python -c "import sys; from torchcodec._core import ops; sys.exit(0 if ops._FFMPEG_AVAILABLE else 1)" >/dev/null 2>&1; then
    print_status "LeRobot conversion ready (torchcodec found FFmpeg)"
else
    print_warning "torchcodec cannot load FFmpeg: 'Convert to LeRobot' will fail until it can (sudo apt install ffmpeg)"
fi
echo ""

# Gripper serial access requires dialout membership and a udev rule.
# The Pico VID/PID rule covers replacement boards and creates stable device
# links. Run setup as the station user and elevate only individual system
# commands so environments and wrappers remain user-owned.















echo -e "${BLUE}Step 4: Gripper serial access${NC}"
echo ""

# Add the invoking station user to dialout. Group membership takes effect
# only after a new login; opening another terminal is insufficient.



NEEDS_RELOGIN=0

if id -nG | tr ' ' '\n' | grep -qx dialout; then
    print_status "$USER is already in dialout"
elif ! command -v sudo &>/dev/null; then
    print_warning "sudo is not available. Add the operator by hand:"
    echo "  sudo usermod -aG dialout $USER   # then log out and back in"
else
    echo "$USER is not in dialout, which is what opens /dev/ttyACM*."
    echo "  sudo usermod -aG dialout $USER"
    echo ""
    # Enter accepts an interactive prompt; EOF declines. Preserve this
    # distinction so unattended setup cannot elevate automatically, and
    # handle read failure without triggering set -e.




    read -r -p "Add $USER to dialout now (needs sudo)? [Y/n]: " add_dialout || add_dialout="n"
    if [[ ! "$add_dialout" =~ ^[Nn] ]]; then
        if sudo usermod -aG dialout "$USER"; then
            print_status "Added $USER to dialout"
            NEEDS_RELOGIN=1
        else
            print_warning "Could not add $USER to dialout. Run the command above by hand."
        fi
    else
        print_warning "Skipped. Gripper calibration fails until $USER is in dialout."
    fi
fi
echo ""

RULES_SRC="$UMI_ROOT/99-umi-gripper.rules"
RULES_DEST="/etc/udev/rules.d/99-umi-gripper.rules"

if [ ! -f "$RULES_SRC" ]; then
    print_warning "No $RULES_SRC in this checkout; skipping."
elif [ -f "$RULES_DEST" ] && cmp -s "$RULES_SRC" "$RULES_DEST"; then
    print_status "udev rule already installed and up to date"
elif ! command -v sudo &>/dev/null; then
    print_warning "sudo is not available. Install the rule by hand:"
    echo "  sudo cp $RULES_SRC /etc/udev/rules.d/"
    echo "  sudo udevadm control --reload-rules && sudo udevadm trigger"
else
    if [ -f "$RULES_DEST" ]; then
        echo "The installed rule differs from this checkout's copy."
    else
        echo "The gripper udev rule is not installed on this station."
    fi
    echo "  sudo cp $RULES_SRC /etc/udev/rules.d/"
    echo "  sudo udevadm control --reload-rules && sudo udevadm trigger"
    echo ""
    # Enter means yes, end of input means no (see the dialout prompt above).
    read -r -p "Install it now (needs sudo)? [Y/n]: " install_rule || install_rule="n"
    if [[ ! "$install_rule" =~ ^[Nn] ]]; then
        # Report rule copying separately from reload failure; an installed rule
        # may only require replugging the gripper.


        if sudo cp "$RULES_SRC" "$RULES_DEST"; then
            print_status "Installed $RULES_DEST"
            if sudo udevadm control --reload-rules && sudo udevadm trigger; then
                print_status "udev rules reloaded"
            else
                print_warning "Rule copied, but the reload failed. Replug the gripper, or run:"
                echo "  sudo udevadm control --reload-rules && sudo udevadm trigger"
            fi
        else
            print_warning "Could not install the rule. Run the two commands above by hand."
        fi
    else
        print_warning "Skipped. The rule is optional if this operator is in dialout."
    fi
fi
echo ""

# Install command shortcuts through install_gumi.sh.




echo -e "${BLUE}Step 5: Putting umi_dc and dexumi_dc on PATH${NC}"
echo ""

bash "$SCRIPT_DIR/install_gumi.sh"
echo ""

# ---------------------------------------------------------------------------
# Done: run instructions
# ---------------------------------------------------------------------------
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Setup complete.${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""

# A new login is required after dialout changes; repeat this in the
# completion instructions.


if [ "$NEEDS_RELOGIN" -eq 1 ]; then
    print_warning "LOG OUT AND BACK IN before recording: $USER joined dialout, and"
    echo "  that does not take effect until the next login. Opening a new"
    echo "  terminal is NOT enough. Verify afterwards with:  id -nG | grep dialout"
    echo ""
fi
echo "Run the station UI with:"
echo ""
echo "  umi_dc               # serves it and opens the browser for you"
echo ""
echo "or, without the link in ~/.local/bin, from the repository root:"
echo ""
echo "  cd $REPO_ROOT"
echo "  uv run umi_dc"
echo ""
echo "It serves http://127.0.0.1:8770/ (--no-open to skip the browser). dexumi_dc"
echo "starts the DexUMI data-collection GUI the same way, at http://127.0.0.1:8791/."
echo "After pulling updates, run 'uv sync' in the repository root again."
echo ""
echo "Recordings are written under $UMI_ROOT/output/ (raw, quarantine, aligned,"
echo "reports). Set UMI_OUTPUT_ROOT to put them somewhere else."
echo ""

#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Brings the GEAR Neck's USB-CAN adapter up as the SocketCAN interface can_neck
# at 1 Mbit/s, the bitrate of the neck's GDK drives.
#
# The Jhoinrch RH-02 runs CANable slcan firmware (USB id 16d0:117e) and shows up
# as a serial port; slcand, from can-utils, attaches it as a network interface.
# Linux only. Nothing is made persistent: run this again after every replug or
# reboot. The tools also work without it, through --interface slcan.
#
# Usage: sudo setup_can.sh [serial port]   (default: the adapter found by its USB id)
set -euo pipefail

IFACE="can_neck"
VID="16d0"
PID="117e"
DFU_ID="0483:df11"  # the adapter's STM32 bootloader

log() { echo "[setup_can] $*"; }
die() {
  echo "[setup_can] ERROR: $*" >&2
  exit 1
}

find_adapter_tty() {
  local tty usb_device
  for tty in /sys/class/tty/ttyACM*; do
    [[ -e "$tty/device" ]] || continue
    usb_device="$(dirname "$(readlink -f "$tty/device")")"
    if [[ "$(cat "$usb_device/idVendor" 2>/dev/null)" == "$VID" &&
          "$(cat "$usb_device/idProduct" 2>/dev/null)" == "$PID" ]]; then
      echo "/dev/$(basename "$tty")"
      return 0
    fi
  done
  return 1
}

[[ $EUID -eq 0 ]] || die "run it as root: sudo $0 $*"
command -v slcand > /dev/null \
  || die "slcand not found: install can-utils (apt install can-utils, dnf install can-utils or" \
         "pacman -S can-utils), or skip this script and run the tools with --interface slcan"

if ip link show "$IFACE" > /dev/null 2>&1; then
  log "$IFACE already exists; bringing it up"
  ip link set "$IFACE" up
  ip -details link show "$IFACE"
  exit 0
fi

TTY="${1:-}"
if [[ -z "$TTY" ]]; then
  if ! TTY="$(find_adapter_tty)"; then
    if command -v lsusb > /dev/null && lsusb | grep -qi "ID $DFU_ID"; then
      die "the adapter is in DFU (bootloader) mode: switch its BOOT DIP switch off and replug it"
    fi
    die "no USB-CAN adapter ($VID:$PID) found: plug it in, or pass its serial port"
  fi
fi
[[ -c "$TTY" ]] || die "$TTY is not a serial port"

modprobe slcan 2> /dev/null \
  || die "this kernel has no slcan module; use the tools with --interface slcan instead"

log "attaching $TTY as $IFACE at 1 Mbit/s"
# -o open the channel, -c close it on exit, -s8 1 Mbit/s.
slcand -o -c -s8 "$TTY" "$IFACE"
for _ in $(seq 30); do
  ip link show "$IFACE" > /dev/null 2>&1 && break
  sleep 0.1
done
ip link show "$IFACE" > /dev/null 2>&1 || die "slcand did not create $IFACE"
ip link set "$IFACE" txqueuelen 1000 up
ip -details link show "$IFACE"
log "done. Next: neck_assign_id scan"

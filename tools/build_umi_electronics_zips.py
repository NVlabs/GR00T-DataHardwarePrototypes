#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the two UMI gripper electronics downloads linked from GEAR_UMI/hardware.html.

  GEAR_UMI/files/firmware/umi-encoder-v2.0.0-source.zip
      README, firmware/ sources
  GEAR_UMI/files/electrical/UMI-Gripper-Rev2617-offline.zip
      UMI-Gripper-Rev2617-offline/: README, hardware/ KiCad projects and library, firmware/

Both are packed from the files in this repository (GEAR_UMI/files/firmware/umi-encoder/,
GEAR_UMI/files/electrical/kicad/ and GEAR_UMI/files/firmware/UMI-Gripper-Rev2617-README.md),
and each one also carries LICENSE, NOTICE and the LICENSES/ texts that the firmware's
third-party portions require. The firmware is Apache-2.0; the KiCad board designs and the
README are CC BY 4.0, whose text is included too. Run it again after changing any of
those files; the zips are rebuilt
with fixed timestamps, so unchanged inputs give identical archives.

  python3 tools/build_umi_electronics_zips.py
"""

import os
import subprocess
import zipfile

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
UMI = os.path.join(REPO, "GEAR_UMI")
FIRMWARE = os.path.join(UMI, "files", "firmware", "umi-encoder")
KICAD = os.path.join(UMI, "files", "electrical", "kicad")
README = os.path.join(UMI, "files", "firmware", "UMI-Gripper-Rev2617-README.md")
TEXTS = os.path.join(REPO, "LICENSES", "GEAR_UMI")
CC_BY = os.path.join(REPO, "LICENSES", "CC-BY-4.0.txt")
SOURCE_ZIP = os.path.join(UMI, "files", "firmware", "umi-encoder-v2.0.0-source.zip")
OFFLINE_ZIP = os.path.join(
    UMI, "files", "electrical", "UMI-Gripper-Rev2617-offline.zip"
)
STAMP = (2026, 1, 1, 0, 0, 0)

# Third-party license texts needed by the firmware sources in the archive.
FIRMWARE_TEXTS = [
    "CRC16-BSD-3-Clause.txt",
    "Pico-stdio-usb-MIT.txt",
    "RaspberryPi-BSD-3-Clause.txt",
    "TinyUSB-MIT.txt",
]

LICENSE_PREAMBLE = """\
Apache-2.0 applies to the NVIDIA-owned firmware source in firmware/. The board
designs in hardware/ (where present) and the README are licensed under the
Creative Commons Attribution 4.0 International license (CC BY 4.0); see
LICENSES/CC-BY-4.0.txt. Third-party portions of the firmware keep their own
terms; see NOTICE and LICENSES/.

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


"""

FIRMWARE_NOTICE = """\
UMI gripper firmware v2.0.0 (RP2040)
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
NVIDIA-owned portions: Apache-2.0; see LICENSE.

The firmware source files below adapt third-party code. Each keeps its original
copyright and license notice at the top of the file; the full license texts are
in LICENSES/.

firmware/crc16.h
  CRC-16/XMODEM table and function matching Redis src/crc16.c.
  Copyright 2001-2010 Georges Menie (www.menie.org)
  Copyright 2010-2012 Salvatore Sanfilippo
  License: BSD-3-Clause (LICENSES/CRC16-BSD-3-Clause.txt)
  Reference: https://github.com/redis/redis/blob/7.2/src/crc16.c

firmware/usb_descriptors.c
  Device, string and reset descriptors adapted from the Raspberry Pi Pico SDK
  stdio_usb_descriptors.c, which was originally part of MicroPython.
  Copyright (c) 2020 Raspberry Pi (Trading) Ltd.
  Copyright (c) 2019 Damien P. George
  License: MIT (LICENSES/Pico-stdio-usb-MIT.txt)
  Reference: https://github.com/raspberrypi/pico-sdk/blob/2.2.0/src/rp2_common/pico_stdio_usb/stdio_usb_descriptors.c
  The stereo microphone descriptor is adapted from TinyUSB src/device/usbd.h.
  Copyright (c) 2019 Ha Thach (tinyusb.org)
  License: MIT (LICENSES/TinyUSB-MIT.txt)

firmware/usb_audio.c
  UAC2 control callbacks adapted from the TinyUSB audio_4_channel_mic example.
  Copyright (c) 2020 Reinhard Panhuber
  License: MIT (LICENSES/TinyUSB-MIT.txt)
  Reference: https://github.com/hathach/tinyusb/blob/0.18.0/examples/device/audio_4_channel_mic/src/main.c

firmware/OpenPDM2PCM/OpenPDMFilter.c, firmware/OpenPDM2PCM/OpenPDMFilter.h
  Copyright (c) 2018 STMicroelectronics
  License: Apache-2.0 (LICENSE)
  Reference: https://github.com/ArmDeveloperEcosystem/microphone-library-for-pico/tree/main/src/OpenPDM2PCM

firmware/pico_sdk_import.cmake
  Copy of the Raspberry Pi Pico SDK import helper.
  Copyright (c) 2020 Raspberry Pi (Trading) Ltd.
  License: BSD-3-Clause (LICENSES/RaspberryPi-BSD-3-Clause.txt)

Building the firmware links code that is not in this archive: the Raspberry Pi
Pico SDK 2.2.0 (BSD-3-Clause, with its MIT-licensed TinyUSB stack and printf)
and the Arm toolchain's Newlib C library and libgcc (GPL-3.0-or-later with the
GCC Runtime Library Exception 3.1). If you distribute a built UF2 or ELF,
include the notices of the SDK and toolchain versions you built with.
"""


def read(path):
    with open(path, "rb") as f:
        return f.read()


def tree(root):
    """(relative posix path, bytes) for every git-tracked file under root, sorted.

    Untracked files (build output, editor backups) never end up in an archive.
    """
    names = (
        subprocess.run(
            ["git", "ls-files", "-z", "--", "."],
            cwd=root,
            capture_output=True,
            check=True,
        )
        .stdout.decode("utf8")
        .split("\0")
    )
    out = [
        (name, read(os.path.join(root, name))) for name in sorted(n for n in names if n)
    ]
    pointers = [
        name for name, data in out if data.startswith(b"version https://git-lfs")
    ]
    if pointers:
        raise SystemExit(
            f"Git LFS pointers, not files, under {root}: {pointers[:3]}. Run `git lfs pull` first."
        )
    return out


def legal_files(notice):
    apache = read(os.path.join(TEXTS, "Apache-2.0.txt")).decode("utf8")
    files = [
        ("LICENSE", (LICENSE_PREAMBLE + apache.lstrip("\n")).encode("utf8")),
        ("NOTICE", notice.encode("utf8")),
    ]
    files += [
        ("LICENSES/" + name, read(os.path.join(TEXTS, name))) for name in FIRMWARE_TEXTS
    ]
    files.append(("LICENSES/CC-BY-4.0.txt", read(CC_BY)))
    return files


def write_zip(path, members):
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in members:
            info = zipfile.ZipInfo(name, STAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            z.writestr(info, data)
    os.replace(tmp, path)
    print(f"wrote {os.path.relpath(path, REPO)} ({len(members)} files)")


def main():
    firmware = [("firmware/" + name, data) for name, data in tree(FIRMWARE)]
    readme = [("README.md", read(README))]

    write_zip(SOURCE_ZIP, readme + legal_files(FIRMWARE_NOTICE) + firmware)

    prefix = "UMI-Gripper-Rev2617-offline/"
    hardware = [("hardware/" + name, data) for name, data in tree(KICAD)]
    members = readme + legal_files(FIRMWARE_NOTICE) + hardware + firmware
    write_zip(OFFLINE_ZIP, [(prefix + name, data) for name, data in members])


if __name__ == "__main__":
    main()

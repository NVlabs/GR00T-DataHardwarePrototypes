# GEAR UMI license and notice texts

This collection covers the UMI software, the gripper firmware source and the
components linked into a built gripper firmware binary. [NOTICE](../../NOTICE)
maps components to these files. The [Apache 2.0 license](../../LICENSE) and its
[scope](../../README.md#license) remain at the repository root.

The Record3D-derived UMI iPhone adapter, including NVIDIA adaptations, is
LGPL-2.1-only. Its [attribution and replacement note](../Record3D/README.md)
reuses the [LGPL text](../DexUMI/record3d-79e6f446/LICENSE) in the DexUMI collection.

`UMI-MIT.txt` preserves the earlier UMI source/release notice with NVIDIA
ownership and the original MIT permission and disclaimer text. It is retained
for provenance and earlier distributions, not offered as an alternative
license for the current NVIDIA-owned project material.
Third-party notices retain their original copyright and terms.

| File | Component / purpose |
| --- | --- |
| [Apache-2.0.txt](Apache-2.0.txt) | Apache License 2.0; shared by the identified Apache-licensed components |
| [CRC16-BSD-3-Clause.txt](CRC16-BSD-3-Clause.txt) | CRC16 original BSD 3-Clause notice (firmware `crc16.h`) |
| [GCC-GPL-3.0.txt](GCC-GPL-3.0.txt) | GCC GPLv3 text; read with the applicable runtime exception and notice |
| [GCC-Runtime-Library-Exception-3.1.txt](GCC-Runtime-Library-Exception-3.1.txt) | GCC Runtime Library Exception 3.1 |
| [GCC-libgcc-ARM-notice.txt](GCC-libgcc-ARM-notice.txt) | GCC/libgcc ARM runtime notice identified in the v2.0.0 firmware binary |
| [Newlib-4.4.0.20231231-COPYING.NEWLIB.txt](Newlib-4.4.0.20231231-COPYING.NEWLIB.txt) | Complete upstream Newlib license compilation |
| [Newlib-4.4.0.20231231-linked-notices.txt](Newlib-4.4.0.20231231-linked-notices.txt) | Notices for the Newlib portions identified in the v2.0.0 firmware binary |
| [Pico-stdio-usb-MIT.txt](Pico-stdio-usb-MIT.txt) | Pico USB stdio source MIT notice (firmware `usb_descriptors.c`) |
| [PicoPrintf-MIT.txt](PicoPrintf-MIT.txt) | Pico printf source MIT notice |
| [RaspberryPi-BSD-3-Clause.txt](RaspberryPi-BSD-3-Clause.txt) | Raspberry Pi Pico SDK BSD 3-Clause notice (firmware `pico_sdk_import.cmake`, SDK runtime) |
| [Three-0.164.1-MIT.txt](Three-0.164.1-MIT.txt) | Earlier calibration viewer Three.js MIT notice; retained for earlier distributions |
| [TinyUSB-MIT.txt](TinyUSB-MIT.txt) | TinyUSB MIT notice (firmware `usb_descriptors.c`, `usb_audio.c`) |
| [TinyUSB-firmware-linked-notices.txt](TinyUSB-firmware-linked-notices.txt) | Original notices for the TinyUSB stack portions identified in the v2.0.0 firmware binary |
| [UMI-MIT.txt](UMI-MIT.txt) | Earlier UMI source/release MIT reference notice; current project default is Apache-2.0 |
| [htmx-0BSD.txt](htmx-0BSD.txt) | Station UI htmx Zero-Clause BSD notice |
| [pyopenxr.txt](pyopenxr.txt) | Original pyopenxr Apache-licensed source attribution |

The [DexUMI collection](../DexUMI/README.md) reuses relevant shared texts
from this directory. Include all applicable texts and notices when packaging
the software or firmware separately: `tools/build_umi_electronics_zips.py`
does this for the firmware and electronics downloads.

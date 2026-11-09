# UMI Gripper Electronics — Rev2617

Offline release of the **umi-oak-cb1** OAK carrier board and **tle5012b-encoder** magnetic encoder board, with RP2040 firmware **v2.0.0**.

## Package Contents

```text
hardware/
  umi-oak-cb1/       KiCad project and manufacturing files
  tle5012b-encoder/  KiCad project and manufacturing files
  library/          Required custom symbols, footprints, and 3D models
firmware/           RP2040 application sources and CMake build inputs
```

Each board's `production/` directory contains Gerbers and drill files in `gerbers/`, plus `bom.csv`, `positions.csv`, and `designators.csv`. These are the existing manufacturing outputs, not newly regenerated exports. No legacy revisions, backups, or reference documents are included.

## Electrical Architecture

The carrier integrates a Luxonis **OAK-SoM (BW1099)**, three camera interfaces, a USB 2.0 hub, and an RP2040 for encoder acquisition, stereo microphones, and board monitoring. The encoder PCB contains the TLE5012B sensor and connects to the carrier through a five-pin cable; it has no separate programmable MCU.

```text
                           UMI-OAK-CB1
                    +-----------------------+
USB-C data input -->| HD3SS3220 SS mux       |-- USB 3 SS --> OAK-SoM
       USB 2 D+/D- ->| CH334P USB 2 hub       |-- Port 1 ---> RP2040
                    |                       |-- Port 2 ---> iPhone DFP
                    |                       |-- Port 3 ---> OAK-SoM USB 2
                    +-----------------------+

OAK-SoM <--- MIPI CSI-2 + camera control ---> CAM_A: 4 lanes
                                        +-> CAM_B: 2 lanes
                                        +-> CAM_C: 2 lanes

RP2040 <-- 3-wire SSC/SPI --> TLE5012B encoder PCB
       <-- PDM data -------- Two on-board microphones
       <-- PG / ADC -------- Input power monitoring
       --- GPIO ----------> Status LED / iPhone VBUS switch

USB-C data-port 5 V --> LM73100 --+
                                +--> SYS_5V --> OAK-SoM
USB-C AUX-port 5 V ---> LM73100 --+          +-> SY6280AAC --> iPhone VBUS
                                           +-> TPS54302 --> 3.3 V
                                                              |
                                              RP2040 / hub / cameras
                                                              |
                                                    XC6206 --> 1.8 V
```

- USB SuperSpeed connects directly to the OAK-SoM; its USB 2.0 fallback runs through the hub. USB 2.0 D+/D- does not pass through the SuperSpeed mux.
- The auxiliary USB-C port supplies power only. Both inputs are **5 V only**; the design does not negotiate higher-voltage USB Power Delivery.
- Preserve the `hardware/` directory structure so the carrier's relative library paths remain valid. Use KiCad 10 or a compatible newer version with standard KiCad libraries installed.

## Communication Diagram

```text
TLE5012B -- SSC/SPI, 1 MHz --> RP2040 Core 1
                              | sample angle + health at 120 Hz
                              v
                         Inter-core frame queue
                              |
                              v
                         RP2040 Core 0 -- USB CDC 1 --> Host application
                              |
                              +--------- USB CDC 0 --> Debug console

Stereo microphones -- PDM --> PIO + DMA --> PDM-to-PCM filter
                                              |
                                              +-- USB UAC2 --> Host audio
                                                  48 kHz, stereo, 16-bit

Camera modules -- MIPI CSI-2 --> OAK-SoM -- USB 3 / USB 2 --> Host vision
```

The RP2040's USB interfaces reach the host through the CH334P hub. Encoder data and microphone audio are separate USB streams; they do not travel through the OAK-SoM.

| Interface | Purpose |
| --- | --- |
| SSC/SPI | Encoder clock GPIO2, shared bidirectional DATA using GPIO3/GPIO4, active-low chip select GPIO5 |
| USB CDC 0 | Human-readable firmware diagnostics |
| USB CDC 1 | Binary encoder and sensor-health frames; CDC index 1, not USB interface number 1 |
| USB UAC2 | Stereo microphone capture at 48 kHz, 16-bit PCM |
| MIPI CSI-2 / I2C | Camera image data and camera configuration through the OAK-SoM |

Encoder frames use **UMI protocol 0x12**:

```text
[Sync 0x5AA5 | Version | Source | Device ID (8 B) | Timestamp ms | Payload length]
[JOINT_ENCODER TLV: float32 angle in degrees]
[SENSOR_HEALTH TLV: health state, error count, safety/status words]
[CRC16]
```

Multi-byte fields are little-endian. Each frame pairs angle data with health information; consumers must check health before trusting an angle. See `firmware/umi_protocol.h` and `firmware/tle5012b.h` for the exact wire layout.

## Firmware Build

Requires CMake, an Arm embedded toolchain, and Raspberry Pi Pico SDK **2.2.0**, including its required submodules. The SDK and toolchain are **not bundled**; preinstall them to build offline.

```sh
export PICO_SDK_PATH=/path/to/pico-sdk
cmake -S firmware -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The build produces `build/umi-encoder.uf2`. Enter RP2040 BOOTSEL mode and copy the UF2 to its USB boot volume. Encoder and microphone acquisition are enabled by default; pin assignments are in `firmware/board.h`. No prebuilt firmware binary is included.

## License

The firmware source in `firmware/` is licensed under the Apache License, Version 2.0 (`LICENSE`). A few of its files adapt third-party code; `NOTICE` lists them and `LICENSES/` holds their license texts. The board designs in `hardware/` and this document are licensed under the Creative Commons Attribution 4.0 International license (CC BY 4.0, `LICENSES/CC-BY-4.0.txt`). In the GR00T Hardware Designs repository these files are at the repository root.

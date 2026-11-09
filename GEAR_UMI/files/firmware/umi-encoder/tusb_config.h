/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _TUSB_CONFIG_H_
#define _TUSB_CONFIG_H_

// TinyUSB configuration

#include <tusb_option.h>

#include "board.h"

#define CFG_TUSB_RHPORT0_MODE (OPT_MODE_DEVICE | OPT_MODE_FULL_SPEED)

#ifndef CFG_TUD_ENDPOINT0_SIZE
#define CFG_TUD_ENDPOINT0_SIZE 64
#endif

// TinyUSB debug: 0=off, 1=error, 2=class-level, 3=all
// Set via CFG_TUSB_DEBUG in CMakeLists.txt (must override SDK default of 0)

// ---- CDC Class ----
#define CFG_TUD_CDC 2
#define CFG_TUD_CDC_RX_BUFSIZE 1024
#define CFG_TUD_CDC_TX_BUFSIZE 1024
#define CFG_TUD_CDC_EP_BUFSIZE 64

// ---- Vendor Class (reset interface) ----
#define CFG_TUD_VENDOR 1
#define CFG_TUD_BOS 1

// ---- Audio Class (UAC2 stereo microphone) ----
#define CFG_TUD_AUDIO 1

// Audio function 1 configuration — sample rate sourced from board.h
#define CFG_TUD_AUDIO_FUNC_1_SAMPLE_RATE BOARD_PDM_SAMPLE_RATE

// Descriptor length for the 2-channel UAC2 mic descriptor (defined in usb_descriptors.c)
// = IAD(8) + STD_AC(9) + CS_AC(9) + CLK_SRC(8) + INPUT_TERM(17) + OUTPUT_TERM(12)
//   + FEATURE_UNIT_TWO_CH(18) + STD_AS_INT(9)*2 + CS_AS_INT(16)
//   + TYPE_I_FORMAT(6) + STD_AS_ISO_EP(7) + CS_AS_ISO_EP(8)  =  136
#define CFG_TUD_AUDIO_FUNC_1_DESC_LEN 136

#define CFG_TUD_AUDIO_FUNC_1_N_AS_INT 1
#define CFG_TUD_AUDIO_FUNC_1_CTRL_BUF_SZ 64

// Enable EP IN for microphone (device -> host)
#define CFG_TUD_AUDIO_ENABLE_EP_IN 1

// Audio format parameters
#define CFG_TUD_AUDIO_FUNC_1_N_BYTES_PER_SAMPLE_TX 2 /* 16-bit */
#define CFG_TUD_AUDIO_FUNC_1_N_CHANNELS_TX 2         /* stereo */

// Endpoint size: (48000/1000 + 1) * 2 bytes * 2 channels = 49 * 4 = 196 bytes
#define CFG_TUD_AUDIO_EP_SZ_IN \
    TUD_AUDIO_EP_SIZE(CFG_TUD_AUDIO_FUNC_1_SAMPLE_RATE, \
                      CFG_TUD_AUDIO_FUNC_1_N_BYTES_PER_SAMPLE_TX, \
                      CFG_TUD_AUDIO_FUNC_1_N_CHANNELS_TX)

// Enable built-in encoding support (Type I PCM interleaving)
#define CFG_TUD_AUDIO_ENABLE_ENCODING 1
#define CFG_TUD_AUDIO_EP_IN_FLOW_CONTROL 1
#define CFG_TUD_AUDIO_ENABLE_TYPE_I_ENCODING 1

#define CFG_TUD_AUDIO_FUNC_1_EP_IN_SZ_MAX CFG_TUD_AUDIO_EP_SZ_IN
#define CFG_TUD_AUDIO_FUNC_1_EP_IN_SW_BUF_SZ CFG_TUD_AUDIO_EP_SZ_IN

// Encoding: 1 FIFO stream with 2 channels per FIFO (stereo I2S)
#define CFG_TUD_AUDIO_FUNC_1_CHANNEL_PER_FIFO_TX 2
#define CFG_TUD_AUDIO_FUNC_1_N_TX_SUPP_SW_FIFO \
    (CFG_TUD_AUDIO_FUNC_1_N_CHANNELS_TX / CFG_TUD_AUDIO_FUNC_1_CHANNEL_PER_FIFO_TX)
#define CFG_TUD_AUDIO_FUNC_1_TX_SUPP_SW_FIFO_SZ \
    (4 * (CFG_TUD_AUDIO_EP_SZ_IN / CFG_TUD_AUDIO_FUNC_1_N_TX_SUPP_SW_FIFO))

#endif // _TUSB_CONFIG_H_

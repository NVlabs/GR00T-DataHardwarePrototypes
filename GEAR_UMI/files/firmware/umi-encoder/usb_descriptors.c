/*
 * USB device/string/reset descriptors adapted from the Raspberry Pi Pico SDK
 * stdio_usb_descriptors.c (originally derived from MicroPython).
 * Upstream reference: https://github.com/raspberrypi/pico-sdk/blob/2.2.0/src/rp2_common/pico_stdio_usb/stdio_usb_descriptors.c
 */

/*
 * This file is based on a file originally part of the
 * MicroPython project, http://micropython.org/
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2020 Raspberry Pi (Trading) Ltd.
 * Copyright (c) 2019 Damien P. George
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/*
 * The stereo microphone descriptor is adapted from TinyUSB
 * TUD_AUDIO_MIC_FOUR_CH_DESCRIPTOR in src/device/usbd.h.
 * Upstream reference: https://github.com/hathach/tinyusb/blob/0.18.0/src/device/usbd.h
 */

/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2019 Ha Thach (tinyusb.org)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * This file is part of the TinyUSB stack.
 */

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT AND Apache-2.0
 *
 * NVIDIA modifications are Apache-2.0-licensed; the upstream terms above are retained.
 */

#include "pico/stdio_usb.h"
#include "pico/stdio_usb/reset_interface.h"
#include "pico/unique_id.h"
#include "tusb.h"

#include "board.h"

#define USBD_VID BOARD_USB_VID
#if PICO_RP2040
#define USBD_PID BOARD_USB_PID_RP2040
#else
#define USBD_PID BOARD_USB_PID_DEFAULT
#endif

#define USBD_MANUFACTURER BOARD_USB_MANUFACTURER
#define USBD_PRODUCT BOARD_USB_PRODUCT

#define TUD_RPI_RESET_DESC_LEN 9
#define TUD_RPI_RESET_DESCRIPTOR(_itfnum, _stridx)                                                                     \
    9, TUSB_DESC_INTERFACE, _itfnum, 0, 0, TUSB_CLASS_VENDOR_SPECIFIC, RESET_INTERFACE_SUBCLASS,                       \
        RESET_INTERFACE_PROTOCOL, _stridx

/* --------------------------------------------------------------------------
 *  UAC2 Stereo Microphone Descriptor (2-channel variant)
 *
 *  Based on TinyUSB's TUD_AUDIO_MIC_FOUR_CH_DESCRIPTOR, adapted for 2 ch.
 *  Topology: Input Terminal (Mic) -> Feature Unit -> Output Terminal (USB)
 *            Clock Source feeds Input Terminal.
 *  Entity IDs: 1=InputTerm, 2=FeatureUnit, 3=OutputTerm, 4=ClockSource
 * -------------------------------------------------------------------------- */

#define TUD_AUDIO_MIC_TWO_CH_DESC_LEN (TUD_AUDIO_DESC_IAD_LEN           \
    + TUD_AUDIO_DESC_STD_AC_LEN                                         \
    + TUD_AUDIO_DESC_CS_AC_LEN                                          \
    + TUD_AUDIO_DESC_CLK_SRC_LEN                                        \
    + TUD_AUDIO_DESC_INPUT_TERM_LEN                                     \
    + TUD_AUDIO_DESC_OUTPUT_TERM_LEN                                    \
    + TUD_AUDIO_DESC_FEATURE_UNIT_TWO_CHANNEL_LEN                       \
    + TUD_AUDIO_DESC_STD_AS_INT_LEN                                     \
    + TUD_AUDIO_DESC_STD_AS_INT_LEN                                     \
    + TUD_AUDIO_DESC_CS_AS_INT_LEN                                      \
    + TUD_AUDIO_DESC_TYPE_I_FORMAT_LEN                                  \
    + TUD_AUDIO_DESC_STD_AS_ISO_EP_LEN                                  \
    + TUD_AUDIO_DESC_CS_AS_ISO_EP_LEN)

#define TUD_AUDIO_MIC_TWO_CH_DESCRIPTOR(_itfnum, _stridx, _nBytesPerSample, _nBitsUsedPerSample, _epin, _epsize) \
    /* Standard Interface Association Descriptor (IAD) */\
    TUD_AUDIO_DESC_IAD(/*_firstitf*/ _itfnum, /*_nitfs*/ 0x02, /*_stridx*/ 0x00),\
    /* Standard AC Interface Descriptor(4.7.1) */\
    TUD_AUDIO_DESC_STD_AC(/*_itfnum*/ _itfnum, /*_nEPs*/ 0x00, /*_stridx*/ _stridx),\
    /* Class-Specific AC Interface Header Descriptor(4.7.2) */\
    TUD_AUDIO_DESC_CS_AC(/*_bcdADC*/ 0x0200, /*_category*/ AUDIO_FUNC_MICROPHONE, \
        /*_totallen*/ TUD_AUDIO_DESC_CLK_SRC_LEN + TUD_AUDIO_DESC_INPUT_TERM_LEN \
                    + TUD_AUDIO_DESC_OUTPUT_TERM_LEN + TUD_AUDIO_DESC_FEATURE_UNIT_TWO_CHANNEL_LEN, \
        /*_ctrl*/ AUDIO_CS_AS_INTERFACE_CTRL_LATENCY_POS),\
    /* Clock Source Descriptor(4.7.2.1) */\
    TUD_AUDIO_DESC_CLK_SRC(/*_clkid*/ 0x04, /*_attr*/ AUDIO_CLOCK_SOURCE_ATT_INT_FIX_CLK, \
        /*_ctrl*/ (AUDIO_CTRL_R << AUDIO_CLOCK_SOURCE_CTRL_CLK_FRQ_POS), \
        /*_assocTerm*/ 0x01, /*_stridx*/ 0x00),\
    /* Input Terminal Descriptor(4.7.2.4) — 2 channels */\
    TUD_AUDIO_DESC_INPUT_TERM(/*_termid*/ 0x01, /*_termtype*/ AUDIO_TERM_TYPE_IN_GENERIC_MIC, \
        /*_assocTerm*/ 0x03, /*_clkid*/ 0x04, /*_nchannelslogical*/ 0x02, \
        /*_channelcfg*/ AUDIO_CHANNEL_CONFIG_NON_PREDEFINED, /*_idxchannelnames*/ 0x00, \
        /*_ctrl*/ AUDIO_CTRL_R << AUDIO_IN_TERM_CTRL_CONNECTOR_POS, /*_stridx*/ 0x00),\
    /* Output Terminal Descriptor(4.7.2.5) */\
    TUD_AUDIO_DESC_OUTPUT_TERM(/*_termid*/ 0x03, /*_termtype*/ AUDIO_TERM_TYPE_USB_STREAMING, \
        /*_assocTerm*/ 0x01, /*_srcid*/ 0x02, /*_clkid*/ 0x04, /*_ctrl*/ 0x0000, /*_stridx*/ 0x00),\
    /* Feature Unit Descriptor(4.7.2.8) — 2 channels + master */\
    TUD_AUDIO_DESC_FEATURE_UNIT_TWO_CHANNEL(/*_unitid*/ 0x02, /*_srcid*/ 0x01, \
        /*_ctrlch0master*/ AUDIO_CTRL_RW << AUDIO_FEATURE_UNIT_CTRL_MUTE_POS | AUDIO_CTRL_RW << AUDIO_FEATURE_UNIT_CTRL_VOLUME_POS, \
        /*_ctrlch1*/       AUDIO_CTRL_RW << AUDIO_FEATURE_UNIT_CTRL_MUTE_POS | AUDIO_CTRL_RW << AUDIO_FEATURE_UNIT_CTRL_VOLUME_POS, \
        /*_ctrlch2*/       AUDIO_CTRL_RW << AUDIO_FEATURE_UNIT_CTRL_MUTE_POS | AUDIO_CTRL_RW << AUDIO_FEATURE_UNIT_CTRL_VOLUME_POS, \
        /*_stridx*/ 0x00),\
    /* Standard AS Interface Descriptor(4.9.1) — Alt 0, zero bandwidth */\
    TUD_AUDIO_DESC_STD_AS_INT(/*_itfnum*/ (uint8_t)((_itfnum)+1), /*_altset*/ 0x00, /*_nEPs*/ 0x00, /*_stridx*/ 0x00),\
    /* Standard AS Interface Descriptor(4.9.1) — Alt 1, data streaming */\
    TUD_AUDIO_DESC_STD_AS_INT(/*_itfnum*/ (uint8_t)((_itfnum)+1), /*_altset*/ 0x01, /*_nEPs*/ 0x01, /*_stridx*/ 0x00),\
    /* Class-Specific AS Interface Descriptor(4.9.2) — 2 physical channels */\
    TUD_AUDIO_DESC_CS_AS_INT(/*_termid*/ 0x03, /*_ctrl*/ AUDIO_CTRL_NONE, \
        /*_formattype*/ AUDIO_FORMAT_TYPE_I, /*_formats*/ AUDIO_DATA_FORMAT_TYPE_I_PCM, \
        /*_nchannelsphysical*/ 0x02, /*_channelcfg*/ AUDIO_CHANNEL_CONFIG_NON_PREDEFINED, /*_stridx*/ 0x00),\
    /* Type I Format Type Descriptor(2.3.1.6) */\
    TUD_AUDIO_DESC_TYPE_I_FORMAT(_nBytesPerSample, _nBitsUsedPerSample),\
    /* Standard AS Isochronous Audio Data Endpoint Descriptor(4.10.1.1) */\
    TUD_AUDIO_DESC_STD_AS_ISO_EP(/*_ep*/ _epin, \
        /*_attr*/ (uint8_t)((uint8_t)TUSB_XFER_ISOCHRONOUS | (uint8_t)TUSB_ISO_EP_ATT_ASYNCHRONOUS | (uint8_t)TUSB_ISO_EP_ATT_DATA), \
        /*_maxEPsize*/ _epsize, /*_interval*/ 0x01),\
    /* Class-Specific AS Isochronous Audio Data Endpoint Descriptor(4.10.1.2) */\
    TUD_AUDIO_DESC_CS_AS_ISO_EP(/*_attr*/ AUDIO_CS_AS_ISO_DATA_EP_ATT_NON_MAX_PACKETS_OK, \
        /*_ctrl*/ AUDIO_CTRL_NONE, \
        /*_lockdelayunit*/ AUDIO_CS_AS_ISO_DATA_EP_LOCK_DELAY_UNIT_UNDEFINED, /*_lockdelay*/ 0x0000)

/* Make the total audio descriptor length available to tusb_config.h */
const uint16_t usbd_audio_desc_total_len = TUD_AUDIO_MIC_TWO_CH_DESC_LEN;

/* --------------------------------------------------------------------------
 *  Endpoint & Interface numbering
 * -------------------------------------------------------------------------- */

#define USBD_DESC_CDC_LEN (TUD_CDC_DESC_LEN * CFG_TUD_CDC)
#define USBD_DESC_LEN (TUD_CONFIG_DESC_LEN + USBD_DESC_CDC_LEN + TUD_RPI_RESET_DESC_LEN + TUD_AUDIO_MIC_TWO_CH_DESC_LEN)
#define USBD_DESC_STR_MAX (32)

#define USBD_CONFIGURATION_DESCRIPTOR_ATTRIBUTE (0)
#define USBD_MAX_POWER_MA BOARD_USB_MAX_POWER_MA

#define USBD_CDC_CMD_MAX_SIZE 8
#define USBD_CDC_IN_OUT_MAX_SIZE CFG_TUD_CDC_EP_BUFSIZE

/* CDC 0 endpoints */
#define USBD_CDC_0_EP_CMD BOARD_CDC_0_EP_CMD
#define USBD_CDC_0_EP_OUT BOARD_CDC_0_EP_OUT
#define USBD_CDC_0_EP_IN  BOARD_CDC_0_EP_IN

/* CDC 1 endpoints */
#define USBD_CDC_1_EP_CMD BOARD_CDC_1_EP_CMD
#define USBD_CDC_1_EP_OUT BOARD_CDC_1_EP_OUT
#define USBD_CDC_1_EP_IN  BOARD_CDC_1_EP_IN

/* Audio endpoint — defined in usb_audio.h */
#include "usb_audio.h"

enum {
    USBD_ITF_CDC_0 = 0,
    USBD_ITF_CDC_0_DATA,
    USBD_ITF_RPI_RESET,
    USBD_ITF_CDC_1,
    USBD_ITF_CDC_1_DATA,
    USBD_ITF_AUDIO_CONTROL,     /* Audio Control interface */
    USBD_ITF_AUDIO_STREAMING,   /* Audio Streaming interface */
    USBD_ITF_MAX
};

enum {
    USBD_STR_0 = 0,
    USBD_STR_MANUF,
    USBD_STR_PRODUCT,
    USBD_STR_SERIAL,
    USBD_STR_CDC,
    USBD_STR_RPI_RESET,
    USBD_STR_AUDIO,
    USBD_STR_COUNT
};

static const tusb_desc_device_t usbd_desc_device = {
    .bLength = sizeof(tusb_desc_device_t),
    .bDescriptorType = TUSB_DESC_DEVICE,
    .bcdUSB = 0x0210,
    .bDeviceClass = TUSB_CLASS_MISC,
    .bDeviceSubClass = MISC_SUBCLASS_COMMON,
    .bDeviceProtocol = MISC_PROTOCOL_IAD,
    .bMaxPacketSize0 = CFG_TUD_ENDPOINT0_SIZE,
    .idVendor = USBD_VID,
    .idProduct = USBD_PID,
    .bcdDevice = 0x0100,
    .iManufacturer = USBD_STR_MANUF,
    .iProduct = USBD_STR_PRODUCT,
    .iSerialNumber = USBD_STR_SERIAL,
    .bNumConfigurations = 1,
};

static const uint8_t usbd_desc_cfg[USBD_DESC_LEN] = {
    TUD_CONFIG_DESCRIPTOR(1, USBD_ITF_MAX, USBD_STR_0, USBD_DESC_LEN, USBD_CONFIGURATION_DESCRIPTOR_ATTRIBUTE,
                          USBD_MAX_POWER_MA),

    /* CDC 0 — stdio debug */
    TUD_CDC_DESCRIPTOR(USBD_ITF_CDC_0, USBD_STR_CDC, USBD_CDC_0_EP_CMD, USBD_CDC_CMD_MAX_SIZE, USBD_CDC_0_EP_OUT,
                       USBD_CDC_0_EP_IN, USBD_CDC_IN_OUT_MAX_SIZE),

    /* Vendor Reset interface */
    TUD_RPI_RESET_DESCRIPTOR(USBD_ITF_RPI_RESET, USBD_STR_RPI_RESET),

    /* CDC 1 — encoder data */
    TUD_CDC_DESCRIPTOR(USBD_ITF_CDC_1, USBD_STR_CDC, USBD_CDC_1_EP_CMD, USBD_CDC_CMD_MAX_SIZE, USBD_CDC_1_EP_OUT,
                       USBD_CDC_1_EP_IN, USBD_CDC_IN_OUT_MAX_SIZE),

    /* UAC2 Stereo Microphone */
    TUD_AUDIO_MIC_TWO_CH_DESCRIPTOR(USBD_ITF_AUDIO_CONTROL, USBD_STR_AUDIO,
                                    CFG_TUD_AUDIO_FUNC_1_N_BYTES_PER_SAMPLE_TX,
                                    CFG_TUD_AUDIO_FUNC_1_N_BYTES_PER_SAMPLE_TX * 8,
                                    USBD_AUDIO_EP_IN, CFG_TUD_AUDIO_EP_SZ_IN),
};

static char usbd_serial_str[PICO_UNIQUE_BOARD_ID_SIZE_BYTES * 2 + 1];

static const char *const usbd_desc_str[USBD_STR_COUNT] = {
    [USBD_STR_MANUF] = USBD_MANUFACTURER,
    [USBD_STR_PRODUCT] = USBD_PRODUCT,
    [USBD_STR_SERIAL] = usbd_serial_str,
    [USBD_STR_CDC] = "Board CDC",
    [USBD_STR_RPI_RESET] = "Reset",
    [USBD_STR_AUDIO] = "UMI Stereo Mic",
};

const uint8_t *tud_descriptor_device_cb(void) { return (const uint8_t *)&usbd_desc_device; }

const uint8_t *tud_descriptor_configuration_cb(uint8_t index) {
    (void)index;
    return usbd_desc_cfg;
}

const uint16_t *tud_descriptor_string_cb(uint8_t index, uint16_t langid) {
    static uint16_t desc_str[USBD_DESC_STR_MAX];
    uint8_t len = 0;

    (void)langid;

    if (!usbd_serial_str[0]) {
        pico_get_unique_board_id_string(usbd_serial_str, sizeof(usbd_serial_str));
    }

    if (index == 0) {
        desc_str[1] = 0x0409;
        len = 1;
    } else {
        if (index >= USBD_STR_COUNT || !usbd_desc_str[index]) {
            return NULL;
        }
        const char *str = usbd_desc_str[index];
        for (len = 0; len < USBD_DESC_STR_MAX - 1 && str[len]; ++len) {
            desc_str[1 + len] = (uint8_t)str[len];
        }
    }

    desc_str[0] = (uint16_t)((TUSB_DESC_STRING << 8) | (2 * len + 2));
    return desc_str;
}

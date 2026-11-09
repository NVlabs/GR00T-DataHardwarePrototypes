/*
 * UAC2 control callbacks adapted from the TinyUSB four-channel microphone
 * example, customized for this project's stereo microphone.
 * Upstream reference: https://github.com/hathach/tinyusb/blob/0.18.0/examples/device/audio_4_channel_mic/src/main.c
 */

/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2020 Reinhard Panhuber
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

/**
 * @file usb_audio.c
 * @brief USB Audio Class 2.0 stereo microphone — TinyUSB callbacks & data path
 */

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT AND Apache-2.0
 *
 * NVIDIA modifications are Apache-2.0-licensed; the upstream terms above are retained.
 */

#include "usb_audio.h"
#include "pdm_mic.h"
#include "tusb.h"
#include "hardware/structs/usb.h"

#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Audio control state                                                */
/* ------------------------------------------------------------------ */

/* +1 for master channel 0 */
static bool g_mute[CFG_TUD_AUDIO_FUNC_1_N_CHANNELS_TX + 1];
static int16_t g_volume[CFG_TUD_AUDIO_FUNC_1_N_CHANNELS_TX + 1];
static uint32_t g_sample_freq = CFG_TUD_AUDIO_FUNC_1_SAMPLE_RATE;
static uint8_t g_clk_valid = 1;

/* Sample frequency range — single fixed rate */
static const audio_control_range_4_n_t(1) g_sample_freq_rng = {
    .wNumSubRanges = 1,
    .subrange = {{
        .bMin = CFG_TUD_AUDIO_FUNC_1_SAMPLE_RATE,
        .bMax = CFG_TUD_AUDIO_FUNC_1_SAMPLE_RATE,
        .bRes = 0,
    }},
};

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

void usb_audio_init(void) {
    memset(g_mute, 0, sizeof(g_mute));
    for (int i = 0; i < (int)(sizeof(g_volume) / sizeof(g_volume[0])); i++) {
        g_volume[i] = 0; /* 0 dB */
    }
}

void usb_audio_feed(const int16_t *pcm_buf, size_t sample_count) {
    if (g_mute[0]) {
        static int16_t silence[PDM_MIC_PCM_BUF_SAMPLES];
        size_t n = (sample_count <= PDM_MIC_PCM_BUF_SAMPLES)
                   ? sample_count : PDM_MIC_PCM_BUF_SAMPLES;
        tud_audio_write_support_ff(0, (const uint8_t *)silence,
                                   (uint16_t)(n * sizeof(int16_t)));
    } else {
        tud_audio_write_support_ff(0, (const uint8_t *)pcm_buf,
                                   (uint16_t)(sample_count * sizeof(int16_t)));
    }
}

/* ------------------------------------------------------------------ */
/*  TinyUSB Audio Class Callbacks (UAC2)                               */
/* ------------------------------------------------------------------ */

bool tud_audio_set_req_ep_cb(uint8_t rhport, tusb_control_request_t const *p_request, uint8_t *pBuff) {
    (void)rhport;
    (void)pBuff;
    TU_VERIFY(p_request->bRequest == AUDIO_CS_REQ_CUR);
    return false;
}

bool tud_audio_set_req_itf_cb(uint8_t rhport, tusb_control_request_t const *p_request, uint8_t *pBuff) {
    (void)rhport;
    (void)pBuff;
    TU_VERIFY(p_request->bRequest == AUDIO_CS_REQ_CUR);
    return false;
}

bool tud_audio_set_req_entity_cb(uint8_t rhport, tusb_control_request_t const *p_request, uint8_t *pBuff) {
    (void)rhport;

    uint8_t channelNum = TU_U16_LOW(p_request->wValue);
    uint8_t ctrlSel = TU_U16_HIGH(p_request->wValue);
    uint8_t entityID = TU_U16_HIGH(p_request->wIndex);

    TU_VERIFY(p_request->bRequest == AUDIO_CS_REQ_CUR);

    if (entityID == 2) {
        switch (ctrlSel) {
            case AUDIO_FU_CTRL_MUTE:
                TU_VERIFY(p_request->wLength == sizeof(audio_control_cur_1_t));
                g_mute[channelNum] = ((audio_control_cur_1_t *)pBuff)->bCur;
                printf("[UAC] Set Mute ch%u = %u\n", channelNum, g_mute[channelNum]);
                return true;

            case AUDIO_FU_CTRL_VOLUME:
                TU_VERIFY(p_request->wLength == sizeof(audio_control_cur_2_t));
                g_volume[channelNum] = ((audio_control_cur_2_t *)pBuff)->bCur;
                printf("[UAC] Set Volume ch%u = %d\n", channelNum, g_volume[channelNum]);
                return true;

            default:
                break;
        }
    }

    if (entityID == 4) {
        if (ctrlSel == AUDIO_CS_CTRL_SAM_FREQ) {
            TU_VERIFY(p_request->wLength == sizeof(audio_control_cur_4_t));
            g_sample_freq = (uint32_t)((audio_control_cur_4_t const *)pBuff)->bCur;
            return true;
        }
    }

    return false;
}

bool tud_audio_get_req_ep_cb(uint8_t rhport, tusb_control_request_t const *p_request) {
    (void)rhport;
    (void)p_request;
    return false;
}

bool tud_audio_get_req_itf_cb(uint8_t rhport, tusb_control_request_t const *p_request) {
    (void)rhport;
    (void)p_request;
    return false;
}

bool tud_audio_get_req_entity_cb(uint8_t rhport, tusb_control_request_t const *p_request) {
    (void)rhport;

    uint8_t channelNum = TU_U16_LOW(p_request->wValue);
    uint8_t ctrlSel = TU_U16_HIGH(p_request->wValue);
    uint8_t entityID = TU_U16_HIGH(p_request->wIndex);

    /* Input Terminal (ID = 1) */
    if (entityID == 1) {
        if (ctrlSel == AUDIO_TE_CTRL_CONNECTOR) {
            audio_desc_channel_cluster_t ret;
            ret.bNrChannels = CFG_TUD_AUDIO_FUNC_1_N_CHANNELS_TX;
            ret.bmChannelConfig = (audio_channel_config_t)0;
            ret.iChannelNames = 0;
            return tud_audio_buffer_and_schedule_control_xfer(rhport, p_request, (void *)&ret, sizeof(ret));
        }
    }

    /* Feature Unit (ID = 2) */
    if (entityID == 2) {
        switch (ctrlSel) {
            case AUDIO_FU_CTRL_MUTE:
                return tud_control_xfer(rhport, p_request, &g_mute[channelNum], 1);

            case AUDIO_FU_CTRL_VOLUME:
                if (p_request->bRequest == AUDIO_CS_REQ_CUR) {
                    return tud_control_xfer(rhport, p_request, &g_volume[channelNum], sizeof(g_volume[channelNum]));
                } else if (p_request->bRequest == AUDIO_CS_REQ_RANGE) {
                    audio_control_range_2_n_t(1) vol_range;
                    vol_range.wNumSubRanges = 1;
                    vol_range.subrange[0].bMin = -90 * 256; /* -90 dB in 1/256 dB units */
                    vol_range.subrange[0].bMax = 0;         /*   0 dB */
                    vol_range.subrange[0].bRes = 256;       /*   1 dB step */
                    return tud_audio_buffer_and_schedule_control_xfer(rhport, p_request, (void *)&vol_range,
                                                                      sizeof(vol_range));
                }
                break;

            default:
                break;
        }
    }

    /* Clock Source (ID = 4) */
    if (entityID == 4) {
        switch (ctrlSel) {
            case AUDIO_CS_CTRL_SAM_FREQ:
                if (p_request->bRequest == AUDIO_CS_REQ_CUR) {
                    return tud_audio_buffer_and_schedule_control_xfer(rhport, p_request, &g_sample_freq,
                                                                      sizeof(g_sample_freq));
                } else if (p_request->bRequest == AUDIO_CS_REQ_RANGE) {
                    return tud_control_xfer(rhport, p_request, (void *)&g_sample_freq_rng,
                                            sizeof(g_sample_freq_rng));
                }
                break;

            case AUDIO_CS_CTRL_CLK_VALID:
                return tud_control_xfer(rhport, p_request, &g_clk_valid, sizeof(g_clk_valid));

            default:
                break;
        }
    }

    return false;
}

bool tud_audio_tx_done_pre_load_cb(uint8_t rhport, uint8_t itf, uint8_t ep_in, uint8_t cur_alt_setting) {
    (void)rhport;
    (void)itf;
    (void)ep_in;
    (void)cur_alt_setting;
    return true;
}

bool tud_audio_tx_done_post_load_cb(uint8_t rhport, uint16_t n_bytes_copied, uint8_t itf, uint8_t ep_in,
                                     uint8_t cur_alt_setting) {
    (void)rhport;
    (void)n_bytes_copied;
    (void)itf;
    (void)ep_in;
    (void)cur_alt_setting;
    return true;
}

bool tud_audio_set_itf_cb(uint8_t rhport, tusb_control_request_t const *p_request) {
    (void)rhport;
    (void)p_request;
    return true;
}

bool tud_audio_set_itf_close_EP_cb(uint8_t rhport, tusb_control_request_t const *p_request) {
    (void)rhport;
    (void)p_request;

    /* RP2040 workaround: with TUP_DCD_EDPT_ISO_ALLOC the audio class driver
     * never calls usbd_edpt_close() for ISO endpoints.  A pending transfer's
     * AVAIL bit stays set in the hardware buffer control register.  If the host
     * later re-opens the interface (SET_INTERFACE alt=1), scheduling a new
     * transfer hits the "ep XX was already available" panic.
     * Fix: clear the buffer control register so the next transfer can arm it. */
    uint8_t const epnum = tu_edpt_number(USBD_AUDIO_EP_IN);
    usb_dpram->ep_buf_ctrl[epnum].in = 0;

    return true;
}

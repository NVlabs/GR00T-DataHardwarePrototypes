/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file pdm_mic.c
 * @brief Stereo PDM DMIC driver — PIO + DMA + OpenPDMFilter
 *
 * When BOARD_PDM_MIC_TEST_MODE == 1, no hardware is used; a 1 kHz stereo
 * sine-wave test tone is generated via a repeating timer instead.
 */

#include "pdm_mic.h"

#include <string.h>

static pdm_mic_callback_t g_callback;
static volatile const int16_t *g_ready_buf;
static volatile size_t g_ready_count;

/* ================================================================== */
#if BOARD_PDM_MIC_TEST_MODE
/* ================================================================== */
/*  TEST MODE — 1 kHz stereo sine (no hardware)                       */
/* ================================================================== */

#include "pico/time.h"
#include <math.h>

#define TEST_TONE_FREQ_HZ  1000
#define TEST_TONE_AMPLITUDE 0x3000

static int16_t g_test_pcm[PDM_MIC_PCM_BUF_SAMPLES];
static struct repeating_timer g_test_timer;

static bool test_timer_callback(struct repeating_timer *t) {
    (void)t;
    if (g_callback) {
        g_callback(g_test_pcm, PDM_MIC_PCM_BUF_SAMPLES);
    } else {
        g_ready_buf = g_test_pcm;
        g_ready_count = PDM_MIC_PCM_BUF_SAMPLES;
    }
    return true;
}

static void test_mode_generate_tone(void) {
    for (int i = 0; i < PDM_MIC_SAMPLES_PER_MS; i++) {
        float phase = 2.0f * (float)M_PI * (float)TEST_TONE_FREQ_HZ
                      * (float)i / (float)PDM_MIC_SAMPLE_RATE;
        int16_t left  = (int16_t)(sinf(phase) * TEST_TONE_AMPLITUDE);
        int16_t right = (int16_t)(cosf(phase) * TEST_TONE_AMPLITUDE);
        g_test_pcm[i * 2 + 0] = left;
        g_test_pcm[i * 2 + 1] = right;
    }
}

void pdm_mic_init(pdm_mic_callback_t callback) {
    g_callback = callback;
    g_ready_buf = NULL;
    g_ready_count = 0;
    test_mode_generate_tone();
    add_repeating_timer_ms(1, test_timer_callback, NULL, &g_test_timer);
}

/* ================================================================== */
#else /* !BOARD_PDM_MIC_TEST_MODE — real hardware path */
/* ================================================================== */

#include "pdm_mic.pio.h"

#include "hardware/clocks.h"
#include "hardware/dma.h"
#include "hardware/irq.h"
#include "hardware/pio.h"

#include "OpenPDM2PCM/OpenPDMFilter.h"

#define PDM_PIO         BOARD_PDM_PIO
#define PDM_SM_MASTER   BOARD_PDM_SM
#define PDM_SM_SLAVE    (BOARD_PDM_SM + 1)

#define PDM_RAW_BUF_COUNT 2

/* Per-channel raw PDM double buffers */
static uint8_t g_raw_buf_l[PDM_RAW_BUF_COUNT][PDM_MIC_RAW_BUF_BYTES];
static uint8_t g_raw_buf_r[PDM_RAW_BUF_COUNT][PDM_MIC_RAW_BUF_BYTES];

static int g_dma_ch_l, g_dma_ch_r;
static volatile int g_buf_write_idx;

/* OpenPDMFilter instances — one per channel (mono) */
static TPDMFilter_InitStruct g_filter_l;
static TPDMFilter_InitStruct g_filter_r;
static uint16_t g_filter_volume;

/* PCM output double buffer */
static int16_t g_pcm_buf[PDM_RAW_BUF_COUNT][PDM_MIC_PCM_BUF_SAMPLES];
static volatile int g_pcm_write_idx;

/* Per-channel temporary PCM before interleaving */
static uint16_t g_pcm_tmp_l[PDM_MIC_SAMPLES_PER_MS];
static uint16_t g_pcm_tmp_r[PDM_MIC_SAMPLES_PER_MS];

static void pdm_filter_init(TPDMFilter_InitStruct *f) {
    f->Fs              = PDM_MIC_SAMPLE_RATE;
    f->LP_HZ           = PDM_MIC_SAMPLE_RATE / 2;
    f->HP_HZ           = 10;
    f->In_MicChannels  = 1;
    f->Out_MicChannels = 1;
    f->Decimation      = PDM_MIC_DECIMATION;
    f->MaxVolume       = BOARD_PDM_FILTER_MAX_VOLUME;
    f->Gain            = BOARD_PDM_FILTER_GAIN;
    Open_PDM_Filter_Init(f);
}

static void pdm_process_and_interleave(int buf_idx) {
    int filter_stride = PDM_MIC_SAMPLE_RATE / 1000;
    int samples = PDM_MIC_SAMPLES_PER_MS;
    uint8_t *in_l = g_raw_buf_l[buf_idx];
    uint8_t *in_r = g_raw_buf_r[buf_idx];
    uint16_t *out_l = g_pcm_tmp_l;
    uint16_t *out_r = g_pcm_tmp_r;

    for (int i = 0; i < samples; i += filter_stride) {
        Open_PDM_Filter_64(in_l, out_l, g_filter_volume, &g_filter_l);
        Open_PDM_Filter_64(in_r, out_r, g_filter_volume, &g_filter_r);
        in_l += filter_stride * (PDM_MIC_DECIMATION / 8);
        in_r += filter_stride * (PDM_MIC_DECIMATION / 8);
        out_l += filter_stride;
        out_r += filter_stride;
    }

    int16_t *pcm = g_pcm_buf[g_pcm_write_idx];
    for (int i = 0; i < PDM_MIC_SAMPLES_PER_MS; i++) {
        pcm[i * 2 + 0] = (int16_t)g_pcm_tmp_l[i];
        pcm[i * 2 + 1] = (int16_t)g_pcm_tmp_r[i];
    }
}

static void pdm_dma_irq_handler(void) {
    bool l_done = dma_irqn_get_channel_status(0, g_dma_ch_l);
    bool r_done = dma_irqn_get_channel_status(0, g_dma_ch_r);

    if (l_done) dma_irqn_acknowledge_channel(0, g_dma_ch_l);
    if (r_done) dma_irqn_acknowledge_channel(0, g_dma_ch_r);

    if (!(l_done && r_done)) return;

    int read_idx = g_buf_write_idx;
    g_buf_write_idx = (g_buf_write_idx + 1) % PDM_RAW_BUF_COUNT;

    /* Restart DMA into the next write buffer */
    dma_channel_transfer_to_buffer_now(g_dma_ch_l,
        g_raw_buf_l[g_buf_write_idx], PDM_MIC_RAW_BUF_BYTES);
    dma_channel_transfer_to_buffer_now(g_dma_ch_r,
        g_raw_buf_r[g_buf_write_idx], PDM_MIC_RAW_BUF_BYTES);

    /* Filter + interleave the just-completed buffer */
    pdm_process_and_interleave(read_idx);

    int pcm_idx = g_pcm_write_idx;
    g_pcm_write_idx = (g_pcm_write_idx + 1) % PDM_RAW_BUF_COUNT;

    if (g_callback) {
        g_callback(g_pcm_buf[pcm_idx], PDM_MIC_PCM_BUF_SAMPLES);
    } else {
        g_ready_buf = g_pcm_buf[pcm_idx];
        g_ready_count = PDM_MIC_PCM_BUF_SAMPLES;
    }
}

void pdm_mic_init(pdm_mic_callback_t callback) {
    g_callback = callback;
    g_ready_buf = NULL;
    g_ready_count = 0;
    g_buf_write_idx = 0;
    g_pcm_write_idx = 0;

    /* --- Filter init --- */
    pdm_filter_init(&g_filter_l);
    pdm_filter_init(&g_filter_r);
    g_filter_volume = BOARD_PDM_FILTER_VOLUME;

    /* --- PIO setup --- */
    float clk_div = (float)clock_get_hz(clk_sys) /
                    (float)(PDM_MIC_SAMPLE_RATE * PDM_MIC_DECIMATION * 4);

    uint offset_master = pio_add_program(PDM_PIO, &pdm_mic_master_program);
    uint offset_slave  = pio_add_program(PDM_PIO, &pdm_mic_slave_program);

    pdm_mic_master_init(PDM_PIO, PDM_SM_MASTER, offset_master,
                        clk_div, BOARD_PDM_PIN_DATL, BOARD_PDM_PIN_CLK);
    pdm_mic_slave_init(PDM_PIO, PDM_SM_SLAVE, offset_slave,
                       clk_div, BOARD_PDM_PIN_DATR);

    /* --- DMA setup --- */
    g_dma_ch_l = dma_claim_unused_channel(true);
    g_dma_ch_r = dma_claim_unused_channel(true);

    dma_channel_config cfg_l = dma_channel_get_default_config(g_dma_ch_l);
    channel_config_set_transfer_data_size(&cfg_l, DMA_SIZE_8);
    channel_config_set_read_increment(&cfg_l, false);
    channel_config_set_write_increment(&cfg_l, true);
    channel_config_set_dreq(&cfg_l, pio_get_dreq(PDM_PIO, PDM_SM_MASTER, false));
    dma_channel_configure(g_dma_ch_l, &cfg_l,
        g_raw_buf_l[0], &PDM_PIO->rxf[PDM_SM_MASTER],
        PDM_MIC_RAW_BUF_BYTES, false);

    dma_channel_config cfg_r = dma_channel_get_default_config(g_dma_ch_r);
    channel_config_set_transfer_data_size(&cfg_r, DMA_SIZE_8);
    channel_config_set_read_increment(&cfg_r, false);
    channel_config_set_write_increment(&cfg_r, true);
    channel_config_set_dreq(&cfg_r, pio_get_dreq(PDM_PIO, PDM_SM_SLAVE, false));
    dma_channel_configure(g_dma_ch_r, &cfg_r,
        g_raw_buf_r[0], &PDM_PIO->rxf[PDM_SM_SLAVE],
        PDM_MIC_RAW_BUF_BYTES, false);

    /* DMA IRQ on both channels */
    dma_irqn_set_channel_enabled(0, g_dma_ch_l, true);
    dma_irqn_set_channel_enabled(0, g_dma_ch_r, true);
    irq_set_exclusive_handler(DMA_IRQ_0, pdm_dma_irq_handler);
    irq_set_enabled(DMA_IRQ_0, true);

    /* Start DMA */
    dma_channel_start(g_dma_ch_l);
    dma_channel_start(g_dma_ch_r);

    /* Start both SMs synchronously */
    pio_enable_sm_mask_in_sync(PDM_PIO,
        (1u << PDM_SM_MASTER) | (1u << PDM_SM_SLAVE));
}

#endif /* BOARD_PDM_MIC_TEST_MODE */

/* ================================================================== */
/*  Common (always compiled)                                           */
/* ================================================================== */

bool pdm_mic_get_ready_buffer(const int16_t **buf, size_t *sample_count) {
    if (!g_ready_buf) return false;
    *buf = (const int16_t *)g_ready_buf;
    *sample_count = g_ready_count;
    g_ready_buf = NULL;
    g_ready_count = 0;
    return true;
}

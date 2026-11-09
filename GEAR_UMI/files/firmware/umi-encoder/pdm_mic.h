/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file pdm_mic.h
 * @brief Stereo PDM DMIC driver using PIO + DMA + OpenPDMFilter
 *
 * Captures audio from two LMD3526T261 PDM DMIC sharing a common CLK line
 * with independent DATA pins (left and right channels).
 *
 * Uses two PIO state machines on PIO0:
 *   SM0 (master) — drives CLK via sideset, reads DATL
 *   SM1 (slave)  — reads DATR only, started in sync with SM0
 *
 * Each SM has its own DMA channel with double-buffering.  When both DMA
 * transfers complete, raw PDM bitstreams are decimated by OpenPDMFilter
 * (sinc3, decimation=64) and interleaved into stereo 16-bit PCM.
 *
 * The user-provided callback receives ready-to-use stereo PCM data.
 */

#ifndef PDM_MIC_H
#define PDM_MIC_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "board.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PDM_MIC_SAMPLE_RATE   BOARD_PDM_SAMPLE_RATE
#define PDM_MIC_CHANNELS      2
#define PDM_MIC_BITS_PER_SAMPLE 16

#define PDM_MIC_DECIMATION    64
#define PDM_MIC_SAMPLES_PER_MS (PDM_MIC_SAMPLE_RATE / 1000)

/* Raw PDM bytes per channel per ms: samples_per_ms * (decimation / 8) */
#define PDM_MIC_RAW_BUF_BYTES (PDM_MIC_SAMPLES_PER_MS * (PDM_MIC_DECIMATION / 8))

/* Stereo PCM buffer: samples_per_ms * 2 channels */
#define PDM_MIC_PCM_BUF_SAMPLES (PDM_MIC_SAMPLES_PER_MS * PDM_MIC_CHANNELS)

/**
 * @brief Callback invoked from DMA IRQ when a stereo PCM buffer is ready.
 *
 * @param pcm_buf  Interleaved stereo 16-bit PCM (L, R, L, R, ...).
 * @param sample_count  Total number of int16_t samples (= frames * 2).
 */
typedef void (*pdm_mic_callback_t)(const int16_t *pcm_buf, size_t sample_count);

/**
 * @brief Initialize and start PDM stereo microphone capture.
 *
 * Sets up PIO state machines, DMA channels, and OpenPDMFilter instances.
 * Capture begins immediately after init returns.
 *
 * @param callback  Called from DMA IRQ context when a PCM buffer is ready.
 *                  May be NULL for polling via pdm_mic_get_ready_buffer().
 */
void pdm_mic_init(pdm_mic_callback_t callback);

/**
 * @brief Check if a completed PCM buffer is available (polling mode).
 *
 * @param[out] buf  Set to point to the ready stereo PCM buffer.
 * @param[out] sample_count  Number of int16_t samples in the buffer.
 * @return true if a buffer was available.
 */
bool pdm_mic_get_ready_buffer(const int16_t **buf, size_t *sample_count);

#ifdef __cplusplus
}
#endif

#endif /* PDM_MIC_H */

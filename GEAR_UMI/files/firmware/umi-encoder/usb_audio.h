/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file usb_audio.h
 * @brief USB Audio Class 2.0 stereo microphone integration
 *
 * Bridges PDM microphone PCM buffers to TinyUSB audio class driver.
 */

#ifndef USB_AUDIO_H
#define USB_AUDIO_H

#include <stddef.h>
#include <stdint.h>

#include "board.h"

#define USBD_AUDIO_EP_IN BOARD_AUDIO_EP_IN

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize USB audio subsystem.
 *
 * Sets up audio control state (volume, mute, sample frequency).
 * Must be called after TinyUSB is initialized.
 */
void usb_audio_init(void);

/**
 * @brief Feed interleaved stereo 16-bit PCM into USB audio pipeline.
 *
 * Writes PCM data directly to TinyUSB audio FIFO.
 * Call from the main loop (same context as tud_task()).
 *
 * @param pcm_buf      Interleaved stereo 16-bit PCM (L, R, L, R, ...).
 * @param sample_count Total number of int16_t samples (frames * 2).
 */
void usb_audio_feed(const int16_t *pcm_buf, size_t sample_count);

#ifdef __cplusplus
}
#endif

#endif /* USB_AUDIO_H */

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file status_led.h
 * @brief Pattern-based LED status indicator with priority-based fault reporting
 *
 * The LED plays a looping blink pattern determined by the highest-priority
 * active fault flag.  Fault flags are split into two banks so that Core 0
 * and Core 1 can write independently without RMW races.
 *
 * Patterns (highest priority first):
 *   Power fault   — 2 Hz rapid blink  (VBUS low + AUX not connected)
 *   Encoder init  — triple-blink + pause
 *   Encoder read  — double-blink + pause
 *   USB disconnect — 1 Hz steady blink
 *   Normal        — brief heartbeat flash every 2 s
 */

#ifndef STATUS_LED_H
#define STATUS_LED_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Blink-pattern primitives ---- */

typedef struct {
    uint16_t on_ms;
    uint16_t off_ms;
} status_led_step_t;

typedef struct {
    const status_led_step_t *steps;
    uint8_t count;
} status_led_pattern_t;

/* ---- Fault flags (bit positions, higher = higher priority) ---- */

#define STATUS_FAULT_POWER      (1u << 4)
#define STATUS_FAULT_ENC_INIT   (1u << 3)
#define STATUS_FAULT_ENC_READ   (1u << 2)
#define STATUS_FAULT_USB        (1u << 1)

/**
 * @brief Initialise the LED pattern player.  Call once after board_gpio_init().
 */
void status_led_init(void);

/**
 * @brief Advance the pattern player.  Call from Core 0 main loop.
 */
void status_led_task(void);

/* Core 0 fault flag accessors (not safe to call from Core 1) */
void status_led_set_fault(uint8_t flag);
void status_led_clear_fault(uint8_t flag);

/* Core 1 fault flag accessors (not safe to call from Core 0) */
void status_led_core1_set_fault(uint8_t flag);
void status_led_core1_clear_fault(uint8_t flag);

#ifdef __cplusplus
}
#endif

#endif /* STATUS_LED_H */

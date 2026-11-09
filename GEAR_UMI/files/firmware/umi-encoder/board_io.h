/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file board_io.h
 * @brief Board-level I/O helpers: LED, DFP VBUS, Power Good, ADC monitoring
 */

#ifndef BOARD_IO_H
#define BOARD_IO_H

#include <stdbool.h>
#include <stdint.h>

#include "board.h"
#include "hardware/gpio.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- LED ---- */

static inline void board_led_set(bool on) {
    gpio_put(BOARD_LED_PIN, on);
}

static inline void board_led_toggle(void) {
    gpio_xor_mask(1u << BOARD_LED_PIN);
}

/* ---- iPhone DFP VBUS load switch (SY6280AAC) ---- */

static inline void board_dfp_vbus_set(bool enable) {
    gpio_put(BOARD_DFP_VBUS_EN_PIN, enable);
}

static inline bool board_dfp_vbus_get(void) {
    return gpio_get(BOARD_DFP_VBUS_EN_PIN);
}

/* ---- Power Good (LM73100) ---- */

typedef struct {
    bool vbus_pg;     /* true = LM73100 #1 output OK */
    bool vaux_pg;     /* true = LM73100 #2 output OK */
} board_pg_status_t;

static inline board_pg_status_t board_pg_read(void) {
    board_pg_status_t s;
    s.vbus_pg = gpio_get(BOARD_VBUS_PG_PIN);
    s.vaux_pg = gpio_get(BOARD_VAUX_PG_PIN);
    return s;
}

/* ---- ADC power monitoring ---- */

typedef struct {
    float vbus_v;     /* VBUS voltage (V), after divider compensation */
    float vbus_i;     /* VBUS current (A) */
    float vaux_v;     /* AUX voltage (V), after divider compensation */
    float vaux_i;     /* AUX current (A) */
} board_power_reading_t;

/**
 * @brief Read all 4 ADC power monitoring channels.
 *
 * Sequentially samples ADC0..3 and converts to physical units.
 * Not ISR-safe — call from main loop only.
 */
board_power_reading_t board_power_read(void);

#ifdef __cplusplus
}
#endif

#endif /* BOARD_IO_H */

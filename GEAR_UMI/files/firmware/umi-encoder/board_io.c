/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file board_io.c
 * @brief Board-level I/O: ADC power monitoring implementation
 *
 * Inline functions (LED, DFP VBUS, PG) live in board_io.h.
 * This file implements the ADC-based power measurement which
 * requires sequential channel switching.
 */

#include "board_io.h"

#include "hardware/adc.h"
#include "hardware/gpio.h"

static float adc_read_voltage(uint channel) {
    adc_select_input(channel);
    uint16_t raw = adc_read();
    return (float)raw * BOARD_ADC_VREF / BOARD_ADC_MAX;
}

board_power_reading_t board_power_read(void) {
    board_power_reading_t r;

    float v_imon_bus  = adc_read_voltage(BOARD_VBUS_IMON_ADC_CH);
    float v_vmon_bus  = adc_read_voltage(BOARD_VBUS_VMON_ADC_CH);
    float v_imon_aux  = adc_read_voltage(BOARD_VAUX_IMON_ADC_CH);
    float v_vmon_aux  = adc_read_voltage(BOARD_VAUX_VMON_ADC_CH);

    r.vbus_v = v_vmon_bus * BOARD_VMON_DIVIDER_RATIO;
    r.vaux_v = v_vmon_aux * BOARD_VMON_DIVIDER_RATIO;

    r.vbus_i = v_imon_bus * BOARD_IMON_SCALE;
    r.vaux_i = v_imon_aux * BOARD_IMON_SCALE;

    return r;
}

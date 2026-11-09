/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file status_led.c
 * @brief Pattern-based LED status indicator implementation
 */

#include "status_led.h"
#include "board_io.h"
#include "pico/time.h"

/* ------------------------------------------------------------------ */
/*  Pattern definitions                                                */
/* ------------------------------------------------------------------ */

static const status_led_step_t s_pat_power[]   = { {250, 250} };
static const status_led_step_t s_pat_enc_init[] = { {100, 100}, {100, 100}, {100, 700} };
static const status_led_step_t s_pat_enc_read[] = { {100, 100}, {100, 700} };
static const status_led_step_t s_pat_usb[]     = { {500, 500} };
static const status_led_step_t s_pat_normal[]  = { {50, 1950} };

#define PAT(arr) { .steps = (arr), .count = sizeof(arr) / sizeof((arr)[0]) }

static const status_led_pattern_t s_patterns[] = {
    [4] = PAT(s_pat_power),
    [3] = PAT(s_pat_enc_init),
    [2] = PAT(s_pat_enc_read),
    [1] = PAT(s_pat_usb),
    [0] = PAT(s_pat_normal),
};

#define PATTERN_COUNT (sizeof(s_patterns) / sizeof(s_patterns[0]))

/* ------------------------------------------------------------------ */
/*  Fault flag banks (separate per core to avoid RMW races)            */
/* ------------------------------------------------------------------ */

static volatile uint8_t g_core0_faults;
static volatile uint8_t g_core1_faults;

void status_led_set_fault(uint8_t flag)         { g_core0_faults |=  flag; }
void status_led_clear_fault(uint8_t flag)       { g_core0_faults &= ~flag; }
void status_led_core1_set_fault(uint8_t flag)   { g_core1_faults |=  flag; }
void status_led_core1_clear_fault(uint8_t flag) { g_core1_faults &= ~flag; }

/* ------------------------------------------------------------------ */
/*  Pattern player state                                               */
/* ------------------------------------------------------------------ */

static uint8_t  s_active_idx;     /* index into s_patterns[] */
static uint8_t  s_step;           /* current step within pattern */
static bool     s_phase_on;       /* true = ON half, false = OFF half */
static absolute_time_t s_next_toggle;

static uint8_t highest_fault_index(uint8_t combined) {
    /* Return index of highest set bit, or 0 (normal) if none set. */
    for (int i = PATTERN_COUNT - 1; i > 0; i--) {
        if (combined & (1u << i))
            return (uint8_t)i;
    }
    return 0;
}

static void start_pattern(uint8_t idx) {
    s_active_idx = idx;
    s_step = 0;
    s_phase_on = true;
    board_led_set(true);
    s_next_toggle = delayed_by_ms(get_absolute_time(),
                                  s_patterns[idx].steps[0].on_ms);
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

void status_led_init(void) {
    g_core0_faults = 0;
    g_core1_faults = 0;
    start_pattern(0);
}

void status_led_task(void) {
    uint8_t combined = g_core0_faults | g_core1_faults;
    uint8_t wanted = highest_fault_index(combined);

    if (wanted != s_active_idx)
        start_pattern(wanted);

    if (absolute_time_diff_us(s_next_toggle, get_absolute_time()) < 0)
        return;

    const status_led_pattern_t *pat = &s_patterns[s_active_idx];

    if (s_phase_on) {
        /* ON phase just ended → turn off, schedule OFF duration */
        board_led_set(false);
        s_phase_on = false;
        s_next_toggle = delayed_by_ms(get_absolute_time(),
                                      pat->steps[s_step].off_ms);
    } else {
        /* OFF phase just ended → advance to next step (wrap), turn on */
        s_step++;
        if (s_step >= pat->count)
            s_step = 0;

        board_led_set(true);
        s_phase_on = true;
        s_next_toggle = delayed_by_ms(get_absolute_time(),
                                      pat->steps[s_step].on_ms);
    }
}

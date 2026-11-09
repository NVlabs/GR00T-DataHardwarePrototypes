/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file encoder_runtime.c
 * @brief Implementation of the encoder runtime (Core 1 state) and the
 *        Core 1 → Core 0 diagnostic log queue.
 *
 * See encoder_runtime.h for the threading model and API contract.
 */

#include "encoder_runtime.h"

#include "board.h"
#include "pico/util/queue.h"
#include "status_led.h"

#include <stdio.h>
#include <string.h>

/* ---------------------------------------------------------------------------
 *  Inter-core diagnostic log queue
 *
 *  Owned by this module so main.c does not need to know how Core 1 surfaces
 *  encoder events to Core 0. Producer (Core 1) is the runtime instance API
 *  below; consumer (Core 0) uses encoder_runtime_pop_diag_log_entry().
 * ------------------------------------------------------------------------- */

#define ENCODER_RUNTIME_DIAG_QUEUE_DEPTH 8

/* One entry in the Core 1 → Core 0 diag log queue. Internal to this module
 * — main.c never sees it; it only triggers drain via the public API. */
typedef struct {
    tle5012b_diag_event_t diag;
    bool recovered; /* true = "fault → recovered" event */
} encoder_diag_log_entry_t;

static queue_t g_diag_queue;

void encoder_runtime_subsystem_init(void) {
    queue_init(&g_diag_queue, sizeof(encoder_diag_log_entry_t), ENCODER_RUNTIME_DIAG_QUEUE_DEPTH);
}

static const char *error_name(tle5012b_error_t err) {
    switch (err) {
        case TLE5012B_OK:
            return "ok";
        case TLE5012B_ERR_SYSTEM:
            return "system error";
        case TLE5012B_ERR_INTERFACE:
            return "interface error";
        case TLE5012B_ERR_INVALID_ANGLE:
            return "invalid angle";
        case TLE5012B_ERR_ANGLE_SPEED:
            return "angle speed error";
        case TLE5012B_ERR_CRC:
            return "crc error";
        case TLE5012B_ERR_INVALID_PARAM:
            return "invalid param";
        case TLE5012B_ERR_SPI_INIT:
            return "spi init error";
        case TLE5012B_ERR_TIMEOUT:
            return "timeout";
        case TLE5012B_ERR_STATUS:
            return "status error";
        case TLE5012B_ERR_NOT_INIT:
            return "not initialized";
        default:
            return "unknown error";
    }
}

/* Decode the bit-flag soup in a STAT register reading into a single
 * "[ERROR][TLE5012B] STAT flags: ..." line. Used when a diag log entry
 * carries a captured status word, so the log shows *which* fault bits the
 * sensor latched, not just the raw hex. */
static void print_status_flags(uint16_t status) {
    bool printed = false;

    printf("[ERROR][TLE5012B] STAT flags:");
    if (status & TLE5012B_STAT_SRST) {
        printf(" SRST");
        printed = true;
    }
    if (status & TLE5012B_STAT_SWD) {
        printf(" SWD");
        printed = true;
    }
    if (status & TLE5012B_STAT_SVR) {
        printf(" SVR");
        printed = true;
    }
    if (status & TLE5012B_STAT_SFUSE) {
        printf(" SFUSE");
        printed = true;
    }
    if (status & TLE5012B_STAT_SDSPU) {
        printf(" SDSPU");
        printed = true;
    }
    if (status & TLE5012B_STAT_SOV) {
        printf(" SOV");
        printed = true;
    }
    if (status & TLE5012B_STAT_SXYOL) {
        printf(" SXYOL");
        printed = true;
    }
    if (status & TLE5012B_STAT_SMAGOL) {
        printf(" SMAGOL");
        printed = true;
    }
    if (status & TLE5012B_STAT_SADCT) {
        printf(" SADCT");
        printed = true;
    }
    if (status & TLE5012B_STAT_SROM) {
        printf(" SROM");
        printed = true;
    }
    if (status & TLE5012B_STAT_NOGMRXY) {
        printf(" NOGMRXY");
        printed = true;
    }
    if (status & TLE5012B_STAT_NOGMRA) {
        printf(" NOGMRA");
        printed = true;
    }
    if (!printed) {
        printf(" none");
    }
    printf("\r\n");
}

void encoder_runtime_drain_diag_log_to_stdout(void) {
    encoder_diag_log_entry_t entry;

    while (queue_try_remove(&g_diag_queue, &entry)) {
        const tle5012b_diag_event_t *diag = &entry.diag;

        if (entry.recovered) {
            printf("[INFO][TLE5012B] Encoder fault cleared: %s cmd=0x%04X safety=0x%04X",
                   error_name(diag->error), diag->command, diag->safety_word);
            if (diag->has_status) {
                printf(" last_status=0x%04X", diag->status_word);
            }
            printf("\r\n");
            continue;
        }

        printf("[ERROR][TLE5012B] Encoder %s: cmd=0x%04X safety=0x%04X", error_name(diag->error), diag->command,
               diag->safety_word);
        if (!(diag->safety_word & TLE5012B_SAFETY_STAT_OK)) {
            printf(" STAT_ERR");
        }
        if (!(diag->safety_word & TLE5012B_SAFETY_INTF_OK)) {
            printf(" INTF_ERR");
        }
        if (!(diag->safety_word & TLE5012B_SAFETY_INV_OK)) {
            printf(" ANG_INVALID");
        }
        if (diag->has_status) {
            printf(" status=0x%04X", diag->status_word);
        }
        printf("\r\n");

        if (diag->has_status) {
            print_status_flags(diag->status_word);
        }
    }
}

/* ---------------------------------------------------------------------------
 *  Internal helpers (Core 1 only)
 * ------------------------------------------------------------------------- */

static bool diag_event_equal(const tle5012b_diag_event_t *a, const tle5012b_diag_event_t *b) {
    return a && b && a->error == b->error && a->command == b->command && a->safety_word == b->safety_word &&
           a->status_word == b->status_word && a->has_status == b->has_status;
}

static void enqueue_diag_log(const tle5012b_diag_event_t *diag, bool recovered) {
    if (!diag) {
        return;
    }
    encoder_diag_log_entry_t entry = {
        .diag = *diag,
        .recovered = recovered,
    };
    queue_try_add(&g_diag_queue, &entry);
}

/* Fold a fresh diag event into the wire-bound health snapshot.
 * Does NOT touch state / fail_streak / queues — that's the caller's job. */
static void apply_diag_to_health(encoder_runtime_t *rt, const tle5012b_diag_event_t *diag) {
    rt->health.last_error = (uint8_t)diag->error;
    rt->health.safety_word = diag->safety_word;
    rt->health.status_word = diag->has_status ? diag->status_word : 0;
    rt->health.error_count++;
}

/* ---------------------------------------------------------------------------
 *  Core 1 instance API
 * ------------------------------------------------------------------------- */

void encoder_runtime_init(encoder_runtime_t *rt) {
    *rt = (encoder_runtime_t){
        .health = {.state = UMI_SENSOR_STATE_OK},
    };
}

void encoder_runtime_handle_init_failure(encoder_runtime_t *rt, tle5012b_dev_t *dev, tle5012b_error_t init_err) {
    tle5012b_diag_event_t init_diag;
    if (!tle5012b_take_diag_event(dev, &init_diag)) {
        init_diag = (tle5012b_diag_event_t){.error = init_err};
    }
    enqueue_diag_log(&init_diag, false);
    status_led_core1_set_fault(STATUS_FAULT_ENC_INIT);

    apply_diag_to_health(rt, &init_diag);
    rt->health.state = UMI_SENSOR_STATE_INIT_FAILED;
    rt->init_failed = true;
}

void encoder_runtime_drain_pending_diag(encoder_runtime_t *rt, tle5012b_dev_t *dev) {
    tle5012b_diag_event_t diag;
    if (!tle5012b_take_diag_event(dev, &diag)) {
        return;
    }
    enqueue_diag_log(&diag, false);
    rt->last_diag = diag;
    rt->diag_active = true;
    apply_diag_to_health(rt, &diag);
}

bool encoder_runtime_observe_read(encoder_runtime_t *rt, tle5012b_dev_t *dev, tle5012b_error_t read_err) {
    /* INIT_FAILED is sticky; skip all per-frame bookkeeping and keep
     * reporting INIT_FAILED in the SENSOR_HEALTH TLV. */
    if (rt->init_failed) {
        return false;
    }

    tle5012b_diag_event_t diag;
    bool have_diag = tle5012b_take_diag_event(dev, &diag);

    if (have_diag) {
        /* De-duplicate identical events to avoid flooding the Core 0 log
         * queue when a fault is stuck (would otherwise emit one message
         * per encoder sample = BOARD_ENCODER_REFRESH_RATE_HZ Hz). */
        if (!rt->diag_active || !diag_event_equal(&diag, &rt->last_diag)) {
            enqueue_diag_log(&diag, false);
            rt->last_diag = diag;
        }
        rt->diag_active = true;
        apply_diag_to_health(rt, &diag);
    } else if (rt->diag_active && read_err == TLE5012B_OK) {
        /* Fault → recovered transition: log it once, then arm the
         * detector for the next fault. */
        enqueue_diag_log(&rt->last_diag, true);
        rt->diag_active = false;
    }

    bool read_ok = (read_err == TLE5012B_OK) && !have_diag;

    /* Drive fail_streak (and the LED that mirrors it). */
    if (!read_ok) {
        if (++rt->fail_streak >= BOARD_ENCODER_FAULT_STREAK_MAX) {
            status_led_core1_set_fault(STATUS_FAULT_ENC_READ);
        }
    } else if (rt->fail_streak) {
        rt->fail_streak = 0;
        status_led_core1_clear_fault(STATUS_FAULT_ENC_READ);
    }

    /* Derive health.state from fail_streak (INIT_FAILED already returned
     * above). LED policy and state policy share the same threshold so the
     * two stay in lock-step. */
    if (rt->fail_streak == 0) {
        rt->health.state = UMI_SENSOR_STATE_OK;
    } else if (rt->fail_streak >= BOARD_ENCODER_FAULT_STREAK_MAX) {
        rt->health.state = UMI_SENSOR_STATE_FAULT;
    } else {
        rt->health.state = UMI_SENSOR_STATE_DEGRADED;
    }

    return read_ok;
}

float encoder_runtime_pick_angle(encoder_runtime_t *rt, bool read_ok, bool spi_returned_ok, float fresh_angle) {
    if (read_ok) {
        rt->health.last_good_angle_deg = fresh_angle;
        rt->health.has_last_good = true;
        return fresh_angle;
    }
    if (spi_returned_ok) {
        return fresh_angle;
    }
    if (rt->health.has_last_good) {
        return rt->health.last_good_angle_deg;
    }
    return 0.0f;
}

size_t encoder_runtime_pack_health_tlv_body(const encoder_runtime_t *rt, uint8_t *buf, size_t cap) {
    const size_t need = sizeof(umi_sensor_health_header_t) + sizeof(tle5012b_health_extra_t);
    if (cap < need) {
        return 0;
    }

    umi_sensor_health_header_t hdr = {
        .source_sensor_type = UMI_SENSOR_TYPE_JOINT_ENCODER,
        .state              = rt->health.state,
        .last_error         = rt->health.last_error,
        .reserved           = 0,
        .error_count        = rt->health.error_count,
    };
    tle5012b_health_extra_t extra = {
        .safety_word = rt->health.safety_word,
        .status_word = rt->health.status_word,
    };
    memcpy(buf,               &hdr,   sizeof(hdr));
    memcpy(buf + sizeof(hdr), &extra, sizeof(extra));
    return need;
}

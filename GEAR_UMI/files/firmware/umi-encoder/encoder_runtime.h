/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file encoder_runtime.h
 * @brief Core 1 state box for the joint encoder, plus its Core 0 diag log sink.
 *
 * This module owns everything the joint encoder needs to remember between
 * samples on Core 1 (health snapshot, fail-streak counter, diag dedup state,
 * sticky init-failure flag) and the inter-core diagnostic log queue that
 * surfaces those events to Core 0 for printing.
 *
 * Threading model:
 *   - All `encoder_runtime_*` instance APIs (the ones taking a
 *     `encoder_runtime_t *`) are Core 1 only. They mutate the runtime,
 *     interact with the TLE5012B driver, and push diag log entries onto an
 *     internal Core 1 → Core 0 queue.
 *   - `encoder_runtime_subsystem_init()` is called once on Core 0 before
 *     `multicore_launch_core1()`.
 *   - `encoder_runtime_drain_diag_log_to_stdout()` is Core 0 only; call it
 *     periodically from the main loop to drain queued diagnostics into
 *     printf / USB CDC stdout.
 *
 * Wire convention reminder: in the SENSOR_HEALTH TLV, status_word == 0 means
 * "STAT register was not captured for the latest fault" (e.g. CRC error
 * paths skip the read), so a separate has_status flag is not transported.
 */

#ifndef ENCODER_RUNTIME_H
#define ENCODER_RUNTIME_H

#include "tle5012b.h"
#include "umi_protocol.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 *  Public, wire-bound encoder health snapshot
 * ------------------------------------------------------------------------- */

/* Mirrored verbatim into the SENSOR_HEALTH TLV body (after the umi_sensor_
 * health_header_t prefix). Owned by encoder_runtime_t.health. */
typedef struct {
    uint8_t state;        /* umi_sensor_state_t */
    uint8_t last_error;   /* tle5012b_error_t */
    uint16_t safety_word; /* last safety word from the sensor */
    uint16_t status_word; /* last STAT value, 0 if not captured */
    uint32_t error_count; /* monotonic, wraps after 2^32 */
    bool has_last_good;
    float last_good_angle_deg;
} encoder_health_t;

/* Opaque-by-convention runtime state. The struct is exposed so callers can
 * stack-allocate it; touch only `health` and `init_failed` from the outside. */
typedef struct {
    encoder_health_t health;
    uint32_t fail_streak;
    tle5012b_diag_event_t last_diag;
    bool diag_active;
    bool init_failed;
} encoder_runtime_t;

/* ---------------------------------------------------------------------------
 *  Inter-core diagnostic log (Core 0 side)
 * ------------------------------------------------------------------------- */

/* Initialise the inter-core diag log queue. Must be called on Core 0 once
 * before launching Core 1. Not safe to call twice. */
void encoder_runtime_subsystem_init(void);

/* Drain every queued diag entry into printf / USB CDC stdout. Each entry
 * renders as one or two human-readable lines describing the latest error
 * (or the fault → recovered transition). Non-blocking — returns once the
 * queue is empty, so the main loop calls it once per iteration. */
void encoder_runtime_drain_diag_log_to_stdout(void);

/* ---------------------------------------------------------------------------
 *  Core 1 instance API
 * ------------------------------------------------------------------------- */

/* Reset the runtime to its post-boot defaults (state = OK, no last_good,
 * counters zeroed). Call once before the first encoder_runtime_observe_read. */
void encoder_runtime_init(encoder_runtime_t *rt);

/* Call once when tle5012b_init() returned an error. Latches the cause into
 * health, queues one diag log entry for Core 0, lights the init-fault LED,
 * and flips the sticky init_failed flag so subsequent frames skip the SPI. */
void encoder_runtime_handle_init_failure(encoder_runtime_t *rt, tle5012b_dev_t *dev,
                                         tle5012b_error_t init_err);

/* Call once after a successful tle5012b_init() to absorb any pending diag
 * the driver raised during init (e.g. a transient that cleared before the
 * first user-level read). No-op if the driver has no pending diag. */
void encoder_runtime_drain_pending_diag(encoder_runtime_t *rt, tle5012b_dev_t *dev);

/* Call once per sample after tle5012b_read_angle(). Folds in any pending
 * diag from the driver, manages "fault → recovered" transitions, updates
 * fail_streak, drives the read-fault LED, and recomputes health.state.
 *
 * Returns read_ok = (read_err == OK) and no diag was raised this frame.
 * The SYSTEM_ERROR soft-failure path (driver returns OK but raises a diag)
 * is therefore treated as a "soft failure" for state-machine purposes — angle
 * data is still publishable (CRC passed) but health.state degrades.
 *
 * No-op when rt->init_failed: returns false so the caller falls back to
 * last_good (or 0.0f). */
bool encoder_runtime_observe_read(encoder_runtime_t *rt, tle5012b_dev_t *dev, tle5012b_error_t read_err);

/* Pick which angle to ship in the JOINT_ENCODER TLV.
 *   - read_ok                              → fresh sample, also memo as last_good
 *   - SPI returned OK but a diag fired     → publish fresh (CRC passed) but
 *                                            do not overwrite last_good
 *   - hard failure / init failure          → fall back to last_good
 *   - no last_good ever recorded           → 0.0f; host should see state != OK
 *                                            and ignore the value */
float encoder_runtime_pick_angle(encoder_runtime_t *rt, bool read_ok, bool spi_returned_ok, float fresh_angle);

/* Serialise the SENSOR_HEALTH TLV body (generic header + TLE5012B private
 * extra) into `buf`. Returns the number of bytes written, or 0 if `cap` is
 * too small. */
size_t encoder_runtime_pack_health_tlv_body(const encoder_runtime_t *rt, uint8_t *buf, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* ENCODER_RUNTIME_H */

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef UMI_PROTOCOL_H
#define UMI_PROTOCOL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define UMI_SYNC_WORD (0x5AA5u)

/* UMI_PROTOCOL_VERSION is a single monotonically-increasing byte that
 * captures every observable change to the wire format. Bump it whenever:
 *
 *   - A new sensor_type is allocated.
 *   - A field is added/removed/resized inside any TLV body.
 *   - The frame header layout changes.
 *   - The semantics of any existing byte change.
 *
 * Hosts should compare against the highest version they understand and
 * refuse to parse frames from devices with a higher version (or fall back
 * to a strict subset). 0xFF is reserved for "experimental".
 *
 * History:
 *   0x11 - Initial release (JOINT_ENCODER only).
 *   0x12 - Added SENSOR_HEALTH TLV (umi_sensor_health_header_t + opaque
 *          device-specific extra payload). */
#define UMI_PROTOCOL_VERSION (0x12u)

#define UMI_MAX_PAYLOAD_BYTES (65535u)
#define UMI_SENSOR_TLV_HEADER_SIZE (4u)
#define UMI_DEVICE_ID_BYTES (8u)

#define UMI_SOURCE_ID(type, instance) ((uint8_t)(((type)&0xF0u) | ((instance)&0x0Fu)))

typedef enum {
    UMI_SENSOR_TYPE_JOINT_ENCODER = 0x01,
    /* 0xF0..0xFF reserved for meta TLVs (not physical sensors).
     * SENSOR_HEALTH carries health/diagnostic information about another
     * sensor instance; the outer TLV's sensor_id must equal the sensor_id
     * of the physical sensor it describes (e.g. encoder #0's health TLV
     * uses sensor_id = 0). */
    UMI_SENSOR_TYPE_SENSOR_HEALTH = 0xF0,
} umi_sensor_type_t;

typedef enum {
    UMI_SOURCE_TYPE_UMI_GRIPPER = 0x10,
    UMI_SOURCE_TYPE_DEX_UMI = 0x20,
} umi_source_type_t;

/* Health state for umi_sensor_health_header_t.state.
 *
 * The state machine is driven by the producer (firmware). Consumers should
 * treat unknown values as FAULT to fail safe. */
typedef enum {
    UMI_SENSOR_STATE_OK = 0,           /* Sensor working normally. */
    UMI_SENSOR_STATE_DEGRADED = 1,     /* Transient errors observed; auto-recovering. */
    UMI_SENSOR_STATE_FAULT = 2,        /* Persistent errors past the streak threshold. */
    UMI_SENSOR_STATE_INIT_FAILED = 3,  /* Sensor did not initialise on boot. */
} umi_sensor_state_t;

#pragma pack(push, 1)
typedef struct {
    uint16_t sync;
    uint8_t version;
    uint8_t source;
    uint8_t device_id[UMI_DEVICE_ID_BYTES];
    uint32_t timestamp_ms;
    uint16_t payload_length;
} umi_frame_header_t;

typedef struct {
    uint8_t sensor_type;
    uint8_t sensor_id;
    uint16_t data_length;
} umi_sensor_tlv_t;

/* Generic header at the start of every SENSOR_HEALTH TLV body.
 *
 * Wire layout: [umi_sensor_health_header_t (8 B) | extra payload (N B)]
 *
 * The extra payload is opaque to the protocol; its layout is private to
 * the driver of `source_sensor_type` (e.g. tle5012b.h defines
 * tle5012b_health_extra_t for TLE5012B encoders). To keep parsing simple,
 * per-sensor variable extra-length is deliberately not supported: each
 * sensor type publishes a fixed-size extra struct, and the wire length of
 * a SENSOR_HEALTH TLV is therefore deterministic for each sensor type.
 * Any change to the extra struct of an existing sensor type is a wire
 * format break and must bump UMI_PROTOCOL_VERSION.
 *
 * Field semantics:
 *   source_sensor_type : umi_sensor_type_t of the physical sensor this
 *                        report describes (e.g. JOINT_ENCODER). Lets a
 *                        dispatcher disambiguate when multiple sensor
 *                        types share an instance id.
 *   state              : umi_sensor_state_t (OK/DEGRADED/FAULT/INIT_FAILED).
 *   last_error         : Last driver-level error code. Its enumeration
 *                        depends on source_sensor_type — consult the
 *                        relevant driver header (e.g. tle5012b_error_t).
 *   reserved           : Padding for 4-byte alignment of error_count;
 *                        must be zero on the wire.
 *   error_count        : Monotonic count of failures since boot. Wraps
 *                        after 2^32. Hosts compute Δ between frames to
 *                        derive an instantaneous error rate. */
typedef struct {
    uint8_t source_sensor_type;
    uint8_t state;
    uint8_t last_error;
    uint8_t reserved;
    uint32_t error_count;
} umi_sensor_health_header_t;
#pragma pack(pop)

typedef struct {
    uint8_t version;
    uint8_t source;
    uint8_t device_id[UMI_DEVICE_ID_BYTES];
    uint32_t timestamp_ms;
} umi_frame_meta_t;

typedef struct {
    uint8_t *buffer;
    size_t capacity;
    uint16_t length;
} umi_payload_builder_t;

void umi_payload_builder_init(umi_payload_builder_t *builder, uint8_t *buffer, size_t capacity);
bool umi_payload_builder_add_segment(umi_payload_builder_t *builder, uint8_t sensor_type, uint8_t sensor_id,
                                     const void *data, uint16_t data_length);
uint16_t umi_payload_builder_size(const umi_payload_builder_t *builder);

size_t umi_frame_encode(const umi_frame_meta_t *meta, const umi_payload_builder_t *payload, uint8_t *out_frame,
                        size_t frame_capacity);

uint16_t umi_crc16_ccitt(const uint8_t *data, size_t length);

#ifdef __cplusplus
}
#endif

#endif /* UMI_PROTOCOL_H */

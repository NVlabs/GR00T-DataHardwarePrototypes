/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "umi_protocol.h"
#include "crc16.h"

#include <string.h>

void umi_payload_builder_init(umi_payload_builder_t *builder, uint8_t *buffer, size_t capacity) {
    if (!builder) {
        return;
    }

    builder->buffer = buffer;
    builder->capacity = capacity;
    builder->length = 0;
}

bool umi_payload_builder_add_segment(umi_payload_builder_t *builder, uint8_t sensor_type, uint8_t sensor_id,
                                     const void *data, uint16_t data_length) {
    if (!builder || !builder->buffer) {
        return false;
    }

    size_t required = (size_t)builder->length + UMI_SENSOR_TLV_HEADER_SIZE + (size_t)data_length;
    if (required > builder->capacity || required > UMI_MAX_PAYLOAD_BYTES) {
        return false;
    }

    uint8_t *cursor = builder->buffer + builder->length;
    cursor[0] = sensor_type;
    cursor[1] = sensor_id;
    cursor[2] = (uint8_t)(data_length & 0xFFu);
    cursor[3] = (uint8_t)(data_length >> 8);

    if (data_length && data) {
        memcpy(cursor + UMI_SENSOR_TLV_HEADER_SIZE, data, data_length);
    }

    builder->length = (uint16_t)required;
    return true;
}

uint16_t umi_payload_builder_size(const umi_payload_builder_t *builder) { return builder ? builder->length : 0u; }

size_t umi_frame_encode(const umi_frame_meta_t *meta, const umi_payload_builder_t *payload, uint8_t *out_frame,
                        size_t frame_capacity) {
    if (!meta || !payload || !out_frame) {
        return 0u;
    }

    uint16_t payload_len = umi_payload_builder_size(payload);
    size_t needed = sizeof(umi_frame_header_t) + (size_t)payload_len + sizeof(uint16_t);

    if (frame_capacity < needed) {
        return 0u;
    }

    umi_frame_header_t header = {
        .sync = UMI_SYNC_WORD,
        .version = meta->version ? meta->version : UMI_PROTOCOL_VERSION,
        .source = meta->source,
        .timestamp_ms = meta->timestamp_ms,
        .payload_length = payload_len,
    };
    memcpy(header.device_id, meta->device_id, UMI_DEVICE_ID_BYTES);
    memcpy(out_frame, &header, sizeof(header));

    if (payload_len && payload->buffer) {
        memcpy(out_frame + sizeof(header), payload->buffer, payload_len);
    }

    const uint8_t *crc_start = out_frame + sizeof(header.sync);
    size_t crc_length = sizeof(header) - sizeof(header.sync) + (size_t)payload_len;
    uint16_t crc = crc16((const char *)crc_start, (int)crc_length);
    memcpy(out_frame + sizeof(header) + payload_len, &crc, sizeof(crc));

    return needed;
}

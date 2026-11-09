/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cdc_uart.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tusb.h"

#include "board.h"

#ifndef CDC_UART_ITF
#define CDC_UART_ITF BOARD_CDC_UART_ITF
#endif

#if CFG_TUD_CDC < 2
#error "cdc_uart module requires CFG_TUD_CDC >= 2"
#endif

static void cdc_uart_service(void) { tud_task(); }

bool cdc_uart_connected(void) { return tud_cdc_n_connected(CDC_UART_ITF); }

size_t cdc_uart_write(const void *data, size_t len) {
    if (!data || !len || !cdc_uart_connected()) {
        return 0;
    }

    const uint8_t *ptr = (const uint8_t *)data;
    size_t written = 0;

    while (len) {
        uint32_t avail = tud_cdc_n_write_available(CDC_UART_ITF);
        if (!avail) {
            cdc_uart_service();
            continue;
        }

        uint32_t chunk = (uint32_t)(len < avail ? len : avail);
        uint32_t pushed = tud_cdc_n_write(CDC_UART_ITF, ptr, chunk);
        tud_cdc_n_write_flush(CDC_UART_ITF);

        ptr += pushed;
        len -= pushed;
        written += pushed;
    }

    return written;
}

int cdc_uart_printf(const char *fmt, ...) {
    if (!fmt) {
        return 0;
    }

    char buffer[128];
    va_list args;
    va_start(args, fmt);
    int needed = vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    if (needed <= 0) {
        return needed;
    }

    if ((size_t)needed < sizeof(buffer)) {
        cdc_uart_write(buffer, (size_t)needed);
        return needed;
    }

    char *dynamic = malloc((size_t)needed + 1u);
    if (!dynamic) {
        return 0;
    }

    va_start(args, fmt);
    vsnprintf(dynamic, (size_t)needed + 1u, fmt, args);
    va_end(args);

    cdc_uart_write(dynamic, (size_t)needed);
    free(dynamic);
    return needed;
}

size_t cdc_uart_read(void *buffer, size_t len) {
    if (!buffer || !len || !tud_cdc_n_available(CDC_UART_ITF)) {
        return 0;
    }

    return tud_cdc_n_read(CDC_UART_ITF, buffer, (uint32_t)len);
}

bool cdc_uart_has_data(void) { return tud_cdc_n_available(CDC_UART_ITF) > 0; }

void cdc_uart_task(void) { cdc_uart_service(); }

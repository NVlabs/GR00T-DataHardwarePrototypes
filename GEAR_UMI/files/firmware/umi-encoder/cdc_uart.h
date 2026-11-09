/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CDC_UART_H
#define CDC_UART_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool cdc_uart_connected(void);
size_t cdc_uart_write(const void *data, size_t len);
int cdc_uart_printf(const char *fmt, ...);
size_t cdc_uart_read(void *buffer, size_t len);
bool cdc_uart_has_data(void);
void cdc_uart_task(void);

#ifdef __cplusplus
}
#endif

#endif /* CDC_UART_H */

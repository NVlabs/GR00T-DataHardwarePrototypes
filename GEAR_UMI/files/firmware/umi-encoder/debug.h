/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file debug.h
 * @brief Global debug logging utilities
 * @author UMI Project
 */

#ifndef DEBUG_H
#define DEBUG_H

#include <stdio.h>

#ifdef DEBUG_INCLUDE_CORE_ID
#include "pico.h"
#endif

/* Global debug level control */
#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 0 // 0=Off, 1=Error, 2=Warning, 3=Info, 4=Debug, 5=Verbose
#endif

/* Debug level macros */
#define DEBUG_LEVEL_OFF 0
#define DEBUG_LEVEL_ERROR 1
#define DEBUG_LEVEL_WARN 2
#define DEBUG_LEVEL_INFO 3
#define DEBUG_LEVEL_DEBUG 4
#define DEBUG_LEVEL_VERBOSE 5

/* Optional core-id prefix helpers */
#ifdef DEBUG_INCLUDE_CORE_ID
#define DEBUG_CORE_FMT "[C%d]"
#define DEBUG_CORE_ARGS get_core_num(),
#else
#define DEBUG_CORE_FMT ""
#define DEBUG_CORE_ARGS
#endif

/* Debug output macros — output goes to stdio (USB CDC) via printf() */
#if DEBUG_LEVEL >= DEBUG_LEVEL_ERROR
#define DEBUG_ERROR(tag, fmt, ...)                                                                                     \
    printf("[ERROR]" DEBUG_CORE_FMT "[%s] " fmt "\r\n", DEBUG_CORE_ARGS tag, ##__VA_ARGS__)
#else
#define DEBUG_ERROR(tag, fmt, ...) ((void)0)
#endif

#if DEBUG_LEVEL >= DEBUG_LEVEL_WARN
#define DEBUG_WARN(tag, fmt, ...)                                                                                      \
    printf("[WARN]" DEBUG_CORE_FMT "[%s] " fmt "\r\n", DEBUG_CORE_ARGS tag, ##__VA_ARGS__)
#else
#define DEBUG_WARN(tag, fmt, ...) ((void)0)
#endif

#if DEBUG_LEVEL >= DEBUG_LEVEL_INFO
#define DEBUG_INFO(tag, fmt, ...)                                                                                      \
    printf("[INFO]" DEBUG_CORE_FMT "[%s] " fmt "\r\n", DEBUG_CORE_ARGS tag, ##__VA_ARGS__)
#else
#define DEBUG_INFO(tag, fmt, ...) ((void)0)
#endif

#if DEBUG_LEVEL >= DEBUG_LEVEL_DEBUG
#define DEBUG_LOG(tag, fmt, ...)                                                                                       \
    printf("[DEBUG]" DEBUG_CORE_FMT "[%s] " fmt "\r\n", DEBUG_CORE_ARGS tag, ##__VA_ARGS__)
#else
#define DEBUG_LOG(tag, fmt, ...) ((void)0)
#endif

#if DEBUG_LEVEL >= DEBUG_LEVEL_VERBOSE
#define DEBUG_VERBOSE(tag, fmt, ...)                                                                                   \
    printf("[VERBOSE]" DEBUG_CORE_FMT "[%s] " fmt "\r\n", DEBUG_CORE_ARGS tag, ##__VA_ARGS__)
#else
#define DEBUG_VERBOSE(tag, fmt, ...) ((void)0)
#endif

/* Hex dump utility for debugging binary data */
#if DEBUG_LEVEL >= DEBUG_LEVEL_VERBOSE
static inline void debug_hex_dump(const char *tag, const char *desc, const uint8_t *data, size_t len) {
    printf("[HEX]" DEBUG_CORE_FMT "[%s] %s: ", DEBUG_CORE_ARGS tag, desc);
    for (size_t i = 0; i < len; i++) {
        printf("%02X ", data[i]);
    }
    printf("\r\n");
}
#else
#define debug_hex_dump(tag, desc, data, len) ((void)0)
#endif

#endif // DEBUG_H

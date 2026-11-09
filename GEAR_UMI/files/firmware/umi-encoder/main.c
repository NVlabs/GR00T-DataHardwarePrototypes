/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pico/multicore.h"
#include "pico/stdlib.h" // IWYU pragma: keep
#include "pico/unique_id.h"
#include "pico/util/queue.h"
#include "hardware/adc.h"
#include "hardware/gpio.h"
#include "hardware/uart.h"
#include "tusb.h"

#include <string.h>

#include "board.h"
#include "board_io.h"
#include "cdc_uart.h"
#include "debug.h"
#include "encoder_runtime.h"
#include "pdm_mic.h"
#include "status_led.h"
#include "tle5012b.h"
#include "umi_protocol.h"
#include "usb_audio.h"

#define TAG "MAIN"

/* ------------------------------------------------------------------ */
/*  Hard Fault handler — bypasses stdio, writes directly to UART0      */
/* ------------------------------------------------------------------ */

static void fault_print(const char *s) {
    while (*s) {
        uart_putc_raw(uart0, *s++);
    }
}

static void fault_hex32(uint32_t v) {
    const char hex[] = "0123456789ABCDEF";
    fault_print("0x");
    for (int i = 28; i >= 0; i -= 4) {
        uart_putc_raw(uart0, hex[(v >> i) & 0xF]);
    }
}

void __attribute__((used)) hardfault_handler_c(uint32_t *stack) {
    fault_print("\r\n!!! HARD FAULT !!!\r\n");
    const char *names[] = {"R0","R1","R2","R3","R12","LR","PC","xPSR"};
    for (int i = 0; i < 8; i++) {
        fault_print("  "); fault_print(names[i]); fault_print(" = ");
        fault_hex32(stack[i]); fault_print("\r\n");
    }

    /* Dump memory at R0 (likely ep struct) — 32 bytes = sizeof(hw_endpoint_t) */
    uint32_t r0 = stack[0];
    if (r0 >= 0x20000000 && r0 < 0x20042000) {
        fault_print("  Memory at R0 (ep struct):\r\n");
        volatile uint32_t *p = (volatile uint32_t *)r0;
        for (int i = 0; i < 8; i++) {
            fault_print("    [+"); fault_hex32(i * 4); fault_print("] = ");
            fault_hex32(p[i]); fault_print("\r\n");
        }
        /* buffer_control is at offset 8 — dereference it */
        uint32_t bc_ptr = p[2]; /* offset 8 / 4 = index 2 */
        fault_print("  buffer_control ptr = "); fault_hex32(bc_ptr); fault_print("\r\n");
        if (bc_ptr >= 0x50100000 && bc_ptr < 0x50101000) {
            fault_print("  *buffer_control = "); fault_hex32(*(volatile uint32_t *)bc_ptr);
            fault_print(" (valid DPRAM)\r\n");
        } else {
            fault_print("  *buffer_control -> INVALID ADDRESS!\r\n");
        }
    }

    while (1) { __asm volatile("nop"); }
}

void __attribute__((naked)) isr_hardfault(void) {
    __asm volatile(
        "mrs r0, msp \n"
        "b hardfault_handler_c \n"
    );
}

#ifndef UMI_SOURCE_INSTANCE
#define UMI_SOURCE_INSTANCE 0
#endif

#define UMI_ENCODER_SENSOR_ID 0x00
/* Buffers are sized with headroom so future TLVs (additional sensors,
 * larger health extras, etc.) can be added without revisiting the
 * constants on every protocol bump. Current usage is ~24 B payload /
 * ~44 B frame; see umi_protocol.h for the frame layout. */
#define UMI_PAYLOAD_BUFFER_SIZE 128
#define UMI_FRAME_BUFFER_SIZE 192

#ifndef UMI_ENCODER_REFRESH_RATE_HZ
#define UMI_ENCODER_REFRESH_RATE_HZ BOARD_ENCODER_REFRESH_RATE_HZ
#endif
#define UMI_ENCODER_SAMPLE_INTERVAL_US (1000000 / UMI_ENCODER_REFRESH_RATE_HZ)

/* ------------------------------------------------------------------ */
/*  Inter-core frame queue (Core 1 → Core 0)                           */
/* ------------------------------------------------------------------ */

typedef struct {
    uint8_t data[UMI_FRAME_BUFFER_SIZE];
    size_t size;
} umi_encoded_frame_t;

/* Frame queue depth chosen so Core 0 can absorb several USB / tud_task
 * stalls without dropping frames. At BOARD_ENCODER_REFRESH_RATE_HZ = 120,
 * a depth of 32 buffers ~266 ms of samples — plenty for any USB hiccup,
 * yet small enough to keep RAM usage trivial (~32 * sizeof(frame) bytes).
 *
 * Frame timestamps are stamped at encode time on Core 1 and remain stable
 * while the frame sits in the queue, so the host sees the original
 * sample time rather than the dequeue time.
 *
 * The encoder diag log queue (Core 1 → Core 0 for human-readable error
 * surfaces) lives inside encoder_runtime.c — main.c only consumes it via
 * encoder_runtime_pop_diag_log_entry(). */
#define UMI_FRAME_QUEUE_DEPTH 32
static queue_t g_frame_queue;

/* Board-unique 64-bit ID, latched once on Core 0 before launching Core 1.
 * Read by Core 1 to stamp every UMI frame so the host can detect silent
 * device swaps mid-session. */
static pico_unique_board_id_t g_board_unique_id;

/* ------------------------------------------------------------------ */
/*  PDM microphone: IRQ-safe handoff to main loop                      */
/*                                                                     */
/*  The DMA callback runs in IRQ context.  TinyUSB FIFO operations    */
/*  (tud_audio_write_support_ff) are NOT safe to call from IRQ while  */
/*  tud_task() may be reconfiguring the same FIFO during              */
/*  SET_INTERFACE.  We therefore only latch the buffer pointer here    */
/*  and do the actual feed in the main loop.                           */
/* ------------------------------------------------------------------ */

#if BOARD_ENABLE_PDM_MIC
static volatile const int16_t *g_mic_pcm_buf;
static volatile size_t g_mic_sample_count;

static void pdm_dma_callback(const int16_t *pcm_buf, size_t sample_count) {
    g_mic_pcm_buf = pcm_buf;
    g_mic_sample_count = sample_count;
}
#endif /* BOARD_ENABLE_PDM_MIC */

/* ------------------------------------------------------------------ */
/*  Core 1 — encoder sampling + protocol encoding                      */
/*                                                                     */
/*  Important: this core must not call any TinyUSB function, including */
/*  tud_task(), tud_cdc_*(), cdc_uart_*(), or printf() (which routes   */
/*  through USB CDC stdio and calls tud_task() internally).            */
/* ------------------------------------------------------------------ */

static void core1_entry(void) {
#if !BOARD_ENABLE_ENCODER
    while (true) {
        sleep_ms(100);
    }
#else
    umi_payload_builder_t payload_builder;
    uint8_t payload_buffer[UMI_PAYLOAD_BUFFER_SIZE];
    uint8_t frame_buffer[UMI_FRAME_BUFFER_SIZE];

    tle5012b_dev_t encoder = {
        .spi_instance = BOARD_ENCODER_SPI_INSTANCE,
        .pin_sck      = BOARD_ENCODER_PIN_SCK,
        .pin_mosi     = BOARD_ENCODER_PIN_MOSI,
        .pin_miso     = BOARD_ENCODER_PIN_MISO,
        .pin_cs       = BOARD_ENCODER_PIN_CS,
        .spi_baudrate = BOARD_ENCODER_SPI_BAUDRATE,
        .initialized  = false,
    };

    encoder_runtime_t rt;
    encoder_runtime_init(&rt);

    tle5012b_error_t init_err = tle5012b_init(&encoder);
    if (init_err != TLE5012B_OK) {
        encoder_runtime_handle_init_failure(&rt, &encoder, init_err);
    } else {
        encoder_runtime_drain_pending_diag(&rt, &encoder);
    }

    absolute_time_t next_sample_time = get_absolute_time();

    while (true) {
        absolute_time_t current_time = get_absolute_time();
        int64_t time_diff_us = absolute_time_diff_us(current_time, next_sample_time);
        if (time_diff_us > UMI_ENCODER_SAMPLE_INTERVAL_US) {
            next_sample_time = current_time;
        }
        sleep_until(next_sample_time);
        next_sample_time = delayed_by_us(next_sample_time, UMI_ENCODER_SAMPLE_INTERVAL_US);

        /* Sample. Skip the SPI hit entirely if init failed — the runtime
         * will keep reporting INIT_FAILED via SENSOR_HEALTH. */
        tle5012b_angle_data_t angle_data = {0};
        tle5012b_error_t read_err = TLE5012B_ERR_NOT_INIT;
        if (!rt.init_failed) {
            read_err = tle5012b_read_angle(&encoder, &angle_data);
        }

        bool read_ok = encoder_runtime_observe_read(&rt, &encoder, read_err);
        float angle_to_send = encoder_runtime_pick_angle(&rt, read_ok, read_err == TLE5012B_OK,
                                                         angle_data.angle_deg);

        /* Build payload: angle TLV first, then SENSOR_HEALTH TLV for the
         * same sensor_id. The frame is atomic — if either segment doesn't
         * fit, the whole frame is dropped so the host never sees an angle
         * without a paired health report. */
        umi_payload_builder_init(&payload_builder, payload_buffer, sizeof(payload_buffer));

        if (!umi_payload_builder_add_segment(&payload_builder, UMI_SENSOR_TYPE_JOINT_ENCODER,
                                             UMI_ENCODER_SENSOR_ID, &angle_to_send,
                                             sizeof(angle_to_send))) {
            continue;
        }

        uint8_t health_buf[sizeof(umi_sensor_health_header_t) + sizeof(tle5012b_health_extra_t)];
        size_t health_n = encoder_runtime_pack_health_tlv_body(&rt, health_buf, sizeof(health_buf));
        if (health_n == 0 ||
            !umi_payload_builder_add_segment(&payload_builder, UMI_SENSOR_TYPE_SENSOR_HEALTH,
                                             UMI_ENCODER_SENSOR_ID, health_buf, health_n)) {
            continue;
        }

        umi_frame_meta_t meta = {
            .version      = UMI_PROTOCOL_VERSION,
            .source       = UMI_SOURCE_ID(UMI_SOURCE_TYPE_UMI_GRIPPER, UMI_SOURCE_INSTANCE),
            .timestamp_ms = to_ms_since_boot(get_absolute_time()),
        };
        memcpy(meta.device_id, g_board_unique_id.id, UMI_DEVICE_ID_BYTES);

        size_t frame_size = umi_frame_encode(&meta, &payload_builder, frame_buffer, sizeof(frame_buffer));
        if (!frame_size) {
            continue;
        }

        umi_encoded_frame_t frame;
        frame.size = frame_size;
        memcpy(frame.data, frame_buffer, frame_size);
        queue_try_add(&g_frame_queue, &frame);
    }
#endif
}

/* ------------------------------------------------------------------ */
/*  Board-level GPIO initialisation                                    */
/* ------------------------------------------------------------------ */

static void board_gpio_init(void) {
    /* LED — output, initially off */
    gpio_init(BOARD_LED_PIN);
    gpio_set_dir(BOARD_LED_PIN, GPIO_OUT);
    gpio_put(BOARD_LED_PIN, 0);

    /* iPhone DFP VBUS load switch — always on */
    gpio_init(BOARD_DFP_VBUS_EN_PIN);
    gpio_set_dir(BOARD_DFP_VBUS_EN_PIN, GPIO_OUT);
    gpio_put(BOARD_DFP_VBUS_EN_PIN, 1);

    /* LM73100 Power Good — digital inputs (open-drain, external pull-up) */
    gpio_init(BOARD_VBUS_PG_PIN);
    gpio_set_dir(BOARD_VBUS_PG_PIN, GPIO_IN);
    gpio_set_pulls(BOARD_VBUS_PG_PIN, true, false);

    gpio_init(BOARD_VAUX_PG_PIN);
    gpio_set_dir(BOARD_VAUX_PG_PIN, GPIO_IN);
    gpio_set_pulls(BOARD_VAUX_PG_PIN, true, false);

    /* ADC for IMON/VMON — GPIO26..29 are muxed to ADC0..3 */
    adc_init();
    adc_gpio_init(BOARD_VBUS_IMON_PIN);
    adc_gpio_init(BOARD_VBUS_VMON_PIN);
    adc_gpio_init(BOARD_VAUX_IMON_PIN);
    adc_gpio_init(BOARD_VAUX_VMON_PIN);
}

/* ------------------------------------------------------------------ */
/*  Power monitoring & DFP VBUS management (called from main loop)     */
/* ------------------------------------------------------------------ */

static absolute_time_t g_next_power_mon;

static void power_mon_task(void) {
    if (absolute_time_diff_us(g_next_power_mon, get_absolute_time()) < 0)
        return;
    g_next_power_mon = delayed_by_ms(get_absolute_time(), BOARD_POWER_MON_INTERVAL_MS);

    board_pg_status_t pg = board_pg_read();
    board_power_reading_t pwr = board_power_read();

    DEBUG_INFO(TAG, "PWR: VBUS=%.2fV/%.2fA PG=%d | AUX=%.2fV/%.2fA PG=%d",
               (double)pwr.vbus_v, (double)pwr.vbus_i, pg.vbus_pg,
               (double)pwr.vaux_v, (double)pwr.vaux_i, pg.vaux_pg);

    if (pwr.vbus_v < BOARD_POWER_FAULT_VBUS_THRESHOLD && !pg.vaux_pg)
        status_led_set_fault(STATUS_FAULT_POWER);
    else
        status_led_clear_fault(STATUS_FAULT_POWER);
}

/* ------------------------------------------------------------------ */
/*  Core 0 — USB servicing (TinyUSB, CDC, Audio) + PDM mic             */
/* ------------------------------------------------------------------ */

int main() {
    stdio_init_all();
    DEBUG_INFO(TAG, "UMI-Gripper boot");

    board_gpio_init();
    status_led_init();
    DEBUG_INFO(TAG, "Board GPIO initialized (LED on)");

    {
        board_pg_status_t pg = board_pg_read();
        DEBUG_INFO(TAG, "Power Good: VBUS=%d AUX=%d", pg.vbus_pg, pg.vaux_pg);
    }
    g_next_power_mon = delayed_by_ms(get_absolute_time(), BOARD_POWER_MON_INTERVAL_MS);

    queue_init(&g_frame_queue, sizeof(umi_encoded_frame_t), UMI_FRAME_QUEUE_DEPTH);
    encoder_runtime_subsystem_init();

    pico_get_unique_board_id(&g_board_unique_id);
    DEBUG_INFO(TAG, "Board ID: %02X%02X%02X%02X%02X%02X%02X%02X",
               g_board_unique_id.id[0], g_board_unique_id.id[1],
               g_board_unique_id.id[2], g_board_unique_id.id[3],
               g_board_unique_id.id[4], g_board_unique_id.id[5],
               g_board_unique_id.id[6], g_board_unique_id.id[7]);

    usb_audio_init();

#if BOARD_ENABLE_PDM_MIC
#if BOARD_PDM_MIC_TEST_MODE
    DEBUG_INFO(TAG, "Initializing PDM mic in TEST MODE (1 kHz sine, no hardware)...");
#else
    DEBUG_INFO(TAG, "Initializing PDM stereo DMIC (PIO + DMA + OpenPDMFilter)...");
#endif
    pdm_mic_init(pdm_dma_callback);
#if BOARD_PDM_MIC_TEST_MODE
    DEBUG_INFO(TAG, "PDM test tone active: 1 kHz stereo sine @ %d Hz", PDM_MIC_SAMPLE_RATE);
#else
    DEBUG_INFO(TAG, "PDM DMIC initialized (CLK=GPIO%d, DATL=GPIO%d, DATR=GPIO%d, %d Hz stereo)",
               BOARD_PDM_PIN_CLK, BOARD_PDM_PIN_DATL, BOARD_PDM_PIN_DATR, PDM_MIC_SAMPLE_RATE);
#endif
#else
    DEBUG_INFO(TAG, "PDM microphone DISABLED");
#endif /* BOARD_ENABLE_PDM_MIC */

    multicore_launch_core1(core1_entry);

#if !BOARD_ENABLE_ENCODER
    DEBUG_INFO(TAG, "Encoder SKIPPED (Core 1 idle)");
#else
    DEBUG_INFO(TAG, "Encoder running on Core 1 (%d Hz)", UMI_ENCODER_REFRESH_RATE_HZ);
#endif

    while (true) {
        tud_task();

#if BOARD_ENABLE_PDM_MIC
        /* Feed PDM-filtered PCM into TinyUSB — must happen in same
         * context as tud_task() to avoid FIFO race during SET_INTERFACE. */
        const int16_t *pcm = (const int16_t *)g_mic_pcm_buf;
        if (pcm) {
            size_t sc = g_mic_sample_count;
            g_mic_pcm_buf = NULL;
            usb_audio_feed(pcm, sc);
        }
#endif

        umi_encoded_frame_t frame;
        if (queue_try_remove(&g_frame_queue, &frame)) {
            cdc_uart_write(frame.data, frame.size);
        }

        encoder_runtime_drain_diag_log_to_stdout();

        power_mon_task();

        if (tud_mounted())
            status_led_clear_fault(STATUS_FAULT_USB);
        else
            status_led_set_fault(STATUS_FAULT_USB);

        status_led_task();
    }
}

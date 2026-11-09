/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file board.h
 * @brief Centralised board-level hardware configuration
 *
 * All GPIO pin assignments, peripheral instance selections, clock rates,
 * and USB device identity live here so that porting to a different PCB
 * only requires editing this single file.
 *
 * GPIO mapping follows the OAK-SoM baseboard schematic.
 */

#ifndef BOARD_H
#define BOARD_H

/* ================================================================== */
/*  Feature Toggles (set to 0 to disable at compile time)             */
/* ================================================================== */

#ifndef BOARD_ENABLE_ENCODER
#define BOARD_ENABLE_ENCODER      1
#endif
#ifndef BOARD_ENABLE_PDM_MIC
#define BOARD_ENABLE_PDM_MIC      1
#endif
#ifndef BOARD_PDM_MIC_TEST_MODE
#define BOARD_PDM_MIC_TEST_MODE   0   /* 1 = 1 kHz sine test tone */
#endif


/* ================================================================== */
/*  TLE5012B Magnetic Encoder (SPI0)                                  */
/*  GPIO0/1 = UART0, GPIO2-5 = SPI encoder                           */
/* ================================================================== */

#define BOARD_ENCODER_SPI_INSTANCE    0
#define BOARD_ENCODER_PIN_SCK         2   /* GPIO2  ENC_SCK  */
#define BOARD_ENCODER_PIN_MOSI        3   /* GPIO3  SPI0_MOSI */
#define BOARD_ENCODER_PIN_MISO        4   /* GPIO4  ENC_DATA */
#define BOARD_ENCODER_PIN_CS          5   /* GPIO5  ENC_CSQ  */
#define BOARD_ENCODER_SPI_BAUDRATE    1000000
#define BOARD_ENCODER_REFRESH_RATE_HZ 120

/* ================================================================== */
/*  Extension IO (active-low accent to J10/J11 header)                */
/* ================================================================== */

#define BOARD_EXT_IO0_PIN             8   /* GPIO8  */
#define BOARD_EXT_IO1_PIN             9   /* GPIO9  */
#define BOARD_EXT_I2C_SDA_PIN         10  /* GPIO10 */
#define BOARD_EXT_I2C_SCL_PIN         11  /* GPIO11 */

/* ================================================================== */
/*  HD3SS3220 I2C (USB Type-C SS Mux control)                        */
/* ================================================================== */

#define BOARD_SS_MUX_I2C_INSTANCE     0
#define BOARD_SS_MUX_I2C_SDA_PIN      12  /* GPIO12 */
#define BOARD_SS_MUX_I2C_SCL_PIN      13  /* GPIO13 */

/* ================================================================== */
/*  iPhone DFP VBUS Load Switch (SY6280AAC)                          */
/* ================================================================== */

#define BOARD_DFP_VBUS_EN_PIN         14  /* GPIO14  active-high enable */

/* ================================================================== */
/*  PDM Stereo DMIC (PIO + DMA)                                      */
/*  Two LMD3526T261 microphones in stereo configuration               */
/* ================================================================== */

#define BOARD_PDM_PIO                 pio0
#define BOARD_PDM_SM                  0
#define BOARD_PDM_PIN_CLK             16  /* GPIO16  MK1 + MK2 shared clock */
#define BOARD_PDM_PIN_DATR            17  /* GPIO17  right channel (MK2)    */
#define BOARD_PDM_PIN_DATL            18  /* GPIO18  left channel  (MK1)    */
#define BOARD_PDM_SAMPLE_RATE         48000

/* OpenPDMFilter gain parameters:
 *   MaxVolume — full-scale reference (filter internal normalization)
 *   Gain      — software gain applied during decimation
 *   Volume    — runtime volume, 0..MaxVolume (MaxVolume = 0dB)
 *
 * Effective output ∝ Volume * Gain.
 *
 *   Gain |  Relative  | Use case
 *   -----+------------+----------------------------------
 *     1  |    0 dB    | Extreme near-field / high SPL
 *     4  |  +12 dB    | Quiet close-range
 *     8  |  +18 dB    | General purpose (recommended)
 *    16  |  +24 dB    | Far-field / low-sensitivity mic   */
#define BOARD_PDM_FILTER_MAX_VOLUME   64
#define BOARD_PDM_FILTER_GAIN         8    /* +18dB, general purpose */
#define BOARD_PDM_FILTER_VOLUME       64   /* = MaxVolume → 0dB atten */

/* ================================================================== */
/*  MCU LED (D5 green)                                                */
/* ================================================================== */

#define BOARD_LED_PIN                 19  /* GPIO19 */

/* ================================================================== */
/*  Power Monitoring — Thresholds & Intervals                         */
/* ================================================================== */

#define BOARD_POWER_MON_INTERVAL_MS       2000   /* ADC + PG poll period (ms) */
#define BOARD_POWER_FAULT_VBUS_THRESHOLD  4.5f   /* VBUS under-voltage (V)    */
#define BOARD_ENCODER_FAULT_STREAK_MAX    10     /* consecutive read failures  */

/* ================================================================== */
/*  Power Monitoring — Digital (LM73100 Power Good, open-drain)       */
/* ================================================================== */

#define BOARD_VBUS_PG_PIN             24  /* GPIO24  LM73100 #1 PG */
#define BOARD_VAUX_PG_PIN             25  /* GPIO25  LM73100 #2 PG */

/* ================================================================== */
/*  Power Monitoring — Analog (LM73100 IMON / VMON via ADC)           */
/*                                                                    */
/*  VMON channels use 4.7k/4.7k resistor divider:                    */
/*    V_actual = ADC_voltage * (4.7k + 4.7k) / 4.7k = ADC_voltage * 2*/
/* ================================================================== */

#define BOARD_VBUS_IMON_PIN           26  /* GPIO26 / ADC0 */
#define BOARD_VBUS_VMON_PIN           27  /* GPIO27 / ADC1 */
#define BOARD_VAUX_IMON_PIN           28  /* GPIO28 / ADC2 */
#define BOARD_VAUX_VMON_PIN           29  /* GPIO29 / ADC3 */

#define BOARD_VBUS_IMON_ADC_CH        0
#define BOARD_VBUS_VMON_ADC_CH        1
#define BOARD_VAUX_IMON_ADC_CH        2
#define BOARD_VAUX_VMON_ADC_CH        3

/* VMON resistor divider: R_high = R_low = 4.7kΩ */
#define BOARD_VMON_R_HIGH_OHM         4700
#define BOARD_VMON_R_LOW_OHM          4700
#define BOARD_VMON_DIVIDER_RATIO      ((float)(BOARD_VMON_R_HIGH_OHM + BOARD_VMON_R_LOW_OHM) / BOARD_VMON_R_LOW_OHM)

/* LM73100 IMON (datasheet §7.3.5):
 *   I_OUT = V_IMON * 1e6 / (R_IMON * G_IMON)
 * where G_IMON = 181 μA/A (typ), R_IMON = external resistor to GND.  */
#define BOARD_IMON_G_UA_PER_A         181    /* μA/A, typ from datasheet */
#define BOARD_IMON_R_OHM              1000   /* 1kΩ external resistor */
#define BOARD_IMON_SCALE              (1e6f / ((float)BOARD_IMON_R_OHM * (float)BOARD_IMON_G_UA_PER_A))

/* RP2040 ADC: 12-bit, 3.3V reference */
#define BOARD_ADC_VREF                3.3f
#define BOARD_ADC_MAX                 4095

/* ================================================================== */
/*  USB Device Identity                                               */
/* ================================================================== */

#define BOARD_USB_VID                 0x2E8A
#define BOARD_USB_PID_RP2040          0x000a
#define BOARD_USB_PID_DEFAULT         0x0009
#define BOARD_USB_MANUFACTURER        "Raspberry Pi"
#define BOARD_USB_PRODUCT             "Pico"
#define BOARD_USB_MAX_POWER_MA        500

/* ================================================================== */
/*  USB Endpoint Assignments                                          */
/* ================================================================== */

#define BOARD_CDC_0_EP_CMD            0x81
#define BOARD_CDC_0_EP_OUT            0x01
#define BOARD_CDC_0_EP_IN             0x82
#define BOARD_CDC_1_EP_CMD            0x83
#define BOARD_CDC_1_EP_OUT            0x03
#define BOARD_CDC_1_EP_IN             0x84
#define BOARD_AUDIO_EP_IN             0x85

/* ================================================================== */
/*  CDC UART                                                          */
/* ================================================================== */

#define BOARD_CDC_UART_ITF            1

#endif /* BOARD_H */

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file tle5012b.c
 * @brief TLE5012B Magnetic Angle Sensor Driver Implementation (3-wire SPI)
 */

#include "tle5012b.h"
#include "debug.h"
#include "hardware/gpio.h"
#include "hardware/spi.h"
#include "pico/stdlib.h"
#include <math.h>
#include <string.h>

/* Debug tag for this module */
#define TAG "TLE5012B"

/* Internal helper functions */
static uint8_t tle5012b_calculate_crc_buffer(uint8_t *data, uint8_t length);
static tle5012b_error_t tle5012b_transfer(tle5012b_dev_t *dev, uint16_t cmd, uint16_t *received_data,
                                          uint16_t num_data_words, bool ignore_safety);
static void tle5012b_store_diag_event(tle5012b_dev_t *dev, tle5012b_error_t error, uint16_t cmd,
                                      uint16_t safety_word, bool capture_status);

/**
 * @brief Calculate CRC-8 for TLE5012B communication
 *
 * The TLE5012B uses CRC-8 with polynomial: x^8 + x^4 + x^3 + x^2 + 1 (0x1D)
 * Seed: 0xFF
 *
 * @param data Pointer to data buffer
 * @param length Length of data in bytes
 * @return 8-bit CRC value
 */
static uint8_t tle5012b_calculate_crc_buffer(uint8_t *data, uint8_t length) {
    uint32_t crc = 0xFF; // CRC seed

    for (uint8_t i = 0; i < length; i++) {
        crc ^= data[i];
        for (uint8_t bit = 0; bit < 8; bit++) {
            if ((crc & 0x80) != 0) {
                crc <<= 1;
                crc ^= 0x1D; // CRC polynomial
            } else {
                crc <<= 1;
            }
        }
    }

    return (~crc) & 0xFF; // Invert and return
}

static void tle5012b_store_diag_event(tle5012b_dev_t *dev, tle5012b_error_t error, uint16_t cmd,
                                      uint16_t safety_word, bool capture_status) {
    if (!dev) {
        return;
    }

    dev->pending_diag.error = error;
    dev->pending_diag.command = cmd;
    dev->pending_diag.safety_word = safety_word;
    dev->pending_diag.status_word = 0;
    dev->pending_diag.has_status = false;

    if (capture_status) {
        uint16_t status;
        if (tle5012b_read_status(dev, &status) == TLE5012B_OK) {
            dev->pending_diag.status_word = status;
            dev->pending_diag.has_status = true;
        }
    }

    dev->pending_diag_valid = true;
}

/**
 * @brief Perform SPI transfer with TLE5012B (3-wire mode)
 *
 * 3-wire SPI protocol with hardware implementation:
 * - MOSI and MISO are tied together in hardware to encoder DATA pin
 * - For READ: Send command, receive data word(s) + safety word
 * - For WRITE: Send command + data word(s), receive safety word
 * - Number of data words is encoded in command[3:0]
 *
 * @param dev Pointer to device structure
 * @param cmd Command word (includes register address and R/W bit)
 * @param data Pointer to data words (for write: send data, for read: receive data)
 * @param num_data_words Number of 16-bit data words to send/receive (excluding safety word)
 * @param ignore_safety If true, skip safety word error checking
 * @return TLE5012B_OK on success, error code otherwise
 */
static tle5012b_error_t tle5012b_transfer(tle5012b_dev_t *dev, uint16_t cmd, uint16_t *data, uint16_t num_data_words,
                                          bool ignore_safety) {
    if (!dev || !dev->initialized) {
        return TLE5012B_ERR_NOT_INIT;
    }

    // Skip safety word checking if ignore_safety is true
    bool check_safety = !ignore_safety;

    spi_inst_t *spi = (spi_inst_t *)dev->spi_handle;
    uint8_t tx_buf[66]; // Buffer for command + data words (max: 1 cmd + 32 data words)
    uint8_t rx_buf[64]; // Buffer for received data (max: multiple data words + safety word)

    bool is_write = (cmd & TLE5012B_CMD_WRITE) == TLE5012B_CMD_WRITE;

    // Prepare command (MSB first)
    tx_buf[0] = (cmd >> 8) & 0xFF;
    tx_buf[1] = cmd & 0xFF;
    uint16_t tx_bytes = 2;

    // For write operations, append data words to transmit buffer
    if (is_write && data) {
        for (uint16_t i = 0; i < num_data_words; i++) {
            tx_buf[tx_bytes++] = (data[i] >> 8) & 0xFF;
            tx_buf[tx_bytes++] = data[i] & 0xFF;
        }
        DEBUG_VERBOSE(TAG, "Write CMD: 0x%04X, %d data words", cmd, num_data_words);
    } else {
        DEBUG_VERBOSE(TAG, "Read CMD: 0x%04X, expecting %d data words", cmd, num_data_words);
    }

    // Assert CS (active low)
    gpio_put(dev->pin_cs, 0);
    sleep_us(1); // CS setup time

    // Phase 1: Send command (and data for write) with MOSI as output
    spi_write_blocking(spi, tx_buf, tx_bytes);

    // Phase 2: Switch MOSI to high-impedance to avoid bus conflict
    // MOSI and MISO are tied together, so MOSI must be high-Z when receiving
    gpio_set_function(dev->pin_mosi, GPIO_FUNC_SIO); // Switch to software I/O
    gpio_set_dir(dev->pin_mosi, GPIO_IN);            // Set as input (high-Z)
    gpio_disable_pulls(dev->pin_mosi);               // Disable pull-up/down

    sleep_us(5); // Small delay for mode switch

    // For read: receive data words + safety word
    // For write: receive only safety word
    uint16_t rx_words = is_write ? 1 : (num_data_words + 1);
    uint16_t rx_bytes = rx_words * 2;

    memset(rx_buf, 0, sizeof(rx_buf));
    spi_read_blocking(spi, 0x00, rx_buf, rx_bytes);

    // Restore MOSI to SPI function for next transfer
    gpio_set_function(dev->pin_mosi, GPIO_FUNC_SPI);

    // Deassert CS
    sleep_us(1); // CS hold time
    gpio_put(dev->pin_cs, 1);
    sleep_us(10); // Inter-frame delay

    // Extract and verify response
    if (is_write) {
        // For write: only safety word received
        uint16_t safety_word = ((uint16_t)rx_buf[0] << 8) | rx_buf[1];
        uint8_t received_crc = rx_buf[1];

        // CRC calculated over command + data words sent
        uint8_t calculated_crc = tle5012b_calculate_crc_buffer(tx_buf, tx_bytes);

        DEBUG_VERBOSE(TAG, "Safety: 0x%04X, CRC recv: 0x%02X, calc: 0x%02X", safety_word, received_crc, calculated_crc);

        if (check_safety) {
            // Verify CRC
            if (received_crc != calculated_crc) {
                tle5012b_store_diag_event(dev, TLE5012B_ERR_CRC, cmd, safety_word, false);
                DEBUG_ERROR(TAG, "Write CRC mismatch! recv=0x%02X calc=0x%02X", received_crc, calculated_crc);
                tle5012b_reset_safety(dev);
                return TLE5012B_ERR_CRC;
            }

            // Check safety word error bits
            if (!(safety_word & TLE5012B_SAFETY_STAT_OK)) {
                tle5012b_store_diag_event(dev, TLE5012B_ERR_SYSTEM, cmd, safety_word, true);
                DEBUG_ERROR(TAG, "Write failed: System error");
                tle5012b_reset_safety(dev);
                return TLE5012B_ERR_SYSTEM;
            }
            if (!(safety_word & TLE5012B_SAFETY_INTF_OK)) {
                tle5012b_store_diag_event(dev, TLE5012B_ERR_INTERFACE, cmd, safety_word, true);
                DEBUG_ERROR(TAG, "Write failed: Interface error");
                tle5012b_reset_safety(dev);
                return TLE5012B_ERR_INTERFACE;
            }
        }
    } else if (data) {
        // For read: extract data words from response
        for (uint16_t i = 0; i < num_data_words; i++) {
            data[i] = ((uint16_t)rx_buf[i * 2] << 8) | rx_buf[i * 2 + 1];
            DEBUG_VERBOSE(TAG, "Data[%d]: 0x%04X", i, data[i]);
        }

        // Extract safety word (last word)
        uint16_t safety_word = ((uint16_t)rx_buf[rx_bytes - 2] << 8) | rx_buf[rx_bytes - 1];
        uint8_t received_crc = rx_buf[rx_bytes - 1];

        // Prepare CRC calculation buffer: command + all received data words
        uint8_t crc_buf[66];
        crc_buf[0] = tx_buf[0];
        crc_buf[1] = tx_buf[1];
        for (uint16_t i = 0; i < num_data_words; i++) {
            crc_buf[2 + i * 2] = (data[i] >> 8) & 0xFF;
            crc_buf[2 + i * 2 + 1] = data[i] & 0xFF;
        }

        uint8_t calculated_crc = tle5012b_calculate_crc_buffer(crc_buf, 2 + num_data_words * 2);

        DEBUG_VERBOSE(TAG, "Safety: 0x%04X, CRC recv: 0x%02X, calc: 0x%02X", safety_word, received_crc, calculated_crc);

        if (check_safety) {
            // Verify CRC
            if (received_crc != calculated_crc) {
                tle5012b_store_diag_event(dev, TLE5012B_ERR_CRC, cmd, safety_word, false);
                DEBUG_ERROR(TAG, "Read CRC mismatch! recv=0x%02X calc=0x%02X", received_crc, calculated_crc);
                tle5012b_reset_safety(dev);
                return TLE5012B_ERR_CRC;
            }

            // Check safety word error bits
            if (!(safety_word & TLE5012B_SAFETY_STAT_OK)) {
                /* Deliberately return OK so the encoder loop does not stall on a
                 * latched system-error bit, but still surface it through the diag
                 * event channel (Core 0 will print it once per status change) and
                 * clear the latch so the next frame can succeed. */
                tle5012b_store_diag_event(dev, TLE5012B_ERR_SYSTEM, cmd, safety_word, true);
                tle5012b_reset_safety(dev);
                return TLE5012B_OK;
            }
            if (!(safety_word & TLE5012B_SAFETY_INTF_OK)) {
                tle5012b_store_diag_event(dev, TLE5012B_ERR_INTERFACE, cmd, safety_word, true);
                DEBUG_ERROR(TAG, "Interface access error");
                tle5012b_reset_safety(dev);
                return TLE5012B_ERR_INTERFACE;
            }
            if (!(safety_word & TLE5012B_SAFETY_INV_OK)) {
                tle5012b_store_diag_event(dev, TLE5012B_ERR_INVALID_ANGLE, cmd, safety_word, true);
                DEBUG_ERROR(TAG, "Invalid angle error");
                tle5012b_reset_safety(dev);
                return TLE5012B_ERR_INVALID_ANGLE;
            }
        }
    }

    return TLE5012B_OK;
}

/**
 * @brief Initialize TLE5012B sensor
 */
tle5012b_error_t tle5012b_init(tle5012b_dev_t *dev) {
    if (!dev) {
        return TLE5012B_ERR_INVALID_PARAM;
    }

    DEBUG_INFO(TAG, "Initializing TLE5012B...");
    DEBUG_LOG(TAG, "SPI%d, SCK=%d, MOSI=%d, MISO=%d, CS=%d, Baudrate=%d", dev->spi_instance, dev->pin_sck,
              dev->pin_mosi, dev->pin_miso, dev->pin_cs, dev->spi_baudrate);

    // Select SPI instance
    spi_inst_t *spi;
    if (dev->spi_instance == 0) {
        spi = spi0;
    } else if (dev->spi_instance == 1) {
        spi = spi1;
    } else {
        return TLE5012B_ERR_INVALID_PARAM;
    }

    dev->spi_handle = spi;

    // Validate baudrate
    if (dev->spi_baudrate == 0 || dev->spi_baudrate > TLE5012B_SPI_MAX_SPEED) {
        dev->spi_baudrate = 1000000; // Default to 1 MHz
    }

    // Initialize SPI
    spi_init(spi, dev->spi_baudrate);

    // Configure SPI format: 8-bit data, CPOL=0, CPHA=1 (SPI Mode 2)
    // Clock idle low, data sampled on falling edge
    spi_set_format(spi, 8, SPI_CPOL_0, SPI_CPHA_1, SPI_MSB_FIRST);

    // Initialize GPIO pins
    gpio_set_function(dev->pin_sck, GPIO_FUNC_SPI);
    gpio_set_function(dev->pin_mosi, GPIO_FUNC_SPI);
    gpio_set_function(dev->pin_miso, GPIO_FUNC_SPI);

    // CS pin as output, initially high
    gpio_init(dev->pin_cs);
    gpio_set_dir(dev->pin_cs, GPIO_OUT);
    gpio_put(dev->pin_cs, 1);

    dev->initialized = true;

    DEBUG_INFO(TAG, "SPI initialization successful");

    tle5012b_error_t err;

    // Reset sensor
    err = tle5012b_reset(dev);
    if (err != TLE5012B_OK) {
        DEBUG_ERROR(TAG, "Sensor reset failed: %d", err);
        return err;
    }

    // Reset safety
    err = tle5012b_reset_safety(dev);
    if (err != TLE5012B_OK) {
        DEBUG_WARN(TAG, "Safety reset failed: %d", err);
    }

    /* Self-heal a stale parameter CRC.
     *
     * If anything (older firmware, for example) wrote a CRC-protected
     * register (MOD_2/MOD_3/OFFX/OFFY/SYNCH/IFAB/MOD_4 at 0x08..0x0E)
     * without rewriting CRCPAR in TCO_Y (0x0F), the sensor boots with
     * STAT.SFUSE = 1 and the safety word reports SYSTEM_ERROR on every
     * frame. Disabling the Fuse CRC check (ACSTAT.ASFUSE = 0) would hide
     * that, but it silences an entire IEC 60730 Class B safety mechanism.
     * Instead, do a one-shot CRC rebuild: read the 8-register CRC block,
     * recompute CRCPAR, and write it back. After this, the fuse CRC check
     * stays armed and will catch real configuration corruption. */
    uint16_t status_check;
    err = tle5012b_read_status(dev, &status_check);
    if (err == TLE5012B_OK && (status_check & TLE5012B_STAT_SFUSE)) {
        DEBUG_WARN(TAG, "Fuse CRC mismatch on boot, rebuilding parameter CRC");
        tle5012b_error_t crc_err = tle5012b_regular_crc_update(dev);
        if (crc_err != TLE5012B_OK) {
            DEBUG_WARN(TAG, "Parameter CRC rebuild failed: %d", crc_err);
        } else {
            DEBUG_LOG(TAG, "Parameter CRC rebuilt successfully");
            tle5012b_reset_safety(dev);
        }
    }

    // Start sensor and configure
    // Write slave number (default to 0)
    err = tle5012b_write_slave_number(dev, 0);
    if (err != TLE5012B_OK) {
        DEBUG_WARN(TAG, "Failed to write slave number: %d", err);
    }

    DEBUG_LOG(TAG, "Initialization complete");
    return TLE5012B_OK;
}

/**
 * @brief Deinitialize TLE5012B sensor
 */
void tle5012b_deinit(tle5012b_dev_t *dev) {
    if (!dev || !dev->initialized) {
        return;
    }

    spi_inst_t *spi = (spi_inst_t *)dev->spi_handle;

    // Deinitialize SPI
    spi_deinit(spi);

    // Reset GPIO pins to input
    gpio_deinit(dev->pin_sck);
    gpio_deinit(dev->pin_mosi);
    gpio_deinit(dev->pin_miso);
    gpio_deinit(dev->pin_cs);

    dev->initialized = false;
    dev->spi_handle = NULL;
}

bool tle5012b_take_diag_event(tle5012b_dev_t *dev, tle5012b_diag_event_t *event) {
    if (!dev || !event || !dev->pending_diag_valid) {
        return false;
    }

    *event = dev->pending_diag;
    dev->pending_diag_valid = false;
    return true;
}

/**
 * @brief Trigger update to create a snapshot of UPD registers
 */
tle5012b_error_t tle5012b_trigger_update(tle5012b_dev_t *dev) {
    if (!dev || !dev->initialized) {
        return TLE5012B_ERR_NOT_INIT;
    }

    DEBUG_VERBOSE(TAG, "Triggering register update snapshot");

    // Set SCK low
    gpio_set_function(dev->pin_sck, GPIO_FUNC_SIO);
    gpio_set_dir(dev->pin_sck, GPIO_OUT);
    gpio_put(dev->pin_sck, 0);

    // Set MOSI high
    gpio_set_function(dev->pin_mosi, GPIO_FUNC_SIO);
    gpio_set_dir(dev->pin_mosi, GPIO_OUT);
    gpio_put(dev->pin_mosi, 1);

    // CS pulse: LOW -> HIGH
    gpio_put(dev->pin_cs, 0);
    sleep_us(5); // Grace period for register snapshot
    gpio_put(dev->pin_cs, 1);

    // Restore pins to SPI function
    gpio_set_function(dev->pin_sck, GPIO_FUNC_SPI);
    gpio_set_function(dev->pin_mosi, GPIO_FUNC_SPI);

    return TLE5012B_OK;
}

/**
 * @brief Write to sensor with optional CRC update
 */
tle5012b_error_t tle5012b_write_to_sensor(tle5012b_dev_t *dev, uint16_t reg_addr, uint16_t data, bool update_crc) {
    if (!dev) {
        return TLE5012B_ERR_INVALID_PARAM;
    }

    // Write data to register
    tle5012b_error_t err = tle5012b_write_register(dev, reg_addr, data, false);
    if (err != TLE5012B_OK) {
        return err;
    }

    // If write affects CRC-protected registers, update CRC
    if (update_crc) {
        err = tle5012b_regular_crc_update(dev);
        if (err != TLE5012B_OK) {
            DEBUG_WARN(TAG, "CRC update failed after write: %d", err);
            return err;
        }
    }

    return TLE5012B_OK;
}

/**
 * @brief Write temperature coefficient with update
 */
tle5012b_error_t tle5012b_write_temp_coeff_update(tle5012b_dev_t *dev, uint16_t data) {
    if (!dev) {
        return TLE5012B_ERR_INVALID_PARAM;
    }

    // Trigger update before write
    tle5012b_error_t err = tle5012b_trigger_update(dev);
    if (err != TLE5012B_OK) {
        return err;
    }

    // Write to TCO_Y register
    err = tle5012b_write_register(dev, TLE5012B_REG_TCO_Y, data, false);
    if (err != TLE5012B_OK) {
        return err;
    }

    // Read status register to check FUSE bit
    uint16_t status;
    err = tle5012b_read_status(dev, &status);
    if (err != TLE5012B_OK) {
        return err;
    }

    // If FUSE error bit (bit 3) is set, perform CRC update
    if (status & TLE5012B_STAT_SFUSE) {
        DEBUG_VERBOSE(TAG, "FUSE bit set, performing CRC update");
        err = tle5012b_regular_crc_update(dev);
        if (err != TLE5012B_OK) {
            DEBUG_WARN(TAG, "CRC update failed: %d", err);
            return err;
        }
    }

    return TLE5012B_OK;
}

/**
 * @brief Write slave number to STAT[14:13]
 */
tle5012b_error_t tle5012b_write_slave_number(tle5012b_dev_t *dev, uint8_t slave_num) {
    if (!dev) {
        return TLE5012B_ERR_INVALID_PARAM;
    }

    if (slave_num > 3) {
        DEBUG_ERROR(TAG, "Invalid slave number: %d (must be 0-3)", slave_num);
        return TLE5012B_ERR_INVALID_PARAM;
    }

    DEBUG_VERBOSE(TAG, "Writing slave number: %d", slave_num);

    /* The Slave Number (SNR) field lives in STAT bits 14:13, not in MOD_2.
     * STAT is outside the CRC-protected range (0x08..0x0F), so no parameter
     * CRC update is required after this write. The other STAT bits are
     * read-only and silently ignore the write, matching the Infineon
     * reference driver's writeSlaveNumber() implementation. */
    uint16_t snr_val = ((uint16_t)slave_num << 13) & TLE5012B_STAT_SNR_MASK;
    return tle5012b_write_to_sensor(dev, TLE5012B_REG_STAT, snr_val, false);
}

/**
 * @brief Read a register from TLE5012B
 */
tle5012b_error_t tle5012b_read_register(tle5012b_dev_t *dev, uint16_t reg_addr, uint16_t *data, bool ignore_safety) {
    if (!dev || !data) {
        return TLE5012B_ERR_INVALID_PARAM;
    }

    tle5012b_error_t err;
    DEBUG_VERBOSE(TAG, "Reading register 0x%04X", reg_addr);

    // Construct read command: READ bit | register address | 1 word count
    uint16_t cmd = TLE5012B_CMD_READ | (reg_addr & 0x3FF0) | 0x0001;

    err = tle5012b_transfer(dev, cmd, data, 1, ignore_safety);
    DEBUG_LOG(TAG, "Read register 0x%04X: 0x%04X", reg_addr, *data);

    return err;
}

/**
 * @brief Write a register to TLE5012B
 */
tle5012b_error_t tle5012b_write_register(tle5012b_dev_t *dev, uint16_t reg_addr, uint16_t data, bool ignore_safety) {
    if (!dev) {
        return TLE5012B_ERR_INVALID_PARAM;
    }

    // Construct write command: WRITE bit | register address | 1 word count
    uint16_t cmd = TLE5012B_CMD_WRITE | (reg_addr & 0x3FF0) | 0x0001;

    DEBUG_LOG(TAG, "Writing 0x%04X to register 0x%04X, cmd=0x%04X", data, reg_addr, cmd);

    return tle5012b_transfer(dev, cmd, &data, 1, ignore_safety);
}

/**
 * @brief Read angle value from TLE5012B
 */
tle5012b_error_t tle5012b_read_angle(tle5012b_dev_t *dev, tle5012b_angle_data_t *angle_data) {
    if (!dev || !angle_data) {
        return TLE5012B_ERR_INVALID_PARAM;
    }

    tle5012b_error_t err;
    uint16_t raw_value;

    // Read raw angle
    err = tle5012b_read_register(dev, TLE5012B_REG_AVAL, &raw_value, false);
    if (err != TLE5012B_OK) {
        return err;
    }

    // Extract 15-bit angle (bit 14-0)
    angle_data->angle_raw = (int16_t)(raw_value & TLE5012B_MASK_DELETE_BIT15);

    // Convert to degrees and radians
    angle_data->angle_deg = tle5012b_raw_to_degrees(angle_data->angle_raw);
    angle_data->angle_rad = tle5012b_raw_to_radians(angle_data->angle_raw);

    DEBUG_VERBOSE(TAG, "Angle: raw=%d, deg=%.2f, rad=%.4f", angle_data->angle_raw, angle_data->angle_deg,
                  angle_data->angle_rad);

    // Read speed
    err = tle5012b_read_register(dev, TLE5012B_REG_ASPD, &raw_value, false);
    if (err != TLE5012B_OK) {
        return err;
    }
    angle_data->speed = (int16_t)raw_value;

    // Read revolution counter
    err = tle5012b_read_register(dev, TLE5012B_REG_AREV, &raw_value, false);
    if (err != TLE5012B_OK) {
        return err;
    }
    angle_data->revolution = raw_value;

    return TLE5012B_OK;
}

/**
 * @brief Read raw angle value (15-bit)
 */
tle5012b_error_t tle5012b_read_angle_raw(tle5012b_dev_t *dev, int16_t *angle) {
    if (!dev || !angle) {
        return TLE5012B_ERR_INVALID_PARAM;
    }

    uint16_t raw_value;
    tle5012b_error_t err = tle5012b_read_register(dev, TLE5012B_REG_AVAL, &raw_value, false);
    if (err != TLE5012B_OK) {
        return err;
    }

    // Extract 15-bit angle
    *angle = (int16_t)(raw_value & TLE5012B_MASK_DELETE_BIT15);

    return TLE5012B_OK;
}

/**
 * @brief Read angle speed
 */
tle5012b_error_t tle5012b_read_speed(tle5012b_dev_t *dev, int16_t *speed) {
    if (!dev || !speed) {
        return TLE5012B_ERR_INVALID_PARAM;
    }

    uint16_t raw_value;
    tle5012b_error_t err = tle5012b_read_register(dev, TLE5012B_REG_ASPD, &raw_value, false);
    if (err != TLE5012B_OK) {
        return err;
    }

    *speed = (int16_t)raw_value;

    return TLE5012B_OK;
}

/**
 * @brief Read revolution counter
 */
tle5012b_error_t tle5012b_read_revolution(tle5012b_dev_t *dev, uint16_t *revolution) {
    if (!dev || !revolution) {
        return TLE5012B_ERR_INVALID_PARAM;
    }

    return tle5012b_read_register(dev, TLE5012B_REG_AREV, revolution, false);
}

/**
 * @brief Read status register
 */
tle5012b_error_t tle5012b_read_status(tle5012b_dev_t *dev, uint16_t *status) {
    if (!dev || !status) {
        return TLE5012B_ERR_INVALID_PARAM;
    }

    return tle5012b_read_register(dev, TLE5012B_REG_STAT, status, true);
}

/**
 * @brief Check if sensor has errors
 */
tle5012b_error_t tle5012b_check_error(tle5012b_dev_t *dev, bool *has_error) {
    if (!dev || !has_error) {
        return TLE5012B_ERR_INVALID_PARAM;
    }

    uint16_t status;
    tle5012b_error_t err = tle5012b_read_status(dev, &status);
    if (err != TLE5012B_OK) {
        return err;
    }

    // Check for any error bits (bits 0-10, excluding bit 8 which is reserved)
    *has_error = (status & 0x07FF) != 0;

    // Print specific errors if any
    if (*has_error) {
        if (status & TLE5012B_STAT_SRST) {
            DEBUG_ERROR(TAG, "System reset error");
        }
        if (status & TLE5012B_STAT_SWD) {
            DEBUG_ERROR(TAG, "System watchdog error");
        }
        if (status & TLE5012B_STAT_SVR) {
            DEBUG_ERROR(TAG, "Supply voltage regulator error");
        }
        if (status & TLE5012B_STAT_SFUSE) {
            DEBUG_ERROR(TAG, "Fuse CRC error");
        }
        if (status & TLE5012B_STAT_SDSPU) {
            DEBUG_ERROR(TAG, "DSPU error");
        }
        if (status & TLE5012B_STAT_SOV) {
            DEBUG_ERROR(TAG, "Overflow error");
        }
        if (status & TLE5012B_STAT_SXYOL) {
            DEBUG_ERROR(TAG, "X/Y data out of limit error");
        }
        if (status & TLE5012B_STAT_SMAGOL) {
            DEBUG_ERROR(TAG, "Magnitude out of limit error");
        }
        if (status & TLE5012B_STAT_SADCT) {
            DEBUG_ERROR(TAG, "ADC test error");
        }
        if (status & TLE5012B_STAT_SROM) {
            DEBUG_ERROR(TAG, "ROM CRC error");
        }
        if (status & TLE5012B_STAT_NOGMRXY) {
            DEBUG_ERROR(TAG, "No valid GMR XY values");
        }
        if (status & TLE5012B_STAT_NOGMRA) {
            DEBUG_ERROR(TAG, "No valid GMR angle value");
        }
    }

    return TLE5012B_OK;
}

/**
 * @brief Software reset
 */
tle5012b_error_t tle5012b_reset(tle5012b_dev_t *dev) {
    if (!dev) {
        return TLE5012B_ERR_INVALID_PARAM;
    }

    // Write to ACSTAT register to trigger reset
    tle5012b_error_t err = tle5012b_write_register(dev, TLE5012B_REG_ACSTAT, 0x0001, true);
    if (err != TLE5012B_OK) {
        return err;
    }

    // Wait for reset to complete
    sleep_ms(10);

    return TLE5012B_OK;
}

/**
 * @brief Read CRC block registers
 */
tle5012b_error_t tle5012b_read_block_crc(tle5012b_dev_t *dev, uint16_t registers[TLE5012B_CRC_NUM_REGISTERS]) {
    if (!dev || !registers) {
        return TLE5012B_ERR_INVALID_PARAM;
    }

    // Use special READ_BLOCK_CRC command (reads 8 registers)
    uint16_t cmd = TLE5012B_CMD_READ_BLOCK_CRC;

    tle5012b_error_t err = tle5012b_transfer(dev, cmd, registers, TLE5012B_CRC_NUM_REGISTERS, false);
    if (err != TLE5012B_OK) {
        return err;
    }

    // Reset safety after CRC read
    tle5012b_reset_safety(dev);

    return TLE5012B_OK;
}

/**
 * @brief Reset safety word
 */
tle5012b_error_t tle5012b_reset_safety(tle5012b_dev_t *dev) {
    if (!dev) {
        return TLE5012B_ERR_INVALID_PARAM;
    }

    DEBUG_VERBOSE(TAG, "Resetting safety word");

    // Trigger update and perform a dummy read to clear safety errors
    tle5012b_trigger_update(dev);

    // Read 3 registers (STAT, ACSTAT, AVAL) to clear safety
    // Ignore safety word errors during this reset operation
    uint16_t dummy[3];
    uint16_t cmd = TLE5012B_CMD_READ | TLE5012B_REG_STAT | 0x0003;

    // Pass true to ignore_safety parameter
    tle5012b_transfer(dev, cmd, dummy, 3, true);

    return TLE5012B_OK;
}

/**
 * @brief Firmware reset
 */
tle5012b_error_t tle5012b_reset_firmware(tle5012b_dev_t *dev) {
    if (!dev) {
        return TLE5012B_ERR_INVALID_PARAM;
    }

    DEBUG_INFO(TAG, "Triggering firmware reset");

    // Write 0x401 to ACSTAT register to trigger firmware reset
    // Bit 10 (ASFRST) = 1, Bit 0 (ASRST) = 1
    return tle5012b_write_register(dev, TLE5012B_REG_ACSTAT, 0x0401, true);
}

/**
 * @brief Regular CRC update
 */
tle5012b_error_t tle5012b_regular_crc_update(tle5012b_dev_t *dev) {
    if (!dev) {
        return TLE5012B_ERR_INVALID_PARAM;
    }

    // Read CRC block (8 registers)
    uint16_t registers[TLE5012B_CRC_NUM_REGISTERS];
    tle5012b_error_t err = tle5012b_read_block_crc(dev, registers);
    if (err != TLE5012B_OK) {
        return err;
    }

    // Convert registers to byte array for CRC calculation
    uint8_t temp[16];
    for (uint8_t i = 0; i < TLE5012B_CRC_NUM_REGISTERS; i++) {
        temp[2 * i] = (registers[i] >> 8) & 0xFF; // High byte
        temp[(2 * i) + 1] = registers[i] & 0xFF;  // Low byte
    }

    // Calculate CRC over first 15 bytes (7.5 registers)
    uint8_t crc = tle5012b_calculate_crc_buffer(temp, 15);

    // Construct new register value: high byte from temp[14], low byte = CRC
    uint16_t val_to_send = ((uint16_t)temp[14] << 8) | crc;

    DEBUG_VERBOSE(TAG, "Regular CRC update: calculated=0x%02X, writing 0x%04X to TCO_Y", crc, val_to_send);

    // Write updated value to temperature coefficient register using special update function
    return tle5012b_write_temp_coeff_update(dev, val_to_send);
}

/**
 * @brief Convert raw angle to degrees
 */
float tle5012b_raw_to_degrees(int16_t angle_raw) {
    // 15-bit resolution: 0-32767 maps to 0-360 degrees
    return ((float)angle_raw * TLE5012B_ANGLE_360) / TLE5012B_POW_2_15;
}

/**
 * @brief Convert raw angle to radians
 */
float tle5012b_raw_to_radians(int16_t angle_raw) {
    // 15-bit resolution: 0-32767 maps to 0-2π radians
    return ((float)angle_raw * 2.0f * M_PI) / TLE5012B_POW_2_15;
}

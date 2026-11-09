/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file tle5012b.h
 * @brief TLE5012B Magnetic Angle Sensor Driver (3-wire SPI)
 *
 * This driver supports the TLE5012B GMR-based angle sensor with 3-wire SPI interface.
 * The TLE5012B provides 15-bit angle resolution and supports up to 1000 rev/s.
 */

#ifndef TLE5012B_H
#define TLE5012B_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* TLE5012B Register Addresses (with command encoding) */
#define TLE5012B_REG_STAT 0x0000    // Status register
#define TLE5012B_REG_ACSTAT 0x0010  // Activation status
#define TLE5012B_REG_AVAL 0x0020    // Angle value (15-bit)
#define TLE5012B_REG_ASPD 0x0030    // Angle speed
#define TLE5012B_REG_AREV 0x0040    // Angle revolution
#define TLE5012B_REG_FSYNC 0x0050   // Frame synchronization
#define TLE5012B_REG_MOD_1 0x0060   // Interface mode 1
#define TLE5012B_REG_SIL 0x0070     // SIL (Safety)
#define TLE5012B_REG_MOD_2 0x0080   // Interface mode 2
#define TLE5012B_REG_MOD_3 0x0090   // Interface mode 3
#define TLE5012B_REG_OFFX 0x00A0    // Offset X
#define TLE5012B_REG_OFFY 0x00B0    // Offset Y
#define TLE5012B_REG_SYNCH 0x00C0   // Synchronization
#define TLE5012B_REG_IFAB 0x00D0    // IFAB configuration
#define TLE5012B_REG_MOD_4 0x00E0   // Interface mode 4
#define TLE5012B_REG_TCO_Y 0x00F0   // Temperature coefficient Y
#define TLE5012B_REG_ADC_X 0x0100   // ADC X raw value
#define TLE5012B_REG_ADC_Y 0x0110   // ADC Y raw value
#define TLE5012B_REG_D_MAG 0x0140   // Magnitude
#define TLE5012B_REG_T_RAW 0x0150   // Temperature sensor raw
#define TLE5012B_REG_IIF_CNT 0x0200 // IIF counter
#define TLE5012B_REG_T25O 0x0300    // Temperature 25°C offset

/* Command bits */
#define TLE5012B_CMD_READ 0x8000  // Read command (bit 15)
#define TLE5012B_CMD_WRITE 0x5000 // Write command (bits 14:12 = 101)
#define TLE5012B_CMD_UPD 0x0400   // Update bit (bit 10) - read from update buffer

/* Special commands */
#define TLE5012B_CMD_READ_BLOCK_CRC 0x8088 // Read block CRC command

/* CRC block register count */
#define TLE5012B_CRC_NUM_REGISTERS 8 // Number of registers in CRC block

/* Command word encoding notes:
 * - Bit 15: READ (1) / WRITE part of 0x5000 pattern
 * - Bits 14-10: Register address (shifted to bits 14:4 in final command)
 * - Bit 10: UPD - read from update buffer when set
 * - Bits 3-0: Number of data words to read/write
 * - Bit 0: When set to 1, safety word is included in response
 */

/* Status register (STAT) bits */
#define TLE5012B_STAT_SRST (1 << 0)        // System reset
#define TLE5012B_STAT_SWD (1 << 1)         // System watchdog
#define TLE5012B_STAT_SVR (1 << 2)         // Supply voltage regulator
#define TLE5012B_STAT_SFUSE (1 << 3)       // Fuse CRC error
#define TLE5012B_STAT_SDSPU (1 << 4)       // DSPU error
#define TLE5012B_STAT_SOV (1 << 5)         // Overflow
#define TLE5012B_STAT_SXYOL (1 << 6)       // X/Y data out of limit
#define TLE5012B_STAT_SMAGOL (1 << 7)      // Magnitude out of limit
#define TLE5012B_STAT_RESERVED (1 << 8)    // Reserved
#define TLE5012B_STAT_SADCT (1 << 9)       // ADC test error
#define TLE5012B_STAT_SROM (1 << 10)       // ROM CRC error
#define TLE5012B_STAT_NOGMRXY (1 << 11)    // No valid GMR XY values
#define TLE5012B_STAT_NOGMRA (1 << 12)     // No valid GMR angle value
#define TLE5012B_STAT_SNR_MASK (0x3 << 13) // Slave number mask
#define TLE5012B_STAT_RDST (1 << 15)       // Read status

/* Interface mode 2 register (MOD_2) bits */
#define TLE5012B_MOD2_AUTOCAL (1 << 15)  // Auto-calibration mode
#define TLE5012B_MOD2_PREDICT (1 << 14)  // Prediction
#define TLE5012B_MOD2_ANGRANGE (1 << 13) // Angle range
/* Note: Slave number is in STAT[14:13] (TLE5012B_STAT_SNR_MASK), not in MOD_2;
 * MOD_2[1:0] is AUTOCAL. See tle5012b_write_slave_number(). */

/* Activation status register (ACSTAT) bits */
#define TLE5012B_ACSTAT_ASRST (1 << 0)    // Activation hardware reset
#define TLE5012B_ACSTAT_ASWD (1 << 1)     // Enable DSPU watchdog
#define TLE5012B_ACSTAT_ASVR (1 << 2)     // Enable voltage regulator check
#define TLE5012B_ACSTAT_ASFUSE (1 << 3)   // Activation fuse CRC
#define TLE5012B_ACSTAT_ASDSPU (1 << 4)   // Activation DSPU BIST
#define TLE5012B_ACSTAT_ASOV (1 << 5)     // Enable overflow check
#define TLE5012B_ACSTAT_ASVECXY (1 << 6)  // Activation X,Y out of limit check
#define TLE5012B_ACSTAT_ASVECMAG (1 << 7) // Activation magnitude check
#define TLE5012B_ACSTAT_ASADCT (1 << 9)   // Enable ADC test vector check
#define TLE5012B_ACSTAT_ASFRST (1 << 10)  // Activation firmware reset

/* Angle value register (AVAL) bits */
#define TLE5012B_AVAL_ANGVAL_MASK 0x7FFF // Angle value mask (15-bit)
#define TLE5012B_AVAL_RDAV (1 << 15)     // Read status angle value

/* Angle speed register (ASPD) bits */
#define TLE5012B_ASPD_ANGSPD_MASK 0x7FFF // Angle speed mask (15-bit signed)
#define TLE5012B_ASPD_RDAS (1 << 15)     // Read status angle speed

/* Angle revolution register (AREV) bits */
#define TLE5012B_AREV_REVOL_MASK 0x01FF // Revolution counter mask (9-bit)
#define TLE5012B_AREV_FCNT_MASK 0x7E00  // Frame counter mask
#define TLE5012B_AREV_FCNT_SHIFT 9
#define TLE5012B_AREV_RDREV (1 << 15) // Read status revolution

/* Bitmasks for safety word error checking */
#define TLE5012B_SAFETY_STAT_OK 0x4000 // System status OK (bit 14)
#define TLE5012B_SAFETY_INTF_OK 0x2000 // Interface access OK (bit 13)
#define TLE5012B_SAFETY_INV_OK 0x1000  // Angle valid (bit 12)

/* CRC parameters
 *
 * For the number of registers in the CRC block use TLE5012B_CRC_NUM_REGISTERS
 * (defined above near the read-block-CRC command). */
#define TLE5012B_CRC_POLYNOMIAL 0x1D // CRC-8 polynomial
#define TLE5012B_CRC_SEED 0xFF       // CRC seed value

/* Utility bitmasks */
#define TLE5012B_MASK_DELETE_BIT15 0x7FFF  // Mask to delete bit 15
#define TLE5012B_MASK_SIGN_BIT14 0x4000    // Check sign bit 14
#define TLE5012B_MASK_SIGN_EXTEND15 0x8000 // Sign extension for 15-bit
#define TLE5012B_MASK_9BIT 0x01FF          // 9-bit mask
#define TLE5012B_MASK_SIGN_BIT8 0x0100     // Check sign bit 8
#define TLE5012B_MASK_SIGN_EXTEND9 0x0200  // Sign extension for 9-bit
#define TLE5012B_MASK_BIT_14_4 0x7FF0      // Bits 14:4 mask

/* Mathematical constants */
#define TLE5012B_POW_2_15 32768.0f // 2^15
#define TLE5012B_POW_2_7 128.0f    // 2^7
#define TLE5012B_ANGLE_360 360.0f  // Full circle degrees

/* Temperature calculation constants */
#define TLE5012B_TEMP_OFFSET 152.0f // Temperature offset
#define TLE5012B_TEMP_DIV 2.776f    // Temperature divisor

/* Configuration */
#define TLE5012B_ANGLE_MAX 32768 // 15-bit resolution (2^15)
#define TLE5012B_ANGLE_360_DEG 32768.0f
#define TLE5012B_SPI_MAX_SPEED 8000000 // 8 MHz max SPI clock
#define TLE5012B_MAX_REGISTER_MEM 48   // Max readable register buffer (0x30)

/* Error codes */
typedef enum {
    TLE5012B_OK = 0,                 // No error
    TLE5012B_ERR_SYSTEM = 1,         // System error (voltage, ROM defect)
    TLE5012B_ERR_INTERFACE = 2,      // Interface access error
    TLE5012B_ERR_INVALID_ANGLE = 3,  // Invalid angle error
    TLE5012B_ERR_ANGLE_SPEED = 4,    // Angle speed calculation error
    TLE5012B_ERR_CRC = 0xFF,         // CRC error
    TLE5012B_ERR_INVALID_PARAM = -1, // Invalid parameter
    TLE5012B_ERR_SPI_INIT = -2,      // SPI initialization failed
    TLE5012B_ERR_TIMEOUT = -4,       // Timeout error
    TLE5012B_ERR_STATUS = -5,        // Status register error
    TLE5012B_ERR_NOT_INIT = -6       // Device not initialized
} tle5012b_error_t;

typedef struct {
    tle5012b_error_t error;
    uint16_t command;
    uint16_t safety_word;
    uint16_t status_word;
    bool has_status;
} tle5012b_diag_event_t;

/* Device-private "extra" payload that follows umi_sensor_health_header_t in
 * a SENSOR_HEALTH TLV when source_sensor_type == UMI_SENSOR_TYPE_JOINT_ENCODER
 * and the encoder is a TLE5012B. Wire layout is little-endian, packed.
 *
 * Hosts must treat any change to this struct as a UMI protocol version bump.
 * Fields are appended-only; existing field positions and sizes do not change. */
#pragma pack(push, 1)
typedef struct {
    uint16_t safety_word;  /* Last safety word received from the sensor. */
    uint16_t status_word;  /* Last STAT register value (0 if not captured). */
} tle5012b_health_extra_t;
#pragma pack(pop)

/* TLE5012B device structure */
typedef struct {
    uint8_t spi_instance;  // SPI instance (0 or 1 for RP2040)
    uint8_t pin_sck;       // SPI clock pin
    uint8_t pin_mosi;      // SPI MOSI pin
    uint8_t pin_miso;      // SPI MISO pin
    uint8_t pin_cs;        // Chip select pin
    uint32_t spi_baudrate; // SPI baudrate
    bool initialized;      // Initialization flag
    void *spi_handle;      // SPI hardware handle (spi_inst_t*)
    tle5012b_diag_event_t pending_diag;
    bool pending_diag_valid;
} tle5012b_dev_t;

/* Angle data structure */
typedef struct {
    int16_t angle_raw;   // Raw angle value (0-32767)
    float angle_deg;     // Angle in degrees (0-360)
    float angle_rad;     // Angle in radians (0-2π)
    uint16_t revolution; // Revolution counter
    int16_t speed;       // Angular speed
} tle5012b_angle_data_t;

/**
 * @brief Initialize TLE5012B sensor
 *
 * @param dev Pointer to device structure
 * @return TLE5012B_OK on success, error code otherwise
 */
tle5012b_error_t tle5012b_init(tle5012b_dev_t *dev);

/**
 * @brief Deinitialize TLE5012B sensor
 *
 * @param dev Pointer to device structure
 */
void tle5012b_deinit(tle5012b_dev_t *dev);

/**
 * @brief Trigger update to create a snapshot of UPD registers
 *
 * This function triggers an update in the register buffer. It should be called
 * once before reading UPD registers, as it generates a snapshot of the UPD
 * register values at the trigger point.
 *
 * Protocol: SCK=LOW, MOSI=HIGH, CS pulse (LOW->HIGH)
 *
 * @param dev Pointer to device structure
 * @return TLE5012B_OK on success, error code otherwise
 */
tle5012b_error_t tle5012b_trigger_update(tle5012b_dev_t *dev);

/**
 * @brief Write to sensor with optional CRC update
 *
 * Generic write function that writes data to a register and optionally
 * updates the CRC if the write affects CRC-protected registers.
 *
 * @param dev Pointer to device structure
 * @param reg_addr Register address to write
 * @param data Data to write
 * @param update_crc If true, perform regular CRC update after write
 * @return TLE5012B_OK on success, error code otherwise
 */
tle5012b_error_t tle5012b_write_to_sensor(tle5012b_dev_t *dev, uint16_t reg_addr, uint16_t data, bool update_crc);

/**
 * @brief Write temperature coefficient with update
 *
 * Writes to TCO_Y register with trigger update. If FUSE error bit (bit 3)
 * is set in status register after write, performs a regular CRC update.
 *
 * @param dev Pointer to device structure
 * @param data Data to write to TCO_Y register
 * @return TLE5012B_OK on success, error code otherwise
 */
tle5012b_error_t tle5012b_write_temp_coeff_update(tle5012b_dev_t *dev, uint16_t data);

/**
 * @brief Write the slave number (STAT[14:13])
 *
 * Sets the slave number field in bits 14:13 of the STAT register.
 * Used for daisy-chain configurations.
 *
 * @param dev Pointer to device structure
 * @param slave_num Slave number (0-3)
 * @return TLE5012B_OK on success, error code otherwise
 */
tle5012b_error_t tle5012b_write_slave_number(tle5012b_dev_t *dev, uint8_t slave_num);

/**
 * @brief Read a register from TLE5012B
 *
 * @param dev Pointer to device structure
 * @param reg_addr Register address (use TLE5012B_REG_* defines)
 * @param data Pointer to store read data
 * @param ignore_safety If true, skip safety word error checking
 * @return TLE5012B_OK on success, error code otherwise
 */
tle5012b_error_t tle5012b_read_register(tle5012b_dev_t *dev, uint16_t reg_addr, uint16_t *data, bool ignore_safety);

/**
 * @brief Write a register to TLE5012B
 *
 * @param dev Pointer to device structure
 * @param reg_addr Register address (use TLE5012B_REG_* defines)
 * @param data Data to write
 * @param ignore_safety If true, skip safety word error checking
 * @return TLE5012B_OK on success, error code otherwise
 */
tle5012b_error_t tle5012b_write_register(tle5012b_dev_t *dev, uint16_t reg_addr, uint16_t data, bool ignore_safety);

/**
 * @brief Read angle value from TLE5012B
 *
 * @param dev Pointer to device structure
 * @param angle_data Pointer to store angle data
 * @return TLE5012B_OK on success, error code otherwise
 */
tle5012b_error_t tle5012b_read_angle(tle5012b_dev_t *dev, tle5012b_angle_data_t *angle_data);

/**
 * @brief Read raw angle value (15-bit)
 *
 * @param dev Pointer to device structure
 * @param angle Pointer to store raw angle value
 * @return TLE5012B_OK on success, error code otherwise
 */
tle5012b_error_t tle5012b_read_angle_raw(tle5012b_dev_t *dev, int16_t *angle);

/**
 * @brief Retrieve and clear the most recent diagnostic event
 *
 * Driver code can populate a best-effort diagnostic event when SPI access
 * succeeds but the sensor reports a system/interface/angle-validity problem.
 * This lets Core 1 hand the event to Core 0 for logging without calling
 * printf() from the encoder sampling loop.
 *
 * @param dev Pointer to device structure
 * @param event Pointer to store diagnostic event
 * @return true if an event was returned, false otherwise
 */
bool tle5012b_take_diag_event(tle5012b_dev_t *dev, tle5012b_diag_event_t *event);

/**
 * @brief Read angle speed
 *
 * @param dev Pointer to device structure
 * @param speed Pointer to store speed value
 * @return TLE5012B_OK on success, error code otherwise
 */
tle5012b_error_t tle5012b_read_speed(tle5012b_dev_t *dev, int16_t *speed);

/**
 * @brief Read revolution counter
 *
 * @param dev Pointer to device structure
 * @param revolution Pointer to store revolution counter
 * @return TLE5012B_OK on success, error code otherwise
 */
tle5012b_error_t tle5012b_read_revolution(tle5012b_dev_t *dev, uint16_t *revolution);

/**
 * @brief Read status register
 *
 * @param dev Pointer to device structure
 * @param status Pointer to store status value
 * @return TLE5012B_OK on success, error code otherwise
 */
tle5012b_error_t tle5012b_read_status(tle5012b_dev_t *dev, uint16_t *status);

/**
 * @brief Check if sensor has errors
 *
 * @param dev Pointer to device structure
 * @param has_error Pointer to store error flag
 * @return TLE5012B_OK on success, error code otherwise
 */
tle5012b_error_t tle5012b_check_error(tle5012b_dev_t *dev, bool *has_error);

/**
 * @brief Software reset
 *
 * @param dev Pointer to device structure
 * @return TLE5012B_OK on success, error code otherwise
 */
tle5012b_error_t tle5012b_reset(tle5012b_dev_t *dev);

/**
 * @brief Read CRC block registers (8 registers)
 *
 * Reads the CRC block containing 8 registers used for CRC calculation.
 *
 * @param dev Pointer to device structure
 * @param registers Array to store 8 register values
 * @return TLE5012B_OK on success, error code otherwise
 */
tle5012b_error_t tle5012b_read_block_crc(tle5012b_dev_t *dev, uint16_t registers[TLE5012B_CRC_NUM_REGISTERS]);

/**
 * @brief Reset safety word
 *
 * Clears any safety word errors by performing a dummy read operation.
 * Should be called after detecting safety word errors.
 *
 * @param dev Pointer to device structure
 * @return TLE5012B_OK on success, error code otherwise
 */
tle5012b_error_t tle5012b_reset_safety(tle5012b_dev_t *dev);

/**
 * @brief Firmware reset
 *
 * Triggers a firmware reset by writing to activation status register.
 *
 * @param dev Pointer to device structure
 * @return TLE5012B_OK on success, error code otherwise
 */
tle5012b_error_t tle5012b_reset_firmware(tle5012b_dev_t *dev);

/**
 * @brief Regular CRC update
 *
 * Reads the CRC block, calculates CRC over the registers, and writes
 * the updated CRC value back to the temperature coefficient register.
 *
 * @param dev Pointer to device structure
 * @return TLE5012B_OK on success, error code otherwise
 */
tle5012b_error_t tle5012b_regular_crc_update(tle5012b_dev_t *dev);

/**
 * @brief Convert raw angle to degrees
 *
 * @param angle_raw Raw angle value (0-32767)
 * @return Angle in degrees (0-360)
 */
float tle5012b_raw_to_degrees(int16_t angle_raw);

/**
 * @brief Convert raw angle to radians
 *
 * @param angle_raw Raw angle value (0-32767)
 * @return Angle in radians (0-2π)
 */
float tle5012b_raw_to_radians(int16_t angle_raw);

#ifdef __cplusplus
}
#endif

#endif // TLE5012B_H

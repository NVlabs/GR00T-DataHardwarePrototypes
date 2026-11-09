# Generated Cmake Pico project file

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Initialise pico_sdk from installed location
# (note this can come from environment, CMake cache etc)

# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.2.0)
set(toolchainVersion 14_2_Rel1)
set(picotoolVersion 2.2.0-a4)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
set(PICO_BOARD pico CACHE STRING "Board type")

# Pull in Raspberry Pi Pico SDK (must be before project)
include(pico_sdk_import.cmake)

project(umi-encoder C CXX ASM)

# Initialise the Raspberry Pi Pico SDK
pico_sdk_init()

# Debug level control
# 0=Off, 1=Error, 2=Warning, 3=Info, 4=Debug, 5=Verbose
set(DEBUG_LEVEL 3 CACHE STRING "Debug output level")

# Feature toggles — override board.h defaults from the command line.
# These translate to BOARD_ENABLE_* macros (0 = disabled, 1 = enabled).
option(ENABLE_ENCODER  "Enable TLE5012B encoder on Core 1"          ON)
option(ENABLE_PDM_MIC  "Enable PDM stereo DMIC (PIO + DMA)"         ON)

# PDM microphone test mode: generates a 1 kHz sine wave instead of reading
# from real hardware.  Only meaningful when ENABLE_PDM_MIC is ON.
option(PDM_MIC_TEST_MODE "Generate test tone instead of real PDM mic" OFF)

# Add the firmware executable. Its program version is set below.

add_executable(umi-encoder 
    main.c
    usb_descriptors.c
    cdc_uart.c
    tle5012b.c
    encoder_runtime.c
    umi_protocol.c
    pdm_mic.c
    usb_audio.c
    board_io.c
    status_led.c
    OpenPDM2PCM/OpenPDMFilter.c
)

# Version schema: v2.x.x denotes the v2 gripper hardware (current design).
# Older v0.x.x / v1.x.x tags refer to legacy / pre-release boards.
pico_set_program_name(umi-encoder "umi-encoder")
pico_set_program_version(umi-encoder "2.0.0")

# Generate PIO header from .pio file
pico_generate_pio_header(umi-encoder ${CMAKE_CURRENT_LIST_DIR}/pdm_mic.pio)

# Modify the below lines to enable/disable output over UART/USB
pico_enable_stdio_uart(umi-encoder 0)
pico_enable_stdio_usb(umi-encoder 1)

# Add the standard library to the build
target_link_libraries(umi-encoder
    pico_stdlib
    pico_multicore
    pico_util
    hardware_spi
    hardware_gpio
    hardware_i2c
    hardware_adc
    hardware_pio
    hardware_dma
    hardware_clocks
)

# Add the standard include files to the build
target_include_directories(umi-encoder PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/OpenPDM2PCM
)

# Define compile-time constants
target_compile_definitions(umi-encoder PRIVATE
    # General Debugging
    DEBUG_LEVEL=${DEBUG_LEVEL}
    DEBUG_INCLUDE_CORE_ID=1
    # OpenPDMFilter needs PICO_BUILD for configurable gain
    PICO_BUILD=1
    # USB Definitions
    CFG_TUSB_CONFIG_FILE=\"tusb_config.h\"
    CFG_TUSB_DEBUG=0
    PICO_STDIO_USB_ENABLE_IRQ_BACKGROUND_TASK=0
    PICO_STDIO_USB_STDOUT_TIMEOUT_US=0
    PICO_STDIO_USB_ENABLE_TINYUSB_INIT=1
    PICO_STDIO_USB_USE_DEFAULT_DESCRIPTORS=0
    PICO_STDIO_USB_ENABLE_RESET_VIA_VENDOR_INTERFACE=1
    PICO_STDIO_USB_RESET_INTERFACE_SUPPORT_RESET_TO_BOOTSEL=1
    PICO_STDIO_USB_RESET_INTERFACE_SUPPORT_RESET_TO_FLASH_BOOT=1
    PICO_STDIO_USB_RESET_INTERFACE_SUPPORT_MS_OS_20_DESCRIPTOR=1
    PICO_STDIO_USB_RESET_BOOTSEL_INTERFACE_DISABLE_MASK=0
    PICO_STDIO_USB_RESET_RESET_TO_FLASH_DELAY_MS=100
)

# Feature toggles → BOARD_ENABLE_* compile definitions
if(ENABLE_ENCODER)
    message(STATUS "Encoder: Enabled (TLE5012B on Core 1)")
else()
    target_compile_definitions(umi-encoder PRIVATE BOARD_ENABLE_ENCODER=0)
    message(STATUS "Encoder: DISABLED (Core 1 idle)")
endif()

if(ENABLE_PDM_MIC)
    if(PDM_MIC_TEST_MODE)
        target_compile_definitions(umi-encoder PRIVATE BOARD_PDM_MIC_TEST_MODE=1)
        message(STATUS "PDM Mic: TEST MODE (1 kHz sine tone, no hardware)")
    else()
        message(STATUS "PDM Mic: Normal mode (PIO + DMA + OpenPDMFilter)")
    endif()
else()
    target_compile_definitions(umi-encoder PRIVATE BOARD_ENABLE_PDM_MIC=0)
    message(STATUS "PDM Mic: DISABLED")
endif()

# Override SDK's default CFG_TUSB_DEBUG=0 from tinyusb_common_base
get_target_property(_tusb_defs tinyusb_common_base INTERFACE_COMPILE_DEFINITIONS)
list(FILTER _tusb_defs EXCLUDE REGEX "^CFG_TUSB_DEBUG=")
set_target_properties(tinyusb_common_base PROPERTIES INTERFACE_COMPILE_DEFINITIONS "${_tusb_defs}")

pico_add_extra_outputs(umi-encoder)

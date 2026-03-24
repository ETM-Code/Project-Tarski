#pragma once

/*
    This file defines the Arduino pin mapping for the T1 devboard interface.
    Values use Arduino digital pin numbers; A0-A7 are represented as 14-21.
*/

// #define PIN_SERIAL_TX         0      //  RESERVED
// #define PIN_SERIAL_RX         1      //  RESERVED
#define PIN_LATCH_DAC            2      //  Latch pulse for DAC update
// #define PIN_XXX               3      //  UNUSED
#define PIN_PARALLEL_LD          4      //  Parallel-load control for SR latches
// #define PIN_XXX               5      //  UNUSED
#define PIN_R_CLK                6      //  Shift/register clock for serial-output devices
// #define PIN_XXX               7      //  UNUSED
// #define PIN_XXX               8      //  UNUSED
#define PIN_RESET_SR             9      //  Reset line for SR latches storing accelerator output
// #define PIN_XXX              10      //  UNUSED
#define PIN_MOSI                11      //  Serial data out from MCU
#define PIN_MISO                12      //  Serial data in to MCU
#define PIN_SCLK                13      //  Serial clock (shares onboard LED on many Arduino boards)
#define PIN_L2_EN_MEAS          14      //  A0: enable/select L2 measurement path
#define PIN_L1_EN_MEAS          15      //  A1: enable/select L1 measurement path
#define PIN_OE_S                16      //  A2: output-enable for synapse shift-register path
#define PIN_SRCLR_S             17      //  A3: clear/reset for synapse shift-register path
#define PIN_I2C_SDA             18      //  A4: reserved for I2C SDA
#define PIN_I2C_SCL             19      //  A5: reserved for I2C SCL
#define PIN_L1_MEAS_OUT         20      //  A6: analog measurement input for L1 output
#define PIN_L2_MEAS_OUT         21      //  A7: analog measurement input for L2 output

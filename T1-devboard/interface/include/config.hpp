#pragma once

/**
 * Configuration values for the T1 devboard interface.
**/

#include "datatypes.hpp"

// Synapse configuration
constexpr usize CONF_SYNAPSE_COUNT = 90;

// DAC configuration
constexpr usize CONF_DAC_COUNT = 12;
constexpr usize CONF_DAC_BITS = 12;                            // DAC resolution in bits
constexpr usize CONF_DAC_BYTE_PER = (CONF_DAC_BITS + 8 - 1) / 8; // Bytes required per DAC value
constexpr usize CONF_DAC_DATA_SIZE = CONF_DAC_COUNT * CONF_DAC_BYTE_PER; // Total DAC bytes

// MCP4728 I2C configuration
constexpr usize CONF_MCP4728_COUNT = 3;                        // Number of MCP4728 devices on the bus
constexpr u8 CONF_MCP4728_ADDRS[CONF_MCP4728_COUNT] = { 0x60, 0x61, 0x62 }; // 7-bit I2C addresses

static_assert(CONF_DAC_COUNT <= (CONF_MCP4728_COUNT * 4));

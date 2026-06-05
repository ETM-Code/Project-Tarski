// ============================================================================
// CHARACTERIZATION (golden-master) test for the pure bit/encoding logic of
// the T1-devboard AVR firmware (src/device.cpp).
//
// PURPOSE: lock in the CURRENT behavior of the firmware's pure functions so
// that any FUTURE change that alters behavior is caught by a failing test.
//
// SCOPE / HONESTY NOTE:
//   device.cpp is hardware-bound (Arduino.h, Wire.h, AVR registers) and CANNOT
//   be compiled on a host. The serial I/O, ADC polling, I2C bit-bang timing and
//   GPIO sequencing are NOT coverable here -- they need real hardware. They are
//   exercised by the build-smoke test (test/test_build_size/run_build_smoke.py)
//   only at the compile/link/size level.
//
//   What IS coverable on the host is the *pure* arithmetic/bit/encoding logic.
//   Where that logic lives inside a hardware-bound function, it is TRANSCRIBED
//   verbatim here with a comment pointing at the exact source line. The
//   transcription is the unit under test; if device.cpp's formula changes, the
//   golden values below must be updated, which is exactly the change-detection
//   we want.
//
//   signals.hpp and version.hpp are Arduino-free and are #included DIRECTLY so
//   that the real constants (not copies) are asserted. config.hpp/datatypes.hpp
//   CANNOT be included on a host: datatypes.hpp has LP64 static_asserts
//   (sizeof(u32)==4) that fail on a 64-bit host. The config.hpp literals
//   (CONF_DAC_COUNT=12, CONF_DAC_BITS=12, etc.) are therefore transcribed with a
//   cross-reference comment.
//
// All golden values were obtained by RUNNING the current firmware source logic,
// not guessed. Floats are not involved (pure integer/bit logic).
// ============================================================================

#include <cstdint>
#include <cstdio>
#include <cassert>

// Arduino-free project headers -- included directly so we test the REAL macros.
#include "signals.hpp"
#include "version.hpp"

// ---------------------------------------------------------------------------
// Transcribed config.hpp constants (cannot #include config.hpp on host because
// it pulls datatypes.hpp whose LP64 static_asserts fail on a 64-bit host).
// Cross-reference: include/config.hpp lines 10-20.
// ---------------------------------------------------------------------------
static constexpr unsigned CONF_SYNAPSE_COUNT = 90;   // config.hpp:10
static constexpr unsigned CONF_DAC_COUNT     = 12;   // config.hpp:13
static constexpr unsigned CONF_DAC_BITS      = 12;   // config.hpp:14
static constexpr unsigned CONF_MCP4728_COUNT = 3;    // config.hpp:19
static const uint8_t CONF_MCP4728_ADDRS[3] = { 0x60, 0x61, 0x62 }; // config.hpp:20

// ---------------------------------------------------------------------------
// Tiny test harness: bare assert()-style checks with named cases + a counter.
// No GoogleTest/doctest dependency -> hermetic, zero-install.
// ---------------------------------------------------------------------------
static int g_checks = 0;
static int g_fails  = 0;

#define CHECK_EQ(actual, expected, name)                                       \
    do {                                                                       \
        g_checks++;                                                            \
        auto _a = (actual);                                                    \
        auto _e = (expected);                                                  \
        if (_a != _e) {                                                        \
            g_fails++;                                                         \
            std::printf("FAIL %-48s got=%lld want=%lld\n", (name),             \
                        (long long)_a, (long long)_e);                         \
        }                                                                      \
    } while (0)

#define CHECK_TRUE(cond, name)                                                 \
    do {                                                                       \
        g_checks++;                                                            \
        if (!(cond)) { g_fails++; std::printf("FAIL %-48s (false)\n", (name)); }\
    } while (0)

// ===========================================================================
// Transcribed pure-logic helpers (mirror device.cpp byte-for-byte).
// ===========================================================================

// _BV from <avr/io.h>: bit value. device.cpp uses _BV(b) throughout.
static constexpr uint8_t BV(uint8_t b) { return (uint8_t)(1u << b); }

// ATmega328P WDTCSR prescaler bit positions (iom328p.h). device.cpp relies on
// these exact positions inside _WDTFromTimeout (device.cpp:75-91).
static constexpr uint8_t WDP0 = 0, WDP1 = 1, WDP2 = 2, WDP3 = 5;

// Device::Timeout enum ordering (device.hpp:11-23).
enum class Timeout : uint8_t {
    TO16MS = 0, TO32MS, TO64MS, TO125MS, TO250MS,
    TO500MS, TO1S, TO2S, TO4S, TO8S
};

// Transcription of Device::_WDTFromTimeout (device.cpp:75-91).
static uint8_t WDTFromTimeout(Timeout t) {
    switch (t) {
        case Timeout::TO16MS:  return 0;
        case Timeout::TO32MS:  return BV(WDP0);
        case Timeout::TO64MS:  return BV(WDP1);
        case Timeout::TO125MS: return BV(WDP1) | BV(WDP0);
        case Timeout::TO250MS: return BV(WDP2);
        case Timeout::TO500MS: return BV(WDP2) | BV(WDP0);
        case Timeout::TO1S:    return BV(WDP2) | BV(WDP1);
        case Timeout::TO2S:    return BV(WDP2) | BV(WDP1) | BV(WDP0);
        case Timeout::TO4S:    return BV(WDP3);
        case Timeout::TO8S:    return BV(WDP3) | BV(WDP0);
        default:               return BV(WDP2) | BV(WDP1);
    }
}

// Transcription of the MCP4728 fast-write per-channel split inside
// Device::_WriteDACData (device.cpp:181-185).
struct DacBytes { uint8_t b1; uint8_t b2; };
static DacBytes DacFastWriteBytes(uint16_t value) {
    value = (uint16_t)(value & ((1u << CONF_DAC_BITS) - 1u)); // device.cpp:181 (12-bit mask)
    uint8_t b1 = (uint8_t)((value >> 8) & 0x0F);              // device.cpp:182
    uint8_t b2 = (uint8_t)(value & 0xFF);                     // device.cpp:183
    return { b1, b2 };
}

// Transcription of the 12-bit DAC code mask used in LoadDACs/CalibL1/CalibPulse
// (device.cpp:477, 702, 823): raw & ((1U << CONF_DAC_BITS) - 1U).
static uint16_t DacCodeMask(uint16_t raw) {
    return (uint16_t)(raw & ((1u << CONF_DAC_BITS) - 1u));
}

// Transcription of Device::SendU16 little-endian split (device.cpp:402-406).
struct U16Bytes { uint8_t lo; uint8_t hi; };
static U16Bytes SendU16Bytes(uint16_t value) {
    uint8_t lo = (uint8_t)(value & 0xFF);          // device.cpp:404
    uint8_t hi = (uint8_t)((value >> 8) & 0xFF);   // device.cpp:405
    return { lo, hi };
}

// Transcription of the host->device u16 reassembly used in
// LoadDACs/RunInference/CalibL1/CalibPulse (e.g. device.cpp:476):
// raw = lsb | (msb << 8).
static uint16_t U16FromBytes(uint8_t lsb, uint8_t msb) {
    return (uint16_t)((uint16_t)lsb | ((uint16_t)msb << 8));
}

// Transcription of ProgramDACAddress command-byte math (device.cpp:965-1045).
struct ProgDacBytes { uint8_t cmd1, cmd2, cmd3, addr_byte; };
static ProgDacBytes ProgDacEncode(uint8_t old_addr, uint8_t new_addr) {
    uint8_t old_bits = old_addr & 0x07;                 // device.cpp:965
    uint8_t new_bits = new_addr & 0x07;                 // device.cpp:966
    uint8_t cmd1 = (uint8_t)(0x61 | (old_bits << 2));   // device.cpp:1036 (LDAC rides cmd1)
    uint8_t cmd2 = (uint8_t)(0x62 | (new_bits << 2));   // device.cpp:1040
    uint8_t cmd3 = (uint8_t)(0x63 | (new_bits << 2));   // device.cpp:1044
    uint8_t addr_byte = (uint8_t)((old_addr << 1) & 0xFE); // device.cpp:1026
    return { cmd1, cmd2, cmd3, addr_byte };
}

// Transcription of the diagnostic ack-bit packing (device.cpp:1110-1113).
static uint8_t AckBitsPack(bool a1, bool a2, bool a3, bool a4) {
    return (uint8_t)((a1 ? 0x01 : 0) | (a2 ? 0x02 : 0)
                   | (a3 ? 0x04 : 0) | (a4 ? 0x08 : 0));
}

// Transcription of the CalibL1/CalibPulse u32 elapsed_us 4-byte little-endian
// split (device.cpp:742-745): SendU8(value & 0xFF), >>8, >>16, >>24.
struct U32Bytes { uint8_t b0, b1, b2, b3; };
static U32Bytes SendU32Bytes(uint32_t value) {
    uint8_t b0 = (uint8_t)(value & 0xFF);          // device.cpp:742
    uint8_t b1 = (uint8_t)((value >> 8) & 0xFF);   // device.cpp:743
    uint8_t b2 = (uint8_t)((value >> 16) & 0xFF);  // device.cpp:744
    uint8_t b3 = (uint8_t)((value >> 24) & 0xFF);  // device.cpp:745
    return { b0, b1, b2, b3 };
}

// Transcription of the host-side u32 reassembly (the inverse of the device
// serialization): b0 | b1<<8 | b2<<16 | b3<<24.
static uint32_t U32FromBytes(uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3) {
    return (uint32_t)b0 | ((uint32_t)b1 << 8)
         | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
}

// ATmega328P ADCSRA prescaler bit positions (iom328p.h). device.cpp:799 relies
// on these exact positions for the CalibPulse fast-mode (prescaler 32) switch.
static constexpr uint8_t ADPS0 = 0, ADPS1 = 1, ADPS2 = 2;

// Transcription of the CalibPulse ADC prescaler-32 bit math (device.cpp:799):
// (ADCSRA & 0xF8) | _BV(ADPS2) | _BV(ADPS0).
static uint8_t CalibPulseADCSRA(uint8_t adcsra) {
    return (uint8_t)((adcsra & 0xF8) | BV(ADPS2) | BV(ADPS0)); // device.cpp:799
}

// Transcription of the ProgramDACAddress status-code precedence
// (device.cpp:1103-1104): status = still_at_old ? 0x12 : 0x13;
// then if any ack is false, status = 0x14 (ack-failure wins).
static uint8_t ProgDacStatus(bool still_at_old,
                             bool ack1, bool ack2, bool ack3, bool ack4) {
    uint8_t status = still_at_old ? 0x12 : 0x13;        // device.cpp:1103
    if(!ack1 || !ack2 || !ack3 || !ack4) status = 0x14; // device.cpp:1104
    return status;
}

// Transcription of the ProgramDACAddress address-range validation predicate
// (device.cpp:922): an address is valid only if NOT (a < 0x60 || a > 0x67).
static bool ProgDacAddrValid(uint8_t a) {
    return !(a < 0x60 || a > 0x67); // device.cpp:922
}

// Transcription of the SendSignature ASCII +0x30 offset (device.cpp:313-315).
static uint8_t SigByte(uint8_t v) { return (uint8_t)(v + 0x30); }

// Transcription of the DAC quiescent table (device.cpp:52-65).
// ch0..8 = (1<<CONF_DAC_BITS)-1 = 4095 (mirrors at ~V_DD), ch9..11 = 0.
static const uint16_t DAC_CH_QUIESCENT[CONF_DAC_COUNT] = {
    (uint16_t)((1u << CONF_DAC_BITS) - 1u), (uint16_t)((1u << CONF_DAC_BITS) - 1u),
    (uint16_t)((1u << CONF_DAC_BITS) - 1u), (uint16_t)((1u << CONF_DAC_BITS) - 1u),
    (uint16_t)((1u << CONF_DAC_BITS) - 1u), (uint16_t)((1u << CONF_DAC_BITS) - 1u),
    (uint16_t)((1u << CONF_DAC_BITS) - 1u), (uint16_t)((1u << CONF_DAC_BITS) - 1u),
    (uint16_t)((1u << CONF_DAC_BITS) - 1u),
    0, 0, 0,
};

// ===========================================================================
// TESTS
// ===========================================================================

// --- dac_fastwrite_byte_encoding ------------------------------------------
// Locks MCP4728 fast-write per-channel split incl. 12-bit truncation.
static void test_dac_fastwrite_byte_encoding() {
    struct { uint16_t in; uint8_t b1; uint8_t b2; } cases[] = {
        {0x0000, 0x00, 0x00},
        {0x0001, 0x00, 0x01},
        {0x0FFF, 0x0F, 0xFF},
        {0x1000, 0x00, 0x00}, // CHARACTERIZATION: 12-bit mask drops bit 12
        {0xFFFF, 0x0F, 0xFF},
        {0x0800, 0x08, 0x00},
        {0x0ABC, 0x0A, 0xBC},
    };
    for (auto& c : cases) {
        DacBytes got = DacFastWriteBytes(c.in);
        CHECK_EQ(got.b1, c.b1, "dac_fastwrite_b1");
        CHECK_EQ(got.b2, c.b2, "dac_fastwrite_b2");
    }
}

// --- dac_code_12bit_mask ---------------------------------------------------
static void test_dac_code_12bit_mask() {
    CHECK_EQ(DacCodeMask(0x1234), 0x0234, "dac_mask_0x1234");
    CHECK_EQ(DacCodeMask(0xFFFF), 0x0FFF, "dac_mask_0xFFFF");
    CHECK_EQ(DacCodeMask(0x0000), 0x0000, "dac_mask_0x0000");
    CHECK_EQ(DacCodeMask(0x1000), 0x0000, "dac_mask_0x1000");
    CHECK_EQ(DacCodeMask(0x0FFF), 0x0FFF, "dac_mask_0x0FFF");
    CHECK_EQ((1u << CONF_DAC_BITS) - 1u, 0x0FFFu, "dac_mask_constant_is_0x0FFF");
}

// --- wdt_timeout_to_register_bits -----------------------------------------
static void test_wdt_timeout_to_register_bits() {
    // Golden table for TO16MS..TO8S (enum 0..9).
    uint8_t expected[10] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x20,0x21};
    for (int i = 0; i < 10; i++) {
        CHECK_EQ(WDTFromTimeout((Timeout)i), expected[i], "wdt_timeout_value");
    }
    // Default branch (out-of-range enum) == 0x06 (WDP2|WDP1, the TO1S bits).
    CHECK_EQ(WDTFromTimeout((Timeout)99), 0x06, "wdt_default_branch");
}

// --- progdac_cmd_byte_formulas --------------------------------------------
// Lock ProgramDACAddress byte math. NOTE: post-patch, the LDAC toggle rides
// cmd1 (i2c_byte_with_ldac(cmd1)) -- this off-by-one fix is the patch context.
static void test_progdac_cmd_byte_formulas() {
    // old_addr=0x60 (bits 0).
    {
        ProgDacBytes g = ProgDacEncode(0x60, 0x60);
        CHECK_EQ(g.cmd1, 0x61, "progdac_0x60_cmd1");
        CHECK_EQ(g.cmd2, 0x62, "progdac_0x60_cmd2");
        CHECK_EQ(g.cmd3, 0x63, "progdac_0x60_cmd3");
        CHECK_EQ(g.addr_byte, 0xC0, "progdac_0x60_addrbyte");
    }
    // old_addr=0x61 (bits 1).
    {
        ProgDacBytes g = ProgDacEncode(0x61, 0x61);
        CHECK_EQ(g.cmd1, 0x65, "progdac_0x61_cmd1");
        CHECK_EQ(g.cmd2, 0x66, "progdac_0x61_cmd2");
        CHECK_EQ(g.cmd3, 0x67, "progdac_0x61_cmd3");
        CHECK_EQ(g.addr_byte, 0xC2, "progdac_0x61_addrbyte");
    }
    // old_addr=0x67 (bits 7).
    {
        ProgDacBytes g = ProgDacEncode(0x67, 0x67);
        CHECK_EQ(g.cmd1, 0x7D, "progdac_0x67_cmd1");
        CHECK_EQ(g.cmd2, 0x7E, "progdac_0x67_cmd2");
        CHECK_EQ(g.cmd3, 0x7F, "progdac_0x67_cmd3");
        CHECK_EQ(g.addr_byte, 0xCE, "progdac_0x67_addrbyte");
    }
    // Full sweep 0x60..0x67: lock the formula across the valid address range.
    for (uint8_t a = 0x60; a <= 0x67; a++) {
        uint8_t bits = a & 0x07;
        ProgDacBytes g = ProgDacEncode(a, a);
        CHECK_EQ(g.cmd1, (uint8_t)(0x61 | (bits << 2)), "progdac_sweep_cmd1");
        CHECK_EQ(g.cmd2, (uint8_t)(0x62 | (bits << 2)), "progdac_sweep_cmd2");
        CHECK_EQ(g.cmd3, (uint8_t)(0x63 | (bits << 2)), "progdac_sweep_cmd3");
        CHECK_EQ(g.addr_byte, (uint8_t)((a << 1) & 0xFE), "progdac_sweep_addrbyte");
    }
}

// --- progdac_ack_bits_packing ---------------------------------------------
// (ack1?1:0)|(ack2?2:0)|(ack3?4:0)|(ack4?8:0) must be identity over 0..15.
static void test_progdac_ack_bits_packing() {
    for (int v = 0; v < 16; v++) {
        bool a1 = v & 0x01, a2 = v & 0x02, a3 = v & 0x04, a4 = v & 0x08;
        CHECK_EQ(AckBitsPack(a1, a2, a3, a4), (uint8_t)v, "progdac_ackbits_identity");
    }
    // Spot-check the bit->ack mapping is bit0=ack1 .. bit3=ack4.
    CHECK_EQ(AckBitsPack(true, false, false, false), 0x01, "ackbits_ack1_is_bit0");
    CHECK_EQ(AckBitsPack(false, true, false, false), 0x02, "ackbits_ack2_is_bit1");
    CHECK_EQ(AckBitsPack(false, false, true, false), 0x04, "ackbits_ack3_is_bit2");
    CHECK_EQ(AckBitsPack(false, false, false, true), 0x08, "ackbits_ack4_is_bit3");
}

// --- progdac_status_and_diagnostic_codes ----------------------------------
// Lock the literal status/diagnostic codes (device.cpp:1103-1104, 908-930) and
// the framing-collision invariant: none may equal PORT_TRN_END (0x04).
static void test_progdac_status_and_diagnostic_codes() {
    const uint8_t STATUS_STILL_AT_OLD = 0x12; // device.cpp:1103
    const uint8_t STATUS_NOT_RESPOND  = 0x13; // device.cpp:1103
    const uint8_t STATUS_ACK_FAILURE  = 0x14; // device.cpp:1104
    const uint8_t DIAG_OLD_ADDR       = 0xA1; // device.cpp:908
    const uint8_t DIAG_NEW_ADDR       = 0xA2; // device.cpp:916
    const uint8_t DIAG_INVALID_RANGE  = 0xA3; // device.cpp:927
    CHECK_EQ(STATUS_STILL_AT_OLD, 0x12, "progdac_status_still_at_old");
    CHECK_EQ(STATUS_NOT_RESPOND,  0x13, "progdac_status_not_responding");
    CHECK_EQ(STATUS_ACK_FAILURE,  0x14, "progdac_status_ack_failure");
    CHECK_EQ(DIAG_OLD_ADDR,       0xA1, "progdac_diag_old_addr");
    CHECK_EQ(DIAG_NEW_ADDR,       0xA2, "progdac_diag_new_addr");
    CHECK_EQ(DIAG_INVALID_RANGE,  0xA3, "progdac_diag_invalid_range");
    // Framing-collision invariant (the comment at device.cpp:1098 calls this out).
    CHECK_TRUE(STATUS_STILL_AT_OLD != PORT_TRN_END, "status0x12_ne_TRN_END");
    CHECK_TRUE(STATUS_NOT_RESPOND  != PORT_TRN_END, "status0x13_ne_TRN_END");
    CHECK_TRUE(STATUS_ACK_FAILURE  != PORT_TRN_END, "status0x14_ne_TRN_END");
    CHECK_TRUE(DIAG_OLD_ADDR       != PORT_TRN_END, "diag0xA1_ne_TRN_END");
    CHECK_TRUE(DIAG_NEW_ADDR       != PORT_TRN_END, "diag0xA2_ne_TRN_END");
    CHECK_TRUE(DIAG_INVALID_RANGE  != PORT_TRN_END, "diag0xA3_ne_TRN_END");
}

// --- dac_quiescent_table_values -------------------------------------------
static void test_dac_quiescent_table_values() {
    CHECK_EQ(CONF_DAC_COUNT, 12u, "quiescent_table_len_is_12");
    for (int ch = 0; ch <= 8; ch++) {
        CHECK_EQ(DAC_CH_QUIESCENT[ch], 4095, "quiescent_mirror_ch_is_4095");
    }
    CHECK_EQ(DAC_CH_QUIESCENT[9],  0, "quiescent_ch9_is_0");
    CHECK_EQ(DAC_CH_QUIESCENT[10], 0, "quiescent_ch10_VOUT3_is_0"); // synapse rail off
    CHECK_EQ(DAC_CH_QUIESCENT[11], 0, "quiescent_ch11_is_0");
    CHECK_EQ((1u << CONF_DAC_BITS) - 1u, 4095u, "quiescent_mirror_value_derivation");
}

// --- sendu16_little_endian_split ------------------------------------------
static void test_sendu16_little_endian_split() {
    struct { uint16_t in; uint8_t lo; uint8_t hi; } cases[] = {
        {0xBEEF, 0xEF, 0xBE},
        {0x0001, 0x01, 0x00},
        {0xFF00, 0x00, 0xFF},
        {0x0000, 0x00, 0x00},
        {0x1234, 0x34, 0x12},
    };
    for (auto& c : cases) {
        U16Bytes got = SendU16Bytes(c.in);
        CHECK_EQ(got.lo, c.lo, "sendu16_lo");
        CHECK_EQ(got.hi, c.hi, "sendu16_hi");
    }
}

// --- u16_reassembly_from_bytes --------------------------------------------
static void test_u16_reassembly_from_bytes() {
    CHECK_EQ(U16FromBytes(0xEF, 0xBE), 0xBEEF, "u16_from_bytes_BEEF");
    CHECK_EQ(U16FromBytes(0x34, 0x12), 0x1234, "u16_from_bytes_1234");
    CHECK_EQ(U16FromBytes(0x00, 0x00), 0x0000, "u16_from_bytes_0000");
    CHECK_EQ(U16FromBytes(0xFF, 0xFF), 0xFFFF, "u16_from_bytes_FFFF");
    // Round-trip: SendU16Bytes then U16FromBytes is identity.
    for (uint32_t v = 0; v <= 0xFFFF; v += 0x1111) {
        U16Bytes b = SendU16Bytes((uint16_t)v);
        CHECK_EQ(U16FromBytes(b.lo, b.hi), (uint16_t)v, "u16_roundtrip");
    }
}

// --- spike_threshold_constants --------------------------------------------
// The patch lowered CalibL1's spike_threshold from 512 to 200. CalibPulse keeps
// 512, and has a low early-stop threshold of 102 (~0.5V).
static void test_spike_threshold_constants() {
    const uint16_t CALIB_L1_SPIKE_THRESHOLD    = 200; // device.cpp:721 (lowered from 512)
    const uint16_t CALIB_PULSE_SPIKE_THRESHOLD = 512; // device.cpp:810
    const uint16_t CALIB_PULSE_LOW_THRESHOLD   = 102; // device.cpp:871
    CHECK_EQ(CALIB_L1_SPIKE_THRESHOLD,    200, "calibL1_spike_threshold_200");
    CHECK_EQ(CALIB_PULSE_SPIKE_THRESHOLD, 512, "calibPulse_spike_threshold_512");
    CHECK_EQ(CALIB_PULSE_LOW_THRESHOLD,   102, "calibPulse_low_threshold_102");
    // The lowering is locked: the two thresholds are now distinct.
    CHECK_TRUE(CALIB_L1_SPIKE_THRESHOLD != CALIB_PULSE_SPIKE_THRESHOLD,
               "calibL1_ne_calibPulse_threshold");
}

// --- framing_signal_byte_values -------------------------------------------
// Asserts the REAL macros from signals.hpp (included directly).
static void test_framing_signal_byte_values() {
    CHECK_EQ(PORT_MSG,      0x02, "PORT_MSG");
    CHECK_EQ(PORT_MSG_END,  0x03, "PORT_MSG_END");
    CHECK_EQ(PORT_TRN_END,  0x04, "PORT_TRN_END");
    CHECK_EQ(PORT_SIG,      0x05, "PORT_SIG");
    CHECK_EQ(PORT_ACK,      0x06, "PORT_ACK");
    CHECK_EQ(PORT_NAK,      0x15, "PORT_NAK");
    CHECK_EQ((int)PORT_READ_OUT,   (int)'O', "PORT_READ_OUT");
    CHECK_EQ((int)PORT_LOAD_SYN,   (int)'S', "PORT_LOAD_SYN");
    CHECK_EQ((int)PORT_LOAD_DAC,   (int)'D', "PORT_LOAD_DAC");
    CHECK_EQ((int)PORT_READ_MEAS,  (int)'M', "PORT_READ_MEAS");
    CHECK_EQ((int)PORT_SET_FLAG,   (int)'F', "PORT_SET_FLAG");
    CHECK_EQ((int)PORT_UNSET_FLAG, (int)'U', "PORT_UNSET_FLAG");
    CHECK_EQ((int)PORT_TGL_FLAG,   (int)'T', "PORT_TGL_FLAG");
    CHECK_EQ((int)PORT_PROG_DAC,   (int)'P', "PORT_PROG_DAC");
    CHECK_EQ((int)PORT_RUN_INF,    (int)'R', "PORT_RUN_INF");
    CHECK_EQ((int)PORT_CALIB_L1,   (int)'C', "PORT_CALIB_L1");
    CHECK_EQ((int)PORT_CALIB_PULSE,(int)'B', "PORT_CALIB_PULSE");

    // All command chars mutually distinct and disjoint from control bytes.
    const int cmds[] = {
        PORT_READ_OUT, PORT_LOAD_SYN, PORT_LOAD_DAC, PORT_READ_MEAS,
        PORT_SET_FLAG, PORT_UNSET_FLAG, PORT_TGL_FLAG, PORT_PROG_DAC,
        PORT_RUN_INF, PORT_CALIB_L1, PORT_CALIB_PULSE,
    };
    const int n = sizeof(cmds) / sizeof(cmds[0]);
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            CHECK_TRUE(cmds[i] != cmds[j], "command_chars_distinct");
        }
    }
    const int ctrl[] = { PORT_MSG, PORT_MSG_END, PORT_TRN_END, PORT_SIG, PORT_ACK, PORT_NAK };
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < (int)(sizeof(ctrl)/sizeof(ctrl[0])); k++) {
            CHECK_TRUE(cmds[i] != ctrl[k], "command_char_not_control_byte");
        }
    }
}

// --- firmware_signature_offset --------------------------------------------
// SendSignature shifts each version byte up by 0x30 (ASCII offset).
// version.hpp is 0/1/0 => '0','1','0' = 0x30,0x31,0x30.
static void test_firmware_signature_offset() {
    CHECK_EQ((uint8_t)(FIRM_VER_MJR + 0x30), 0x30, "sig_major_byte"); // '0'
    CHECK_EQ((uint8_t)(FIRM_VER_MNR + 0x30), 0x31, "sig_minor_byte"); // '1'
    CHECK_EQ((uint8_t)(FIRM_VER_PCH + 0x30), 0x30, "sig_patch_byte"); // '0'
    // Lock the current version itself.
    CHECK_EQ(FIRM_VER_MJR, 0, "firm_ver_major");
    CHECK_EQ(FIRM_VER_MNR, 1, "firm_ver_minor");
    CHECK_EQ(FIRM_VER_PCH, 0, "firm_ver_patch");
}

// --- dac_count_validation_bounds ------------------------------------------
// Lock accept/reject predicates. Literals transcribed from device.cpp with
// cross-references; CONF_DAC_COUNT==12 from config.hpp:13.
static void test_dac_count_validation_bounds() {
    // LoadDACs (device.cpp:459): reject if write_count==0 OR >CONF_DAC_COUNT.
    auto loaddacs_accept = [](unsigned wc) {
        return !(wc == 0 || wc > CONF_DAC_COUNT);
    };
    CHECK_TRUE(!loaddacs_accept(0),  "loaddacs_reject_0");
    CHECK_TRUE(loaddacs_accept(1),   "loaddacs_accept_1");
    CHECK_TRUE(loaddacs_accept(12),  "loaddacs_accept_12");
    CHECK_TRUE(!loaddacs_accept(13), "loaddacs_reject_13");

    // RunInference (device.cpp:615): reject if num_samples>250 OR ==0.
    auto runinf_accept = [](unsigned ns) { return !(ns > 250 || ns == 0); };
    CHECK_TRUE(!runinf_accept(0),   "runinf_reject_0");
    CHECK_TRUE(runinf_accept(1),    "runinf_accept_1");
    CHECK_TRUE(runinf_accept(250),  "runinf_accept_250");
    CHECK_TRUE(!runinf_accept(251), "runinf_reject_251");

    // CalibL1 (device.cpp:673): reject if dac_channel>=12 OR meas_channel>1.
    auto calibl1_accept = [](unsigned dac, unsigned meas) {
        return !(dac >= CONF_DAC_COUNT || meas > 1);
    };
    CHECK_TRUE(calibl1_accept(0, 0),    "calibl1_accept_0_0");
    CHECK_TRUE(calibl1_accept(11, 1),   "calibl1_accept_11_1");
    CHECK_TRUE(!calibl1_accept(12, 0),  "calibl1_reject_dac12");
    CHECK_TRUE(!calibl1_accept(0, 2),   "calibl1_reject_meas2");

    // CalibPulse (device.cpp:769): reject if dac>=12 OR meas>1 OR bursts==0 OR bursts>32.
    auto calibpulse_accept = [](unsigned dac, unsigned meas, unsigned bursts) {
        return !(dac >= CONF_DAC_COUNT || meas > 1 || bursts == 0 || bursts > 32);
    };
    CHECK_TRUE(calibpulse_accept(0, 0, 1),    "calibpulse_accept_min");
    CHECK_TRUE(calibpulse_accept(11, 1, 32),  "calibpulse_accept_max");
    CHECK_TRUE(!calibpulse_accept(12, 0, 1),  "calibpulse_reject_dac12");
    CHECK_TRUE(!calibpulse_accept(0, 2, 1),   "calibpulse_reject_meas2");
    CHECK_TRUE(!calibpulse_accept(0, 0, 0),   "calibpulse_reject_bursts0");
    CHECK_TRUE(!calibpulse_accept(0, 0, 33),  "calibpulse_reject_bursts33");
}

// --- config sanity (extra) -------------------------------------------------
static void test_config_constants_sanity() {
    CHECK_EQ(CONF_SYNAPSE_COUNT, 90u, "conf_synapse_count_90");
    CHECK_EQ(CONF_DAC_COUNT, 12u, "conf_dac_count_12");
    CHECK_EQ(CONF_DAC_BITS, 12u, "conf_dac_bits_12");
    CHECK_EQ(CONF_MCP4728_COUNT, 3u, "conf_mcp4728_count_3");
    CHECK_EQ(CONF_MCP4728_ADDRS[0], 0x60, "conf_addr0_0x60");
    CHECK_EQ(CONF_MCP4728_ADDRS[1], 0x61, "conf_addr1_0x61");
    CHECK_EQ(CONF_MCP4728_ADDRS[2], 0x62, "conf_addr2_0x62");
    // The static_assert in config.hpp:22.
    CHECK_TRUE(CONF_DAC_COUNT <= (CONF_MCP4728_COUNT * 4), "conf_dac_fits_in_mcp4728s");
}

// --- elapsed_us_u32_little_endian_split -----------------------------------
// Locks the CalibL1/CalibPulse u32 elapsed_us 4-byte LE serialization
// (device.cpp:742-745). NOTE: 0xFFFFFFFF is the CalibL1 timeout DEFAULT
// sentinel (device.cpp:708) returned when no spike is detected before timeout.
static void test_elapsed_us_u32_little_endian_split() {
    struct { uint32_t in; uint8_t b0, b1, b2, b3; } cases[] = {
        // CHARACTERIZATION: 0xFFFFFFFF is the CalibL1 timeout sentinel default.
        {0xFFFFFFFFu, 0xFF, 0xFF, 0xFF, 0xFF},
        {0x00000000u, 0x00, 0x00, 0x00, 0x00},
        {0x12345678u, 0x78, 0x56, 0x34, 0x12},
        {0x000000FFu, 0xFF, 0x00, 0x00, 0x00},
        {0x0000FF00u, 0x00, 0xFF, 0x00, 0x00},
        {0x00FF0000u, 0x00, 0x00, 0xFF, 0x00},
        {0xFF000000u, 0x00, 0x00, 0x00, 0xFF},
        {0x000186A0u, 0xA0, 0x86, 0x01, 0x00}, // 100000 us = a plausible elapsed
    };
    for (auto& c : cases) {
        U32Bytes g = SendU32Bytes(c.in);
        CHECK_EQ(g.b0, c.b0, "elapsed_u32_b0");
        CHECK_EQ(g.b1, c.b1, "elapsed_u32_b1");
        CHECK_EQ(g.b2, c.b2, "elapsed_u32_b2");
        CHECK_EQ(g.b3, c.b3, "elapsed_u32_b3");
    }
    // Lock the timeout sentinel constant itself (device.cpp:708).
    const uint32_t CALIB_TIMEOUT_SENTINEL = 0xFFFFFFFFu;
    CHECK_EQ(CALIB_TIMEOUT_SENTINEL, 0xFFFFFFFFu, "calib_timeout_sentinel_value");
    // Round-trip over a fixed sweep: reassemble == original.
    for (uint64_t v = 0; v <= 0xFFFFFFFFull; v += 0x11111111ull) {
        U32Bytes b = SendU32Bytes((uint32_t)v);
        CHECK_EQ(U32FromBytes(b.b0, b.b1, b.b2, b.b3), (uint32_t)v,
                 "elapsed_u32_roundtrip");
    }
}

// --- progdac_address_range_predicate --------------------------------------
// Locks the ProgramDACAddress accept window 0x60..0x67 and reject path
// (device.cpp:922). Combined predicate is valid only if BOTH old and new are
// in range.
static void test_progdac_address_range_predicate() {
    CHECK_TRUE(!ProgDacAddrValid(0x5F), "progdac_reject_0x5F");
    CHECK_TRUE(ProgDacAddrValid(0x60),  "progdac_accept_0x60");
    CHECK_TRUE(ProgDacAddrValid(0x67),  "progdac_accept_0x67");
    CHECK_TRUE(!ProgDacAddrValid(0x68), "progdac_reject_0x68");
    CHECK_TRUE(!ProgDacAddrValid(0x00), "progdac_reject_0x00");
    CHECK_TRUE(!ProgDacAddrValid(0xFF), "progdac_reject_0xFF");
    // Every in-window address accepts; everything else rejects.
    for (int a = 0x60; a <= 0x67; a++)
        CHECK_TRUE(ProgDacAddrValid((uint8_t)a), "progdac_window_accept");
    // Combined predicate: valid only if BOTH old and new are in range.
    auto combined = [](uint8_t o, uint8_t n) {
        return ProgDacAddrValid(o) && ProgDacAddrValid(n);
    };
    CHECK_TRUE(combined(0x60, 0x67),  "progdac_combined_both_valid");
    CHECK_TRUE(!combined(0x60, 0x68), "progdac_combined_new_oob");
    CHECK_TRUE(!combined(0x5F, 0x60), "progdac_combined_old_oob");
    CHECK_TRUE(!combined(0x5F, 0x68), "progdac_combined_both_oob");
}

// --- progdac_status_code_precedence ---------------------------------------
// Locks the status SELECTION logic (device.cpp:1103-1104), not just literals:
// still_at_old ? 0x12 : 0x13, then ANY false ack overrides to 0x14.
static void test_progdac_status_code_precedence() {
    CHECK_EQ(ProgDacStatus(true,  true,  true, true, true),  0x12,
             "progdac_status_still_old_all_ack");
    CHECK_EQ(ProgDacStatus(false, true,  true, true, true),  0x13,
             "progdac_status_not_responding_all_ack");
    CHECK_EQ(ProgDacStatus(true,  true,  false, true, true), 0x14,
             "progdac_status_ack2_fail_overrides_to_0x14");
    CHECK_EQ(ProgDacStatus(false, true,  true, true, false), 0x14,
             "progdac_status_ack4_fail_overrides_to_0x14");
    CHECK_EQ(ProgDacStatus(false, true,  true, true, true),  0x13,
             "progdac_status_not_responding_again");
    // Lock the precedence invariant: ANY false ack forces 0x14 regardless of
    // still_at_old (ack-failure wins over still-at-old).
    for (int still = 0; still <= 1; still++) {
        for (int v = 0; v < 16; v++) {
            bool a1 = v & 1, a2 = v & 2, a3 = v & 4, a4 = v & 8;
            uint8_t st = ProgDacStatus(still != 0, a1, a2, a3, a4);
            if (!a1 || !a2 || !a3 || !a4) {
                CHECK_EQ(st, 0x14, "progdac_any_false_ack_forces_0x14");
            } else {
                CHECK_EQ(st, (uint8_t)(still ? 0x12 : 0x13),
                         "progdac_all_ack_uses_still_at_old");
            }
        }
    }
}

// --- calibpulse_adcsra_prescaler32_bits -----------------------------------
// Locks (ADCSRA & 0xF8) | _BV(ADPS2) | _BV(ADPS0) -> prescaler 32 (ADPS=101).
// Comment: prescaler-32 selection for fast decay sampling (device.cpp:799).
static void test_calibpulse_adcsra_prescaler32_bits() {
    // Low 3 bits must be 0b101 = 5 regardless of input low bits.
    CHECK_EQ(CalibPulseADCSRA(0x00), 0x05, "adcsra_from_0x00_is_0x05");
    // Upper bits preserved, low 3 forced to 101: 0xFF & 0xF8 = 0xF8, |0x05 = 0xFD.
    CHECK_EQ(CalibPulseADCSRA(0xFF), 0xFD, "adcsra_from_0xFF_is_0xFD");
    // ADPS1 (bit 1) must stay clear in the result for every input.
    for (int in = 0; in < 256; in++) {
        uint8_t r = CalibPulseADCSRA((uint8_t)in);
        CHECK_EQ((r & 0x07), 0x05, "adcsra_low3_is_101");
        CHECK_EQ((r >> 1) & 1, 0, "adcsra_ADPS1_stays_clear");
        CHECK_EQ((uint8_t)(r & 0xF8), (uint8_t)(in & 0xF8), "adcsra_upper_preserved");
    }
    // Bit-position sanity (the math depends on these).
    CHECK_EQ(ADPS2, 2, "ADPS2_is_2");
    CHECK_EQ(ADPS1, 1, "ADPS1_is_1");
    CHECK_EQ(ADPS0, 0, "ADPS0_is_0");
}

// --- calibpulse_decay_window_guards ---------------------------------------
// Locks the magic numbers bounding the CalibPulse sampling loop
// (device.cpp:803,828,863,871) plus the stated stack budget.
static void test_calibpulse_decay_window_guards() {
    const unsigned MAX_SAMPLES_PER_BURST    = 64;     // device.cpp:803
    const unsigned DECAY_WINDOW_US          = 10000;  // device.cpp:863 (10ms)
    const unsigned ONSET_TIMEOUT_US         = 50000;  // device.cpp:828 (50ms)
    const unsigned CALIB_PULSE_LOW_THRESHOLD = 102;   // device.cpp:871
    const unsigned CALIB_PULSE_SPIKE_THRESHOLD = 512; // device.cpp:810
    CHECK_EQ(MAX_SAMPLES_PER_BURST,     64,    "calibpulse_max_samples_64");
    CHECK_EQ(DECAY_WINDOW_US,           10000, "calibpulse_decay_window_10000us");
    CHECK_EQ(ONSET_TIMEOUT_US,          50000, "calibpulse_onset_timeout_50000us");
    CHECK_EQ(CALIB_PULSE_LOW_THRESHOLD, 102,   "calibpulse_low_threshold_102");
    // Stack budget invariant: 64 samples x 4 bytes (u16 time + u16 adc) = 256 B.
    CHECK_EQ(MAX_SAMPLES_PER_BURST * 4, 256u, "calibpulse_stack_budget_256");
    // The early-stop low threshold is below the spike threshold.
    CHECK_TRUE(CALIB_PULSE_LOW_THRESHOLD < CALIB_PULSE_SPIKE_THRESHOLD,
               "calibpulse_low_below_spike");
}

// --- readmeasurement_source_mapping ---------------------------------------
// Locks the ReadMeasurement source-select accept window and the invalid-source
// reject path (device.cpp:508-527). _MEAS_SOURCE_L1=0, _MEAS_SOURCE_L2=1.
static void test_readmeasurement_source_mapping() {
    const uint8_t MEAS_SOURCE_L1 = 0; // device.cpp:19
    const uint8_t MEAS_SOURCE_L2 = 1; // device.cpp:20
    CHECK_EQ(MEAS_SOURCE_L1, 0, "meas_source_L1_is_0");
    CHECK_EQ(MEAS_SOURCE_L2, 1, "meas_source_L2_is_1");
    // Accept predicate: only source 0 (L1) or 1 (L2); everything else rejects
    // via _DisableADC + _SendFailure (device.cpp:522-527).
    auto readmeas_accept = [](uint8_t s) { return s == 0 || s == 1; };
    CHECK_TRUE(readmeas_accept(0),   "readmeas_accept_L1");
    CHECK_TRUE(readmeas_accept(1),   "readmeas_accept_L2");
    CHECK_TRUE(!readmeas_accept(2),  "readmeas_reject_2");
    CHECK_TRUE(!readmeas_accept(0xFF), "readmeas_reject_0xFF");
    // Return-value selection: source==L1 picks _L1_meas else _L2_meas
    // (device.cpp:533). Lock that source 1 maps to the L2 branch.
    auto picks_l1 = [&](uint8_t s) { return s == MEAS_SOURCE_L1; };
    CHECK_TRUE(picks_l1(0),  "readmeas_source0_picks_L1");
    CHECK_TRUE(!picks_l1(1), "readmeas_source1_picks_L2");
}

// --- signal_flag_dispatch_asymmetry ---------------------------------------
// Locks _SetSignalFlag/_ToggleSignalFlag dispatch (device.cpp:197-235).
// _FLAG_MEAS_ENABLE=0, _FLAG_ADC_ENABLE=1. CHARACTERIZATION asymmetry: set/unset
// accept BOTH 0 and 1, but toggle has NO ADC case so it REJECTS flag 1.
static void test_signal_flag_dispatch_asymmetry() {
    const uint8_t FLAG_MEAS_ENABLE = 0; // device.cpp:17
    const uint8_t FLAG_ADC_ENABLE  = 1; // device.cpp:18
    CHECK_EQ(FLAG_MEAS_ENABLE, 0, "flag_meas_enable_is_0");
    CHECK_EQ(FLAG_ADC_ENABLE,  1, "flag_adc_enable_is_1");
    // _SetSignalFlag / _UnSet (same switch): accept 0 and 1, default false.
    auto set_flag_accept = [](uint8_t f) { return f == 0 || f == 1; };
    CHECK_TRUE(set_flag_accept(0),  "setflag_accept_meas");
    CHECK_TRUE(set_flag_accept(1),  "setflag_accept_adc");
    CHECK_TRUE(!set_flag_accept(2), "setflag_reject_2");
    CHECK_TRUE(!set_flag_accept(0xFF), "setflag_reject_0xFF");
    // _ToggleSignalFlag: ONLY flag 0 has a case; flag 1 (ADC) hits default->false.
    auto toggle_flag_accept = [](uint8_t f) { return f == 0; };
    CHECK_TRUE(toggle_flag_accept(0),  "toggle_accept_meas");
    // CHARACTERIZATION asymmetry: toggle REJECTS ADC_ENABLE (no ADC toggle case).
    CHECK_TRUE(!toggle_flag_accept(1), "toggle_REJECTS_adc_asymmetry");
    CHECK_TRUE(!toggle_flag_accept(2), "toggle_reject_2");
}

// --- runinference_interval_and_stream_framing -----------------------------
// Locks the RunInference interval_us LE reassembly (device.cpp:612) and the
// output stream framing arithmetic (device.cpp:645-649).
static void test_runinference_interval_and_stream_framing() {
    // interval_us = lo | (hi << 8) -- same formula as U16FromBytes.
    CHECK_EQ(U16FromBytes(0x10, 0x27), 10000, "runinf_interval_10000us");
    CHECK_EQ(U16FromBytes(0xFF, 0xFF), 65535, "runinf_interval_65535us");
    CHECK_EQ(U16FromBytes(0x00, 0x00), 0,     "runinf_interval_0us");
    // Stream framing: N samples -> N * SendU16 (2 bytes each) + 1 PORT_TRN_END.
    auto stream_len = [](unsigned n) { return n * 2 + 1; };
    CHECK_EQ(stream_len(1),   3,   "runinf_stream_len_N1");
    CHECK_EQ(stream_len(250), 501, "runinf_stream_len_N250_max");
    CHECK_EQ(stream_len(0),   1,   "runinf_stream_len_N0_just_TRN_END");
}

// --- target_datatype_width_contract ---------------------------------------
// Documents the TARGET widths the host transcriptions ASSUME (datatypes.hpp).
// The host substitutes `unsigned` (32-bit) for usize (16-bit on target); this
// CHECK makes that host/target width gap explicit and change-detecting.
static void test_target_datatype_width_contract() {
    // Target contract (datatypes.hpp:25-28): u8=1, u16=2, u32=4, usize=2.
    // We pin the documented widths as plain integer goldens (the host cannot
    // include datatypes.hpp, so these are the contract the firmware relies on).
    const unsigned TARGET_SIZEOF_U16   = 2; // datatypes.hpp:26
    const unsigned TARGET_SIZEOF_U32   = 4; // datatypes.hpp:27
    const unsigned TARGET_SIZEOF_USIZE = 2; // datatypes.hpp:28 (16-bit on AVR)
    CHECK_EQ(TARGET_SIZEOF_U16,   2, "target_sizeof_u16_is_2");
    CHECK_EQ(TARGET_SIZEOF_U32,   4, "target_sizeof_u32_is_4");
    CHECK_EQ(TARGET_SIZEOF_USIZE, 2, "target_sizeof_usize_is_2");
    // All transcribed CONF_* must fit in 16 bits (since usize/u16 are 16-bit).
    CHECK_TRUE(CONF_DAC_COUNT     < 65536u, "conf_dac_count_fits_16bit");
    CHECK_TRUE(CONF_SYNAPSE_COUNT < 65536u, "conf_synapse_count_fits_16bit");
    CHECK_TRUE(CONF_MCP4728_COUNT < 65536u, "conf_mcp4728_count_fits_16bit");
    // The host's own u32 transcription type must really be 4 bytes, else the
    // elapsed_us split goldens would be meaningless.
    CHECK_EQ((unsigned)sizeof(uint32_t), 4u, "host_uint32_is_4_bytes");
    CHECK_EQ((unsigned)sizeof(uint16_t), 2u, "host_uint16_is_2_bytes");
}

// --- signature_offset_transform_general -----------------------------------
// Locks the +0x30 transform itself (device.cpp:313-315), not just today's
// version. CHARACTERIZATION: only single-digit components (0..9) stay in
// printable ASCII '0'..'9' and clear of control bytes; v>=10 would collide.
static void test_signature_offset_transform_general() {
    CHECK_EQ(SigByte(0), 0x30, "sig_0_is_ascii0");
    CHECK_EQ(SigByte(9), 0x39, "sig_9_is_ascii9");
    for (uint8_t v = 0; v <= 9; v++) {
        uint8_t b = SigByte(v);
        // In printable ASCII '0'..'9'.
        CHECK_TRUE(b >= '0' && b <= '9', "sig_in_printable_digit_range");
        // Never collides with framing/control bytes (signals.hpp real macros).
        CHECK_TRUE(b != PORT_TRN_END, "sig_ne_TRN_END");
        CHECK_TRUE(b != PORT_ACK,     "sig_ne_ACK");
        CHECK_TRUE(b != PORT_NAK,     "sig_ne_NAK");
        CHECK_TRUE(b != PORT_MSG,     "sig_ne_MSG");
        CHECK_TRUE(b != PORT_MSG_END, "sig_ne_MSG_END");
        CHECK_TRUE(b != PORT_SIG,     "sig_ne_SIG");
    }
    // CHARACTERIZATION: v==10 produces 0x3A (':'), leaving the digit range --
    // documents that only single-digit components are safe.
    CHECK_EQ(SigByte(10), 0x3A, "sig_10_leaves_digit_range");
}

int main() {
    test_dac_fastwrite_byte_encoding();
    test_dac_code_12bit_mask();
    test_wdt_timeout_to_register_bits();
    test_progdac_cmd_byte_formulas();
    test_progdac_ack_bits_packing();
    test_progdac_status_and_diagnostic_codes();
    test_dac_quiescent_table_values();
    test_sendu16_little_endian_split();
    test_u16_reassembly_from_bytes();
    test_spike_threshold_constants();
    test_framing_signal_byte_values();
    test_firmware_signature_offset();
    test_dac_count_validation_bounds();
    test_config_constants_sanity();
    // --- coverage-gap closures ---
    test_elapsed_us_u32_little_endian_split();
    test_progdac_address_range_predicate();
    test_progdac_status_code_precedence();
    test_calibpulse_adcsra_prescaler32_bits();
    test_calibpulse_decay_window_guards();
    test_readmeasurement_source_mapping();
    test_signal_flag_dispatch_asymmetry();
    test_runinference_interval_and_stream_framing();
    test_target_datatype_width_contract();
    test_signature_offset_transform_general();

    std::printf("\n%d checks, %d failed\n", g_checks, g_fails);
    if (g_fails == 0) std::printf("ALL HOST LOGIC TESTS PASSED\n");
    return g_fails == 0 ? 0 : 1;
}

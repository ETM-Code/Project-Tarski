# T1 Devboard Interface Firmware

This repository contains Arduino firmware for a serial-controlled interface board.

The firmware:
- receives 1-byte serial commands from a host application,
- controls external hardware (SR latches, synapse shift path, MCP4728 DACs),
- returns binary responses using ACK/NAK + end-of-transmission framing.

## Runtime Overview

Main control flow is in `src/main.cpp`:
1. `setup()` calls `Device::Init(1000)` and `Serial.begin(9600)`.
2. `loop()` blocks until one serial byte is available.
3. That byte is treated as a command and dispatched to a `Device::*` handler.
4. Unknown commands return `NAK` and end-of-transmission.

## Serial Link

Current serial settings:
- Baud: `9600`
- Data bits: `8`
- Parity: `None`
- Stop bits: `1`
- Flow control: `None`

Command bytes are defined in `include/signals.hpp`.

### Control Bytes

- `PORT_ACK` = `0x06`
- `PORT_NAK` = `0x15`
- `PORT_TRN_END` = `0x04`
- `PORT_SIG` = `0x05`
- `PORT_READ_OUT` = `'O'`
- `PORT_LOAD_SYN` = `'S'`
- `PORT_LOAD_DAC` = `'D'`
- `PORT_READ_MEAS` = `'M'`
- `PORT_SET_FLAG` = `'F'`
- `PORT_UNSET_FLAG` = `'U'`
- `PORT_TGL_FLAG` = `'T'`

## Protocol Rules

### Framing

For each command, host sends:
1. 1-byte command
2. optional payload bytes (depends on command)

Firmware responses are command-specific but follow these patterns:
- Success-with-data: `ACK` + data bytes + `TRN_END`
- Success-no-data: `ACK` + `TRN_END`
- Failure: `NAK` + `TRN_END`

Some commands immediately send `ACK` before reading payload. This is a handshake that confirms command receipt.

### Timeout behavior

`Device::Init(1000)` sets a 1000 ms timeout used by `AwaitData()`/`ReadU8()`.

Important detail: timeout is per byte read. If the next expected byte is not received within ~1000 ms, the command fails (`NAK`, `TRN_END`).

## Command-by-Command Spec

### 1) `PORT_SIG` (`0x05`) - Firmware signature
Host sends:
- `0x05`

Device responds:
- `ACK`
- 1 byte: `FIRM_VER_MJR + 0x30`
- 1 byte: `FIRM_VER_MNR + 0x30`
- 1 byte: `FIRM_VER_PCH + 0x30`
- `TRN_END`

With current version (`0.0.1`) this is: `0x06 0x30 0x30 0x31 0x04`.

### 2) `PORT_READ_OUT` (`'O'`) - Read accelerator output latch
Host sends:
- `'O'`

Device behavior:
- pulses/latches SR path (`PIN_PARALLEL_LD`),
- shifts in 16 bits via `PIN_MISO` clocked by `PIN_SCLK`,
- resets SR latches (`PIN_RESET_SR`).

Device responds:
- `ACK`
- 2 bytes output (`u16`, little-endian)
- `TRN_END`

### 3) `PORT_LOAD_SYN` (`'S'`) - Load synapse weights
Host sends:
- `'S'`

Device immediately responds:
- `ACK`

Host must then send:
- exactly `CONF_SYNAPSE_COUNT` bytes (`90` bytes currently)

Device behavior:
- stores bytes,
- shifts them out on `PIN_MOSI` using `PIN_SCLK`.

Final response:
- success: `ACK` + `TRN_END`
- failure (timeout while reading payload): `NAK` + `TRN_END`

### 4) `PORT_LOAD_DAC` (`'D'`) - Load DAC values over I2C
Host sends:
- `'D'`

Device immediately responds:
- `ACK`

Host then sends payload:
1. `write_count` (1 byte), valid range `1..CONF_DAC_COUNT` (`1..12` currently)
2. for each DAC entry `i` in `[0, write_count)`:
   - `lsb` (1 byte)
   - `msb` (1 byte)

Each DAC value is interpreted as:
- `raw = lsb | (msb << 8)`
- value masked to 12 bits.

I2C write behavior:
- devices are MCP4728 (`4 channels/device`),
- I2C addresses come from `CONF_MCP4728_ADDRS` in `include/config.hpp`,
- values are assigned sequentially across devices/channels,
- transmission failure (`Wire.endTransmission() != 0`) returns failure.

Final response:
- success: `ACK` + `TRN_END`
- failure: `NAK` + `TRN_END`

### 5) `PORT_READ_MEAS` (`'M'`) - Read measurement channel
Host sends:
- `'M'`

Device immediately responds:
- `ACK`

Host then sends:
- `measurement_source` (1 byte):
  - `0` => L1
  - `1` => L2

Device behavior:
- enables selected measurement path,
- reads analog input (`PIN_L1_MEAS_OUT` or `PIN_L2_MEAS_OUT`),
- disables both measurement enable pins afterward.

Final response:
- success: 2-byte measurement (`u16`, little-endian) + `TRN_END`
- failure (bad source or timeout): `NAK` + `TRN_END`

### 6) Flag Commands (`'F'`, `'U'`, `'T'`)
Common payload:
- `flag` (1 byte)

Current implemented flag IDs:
- `0` => measurement enable signal (`PIN_L1_EN_MEAS` and `PIN_L2_EN_MEAS` together)

Behavior:
- `'F'` (`PORT_SET_FLAG`): set signal high
- `'U'` (`PORT_UNSET_FLAG`): set signal low
- `'T'` (`PORT_TGL_FLAG`): toggle signal

Response pattern:
1. device sends immediate `ACK` after command byte,
2. host sends `flag` byte,
3. device sends final result:
   - success: `ACK` + `TRN_END`
   - invalid flag / timeout: `NAK` + `TRN_END`

## Host Application Implementation Guide

A host application should implement a strict request/response state machine.

### Recommended host flow per command
1. Open serial at `9600 8N1`.
2. Send command byte.
3. If command expects immediate ACK (`S`, `D`, `M`, `F`, `U`, `T`), wait for it.
4. Send payload bytes (if any).
5. Read and validate final response:
   - parse data bytes where applicable,
   - require trailing `TRN_END`.
6. On timeout or malformed framing, treat transaction as failed and resync by scanning for `TRN_END`.

### Minimal packet parser requirements
- Byte-level reads with timeouts.
- Verify `ACK`/`NAK` exactly where expected.
- Verify `TRN_END` terminator.
- Decode little-endian `u16` values.

### Python-style pseudocode
```python
# Pseudocode, not drop-in code
send_byte(PORT_LOAD_DAC)
expect_byte(PORT_ACK)

write_count = 4
send_byte(write_count)
for value in [100, 200, 300, 400]:
    send_byte(value & 0xFF)         # lsb
    send_byte((value >> 8) & 0xFF)  # msb

status = expect_byte_any([PORT_ACK, PORT_NAK])
expect_byte(PORT_TRN_END)
if status == PORT_NAK:
    raise RuntimeError("DAC write failed")
```

## Hardware/Pin Mapping

Pin constants are in `include/pinout.hpp`.

Main active groups:
- SR latch output read path: `PIN_PARALLEL_LD`, `PIN_MISO`, `PIN_SCLK`, `PIN_RESET_SR`
- Synapse shift output path: `PIN_OE_S`, `PIN_SRCLR_S`, `PIN_MOSI`, `PIN_SCLK`
- Measurement select/read: `PIN_L1_EN_MEAS`, `PIN_L2_EN_MEAS`, `PIN_L1_MEAS_OUT`, `PIN_L2_MEAS_OUT`
- I2C DAC bus: `PIN_I2C_SDA`, `PIN_I2C_SCL`

## Configuration

All interface constants live in `include/config.hpp`:
- `CONF_SYNAPSE_COUNT`
- `CONF_DAC_COUNT`, `CONF_DAC_BITS`
- `CONF_MCP4728_COUNT`
- `CONF_MCP4728_ADDRS[]`

If DAC addresses are changed by your external tooling, update `CONF_MCP4728_ADDRS` to match.

## Notes

- LED-related command/circuit code has been removed.
- `Device::AwaitResponse()` exists but is not currently used by command handlers.
- There is an internal variable `_command_impl_approved` currently not used by command handlers.

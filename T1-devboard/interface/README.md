# T1 Devboard Interface Firmware

Arduino firmware for the T1 serial-controlled interface board used to load weights, drive DACs, and read board outputs.

## Core Behavior

- Receives one-byte commands over UART
- Controls shift-register paths, measurement routing, and MCP4728 DAC writes
- Returns framed responses using `ACK`/`NAK` plus `TRN_END`

## Runtime

Main control flow is implemented in `src/main.cpp` with command dispatch through `Device::*` handlers.

## Serial Settings

- Baud: `115200`
- Data bits: `8`
- Parity: `None`
- Stop bits: `1`

Command identifiers and control bytes are defined in `include/signals.hpp`.

## Implemented Command Groups

- Signature/version query
- Output latch readback
- Synapse-weight payload load
- DAC write payload load
- Measurement-channel read
- Flag set/unset/toggle commands

## Protocol Summary

Command and control constants are defined in `include/signals.hpp`.

- `PORT_ACK` (`0x06`)
- `PORT_NAK` (`0x15`)
- `PORT_TRN_END` (`0x04`)
- `PORT_SIG` (`0x05`)
- `PORT_READ_OUT` (`'O'`)
- `PORT_LOAD_SYN` (`'S'`)
- `PORT_LOAD_DAC` (`'D'`)
- `PORT_READ_MEAS` (`'M'`)
- `PORT_SET_FLAG` (`'F'`)
- `PORT_UNSET_FLAG` (`'U'`)
- `PORT_TGL_FLAG` (`'T'`)

## Response Framing

Responses follow one of:

- Success with data: `ACK` + data + `TRN_END`
- Success without data: `ACK` + `TRN_END`
- Failure: `NAK` + `TRN_END`

Some commands emit an immediate `ACK` before receiving payload bytes.

## Timeouts

Timeout behavior is configured through `Device::Init(Device::Timeout::<preset>)` and applied per expected byte.

Supported timeout presets:

- `TO16MS`
- `TO32MS`
- `TO64MS`
- `TO125MS`
- `TO250MS`
- `TO500MS`
- `TO1S`
- `TO2S`
- `TO4S`
- `TO8S`

## Configuration and Pinout

- `include/config.hpp`: interface constants such as synapse count, DAC count, and MCP4728 addresses
- `include/pinout.hpp`: pin mapping for SR paths, measurement routing, and I2C lines

## Host Integration Note

Host software should implement a strict byte-level state machine that validates command-specific framing and always requires the terminating `TRN_END`.

Recommended host flow:

1. Open serial as `115200 8N1`.
2. Send one-byte command.
3. For payload commands, wait for immediate `ACK` before payload.
4. Send payload bytes with per-byte timeout handling.
5. Require a final status (`ACK`/`NAK`) and `TRN_END`.
6. On malformed framing, resynchronize on `TRN_END`.

For command-level behavior, use `include/signals.hpp` and handler implementations in `src/` as the canonical reference.

# Characterization (golden-master) tests — firmware-cpp (T1-devboard/interface)

These tests LOCK IN the current behavior of the AVR firmware so future changes
that alter behavior fail loudly. They never modify production source.

## Run everything

From `T1-devboard/interface/`:

```bash
uv run --with pytest pytest test/test_build_size/run_build_smoke.py -v
```

That single command runs both tracks:

| Track | File | What it locks |
|-------|------|---------------|
| Build smoke | `test/test_build_size/run_build_smoke.py` | `pio run` exits 0; Flash=7992±16 B; RAM=526±16 B (avr-size text+data / data+bss) |
| Host pure-logic | `test/host/test_device_logic.cpp` (driven by the pytest file) | 1312 assertions over the firmware's pure bit/encoding logic |

`pytest` reports **4 test functions** (3 build-smoke + 1 host-logic driver). The
host-logic driver internally runs **1312** C++ assertions; the two counts are
distinct and must not be conflated (4 pytest units vs 1312 host checks).

The host test can also be run directly:

```bash
clang++ -std=c++14 -Wall -Iinclude test/host/test_device_logic.cpp -o /tmp/t && /tmp/t
```

## Golden values (observed, not guessed)

`pio run` on current code reports RAM 526 / Flash 7992. `avr-size` on
`.pio/build/nanoatmega328new/firmware.elf`:

```
text   data   bss    dec    hex
7952   40     486    8478   211e
```

Flash = text+data = 7992. RAM = data+bss = 526.

## What is covered (host pure-logic)

- MCP4728 fast-write byte split + 12-bit truncation (`_WriteDACData`)
- 12-bit DAC code mask (LoadDACs/CalibL1/CalibPulse)
- `_WDTFromTimeout` register-bit table incl. default branch
- `ProgramDACAddress` cmd1/cmd2/cmd3/addr_byte math (the LDAC-on-cmd1 off-by-one fix context)
- ack-bit packing diagnostic byte
- status/diagnostic codes 0x12/0x13/0x14/0xA1/0xA2/0xA3 + the "must not equal PORT_TRN_END" framing invariant
- DAC quiescent table (the DAC-quiescing patch): ch0..8=4095, ch9..11=0
- `SendU16` little-endian split + host->device u16 reassembly (round-trip)
- spike thresholds: CalibL1=200 (lowered from 512), CalibPulse=512, low=102
- all signals.hpp framing/command bytes (real macros, distinctness invariants)
- SendSignature version-byte +0x30 offset and current version 0/1/0
- accept/reject bounds for LoadDACs / RunInference / CalibL1 / CalibPulse
- CalibL1/CalibPulse `elapsed_us` u32 4-byte little-endian split + the
  `0xFFFFFFFF` timeout sentinel (device.cpp:708,742-745) + round-trip
- `ProgramDACAddress` address-range predicate (accept 0x60..0x67, reject path)
- `ProgramDACAddress` status-code PRECEDENCE: `still_at_old ? 0x12 : 0x13` then
  any false ack overrides to 0x14 (ack-failure wins)
- CalibPulse ADCSRA prescaler-32 bit math `(ADCSRA & 0xF8)|_BV(ADPS2)|_BV(ADPS0)`
- CalibPulse loop bounds: MAX_SAMPLES_PER_BURST=64, DECAY_WINDOW=10000us,
  ONSET_TIMEOUT=50000us, low=102, 256-byte stack budget
- ReadMeasurement source mapping (L1=0/L2=1) + invalid-source (>=2) reject
- `_SetSignalFlag`/`_ToggleSignalFlag` dispatch asymmetry (set/unset accept
  {0,1}; toggle accepts only 0, REJECTS ADC_ENABLE=1)
- RunInference interval_us LE reassembly + N*2+1 stream framing arithmetic
- target datatype width contract (u16=2, u32=4, usize=2) the host transcriptions
  assume, and that CONF_* fit in 16 bits
- SendSignature +0x30 transform generalized over 0..9 (printable, control-safe;
  CHARACTERIZATION: v>=10 collides out of the digit range)

## What is NOT covered (honest limitations)

`device.cpp` is hardware-bound and cannot be host-compiled, so the following are
only exercised at the compile/link/size level by the build-smoke test, NOT
behaviorally:

- Serial I/O framing sequences, WDT sleep/wake timing, ADC polling/prescaler
- I2C bit-bang LDAC timing in `ProgramDACAddress`, GPIO/shift-register sequencing
- The actual analog spike detection and decay sampling

`config.hpp`/`datatypes.hpp` cannot be `#include`d on a 64-bit host because
`datatypes.hpp` asserts `sizeof(u32)==4` (u32 = unsigned long = 8 B on LP64), so
their constants are transcribed in the host test with cross-reference comments.

CHARACTERIZATION: where current behavior looks like a quirk (e.g. 12-bit mask
silently dropping bit 12), it is locked anyway to detect change.

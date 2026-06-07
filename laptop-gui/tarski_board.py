#!/usr/bin/env python3
"""Tarski board interface — connects to the real hardware via serial.

Usage:
    python3 tarski_board.py --port /dev/tty.usbserial-XXX
    python3 tarski_board.py --port /dev/tty.usbserial-XXX --calibrate
    python3 tarski_board.py --port /dev/tty.usbserial-XXX --infer --checkpoint ../gilgamesh/models/my_model.json
"""

import argparse
import json
import sys
import time
from itertools import combinations

import numpy as np

try:
    import serial
except ImportError:
    print("Install pyserial: pip install pyserial")
    sys.exit(1)

# Serial protocol constants (must match T1-devboard signals.hpp)
PORT_TRN_END = 0x04
PORT_SIG     = 0x05
PORT_ACK     = 0x06
PORT_NAK     = 0x15
PORT_READ_OUT  = ord('O')
PORT_LOAD_SYN  = ord('S')
PORT_LOAD_DAC  = ord('D')
PORT_READ_MEAS = ord('M')
PORT_SET_FLAG  = ord('F')
PORT_UNSET_FLAG = ord('U')

# Firmware signal flags (match device.cpp _FLAG_*).
FLAG_MEAS_ENABLE = 0
FLAG_ADC_ENABLE = 1
FLAG_RESET_SR_ACTIVE_LOW = 2  # diagnostic: flip RESET_SR polarity at runtime
PORT_PROG_DAC  = ord('P')
PORT_SCAN_I2C  = ord('I')
PORT_MEAS_SPKS = ord('K')
PORT_CH10_BURST = ord('H')

# Hardware constants
V_DD = 5.0
V_BE = 0.65
R_SET_INPUT = 174e3
R_LEAK = 120e3
R_TOP = 820e3
R_BOTTOM = 150e3

THETA_0 = ((V_DD / R_TOP + 2.5 / R_BOTTOM) / (1/R_TOP + 1/R_BOTTOM)) - 2.5

# DAC configuration
# MCP4728s on this board are configured with internal VREF = 2.048 V and
# gain = 2, so the full-scale output is ~4.096 V (NOT the 5.0 V of the
# factory default with VDD reference). `code = 4095` ≈ 4.0 V,
# `code = 2048` ≈ 2.0 V. The firmware issues Fast Write commands, which
# don't touch the VREF/gain bits — those were set once in EEPROM and the
# firmware is expected to match whatever's there.
DAC_VREF = 4.096
DAC_BITS = 12
DAC_MAX_CODE = (1 << DAC_BITS) - 1
NUM_DACS = 9        # 9 hidden neurons
NUM_SYNAPSES = 90   # 9 hidden × 10 output

# Quiescent SR byte: bits 5,6 HIGH (inhibitory NPN collectors routed to
# VCC, disengaged), bits 1..3 LOW (excitatory PNP collectors routed to
# GND, disengaged). This is the "no synapse active" state. Loading
# literal 0x00 instead leaves the entire inhibitory stage engaged —
# see `weight_to_sr_byte` for the polarity derivation.
SR_IDLE_BYTE = 0x70

# Synapse SR byte → physical (output_neuron, hidden_source) mapping.
#
# Traced from testernetter.net: the 90 74HC595 shift registers are grouped
# in the hierarchy as /Neuron_Layer1/Synapse Layer/Synapses NeuronN/SynapseM
# with N=1..10 (OUTPUT neuron index) and M=1..9 (hidden-source index). The
# daisy chain walks N1S1→N1S2→…→N1S9→N2S1→…→N10S9 with MOSI entering at
# IC3902 (N1S1) and exiting at IC13802 (N10S9). So chip position
#     k = 9*(N-1) + (M-1)   (0-indexed)
# holds the byte intended for output neuron N-1, hidden source M-1.
#
# The firmware calls `shiftOut(..., MSBFIRST, weights[i])` in a loop over
# i=0..89. Because daisy-chain data propagates, the FIRST byte sent ends
# up in the LAST chip — so chip k gets byte index (89 - k). Combined:
#     byte_index(output, hidden) = 89 - (9*output + hidden)
# with output ∈ 0..9 and hidden ∈ 0..8.
#
# This replaces the earlier (wrong) `hidden*10 + output` assumption, which
# treated the layout as 9 hiddens × 10 outputs hidden-major with no chain
# reversal. Any code that builds an sr_byte array at a specific (output,
# hidden) slot MUST go through `phys_byte`.

def phys_byte(output: int, hidden: int) -> int:
    """Return the sr_bytes index that physically lands on the synapse
    from hidden neuron `hidden` (0..8) to output neuron `output` (0..9)."""
    assert 0 <= output < 10, f"output must be 0..9, got {output}"
    assert 0 <= hidden < 9, f"hidden must be 0..8, got {hidden}"
    return 89 - (9 * output + hidden)


def byte_to_physical(byte_index: int) -> tuple[int, int]:
    """Inverse of phys_byte: (output, hidden) for a given sr_bytes index."""
    assert 0 <= byte_index < NUM_SYNAPSES
    chip = 89 - byte_index
    return (chip // 9, chip % 9)

# DAC channel roles (12 channels total across the three MCP4728s).
# Channels 0..8  drive the 9 hidden-neuron PNP current mirrors. Because
#                the mirror emitter is on +5 V and the DAC drives the
#                base through R_set, LOW V_DAC = HIGH input current.
#                Quiescent ("no stim") is therefore DAC_MAX_CODE (~V_DD).
# Channel  9    = DAC#3 VOUTB → V_SET2_1 (reserved / unused on this
#                board). Idle at 0 V.
# Channel 10    = DAC#3 VOUTC → V_OUT3 synapse rail (drives the anodes of
#                D1806..D1809 which force Neuron6..Neuron9's V_out when
#                HIGH). MUST be 0 V when idle — otherwise hidden neurons
#                6..9 are permanently spike-forced and every measurement
#                baseline is saturated.
# Channel 11    = DAC#3 VOUTD (no-connect per netlist). Idle at 0 V.
DAC_ROLE_PNP_INVERTED = "pnp_inverted"
DAC_ROLE_DIRECT       = "direct"
DAC_CHANNEL_ROLES = (
    [DAC_ROLE_PNP_INVERTED] * 9       # ch 0..8: hidden-neuron current mirrors
    + [DAC_ROLE_DIRECT]                # ch 9:   V_SET2_1 (unused, 0 V)
    + [DAC_ROLE_DIRECT]                # ch 10:  synapse V_OUT3
    + [DAC_ROLE_DIRECT]                # ch 11:  no-connect
)
DAC_QUIESCENT_CODES = [
    DAC_MAX_CODE if role == DAC_ROLE_PNP_INVERTED else 0
    for role in DAC_CHANNEL_ROLES
]

# Mapping from "output neuron index" (0..9) to bit position inside the
# 16-bit word read back from the 74HC165 spike-latch chain.
#
# Per the netlist (testernetter.net):
#   U15001 (upstream 74HC165) has D0..D5 unconnected (floating), D6=L1,
#   D7=L2; its Q7 feeds U15002.DS.
#   U15002 (downstream, nearest MCU) has D0=L3, D1=L4, D2=L5, D3=L6,
#   D4=L7, D5=L8, D6=L9, D7=L10; its Q7 goes to MISO.
#
# 74HC165 shifts MSB first on every clock, so the serial sequence is:
#   U15002.D7 (L10), D6 (L9), D5 (L8), D4 (L7), D3 (L6),
#   D2 (L5),  D1 (L4), D0 (L3),
#   U15001.D7 (L2),  D6 (L1),
#   D5 (NC), D4 (NC), D3 (NC), D2 (NC), D1 (NC), D0 (NC)
#
# The firmware's `_ReadShiftRegisterWord` uses `value <<= 1; value |= bit;`
# so the FIRST bit read lands in bit 15 of the 16-bit word. That gives
# the final word-bit → latch mapping below.
# Outputs are numbered 0..9 (O0..O9 in the host API), but the hardware
# labels them L1..L10 — we subtract 1 to align.
OUTPUT_BIT_MAP = {
    0: 6,   # O0 → L1  → word bit 6
    1: 7,   # O1 → L2  → word bit 7
    2: 8,   # O2 → L3  → word bit 8
    3: 9,   # O3 → L4  → word bit 9
    4: 10,  # O4 → L5  → word bit 10
    5: 11,  # O5 → L6  → word bit 11
    6: 12,  # O6 → L7  → word bit 12
    7: 13,  # O7 → L8  → word bit 13
    8: 14,  # O8 → L9  → word bit 14
    9: 15,  # O9 → L10 → word bit 15
}

def extract_output_spikes(word: int) -> list[int]:
    """Decode a raw 16-bit spike word into a 10-element 0/1 list of
    output-neuron spike flags, using the netlist-derived bit mapping.

    POLARITY: the spike latches are cross-coupled 74HC02 NOR SR latches
    (U15101..U15105). At reset their output reads HIGH; a spike event
    SETS them LOW. So a read of `HIGH` means "latch is in the idle
    (reset) state — no spike captured", and `LOW` means "a spike was
    captured since the last reset". We invert here so callers get the
    intuitive `1 = spiked`, `0 = idle`.

    Bits 0..5 of the raw word are floating NC pins (U15001 D0..D5 are
    unconnected per the netlist) and are ignored.
    """
    return [1 - ((word >> OUTPUT_BIT_MAP[i]) & 1) for i in range(10)]


# Channel 10 is the synapse V_OUT3 rail. Driving it HIGH (DAC_MAX_CODE)
# simulates every hidden neuron firing at once — see sample_with_synapse_pulse.
CH10_CHANNEL = 10


def ch10_drive_codes(code: int = DAC_MAX_CODE) -> list[int]:
    """Quiescent DAC codes with channel 10 (the synapse V_OUT3 rail) asserted
    at `code`. This is the standard "drive the synapse rail" stimulus pattern."""
    codes = list(DAC_QUIESCENT_CODES)
    codes[CH10_CHANNEL] = code
    return codes


def fire_glyphs(flags) -> str:
    """Render a sequence of truthy/falsy fire flags as ●/· glyphs."""
    return ''.join('●' if x else '·' for x in flags)


def adc_to_volts(raw: int) -> float:
    """Convert a raw 10-bit ADC reading to volts against the 5 V reference."""
    return raw * 5.0 / 1023


def count_observable_fires(snapshots, output: int) -> int:
    """Count snapshots in which `output`'s latch was captured in the SPIKED
    state, decoding each raw spike word via extract_output_spikes."""
    return sum(1 for w in snapshots if extract_output_spikes(w)[output])


def accumulate_spike_counts(snapshots) -> list[int]:
    """Sum per-output spike flags across `snapshots` into a 10-element list."""
    counts = [0] * 10
    for word in snapshots:
        bits = extract_output_spikes(word)
        for i in range(10):
            counts[i] += bits[i]
    return counts


BAUD_RATE = 115200  # Must match firmware Serial.begin(115200)


class TarskiBoard:
    """Interface to the Tarski neuromorphic board via serial."""

    def __init__(self, port: str, timeout: float = 2.0, verbose: bool = False):
        self.ser = serial.Serial(port, BAUD_RATE, timeout=timeout)
        time.sleep(2)  # Wait for Arduino reset
        self.ser.reset_input_buffer()
        self.verbose = verbose

    def close(self):
        self.ser.close()

    # ── Low-level protocol ──

    def _send(self, data: bytes):
        if self.verbose:
            print(f"  TX[{len(data)}]: {data.hex(' ')}")
        self.ser.write(data)

    def _read(self, n: int) -> bytes:
        data = self.ser.read(n)
        if self.verbose:
            short = " (short!)" if len(data) < n else ""
            print(f"  RX[{len(data)}/{n}]: {data.hex(' ')}{short}")
        if len(data) < n:
            raise TimeoutError(f"Expected {n} bytes, got {len(data)}")
        return data

    def _expect_final_ack(self, op: str):
        """Consume the post-payload response from the firmware and raise if
        it's a NAK. Command handlers that end in _SendSuccess() emit
        [ACK, TRN_END], _SendFailure() emits [NAK, TRN_END]. Earlier versions
        of this driver discarded the response shape entirely via
        `_read_until_trn_end()` and silently treated every NAK as success —
        that's how the broken --verify-dacs check slipped through."""
        resp = self._read_until_trn_end()
        if not resp:
            raise RuntimeError(f"{op}: empty response (expected ACK or NAK)")
        if resp[0] == PORT_NAK:
            tail = f" [{resp.hex(' ')}]" if len(resp) > 1 else ""
            raise RuntimeError(f"{op}: NAK'd by firmware{tail}")
        if resp[0] != PORT_ACK:
            raise RuntimeError(f"{op}: unexpected response {resp.hex(' ')}")

    def _expect_ack(self) -> bool:
        resp = self._read(1)
        return resp[0] == PORT_ACK

    def _send_cmd(self, opcode: int, nak_msg: str):
        """Write a command opcode byte and require the firmware's command-byte
        ACK, raising RuntimeError(`nak_msg`) if it NAKs. This is the preamble
        shared by nearly every command handler."""
        self._send(bytes([opcode]))
        if not self._expect_ack():
            raise RuntimeError(nak_msg)

    def _read_until_trn_end(self) -> bytes:
        """Read bytes until PORT_TRN_END."""
        buf = bytearray()
        while True:
            b = self._read(1)
            if b[0] == PORT_TRN_END:
                return bytes(buf)
            buf.append(b[0])

    # ── Commands ──

    def get_signature(self) -> str:
        """Request firmware version. Returns version string like '0.1.0'."""
        self._send_cmd(PORT_SIG, "Signature request NAK'd")
        data = self._read_until_trn_end()
        # Version bytes are shifted +0x30 to avoid control char collisions
        return '.'.join(str(b - 0x30) for b in data)

    def load_weights(self, sr_bytes: list[int]):
        """Load 90 synapse weight bytes into the shift register chain."""
        assert len(sr_bytes) == NUM_SYNAPSES, f"Expected {NUM_SYNAPSES} bytes, got {len(sr_bytes)}"
        self._send_cmd(PORT_LOAD_SYN, "Weight load NAK'd")
        self._send(bytes(sr_bytes))
        # Consume the firmware's ACK + TRN_END response off the wire.
        self._read_until_trn_end()
        return True

    def sample_with_synapse_pulse(self, weights: list[int],
                                  num_samples: int = 50,
                                  interval_us: int = 500,
                                  ch10_code: int = None) -> list[int]:
        """Drive DAC channel 10 HIGH and sample output spike latches.

        Channel 10 is wired directly into the synapse "V_out" bus: while
        it is held HIGH, every synapse whose shift-register bit is set
        sources current into its destination output membrane, exactly
        as a real hidden spike would. This bypasses the hidden-neuron
        layer entirely, so per-synapse calibration is independent of
        whether individual hidden neurons are actually firing (which is
        hard to observe through the 9-way wired-OR MEAS_OUT bus anyway).

        `weights` is the full 90-byte shift-register pattern; pass a
        single-synapse pattern to probe one connection, or all-+7 for
        the smoke test. `ch10_code` defaults to DAC_MAX_CODE (~4.1 V).
        """
        assert len(weights) == NUM_SYNAPSES
        if ch10_code is None:
            ch10_code = DAC_MAX_CODE
        self.load_weights(weights)
        self.load_dacs(ch10_drive_codes(ch10_code))
        time.sleep(0.001)  # DAC settle
        snapshots = self.run_inference(num_samples, interval_us)
        # Release ch10 and leave the board quiescent.
        self.load_dacs(list(DAC_QUIESCENT_CODES))
        return snapshots

    def quiesce(self, zero_weights: bool = True,
                reset_output_latches: bool = True) -> None:
        """Drive the board into a fully-idle state.

        Everything that can hold state between operations gets cleared:

        1. **DACs** → `DAC_QUIESCENT_CODES` (PNP mirrors off on ch0..8,
           ch10/ch11 at 0 V). Without this, ch10 retains whatever value
           the prior phase left it at — often ~4 V — which keeps the
           synapse V_out rail asserted and produces ghost drive.
        2. **74HC595 weight shift registers** → quiescent 0x70 per byte
           (if `zero_weights`), which corresponds to `weight_to_sr_byte(0)`.
           This is NOT `0x00`: bits 5, 6 must be HIGH to route the ×2
           and ×4 inhibitory NPN collectors to VCC instead of the
           inhibitory summing bus. Loading `0x00` leaves every synapse's
           inhibitory stage fully engaged, which burns ~235 µA through
           the NPN mirrors continuously and pulls every output membrane
           down during hidden spikes. Residual programmed synapse bits
           leave the analog-switch matrix in a non-deterministic state
           that can route spike energy in unexpected ways between phases.
        3. **74HC02 output SR latches** → cleared (if
           `reset_output_latches`). We piggy-back on `read_output()`
           whose firmware handler ends with `_PulseResetSR()`, so the
           act of reading the word also clears it. Without this step,
           stale spike captures from a prior run can masquerade as
           fresh fires in the next one.

        The 74HC165 parallel-in side doesn't need clearing — it's
        read-only and re-samples on every parallel-load pulse.
        """
        self.load_dacs(list(DAC_QUIESCENT_CODES))
        if zero_weights:
            quiescent = self.weight_to_sr_byte(0)
            self.load_weights([quiescent] * NUM_SYNAPSES)
        if reset_output_latches:
            # The firmware's ReadOutput handler ends with _PulseResetSR(),
            # so just reading the output word clears all 10 latches as a
            # side effect. Discard the returned value.
            self.read_output()

    def scan_i2c(self) -> list[int]:
        """Scan MCP4728 address range 0x60..0x67 and return the list of
        addresses that ACKed. Uses PORT_SCAN_I2C on the firmware."""
        self._send_cmd(PORT_SCAN_I2C, "Scan I2C NAK'd on command byte")
        # Response shape: [ACK, bitmap_byte, TRN_END]
        resp = self._read_until_trn_end()
        if len(resp) < 2 or resp[0] != PORT_ACK:
            raise RuntimeError(f"scan_i2c: unexpected response {resp.hex(' ')}")
        bitmap = resp[1]
        return [0x60 + i for i in range(8) if (bitmap >> i) & 1]

    def load_dacs(self, codes: list[int]):
        """Load DAC codes (12-bit values) for hidden neuron input currents.

        Raises RuntimeError if the firmware reports an I2C failure (missing
        DAC, wrong address, bus NAK, etc). The count determines which
        MCP4728 chips get written — 1..4 = DAC #1 only, 5..8 = DAC #1+#2,
        9..12 = all three.
        """
        n = len(codes)
        assert 1 <= n <= 12, f"DAC count must be 1-12, got {n}"
        self._send_cmd(PORT_LOAD_DAC, "DAC load NAK'd on command byte")
        self._send(bytes([n]))
        for code in codes:
            code = max(0, min(DAC_MAX_CODE, code))
            self._send(bytes([code & 0xFF, (code >> 8) & 0xFF]))
        self._expect_final_ack(f"load_dacs({n})")
        return True

    def read_output(self) -> int:
        """Read the 16-bit spike output word from the 74HC165 latches."""
        self._send_cmd(PORT_READ_OUT, "Read output NAK'd")
        data = self._read(2)  # LSB, MSB
        trn = self._read(1)   # TRN_END
        return data[0] | (data[1] << 8)

    def read_measurement(self, channel: int) -> int:
        """Read ADC measurement. channel: 0=L1, 1=L2."""
        self._send_cmd(PORT_READ_MEAS, "Measurement NAK'd")
        self._send(bytes([channel]))
        data = self._read(2)
        trn = self._read(1)
        return data[0] | (data[1] << 8)

    def set_flag(self, flag: int):
        self._send_cmd(PORT_SET_FLAG, "Set flag NAK'd on command byte")
        self._send(bytes([flag]))
        self._expect_final_ack(f"set_flag({flag})")

    def unset_flag(self, flag: int):
        self._send_cmd(PORT_UNSET_FLAG, "Unset flag NAK'd on command byte")
        self._send(bytes([flag]))
        self._expect_final_ack(f"unset_flag({flag})")

    def program_dac_address(self, old_addr: int, new_addr: int,
                            max_retries: int = 3) -> tuple[bool, str]:
        """Program a MCP4728 DAC I2C address.

        All MCP4728s ship with factory address 0x60. To use three on the
        same bus, each must be programmed to a unique address.

        Procedure:
            1. Disconnect all DACs except the target via jumpers
            2. Call program_dac_address(0x60, target_addr)
            3. Reconnect and repeat for each DAC

        Returns (success, message). The firmware retries internally and reports
        detailed status if it fails:
            0x01 = verified at new address (success)
            0x02 = failed, device still at old address
            0x03 = failed, device not responding at either address
            0x04 = I2C framing issue (ACK missing)
        """
        assert 0x60 <= old_addr <= 0x67, f"Invalid old address: 0x{old_addr:02X}"
        assert 0x60 <= new_addr <= 0x67, f"Invalid new address: 0x{new_addr:02X}"

        # Status codes match firmware ProgramDACAddress; intentionally avoid
        # 0x04 because it collides with PORT_TRN_END in the stream parser.
        STATUS_MSGS = {
            0x12: "Device still at old address — LDAC timing may need adjustment",
            0x13: "Device not responding at either address — check connections",
            0x14: "Partial ACK failure during bit-bang (I2C framing)",
        }

        last_resp: bytes = b""
        for attempt in range(max_retries):
            # Drop any leftover bytes from a previous aborted transaction so
            # a stale TRN_END doesn't get read as this attempt's response.
            self.ser.reset_input_buffer()

            self._send(bytes([PORT_PROG_DAC]))
            if not self._expect_ack():
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
                return False, "Command not acknowledged"

            # Send the two address bytes as separate writes with a brief
            # gap. Sending them as a single 2-byte write races with the
            # firmware's AwaitData sleep loop in ProgramDACAddress.
            self._send(bytes([old_addr]))
            time.sleep(0.005)
            self._send(bytes([new_addr]))
            try:
                resp = self._read_until_trn_end()
            except TimeoutError as e:
                last_resp = b""
                if attempt < max_retries - 1:
                    time.sleep(1.0)
                    continue
                return False, f"Timeout waiting for response: {e}"
            last_resp = resp

            if len(resp) >= 1 and resp[0] == PORT_ACK:
                return True, f"Programmed 0x{old_addr:02X} → 0x{new_addr:02X}"

            # NAK with status byte (and optional ack_bits diagnostic byte)
            if len(resp) >= 2 and resp[0] == PORT_NAK:
                status = resp[1]
                msg = STATUS_MSGS.get(status, f"Unknown status 0x{status:02X}")
                if len(resp) >= 3:
                    ack_bits = resp[2]
                    labels = ['addr', 'cmd1', 'cmd2_ldac', 'cmd3']
                    detail = ', '.join(
                        f"{labels[i]}={'OK' if (ack_bits >> i) & 1 else 'NAK'}"
                        for i in range(4)
                    )
                    msg = f"{msg} [ack_bits=0b{ack_bits:04b}: {detail}]"
                if attempt < max_retries - 1:
                    time.sleep(1.0)  # Wait before retry
                    continue
                return False, f"Failed after {max_retries} attempts: {msg}"

            # Unexpected response shape — fall through to retry
            if attempt < max_retries - 1:
                time.sleep(1.0)
                continue

        hex_resp = last_resp.hex() if last_resp else "<empty>"
        return False, (
            f"Max retries exceeded. Last response from firmware: {hex_resp} "
            f"(expected ACK+TRN_END or NAK+status+TRN_END)"
        )

    def run_inference(self, num_samples: int = 25, interval_us: int = 1000) -> list[int]:
        """Run inference with rapid spike sampling.

        DAC values and weights must already be loaded.
        Returns a list of num_samples 16-bit spike words, one per sampling interval.
        Each bit in a word indicates whether that output neuron spiked since the last sample.

        To get spike counts: sum the bits across all samples.
        """
        assert 1 <= num_samples <= 250
        self._send_cmd(ord('R'), "RunInference NAK'd")  # PORT_RUN_INF
        self._send(bytes([
            num_samples,
            interval_us & 0xFF,
            (interval_us >> 8) & 0xFF,
        ]))
        # Read num_samples × 2 bytes + TRN_END
        samples = []
        for _ in range(num_samples):
            data = self._read(2)
            samples.append(data[0] | (data[1] << 8))
        self._read(1)  # TRN_END
        return samples

    def measure_spikes(self, target_channel: int, dac_code: int,
                       window_ms: int = 50, meas_source: int = 0,
                       drive_ch10: bool = False) -> dict:
        """Differential wired-OR sampling via the firmware.

        Arguments:
            target_channel: DAC channel to stim during the driven window (0..11)
            dac_code:       u12 value to write to target_channel
            window_ms:      duration of each (baseline, driven) window
            meas_source:    0 = L1 wired-OR  (PIN_L1_MEAS_OUT, hidden V_outs)
                            1 = L2 wired-OR  (PIN_L2_MEAS_OUT, output V_outs)
            drive_ch10:     if True, also assert ch10 HIGH during the DRIVEN
                            window. Used for L2 measurements where we want
                            the synapse rail lifted to simulate all hidden
                            neurons firing. Uses bit 0 of the firmware's
                            driven_mask parameter.

        Returns a dict with:
            baseline_count, baseline_min, baseline_max,
            driven_count,   driven_min,   driven_max
        where counts are samples above a fixed ADC threshold (~0.98 V)
        and min/max are raw ADC extrema over the window.

        Requires firmware v0.1.14 or later. The protocol takes 7
        parameter bytes (was 5 in v0.1.13): target, code_lo, code_hi,
        win_lo, win_hi, meas_source, driven_mask.

        The response is read as a FIXED-LENGTH frame (ACK + 6×u16 + TRN_END
        = 14 bytes), not terminated by TRN_END in the stream, because
        the u16 payload bytes can legitimately contain 0x04 = PORT_TRN_END
        and the old parser would truncate mid-packet.
        """
        assert 0 <= target_channel < 12
        assert 0 <= dac_code <= DAC_MAX_CODE
        assert 1 <= window_ms <= 1000
        assert meas_source in (0, 1)
        driven_mask = 0x01 if drive_ch10 else 0x00
        self._send_cmd(PORT_MEAS_SPKS, "measure_spikes NAK'd on command byte")
        self._send(bytes([
            target_channel,
            dac_code & 0xFF, (dac_code >> 8) & 0xFF,
            window_ms & 0xFF, (window_ms >> 8) & 0xFF,
            meas_source & 0xFF,
            driven_mask & 0xFF,
        ]))
        # Fixed-length frame: 1 (ACK) + 6×2 (u16 LE) + 1 (TRN_END) = 14
        resp = self._read(14)
        if resp[0] != PORT_ACK:
            raise RuntimeError(
                f"measure_spikes: bad ACK in response {resp.hex(' ')}"
            )
        if resp[13] != PORT_TRN_END:
            raise RuntimeError(
                f"measure_spikes: missing TRN_END in response {resp.hex(' ')}"
            )
        def _u16(i):
            return resp[i] | (resp[i + 1] << 8)
        return {
            'baseline_count': _u16(1),
            'baseline_min':   _u16(3),
            'baseline_max':   _u16(5),
            'driven_count':   _u16(7),
            'driven_min':     _u16(9),
            'driven_max':     _u16(11),
        }

    def ch10_burst(self, num_cycles: int, high_us: int = 200,
                   low_us: int = 200) -> dict:
        """Fire the firmware-side ch10 toggle burst and return the final
        latch word plus elapsed microseconds.

        Requires firmware ≥ 0.1.15 (PORT_CH10_BURST handler). Weights
        must be loaded before calling. Other DAC channels are not
        touched — make sure ch0..ch8 are at their quiescent codes
        beforehand so no stray hidden-neuron current interferes.
        """
        assert 0 <= num_cycles <= 0xFFFF
        assert 0 <= high_us <= 0xFFFF
        assert 0 <= low_us <= 0xFFFF
        self._send_cmd(PORT_CH10_BURST, "ch10_burst NAK'd on command byte")
        self._send(bytes([
            num_cycles & 0xFF, (num_cycles >> 8) & 0xFF,
            high_us & 0xFF, (high_us >> 8) & 0xFF,
            low_us & 0xFF, (low_us >> 8) & 0xFF,
        ]))
        # Response: word_lo, word_hi, elapsed_b0..b3, TRN_END = 7 bytes
        resp = self._read(7)
        if resp[6] != PORT_TRN_END:
            raise RuntimeError(
                f"ch10_burst: missing TRN_END in response {resp.hex(' ')}"
            )
        word = resp[0] | (resp[1] << 8)
        elapsed = (resp[2] | (resp[3] << 8) |
                   (resp[4] << 16) | (resp[5] << 24))
        return {'word': word, 'elapsed_us': elapsed}

    def calib_l1_single(self, dac_channel: int, dac_code: int,
                        max_wait_ms: int = 50, meas_channel: int = 0) -> int | None:
        """Measure time to first spike for one DAC channel.

        Returns elapsed microseconds, or None if no spike within timeout.
        """
        self._send_cmd(ord('C'), "CalibL1 NAK'd")  # PORT_CALIB_L1
        self._send(bytes([
            dac_channel,
            dac_code & 0xFF, (dac_code >> 8) & 0xFF,
            max_wait_ms & 0xFF, (max_wait_ms >> 8) & 0xFF,
            meas_channel,
        ]))
        # Read 4 bytes (u32 little-endian) + TRN_END
        data = self._read(4)
        self._read(1)  # TRN_END
        elapsed = data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24)
        return None if elapsed == 0xFFFFFFFF else elapsed

    # ── High-level operations ──

    def fc1_to_dac_codes(self, fc1_outputs: list[float], dac_scale: float = 1.0) -> list[int]:
        """Convert gilgamesh fc1 outputs to 12-bit DAC codes.

        Threshold-independent scaling:
            We pick V_DAC so that the membrane asymptote
            `u_inf = R_LEAK × I = (V_DAC_drop − V_BE) × R_LEAK/R_SET` equals
            the fc1 output `g` (in volts). Solving:
                V_DAC_drop = g × R_SET/R_LEAK + V_BE
            This is the physically-correct asymptote-inverse for any
            threshold — no theta factor, matching hw_forward.rs and the
            netlist's current-mirror equation.

        PNP inversion:
            The hidden-neuron input stage uses BCM857BS PNP current mirrors,
            so  I_in ≈ (V_DD − V_DAC_actual − V_BE) / R_SET_INPUT.
            Lower V_DAC → larger current. We compute the conceptual
            "drop" above V_BE (V_DAC_drop) and then write
            `V_DAC_actual = DAC_VREF − V_DAC_drop`, so that the board's PNP
            produces exactly the current we asked for.

        `dac_scale` (default 1.0) is a small per-board headroom factor —
        bump it above 1.0 if parts tolerance bites.
        """
        nominal_scale = R_SET_INPUT / R_LEAK
        scale = nominal_scale * dac_scale
        codes = []
        for g in fc1_outputs:
            v_drop = max(0.0, min(DAC_VREF, g * scale + V_BE))
            v_dac = DAC_VREF - v_drop
            code = int(round(v_dac / DAC_VREF * DAC_MAX_CODE))
            codes.append(max(0, min(DAC_MAX_CODE, code)))
        return codes

    def weight_to_sr_byte(self, w: int) -> int:
        """Convert quantized weight (-7 to +7) to shift register byte.

        Excitatory side (Q1..Q3 → bits 1..3): switches route PNP mirror
        collectors. Bit HIGH = routed to excitatory summing bus (engaged).
        Bit LOW = routed to GND (disengaged).

        Inhibitory side (Q5, Q6 → bits 5, 6): switches route NPN mirror
        collectors with INVERTED polarity. Bit HIGH = routed to VCC
        (disengaged). Bit LOW = routed to inhibitory summing bus
        (engaged). So "no inhibition" requires bits 5,6 driven HIGH.

        Bit 4 (Q4, the ×1 inhibitory unit) is physically disconnected on
        this board, so the inhibitory side only has ×2 and ×4 stages —
        magnitudes 0, 2, 4, 6 are realizable exactly; odd magnitudes get
        rounded to the nearest even value and one LSB of precision is
        lost. Bit 4 is still driven HIGH for cleanliness.

        Bits 0 and 7 of the byte hit Q0 and Q7 of the 74HC595, both of
        which are unconnected on the PCB, so they don't matter.
        """
        mag = min(abs(w), 7)
        # Inhibitory "all off" mask: drive Q4, Q5, Q6 HIGH so every NPN
        # unit switch routes its collector to VCC rather than the
        # inhibitory summing bus.
        INH_OFF = 0x70
        if w > 0:
            exc = ((mag & 1) << 1) | ((mag & 2) << 1) | ((mag & 4) << 1)
            return exc | INH_OFF
        if w < 0:
            # Engage inhibitory units by clearing the relevant bits.
            # Q4 (bit 4) is disconnected; only Q5 (×2) and Q6 (×4) are
            # physically realised. Round |w| to the nearest available
            # combination of {2, 4, 6}.
            inh_engage = ((mag & 2) << 4) | ((mag & 4) << 4)
            return INH_OFF ^ inh_engage
        return INH_OFF

    def load_checkpoint(self, checkpoint_path: str, dac_scale: float = 1.16):
        """Load a gilgamesh checkpoint's fc2 weights into the board."""
        with open(checkpoint_path) as f:
            cp = json.load(f)

        fc2_q = cp['quantized']['fc2_weight']
        sr_bytes = [SR_IDLE_BYTE] * NUM_SYNAPSES
        for h in range(9):
            for o in range(10):
                sr_bytes[phys_byte(o, h)] = self.weight_to_sr_byte(fc2_q[h][o])

        self.load_weights(sr_bytes)
        print(f"Loaded {len(sr_bytes)} synapse weights from {checkpoint_path}")
        return cp

    def infer(self, pixels: np.ndarray, fc1_weights: np.ndarray,
              calib: dict = None, num_samples: int = 50,
              interval_us: int = 500) -> tuple[int, list[int]]:
        """Run a single inference using rapid spike sampling.

        Args:
            pixels: 6x6 normalized pixel array (36 floats)
            fc1_weights: [36, 9] fc1 weight matrix from checkpoint
            calib: calibration data (per-channel DAC scales). If None, uses default.
            num_samples: number of spike sampling intervals (default 50)
            interval_us: microseconds between samples (default 500 = 25ms total)

        Returns:
            (prediction, spike_counts) — predicted digit and per-neuron spike counts
        """
        # fc1 weighted sum (laptop-side preprocessing)
        fc1_out = pixels.flatten() @ fc1_weights

        # Convert to DAC codes using calibration data
        if calib and 'dac_scales' in calib:
            dac_codes = self.fc1_to_dac_codes_calibrated(fc1_out.tolist(), calib['dac_scales'])
        else:
            dac_codes = self.fc1_to_dac_codes(fc1_out.tolist())
        self.load_dacs(dac_codes)

        # Small delay for DAC to settle
        time.sleep(0.001)

        # Run inference with rapid spike sampling
        snapshots = self.run_inference(num_samples, interval_us)

        # Count spikes per output neuron across all snapshots
        spike_counts = accumulate_spike_counts(snapshots)

        # Prediction: argmax of spike counts
        prediction = max(range(10), key=lambda i: spike_counts[i])

        return prediction, spike_counts

    def fc1_to_dac_codes_calibrated(self, fc1_outputs: list[float],
                                     dac_scales: list[float]) -> list[int]:
        """Convert fc1 outputs to DAC codes using per-channel calibrated scales.

        The calibration step fits a per-channel multiplier that wraps the
        baseline `R_SET_INPUT / R_LEAK` factor so that every hidden neuron
        reaches the same effective asymptote for a given `g`. See
        `fc1_to_dac_codes` for the threshold-independent physics."""
        base = R_SET_INPUT / R_LEAK
        codes = []
        for i, g in enumerate(fc1_outputs):
            per_ch = dac_scales[i] if i < len(dac_scales) else dac_scales[0]
            scale = base * per_ch
            v_drop = max(0.0, min(DAC_VREF, g * scale + V_BE))
            v_dac = DAC_VREF - v_drop
            code = int(round(v_dac / DAC_VREF * DAC_MAX_CODE))
            codes.append(max(0, min(DAC_MAX_CODE, code)))
        return codes

    def probe_sr_topology(self, observable_output: int = 9,
                          n_samples: int = 50, interval_us: int = 500) -> dict:
        """Reverse-engineer the SR byte→synapse mapping empirically.

        Background: `weight_to_sr_byte`'s docstring says "(will be verified
        against schematic)" — the bit layout inside each byte has never
        been checked, and the 90-byte→physical-synapse mapping
        (`src*10+j` vs `j*9+src` vs reversed-endian) has never been
        confirmed either. This probe doesn't trust any of that. It just
        sweeps all 90 SR byte positions one at a time and records which
        byte positions (if any) cause `observable_output` to fire under
        ch10 drive.

        Because only one output neuron has been reworked with a 5.6 nF
        parallel cap, `observable_output` is the ONLY latch-visible
        output. Fires on O0..O8 exist in the analog world but can't be
        captured. That's OK — we only need one observable to disambiguate
        the SR layout.

        What the results tell us:

        * If EXACTLY ONE byte position fires the observable output, each
          SR byte maps 1-to-1 to a physical synapse and synapses are
          independent (topology A). The winning byte index tells us
          whether the layout is hidden-major (`src*10+j`) or
          output-major (`j*9+src`) or some rotation/reversal of either.

        * If NO single byte fires but ALL 10 bytes in one "row" fire
          together, synapses in that row share a single current source
          that needs multiple bits set to overcome some bias/threshold
          (topology B, or a shared pull-up/pull-down that disables when
          bits are floating). This matches the observed 4.3-fires-but-
          4.5-quiet anomaly.

        * If MULTIPLE non-consecutive bytes fire, there's aliasing or
          cross-talk — each byte may control more than one physical
          synapse.

        * If the observable fires at every byte position equally, the
          SR output is stuck at all-+7 regardless of programming.

        Also does a within-byte bit sweep on the first byte that fires,
        to verify which bit positions encode the magnitude.
        """
        assert 0 <= observable_output < 10
        print(f"\n=== SR topology probe (observable = O{observable_output}) ===")
        results = {
            'observable_output': observable_output,
            'fires_per_byte': [0] * NUM_SYNAPSES,
            'winning_bytes': [],
            'within_byte_bit_sweep': {},
        }
        # Ensure idle before we start.
        self.quiesce(zero_weights=True)
        time.sleep(0.02)

        w7_byte = self.weight_to_sr_byte(7)

        # Warm-up: the --calibrate flow runs Phases 1/2 + diagnose_output_path
        # before Phase 3, which means by the time Phase 3 fires weights the
        # output integrator has been exposed to ~600 ms of sustained ch10-HIGH
        # stimulation (via the tight-loop L2 probe). A cold-start probe that
        # skips this doesn't reproduce the fire behavior. So we reproduce the
        # warm-up here: load all-+7 weights, assert ch10, hold it HIGH via a
        # long calib_l1_single poll. Do this three times.
        print("\n  Warm-up: hold ch10 HIGH with all-+7 weights for ~600 ms")
        print("  (reproduces the --calibrate pre-Phase-3 integrator state).")
        for _ in range(3):
            self.load_weights([w7_byte] * NUM_SYNAPSES)
            try:
                self.calib_l1_single(
                    dac_channel=10, dac_code=DAC_MAX_CODE,
                    max_wait_ms=200, meas_channel=1,
                )
            except Exception:
                pass
        self.load_weights([SR_IDLE_BYTE] * NUM_SYNAPSES)
        self.load_dacs(list(DAC_QUIESCENT_CODES))
        time.sleep(0.01)
        print(f"  weight_to_sr_byte(+7) = 0x{w7_byte:02X} "
              f"(binary {w7_byte:08b})")

        # Baseline sanity: all 90 bytes at +7 must fire the observable,
        # otherwise the board is in a state that the earlier --calibrate
        # phases somehow warm up and our cold-start probe is invalid.
        print("\n  Baseline: all 90 bytes at +7 (matches Phase 3 config):")
        sr_all = [w7_byte] * NUM_SYNAPSES
        fires_all = 0
        for _ in range(3):
            snaps = self.sample_with_synapse_pulse(
                sr_all, num_samples=n_samples, interval_us=interval_us,
            )
            fires_all += count_observable_fires(snaps, observable_output)
        results['baseline_all_plus7_fires'] = fires_all
        print(f"    all-90 @ +7 → O{observable_output}: "
              f"{fires_all}/{3*n_samples} fires across 3 runs")
        if fires_all == 0:
            print(f"    WARNING: baseline all-+7 doesn't fire O{observable_output}!")
            print(f"    This differs from Phase 3 of the most recent --calibrate")
            print(f"    run (which saw 14/50 fires). Board state or warm-up")
            print(f"    differs — aborting sweep, it would be meaningless.")
            self.quiesce(zero_weights=True)
            return results

        # Sweep all 90 byte positions, one at a time.
        print(f"\n  Single-byte sweep: enable exactly one SR byte at +7,")
        print(f"  drive ch10 HIGH, count fires on O{observable_output}.")
        print(f"  (The other 89 bytes are held at 0.)\n")

        for byte_idx in range(NUM_SYNAPSES):
            sr_bytes = [SR_IDLE_BYTE] * NUM_SYNAPSES
            sr_bytes[byte_idx] = w7_byte
            snaps = self.sample_with_synapse_pulse(
                sr_bytes, num_samples=n_samples, interval_us=interval_us,
            )
            fires = count_observable_fires(snaps, observable_output)
            results['fires_per_byte'][byte_idx] = fires
            if fires > 0:
                results['winning_bytes'].append(byte_idx)
                phys_out, phys_hid = byte_to_physical(byte_idx)
                print(f"    byte[{byte_idx:2d}] +7 → O{observable_output} "
                      f"fired {fires}/{n_samples}  "
                      f"(phys mapping: H{phys_hid}→O{phys_out})")

        if not results['winning_bytes']:
            print("    No single-byte +7 pattern fired the observable.")
            print("    → synapses are NOT independent: need ≥2 bits set")
            print("      for any row to produce enough current (shared-")
            print("      source / biased topology). Falling back to the")
            print("      pairs-of-bytes scan next.")

        # If no single byte fired, scan pairs to find the smallest set
        # that does fire. This runs 90 * 89 / 2 = 4005 tests in the worst
        # case — cap it by only scanning pairs within the same 10-byte
        # block (either byte_i % 10 == byte_j % 10 for column pairs, or
        # byte_i // 10 == byte_j // 10 for row pairs) to keep runtime
        # bounded.
        if not results['winning_bytes']:
            print("\n  Row-pair sweep (pairs within same 10-byte block):")
            results['row_pair_fires'] = {}
            for row in range(9):
                base = row * 10
                row_bytes = list(range(base, base + 10))
                # Enable all 10 bytes in this row.
                sr_bytes = [SR_IDLE_BYTE] * NUM_SYNAPSES
                for b in row_bytes:
                    sr_bytes[b] = w7_byte
                snaps = self.sample_with_synapse_pulse(
                    sr_bytes, num_samples=n_samples, interval_us=interval_us,
                )
                fires = count_observable_fires(snaps, observable_output)
                results['row_pair_fires'][f'row{row}_all10'] = fires
                mark = '●' if fires > 0 else '·'
                print(f"    all 10 bytes in row {row} (bytes {base}..{base+9}): "
                      f"{mark}  ({fires}/{n_samples})")

            # If "row N, all 10 enabled" fires, try progressively smaller
            # subsets within that row to find the minimum set.
            firing_rows = [
                int(k.split('row')[1].split('_')[0])
                for k, v in results['row_pair_fires'].items() if v > 0
            ]
            if firing_rows:
                r = firing_rows[0]
                base = r * 10
                print(f"\n  Minimum-subset scan in row {r}:")
                for k in range(1, 11):
                    sr_bytes = [SR_IDLE_BYTE] * NUM_SYNAPSES
                    for b in range(base, base + k):
                        sr_bytes[b] = w7_byte
                    snaps = self.sample_with_synapse_pulse(
                        sr_bytes, num_samples=n_samples, interval_us=interval_us,
                    )
                    fires = count_observable_fires(snaps, observable_output)
                    mark = '●' if fires > 0 else '·'
                    print(f"    first {k:2d} bytes of row {r} enabled: "
                          f"{mark}  ({fires}/{n_samples})")
                    results[f'row{r}_first_{k}'] = fires
                    if fires > 0:
                        print(f"    → minimum firing subset = "
                              f"{k} byte(s) of row {r}")
                        break

        # Within-byte bit sweep: pick the first winning byte (if any)
        # and sweep byte values 1..255 to verify bit encoding.
        if results['winning_bytes']:
            b = results['winning_bytes'][0]
            print(f"\n  Within-byte bit sweep on byte[{b}] "
                  f"(finds which bits actually carry current):")
            per_bit_fires = {}
            for bit in range(8):
                sr_bytes = [SR_IDLE_BYTE] * NUM_SYNAPSES
                sr_bytes[b] = 1 << bit
                snaps = self.sample_with_synapse_pulse(
                    sr_bytes, num_samples=n_samples, interval_us=interval_us,
                )
                fires = count_observable_fires(snaps, observable_output)
                per_bit_fires[bit] = fires
                mark = '●' if fires > 0 else '·'
                print(f"    byte[{b}] = 0b{1<<bit:08b} (bit {bit}): "
                      f"{mark}  ({fires}/{n_samples})")
            results['within_byte_bit_sweep'] = per_bit_fires

        self.quiesce(zero_weights=True)

        # Summary
        print("\n  --- Summary ---")
        print(f"  Winning bytes (single-+7 fires O{observable_output}): "
              f"{results['winning_bytes']}")
        if results['winning_bytes']:
            print(f"  → Synapses appear INDEPENDENT (topology A).")
            print(f"    Use the fire locations above to validate the")
            print(f"    `src*10+j` indexing in set_subset_weights.")
        elif any(v > 0 for v in results.get('row_pair_fires', {}).values()):
            print(f"  → Synapses appear to need multiple bits to fire")
            print(f"    the observable. Either shared row current or")
            print(f"    the SR storage is not fully latching single-bit")
            print(f"    changes (e.g. cleared bits leaking HIGH).")
        else:
            print(f"  → NOTHING fired the observable across the whole")
            print(f"    sweep — the H_→O{observable_output} synapse itself")
            print(f"    is likely broken, or weight encoding is wrong.")

        return results

    def verify_mapping(self, observable_output: int = 9,
                       n_samples: int = 50, interval_us: int = 500) -> dict:
        """Enable one (output, hidden) synapse at a time via the corrected
        `phys_byte` mapping and check which light up the observable output.

        Because only `observable_output` (reworked with a 5.6 nF parallel
        cap) has a latch-visible pulse extender, we can only directly
        confirm the 9 synapses targeting that output. For each hidden
        source h ∈ 0..8, we call `phys_byte(observable_output, h)`, set
        that single SR byte to +7, pulse ch10, and count fires. If the
        mapping is correct, every one of the 9 (with sufficient drive
        headroom per synapse) should produce a non-zero fire count.

        As a cross-check, we also exercise one (output, hidden) pair per
        other output column at +7 to observe analog activity via
        `measure_spikes(meas_source=1, drive_ch10=True)` — a Δ-count
        above the idle baseline means the synapse current is landing
        on the intended output membrane, even though its latch can't
        capture the narrow 10 pF pulse.
        """
        print("\n=== Mapping Verification ===")
        print(f"  Target observable output: O{observable_output}")
        print(f"  Using phys_byte(output, hidden) = 89 - (9*output + hidden)\n")

        self.quiesce(zero_weights=True)
        w7_byte = self.weight_to_sr_byte(7)
        results = {
            'observable_output': observable_output,
            'per_hidden': {},
            'other_outputs_analog': {},
        }

        # 1. Sweep hidden sources targeting the observable output.
        print(f"  [1] Per-hidden fire counts targeting O{observable_output} "
              f"(latch visible):")
        for h in range(9):
            sr_bytes = [SR_IDLE_BYTE] * NUM_SYNAPSES
            idx = phys_byte(observable_output, h)
            sr_bytes[idx] = w7_byte
            snaps = self.sample_with_synapse_pulse(
                sr_bytes, num_samples=n_samples, interval_us=interval_us,
            )
            fires = count_observable_fires(snaps, observable_output)
            results['per_hidden'][h] = {'byte_index': idx, 'fires': fires}
            mark = '●' if fires > 0 else '·'
            print(f"    H{h} → O{observable_output}  "
                  f"byte[{idx:2d}]  {mark}  ({fires}/{n_samples})")

        fired_hiddens = [h for h, d in results['per_hidden'].items()
                         if d['fires'] > 0]
        print(f"\n  {len(fired_hiddens)}/9 hidden sources fired the observable "
              f"under the new mapping.")

        # 2. Analog cross-check: for each other output, try H0→Oj and
        # look at the driven−baseline Δ in the L2 wired-OR bus from
        # `measure_spikes(meas_source=1, drive_ch10=True)`.
        print(f"\n  [2] Analog cross-check on non-observable outputs "
              f"(L2 wired-OR driven−baseline):")
        for j in range(10):
            if j == observable_output:
                continue
            sr_bytes = [SR_IDLE_BYTE] * NUM_SYNAPSES
            idx = phys_byte(j, 0)
            sr_bytes[idx] = w7_byte
            self.load_weights(sr_bytes)
            res = self.measure_spikes(
                target_channel=0, dac_code=0,
                window_ms=50, meas_source=1, drive_ch10=True,
            )
            base_c = res['baseline_count']
            drv_c = res['driven_count']
            delta = drv_c - base_c
            results['other_outputs_analog'][j] = {
                'byte_index': idx,
                'baseline_count': base_c, 'driven_count': drv_c,
                'delta': delta,
            }
            print(f"    H0 → O{j}  byte[{idx:2d}]  "
                  f"base={base_c} drv={drv_c} Δ={delta:+d}")
            self.load_weights([SR_IDLE_BYTE] * NUM_SYNAPSES)

        self.quiesce(zero_weights=True)
        return results

    def diagnose_output_path(self) -> dict:
        """Host-side output-latch diagnostics.

        Phase 3 has been seeing all 10 output latches stuck at their
        cleared state (raw word 0xFFC0, polarity inverted so HIGH = idle)
        regardless of weights and ch10 drive. This probes four things
        without needing a firmware reflash:

          1. L2 output wired-OR analog level vs ch10 drive + weight
             pattern. Uses `read_measurement(1)` (PIN_L2_MEAS_OUT, A7 =
             output-layer wired-OR across the 10 L2 V_outs).
             This bypasses the SR latch entirely and reports whether
             the synapses are actually sourcing current into the output
             membranes as a function of weight.
          2. Weight loopback: run test (1) with three different SR
             patterns (all-0, all-+7, alternating-+7). If L2 is
             identical across patterns, the 74HC595 storage register
             isn't receiving data.
          3. Reset-line stuck check: do back-to-back read_output() calls
             with no stimulus, confirm 0xFFC0 idle both times, then
             stimulate and watch whether the word ever deviates.
          4. Polarity flip: set FLAG_RESET_SR_ACTIVE_LOW and rerun test
             (3). If the flipped polarity reveals spikes, our original
             pulse direction was wrong.

        Prints a readable report and returns a dict of the measurements.
        """
        print("\n=== Output-path diagnostics ===")
        results = {}
        self.quiesce(zero_weights=True)
        time.sleep(0.01)

        # Helper: measure L2 wired-OR under a given weight pattern with
        # ch10 asserted HIGH.
        def l2_under(weights: list[int], label: str) -> int:
            self.load_weights(weights)
            self.load_dacs(ch10_drive_codes())
            time.sleep(0.005)
            # L2 wired-OR: enable meas path, sample via firmware's
            # ReadMeasurement(source=1).
            v = self.read_measurement(1)
            # Release ch10 / weights back to idle.
            self.load_dacs(list(DAC_QUIESCENT_CODES))
            self.load_weights([SR_IDLE_BYTE] * NUM_SYNAPSES)
            time.sleep(0.002)
            print(f"  [L2 wired-OR] {label:<24}  ADC={v:4d}  (~{adc_to_volts(v):.2f} V)")
            return v

        # 1+2. L2 probe with three weight patterns.
        print("\n  (1+2) Does L2 output-bus swing with weight pattern?")
        w_zero = [SR_IDLE_BYTE] * NUM_SYNAPSES
        w_plus7 = [self.weight_to_sr_byte(7)] * NUM_SYNAPSES
        w_alt = [self.weight_to_sr_byte(7) if (i % 2 == 0) else 0
                 for i in range(NUM_SYNAPSES)]
        # Baseline (no ch10) for reference.
        self.quiesce(zero_weights=True)
        time.sleep(0.005)
        l2_idle = self.read_measurement(1)
        print(f"  [L2 wired-OR] {'idle (ch10=0, w=0)':<24}  ADC={l2_idle:4d}  (~{adc_to_volts(l2_idle):.2f} V)")
        results['l2_idle'] = l2_idle

        results['l2_w0_ch10hi'] = l2_under(w_zero,  "w=0       ch10=HI")
        results['l2_w7_ch10hi'] = l2_under(w_plus7, "w=+7 all  ch10=HI")
        results['l2_alt_ch10hi'] = l2_under(w_alt,  "w=+7 alt  ch10=HI")

        l2_swing = results['l2_w7_ch10hi'] - results['l2_w0_ch10hi']
        l2_alt_vs_7 = abs(results['l2_alt_ch10hi'] - results['l2_w7_ch10hi'])
        if l2_swing < 20:
            print(f"  → L2 barely moves ({l2_swing} counts) when weights flip 0→+7.")
            print("    Either (a) the 74HC595 isn't latching the weight data,")
            print("    (b) the analog-switch matrix is broken/stuck, or")
            print("    (c) synapse current is below the wired-OR sense floor.")
        else:
            print(f"  → L2 moves by {l2_swing} counts for all-+7 vs all-0.")
            print(f"    Synapse current IS reaching the output membrane.")
        if l2_alt_vs_7 < 10 and l2_swing >= 20:
            print(f"  → Alt vs all-+7 differ by only {l2_alt_vs_7} counts —")
            print("    weight pattern _might_ not matter much (saturation?),")
            print("    but SR is at least receiving *something*.")

        # 3. Back-to-back read_output with no stimulus.
        print("\n  (3) Latch idle state + stimulated-read sanity:")
        self.quiesce(zero_weights=True)
        time.sleep(0.005)
        w_idle_a = self.read_output()
        w_idle_b = self.read_output()
        print(f"  idle read #1 = 0x{w_idle_a:04X}   idle read #2 = 0x{w_idle_b:04X}")
        results['idle_read_a'] = w_idle_a
        results['idle_read_b'] = w_idle_b

        # Now fire with all-+7 and sample many output words without
        # explicitly resetting between reads. sample_with_synapse_pulse
        # uses run_inference which DOES pulse reset — so we also do a
        # single "raw" read_output right after loading, before any
        # reset.
        self.load_weights(w_plus7)
        self.load_dacs(ch10_drive_codes())
        time.sleep(0.01)
        raw_words = [self.read_output() for _ in range(6)]
        self.quiesce(zero_weights=True)
        print(f"  stimulated raw words (ch10=HI, w=+7, no run_inference):")
        print(f"    {[f'0x{w:04X}' for w in raw_words]}")
        results['stim_raw_words'] = raw_words
        uniq = set(raw_words)
        if uniq == {w_idle_a} or uniq == {0xFFC0}:
            print("  → Every stimulated read matches idle state —")
            print("    latch SET inputs never go HIGH. Either no spike edges")
            print("    arrive at U15101..U15105, or reset is held asserted.")
        elif len(uniq) > 1:
            print(f"  → Stimulated reads vary ({len(uniq)} distinct words) —")
            print("    spike capture path IS working; investigate decoding.")

        # 3b. Tight-loop L2 probe via calib_l1_single(meas_channel=1).
        # analogRead in a busy-wait catches ~13k samples/sec — two orders
        # of magnitude more than the one-shot read_measurement used above,
        # which is crucial when the output pulse extender has a 10 pF cap
        # (≈2 µs stretched pulse vs ~1 ms on the hidden layer).
        #
        # Firmware CalibL1 quiesces all DACs, then sets dac_channel=10 to
        # dac_code=MAX, then polls the selected meas pin for threshold
        # (ADC>200, ~0.98 V) until first hit or timeout. We first load
        # weights (persistent across commands), then call it.
        print("\n  (3b) Tight-loop L2 probe (catches narrow output pulses):")

        def l2_probe_first_spike(weights: list[int], label: str,
                                 max_wait_ms: int = 200) -> int | None:
            self.load_weights(weights)
            # dac_channel=10 (V_OUT1 synapse rail), dac_code=MAX (HIGH),
            # meas_channel=1 (PIN_L2_MEAS_OUT).
            try:
                elapsed = self.calib_l1_single(
                    dac_channel=10, dac_code=DAC_MAX_CODE,
                    max_wait_ms=max_wait_ms, meas_channel=1,
                )
            except Exception as e:
                print(f"    {label}: error ({e})")
                return None
            # Clean up weights.
            self.load_weights([SR_IDLE_BYTE] * NUM_SYNAPSES)
            if elapsed is None:
                print(f"    {label:<20} → no L2 pulse > 0.98 V within {max_wait_ms} ms")
            else:
                print(f"    {label:<20} → first L2 pulse at t={elapsed} µs")
            return elapsed

        # Control: weights=0 should NEVER fire (no synapse current path).
        # If this DOES fire, ch10 is leaking directly onto the L2 bus and
        # all our "fire" readings are bogus.
        results['l2_tight_w0']  = l2_probe_first_spike(w_zero,  "w=0      (control)")
        results['l2_tight_w7']  = l2_probe_first_spike(w_plus7, "w=+7 all          ")
        results['l2_tight_alt'] = l2_probe_first_spike(w_alt,   "w=+7 alt          ")
        if results.get('l2_tight_w7') is not None and results.get('l2_tight_w0') is None:
            print("  → Output neurons ARE firing under w=+7 drive "
                  "(L2 pulse caught above 0.98 V).")
            print("    The SR-latch spike-capture path is what's broken —")
            print("    likely: 10 pF output pulse extender makes V_out peak")
            print("    below HC02 V_IH, so latch SET never registers.")
        elif (results.get('l2_tight_w7') is None
              and results.get('l2_tight_w0') is None):
            print("  → Even tight-loop L2 probe sees nothing above 0.98 V.")
            print("    Either output neurons never cross V_mem threshold,")
            print("    OR the 10 pF pulse extender makes V_out peak so")
            print("    briefly/narrowly that even 13 kHz polling misses it.")
        elif (results.get('l2_tight_w0') is not None
              and results.get('l2_tight_w7') is not None):
            print("  → WARNING: w=0 control also fires. ch10 is leaking")
            print("    directly onto the L2 bus — this measurement is unreliable.")

        # 3c. Differential L2 wired-OR spike count via firmware
        # measure_spikes with meas_source=1 + drive_ch10=True. This is
        # the output-layer analogue of the L1 baseline-vs-driven count
        # already used in Phase 1/2, and it samples A7 at full firmware
        # analogRead rate across the entire window (not just a single
        # Python-side ADC sample). Much better coverage than (1+2)'s
        # `read_measurement` snapshots.
        print("\n  (3c) L2 differential spike count "
              "(firmware analog sample, ch10 driven):")

        def l2_diff_count(weights: list[int], label: str,
                          window_ms: int = 100) -> dict | None:
            self.load_weights(weights)
            try:
                # target_channel=10 is ignored for our purposes because
                # drive_ch10=True overrides it during the driven window,
                # but we still have to pass a valid channel. The
                # baseline window uses _SetDacsQuiescent so ch10 is 0 V
                # there. Using 11 (no-connect) keeps the
                # target-programming line a no-op electrically.
                r = self.measure_spikes(
                    target_channel=11, dac_code=0,
                    window_ms=window_ms,
                    meas_source=1, drive_ch10=True,
                )
            except Exception as e:
                print(f"    {label}: error ({e})")
                return None
            finally:
                self.load_weights([SR_IDLE_BYTE] * NUM_SYNAPSES)
                self.load_dacs(list(DAC_QUIESCENT_CODES))
            delta = r['driven_count'] - r['baseline_count']
            print(f"    {label:<20} "
                  f"base={r['baseline_count']:4d} [{r['baseline_min']:4d}..{r['baseline_max']:4d}]  "
                  f"drv={r['driven_count']:4d} [{r['driven_min']:4d}..{r['driven_max']:4d}]  "
                  f"Δ={delta:+5d}")
            return r

        try:
            results['l2_diff_w0']  = l2_diff_count(w_zero,  "w=0      (control)")
            results['l2_diff_w7']  = l2_diff_count(w_plus7, "w=+7 all          ")
            results['l2_diff_alt'] = l2_diff_count(w_alt,   "w=+7 alt          ")
        except RuntimeError as e:
            print(f"    (skipped — firmware may be too old: {e})")

        w7_r = results.get('l2_diff_w7')
        w0_r = results.get('l2_diff_w0')
        if w7_r and w0_r:
            w7_delta = w7_r['driven_count'] - w7_r['baseline_count']
            w0_delta = w0_r['driven_count'] - w0_r['baseline_count']
            if w7_delta > w0_delta + 5:
                print(f"  → L2 Δ-count shows real synapse→output activity "
                      f"({w7_delta} vs {w0_delta} control).")
            elif w0_delta > 5:
                print(f"  → Even the w=0 control has Δ={w0_delta} — ch10 may")
                print(f"    be leaking onto the L2 bus directly.")
            else:
                print(f"  → Neither driven nor control shows L2 activity.")

        # 4. Polarity flip: flip FLAG_RESET_SR_ACTIVE_LOW and retry.
        print("\n  (4) Retry stimulated reads with RESET_SR polarity flipped:")
        try:
            self.set_flag(FLAG_RESET_SR_ACTIVE_LOW)
        except Exception as e:
            print(f"  (could not set polarity flag: {e})")
            results['polarity_flip_supported'] = False
            return results
        results['polarity_flip_supported'] = True
        try:
            self.quiesce(zero_weights=True)
            time.sleep(0.005)
            idle_flip = self.read_output()
            self.load_weights(w_plus7)
            self.load_dacs(ch10_drive_codes())
            time.sleep(0.01)
            flip_words = [self.read_output() for _ in range(6)]
            self.quiesce(zero_weights=True)
            print(f"  idle (flipped) = 0x{idle_flip:04X}")
            print(f"  stimulated (flipped) = {[f'0x{w:04X}' for w in flip_words]}")
            results['flip_idle'] = idle_flip
            results['flip_stim_words'] = flip_words
            if set(flip_words) != {idle_flip} and len(set(flip_words)) > 1:
                print("  → Flipped polarity shows varying output! "
                      "Original reset polarity was probably inverted.")
        finally:
            self.unset_flag(FLAG_RESET_SR_ACTIVE_LOW)

        return results

    def full_calibration(self, max_wait_ms: int = 50,
                         dac_voltages: list[float] = None) -> dict:
        """Run the full board calibration procedure.

        PNP inversion note: the hidden-neuron input stage uses PNP current
        mirrors, so low V_DAC = high stimulation current. "Max stim" is
        code 0 (~0 V), "no stim" is code 4095 (~V_DD). The voltage sweep
        therefore goes from HIGH V_DAC (weak stim) to LOW V_DAC (strong).

        Wired-OR measurement note: `MEAS_OUT` (A6) is shared across all
        9 hidden-neuron V_out stretch nodes through 2N7002 switches with
        a single global enable. A plain `analogRead` can't isolate one
        channel — it always sees the OR of everyone's recent activity.
        Phase 1/2 therefore use `measure_spikes`, which samples the bus
        twice (quiescent vs target-driven) and reports a differential
        count. A channel is "active" when `driven - baseline >= DELTA`.

        1. Drive each DAC channel at max stim (V_DAC ≈ 0) and compare
           bus activity vs baseline.
        2. Sweep V_DAC from high→low, recording the same delta per
           voltage.
        3. Compute per-channel scale factors from those curves.
        4. Test synapse current delivery by programming weights and
           observing output spikes.
        """
        if dac_voltages is None:
            # Sweep from weak (high V_DAC) to strong (low V_DAC).
            dac_voltages = [v * 0.2 for v in range(20, 0, -1)]  # 4.0V → 0.2V

        calib = {
            'dac_channel_map': {},   # channel_idx → neuron response data
            'dac_scales': [],        # per-channel scale for fc1→DAC mapping
            'spike_time_curves': {}, # channel_idx → [(v_dac, time_us), ...]
            'synapse_test': {},      # output neuron test results
        }

        # The firmware disables the ADC at boot to save power. CalibL1 and
        # ReadMeasurement both use analogRead() and silently return 0 on a
        # disabled ADC, which looks identical to "never spiked". Enable it
        # for the duration of calibration.
        self.set_flag(1)  # flag 1 = ADC enable

        # Clean slate: all 12 DAC channels quiescent (ch 10 synapse setup
        # rail forced to 0 V), all 90 synapse weights zero. Without this,
        # previous weight patterns or ghost DAC channel states bleed into
        # the hidden-layer tests.
        print("\nQuiescing board before calibration...")
        self.quiesce(zero_weights=True)
        time.sleep(0.01)

        print("\n=== Phase 1: DAC Channel Discovery ===")
        print("Driving each DAC channel to max stim (V_DAC ≈ 0, PNP inverted)")
        print("and measuring the wired-OR bus as baseline-vs-driven counts.\n")

        MAX_STIM_CODE = 0
        WINDOW_MS = 50
        ACTIVE_DELTA = 20  # min (driven - baseline) samples to call a channel active

        for ch in range(9):
            r = self.measure_spikes(ch, MAX_STIM_CODE, WINDOW_MS)
            delta = r['driven_count'] - r['baseline_count']
            active = delta >= ACTIVE_DELTA
            mark = "ACTIVE" if active else "inactive"
            print(
                f"  Ch{ch}: base cnt={r['baseline_count']:>4} "
                f"[{r['baseline_min']:>4}..{r['baseline_max']:>4}]  "
                f"drv cnt={r['driven_count']:>4} "
                f"[{r['driven_min']:>4}..{r['driven_max']:>4}]  "
                f"Δ={delta:+5}  {mark}"
            )
            calib['dac_channel_map'][ch] = {
                **r,
                'delta': delta,
                'active': active,
            }

        active_channels = [ch for ch, d in calib['dac_channel_map'].items() if d['active']]
        print(f"\n  Active channels: {active_channels} ({len(active_channels)}/9)")
        if not active_channels:
            print("  WARNING: all channels dark. Possible causes:")
            print("   - Background self-oscillation saturating the wired-OR bus")
            print("     (baseline ≈ driven because the bus is already HIGH)")
            print("   - Current budget too marginal (input ~= leak at threshold)")
            print("   - Measurement path dead")
            print("  Try scoping individual V_out / N_OUT test points.")

        print("\n=== Phase 2: Spike-Rate Response Curves ===")
        print("Sweeping V_DAC per active channel, measuring Δ-count.\n")

        for ch in active_channels:
            curve = []
            for v_dac in dac_voltages:
                code = int(round(v_dac / DAC_VREF * DAC_MAX_CODE))
                r = self.measure_spikes(ch, code, WINDOW_MS)
                delta = r['driven_count'] - r['baseline_count']
                curve.append({'v_dac': v_dac, **r, 'delta': delta})
                print(
                    f"  Ch{ch} V={v_dac:.2f}V  base={r['baseline_count']:>4} "
                    f"drv={r['driven_count']:>4} Δ={delta:+5} "
                    f"drv[{r['driven_min']:>4}..{r['driven_max']:>4}]"
                )

            calib['spike_rate_curves'] = calib.get('spike_rate_curves', {})
            calib['spike_rate_curves'][ch] = curve

            best_delta = max((p['delta'] for p in curve), default=0)
            best_v = next(
                (p['v_dac'] for p in curve if p['delta'] == best_delta),
                dac_voltages[-1],
            )
            # Rough scale: V_stim_drop = (V_DD_assumed - best_v)
            # Store as a divisor so the existing fc1_to_dac_codes logic
            # ("v_drop = max(0, min(DAC_VREF, g * scale + V_BE))") stays
            # consistent with what Phase 1/2 observed as a good stim point.
            v_drop = max(0.1, DAC_VREF - best_v)
            scale = v_drop / max(abs(THETA_0), 1e-3)
            calib['dac_scales'].append(scale)
            print(f"  Ch{ch} best Δ={best_delta} at V_DAC={best_v:.2f}V "
                  f"(v_drop={v_drop:.2f}V)  scale={scale:.4f}\n")

        # Fill remaining scales with average for inactive channels
        if calib['dac_scales']:
            avg_scale = sum(calib['dac_scales']) / len(calib['dac_scales'])
        else:
            avg_scale = THETA_0 * R_SET_INPUT / R_LEAK  # nominal
        while len(calib['dac_scales']) < 9:
            calib['dac_scales'].append(avg_scale)

        # Fully quiesce between Phase 2 and Phase 3 so leftover synapse
        # bits or driven DAC channels from Phase 2 don't bias the
        # all-excitatory smoke test.
        self.quiesce(zero_weights=True)
        time.sleep(0.01)

        # Phases 3+ factored out so they can be re-run in isolation
        # via --calibrate-phase3 without re-doing the (working, slow)
        # Phase 1/2 DAC sweep.
        self._run_phases_3_and_4(calib)
        return calib

    def partial_calibration(self, existing_calib: dict = None) -> dict:
        """Re-run Phases 3 + 4 only, reusing cached dac_scales.

        Use this when Phases 1/2 are known to work and you're iterating
        on the synapse-verification / combinatorial-calibration path
        (typically after a hardware rework on the output layer). If
        `existing_calib` is None, loads `calibration_results.json`.
        """
        if existing_calib is None:
            try:
                with open('calibration_results.json') as f:
                    existing_calib = json.load(f)
                print("Loaded prior calibration from calibration_results.json")
            except FileNotFoundError:
                raise RuntimeError(
                    "No calibration_results.json found — run full "
                    "--calibrate at least once before --calibrate-phase3."
                )
        calib = dict(existing_calib)
        # Stale phase-3/4 fields from the prior run shouldn't leak in.
        for k in ('synapse_test', 'synapse_combos', 'synapse_curves',
                  'output_path_diag'):
            calib.pop(k, None)
        calib.setdefault('synapse_test', {})

        # ADC stays on for the duration — firmware disables it at boot
        # to save power, and we need fast analog reads in phases 3/4.
        self.set_flag(FLAG_ADC_ENABLE)
        try:
            print("\nClearing board state (DACs, weight SR, output latches)...")
            self.quiesce(zero_weights=True, reset_output_latches=True)
            time.sleep(0.02)
            self._run_phases_3_and_4(calib)
        finally:
            self.unset_flag(FLAG_ADC_ENABLE)
        return calib

    def _run_phases_3_and_4(self, calib: dict) -> None:
        """Inner: Phase 3 + Phase 4 over an already-populated calib dict."""
        # Host-side output-path diagnostics (probes L2 wired-OR,
        # weight-SR loopback behavior, reset-line, and polarity).
        calib['output_path_diag'] = self.diagnose_output_path()
        self.quiesce(zero_weights=True, reset_output_latches=True)
        time.sleep(0.01)

        print("\n=== Phase 3: Synapse Verification ===")
        print("All-excitatory weights + DAC ch10 trigger, counting output spikes.\n")

        # Diagnostic: confirm ch10 actually drives MEAS_OUT high via
        # D1801..D1809. If the bus doesn't shift when we assert ch10,
        # there's a host/firmware plumbing bug and Phase 3 can't work.
        print("  [ch10 sanity] reading MEAS_OUT before/after asserting ch10:")
        self.quiesce(zero_weights=True)
        time.sleep(0.02)
        l1_off = self.read_measurement(0)
        self.load_dacs(ch10_drive_codes())
        time.sleep(0.02)
        l1_on = self.read_measurement(0)
        # Also dump raw output latch word while ch10 is still HIGH + all
        # weights zero, so we can see if the SR latches flip from
        # anything-other-than-the-ch10-drive.
        raw_ch10_on_noweights = self.read_output()
        self.quiesce(zero_weights=True)
        print(f"    ch10=0V  → L1 ADC = {l1_off}  (~{adc_to_volts(l1_off):.2f} V)")
        print(f"    ch10=HI  → L1 ADC = {l1_on}   (~{adc_to_volts(l1_on):.2f} V)")
        print(f"    raw output word with ch10=HI (no weights) = 0x{raw_ch10_on_noweights:04X}")
        if l1_on - l1_off < 50:
            print("    WARNING: MEAS_OUT barely moves when ch10 goes HIGH.")
            print("    Either ch10 isn't actually being driven (host plumbing"
                  " bug) or D1801..D1809 aren't forward-biasing the hidden"
                  " V_outs (physical). Phase 3/4 results will be unreliable.")
        else:
            print(f"    OK — ch10 drive lifts MEAS_OUT by ~{l1_on-l1_off} ADC counts.")

        # Dump raw output words from the actual all-excitatory sample run
        # so we can see what's really in the latches (before our bit
        # mapping / polarity inversion).
        test_weights = [self.weight_to_sr_byte(7)] * 90
        snap_dbg = self.sample_with_synapse_pulse(
            test_weights, num_samples=5, interval_us=500,
        )
        print(f"    raw output words (ch10=HI, all weights=+7): "
              f"{[f'0x{w:04X}' for w in snap_dbg]}")

        # Bypass the hidden layer entirely: hold the 9 hidden DACs at
        # their quiescent PNP-off state and instead drive DAC ch10 HIGH
        # to assert the synapse V_out rail directly. Every SR-enabled
        # synapse then sources current into its output membrane as if a
        # hidden neuron had just spiked — but under our direct,
        # deterministic control, independent of hidden-layer firing.
        snapshots = self.sample_with_synapse_pulse(
            test_weights, num_samples=50, interval_us=500,
        )
        spike_counts = accumulate_spike_counts(snapshots)

        print(f"  All-excitatory test: spike counts = {spike_counts}")
        calib['synapse_test']['all_exc_counts'] = spike_counts

        responding = sum(1 for c in spike_counts if c > 0)
        print(f"  {responding}/10 output neurons responding")

        print("\n=== Phase 4: Combinatorial Synapse Calibration ===")
        print("DAC ch10 (V_OUT1) drives the anodes of D1801..D1809, which")
        print("forward-bias into all 9 hidden-neuron V_out stretch nets. So")
        print("when ch10 is asserted HIGH, every hidden neuron behaves as if")
        print("it had just spiked, and we can enable arbitrary subsets of the")
        print("90 synapses via the shift register. For each (subset, weight)")
        print("pattern we record the boolean fire/no-fire observation on each")
        print("output, then solve a per-output linear program for the 9")
        print("per-synapse currents s_{i,j} and the threshold T_j.\n")

        # DAC ch10 (V_OUT1 net per netlist) drives ALL 9 hidden V_outs via
        # D1801..D1809 — not just 4. So Phase 4 has access to every
        # (H_i, O_j) synapse in the 9×10 matrix.
        CH10_SOURCES = list(range(9))
        calib['synapse_combos'] = {}

        def set_subset_weights(subset, weight):
            """Build an SR byte array with only (src→output_j) synapses
            enabled for src in `subset`, at the given weight, for ALL
            output neurons j. Returns the 90-byte pattern."""
            bytes_out = [SR_IDLE_BYTE] * NUM_SYNAPSES
            wbyte = self.weight_to_sr_byte(weight)
            for src in subset:
                for j in range(10):
                    bytes_out[phys_byte(j, src)] = wbyte
            return bytes_out

        def measure_fire_pattern(subset, weight, n_samples=50, interval_us=500):
            """Enable all (src→O*) synapses for src ∈ subset at the given
            weight, pulse ch10 HIGH, sample, return (fired_bool, counts)
            where `counts[j]` is the number of samples out of `n_samples`
            where O_j's latch was captured in the SPIKED state."""
            sr_bytes = set_subset_weights(subset, weight)
            snaps = self.sample_with_synapse_pulse(
                sr_bytes, num_samples=n_samples, interval_us=interval_us,
            )
            counts = [0] * 10
            for word in snaps:
                bits = extract_output_spikes(word)
                for j in range(10):
                    counts[j] += bits[j]
            fired = [c > 0 for c in counts]
            return fired, counts

        # --- 4.1 Single-subset threshold sweep ---
        # Enable all 9 sources at a given weight, sweep weight 1..7,
        # record which outputs fire at each step. The first weight at
        # which output O_j fires is an upper bound on T_j / Σ s_{i,j}.
        print(f"  [4.1] All {len(CH10_SOURCES)} ch10 sources (H0..H{CH10_SOURCES[-1]}) enabled, weight sweep:")
        all_sources = tuple(CH10_SOURCES)
        weight_threshold = [None] * 10
        weight_sweep_counts = {}
        for weight in range(1, 8):
            fired, counts = measure_fire_pattern(all_sources, weight)
            weight_sweep_counts[weight] = counts
            mark = ''.join('●' if f else '·' for f in fired)
            print(f"    w={weight}: outputs {mark}   counts={counts}")
            for j in range(10):
                if fired[j] and weight_threshold[j] is None:
                    weight_threshold[j] = weight
        for j in range(10):
            if weight_threshold[j] is None:
                print(f"    O{j}: never fires at any weight "
                      f"(all {len(CH10_SOURCES)} sources @ +7 insufficient)")
            else:
                print(f"    O{j}: first fires at weight w={weight_threshold[j]}")
        calib['synapse_combos']['all_sources_weight_threshold'] = weight_threshold

        # --- 4.2 Leave-one-out at weight 7 ---
        # Drop each source in turn, keep the other 3 at +7. If O_j still
        # fires, the dropped source is "non-essential" (its contribution
        # doesn't push past the threshold). Gives a per-source necessity
        # map at the 3-source drive level.
        print("\n  [4.2] Leave-one-out @ w=+7 (which sources are necessary):")
        leave_one_out = {}
        for dropped in CH10_SOURCES:
            remaining = tuple(s for s in CH10_SOURCES if s != dropped)
            fired, counts = measure_fire_pattern(remaining, 7)
            leave_one_out[dropped] = {'fired': fired, 'counts': counts}
            mark = ''.join('●' if f else '·' for f in fired)
            print(f"    drop H{dropped} (keep {remaining}): {mark}")
        calib['synapse_combos']['leave_one_out_w7'] = leave_one_out

        # --- 4.3 Single-source @ w=+7 ---
        # Is any single source strong enough alone? Almost certainly no
        # (1 synapse ≈ 3 µA << leak), but it's cheap to verify.
        print("\n  [4.3] Single source only @ w=+7:")
        single_source = {}
        for src in CH10_SOURCES:
            fired, counts = measure_fire_pattern((src,), 7)
            single_source[src] = {'fired': fired, 'counts': counts}
            mark = ''.join('●' if f else '·' for f in fired)
            print(f"    only H{src}: {mark}")
        calib['synapse_combos']['single_source_w7'] = single_source

        # --- 4.4 Pairwise @ w=+7 ---
        # All 6 pairs. Gives finer bounds on pairwise synapse sums.
        print("\n  [4.4] Pairs of sources @ w=+7:")
        from itertools import combinations
        pairs = {}
        for pair in combinations(CH10_SOURCES, 2):
            fired, counts = measure_fire_pattern(pair, 7)
            pairs[pair] = {'fired': fired, 'counts': counts}
            mark = ''.join('●' if f else '·' for f in fired)
            print(f"    H{pair[0]}+H{pair[1]}: {mark}")
        calib['synapse_combos']['pairs_w7'] = {
            f"{a}_{b}": v for (a, b), v in pairs.items()
        }

        # Legacy field for compatibility with downstream tools.
        calib['synapse_curves'] = {}

        # --- 4.5 Inhibitory smoke test ---
        # Set one exc (+7) synapse that fires O_j, then add a same-
        # magnitude inhibitory from another source. Inhibitory should
        # subtract current; if it drops the total below T_j the firing
        # should stop.
        print("\n  [4.5] Inhibitory smoke test:")
        # Find an output that fires with just 2 sources at +7 — pairs
        # is now dict-of-dicts (fired, counts), so dereference ['fired'].
        test_output = None
        test_pair = None
        for pair in combinations(CH10_SOURCES, 2):
            fired = pairs[pair]['fired']
            for j in range(10):
                if fired[j]:
                    test_output = j
                    test_pair = pair
                    break
            if test_output is not None:
                break

        if test_output is None:
            print("    (no pair fires any output — skipping inhibitory test)")
        else:
            src_a, src_b = test_pair
            # Try adding a 3rd source at −7 inhibitory
            inh_source = next(s for s in CH10_SOURCES if s not in test_pair)
            sr_bytes = [SR_IDLE_BYTE] * NUM_SYNAPSES
            sr_bytes[phys_byte(test_output, src_a)] = self.weight_to_sr_byte(7)
            sr_bytes[phys_byte(test_output, src_b)] = self.weight_to_sr_byte(7)
            snap_exc = self.sample_with_synapse_pulse(sr_bytes, num_samples=50, interval_us=500)
            exc_fired = any(extract_output_spikes(w)[test_output] for w in snap_exc)
            print(f"    H{src_a}+H{src_b} @ +7 → O{test_output}: {'fires' if exc_fired else 'quiet'}")

            sr_bytes[phys_byte(test_output, inh_source)] = self.weight_to_sr_byte(-7)
            snap_both = self.sample_with_synapse_pulse(sr_bytes, num_samples=50, interval_us=500)
            both_fired = any(extract_output_spikes(w)[test_output] for w in snap_both)
            print(f"    +H{inh_source} @ −7 inhibitory: {'still fires' if both_fired else 'suppressed'}")

            calib['synapse_test']['inh_test'] = {
                'test_output': test_output,
                'exc_sources': list(test_pair),
                'inh_source': inh_source,
                'exc_fired': exc_fired,
                'both_fired': both_fired,
            }

        # --- 4.6 Algebraic solve for per-synapse currents s_ij and
        #         per-output thresholds T_j ---
        #
        # Model per output j:
        #     I_total(j, subset, weight) = weight × sum_{s ∈ subset} s_{s,j}
        #     fires_j(subset, weight) = 1  iff  I_total > T_j
        #
        # From every observation we get a linear constraint on
        # (s_{5,j}, s_{6,j}, s_{7,j}, s_{8,j}, T_j):
        #     fired → weight × Σ s_{k,j} > T_j   (strict, we use ≥ slack)
        #     quiet → weight × Σ s_{k,j} ≤ T_j
        #
        # We solve per-output via a small linear program with a slack
        # variable to find a feasible assignment. If the LP is infeasible
        # (contradictory observations — noisy or bad data), we report
        # bounds from the 4.1 weight sweep only.
        print("\n  [4.6] Solving per-output synapse currents + thresholds:")
        try:
            import numpy as np
            from scipy.optimize import linprog
        except ImportError:
            print("    scipy/numpy not available — skipping algebraic solve.")
        else:
            # Assemble every (subset, weight, fired) observation.
            obs = []
            for w, counts_w in weight_sweep_counts.items():
                fired_w = [c > 0 for c in counts_w]
                for j in range(10):
                    obs.append((all_sources, w, j, fired_w[j]))
            for dropped, d in leave_one_out.items():
                remaining = tuple(s for s in CH10_SOURCES if s != dropped)
                for j in range(10):
                    obs.append((remaining, 7, j, d['fired'][j]))
            for src, d in single_source.items():
                for j in range(10):
                    obs.append(((src,), 7, j, d['fired'][j]))
            for pair, d in pairs.items():
                for j in range(10):
                    obs.append((pair, 7, j, d['fired'][j]))

            # Per-output LP. Variables are:
            #   [s_{CH10_SOURCES[0], j}, ..., s_{CH10_SOURCES[-1], j},
            #    T_j,
            #    slack]
            # slack ≥ 0 absorbs contradictions. Minimise slack.
            n_sources = len(CH10_SOURCES)
            src_index = {s: i for i, s in enumerate(CH10_SOURCES)}
            solved = {}
            for j in range(10):
                A_ub, b_ub = [], []
                for (subset, weight, tgt_j, fired_bool) in obs:
                    if tgt_j != j:
                        continue
                    coefs = [0.0] * n_sources
                    for s in subset:
                        if s in src_index:
                            coefs[src_index[s]] = float(weight)
                    # Model: I = weight × Σ s_{s,j} for s ∈ subset
                    #   fired → I > T_j → -I + T_j ≤ slack
                    #   quiet → I ≤ T_j →  I - T_j ≤ slack
                    if fired_bool:
                        row = [-c for c in coefs] + [1.0, -1.0]
                    else:
                        row = list(coefs) + [-1.0, -1.0]
                    A_ub.append(row); b_ub.append(0.0)

                if not A_ub:
                    continue

                c_vec = [0.0] * n_sources + [0.0, 1.0]
                bounds = [(0, None)] * n_sources + [(0, None), (0, None)]
                res = linprog(c=c_vec, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                              bounds=bounds, method='highs')
                if res.success:
                    s_vec = res.x[:n_sources]
                    T, slack = res.x[n_sources], res.x[n_sources + 1]
                    solved[j] = {
                        's': {CH10_SOURCES[k]: float(s_vec[k]) for k in range(n_sources)},
                        'T': float(T),
                        'slack': float(slack),
                        'feasible': slack < 1e-6,
                    }
                else:
                    solved[j] = {'error': res.message}

            # Print header with all source columns
            header = "    O "
            for s in CH10_SOURCES:
                header += f"{'s['+str(s)+']':>8} "
            header += f"{'T':>8} {'slack':>8}"
            print(header)
            for j in range(10):
                r = solved.get(j)
                if not r:
                    print(f"    O{j:<2}  (no data)")
                elif 'error' in r:
                    print(f"    O{j:<2}  LP failed: {r['error']}")
                else:
                    row = f"    O{j:<2}"
                    for s in CH10_SOURCES:
                        row += f" {r['s'][s]:8.3f}"
                    tag = "" if r['feasible'] else "  ←contradictory (noisy)"
                    row += f" {r['T']:8.3f} {r['slack']:8.3f}{tag}"
                    print(row)
            calib['synapse_combos']['solved'] = solved

        # Final cleanup: drive the board into a fully-idle state so the
        # user isn't left with oscillating neurons. ch 10 synapse rail
        # back to 0 V, all weight bits cleared, output latches reset.
        print("\nQuiescing board (all DACs idle, all weights zero, latches reset)...")
        self.quiesce(zero_weights=True, reset_output_latches=True)

        print("\n=== Phases 3–4 Complete ===")

    def _compute_channel_scale(self, curve: list[tuple], target_time_us: int) -> float:
        """Compute the DAC scale factor from a calibration curve.

        Finds the V_DAC that produces a spike at approximately target_time_us,
        then computes what scale maps gilgamesh threshold=1.0 to that voltage.
        """
        # Find two points bracketing the target time
        for i in range(len(curve) - 1):
            v1, t1 = curve[i]
            v2, t2 = curve[i + 1]
            if t1 is not None and t2 is not None:
                if t1 <= target_time_us <= t2 or t2 <= target_time_us <= t1:
                    # Linear interpolation
                    frac = (target_time_us - t1) / (t2 - t1) if t2 != t1 else 0.5
                    v_target = v1 + frac * (v2 - v1)
                    # scale = (v_target - V_BE) / (1.0)
                    # Because gilgamesh threshold=1.0 should map to this voltage
                    return max(0.01, v_target - V_BE)

        # Fallback: use the fastest spiking point
        for v, t in curve:
            if t is not None:
                return max(0.01, v - V_BE)

        # No spikes at all: use nominal
        return THETA_0 * R_SET_INPUT / R_LEAK


def main():
    parser = argparse.ArgumentParser(description='Tarski board interface')
    parser.add_argument('--port', required=True, help='Serial port (e.g., /dev/tty.usbserial-XXX)')
    parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--checkpoint', help='Gilgamesh checkpoint path')
    parser.add_argument('--dac-scale', type=float, default=1.16)
    parser.add_argument('--setup-dacs', action='store_true',
                        help='Program MCP4728 I2C addresses (requires jumper isolation)')
    parser.add_argument('--calibrate', action='store_true', help='Run L1 calibration')
    parser.add_argument('--calibrate-phase3', action='store_true',
                        help='Re-run Phases 3+4 only (synapse verification + '
                             'combinatorial calibration), reusing cached '
                             'dac_scales from calibration_results.json. '
                             'Skips the slow DAC channel-discovery + '
                             'response-curve sweeps.')
    parser.add_argument('--infer', action='store_true', help='Run inference on MNIST samples')
    parser.add_argument('--data-dir', default='../gilgamesh/data', help='MNIST data directory')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples to infer')
    parser.add_argument('--reprogram-dac', nargs=2, metavar=('OLD', 'NEW'),
                        help='Reprogram a single DAC address (e.g. --reprogram-dac 0x60 0x61). '
                             'Isolate the target DAC first.')
    parser.add_argument('--verify-dacs', action='store_true',
                        help='Smoke-test all 3 DACs by writing 0 to all 12 channels. '
                             'NAK means an address is wrong or a DAC is missing.')
    parser.add_argument('--infer-pixels', metavar='FILE',
                        help='Run one inference on a 6x6 normalised pixel array loaded from '
                             'FILE (.npy or whitespace-separated text, 36 floats). '
                             'Requires --checkpoint.')
    parser.add_argument('--set-dacs', metavar='CODES',
                        help='Write raw DAC codes and exit, leaving the outputs '
                             'held until the next command. CODES is a '
                             'comma-separated list of 1..12 12-bit values '
                             '(e.g. --set-dacs 0,0,0,0,2048,2048,2048,2048,4095). '
                             'Values can be decimal or 0x-hex, or "Nv" / "Nmv" '
                             'for volts relative to the 5V VDD reference.')
    parser.add_argument('--ch10-toggle', type=int, metavar='N', default=None,
                        help='Load w=+7 everywhere, then toggle ch10 '
                             'HIGH/LOW N times via I2C DAC writes (~12 Hz '
                             'from host due to serial + full-chip writes), '
                             'then read the output latches. Prefer '
                             '--ch10-burst for fast firmware-side toggling.')
    parser.add_argument('--ch10-burst', type=int, metavar='N', default=None,
                        help='Firmware-side fast ch10 toggle (requires fw '
                             '≥ 0.1.15). Loads w=+7 everywhere, then runs '
                             'the firmware burst handler to toggle ch10 '
                             'HIGH/LOW N times at ~kHz rates, then reads '
                             'the latch word. Use --ch10-burst-high-us and '
                             '--ch10-burst-low-us to tune the pulse timing.')
    parser.add_argument('--ch10-burst-high-us', type=int, default=200,
                        help='ch10 HIGH duration per cycle (µs, default 200).')
    parser.add_argument('--ch10-burst-low-us', type=int, default=200,
                        help='ch10 LOW duration per cycle (µs, default 200).')
    parser.add_argument('--l2-probe', action='store_true',
                        help='Debug tool for SR-latch investigation: runs '
                             'the same load-weights/set-ch10 sequence as '
                             '--pulse-test, but instead of reading the '
                             'latch word it samples the L2 wired-OR analog '
                             'bus (A7, averaged across all 10 output '
                             'membranes) before drive, at 100 ms and 1 s '
                             'after drive. Use this to confirm the output '
                             'membranes are actually being driven even '
                             'when the latches show nothing.')
    parser.add_argument('--pulse-test', action='store_true',
                        help='In one session: load all 90 synapses at w=+7, '
                             'drive ch10 HIGH (4 V), wait 5 s, read the '
                             'output latch word, then quiesce. Avoids the '
                             'DTR-reset problem of chaining separate CLI '
                             'calls.')
    parser.add_argument('--pulse-wait', type=float, default=5.0,
                        help='Seconds to hold ch10 HIGH before reading in '
                             '--pulse-test (default 5.0).')
    parser.add_argument('--weights-max', action='store_true',
                        help='Load every synapse at w=+7 and exit (leaves '
                             'the SR holding the pattern until overwritten).')
    parser.add_argument('--weights-zero', action='store_true',
                        help='Load every synapse at w=0 (SR_IDLE_BYTE) and '
                             'exit. This is the true idle pattern — bits 5/6 '
                             'HIGH so inhibitory NPNs are disengaged.')
    parser.add_argument('--read-output', action='store_true',
                        help='Read the output latch word and print raw + '
                             'decoded spike bits. NOTE: the firmware '
                             'ReadOutput handler pulses RESET_SR as a side '
                             'effect, so this also clears the SR latches.')
    parser.add_argument('--clear-latches', action='store_true',
                        help='Clear the output SR latches and synapse SR '
                             '(loads idle weights, then pulses RESET_SR via '
                             'a discarded read_output).')
    parser.add_argument('--verify-mapping', action='store_true',
                        help='Load one (output, hidden) synapse at a time '
                             'via phys_byte() and confirm the corrected SR '
                             'byte mapping. Prints per-hidden fire counts '
                             'on the observable output and analog Δ-counts '
                             'on the other 9 outputs.')
    parser.add_argument('--probe-topology', action='store_true',
                        help='Reverse-engineer SR byte→synapse mapping by '
                             'enabling one byte at a time and watching which '
                             'outputs fire. Only reliable for outputs with '
                             'a reworked 5.6 nF parallel stretch cap.')
    parser.add_argument('--probe-output', type=int, default=9,
                        help='Which output neuron to treat as observable '
                             'during --probe-topology (default 9 = O9, the '
                             'one with the 5.6 nF rework).')
    parser.add_argument('--scan-i2c', action='store_true',
                        help='Scan I2C addresses 0x60..0x67 and report which '
                             'MCP4728s (if any) respond.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print hex of every byte sent/received on the serial link.')
    args = parser.parse_args()

    def _parse_addr(s: str) -> int:
        return int(s, 0)

    board = TarskiBoard(args.port, verbose=args.verbose)

    try:
        # Handshake
        version = board.get_signature()
        print(f"Connected to Tarski board, firmware v{version}")

        if args.setup_dacs:
            print("\n=== MCP4728 I2C Address Programming ===")
            print("DAC #1 stays at the factory default 0x60 — we only program")
            print("DAC #2 (→ 0x61) and DAC #3 (→ 0x62). Isolate each target DAC")
            print("on the bus before programming it.\n")

            steps = [(2, 0x61), (3, 0x62)]
            for dac_num, addr in steps:
                input(f"Step: Connect ONLY DAC #{dac_num} (disconnect others). Press Enter...")
                success, msg = board.program_dac_address(0x60, addr)
                if success:
                    print(f"  DAC #{dac_num}: {msg}")
                else:
                    print(f"  DAC #{dac_num} FAILED: {msg}")
                    retry = input("  Retry? (y/n): ").strip().lower()
                    if retry == 'y':
                        success, msg = board.program_dac_address(0x60, addr)
                        print(f"  Retry: {'OK' if success else 'FAILED'} — {msg}")

            input("\nReconnect ALL DACs. Press Enter to verify...")
            try:
                board.load_dacs([0] * 12)
                print("  All 3 DACs responding on expected addresses — OK")
            except RuntimeError as e:
                print(f"  Verify FAILED: {e}")

        if args.reprogram_dac:
            old_addr = _parse_addr(args.reprogram_dac[0])
            new_addr = _parse_addr(args.reprogram_dac[1])
            print(f"\nReprogramming DAC 0x{old_addr:02X} → 0x{new_addr:02X}")
            print("Make sure only the target DAC is live on the bus.")
            success, msg = board.program_dac_address(old_addr, new_addr)
            print(f"  {'OK' if success else 'FAILED'}: {msg}")
            if not success:
                sys.exit(1)

        if args.set_dacs:
            def _parse_code(tok: str) -> int:
                tok = tok.strip().lower()
                if tok.endswith('mv'):
                    v = float(tok[:-2]) / 1000.0
                    return max(0, min(DAC_MAX_CODE, int(round(v / DAC_VREF * DAC_MAX_CODE))))
                if tok.endswith('v'):
                    v = float(tok[:-1])
                    return max(0, min(DAC_MAX_CODE, int(round(v / DAC_VREF * DAC_MAX_CODE))))
                return int(tok, 0)

            codes = [_parse_code(t) for t in args.set_dacs.split(',')]
            if not 1 <= len(codes) <= 12:
                print(f"--set-dacs: need 1..12 codes, got {len(codes)}")
                sys.exit(1)
            board.load_dacs(codes)
            print(f"\nHeld {len(codes)} DAC channels:")
            for i, c in enumerate(codes):
                v = c * DAC_VREF / DAC_MAX_CODE
                print(f"  ch{i}: code={c:4d}  (~{v:.3f} V at VDD ref)")
            print("\nOutputs will stay at these values until the next --set-dacs,")
            print("--calibrate, or --infer invocation (or board reset). Measure now.")

        if args.scan_i2c:
            print("\nScanning I2C bus for MCP4728 devices (0x60..0x67)...")
            present = board.scan_i2c()
            if not present:
                print("  No devices responded.")
                print("  Every MCP4728 is either powered off, physically")
                print("  disconnected, or using an address outside 0x60..0x67.")
                sys.exit(1)
            print(f"  Found {len(present)} device(s):")
            for addr in present:
                note = ""
                if addr == 0x60:
                    note = "  (expected: DAC #1 factory default)"
                elif addr == 0x61:
                    note = "  (expected: DAC #2)"
                elif addr == 0x62:
                    note = "  (expected: DAC #3)"
                else:
                    note = "  (UNEXPECTED — stray address)"
                print(f"    0x{addr:02X}{note}")
            expected = {0x60, 0x61, 0x62}
            found = set(present)
            missing = expected - found
            stray = found - expected
            if missing:
                print(f"\n  Missing: {', '.join(f'0x{a:02X}' for a in sorted(missing))}")
            if stray:
                print(f"  Stray:   {', '.join(f'0x{a:02X}' for a in sorted(stray))}")
                print("  A stray address means one of the DACs was programmed "
                      "to an unexpected value earlier. You can move it back "
                      "with --reprogram-dac STRAY_ADDR WANTED_ADDR "
                      "(isolate it on the bus first).")

        if args.verify_dacs:
            print("\nVerifying DAC bus (one MCP4728 at a time)...")
            # Test each MCP4728 individually by targeting a count that only
            # reaches that specific chip:
            #   n=1  hits DAC #1 (0x60) only
            #   n=5  hits DAC #1 + DAC #2 (0x60 + 0x61)
            #   n=9  hits all three
            # We narrow a failure to the missing chip by probing each count.
            probes = [
                (1, "DAC #1 (0x60)"),
                (5, "DAC #1 + #2 (0x60, 0x61)"),
                (9, "DAC #1 + #2 + #3 (0x60, 0x61, 0x62)"),
            ]
            first_failure = None
            for n, label in probes:
                try:
                    board.load_dacs([0] * n)
                    print(f"  {label}: OK")
                except RuntimeError as e:
                    print(f"  {label}: FAILED ({e})")
                    if first_failure is None:
                        first_failure = label
                    break
            if first_failure:
                print(f"\n  Verification failed at: {first_failure}")
                print("  The last chip in that probe is the one that isn't responding.")
                print("  Re-run --setup-dacs or --reprogram-dac for the missing DAC.")
                sys.exit(1)
            print("\n  All 3 DACs responding at their expected addresses.")

        if args.infer_pixels:
            if not args.checkpoint:
                print("--infer-pixels requires --checkpoint")
                sys.exit(1)
            path = args.infer_pixels
            if path.endswith('.npy'):
                pixels_norm = np.load(path).astype(np.float32).reshape(6, 6)
            else:
                pixels_norm = np.loadtxt(path, dtype=np.float32).reshape(6, 6)

            calib = None
            try:
                with open('calibration_results.json') as f:
                    calib = json.load(f)
            except FileNotFoundError:
                print("No calibration_results.json — using default DAC mapping.")

            cp = board.load_checkpoint(args.checkpoint, args.dac_scale)
            fc1_w = np.array(cp['weights']['fc1_weight'], dtype=np.float32)
            prediction, spike_counts = board.infer(
                pixels_norm, fc1_w, calib,
                num_samples=50, interval_us=500,
            )
            print(f"Prediction: {prediction}")
            print(f"Spike counts: {list(spike_counts)}")

        if args.ch10_burst is not None:
            n = args.ch10_burst
            hi = args.ch10_burst_high_us
            lo = args.ch10_burst_low_us
            print(f"\n=== ch10 Burst (firmware) N={n} hi={hi}µs lo={lo}µs ===")
            board.quiesce(zero_weights=True, reset_output_latches=True)
            wbyte = board.weight_to_sr_byte(7)
            board.load_weights([wbyte] * NUM_SYNAPSES)
            print(f"  Loaded 90 synapses at w=+7")
            # Make sure the non-ch10 DACs are at their quiescent state.
            codes = list(DAC_QUIESCENT_CODES)
            board.load_dacs(codes)
            print(f"  Other DACs set to quiescent (ch0..8 at DAC_MAX, ch9..11 at 0)")
            res = board.ch10_burst(n, high_us=hi, low_us=lo)
            rate = (n * 2) / (res['elapsed_us'] / 1e6) if res['elapsed_us'] > 0 else 0
            print(f"  Burst done: {n} cycles in {res['elapsed_us']} µs "
                  f"(~{rate:.0f} edges/s)")
            bits = extract_output_spikes(res['word'])
            bits_str = ''.join('●' if b else '·' for b in bits)
            print(f"  Raw output: 0x{res['word']:04X}")
            print(f"  Decoded O0..O9:  {bits_str}")
            print(f"  {sum(bits)}/10 latches fired")
            board.quiesce(zero_weights=True, reset_output_latches=True)

        if args.ch10_toggle is not None:
            n = args.ch10_toggle
            print(f"\n=== ch10 Toggle Test ({n} cycles) ===")
            board.quiesce(zero_weights=True, reset_output_latches=True)
            wbyte = board.weight_to_sr_byte(7)
            board.load_weights([wbyte] * NUM_SYNAPSES)
            print(f"  Loaded 90 synapses at w=+7")
            codes_lo = list(DAC_QUIESCENT_CODES)  # ch10 = 0
            codes_hi = list(DAC_QUIESCENT_CODES)
            codes_hi[10] = DAC_MAX_CODE
            import time as _t
            t0 = _t.time()
            for i in range(n):
                board.load_dacs(codes_hi)
                board.load_dacs(codes_lo)
            elapsed = _t.time() - t0
            rate = n / elapsed if elapsed > 0 else 0
            print(f"  Toggled ch10 HIGH/LOW {n} times in {elapsed:.2f} s "
                  f"(~{rate:.0f} Hz)")
            # Final read — don't clear before this
            word = board.read_output()
            bits = extract_output_spikes(word)
            bits_str = ''.join('●' if b else '·' for b in bits)
            print(f"  Raw output: 0x{word:04X}")
            print(f"  Decoded O0..O9:  {bits_str}")
            print(f"  {sum(bits)}/10 latches fired")
            board.quiesce(zero_weights=True, reset_output_latches=True)

        if args.l2_probe:
            print("\n=== L2 Analog Probe ===")
            board.quiesce(zero_weights=True, reset_output_latches=True)
            time.sleep(0.01)
            # Baseline: everything idle, no drive
            baseline = board.read_measurement(1)
            print(f"  baseline L2 (idle, ch10=0V):        ADC={baseline:4d}  "
                  f"(~{baseline*5.0/1023:.2f} V)")
            # Load w=+7 but leave ch10 at 0: no current should flow yet
            wbyte = board.weight_to_sr_byte(7)
            board.load_weights([wbyte] * NUM_SYNAPSES)
            time.sleep(0.01)
            preload = board.read_measurement(1)
            print(f"  weights=+7, ch10=0V:                ADC={preload:4d}  "
                  f"(~{preload*5.0/1023:.2f} V)")
            # Assert ch10 HIGH and sample L2 at several time points
            codes = list(DAC_QUIESCENT_CODES)
            codes[10] = DAC_MAX_CODE
            board.load_dacs(codes)
            time.sleep(0.001)
            t_1ms = board.read_measurement(1)
            print(f"  weights=+7, ch10=HIGH (t=1 ms):     ADC={t_1ms:4d}  "
                  f"(~{t_1ms*5.0/1023:.2f} V)")
            time.sleep(0.1)
            t_100ms = board.read_measurement(1)
            print(f"  weights=+7, ch10=HIGH (t=100 ms):   ADC={t_100ms:4d}  "
                  f"(~{t_100ms*5.0/1023:.2f} V)")
            time.sleep(1.0)
            t_1s = board.read_measurement(1)
            print(f"  weights=+7, ch10=HIGH (t=1.1 s):    ADC={t_1s:4d}  "
                  f"(~{t_1s*5.0/1023:.2f} V)")
            time.sleep(2.0)
            t_3s = board.read_measurement(1)
            print(f"  weights=+7, ch10=HIGH (t=3.1 s):    ADC={t_3s:4d}  "
                  f"(~{t_3s*5.0/1023:.2f} V)")
            # Drop ch10, see membranes decay
            board.load_dacs(list(DAC_QUIESCENT_CODES))
            time.sleep(0.01)
            after = board.read_measurement(1)
            print(f"  ch10 back to 0V (t=10 ms after):    ADC={after:4d}  "
                  f"(~{after*5.0/1023:.2f} V)")
            # Also check the final latch state for reference
            word = board.read_output()
            bits = extract_output_spikes(word)
            bits_str = ''.join('●' if b else '·' for b in bits)
            print(f"\n  latch word at end of run: 0x{word:04X}  "
                  f"decoded: {bits_str}")
            print(f"  {sum(bits)}/10 latches captured something")
            print()
            print("  Interpretation:")
            print("   - Baseline and preload ADC should be nearly identical")
            print("     (both have no ch10 drive).")
            print("   - At t=1 ms with full drive, L2 should jump up to show")
            print("     membranes charging; if it stays at baseline, no")
            print("     synapse current is reaching any membrane.")
            print("   - At t=100 ms, L2 reflects the time-average of 10")
            print("     output membranes. If neurons are firing + resetting")
            print("     repeatedly, L2 should oscillate around some mid")
            print("     value. If membranes stick at the rail, L2 sits high.")
            print("   - The final latch word tells us which (if any) output")
            print("     latches caught a spike across the full run.")
            board.quiesce(zero_weights=True, reset_output_latches=True)

        if args.pulse_test:
            print("\n=== Pulse Test ===")
            # Output neurons fire exactly once per ch10 rising edge under
            # w=+7 drive (the 27 µA synapse current overpowers the SPST
            # reset switch, so once the membrane latches high it stays
            # there). So we MUST clear the latches BEFORE asserting ch10,
            # not after — otherwise the reset destroys the one spike
            # every output produces. Let the wait then accumulate any
            # stretched pulse-capture the marginal 10 pF stretch caps
            # manage to squeeze into the 74HC02 SR latches.
            board.quiesce(zero_weights=True, reset_output_latches=True)
            wbyte = board.weight_to_sr_byte(7)
            board.load_weights([wbyte] * NUM_SYNAPSES)
            print(f"  [1] Loaded 90 synapses at w=+7 (byte 0x{wbyte:02X}).")
            # Latches cleared as a side effect of quiesce. DO NOT clear
            # again between here and the wait.
            codes = list(DAC_QUIESCENT_CODES)
            codes[10] = DAC_MAX_CODE
            board.load_dacs(codes)
            print(f"  [2] ch10 → DAC_MAX_CODE (~4 V); latches already clean.")
            print(f"  [3] Holding ch10 HIGH for {args.pulse_wait:.1f} s...")
            time.sleep(args.pulse_wait)
            word = board.read_output()
            bits = extract_output_spikes(word)
            bits_str = ''.join('●' if b else '·' for b in bits)
            print(f"  [4] Raw output: 0x{word:04X}  (binary {word:016b})")
            print(f"      Decoded O0..O9:  {bits_str}")
            print(f"      Per-output list: {bits}")
            print(f"      {sum(bits)}/10 latches fired")
            board.quiesce(zero_weights=True, reset_output_latches=True)
            print("  [5] Board quiesced.")

        if args.weights_max:
            wbyte = board.weight_to_sr_byte(7)
            board.load_weights([wbyte] * NUM_SYNAPSES)
            print(f"Loaded all 90 synapses at w=+7 (byte 0x{wbyte:02X}).")

        if args.weights_zero:
            wbyte = board.weight_to_sr_byte(0)
            board.load_weights([wbyte] * NUM_SYNAPSES)
            print(f"Loaded all 90 synapses at w=0 (byte 0x{wbyte:02X}).")

        if args.read_output:
            word = board.read_output()
            bits = extract_output_spikes(word)
            bits_str = ''.join('●' if b else '·' for b in bits)
            print(f"Raw output word: 0x{word:04X}  (binary {word:016b})")
            print(f"Decoded O0..O9:  {bits_str}")
            print(f"Per-output list: {bits}")

        if args.clear_latches:
            idle = board.weight_to_sr_byte(0)
            board.load_weights([idle] * NUM_SYNAPSES)
            board.read_output()  # side effect: pulses RESET_SR
            print("Cleared synapse SR (idle) + output SR latches (RESET_SR).")

        if args.verify_mapping:
            print("\nRunning mapping verification...")
            vm = board.verify_mapping(observable_output=args.probe_output)
            with open('mapping_verification.json', 'w') as f:
                json.dump(vm, f, indent=2, default=str)
            print("\nResults saved to mapping_verification.json")

        if args.probe_topology:
            print("\nRunning SR topology probe...")
            probe = board.probe_sr_topology(observable_output=args.probe_output)
            with open('topology_probe.json', 'w') as f:
                json.dump(probe, f, indent=2, default=str)
            print("\nProbe results saved to topology_probe.json")

        if args.calibrate_phase3:
            print("\nRunning Phase 3+4 partial calibration...")
            calib = board.partial_calibration()
            with open('calibration_results.json', 'w') as f:
                json.dump(calib, f, indent=2, default=str)
            print("\nCalibration saved to calibration_results.json")

        if args.calibrate:
            print("\nRunning full board calibration...")
            calib = board.full_calibration()
            with open('calibration_results.json', 'w') as f:
                json.dump(calib, f, indent=2, default=str)
            print("\nCalibration saved to calibration_results.json")
            print(f"Per-channel DAC scales: {[f'{s:.4f}' for s in calib['dac_scales']]}")

        if args.infer and args.checkpoint:
            # Load calibration if available
            calib = None
            try:
                with open('calibration_results.json') as f:
                    calib = json.load(f)
                print(f"\nLoaded calibration from calibration_results.json")
            except FileNotFoundError:
                print("\nNo calibration file found — using default DAC mapping.")
                print("Run --calibrate first for best accuracy.")

            print(f"Loading checkpoint: {args.checkpoint}")
            cp = board.load_checkpoint(args.checkpoint, args.dac_scale)

            # Load fc1 weights
            fc1_w = np.array(cp['weights']['fc1_weight'], dtype=np.float32)

            # Load MNIST test data
            import struct as st
            images_path = f"{args.data_dir}/t10k-images-idx3-ubyte"
            labels_path = f"{args.data_dir}/t10k-labels-idx1-ubyte"

            with open(images_path, 'rb') as f:
                magic, n, rows, cols = st.unpack('>4I', f.read(16))
                images = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)

            with open(labels_path, 'rb') as f:
                magic, n = st.unpack('>2I', f.read(8))
                labels = np.frombuffer(f.read(), dtype=np.uint8)

            # Downsample to 6x6 and normalize
            correct = 0
            for i in range(min(args.samples, len(images))):
                img = images[i].astype(np.float32)
                # Downsample 28x28 → 6x6 by float-scaled block averaging
                # (matches Rust MnistData::load exactly)
                scale = 28.0 / 6.0
                pixels_6x6 = np.zeros((6, 6), dtype=np.float32)
                for ty in range(6):
                    for tx in range(6):
                        y0 = int(ty * scale)
                        y1 = min(int((ty + 1) * scale), 28)
                        x0 = int(tx * scale)
                        x1 = min(int((tx + 1) * scale), 28)
                        pixels_6x6[ty, tx] = img[y0:y1, x0:x1].mean()
                # MNIST normalization
                pixels_norm = (pixels_6x6 / 255.0 - 0.1307) / 0.3081

                prediction, spike_counts = board.infer(
                    pixels_norm, fc1_w, calib,
                    num_samples=50, interval_us=500,  # 50×500µs = 25ms
                )
                label = labels[i]
                ok = '✓' if prediction == label else '✗'
                if prediction == label:
                    correct += 1
                top3 = sorted(range(10), key=lambda j: spike_counts[j], reverse=True)[:3]
                top3_str = ' '.join(f'{j}:{spike_counts[j]}' for j in top3)
                print(f"  #{i:>4}: label={label} pred={prediction} {ok}  [{top3_str}]")

            print(f"\nAccuracy: {correct}/{args.samples} "
                  f"({correct/args.samples*100:.1f}%)")

    finally:
        board.close()


if __name__ == '__main__':
    main()

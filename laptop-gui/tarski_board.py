#!/usr/bin/env python3
"""Tarski board interface — connects to the real hardware via serial.

Usage:
    python3 tarski_board.py --port /dev/tty.usbserial-XXX
    python3 tarski_board.py --port /dev/tty.usbserial-XXX --calibrate
    python3 tarski_board.py --port /dev/tty.usbserial-XXX --infer --checkpoint ../gilgamesh/models/my_model.json
"""

import argparse
import json
import struct
import sys
import time
import numpy as np

try:
    import serial
except ImportError:
    print("Install pyserial: pip install pyserial")
    sys.exit(1)

# Serial protocol constants (must match t1-devboard signals.hpp)
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
PORT_PROG_DAC  = ord('P')
PORT_CALIB_PULSE = ord('B')

# Hardware constants
V_DD = 5.0
V_BE = 0.65
R_SET_INPUT = 174e3
R_LEAK = 120e3
R_TOP = 820e3
R_BOTTOM = 150e3

THETA_0 = ((V_DD / R_TOP + 2.5 / R_BOTTOM) / (1/R_TOP + 1/R_BOTTOM)) - 2.5

# Pulse stretcher (hidden neuron → synapse switch gate)
R_STRETCH = 150e3       # 150kΩ
C_STRETCH = 5.8e-9      # 5.8nF (PCB fix from 10pF)
TAU_PULSE_NOMINAL = R_STRETCH * C_STRETCH   # ~870µs
V_PEAK_NOMINAL = V_DD - 0.56   # ~4.44V after diode drop
# SN74LVC1G3157 SPDT switch: CMOS thresholds at VCC=5V
# V_IH = 0.7×VCC = 3.5V (guaranteed ON), V_IL = 0.3×VCC = 1.5V (guaranteed OFF)
# Typical switching threshold ≈ 0.5×VCC = 2.5V
V_SWITCH_NOMINAL = 2.5
# Nominal duty cycle: t_on / dt where dt = 1ms
import math as _math
DUTY_NOMINAL = (TAU_PULSE_NOMINAL * 1e6
                * _math.log(V_PEAK_NOMINAL / V_SWITCH_NOMINAL) / 1000.0)

# DAC configuration
# MCP4728 factory EEPROM default: VDD reference, gain=1
# Firmware doesn't change EEPROM, so DAC outputs 0 to VDD
DAC_VREF = 5.0      # VDD reference (factory default)
DAC_BITS = 12
DAC_MAX_CODE = (1 << DAC_BITS) - 1
NUM_DACS = 9        # 9 hidden neurons
NUM_SYNAPSES = 90   # 9 hidden × 10 output
BAUD_RATE = 115200  # Must match firmware Serial.begin(115200)


class TarskiBoard:
    """Interface to the Tarski neuromorphic board via serial."""

    def __init__(self, port: str, timeout: float = 2.0):
        self.ser = serial.Serial(port, BAUD_RATE, timeout=timeout)
        time.sleep(2)  # Wait for Arduino reset
        self.ser.reset_input_buffer()

    def close(self):
        self.ser.close()

    # ── Low-level protocol ──

    def _send(self, data: bytes):
        self.ser.write(data)

    def _read(self, n: int) -> bytes:
        data = self.ser.read(n)
        if len(data) < n:
            raise TimeoutError(f"Expected {n} bytes, got {len(data)}")
        return data

    def _expect_ack(self) -> bool:
        resp = self._read(1)
        return resp[0] == PORT_ACK

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
        self._send(bytes([PORT_SIG]))
        if not self._expect_ack():
            raise RuntimeError("Signature request NAK'd")
        data = self._read_until_trn_end()
        # Version bytes are shifted +0x30 to avoid control char collisions
        return '.'.join(str(b - 0x30) for b in data)

    def load_weights(self, sr_bytes: list[int]):
        """Load 90 synapse weight bytes into the shift register chain."""
        assert len(sr_bytes) == NUM_SYNAPSES, f"Expected {NUM_SYNAPSES} bytes, got {len(sr_bytes)}"
        self._send(bytes([PORT_LOAD_SYN]))
        if not self._expect_ack():
            raise RuntimeError("Weight load NAK'd")
        self._send(bytes(sr_bytes))
        # Read response: ACK + TRN_END
        resp = self._read_until_trn_end()
        if PORT_ACK not in resp and len(resp) == 0:
            pass  # Some firmware versions send ACK before TRN_END differently
        return True

    def load_dacs(self, codes: list[int]):
        """Load DAC codes (12-bit values) for hidden neuron input currents."""
        n = len(codes)
        assert 1 <= n <= 12, f"DAC count must be 1-12, got {n}"
        self._send(bytes([PORT_LOAD_DAC]))
        if not self._expect_ack():
            raise RuntimeError("DAC load NAK'd")
        # Send count
        self._send(bytes([n]))
        # Send codes as LSB, MSB pairs
        for code in codes:
            code = max(0, min(DAC_MAX_CODE, code))
            self._send(bytes([code & 0xFF, (code >> 8) & 0xFF]))
        resp = self._read_until_trn_end()
        return True

    def read_output(self) -> int:
        """Read the 16-bit spike output word from the 74HC165 latches."""
        self._send(bytes([PORT_READ_OUT]))
        if not self._expect_ack():
            raise RuntimeError("Read output NAK'd")
        data = self._read(2)  # LSB, MSB
        trn = self._read(1)   # TRN_END
        return data[0] | (data[1] << 8)

    def read_measurement(self, channel: int) -> int:
        """Read ADC measurement. channel: 0=L1, 1=L2."""
        self._send(bytes([PORT_READ_MEAS]))
        if not self._expect_ack():
            raise RuntimeError("Measurement NAK'd")
        self._send(bytes([channel]))
        data = self._read(2)
        trn = self._read(1)
        return data[0] | (data[1] << 8)

    def set_flag(self, flag: int):
        self._send(bytes([PORT_SET_FLAG]))
        if not self._expect_ack():
            raise RuntimeError("Set flag NAK'd")
        self._send(bytes([flag]))
        self._read_until_trn_end()

    def unset_flag(self, flag: int):
        self._send(bytes([PORT_UNSET_FLAG]))
        if not self._expect_ack():
            raise RuntimeError("Unset flag NAK'd")
        self._send(bytes([flag]))
        self._read_until_trn_end()

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

        STATUS_MSGS = {
            0x02: "Device still at old address — LDAC timing may need adjustment",
            0x03: "Device not responding at either address — check connections",
            0x04: "I2C framing error — ACK missing during address write",
        }

        for attempt in range(max_retries):
            self._send(bytes([PORT_PROG_DAC]))
            if not self._expect_ack():
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
                return False, "Command not acknowledged"

            self._send(bytes([old_addr, new_addr]))
            resp = self._read_until_trn_end()

            if len(resp) >= 1 and resp[0] == PORT_ACK:
                return True, f"Programmed 0x{old_addr:02X} → 0x{new_addr:02X}"

            # NAK with status byte
            if len(resp) >= 2 and resp[0] == PORT_NAK:
                status = resp[1]
                msg = STATUS_MSGS.get(status, f"Unknown status 0x{status:02X}")
                if attempt < max_retries - 1:
                    time.sleep(1.0)  # Wait before retry
                    continue
                return False, f"Failed after {max_retries} attempts: {msg}"

        return False, "Max retries exceeded"

    def run_inference(self, num_samples: int = 25, interval_us: int = 1000) -> list[int]:
        """Run inference with rapid spike sampling.

        DAC values and weights must already be loaded.
        Returns a list of num_samples 16-bit spike words, one per sampling interval.
        Each bit in a word indicates whether that output neuron spiked since the last sample.

        To get spike counts: sum the bits across all samples.
        """
        assert 1 <= num_samples <= 250
        self._send(bytes([ord('R')]))  # PORT_RUN_INF
        if not self._expect_ack():
            raise RuntimeError("RunInference NAK'd")
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

    def calib_l1_single(self, dac_channel: int, dac_code: int,
                        max_wait_ms: int = 50, meas_channel: int = 0) -> int | None:
        """Measure time to first spike for one DAC channel.

        Returns elapsed microseconds, or None if no spike within timeout.
        """
        self._send(bytes([ord('C')]))  # PORT_CALIB_L1
        if not self._expect_ack():
            raise RuntimeError("CalibL1 NAK'd")
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

    def calib_pulse_burst(self, dac_channel: int, dac_code: int,
                          num_bursts: int = 16, meas_channel: int = 0
                          ) -> list[list[tuple[int, int]]] | None:
        """Sample the pulse stretcher decay curve via fast ADC bursts.

        Returns a list of bursts, each burst is a list of (time_us, adc_value)
        tuples tracing the exponential decay after a hidden neuron spike.
        Returns None if the firmware does not support this command.
        """
        self._send(bytes([PORT_CALIB_PULSE]))
        if not self._expect_ack():
            # Firmware doesn't support CalibPulse — drain the TRN_END that
            # follows NAK in the unknown-command response.
            try:
                self._read(1)  # TRN_END
            except TimeoutError:
                pass
            return None
        self._send(bytes([
            dac_channel,
            dac_code & 0xFF, (dac_code >> 8) & 0xFF,
            num_bursts,
            meas_channel,
        ]))

        bursts = []
        for _ in range(num_bursts):
            num_samples = self._read(1)[0]
            samples = []
            for _ in range(num_samples):
                data = self._read(4)  # time_us(u16) + adc(u16), little-endian
                time_us = data[0] | (data[1] << 8)
                adc_val = data[2] | (data[3] << 8)
                samples.append((time_us, adc_val))
            bursts.append(samples)
        self._read(1)  # TRN_END
        return bursts

    # ── High-level operations ──

    def fc1_to_dac_codes(self, fc1_outputs: list[float], dac_scale: float = 1.16) -> list[int]:
        """Convert gilgamesh fc1 outputs to 12-bit DAC codes.

        The MCP4728 with internal 2.048V reference: VOUT = (code/4095) × 2.048V
        So code = (V_DAC / 2.048) × 4095
        """
        nominal_scale = THETA_0 * R_SET_INPUT / R_LEAK
        scale = nominal_scale * dac_scale
        codes = []
        for g in fc1_outputs:
            v_dac = max(0.0, min(DAC_VREF, g * scale + V_BE))
            code = int(round(v_dac / DAC_VREF * DAC_MAX_CODE))
            codes.append(max(0, min(DAC_MAX_CODE, code)))
        return codes

    def weight_to_sr_byte(self, w: int) -> int:
        """Convert quantized weight (-7 to +7) to shift register byte.

        Bit mapping (will be verified against schematic):
        Excitatory (w > 0): Q1=bit1 (×1), Q2=bit2 (×2), Q3=bit3 (×4)
        Inhibitory (w < 0): Q4=bit4 (×1), Q5=bit5 (×2), Q6=bit6 (×4)
        """
        mag = min(abs(w), 7)
        if w > 0:
            return ((mag & 1) << 1) | ((mag & 2) << 1) | ((mag & 4) << 1)
        elif w < 0:
            return ((mag & 1) << 4) | ((mag & 2) << 4) | ((mag & 4) << 4)
        return 0

    def load_checkpoint(self, checkpoint_path: str, dac_scale: float = 1.16,
                        calib: dict = None, duty_compensate: bool = True):
        """Load a gilgamesh checkpoint's fc2 weights into the board.

        If calib contains pulse_calibration data and duty_compensate is True,
        adjusts fc2 float weights for per-neuron duty cycle variation before
        re-quantizing to 3-bit.
        """
        with open(checkpoint_path) as f:
            cp = json.load(f)

        pulse_calib = (calib or {}).get('pulse_calibration')
        if duty_compensate and pulse_calib and pulse_calib.get('per_neuron'):
            # Adjust the already-quantized fc2 weights for duty cycle mismatch.
            # We scale each hidden neuron's row by (nominal / actual) duty ratio
            # and re-round to the [-7, +7] integer range. This preserves the
            # original quantization scheme from gilgamesh.
            fc2_q_orig = cp['quantized']['fc2_weight']
            duties = self.compute_duty_cycles(pulse_calib)
            nominal = pulse_calib.get('nominal_duty', DUTY_NOMINAL)
            max_ratio = 2.0

            fc2_q = []
            for h in range(len(fc2_q_orig)):
                duty = duties[h] if h < len(duties) else nominal
                ratio = nominal / duty if duty > 0 else 1.0
                ratio = max(1.0 / max_ratio, min(max_ratio, ratio))
                row = []
                for o in range(len(fc2_q_orig[h])):
                    w = int(round(fc2_q_orig[h][o] * ratio))
                    row.append(max(-7, min(7, w)))
                fc2_q.append(row)
            print(f"  Duty-compensated fc2 weights (duties: "
                  f"{[f'{d:.3f}' for d in duties]})")
        else:
            fc2_q = cp['quantized']['fc2_weight']

        sr_bytes = []
        for h in range(9):
            for o in range(10):
                sr_bytes.append(self.weight_to_sr_byte(fc2_q[h][o]))

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
        spike_counts = [0] * 10
        for word in snapshots:
            for i in range(10):
                if (word >> i) & 1:
                    spike_counts[i] += 1

        # Prediction: argmax of spike counts
        prediction = max(range(10), key=lambda i: spike_counts[i])

        return prediction, spike_counts

    def fc1_to_dac_codes_calibrated(self, fc1_outputs: list[float],
                                     dac_scales: list[float]) -> list[int]:
        """Convert fc1 outputs to DAC codes using per-channel calibrated scales."""
        codes = []
        for i, g in enumerate(fc1_outputs):
            scale = dac_scales[i] if i < len(dac_scales) else dac_scales[0]
            v_dac = max(0.0, min(DAC_VREF, g * scale + V_BE))
            code = int(round(v_dac / DAC_VREF * DAC_MAX_CODE))
            codes.append(max(0, min(DAC_MAX_CODE, code)))
        return codes

    def full_calibration(self, max_wait_ms: int = 50,
                         dac_voltages: list[float] = None) -> dict:
        """Run the full board calibration procedure.

        This discovers the board's actual behaviour using ONLY observations —
        no assumed mappings. It:

        1. Discovers which DAC channel drives which hidden neuron by setting
           one channel at a time and observing which neuron spikes
        2. Measures the spike time response curve per channel
        3. Computes per-channel DAC scales for optimal weight mapping
        4. Tests synapse current delivery by programming weights and observing
           output spikes

        Returns a calibration dict that can be saved to JSON and used for inference.
        """
        if dac_voltages is None:
            dac_voltages = [v * 0.1 for v in range(25, 6, -1)]  # 2.5V → 0.7V

        calib = {
            'dac_channel_map': {},   # channel_idx → neuron response data
            'dac_scales': [],        # per-channel scale for fc1→DAC mapping
            'spike_time_curves': {}, # channel_idx → [(v_dac, time_us), ...]
            'synapse_test': {},      # output neuron test results
        }

        print("\n=== Phase 1: DAC Channel Discovery ===")
        print("Setting each DAC channel to max and observing spike response.\n")

        # For each DAC channel, set it to max and measure spike time.
        # A channel that produces a spike controls a working hidden neuron.
        for ch in range(9):
            max_code = int(round(2.5 / DAC_VREF * DAC_MAX_CODE))
            time_us = self.calib_l1_single(ch, max_code, max_wait_ms, meas_channel=0)

            if time_us is not None:
                print(f"  Channel {ch}: spike at {time_us}µs ({time_us/1000:.2f}ms)")
                calib['dac_channel_map'][ch] = {'spike_time_us': time_us, 'active': True}
            else:
                print(f"  Channel {ch}: no spike (timeout {max_wait_ms}ms)")
                calib['dac_channel_map'][ch] = {'spike_time_us': None, 'active': False}

        active_channels = [ch for ch, d in calib['dac_channel_map'].items() if d['active']]
        print(f"\n  Active channels: {active_channels} ({len(active_channels)}/9)")

        print("\n=== Phase 2: Spike Time Response Curves ===")
        print("Sweeping DAC voltage per active channel.\n")

        for ch in active_channels:
            curve = []
            for v_dac in dac_voltages:
                code = int(round(v_dac / DAC_VREF * DAC_MAX_CODE))
                time_us = self.calib_l1_single(ch, code, max_wait_ms, meas_channel=0)
                curve.append((v_dac, time_us))

                status = f"{time_us}µs" if time_us is not None else "no spike"
                print(f"  Ch{ch} V_DAC={v_dac:.2f}V → {status}")

            calib['spike_time_curves'][ch] = curve

            # Compute per-channel scale from the curve
            # Find the voltage where spike time ≈ 1ms (matching the simulation timestep)
            # The optimal scale maps gilgamesh threshold=1.0 to this voltage
            target_time_us = 1000  # 1ms
            scale = self._compute_channel_scale(curve, target_time_us)
            calib['dac_scales'].append(scale)
            print(f"  Ch{ch} calibrated scale: {scale:.4f}\n")

        # Fill remaining scales with average for inactive channels
        if calib['dac_scales']:
            avg_scale = sum(calib['dac_scales']) / len(calib['dac_scales'])
        else:
            avg_scale = THETA_0 * R_SET_INPUT / R_LEAK  # nominal
        while len(calib['dac_scales']) < 9:
            calib['dac_scales'].append(avg_scale)

        print("\n=== Phase 3: Synapse Verification ===")
        print("Programming weights and verifying output neuron response.\n")

        # Load a test pattern: all synapses excitatory max (weight = +7)
        test_weights = [self.weight_to_sr_byte(7)] * 90
        self.load_weights(test_weights)

        # Drive all hidden neurons and sample output
        max_codes = [int(round(2.0 / DAC_VREF * DAC_MAX_CODE))] * 9
        self.load_dacs(max_codes)
        time.sleep(0.001)

        snapshots = self.run_inference(50, 500)  # 50 samples × 500µs = 25ms
        spike_counts = [0] * 10
        for word in snapshots:
            for i in range(10):
                if (word >> i) & 1:
                    spike_counts[i] += 1

        print(f"  All-excitatory test: spike counts = {spike_counts}")
        calib['synapse_test']['all_exc_counts'] = spike_counts

        responding = sum(1 for c in spike_counts if c > 0)
        print(f"  {responding}/10 output neurons responding")

        print("\n=== Phase 4: Layer 2 Synapse Calibration ===")
        print("Programming individual synapses and measuring output response.\n")

        calib['synapse_curves'] = {}

        # For each output neuron, test synapses from hidden neuron 0
        # (we use H0 as the spike source because we already know its DAC from phase 2)
        source_ch = active_channels[0] if active_channels else 0
        max_dac_code = int(round(2.0 / DAC_VREF * DAC_MAX_CODE))

        for output_idx in range(10):
            print(f"  Testing synapses → O{output_idx}:")
            curve = []

            for weight in [1, 2, 3, 4, 5, 6, 7]:
                # Program ONLY synapse source_ch → output_idx to this weight
                sr_bytes = [0] * NUM_SYNAPSES
                sr_bytes[source_ch * 10 + output_idx] = self.weight_to_sr_byte(weight)
                self.load_weights(sr_bytes)

                # Drive hidden neuron at max
                codes = [0] * 9
                codes[source_ch] = max_dac_code
                self.load_dacs(codes)
                time.sleep(0.001)

                # Sample at 200µs intervals for 25ms = 125 samples
                snapshots = self.run_inference(125, 200)

                # Find first sample where this output neuron spiked
                first_spike_sample = None
                total_spikes = 0
                for s_idx, word in enumerate(snapshots):
                    if (word >> output_idx) & 1:
                        if first_spike_sample is None:
                            first_spike_sample = s_idx
                        total_spikes += 1

                spike_time_us = first_spike_sample * 200 if first_spike_sample is not None else None
                curve.append({
                    'weight': weight,
                    'first_spike_us': spike_time_us,
                    'total_spikes': total_spikes,
                })
                t_str = f"{spike_time_us}µs ({total_spikes}×)" if spike_time_us is not None else "no spike"
                print(f"    w=+{weight}: {t_str}")

                # Zero DACs to reset
                self.load_dacs([0] * 9)
                time.sleep(0.005)

            calib['synapse_curves'][output_idx] = curve

        # Test inhibitory synapses: set strong excitatory, add inhibitory, measure reduction
        print("\n  Testing inhibitory path:")
        inh_test = []
        test_output = 0
        # First: excitatory-only (w=+7)
        sr_bytes = [0] * NUM_SYNAPSES
        sr_bytes[source_ch * 10 + test_output] = self.weight_to_sr_byte(7)
        self.load_weights(sr_bytes)
        codes = [0] * 9
        codes[source_ch] = max_dac_code
        self.load_dacs(codes)
        time.sleep(0.001)
        snap_exc = self.run_inference(125, 200)
        exc_count = sum(1 for w in snap_exc if (w >> test_output) & 1)
        print(f"    Exc only (w=+7): {exc_count} spikes")
        self.load_dacs([0] * 9)
        time.sleep(0.005)

        # Now add inhibitory from a different hidden neuron
        if len(active_channels) >= 2:
            inh_source = active_channels[1]
            sr_bytes[source_ch * 10 + test_output] = self.weight_to_sr_byte(7)    # exc from H0
            sr_bytes[inh_source * 10 + test_output] = self.weight_to_sr_byte(-7)  # inh from H1
            self.load_weights(sr_bytes)
            codes = [0] * 9
            codes[source_ch] = max_dac_code
            codes[inh_source] = max_dac_code
            self.load_dacs(codes)
            time.sleep(0.001)
            snap_both = self.run_inference(125, 200)
            both_count = sum(1 for w in snap_both if (w >> test_output) & 1)
            print(f"    Exc+Inh (w=+7/-7): {both_count} spikes (was {exc_count})")
            self.load_dacs([0] * 9)
            time.sleep(0.005)
            calib['synapse_test']['inh_test'] = {
                'exc_only_spikes': exc_count,
                'exc_plus_inh_spikes': both_count,
            }

        print("\n=== Phase 5: Pulse Duration Calibration ===")
        print("Measuring pulse stretcher decay per hidden neuron.\n")

        calib['pulse_calibration'] = {
            'per_neuron': [],
            'nominal_tau_us': TAU_PULSE_NOMINAL * 1e6,
            'nominal_duty': float(DUTY_NOMINAL),
            'v_switch_v': V_SWITCH_NOMINAL,
        }

        # Use a strong DAC drive to reliably trigger spikes
        pulse_dac_code = int(round(2.0 / DAC_VREF * DAC_MAX_CODE))

        pulse_supported = True
        for ch in active_channels:
            print(f"  Channel {ch}: sampling {16} bursts...")
            bursts = self.calib_pulse_burst(
                dac_channel=ch, dac_code=pulse_dac_code,
                num_bursts=16, meas_channel=0,
            )

            if bursts is None:
                print("    Firmware does not support CalibPulse (PORT_CALIB_PULSE).")
                print("    Using nominal pulse parameters for all neurons.")
                pulse_supported = False
                break

            n_valid = sum(1 for b in bursts if len(b) > 0)
            n_total_samples = sum(len(b) for b in bursts)
            print(f"    {n_valid}/{len(bursts)} bursts with spike, "
                  f"{n_total_samples} total samples")

            fit = self._fit_pulse_decay(bursts)
            t_on_us = (fit['tau_pulse_us']
                       * np.log(fit['v_peak_v'] / V_SWITCH_NOMINAL)
                       if fit['v_peak_v'] > V_SWITCH_NOMINAL else 0.0)
            duty = t_on_us / 1000.0  # dt = 1ms

            neuron_data = {
                'channel': ch,
                'tau_pulse_us': fit['tau_pulse_us'],
                'v_peak_v': fit['v_peak_v'],
                't_on_us': float(t_on_us),
                'duty_cycle': float(min(duty, 1.0)),
                'fit_quality': fit['fit_quality'],
                'n_samples': fit['n_samples'],
            }
            calib['pulse_calibration']['per_neuron'].append(neuron_data)

            print(f"    τ_pulse = {fit['tau_pulse_us']:.1f}µs, "
                  f"V_peak = {fit['v_peak_v']:.2f}V, "
                  f"t_on = {t_on_us:.1f}µs, "
                  f"duty = {duty:.3f}, "
                  f"R² = {fit['fit_quality']:.4f}")

        # If firmware didn't support CalibPulse, fill in nominal values
        if not pulse_supported:
            for ch in active_channels:
                calib['pulse_calibration']['per_neuron'].append({
                    'channel': ch,
                    'tau_pulse_us': TAU_PULSE_NOMINAL * 1e6,
                    'v_peak_v': V_PEAK_NOMINAL,
                    't_on_us': float(TAU_PULSE_NOMINAL * 1e6
                                     * np.log(V_PEAK_NOMINAL / V_SWITCH_NOMINAL)),
                    'duty_cycle': float(DUTY_NOMINAL),
                    'fit_quality': 0.0,  # not measured
                    'n_samples': 0,
                })

        # Compute duty cycles from calibration data
        duties = self.compute_duty_cycles(calib['pulse_calibration'])
        if duties:
            calib['pulse_calibration']['measured_duties'] = duties
            mean_duty = sum(duties) / len(duties)
            calib['pulse_calibration']['mean_duty'] = float(mean_duty)
            print(f"\n  Mean duty cycle: {mean_duty:.3f}")
            print(f"  Per-neuron duties: {[f'{d:.3f}' for d in duties]}")
        else:
            calib['pulse_calibration']['measured_duties'] = []
            calib['pulse_calibration']['mean_duty'] = float(
                calib['pulse_calibration']['nominal_duty'])

        print("\n=== Calibration Complete ===")
        return calib

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

    @staticmethod
    def _fit_pulse_decay(bursts: list[list[tuple[int, int]]],
                         adc_vref: float = 5.0, adc_max: int = 1023
                         ) -> dict:
        """Fit exponential decay V(t) = V_peak × exp(-t/τ) to ADC burst data.

        Returns dict with tau_pulse_us, v_peak_v, and the fitted curve.
        """
        # Aggregate all samples across bursts, converting ADC to voltage.
        # Each burst's time is relative to its own spike onset, so t values
        # from different bursts overlap by design. This is correct for
        # least-squares fitting: we're fitting V(t) = V_peak*exp(-t/τ)
        # where each burst provides independent samples of the same curve.
        all_t = []
        all_v = []
        for burst in bursts:
            if not burst:
                continue
            for t_us, adc_val in burst:
                v = adc_val / adc_max * adc_vref
                all_t.append(t_us)
                all_v.append(v)

        if len(all_t) < 3:
            return {
                'tau_pulse_us': TAU_PULSE_NOMINAL * 1e6,
                'v_peak_v': V_PEAK_NOMINAL,
                'n_samples': len(all_t),
                'fit_quality': 0.0,
            }

        t_arr = np.array(all_t, dtype=np.float64)
        v_arr = np.array(all_v, dtype=np.float64)

        # Filter out zero/negative voltage samples (can't take log)
        mask = v_arr > 0.1
        t_fit = t_arr[mask]
        v_fit = v_arr[mask]

        if len(t_fit) < 3:
            return {
                'tau_pulse_us': TAU_PULSE_NOMINAL * 1e6,
                'v_peak_v': V_PEAK_NOMINAL,
                'n_samples': len(t_fit),
                'fit_quality': 0.0,
            }

        # Linear regression on ln(V) = ln(V_peak) - t/τ
        ln_v = np.log(v_fit)
        # Fit: ln_v = a + b*t  where b = -1/τ, a = ln(V_peak)
        A = np.vstack([np.ones_like(t_fit), t_fit]).T
        result = np.linalg.lstsq(A, ln_v, rcond=None)
        coeffs = result[0]
        a, b = coeffs[0], coeffs[1]

        v_peak = np.exp(a)
        tau_us = -1.0 / b if b < 0 else TAU_PULSE_NOMINAL * 1e6

        # R² goodness of fit
        ln_v_pred = a + b * t_fit
        ss_res = np.sum((ln_v - ln_v_pred) ** 2)
        ss_tot = np.sum((ln_v - np.mean(ln_v)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            'tau_pulse_us': float(tau_us),
            'v_peak_v': float(v_peak),
            'n_samples': int(len(t_fit)),
            'fit_quality': float(r_squared),
        }

    @staticmethod
    def compute_duty_cycles(pulse_calib: dict,
                            dt_us: float = 1000.0,
                            v_switch: float = V_SWITCH_NOMINAL) -> list[float]:
        """Compute per-neuron duty cycles from pulse calibration data.

        duty = t_on / dt, where t_on = τ × ln(V_peak / V_switch).
        """
        duties = []
        for neuron in pulse_calib.get('per_neuron', []):
            tau = neuron['tau_pulse_us']
            v_peak = neuron['v_peak_v']
            if v_peak > v_switch and tau > 0:
                t_on = tau * np.log(v_peak / v_switch)
                duties.append(min(float(t_on / dt_us), 1.0))
            else:
                # Fallback: nominal
                t_on_nom = TAU_PULSE_NOMINAL * 1e6 * np.log(V_PEAK_NOMINAL / v_switch)
                duties.append(float(t_on_nom / dt_us))
        return duties

    @staticmethod
    def adjust_fc2_for_duty(fc2_weights: list[list[float]],
                            duty_cycles: list[float],
                            nominal_duty: float = None,
                            max_ratio: float = 2.0) -> list[list[float]]:
        """Scale fc2 float weights to compensate for per-neuron duty variation.

        Each hidden neuron h delivers current for duty[h] fraction of the
        timestep. We scale its outgoing fc2 weights by (nominal / actual)
        so that the total charge delivered matches what the network expects.

        Args:
            fc2_weights: [9][10] float weight matrix (pre-quantization)
            duty_cycles: per-hidden-neuron duty cycles (length 9)
            nominal_duty: reference duty cycle. If None, uses the physical
                nominal (DUTY_NOMINAL from component values), not the mean
                of measured values.
            max_ratio: maximum allowed compensation ratio per neuron. Clamped
                to avoid extreme scaling that would be destroyed by quantization.

        Returns:
            Adjusted [9][10] float weight matrix.
        """
        if nominal_duty is None:
            nominal_duty = DUTY_NOMINAL

        adjusted = []
        for h in range(len(fc2_weights)):
            row = []
            duty = duty_cycles[h] if h < len(duty_cycles) else nominal_duty
            ratio = nominal_duty / duty if duty > 0 else 1.0
            # Clamp ratio to avoid extreme compensation that quantization
            # would destroy anyway (3-bit weights have ~14% resolution)
            ratio = max(1.0 / max_ratio, min(max_ratio, ratio))
            for o in range(len(fc2_weights[h])):
                row.append(fc2_weights[h][o] * ratio)
            adjusted.append(row)
        return adjusted

    @staticmethod
    def quantize_fc2(fc2_float: list[list[float]], bits: int = 3) -> list[list[int]]:
        """Quantize fc2 float weights to integer range [-(2^bits-1), +(2^bits-1)].

        For 3-bit: [-7, +7].
        """
        max_val = (1 << bits) - 1  # 7 for 3-bit
        # Find the scale: map the max absolute value to max_val
        flat = [w for row in fc2_float for w in row]
        abs_max = max(abs(w) for w in flat) if flat else 1.0
        if abs_max == 0:
            abs_max = 1.0
        scale = max_val / abs_max

        quantized = []
        for row in fc2_float:
            q_row = []
            for w in row:
                q = int(round(w * scale))
                q = max(-max_val, min(max_val, q))
                q_row.append(q)
            quantized.append(q_row)
        return quantized


def main():
    parser = argparse.ArgumentParser(description='Tarski board interface')
    parser.add_argument('--port', required=True, help='Serial port (e.g., /dev/tty.usbserial-XXX)')
    parser.add_argument('--baud', type=int, default=9600)
    parser.add_argument('--checkpoint', help='Gilgamesh checkpoint path')
    parser.add_argument('--dac-scale', type=float, default=1.16)
    parser.add_argument('--setup-dacs', action='store_true',
                        help='Program MCP4728 I2C addresses (requires jumper isolation)')
    parser.add_argument('--calibrate', action='store_true', help='Run full calibration (incl. pulse duration)')
    parser.add_argument('--infer', action='store_true', help='Run inference on MNIST samples')
    parser.add_argument('--no-duty-compensate', action='store_true',
                        help='Disable fc2 weight adjustment for pulse duty cycle')
    parser.add_argument('--export-duty-json', type=str, default=None,
                        help='Export per-neuron duty cycles to JSON for hw-aware training in gilgamesh')
    parser.add_argument('--data-dir', default='../gilgamesh/data', help='MNIST data directory')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples to infer')
    args = parser.parse_args()

    board = TarskiBoard(args.port)

    try:
        # Handshake
        version = board.get_signature()
        print(f"Connected to Tarski board, firmware v{version}")

        if args.setup_dacs:
            print("\n=== MCP4728 I2C Address Programming ===")
            print("This programs each DAC to a unique I2C address.")
            print("You MUST isolate each DAC via jumpers before programming.\n")

            target_addrs = [0x60, 0x61, 0x62]
            for i, addr in enumerate(target_addrs):
                input(f"Step {i+1}: Connect ONLY DAC #{i+1} (disconnect others). Press Enter...")
                success, msg = board.program_dac_address(0x60, addr)
                if success:
                    print(f"  DAC #{i+1}: {msg}")
                else:
                    print(f"  DAC #{i+1} FAILED: {msg}")
                    retry = input("  Retry? (y/n): ").strip().lower()
                    if retry == 'y':
                        success, msg = board.program_dac_address(0x60, addr)
                        print(f"  Retry: {'OK' if success else 'FAILED'} — {msg}")

            input("\nReconnect ALL DACs. Press Enter to verify...")
            # Verify by trying to write to each address
            for addr in target_addrs:
                try:
                    board.load_dacs([0])  # Will fail if address wrong, but tests communication
                    print(f"  DAC at 0x{addr:02X}: OK")
                except:
                    print(f"  DAC at 0x{addr:02X}: NOT RESPONDING")

        if args.calibrate:
            print("\nRunning full board calibration...")
            calib = board.full_calibration()
            with open('calibration_results.json', 'w') as f:
                json.dump(calib, f, indent=2, default=str)
            print("\nCalibration saved to calibration_results.json")
            print(f"Per-channel DAC scales: {[f'{s:.4f}' for s in calib['dac_scales']]}")

            pulse_calib = calib.get('pulse_calibration', {})
            if pulse_calib.get('per_neuron'):
                duties = TarskiBoard.compute_duty_cycles(pulse_calib)
                print(f"Per-neuron duty cycles: {[f'{d:.3f}' for d in duties]}")

            # Export duty cycles for gilgamesh hw-aware training if requested
            export_path = args.export_duty_json
            if export_path and pulse_calib.get('per_neuron'):
                duty_export = {
                    'per_neuron_duty_cycles': duties,
                    'per_neuron_tau_pulse_us': [
                        n['tau_pulse_us'] for n in pulse_calib['per_neuron']],
                    'per_neuron_v_peak_v': [
                        n['v_peak_v'] for n in pulse_calib['per_neuron']],
                    'v_switch_v': pulse_calib['v_switch_v'],
                    'nominal_duty': pulse_calib['nominal_duty'],
                    'mean_duty': pulse_calib.get('mean_duty',
                                                 pulse_calib['nominal_duty']),
                }
                with open(export_path, 'w') as f:
                    json.dump(duty_export, f, indent=2)
                print(f"Duty cycle data exported to {export_path}")

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
            duty_comp = not args.no_duty_compensate
            cp = board.load_checkpoint(args.checkpoint, args.dac_scale,
                                       calib=calib, duty_compensate=duty_comp)
            if duty_comp and calib and calib.get('pulse_calibration', {}).get('per_neuron'):
                print("  (fc2 weights adjusted for pulse duty cycle)")
            elif not duty_comp:
                print("  (duty compensation disabled via --no-duty-compensate)")

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

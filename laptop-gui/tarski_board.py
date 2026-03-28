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

# Hardware constants
V_DD = 5.0
V_BE = 0.65
R_SET_INPUT = 174e3
R_LEAK = 120e3
R_TOP = 820e3
R_BOTTOM = 150e3

THETA_0 = ((V_DD / R_TOP + 2.5 / R_BOTTOM) / (1/R_TOP + 1/R_BOTTOM)) - 2.5

# DAC configuration
# MCP4728 factory EEPROM default: VDD reference, gain=1
# Firmware doesn't change EEPROM, so DAC outputs 0 to VDD
DAC_VREF = 5.0      # VDD reference (factory default)
DAC_BITS = 12
DAC_MAX_CODE = (1 << DAC_BITS) - 1
NUM_DACS = 9        # 9 hidden neurons
NUM_SYNAPSES = 90   # 9 hidden × 10 output
BAUD_RATE = 9600


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

    def load_checkpoint(self, checkpoint_path: str, dac_scale: float = 1.16):
        """Load a gilgamesh checkpoint's fc2 weights into the board."""
        with open(checkpoint_path) as f:
            cp = json.load(f)

        fc2_q = cp['quantized']['fc2_weight']
        sr_bytes = []
        for h in range(9):
            for o in range(10):
                sr_bytes.append(self.weight_to_sr_byte(fc2_q[h][o]))

        self.load_weights(sr_bytes)
        print(f"Loaded {len(sr_bytes)} synapse weights from {checkpoint_path}")
        return cp

    def infer(self, pixels: np.ndarray, fc1_weights: np.ndarray,
              dac_scale: float = 1.16, wait_ms: float = 25.0) -> tuple[int, int]:
        """Run a single inference.

        Args:
            pixels: 6x6 normalized pixel array (36 floats)
            fc1_weights: [36, 9] fc1 weight matrix from checkpoint
            dac_scale: DAC mapping scale factor
            wait_ms: time to wait for neurons to compute

        Returns:
            (prediction, spike_word) — predicted digit and raw 16-bit spike output
        """
        # fc1 weighted sum (laptop-side preprocessing)
        fc1_out = pixels.flatten() @ fc1_weights

        # Convert to DAC codes and send
        dac_codes = self.fc1_to_dac_codes(fc1_out.tolist(), dac_scale)
        self.load_dacs(dac_codes)

        # Wait for analog neurons to compute
        time.sleep(wait_ms / 1000.0)

        # Read output spikes
        spike_word = self.read_output()

        # Prediction: argmax of spike bits
        # Each bit represents whether output neuron i spiked
        spike_counts = [(spike_word >> i) & 1 for i in range(10)]
        prediction = max(range(10), key=lambda i: spike_counts[i])

        return prediction, spike_word

    def calibrate_l1(self, dac_voltages: list[float] = None,
                     max_wait_ms: float = 50.0) -> dict:
        """Run Layer 1 calibration: sweep DAC voltage per hidden neuron.

        For each neuron, sets a DAC voltage and measures whether a spike occurs
        within max_wait_ms using the MEAS_OUT pins.

        Returns dict of {neuron_idx: [(dac_voltage, adc_reading), ...]}.
        """
        if dac_voltages is None:
            dac_voltages = [v/10 for v in range(25, 6, -1)]  # 2.5V down to 0.7V

        # Enable measurement path
        self.set_flag(0)  # FLAG_MEAS_ENABLE

        results = {}
        for neuron_idx in range(9):
            points = []
            for v_dac in dac_voltages:
                # Set only this neuron's DAC, zero all others
                codes = [0] * 9
                codes[neuron_idx] = int(round(v_dac / V_DD * DAC_MAX_CODE))
                self.load_dacs(codes)

                # Wait for spike
                time.sleep(max_wait_ms / 1000.0)

                # Read measurement (L1 channel)
                adc = self.read_measurement(0)
                points.append((v_dac, adc))

                # Zero the DAC to reset neuron
                codes[neuron_idx] = 0
                self.load_dacs(codes)
                time.sleep(5.0 / 1000.0)  # Brief reset

            results[neuron_idx] = points
            print(f"  H{neuron_idx}: {len(points)} measurements")

        # Disable measurement path
        self.unset_flag(0)
        return results


def main():
    parser = argparse.ArgumentParser(description='Tarski board interface')
    parser.add_argument('--port', required=True, help='Serial port (e.g., /dev/tty.usbserial-XXX)')
    parser.add_argument('--baud', type=int, default=9600)
    parser.add_argument('--checkpoint', help='Gilgamesh checkpoint path')
    parser.add_argument('--dac-scale', type=float, default=1.16)
    parser.add_argument('--calibrate', action='store_true', help='Run L1 calibration')
    parser.add_argument('--infer', action='store_true', help='Run inference on MNIST samples')
    parser.add_argument('--data-dir', default='../gilgamesh/data', help='MNIST data directory')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples to infer')
    args = parser.parse_args()

    board = TarskiBoard(args.port)

    try:
        # Handshake
        version = board.get_signature()
        print(f"Connected to Tarski board, firmware v{version}")

        if args.calibrate:
            print("\nRunning L1 calibration...")
            results = board.calibrate_l1()
            with open('calibration_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print("Saved to calibration_results.json")

        if args.infer and args.checkpoint:
            print(f"\nLoading checkpoint: {args.checkpoint}")
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
                # Downsample 28x28 → 6x6 by averaging blocks
                h_blocks = np.array_split(img, 6, axis=0)
                small = np.array([np.array_split(hb, 6, axis=1) for hb in h_blocks])
                pixels_6x6 = np.array([[block.mean() for block in row] for row in small])
                # MNIST normalization
                pixels_norm = (pixels_6x6 / 255.0 - 0.1307) / 0.3081

                prediction, spike_word = board.infer(
                    pixels_norm, fc1_w, args.dac_scale
                )
                label = labels[i]
                ok = '✓' if prediction == label else '✗'
                if prediction == label:
                    correct += 1
                print(f"  Sample {i:>4}: label={label}, pred={prediction}, "
                      f"spikes=0x{spike_word:04X} {ok}")

            print(f"\nAccuracy: {correct}/{args.samples} "
                  f"({correct/args.samples*100:.1f}%)")

    finally:
        board.close()


if __name__ == '__main__':
    main()

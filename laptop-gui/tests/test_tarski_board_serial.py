"""Characterization (golden-master) tests for the SERIAL-touching methods of
tarski_board.py — the on-the-wire command serializers and response decoders.

Purpose: lock in the CURRENT behavior of the byte-level protocol so any future
change that alters the emitted TX bytes or the parsed RX decoding is caught by
a failing test. These close the coverage gaps the completeness critic flagged:
load_dacs, measure_spikes, read_output/run_inference, ch10_burst,
_expect_final_ack NAK detection, load_checkpoint round-trip,
program_dac_address status decode, get_signature/scan_i2c/calib_l1_single
decoders, the _read/_read_until_trn_end primitives, weight_to_sr_byte bit
layout, and the full infer DAC-code feed path.

Every golden value below was obtained by actually RUNNING the current code
against a fake serial that records TX and replays canned RX bytes — never
hand-derived. No production source file is modified; no real hardware is
touched.

Rules honored:
  * Deterministic: fixed canned bytes, fixed seeds, time.sleep stubbed.
  * Where current behavior looks like a bug it is locked ANYWAY, flagged
    with a CHARACTERIZATION comment.
"""

import json

import numpy as np
import pytest

import tarski_board as tb

ACK = tb.PORT_ACK        # 0x06
NAK = tb.PORT_NAK        # 0x15
TRN = tb.PORT_TRN_END    # 0x04


class FakeSerial:
    """Minimal stand-in for serial.Serial.

    `rx` is the byte stream the firmware would send back; reads consume it
    sequentially and short reads return fewer bytes (so _read raises
    TimeoutError exactly as it would on the wire). All writes are appended to
    `tx` so tests can assert the exact bytes the driver emitted.
    """

    def __init__(self, rx=b""):
        self.rx = bytearray(rx)
        self.tx = bytearray()
        self._i = 0

    def write(self, data):
        self.tx += bytes(data)

    def read(self, n):
        chunk = self.rx[self._i:self._i + n]
        self._i += len(chunk)
        return bytes(chunk)

    def reset_input_buffer(self):
        pass

    def close(self):
        pass


def make_board(rx=b""):
    """A TarskiBoard built via __new__ (no real __init__/serial), wired to a
    FakeSerial replaying `rx`."""
    b = tb.TarskiBoard.__new__(tb.TarskiBoard)
    b.ser = FakeSerial(rx)
    b.verbose = False
    return b


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """All serial methods here call time.sleep for hardware settle; stub it so
    tests are instant and never assert on wall-clock time."""
    monkeypatch.setattr(tb.time, "sleep", lambda *_a, **_k: None)


# ---------------------------------------------------------------------------
# load_dacs — DAC code serialization + per-code clamp + count byte
# ---------------------------------------------------------------------------

def test_load_dacs_emits_cmd_count_and_clamped_le_codes():
    # cmd 'D' (0x44), count=3, then per-code clamp then little-endian split:
    #   -5   -> clamp to 0     -> 00 00
    #   4096 -> clamp to 0x0FFF -> ff 0f
    #   2048 -> 0x0800         -> 00 08
    b = make_board([ACK, ACK, TRN])  # cmd-ACK, then final ACK + TRN
    b.load_dacs([-5, 4096, 2048])
    assert b.ser.tx.hex(' ') == "44 03 00 00 ff 0f 00 08"


def test_load_dacs_count_byte_matches_len():
    b = make_board([ACK, ACK, TRN])
    b.load_dacs([2048] * 9)
    # cmd 'D' then count 0x09 then 9 LE codes of 0x0800.
    assert b.ser.tx[0] == ord('D')
    assert b.ser.tx[1] == 9
    assert b.ser.tx[2:].hex(' ') == ' '.join(["00 08"] * 9)


@pytest.mark.parametrize("n", [0, 13])
def test_load_dacs_rejects_count_out_of_1_to_12(n):
    b = make_board()
    with pytest.raises(AssertionError):
        b.load_dacs([0] * n)


def test_load_dacs_returns_true_on_success():
    b = make_board([ACK, ACK, TRN])
    assert b.load_dacs([0]) is True


# ---------------------------------------------------------------------------
# measure_spikes — fixed-length 14-byte frame parse (the just-ported fix)
# ---------------------------------------------------------------------------

def _measure_spikes_frame():
    # ACK + 6×u16 LE + TRN. baseline_min=0x04 deliberately equals PORT_TRN_END
    # to prove the fixed-length parser does NOT truncate mid-packet.
    return [ACK, 10, 0, 4, 0, 250, 0, 99, 0, 1, 0, 0xFF, 0xFF, TRN]


def test_measure_spikes_fixed_length_frame_no_truncation_on_embedded_trn_end():
    b = make_board([ACK] + _measure_spikes_frame())
    assert b.measure_spikes(0, 100, drive_ch10=False) == {
        'baseline_count': 10,
        'baseline_min': 4,      # == 0x04 == TRN_END; survives the fixed parse
        'baseline_max': 250,
        'driven_count': 99,
        'driven_min': 1,
        'driven_max': 65535,    # 0xFFFF u16 max
    }


def test_measure_spikes_v0114_seven_param_payload_and_driven_mask_off():
    b = make_board([ACK] + _measure_spikes_frame())
    b.measure_spikes(0, 100, window_ms=50, meas_source=0, drive_ch10=False)
    # 7-byte payload: target, code_lo, code_hi, win_lo, win_hi, meas_source,
    # driven_mask. cmd 'K'(0x4b) then target=0, code=100(0x64), win=50(0x32),
    # meas_source=0, driven_mask=0x00.
    assert b.ser.tx.hex(' ') == "4b 00 64 00 32 00 00 00"


def test_measure_spikes_drive_ch10_sets_driven_mask_bit0():
    b = make_board([ACK] + _measure_spikes_frame())
    b.measure_spikes(0, 100, window_ms=50, meas_source=0, drive_ch10=True)
    # Last payload byte (driven_mask) flips from 0x00 to 0x01.
    assert b.ser.tx.hex(' ') == "4b 00 64 00 32 00 00 01"


def test_measure_spikes_raises_on_bad_ack():
    frame = [0x00, 10, 0, 4, 0, 250, 0, 99, 0, 1, 0, 0xFF, 0xFF, TRN]
    b = make_board([ACK] + frame)
    with pytest.raises(RuntimeError, match="bad ACK"):
        b.measure_spikes(0, 100)


def test_measure_spikes_raises_on_missing_trn_end():
    frame = [ACK, 10, 0, 4, 0, 250, 0, 99, 0, 1, 0, 0xFF, 0xFF, 0x00]
    b = make_board([ACK] + frame)
    with pytest.raises(RuntimeError, match="missing TRN_END"):
        b.measure_spikes(0, 100)


# ---------------------------------------------------------------------------
# read_output / run_inference — (LSB, MSB) word assembly + RESET_SR side effect
# ---------------------------------------------------------------------------

def test_read_output_assembles_lsb_msb_word():
    # cmd-ACK + LSB=0xC0 + MSB=0xFF + TRN -> 0xFF<<8 | 0xC0 == 0xFFC0.
    b = make_board([ACK, 0xC0, 0xFF, TRN])
    assert b.read_output() == 0xFFC0


def test_run_inference_assembles_two_words_le():
    # cmd-ACK + sample0 (0x0000) + sample1 (0x0040) + TRN.
    b = make_board([ACK, 0, 0, 0x40, 0x00, TRN])
    assert b.run_inference(2, 1000) == [0x0000, 0x0040]


def test_run_inference_emits_count_and_interval_le():
    b = make_board([ACK, 0, 0, 0x40, 0x00, TRN])
    b.run_inference(2, 1000)
    # cmd 'R'(0x52) then num_samples=2, interval=1000(0x03e8) LE.
    assert b.ser.tx.hex(' ') == "52 02 e8 03"


@pytest.mark.parametrize("num_samples", [0, 251])
def test_run_inference_rejects_out_of_bounds_sample_count(num_samples):
    b = make_board()
    with pytest.raises(AssertionError):
        b.run_inference(num_samples, 1000)


# ---------------------------------------------------------------------------
# ch10_burst — 7-byte response (word + elapsed_us LE assembly)
# ---------------------------------------------------------------------------

def test_ch10_burst_assembles_word_and_elapsed_us():
    # cmd-ACK + word(0x0040 LE) + elapsed(0x000003E8 LE = 1000) + TRN.
    b = make_board([ACK, 0x40, 0x00, 0xE8, 0x03, 0x00, 0x00, TRN])
    assert b.ch10_burst(5) == {'word': 64, 'elapsed_us': 1000}


def test_ch10_burst_raises_on_missing_trn_end():
    b = make_board([ACK, 0x40, 0x00, 0xE8, 0x03, 0x00, 0x00, 0x00])
    with pytest.raises(RuntimeError, match="missing TRN_END"):
        b.ch10_burst(5)


def test_ch10_burst_rejects_num_cycles_above_u16():
    b = make_board()
    with pytest.raises(AssertionError):
        b.ch10_burst(0x10000)


# ---------------------------------------------------------------------------
# _expect_final_ack — NAK detection (the explicit just-ported correctness fix)
# ---------------------------------------------------------------------------

def test_expect_final_ack_raises_on_bare_nak():
    b = make_board([NAK, TRN])
    with pytest.raises(RuntimeError, match=r"NAK'd by firmware"):
        b._expect_final_ack("op")


def test_expect_final_ack_appends_tail_hex_on_nak_with_status():
    b = make_board([NAK, 0x12, TRN])
    with pytest.raises(RuntimeError, match=r"\[15 12\]"):
        b._expect_final_ack("op")


def test_expect_final_ack_raises_on_empty_response():
    b = make_board([TRN])
    with pytest.raises(RuntimeError, match="empty response"):
        b._expect_final_ack("op")


def test_expect_final_ack_raises_on_unexpected_response():
    b = make_board([0x99, TRN])
    with pytest.raises(RuntimeError, match="unexpected response 99"):
        b._expect_final_ack("op")


def test_expect_final_ack_returns_none_on_ack():
    b = make_board([ACK, TRN])
    assert b._expect_final_ack("op") is None


# ---------------------------------------------------------------------------
# load_checkpoint — fc2_weight -> phys_byte serialization round-trip
# ---------------------------------------------------------------------------

def test_load_checkpoint_maps_fc2_through_phys_byte(tmp_path):
    # Deterministic fc2[h][o] = ((h*10+o) % 15) - 7, h in 0..8, o in 0..9.
    fc2 = [[((h * 10 + o) % 15) - 7 for o in range(10)] for h in range(9)]
    cp_data = {'quantized': {'fc2_weight': fc2}}
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(cp_data))

    b = make_board()
    captured = {}
    b.load_weights = lambda sr: captured.setdefault('sr', list(sr))

    returned = b.load_checkpoint(str(path))
    sr = captured['sr']

    assert len(sr) == tb.NUM_SYNAPSES == 90
    # phys_byte(0,0) == 89 carries fc2[0][0]; phys_byte(9,8) == 0 carries
    # fc2[8][9]. This pins the corrected 89-(9*output+hidden) mapping
    # end-to-end (replaces the earlier wrong hidden*10+output assumption).
    assert sr[89] == b.weight_to_sr_byte(fc2[0][0])
    assert sr[0] == b.weight_to_sr_byte(fc2[8][9])
    # The full mapping holds for every synapse.
    for h in range(9):
        for o in range(10):
            assert sr[tb.phys_byte(o, h)] == b.weight_to_sr_byte(fc2[h][o])
    # load_checkpoint returns the parsed checkpoint dict unchanged.
    assert returned == cp_data


# ---------------------------------------------------------------------------
# program_dac_address — status decode + ack_bits diagnostic + address asserts
# ---------------------------------------------------------------------------

def test_program_dac_address_success_message():
    b = make_board([ACK, ACK, TRN])  # cmd-ACK, then ACK + TRN response
    ok, msg = b.program_dac_address(0x60, 0x61)
    assert ok is True
    assert msg == "Programmed 0x60 → 0x61"


def test_program_dac_address_nak_with_ack_bits_diagnostic():
    # cmd-ACK then NAK + status 0x12 (LDAC timing) + ack_bits 0b1011 + TRN.
    b = make_board([ACK, NAK, 0x12, 0b1011, TRN])
    ok, msg = b.program_dac_address(0x60, 0x61, max_retries=1)
    assert ok is False
    assert "LDAC timing" in msg
    assert "ack_bits=0b1011" in msg
    assert "addr=OK, cmd1=OK, cmd2_ldac=NAK, cmd3=OK" in msg


def test_program_dac_address_unknown_status():
    b = make_board([ACK, NAK, 0x99, TRN])
    ok, msg = b.program_dac_address(0x60, 0x61, max_retries=1)
    assert ok is False
    assert "Unknown status 0x99" in msg


def test_program_dac_address_rejects_bad_old_address():
    b = make_board()
    with pytest.raises(AssertionError, match="Invalid old address"):
        b.program_dac_address(0x59, 0x61)


def test_program_dac_address_rejects_bad_new_address():
    b = make_board()
    with pytest.raises(AssertionError, match="Invalid new address"):
        b.program_dac_address(0x60, 0x68)


# ---------------------------------------------------------------------------
# get_signature / scan_i2c / calib_l1_single — response decoders
# ---------------------------------------------------------------------------

def test_get_signature_decodes_minus_0x30_byte_shift():
    # cmd-ACK + bytes 0x30,0x31,0x30 -> str(b-0x30) joined with '.' -> '0.1.0'.
    b = make_board([ACK, 0x30, 0x31, 0x30, TRN])
    assert b.get_signature() == "0.1.0"


def test_scan_i2c_decodes_bitmap_to_0x60_base_addresses():
    # cmd-ACK + [ACK, bitmap, TRN]; bitmap 0b00000101 -> bits 0,2 -> 0x60,0x62.
    b = make_board([ACK, ACK, 0b00000101, TRN])
    assert b.scan_i2c() == [0x60, 0x62]


def test_calib_l1_single_sentinel_maps_to_none():
    # cmd-ACK + u32 0xFFFFFFFF + TRN -> "no spike" sentinel -> None.
    b = make_board([ACK, 0xFF, 0xFF, 0xFF, 0xFF, TRN])
    assert b.calib_l1_single(0, 0) is None


def test_calib_l1_single_decodes_u32_le():
    # cmd-ACK + 0x00000010 LE + TRN -> 16 microseconds.
    b = make_board([ACK, 0x10, 0, 0, 0, TRN])
    assert b.calib_l1_single(0, 0) == 16


# ---------------------------------------------------------------------------
# _read / _read_until_trn_end — low-level read primitives
# ---------------------------------------------------------------------------

def test_read_raises_timeout_on_short_read():
    b = make_board([1, 2])
    with pytest.raises(TimeoutError, match="Expected 5 bytes, got 2"):
        b._read(5)


def test_read_until_trn_end_accumulates_until_0x04():
    b = make_board([0x41, 0x42, TRN])
    assert b._read_until_trn_end() == b"AB"


def test_read_until_trn_end_empty_when_first_byte_is_trn_end():
    b = make_board([TRN])
    assert b._read_until_trn_end() == b""


# ---------------------------------------------------------------------------
# weight_to_sr_byte — excitatory/inhibitory BIT-POSITION layout (binary)
# ---------------------------------------------------------------------------

def test_weight_to_sr_byte_plus7_bit_layout():
    # +7: magnitude into bits 1..3, INH_OFF (bits 4,5,6) HIGH -> 0b01111110.
    assert format(tb.TarskiBoard.weight_to_sr_byte(None, 7), '08b') == "01111110"


def test_weight_to_sr_byte_minus6_bit_layout():
    # -6: only bit 4 stays HIGH; bits 5,6 cleared to engage x2+x4 NPN inhibit.
    assert format(tb.TarskiBoard.weight_to_sr_byte(None, -6), '08b') == "00010000"


@pytest.mark.parametrize(
    "w,expected_bits",
    [
        (1, "01110010"),  # mag 1 -> bit 1 set, INH_OFF intact
        (2, "01110100"),  # mag 2 -> bit 2 set
        (4, "01111000"),  # mag 4 -> bit 3 set
    ],
)
def test_weight_to_sr_byte_excitatory_single_magnitude_bits(w, expected_bits):
    assert format(tb.TarskiBoard.weight_to_sr_byte(None, w), '08b') == expected_bits


@pytest.mark.parametrize(
    "w,expected_bits",
    [
        (-2, "01010000"),  # engage x2: clear bit 5 -> 0x50
        (-4, "00110000"),  # engage x4: clear bit 6 -> 0x30
    ],
)
def test_weight_to_sr_byte_inhibitory_single_magnitude_bits(w, expected_bits):
    assert format(tb.TarskiBoard.weight_to_sr_byte(None, w), '08b') == expected_bits


# ---------------------------------------------------------------------------
# fc1_to_dac_codes / fc1_to_dac_codes_calibrated — edge/boundary behavior
# ---------------------------------------------------------------------------

def test_fc1_to_dac_codes_empty_list_returns_empty():
    b = make_board()
    assert b.fc1_to_dac_codes([]) == []


def test_fc1_to_dac_codes_calibrated_empty_scales_raises_indexerror():
    # CHARACTERIZATION: current behavior, possibly a bug, locked to detect
    # change. With dac_scales=[] the i<len fallback uses dac_scales[0], which
    # is itself out of range, so the call blows up rather than degrading
    # gracefully.
    b = make_board()
    with pytest.raises(IndexError):
        b.fc1_to_dac_codes_calibrated([0.5], [])


# ---------------------------------------------------------------------------
# infer — full DAC-code feed path with non-zero pixels/fc1 (matmul -> load_dacs)
# ---------------------------------------------------------------------------

def test_infer_feeds_fc1_to_dac_codes_into_load_dacs():
    b = make_board()
    # Fixed inputs, no randomness. fc1 weights chosen so codes span the range
    # rather than clamping to all-zero (which the existing all-zero infer
    # tests never exercise).
    pixels = (np.arange(36).reshape(6, 6) / 35.0).astype(np.float32)
    fc1 = (np.arange(36 * 9).reshape(36, 9).astype(np.float32) - 162.0) / 5000.0

    expected_codes = b.fc1_to_dac_codes((pixels.flatten() @ fc1).tolist())
    # Golden snapshot of the codes computed inside infer.
    assert expected_codes == [3179, 3174, 3169, 3163, 3158, 3153, 3148, 3142, 3137]

    captured = {}
    b.load_dacs = lambda codes: captured.setdefault('codes', list(codes))
    b.run_inference = lambda num_samples, interval_us: [0xFFFF] * 5

    b.infer(pixels, fc1, num_samples=5, interval_us=500)

    # The DAC codes fed to the board inside infer equal fc1_to_dac_codes of the
    # matmul output — locking the full matmul-to-DAC bridge.
    assert captured['codes'] == expected_codes

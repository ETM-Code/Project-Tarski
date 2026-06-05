"""Characterization (golden-master) tests for tarski_board.py.

Purpose: lock in the CURRENT behavior of the board driver's pure functions
and the just-ported correctness fixes (synapse mapping, SR byte encoding,
PNP-inverted DAC codes, spike decode/polarity). Every golden value below was
obtained by actually running the current code, not hand-derived.

Rules honored:
  * No production source file is modified.
  * Tests are deterministic (no wall clock, no unseeded randomness, fixed
    inputs). Floats are compared with explicit tolerance.
  * Where current behavior looks like a bug it is locked ANYWAY, flagged
    with a CHARACTERIZATION comment.
"""

import numpy as np
import pytest

import tarski_board as tb


# ---------------------------------------------------------------------------
# phys_byte / byte_to_physical  (synapse SR-index mapping, chain-reversed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "output,hidden,expected",
    [
        (0, 0, 89),
        (0, 8, 81),
        (9, 8, 0),
        (9, 0, 8),
        (5, 4, 40),
        (1, 2, 78),
    ],
)
def test_phys_byte_golden_table(output, hidden, expected):
    # Locks the full formula 89 - (9*output + hidden).
    assert tb.phys_byte(output, hidden) == expected


def test_phys_byte_full_formula_holds_everywhere():
    for output in range(10):
        for hidden in range(9):
            assert tb.phys_byte(output, hidden) == 89 - (9 * output + hidden)


@pytest.mark.parametrize("output,hidden", [(10, 0), (0, 9), (-1, 0)])
def test_phys_byte_raises_on_out_of_range(output, hidden):
    with pytest.raises(AssertionError):
        tb.phys_byte(output, hidden)


@pytest.mark.parametrize(
    "byte_index,expected",
    [
        (0, (9, 8)),
        (89, (0, 0)),
        (1, (9, 7)),
        (80, (1, 0)),
        (45, (4, 8)),
    ],
)
def test_byte_to_physical_golden_table(byte_index, expected):
    assert tb.byte_to_physical(byte_index) == expected


def test_byte_to_physical_is_exact_inverse_of_phys_byte():
    for output in range(10):
        for hidden in range(9):
            idx = tb.phys_byte(output, hidden)
            assert tb.byte_to_physical(idx) == (output, hidden)


@pytest.mark.parametrize("byte_index", [-1, 90])
def test_byte_to_physical_raises_out_of_range(byte_index):
    with pytest.raises(AssertionError):
        tb.byte_to_physical(byte_index)


# ---------------------------------------------------------------------------
# weight_to_sr_byte  (SR byte encoding incl. inhibitory polarity)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "w,expected",
    [
        (0, 0x70),
        (1, 0x72),
        (2, 0x74),
        (3, 0x76),
        (4, 0x78),
        (5, 0x7A),
        (6, 0x7C),
        (7, 0x7E),
    ],
)
def test_weight_to_sr_byte_excitatory_table(board, w, expected):
    assert board.weight_to_sr_byte(w) == expected


@pytest.mark.parametrize(
    "w,expected",
    [
        (-1, 0x70),
        (-2, 0x50),
        (-3, 0x50),
        (-4, 0x30),
        (-5, 0x30),
        (-6, 0x10),
        (-7, 0x10),
    ],
)
def test_weight_to_sr_byte_inhibitory_table(board, w, expected):
    # CHARACTERIZATION: current behavior, possibly a bug, locked to detect
    # change. Odd inhibitory magnitudes round DOWN to the nearest even value
    # because bit 4 (the x1 inhibitory unit) is physically disconnected, so
    # one LSB of inhibitory precision is silently lost (-1 -> "no inhibition").
    assert board.weight_to_sr_byte(w) == expected


def test_weight_to_sr_byte_zero_equals_idle_byte(board):
    b = board.weight_to_sr_byte(0)
    assert b == 0x70
    assert b == tb.SR_IDLE_BYTE
    # Bits 5 and 6 HIGH => inhibitory stage fully disengaged.
    assert (b >> 5) & 1 == 1
    assert (b >> 6) & 1 == 1


@pytest.mark.parametrize(
    "w,expected",
    [
        (100, 0x7E),    # min(abs(w), 7) caps the excitatory magnitude at 7
        (-100, 0x10),   # ... and the inhibitory magnitude at 7 (-> even 6)
    ],
)
def test_weight_to_sr_byte_clamps_out_of_range_magnitude(board, w, expected):
    assert board.weight_to_sr_byte(w) == expected


# ---------------------------------------------------------------------------
# extract_output_spikes / OUTPUT_BIT_MAP  (spike decode + inverted polarity)
# ---------------------------------------------------------------------------

def test_extract_output_spikes_all_idle_word_is_all_spiked():
    # Inverted polarity: latch HIGH = idle, so word=0x0000 (all low) decodes
    # to every neuron "spiked".
    assert tb.extract_output_spikes(0x0000) == [1] * 10


def test_extract_output_spikes_all_high_word_is_no_spikes():
    assert tb.extract_output_spikes(0xFFFF) == [0] * 10


def test_extract_output_spikes_O0_maps_to_bit6():
    # bit 6 set HIGH => O0 reads idle (0), all others spiked.
    assert tb.extract_output_spikes(0x0040) == [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]


def test_extract_output_spikes_O9_maps_to_bit15():
    # bit 15 set HIGH => O9 reads idle (0), all others spiked.
    assert tb.extract_output_spikes(0x8000) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]


def test_extract_output_spikes_ignores_floating_low_6_nc_bits():
    # Low 6 bits (0x003F) are unmapped NC pins and must not affect output.
    assert tb.extract_output_spikes(0x003F) == [1] * 10


def test_output_bit_map_constant_locked():
    assert tb.OUTPUT_BIT_MAP == {
        0: 6, 1: 7, 2: 8, 3: 9, 4: 10,
        5: 11, 6: 12, 7: 13, 8: 14, 9: 15,
    }


# ---------------------------------------------------------------------------
# fc1_to_dac_codes  (PNP-inverted DAC codes)
# ---------------------------------------------------------------------------

def test_fc1_to_dac_codes_pnp_inverted_golden_vector(board):
    # Locks baseline offset, inversion, and BOTH clamps (0 and DAC_MAX_CODE).
    assert board.fc1_to_dac_codes([0.0, 1.0, -1.0, 2.5, 10.0]) == [
        3445, 1996, 4095, 0, 0,
    ]


def test_fc1_to_dac_codes_honors_dac_scale_headroom(board):
    assert board.fc1_to_dac_codes([0.5], dac_scale=1.16) == [2604]


def test_fc1_to_dac_codes_clamps_negative_input_to_max(board):
    # Negative g -> v_drop floors at 0 -> v_dac = DAC_VREF -> code = MAX.
    assert board.fc1_to_dac_codes([-100.0]) == [4095]


def test_fc1_to_dac_codes_clamps_huge_input_to_zero(board):
    assert board.fc1_to_dac_codes([1e6]) == [0]


# ---------------------------------------------------------------------------
# fc1_to_dac_codes_calibrated  (per-channel scales)
# ---------------------------------------------------------------------------

def test_fc1_to_dac_codes_calibrated_per_channel_scales(board):
    assert board.fc1_to_dac_codes_calibrated(
        [0.0, 1.0, 0.5], [1.0, 1.16, 2.0]
    ) == [3445, 1764, 1996]


def test_fc1_to_dac_codes_calibrated_index_past_end_falls_back_to_scales0(board):
    # CHARACTERIZATION: current behavior, possibly a bug, locked to detect
    # change. When the channel index runs past the end of dac_scales, the
    # code uses dac_scales[0] (NOT dac_scales[i]); here both 0.5 inputs map
    # to the same code because both fall back to scale 1.0.
    assert board.fc1_to_dac_codes_calibrated([0.5, 0.5], [1.0]) == [2720, 2720]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

def test_dac_quiescent_codes_constant_locked():
    assert tb.DAC_QUIESCENT_CODES == [4095] * 9 + [0, 0, 0]
    assert len(tb.DAC_QUIESCENT_CODES) == 12


def test_dac_and_sr_idle_constants_locked():
    assert tb.DAC_MAX_CODE == 4095
    assert tb.SR_IDLE_BYTE == 0x70


def test_theta_0_derived_constant_value():
    assert abs(tb.THETA_0 - 0.38659793814432986) < 1e-9


# ---------------------------------------------------------------------------
# _compute_channel_scale  (calibration curve interpolation / fallbacks)
# ---------------------------------------------------------------------------

def test_compute_channel_scale_linear_interpolation_between_bracketing_points(board):
    result = board._compute_channel_scale([(1.0, 100), (2.0, 200)], 150)
    assert result == pytest.approx(0.85, abs=1e-9)


def test_compute_channel_scale_fallback_to_fastest_spiking_point(board):
    # No bracket found (first point has t=None), so the first non-None point
    # (v=2.0) is used: max(0.01, 2.0 - V_BE) = 1.35.
    result = board._compute_channel_scale([(1.0, None), (2.0, 300)], 150)
    assert result == pytest.approx(1.35, abs=1e-9)


def test_compute_channel_scale_no_spike_nominal_fallback(board):
    # No spikes at all -> nominal = THETA_0 * R_SET_INPUT / R_LEAK.
    result = board._compute_channel_scale([(1.0, None)], 150)
    assert result == pytest.approx(0.5605670103092784, abs=1e-9)
    assert result == pytest.approx(
        tb.THETA_0 * tb.R_SET_INPUT / tb.R_LEAK, abs=1e-12
    )


# ---------------------------------------------------------------------------
# MNIST 28x28 -> 6x6 downsample + normalization
# ---------------------------------------------------------------------------

def test_mnist_downsample_and_normalization_formula():
    # NOTE: This reproduces the INLINE main() logic (not a callable function),
    # so it is a weaker lock than the rest of the suite — it duplicates the
    # source rather than calling it. If the source formula changes, update
    # both. Golden values captured by running the source's exact block.
    img = (np.arange(784) % 256).reshape(28, 28).astype(np.float32)

    scale = 28.0 / 6.0
    pixels_6x6 = np.zeros((6, 6), dtype=np.float32)
    for ty in range(6):
        for tx in range(6):
            y0 = int(ty * scale)
            y1 = min(int((ty + 1) * scale), 28)
            x0 = int(tx * scale)
            x1 = min(int((tx + 1) * scale), 28)
            pixels_6x6[ty, tx] = img[y0:y1, x0:x1].mean()
    pixels_norm = (pixels_6x6 / 255.0 - 0.1307) / 0.3081

    assert float(pixels_norm[0, 0]) == pytest.approx(0.129465, abs=1e-4)
    assert float(pixels_norm.sum()) == pytest.approx(41.996223, abs=1e-3)


# ---------------------------------------------------------------------------
# TarskiBoard.infer  (spike-count + argmax decode, mocked serial)
# ---------------------------------------------------------------------------

def _word_with_spiked(spiked_indices):
    """Build a 16-bit spike word in which exactly `spiked_indices` read as
    spiked (their mapped bit driven LOW) and everything else reads idle."""
    word = 0xFFFF
    for i in spiked_indices:
        word &= ~(1 << tb.OUTPUT_BIT_MAP[i])
    return word


def test_infer_spike_count_and_argmax_decode(board, monkeypatch):
    # Canned snapshots: O3 fires 3x, O5 2x, O0 1x. argmax -> 3.
    canned = [
        _word_with_spiked({3}),
        _word_with_spiked({3, 5}),
        _word_with_spiked({3}),
        _word_with_spiked({5}),
        _word_with_spiked({0}),
    ]
    board.load_dacs = lambda codes: None
    board.run_inference = lambda num_samples, interval_us: canned
    # Avoid the real settle delay; deterministic, no wall-clock assertion.
    monkeypatch.setattr(tb.time, "sleep", lambda *_a, **_k: None)

    pixels = np.zeros((6, 6), dtype=np.float32)
    fc1_weights = np.zeros((36, 9), dtype=np.float32)

    prediction, spike_counts = board.infer(pixels, fc1_weights)

    assert spike_counts == [1, 0, 0, 3, 0, 2, 0, 0, 0, 0]
    assert prediction == 3


def test_infer_argmax_tie_break_picks_lowest_index(board, monkeypatch):
    # CHARACTERIZATION: max(range(10), key=...) returns the FIRST index on a
    # tie, so when two neurons tie the LOWEST index wins. Lock that.
    canned = [
        _word_with_spiked({2}),
        _word_with_spiked({2, 7}),
        _word_with_spiked({7}),
    ]
    board.load_dacs = lambda codes: None
    board.run_inference = lambda num_samples, interval_us: canned
    monkeypatch.setattr(tb.time, "sleep", lambda *_a, **_k: None)

    pixels = np.zeros((6, 6), dtype=np.float32)
    fc1_weights = np.zeros((36, 9), dtype=np.float32)

    prediction, spike_counts = board.infer(pixels, fc1_weights)

    assert spike_counts == [0, 0, 2, 0, 0, 0, 0, 2, 0, 0]
    assert prediction == 2  # tie between 2 and 7 -> lowest index

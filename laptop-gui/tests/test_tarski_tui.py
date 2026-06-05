"""Characterization (golden-master) tests for tarski_tui.py pure logic.

Purpose: lock in the CURRENT behavior of the TUI's two deterministic, pure
pieces of logic that need no Textual event loop:

  * ``TarskiTUI._prepare_sample`` — the np.array_split-based 28->6 MNIST
    downsample + normalization, which DIVERGES from tarski_board.main()'s
    int(ty*scale) floor-block downsample.
  * ``PredictionDisplay.render`` — the spike-count bar renderer
    (bar_len = int(c/max_c*15) with a max_c==0 divide-guard).

Both methods are exercised by calling the REAL source functions UNBOUND on a
lightweight plain object that carries the attributes they read. This avoids
Textual's reactive/Node machinery (which requires a running app) while still
locking the genuine source code rather than a reproduction.

Every golden value was obtained by RUNNING the current code. No production
source file is modified.

Rules honored:
  * Deterministic: fixed integer-ramp input images, no randomness, no clock.
  * Where current behavior looks like a bug it is locked ANYWAY, flagged with
    a CHARACTERIZATION comment.
"""

import os
import sys

import numpy as np
import pytest

_COMPONENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _COMPONENT_ROOT not in sys.path:
    sys.path.insert(0, _COMPONENT_ROOT)

import tarski_tui as tt  # noqa: E402


class _Plain:
    """A bare attribute bag, so we can invoke bound methods unbound without
    constructing a full Textual App / Static (which needs an event loop)."""


# ---------------------------------------------------------------------------
# TarskiTUI._prepare_sample — np.array_split downsample (DIVERGES from board)
# ---------------------------------------------------------------------------

def test_prepare_sample_returns_none_when_no_images_loaded():
    holder = _Plain()
    holder.images = None
    assert tt.TarskiTUI._prepare_sample(holder, 0) is None


def test_prepare_sample_raises_valueerror_on_standard_28x28_mnist():
    # CHARACTERIZATION: current behavior, possibly a bug, locked to detect
    # change. Under the pinned numpy (2.x), np.array_split of a length-28 axis
    # into 6 produces UNEVEN blocks ([5,5,5,5,4,4]), so the nested
    # `np.array([... for hb in h_blocks])` has an inhomogeneous shape and
    # raises ValueError. The standard MNIST image is exactly 28x28, so the TUI
    # downsample path actually blows up on real input. (This is also why the
    # board-path downsample in tarski_board.main(), which uses int(ty*scale)
    # floor blocks, DIVERGES: for the same 28x28 input the board path produces
    # a finite normalized array, sum ~= 41.996, while this TUI path raises.)
    holder = _Plain()
    holder.images = (np.arange(784) % 256).reshape(1, 28, 28).astype(np.float32)
    with pytest.raises(ValueError):
        tt.TarskiTUI._prepare_sample(holder, 0)


def test_prepare_sample_divisible_shape_block_mean_and_normalization():
    # A 36x36 image splits EVENLY into 6 blocks of 6, so np.array stays
    # homogeneous and the block-mean + normalization math runs. This pins the
    # np.array_split block-mean and the (x/255 - 0.1307)/0.3081 normalization
    # constants. Golden values captured from the current code.
    holder = _Plain()
    holder.images = (np.arange(36 * 36) % 256).reshape(1, 36, 36).astype(np.float32)
    out = tt.TarskiTUI._prepare_sample(holder, 0)
    assert out.shape == (6, 6)
    assert float(out.sum()) == pytest.approx(42.47208786, abs=1e-3)
    assert float(out[0, 0]) == pytest.approx(0.75314867, abs=1e-4)


# ---------------------------------------------------------------------------
# PredictionDisplay.render — spike-count bar renderer (line 47-60)
# ---------------------------------------------------------------------------

def _render_prediction(prediction, true_label, spike_counts):
    holder = _Plain()
    holder.prediction = prediction
    holder.true_label = true_label
    holder.spike_counts = list(spike_counts)
    return tt.PredictionDisplay.render(holder)


def test_render_no_prediction_yet_when_prediction_negative():
    assert _render_prediction(-1, -1, [0] * 10) == "[dim]No prediction yet[/]"


def test_render_max_element_is_15_full_blocks_and_zero_is_15_empty():
    counts = [0, 5, 10, 0, 0, 0, 0, 0, 0, 0]
    out = _render_prediction(prediction=2, true_label=-1, spike_counts=counts)
    # Max element (i=2, c=10) -> bar_len 15 -> 15 filled blocks, 0 empty.
    assert "  [bold green]2: " + "█" * 15 + "░" * 0 + " 10[/]" in out
    # A zero element (i=3, c=0) -> bar_len 0 -> 15 empty blocks.
    assert "  [dim]3: " + "█" * 0 + "░" * 15 + " 0[/]" in out
    # A mid element (i=1, c=5) -> int(5/10*15) == 7 filled, 8 empty.
    assert "  [dim]1: " + "█" * 7 + "░" * 8 + " 5[/]" in out


def test_render_full_golden_string_for_counts():
    counts = [0, 5, 10, 0, 0, 0, 0, 0, 0, 0]
    out = _render_prediction(prediction=2, true_label=-1, spike_counts=counts)
    expected_bars = (
        "  [dim]0: " + "░" * 15 + " 0[/]\n"
        "  [dim]1: " + "█" * 7 + "░" * 8 + " 5[/]\n"
        "  [bold green]2: " + "█" * 15 + " 10[/]\n"
        "  [dim]3: " + "░" * 15 + " 0[/]\n"
        "  [dim]4: " + "░" * 15 + " 0[/]\n"
        "  [dim]5: " + "░" * 15 + " 0[/]\n"
        "  [dim]6: " + "░" * 15 + " 0[/]\n"
        "  [dim]7: " + "░" * 15 + " 0[/]\n"
        "  [dim]8: " + "░" * 15 + " 0[/]\n"
        "  [dim]9: " + "░" * 15 + " 0[/]\n"
    )
    # prediction==true_label is False here (true_label=-1) -> "blue" color and
    # no label suffix.
    assert out == f"[bold blue]  Prediction: 2[/]\n\n{expected_bars}"


def test_render_all_zero_counts_no_zero_division_and_all_empty_bars():
    # max_c guard: any(spike_counts) is False -> max_c defaults to 1, so
    # bar_len = int(0/1*15) == 0 for every row. No ZeroDivisionError.
    out = _render_prediction(prediction=0, true_label=-1, spike_counts=[0] * 10)
    for i in range(10):
        marker = "[bold green]" if i == 0 else "[dim]"
        assert f"  {marker}{i}: " + "░" * 15 + " 0[/]" in out


def test_render_correct_prediction_is_green_with_label_suffix():
    out = _render_prediction(prediction=3, true_label=3, spike_counts=[0] * 10)
    assert out.startswith("[bold green]  Prediction: 3[/]  (label: 3)")


def test_render_wrong_prediction_is_red_with_label_suffix():
    out = _render_prediction(prediction=3, true_label=5, spike_counts=[0] * 10)
    assert out.startswith("[bold red]  Prediction: 3[/]  (label: 5)")

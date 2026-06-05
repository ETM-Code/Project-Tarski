"""
Characterization tests for lif_neuron_generator.py.

Golden-master: locks the CURRENT generated netlist text and helper outputs.
All golden values were obtained by running the current code, never hand-derived.
"""
from __future__ import annotations

import copy

import pytest

import lif_neuron_generator as M
from conftest import golden


# ----------------------------- fixtures -----------------------------

@pytest.fixture
def default_cfg():
    return M.NeuronConfig.load("defaults/neuron_default.json")


# ----------------------- full snapshots -----------------------------

def test_generate_detailed_neuron_default_golden_snapshot(default_cfg):
    out = M.generate_detailed_neuron(default_cfg)
    assert out == golden("detailed_default.txt")
    # 46 lines locked (header + subckt + body + .ends).
    assert len(out.splitlines()) == 46


def test_generate_detailed_neuron_computed_resistors(default_cfg):
    out = M.generate_detailed_neuron(default_cfg)
    # divider_scale=5.0 -> Rvh=681k*5=3.405M -> :.3g -> 3.4e+06 ; Rvl=316k*5=1.58M.
    # beta=0.05/5.0=0.01 ; Rf from beta/g_div math -> 1.07e+08.
    assert "Rvh  vdd     vth_node 3.4e+06\n" in out
    assert "Rvl  vth_node vref    1.58e+06\n" in out
    assert "Rf   comp_out vth_node 1.07e+08\n" in out


def test_generate_fast_neuron_is_renamed_detailed(default_cfg):
    fast = M.generate_fast_neuron(default_cfg)
    detailed = M.generate_detailed_neuron(default_cfg)
    # Only the subckt header differs (_detailed -> _fast).
    assert fast == detailed.replace(
        ".subckt demo_neuron_detailed", ".subckt demo_neuron_fast", 1
    )
    assert fast == golden("fast_default.txt")
    header = fast.splitlines()[1]
    assert header == ".subckt demo_neuron_fast mem vref vdd comp_out analog_out sum"
    # Bodies identical apart from that one header line.
    fast_body = "\n".join(fast.splitlines()[2:])
    det_body = "\n".join(detailed.splitlines()[2:])
    assert fast_body == det_body


# ----------------------- analog output stage ------------------------

def test_generate_detailed_neuron_inverting_analog_stage(default_cfg):
    out = M.generate_detailed_neuron(default_cfg)
    # inverting, gain=2.0, R1=10k -> R2=R1*gain=20k.
    assert "R1_ana ana_in ana_inv_in 1e+04\n" in out
    assert "R2_ana analog_out ana_inv_in 2e+04\n" in out
    assert "Eana_opamp ana_opamp_out 0 0 ana_inv_in 1e5\n" in out
    assert "Bdiff ana_in 0 V = V(mem) - V(vref)\n" in out


def test_generate_detailed_neuron_noninverting_analog_stage(default_cfg):
    cfg = copy.deepcopy(default_cfg)
    cfg.analog_out.inverting = False
    cfg.analog_out.gain = 3.0
    out = M.generate_detailed_neuron(cfg)
    # non-inverting gain = 1+R2/R1 -> R2 = R1*(gain-1) = 10k*2 = 20k.
    assert "R2_ana analog_out ana_fb 2e+04\n" in out
    assert "R1_ana ana_fb 0 1e+04\n" in out
    assert "Eana_opamp ana_opamp_out 0 ana_in ana_fb 1e5\n" in out


def test_generate_detailed_neuron_analog_sign_negative(default_cfg):
    cfg = copy.deepcopy(default_cfg)
    cfg.analog_out.sign = -1
    out = M.generate_detailed_neuron(cfg)
    assert "Bdiff ana_in 0 V = V(vref) - V(mem)\n" in out
    assert "Bdiff ana_in 0 V = V(mem) - V(vref)\n" not in out


def test_generate_detailed_neuron_clamp_to_rails(default_cfg):
    cfg = copy.deepcopy(default_cfg)
    cfg.analog_out.clamp_to_rails = True
    out = M.generate_detailed_neuron(cfg)
    assert ".model DCLAMP D(Is=1e-15 N=1.8 Rs=1)\n" in out
    assert "Dcl_lo  0   analog_out DCLAMP\n" in out
    assert "Dcl_hi  analog_out vdd DCLAMP\n" in out
    # absent in default (clamp_to_rails False)
    default_out = M.generate_detailed_neuron(default_cfg)
    assert "DCLAMP" not in default_out


# ----------------------- reset path ---------------------------------

def test_generate_detailed_neuron_reset_disabled(default_cfg):
    cfg = copy.deepcopy(default_cfg)
    cfg.reset.enable = False
    out = M.generate_detailed_neuron(cfg)
    for line in out.splitlines():
        assert not line.startswith("Sreset")
        assert not line.startswith("Rreset")
        assert not line.startswith(".model SWMUX")
    # default (enabled) DOES emit them with JSON values.
    default_out = M.generate_detailed_neuron(default_cfg)
    assert "Rreset mem reset_node 200.0\n" in default_out
    assert "Sreset reset_node vref comp_out 0 SWMUX\n" in default_out
    assert (
        ".model SWMUX SW(Ron=10.0 Roff=1000000000.0 Vt=2.5 Vh=0.1)\n" in default_out
    )


# ----------------------- pulse stretch (main module routing) --------

def test_generate_detailed_neuron_pulse_stretch_enabled_main():
    # neuron_pulse_stretch.json has pulse_stretch.R_load_ohm; NeuronConfig.load
    # in THIS module filters it (unlike the dedicated pulse_stretch module).
    cfg = M.NeuronConfig.load("defaults/neuron_pulse_stretch.json")
    out = M.generate_detailed_neuron(cfg)
    assert out == golden("main_pulse_stretch_detailed.txt")
    # pulse-stretch branch + comp_shaped reset routing
    assert "Rcout comp_raw comp_shaped 10.0\n" in out
    assert "Ccout comp_shaped 0 4e-09\n" in out
    assert ".model DPW D(Is=1e-12 N=1.05 Rs=10)\n" in out
    assert "Dpw comp_shaped comp_out DPW\n" in out
    assert "Cpw comp_out 0 1e-07\n" in out
    assert "Rpw comp_out 0 20000.0\n" in out
    assert "Sreset reset_node vref comp_shaped 0 SWMUX\n" in out


# ----------------------- sanity checks ------------------------------

def test_lif_sanity_checks_clean_default(default_cfg):
    assert M.lif_sanity_checks(default_cfg) == []


def test_lif_sanity_checks_flags_low_vdd_and_threshold_window(default_cfg):
    cfg = copy.deepcopy(default_cfg)
    cfg.supplies.vdd = 1.0
    issues = M.lif_sanity_checks(cfg)
    assert issues == [
        "Comparator supply vdd=1.0 V outside 1.8–5.5 V.",
        "Threshold window [3.275,3.325] not inside [0,1.0].",
    ]


def test_lif_sanity_checks_flags_small_tau(default_cfg):
    cfg = copy.deepcopy(default_cfg)
    cfg.membrane.R_leak_ohm = 1000.0
    cfg.membrane.C_mem_F = 1e-8  # tau = 1e-5 < 1e-4
    issues = M.lif_sanity_checks(cfg)
    assert "Membrane tau=0.01 ms is very small." in issues


# ----------------------- config loading -----------------------------

def test_NeuronConfig_load_parses_default_json(default_cfg):
    cfg = default_cfg
    assert cfg.name == "demo_neuron"
    assert cfg.supplies.vdd == 5.0
    assert cfg.threshold.divider_scale == 5.0
    assert cfg.threshold.C_adapt_F == 2.2e-8
    assert cfg.analog_out.inverting is True
    assert cfg.bias_currents.tia_A == 2e-5
    # pulse_stretch absent in default json -> default (disabled)
    assert cfg.pulse_stretch.enable is False


def test_NeuronConfig_load_filters_pulse_stretch_R_load_ohm():
    # The ps JSON has pulse_stretch.R_load_ohm; this module's load() keeps it
    # by passing through a known-keys-only dict, so no error here.
    cfg = M.NeuronConfig.load("defaults/neuron_pulse_stretch.json")
    assert cfg.pulse_stretch.enable is True
    assert cfg.pulse_stretch.R_pw_ohm == 20000.0
    assert cfg.pulse_stretch.C_pw_F == 1e-7
    assert cfg.pulse_stretch.R_load_ohm == 10000.0


def test_header_helper_format():
    assert M._header("X") == "* ---------- X ----------\n"


# ===================================================================
# COVERAGE-GAP CLOSERS. Golden values obtained by running current code.
# ===================================================================


def test_lif_sanity_checks_negative_threshold_lower_bound_only(default_cfg):
    """vth_lo < 0 (via negative over_vref_V) is the SOLE trigger; vdd stays in
    range so the supply check does NOT fire."""
    cfg = copy.deepcopy(default_cfg)
    cfg.threshold.over_vref_V = -3.0
    assert M.lif_sanity_checks(cfg) == [
        "Threshold window [-0.525,-0.475] not inside [0,5.0]."
    ]


def test_generate_detailed_neuron_divider_scale_lower_clamp(default_cfg):
    """divider_scale=0.0 -> max(scale, 1e-3) clamp -> Rvh=681e3*1e-3=681,
    Rvl=316e3*1e-3=316."""
    cfg = copy.deepcopy(default_cfg)
    cfg.threshold.divider_scale = 0.0
    out = M.generate_detailed_neuron(cfg)
    assert "Rvh  vdd     vth_node 681\n" in out
    assert "Rvl  vth_node vref    316\n" in out

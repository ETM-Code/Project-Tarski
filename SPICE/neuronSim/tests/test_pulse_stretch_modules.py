"""
Characterization tests for the pulse-stretch variant modules:
  - lif_neuron_generator_pulse_stretch.py
  - lif_network_generator_pulse_stretch.py

These modules DUPLICATE class/function names from the main modules
(NeuronConfig, generate_detailed_neuron, build_netlist, ...). To avoid name
collisions we import them under distinct aliases via importlib.
"""
from __future__ import annotations

import copy
import importlib

import pytest

from conftest import golden

ps = importlib.import_module("lif_neuron_generator_pulse_stretch")
psn = importlib.import_module("lif_network_generator_pulse_stretch")


def _make_cfg():
    """Construct the pulse_stretch module's NeuronConfig directly from the
    neuron_default values, using the default (enabled) PulseStretchCfg.
    We cannot use NeuronConfig.load on the ps JSON because it raises (see below).
    """
    return ps.NeuronConfig(
        name="demo_neuron",
        supplies=ps.Supplies(vdd=5.0, vref=2.5),
        membrane=ps.Membrane(C_mem_F=1e-8, R_leak_ohm=120000.0),
        threshold=ps.Threshold(
            over_vref_V=0.8, hysteresis_V=0.05, C_adapt_F=2.2e-8, divider_scale=5.0
        ),
        reset=ps.ResetPath(
            enable=True,
            series_R_ohm=200.0,
            mux_Ron_ohm=10.0,
            mux_Roff_ohm=1e9,
            mux_Coff_F=7e-12,
            switch_Vt=2.5,
            switch_Vh=0.1,
        ),
        comparator=ps.ComparatorCfg(
            offset_V=0.003, prop_delay_s=4e-8, vlow_V=0.0, vhigh_V=5.0, gain_fast=2000.0
        ),
        simulation=ps.SimCfg(tstop_s=0.05, tstep_s=1e-6),
        analog_out=ps.AnalogOutCfg(
            gain=2.0,
            sign=1,
            clamp_to_rails=False,
            analog_out_R_ohm=10000.0,
            inverting=True,
            R1_ohm=10000.0,
            R2_ohm=40000.0,
            bias_current_A=1e-5,
        ),
        bias_currents=ps.BiasCurrents(tia_A=2e-5, comparator_A=2e-6),
    )


# --------- the R_load_ohm load() bug (locked as current behavior) ---------

def test_pulse_stretch_NeuronConfig_load_raises_on_R_load_ohm():
    # CHARACTERIZATION: current behavior, possibly a bug, locked to detect change.
    # This module's PulseStretchCfg has no R_load_ohm field, and its load()
    # forwards the JSON pulse_stretch dict unfiltered, so loading the ps JSON
    # (which contains R_load_ohm) raises TypeError.
    with pytest.raises(TypeError, match="R_load_ohm"):
        ps.NeuronConfig.load("defaults/neuron_pulse_stretch.json")


# --------- detailed subckt snapshot (distinct variant) ---------

def test_pulse_stretch_generate_detailed_neuron_snapshot():
    out = ps.generate_detailed_neuron(_make_cfg())
    assert out == golden("psmod_detailed.txt")
    assert len(out.splitlines()) == 53
    assert (
        out.splitlines()[0]
        == "* ---------- demo_neuron DETAILED neuron subcircuit (with pulse stretching) ----------"
    )
    assert (
        ".subckt demo_neuron_detailed mem vref vdd comp_pulse analog_out sum\n" in out
    )
    # NOTE: opposite Bdef sign vs main module.
    assert "Bdef vdef 0 V = V(vref) - V(mem)\n" in out
    assert ".model DPW D(Is=1e-12 N=1.05 Rs=10 Cjo=1p)\n" in out
    assert "Dpw comp_out comp_pulse DPW\n" in out
    assert "Rpw comp_pulse 0 1e+05\n" in out
    assert "Cpw comp_pulse 0 1e-07\n" in out
    assert "Sreset reset_node vref comp_out 0 SWMUX\n" in out


def test_pulse_stretch_generate_detailed_neuron_disabled_bypass():
    cfg = copy.deepcopy(_make_cfg())
    cfg.pulse_stretch.enable = False
    out = ps.generate_detailed_neuron(cfg)
    assert "Rpw_bypass comp_out comp_pulse 1\n" in out
    assert "Dpw" not in out
    assert "Rpw comp_pulse" not in out
    assert "Cpw comp_pulse" not in out


def test_pulse_stretch_generate_fast_neuron_rename():
    cfg = _make_cfg()
    fast = ps.generate_fast_neuron(cfg)
    detailed = ps.generate_detailed_neuron(cfg)
    assert fast == detailed.replace(
        ".subckt demo_neuron_detailed", ".subckt demo_neuron_fast", 1
    )
    assert fast == golden("psmod_fast.txt")
    assert (
        fast.splitlines()[1]
        == ".subckt demo_neuron_fast mem vref vdd comp_pulse analog_out sum"
    )


# --------- ps network module ---------

def test_network_pulse_stretch_build_netlist_snapshot():
    cfgN = psn.NetworkConfig.load("defaults/network_rapid_pulses.json")
    out = psn.build_netlist(cfgN, _make_cfg(), "detailed", "/FIXED/out.csv")
    assert out == golden("psnet_build_detailed.txt")
    assert out.splitlines()[0] == "* lif_rapid_pulse_test [DETAILED]"
    assert "XNEU mem vref vdd comp ana sum demo_neuron_detailed" in out


def test_network_pulse_stretch_gen_spike_source_rapid():
    import lif_network_generator as base

    spikes = [
        psn.Spike(5.0, 0.3, 0.5),
        psn.Spike(5.5, 0.3, 0.5),
        psn.Spike(6.0, 0.3, 0.5),
        psn.Spike(6.5, 0.3, 0.5),
        psn.Spike(7.0, 0.3, 0.5),
    ]
    ps_out = psn.gen_spike_source("syn1", spikes, 1.0)
    base_spikes = [
        base.Spike(5.0, 0.3, 0.5),
        base.Spike(5.5, 0.3, 0.5),
        base.Spike(6.0, 0.3, 0.5),
        base.Spike(6.5, 0.3, 0.5),
        base.Spike(7.0, 0.3, 0.5),
    ]
    base_out = base.gen_spike_source("syn1", base_spikes, 1.0)
    # Confirms the ported copy is byte-identical to the base implementation.
    assert ps_out == base_out
    assert ps_out == (
        "Vsrc_syn1 n_syn1 vref PWL(0.0 0.0 0.005 0.0 0.005001 0.5 0.005299 0.5 "
        "0.0053 0.0 0.0055 0.0 0.005501 0.5 0.005799 0.5 0.0058 0.0 0.006 0.0 "
        "0.006001 0.5 0.006299 0.5 0.0063 0.0 0.0065 0.0 0.006501 0.5 0.006799 0.5 "
        "0.0068 0.0 0.007 0.0 0.007001 0.5 0.007299 0.5 0.0073 0.0)\n"
    )


# ===================================================================
# COVERAGE-GAP CLOSERS. The pulse_stretch neuron module has its OWN copies of
# lif_sanity_checks / NeuronConfig.load / the analog stage; these pin them so a
# divergence from the main module is caught. Golden values from running the code.
# ===================================================================


def test_ps_lif_sanity_checks_clean_default():
    assert ps.lif_sanity_checks(_make_cfg()) == []


def test_ps_lif_sanity_checks_flags_low_vdd_and_window():
    cfg = copy.deepcopy(_make_cfg())
    cfg.supplies.vdd = 1.0
    # Same two-issue list the main module produces for equivalent inputs.
    assert ps.lif_sanity_checks(cfg) == [
        "Comparator supply vdd=1.0 V outside 1.8–5.5 V.",
        "Threshold window [3.275,3.325] not inside [0,1.0].",
    ]


def test_ps_NeuronConfig_load_success_without_R_load_ohm(tmp_path):
    """SUCCESS path of this module's load(): a pulse_stretch dict WITHOUT
    R_load_ohm (which this module's PulseStretchCfg accepts) loads cleanly."""
    import json

    data = json.loads(
        open("defaults/neuron_pulse_stretch.json").read()
    )
    del data["pulse_stretch"]["R_load_ohm"]
    p = tmp_path / "neuron_ps_no_rload.json"
    p.write_text(json.dumps(data))
    cfg = ps.NeuronConfig.load(str(p))
    assert cfg.pulse_stretch.enable is True
    assert cfg.pulse_stretch.R_pw_ohm == 20000.0
    assert cfg.pulse_stretch.C_pw_F == 1e-7


def test_ps_analog_stage_non_inverting():
    cfg = copy.deepcopy(_make_cfg())
    cfg.analog_out.inverting = False
    cfg.analog_out.gain = 3.0
    out = ps.generate_detailed_neuron(cfg)
    # non-inverting gain = 1+R2/R1 -> R2 = R1*(gain-1) = 10k*2 = 20k.
    assert "R2_ana analog_out ana_fb 2e+04\n" in out
    assert "Eana_opamp ana_opamp_out 0 ana_in ana_fb 1e5\n" in out


def test_ps_analog_stage_sign_negative():
    cfg = copy.deepcopy(_make_cfg())
    cfg.analog_out.sign = -1
    out = ps.generate_detailed_neuron(cfg)
    assert "Bdiff ana_in 0 V = V(vref) - V(mem)\n" in out


def test_ps_analog_stage_clamp_to_rails():
    cfg = copy.deepcopy(_make_cfg())
    cfg.analog_out.clamp_to_rails = True
    out = ps.generate_detailed_neuron(cfg)
    assert ".model DCLAMP D(Is=1e-15 N=1.8 Rs=1)\n" in out
    assert "Dcl_lo  0   analog_out DCLAMP\n" in out
    assert "Dcl_hi  analog_out vdd DCLAMP\n" in out

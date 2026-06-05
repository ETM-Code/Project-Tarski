"""
Characterization tests for lif_neuron_generator_passive.py.

Golden-master: locks the passive (no op-amp) subcircuit + test netlist text
and the shipped default config values.
"""
from __future__ import annotations

import copy

import pytest

import lif_neuron_generator_passive as P
from conftest import golden


@pytest.fixture
def default_cfg():
    return P.PassiveNeuronConfig.default()


def test_generate_passive_neuron_default_snapshot(default_cfg):
    out = P.generate_passive_neuron(default_cfg)
    assert out == golden("passive_default.txt")
    assert len(out.splitlines()) == 33
    assert out.splitlines()[0] == (
        "* ---------- lif_passive PASSIVE neuron subcircuit (no op-amps) ----------"
    )
    assert ".subckt lif_passive_passive mem vref vdd comp_pulse sum\n" in out


def test_generate_passive_neuron_pulse_stretch_disabled(default_cfg):
    cfg = copy.deepcopy(default_cfg)
    cfg.pulse_stretch.enable = False
    out = P.generate_passive_neuron(cfg)
    assert "Rpw_bypass comp_out comp_pulse 1\n" in out
    assert "Dpw" not in out
    assert "Cpw" not in out
    assert ".model DPW" not in out


def test_generate_passive_neuron_reset_disabled(default_cfg):
    cfg = copy.deepcopy(default_cfg)
    cfg.reset.enable = False
    out = P.generate_passive_neuron(cfg)
    for line in out.splitlines():
        assert not line.startswith("Rreset")
        assert not line.startswith("Sreset")
        assert not line.startswith(".model SWRESET")
    # enabled default DOES emit them
    enabled = P.generate_passive_neuron(default_cfg)
    assert "Sreset reset_node vref comp_pulse 0 SWRESET\n" in enabled


def test_generate_test_netlist_passive_snapshot(default_cfg):
    out = P.generate_test_netlist(
        default_cfg, subckt_path="/X/sub.subckt", output_csv="/X/out.csv"
    )
    assert out == golden("passive_test_netlist.txt")
    assert len(out.splitlines()) == 29
    assert ".include /X/sub.subckt" in out
    assert "Xneuron mem vref vdd comp_pulse sum lif_passive_passive" in out
    assert "Iin 0 sum DC 1e-6" in out
    assert ".ic V(mem)=2.5" in out
    assert "wrdata /X/out.csv v(mem) v(comp_pulse)" in out


def test_PassiveNeuronConfig_default_values(default_cfg):
    cfg = default_cfg
    assert cfg.name == "lif_passive"
    assert cfg.supplies.vdd == 5.0
    assert cfg.supplies.vref == 2.5
    assert cfg.membrane.C_mem_F == 10e-9
    assert cfg.membrane.R_leak_ohm == 120e3
    assert cfg.threshold.over_vref_V == 0.8
    assert cfg.threshold.hysteresis_V == 0.05
    assert cfg.reset.enable is True
    assert cfg.reset.series_R_ohm == 200.0
    assert cfg.reset.mux_Ron_ohm == 10.0
    assert cfg.reset.mux_Roff_ohm == 1e9
    assert cfg.simulation.tstop_s == 0.05
    assert cfg.simulation.tstep_s == 1e-6
    assert cfg.pulse_stretch.enable is True
    assert cfg.pulse_stretch.R_pw_ohm == 16.7e3
    assert cfg.pulse_stretch.C_pw_F == 100e-9


def test_PassiveNeuronConfig_load_from_json(tmp_path):
    import json

    data = {
        "name": "loaded_passive",
        "supplies": {"vdd": 3.3, "vref": 1.65},
        "membrane": {"C_mem_F": 5e-9, "R_leak_ohm": 100e3},
        "threshold": {"over_vref_V": 0.5, "hysteresis_V": 0.02},
        "reset": {
            "enable": True,
            "series_R_ohm": 150.0,
            "mux_Ron_ohm": 5.0,
            "mux_Roff_ohm": 1e8,
        },
        "simulation": {"tstop_s": 0.01, "tstep_s": 2e-6},
    }
    p = tmp_path / "passive.json"
    p.write_text(json.dumps(data))
    cfg = P.PassiveNeuronConfig.load(str(p))
    assert cfg.name == "loaded_passive"
    assert cfg.supplies.vdd == 3.3
    assert cfg.membrane.C_mem_F == 5e-9
    assert cfg.threshold.over_vref_V == 0.5
    assert cfg.reset.series_R_ohm == 150.0
    assert cfg.simulation.tstep_s == 2e-6
    # pulse_stretch absent -> falls back to PulseStretchCfg() defaults
    assert cfg.pulse_stretch.enable is True
    assert cfg.pulse_stretch.R_pw_ohm == 100e3
    assert cfg.pulse_stretch.C_pw_F == 100e-9


# ===================================================================
# COVERAGE-GAP CLOSERS. Golden values obtained by running current code.
# ===================================================================


def test_generate_passive_neuron_absolute_threshold_and_polarity(default_cfg):
    """Pin the passive-specific computed values in isolation:
    .param VTH = vref + over_vref_V = 2.5 + 0.8 = 3.3, the Bcomp threshold-direction
    (fires when V(mem) > VTH), and the reset switch driven by comp_pulse (stretched)."""
    out = P.generate_passive_neuron(default_cfg)
    assert ".param VTH=3.3\n" in out
    assert (
        "Bcomp comp_raw 0 V = VLO + (VHI - VLO) * "
        "(0.5 * (1 + tanh((V(mem) - VTH) / VSW)))\n" in out
    )
    assert "Sreset reset_node vref comp_pulse 0 SWRESET\n" in out


def test_generate_passive_neuron_membrane_tau_comment(default_cfg):
    """tau_m comment string: 120000 * 1e-08 * 1000 = 1.200 ms."""
    out = P.generate_passive_neuron(default_cfg)
    assert "* tau_m = 1.2e+05 * 1e-08 = 1.200 ms\n" in out


def test_PassiveNeuronConfig_load_reset_switch_defaults(tmp_path):
    """reset omits switch_vt/switch_vh -> ResetPath defaults 2.5 / 0.1."""
    import json

    data = {
        "name": "p",
        "supplies": {"vdd": 5.0, "vref": 2.5},
        "membrane": {"C_mem_F": 1e-8, "R_leak_ohm": 1.2e5},
        "threshold": {"over_vref_V": 0.8, "hysteresis_V": 0.05},
        "reset": {
            "enable": True,
            "series_R_ohm": 200.0,
            "mux_Ron_ohm": 10.0,
            "mux_Roff_ohm": 1e9,
        },
        "simulation": {"tstop_s": 0.05, "tstep_s": 1e-6},
    }
    p = tmp_path / "passive_reset.json"
    p.write_text(json.dumps(data))
    cfg = P.PassiveNeuronConfig.load(str(p))
    assert cfg.reset.switch_vt == 2.5
    assert cfg.reset.switch_vh == 0.1


def test_PassiveNeuronConfig_load_pulse_stretch_override(tmp_path):
    """pulse_stretch IS supplied (override branch of d.get('pulse_stretch', {})):
    enable=False, R_pw_ohm=50000.0; C_pw_F absent -> default 100e-9."""
    import json

    data = {
        "name": "p",
        "supplies": {"vdd": 5.0, "vref": 2.5},
        "membrane": {"C_mem_F": 1e-8, "R_leak_ohm": 1.2e5},
        "threshold": {"over_vref_V": 0.8, "hysteresis_V": 0.05},
        "reset": {
            "enable": True,
            "series_R_ohm": 200.0,
            "mux_Ron_ohm": 10.0,
            "mux_Roff_ohm": 1e9,
        },
        "simulation": {"tstop_s": 0.05, "tstep_s": 1e-6},
        "pulse_stretch": {"enable": False, "R_pw_ohm": 50000.0},
    }
    p = tmp_path / "passive_ps.json"
    p.write_text(json.dumps(data))
    cfg = P.PassiveNeuronConfig.load(str(p))
    assert cfg.pulse_stretch.enable is False
    assert cfg.pulse_stretch.R_pw_ohm == 50000.0
    assert cfg.pulse_stretch.C_pw_F == 100e-9

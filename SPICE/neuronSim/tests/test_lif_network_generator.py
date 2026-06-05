"""
Characterization tests for lif_network_generator.py.

Golden-master: locks generated netlists, PWL spike strings, the validation
report text, CSV parsing, and the power-stats numeric pipeline.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lif_network_generator as N
from conftest import golden


# ----------------------------- fixtures -----------------------------

@pytest.fixture
def default_neuron():
    return N.NeuronConfig.load("defaults/neuron_default.json")


@pytest.fixture
def default_network():
    return N.NetworkConfig.load("defaults/network_default.json")


def _write_csv(tmp_path, content, name="sim.csv"):
    p = tmp_path / name
    p.write_text(content)
    return str(p)


# ----------------------- spike source PWL ---------------------------

def test_gen_spike_source_excitatory_pwl():
    out = N.gen_spike_source("syn1", [N.Spike(5.0, 0.4, 1.2)], 1.0)
    assert out == (
        "Vsrc_syn1 n_syn1 vref PWL(0.0 0.0 0.005 0.0 0.005001 1.2 "
        "0.005399 1.2 0.0054 0.0)\n"
    )


def test_gen_spike_source_inhibitory_sign_flip():
    out = N.gen_spike_source("syn1", [N.Spike(5.0, 0.4, 1.2)], -1.0)
    assert out == (
        "Vsrc_syn1 n_syn1 vref PWL(0.0 0.0 0.005 0.0 0.005001 -1.2 "
        "0.005399 -1.2 0.0054 0.0)\n"
    )


def test_gen_spike_source_multi_spike_dedup_sort():
    out = N.gen_spike_source(
        "syn2", [N.Spike(5.0, 0.35, 1.2), N.Spike(11.5, 0.35, 1.2)], 1.0
    )
    assert out == (
        "Vsrc_syn2 n_syn2 vref PWL(0.0 0.0 0.005 0.0 0.005001 1.2 0.005349 1.2 "
        "0.00535 0.0 0.0115 0.0 0.011501 1.2 0.011849 1.2 0.01185 0.0)\n"
    )


# ----------------------- control block ------------------------------

def test_head_controls_csv_control_block():
    out = N.head_controls_csv("/FIXED/out.csv", 1e-6, 0.05, ["v(mem)", "v(comp)"])
    assert out == (
        "\n.control\n"
        "  set filetype=ascii\n"
        "  set wr_singlescale\n"
        "  set wr_vecnames\n"
        "  * Force uniform output grid and small files\n"
        "  tran 1e-06 0.05 0 1e-06\n"
        "  linearize\n"
        "  wrdata /FIXED/out.csv time v(mem) v(comp)\n"
        "  quit\n"
        ".endc\n"
    )


# ----------------------- full netlist snapshots ---------------------

def test_build_netlist_detailed_full_snapshot(default_network, default_neuron):
    out = N.build_netlist(default_network, default_neuron, "detailed", "/FIXED/out.csv")
    assert out == golden("build_detailed.txt")
    assert len(out.splitlines()) == 79


def test_build_netlist_fast_uses_fast_subckt(default_network, default_neuron):
    out = N.build_netlist(default_network, default_neuron, "fast", "/FIXED/out.csv")
    assert out == golden("build_fast.txt")
    assert "XNEU mem vref vdd comp ana sum demo_neuron_fast" in out
    assert ".subckt demo_neuron_fast mem vref vdd comp_out analog_out sum" in out
    assert ".subckt demo_neuron_detailed" not in out


# ----------------------- numeric helpers ----------------------------

def test_estimate_membrane_step_value(default_neuron):
    val = N.estimate_membrane_step(default_neuron, Rw=33000.0, A=1.2, width_s=0.4e-3)
    assert val == pytest.approx(0.7405035921746238, abs=1e-9)


def test_safety_and_math_validation_text_snapshot(default_network, default_neuron):
    out = N.safety_and_math_validation(default_network, default_neuron)["text"]
    assert out == golden("safety_default.txt")
    assert "Membrane tau (Rleak*C): 1.20 ms" in out
    assert "Comparator nominal Vth: 3.300 V (hyst ±0.025 V)" in out
    assert "Reset path: I≈4.76 mA/V. With 1 V delta, 4.76 mA flows." in out


def test_safety_and_math_validation_high_current_warning(default_neuron):
    cfgN = N.NetworkConfig(
        title="t",
        neuron_json="x",
        synapses=[N.Synapse("synX", "excitatory", 100.0, [N.Spike(1.0, 0.4, 3.2)])],
    )
    out = N.safety_and_math_validation(cfgN, default_neuron)["text"]
    assert (
        "WARNING: synX Ipk=29.50 mA exceeds 80% of MCP41HV51(50k) "
        "continuous (6.5 mA)." in out
    )


# ----------------------- CSV loading --------------------------------

def test_load_sim_data_named_header(tmp_path):
    csv = "time v(mem) i(vdd)\n0.0 2.5 0.001\n1e-6 2.6 0.0011\n2e-6 2.7 0.0012\n"
    sim = N.load_sim_data(_write_csv(tmp_path, csv))
    assert sim.source == "named"
    assert sim.names == ["time", "v(mem)", "i(vdd)"]
    np.testing.assert_allclose(sim.get("v(mem)"), [2.5, 2.6, 2.7], atol=1e-12)
    np.testing.assert_allclose(sim.time, [0.0, 1e-6, 2e-6], atol=1e-12)


def test_load_sim_data_fallback_multiscale(tmp_path):
    # headerless: time vref time vmem  (multi time-column wrdata form)
    csv = "0.0 2.5 0.0 0.1\n1e-6 2.6 1e-6 0.2\n2e-6 2.7 2e-6 0.3\n"
    sim = N.load_sim_data(_write_csv(tmp_path, csv))
    assert sim.source == "fallback"
    np.testing.assert_allclose(sim.get("v(vref)"), [2.5, 2.6, 2.7], atol=1e-12)
    np.testing.assert_allclose(sim.get("v(mem)"), [0.1, 0.2, 0.3], atol=1e-12)


def test_load_sim_data_index_column_prefix(tmp_path):
    csv = "index time v(mem)\n0 0.0 2.5\n1 1e-6 2.6\n2 2e-6 2.7\n"
    sim = N.load_sim_data(_write_csv(tmp_path, csv))
    assert sim.source == "named"
    assert sim.names == ["index", "time", "v(mem)"]
    np.testing.assert_allclose(sim.get("v(mem)"), [2.5, 2.6, 2.7], atol=1e-12)


def test_load_sim_data_empty_raises(tmp_path):
    csv = "* comment only\n* another\n"
    with pytest.raises(ValueError, match=r"CSV parse error: got shape"):
        N.load_sim_data(_write_csv(tmp_path, csv))


# ----------------------- power stats --------------------------------

@pytest.fixture(autouse=True)
def _ensure_module_numpy(monkeypatch):
    """compute_power_stats() uses the module-global ``np`` (set to None if the
    optional pandas/matplotlib import block fails at import time). Pin it to the
    real numpy so these tests are hermetic regardless of whether pandas is
    installed in the runner environment."""
    monkeypatch.setattr(N, "np", np, raising=False)
    yield


def _simdata(time, vectors):
    return N.SimData(time=time, vectors=vectors, names=None, raw=None, source="named")


def test_compute_power_stats_named_vdd_vref():
    time = np.array([0.0, 1e-3, 2e-3])
    sim = _simdata(
        time,
        {
            "time": time,
            "v(mem)": np.array([2.5, 2.6, 2.7]),
            "i(vdd)": np.array([-1e-3, -1e-3, -1e-3]),
            "i(vref)": np.array([-2e-3, -2e-3, -2e-3]),
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # numpy.trapz DeprecationWarning under numpy 2.x
        stats = N.compute_power_stats(
            sim, N.Supplies(5.0, 2.5), skip_rails=N.POWER_RAIL_SKIPS
        )
    assert stats["total"]["counted_rails"] == 1
    rails = {r["rail"]: r for r in stats["rails"]}
    # VDD: 5 V * 1e-3 A = 5e-3 W const over 2e-3 s -> 1e-5 J.
    assert rails["VDD"]["energy_J"] == pytest.approx(1e-5, abs=1e-12)
    assert rails["VDD"]["include"] is True
    assert rails["VREF"]["include"] is False
    assert rails["VREF"]["reason"] == (
        "reference buffer uses an idealized op-amp, so the supply current "
        "is non-physical"
    )


def test_compute_power_stats_returns_none_on_short_time():
    sim1 = _simdata(np.array([0.0]), {})
    assert N.compute_power_stats(sim1, N.Supplies(5.0, 2.5)) is None
    sim2 = _simdata(None, {})
    assert N.compute_power_stats(sim2, N.Supplies(5.0, 2.5)) is None


def test_compute_power_stats_extra_rail_dynamic_voltage():
    time = np.array([0.0, 1e-3, 2e-3])
    sim = _simdata(
        time,
        {
            "time": time,
            "i(vsrc_x)": np.array([-1e-3, -1e-3, -1e-3]),
            "v(n_x)": np.array([3.0, 3.0, 3.0]),
            "v(vref)": np.array([2.5, 2.5, 2.5]),
        },
    )
    extra = [
        {
            "label": "SRC_X",
            "current_vec": "i(vsrc_x)",
            "voltage_vec": "v(n_x)",
            "voltage_ref": "v(vref)",
            "include": True,
            "reason": "syn",
        }
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats = N.compute_power_stats(
            sim, N.Supplies(5.0, 2.5), skip_rails=N.POWER_RAIL_SKIPS, extra_rails=extra
        )
    rails = {r["rail"]: r for r in stats["rails"]}
    src = rails["SRC_X"]
    # (3.0-2.5) V * 1e-3 A = 5e-4 W const over 2e-3 s -> 1e-6 J.
    assert src["energy_J"] == pytest.approx(1e-6, abs=1e-12)
    assert src["voltage"] is None  # dynamic voltage -> no fixed report
    assert src["include"] is True


# ----------------------- formatting helpers -------------------------

@pytest.mark.parametrize(
    "value,expected",
    [
        (1.5e-7, "150 nW"),
        (2.3e-4, "230 µW"),
        (0.5, "500 mW"),
        (3.0, "3 W"),
    ],
)
def test_fmt_watts_unit_thresholds(value, expected):
    assert N._fmt_watts(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (5e-13, "0.5 pJ"),
        (5e-10, "500 pJ"),
        (5e-7, "500 nJ"),
        (5e-4, "500 µJ"),
        (2.0, "2 J"),
    ],
)
def test_fmt_energy_unit_thresholds(value, expected):
    assert N._fmt_energy(value) == expected


def test_pretty_and_canonical_signal_name():
    assert N.pretty(123456.0) == "1.23e+05"
    assert N.canonical_signal_name("  V(MEM) ") == "v(mem)"


# ----------------------- config + SimData ---------------------------

def test_NetworkConfig_load_default_json(default_network):
    cfgN = default_network
    assert cfgN.title == "lif_demo_network"
    assert len(cfgN.synapses) == 3
    assert len(cfgN.synapses[1].spikes) == 3  # syn2
    assert cfgN.synapses[0].weight_ohm == 33000
    assert all(s.type == "excitatory" for s in cfgN.synapses)
    sp = cfgN.synapses[0].spikes[0]
    assert (sp.t_ms, sp.width_ms, sp.amp_V) == (5.0, 0.4, 1.2)


def test_SimData_get_case_insensitive():
    arr = np.array([1.0, 2.0, 3.0])
    sim = N.SimData(time=None, vectors={"v(mem)": arr}, names=None, raw=None)
    got = sim.get("  V(MEM) ")
    assert got is arr


# ===================================================================
# COVERAGE-GAP CLOSERS (added to pin previously-unlocked branches).
# All golden values obtained by running the current code.
# ===================================================================


# ----------------- build_netlist inhibitory + multi-synapse ----------

def test_build_netlist_inhibitory_synapse_polarity_and_signal_order(default_neuron):
    """The sign=-1 (inhibitory) branch inside build_netlist plus the per-synapse
    signal ordering appended to the wrdata line."""
    cfgN = N.NetworkConfig(
        title="t",
        neuron_json="x",
        synapses=[
            N.Synapse("exc", "excitatory", 50000.0, [N.Spike(5.0, 0.4, 1.2)]),
            N.Synapse("inh", "inhibitory", 50000.0, [N.Spike(6.0, 0.4, 1.2)]),
        ],
    )
    out = N.build_netlist(cfgN, default_neuron, "detailed", "/X/o.csv")
    # Excitatory: amplitude stays +1.2; inhibitory: polarity flipped to -1.2.
    assert (
        "Vsrc_exc n_exc vref PWL(0.0 0.0 0.005 0.0 0.005001 1.2 "
        "0.005399 1.2 0.0054 0.0)\n" in out
    )
    assert "R_exc n_exc sum 50000.0" in out
    assert (
        "Vsrc_inh n_inh vref PWL(0.0 0.0 0.006 0.0 0.006001 -1.2 "
        "0.006399 -1.2 0.0064 0.0)\n" in out
    )
    assert "R_inh n_inh sum 50000.0" in out


def test_build_netlist_inhibitory_wrdata_signal_ordering(default_neuron):
    cfgN = N.NetworkConfig(
        title="t",
        neuron_json="x",
        synapses=[
            N.Synapse("exc", "excitatory", 50000.0, [N.Spike(5.0, 0.4, 1.2)]),
            N.Synapse("inh", "inhibitory", 50000.0, [N.Spike(6.0, 0.4, 1.2)]),
        ],
    )
    out = N.build_netlist(cfgN, default_neuron, "detailed", "/X/o.csv")
    wrdata = [l for l in out.splitlines() if l.strip().startswith("wrdata")][0]
    assert wrdata == (
        "  wrdata /X/o.csv time v(vref) v(mem) v(comp) v(ana) v(n_outmix) v(sum) "
        "v(n_exc) i(Vsrc_exc) v(n_inh) i(Vsrc_inh) i(VDD) i(VREF)"
    )


# ----------------- gen_spike_source empty list ----------------------

def test_gen_spike_source_empty_spike_list_seed_point_only():
    # Boundary: zero spikes -> only the (0.0, 0.0) seed point is emitted.
    assert N.gen_spike_source("s", [], 1.0) == "Vsrc_s n_s vref PWL(0.0 0.0)\n"


# ----------------- _fmt_watts / _fmt_energy negatives + boundaries ---

@pytest.mark.parametrize(
    "value,expected",
    [
        (-3.0, "-3 W"),       # negative: abs() chooses branch, sign preserved
        (1e-6, "1 µW"),       # boundary: NOT < 1e-6 -> falls to µW branch
        (1e-3, "1 mW"),       # boundary: NOT < 1e-3 -> falls to mW branch
        (1.0, "1 W"),         # boundary: NOT < 1 -> falls to W branch
    ],
)
def test_fmt_watts_negative_and_boundary(value, expected):
    assert N._fmt_watts(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (-2.0, "-2 J"),       # negative
        (1e-9, "1 nJ"),       # boundary: NOT < 1e-9 -> nJ
        (1e-6, "1 µJ"),       # boundary: NOT < 1e-6 -> µJ
        (1e-3, "1 mJ"),       # boundary: NOT < 1e-3 -> mJ
    ],
)
def test_fmt_energy_negative_and_boundary(value, expected):
    assert N._fmt_energy(value) == expected


# ----------------- compute_power_stats no-rails / counted==0 --------

def test_compute_power_stats_no_rails_returns_none():
    """Neither i(vdd) nor i(vref) present -> 'if not rails: return None' path."""
    time = np.array([0.0, 1e-3, 2e-3])
    sim = _simdata(time, {"time": time, "v(mem)": np.array([2.5, 2.6, 2.7])})
    assert (
        N.compute_power_stats(sim, N.Supplies(5.0, 2.5), skip_rails=N.POWER_RAIL_SKIPS)
        is None
    )


def test_compute_power_stats_only_skipped_rail_counted_zero():
    """Only i(vref) present, which POWER_RAIL_SKIPS excludes -> counted_rails==0,
    peak_power_W==0.0 (total_samples stays None)."""
    time = np.array([0.0, 1e-3, 2e-3])
    sim = _simdata(time, {"time": time, "i(vref)": np.array([-2e-3, -2e-3, -2e-3])})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats = N.compute_power_stats(
            sim, N.Supplies(5.0, 2.5), skip_rails=N.POWER_RAIL_SKIPS
        )
    assert stats["total"]["counted_rails"] == 0
    assert stats["total"]["peak_power_W"] == 0.0


def test_print_power_report_no_rails_included_branch(capsys):
    time = np.array([0.0, 1e-3, 2e-3])
    sim = _simdata(time, {"time": time, "i(vref)": np.array([-2e-3, -2e-3, -2e-3])})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats = N.compute_power_stats(
            sim, N.Supplies(5.0, 2.5), skip_rails=N.POWER_RAIL_SKIPS
        )
    N.print_power_report("DETAILED", stats)
    out = capsys.readouterr().out
    assert "[power] DETAILED: no supply rails included in totals (check configuration)." in out
    assert (
        "VREF: (excluded) avg 5 mW, peak 5 mW, energy 10 µJ @ 2.5 V — "
        "reference buffer uses an idealized op-amp, so the supply current "
        "is non-physical" in out
    )


# ----------------- print_power_report full VDD/VREF text ------------

def test_print_power_report_named_vdd_vref_text(capsys):
    time = np.array([0.0, 1e-3, 2e-3])
    sim = _simdata(
        time,
        {
            "time": time,
            "i(vdd)": np.array([-1e-3, -1e-3, -1e-3]),
            "i(vref)": np.array([-2e-3, -2e-3, -2e-3]),
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats = N.compute_power_stats(
            sim, N.Supplies(5.0, 2.5), skip_rails=N.POWER_RAIL_SKIPS
        )
    N.print_power_report("DETAILED", stats)
    out = capsys.readouterr().out
    assert "[power] DETAILED: avg 5 mW, peak 5 mW, energy 10 µJ over 0.002 s" in out
    assert "VDD: avg 5 mW, peak 5 mW, energy 10 µJ @ 5 V" in out
    assert (
        "VREF: (excluded) avg 5 mW, peak 5 mW, energy 10 µJ @ 2.5 V — "
        "reference buffer uses an idealized op-amp" in out
    )


# ----------------- load_sim_data name/column mismatch fallback ------

def test_load_sim_data_name_count_mismatch_falls_back_to_positional(tmp_path):
    """Header has 4 names but data has 2 columns -> names=None fallback,
    source='fallback', first value column maps to v(vref)."""
    csv = "time v(mem) v(comp) extra\n0.0 2.5\n1e-6 2.6\n"
    sim = N.load_sim_data(_write_csv(tmp_path, csv))
    assert sim.source == "fallback"
    assert sim.names is None
    np.testing.assert_allclose(sim.get("v(vref)"), [2.5, 2.6], atol=1e-12)


# ----------------- estimate_membrane_step tau==0 alpha=1.0 ----------

def test_estimate_membrane_step_zero_capacitance_alpha_forced(default_neuron):
    """C_mem_F=0 -> tau=0 -> alpha forced to 1.0 -> result == Vinf."""
    import copy

    cfg = copy.deepcopy(default_neuron)
    cfg.membrane.C_mem_F = 0.0
    val = N.estimate_membrane_step(cfg, 33000.0, 1.2, 0.4e-3)
    assert val == pytest.approx(0.9411764705882353, abs=1e-9)


# ----------------- safety_and_math_validation reset disabled --------

def test_safety_and_math_validation_reset_disabled_omits_reset_path(default_neuron):
    import copy

    cfg = copy.deepcopy(default_neuron)
    cfg.reset.enable = False
    cfgN = N.NetworkConfig(title="t", neuron_json="x", synapses=[])
    out = N.safety_and_math_validation(cfgN, cfg)["text"]
    assert "Reset path" not in out
    # Report ends after the Comparator nominal Vth line (no synapses, no reset).
    assert out.splitlines()[-1] == (
        "Comparator nominal Vth: 3.300 V (hyst ±0.025 V)"
    )


# ----------------- run_ngspice absent path ---------------------------

def test_run_ngspice_not_on_path_returns_127(monkeypatch, capsys):
    monkeypatch.setattr(N.shutil, "which", lambda *_: None)
    rc = N.run_ngspice("/x.cir", "/x.log")
    assert rc == 127
    assert "ngspice not found on PATH." in capsys.readouterr().out

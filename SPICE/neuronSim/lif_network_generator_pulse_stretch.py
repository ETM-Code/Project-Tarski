#!/usr/bin/env python3
"""
Pulse-stretch variant of the LIF network generator.

This is a thin wrapper over :mod:`lif_network_generator`. The network assembly,
CSV parsing, plotting and power-analysis code is shared; only the neuron backend
differs (it uses the pulse-stretched neuron subcircuits from
:mod:`lif_neuron_generator_pulse_stretch`).

Usage:
  python lif_network_generator_pulse_stretch.py
  python lif_network_generator_pulse_stretch.py --network defaults/network_default.json --neuron defaults/neuron_default.json --mode detailed --yes
"""
from __future__ import annotations

import lif_network_generator as _core

# Shared network/analysis core re-exported so existing import paths keep working.
from lif_network_generator import (  # noqa: F401  (re-exported for stable import paths)
    Spike,
    Synapse,
    NetworkConfig,
    SimData,
    ensure_outputs_dir,
    pretty,
    estimate_membrane_step,
    head_controls_csv,
    gen_spike_source,
    run_ngspice,
    parse_args,
    safety_and_math_validation,
    canonical_signal_name,
    load_sim_data,
    plot_results,
    POWER_RAIL_SKIPS,
    compute_power_stats,
    print_power_report,
)

# Neuron configuration + subcircuit emitters from the pulse-stretched generator.
from lif_neuron_generator_pulse_stretch import (  # noqa: F401  (re-exported)
    Supplies,
    Membrane,
    Threshold,
    ResetPath,
    ComparatorCfg,
    PulseStretchCfg,
    SimCfg,
    NeuronConfig,
    generate_fast_neuron,
    generate_detailed_neuron,
)


def build_netlist(cfgN: NetworkConfig, cfg: NeuronConfig, mode: str, out_csv: str) -> str:
    """Build the network netlist using the pulse-stretched neuron backend."""
    return _core.build_netlist(
        cfgN, cfg, mode, out_csv,
        fast_emitter=generate_fast_neuron,
        detailed_emitter=generate_detailed_neuron,
    )


def main():
    """Run the shared driver with the pulse-stretched neuron backend."""
    _core.main(neuron_config_loader=NeuronConfig.load, netlist_builder=build_netlist)


if __name__ == "__main__":
    main()

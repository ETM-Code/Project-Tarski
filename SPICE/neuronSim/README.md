# LIF Neuron SPICE Workspace

SPICE and Python tooling for analogue LIF neuron experiments and validation.

This directory captures an earlier voltage-injection modelling path used during design exploration. It remains useful for regression checks and report traceability, but the current hardware architecture is documented in the main report and implementation repositories.

## Main Scripts and Inputs

- `lif_network_generator.py`: generate and run neuron-plus-synapse simulations
- `lif_neuron_generator.py`: generate neuron-only netlists
- `defaults/*.json`: default simulation parameters
- `models/`: component-model notes used by detailed runs

## Simulation Modes

- Fast mode: behavioural sources and idealized switching for quick iteration
- Detailed mode: non-ideal component behavior approximations (offsets, delays, parasitics)

Both modes are kept so behavior changes can be separated from solver/runtime effects.

## Quick Start

```bash
python lif_network_generator.py
```

For neuron-only generation:

```bash
python lif_neuron_generator.py
```

## Outputs

Artifacts are written to `outputs/`:

- generated netlists (`*.cir`)
- transient waveform exports (`*.csv`)
- plotting artifacts (`*.png`)

These files are used by report figures and by cross-checks against higher-level simulators.

## Requirements

- Python 3.x
- `ngspice` on `PATH`
- Python packages: `numpy`, `pandas`, `matplotlib`

## Notes

- If `ngspice` is unavailable, generators can still emit netlists for manual simulation.
- Parameter edits should be made in JSON defaults or explicit CLI arguments, not hardcoded in scripts.

# Gilgamesh Neural Network Toolkit

This repository contains a software stack for compiling neuromorphic design descriptions, simulating them, training synaptic weights, and comparing behaviour against SPICE-based references.

## Quick Start

```bash
# Train using defaults from configs/default_design.json
cargo run -- train

# Visualise a single training sample (plots and JSON traces)
cargo run -- visualize --sample 0 --neurons 8

# Resume training from the latest checkpoint and keep going for 10 more epochs
cargo run -- train --resume output/checkpoints/latest.json --epochs 10
```

All CLI subcommands honour the design-level defaults declared in `configs/default_design.json`. Override any option on the command line when needed.

## Design File Schema

A design JSON describes layers, connectivity rules, neuron templates, and training hints. Example (`configs/default_design.json`):

```json
"training": {
  "dataset": "../src/inputs/training_set/converted/train_7x7.bin",
  "test_dataset": "../src/inputs/testing_set/converted/test_7x7.bin",
  "input_layer": "input",
  "target_readout": "logits",
  "checkpoint_dir": "../output/checkpoints",
  "checkpoint_every": 5,
  "config": {
    "learning_rate": 0.003,
    "epochs": 30,
    "row_spacing_start": 0.0015,
    "row_spacing_end": 0.0008,
    "pulse_width": 0.45,
    "input_scale": 1.0,
    "sample_limit": 2048
  }
}
```

Key fields:

- `dataset`, `test_dataset`: training and evaluation datasets. Paths may be relative to the design file.
- `input_layer`, `target_readout`: layer and readout IDs used for temporal encoding and classification.
- `checkpoint_dir`, `checkpoint_every`: automatic checkpoint output path and frequency (epochs).
- `config`: default hyperparameters (learning rate, epoch count, temporal spacing schedule, etc.).

## Training CLI

`cargo run -- train` accepts the options below; all default to the values above unless overridden:

| Flag | Description |
| --- | --- |
| `--design <path>` | Design JSON (defaults to `configs/default_design.json`). |
| `--dataset <path>` / `--test-dataset <path>` | Override dataset locations. |
| `--resume <checkpoint.json>` | Resume from a saved compiled network. If omitted, the trainer tries `checkpoint_dir/latest.json`. |
| `--checkpoint-dir <dir>` | Where to write checkpoints (JSON). |
| `--checkpoint-every <n>` | Save weights every `n` epochs (0 disables periodic saves). |
| `--compiled-out <path>` | Write the final compiled network to disk. |
| `--log <path>` | Emit per-epoch loss logs as JSON. |
| `--epochs`, `--learning-rate`, ... | Override hyperparameters on the command line. |

### Checkpoints and Resume

- Checkpoints are stored as JSON under `checkpoint_dir` (defaults to `output/checkpoints`).
- `checkpoint_epoch_####.json` captures weights at that epoch; `latest.json` always mirrors the most recent checkpoint.
- To continue training, use `--resume path/to/checkpoint.json` or simply run `cargo run -- train` again (it auto-detects `latest.json`).

## Visualisation CLI

`cargo run -- visualize` feeds a single sample through the compiled network and emits JSON + PNG line charts:

```
cargo run -- visualize --sample 12 --layer hidden --neurons 16 --test
```

Outputs land under `output/visualizations/{train|test}/sample_####/` and include:

- `visualization.json` with times, readout histories, final readout values, and traced neuron voltages.
- `readout_<id>.png` and `neuron_<index>.png` plots generated through Plotters.

Useful flags:

| Flag | Description |
| --- | --- |
| `--sample <n>` | (Optional) dataset index to visualise (default `0`). |
| `--layer <id>` | (Optional) layer whose first N neurons are traced. |
| `--neurons <n>` | Number of neurons to trace (default `8`). |
| `--test` | Use the test dataset instead of the training set. |
| `--output-dir <dir>` | Root directory for visualization exports. |

## Simulation Exports

`SimulationOptions::trace_neurons` and the `SimulationResult` now retain per-neuron traces. Any component calling `simulate_network` can request additional neurons to be recorded for bespoke tooling outside the Visualize CLI.

## Development Commands

```bash
# Run the full test suite
cargo test

# Static checks
cargo check

# Short training sanity check
cargo run -- train --epochs 1 --sample-limit 10
```

For further analysis you can read `visualization.json` in a Python notebook or other tools, leveraging the timestamps, readout vectors, and neuron traces that the runtime captures.

## Pulse Stretch Configuration

The pulse stretch circuit extends spike duration for meaningful charge transfer to downstream neurons.

```json
"pulse_stretch": {
  "enable": true,
  "R_pw_ohm": 20000,
  "C_pw_F": 1e-7,
  "R_load_ohm": 10000
}
```

Parameters:
- `R_pw_ohm`: Pulldown resistor (determines decay when no load)
- `C_pw_F`: Pulse capacitor
- `R_load_ohm`: Optional load resistance (accounts for parallel discharge path)

The effective time constant is: τ_eff = (R_pw || R_load) × C_pw

Example: R_pw=20kΩ, R_load=10kΩ, C_pw=100nF → τ_eff = 6.67kΩ × 100nF = 0.67ms

## SPICE vs. Rust Comparator (Python harness)

Prereqs: ngspice on PATH; Python 3.11 recommended. Install matplotlib if you want plots (otherwise they are skipped).

Typical commands (from repo root):

```bash
# Generate SPICE CSV + compare with Rust (detailed mode, release build, write JSON/CSV/plots)
python3 tools/spice_compare.py --mode detailed

# Reuse an existing SPICE CSV (skip ngspice) and still emit outputs
python3 tools/spice_compare.py --mode detailed --skip-spice

# Benchmark only the Rust core (no artifacts), assuming SPICE CSV already exists
python3 tools/spice_compare.py --core-only --skip-spice --no-prebuild

# Force debug build instead of release (slower, for debugging)
python3 tools/spice_compare.py --debug --mode detailed

# Use a different SPICE CSV
python3 tools/spice_compare.py --mode detailed --spice-csv /path/to/lif_detailed.csv

# Specify network/neuron JSONs explicitly
python3 tools/spice_compare.py --network SPICE/neuronSim/defaults/network_default.json \
  --neuron SPICE/neuronSim/defaults/neuron_default.json --mode detailed
```

Notes:
- Artifacts default to `comparison_outputs/` (comparison JSON, equivalent CSV, plots if matplotlib is present).
- `--core-only` prints metrics and timings (including simulate-only speedup) without writing JSON/CSV. Add `--no-prebuild` to skip `cargo build` if the binary is already built.
- If `python3` points to 3.14 on your system, prefer the 3.11 interpreter you used earlier, e.g. `/opt/homebrew/opt/python@3.11/bin/python3.11 tools/spice_compare.py ...`.

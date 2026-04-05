# Scope Probe Toolkit

Utilities for oscilloscope-based current, power, and energy measurements used in the Project Tarski validation workflow.

## Core Flow

1. Configure scope and defaults with `scope-config.py`
2. Capture waveforms with `scope-capture.py`
3. Convert captures to current/power with `capture-power-analysis.py`
4. Summarize and integrate with `postprocess-summary.py` and `postprocess-enery-analysis.py`

## Main Scripts

- `setup.sh`: install Python dependencies
- `scope-config.py`: VISA discovery, config storage, scope setup
- `scope-capture.py`: raw waveform capture to CSV
- `capture-waveform-analysis.py`: waveform plotting
- `capture-power-analysis.py`: current and power derivation
- `postprocess-summary.py`: statistical summaries
- `postprocess-enery-analysis.py`: energy integration from power traces

## Quick Start

```bash
./setup.sh venv
source .venv/bin/activate
./.venv/bin/python scope-config.py --search
./.venv/bin/python scope-capture.py
./.venv/bin/python capture-power-analysis.py output/<capture>.raw.csv
./.venv/bin/python postprocess-summary.py output/<capture>.proc.csv --console
./.venv/bin/python postprocess-enery-analysis.py output/<capture>.proc.csv
```

## Notes

- Default configuration files are `scope-config.toml` and `analysis-config.toml`.
- The `postprocess-enery-analysis.py` filename is intentionally preserved for compatibility with existing scripts.

### `analysis-config.toml`
Used by:
- `capture-power-analysis.py`

Typical keys:
- `resistance`

## File Naming Stages

- Raw capture: `*.raw.csv`
- Processed power/current: `*.proc.csv`
- Energy integration: `*.energy.csv`

## Gotchas / Troubleshooting

- USB/VISA permissions:
  - If the scope is not found or access fails, try `sudo` or fix udev permissions.
  - `scope-config.py` prints permission-specific errors.

- Running with `sudo`:
  - Scope scripts attempt to `chown` generated files/directories back to your user (via `SUDO_UID/SUDO_GID`).

- `-h` is disabled on several scripts:
  - Use `--help` instead.

- Unit parsing in `scope-config.py` is strict:
  - `--time-scale 10us` works.
  - `--time-scale 10` fails (no unit).
  - Time args reject voltage units and vice versa.

- `capture-power-analysis.py --graph` behavior:
  - In graph mode, it displays plots and does **not** write processed CSV.

- Stitch mode on long captures:
  - `-s mean`: one averaged point per captured frame.
  - `-s raw`: all points from each frame are kept.

- Matplotlib cache warning (`/home/.../.config/matplotlib not writable`):
  - If seen, set `MPLCONFIGDIR` to a writable dir, e.g.:
    - `export MPLCONFIGDIR=/tmp/matplotlib-cache`

- Optional math channel in capture:
  - `-m` only captures MATH if MATH is enabled/configured on the scope.

## Minimal End-to-End Example

```bash
# 1) Setup
./setup.sh venv
source .venv/bin/activate

# 2) Configure defaults and scope
./.venv/bin/python scope-config.py --search
./.venv/bin/python scope-config.py --set-resource "USB0::...::INSTR"
./.venv/bin/python scope-config.py --set-duration 30s --time-scale 10us --ch-scale all 500mV --math sub

# 3) Capture raw waveform
./.venv/bin/python scope-capture.py -m

# 4) Process to power/current
./.venv/bin/python capture-power-analysis.py output/2026-04-02-12-00-00.raw.csv -r 9.96

# 5) Summarize
./.venv/bin/python postprocess-summary.py output/2026-04-02-12-00-00.proc.csv --console

# 6) Integrate energy
./.venv/bin/python postprocess-enery-analysis.py output/2026-04-02-12-00-00.proc.csv
```

# Scope Probe Toolkit

Scripts for configuring a Keysight DSO-X 2012A, capturing waveform CSVs, deriving current/power, and producing summaries/energy totals.

## Scripts At A Glance

- `setup.sh`: installs Python dependencies (venv or global with `.venv` shims).
- `scope-config.py`: configure/read/store/load oscilloscope settings.
- `scope-capture.py`: capture CH1/CH2/(optional MATH) waveforms to CSV.
- `capture-waveform-analysis.py`: quick waveform plotting from captured CSVs.
- `capture-power-analysis.py`: compute current and power from captures.
- `postprocess-summary.py`: min/max/avg/stddev summaries of processed data.
- `postprocess-enery-analysis.py`: integrate power over time to compute energy.

## Expected Workflow Order

1. Configure environment: `setup.sh`
2. Discover/configure scope + defaults: `scope-config.py`
3. Capture raw data: `scope-capture.py`
4. (Optional) inspect captured traces: `capture-waveform-analysis.py`
5. Convert capture to current/power: `capture-power-analysis.py`
6. Summarize processed output: `postprocess-summary.py`
7. Integrate energy: `postprocess-enery-analysis.py`

## Python Environment Setup

### Option A (recommended): virtualenv

```bash
./setup.sh venv
source .venv/bin/activate
```

### Option B: global install + shebang shims

```bash
./setup.sh global
```

This installs packages globally and creates `.venv/bin/{python,pip}` symlinks so repo shebangs still work.

Installed dependencies include: `pyvisa`, `pyvisa-py`, `pyusb`, `psutil`, `zeroconf`, `matplotlib`.

## Script Usage

### 1) `scope-config.py`

Purpose:
- Discover VISA resources
- Persist default USB resource and duration to config
- Apply/read/store/load scope settings
- Configure math mode (CH1-CH2)

Config file default: `scope-config.toml` (override with `-C`).

Common commands:

```bash
# Find connected VISA resources
./.venv/bin/python scope-config.py --search

# Persist default resource and capture duration in config
./.venv/bin/python scope-config.py --set-resource "USB0::...::INSTR"
./.venv/bin/python scope-config.py --set-duration 30s

# Apply settings directly to scope
./.venv/bin/python scope-config.py \
  --time-scale 10us \
  --ch-scale all 500mV \
  --ch-offset 1 0mV \
  --ch-offset 2 0mV \
  --math sub

# Read current scope settings only
./.venv/bin/python scope-config.py --read-only

# Store current scope settings to config, then load later
./.venv/bin/python scope-config.py --store
./.venv/bin/python scope-config.py --load
```

Important flags:
- `-C <file>` config path
- `-T <ms>` VISA timeout
- `--resource <visa_resource>` one-off resource
- `--set-resource <visa_resource>` persist resource to config
- `--set-duration <time_with_units>` persist default capture duration
- `--ch-scale <1|2|all> <value_with_units>`
- `--ch-offset <1|2|all> <value_with_units>`
- `--math off|sub|subtract`

Units are mandatory for time/voltage values (`ns/us/ms/s`, `uV/mV/V`).

### 2) `scope-capture.py`

Purpose:
- Capture waveform data from scope and write CSV.

Output:
- Default path: `output/YYYY-MM-DD-hh-mm-ss.raw.csv`
- Single-capture CSV columns: `sample_index,time_s,CHAN1_v,CHAN2_v[,MATH_v]`
- Duration mode CSV columns: `capture_index,capture_elapsed_s,sample_index,time_s,source,voltage_v`

Examples:

```bash
# Single capture using resource from scope-config.toml
./.venv/bin/python scope-capture.py

# Include MATH trace and custom points
./.venv/bin/python scope-capture.py -m --points 2000

# Duration capture for 30s
./.venv/bin/python scope-capture.py -d 30

# Multiple runs (prompts Enter between runs)
./.venv/bin/python scope-capture.py -n 5

# Fast mode (higher frame rate)
./.venv/bin/python scope-capture.py -f
```

Important flags:
- `-C <file>` config path
- `--resource <visa_resource>`
- `-o, --output <csv_path>`
- `-T <ms>` timeout
- `--points <n>`
- `-m, --include-math`
- `-t, --acquire-type NORM|AVER|HRES`
- `-A <n>` average count (for `AVER`)
- `-d, --duration <seconds>`
- `-n, --num-runs <count>`
- `-f, --fast`

### 3) `capture-waveform-analysis.py`

Purpose:
- Plot raw captured waveforms (`.raw.csv` or long-format duration CSV).

Examples:

```bash
# Interactive plot
./.venv/bin/python capture-waveform-analysis.py output/2026-04-01-12-00-00.raw.csv --graph

# Save PNG only
./.venv/bin/python capture-waveform-analysis.py output/file.raw.csv -o output/file-waveform.png

# Select one channel and one capture index from long-format data
./.venv/bin/python capture-waveform-analysis.py output/file.raw.csv -c CHAN2 -i 3 --graph
```

Important flags:
- `-i, --cap-idx <int>` select capture index for long-format data
- `-c, --channel <name>` channel/source filter
- `-o <png_path>` save plot
- `--graph` show interactive window

### 4) `capture-power-analysis.py`

Purpose:
- Convert captured voltage data into current and power.
- Current is computed by Ohm's law from shunt voltage (`MATH` or derived `CH1-CH2`).
- Power uses `CHAN2 * current`.

Input:
- Capture CSV(s) from `scope-capture.py`.

Output:
- Default: `output/<input>.proc.csv`
- Columns: `input_file,sample_index,time_from_start_s,chan2_voltage_v,current_a,power_w`

Examples:

```bash
# Process one capture to CSV (default)
./.venv/bin/python capture-power-analysis.py output/file.raw.csv

# Override shunt resistance and output path
./.venv/bin/python capture-power-analysis.py output/file.raw.csv -r 9.96 -o output/file.proc.csv

# Use derived CH1-CH2 instead of MATH
./.venv/bin/python capture-power-analysis.py output/file.raw.csv -d

# Long-format stitch mode
./.venv/bin/python capture-power-analysis.py output/file.raw.csv -s mean
./.venv/bin/python capture-power-analysis.py output/file.raw.csv -s raw

# Plot interactively (no CSV written when --graph is set)
./.venv/bin/python capture-power-analysis.py output/file.raw.csv --graph
```

Important flags:
- `-C <file>` analysis config (default `analysis-config.toml`)
- `-r, --resistance <ohms>`
- `-i, --cap-idx <int>` select one capture index
- `-s, --stitch-mode mean|raw`
- `-d, --derive-diff` use CH1-CH2 instead of MATH
- `-c, --column <ch1|ch2|math> <name>` rename source columns
- `-o <csv_path>` processed CSV output
- `--graph` interactive plot only

### 5) `postprocess-summary.py`

Purpose:
- Group processed rows by `input_file` and compute min/max/avg/stddev for:
  - `chan2_voltage_v`
  - `current_a`
  - `power_w`

Examples:

```bash
# Console table
./.venv/bin/python postprocess-summary.py output/file.proc.csv --console

# Markdown and image outputs
./.venv/bin/python postprocess-summary.py output/file.proc.csv --md --image
```

Output files:
- Markdown: `<input>-summary.md`
- Image: `<input>-summary.png`

At least one output mode is required: `--console`, `--md`, or `--image`.

### 6) `postprocess-enery-analysis.py`

Purpose:
- Integrate `power_w` over `time_from_start_s` (trapezoidal integration) to compute energy.

Input:
- Processed CSV from `capture-power-analysis.py`.

Output:
- Default: `output/<input>.energy.csv`
- Columns: `source_file,input_file,duration_s,energy_j,energy_wh,energy_mwh`

Examples:

```bash
# Integrate one file
./.venv/bin/python postprocess-enery-analysis.py output/file.proc.csv

# Custom output path
./.venv/bin/python postprocess-enery-analysis.py output/file.proc.csv -o output/file.energy.csv
```

## Config Files

### `scope-config.toml`
Used by:
- `scope-config.py`
- `scope-capture.py`

Typical keys:
- `resource`
- `duration_seconds`
- optionally stored scope settings (`time_scale`, `ch1_scale`, etc.)

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

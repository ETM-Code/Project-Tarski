# Tarski Laptop Interface (Python + TUI)

Host-side tools for talking to the T1 devboard over serial, running calibration, and doing inference from the terminal.

This folder contains:

- `tarski_board.py`: Python board client and CLI calibration/inference workflow
- `tarski_tui.py`: Interactive Textual terminal UI built on top of `tarski_board.py`
- `main.py`: Minimal placeholder entrypoint (currently prints a hello message)

## What This Interface Does

The Python interface controls the board using the serial protocol implemented by `T1-devboard/interface` firmware. It can:

- connect and handshake with the board firmware
- program synapse shift-register bytes
- load DAC codes
- read measurements and output words
- run timed spike-sampling inference
- run full calibration (including layer-2 synapse characterization)

## Requirements

- Python `3.11+`
- A flashed T1 devboard connected over USB serial
- Python dependencies:
  - `pyserial`
  - `numpy`
  - `textual` (for TUI mode)

If you use `uv`, dependencies come from `pyproject.toml`.

## Setup

From this directory:

```bash
uv sync
```

If you prefer pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pyserial numpy textual
```

## Find Your Serial Port

On macOS, common device names look like:

- `/dev/tty.usbserial-XXXX`
- `/dev/tty.usbmodemXXXX`

Example check:

```bash
ls /dev/tty.usb*
```

## Python CLI (`tarski_board.py`)

`tarski_board.py` is the main automation interface. All high-level workflows are exposed via flags.

### 1) Handshake only

```bash
uv run python tarski_board.py --port /dev/tty.usbserial-XXXX
```

This connects and prints the firmware signature.

### 2) Program DAC addresses (hardware setup step)

```bash
uv run python tarski_board.py --port /dev/tty.usbserial-XXXX --setup-dacs
```

You must isolate DACs with jumpers when prompted. This procedure assigns unique I2C addresses (`0x60`, `0x61`, `0x62`).

### 3) Run full calibration

```bash
uv run python tarski_board.py --port /dev/tty.usbserial-XXXX --calibrate
```

Writes `calibration_results.json` in this folder.

Calibration phases include:

1. DAC channel discovery (active/inactive channels)
2. spike-time response curves and per-channel DAC scales
3. bulk synapse response verification
4. layer-2 synapse calibration (iterative weight programming and spike timing/count checks)

### 4) Run inference using a checkpoint

```bash
uv run python tarski_board.py \
  --port /dev/tty.usbserial-XXXX \
  --infer \
  --checkpoint ../gilgamesh/models/my_model.json \
  --data-dir ../gilgamesh/data \
  --samples 100
```

If `calibration_results.json` exists, it is loaded automatically for calibrated DAC mapping.

## Interactive TUI (`tarski_tui.py`)

The TUI is the easiest way to run connect/calibrate/model/infer flows interactively.

### Launch interactive mode

```bash
uv run python tarski_tui.py --port /dev/tty.usbserial-XXXX --checkpoint ../gilgamesh/models/my_model.json
```

Main actions:

- `Connect`: open serial and read firmware version
- `Setup DACs`: run DAC address programming flow
- `Calibrate`: run full calibration and save `calibration_results.json`
- `Load Model`: load checkpoint and optional saved calibration
- `Infer`: run one sample and show prediction/spike bars
- `Run 100`: run a short batch and report progress/accuracy

Keyboard shortcuts:

- `c`: calibrate
- `i`: infer current sample
- `n`: next sample
- `p`: previous sample
- `q`: quit

### Non-interactive TUI passthrough

`tarski_tui.py` can proxy to the CLI workflow:

```bash
uv run python tarski_tui.py \
  --port /dev/tty.usbserial-XXXX \
  --non-interactive \
  --calibrate \
  --infer \
  --checkpoint ../gilgamesh/models/my_model.json
```

This internally delegates to `tarski_board.py`.

## File Roles

- `tarski_board.py`
  - low-level serial framing (`ACK`/`NAK`/`TRN_END`)
  - command wrappers (`load_weights`, `load_dacs`, `read_output`, `run_inference`, `calib_l1_single`)
  - high-level workflows (`full_calibration`, `infer`, checkpoint loading)
- `tarski_tui.py`
  - Textual app for interactive operation
  - wraps board methods and logs live status
- `main.py`
  - currently a placeholder and not used by calibration/inference workflows

## Typical Workflow

1. Flash board firmware and connect USB serial.
2. Run DAC setup once (if needed for fresh hardware).
3. Run calibration and save `calibration_results.json`.
4. Load model checkpoint and run inference (CLI or TUI).

## Troubleshooting

- No serial connection:
  - verify the correct `--port`
  - replug the board and retry
- Timeout or missing ACK:
  - board may not be running matching firmware protocol
  - check baud and command framing consistency
- Poor inference quality:
  - rerun `--calibrate`
  - verify checkpoint path and data directory
- TUI starts but actions fail:
  - ensure `Connect` succeeds first
  - ensure model and MNIST files are reachable

## Notes

- `calibration_results.json` is local runtime output and should be treated as board-specific calibration data.
- `pyproject.toml` expects this `README.md` as project metadata readme.

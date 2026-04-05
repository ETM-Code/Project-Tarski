# Training Assets

This directory contains Python-side training/comparison assets used for Arduino-compatible network experiments.

## Layout

- `comparison/`: snnTorch/PyTorch comparison scripts, dependency list, and historical result snapshots.

## Run Comparison

From repository root:

```bash
./arduino-mnist/training/comparison/run_comparison.sh
```

Or run directly:

```bash
python3.11 arduino-mnist/training/comparison/snntorch_comparison.py --models baseline ann
```

# Training Assets

Python-side training and comparison assets for Arduino-compatible network experiments.

## Layout

- `comparison/`: snnTorch/PyTorch comparison scripts, dependency list, and historical result snapshots.

## Run

From repository root:

```bash
./arduino-mnist/training/comparison/run_comparison.sh
```

Or run the script directly:

```bash
python3.11 arduino-mnist/training/comparison/snntorch_comparison.py --models baseline ann
```

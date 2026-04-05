# Gilgamesh Neural Network Toolkit

Legacy Gilgamesh codebase retained for historical traceability.

This archive includes the earlier training/runtime stack that preceded the current standalone `Gilgamesh` repository layout.

## What Is Kept Here

- historical CLI and runtime code
- legacy configuration and dataset path conventions
- older SPICE comparison harnesses and outputs

## Legacy CLI Surface

Common operations retained in this archive:

- `train`: run training from design/config defaults
- `visualize`: emit sample traces and plots
- resume flow via checkpoint JSONs in `output/checkpoints/`

## Usage

```bash
cargo run -- train
cargo run -- visualize --sample 0 --neurons 8
cargo run -- train --resume output/checkpoints/latest.json --epochs 10
```

## Where to Look

- `configs/`: archived design/config JSON files
- `tools/`: legacy SPICE comparison utilities
- `output/` or `comparison_outputs/`: generated artifacts from older runs

## Note

For active development, use the standalone `Gilgamesh` repository.

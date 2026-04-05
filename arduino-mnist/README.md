# Arduino MNIST

Embedded inference project for Arduino-compatible targets, including:

- firmware/runtime code under `src/` and `include/`
- host-side serial/testing tooling under `software/`
- exported model artifacts under `model/`
- Python comparison and quantized training assets under `training/comparison/`

## Quick Start

From repository root:

```bash
# Build and flash with PlatformIO (if configured for your board)
cd arduino-mnist
platformio run

# Run host-side tooling
cd software
make
```

## Training / Comparison Workflow

The Python workflow previously under `gilgamesh/comparison/` now lives in:

- `arduino-mnist/training/comparison/`

See `arduino-mnist/training/README.md` for usage notes.

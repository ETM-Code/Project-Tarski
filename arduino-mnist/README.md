# Arduino MNIST

Arduino-compatible inference and board-interface project used as the digital baseline and integration path in Project Tarski.

## Contents

- firmware/runtime code under `src/` and `include/`
- host-side serial/testing tooling under `software/`
- exported model artifacts under `model/`
- training/comparison Python assets under `training/comparison/`

## Quick Start

From repository root:

```bash
cd arduino-mnist
platformio run

cd software
make
```

## Dataset

The MNIST dataset is not vendored in this repository. The training scripts
download it on first run (via `torchvision.datasets.MNIST(root="data", download=True)`),
so the `data/MNIST/` tree is created locally the first time you train.

## Training and Comparison

The Python training/comparison workflow lives in `arduino-mnist/training/comparison/`. See `training/README.md` for commands and environment guidance.

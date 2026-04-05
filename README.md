# Tarski

Tarski is a mixed hardware and software repository for a neuromorphic computing project: board design, emulator work, training code, measurement tooling, and report materials all live here.

This repository has accumulated a lot of parallel experiments over time. The goal of this cleanup pass is not to rewrite history, but to make the current layout understandable and presentable.

## Start Here

- [`emulator/README.md`](emulator/README.md): hardware-accurate board emulator and web UI
- [`gilgamesh/README.md`](gilgamesh/README.md): current Rust SNN training and hardware-matched model work
- [`arduino-mnist/README.md`](arduino-mnist/README.md): Arduino inference project and associated training assets
- [`scope-probe/README.md`](scope-probe/README.md): oscilloscope capture and power analysis tooling
- [`report/README.md`](report/README.md): final report sources and figure/material organisation
- [`docs/README.md`](docs/README.md): repository-level notes and top-level project documents
- [`references/README.md`](references/README.md): datasheets, primer material, vendored references, and shared assets
- [`archive/README.md`](archive/README.md): archived experiments and historical side tracks
- [`assets/README.md`](assets/README.md): legacy project assets and supporting documents

## Repository Map

### Top-level docs

- `docs/urgent-circuit-surgery.md`: hardware fixes discovered during validation
- `docs/cost-estimate.md`: project cost planning
- `docs/mnist-training-system-plan.md`: longer-form training system planning notes

### Active project areas

- `emulator/`: main emulator workspace, tests, frontend, configs, and analysis notes
- `gilgamesh/`: current training/runtime code for the spiking network stack
- `scope-probe/`: scripts for waveform capture and energy/power post-processing
- `report/`: final report drafts, figures, scripts, and source notes
- `Schematics/`: KiCad-era schematics and board-level design assets
- `t1-devboard/`: firmware/interface work for the hardware control board

### Supporting or historical areas

- `archive/gilgamesh-legacy/`: older generation of the training/simulation stack retained for reference
- `SPICE/`: SPICE experiments and neuron-level validation work
- `assets/`: presentations, exported documents, images, and assorted project collateral
- `references/`: datasheets, primer material, vendored libraries, and shared standalone assets
- `archive/`: exploratory or superseded side projects kept for historical context
- `arduino-mnist/`: embedded Arduino inference project and training artifacts
- `laptop-gui/`: separate GUI prototype work

## Working Conventions

For new additions, prefer:

- Markdown/docs: lowercase kebab-case names
- Scripts: snake_case names
- Directory names without spaces where practical
- Relative repository paths in documentation so references stay portable

Older historical directories keep their existing names to avoid unnecessary churn.

## Repo Hygiene Notes

- Build outputs, caches, virtual environments, and frontend bundles are ignored and should not be committed.
- The `full-version` branch preserves the pre-cleanup state of the repository.
- Use `python3 Utilities/repo_inventory.py` to print a quick map of the key project areas and important files.

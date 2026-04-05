# Project Tarski

Project Tarski is the integration repository for a discrete-component analogue spiking neural network platform for MNIST classification.

This repository keeps the project-level documentation, report workspace, hardware support material, and integration context. The main implementation codebases are linked as submodules:

- `emulator/` -> [Tarski-Emulator](https://github.com/ETM-Code/Tarski-Emulator)
- `gilgamesh/` -> [Gilgamesh](https://github.com/ETM-Code/Gilgamesh)
- `Schematics/` -> [Tarski-Schematics](https://github.com/ETM-Code/Tarski-Schematics)

## Report-Aligned Scope

The report covers the full workflow end-to-end:

1. LIF analogue circuit design and SPICE validation
2. Training and hardware-aware modelling in Gilgamesh
3. PCB design and hardware implementation
4. Emulator and firmware integration
5. Power and accuracy evaluation

Use this repository as the anchor for that full narrative, then work inside the dedicated repos for implementation changes.

## Repository Layout

- `report/`: final report drafting workspace and supporting assets
- `SPICE/`: circuit-level simulation workspaces and generators
- `arduino-mnist/`: Arduino-compatible inference project and host tooling
- `T1-devboard/`: interface firmware for board control
- `scope-probe/`: oscilloscope capture and power-analysis tooling
- `references/`: datasheets and stable background material
- `assets/`: collateral and exported artifacts
- `archive/`: superseded/legacy material kept for traceability
- `Utilities/`: lightweight maintenance scripts

## Clone With Submodules

```bash
git clone --recurse-submodules https://github.com/ETM-Code/Project-Tarski.git
```

If already cloned:

```bash
git submodule update --init --recursive
```

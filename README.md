# Project Tarski

Project Tarski is an attempt to build an analogue neuromorphic accelerator for low-power computing. The idea is simple to state and hard to do. Run a spiking neural network in analogue electronics, on ordinary discrete components, and classify MNIST digits while drawing a fraction of the power a digital chip needs for the same work.

Most AI runs on digital hardware that burns a lot of power. Brains don't. A brain computes in analogue, with spikes, on roughly 20 watts. Tarski takes that literally and builds it from parts you can buy off a reel: op-amps, transistors, resistors, capacitors. No custom silicon.

This repository is the anchor. It holds the project-level context and the report workspace, and it pulls the real implementation together as submodules.

## Status

This is a proof of concept, not a finished product. The board exists: a 350 by 350 mm, four-layer, 3,442-component PCB. It has been powered, reworked, and it spikes on stimulation. What it has not done yet is a full end-to-end MNIST classification on the bench. The board arrived two days after the project deadline, so bring-up is still ongoing.

So read the numbers accordingly. The accuracy figures (mid-80s on 6×6 MNIST) and the power figures come from the trained model and the hardware-accurate emulator. They are what the board is expected to reach, not a measured bench result. Bench validation is the next step.

## The submodules

- `emulator/`: [Tarski-Emulator](https://github.com/ETM-Code/Tarski-Emulator). Boots the real board firmware in simulation and couples it to a physics model of the analogue neurons.
- `gilgamesh/`: [Gilgamesh](https://github.com/ETM-Code/Gilgamesh). Trains the networks with the hardware's constraints built in, not an idealised model.
- `Schematics/`: [Tarski-Schematics](https://github.com/ETM-Code/Tarski-Schematics). The KiCad schematics and PCB layouts for the board itself.

## How the pieces fit

The work runs end to end.

1. Design the analogue leaky integrate-and-fire neuron and check it in SPICE.
2. Train a network in Gilgamesh against the real circuit behaviour.
3. Lay out the board in Tarski-Schematics and build it.
4. Emulate firmware and board together, before and after fabrication.
5. Measure power and accuracy.

Start here for the whole picture. Work inside the dedicated repos when you want to change something.

## Repository layout

- `report/`: report drafting workspace and supporting assets
- `SPICE/`: circuit-level simulation workspaces and generators
- `arduino-mnist/`: the Arduino digital baseline and host tooling
- `T1-devboard/`: interface firmware for board control
- `scope-probe/`: oscilloscope capture and power-analysis tooling
- `references/`: datasheets and background material
- `assets/`: collateral and exported artifacts
- `archive/`: superseded material, kept for traceability
- `Utilities/`: small maintenance scripts

## Clone with submodules

```bash
git clone --recurse-submodules https://github.com/ETM-Code/Project-Tarski.git
```

If you already cloned it:

```bash
git submodule update --init --recursive
```

## License

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

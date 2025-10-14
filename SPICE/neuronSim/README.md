
# Leaky Integrate-and-Fire (LIF) Analogue Neuron — ngspice + Python

This repository lets you generate and simulate a leaky integrate‑and‑fire analogue neuron in **ngspice**, with synapses whose **weights are set exclusively by resistances (digital potentiometers)**. 

You get two versions of the circuit:
1. **Fast/Easy-to-simulate**: behavioral models (ideal Vref, comparator as smooth nonlinearity, ideal switches).
2. **Detailed/Physically-aware**: includes non-idealities approximating the listed parts (OPA604 buffer for Vref, MAX4617 MUX Ron/Coff for reset path, MCP41HV51 digipot wiper/end resistance, comparator delay & offset, etc.).

The **overall network** (neuron + synapses + spike sources) is produced by `lif_network_generator.py`. The **neuron-only** netlist is produced by `lif_neuron_generator.py`. Both scripts read simple JSON files so you can quickly tweak values (leak resistor, threshold, virtual ground, synaptic spikes, etc.).

> **Default run:**  
> ```bash
> python lif_network_generator.py            # generates & runs FAST version, prompts to continue to DETAILED
> ```
> Or to generate the neuron alone:
> ```bash
> python lif_neuron_generator.py             # writes FAST and DETAILED neuron-only netlists
> ```

After a run, CSVs and plots are in `outputs/`.

---

## Structure

```
lif/
├─ lif_neuron_generator.py
├─ lif_network_generator.py
├─ defaults/
│  ├─ neuron_default.json
│  └─ network_default.json
├─ models/
│  └─ readme.txt
└─ outputs/
```

---

## Requirements

- Python 3.9+
- `ngspice` in your PATH (for simulation)
- Python libs: `numpy`, `pandas`, `matplotlib`

Install (example):
```bash
pip install numpy pandas matplotlib
# ngspice install depends on OS (e.g., brew install ngspice on macOS)
```

---

## Quick Start

1. **Generate + simulate fast model** (default):
   ```bash
   python lif_network_generator.py
   ```
   - Writes `outputs/lif_fast.cir`
   - Runs transient sim and saves `outputs/lif_fast.csv`
   - Plots: spikes, membrane voltage, membrane charge, comparator output

2. **Continue to detailed model** (prompted):
   - Writes `outputs/lif_detailed.cir`
   - Runs transient sim and saves `outputs/lif_detailed.csv`
   - Plots the same set with parasitic effects enabled

3. **Change parameters**: edit `defaults/*.json` or supply your own JSONs.
   ```bash
   python lif_network_generator.py --network defaults/network_default.json --neuron defaults/neuron_default.json --mode detailed --yes
   ```

---

## Design Notes (what’s modeled)

- **Virtual ground (Vref):** Default 2.5 V. In FAST: ideal DC source. In DETAILED: buffered by an `OPA604`-like op-amp macro (finite gain/GBW, output swing, and bias currents approximated).
- **Membrane:** `C_mem` to Vref in parallel with `R_leak` to Vref.
- **Synapses (excitatory by default):** Each spike source is a PWL **voltage** referenced to Vref, routed to the membrane through a **resistor only** (the *digital potentiometer*). Weight = 1/R.  
  - Inhibitory support: set `"type": "inhibitory"` in the JSON (source pulses go below Vref; the resistor weight is still the only gain element).
- **Comparator & reset:** When `V(mem) - Vref > V_th`, the comparator toggles high. Hysteresis prevents chatter. The output drives a **reset switch** closing a path from `mem` to `Vref` via a series resistor (to limit current).  
  - FAST: behavioral tanh comparator + ideal V‑controlled switch.  
  - DETAILED: adds input offset, propagation delay, limited output rails; reset path goes through a MAX4617‑like Ron plus external series resistor and Coff.

---

## Outputs

- `lif_fast.cir`, `lif_detailed.cir` — SPICE netlists
- `lif_*.csv` — time series (time, Vref, Vmem, comparator, each spike)
- `lif_*.png` — four figures (spikes, Vmem, charge, comparator output)

---

## Troubleshooting

- If ngspice is not found, the script will still emit the netlist(s). You can run them manually:
  ```bash
  ngspice -b -o outputs/ngspice_fast.log outputs/lif_fast.cir
  ```
- If plots don’t appear, ensure `matplotlib` is installed and you have permission to write into `outputs/`.

---

## License

MIT — do anything reasonable, attribution appreciated.

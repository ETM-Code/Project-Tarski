#!/usr/bin/env python3
"""
Generate a PASSIVE leaky integrate-and-fire (LIF) neuron subcircuit for ngspice.

This version uses NO op-amps - just passive RC integration with a comparator
for threshold detection. This matches what would be used in a real neuromorphic
ASIC or a simple PCB prototype.

Key differences from the TIA version:
- Membrane is a simple RC circuit (cap + leak resistor to Vref)
- No virtual ground - membrane voltage moves with input
- Current into membrane directly charges the capacitor
- Simpler, more power-efficient, but less linear current summation
"""
from __future__ import annotations
import json, argparse, os
from dataclasses import dataclass, field

# -------------------- Config dataclasses --------------------

@dataclass
class Supplies:
    vdd: float
    vref: float

@dataclass
class Membrane:
    C_mem_F: float
    R_leak_ohm: float

@dataclass
class Threshold:
    over_vref_V: float       # threshold above Vref
    hysteresis_V: float      # total hysteresis span

@dataclass
class ResetPath:
    enable: bool
    series_R_ohm: float
    mux_Ron_ohm: float
    mux_Roff_ohm: float

@dataclass
class PulseStretchCfg:
    enable: bool = True
    R_pw_ohm: float = 100e3
    C_pw_F: float = 100e-9

@dataclass
class SimCfg:
    tstop_s: float
    tstep_s: float

@dataclass
class PassiveNeuronConfig:
    name: str
    supplies: Supplies
    membrane: Membrane
    threshold: Threshold
    reset: ResetPath
    simulation: SimCfg
    pulse_stretch: PulseStretchCfg = field(default_factory=PulseStretchCfg)

    @staticmethod
    def load(json_path: str) -> "PassiveNeuronConfig":
        with open(json_path, "r") as f:
            d = json.load(f)
        return PassiveNeuronConfig(
            name=d.get("name", "lif_passive"),
            supplies=Supplies(**d["supplies"]),
            membrane=Membrane(**d["membrane"]),
            threshold=Threshold(**d["threshold"]),
            reset=ResetPath(**d["reset"]),
            simulation=SimCfg(**d["simulation"]),
            pulse_stretch=PulseStretchCfg(**d.get("pulse_stretch", {})),
        )

    @staticmethod
    def default() -> "PassiveNeuronConfig":
        """Create default configuration matching typical values."""
        return PassiveNeuronConfig(
            name="lif_passive",
            supplies=Supplies(vdd=5.0, vref=2.5),
            membrane=Membrane(C_mem_F=10e-9, R_leak_ohm=120e3),  # tau = 1.2ms
            threshold=Threshold(over_vref_V=0.8, hysteresis_V=0.05),
            reset=ResetPath(
                enable=True,
                series_R_ohm=200.0,
                mux_Ron_ohm=10.0,
                mux_Roff_ohm=1e9,
            ),
            simulation=SimCfg(tstop_s=0.05, tstep_s=1e-6),
            pulse_stretch=PulseStretchCfg(
                enable=True,
                R_pw_ohm=16.7e3,  # tau_pulse = 1.67ms
                C_pw_F=100e-9,
            ),
        )


def generate_passive_neuron(cfg: PassiveNeuronConfig) -> str:
    """
    Generate a PASSIVE LIF neuron subcircuit (no op-amps).

    Circuit topology:
    - Membrane: simple RC to Vref (cap from mem to ground, resistor from mem to vref)
    - Input: current injected directly into membrane node
    - Threshold: comparator (behavioral) comparing mem to Vref + threshold
    - Reset: switch that connects mem to vref through resistor when spike occurs
    - Pulse stretch: RC circuit on comparator output

    Pins: mem vref vdd comp_pulse sum
    - mem: membrane voltage node
    - vref: reference voltage
    - vdd: supply
    - comp_pulse: stretched spike output for downstream neurons
    - sum: current injection node (same as mem for passive)
    """
    s = []
    s.append(f"* ---------- {cfg.name} PASSIVE neuron subcircuit (no op-amps) ----------\n")
    s.append(f".subckt {cfg.name}_passive mem vref vdd comp_pulse sum\n")

    # For passive neuron, sum and mem are the same node
    # We'll use a small resistor to connect them (allows current measurement if needed)
    s.append("* Sum node connection (passive: sum connects directly to membrane)\n")
    s.append("Rsum sum mem 1\n")  # ~0 ohm, just for node separation

    # Passive membrane: capacitor to ground, leak resistor to Vref
    s.append("\n* Passive RC membrane (no op-amp)\n")
    s.append(f"* tau_m = {cfg.membrane.R_leak_ohm:.3g} * {cfg.membrane.C_mem_F:.3g} = {cfg.membrane.R_leak_ohm * cfg.membrane.C_mem_F * 1000:.3f} ms\n")
    s.append(f"Cmem mem 0 {cfg.membrane.C_mem_F}\n")
    s.append(f"Rleak mem vref {cfg.membrane.R_leak_ohm}\n")

    # Comparator (behavioral - soft tanh for numerical stability)
    # Fires when mem > vref + threshold
    vth = cfg.threshold.over_vref_V
    hyst = cfg.threshold.hysteresis_V

    s.append("\n* Threshold comparator (behavioral, with hysteresis)\n")
    s.append(f"* Threshold = Vref + {vth:.3f}V = {cfg.supplies.vref + vth:.3f}V\n")
    s.append(".param VLO=0.0 VHI=5.0\n")
    s.append(".param VSW=0.02\n")  # Soft transition width

    # Comparator: output high when V(mem) > V(vref) + threshold
    s.append(f"Bcomp comp_raw 0 V = VLO + (VHI - VLO) * (0.5 * (1 + tanh((V(mem) - V(vref) - {vth}) / VSW)))\n")

    # Small RC on comparator output for stability
    s.append("Rcomp comp_raw comp_out 10\n")
    s.append("Ccomp comp_out 0 1p\n")

    # Pulse stretching circuit
    if cfg.pulse_stretch.enable:
        tau_pulse = cfg.pulse_stretch.R_pw_ohm * cfg.pulse_stretch.C_pw_F
        s.append(f"\n* Pulse stretching circuit (tau_pulse = {tau_pulse*1000:.2f} ms)\n")
        s.append(".model DPW D(Is=1e-12 N=1.05 Rs=10 Cjo=1p)\n")
        s.append("Dpw comp_out comp_pulse DPW\n")  # Fast charge
        s.append(f"Rpw comp_pulse 0 {cfg.pulse_stretch.R_pw_ohm:.3g}\n")  # Slow discharge
        s.append(f"Cpw comp_pulse 0 {cfg.pulse_stretch.C_pw_F:.3g}\n")
    else:
        s.append("\n* No pulse stretching - direct connection\n")
        s.append("Rpw_bypass comp_out comp_pulse 1\n")

    # Reset path: switch that pulls membrane to Vref when spiking
    if cfg.reset.enable:
        s.append("\n* Reset path (pulls membrane to Vref on spike)\n")
        s.append(f"Rreset mem reset_node {cfg.reset.series_R_ohm}\n")
        s.append("Sreset reset_node vref comp_out 0 SWRESET\n")
        s.append(f".model SWRESET SW(Ron={cfg.reset.mux_Ron_ohm} Roff={cfg.reset.mux_Roff_ohm} Vt=2.5 Vh=0.1)\n")

    s.append("\n.ends\n")
    return "".join(s)


def generate_test_netlist(cfg: PassiveNeuronConfig, subckt_path: str, output_csv: str) -> str:
    """Generate a test netlist for the passive neuron."""
    return f"""* Passive LIF Neuron Test
* Generated by lif_neuron_generator_passive.py

.include {subckt_path}

* Power supplies
Vdd vdd 0 DC {cfg.supplies.vdd}
Vref vref 0 DC {cfg.supplies.vref}

* Neuron instance
* Pins: mem vref vdd comp_pulse sum
Xneuron mem vref vdd comp_pulse sum {cfg.name}_passive

* Constant current input (into sum/membrane node)
Iin 0 sum DC 1e-6

* Initial conditions (membrane at Vref)
.ic V(mem)={cfg.supplies.vref}

* Transient analysis
.tran {cfg.simulation.tstep_s} {cfg.simulation.tstop_s} 0 {cfg.simulation.tstep_s}

* Save data
.control
run
wrdata {output_csv} v(mem) v(comp_pulse)
.endc

.end
"""


def main():
    ap = argparse.ArgumentParser(description="Generate PASSIVE ngspice subcircuit for a LIF neuron (no op-amps).")
    ap.add_argument("--config", default=None, help="Path to neuron JSON config (uses defaults if not specified).")
    ap.add_argument("--outdir", default="outputs", help="Directory to write files.")
    ap.add_argument("--name", default="lif_passive", help="Neuron name.")
    args = ap.parse_args()

    if args.config:
        cfg = PassiveNeuronConfig.load(args.config)
    else:
        cfg = PassiveNeuronConfig.default()
        cfg.name = args.name

    os.makedirs(args.outdir, exist_ok=True)

    subckt = generate_passive_neuron(cfg)
    subckt_path = os.path.join(args.outdir, f"{cfg.name}_passive.subckt")
    with open(subckt_path, "w") as f:
        f.write(subckt)

    # Also generate a test netlist
    test_netlist = generate_test_netlist(
        cfg,
        subckt_path,
        os.path.join(args.outdir, "passive_output.csv")
    )
    test_path = os.path.join(args.outdir, f"{cfg.name}_test.cir")
    with open(test_path, "w") as f:
        f.write(test_netlist)

    print(f"Generated passive neuron subcircuit: {subckt_path}")
    print(f"Generated test netlist: {test_path}")
    print(f"\nTo run: ngspice -b {test_path}")


if __name__ == "__main__":
    main()

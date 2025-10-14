
#!/usr/bin/env python3
"""
Generate a leaky integrate-and-fire (LIF) neuron *subcircuit* netlist for ngspice.

- FAST/EASY: ideal Vref, behavioral comparator, ideal switch
- DETAILED: non-idealities approximating OPA604 buffer, MAX4617 switch Ron/Coff, comparator delay/offset

This file *only* builds the NEURON. The overall network (neuron + synapses + spikes) is created by lif_network_generator.py.
"""
from __future__ import annotations
import json, argparse, math, os, textwrap, sys
from dataclasses import dataclass

@dataclass
class Supplies:
    vdd: float
    vref: float
    opa604_vpos: float
    opa604_vneg: float

@dataclass
class Membrane:
    C_mem_F: float
    R_leak_ohm: float

@dataclass
class Threshold:
    over_vref_V: float
    hysteresis_V: float

@dataclass
class ResetPath:
    enable: bool
    series_R_ohm: float
    mux_Ron_ohm: float
    mux_Roff_ohm: float
    mux_Coff_F: float
    switch_Vt: float
    switch_Vh: float

@dataclass
class ComparatorCfg:
    offset_V: float
    prop_delay_s: float
    vlow_V: float
    vhigh_V: float
    gain_fast: float

@dataclass
class SimCfg:
    tstop_s: float
    tstep_s: float

@dataclass
class NeuronConfig:
    name: str
    supplies: Supplies
    membrane: Membrane
    threshold: Threshold
    reset: ResetPath
    comparator: ComparatorCfg
    simulation: SimCfg

    @staticmethod
    def load(json_path: str) -> "NeuronConfig":
        with open(json_path, "r") as f:
            d = json.load(f)
        return NeuronConfig(
            name=d.get("name","lif_neuron"),
            supplies=Supplies(**d["supplies"]),
            membrane=Membrane(**d["membrane"]),
            threshold=Threshold(**d["threshold"]),
            reset=ResetPath(**d["reset"]),
            comparator=ComparatorCfg(**d["comparator"]),
            simulation=SimCfg(**d["simulation"]),
        )

def lif_sanity_checks(cfg: NeuronConfig) -> list[str]:
    """Basic physical sanity checks vs. listed components."""
    issues = []
    # Comparator (NCS2250): VDD 1.8–5.5 V, rail-to-rail ±0.2 V beyond rails.
    if not (1.8 <= cfg.supplies.vdd <= 5.5):
        issues.append(f"Comparator supply vdd={cfg.supplies.vdd} V outside NCS2250 range (1.8–5.5V).")
    # OPA604 supply requirements ±4.5 to ±24 V
    if (cfg.supplies.opa604_vpos - cfg.supplies.opa604_vneg) < 9.0:
        issues.append("OPA604 total supply < 9 V — OPA604 needs at least ±4.5 V.")
    # Threshold location
    vth_hi = cfg.supplies.vref + cfg.threshold.over_vref_V + cfg.threshold.hysteresis_V/2
    vth_lo = cfg.supplies.vref + cfg.threshold.over_vref_V - cfg.threshold.hysteresis_V/2
    if vth_hi > cfg.supplies.vdd or vth_lo < 0:
        issues.append(f"Threshold window [{vth_lo:.3f}, {vth_hi:.3f}] V not contained in [0,{cfg.supplies.vdd}] comparator rails.")
    # RC
    tau = cfg.membrane.C_mem_F * cfg.membrane.R_leak_ohm
    if tau < 1e-4:
        issues.append(f"Membrane tau={tau*1e3:.2f} ms is very small; spikes may discharge too fast.")
    return issues

def _header(title: str) -> str:
    return f"* ---------- {title} ----------\n"

def generate_fast_neuron(cfg: NeuronConfig) -> str:
    """
    FAST neuron subckt:
      Nodes: mem vref vdd comp_out
      - ideal Vref (external in top-level netlist)
      - Membrane RC to vref
      - Behavioral comparator with tanh + external hysteresis node
      - Reset switch (ideal) controlled by comp_out (optional)
    """
    s = []
    s.append(_header(f"{cfg.name} FAST neuron subcircuit"))
    s.append(f".subckt {cfg.name}_fast mem vref vdd comp_out\n")
    # Membrane
    s.append(f"Cmem mem vref {cfg.membrane.C_mem_F}\n")
    s.append(f"Rleak mem vref {cfg.membrane.R_leak_ohm}\n")
    # Threshold & hysteresis reference
    s.append(f".param VTH0={cfg.threshold.over_vref_V}\n")
    s.append(f".param HYST={cfg.threshold.hysteresis_V}\n")
    # Comparator rails
    s.append(f".param VLO={cfg.comparator.vlow_V} VHI={cfg.comparator.vhigh_V}\n")
    # Dynamic threshold depends on output level -> simple Schmitt behavior
    s.append("* Threshold node includes hysteresis based on comparator output level\n")
    s.append("Bvth vth 0 V = V(vref) + VTH0 + HYST*((V(comp_out)-(VLO+VHI)/2)/(VHI-VLO))\n")
    # Behavioral comparator
    s.append("* Smooth comparator (tanh) to aid convergence\n")
    s.append(f"Bcomp comp_raw 0 V = VLO + (VHI-VLO)*0.5*(1+tanh({cfg.comparator.gain_fast}*(V(mem)-V(vth))))\n")
    # Small RC to emulate finite delay and avoid algebraic loop
    s.append("Rcout comp_raw comp_out 1k\n")
    s.append("Ccout comp_out 0 1n\n")
    # Reset path (ideal V-controlled switch)
    if cfg.reset.enable:
        s.append("* Reset path: mem -> series R -> ideal switch -> vref\n")
        s.append(f"Rreset mem reset_node {cfg.reset.series_R_ohm}\n")
        s.append(f"Sreset reset_node vref comp_out 0 SWRESET\n")
        s.append(f".model SWRESET SW(Ron=0.1 Roff=1e12 Vt={cfg.reset.switch_Vt} Vh={cfg.reset.switch_Vh})\n")
    s.append(".ends\n")
    return "".join(s)

def generate_detailed_neuron(cfg: NeuronConfig) -> str:
    """
    DETAILED neuron subckt:
      - Vref buffered by a simple OPA604-like macromodel (finite gain and GBW)
      - Comparator has input offset and explicit delay (via RC)
      - Reset through MAX4617-like Ron (+ external series resistor) and Coff
    """
    s = []
    s.append(_header(f"{cfg.name} DETAILED neuron subcircuit"))
    s.append(f".subckt {cfg.name}_detailed mem vref vdd comp_out vpos vneg\n")
    # Vref buffer (OPA604-like): simple single-pole op-amp model in unity gain
    s.append("* OPA604-like buffer for Vref (unity gain follower)\n")
    s.append("Ebuf vref_buf 0 vref 0 1e5\n")                 # finite DC gain
    s.append("Rbuf vref_buf vref 1k\n")
    s.append("Cbuf vref 0 80p\n")                            # sets a pole ~20 MHz for 1e5 gain/GBW notionally
    s.append("* Tie the external vref node to the buffered node\n")
    s.append("Rvr vref vref_buf 1m\n")
    # Membrane
    s.append(f"Cmem mem vref {cfg.membrane.C_mem_F}\n")
    s.append(f"Rleak mem vref {cfg.membrane.R_leak_ohm}\n")
    # Threshold and hysteresis
    s.append(f".param VTH0={cfg.threshold.over_vref_V}\n")
    s.append(f".param HYST={cfg.threshold.hysteresis_V}\n")
    s.append(f".param VLO={cfg.comparator.vlow_V} VHI={cfg.comparator.vhigh_V}\n")
    s.append("* Dynamic threshold w/ hysteresis\n")
    s.append("Bvth vth 0 V = V(vref) + VTH0 + HYST*((V(comp_out)-(VLO+VHI)/2)/(VHI-VLO))\n")
    # Comparator core: tanh + input offset + output RC for delay
    s.append("* Comparator with offset and RC delay\n")
    s.append(f"Bcomp comp_raw 0 V = VLO + (VHI-VLO)*0.5*(1+tanh(2000*(V(mem)-V(vth)-{cfg.comparator.offset_V})))\n")
    # Simple delay: RC on output
    rc = max(cfg.comparator.prop_delay_s/10, 1e-9)
    rout = max(cfg.comparator.prop_delay_s/rc, 1.0)
    s.append(f"Rcout comp_raw comp_out {rout}\n")
    s.append(f"Ccout comp_out 0 {rc}\n")
    # Reset through MAX4617-like switch (Ron, Coff) in series with external resistor
    if cfg.reset.enable:
        s.append("* Reset path through MAX4617-like switch in series with a limiting resistor\n")
        s.append(f"Rreset mem reset_node {cfg.reset.series_R_ohm}\n")
        s.append("Coff_reset reset_node vref {Coff}\n".replace("{Coff}", f"{cfg.reset.mux_Coff_F}"))
        s.append("Sreset reset_node vref comp_out 0 SWMUX\n")
        s.append(f".model SWMUX SW(Ron={cfg.reset.mux_Ron_ohm} Roff={cfg.reset.mux_Roff_ohm} Vt={cfg.reset.switch_Vt} Vh={cfg.reset.switch_Vh})\n")
    s.append(".ends\n")
    return "".join(s)

def main():
    ap = argparse.ArgumentParser(description="Generate ngspice subcircuits for a LIF neuron.")
    ap.add_argument("--neuron", default="defaults/neuron_default.json", help="Path to neuron JSON config.")
    ap.add_argument("--outdir", default="outputs", help="Directory to write files.")
    args = ap.parse_args()

    cfg = NeuronConfig.load(args.neuron)
    issues = lif_sanity_checks(cfg)
    if issues:
        print("Sanity checks found:", *("\n - "+x for x in issues), sep="")
    else:
        print("Sanity checks OK.")

    os.makedirs(args.outdir, exist_ok=True)
    fast = generate_fast_neuron(cfg)
    detailed = generate_detailed_neuron(cfg)

    with open(os.path.join(args.outdir, f"{cfg.name}_fast.subckt"), "w") as f:
        f.write(fast)
    with open(os.path.join(args.outdir, f"{cfg.name}_detailed.subckt"), "w") as f:
        f.write(detailed)

    print(f"Wrote: {args.outdir}/{cfg.name}_fast.subckt and {args.outdir}/{cfg.name}_detailed.subckt")

if __name__ == "__main__":
    main()

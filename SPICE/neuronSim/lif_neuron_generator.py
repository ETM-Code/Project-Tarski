#!/usr/bin/env python3
"""
Generate a leaky integrate-and-fire (LIF) neuron *subcircuit* netlist for ngspice.

- FAST/EASY: ideal Vref, behavioral comparator + small Schmitt, ideal switch
  (now also exposes a buffered analog_out)
- DETAILED: non-idealities approximating OPA604 buffer, MAX4617 switch Ron/Coff,
  comparator delay/offset, *physical hysteresis network*, and *adaptive threshold (Option C)*
  via RC-to-Vref and one-way injection from comp_out.

This file *only* builds the NEURON. The overall network (neuron + synapses + spikes)
is created by lif_network_generator.py.
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
    over_vref_V: float       # desired center above Vref (e.g., 0.8 V)
    hysteresis_V: float      # total Schmitt span (e.g., 0.05 V -> ±25 mV)

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
    # Comparator (NCS2250): VDD 1.8–5.5 V
    if not (1.8 <= cfg.supplies.vdd <= 5.5):
        issues.append(f"Comparator supply vdd={cfg.supplies.vdd} V outside NCS2250 range (1.8–5.5V).")
    # OPA604 supply requirements ±4.5 to ±24 V (only relevant if you actually use OPA604 rails)
    if (cfg.supplies.opa604_vpos - cfg.supplies.opa604_vneg) < 9.0:
        issues.append("OPA604 total supply < 9 V — OPA604 needs at least ±4.5 V.")
    # Threshold window inside rails
    vth_hi = cfg.supplies.vref + cfg.threshold.over_vref_V + cfg.threshold.hysteresis_V/2
    vth_lo = cfg.supplies.vref + cfg.threshold.over_vref_V - cfg.threshold.hysteresis_V/2
    if vth_hi > cfg.supplies.vdd or vth_lo < 0:
        issues.append(f"Threshold window [{vth_lo:.3f}, {vth_hi:.3f}] V not contained in [0,{cfg.supplies.vdd}] comparator rails.")
    # RC time constant not absurdly small
    tau = cfg.membrane.C_mem_F * cfg.membrane.R_leak_ohm
    if tau < 1e-4:
        issues.append(f"Membrane tau={tau*1e3:.2f} ms is very small; spikes may discharge too fast.")
    return issues

def _header(title: str) -> str:
    return f"* ---------- {title} ----------\n"

def generate_fast_neuron(cfg: NeuronConfig) -> str:
    """
    FAST neuron subckt:
      Nodes: mem vref vdd comp_out analog_out
      - ideal Vref (external in top-level netlist)
      - Membrane RC to vref
      - Behavioral comparator with tanh + external hysteresis contribution
      - Reset switch (ideal) controlled by comp_out (optional)
      - Buffered analog_out (Option A)
    """
    s = []
    s.append(_header(f"{cfg.name} FAST neuron subcircuit"))
    s.append(f".subckt {cfg.name}_fast mem vref vdd comp_out analog_out\n")
    # Membrane
    s.append(f"Cmem mem vref {cfg.membrane.C_mem_F}\n")
    s.append(f"Rleak mem vref {cfg.membrane.R_leak_ohm}\n")
    # Threshold & hysteresis reference (behavioral)
    s.append(f".param VTH0={cfg.threshold.over_vref_V}\n")
    s.append(f".param HYST={cfg.threshold.hysteresis_V}\n")
    s.append(f".param VLO={cfg.comparator.vlow_V} VHI={cfg.comparator.vhigh_V}\n")
    s.append("* Behavioral Schmitt-like threshold referenced to Vref\n")
    s.append("Bvth vth 0 V = V(vref) + VTH0 + HYST*((V(comp_out)-(VLO+VHI)/2)/(VHI-VLO))\n")
    # Comparator core (smooth tanh), small output RC
    s.append("* Smooth comparator (tanh) to aid convergence\n")
    s.append(f"Bcomp comp_raw 0 V = VLO + (VHI-VLO)*0.5*(1+tanh({cfg.comparator.gain_fast}*(V(mem)-V(vth))))\n")
    s.append("Rcout comp_raw comp_out 1k\n")
    s.append("Ccout comp_out 0 1n\n")
    # Reset path (ideal V-controlled switch)
    if cfg.reset.enable:
        s.append("* Reset path: mem -> series R -> ideal switch -> vref\n")
        s.append(f"Rreset mem reset_node {cfg.reset.series_R_ohm}\n")
        s.append(f"Sreset reset_node vref comp_out 0 SWRESET\n")
        s.append(f".model SWRESET SW(Ron=0.1 Roff=1e12 Vt={cfg.reset.switch_Vt} Vh={cfg.reset.switch_Vh})\n")
    # Analog output buffer (Option A) – 1× gain, small Rout
    s.append("* Buffered analog tap of (mem - vref)\n")
    s.append("Eana analog_out 0 mem vref 1\n")
    s.append("Rana analog_out 0 150\n")
    s.append(".ends\n")
    return "".join(s)

def generate_detailed_neuron(cfg: NeuronConfig) -> str:
    """
    DETAILED neuron subckt:
      Nodes: mem vref vdd comp_out vpos vneg analog_out
      - Vref buffered (OPA604-like follower)
      - Membrane RC to Vref
      - Real comparator hysteresis network (non-inverting Schmitt) centered near Vref+over_vref_V
      - Output RC to emulate finite delay and small offset
      - Reset through MAX4617-like Ron/Coff + series resistor
      - Adaptive threshold (Option C): C_adapt to Vref and one-way injection from comp_out
      - Buffered analog_out (Option A)
    """
    s = []
    s.append(_header(f"{cfg.name} DETAILED neuron subcircuit"))
    s.append(f".subckt {cfg.name}_detailed mem vref vdd comp_out vpos vneg analog_out\n")

    # --- Vref buffer (OPA604-like): unity follower with finite gain/bandwidth ---
    s.append("* OPA604-like buffer for Vref (unity gain follower)\n")
    s.append("Ebuf vref_buf 0 vref 0 1e5\n")       # large DC gain
    s.append("Rbuf vref_buf vref 1k\n")
    s.append("Cbuf vref 0 80p\n")
    s.append("* Tie external vref node to buffered node\n")
    s.append("Rvr vref vref_buf 1m\n")

    # --- Membrane RC ---
    s.append(f"Cmem mem vref {cfg.membrane.C_mem_F}\n")
    s.append(f"Rleak mem vref {cfg.membrane.R_leak_ohm}\n")

    # --- Real hysteresis network around comparator (+) node vth_node ---
    vhi = cfg.comparator.vhigh_V
    vlo = cfg.comparator.vlow_V
    dv_out = max(vhi - vlo, 1e-6)
    beta = cfg.threshold.hysteresis_V / dv_out   # ~0.01 for 50 mV over 5 V

    # Divider choice: 133k/62k centers ~Vref+0.8 V with VDD=5 V
    s.append("* Divider to set center threshold near Vref+over_vref_V\n")
    s.append("Rvh  vdd     vth_node 133k\n")      # VDD -> vth_node
    s.append("Rvl  vth_node vref    62k\n")       # vth_node -> Vref

    # Compute Rf for desired beta: beta = (1/Rf)/(1/Rf + 1/Rvh + 1/Rvl)
    g_div = (1.0/133000.0) + (1.0/62000.0)
    Rf = (1.0 / (beta * g_div / max(1.0 - beta, 1e-6)))
    Rf_ohm = max(min(Rf, 50e6), 100e3)
    s.append(f"Rf   comp_out vth_node {Rf_ohm:.3g}\n")  # ~4.22e6 for beta~0.01

    # --- Adaptive threshold (Option C) ---
    # (i) C_adapt from vth_node to Vref; (ii) one-way injection from comp_out via R_inject + Schottky
    s.append("* Adaptive threshold: RC to Vref + unidirectional injection from comp_out\n")
    s.append("Cadapt vth_node vref 100n\n")
    s.append(".model DADAPT D(Is=1e-6 N=1.05 Rs=2 Cjo=1p Eg=0.69)\n")   # simple Schottky-like diode
    s.append("Rinj  comp_out ninj 2.2Meg\n")
    s.append("Dinj  ninj vth_node DADAPT  ; anode at comp_out side, cathode at vth_node\n")

    # --- Comparator behavioral core (tanh), with offset and output RC for prop delay ---
    s.append(f".param VLO={vlo} VHI={vhi}\n")
    rc = max(cfg.comparator.prop_delay_s/10.0, 1e-9)
    rout = max(cfg.comparator.prop_delay_s/rc, 10.0)
    s.append("* Comparator core: input = mem - vth_node - offset\n")
    s.append(f"Bcomp comp_raw 0 V = VLO + (VHI-VLO)*0.5*(1+tanh(2000*( V(mem) - V(vth_node) - {cfg.comparator.offset_V} )))\n")
    s.append(f"Rcout comp_raw comp_out {rout}\n")
    s.append(f"Ccout comp_out 0 {rc}\n")

    # --- Reset path through MAX4617-like switch (Ron/Coff) ---
    if cfg.reset.enable:
        s.append("* Reset path: mem -> Rseries -> (switch Ron/Coff) -> Vref\n")
        s.append(f"Rreset mem reset_node {cfg.reset.series_R_ohm}\n")
        s.append(f"Coff_reset reset_node vref {cfg.reset.mux_Coff_F}\n")
        s.append("Sreset reset_node vref comp_out 0 SWMUX\n")
        s.append(f".model SWMUX SW(Ron={cfg.reset.mux_Ron_ohm} Roff={cfg.reset.mux_Roff_ohm} "
                 f"Vt={cfg.reset.switch_Vt} Vh={cfg.reset.switch_Vh})\n")

    # --- Analog output buffer (Option A) ---
    s.append("* Buffered analog tap of (mem - vref)\n")
    s.append("Eana analog_out 0 mem vref 1\n")
    s.append("Rana analog_out 0 150\n")

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


#!/usr/bin/env python3
"""
Build the full LIF circuit (neuron + synapses + spikes) and optionally run ngspice.

Usage:
  python lif_network_generator.py
  python lif_network_generator.py --network defaults/network_default.json --neuron defaults/neuron_default.json --mode detailed --yes
"""
from __future__ import annotations
import json, argparse, os, subprocess, shutil, math, textwrap, sys
from dataclasses import dataclass, field
from typing import List, Dict
import csv

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
except Exception as e:
    # Delay import errors until plotting stage
    np = pd = plt = None

# Import neuron configuration classes from neuron generator
from lif_neuron_generator import (
    Supplies, 
    Membrane, 
    Threshold, 
    ResetPath, 
    ComparatorCfg, 
    SimCfg, 
    NeuronConfig,
    generate_fast_neuron,
    generate_detailed_neuron
)

@dataclass
class Spike:
    t_ms: float
    width_ms: float
    amp_V: float

@dataclass
class Synapse:
    name: str
    type: str          # "excitatory" or "inhibitory"
    weight_ohm: float
    spikes: List[Spike]

@dataclass
class NetworkConfig:
    title: str
    neuron_json: str
    synapses: List[Synapse] = field(default_factory=list)

    @staticmethod
    def load(path: str) -> "NetworkConfig":
        with open(path, "r") as f:
            d = json.load(f)
        synapses = [Synapse(
            name=s["name"],
            type=s.get("type","excitatory"),
            weight_ohm=float(s["weight_ohm"]),
            spikes=[Spike(**sp) for sp in s.get("spikes",[])]
        ) for s in d.get("synapses",[])]
        return NetworkConfig(title=d.get("title","lif_network"), neuron_json=d["neuron_json"], synapses=synapses)

# -------- Utility --------
def ensure_outputs_dir():
    outdir = "outputs"
    os.makedirs(outdir, exist_ok=True)
    return outdir

def pretty(v): return f"{v:.3g}"

def estimate_membrane_step(cfg: NeuronConfig, Rw: float, A: float, width_s: float) -> float:
    """
    Approximate peak ΔVmem (above Vref) after a PWL step of amplitude A for duration width_s
    through a series resistor Rw into C to Vref with R_leak to Vref.
    """
    Rpar = 1.0 / (1.0/Rw + 1.0/cfg.membrane.R_leak_ohm)
    tau = Rpar * cfg.membrane.C_mem_F
    Vinf = A * (cfg.membrane.R_leak_ohm / (cfg.membrane.R_leak_ohm + Rw))
    alpha = 1.0 - math.exp(-width_s/tau) if tau > 0 else 1.0
    return Vinf * alpha

def head_controls_csv(filename_csv: str, tstep: float, tstop: float, signals: List[str]) -> str:
    sigs = " ".join(signals)
    return textwrap.dedent(f"""
    .control
      set filetype=ascii
      tran {tstep} {tstop}
      wrdata {filename_csv} time {sigs}
      quit
    .endc
    """)

def gen_spike_source(name: str, spikes: List[Spike], sign: float) -> str:
    """
    Build a PWL voltage source referenced to Vref.
    sign = +1 for excitatory (above Vref) or -1 for inhibitory (below Vref).
    """
    points = [(0.0, 0.0)]
    for sp in spikes:
        t0 = sp.t_ms/1000.0
        t1 = t0 + sp.width_ms/1000.0
        A  = sign * sp.amp_V
        # Piecewise: (t0,0) -> (t0+1us, A) -> (t1-1us, A) -> (t1,0) for clean edges
        eps = 1e-6
        points += [(t0, 0.0), (t0+eps, A), (max(t0+eps, t1-eps), A), (t1, 0.0)]
    # Deduplicate/monotonic
    points = sorted({(round(t,12), round(v,9)) for (t,v) in points})
    # Build PWL string
    flat = " ".join(f"{t} {v}" for t,v in points)
    return f"Vsrc_{name} n_{name} vref PWL({flat})\n"

def neuron_subckt_includes(neuron_subckt_fast: str, neuron_subckt_detailed: str) -> str:
    return neuron_subckt_fast + "\n" + neuron_subckt_detailed + "\n"

def build_netlist(cfgN: NetworkConfig, cfg: NeuronConfig, mode: str, out_csv: str) -> str:
    """
    mode: 'fast' or 'detailed'
    Synapses use series diode to block idle leak:
      excitatory:  Vsrc → Rweight → D(anode→cathode) → mem
      inhibitory:  Vsrc → Rweight → D(cathode→anode) → mem
    """
    title = f"* {cfgN.title} [{mode.upper()}]"
    lines = [title, ""]

    # Supplies
    lines.append(f"VDD vdd 0 {cfg.supplies.vdd}")
    lines.append(f"VREF vref 0 {cfg.supplies.vref}")
    if mode == "detailed":
        lines.append(f"VOPP vpos 0 {cfg.supplies.opa604_vpos}")
        lines.append(f"VOPN vneg 0 {cfg.supplies.opa604_vneg}")

    # Include subcircuits (regenerate on the fly)
    neuron_fast = generate_fast_neuron(cfg)
    neuron_det  = generate_detailed_neuron(cfg)
    lines.append(neuron_subckt_includes(neuron_fast, neuron_det))

    # Neuron instance
    if mode == "fast":
        lines.append(f"XNEU mem vref vdd comp ana {cfg.name}_fast")
    else:
        lines.append(f"XNEU mem vref vdd comp vpos vneg ana {cfg.name}_detailed")

    # ---- Hybrid-output monitor: comp_out (through diode) + ana (through small R) -> n_outmix
    # This does not alter your synapses. It only exposes what a shared hybrid line would look like.
    # Digital path: one-way via Schottky to avoid back-drive into comparator
    lines.append("Ra_out ana n_outmix 1k")
    lines.append("Dout   comp n_outmix D_SCHOTTKY")
    # Weak return so the node is numerically well-defined in SPICE even if both drivers are idle
    lines.append("Rout_weak n_outmix vref 1Meg")

    # One Schottky diode model (tuned to be gentle; tweak as needed)
    # ~0.25 V at ~0.1 mA, small Rs and Cjo to help convergence
    lines.append(".model D_SCHOTTKY D(Is=1e-6 N=1.05 Rs=2 Cjo=2p Vj=0.3 M=0.3 Eg=0.69)")

    # Synapses
    signals = ["v(vref)", "v(mem)", "v(comp)", "v(ana)", "v(n_outmix)"]

    for syn in cfgN.synapses:
        sign = +1.0 if syn.type.lower().startswith("excit") else -1.0

        # 1) Spike source relative to Vref (node that also serves as diode control)
        lines.append(gen_spike_source(syn.name, syn.spikes, sign))
        signals.append(f"v(n_{syn.name})")  # plot the absolute spike node

        # 2) Series weight + diode
        if mode == "fast":
            # Vsrc -> R -> (node) -> D -> mem
            lines.append(f"R_{syn.name} n_{syn.name} n_{syn.name}_r {syn.weight_ohm}")
            if sign > 0:
                # excitatory: anode at resistor side, cathode at mem
                lines.append(f"D_{syn.name} n_{syn.name}_r mem D_SCHOTTKY")
            else:
                # inhibitory: reverse orientation (current mem<-res when pulse is negative)
                lines.append(f"D_{syn.name} mem n_{syn.name}_r D_SCHOTTKY")
        else:
            # Include digipot parasitics in series before diode
            # Vsrc -> Rend(75) -> Rwiper(weight) -> node_r -> D -> mem
            lines.append(f"R_{syn.name}_A n_{syn.name} n_{syn.name}_w 75")
            lines.append(f"R_{syn.name}_W n_{syn.name}_w n_{syn.name}_r {syn.weight_ohm}")
            if sign > 0:
                lines.append(f"D_{syn.name} n_{syn.name}_r mem D_SCHOTTKY")
            else:
                lines.append(f"D_{syn.name} mem n_{syn.name}_r D_SCHOTTKY")

    # CSV write + transient control
    lines.append(head_controls_csv(out_csv, cfg.simulation.tstep_s, cfg.simulation.tstop_s, signals))
    lines.append(".end\n")
    return "\n".join(lines)

def run_ngspice(netlist_path: str, log_path: str) -> int:
    exe = shutil.which("ngspice")
    if not exe:
        print("ngspice not found on PATH. Netlist has been generated; you can run it manually later.")
        return 127
    cmd = [exe, "-b", "-o", log_path, netlist_path]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="defaults/network_default.json", help="Network JSON (synapses/spikes).")
    ap.add_argument("--neuron", default="defaults/neuron_default.json", help="Neuron JSON (R, C, thresholds, supplies).")
    ap.add_argument("--mode", choices=["fast","detailed","both"], default="both", help="Which model to run.")
    ap.add_argument("--yes", action="store_true", help="If set and mode=both, skip the prompt and run detailed after fast.")
    ap.add_argument("--norun", action="store_true", help="Generate netlists only; do not run ngspice.")
    args = ap.parse_args()
    return args

def safety_and_math_validation(cfgN: NetworkConfig, cfg: NeuronConfig) -> Dict[str,str]:
    """Compute key checks & predictions (text report) for series-diode synapses."""
    lines = []
    lines.append("== Sanity & math checks ==")
    if not (1.8 <= cfg.supplies.vdd <= 5.5):
        lines.append(f"WARNING: vdd={cfg.supplies.vdd} outside NCS2250 allowed range (1.8..5.5 V).")
    tau_leak = cfg.membrane.R_leak_ohm * cfg.membrane.C_mem_F
    lines.append(f"Membrane tau (Rleak*C): {tau_leak*1e3:.2f} ms")
    vth = cfg.supplies.vref + cfg.threshold.over_vref_V
    lines.append(f"Comparator nominal Vth: {vth:.3f} V (hyst ±{cfg.threshold.hysteresis_V/2:.3f} V)")

    # Conservative Schottky drop assumption
    VF = 0.25  # V, typical small-signal drop for our D_SCHOTTKY

    imax_cont = 6.5e-3
    for syn in cfgN.synapses:
        Amax = max((sp.amp_V for sp in syn.spikes), default=0.0)
        Aeff = max(Amax - VF, 0.0)
        Ipk = Aeff / max(syn.weight_ohm, 1e-12)
        if Ipk > 0.8*imax_cont:
            lines.append(f"WARNING: {syn.name} Ipk={Ipk*1e3:.2f} mA exceeds 80% of MCP41HV51(50k) continuous limit ({imax_cont*1e3:.1f} mA).")

        width_s = max((sp.width_ms for sp in syn.spikes), default=0)/1000.0

        # With diode isolation, inactive synapses do not shunt. Use Rw || Rleak only.
        Rw = syn.weight_ohm
        Rpar = 1.0 / (1.0/Rw + 1.0/cfg.membrane.R_leak_ohm)
        tau = Rpar * cfg.membrane.C_mem_F
        Vinf = Aeff * (cfg.membrane.R_leak_ohm / (cfg.membrane.R_leak_ohm + Rw))
        dv = Vinf * (1.0 - math.exp(-width_s/tau)) if tau > 0 else Vinf

        lines.append(
            f"{syn.name}: A_eff≈{Aeff:.3f} V, Rpar≈{Rpar/1e3:.2f} kΩ, τ≈{tau*1e3:.3f} ms, "
            f"V∞≈{Vinf:.3f} V, ΔV({width_s*1e3:.1f} ms)≈{dv:.3f} V"
        )

    if cfg.reset.enable:
        ron = cfg.reset.mux_Ron_ohm
        rser = cfg.reset.series_R_ohm
        ireset = 1.0/(ron+rser)  # A per volt
        lines.append(f"Reset path: I≈{ireset*1e3:.2f} mA/V. With 1 V delta, {ireset*1e3:.2f} mA flows.")
    return {"text":"\n".join(lines)}


def plot_results(csv_path: str, png_prefix: str):
    import numpy as np, matplotlib.pyplot as plt

    # Load numeric rows only (ignore comments/headers)
    rows = []
    with open(csv_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("*") or s[0].isalpha():
                continue
            try:
                rows.append([float(x) for x in s.split()])
            except ValueError:
                continue
    arr = np.array(rows, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        print(f"CSV parse error: got shape {arr.shape}")
        return

    ncols = arr.shape[1]
    paired_format = (ncols % 2 == 0) and np.allclose(arr[:,0], arr[:,2], atol=1e-14)

    # We always write base signals in this order from wrdata:
    # time, vref, vmem, vcomp, vana, v(n_outmix), synapses...
    if paired_format:
        val_cols = [2*i+1 for i in range(ncols//2)]
        time  = arr[:, val_cols[0]]
        vref  = arr[:, val_cols[1]]
        vmem  = arr[:, val_cols[2]]
        vcomp = arr[:, val_cols[3]]
        vana  = arr[:, val_cols[4]]
        vcomb = arr[:, val_cols[5]] if len(val_cols) > 5 else None
        syn   = arr[:, val_cols[6:]] if len(val_cols) > 6 else None
    else:
        time  = arr[:, 0]
        vref  = arr[:, 1]
        vmem  = arr[:, 2]
        vcomp = arr[:, 3]
        vana  = arr[:, 4]
        vcomb = arr[:, 5] if ncols > 5 else None
        syn   = arr[:, 6:] if ncols > 6 else None

    # Derived signals
    dv = vmem - vref
    C  = 33e-9
    Q  = C * dv

    # ---- Synaptic spikes ----
    if syn is not None and syn.shape[1] > 0:
        plt.figure()
        for k in range(syn.shape[1]):
            plt.plot(time*1e3, syn[:, k], label=f"syn{k+1}")
        plt.title("Synaptic spikes (absolute)")
        plt.xlabel("Time (ms)"); plt.ylabel("Voltage (V)"); plt.grid(True); plt.legend()
        plt.savefig(f"{png_prefix}_spikes.png", dpi=160); plt.close()

    # ---- Membrane ----
    plt.figure()
    plt.plot(time*1e3, vmem, label="Vmem")
    plt.plot(time*1e3, vref, '--', label="Vref")
    plt.title("Membrane voltage"); plt.xlabel("Time (ms)"); plt.ylabel("Voltage (V)")
    plt.grid(True); plt.legend(); plt.savefig(f"{png_prefix}_vmem.png", dpi=160); plt.close()

    # ---- Comparator (digital) output ----
    plt.figure()
    plt.plot(time*1e3, vcomp, label="Vcomp (digital output)")
    plt.title("Digital output (from comparator)")
    plt.xlabel("Time (ms)"); plt.ylabel("Voltage (V)"); plt.grid(True); plt.legend()
    plt.savefig(f"{png_prefix}_comp.png", dpi=160); plt.close()

    # ---- Analog output ----
    plt.figure()
    plt.plot(time*1e3, vana, color="orange", label="Analog output (Vmem - Vref)")
    plt.title("Analog output"); plt.xlabel("Time (ms)"); plt.ylabel("Voltage (V)")
    plt.grid(True); plt.legend(); plt.savefig(f"{png_prefix}_analog.png", dpi=160); plt.close()

    # ---- Combined output after diode mixing ----
    if vcomb is not None:
        plt.figure()
        plt.plot(time*1e3, vana, label="Analog out (pre-mix)")
        plt.plot(time*1e3, vcomp, label="Comparator out (pre-mix)")
        plt.plot(time*1e3, vcomb, '--', linewidth=2, label="Combined output (post diode mix)")
        plt.title("Hybrid output (analog + digital)")
        plt.xlabel("Time (ms)"); plt.ylabel("Voltage (V)")
        plt.grid(True); plt.legend()
        plt.savefig(f"{png_prefix}_hybrid.png", dpi=160); plt.close()

    # ---- Membrane charge ----
    plt.figure()
    plt.plot(time*1e3, Q, label="Charge Q = C·ΔV")
    plt.title("Membrane charge accumulation")
    plt.xlabel("Time (ms)"); plt.ylabel("Coulombs")
    plt.grid(True); plt.legend(); plt.savefig(f"{png_prefix}_charge.png", dpi=160); plt.close()


def main():
    args = parse_args()
    outdir = ensure_outputs_dir()
    cfgN = NetworkConfig.load(args.network)
    cfg = NeuronConfig.load(args.neuron)

    # Validation + predictions
    report = safety_and_math_validation(cfgN, cfg)["text"]
    print(report)

    # Build & maybe run the requested modes
    wants = ["fast"] if args.mode == "fast" else (["detailed"] if args.mode == "detailed" else ["fast","detailed"])

    for i,mode in enumerate(wants):
        csv_name = f"lif_{mode}.csv"
        cir_name = f"lif_{mode}.cir"
        cir_path = os.path.join(outdir, cir_name)
        csv_path = os.path.join(outdir, csv_name)
        csv_abs  = os.path.abspath(csv_path)
        netlist = build_netlist(cfgN, cfg, mode, csv_abs)
        with open(cir_path, "w") as f:
            f.write(netlist)
        print(f"Wrote {cir_path}")
        if not args.norun:
            log_path = os.path.join(outdir, f"ngspice_{mode}.log")
            rc = run_ngspice(cir_path, log_path)
            # If ngspice wrote to CWD, pull it into outputs for plotting
            if rc == 0 and not os.path.exists(csv_path):
                alt = os.path.abspath(os.path.basename(csv_path))
                if os.path.exists(alt):
                    try:
                        shutil.move(alt, csv_path)
                    except Exception:
                        pass
            if rc == 0 and os.path.exists(csv_path):
                print(f"Sim OK -> {csv_path}")
                try:
                    plot_results(csv_path, os.path.join(outdir, f"lif_{mode}"))
                    print(f"Plots saved to {outdir}/lif_{mode}_*.png")
                except Exception as e:
                    print(f"Plotting failed: {e}")
            else:
                print(f"ngspice returned {rc}. Check log: {log_path}")
        # If both requested, and we just finished fast, prompt unless --yes
        if args.mode == "both" and i == 0 and not args.yes and not args.norun:
            try:
                go = input("Run DETAILED model now? [y/N]: ").strip().lower()
            except EOFError:
                go = "n"
            if go != "y":
                print("Skipping DETAILED run.")
                break

if __name__ == "__main__":
    main()

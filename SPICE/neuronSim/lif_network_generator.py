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
except Exception:
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

# ------------------ Data classes ------------------

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

# ------------------ Utilities ------------------

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
    """
    Emit the ngspice control block. We force a single time scale and ask for
    vector names in the file header so the parser can map columns by name.
    """
    sigs = " ".join(signals)
    return textwrap.dedent(f"""
    .control
      set filetype=ascii
      set wr_singlescale
      set wr_vecnames
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
        eps = 1e-6
        points += [(t0, 0.0), (t0+eps, A), (max(t0+eps, t1-eps), A), (t1, 0.0)]
    points = sorted({(round(t,12), round(v,9)) for (t,v) in points})
    flat = " ".join(f"{t} {v}" for t,v in points)
    return f"Vsrc_{name} n_{name} vref PWL({flat})\n"

def neuron_subckt_includes(neuron_subckt_fast: str, neuron_subckt_detailed: str) -> str:
    return neuron_subckt_fast + "\n" + neuron_subckt_detailed + "\n"

# ------------------ Netlist builder ------------------

def build_netlist(cfgN: NetworkConfig, cfg: NeuronConfig, mode: str, out_csv: str) -> str:
    title = f"* {cfgN.title} [{mode.upper()}]"
    lines = [title, ""]

    # Supplies
    lines.append(f"VDD vdd 0 {cfg.supplies.vdd}")
    lines.append(f"VREF vref 0 {cfg.supplies.vref}")

    # Include subcircuits (regen on the fly)
    lines.append(generate_fast_neuron(cfg))
    lines.append(generate_detailed_neuron(cfg))

    # Neuron instance
    if mode == "fast":
        lines.append(f"XNEU mem vref vdd comp ana sum {cfg.name}_fast")
    else:
        lines.append(f"XNEU mem vref vdd comp vpos vneg ana sum {cfg.name}_detailed")

    # --- Hybrid combiner: match PCB ---
    # Analog path: 100 Ω from analog op-amp to the output node
    # Digital path: Schottky from comparator to the same node (diode-OR)
    # Light load to ground so the node is defined when nobody drives it
    lines += [
        "Ra_out   ana n_outmix 100",
        "Diso     comp n_outmix D_SCHOTTKY",
        "Rload    n_outmix 0 10k",
        ".model   D_SCHOTTKY D(Is=1e-6 N=1.05 Rs=2 Cjo=2p Vj=0.3 M=0.3 Eg=0.69)"
    ]

    # Signals to write (order matters for the plotter)
    signals = ["v(vref)", "v(mem)", "v(comp)", "v(ana)", "v(n_outmix)", "v(sum)"]

    # Synapses
    for syn in cfgN.synapses:
        sign = +1.0 if syn.type.lower().startswith("excit") else -1.0
        lines.append(gen_spike_source(syn.name, syn.spikes, sign))
        signals.append(f"v(n_{syn.name})")
        lines.append(f"R_{syn.name} n_{syn.name} sum {syn.weight_ohm}")

    # Control block (single time column)
    lines.append(head_controls_csv(out_csv, cfg.simulation.tstep_s, cfg.simulation.tstop_s, signals))
    lines.append(".end\n")
    return "\n".join(lines)

# ------------------ Runner ------------------

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
    ap.add_argument("--neuron",  default="defaults/neuron_default.json",   help="Neuron JSON (R, C, thresholds, supplies).")
    ap.add_argument("--mode", choices=["fast","detailed","both"], default="both", help="Which model to run.")
    ap.add_argument("--yes", action="store_true", help="If set and mode=both, skip the prompt and run detailed after fast.")
    ap.add_argument("--norun", action="store_true", help="Generate netlists only; do not run ngspice.")
    return ap.parse_args()

# ------------------ Validation ------------------

def safety_and_math_validation(cfgN: NetworkConfig, cfg: NeuronConfig) -> Dict[str,str]:
    lines = []
    lines.append("== Sanity & math checks ==")
    if not (1.8 <= cfg.supplies.vdd <= 5.5):
        lines.append(f"WARNING: vdd={cfg.supplies.vdd} outside NCS2250 allowed range (1.8..5.5 V).")
    tau_leak = cfg.membrane.R_leak_ohm * cfg.membrane.C_mem_F
    lines.append(f"Membrane tau (Rleak*C): {tau_leak*1e3:.2f} ms")
    vth = cfg.supplies.vref + cfg.threshold.over_vref_V
    lines.append(f"Comparator nominal Vth: {vth:.3f} V (hyst ±{cfg.threshold.hysteresis_V/2:.3f} V)")

    VF = 0.25
    imax_cont = 6.5e-3
    for syn in cfgN.synapses:
        Amax = max((sp.amp_V for sp in syn.spikes), default=0.0)
        Aeff = max(Amax - VF, 0.0)
        Ipk = Aeff / max(syn.weight_ohm, 1e-12)
        if Ipk > 0.8*imax_cont:
            lines.append(f"WARNING: {syn.name} Ipk={Ipk*1e3:.2f} mA exceeds 80% of MCP41HV51(50k) continuous ({imax_cont*1e3:.1f} mA).")

        width_s = max((sp.width_ms for sp in syn.spikes), default=0)/1000.0
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
        ireset = 1.0/(ron+rser)
        lines.append(f"Reset path: I≈{ireset*1e3:.2f} mA/V. With 1 V delta, {ireset*1e3:.2f} mA flows.")
    return {"text":"\n".join(lines)}

# ------------------ Plotter ------------------

def plot_results(csv_path: str, png_prefix: str):
    import numpy as np, matplotlib.pyplot as plt

    # -------- Load file, grab vector names if present, collect numeric rows --------
    names = None
    rows = []
    with open(csv_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # Header with vector names (enabled by set wr_vecnames)
            if s.lower().startswith("index "):          # e.g., "Index   time   v(vref)  ..."
                parts = s.split()
                # Some ngspice builds write "Index" then vector names; first is "Index", second "time"
                names = [p for p in parts[1:]]          # drop "Index"
                continue
            # Skip comments
            if s.startswith("*"):
                continue
            # Data row
            try:
                rows.append([float(x) for x in s.split()])
            except ValueError:
                pass

    arr = np.array(rows, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        print(f"CSV parse error: got shape {arr.shape}")
        return

    # -------- Recover columns by name (preferred) or by layout fallback --------
    def build_by_name():
        # With wr_vecnames and wr_singlescale, names should look like:
        # ["time", "v(vref)", "v(mem)", "v(comp)", "v(ana)", "v(n_outmix)", "v(sum)", "v(xneu.vdef)", ...]
        if names is None:
            return None
        # Some ngspice prints duplicate "time" column once again; accept either 1 or 2
        # If two times exist, they will be the first two columns of arr.
        # Map every requested vector if present.
        want = [
            "time", "v(vref)", "v(mem)", "v(comp)", "v(ana)", "v(n_outmix)", "v(sum)",
            "v(xneu.vdef)", "v(xneu.vtheta)", "v(xneu.vtheta_rel)", "v(xneu.comp_raw)"
        ]
        # When there are two 'time' columns, keep the first data time (the second column in arr)
        idxmap = {}
        # Build a case-insensitive map
        lower_map = {n.lower(): i for i, n in enumerate(names)}
        for key in want:
            i = lower_map.get(key.lower(), None)
            if i is not None:
                # If ngspice wrote two time columns, arr has 1 more column at the front than names suggests.
                idxmap[key] = i if names[0].lower() != "time" or arr.shape[1] == len(names) else i+1
        return idxmap

    idx = build_by_name()

    if idx:
        time = arr[:, idx["time"]]
        def get(name): 
            j = idx.get(name); 
            return arr[:, j] if j is not None else None
        vref  = get("v(vref)")
        vmem  = get("v(mem)")
        vcomp = get("v(comp)")
        vana  = get("v(ana)")
        vcomb = get("v(n_outmix)")
        vsum  = get("v(sum)")
        vdef_dbg   = get("v(xneu.vdef)")
        vtheta_dbg = get("v(xneu.vtheta)") or get("v(xneu.vtheta_rel)")
        vcomp_raw  = get("v(xneu.comp_raw)")

        # Any remaining columns (spikes) are everything not in idx and not "time"
        used = set(idx.values())
        syn_cols = [k for k in range(arr.shape[1]) if k not in used]
        syn = arr[:, syn_cols] if len(syn_cols) else None
    else:
        # -------- Fallback: original robust pairing detector --------
        time = arr[:, 0]
        def is_time(col, tol=1e-12): return np.allclose(col, time, atol=tol, rtol=0)
        vals_cols, c = [], 1
        while c < arr.shape[1]:
            if is_time(arr[:, c]):
                if c + 1 < arr.shape[1]: vals_cols.append(c + 1)
                c += 2
            else:
                vals_cols.append(c); c += 1
        vals = arr[:, vals_cols]
        def col(i): return vals[:, i] if i < vals.shape[1] else None
        vref  = col(0); vmem = col(1); vcomp = col(2); vana = col(3); vcomb = col(4); vsum = col(5)
        syn   = vals[:, 6:] if vals.shape[1] > 6 else None
        vdef_dbg = vtheta_dbg = vcomp_raw = None

    print(f"Simulated time range: {time[0]:.4f} s → {time[-1]:.4f} s")

    # -------- Sanity print --------
    def rng(x): 
        return (float(np.nanmin(x)), float(np.nanmax(x))) if x is not None else None
    print("Ranges:",
          f"vref {rng(vref)}  vmem {rng(vmem)}  vcomp {rng(vcomp)}  "
          f"vana {rng(vana)}  v(n_outmix) {rng(vcomb)}  v(sum) {rng(vsum)}")

    # -------- Basic derived --------
    dv_rc  = vmem - vref if (vmem is not None and vref is not None) else None
    if vsum is None and vref is not None: vsum = vref
    dv_cap = (vmem - vsum) if (vmem is not None and vsum is not None) else None
    C = 33e-9
    Q = C * dv_cap if dv_cap is not None else None

    # -------- Plots --------
    if syn is not None and syn.size:
        plt.figure()
        for k in range(syn.shape[1]): plt.plot(time*1e3, syn[:, k], label=f"syn{k+1}")
        plt.title("Synaptic spikes (absolute)"); plt.xlabel("Time (ms)"); plt.ylabel("Voltage (V)")
        plt.grid(True); plt.legend(); plt.savefig(f"{png_prefix}_spikes.png", dpi=160); plt.close()

    if vmem is not None and vref is not None:
        plt.figure()
        plt.plot(time*1e3, vmem, label="Vmem")
        plt.plot(time*1e3, vref, '--', label="Vref")
        plt.title("Membrane voltage"); plt.xlabel("Time (ms)"); plt.ylabel("Voltage (V)")
        plt.grid(True); plt.legend(); plt.savefig(f"{png_prefix}_vmem.png", dpi=160); plt.close()

    if vcomp is not None:
        plt.figure()
        plt.plot(time*1e3, vcomp, label="Vcomp (comparator out)")
        plt.title("Digital output (from comparator)"); plt.xlabel("Time (ms)"); plt.ylabel("Voltage (V)")
        plt.grid(True); plt.legend(); plt.savefig(f"{png_prefix}_comp.png", dpi=160); plt.close()

    if vana is not None:
        plt.figure()
        plt.plot(time*1e3, vana, label="Analog out (Vmem - Vref)")
        plt.title("Analog output"); plt.xlabel("Time (ms)"); plt.ylabel("Voltage (V)")
        plt.grid(True); plt.legend(); plt.savefig(f"{png_prefix}_analog.png", dpi=160); plt.close()

    if vcomb is not None:
        plt.figure()
        plt.plot(time*1e3, vcomb, '--', linewidth=2, label="Combined (post-diode mix)")
        if vcomp is not None: plt.plot(time*1e3, vcomp, label="Comparator (pre-mix)")
        if vana  is not None: plt.plot(time*1e3, vana,  label="Analog (pre-mix)")
        plt.title("Hybrid output (analog + digital)"); plt.xlabel("Time (ms)"); plt.ylabel("Voltage (V)")
        plt.grid(True); plt.legend(); plt.savefig(f"{png_prefix}_hybrid.png", dpi=160); plt.close()

    if dv_rc is not None or dv_cap is not None:
        plt.figure()
        if dv_rc  is not None:  plt.plot(time*1e3, dv_rc,  label="ΔV (Vmem - Vref) [RC]")
        if dv_cap is not None:  plt.plot(time*1e3, dv_cap, label="Vcap = Vmem - Vsum [TIA]")
        plt.title("Membrane / integrator deltas"); plt.xlabel("Time (ms)"); plt.ylabel("Volts (V)")
        plt.grid(True); plt.legend(); plt.savefig(f"{png_prefix}_deltas.png", dpi=160); plt.close()

    if Q is not None:
        plt.figure()
        plt.plot(time*1e3, Q, label="Charge Q = C·(Vmem - Vsum)")
        plt.title("Membrane charge accumulation"); plt.xlabel("Time (ms)"); plt.ylabel("Coulombs")
        plt.grid(True); plt.legend(); plt.savefig(f"{png_prefix}_charge.png", dpi=160); plt.close()

    # -------- Optional comparator debug (if we exported it) --------
    if vdef_dbg is not None or (vtheta_dbg is not None):
        plt.figure()
        if vdef_dbg   is not None: plt.plot(time*1e3, vdef_dbg,   label="vdef  = Vref - Vmem")
        if vtheta_dbg is not None: plt.plot(time*1e3, vtheta_dbg, label="vtheta (rel)")
        if (vdef_dbg is not None) and (vtheta_dbg is not None):
            plt.plot(time*1e3, vdef_dbg - vtheta_dbg, label="margin (vdef - vtheta)")
        if vcomp_raw is not None:
            plt.plot(time*1e3, vcomp_raw, '--', label="comp_raw (internal)")
        if vcomp is not None:
            plt.plot(time*1e3, vcomp, ':', label="comp_out (pin)")
        plt.title("Comparator internals"); plt.xlabel("Time (ms)"); plt.ylabel("Volts (V)")
        plt.grid(True); plt.legend(); plt.savefig(f"{png_prefix}_comp_debug.png", dpi=160); plt.close()

        # Quick sanity: comp should be ~0 or ~VDD most of the time
        if vcomp is not None:
            vmin, vmax = float(np.nanmin(vcomp)), float(np.nanmax(vcomp))
            if not (vmin > -0.1 and vmax < 5.1):  # loose bounds around 0..5
                print(f"[warn] v(comp) range looks off for a digital node: {vmin:.3g}..{vmax:.3g}")

# ------------------ Main ------------------

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

if __name__ == "__main__":
    main()
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
from typing import List, Dict, Optional, Any

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

@dataclass
class SimData:
    time: Any
    vectors: Dict[str, Any]
    names: Optional[List[str]]
    raw: Any
    source: str = "fallback"
    labels: Dict[str, str] = field(default_factory=dict)

    def get(self, name: str) -> Any:
        """Case-insensitive accessor for a recorded vector."""
        return self.vectors.get(name.strip().lower())

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
      * Force uniform output grid and small files
      tran {tstep} {tstop} 0 {tstep}
      linearize
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

    # Global simulation options to keep output compact & stable
    lines.append(".options method=gear maxord=2 reltol=2e-3 trtol=7")

    # Include only the subcircuit for the selected mode (avoid duplicate macro defs)
    if mode == "fast":
        lines.append(generate_fast_neuron(cfg))
    else:
        lines.append(generate_detailed_neuron(cfg))

    # Neuron instance
    if mode == "fast":
        lines.append(f"XNEU mem vref vdd comp ana sum {cfg.name}_fast")
    else:
        # inside build_netlist()
        lines.append(f"XNEU mem vref vdd comp ana sum {cfg.name}_detailed")

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
        signals.append(f"i(Vsrc_{syn.name})")
        lines.append(f"R_{syn.name} n_{syn.name} sum {syn.weight_ohm}")

    # Record supply currents for power estimation
    signals.append("i(VDD)")
    signals.append("i(VREF)")

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

# ------------------ CSV helpers ------------------

def canonical_signal_name(name: str) -> str:
    return name.strip().lower()


def load_sim_data(csv_path: str) -> SimData:
    import numpy as np

    names = None
    rows = []
    with open(csv_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("*"):
                continue
            parts = s.split()
            if not parts:
                continue
            if names is None:
                try:
                    [float(x) for x in parts]
                except ValueError:
                    if parts[0].lower() == "index" and len(parts) > 1:
                        parts = parts[1:]
                    names = parts
                    continue
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                pass

    arr = np.array(rows, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"CSV parse error: got shape {arr.shape}")

    if names and len(names) + 1 == arr.shape[1]:
        names = ["index"] + names
    if names and len(names) != arr.shape[1]:
        names = None

    vectors: Dict[str, Any] = {}
    labels: Dict[str, str] = {}
    if names:
        for idx, raw_name in enumerate(names):
            lname = canonical_signal_name(raw_name)
            if lname == "index":
                continue
            vectors[lname] = arr[:, idx]
            labels[lname] = raw_name
        time = vectors.get("time", arr[:, 0])
        source = "named"
    else:
        time = arr[:, 0]
        vectors["time"] = time
        labels["time"] = "time"

        def is_time(col, tol=1e-12):
            return np.allclose(col, time, atol=tol, rtol=0)

        vals_cols, c = [], 1
        while c < arr.shape[1]:
            if is_time(arr[:, c]):
                if c + 1 < arr.shape[1]:
                    vals_cols.append(c + 1)
                c += 2
            else:
                vals_cols.append(c)
                c += 1
        vals = arr[:, vals_cols] if vals_cols else arr[:, 1:]
        base = ["v(vref)", "v(mem)", "v(comp)", "v(ana)", "v(n_outmix)", "v(sum)"]

        def val_col(i):
            if vals.ndim == 1:
                return vals if i == 0 else None
            return vals[:, i] if 0 <= i < vals.shape[1] else None

        for idx, name in enumerate(base):
            vec = val_col(idx)
            if vec is not None:
                cname = canonical_signal_name(name)
                vectors[cname] = vec
                labels[cname] = name
        source = "fallback"

    return SimData(time=time, vectors=vectors, names=names, raw=arr, source=source, labels=labels)


# ------------------ Plotter ------------------

def plot_results(csv_path: str, png_prefix: str, synapses: Optional[List[Synapse]] = None, sim_data: Optional[SimData] = None) -> Optional[SimData]:
    import numpy as np, matplotlib.pyplot as plt

    sim = sim_data or load_sim_data(csv_path)
    time = sim.time
    if time is None or len(time) == 0:
        print(f"CSV parse error: got shape {sim.raw.shape}")
        return sim

    vref = sim.get("v(vref)")
    vmem = sim.get("v(mem)")
    vcomp = sim.get("v(comp)")
    vana = sim.get("v(ana)")
    vcomb = sim.get("v(n_outmix)")
    vsum = sim.get("v(sum)")
    vdef_dbg = sim.get("v(xneu.vdef)")
    vtheta_dbg = sim.get("v(xneu.vtheta)") or sim.get("v(xneu.vtheta_rel)")
    vcomp_raw = sim.get("v(xneu.comp_raw)")

    print(f"Simulated time range: {time[0]:.4f} s → {time[-1]:.4f} s")

    def rng(x):
        return (float(np.nanmin(x)), float(np.nanmax(x))) if x is not None else None

    print("Ranges:",
          f"vref {rng(vref)}  vmem {rng(vmem)}  vcomp {rng(vcomp)}  "
          f"vana {rng(vana)}  v(n_outmix) {rng(vcomb)}  v(sum) {rng(vsum)}")

    dv_rc = vmem - vref if (vmem is not None and vref is not None) else None
    if vsum is None and vref is not None:
        vsum = vref
    dv_cap = (vmem - vsum) if (vmem is not None and vsum is not None) else None
    C = 33e-9
    Q = C * dv_cap if dv_cap is not None else None

    syn_series: List[Any] = []
    syn_labels: List[str] = []
    if synapses:
        for idx, syn in enumerate(synapses, 1):
            key = canonical_signal_name(f"v(n_{syn.name})")
            vec = sim.get(key)
            if vec is None:
                vec = sim.get(f"v({syn.name})")
            if vec is not None:
                syn_series.append(vec)
                syn_labels.append(syn.name or f"syn{idx}")
    if not syn_series:
        for key, label in sim.labels.items():
            if key.startswith("v(n_") and "syn" in key:
                syn_series.append(sim.vectors[key])
                syn_labels.append(label)

    if syn_series:
        plt.figure()
        for sig, label in zip(syn_series, syn_labels):
            plt.plot(time*1e3, sig, label=label)
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
        if vcomp is not None:
            plt.plot(time*1e3, vcomp, label="Comparator (pre-mix)")
        if vana is not None:
            plt.plot(time*1e3, vana, label="Analog (pre-mix)")
        plt.title("Hybrid output (analog + digital)"); plt.xlabel("Time (ms)"); plt.ylabel("Voltage (V)")
        plt.grid(True); plt.legend(); plt.savefig(f"{png_prefix}_hybrid.png", dpi=160); plt.close()

    if dv_rc is not None or dv_cap is not None:
        plt.figure()
        if dv_rc is not None:
            plt.plot(time*1e3, dv_rc, label="ΔV (Vmem - Vref) [RC]")
        if dv_cap is not None:
            plt.plot(time*1e3, dv_cap, label="Vcap = Vmem - Vsum [TIA]")
        plt.title("Membrane / integrator deltas"); plt.xlabel("Time (ms)"); plt.ylabel("Volts (V)")
        plt.grid(True); plt.legend(); plt.savefig(f"{png_prefix}_deltas.png", dpi=160); plt.close()

    if Q is not None:
        plt.figure()
        plt.plot(time*1e3, Q, label="Charge Q = C·(Vmem - Vsum)")
        plt.title("Membrane charge accumulation"); plt.xlabel("Time (ms)"); plt.ylabel("Coulombs")
        plt.grid(True); plt.legend(); plt.savefig(f"{png_prefix}_charge.png", dpi=160); plt.close()

    if vdef_dbg is not None or vtheta_dbg is not None:
        plt.figure()
        if vdef_dbg is not None:
            plt.plot(time*1e3, vdef_dbg, label="vdef  = Vref - Vmem")
        if vtheta_dbg is not None:
            plt.plot(time*1e3, vtheta_dbg, label="vtheta (rel)")
        if (vdef_dbg is not None) and (vtheta_dbg is not None):
            plt.plot(time*1e3, vdef_dbg - vtheta_dbg, label="margin (vdef - vtheta)")
        if vcomp_raw is not None:
            plt.plot(time*1e3, vcomp_raw, '--', label="comp_raw (internal)")
        if vcomp is not None:
            plt.plot(time*1e3, vcomp, ':', label="comp_out (pin)")
        plt.title("Comparator internals"); plt.xlabel("Time (ms)"); plt.ylabel("Volts (V)")
        plt.grid(True); plt.legend(); plt.savefig(f"{png_prefix}_comp_debug.png", dpi=160); plt.close()

        if vcomp is not None:
            vmin, vmax = float(np.nanmin(vcomp)), float(np.nanmax(vcomp))
            if not (vmin > -0.1 and vmax < 5.1):
                print(f"[warn] v(comp) range looks off for a digital node: {vmin:.3g}..{vmax:.3g}")

    return sim


# ------------------ Power metrics ------------------

POWER_RAIL_SKIPS = {
    "vref": "reference buffer uses an idealized op-amp, so the supply current is non-physical"
}

def compute_power_stats(
    sim: SimData,
    supplies: Supplies,
    skip_rails: Optional[Dict[str, str]] = None,
    extra_rails: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:

    time = sim.time
    if time is None or len(time) < 2:
        return None

    rails = []
    total_samples = None
    duration = float(time[-1] - time[0])
    if duration <= 0:
        duration = 0.0

    skip_map = {k.lower(): v for k, v in (skip_rails or {}).items()}

    def add_samples(existing, new):
        if existing is None:
            return new.copy()
        return existing + new

    rail_entries: List[Dict[str, Any]] = [
        {"label": "VDD", "current_vec": "i(vdd)", "voltage": supplies.vdd},
        {"label": "VREF", "current_vec": "i(vref)", "voltage": supplies.vref},
    ]
    if extra_rails:
        rail_entries.extend(extra_rails)

    for rail in rail_entries:
        vec_name = rail.get("current_vec")
        if not vec_name:
            continue
        current = sim.get(vec_name)
        if current is None:
            continue

        voltage_vec = None
        if rail.get("voltage_vec"):
            voltage_vec = sim.get(rail["voltage_vec"])
            if voltage_vec is None:
                continue
            if rail.get("voltage_ref"):
                ref_vec = sim.get(rail["voltage_ref"])
                if ref_vec is None:
                    continue
                voltage_vec = voltage_vec - ref_vec
        voltage = rail.get("voltage")
        if voltage_vec is not None:
            inst_voltage = voltage_vec
            voltage_report = None
        else:
            inst_voltage = voltage
            voltage_report = voltage
            if inst_voltage is None:
                continue

        inst_power = inst_voltage * (-current)
        energy = float(np.trapz(inst_power, time))
        avg_power = energy / duration if duration > 0 else 0.0
        peak_power = float(np.max(inst_power)) if inst_power.size else 0.0
        label = rail.get("label") or (vec_name[2:-1].upper() if vec_name.startswith("i(") and vec_name.endswith(")") else vec_name.upper())
        label_l = label.lower()
        reason = rail.get("reason")
        skip_reason = skip_map.get(label_l)
        if reason is None:
            reason = skip_reason
        include = rail.get("include")
        if include is None:
            include = skip_reason is None
        else:
            include = include and (skip_reason is None)
        rails.append({
            "rail": label,
            "voltage": voltage_report,
            "energy_J": energy,
            "avg_power_W": avg_power,
            "peak_power_W": peak_power,
            "include": include,
            "reason": reason
        })
        if include:
            total_samples = add_samples(total_samples, inst_power)

    if not rails:
        return None

    counted = [r for r in rails if r["include"]]
    total_energy = sum(r["energy_J"] for r in counted)
    avg_total = total_energy / duration if duration > 0 else 0.0
    peak_total = float(np.max(total_samples)) if total_samples is not None else 0.0

    return {
        "rails": rails,
        "total": {
            "energy_J": total_energy,
            "avg_power_W": avg_total,
            "peak_power_W": peak_total,
            "counted_rails": len(counted)
        },
        "duration_s": duration
    }


def _fmt_watts(value: float) -> str:
    abs_v = abs(value)
    if abs_v < 1e-6:
        return f"{value*1e9:.3g} nW"
    if abs_v < 1e-3:
        return f"{value*1e6:.3g} µW"
    if abs_v < 1:
        return f"{value*1e3:.3g} mW"
    return f"{value:.3g} W"


def _fmt_energy(value: float) -> str:
    abs_v = abs(value)
    if abs_v < 1e-9:
        return f"{value*1e12:.3g} pJ"
    if abs_v < 1e-6:
        return f"{value*1e9:.3g} nJ"
    if abs_v < 1e-3:
        return f"{value*1e6:.3g} µJ"
    if abs_v < 1:
        return f"{value*1e3:.3g} mJ"
    return f"{value:.3g} J"


def print_power_report(stage: str, stats: Dict[str, Any]) -> None:
    total = stats["total"]
    duration = stats.get("duration_s", 0.0)
    counted = total.get("counted_rails", len(stats["rails"]))
    if counted:
        print(
            f"[power] {stage}: avg {_fmt_watts(total['avg_power_W'])}, peak {_fmt_watts(total['peak_power_W'])}, "
            f"energy {_fmt_energy(total['energy_J'])} over {duration:.3g} s"
        )
    else:
        print(f"[power] {stage}: no supply rails included in totals (check configuration).")
    for rail in stats["rails"]:
        prefix = " " * 9
        detail = (
            f"avg {_fmt_watts(rail['avg_power_W'])}, peak {_fmt_watts(rail['peak_power_W'])}, "
            f"energy {_fmt_energy(rail['energy_J'])}"
        )
        voltage = rail.get("voltage")
        if voltage is not None:
            detail += f" @ {voltage:.3g} V"
        else:
            detail += " @ dynamic V"
        if not rail.get("include", True):
            reason = rail.get("reason") or "excluded from totals"
            detail = f"(excluded) {detail} — {reason}"
        print(f"{prefix}{rail['rail']}: {detail}")

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
    power_reports: List[tuple[str, Dict[str, Any]]] = []

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
                sim_data = None
                try:
                    sim_data = load_sim_data(csv_path)
                except Exception as e:
                    print(f"Failed to pre-load CSV for analysis: {e}")

                try:
                    sim_data = plot_results(
                        csv_path,
                        os.path.join(outdir, f"lif_{mode}"),
                        synapses=cfgN.synapses,
                        sim_data=sim_data
                    )
                    print(f"Plots saved to {outdir}/lif_{mode}_*.png")
                except Exception as e:
                    sim_data = None
                    print(f"Plotting failed: {e}")

                if sim_data is not None:
                    try:
                        extra_rails = []
                        for syn in cfgN.synapses:
                            syn_name = syn.name.lower()
                            extra_rails.append({
                                "label": f"SRC_{syn.name.upper()}",
                                "current_vec": f"i(vsrc_{syn_name})",
                                "voltage_vec": f"v(n_{syn_name})",
                                "voltage_ref": "v(vref)",
                                "include": True,
                                "reason": "synaptic source energy (included)"
                            })
                        stats = compute_power_stats(
                            sim_data,
                            cfg.supplies,
                            skip_rails=POWER_RAIL_SKIPS,
                            extra_rails=extra_rails,
                        )
                    except Exception as e:
                        stats = None
                        print(f"Power analysis failed: {e}")
                    if stats:
                        print_power_report(mode.upper(), stats)
                        power_reports.append((mode, stats))
                    else:
                        print(f"[power] {mode}: supply current vectors missing; skipping stats.")
            else:
                print(f"ngspice returned {rc}. Check log: {log_path}")

    if power_reports:
        total_energy = sum(stats["total"]["energy_J"] for _, stats in power_reports)
        total_duration = sum(stats.get("duration_s", 0.0) for _, stats in power_reports)
        avg_power = total_energy / total_duration if total_duration > 0 else 0.0
        peak_power = max((stats["total"]["peak_power_W"] for _, stats in power_reports), default=0.0)
        print(
            f"[power] cumulative ({len(power_reports)} stages): avg {_fmt_watts(avg_power)}, "
            f"peak {_fmt_watts(peak_power)}, energy {_fmt_energy(total_energy)} over {total_duration:.3g} s"
        )

if __name__ == "__main__":
    main()

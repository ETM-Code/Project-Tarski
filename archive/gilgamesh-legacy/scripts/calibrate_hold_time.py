#!/usr/bin/env python3
"""
Calibrate hold_time_multiplier by running SPICE simulation.

This script measures the actual hold time from SPICE and calculates the
multiplier needed for the Rust model: hold_time = multiplier × τ_mem

Usage:
    python scripts/calibrate_hold_time.py path/to/neuron.json

    # Or with explicit paths:
    python scripts/calibrate_hold_time.py --neuron path/to/neuron.json --spice-dir ../SPICE/neuronSim

The script will:
1. Generate a minimal SPICE netlist with a single strong spike
2. Run ngspice to simulate
3. Analyze v_comp to find hold time (time at peak before decay)
4. Output the calibrated multiplier

Output can be used to set pulse_stretch.hold_time_multiplier in the neuron config.
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Try to import numpy for analysis
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


@dataclass
class NeuronParams:
    """Extracted neuron parameters needed for calibration."""
    name: str
    vdd: float
    vref: float
    c_mem_f: float
    r_leak_ohm: float
    threshold_v: float
    r_pw_ohm: float
    c_pw_f: float
    pulse_stretch_enabled: bool

    @property
    def tau_mem(self) -> float:
        """Membrane time constant in seconds."""
        return self.c_mem_f * self.r_leak_ohm

    @property
    def tau_pulse(self) -> float:
        """Pulse stretch time constant in seconds."""
        return self.r_pw_ohm * self.c_pw_f

    @classmethod
    def from_json(cls, path: str) -> "NeuronParams":
        with open(path) as f:
            d = json.load(f)

        supplies = d.get("supplies", {})
        membrane = d.get("membrane", {})
        threshold = d.get("threshold", {})
        pulse_stretch = d.get("pulse_stretch", {})

        return cls(
            name=d.get("name", "neuron"),
            vdd=supplies.get("vdd", 5.0),
            vref=supplies.get("vref", 2.5),
            c_mem_f=membrane.get("C_mem_F", 10e-9),
            r_leak_ohm=membrane.get("R_leak_ohm", 120e3),
            threshold_v=threshold.get("over_vref_V", 0.8),
            r_pw_ohm=pulse_stretch.get("R_pw_ohm", 20e3),
            c_pw_f=pulse_stretch.get("C_pw_F", 100e-9),
            pulse_stretch_enabled=pulse_stretch.get("enable", True),
        )


def find_spice_generator(spice_dir: Optional[str] = None) -> Path:
    """Find the SPICE neuron generator script."""
    candidates = [
        spice_dir,
        "../SPICE/neuronSim",
        "../../SPICE/neuronSim",
        os.path.expanduser("~/Tarskii/SPICE/neuronSim"),
    ]

    for candidate in candidates:
        if candidate is None:
            continue
        p = Path(candidate)
        gen_script = p / "lif_neuron_generator_pulse_stretch.py"
        if gen_script.exists():
            return p

    raise FileNotFoundError(
        "Could not find SPICE neuron generator. "
        "Use --spice-dir to specify the path to SPICE/neuronSim"
    )


def generate_calibration_netlist(params: NeuronParams, spice_dir: Path, output_csv: str) -> str:
    """Generate a minimal SPICE netlist for calibration."""

    # We need a single strong spike that will definitely trigger the neuron
    # Use a spike amplitude that's well above threshold
    spike_amp = params.threshold_v + 0.3  # 0.3V above threshold
    spike_width_ms = 2.0  # Long enough to charge membrane
    spike_time_ms = 5.0   # Start after settling
    sim_stop_ms = 50.0    # Enough time to see full decay

    # Synapse weight - low enough to let spike amplitude work
    syn_weight = 20e3  # 20kΩ

    netlist = f"""* Calibration netlist for hold_time_multiplier
* Neuron: {params.name}
* tau_mem = {params.tau_mem*1e3:.3f} ms
* tau_pulse = {params.tau_pulse*1e3:.3f} ms

VDD vdd 0 {params.vdd}
VREF vref 0 {params.vref}

.options method=gear maxord=2 reltol=2e-3 trtol=7

* Include the pulse-stretch neuron subcircuit
.subckt calibration_neuron mem vref vdd comp_pulse analog_out sum

* Vref buffer
Ebuf vref_buf 0 vref 0 1e5
Rbuf vref_buf vref 1k
Cbuf vref 0 80p
Rvr vref vref_buf 1m

* TIA op-amp
Eint mem 0 sum vref 2e5
Rout_int mem 0 20
Cint mem 0 5p
Cmem mem sum {params.c_mem_f}
Rleak mem sum {params.r_leak_ohm}
Iint_bias vdd 0 2e-05

* Threshold divider (simplified)
Rvh vdd vth_node 3.4e+06
Rvl vth_node vref 1.58e+06
Rf comp_out vth_node 1.07e+08
Cadapt vth_node vref 2.2e-08
.model DADAPT D(Is=1e-6 N=1.05 Rs=2 Cjo=1p Eg=0.69)
Rinj comp_out ninj 2.2Meg
Dinj ninj vth_node DADAPT

* Comparator
.param VLO=0.0 VHI={params.vdd}
.param VSW=0.01
Bdef vdef 0 V = V(vref) - V(mem)
Bcomp comp_raw 0 V = VLO + (VHI - VLO)*(0.5*(1 + tanh( ( V(vdef) - {params.threshold_v} ) / VSW )))
Rcout comp_raw comp_out 10.0
Ccout comp_out 0 4e-09
Icomp_bias vdd 0 2e-06

* Pulse stretching circuit
.model DPW D(Is=1e-12 N=1.05 Rs=10 Cjo=1p)
Dpw comp_out comp_pulse DPW
Rpw comp_pulse 0 {params.r_pw_ohm}
Cpw comp_pulse 0 {params.c_pw_f}

* Reset
Rreset mem reset_node 200.0
Coff_reset reset_node vref 7e-12
Sreset reset_node vref comp_out 0 SWMUX
.model SWMUX SW(Ron=10.0 Roff=1000000000.0 Vt=2.5 Vh=0.1)

* Analog output (simplified)
Bdiff ana_in 0 V = V(mem) - V(vref)
Eana_opamp ana_opamp_out 0 0 ana_inv_in 1e5
Rana_int ana_opamp_out analog_out 50
Cana_comp analog_out 0 2p
R1_ana ana_in ana_inv_in 1e+04
R2_ana analog_out ana_inv_in 2e+04
Rana_load analog_out 0 10000.0
Iana_bias vdd 0 1e-05

.ends

XNEU mem vref vdd comp ana sum calibration_neuron

* Single calibration spike
Vsrc_cal n_cal vref PWL(0 0 {spike_time_ms/1000} 0 {spike_time_ms/1000 + 1e-6} {spike_amp} {(spike_time_ms + spike_width_ms)/1000 - 1e-6} {spike_amp} {(spike_time_ms + spike_width_ms)/1000} 0)
R_cal n_cal sum {syn_weight}

.control
  set filetype=ascii
  set wr_singlescale
  set wr_vecnames
  tran 1e-06 {sim_stop_ms/1000} 0 1e-06
  linearize
  wrdata {output_csv} time v(comp) v(mem) v(vref)
  quit
.endc

.end
"""
    return netlist


def run_ngspice(netlist_path: str, log_path: str) -> int:
    """Run ngspice on a netlist."""
    exe = shutil.which("ngspice")
    if not exe:
        raise RuntimeError("ngspice not found on PATH")

    cmd = [exe, "-b", "-o", log_path, netlist_path]
    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd)


def load_csv_data(csv_path: str) -> tuple:
    """Load CSV data from ngspice output."""
    if not HAS_NUMPY:
        raise ImportError("numpy is required for CSV analysis")

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
                    # This is a header line
                    if parts[0].lower() == "index":
                        parts = parts[1:]
                    names = [p.lower() for p in parts]
                    continue

            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                pass

    arr = np.array(rows, dtype=float)
    return names, arr


def analyze_hold_time(csv_path: str, tau_mem: float, vdd: float) -> dict:
    """
    Analyze the SPICE output to find hold time.

    Hold time = time from when v_comp reaches peak until it starts decaying.
    """
    if not HAS_NUMPY:
        raise ImportError("numpy is required for analysis")

    names, arr = load_csv_data(csv_path)

    # Find column indices
    time_idx = 0
    comp_idx = None

    if names:
        for i, name in enumerate(names):
            if "time" in name:
                time_idx = i
            elif "v(comp)" in name or "comp" in name:
                comp_idx = i

    if comp_idx is None:
        # Try to guess - usually time, v(comp), v(mem), v(vref)
        comp_idx = 1

    time = arr[:, time_idx]
    v_comp = arr[:, comp_idx]

    # Find peak voltage (should be ~vdd - diode_drop)
    peak_v = np.max(v_comp)

    # Threshold for "high" - 90% of peak
    high_threshold = 0.9 * peak_v

    # Find when v_comp first goes high
    high_mask = v_comp > high_threshold
    if not np.any(high_mask):
        raise ValueError("v_comp never went high - spike didn't fire")

    first_high_idx = np.argmax(high_mask)
    t_spike = time[first_high_idx]

    # Find when v_comp starts decaying (drops below 90% of peak after being high)
    # Look for the point where it's been high and then drops
    decay_threshold = 0.85 * peak_v  # A bit lower to detect decay start

    # Find indices where we're still high
    high_indices = np.where(high_mask)[0]

    # Find the last continuous high region
    last_high_idx = first_high_idx
    for i in range(first_high_idx, len(v_comp)):
        if v_comp[i] > decay_threshold:
            last_high_idx = i
        else:
            break

    t_decay_start = time[last_high_idx]

    # Hold time is from spike to decay start
    hold_time = t_decay_start - t_spike

    # Calculate multiplier
    multiplier = hold_time / tau_mem if tau_mem > 0 else 0

    return {
        "t_spike_s": float(t_spike),
        "t_decay_start_s": float(t_decay_start),
        "hold_time_s": float(hold_time),
        "hold_time_ms": float(hold_time * 1000),
        "tau_mem_s": float(tau_mem),
        "tau_mem_ms": float(tau_mem * 1000),
        "multiplier": float(multiplier),
        "peak_v": float(peak_v),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate hold_time_multiplier from SPICE simulation"
    )
    parser.add_argument(
        "neuron_json",
        nargs="?",
        default=None,
        help="Path to neuron JSON config"
    )
    parser.add_argument(
        "--neuron",
        default=None,
        help="Path to neuron JSON config (alternative to positional arg)"
    )
    parser.add_argument(
        "--spice-dir",
        default=None,
        help="Path to SPICE/neuronSim directory"
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Write calibration results to JSON file"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files for debugging"
    )

    args = parser.parse_args()

    # Determine neuron config path
    neuron_path = args.neuron_json or args.neuron
    if not neuron_path:
        # Try default location
        default_paths = [
            "../SPICE/neuronSim/defaults/neuron_pulse_stretch.json",
            "../../SPICE/neuronSim/defaults/neuron_pulse_stretch.json",
        ]
        for p in default_paths:
            if os.path.exists(p):
                neuron_path = p
                break

    if not neuron_path or not os.path.exists(neuron_path):
        print("Error: neuron JSON config not found")
        print("Usage: python calibrate_hold_time.py path/to/neuron.json")
        sys.exit(1)

    print(f"Loading neuron config: {neuron_path}")
    params = NeuronParams.from_json(neuron_path)

    print(f"\nNeuron parameters:")
    print(f"  tau_mem  = {params.tau_mem*1e3:.3f} ms  (R_leak={params.r_leak_ohm/1e3:.1f}kΩ, C_mem={params.c_mem_f*1e9:.1f}nF)")
    print(f"  tau_pulse = {params.tau_pulse*1e3:.3f} ms  (R_pw={params.r_pw_ohm/1e3:.1f}kΩ, C_pw={params.c_pw_f*1e9:.1f}nF)")
    print(f"  threshold = {params.threshold_v:.3f} V over Vref")

    if not params.pulse_stretch_enabled:
        print("\nWarning: pulse_stretch is disabled in config, but calibrating anyway")

    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        if args.keep_temp:
            tmpdir = os.path.join(os.getcwd(), "calibration_temp")
            os.makedirs(tmpdir, exist_ok=True)
            print(f"\nKeeping temp files in: {tmpdir}")

        csv_path = os.path.join(tmpdir, "calibration.csv")
        cir_path = os.path.join(tmpdir, "calibration.cir")
        log_path = os.path.join(tmpdir, "calibration.log")

        # Generate netlist
        print("\nGenerating calibration netlist...")
        netlist = generate_calibration_netlist(params, Path("."), csv_path)

        with open(cir_path, "w") as f:
            f.write(netlist)

        # Run ngspice
        print("Running SPICE simulation...")
        try:
            rc = run_ngspice(cir_path, log_path)
        except RuntimeError as e:
            print(f"Error: {e}")
            sys.exit(1)

        if rc != 0:
            print(f"ngspice failed with code {rc}")
            print(f"Check log: {log_path}")
            sys.exit(1)

        if not os.path.exists(csv_path):
            print(f"Error: CSV output not created")
            print(f"Check log: {log_path}")
            sys.exit(1)

        # Analyze results
        print("\nAnalyzing simulation output...")
        try:
            results = analyze_hold_time(csv_path, params.tau_mem, params.vdd)
        except Exception as e:
            print(f"Analysis failed: {e}")
            sys.exit(1)

        # Print results
        print("\n" + "="*50)
        print("CALIBRATION RESULTS")
        print("="*50)
        print(f"  Spike time:      {results['t_spike_s']*1e3:.3f} ms")
        print(f"  Decay start:     {results['t_decay_start_s']*1e3:.3f} ms")
        print(f"  Hold time:       {results['hold_time_ms']:.3f} ms")
        print(f"  tau_mem:         {results['tau_mem_ms']:.3f} ms")
        print(f"  Peak voltage:    {results['peak_v']:.3f} V")
        print()
        print(f"  hold_time_multiplier = {results['multiplier']:.2f}")
        print("="*50)

        # Suggest config update
        print(f"\nTo use this calibration, add to your neuron JSON config:")
        print(f'  "pulse_stretch": {{')
        print(f'    "enable": true,')
        print(f'    "R_pw_ohm": {params.r_pw_ohm},')
        print(f'    "C_pw_F": {params.c_pw_f},')
        print(f'    "hold_time_multiplier": {results["multiplier"]:.2f}')
        print(f'  }}')

        # Write JSON output if requested
        if args.output_json:
            output_data = {
                "neuron_config": neuron_path,
                "params": {
                    "tau_mem_s": params.tau_mem,
                    "tau_pulse_s": params.tau_pulse,
                    "threshold_v": params.threshold_v,
                },
                "results": results,
            }
            with open(args.output_json, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults written to: {args.output_json}")


if __name__ == "__main__":
    main()

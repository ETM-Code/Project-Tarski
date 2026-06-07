#!/usr/bin/env python3
"""
Generate a leaky integrate-and-fire (LIF) neuron *subcircuit* netlist for ngspice.

FAST  : realistic comparator (hard 0/VDD with Schmitt), ideal TIA
DETAIL: adds Vref buffer, finite op-amp gain/pole, physical hysteresis network,
        adaptive threshold injection, and comparator propagation shaping.

This file builds the NEURON subckts. The network (spikes/synapses) is in lif_network_generator.py.
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
    over_vref_V: float       # center above Vref (e.g., 0.8V)
    hysteresis_V: float      # total span (e.g., 0.05V -> ±25 mV)
    C_adapt_F: float = 100e-9          # was hard-coded 100n
    divider_scale: float = 1.0         # scale Rvh/Rvl/Rf together (e.g., 0.5 → 2× faster)
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
    offset_V: float          # input offset (adds to threshold)
    prop_delay_s: float      # used to set output RC shaping
    vlow_V: float
    vhigh_V: float
    gain_fast: float         # (unused now; we’re hard-digital)

@dataclass
class SimCfg:
    tstop_s: float
    tstep_s: float

@dataclass
class AnalogOutCfg:
    """
    Configuration for the analog output buffer/amplifier stage.
    
    Inverting topology:
        Vin = (Vmem - Vref) or (Vref - Vmem) depending on 'sign'
        Vout = -gain * Vin (inverted)
        Circuit: op-amp in inverting configuration with gain = -R2/R1
    
    Non-inverting topology (legacy):
        Vout = gain * Vin (same polarity)
        Circuit: op-amp in non-inverting configuration with gain = 1 + R2/R1
    """
    gain: float = 1.0        # magnitude of gain (|Vout/Vin|)
    sign: int = 1           # +1 => input is (Vmem - Vref),  -1 => input is (Vref - Vmem)
    clamp_to_rails: bool = False  # add diode clamps to [0, VDD]
    analog_out_R_ohm: float = 150.0  # output load resistance
    inverting: bool = True   # True => inverting amplifier, False => non-inverting
    R1_ohm: float = 10e3     # input resistor (inverting) or ground resistor (non-inverting)
    R2_ohm: float = 10e3     # feedback resistor (gain = R2/R1 for inverting, 1+R2/R1 for non-inv)
    bias_current_A: float = 400e-6  # nominal quiescent supply draw of the analog buffer


@dataclass
class BiasCurrents:
    tia_A: float = 200e-6
    comparator_A: float = 150e-6

@dataclass
class PulseStretchCfg:
    """
    RC pulse stretch circuit configuration.

    When comparator fires HIGH, diode Dpw forward-biases and charges Cpw to V_peak.
    When comparator goes LOW, diode reverse-biases and Cpw decays through R_eff.

    R_eff = R_pw || R_load (parallel combination for discharge)
    tau_eff = R_eff * C_pw
    """
    enable: bool = False
    R_pw_ohm: float = 20000.0      # Pulldown resistor (20kΩ typical)
    C_pw_F: float = 100e-9         # Pulse capacitor (100nF typical)
    R_load_ohm: float = 10000.0    # Load resistance (affects discharge rate)
@dataclass
class NeuronConfig:
    name: str
    supplies: Supplies
    membrane: Membrane
    threshold: Threshold
    reset: ResetPath
    comparator: ComparatorCfg
    simulation: SimCfg
    analog_out: AnalogOutCfg = field(default_factory=AnalogOutCfg)
    bias_currents: BiasCurrents = field(default_factory=BiasCurrents)
    pulse_stretch: PulseStretchCfg = field(default_factory=PulseStretchCfg)

    @staticmethod
    def load(json_path: str) -> "NeuronConfig":
        with open(json_path, "r") as f:
            d = json.load(f)
        # Parse pulse_stretch config, handling both snake_case and PascalCase keys
        ps_dict = d.get("pulse_stretch", {})
        ps_kwargs = {
            "enable": ps_dict.get("enable", False),
            "R_pw_ohm": ps_dict.get("R_pw_ohm", ps_dict.get("r_pw_ohm", 20000.0)),
            "C_pw_F": ps_dict.get("C_pw_F", ps_dict.get("c_pw_f", 100e-9)),
            "R_load_ohm": ps_dict.get("R_load_ohm", ps_dict.get("r_load_ohm", 10000.0)),
        }
        return NeuronConfig(
            name=d.get("name","lif_neuron"),
            supplies=Supplies(**d["supplies"]),
            membrane=Membrane(**d["membrane"]),
            threshold=Threshold(**d["threshold"]),
            reset=ResetPath(**d["reset"]),
            comparator=ComparatorCfg(**d["comparator"]),
            simulation=SimCfg(**d["simulation"]),
            analog_out=AnalogOutCfg(**d.get("analog_out", {})),
            bias_currents=BiasCurrents(**d.get("bias_currents", {})),
            pulse_stretch=PulseStretchCfg(**ps_kwargs),
        )

# -------------------- Helpers --------------------

def lif_sanity_checks(cfg: NeuronConfig) -> list[str]:
    issues = []
    if not (1.8 <= cfg.supplies.vdd <= 5.5):
        issues.append(f"Comparator supply vdd={cfg.supplies.vdd} V outside 1.8–5.5 V.")
    vth_hi = cfg.supplies.vref + cfg.threshold.over_vref_V + cfg.threshold.hysteresis_V/2
    vth_lo = cfg.supplies.vref + cfg.threshold.over_vref_V - cfg.threshold.hysteresis_V/2
    if vth_hi > cfg.supplies.vdd or vth_lo < 0:
        issues.append(f"Threshold window [{vth_lo:.3f},{vth_hi:.3f}] not inside [0,{cfg.supplies.vdd}].")
    tau = cfg.membrane.C_mem_F * cfg.membrane.R_leak_ohm
    if tau < 1e-4:
        issues.append(f"Membrane tau={tau*1e3:.2f} ms is very small.")
    return issues

def _header(title: str) -> str:
    return f"* ---------- {title} ----------\n"

def _analog_diff_input(cfg: NeuronConfig) -> str:
    """Differential input line for the analog stage: V(mem)-V(vref) for sign>=0,
    else V(vref)-V(mem)."""
    if cfg.analog_out.sign >= 0:
        return "Bdiff ana_in 0 V = V(mem) - V(vref)\n"
    return "Bdiff ana_in 0 V = V(vref) - V(mem)\n"

# -------------------- FAST subcircuit --------------------

def generate_fast_neuron(cfg: NeuronConfig) -> str:
    """
    Make FAST an alias of the detailed subcircuit (same internals, different name).
    This keeps CLI/back-compat but uses the numerically stable model.
    """
    det = generate_detailed_neuron(cfg)
    # rename only the subckt header/footer so both can coexist in the netlist
    det = det.replace(f".subckt {cfg.name}_detailed",
                      f".subckt {cfg.name}_fast", 1)
    return det

# -------------------- DETAILED subcircuit --------------------

def generate_detailed_neuron(cfg: NeuronConfig) -> str:
    s = []
    s.append(_header(f"{cfg.name} DETAILED neuron subcircuit"))
    # pins: mem vref vdd comp_out analog_out sum
    s.append(f".subckt {cfg.name}_detailed mem vref vdd comp_out analog_out sum\n")

    # Vref buffer (opa-like follower to give vref some source stiffness)
    s.append("* Vref buffer (OPA604-like)\n")
    s.append("Ebuf vref_buf 0 vref 0 1e5\n")
    s.append("Rbuf vref_buf vref 1k\n")
    s.append("Cbuf vref 0 80p\n")
    s.append("Rvr  vref vref_buf 1m\n")

    # TIA with finite DC gain and small pole to avoid algebraic loops
    s.append("* TIA op-amp with finite gain and compensation\n")
    s.append("Eint mem 0 sum vref 2e5\n")
    s.append("Rout_int mem 0 20\n")
    s.append("Cint mem 0 5p\n")
    s.append(f"Cmem  mem sum {cfg.membrane.C_mem_F}\n")
    s.append(f"Rleak mem sum {cfg.membrane.R_leak_ohm}\n")
    s.append(f"Iint_bias vdd 0 {cfg.bias_currents.tia_A}\n")

    # Physical hysteresis divider around a threshold node in absolute volts
    vhi = cfg.comparator.vhigh_V
    vlo = cfg.comparator.vlow_V
    dv_out = max(vhi - vlo, 1e-6)
    beta = cfg.threshold.hysteresis_V / dv_out

    s.append("* Divider to place threshold near Vref + over_vref_V\n")
    scale = cfg.threshold.divider_scale
    Rvh_val = 681e3 * max(scale, 1e-3)
    Rvl_val = 316e3 * max(scale, 1e-3)
    s.append(f"Rvh  vdd     vth_node {Rvh_val:.3g}\n")
    s.append(f"Rvl  vth_node vref    {Rvl_val:.3g}\n")

    # Hysteresis feedback resistor sized from target beta using the ACTUAL divider
    g_div = (1.0/Rvh_val) + (1.0/Rvl_val)
    Rf = (1.0 / (beta * g_div / max(1.0 - beta, 1e-6)))
    # Remove the previous upper clamp at 50 MΩ to match the Rust model.
    # Keep a small lower bound for numerical stability only.
    Rf_ohm = max(Rf, 100e3)
    s.append(f"Rf   comp_out vth_node {Rf_ohm:.3g}\n")

    # Adaptive threshold injection (one-way from comp_out)
    s.append(f"Cadapt vth_node vref {cfg.threshold.C_adapt_F}\n")
    s.append(".model DADAPT D(Is=1e-6 N=1.05 Rs=2 Cjo=1p Eg=0.69)\n")
    s.append("Rinj  comp_out ninj 2.2Meg\n")
    s.append("Dinj  ninj vth_node DADAPT\n")

    # Comparator in deflection space (hard digital with RC shaping)
    s.append(f".param VLO={vlo} VHI={vhi}\n")
    # Smooth comparator transfer (avoids hard discontinuity that slows the solver)
    s.append(".param VSW=0.01\n")
    # Comparator RC shaping: small RC derived from propagation delay (previous behaviour)
    rc   = max(cfg.comparator.prop_delay_s/10.0, 1e-9)
    rout = max(cfg.comparator.prop_delay_s/rc, 10.0)

    # Membrane deflection: positive when membrane rises above Vref
    s.append("Bdef vdef 0 V = V(mem) - V(vref)\n")
    s.append("Btheta_rel vtheta_rel 0 V = V(vth_node) - V(vref)\n")
    # Soft comparator: fires HIGH when vdef > threshold (membrane above Vref + over_vref_V)
    s.append("* Soft comparator: fires when V(mem) > V(vref) + over_vref_V (align with Rust model)\n")
    s.append(
        f"Bcomp comp_raw 0 V = VLO + (VHI - VLO)*(0.5*(1 + tanh( ( V(vdef) - ( {cfg.threshold.over_vref_V} + {cfg.comparator.offset_V} ) ) / VSW )))\n"
    )
    # Pulse stretch circuit: RC decay after comparator
    if cfg.pulse_stretch.enable:
        # Route comparator through pulse stretch circuit
        # comp_raw -> Rcout -> comp_shaped -> Dpw -> comp_out
        # Cpw and Rpw provide the RC decay when comparator goes low
        s.append(f"Rcout comp_raw comp_shaped {rout}\n")
        s.append(f"Ccout comp_shaped 0 {rc}\n")

        # Pulse stretch diode: charges Cpw when comp_shaped goes HIGH
        s.append("* Pulse stretch circuit (RC decay)\n")
        s.append(".model DPW D(Is=1e-12 N=1.05 Rs=10)\n")
        s.append("Dpw comp_shaped comp_out DPW\n")

        # Pulse stretch capacitor and discharge resistor
        s.append(f"Cpw comp_out 0 {cfg.pulse_stretch.C_pw_F}\n")
        s.append(f"Rpw comp_out 0 {cfg.pulse_stretch.R_pw_ohm}\n")

        # Note: R_load is typically external (in the network), but we can add
        # a parallel path if specified for internal discharge modeling
    else:
        # No pulse stretch - direct comparator output
        s.append(f"Rcout comp_raw comp_out {rout}\n")
        s.append(f"Ccout comp_out 0 {rc}\n")

    s.append(f"Icomp_bias vdd 0 {cfg.bias_currents.comparator_A}\n")

    # Reset path - use comp_shaped (raw comparator) for reset control, not stretched output
    if cfg.reset.enable:
        s.append(f"Rreset mem reset_node {cfg.reset.series_R_ohm}\n")
        s.append(f"Coff_reset reset_node vref {cfg.reset.mux_Coff_F}\n")
        # Use comp_shaped for reset control when pulse stretch is enabled
        reset_control = "comp_shaped" if cfg.pulse_stretch.enable else "comp_out"
        s.append(f"Sreset reset_node vref {reset_control} 0 SWMUX\n")
        s.append(f".model SWMUX SW(Ron={cfg.reset.mux_Ron_ohm} Roff={cfg.reset.mux_Roff_ohm} Vt={cfg.reset.switch_Vt} Vh={cfg.reset.switch_Vh})\n")

    # Analog output (scaled (Vmem - Vref) for plotting/hybrid)
    s.append("* Analog output stage\n")
    
    if cfg.analog_out.inverting:
        # Inverting op-amp amplifier topology
        # Gain = -R2/R1
        desired_gain = abs(cfg.analog_out.gain)
        R1 = cfg.analog_out.R1_ohm
        R2 = R1 * desired_gain  # gain magnitude = R2/R1

        # Differential input stage: create Vin = (Vmem - Vref) or (Vref - Vmem)
        s.append(_analog_diff_input(cfg))

        # Inverting amplifier with finite gain op-amp model
        # Op-amp: high gain, finite bandwidth
        s.append("* Inverting amplifier op-amp (OPA-like)\n")
        s.append("Eana_opamp ana_opamp_out 0 0 ana_inv_in 1e5\n")  # V+ = 0, V- = ana_inv_in
        s.append("Rana_int ana_opamp_out analog_out 50\n")  # output impedance
        s.append("Cana_comp analog_out 0 2p\n")  # compensation cap
        
        # Inverting input network
        s.append(f"R1_ana ana_in ana_inv_in {R1:.3g}\n")  # input resistor
        s.append(f"R2_ana analog_out ana_inv_in {R2:.3g}\n")  # feedback resistor
        
        # Output load
        s.append(f"Rana_load analog_out 0 {cfg.analog_out.analog_out_R_ohm}\n")
        
    else:
        # Non-inverting op-amp amplifier topology
        # Gain = 1 + R2/R1
        desired_gain = abs(cfg.analog_out.gain)
        R1 = cfg.analog_out.R1_ohm
        R2 = R1 * max(desired_gain - 1.0, 0.0)  # gain = 1 + R2/R1

        # Differential input stage: create Vin = (Vmem - Vref) or (Vref - Vmem)
        s.append(_analog_diff_input(cfg))

        # Non-inverting amplifier with finite gain op-amp model
        s.append("* Non-inverting amplifier op-amp (OPA-like)\n")
        s.append("Eana_opamp ana_opamp_out 0 ana_in ana_fb 1e5\n")
        s.append("Rana_int ana_opamp_out analog_out 50\n")
        s.append("Cana_comp analog_out 0 2p\n")
        
        # Feedback network: R2 from output to feedback node, R1 from feedback to ground
        s.append(f"R2_ana analog_out ana_fb {R2:.3g}\n")
        s.append(f"R1_ana ana_fb 0 {R1:.3g}\n")
        
        # Output load
        s.append(f"Rana_load analog_out 0 {cfg.analog_out.analog_out_R_ohm}\n")

    if cfg.analog_out.clamp_to_rails:
        # simple diode clamps to 0..VDD so the output can't swing to infinity
        s.append(".model DCLAMP D(Is=1e-15 N=1.8 Rs=1)\n")
        s.append("Dcl_lo  0   analog_out DCLAMP\n")
        s.append("Dcl_hi  analog_out vdd DCLAMP\n")

    s.append(f"Iana_bias vdd 0 {cfg.analog_out.bias_current_A}\n")

    s.append(".ends\n")
    return "".join(s)
# -------------------- CLI --------------------

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

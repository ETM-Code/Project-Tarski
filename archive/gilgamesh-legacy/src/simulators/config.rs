use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Supplies {
    pub vdd: f64,
    pub vref: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Membrane {
    #[serde(alias = "C_mem_F")]
    pub c_mem_f: f64,
    #[serde(alias = "R_leak_ohm")]
    pub r_leak_ohm: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Threshold {
    #[serde(alias = "over_vref_V")]
    pub over_vref_v: f64,
    #[serde(alias = "hysteresis_V")]
    pub hysteresis_v: f64,
    #[serde(default = "default_c_adapt")]
    #[serde(alias = "C_adapt_F")]
    pub c_adapt_f: f64,
    #[serde(default = "default_divider_scale")]
    pub divider_scale: f64,
}

fn default_c_adapt() -> f64 {
    100e-9
}

fn default_divider_scale() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResetPath {
    pub enable: bool,
    #[serde(alias = "series_R_ohm")]
    pub series_r_ohm: f64,
    #[serde(alias = "mux_Ron_ohm")]
    pub mux_ron_ohm: f64,
    #[serde(alias = "mux_Roff_ohm")]
    pub mux_roff_ohm: f64,
    #[serde(alias = "mux_Coff_F")]
    pub mux_coff_f: f64,
    #[serde(alias = "switch_Vt")]
    pub switch_vt: f64,
    #[serde(alias = "switch_Vh")]
    pub switch_vh: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparatorCfg {
    #[serde(alias = "offset_V")]
    pub offset_v: f64,
    pub prop_delay_s: f64,
    #[serde(alias = "vlow_V")]
    pub vlow_v: f64,
    #[serde(alias = "vhigh_V")]
    pub vhigh_v: f64,
}

/// Pulse stretching circuit configuration (RC decay).
/// Models the hardware RC pulse stretcher that extends spike duration.
///
/// Physics: When comparator fires, diode charges C_pw to V_peak. When comparator
/// goes low, C_pw discharges exponentially through parallel resistance R_eff.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PulseStretchCfg {
    #[serde(default)]
    pub enable: bool,
    /// Pulse stretch resistor in ohms (R_pw, typically 20kΩ)
    #[serde(default, alias = "R_pw_ohm")]
    pub r_pw_ohm: f64,
    /// Pulse stretch capacitor in farads (C_pw, typically 100nF)
    #[serde(default, alias = "C_pw_F")]
    pub c_pw_f: f64,
    /// Optional load resistance in ohms (R_load, typically 100kΩ synapse input).
    /// If provided, effective resistance is R_eff = R_pw || R_load (parallel).
    /// If None, uses R_pw alone (conservative, slower decay).
    #[serde(default, alias = "R_load_ohm")]
    pub r_load_ohm: Option<f64>,
}

impl PulseStretchCfg {
    /// Compute effective time constant accounting for parallel discharge paths.
    ///
    /// The pulse stretch capacitor discharges through:
    /// - R_pw (pulldown resistor to ground)
    /// - R_load (downstream synapse input resistance)
    ///
    /// Effective resistance: R_eff = R_pw || R_load
    /// Time constant: τ_eff = R_eff × C_pw
    pub fn tau_eff_s(&self) -> f64 {
        if !self.enable || self.c_pw_f == 0.0 {
            return 0.0;
        }

        let r_eff = if let Some(r_load) = self.r_load_ohm {
            // Parallel combination: 1/R_eff = 1/R_pw + 1/R_load
            if r_load > 0.0 && self.r_pw_ohm > 0.0 {
                (self.r_pw_ohm * r_load) / (self.r_pw_ohm + r_load)
            } else {
                self.r_pw_ohm
            }
        } else {
            // Conservative: use R_pw alone (slower decay)
            self.r_pw_ohm
        };

        r_eff * self.c_pw_f
    }

    /// Compute the nominal RC time constant τ = R_pw × C_pw (no load)
    pub fn tau_s(&self) -> f64 {
        self.r_pw_ohm * self.c_pw_f
    }

    /// Convert to the physics module config with pure exponential decay.
    /// No hold time - decay starts when comparator output goes low.
    pub fn to_physics_config(&self) -> crate::math_functions::neuron_physics::PulseStretchConfig {
        if self.enable {
            crate::math_functions::neuron_physics::PulseStretchConfig::from_tau(
                self.tau_eff_s()
            )
        } else {
            crate::math_functions::neuron_physics::PulseStretchConfig::default()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalogOutCfg {
    pub gain: f64,
    pub sign: i32,
    #[serde(default)]
    pub clamp_to_rails: bool,
    #[serde(default = "default_analog_out_r")]
    #[serde(alias = "analog_out_R_ohm")]
    pub analog_out_r_ohm: f64,
    #[serde(default = "default_inverting")]
    pub inverting: bool,
    #[serde(default = "default_r1")]
    #[serde(alias = "R1_ohm")]
    pub r1_ohm: f64,
    #[serde(default = "default_r2")]
    #[serde(alias = "R2_ohm")]
    pub r2_ohm: f64,
}

fn default_analog_out_r() -> f64 {
    150.0
}

fn default_inverting() -> bool {
    true
}

fn default_r1() -> f64 {
    10_000.0
}

fn default_r2() -> f64 {
    10_000.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimCfg {
    pub tstop_s: f64,
    pub tstep_s: f64,
    #[serde(default)]
    pub min_dt_s: Option<f64>,
    #[serde(default)]
    pub max_dt_s: Option<f64>,
    #[serde(default = "default_max_step_factor")]
    pub max_step_factor: f64,
}

fn default_max_step_factor() -> f64 {
    20.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spike {
    pub t_ms: f64,
    pub width_ms: f64,
    #[serde(alias = "amp_V")]
    pub amp_v: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Synapse {
    pub name: String,
    #[serde(default = "default_syn_type")]
    pub r#type: SynapseType,
    pub weight_ohm: f64,
    #[serde(default)]
    pub spikes: Vec<Spike>,
}

fn default_syn_type() -> SynapseType {
    SynapseType::Excitatory
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SynapseType {
    Excitatory,
    Inhibitory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronConfig {
    pub name: String,
    pub supplies: Supplies,
    pub membrane: Membrane,
    pub threshold: Threshold,
    pub reset: ResetPath,
    pub comparator: ComparatorCfg,
    pub analog_out: AnalogOutCfg,
    pub simulation: SimCfg,
    /// Pulse stretching circuit configuration (optional)
    #[serde(default)]
    pub pulse_stretch: PulseStretchCfg,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub title: String,
    pub neuron_json: String,
    #[serde(default)]
    pub synapses: Vec<Synapse>,
}

impl NeuronConfig {
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }
}

impl NetworkConfig {
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }
}

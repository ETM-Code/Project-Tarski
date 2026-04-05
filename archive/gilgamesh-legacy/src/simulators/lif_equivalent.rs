use crate::math_functions::lif::{adapt, crossing_time, relax_towards};
use crate::math_functions::neuron_physics::{
    compute_comparator_with_pulse_stretch, ComparatorConfig, PulseStretchConfig,
};
use crate::simulators::config::{
    AnalogOutCfg, ComparatorCfg, Membrane, NetworkConfig, NeuronConfig, SimCfg, Synapse,
    SynapseType,
};
use crate::simulators::constants::{
    COMPARATOR_RAIL_DROP_V, DEFAULT_ANALOG_LOAD_R_OHM, DEFAULT_ANALOG_SERIES_R_OHM,
    DEFAULT_DIODE_DROP_V, DEFAULT_INJECTION_R_OHM, DIVIDER_R_VH_OHM, DIVIDER_R_VL_OHM,
};
use serde::{
    ser::{SerializeMap, SerializeSeq, SerializeStruct},
    Deserialize, Deserializer, Serialize, Serializer,
};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct SimulationSample {
    pub time_s: f64,
    pub v_mem: f64,
    pub u_deflection: f64,
    pub v_comp: f64,
    pub v_ana: f64,
    pub v_outmix: f64,
    pub theta_over_vref: f64,
    pub v_syn: Vec<f64>,
}

pub struct RuntimeSample<'a> {
    pub time_s: f64,
    pub v_mem: f64,
    pub u_deflection: f64,
    pub v_comp: f64,
    pub v_ana: f64,
    pub v_outmix: f64,
    pub theta_over_vref: f64,
    pub v_syn: &'a [f64],
}

#[derive(Debug, Clone)]
pub struct EquivalentNeuron {
    pub membrane: Membrane,
    pub comparator: ComparatorCfg,
    pub threshold: ThresholdModel,
    pub analog: AnalogModel,
    pub supplies: crate::simulators::config::Supplies,
    pub sim: SimCfg,
    synapses: Vec<SynapseRuntime>,
    synapse_names: Arc<Vec<Arc<str>>>,
    pub tau_membrane: f64,
    pub comparator_high: f64,
    /// Pulse stretching configuration for RC decay
    pub pulse_stretch: PulseStretchConfig,
}

#[derive(Debug, Clone)]
pub struct ThresholdModel {
    pub theta_low: f64,
    pub theta_high: f64,
    pub tau_theta: f64,
    pub c_adapt: f64,
}

#[derive(Debug, Clone)]
pub struct AnalogModel {
    pub scale: f64,
    pub clamp_to_rails: bool,
}

#[derive(Debug, Clone)]
struct SpikeWindow {
    start: f64,
    end: f64,
    delta_v: f64,
}

#[derive(Debug, Clone)]
struct SynapseRuntime {
    weight_inv: f64,
    spikes: Vec<SpikeWindow>,
}

struct SimulationRecorder {
    capture: bool,
    samples: Vec<SimulationSample>,
    synapse_names: Arc<Vec<Arc<str>>>,
}

impl SimulationRecorder {
    fn new(names: Arc<Vec<Arc<str>>>, capture: bool, capacity: usize) -> Self {
        let samples = if capture {
            Vec::with_capacity(capacity)
        } else {
            Vec::new()
        };
        Self {
            capture,
            samples,
            synapse_names: names,
        }
    }

    fn record(&mut self, sample: RuntimeSample<'_>) {
        if !self.capture {
            return;
        }
        self.samples.push(SimulationSample {
            time_s: sample.time_s,
            v_mem: sample.v_mem,
            u_deflection: sample.u_deflection,
            v_comp: sample.v_comp,
            v_ana: sample.v_ana,
            v_outmix: sample.v_outmix,
            theta_over_vref: sample.theta_over_vref,
            v_syn: sample.v_syn.to_vec(),
        });
    }

    fn finish(self) -> SimulationResult {
        SimulationResult {
            samples: self.samples,
            synapse_names: self.synapse_names,
        }
    }
}

impl From<&AnalogOutCfg> for AnalogModel {
    fn from(cfg: &AnalogOutCfg) -> Self {
        let magnitude = if cfg.gain.abs() > f64::EPSILON {
            cfg.gain.abs()
        } else if cfg.inverting {
            (cfg.r2_ohm / cfg.r1_ohm).abs()
        } else {
            1.0 + cfg.r2_ohm / cfg.r1_ohm
        };

        let input_sign = if cfg.sign >= 0 { 1.0 } else { -1.0 };
        let scale = if cfg.inverting {
            -magnitude * input_sign
        } else {
            magnitude * input_sign
        };

        Self {
            clamp_to_rails: cfg.clamp_to_rails,
            scale,
        }
    }
}

impl EquivalentNeuron {
    pub fn from_configs(neuron: &NeuronConfig, network: &NetworkConfig) -> Self {
        let theta_model = ThresholdModel::from_components(
            neuron.supplies.vref,
            neuron.supplies.vdd,
            &neuron.threshold,
            &neuron.comparator,
        );

        let (synapses, synapse_names) = build_synapse_runtimes(&network.synapses);

        let tau_membrane = neuron.membrane.c_mem_f * neuron.membrane.r_leak_ohm;
        Self {
            membrane: neuron.membrane.clone(),
            comparator: neuron.comparator.clone(),
            threshold: theta_model,
            analog: AnalogModel::from(&neuron.analog_out),
            supplies: neuron.supplies.clone(),
            sim: neuron.simulation.clone(),
            synapses,
            synapse_names,
            tau_membrane,
            comparator_high: (neuron.comparator.vhigh_v - COMPARATOR_RAIL_DROP_V)
                .max(neuron.comparator.vlow_v),
            pulse_stretch: neuron.pulse_stretch.to_physics_config(),
        }
    }

    pub fn run(&self) -> SimulationResult {
        let expected_steps = (self.sim.tstop_s / self.sim.tstep_s).ceil() as usize + 1;
        let mut recorder =
            SimulationRecorder::new(Arc::clone(&self.synapse_names), true, expected_steps);
        self.simulate(|sample| recorder.record(sample));
        recorder.finish()
    }

    pub fn simulate<F>(&self, mut on_sample: F)
    where
        F: FnMut(RuntimeSample<'_>),
    {
        let synapse_count = self.synapses.len();
        let mut spike_indices = vec![0usize; synapse_count];
        let mut syn_voltages = vec![self.supplies.vref; synapse_count];
        let mut total_current = 0.0;
        let vref = self.supplies.vref;

        let base_dt = self.sim.min_dt_s.unwrap_or(self.sim.tstep_s);
        let mut max_dt = self
            .sim
            .max_dt_s
            .unwrap_or(self.sim.tstep_s * self.sim.max_step_factor.max(1.0));
        if max_dt < base_dt {
            max_dt = base_dt;
        }

        let mut time = 0.0;
        let mut u = 0.0; // deflection above vref
        let mut theta = self.threshold.theta_low;
        let mut v_comp = self.comparator.vlow_v;

        const TIME_EPS: f64 = 1e-12;

        while time < self.sim.tstop_s - TIME_EPS {
            let mut next_syn_event = f64::INFINITY;

            for (idx, syn) in self.synapses.iter().enumerate() {
                let tooling = &mut spike_indices[idx];

                while *tooling < syn.spikes.len() && time > syn.spikes[*tooling].end + TIME_EPS {
                    *tooling += 1;
                }

                let mut voltage = vref;
                let prev_voltage = syn_voltages[idx];
                if *tooling < syn.spikes.len() {
                    let window = &syn.spikes[*tooling];
                    if time < window.start - TIME_EPS {
                        next_syn_event = next_syn_event.min((window.start - time).max(0.0));
                    }
                    if time >= window.start - TIME_EPS && time <= window.end + TIME_EPS {
                        voltage = vref + window.delta_v;
                        next_syn_event = next_syn_event.min((window.end - time).max(0.0));
                    }
                }

                if (voltage - prev_voltage).abs() > 1e-12 {
                    total_current -= (prev_voltage - vref) * syn.weight_inv;
                    total_current += (voltage - vref) * syn.weight_inv;
                    syn_voltages[idx] = voltage;
                }
            }

            let u_inf = self.membrane.r_leak_ohm * total_current;

            // Determine adaptive step duration
            let mut dt = if v_comp > self.comparator.vlow_v + 1e-9 {
                base_dt
            } else {
                max_dt
            };

            if next_syn_event.is_finite() {
                if next_syn_event < base_dt {
                    dt = dt.min(next_syn_event.max(TIME_EPS));
                } else {
                    dt = dt.min(next_syn_event);
                }
            }

            let theta_eff = theta + self.comparator.offset_v;
            if let Some(mut cross_dt) = crossing_time(u, u_inf, self.tau_membrane, theta_eff) {
                if cross_dt.is_sign_negative() {
                    cross_dt = TIME_EPS;
                }
                if cross_dt < base_dt {
                    dt = dt.min(cross_dt.max(TIME_EPS));
                } else {
                    dt = dt.min(cross_dt);
                }
            }

            let remaining = (self.sim.tstop_s - time).max(0.0);
            if remaining <= TIME_EPS {
                break;
            }
            if remaining < dt {
                dt = remaining;
            }

            if dt <= TIME_EPS {
                dt = base_dt.min(remaining);
                if dt <= TIME_EPS {
                    break;
                }
            }

            let u_next = relax_towards(u, u_inf, self.tau_membrane, dt);

            // Comparator logic with pulse stretching (unified physics)
            let comp_physics_cfg = ComparatorConfig {
                v_low: self.comparator.vlow_v,
                v_high: self.comparator_high,
                offset: 0.0, // offset already included in theta_eff
            };
            let comp_result = compute_comparator_with_pulse_stretch(
                u_next,
                theta_eff,
                v_comp,
                &comp_physics_cfg,
                &self.pulse_stretch,
                dt,
            );
            let v_comp_next = comp_result.v_comp;

            let theta_target =
                if v_comp_next > (self.comparator.vlow_v + self.comparator.vhigh_v) * 0.5 {
                    self.threshold.theta_high
                } else {
                    self.threshold.theta_low
                };

            let mut theta_next = adapt(theta, theta_target, self.threshold.tau_theta, dt);

            if v_comp_next > self.comparator.vlow_v + 1e-9 {
                let overdrive =
                    (v_comp_next - (self.supplies.vref + theta_next) - DEFAULT_DIODE_DROP_V)
                        .max(0.0);
                let dtheta = (overdrive / DEFAULT_INJECTION_R_OHM) * (dt / self.threshold.c_adapt);
                theta_next = (theta_next + dtheta).min(self.threshold.theta_high);
            }

            let v_mem = self.supplies.vref - u_next;
            let v_ana = self.analog_output(v_mem);
            let v_outmix = mixed_output(v_ana, v_comp_next);

            time += dt;

            on_sample(RuntimeSample {
                time_s: time,
                v_mem,
                u_deflection: u_next,
                v_comp: v_comp_next,
                v_ana,
                v_outmix,
                theta_over_vref: theta_next,
                v_syn: &syn_voltages,
            });

            u = u_next;
            theta = theta_next;
            v_comp = v_comp_next;

            if time >= self.sim.tstop_s - TIME_EPS {
                break;
            }
        }
    }

    pub fn simulate_core(&self) {
        self.simulate(|_| {});
    }

    fn analog_output(&self, v_mem: f64) -> f64 {
        let vin = v_mem - self.supplies.vref;
        let mut vout = self.analog.scale * vin;
        if self.analog.clamp_to_rails {
            vout = vout.clamp(0.0, self.supplies.vdd);
        }

        vout
    }
}

impl ThresholdModel {
    pub fn from_components(
        vref: f64,
        vdd: f64,
        th: &crate::simulators::config::Threshold,
        comp: &ComparatorCfg,
    ) -> Self {
        let scale = th.divider_scale.max(1e-6);
        let r_vh = DIVIDER_R_VH_OHM * scale;
        let r_vl = DIVIDER_R_VL_OHM * scale;
        let dv_out = (comp.vhigh_v - comp.vlow_v).max(1e-6);
        let beta = (th.hysteresis_v / dv_out).clamp(1e-6, 0.999999);
        let g_div = (1.0 / r_vh) + (1.0 / r_vl);
        let r_f = 1.0 / (beta * g_div / (1.0 - beta));

        let g_total = g_div + 1.0 / r_f;
        let tau_theta = th.c_adapt_f / g_total;

        let numerator_low = (vdd - vref) / r_vh + (comp.vlow_v - vref) / r_f;
        let theta_low = numerator_low / g_total;
        let numerator_high = (vdd - vref) / r_vh + (comp.vhigh_v - vref) / r_f;
        let theta_high = numerator_high / g_total;

        Self {
            theta_low,
            theta_high,
            tau_theta,
            c_adapt: th.c_adapt_f,
        }
    }
}

fn build_synapse_runtimes(synapses: &[Synapse]) -> (Vec<SynapseRuntime>, Arc<Vec<Arc<str>>>) {
    let mut runtimes = Vec::with_capacity(synapses.len());
    let mut names = Vec::with_capacity(synapses.len());

    for syn in synapses {
        let name: Arc<str> = Arc::from(syn.name.clone());
        let sign = if syn.r#type == SynapseType::Excitatory {
            1.0
        } else {
            -1.0
        };

        let mut spikes = Vec::with_capacity(syn.spikes.len());
        for spike in &syn.spikes {
            let start = spike.t_ms * 1e-3;
            let end = start + spike.width_ms * 1e-3;
            spikes.push(SpikeWindow {
                start,
                end,
                delta_v: sign * spike.amp_v,
            });
        }

        runtimes.push(SynapseRuntime {
            weight_inv: if syn.weight_ohm.abs() > 1e-12 {
                1.0 / syn.weight_ohm
            } else {
                0.0
            },
            spikes,
        });
        names.push(name);
    }

    (runtimes, Arc::new(names))
}

fn mixed_output(v_ana: f64, v_comp: f64) -> f64 {
    let alpha =
        DEFAULT_ANALOG_LOAD_R_OHM / (DEFAULT_ANALOG_LOAD_R_OHM + DEFAULT_ANALOG_SERIES_R_OHM);
    let analog_path = alpha * v_ana;
    let spike_path = (v_comp - DEFAULT_DIODE_DROP_V).max(analog_path);
    spike_path
}

#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub samples: Vec<SimulationSample>,
    synapse_names: Arc<Vec<Arc<str>>>,
}

impl SimulationResult {
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn to_csv(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let mut wtr = csv::Writer::from_path(path)?;
        wtr.write_record([
            "time_s",
            "v_mem",
            "u_deflection",
            "v_comp",
            "v_ana",
            "v_outmix",
            "theta_over_vref",
        ])?;

        for sample in &self.samples {
            wtr.write_record([
                sample.time_s.to_string(),
                sample.v_mem.to_string(),
                sample.u_deflection.to_string(),
                sample.v_comp.to_string(),
                sample.v_ana.to_string(),
                sample.v_outmix.to_string(),
                sample.theta_over_vref.to_string(),
            ])?;
        }

        wtr.flush()?;
        Ok(())
    }

    pub fn synapse_names(&self) -> &[Arc<str>] {
        self.synapse_names.as_ref().as_slice()
    }
}

impl Serialize for SimulationResult {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("SimulationResult", 1)?;
        state.serialize_field(
            "samples",
            &SerializableSamples {
                samples: &self.samples,
                names: self.synapse_names.as_ref(),
            },
        )?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for SimulationResult {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct SimulationResultRepr {
            samples: Vec<SimulationSampleRepr>,
        }

        #[derive(Deserialize)]
        struct SimulationSampleRepr {
            time_s: f64,
            v_mem: f64,
            u_deflection: f64,
            v_comp: f64,
            v_ana: f64,
            v_outmix: f64,
            theta_over_vref: f64,
            v_syn: HashMap<String, f64>,
        }

        let repr = SimulationResultRepr::deserialize(deserializer)?;
        let mut names: Vec<Arc<str>> = Vec::new();

        if let Some(first) = repr.samples.first() {
            let mut ordered: Vec<String> = first.v_syn.keys().cloned().collect();
            ordered.sort();
            names = ordered.into_iter().map(Arc::<str>::from).collect();
        }

        let mut samples = Vec::with_capacity(repr.samples.len());
        for sample in repr.samples {
            if names.is_empty() {
                let mut ordered: Vec<String> = sample.v_syn.keys().cloned().collect();
                ordered.sort();
                names = ordered.into_iter().map(Arc::<str>::from).collect();
            }

            let mut v_syn = Vec::with_capacity(names.len());
            for name in &names {
                v_syn.push(*sample.v_syn.get(name.as_ref()).unwrap_or(&0.0));
            }

            samples.push(SimulationSample {
                time_s: sample.time_s,
                v_mem: sample.v_mem,
                u_deflection: sample.u_deflection,
                v_comp: sample.v_comp,
                v_ana: sample.v_ana,
                v_outmix: sample.v_outmix,
                theta_over_vref: sample.theta_over_vref,
                v_syn,
            });
        }

        Ok(SimulationResult {
            samples,
            synapse_names: Arc::new(names),
        })
    }
}

struct SerializableSamples<'a> {
    samples: &'a [SimulationSample],
    names: &'a [Arc<str>],
}

impl<'a> Serialize for SerializableSamples<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.samples.len()))?;
        for sample in self.samples {
            seq.serialize_element(&SerializableSample {
                sample,
                names: self.names,
            })?;
        }
        seq.end()
    }
}

struct SerializableSample<'a> {
    sample: &'a SimulationSample,
    names: &'a [Arc<str>],
}

impl<'a> Serialize for SerializableSample<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("SimulationSample", 7)?;
        state.serialize_field("time_s", &self.sample.time_s)?;
        state.serialize_field("v_mem", &self.sample.v_mem)?;
        state.serialize_field("u_deflection", &self.sample.u_deflection)?;
        state.serialize_field("v_comp", &self.sample.v_comp)?;
        state.serialize_field("v_ana", &self.sample.v_ana)?;
        state.serialize_field("v_outmix", &self.sample.v_outmix)?;
        state.serialize_field("theta_over_vref", &self.sample.theta_over_vref)?;
        state.serialize_field(
            "v_syn",
            &SynapseMapSerializer {
                names: self.names,
                values: &self.sample.v_syn,
            },
        )?;
        state.end()
    }
}

struct SynapseMapSerializer<'a> {
    names: &'a [Arc<str>],
    values: &'a [f64],
}

impl<'a> Serialize for SynapseMapSerializer<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(self.names.len()))?;
        for (name, value) in self.names.iter().zip(self.values.iter()) {
            map.serialize_entry(name.as_ref(), value)?;
        }
        map.end()
    }
}

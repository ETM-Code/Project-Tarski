use crate::network::compiled::{
    CompiledLayer, CompiledNetwork, CompiledReadout, IncomingConnectivity, InitialState,
    LayerRuntimeType, NeuronParameters, OutgoingConnectivity, ReadoutRuntimeType, StimulusChannel,
};
use crate::network::design::{
    ConnectRule, Design, IndexRef, IndexSelector, LayerSpec, LayerType, NeuronTemplate,
    ReadoutType, SynapseType,
};
use anyhow::{anyhow, Result};
use std::collections::{BTreeMap, HashMap};

pub struct NetworkCompiler<'a> {
    design: &'a Design,
    layer_layout: Vec<LayerLayout<'a>>,
    layer_lookup: HashMap<&'a str, usize>,
}

struct LayerLayout<'a> {
    spec: &'a LayerSpec,
    template: &'a NeuronTemplate,
    offset: usize,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct EdgeKey {
    src: usize,
    dst: usize,
}

#[derive(Debug, Clone)]
struct EdgeValue {
    g: f64,
    synapse_type: u8,
}

impl<'a> NetworkCompiler<'a> {
    pub fn new(design: &'a Design) -> Result<Self> {
        if design.layers.is_empty() {
            return Err(anyhow!("design must contain at least one layer"));
        }

        let mut neuron_templates: HashMap<&str, &NeuronTemplate> = HashMap::new();
        for (name, template) in &design.neuron_templates {
            neuron_templates.insert(name.as_str(), template);
        }
        if neuron_templates.get("default").is_none() {
            if let Some(first) = design.neuron_templates.values().next() {
                neuron_templates.insert("default", first);
            }
        }

        let mut offset = 0usize;
        let mut layer_layout = Vec::with_capacity(design.layers.len());
        let mut layer_lookup = HashMap::new();
        for (idx, layer) in design.layers.iter().enumerate() {
            let template_name = layer.neuron.as_str();
            let template = neuron_templates
                .get(template_name)
                .copied()
                .ok_or_else(|| anyhow!("unknown neuron template '{}'", template_name))?;
            layer_layout.push(LayerLayout {
                spec: layer,
                template,
                offset,
            });
            layer_lookup.insert(layer.id.as_str(), idx);
            offset += layer.size;
        }

        Ok(Self {
            design,
            layer_layout,
            layer_lookup,
        })
    }

    pub fn compile(self) -> Result<CompiledNetwork> {
        let neuron_count = self
            .layer_layout
            .last()
            .map(|last| last.offset + last.spec.size)
            .unwrap_or(0);

        let mut neuron_cm = Vec::with_capacity(neuron_count);
        let mut neuron_r_leak = Vec::with_capacity(neuron_count);
        let mut neuron_tau = Vec::with_capacity(neuron_count);
        let mut neuron_theta_mode = Vec::with_capacity(neuron_count);
        let mut neuron_theta0 = Vec::with_capacity(neuron_count);

        for layout in &self.layer_layout {
            let tmpl = layout.template;
            let tau = tmpl.cm * tmpl.r_leak;
            let (theta_mode, theta0) = match &tmpl.theta {
                crate::network::design::ThetaTemplate::Constant { theta0 } => (0u8, *theta0),
                crate::network::design::ThetaTemplate::Adaptive { theta0, .. } => (1u8, *theta0),
            };

            for _ in 0..layout.spec.size {
                neuron_cm.push(tmpl.cm);
                neuron_r_leak.push(tmpl.r_leak);
                neuron_tau.push(tau);
                neuron_theta_mode.push(theta_mode);
                neuron_theta0.push(theta0);
            }
        }

        let mut edge_map: BTreeMap<EdgeKey, EdgeValue> = BTreeMap::new();

        for connect in &self.design.connect {
            let src_layer = self.layer_by_id(&connect.from)?;
            let dst_layer = self.layer_by_id(&connect.to)?;
            let src_indices = self.layer_indices(src_layer);
            let dst_indices = self.layer_indices(dst_layer);
            let synapse_type = match connect.synapse.r#type {
                SynapseType::Excitatory => 0u8,
                SynapseType::Inhibitory => 1u8,
            };

            match &connect.rule {
                ConnectRule::Dense => {
                    for &src in &src_indices {
                        for &dst in &dst_indices {
                            let g = synaptic_g(
                                connect.synapse.resistance,
                                neuron_cm[dst],
                                synapse_type,
                            );
                            edge_map.insert(EdgeKey { src, dst }, EdgeValue { g, synapse_type });
                        }
                    }
                }
                ConnectRule::DenseNoSelf => {
                    for &src in &src_indices {
                        for &dst in &dst_indices {
                            if src == dst {
                                continue; // Skip self-connections
                            }
                            let g = synaptic_g(
                                connect.synapse.resistance,
                                neuron_cm[dst],
                                synapse_type,
                            );
                            edge_map.insert(EdgeKey { src, dst }, EdgeValue { g, synapse_type });
                        }
                    }
                }
                ConnectRule::FixedFanIn { fan_in } => {
                    if *fan_in == 0 {
                        continue;
                    }
                    if src_indices.is_empty() {
                        continue;
                    }
                    for (idx, &dst) in dst_indices.iter().enumerate() {
                        for i in 0..*fan_in {
                            let choice = (idx + i) % src_indices.len();
                            let src = src_indices[choice];
                            let g = synaptic_g(
                                connect.synapse.resistance,
                                neuron_cm[dst],
                                synapse_type,
                            );
                            edge_map.insert(EdgeKey { src, dst }, EdgeValue { g, synapse_type });
                        }
                    }
                }
                ConnectRule::Prob { probability, seed } => {
                    let mut rng = Lcg::new(*seed, &connect.from, &connect.to);
                    for &src in &src_indices {
                        for &dst in &dst_indices {
                            if rng.next_f64() < *probability {
                                let g = synaptic_g(
                                    connect.synapse.resistance,
                                    neuron_cm[dst],
                                    synapse_type,
                                );
                                edge_map
                                    .insert(EdgeKey { src, dst }, EdgeValue { g, synapse_type });
                            }
                        }
                    }
                }
                ConnectRule::Ring { shift } => {
                    if dst_indices.is_empty() || src_indices.is_empty() {
                        continue;
                    }
                    let len = src_indices.len() as isize;
                    let base_shift = *shift as isize;
                    for (idx, &dst) in dst_indices.iter().enumerate() {
                        let sel = (idx as isize + base_shift).rem_euclid(len);
                        let src_idx = src_indices[sel as usize];
                        let g =
                            synaptic_g(connect.synapse.resistance, neuron_cm[dst], synapse_type);
                        edge_map
                            .insert(EdgeKey { src: src_idx, dst }, EdgeValue { g, synapse_type });
                    }
                }
            }
        }

        for override_edge in &self.design.overrides {
            let sources = self.resolve_index(&override_edge.from)?;
            let targets = self.resolve_index(&override_edge.to)?;
            for src in &sources {
                for dst in &targets {
                    let synapse_type = match override_edge.synapse.r#type {
                        SynapseType::Excitatory => 0u8,
                        SynapseType::Inhibitory => 1u8,
                    };
                    let g = synaptic_g(
                        override_edge.synapse.resistance,
                        neuron_cm[*dst],
                        synapse_type,
                    );
                    edge_map.insert(
                        EdgeKey {
                            src: *src,
                            dst: *dst,
                        },
                        EdgeValue { g, synapse_type },
                    );
                }
            }
        }

        let edges: Vec<(EdgeKey, EdgeValue)> = edge_map.into_iter().collect();
        let synapse_count = edges.len();

        let (incoming, edge_lookup) = build_incoming(neuron_count, &edges);
        let outgoing = build_outgoing(neuron_count, &edges, &edge_lookup);

        let mut globals = self.design.globals.values.clone();
        if !globals.contains_key("alpha_out") {
            globals.insert("alpha_out".to_string(), 1.0);
        }
        if !globals.contains_key("delta") {
            globals.insert("delta".to_string(), 0.0);
        }

        // Get pulse stretch duration before moving globals
        let pulse_stretch_duration_s = globals.get("pulse_stretch_duration_s").copied().unwrap_or(0.0);

        let layers: Vec<CompiledLayer> = self
            .layer_layout
            .iter()
            .map(|layout| CompiledLayer {
                id: layout.spec.id.clone(),
                offset: layout.offset,
                size: layout.spec.size,
                layer_type: match layout.spec.r#type {
                    LayerType::Input => LayerRuntimeType::Input,
                    LayerType::Hidden => LayerRuntimeType::Hidden,
                    LayerType::Readout => LayerRuntimeType::Readout,
                },
            })
            .collect();

        let readouts = self.compile_readouts(&layers)?;
        let stimuli = self.compile_stimuli()?;

        let state0 = InitialState {
            t: 0.0,
            u: vec![0.0; neuron_count],
            theta: neuron_theta0.clone(),
            comp: vec![0.0; neuron_count],
        };

        let compiled = CompiledNetwork {
            version: self
                .design
                .version
                .clone()
                .unwrap_or_else(|| "0.1".to_string()),
            timebase: self
                .design
                .units
                .as_ref()
                .and_then(|u| u.time.clone())
                .unwrap_or_else(|| "s".to_string()),
            neuron_count,
            synapse_count,
            globals,
            layers,
            neuron: NeuronParameters {
                cm: neuron_cm,
                r_leak: neuron_r_leak,
                tau_m: neuron_tau,
                theta_mode: neuron_theta_mode,
                theta0: neuron_theta0,
                // Default pulse stretch duration: 0 = no stretching (single timestep)
                // Can be configured via globals["pulse_stretch_duration_s"] or per-template settings
                pulse_stretch_duration: vec![pulse_stretch_duration_s; neuron_count],
            },
            incoming,
            outgoing: Some(outgoing),
            readouts,
            stimuli,
            state0,
            training_checkpoint: None,
        };

        Ok(compiled)
    }

    fn layer_by_id(&self, name: &str) -> Result<&LayerLayout<'a>> {
        let idx = self
            .layer_lookup
            .get(name)
            .copied()
            .ok_or_else(|| anyhow!("layer '{}' not found", name))?;
        Ok(&self.layer_layout[idx])
    }

    fn layer_indices(&self, layout: &LayerLayout<'a>) -> Vec<usize> {
        (layout.offset..layout.offset + layout.spec.size).collect()
    }

    fn resolve_index(&self, spec: &str) -> Result<Vec<usize>> {
        let parsed = IndexRef::parse(spec)?;
        let layer = self.layer_by_id(&parsed.layer)?;
        let base = layer.offset;
        let size = layer.spec.size;
        let mut indices = Vec::new();
        match parsed.selector {
            IndexSelector::All => {
                indices.extend(base..base + size);
            }
            IndexSelector::Single(i) => {
                if i >= size {
                    return Err(anyhow!(
                        "index {} out of bounds for layer '{}' (size {})",
                        i,
                        layer.spec.id,
                        size
                    ));
                }
                indices.push(base + i);
            }
            IndexSelector::Range { start, end } => {
                if end >= size {
                    return Err(anyhow!(
                        "range {}:{} exceeds size {} for layer '{}'",
                        start,
                        end,
                        size,
                        layer.spec.id
                    ));
                }
                for idx in start..=end {
                    indices.push(base + idx);
                }
            }
        }
        Ok(indices)
    }

    fn compile_readouts(&self, layers: &[CompiledLayer]) -> Result<Vec<CompiledReadout>> {
        if self.design.readouts.is_empty() {
            return Ok(Vec::new());
        }
        let mut lookup: HashMap<&str, &CompiledLayer> = HashMap::new();
        for layer in layers {
            lookup.insert(layer.id.as_str(), layer);
        }
        let mut compiled = Vec::new();
        for readout in &self.design.readouts {
            let source_layer = readout
                .source_layer
                .as_ref()
                .ok_or_else(|| anyhow!("readout '{}' missing source_layer", readout.id))?;
            let layer = lookup
                .get(source_layer.as_str())
                .copied()
                .ok_or_else(|| anyhow!("unknown readout layer '{}'", source_layer))?;
            let indices = (layer.offset..layer.offset + layer.size).collect();
            let r#type = match readout.r#type {
                ReadoutType::Analog => ReadoutRuntimeType::Analog,
                ReadoutType::Spike => ReadoutRuntimeType::Spike,
            };
            compiled.push(CompiledReadout {
                id: readout.id.clone(),
                r#type,
                indices,
                signal: readout.signal.clone(),
                post: readout.post.clone(),
                tau_s: readout.tau_s,
            });
        }
        Ok(compiled)
    }

    fn compile_stimuli(&self) -> Result<Vec<StimulusChannel>> {
        if self.design.external_inputs.is_empty() {
            return Ok(Vec::new());
        }
        let mut channels = Vec::with_capacity(self.design.external_inputs.len());
        for stim in &self.design.external_inputs {
            let targets = self.resolve_index(&stim.target)?;
            if targets.is_empty() {
                continue;
            }
            let (mut times, values) = match stim.wave.as_str() {
                "pwl_ref" => {
                    let t_on = require_param(&stim.id, "t_on", &stim.params)?;
                    let t_off = require_param(&stim.id, "t_off", &stim.params)?;
                    let amp = require_param(&stim.id, "amp", &stim.params)?;
                    if !(t_off > t_on) {
                        return Err(anyhow!("stimulus '{}' requires t_off > t_on", stim.id));
                    }
                    (
                        vec![0.0, t_on, t_off, t_off + 1e-12],
                        vec![0.0, amp, amp, 0.0],
                    )
                }
                "step" => {
                    let t = require_param(&stim.id, "time", &stim.params)?;
                    let amp = require_param(&stim.id, "amp", &stim.params)?;
                    (vec![0.0, t, t + 1e-12], vec![0.0, amp, amp])
                }
                other => {
                    return Err(anyhow!("unsupported stimulus wave '{}'", other));
                }
            };
            enforce_monotonic(&mut times)?;
            channels.push(StimulusChannel {
                id: stim.id.clone(),
                target_indices: targets,
                times,
                values,
            });
        }
        Ok(channels)
    }
}

fn synaptic_g(resistance: f64, cm: f64, synapse_type: u8) -> f64 {
    if resistance.abs() < f64::EPSILON || cm.abs() < f64::EPSILON {
        return 0.0;
    }
    let base = 1.0 / (cm * resistance);
    if synapse_type == 1 {
        -base
    } else {
        base
    }
}

/// Initialize weights with random perturbation to break symmetry.
/// Uses aggressive randomization to create differentiated receptive fields.
pub fn randomize_weights(network: &mut CompiledNetwork, seed: Option<u64>) {
    use rand::{Rng, SeedableRng};
    let mut rng = match seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_os_rng(),
    };

    // Use much larger variance to create differentiated receptive fields
    // Some weights will be near zero, others will be strong
    // This is essential for hidden neurons to develop different response patterns
    for (idx, g_val) in network.incoming.g.iter_mut().enumerate() {
        let base = *g_val;
        let synapse_type = network.incoming.synapse_type[idx];

        // Sample from uniform [0, 2*base] - centered on base but with high variance
        // This gives some weights near 0 and others near 2*base
        let random_factor: f64 = rng.random_range(0.0..2.0);
        let perturbed = base * random_factor;

        // Ensure sign constraints
        if synapse_type == 0 {
            *g_val = perturbed.abs().max(1e-6); // Excitatory: positive
        } else {
            *g_val = -perturbed.abs().max(1e-6); // Inhibitory: negative
        }
    }
}

fn build_incoming(
    neuron_count: usize,
    edges: &[(EdgeKey, EdgeValue)],
) -> (IncomingConnectivity, HashMap<EdgeKey, usize>) {
    let mut counts = vec![0usize; neuron_count];
    for (key, _) in edges {
        counts[key.dst] += 1;
    }
    let mut row_ptr = vec![0usize; neuron_count + 1];
    for i in 0..neuron_count {
        row_ptr[i + 1] = row_ptr[i] + counts[i];
    }
    let mut tooling = vec![0usize; neuron_count];
    let mut src = vec![0usize; edges.len()];
    let mut g = vec![0f64; edges.len()];
    let mut syn = vec![0u8; edges.len()];
    let mut index_map = HashMap::with_capacity(edges.len());
    for (key, value) in edges.iter() {
        let pos = row_ptr[key.dst] + tooling[key.dst];
        src[pos] = key.src;
        g[pos] = value.g;
        syn[pos] = value.synapse_type;
        tooling[key.dst] += 1;
        index_map.insert(key.clone(), pos);
    }
    (
        IncomingConnectivity {
            row_ptr,
            src,
            g,
            synapse_type: syn,
        },
        index_map,
    )
}

fn build_outgoing(
    neuron_count: usize,
    edges: &[(EdgeKey, EdgeValue)],
    index_lookup: &HashMap<EdgeKey, usize>,
) -> OutgoingConnectivity {
    let mut counts = vec![0usize; neuron_count];
    for (key, _) in edges {
        counts[key.src] += 1;
    }
    let mut col_ptr = vec![0usize; neuron_count + 1];
    for i in 0..neuron_count {
        col_ptr[i + 1] = col_ptr[i] + counts[i];
    }
    let mut tooling = vec![0usize; neuron_count];
    let mut dst = vec![0usize; edges.len()];
    let mut syn_index = vec![0usize; edges.len()];
    for (key, _) in edges {
        let idx = *index_lookup.get(key).expect("edge index");
        let pos = col_ptr[key.src] + tooling[key.src];
        dst[pos] = key.dst;
        syn_index[pos] = idx;
        tooling[key.src] += 1;
    }
    OutgoingConnectivity {
        col_ptr,
        dst,
        synapse_index: syn_index,
    }
}

fn require_param(id: &str, key: &str, params: &BTreeMap<String, serde_json::Value>) -> Result<f64> {
    params
        .get(key)
        .ok_or_else(|| anyhow!("stimulus '{}' missing parameter '{}'", id, key))
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_i64().map(|v| v as f64))
                .or_else(|| value.as_u64().map(|v| v as f64))
                .ok_or_else(|| anyhow!("stimulus '{}' parameter '{}' must be numeric", id, key))
        })
}

fn enforce_monotonic(times: &mut Vec<f64>) -> Result<()> {
    if times.is_empty() {
        return Err(anyhow!("stimulus requires at least one timestamp"));
    }
    let mut last = -f64::INFINITY;
    for value in times.iter_mut() {
        if *value <= last {
            *value = (last + 1e-12).max(*value + 1e-12);
        }
        last = *value;
    }
    Ok(())
}

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: Option<u64>, from: &str, to: &str) -> Self {
        let mut hash = 0xcbf29ce484222325u64;
        for b in from.bytes().chain(to.bytes()) {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        let state = seed.unwrap_or(hash);
        Self { state: state | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64();
        ((bits >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

pub fn compile_design(design: &Design) -> Result<CompiledNetwork> {
    NetworkCompiler::new(design)?.compile()
}

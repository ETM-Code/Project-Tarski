//! Architecture presets for common network configurations.
//!
//! Builds network designs from preset configurations like Mnist7x7.

use crate::network::design::{
    ConnectRule, ConnectSpec, Design, Globals, LayerSpec, LayerType, NeuronTemplate,
    ReadoutSpec, ReadoutType, ResetTemplate, SynapseSpec, SynapseType, ThetaTemplate,
};
use crate::training::config::ArchitecturePreset;
use std::collections::HashMap;

/// Default neuron parameters for training.
#[derive(Debug, Clone)]
pub struct NeuronParams {
    /// Membrane capacitance (F)
    pub cm: f64,
    /// Leak resistance (Ω)
    pub r_leak: f64,
    /// Spike threshold (V)
    pub theta0: f64,
    /// Reset resistance (Ω)
    pub r_reset: f64,
    /// Switch-on resistance (Ω)
    pub ron_switch: f64,
}

impl Default for NeuronParams {
    fn default() -> Self {
        Self {
            cm: 3.3e-8,      // 33nF
            r_leak: 120_000.0, // 120kΩ
            theta0: 0.65,     // 650mV threshold
            r_reset: 750.0,   // 750Ω reset
            ron_switch: 30.0, // 30Ω switch
        }
    }
}

/// Default synapse resistance (Ω).
pub const DEFAULT_SYNAPSE_RESISTANCE: f64 = 15_000.0; // 15kΩ

/// Build a network design from an architecture preset.
pub fn build_design_from_preset(
    preset: &ArchitecturePreset,
    neuron_params: Option<NeuronParams>,
) -> Design {
    let params = neuron_params.unwrap_or_default();

    match preset {
        ArchitecturePreset::Custom => {
            // Return minimal design for custom
            build_minimal_design(&params)
        }
        ArchitecturePreset::Mnist7x7 {
            hidden_neurons,
            excitatory_only,
        } => build_mnist7x7_design(*hidden_neurons, *excitatory_only, &params),
        ArchitecturePreset::Mnist7x7Rate {
            hidden_neurons,
            excitatory_only,
        } => build_mnist7x7_rate_design(*hidden_neurons, *excitatory_only, &params),
        ArchitecturePreset::Feedforward {
            input_size,
            hidden_sizes,
            output_size,
        } => build_feedforward_design(*input_size, hidden_sizes, *output_size, &params),
    }
}

fn build_neuron_template(params: &NeuronParams) -> NeuronTemplate {
    NeuronTemplate {
        cm: params.cm,
        r_leak: params.r_leak,
        theta: ThetaTemplate::Constant {
            theta0: params.theta0,
        },
        reset: ResetTemplate {
            r_reset: params.r_reset,
            ron_switch: params.ron_switch,
        },
    }
}

fn build_minimal_design(params: &NeuronParams) -> Design {
    let mut neuron_templates = HashMap::new();
    neuron_templates.insert("default".to_string(), build_neuron_template(params));

    Design {
        version: Some("0.2".to_string()),
        units: None,
        globals: Globals::default(),
        neuron_templates,
        layers: vec![
            LayerSpec {
                id: "input".to_string(),
                r#type: LayerType::Input,
                size: 1,
                neuron: "default".to_string(),
            },
            LayerSpec {
                id: "output".to_string(),
                r#type: LayerType::Readout,
                size: 1,
                neuron: "default".to_string(),
            },
        ],
        connect: vec![ConnectSpec {
            from: "input".to_string(),
            to: "output".to_string(),
            rule: ConnectRule::Dense,
            synapse: SynapseSpec {
                r#type: SynapseType::Excitatory,
                resistance: DEFAULT_SYNAPSE_RESISTANCE,
            },
        }],
        overrides: Vec::new(),
        external_inputs: Vec::new(),
        readouts: vec![ReadoutSpec {
            id: "logits".to_string(),
            r#type: ReadoutType::Analog,
            source_layer: Some("output".to_string()),
            signal: "analog_background".to_string(),
            post: None,
            tau_s: Some(2e-3),
        }],
        checkpoint: None,
        training: None,
    }
}

/// Build MNIST 7x7 time-multiplexed network.
///
/// Architecture: 7 (input) → hidden_neurons → 10 (output)
/// Uses time-multiplexed input where each of 7 input neurons
/// receives one column of the 7x7 image row by row.
fn build_mnist7x7_design(
    hidden_neurons: usize,
    excitatory_only: bool,
    params: &NeuronParams,
) -> Design {
    let mut neuron_templates = HashMap::new();
    neuron_templates.insert("default".to_string(), build_neuron_template(params));

    // For hardware deployment, excitatory-only is preferred
    // If not excitatory_only, we still use Excitatory as the default
    // (Inhibitory connections would need separate handling)
    let synapse_type = SynapseType::Excitatory;
    let _ = excitatory_only; // May be used for future mixed networks

    let mut globals = Globals::default();
    globals.values.insert("alpha_out".to_string(), 1.0.into());
    globals.values.insert("delta".to_string(), 5.0e-4.into());

    Design {
        version: Some("0.2".to_string()),
        units: None,
        globals,
        neuron_templates,
        layers: vec![
            LayerSpec {
                id: "input".to_string(),
                r#type: LayerType::Input,
                size: 7, // 7 columns for 7x7 MNIST
                neuron: "default".to_string(),
            },
            LayerSpec {
                id: "hidden".to_string(),
                r#type: LayerType::Hidden,
                size: hidden_neurons,
                neuron: "default".to_string(),
            },
            LayerSpec {
                id: "output".to_string(),
                r#type: LayerType::Readout,
                size: 10, // 10 digit classes
                neuron: "default".to_string(),
            },
        ],
        connect: vec![
            ConnectSpec {
                from: "input".to_string(),
                to: "hidden".to_string(),
                rule: ConnectRule::Dense,
                synapse: SynapseSpec {
                    r#type: synapse_type.clone(),
                    resistance: DEFAULT_SYNAPSE_RESISTANCE,
                },
            },
            ConnectSpec {
                from: "hidden".to_string(),
                to: "output".to_string(),
                rule: ConnectRule::Dense,
                synapse: SynapseSpec {
                    r#type: synapse_type,
                    resistance: 22_000.0, // Slightly higher for output
                },
            },
        ],
        overrides: Vec::new(),
        external_inputs: Vec::new(),
        readouts: vec![ReadoutSpec {
            id: "logits".to_string(),
            r#type: ReadoutType::Analog,
            source_layer: Some("output".to_string()),
            signal: "analog_background".to_string(),
            post: None,
            tau_s: Some(2e-3),
        }],
        checkpoint: None,
        training: None,
    }
}

/// Build MNIST 7x7 rate-coded network.
///
/// Architecture: 49 (input) → hidden_neurons → 10 (output)
/// All 49 pixels are presented simultaneously (rate-coded input).
fn build_mnist7x7_rate_design(
    hidden_neurons: usize,
    excitatory_only: bool,
    params: &NeuronParams,
) -> Design {
    let mut neuron_templates = HashMap::new();
    neuron_templates.insert("default".to_string(), build_neuron_template(params));

    let synapse_type = SynapseType::Excitatory;
    let _ = excitatory_only;

    let mut globals = Globals::default();
    globals.values.insert("alpha_out".to_string(), 1.0.into());
    globals.values.insert("delta".to_string(), 5.0e-4.into());

    Design {
        version: Some("0.2".to_string()),
        units: None,
        globals,
        neuron_templates,
        layers: vec![
            LayerSpec {
                id: "input".to_string(),
                r#type: LayerType::Input,
                size: 49, // All 49 pixels of 7x7 MNIST
                neuron: "default".to_string(),
            },
            LayerSpec {
                id: "hidden".to_string(),
                r#type: LayerType::Hidden,
                size: hidden_neurons,
                neuron: "default".to_string(),
            },
            LayerSpec {
                id: "output".to_string(),
                r#type: LayerType::Readout,
                size: 10, // 10 digit classes
                neuron: "default".to_string(),
            },
        ],
        connect: vec![
            ConnectSpec {
                from: "input".to_string(),
                to: "hidden".to_string(),
                rule: ConnectRule::Dense,
                synapse: SynapseSpec {
                    r#type: synapse_type.clone(),
                    resistance: DEFAULT_SYNAPSE_RESISTANCE,
                },
            },
            ConnectSpec {
                from: "hidden".to_string(),
                to: "output".to_string(),
                rule: ConnectRule::Dense,
                synapse: SynapseSpec {
                    r#type: synapse_type,
                    resistance: 22_000.0,
                },
            },
        ],
        overrides: Vec::new(),
        external_inputs: Vec::new(),
        readouts: vec![ReadoutSpec {
            id: "logits".to_string(),
            r#type: ReadoutType::Analog,
            source_layer: Some("output".to_string()),
            signal: "analog_background".to_string(),
            post: None,
            tau_s: Some(2e-3),
        }],
        checkpoint: None,
        training: None,
    }
}

/// Build a generic feedforward network.
fn build_feedforward_design(
    input_size: usize,
    hidden_sizes: &[usize],
    output_size: usize,
    params: &NeuronParams,
) -> Design {
    let mut neuron_templates = HashMap::new();
    neuron_templates.insert("default".to_string(), build_neuron_template(params));

    let mut layers = Vec::new();
    let mut connections = Vec::new();

    // Input layer
    layers.push(LayerSpec {
        id: "input".to_string(),
        r#type: LayerType::Input,
        size: input_size,
        neuron: "default".to_string(),
    });

    let mut prev_layer = "input".to_string();

    // Hidden layers
    for (i, &size) in hidden_sizes.iter().enumerate() {
        let layer_id = format!("hidden{}", i);
        layers.push(LayerSpec {
            id: layer_id.clone(),
            r#type: LayerType::Hidden,
            size,
            neuron: "default".to_string(),
        });

        connections.push(ConnectSpec {
            from: prev_layer,
            to: layer_id.clone(),
            rule: ConnectRule::Dense,
            synapse: SynapseSpec {
                r#type: SynapseType::Excitatory,
                resistance: DEFAULT_SYNAPSE_RESISTANCE,
            },
        });

        prev_layer = layer_id;
    }

    // Output layer
    layers.push(LayerSpec {
        id: "output".to_string(),
        r#type: LayerType::Readout,
        size: output_size,
        neuron: "default".to_string(),
    });

    connections.push(ConnectSpec {
        from: prev_layer,
        to: "output".to_string(),
        rule: ConnectRule::Dense,
        synapse: SynapseSpec {
            r#type: SynapseType::Excitatory,
            resistance: DEFAULT_SYNAPSE_RESISTANCE,
        },
    });

    Design {
        version: Some("0.2".to_string()),
        units: None,
        globals: Globals::default(),
        neuron_templates,
        layers,
        connect: connections,
        overrides: Vec::new(),
        external_inputs: Vec::new(),
        readouts: vec![ReadoutSpec {
            id: "logits".to_string(),
            r#type: ReadoutType::Analog,
            source_layer: Some("output".to_string()),
            signal: "analog_background".to_string(),
            post: None,
            tau_s: Some(2e-3),
        }],
        checkpoint: None,
        training: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::compiler::compile_design;

    #[test]
    fn mnist7x7_preset_compiles() {
        let preset = ArchitecturePreset::Mnist7x7 {
            hidden_neurons: 10,
            excitatory_only: true,
        };
        let design = build_design_from_preset(&preset, None);
        let compiled = compile_design(&design).expect("should compile");

        // 7 input + 10 hidden + 10 output = 27 neurons
        assert_eq!(compiled.neuron_count(), 27);

        // 7*10 + 10*10 = 70 + 100 = 170 synapses
        assert_eq!(compiled.synapse_count(), 170);
    }

    #[test]
    fn mnist7x7_preset_has_correct_layers() {
        let preset = ArchitecturePreset::Mnist7x7 {
            hidden_neurons: 64,
            excitatory_only: true,
        };
        let design = build_design_from_preset(&preset, None);

        assert_eq!(design.layers.len(), 3);
        assert_eq!(design.layers[0].size, 7);  // input
        assert_eq!(design.layers[1].size, 64); // hidden
        assert_eq!(design.layers[2].size, 10); // output
    }

    #[test]
    fn feedforward_preset_compiles() {
        let preset = ArchitecturePreset::Feedforward {
            input_size: 10,
            hidden_sizes: vec![20, 15],
            output_size: 5,
        };
        let design = build_design_from_preset(&preset, None);
        let compiled = compile_design(&design).expect("should compile");

        // 10 + 20 + 15 + 5 = 50 neurons
        assert_eq!(compiled.neuron_count(), 50);

        // 10*20 + 20*15 + 15*5 = 200 + 300 + 75 = 575 synapses
        assert_eq!(compiled.synapse_count(), 575);
    }

    #[test]
    fn custom_preset_returns_minimal() {
        let preset = ArchitecturePreset::Custom;
        let design = build_design_from_preset(&preset, None);

        assert_eq!(design.layers.len(), 2);
    }

    #[test]
    fn custom_neuron_params() {
        let params = NeuronParams {
            theta0: 1.0,
            ..Default::default()
        };
        let preset = ArchitecturePreset::Mnist7x7 {
            hidden_neurons: 10,
            excitatory_only: true,
        };
        let design = build_design_from_preset(&preset, Some(params));

        let template = design.neuron_templates.get("default").unwrap();
        match &template.theta {
            ThetaTemplate::Constant { theta0 } => {
                assert!((*theta0 - 1.0).abs() < 1e-9);
            }
            _ => panic!("Expected constant theta"),
        }
    }
}

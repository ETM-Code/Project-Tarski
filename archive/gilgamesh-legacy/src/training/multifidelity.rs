//! Multi-fidelity training with adaptive fidelity scheduling.
//!
//! Provides training that automatically adjusts simulation fidelity
//! based on training progress, starting fast and increasing accuracy
//! as the model converges.

use crate::network::compiled::CompiledNetwork;
use crate::network::training::{train_network_with_callback, TrainerConfig, TrainingExample};
use crate::training::fidelity::{AdaptiveFidelityScheduler, FidelityLevel};
use crate::training::quantize::{apply_quantized_weights, quantize_network_weights, QuantizeConfig};
use anyhow::Result;

/// Configuration for multi-fidelity training.
#[derive(Debug, Clone)]
pub struct MultiFidelityConfig {
    /// Base trainer configuration
    pub trainer: TrainerConfig,

    /// Starting fidelity level
    pub starting_fidelity: FidelityLevel,

    /// Enable adaptive fidelity scheduling
    pub adaptive: bool,

    /// Minimum epochs per fidelity level before considering increase
    pub min_epochs_per_level: usize,

    /// Plateau detection window size
    pub plateau_window: usize,

    /// Minimum improvement threshold for plateau detection
    pub plateau_threshold: f64,

    /// Enable weight quantization during training
    pub quantize_weights: bool,

    /// Quantization configuration (if quantize_weights is true)
    pub quantize_config: Option<QuantizeConfig>,
}

impl Default for MultiFidelityConfig {
    fn default() -> Self {
        Self {
            trainer: TrainerConfig::default(),
            starting_fidelity: FidelityLevel::Turbo,
            adaptive: true,
            min_epochs_per_level: 5,
            plateau_window: 10,
            plateau_threshold: 0.001,
            quantize_weights: false,
            quantize_config: None,
        }
    }
}

/// Extended training log with fidelity information.
#[derive(Debug, Clone)]
pub struct MultiFidelityLog {
    /// Epoch number
    pub epoch: usize,
    /// Loss value
    pub loss: f64,
    /// Fidelity level used for this epoch
    pub fidelity: FidelityLevel,
    /// Whether fidelity increased after this epoch
    pub fidelity_increased: bool,
}

/// Train a network with multi-fidelity simulation.
///
/// Starts at a fast/coarse fidelity level and progressively increases
/// accuracy as training converges.
pub fn train_multifidelity(
    network: &mut CompiledNetwork,
    dataset: &[TrainingExample],
    config: &MultiFidelityConfig,
) -> Result<Vec<MultiFidelityLog>> {
    train_multifidelity_with_callback(network, dataset, config, |_, _, _| Ok(()))
}

/// Train with multi-fidelity and a per-epoch callback.
pub fn train_multifidelity_with_callback<F>(
    network: &mut CompiledNetwork,
    dataset: &[TrainingExample],
    config: &MultiFidelityConfig,
    mut on_epoch: F,
) -> Result<Vec<MultiFidelityLog>>
where
    F: FnMut(&CompiledNetwork, &MultiFidelityConfig, &MultiFidelityLog) -> Result<()>,
{
    let mut scheduler = AdaptiveFidelityScheduler::with_params(
        config.starting_fidelity,
        config.min_epochs_per_level,
        config.plateau_window,
        config.plateau_threshold,
    );

    let mut all_logs = Vec::with_capacity(config.trainer.epochs);
    let mut epoch_offset = 0;

    // Train at each fidelity level until we reach max epochs or Detailed
    while epoch_offset < config.trainer.epochs {
        let current_fidelity = scheduler.current_level();
        let fidelity_config = current_fidelity.config();

        // Create a modified trainer config with fidelity-appropriate dt
        let mut level_trainer = config.trainer.clone();

        // Calculate remaining epochs
        let remaining = config.trainer.epochs - epoch_offset;
        level_trainer.epochs = remaining;

        // Prepare dataset with fidelity-adjusted simulation options
        let adjusted_dataset: Vec<TrainingExample> = dataset
            .iter()
            .map(|ex| {
                let mut adjusted = ex.clone();
                // Apply fidelity dt if it's smaller than current (more precise)
                if fidelity_config.dt < adjusted.simulation.dt {
                    adjusted.simulation.dt = fidelity_config.dt;
                }
                adjusted
            })
            .collect();

        // Optionally quantize weights before training at this level
        if config.quantize_weights && fidelity_config.quantization {
            if let Some(ref qconfig) = config.quantize_config {
                let result = quantize_network_weights(network, qconfig);
                apply_quantized_weights(network, &result);
            }
        }

        // Train for this fidelity level
        let mut epochs_this_level = 0;
        let logs = train_network_with_callback(
            network,
            &adjusted_dataset,
            &level_trainer,
            |net, _cfg, log| {
                epochs_this_level += 1;
                let global_epoch = epoch_offset + log.epoch;

                // Check if we should increase fidelity
                let fidelity_increased = if config.adaptive {
                    scheduler.record_epoch(log.loss, None)
                } else {
                    false
                };

                let mf_log = MultiFidelityLog {
                    epoch: global_epoch,
                    loss: log.loss,
                    fidelity: current_fidelity,
                    fidelity_increased,
                };

                on_epoch(net, config, &mf_log)?;
                all_logs.push(mf_log);

                // If fidelity increased, we need to restart with new fidelity
                if fidelity_increased {
                    // Signal to break out of this training loop
                    return Err(anyhow::anyhow!("__fidelity_increase__"));
                }

                Ok(())
            },
        );

        match logs {
            Ok(_) => {
                // Training completed all remaining epochs
                break;
            }
            Err(e) if e.to_string() == "__fidelity_increase__" => {
                // Fidelity increased, continue with new level
                epoch_offset += epochs_this_level;
                continue;
            }
            Err(e) => return Err(e),
        }
    }

    Ok(all_logs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::compiler::compile_design;
    use crate::network::design::{
        ConnectRule, ConnectSpec, Design, Globals, LayerSpec, LayerType, NeuronTemplate,
        ReadoutSpec, ReadoutType, ResetTemplate, SynapseSpec, SynapseType, ThetaTemplate,
    };
    use crate::network::runtime::SimulationOptions;
    use std::collections::HashMap;

    fn test_design() -> Design {
        let mut neuron_templates = HashMap::new();
        neuron_templates.insert(
            "default".to_string(),
            NeuronTemplate {
                cm: 1e-9,
                r_leak: 1e6,
                theta: ThetaTemplate::Constant { theta0: 1.0 },
                reset: ResetTemplate {
                    r_reset: 1e9,
                    ron_switch: 1e9,
                },
            },
        );

        Design {
            version: Some("0.2".to_string()),
            units: None,
            globals: Globals::default(),
            neuron_templates,
            layers: vec![
                LayerSpec {
                    id: "input".to_string(),
                    r#type: LayerType::Input,
                    size: 2,
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
                    resistance: 2.5e3,
                },
            }],
            overrides: Vec::new(),
            external_inputs: Vec::new(),
            readouts: vec![ReadoutSpec {
                id: "out".to_string(),
                r#type: ReadoutType::Analog,
                source_layer: Some("output".to_string()),
                signal: "analog_background".to_string(),
                post: None,
                tau_s: Some(1e-3),
            }],
            checkpoint: None,
            training: None,
        }
    }

    #[test]
    fn multifidelity_training_runs() {
        let design = test_design();
        let mut network = compile_design(&design).unwrap();

        let mut targets = HashMap::new();
        targets.insert("out".to_string(), vec![0.5]);

        let mut sim_opts = SimulationOptions::default();
        sim_opts.dt = 1e-4;
        sim_opts.t_end = 5e-3;

        let example = TrainingExample::new(targets, sim_opts);

        let config = MultiFidelityConfig {
            trainer: TrainerConfig {
                learning_rate: 0.1,
                epochs: 5,
                ..TrainerConfig::default()
            },
            starting_fidelity: FidelityLevel::Turbo,
            adaptive: false, // Disable for predictable test
            ..MultiFidelityConfig::default()
        };

        let logs = train_multifidelity(&mut network, &[example], &config).unwrap();

        assert_eq!(logs.len(), 5);
        assert!(logs[0].loss >= logs[4].loss); // Loss should decrease
        assert_eq!(logs[0].fidelity, FidelityLevel::Turbo);
    }

    #[test]
    fn fidelity_config_applies_dt() {
        let turbo = FidelityLevel::Turbo.config();
        let detailed = FidelityLevel::Detailed.config();

        let mut opts = SimulationOptions::default();
        turbo.apply_to(&mut opts);
        assert!((opts.dt - 200e-6).abs() < 1e-9);

        detailed.apply_to(&mut opts);
        assert!((opts.dt - 5e-6).abs() < 1e-9);
    }
}

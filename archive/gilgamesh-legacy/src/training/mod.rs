//! Training algorithms for spiking neural networks.
//!
//! This module provides:
//! - Surrogate gradient functions for backpropagation through spikes
//! - Quantization-aware training (QAT) for hardware deployment
//! - Network weight quantization utilities
//! - Multi-fidelity simulation configuration
//! - TOML configuration parsing
//! - Multi-fidelity training with adaptive scheduling

pub mod config;
pub mod fidelity;
pub mod multifidelity;
pub mod noise;
pub mod presets;
pub mod qat;
pub mod quantize;
pub mod surrogate;

pub use config::{TrainingToml, TrainingParams, ArchitecturePreset, FidelityParams, SurrogateParams};
pub use fidelity::{FidelityLevel, FidelityConfig, ThresholdMode, AdaptiveFidelityScheduler};
pub use multifidelity::{train_multifidelity, MultiFidelityConfig, MultiFidelityLog};
pub use presets::{build_design_from_preset, NeuronParams};
pub use qat::QATWeight;
pub use quantize::{quantize_network_weights, apply_quantized_weights, QuantizeConfig, QuantizationResult};
pub use surrogate::{FastSigmoid, SurrogateGradient, SurrogateType};

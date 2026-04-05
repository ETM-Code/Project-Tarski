//! TOML configuration parser for training.
//!
//! Supports loading training configurations from TOML files with presets
//! for common architectures (e.g., MNIST 7x7 time-multiplexed).

use crate::training::fidelity::FidelityLevel;
use crate::training::surrogate::FastSigmoid;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Top-level training configuration from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingToml {
    /// Training parameters
    pub training: TrainingParams,

    /// Network architecture preset
    #[serde(default)]
    pub architecture: ArchitecturePreset,

    /// Multi-fidelity settings
    #[serde(default)]
    pub fidelity: FidelityParams,

    /// Surrogate gradient settings
    #[serde(default)]
    pub surrogate: SurrogateParams,
}

/// Training parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParams {
    /// Learning rate
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,

    /// Number of epochs
    #[serde(default = "default_epochs")]
    pub epochs: usize,

    /// L2 regularization strength
    #[serde(default)]
    pub regularization: f64,

    /// Batch size (None = full batch)
    #[serde(default)]
    pub batch_size: Option<usize>,

    /// Row spacing at start of training (seconds)
    #[serde(default = "default_row_spacing")]
    pub row_spacing_start: f64,

    /// Row spacing at end of training (seconds)
    #[serde(default = "default_row_spacing")]
    pub row_spacing_end: f64,

    /// Pulse width as fraction of row spacing (0-1)
    #[serde(default = "default_pulse_width")]
    pub pulse_width: f64,

    /// Input amplitude scale
    #[serde(default = "default_input_scale")]
    pub input_scale: f64,

    /// Maximum training samples (None = use all)
    #[serde(default)]
    pub sample_limit: Option<usize>,
}

fn default_learning_rate() -> f64 {
    1e-3
}

fn default_epochs() -> usize {
    25
}

fn default_row_spacing() -> f64 {
    1e-3
}

fn default_pulse_width() -> f64 {
    0.6
}

fn default_input_scale() -> f64 {
    1.0
}

impl Default for TrainingParams {
    fn default() -> Self {
        Self {
            learning_rate: default_learning_rate(),
            epochs: default_epochs(),
            regularization: 0.0,
            batch_size: None,
            row_spacing_start: default_row_spacing(),
            row_spacing_end: default_row_spacing(),
            pulse_width: default_pulse_width(),
            input_scale: default_input_scale(),
            sample_limit: None,
        }
    }
}

/// Network architecture preset.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ArchitecturePreset {
    /// Custom architecture (not a preset)
    Custom,

    /// MNIST 7x7 with time-multiplexed input (7 input neurons, row-by-row)
    Mnist7x7 {
        /// Number of hidden neurons
        #[serde(default = "default_hidden_neurons")]
        hidden_neurons: usize,

        /// Use excitatory-only connections (for hardware)
        #[serde(default)]
        excitatory_only: bool,
    },

    /// MNIST 7x7 with rate-coded input (49 input neurons, all at once)
    Mnist7x7Rate {
        /// Number of hidden neurons
        #[serde(default = "default_hidden_neurons")]
        hidden_neurons: usize,

        /// Use excitatory-only connections (for hardware)
        #[serde(default)]
        excitatory_only: bool,
    },

    /// Simple feedforward for testing
    Feedforward {
        /// Input size
        input_size: usize,
        /// Hidden layer sizes
        hidden_sizes: Vec<usize>,
        /// Output size
        output_size: usize,
    },
}

fn default_hidden_neurons() -> usize {
    64
}

impl Default for ArchitecturePreset {
    fn default() -> Self {
        ArchitecturePreset::Custom
    }
}

/// Multi-fidelity training parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FidelityParams {
    /// Starting fidelity level
    #[serde(default)]
    pub starting_level: FidelityLevel,

    /// Enable adaptive fidelity scheduling
    #[serde(default)]
    pub adaptive: bool,

    /// Minimum epochs before increasing fidelity
    #[serde(default = "default_min_epochs_per_level")]
    pub min_epochs_per_level: usize,

    /// Loss improvement threshold to detect plateau
    #[serde(default = "default_plateau_threshold")]
    pub plateau_threshold: f64,
}

fn default_min_epochs_per_level() -> usize {
    5
}

fn default_plateau_threshold() -> f64 {
    0.001
}

impl Default for FidelityParams {
    fn default() -> Self {
        Self {
            starting_level: FidelityLevel::Turbo,
            adaptive: true,
            min_epochs_per_level: default_min_epochs_per_level(),
            plateau_threshold: default_plateau_threshold(),
        }
    }
}

/// Surrogate gradient parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrogateParams {
    /// Enable surrogate gradients
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Surrogate function type
    #[serde(default)]
    pub function: SurrogateFunctionType,

    /// FastSigmoid slope parameter
    #[serde(default = "default_slope")]
    pub slope: f64,

    /// Triangular width parameter
    #[serde(default = "default_width")]
    pub width: f64,
}

fn default_enabled() -> bool {
    true
}

fn default_slope() -> f64 {
    25.0
}

fn default_width() -> f64 {
    1.0
}

/// Type of surrogate gradient function.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SurrogateFunctionType {
    #[default]
    FastSigmoid,
    Triangular,
}

impl Default for SurrogateParams {
    fn default() -> Self {
        Self {
            enabled: default_enabled(),
            function: SurrogateFunctionType::default(),
            slope: default_slope(),
            width: default_width(),
        }
    }
}

impl SurrogateParams {
    /// Convert to SurrogateType for use in training.
    pub fn to_surrogate_type(&self) -> crate::training::surrogate::SurrogateType {
        use crate::training::surrogate::{SurrogateType, Triangular};

        match self.function {
            SurrogateFunctionType::FastSigmoid => {
                SurrogateType::FastSigmoid(FastSigmoid::new(self.slope))
            }
            SurrogateFunctionType::Triangular => {
                SurrogateType::Triangular(Triangular::new(self.width))
            }
        }
    }
}

impl TrainingToml {
    /// Load configuration from a TOML file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| anyhow!("Failed to read config file: {}", e))?;
        Self::from_str(&content)
    }

    /// Parse configuration from a TOML string.
    pub fn from_str(content: &str) -> Result<Self> {
        toml::from_str(content).map_err(|e| anyhow!("Failed to parse TOML: {}", e))
    }

    /// Create a default MNIST 7x7 configuration.
    pub fn mnist_7x7_default() -> Self {
        Self {
            training: TrainingParams {
                learning_rate: 1e-3,
                epochs: 50,
                regularization: 1e-5,
                batch_size: Some(32),
                row_spacing_start: 1e-3,
                row_spacing_end: 5e-4,
                pulse_width: 0.6,
                input_scale: 1.0,
                sample_limit: None,
            },
            architecture: ArchitecturePreset::Mnist7x7 {
                hidden_neurons: 64,
                excitatory_only: true,
            },
            fidelity: FidelityParams {
                starting_level: FidelityLevel::Turbo,
                adaptive: true,
                min_epochs_per_level: 5,
                plateau_threshold: 0.001,
            },
            surrogate: SurrogateParams::default(),
        }
    }

    /// Generate example TOML content for documentation.
    pub fn example_toml() -> String {
        r#"# Training configuration for MNIST 7x7

[training]
learning_rate = 0.001
epochs = 50
regularization = 0.00001
batch_size = 32
row_spacing_start = 0.001
row_spacing_end = 0.0005
pulse_width = 0.6
input_scale = 1.0
# sample_limit = 1000  # Uncomment to limit training samples

[architecture]
type = "mnist7x7"
hidden_neurons = 64
excitatory_only = true

[fidelity]
starting_level = "turbo"
adaptive = true
min_epochs_per_level = 5
plateau_threshold = 0.001

[surrogate]
enabled = true
function = "fast_sigmoid"
slope = 25.0
"#
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_config() {
        let toml = r#"
            [training]
            epochs = 10
        "#;
        let config = TrainingToml::from_str(toml).unwrap();
        assert_eq!(config.training.epochs, 10);
        assert_eq!(config.training.learning_rate, default_learning_rate());
    }

    #[test]
    fn parse_full_config() {
        let toml = r#"
            [training]
            learning_rate = 0.01
            epochs = 100
            regularization = 0.0001
            batch_size = 64
            row_spacing_start = 0.002
            row_spacing_end = 0.001
            pulse_width = 0.5
            input_scale = 2.0
            sample_limit = 500

            [architecture]
            type = "mnist7x7"
            hidden_neurons = 128
            excitatory_only = false

            [fidelity]
            starting_level = "fast"
            adaptive = false
            min_epochs_per_level = 10
            plateau_threshold = 0.01

            [surrogate]
            enabled = true
            function = "triangular"
            width = 0.5
        "#;
        let config = TrainingToml::from_str(toml).unwrap();

        assert_eq!(config.training.learning_rate, 0.01);
        assert_eq!(config.training.epochs, 100);
        assert_eq!(config.training.batch_size, Some(64));
        assert_eq!(config.training.sample_limit, Some(500));

        match config.architecture {
            ArchitecturePreset::Mnist7x7 {
                hidden_neurons,
                excitatory_only,
            } => {
                assert_eq!(hidden_neurons, 128);
                assert!(!excitatory_only);
            }
            _ => panic!("Expected Mnist7x7 preset"),
        }

        assert_eq!(config.fidelity.starting_level, FidelityLevel::Fast);
        assert!(!config.fidelity.adaptive);

        assert!(config.surrogate.enabled);
        assert!(matches!(
            config.surrogate.function,
            SurrogateFunctionType::Triangular
        ));
    }

    #[test]
    fn parse_feedforward_preset() {
        let toml = r#"
            [training]
            epochs = 20

            [architecture]
            type = "feedforward"
            input_size = 10
            hidden_sizes = [32, 16]
            output_size = 5
        "#;
        let config = TrainingToml::from_str(toml).unwrap();

        match config.architecture {
            ArchitecturePreset::Feedforward {
                input_size,
                hidden_sizes,
                output_size,
            } => {
                assert_eq!(input_size, 10);
                assert_eq!(hidden_sizes, vec![32, 16]);
                assert_eq!(output_size, 5);
            }
            _ => panic!("Expected Feedforward preset"),
        }
    }

    #[test]
    fn surrogate_to_type() {
        let params = SurrogateParams {
            enabled: true,
            function: SurrogateFunctionType::FastSigmoid,
            slope: 30.0,
            width: 1.0,
        };
        let st = params.to_surrogate_type();
        match st {
            crate::training::surrogate::SurrogateType::FastSigmoid(fs) => {
                assert_eq!(fs.slope, 30.0);
            }
            _ => panic!("Expected FastSigmoid"),
        }
    }

    #[test]
    fn example_toml_parses() {
        let example = TrainingToml::example_toml();
        let config = TrainingToml::from_str(&example).unwrap();
        assert_eq!(config.training.epochs, 50);
    }

    #[test]
    fn mnist_default_config() {
        let config = TrainingToml::mnist_7x7_default();
        assert_eq!(config.training.epochs, 50);
        match config.architecture {
            ArchitecturePreset::Mnist7x7 { hidden_neurons, .. } => {
                assert_eq!(hidden_neurons, 64);
            }
            _ => panic!("Expected Mnist7x7"),
        }
    }
}

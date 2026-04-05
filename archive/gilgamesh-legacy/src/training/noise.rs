//! Runtime noise models for high-fidelity simulation.
//!
//! Models realistic noise sources in analog neuromorphic hardware:
//! - Membrane voltage noise (thermal/shot noise)
//! - Synaptic weight variation (memristor variability)
//! - Threshold noise (comparator offset)

use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for runtime noise injection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseConfig {
    /// Enable noise injection
    pub enabled: bool,

    /// Membrane voltage noise (standard deviation in volts)
    /// Typical: 1-10mV for analog circuits
    pub membrane_sigma: f64,

    /// Synaptic weight noise (relative standard deviation, 0-1)
    /// Represents cycle-to-cycle variation in memristors
    /// Typical: 0.01-0.05 (1-5%)
    pub synapse_sigma: f64,

    /// Threshold noise (standard deviation in volts)
    /// Comparator offset variation
    /// Typical: 5-20mV
    pub threshold_sigma: f64,

    /// Random seed for reproducibility (None = random)
    pub seed: Option<u64>,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            membrane_sigma: 5e-3,    // 5mV
            synapse_sigma: 0.02,     // 2% variation
            threshold_sigma: 10e-3,  // 10mV
            seed: None,
        }
    }
}

impl NoiseConfig {
    /// Create a noise config for DETAILED fidelity level.
    pub fn detailed() -> Self {
        Self {
            enabled: true,
            membrane_sigma: 5e-3,
            synapse_sigma: 0.02,
            threshold_sigma: 10e-3,
            seed: None,
        }
    }

    /// Create a low-noise config for testing.
    pub fn low_noise() -> Self {
        Self {
            enabled: true,
            membrane_sigma: 1e-3,
            synapse_sigma: 0.005,
            threshold_sigma: 2e-3,
            seed: None,
        }
    }

    /// Disabled noise (for TURBO/FAST/MEDIUM fidelity).
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }
}

/// Runtime noise generator for simulation.
#[derive(Debug)]
pub struct NoiseGenerator {
    config: NoiseConfig,
    rng: rand::rngs::StdRng,
}

impl NoiseGenerator {
    /// Create a new noise generator with the given config.
    pub fn new(config: NoiseConfig) -> Self {
        use rand::SeedableRng;
        let rng = match config.seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_os_rng(),
        };
        Self { config, rng }
    }

    /// Create a disabled noise generator.
    pub fn disabled() -> Self {
        Self::new(NoiseConfig::disabled())
    }

    /// Check if noise is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Generate membrane voltage noise.
    /// Returns noise value to add to membrane voltage.
    #[inline]
    pub fn membrane_noise(&mut self) -> f64 {
        if !self.config.enabled || self.config.membrane_sigma <= 0.0 {
            return 0.0;
        }
        self.gaussian(self.config.membrane_sigma)
    }

    /// Generate synaptic weight noise.
    /// Returns multiplicative factor (1.0 + noise) to apply to weight.
    #[inline]
    pub fn synapse_noise(&mut self) -> f64 {
        if !self.config.enabled || self.config.synapse_sigma <= 0.0 {
            return 1.0;
        }
        1.0 + self.gaussian(self.config.synapse_sigma)
    }

    /// Generate threshold noise.
    /// Returns noise value to add to threshold.
    #[inline]
    pub fn threshold_noise(&mut self) -> f64 {
        if !self.config.enabled || self.config.threshold_sigma <= 0.0 {
            return 0.0;
        }
        self.gaussian(self.config.threshold_sigma)
    }

    /// Generate Gaussian noise with given standard deviation.
    #[inline]
    fn gaussian(&mut self, sigma: f64) -> f64 {
        // Box-Muller transform for Gaussian
        let u1: f64 = self.rng.random_range(1e-10..1.0);
        let u2: f64 = self.rng.random();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        z * sigma
    }

    /// Reset the RNG with a new seed.
    pub fn reseed(&mut self, seed: u64) {
        use rand::SeedableRng;
        self.rng = rand::rngs::StdRng::seed_from_u64(seed);
    }
}

/// Per-neuron noise state for persistent noise effects.
#[derive(Debug, Clone)]
pub struct NeuronNoiseState {
    /// Fixed threshold offset (set once per neuron, represents device mismatch)
    pub threshold_offset: f64,
}

impl Default for NeuronNoiseState {
    fn default() -> Self {
        Self {
            threshold_offset: 0.0,
        }
    }
}

/// Initialize per-neuron noise state for a network.
pub fn init_neuron_noise(neuron_count: usize, generator: &mut NoiseGenerator) -> Vec<NeuronNoiseState> {
    (0..neuron_count)
        .map(|_| NeuronNoiseState {
            threshold_offset: generator.threshold_noise(),
        })
        .collect()
}

/// Per-synapse noise state for persistent weight variation.
#[derive(Debug, Clone)]
pub struct SynapseNoiseState {
    /// Fixed weight scaling factor (represents device-to-device variation)
    pub weight_scale: f64,
}

impl Default for SynapseNoiseState {
    fn default() -> Self {
        Self { weight_scale: 1.0 }
    }
}

/// Initialize per-synapse noise state for a network.
pub fn init_synapse_noise(synapse_count: usize, generator: &mut NoiseGenerator) -> Vec<SynapseNoiseState> {
    (0..synapse_count)
        .map(|_| SynapseNoiseState {
            weight_scale: generator.synapse_noise(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_noise_returns_zero() {
        let mut gen = NoiseGenerator::disabled();
        assert!(!gen.is_enabled());
        assert_eq!(gen.membrane_noise(), 0.0);
        assert_eq!(gen.synapse_noise(), 1.0);
        assert_eq!(gen.threshold_noise(), 0.0);
    }

    #[test]
    fn enabled_noise_returns_nonzero() {
        let config = NoiseConfig {
            enabled: true,
            membrane_sigma: 0.01,
            synapse_sigma: 0.05,
            threshold_sigma: 0.01,
            seed: Some(42),
        };
        let mut gen = NoiseGenerator::new(config);

        // With seed=42, we should get consistent values
        let m1 = gen.membrane_noise();
        let s1 = gen.synapse_noise();
        let t1 = gen.threshold_noise();

        // Reseed and check reproducibility
        gen.reseed(42);
        let m2 = gen.membrane_noise();
        let s2 = gen.synapse_noise();
        let t2 = gen.threshold_noise();

        assert!((m1 - m2).abs() < 1e-10);
        assert!((s1 - s2).abs() < 1e-10);
        assert!((t1 - t2).abs() < 1e-10);
    }

    #[test]
    fn noise_statistics() {
        let config = NoiseConfig {
            enabled: true,
            membrane_sigma: 1.0,
            synapse_sigma: 0.0,
            threshold_sigma: 0.0,
            seed: Some(123),
        };
        let mut gen = NoiseGenerator::new(config);

        // Generate many samples and check statistics
        let n = 10000;
        let samples: Vec<f64> = (0..n).map(|_| gen.membrane_noise()).collect();

        let mean = samples.iter().sum::<f64>() / n as f64;
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std = variance.sqrt();

        // Mean should be close to 0
        assert!(mean.abs() < 0.05, "Mean {} should be close to 0", mean);
        // Std should be close to 1.0 (the sigma we set)
        assert!((std - 1.0).abs() < 0.05, "Std {} should be close to 1.0", std);
    }

    #[test]
    fn init_neuron_noise_creates_offsets() {
        let mut gen = NoiseGenerator::new(NoiseConfig::detailed());
        let states = init_neuron_noise(100, &mut gen);

        assert_eq!(states.len(), 100);

        // Not all offsets should be zero (with high probability)
        let nonzero = states.iter().filter(|s| s.threshold_offset.abs() > 1e-10).count();
        assert!(nonzero > 90, "Most offsets should be nonzero");
    }

    #[test]
    fn init_synapse_noise_creates_scales() {
        let mut gen = NoiseGenerator::new(NoiseConfig::detailed());
        let states = init_synapse_noise(100, &mut gen);

        assert_eq!(states.len(), 100);

        // Scales should be close to 1.0 but not exactly 1.0
        let varied = states.iter().filter(|s| (s.weight_scale - 1.0).abs() > 1e-10).count();
        assert!(varied > 90, "Most scales should vary from 1.0");

        // Average should be close to 1.0
        let avg = states.iter().map(|s| s.weight_scale).sum::<f64>() / 100.0;
        assert!((avg - 1.0).abs() < 0.1, "Average scale should be close to 1.0");
    }

    #[test]
    fn detailed_config() {
        let config = NoiseConfig::detailed();
        assert!(config.enabled);
        assert!(config.membrane_sigma > 0.0);
        assert!(config.synapse_sigma > 0.0);
        assert!(config.threshold_sigma > 0.0);
    }

    #[test]
    fn low_noise_config() {
        let low = NoiseConfig::low_noise();
        let detailed = NoiseConfig::detailed();

        assert!(low.membrane_sigma < detailed.membrane_sigma);
        assert!(low.synapse_sigma < detailed.synapse_sigma);
        assert!(low.threshold_sigma < detailed.threshold_sigma);
    }
}

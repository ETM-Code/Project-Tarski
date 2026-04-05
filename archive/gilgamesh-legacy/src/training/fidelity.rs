//! Multi-fidelity training configuration.
//!
//! Supports progressive fidelity increase during training:
//! - TURBO: Fast, low-precision (dt=200µs, no hardware models)
//! - FAST: Moderate speed (dt=100µs, basic quantization)
//! - MEDIUM: Balanced (dt=20µs, full hardware models)
//! - DETAILED: High accuracy (dt=5µs, noise models)

use crate::network::runtime::SimulationOptions;
use serde::{Deserialize, Serialize};

/// Fidelity level for simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FidelityLevel {
    /// Fastest: dt=200µs, constant threshold, no quantization
    Turbo,
    /// Fast: dt=100µs, constant threshold, basic quantization
    Fast,
    /// Balanced: dt=20µs, adaptive threshold, full hardware models
    Medium,
    /// Highest accuracy: dt=5µs, adaptive threshold, noise models
    Detailed,
}

impl Default for FidelityLevel {
    fn default() -> Self {
        FidelityLevel::Turbo
    }
}

/// Configuration for a specific fidelity level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FidelityConfig {
    /// Simulation timestep (seconds)
    pub dt: f64,

    /// Threshold mode: "constant" or "adaptive"
    pub threshold_mode: ThresholdMode,

    /// Enable analog output stage modeling
    pub analog_output: bool,

    /// Enable weight quantization
    pub quantization: bool,

    /// Enable noise models
    pub noise: bool,
}

/// Threshold adaptation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThresholdMode {
    /// Fixed threshold (faster)
    Constant,
    /// Adaptive threshold dynamics (more realistic)
    Adaptive,
}

impl Default for ThresholdMode {
    fn default() -> Self {
        ThresholdMode::Constant
    }
}

impl FidelityConfig {
    /// Apply this fidelity configuration to simulation options.
    ///
    /// Updates dt and other simulation parameters based on fidelity level.
    pub fn apply_to(&self, opts: &mut SimulationOptions) {
        opts.dt = self.dt;
        // Note: analog_output, quantization, and noise are handled
        // separately in the training loop, not in SimulationOptions
    }

    /// Create new simulation options with this fidelity's dt.
    pub fn simulation_options(&self) -> SimulationOptions {
        let mut opts = SimulationOptions::default();
        self.apply_to(&mut opts);
        opts
    }
}

impl FidelityLevel {
    /// Get the default configuration for this fidelity level.
    pub fn config(&self) -> FidelityConfig {
        match self {
            FidelityLevel::Turbo => FidelityConfig {
                dt: 200e-6,  // 200µs
                threshold_mode: ThresholdMode::Constant,
                analog_output: false,
                quantization: false,
                noise: false,
            },
            FidelityLevel::Fast => FidelityConfig {
                dt: 100e-6,  // 100µs
                threshold_mode: ThresholdMode::Constant,
                analog_output: true,
                quantization: true,
                noise: false,
            },
            FidelityLevel::Medium => FidelityConfig {
                dt: 20e-6,   // 20µs
                threshold_mode: ThresholdMode::Adaptive,
                analog_output: true,
                quantization: true,
                noise: false,
            },
            FidelityLevel::Detailed => FidelityConfig {
                dt: 5e-6,    // 5µs
                threshold_mode: ThresholdMode::Adaptive,
                analog_output: true,
                quantization: true,
                noise: true,
            },
        }
    }

    /// Get the timestep in seconds.
    pub fn dt(&self) -> f64 {
        self.config().dt
    }

    /// Get the timestep in milliseconds.
    pub fn dt_ms(&self) -> f64 {
        self.config().dt * 1000.0
    }

    /// Approximate speedup factor relative to DETAILED.
    /// Larger dt = fewer timesteps = faster simulation.
    pub fn speedup(&self) -> f64 {
        let detailed_dt = FidelityLevel::Detailed.dt();
        self.dt() / detailed_dt
    }

    /// Get all levels in order from fastest to most accurate.
    pub fn all_levels() -> &'static [FidelityLevel] {
        &[
            FidelityLevel::Turbo,
            FidelityLevel::Fast,
            FidelityLevel::Medium,
            FidelityLevel::Detailed,
        ]
    }

    /// Get the next higher fidelity level, if any.
    pub fn next(&self) -> Option<FidelityLevel> {
        match self {
            FidelityLevel::Turbo => Some(FidelityLevel::Fast),
            FidelityLevel::Fast => Some(FidelityLevel::Medium),
            FidelityLevel::Medium => Some(FidelityLevel::Detailed),
            FidelityLevel::Detailed => None,
        }
    }

    /// Get the previous lower fidelity level, if any.
    pub fn prev(&self) -> Option<FidelityLevel> {
        match self {
            FidelityLevel::Turbo => None,
            FidelityLevel::Fast => Some(FidelityLevel::Turbo),
            FidelityLevel::Medium => Some(FidelityLevel::Fast),
            FidelityLevel::Detailed => Some(FidelityLevel::Medium),
        }
    }
}

/// Adaptive fidelity scheduler for progressive training.
#[derive(Debug, Clone)]
pub struct AdaptiveFidelityScheduler {
    /// Current fidelity level
    current_level: FidelityLevel,

    /// Epochs spent at current level
    epochs_at_level: usize,

    /// Minimum epochs before considering fidelity increase
    min_epochs_per_level: usize,

    /// Loss history for plateau detection
    loss_history: Vec<f64>,

    /// Window size for plateau detection
    plateau_window: usize,

    /// Minimum improvement to not be considered plateau (e.g., 0.001 = 0.1%)
    plateau_threshold: f64,

    /// Maximum allowed drift before forcing fidelity increase
    max_drift: f64,
}

impl Default for AdaptiveFidelityScheduler {
    fn default() -> Self {
        Self {
            current_level: FidelityLevel::Turbo,
            epochs_at_level: 0,
            min_epochs_per_level: 5,
            loss_history: Vec::new(),
            plateau_window: 10,
            plateau_threshold: 0.001,
            max_drift: 0.05,
        }
    }
}

impl AdaptiveFidelityScheduler {
    /// Create a new scheduler starting at the given level.
    pub fn new(starting_level: FidelityLevel) -> Self {
        Self {
            current_level: starting_level,
            ..Default::default()
        }
    }

    /// Create a scheduler with custom parameters.
    pub fn with_params(
        starting_level: FidelityLevel,
        min_epochs_per_level: usize,
        plateau_window: usize,
        plateau_threshold: f64,
    ) -> Self {
        Self {
            current_level: starting_level,
            min_epochs_per_level,
            plateau_window,
            plateau_threshold,
            ..Default::default()
        }
    }

    /// Get the current fidelity level.
    pub fn current_level(&self) -> FidelityLevel {
        self.current_level
    }

    /// Get the current fidelity configuration.
    pub fn current_config(&self) -> FidelityConfig {
        self.current_level.config()
    }

    /// Record a training epoch and determine if fidelity should change.
    ///
    /// Returns true if fidelity was increased.
    pub fn record_epoch(&mut self, loss: f64, validation_drift: Option<f64>) -> bool {
        self.epochs_at_level += 1;
        self.loss_history.push(loss);

        // Keep only recent history
        if self.loss_history.len() > self.plateau_window * 2 {
            self.loss_history.remove(0);
        }

        // Check if we should increase fidelity
        if self.should_increase_fidelity(validation_drift) {
            if let Some(next) = self.current_level.next() {
                self.current_level = next;
                self.epochs_at_level = 0;
                self.loss_history.clear();
                return true;
            }
        }

        false
    }

    /// Check if fidelity should be increased.
    fn should_increase_fidelity(&self, validation_drift: Option<f64>) -> bool {
        // Don't increase if we haven't spent enough time at this level
        if self.epochs_at_level < self.min_epochs_per_level {
            return false;
        }

        // Force increase if validation drift is too high
        if let Some(drift) = validation_drift {
            if drift >= self.max_drift {
                return true;
            }
        }

        // Check for loss plateau
        self.is_plateau()
    }

    /// Check if training has plateaued (loss not improving).
    fn is_plateau(&self) -> bool {
        if self.loss_history.len() < self.plateau_window {
            return false;
        }

        let recent = &self.loss_history[self.loss_history.len() - self.plateau_window..];
        let first_half: f64 = recent[..self.plateau_window / 2].iter().sum::<f64>()
            / (self.plateau_window / 2) as f64;
        let second_half: f64 = recent[self.plateau_window / 2..].iter().sum::<f64>()
            / (self.plateau_window - self.plateau_window / 2) as f64;

        if first_half <= 0.0 {
            return false;
        }

        let improvement = (first_half - second_half) / first_half;
        improvement < self.plateau_threshold
    }

    /// Force increase to next fidelity level.
    pub fn force_increase(&mut self) -> bool {
        if let Some(next) = self.current_level.next() {
            self.current_level = next;
            self.epochs_at_level = 0;
            self.loss_history.clear();
            true
        } else {
            false
        }
    }

    /// Reset to starting level.
    pub fn reset(&mut self, starting_level: FidelityLevel) {
        self.current_level = starting_level;
        self.epochs_at_level = 0;
        self.loss_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fidelity_levels_ordered() {
        let levels = FidelityLevel::all_levels();
        for i in 1..levels.len() {
            assert!(
                levels[i - 1].dt() > levels[i].dt(),
                "Higher fidelity should have smaller dt"
            );
        }
    }

    #[test]
    fn turbo_is_fastest() {
        let turbo = FidelityLevel::Turbo;
        let detailed = FidelityLevel::Detailed;

        assert!(turbo.dt() > detailed.dt());
        assert!(turbo.speedup() > 1.0);
    }

    #[test]
    fn speedup_factors() {
        // TURBO should be ~40x faster than DETAILED (200µs vs 5µs)
        let speedup = FidelityLevel::Turbo.speedup();
        assert!(
            (speedup - 40.0).abs() < 0.1,
            "Expected 40x speedup, got {}",
            speedup
        );
    }

    #[test]
    fn fidelity_progression() {
        let mut level = FidelityLevel::Turbo;

        level = level.next().unwrap();
        assert_eq!(level, FidelityLevel::Fast);

        level = level.next().unwrap();
        assert_eq!(level, FidelityLevel::Medium);

        level = level.next().unwrap();
        assert_eq!(level, FidelityLevel::Detailed);

        assert!(level.next().is_none());
    }

    #[test]
    fn scheduler_starts_at_turbo() {
        let scheduler = AdaptiveFidelityScheduler::default();
        assert_eq!(scheduler.current_level(), FidelityLevel::Turbo);
    }

    #[test]
    fn scheduler_respects_min_epochs() {
        let mut scheduler = AdaptiveFidelityScheduler::with_params(
            FidelityLevel::Turbo,
            5,  // min 5 epochs
            10,
            0.001,
        );

        // Record epochs with decreasing loss - shouldn't increase yet
        for i in 0..4 {
            let increased = scheduler.record_epoch(1.0 - i as f64 * 0.1, None);
            assert!(!increased, "Should not increase before min_epochs");
        }
    }

    #[test]
    fn scheduler_increases_on_plateau() {
        let mut scheduler = AdaptiveFidelityScheduler::with_params(
            FidelityLevel::Turbo,
            1,  // min 1 epoch
            4,  // small window
            0.01,
        );

        // Fill history with flat loss
        for _ in 0..10 {
            scheduler.record_epoch(0.5, None);
        }

        // Should have increased due to plateau
        assert_ne!(scheduler.current_level(), FidelityLevel::Turbo);
    }

    #[test]
    fn scheduler_increases_on_drift() {
        let mut scheduler = AdaptiveFidelityScheduler::with_params(
            FidelityLevel::Turbo,
            1,
            10,
            0.001,
        );
        scheduler.max_drift = 0.05;

        // Record epoch with high drift
        let increased = scheduler.record_epoch(0.5, Some(0.1));
        assert!(increased, "Should increase on high drift");
    }

    #[test]
    fn force_increase() {
        let mut scheduler = AdaptiveFidelityScheduler::new(FidelityLevel::Turbo);

        assert!(scheduler.force_increase());
        assert_eq!(scheduler.current_level(), FidelityLevel::Fast);

        assert!(scheduler.force_increase());
        assert_eq!(scheduler.current_level(), FidelityLevel::Medium);

        assert!(scheduler.force_increase());
        assert_eq!(scheduler.current_level(), FidelityLevel::Detailed);

        // Can't go higher
        assert!(!scheduler.force_increase());
    }

    #[test]
    fn dt_values_match_plan() {
        // Verify dt values match the plan
        assert!((FidelityLevel::Turbo.dt() - 200e-6).abs() < 1e-9);
        assert!((FidelityLevel::Fast.dt() - 100e-6).abs() < 1e-9);
        assert!((FidelityLevel::Medium.dt() - 20e-6).abs() < 1e-9);
        assert!((FidelityLevel::Detailed.dt() - 5e-6).abs() < 1e-9);
    }
}

//! Network weight quantization utilities.
//!
//! Provides functions to quantize trained weights for hardware deployment.

use crate::hardware::MCP4661;
use crate::network::compiled::CompiledNetwork;

/// Configuration for weight quantization.
#[derive(Debug, Clone)]
pub struct QuantizeConfig {
    /// MCP4661 digipot model to use for quantization
    pub digipot: MCP4661,

    /// Weight range for mapping [min, max] -> [0, max_conductance]
    pub weight_min: f64,
    pub weight_max: f64,

    /// Whether to use signed weights (map to [-1, 1] with zero at center code)
    pub signed: bool,
}

impl Default for QuantizeConfig {
    fn default() -> Self {
        Self {
            digipot: MCP4661::mcp4661_104(),
            weight_min: 0.0,
            weight_max: 1.0,
            signed: false,
        }
    }
}

impl QuantizeConfig {
    /// Create config for unsigned weights [0, 1].
    pub fn unsigned() -> Self {
        Self::default()
    }

    /// Create config for signed weights [-1, 1].
    pub fn signed() -> Self {
        Self {
            signed: true,
            weight_min: -1.0,
            weight_max: 1.0,
            ..Default::default()
        }
    }

    /// Quantize a single weight value to a digipot code.
    pub fn weight_to_code(&self, weight: f64) -> u16 {
        let range = self.weight_max - self.weight_min;
        if range <= 0.0 {
            return 0;
        }

        // Normalize to [0, 1]
        let normalized = (weight - self.weight_min) / range;
        let clamped = normalized.clamp(0.0, 1.0);

        // Map to digipot steps
        let max_step = (self.digipot.steps - 1) as f64;
        (clamped * max_step).round() as u16
    }

    /// Convert a digipot code back to weight value.
    pub fn code_to_weight(&self, code: u16) -> f64 {
        let max_step = (self.digipot.steps - 1) as f64;
        let normalized = code as f64 / max_step;
        let range = self.weight_max - self.weight_min;
        self.weight_min + normalized * range
    }

    /// Get the weight step size (quantization resolution).
    pub fn weight_step_size(&self) -> f64 {
        let range = self.weight_max - self.weight_min;
        range / (self.digipot.steps - 1) as f64
    }

    /// Quantize a weight to the nearest representable value.
    pub fn quantize_weight(&self, weight: f64) -> f64 {
        let code = self.weight_to_code(weight);
        self.code_to_weight(code)
    }

    /// Get the conductance for a given digipot code.
    pub fn code_to_conductance(&self, code: u16) -> f64 {
        self.digipot.conductance(code)
    }

    /// Get the resistance for a given digipot code.
    pub fn code_to_resistance(&self, code: u16) -> f64 {
        self.digipot.rwb(code)
    }
}

/// Result of quantizing a network.
#[derive(Debug, Clone)]
pub struct QuantizationResult {
    /// Original weights
    pub original_weights: Vec<f64>,

    /// Quantized weights
    pub quantized_weights: Vec<f64>,

    /// Digipot codes (0-256)
    pub codes: Vec<u16>,

    /// Quantization error statistics
    pub stats: QuantizationStats,
}

/// Statistics about quantization error.
#[derive(Debug, Clone, Default)]
pub struct QuantizationStats {
    /// Mean absolute quantization error
    pub mean_abs_error: f64,

    /// Maximum absolute quantization error
    pub max_abs_error: f64,

    /// RMS quantization error
    pub rms_error: f64,

    /// Number of weights that saturated (hit min/max)
    pub saturated_count: usize,
}

/// Quantize all weights in a network.
///
/// Returns the quantized weights and statistics without modifying the network.
pub fn quantize_network_weights(
    network: &CompiledNetwork,
    config: &QuantizeConfig,
) -> QuantizationResult {
    let original = network.incoming.g.clone();
    let mut quantized = Vec::with_capacity(original.len());
    let mut codes = Vec::with_capacity(original.len());
    let mut saturated_count = 0;

    for &w in &original {
        let code = config.weight_to_code(w);
        let q_weight = config.code_to_weight(code);

        // Check for saturation
        if code == 0 || code == config.digipot.steps - 1 {
            let normalized = (w - config.weight_min) / (config.weight_max - config.weight_min);
            if normalized < 0.0 || normalized > 1.0 {
                saturated_count += 1;
            }
        }

        quantized.push(q_weight);
        codes.push(code);
    }

    // Compute statistics
    let mut sum_abs_error: f64 = 0.0;
    let mut max_abs_error: f64 = 0.0;
    let mut sum_sq_error: f64 = 0.0;

    for (i, &orig) in original.iter().enumerate() {
        let error = (orig - quantized[i]).abs();
        sum_abs_error += error;
        max_abs_error = max_abs_error.max(error);
        sum_sq_error += error * error;
    }

    let n = original.len() as f64;
    let stats = QuantizationStats {
        mean_abs_error: if n > 0.0 { sum_abs_error / n } else { 0.0 },
        max_abs_error,
        rms_error: if n > 0.0 { (sum_sq_error / n).sqrt() } else { 0.0 },
        saturated_count,
    };

    QuantizationResult {
        original_weights: original,
        quantized_weights: quantized,
        codes,
        stats,
    }
}

/// Apply quantized weights back to a network (mutates in place).
pub fn apply_quantized_weights(network: &mut CompiledNetwork, result: &QuantizationResult) {
    if result.quantized_weights.len() == network.incoming.g.len() {
        network.incoming.g = result.quantized_weights.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_to_code_boundaries() {
        let config = QuantizeConfig::unsigned();

        // Minimum weight -> code 0
        assert_eq!(config.weight_to_code(0.0), 0);

        // Maximum weight -> code 256
        assert_eq!(config.weight_to_code(1.0), 256);

        // Middle -> code 128
        assert_eq!(config.weight_to_code(0.5), 128);
    }

    #[test]
    fn weight_to_code_round_trip() {
        let config = QuantizeConfig::unsigned();

        for code in [0, 1, 64, 128, 192, 255, 256] {
            let weight = config.code_to_weight(code);
            let recovered = config.weight_to_code(weight);
            assert_eq!(code, recovered, "Round-trip failed for code {}", code);
        }
    }

    #[test]
    fn signed_weights() {
        let config = QuantizeConfig::signed();

        // -1 -> code 0
        assert_eq!(config.weight_to_code(-1.0), 0);

        // 0 -> code 128
        assert_eq!(config.weight_to_code(0.0), 128);

        // +1 -> code 256
        assert_eq!(config.weight_to_code(1.0), 256);
    }

    #[test]
    fn quantize_clamps_out_of_range() {
        let config = QuantizeConfig::unsigned();

        // Below min -> code 0
        assert_eq!(config.weight_to_code(-0.5), 0);

        // Above max -> code 256
        assert_eq!(config.weight_to_code(1.5), 256);
    }

    #[test]
    fn quantization_step_size() {
        let config = QuantizeConfig::unsigned();
        let step = config.weight_step_size();

        // For [0, 1] with 257 levels: step = 1/256
        let expected = 1.0 / 256.0;
        assert!(
            (step - expected).abs() < 1e-10,
            "Expected step {}, got {}",
            expected,
            step
        );
    }

    #[test]
    fn quantize_weight_discrete() {
        let config = QuantizeConfig::unsigned();

        // Any weight should quantize to a discrete level
        let w = 0.333;
        let q = config.quantize_weight(w);

        // Check it's a multiple of step size
        let step = config.weight_step_size();
        let steps_from_zero = q / step;
        let rounded = steps_from_zero.round();

        assert!(
            (steps_from_zero - rounded).abs() < 1e-9,
            "Quantized {} to {} which is not discrete",
            w,
            q
        );
    }
}

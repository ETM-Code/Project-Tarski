//! Quantization-Aware Training (QAT) for hardware deployment.
//!
//! Trains with quantization from the start using the Straight-Through Estimator (STE).
//! This ensures trained weights map directly to digipot settings without accuracy loss.

/// A weight with quantization-aware training support.
///
/// Maintains a full-precision float value for gradient accumulation,
/// but provides a quantized value for forward passes.
#[derive(Debug, Clone, Copy)]
pub struct QATWeight {
    /// Full-precision weight for gradient accumulation
    float_value: f64,
    /// Number of quantization levels (e.g., 257 for 8-bit MCP4661)
    levels: u16,
    /// Minimum weight value
    min_val: f64,
    /// Maximum weight value
    max_val: f64,
}

impl QATWeight {
    /// Create a new QAT weight.
    ///
    /// # Arguments
    /// * `initial` - Initial weight value
    /// * `levels` - Number of quantization levels (e.g., 257 for 8-bit)
    /// * `min_val` - Minimum weight value
    /// * `max_val` - Maximum weight value
    pub fn new(initial: f64, levels: u16, min_val: f64, max_val: f64) -> Self {
        let clamped = initial.clamp(min_val, max_val);
        Self {
            float_value: clamped,
            levels,
            min_val,
            max_val,
        }
    }

    /// Create a QAT weight for MCP4661 digipot (257 levels, [0, 1] range).
    pub fn for_mcp4661(initial: f64) -> Self {
        Self::new(initial, 257, 0.0, 1.0)
    }

    /// Create a QAT weight for signed weights (257 levels, [-1, 1] range).
    pub fn for_mcp4661_signed(initial: f64) -> Self {
        Self::new(initial, 257, -1.0, 1.0)
    }

    /// Get the full-precision float value.
    pub fn float_value(&self) -> f64 {
        self.float_value
    }

    /// Get the quantized value (for forward pass).
    ///
    /// Uses round-to-nearest quantization.
    pub fn quantized(&self) -> f64 {
        if self.levels <= 1 {
            return self.min_val;
        }
        let range = self.max_val - self.min_val;
        if range <= 0.0 {
            return self.min_val;
        }

        // Normalize to [0, 1]
        let normalized = (self.float_value - self.min_val) / range;

        // Quantize to discrete steps
        let step = (normalized * (self.levels - 1) as f64).round() / (self.levels - 1) as f64;

        // Map back to original range
        self.min_val + step * range
    }

    /// Apply gradient update using Straight-Through Estimator (STE).
    ///
    /// The gradient flows through as if quantization didn't exist,
    /// but the weight is still quantized in the forward pass.
    pub fn apply_gradient(&mut self, grad: f64, lr: f64) {
        self.float_value -= lr * grad;
        self.float_value = self.float_value.clamp(self.min_val, self.max_val);
    }

    /// Convert to digipot step code (0 to levels-1).
    ///
    /// For MCP4661: returns 0-256 (u16 to handle 257 steps).
    pub fn to_digipot_step(&self) -> u16 {
        if self.levels <= 1 {
            return 0;
        }
        let range = self.max_val - self.min_val;
        if range <= 0.0 {
            return 0;
        }

        let normalized = (self.quantized() - self.min_val) / range;
        let max_step = (self.levels - 1) as f64;
        (normalized * max_step).round().clamp(0.0, max_step) as u16
    }

    /// Get the quantization step size.
    pub fn step_size(&self) -> f64 {
        if self.levels <= 1 {
            return 0.0;
        }
        (self.max_val - self.min_val) / (self.levels - 1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qat_quantizes_to_nearest() {
        let w = QATWeight::new(0.51, 257, 0.0, 1.0);
        let step_size: f64 = 1.0 / 256.0;
        let expected = (0.51_f64 / step_size).round() * step_size;
        let quantized = w.quantized();
        assert!(
            (quantized - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            quantized
        );
    }

    #[test]
    fn qat_at_boundaries() {
        // At minimum
        let w_min = QATWeight::new(0.0, 257, 0.0, 1.0);
        assert!((w_min.quantized() - 0.0).abs() < 1e-10);
        assert_eq!(w_min.to_digipot_step(), 0);

        // At maximum
        let w_max = QATWeight::new(1.0, 257, 0.0, 1.0);
        assert!((w_max.quantized() - 1.0).abs() < 1e-10);
        assert_eq!(w_max.to_digipot_step(), 256);
    }

    #[test]
    fn qat_gradient_updates_float() {
        let mut w = QATWeight::new(0.5, 257, 0.0, 1.0);
        let initial = w.float_value();

        w.apply_gradient(1.0, 0.1); // grad=1, lr=0.1 -> decrease by 0.1
        assert!((w.float_value() - (initial - 0.1)).abs() < 1e-10);
    }

    #[test]
    fn qat_clamps_on_update() {
        let mut w = QATWeight::new(0.1, 257, 0.0, 1.0);

        // Apply large positive gradient (should clamp at 0)
        w.apply_gradient(100.0, 0.1);
        assert!((w.float_value() - 0.0).abs() < 1e-10, "Should clamp at min");

        // Reset and apply large negative gradient (should clamp at 1)
        let mut w2 = QATWeight::new(0.9, 257, 0.0, 1.0);
        w2.apply_gradient(-100.0, 0.1);
        assert!((w2.float_value() - 1.0).abs() < 1e-10, "Should clamp at max");
    }

    #[test]
    fn qat_digipot_step_covers_full_range() {
        // Step 0
        let w0 = QATWeight::new(0.0, 257, 0.0, 1.0);
        assert_eq!(w0.to_digipot_step(), 0);

        // Middle step (128)
        let w_mid = QATWeight::new(0.5, 257, 0.0, 1.0);
        assert_eq!(w_mid.to_digipot_step(), 128);

        // Maximum step (256)
        let w_max = QATWeight::new(1.0, 257, 0.0, 1.0);
        assert_eq!(w_max.to_digipot_step(), 256);
    }

    #[test]
    fn qat_signed_range() {
        let w = QATWeight::for_mcp4661_signed(0.0);
        assert!((w.quantized() - 0.0).abs() < 1e-10);
        assert_eq!(w.to_digipot_step(), 128); // Middle of 0-256 range

        let w_neg = QATWeight::for_mcp4661_signed(-1.0);
        assert_eq!(w_neg.to_digipot_step(), 0);

        let w_pos = QATWeight::for_mcp4661_signed(1.0);
        assert_eq!(w_pos.to_digipot_step(), 256);
    }

    #[test]
    fn qat_step_size() {
        let w = QATWeight::new(0.5, 257, 0.0, 1.0);
        let expected = 1.0 / 256.0;
        assert!((w.step_size() - expected).abs() < 1e-10);
    }

    #[test]
    fn qat_quantized_values_are_discrete() {
        // Any weight value should quantize to one of 257 discrete levels
        for i in 0..1000 {
            let val = i as f64 / 999.0; // 0.0 to 1.0
            let w = QATWeight::new(val, 257, 0.0, 1.0);
            let quantized = w.quantized();

            // Check it's a multiple of step_size from min
            let steps_from_min = (quantized - 0.0) / w.step_size();
            let rounded_steps = steps_from_min.round();
            assert!(
                (steps_from_min - rounded_steps).abs() < 1e-9,
                "Value {} quantized to {} which is not a discrete step",
                val,
                quantized
            );
        }
    }
}

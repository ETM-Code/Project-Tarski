//! Inverting op-amp output stage model.
//!
//! Models the analog output stage that converts membrane voltage to
//! an observable output signal, including:
//! - Inverting amplifier topology
//! - Rail clamping
//! - DC offset

/// Inverting op-amp output stage.
///
/// Transfer function: Vout = Vref - (R2/R1) × (Vmem - Vref)
///
/// At Vmem = Vref: Vout = Vref (centered at reference)
#[derive(Debug, Clone)]
pub struct InvertingOpAmp {
    /// Input resistor (Ohms)
    pub r1: f64,

    /// Feedback resistor (Ohms)
    pub r2: f64,

    /// Reference voltage (V) - connected to non-inverting input
    pub v_ref: f64,

    /// Supply voltage (V)
    pub v_dd: f64,

    /// DC offset (mV) - models op-amp input offset voltage
    pub v_offset_mv: f64,

    /// Whether to clamp output to rails [0, VDD]
    pub clamp_to_rails: bool,
}

impl InvertingOpAmp {
    /// Create a new inverting op-amp with the given parameters.
    pub fn new(r1: f64, r2: f64, v_ref: f64, v_dd: f64) -> Self {
        Self {
            r1,
            r2,
            v_ref,
            v_dd,
            v_offset_mv: 0.0,
            clamp_to_rails: true,
        }
    }

    /// Create with default values from SPICE model.
    ///
    /// R1 = R2 = 10kΩ (unity gain), Vref = 2.5V, VDD = 5V
    pub fn default_unity() -> Self {
        Self::new(10_000.0, 10_000.0, 2.5, 5.0)
    }

    /// Create with gain = 4 (R2 = 40kΩ, R1 = 10kΩ).
    pub fn default_gain4() -> Self {
        Self::new(10_000.0, 40_000.0, 2.5, 5.0)
    }

    /// Set the DC offset (mV).
    pub fn with_offset(mut self, offset_mv: f64) -> Self {
        self.v_offset_mv = offset_mv;
        self
    }

    /// Disable rail clamping (for analysis).
    pub fn without_clamping(mut self) -> Self {
        self.clamp_to_rails = false;
        self
    }

    /// Compute the gain magnitude |R2/R1|.
    pub fn gain(&self) -> f64 {
        self.r2 / self.r1
    }

    /// Compute the output voltage for a given membrane voltage.
    ///
    /// Vout = Vref - (R2/R1) × (Vmem - Vref + offset)
    pub fn compute(&self, v_mem: f64) -> f64 {
        let v_in = v_mem - self.v_ref + self.v_offset_mv * 1e-3;
        let v_out = self.v_ref - self.gain() * v_in;

        if self.clamp_to_rails {
            v_out.clamp(0.0, self.v_dd)
        } else {
            v_out
        }
    }

    /// Compute the inverse: what Vmem produces a given Vout?
    ///
    /// Useful for understanding the output range.
    pub fn inverse(&self, v_out: f64) -> f64 {
        // Vout = Vref - gain × (Vmem - Vref)
        // Vref - Vout = gain × (Vmem - Vref)
        // (Vref - Vout) / gain = Vmem - Vref
        // Vmem = Vref + (Vref - Vout) / gain
        self.v_ref + (self.v_ref - v_out) / self.gain()
    }

    /// Get the valid Vmem range that produces non-clipped output.
    ///
    /// Returns (v_mem_min, v_mem_max) where output is in [0, VDD].
    pub fn valid_input_range(&self) -> (f64, f64) {
        let v_mem_for_vout_max = self.inverse(self.v_dd); // Vout = VDD
        let v_mem_for_vout_min = self.inverse(0.0);       // Vout = 0

        if v_mem_for_vout_max < v_mem_for_vout_min {
            (v_mem_for_vout_max, v_mem_for_vout_min)
        } else {
            (v_mem_for_vout_min, v_mem_for_vout_max)
        }
    }
}

impl Default for InvertingOpAmp {
    fn default() -> Self {
        Self::default_unity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unity_gain_at_vref() {
        let amp = InvertingOpAmp::default_unity();

        // At Vmem = Vref, output should be Vref
        let v_out = amp.compute(amp.v_ref);
        assert!(
            (v_out - amp.v_ref).abs() < 1e-10,
            "At Vmem=Vref, Vout should be Vref. Got {}",
            v_out
        );
    }

    #[test]
    fn unity_gain_inverts() {
        let amp = InvertingOpAmp::default_unity();
        let v_ref = amp.v_ref;

        // Above Vref -> output below Vref
        let v_out_above = amp.compute(v_ref + 0.5);
        assert!(
            v_out_above < v_ref,
            "Above Vref should produce output below Vref"
        );

        // Below Vref -> output above Vref
        let v_out_below = amp.compute(v_ref - 0.5);
        assert!(
            v_out_below > v_ref,
            "Below Vref should produce output above Vref"
        );
    }

    #[test]
    fn unity_gain_magnitude() {
        let amp = InvertingOpAmp::default_unity().without_clamping();
        let v_ref = amp.v_ref;

        let delta = 0.5;
        let v_out = amp.compute(v_ref + delta);

        // With unity gain: Vout = Vref - 1 × delta
        let expected = v_ref - delta;
        assert!(
            (v_out - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            v_out
        );
    }

    #[test]
    fn gain_4_amplifies() {
        let amp = InvertingOpAmp::default_gain4().without_clamping();
        let v_ref = amp.v_ref;

        let delta = 0.2;
        let v_out = amp.compute(v_ref + delta);

        // With gain 4: Vout = Vref - 4 × delta
        let expected = v_ref - 4.0 * delta;
        assert!(
            (v_out - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            v_out
        );
    }

    #[test]
    fn rail_clamping() {
        let amp = InvertingOpAmp::default_unity();

        // Large positive Vmem should clamp to 0V
        let v_out_high = amp.compute(10.0);
        assert!(
            (v_out_high - 0.0).abs() < 1e-10,
            "Should clamp to 0V, got {}",
            v_out_high
        );

        // Large negative Vmem should clamp to VDD
        let v_out_low = amp.compute(-10.0);
        assert!(
            (v_out_low - amp.v_dd).abs() < 1e-10,
            "Should clamp to VDD, got {}",
            v_out_low
        );
    }

    #[test]
    fn offset_shifts_output() {
        let amp_no_offset = InvertingOpAmp::default_unity();
        let amp_with_offset = InvertingOpAmp::default_unity().with_offset(10.0); // 10mV offset

        let v_mem = 2.0;
        let v_out_no = amp_no_offset.compute(v_mem);
        let v_out_yes = amp_with_offset.compute(v_mem);

        // 10mV offset with unity gain should shift output by ~10mV
        let expected_shift = 10.0e-3; // 10mV in volts
        let actual_shift = (v_out_no - v_out_yes).abs();

        assert!(
            (actual_shift - expected_shift).abs() < 1e-6,
            "Expected shift of {} V, got {} V",
            expected_shift,
            actual_shift
        );
    }

    #[test]
    fn inverse_round_trip() {
        let amp = InvertingOpAmp::default_unity().without_clamping();

        for v_mem in [1.0, 2.0, 2.5, 3.0, 4.0] {
            let v_out = amp.compute(v_mem);
            let v_mem_recovered = amp.inverse(v_out);
            assert!(
                (v_mem - v_mem_recovered).abs() < 1e-10,
                "Round-trip failed for Vmem={}: got {}",
                v_mem,
                v_mem_recovered
            );
        }
    }

    #[test]
    fn valid_input_range_unity() {
        let amp = InvertingOpAmp::default_unity();
        let (v_min, v_max) = amp.valid_input_range();

        // For unity gain with Vref=2.5V, VDD=5V:
        // Vout=0 when Vmem=5V, Vout=5V when Vmem=0V
        assert!(
            (v_min - 0.0).abs() < 1e-10,
            "Min valid input should be ~0V, got {}",
            v_min
        );
        assert!(
            (v_max - 5.0).abs() < 1e-10,
            "Max valid input should be ~5V, got {}",
            v_max
        );
    }
}

//! MCP4661 Digital Potentiometer model.
//!
//! The MCP4661 is a dual 8-bit I²C digital potentiometer with:
//! - 257 tap points (0x00 to 0x100)
//! - RAB options: 5kΩ, 10kΩ, 50kΩ, 100kΩ (±20% tolerance)
//! - Wiper resistance: 75Ω typical, 160Ω max
//! - Temperature coefficient: 50 ppm/°C (rheostat mode)

use rand::Rng;

/// MCP4661 digital potentiometer model.
#[derive(Debug, Clone)]
pub struct MCP4661 {
    /// Nominal RAB resistance (Ohms)
    pub r_ab: f64,

    /// Wiper resistance (Ohms) - 75Ω typical
    pub r_wiper: f64,

    /// Number of steps (257 for 8-bit)
    pub steps: u16,

    /// Temperature coefficient (ppm/°C)
    pub tempco_ppm: f64,

    /// RAB tolerance (e.g., 0.20 for ±20%)
    pub tolerance: f64,

    /// Actual RAB after tolerance applied (for this instance)
    pub r_ab_actual: f64,
}

impl MCP4661 {
    /// Create a new MCP4661 with specified RAB value.
    pub fn new(r_ab: f64) -> Self {
        Self {
            r_ab,
            r_wiper: 75.0,
            steps: 257, // 8-bit = 256 segments = 257 tap points
            tempco_ppm: 50.0,
            tolerance: 0.20,
            r_ab_actual: r_ab, // No tolerance applied by default
        }
    }

    /// Create the 100kΩ variant (MCP4661-104).
    pub fn mcp4661_104() -> Self {
        Self::new(100_000.0)
    }

    /// Create the 50kΩ variant (MCP4661-503).
    pub fn mcp4661_503() -> Self {
        Self::new(50_000.0)
    }

    /// Create the 10kΩ variant (MCP4661-103).
    pub fn mcp4661_103() -> Self {
        Self::new(10_000.0)
    }

    /// Create the 5kΩ variant (MCP4661-502).
    pub fn mcp4661_502() -> Self {
        Self::new(5_000.0)
    }

    /// Apply random tolerance variation based on the device tolerance spec.
    ///
    /// Call this once during instantiation to model part-to-part variation.
    pub fn with_random_tolerance(mut self, rng: &mut impl Rng) -> Self {
        let variation = rng.random_range(-self.tolerance..self.tolerance);
        self.r_ab_actual = self.r_ab * (1.0 + variation);
        self
    }

    /// Apply a specific tolerance offset (e.g., -0.1 for -10%).
    pub fn with_tolerance_offset(mut self, offset: f64) -> Self {
        self.r_ab_actual = self.r_ab * (1.0 + offset.clamp(-self.tolerance, self.tolerance));
        self
    }

    /// Step resistance Rs = RAB / 256.
    pub fn step_resistance(&self) -> f64 {
        self.r_ab_actual / (self.steps - 1) as f64
    }

    /// Wiper-to-B resistance for a given code (0 to 256).
    ///
    /// RWB(n) = Rs × n + Rw
    pub fn rwb(&self, code: u16) -> f64 {
        let code = code.min(self.steps - 1);
        let rs = self.step_resistance();
        rs * code as f64 + self.r_wiper
    }

    /// Wiper-to-A resistance for a given code.
    ///
    /// RWA(n) = RAB - Rs × n + Rw
    pub fn rwa(&self, code: u16) -> f64 {
        let code = code.min(self.steps - 1);
        let rs = self.step_resistance();
        self.r_ab_actual - rs * code as f64 + self.r_wiper
    }

    /// Conductance (Siemens) for a given code.
    ///
    /// g = 1 / RWB(n)
    pub fn conductance(&self, code: u16) -> f64 {
        1.0 / self.rwb(code)
    }

    /// Convert a target resistance to the nearest digipot code.
    ///
    /// n = round((R_target - Rw) / Rs)
    pub fn resistance_to_code(&self, target_resistance: f64) -> u16 {
        let rs = self.step_resistance();
        if rs <= 0.0 {
            return 0;
        }
        let n = ((target_resistance - self.r_wiper) / rs).round();
        n.clamp(0.0, (self.steps - 1) as f64) as u16
    }

    /// Convert a target conductance to the nearest digipot code.
    pub fn conductance_to_code(&self, target_conductance: f64) -> u16 {
        if target_conductance <= 0.0 {
            return self.steps - 1; // Maximum resistance = minimum conductance
        }
        let target_resistance = 1.0 / target_conductance;
        self.resistance_to_code(target_resistance)
    }

    /// Get minimum achievable resistance (code 0).
    pub fn min_resistance(&self) -> f64 {
        self.rwb(0)
    }

    /// Get maximum achievable resistance (code 256).
    pub fn max_resistance(&self) -> f64 {
        self.rwb(self.steps - 1)
    }

    /// Get minimum achievable conductance.
    pub fn min_conductance(&self) -> f64 {
        1.0 / self.max_resistance()
    }

    /// Get maximum achievable conductance.
    pub fn max_conductance(&self) -> f64 {
        1.0 / self.min_resistance()
    }

    /// Apply temperature drift to the resistance at a given code.
    ///
    /// R_temp = R_nominal × (1 + tempco × ΔT / 1e6)
    pub fn resistance_with_temperature(&self, code: u16, delta_temp_c: f64) -> f64 {
        let r_nominal = self.rwb(code);
        let drift = self.tempco_ppm * delta_temp_c / 1e6;
        r_nominal * (1.0 + drift)
    }
}

impl Default for MCP4661 {
    fn default() -> Self {
        Self::mcp4661_104() // Default to 100kΩ variant
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_resistance_100k() {
        let pot = MCP4661::mcp4661_104();
        let rs = pot.step_resistance();
        // Rs = 100kΩ / 256 ≈ 390.625Ω
        let expected = 100_000.0 / 256.0;
        assert!(
            (rs - expected).abs() < 0.01,
            "Expected {} Ω, got {} Ω",
            expected,
            rs
        );
    }

    #[test]
    fn rwb_at_code_zero() {
        let pot = MCP4661::mcp4661_104();
        let rwb = pot.rwb(0);
        // RWB(0) = Rs × 0 + Rw = 75Ω
        assert!(
            (rwb - 75.0).abs() < 0.01,
            "Expected 75 Ω at code 0, got {} Ω",
            rwb
        );
    }

    #[test]
    fn rwb_at_code_max() {
        let pot = MCP4661::mcp4661_104();
        let rwb = pot.rwb(256);
        // RWB(256) = Rs × 256 + Rw = 100kΩ + 75Ω = 100075Ω
        let expected = 100_000.0 + 75.0;
        assert!(
            (rwb - expected).abs() < 0.01,
            "Expected {} Ω at code 256, got {} Ω",
            expected,
            rwb
        );
    }

    #[test]
    fn conductance_range() {
        let pot = MCP4661::mcp4661_104();

        let g_min = pot.min_conductance();
        let g_max = pot.max_conductance();

        // g_max = 1/75Ω ≈ 13.3 mS
        // g_min = 1/100075Ω ≈ 10 µS
        assert!(g_max > g_min, "Max conductance should be > min");
        assert!(
            (g_max - 1.0 / 75.0).abs() < 1e-6,
            "Max conductance ≈ 1/Rw"
        );
    }

    #[test]
    fn resistance_to_code_round_trip() {
        let pot = MCP4661::mcp4661_104();

        for code in [0, 1, 128, 255, 256] {
            let r = pot.rwb(code);
            let recovered_code = pot.resistance_to_code(r);
            assert_eq!(
                code, recovered_code,
                "Round-trip failed for code {}: got {}",
                code, recovered_code
            );
        }
    }

    #[test]
    fn tolerance_affects_actual_rab() {
        let pot = MCP4661::mcp4661_104().with_tolerance_offset(0.1); // +10%
        let expected = 100_000.0 * 1.1;
        assert!(
            (pot.r_ab_actual - expected).abs() < 0.01,
            "Expected {} Ω, got {} Ω",
            expected,
            pot.r_ab_actual
        );
    }

    #[test]
    fn tolerance_clamped_to_spec() {
        let pot = MCP4661::mcp4661_104().with_tolerance_offset(0.5); // Exceeds ±20%
        let expected = 100_000.0 * 1.2; // Should clamp to +20%
        assert!(
            (pot.r_ab_actual - expected).abs() < 0.01,
            "Tolerance should be clamped to ±20%"
        );
    }

    #[test]
    fn temperature_drift() {
        let pot = MCP4661::mcp4661_104();
        let code = 128;

        let r_25c = pot.rwb(code);
        let r_75c = pot.resistance_with_temperature(code, 50.0); // +50°C

        // At 50 ppm/°C and ΔT=50°C: drift = 50 × 50 / 1e6 = 0.0025 = 0.25%
        let expected_drift = 50.0 * 50.0 / 1e6;
        let actual_drift = (r_75c - r_25c) / r_25c;

        assert!(
            (actual_drift - expected_drift).abs() < 1e-6,
            "Expected drift {}, got {}",
            expected_drift,
            actual_drift
        );
    }

    #[test]
    fn different_rab_variants() {
        let pot_5k = MCP4661::mcp4661_502();
        let pot_100k = MCP4661::mcp4661_104();

        // Step resistance should scale with RAB
        let ratio = pot_100k.step_resistance() / pot_5k.step_resistance();
        assert!(
            (ratio - 20.0).abs() < 0.01,
            "100k/5k step resistance ratio should be 20"
        );
    }
}

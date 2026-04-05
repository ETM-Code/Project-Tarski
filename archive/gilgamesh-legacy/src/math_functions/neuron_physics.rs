//! Unified neuron physics module.
//!
//! This module contains the core physics simulation functions used by both
//! `lif_equivalent` (SPICE-matching single neuron simulator) and `runtime`
//! (network training simulator). Changes here propagate to both systems.
//!
//! The physics is designed to match the SPICE circuit behavior exactly (besides some variation for efficiency).

// Note: relax_towards is available from super::lif if needed for physics computations

/// Diode forward voltage drop in the pulse stretch circuit.
/// The SPICE circuit has a diode (Dpw) between comp_out and comp_pulse,
/// which causes a voltage drop when charging the pulse stretch capacitor.
/// The SPICE diode model (.model DPW D(Is=1e-12 N=1.05 Rs=10)) has a lower
/// forward voltage than a standard silicon diode due to low saturation current.
/// Empirically matched to SPICE: ~0.37V drop at typical operating currents.
pub const PULSE_STRETCH_DIODE_DROP_V: f64 = 0.37;

/// Configuration for pulse stretching circuit (RC decay).
///
/// Models the hardware RC pulse stretcher with physically correct exponential decay:
/// - When comparator fires (u > theta): diode conducts, capacitor charges to V_peak
/// - While comparator stays high: capacitor held at V_peak (diode actively recharges)
/// - When comparator goes low (u ≤ theta): diode reverse-biases, capacitor decays
///   exponentially through parallel resistance R_eff with τ = R_eff × C_pw
///
/// This matches SPICE circuit behavior where decay starts when the comparator output
/// drops, not after an arbitrary hold time. The comparator naturally stays high during
/// membrane recovery due to threshold adaptation and hysteresis.
#[derive(Debug, Clone, Copy)]
pub struct PulseStretchConfig {
    /// Whether pulse stretching is enabled
    pub enable: bool,
    /// Effective RC decay time constant τ_eff = R_eff × C_pw in seconds
    /// where R_eff accounts for parallel discharge paths (R_pw || R_load)
    pub tau_s: f64,
}

impl Default for PulseStretchConfig {
    fn default() -> Self {
        Self {
            enable: false,
            tau_s: 0.0,
        }
    }
}

impl PulseStretchConfig {
    /// Create from effective time constant.
    /// Use this with τ_eff computed from parallel discharge paths.
    pub fn from_tau(tau_s: f64) -> Self {
        Self {
            enable: tau_s > 0.0,
            tau_s,
        }
    }

    /// Create from resistance and capacitance values.
    /// For accurate modeling, use effective resistance R_eff = R_pw || R_load.
    pub fn from_rc(r_eff_ohm: f64, c_f: f64) -> Self {
        Self {
            enable: r_eff_ohm > 0.0 && c_f > 0.0,
            tau_s: r_eff_ohm * c_f,
        }
    }
}

/// State for pulse stretching RC circuit.
#[derive(Debug, Clone, Copy, Default)]
pub struct PulseStretchState {
    /// Current voltage on the pulse stretch capacitor
    pub v_pulse: f64,
}

/// Configuration for comparator behavior.
#[derive(Debug, Clone, Copy)]
pub struct ComparatorConfig {
    /// Low rail voltage (typically 0V)
    pub v_low: f64,
    /// High rail voltage (typically VDD minus rail drop)
    pub v_high: f64,
    /// Offset added to threshold for comparison
    pub offset: f64,
}

impl Default for ComparatorConfig {
    fn default() -> Self {
        Self {
            v_low: 0.0,
            v_high: 5.0,
            offset: 0.0,
        }
    }
}

/// Result of comparator + pulse stretch computation for one timestep.
#[derive(Debug, Clone, Copy)]
pub struct ComparatorOutput {
    /// The comparator output voltage (including pulse stretch decay)
    pub v_comp: f64,
    /// Whether a new spike was detected this timestep
    pub spike_detected: bool,
}

/// Compute comparator output with RC pulse stretching (physically correct).
///
/// This models the SPICE circuit behavior:
/// 1. When u > theta: comparator output high → diode conducts → cap charges/holds at V_peak
/// 2. When u ≤ theta: comparator output low → diode reverse-biases → cap decays exponentially
///
/// The key insight: decay starts when the comparator *output* goes low, which happens
/// naturally when membrane voltage recovers due to threshold adaptation and hysteresis.
/// No artificial "hold time" needed - the physics handles it.
///
/// # Arguments
/// * `u` - Current membrane deflection (voltage above Vref)
/// * `theta_eff` - Effective threshold (theta + offset)
/// * `v_comp_prev` - Previous comparator output voltage
/// * `comp_cfg` - Comparator configuration
/// * `pulse_cfg` - Pulse stretch configuration
/// * `dt` - Time step in seconds
///
/// # Returns
/// The new comparator output voltage and whether a spike was detected.
pub fn compute_comparator_with_pulse_stretch(
    u: f64,
    theta_eff: f64,
    v_comp_prev: f64,
    comp_cfg: &ComparatorConfig,
    pulse_cfg: &PulseStretchConfig,
    dt: f64,
) -> ComparatorOutput {
    let threshold_crossed = u > theta_eff;
    let was_low = v_comp_prev <= comp_cfg.v_low + 1e-9;

    // Spike detection: rising edge (was low, now crossed threshold)
    let spike_detected = was_low && threshold_crossed;

    let v_peak = (comp_cfg.v_high - PULSE_STRETCH_DIODE_DROP_V).max(comp_cfg.v_low);

    let v_comp = if pulse_cfg.enable && pulse_cfg.tau_s > 0.0 {
        // RC pulse stretching mode (physically correct)
        if threshold_crossed {
            // Comparator high: diode forward-biased, cap charges to V_peak
            // (or stays at V_peak if already there)
            v_peak
        } else if v_comp_prev > comp_cfg.v_low + 1e-9 {
            // Comparator low: diode reverse-biased, cap decays exponentially
            // V(t+dt) = V(t) × exp(-dt/τ_eff)
            let v_decayed = v_comp_prev * (-dt / pulse_cfg.tau_s).exp();
            // Clamp to low rail to avoid numerical underflow
            if v_decayed < comp_cfg.v_low + 1e-9 {
                comp_cfg.v_low
            } else {
                v_decayed
            }
        } else {
            // Already at low rail
            comp_cfg.v_low
        }
    } else {
        // No pulse stretching: instantaneous comparator
        if threshold_crossed {
            comp_cfg.v_high
        } else {
            comp_cfg.v_low
        }
    };

    ComparatorOutput { v_comp, spike_detected }
}

/// Compute the pre-synaptic signal from a spiking neuron with pulse stretching.
///
/// This is the signal seen by downstream neurons through synapses.
/// Models RC exponential decay after a spike.
///
/// # Arguments
/// * `spike_time` - Time when the spike occurred (pulse started)
/// * `current_time` - Current simulation time
/// * `pulse_tau` - RC time constant (τ = R_pw × C_pw)
///
/// # Returns
/// Signal amplitude (0.0 to 1.0, where 1.0 is immediately after spike)
pub fn compute_pulse_stretched_signal(
    spike_time: f64,
    current_time: f64,
    pulse_tau: f64,
) -> f64 {
    if pulse_tau <= 0.0 {
        return 0.0;
    }

    let elapsed = current_time - spike_time;
    if elapsed < 0.0 {
        return 0.0;
    }

    // RC exponential decay: V(t) = V0 * exp(-t/τ)
    // We consider the pulse "fully decayed" after 5τ (< 1% remaining)
    if elapsed > 5.0 * pulse_tau {
        return 0.0;
    }

    (-elapsed / pulse_tau).exp()
}

/// Determine if a pulse is still active (hasn't fully decayed).
///
/// # Arguments
/// * `spike_time` - Time when the spike occurred
/// * `current_time` - Current simulation time
/// * `pulse_tau` - RC time constant
///
/// # Returns
/// True if the pulse is still active (within 5τ of spike time)
#[inline]
pub fn is_pulse_active(spike_time: f64, current_time: f64, pulse_tau: f64) -> bool {
    if pulse_tau <= 0.0 {
        return false;
    }
    let elapsed = current_time - spike_time;
    elapsed >= 0.0 && elapsed < 5.0 * pulse_tau
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pulse_stretch_decay() {
        let tau = 0.002; // 2ms

        // At t=0, signal should be 1.0
        let sig0 = compute_pulse_stretched_signal(0.0, 0.0, tau);
        assert!((sig0 - 1.0).abs() < 1e-9);

        // At t=τ, signal should be ~0.368 (1/e)
        let sig_tau = compute_pulse_stretched_signal(0.0, tau, tau);
        assert!((sig_tau - (-1.0_f64).exp()).abs() < 1e-6);

        // At t=5τ, signal should be nearly 0
        let sig_5tau = compute_pulse_stretched_signal(0.0, 5.0 * tau, tau);
        assert!(sig_5tau < 0.01);
    }

    #[test]
    fn test_comparator_spike_detection() {
        let comp_cfg = ComparatorConfig {
            v_low: 0.0,
            v_high: 5.0,
            offset: 0.0,
        };
        let pulse_cfg = PulseStretchConfig::from_tau(0.002); // 2ms time constant

        // Below threshold, no spike
        let out1 = compute_comparator_with_pulse_stretch(
            0.5, 0.8, 0.0, &comp_cfg, &pulse_cfg, 1e-4
        );
        assert!(!out1.spike_detected);
        assert!(out1.v_comp < 0.1);

        // Cross threshold, spike detected
        let out2 = compute_comparator_with_pulse_stretch(
            0.9, 0.8, 0.0, &comp_cfg, &pulse_cfg, 1e-4
        );
        assert!(out2.spike_detected);
        // With pulse stretching, v_comp should be v_high minus diode drop
        let expected_v = 5.0 - PULSE_STRETCH_DIODE_DROP_V;
        assert!((out2.v_comp - expected_v).abs() < 1e-9);

        // Stay above threshold, v_comp stays high (no new spike, no decay while high)
        let out3 = compute_comparator_with_pulse_stretch(
            0.9, 0.8, expected_v, &comp_cfg, &pulse_cfg, 1e-4
        );
        assert!(!out3.spike_detected);
        assert!((out3.v_comp - expected_v).abs() < 1e-9); // Still at peak

        // Drop below threshold, v_comp should decay
        let out4 = compute_comparator_with_pulse_stretch(
            0.7, 0.8, expected_v, &comp_cfg, &pulse_cfg, 1e-4
        );
        assert!(!out4.spike_detected);
        let expected_decay = expected_v * (-1e-4 / 0.002_f64).exp();
        assert!((out4.v_comp - expected_decay).abs() < 1e-6);
        assert!(out4.v_comp < expected_v); // Should have decayed
    }
}

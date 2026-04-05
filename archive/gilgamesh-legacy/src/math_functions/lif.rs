//! Mathematical helper functions for leaky integrate-and-fire dynamics.

/// Compute the membrane deflection after a time step `dt` for a first-order system
/// with time constant `tau` and steady-state value `u_inf`.
#[inline]
pub fn relax_towards(u0: f64, u_inf: f64, tau: f64, dt: f64) -> f64 {
    if tau <= 0.0 {
        return u_inf;
    }
    let alpha = (-dt / tau).exp();
    u_inf + (u0 - u_inf) * alpha
}

/// Solve for the time (relative to `t0`) at which the membrane deflection crosses `target`.
/// Returns `None` if the target is not approached within the interval.
pub fn crossing_time(u0: f64, u_inf: f64, tau: f64, target: f64) -> Option<f64> {
    if tau <= 0.0 {
        return None;
    }
    if (u0 - target).abs() < f64::EPSILON {
        return Some(0.0);
    }
    let denom = u_inf - target;
    if denom.abs() < f64::EPSILON {
        return None;
    }
    let ratio = (u_inf - u0) / denom;
    if ratio <= 0.0 {
        return None;
    }
    Some(tau * ratio.ln())
}

/// Update a first-order adaptive variable towards `theta_inf` with time constant `tau`.
#[inline]
pub fn adapt(theta: f64, theta_inf: f64, tau: f64, dt: f64) -> f64 {
    relax_towards(theta, theta_inf, tau, dt)
}

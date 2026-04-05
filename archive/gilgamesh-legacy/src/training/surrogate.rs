//! Surrogate gradient functions for backpropagation through spiking neurons.
//!
//! Since the Heaviside step function (used for spike generation) has zero gradient
//! almost everywhere, we use surrogate gradients during backpropagation.

/// Trait for surrogate gradient functions.
pub trait SurrogateGradient {
    /// Compute the surrogate gradient at membrane potential `u` relative to threshold `theta`.
    ///
    /// Returns a value in approximately [0, 1] representing how "close" to spiking the neuron was.
    fn gradient(&self, u: f64, theta: f64) -> f64;
}

/// Enum wrapper for surrogate gradient functions (for use in configs).
#[derive(Debug, Clone, Copy)]
pub enum SurrogateType {
    /// FastSigmoid with specified slope.
    FastSigmoid(FastSigmoid),
    /// Triangular with specified width.
    Triangular(Triangular),
}

impl Default for SurrogateType {
    fn default() -> Self {
        SurrogateType::FastSigmoid(FastSigmoid::default())
    }
}

impl SurrogateType {
    /// Compute the surrogate gradient.
    pub fn gradient(&self, u: f64, theta: f64) -> f64 {
        match self {
            SurrogateType::FastSigmoid(s) => s.gradient(u, theta),
            SurrogateType::Triangular(s) => s.gradient(u, theta),
        }
    }
}

/// FastSigmoid surrogate gradient.
///
/// Gradient: 1 / (1 + k|u - θ|)²
///
/// This is computationally efficient and works well for SNNs.
/// Default slope k=25 is recommended for most applications.
#[derive(Debug, Clone, Copy)]
pub struct FastSigmoid {
    /// Slope parameter (steepness). Higher = sharper gradient around threshold.
    pub slope: f64,
}

impl FastSigmoid {
    /// Create a new FastSigmoid with the given slope.
    pub fn new(slope: f64) -> Self {
        Self { slope }
    }

    /// Create with default slope (k=25).
    pub fn default_slope() -> Self {
        Self { slope: 25.0 }
    }
}

impl Default for FastSigmoid {
    fn default() -> Self {
        Self::default_slope()
    }
}

impl SurrogateGradient for FastSigmoid {
    #[inline]
    fn gradient(&self, u: f64, theta: f64) -> f64 {
        let x = u - theta;
        let denom = 1.0 + self.slope * x.abs();
        1.0 / (denom * denom)
    }
}

/// Triangular surrogate gradient (piecewise linear).
///
/// Gradient: max(0, 1 - |u - θ| / width)
///
/// Simple and effective, with finite support.
#[derive(Debug, Clone, Copy)]
pub struct Triangular {
    /// Width of the triangle. Gradient is zero outside [-width, width] from threshold.
    pub width: f64,
}

impl Triangular {
    pub fn new(width: f64) -> Self {
        Self { width: width.abs().max(1e-6) }
    }
}

impl Default for Triangular {
    fn default() -> Self {
        Self { width: 1.0 }
    }
}

impl SurrogateGradient for Triangular {
    #[inline]
    fn gradient(&self, u: f64, theta: f64) -> f64 {
        let x = (u - theta).abs();
        if x < self.width {
            1.0 - x / self.width
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_sigmoid_at_threshold() {
        let sg = FastSigmoid::default();
        // At threshold, gradient should be maximum (1.0)
        let grad = sg.gradient(1.0, 1.0);
        assert!((grad - 1.0).abs() < 1e-10, "Expected 1.0 at threshold, got {}", grad);
    }

    #[test]
    fn fast_sigmoid_symmetric() {
        let sg = FastSigmoid::default();
        let theta = 1.0;
        // Gradient should be symmetric around threshold
        let above = sg.gradient(theta + 0.1, theta);
        let below = sg.gradient(theta - 0.1, theta);
        assert!((above - below).abs() < 1e-10, "Should be symmetric: {} vs {}", above, below);
    }

    #[test]
    fn fast_sigmoid_decreases_away_from_threshold() {
        let sg = FastSigmoid::default();
        let theta = 1.0;
        let at_theta = sg.gradient(theta, theta);
        let near = sg.gradient(theta + 0.01, theta);
        let far = sg.gradient(theta + 0.1, theta);
        assert!(at_theta > near, "Gradient should decrease away from threshold");
        assert!(near > far, "Gradient should continue decreasing");
    }

    #[test]
    fn fast_sigmoid_nonzero_everywhere() {
        let sg = FastSigmoid::default();
        // FastSigmoid never reaches exactly zero (unlike triangular)
        let far = sg.gradient(100.0, 1.0);
        assert!(far > 0.0, "FastSigmoid should be non-zero far from threshold");
    }

    #[test]
    fn fast_sigmoid_slope_affects_width() {
        let narrow = FastSigmoid::new(100.0);
        let wide = FastSigmoid::new(1.0);
        let theta = 1.0;
        let offset = 0.1;

        let narrow_grad = narrow.gradient(theta + offset, theta);
        let wide_grad = wide.gradient(theta + offset, theta);

        // Higher slope = narrower peak = lower gradient away from threshold
        assert!(narrow_grad < wide_grad, "Higher slope should give narrower gradient");
    }

    #[test]
    fn triangular_at_threshold() {
        let sg = Triangular::default();
        let grad = sg.gradient(1.0, 1.0);
        assert!((grad - 1.0).abs() < 1e-10, "Expected 1.0 at threshold, got {}", grad);
    }

    #[test]
    fn triangular_zero_outside_width() {
        let sg = Triangular::new(0.5);
        let theta = 1.0;
        // Outside width, gradient should be zero
        let outside = sg.gradient(theta + 1.0, theta);
        assert!((outside).abs() < 1e-10, "Expected 0 outside width, got {}", outside);
    }

    #[test]
    fn triangular_linear_decay() {
        let sg = Triangular::new(1.0);
        let theta = 1.0;
        // At half width, gradient should be 0.5
        let half = sg.gradient(theta + 0.5, theta);
        assert!((half - 0.5).abs() < 1e-10, "Expected 0.5 at half width, got {}", half);
    }
}

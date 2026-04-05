//! Centralised constants for the equivalent LIF simulator.

pub const DEFAULT_ANALOG_SERIES_R_OHM: f64 = 100.0;
pub const DEFAULT_ANALOG_LOAD_R_OHM: f64 = 10_000.0;
pub const DEFAULT_DIODE_DROP_V: f64 = 0.3;
// Effective resistance used when translating comparator overdrive into threshold pumping.
// Raised to soften adaptation so spikes in the equivalent model track SPICE closer.
pub const DEFAULT_INJECTION_R_OHM: f64 = 220_000_000.0;

pub const DIVIDER_R_VH_OHM: f64 = 681_000.0;
pub const DIVIDER_R_VL_OHM: f64 = 316_000.0;

/// Effective headroom lost on the comparator rail in SPICE due to output RC and loading.
pub const COMPARATOR_RAIL_DROP_V: f64 = 0.21;

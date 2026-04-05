//! Hardware models for realistic SNN simulation.
//!
//! This module provides models for:
//! - MCP4661 digital potentiometer (weight storage)
//! - Inverting op-amp output stage

pub mod mcp4661;
pub mod opamp;

pub use mcp4661::MCP4661;
pub use opamp::InvertingOpAmp;

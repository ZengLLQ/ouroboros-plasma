//! Toroidal Alfvén Eigenmodes (TAE) and MHD spectral stability module.
//!
//! Provides Alfvén continuum dispersion analysis, continuum gap identification,
//! discrete TAE eigenmode ODE solvers, thermal plasma damping breakdowns,
//! and energetic alpha particle resonant drive / linear threshold calculations.

pub mod alfven;
pub mod damping;
pub mod drive;
pub mod tae;

pub use alfven::AlfvenContinuum;
pub use damping::{DampingBreakdown, DampingCalculator};
pub use drive::{FastIonDriveCalculator, ModeStability, TaeStabilityReport};
pub use tae::{TaeMode, TaeSolver};

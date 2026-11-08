//! Magnetohydrodynamics (MHD) fluid simulation module.
//!
//! Provides ideal and resistive MHD equations, conservative/primitive variables,
//! 2D cylindrical/toroidal spatial discretization, Runge-Kutta time stepping,
//! and divergence-free magnetic field diagnostics.

pub mod equations;
pub mod grid;
pub mod state;
pub mod stepper;

pub use equations::{evaluate_mhd_equations, MhdDerivatives};
pub use grid::{BoundaryCondition, MhdGrid};
pub use state::MhdState;
pub use stepper::{MhdStepper, MhdStepperConfig};

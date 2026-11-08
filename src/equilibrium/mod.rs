//! Tokamak magnetohydrodynamic equilibrium module.
//!
//! Provides the 2D Grad-Shafranov elliptic PDE solver, Soloviev analytic equilibrium,
//! Miller geometry flux surface parameterization, and plasma profile analysis.

pub mod flux_surface;
pub mod profiles;
pub mod solver;

pub use flux_surface::{FluxSurface, MillerGeometry};
pub use profiles::PlasmaProfiles;
pub use solver::{GradShafranovSolver, SolovievEquilibrium};

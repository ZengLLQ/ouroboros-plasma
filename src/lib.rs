//! # OuroborosPlasma
//!
//! Computational Fusion Physics & Magnetohydrodynamics (MHD) library in pure Rust (2021 edition).
//!
//! Provides:
//! - **Equilibrium**: 2D/3D Grad-Shafranov elliptic PDE solver (\Delta^* \psi = -\mu_0 R^2 p' - F F'),
//!   Soloviev analytic equilibria, Miller flux surface shaping, and safety factor q(\psi) profile analysis.
//! - **MHD**: Ideal and resistive magnetohydrodynamics fluid conservation equations (mass continuity,
//!   momentum with Lorentz \mathbf{J} \times \mathbf{B} force, Faraday induction, energy conservation)
//!   solved with 4th-order Runge-Kutta time integration.
//! - **Particles**: Relativistic Boris velocity leapfrog integrator for energetic fusion products (e.g., 3.5 MeV \alpha),
//!   guiding center drifts (\nabla B, curvature, \mathbf{E}\times\mathbf{B}), and loss cone kinetics.
//! - **Modes**: Toroidal Alfvén Eigenmodes (TAE) spectral eigen-analysis, continuum gap formation,
//!   multicomponent damping (continuum, Landau, radiative), and fast-ion resonant drive.
//! - **Visualizer**: High-resolution 7-bit ASCII cross-sectional flux contour maps, q-profile plots,
//!   and ITER-grade fusion performance diagnostics.

pub mod cli;
pub mod constants;
pub mod equilibrium;
pub mod math;
pub mod mhd;
pub mod modes;
pub mod particles;
pub mod visualizer;

// Convenient re-exports
pub use constants::*;
pub use equilibrium::{FluxSurface, GradShafranovSolver, MillerGeometry, PlasmaProfiles, SolovievEquilibrium};
pub use math::Vector3D;
pub use mhd::{BoundaryCondition, MhdGrid, MhdState, MhdStepper, MhdStepperConfig};
pub use modes::{AlfvenContinuum, DampingBreakdown, DampingCalculator, FastIonDriveCalculator, ModeStability, TaeMode, TaeSolver};
pub use particles::{GuidingCenter, LossConeAnalyzer, OrbitTopology, ParticleEnsemble, ParticleState, RelativisticBorisIntegrator};
pub use visualizer::{AsciiVisualizer, FusionPerformanceReport};

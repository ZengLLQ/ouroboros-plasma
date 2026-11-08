//! Energetic particle kinetics and relativistic trajectory integration module.
//!
//! Provides the relativistic Boris velocity leapfrog integrator, guiding center drift theory,
//! magnetic mirror trapping, banana orbit kinetics, and fusion alpha particle ensemble simulation.

pub mod boris;
pub mod ensemble;
pub mod guiding_center;
pub mod loss_cone;

pub use boris::{ParticleState, RelativisticBorisIntegrator};
pub use ensemble::{ConfinementReport, ParticleEnsemble};
pub use guiding_center::GuidingCenter;
pub use loss_cone::{LossConeAnalyzer, OrbitTopology};

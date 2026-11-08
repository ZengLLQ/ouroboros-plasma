//! Energetic fusion alpha particle ensemble simulation and confinement statistics.

use std::f64::consts::PI;
use serde::{Deserialize, Serialize};
use crate::constants::{ALPHA_BIRTH_ENERGY_JOULES, ALPHA_CHARGE, ALPHA_MASS};
use crate::math::Vector3D;
use super::boris::{ParticleState, RelativisticBorisIntegrator};
use super::loss_cone::{LossConeAnalyzer, OrbitTopology};

/// Confinement statistics summary for energetic fusion products.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfinementReport {
    pub total_particles: usize,
    pub passing_count: usize,
    pub trapped_count: usize,
    pub prompt_loss_count: usize,
    pub passing_fraction: f64,
    pub trapped_fraction: f64,
    pub prompt_loss_fraction: f64,
    pub confinement_fraction: f64,
    pub total_birth_energy_joules: f64,
    pub confined_energy_joules: f64,
    pub lost_energy_joules: f64,
}

/// Ensemble of energetic fusion alpha particles.
#[derive(Debug, Clone)]
pub struct ParticleEnsemble {
    pub particles: Vec<ParticleState>,
    pub topologies: Vec<OrbitTopology>,
    pub r0: f64,
    pub a: f64,
    pub b0: f64,
    pub q_safety: f64,
}

impl ParticleEnsemble {
    /// Generates a realistic isotropic 3.52 MeV D-T alpha particle ensemble.
    ///
    /// Spatial birth distribution follows thermonuclear fusion profile: S(r) \propto (1 - (r/a)^2)^4.
    pub fn generate_dt_alphas(
        num_particles: usize,
        r0: f64,
        a: f64,
        b0: f64,
        q_safety: f64,
        seed: u64,
    ) -> Self {
        let mut particles = Vec::with_capacity(num_particles);
        let mut topologies = Vec::with_capacity(num_particles);
        let analyzer = LossConeAnalyzer::new(r0, a, b0, q_safety);

        // Alpha birth speed from 3.52 MeV
        let v_alpha = (2.0 * ALPHA_BIRTH_ENERGY_JOULES / ALPHA_MASS).sqrt();

        // Deterministic pseudo-random sequence for reproducible physics
        let mut state = seed.max(1);
        let mut lcg = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };

        for _ in 0..num_particles {
            // Rejection sampling for S(rho) \propto rho * (1 - rho^2)^4
            let rho = loop {
                let r_cand = lcg();
                let prob_cand = lcg();
                let f_profile = (1.0 - r_cand * r_cand).powi(4);
                if prob_cand <= f_profile {
                    break r_cand;
                }
            };

            let r_minor = rho * a;
            let theta = lcg() * 2.0 * PI;
            let phi = lcg() * 2.0 * PI;

            // Position in torus: (R, \phi, Z)
            let r_major = r0 + r_minor * theta.cos();
            let z_pos = r_minor * theta.sin();
            let pos = Vector3D::new(r_major * phi.cos(), r_major * phi.sin(), z_pos);

            // Isotropic velocity direction on sphere
            let pitch = lcg() * 2.0 - 1.0; // cos(\chi) \in [-1, 1]
            let gyro_angle = lcg() * 2.0 * PI;
            let v_par = v_alpha * pitch;
            let v_perp = v_alpha * (1.0 - pitch * pitch).max(0.0).sqrt();

            // Transform velocity to Cartesian (approximating toroidal direction as parallel)
            let vr = v_perp * gyro_angle.cos();
            let vz = v_perp * gyro_angle.sin();
            let vphi = v_par;
            let vel = Vector3D::cylindrical_to_cartesian_vector(vr, vphi, vz, phi);

            let particle = ParticleState::new(pos, vel, ALPHA_MASS, ALPHA_CHARGE);
            let topo = analyzer.classify_orbit(r_minor, pitch, v_alpha, ALPHA_MASS, ALPHA_CHARGE);

            particles.push(particle);
            topologies.push(topo);
        }

        Self {
            particles,
            topologies,
            r0,
            a,
            b0,
            q_safety,
        }
    }

    /// Evaluates the confinement statistics of the ensemble.
    pub fn compute_confinement_report(&self) -> ConfinementReport {
        let total = self.particles.len();
        let mut passing = 0;
        let mut trapped = 0;
        let mut prompt_loss = 0;

        for &topo in &self.topologies {
            match topo {
                OrbitTopology::Passing => passing += 1,
                OrbitTopology::TrappedBanana => trapped += 1,
                OrbitTopology::PromptLoss => prompt_loss += 1,
            }
        }

        let n = total.max(1) as f64;
        let passing_fraction = passing as f64 / n;
        let trapped_fraction = trapped as f64 / n;
        let prompt_loss_fraction = prompt_loss as f64 / n;
        let confinement_fraction = 1.0 - prompt_loss_fraction;

        let total_birth_energy = total as f64 * ALPHA_BIRTH_ENERGY_JOULES;
        let lost_energy = prompt_loss as f64 * ALPHA_BIRTH_ENERGY_JOULES;
        let confined_energy = total_birth_energy - lost_energy;

        ConfinementReport {
            total_particles: total,
            passing_count: passing,
            trapped_count: trapped,
            prompt_loss_count: prompt_loss,
            passing_fraction,
            trapped_fraction,
            prompt_loss_fraction,
            confinement_fraction,
            total_birth_energy_joules: total_birth_energy,
            confined_energy_joules: confined_energy,
            lost_energy_joules: lost_energy,
        }
    }

    /// Advances the entire particle ensemble by `num_steps` Boris integrator steps.
    pub fn advance_ensemble<FEb>(&mut self, dt: f64, num_steps: usize, field_fn: FEb)
    where
        FEb: Fn(Vector3D) -> (Vector3D, Vector3D) + Copy,
    {
        for p in self.particles.iter_mut() {
            for _ in 0..num_steps {
                RelativisticBorisIntegrator::step(p, dt, field_fn);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_ensemble_confinement() {
        // ITER-like tokamak: R0 = 6.2 m, a = 2.0 m, B0 = 5.3 T, q = 1.8
        let ensemble = ParticleEnsemble::generate_dt_alphas(500, 6.2, 2.0, 5.3, 1.8, 42);
        let report = ensemble.compute_confinement_report();

        assert_eq!(report.total_particles, 500);
        assert!(report.confinement_fraction > 0.85, "ITER should confine > 85% of 3.5 MeV alphas: {}", report.confinement_fraction);
        assert!(report.trapped_fraction > 0.2 && report.trapped_fraction < 0.6);
        assert!(report.prompt_loss_fraction < 0.15);
    }
}

//! Relativistic Boris velocity integrator for charged particle dynamics in electromagnetic fields.
//!
//! Solves the relativistic Lorentz force equations:
//! d\mathbf{x} / dt = \mathbf{v} = \mathbf{u} / \gamma
//! d\mathbf{u} / dt = (q / m) [\mathbf{E} + (\mathbf{u} / \gamma) \times \mathbf{B}]
//! where \mathbf{u} = \gamma \mathbf{v} and \gamma = \sqrt{1 + |\mathbf{u}|^2 / c^2}.

use serde::{Deserialize, Serialize};
use crate::constants::SPEED_OF_LIGHT;
use crate::math::Vector3D;

/// State of an energetic relativistic charged particle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleState {
    /// Cartesian position \mathbf{x} = (x, y, z) in meters.
    pub position: Vector3D,
    /// Relativistic spatial momentum per unit mass \mathbf{u} = \gamma \mathbf{v} in m/s.
    pub proper_velocity: Vector3D,
    /// Rest mass m in kg.
    pub mass: f64,
    /// Electric charge q in Coulombs.
    pub charge: f64,
    /// Current particle time t in seconds.
    pub time: f64,
}

impl ParticleState {
    /// Creates a new particle with given position and velocity.
    pub fn new(position: Vector3D, velocity: Vector3D, mass: f64, charge: f64) -> Self {
        let v2 = velocity.norm_squared();
        let c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;
        assert!(v2 < c2, "Particle velocity cannot exceed speed of light: {}", velocity.norm());

        let gamma = 1.0 / (1.0 - v2 / c2).sqrt();
        let proper_velocity = velocity * gamma;

        Self {
            position,
            proper_velocity,
            mass,
            charge,
            time: 0.0,
        }
    }

    /// Evaluates the relativistic Lorentz factor \gamma = \sqrt{1 + u^2 / c^2}.
    pub fn gamma(&self) -> f64 {
        let u2 = self.proper_velocity.norm_squared();
        let c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;
        (1.0 + u2 / c2).sqrt()
    }

    /// Physical velocity vector \mathbf{v} = \mathbf{u} / \gamma in m/s.
    pub fn velocity(&self) -> Vector3D {
        self.proper_velocity / self.gamma()
    }

    /// Kinetic energy E_{kin} = (\gamma - 1) m c^2 in Joules.
    pub fn kinetic_energy(&self) -> f64 {
        let gamma = self.gamma();
        let c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;
        (gamma - 1.0) * self.mass * c2
    }

    /// Kinetic energy in electron-volts (eV).
    pub fn kinetic_energy_ev(&self) -> f64 {
        self.kinetic_energy() / crate::constants::ELEMENTARY_CHARGE
    }

    /// Cylindrical coordinates (R, \phi, Z).
    pub fn cylindrical_position(&self) -> (f64, f64, f64) {
        self.position.to_cylindrical()
    }

    /// Evaluates toroidal canonical angular momentum P_\phi = R m \gamma v_\phi - q \psi(R, Z).
    pub fn canonical_angular_momentum(&self, poloidal_flux_psi: f64) -> f64 {
        let (r, phi, _z) = self.cylindrical_position();
        let (_vr, vphi, _vz) = self.velocity().cartesian_to_cylindrical_vector(phi);
        let gamma = self.gamma();
        r * self.mass * gamma * vphi - self.charge * poloidal_flux_psi
    }
}

/// Relativistic Boris leapfrog integrator.
#[derive(Debug, Clone)]
pub struct RelativisticBorisIntegrator;

impl RelativisticBorisIntegrator {
    /// Advances particle state by time step \Delta t using the Boris algorithm.
    ///
    /// Evaluates electric field \mathbf{E}(\mathbf{x}) and magnetic field \mathbf{B}(\mathbf{x})
    /// at the particle's current spatial position.
    pub fn step<FEb>(particle: &mut ParticleState, dt: f64, field_fn: FEb)
    where
        FEb: Fn(Vector3D) -> (Vector3D, Vector3D), // (E, B)
    {
        let (e_field, b_field) = field_fn(particle.position);

        let q_over_m = particle.charge / particle.mass;
        let c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;

        // 1. First half-step electric acceleration: u^- = u^n + (q E / m) * (dt / 2)
        let u_minus = particle.proper_velocity + e_field * (q_over_m * 0.5 * dt);

        // 2. Relativistic gamma^-
        let gamma_minus = (1.0 + u_minus.norm_squared() / c2).sqrt();

        // 3. Magnetic rotation angle vector: t = (q B / m) * (dt / (2 gamma^-))
        let t_vec = b_field * (q_over_m * 0.5 * dt / gamma_minus);
        let t_mag2 = t_vec.norm_squared();

        // 4. Vector s = 2 t / (1 + |t|^2)
        let s_vec = t_vec * (2.0 / (1.0 + t_mag2));

        // 5. Rotation:
        // u' = u^- + u^- x t
        // u^+ = u^- + u' x s
        let u_prime = u_minus + u_minus.cross(&t_vec);
        let u_plus = u_minus + u_prime.cross(&s_vec);

        // 6. Second half-step electric acceleration: u^{n+1/2} = u^+ + (q E / m) * (dt / 2)
        let u_next = u_plus + e_field * (q_over_m * 0.5 * dt);

        // 7. Compute updated relativistic factor gamma^{n+1/2}
        let gamma_next = (1.0 + u_next.norm_squared() / c2).sqrt();

        // 8. Position update: x^{n+1} = x^n + (u^{n+1/2} / gamma^{n+1/2}) * dt
        let v_next = u_next / gamma_next;
        particle.position += v_next * dt;
        particle.proper_velocity = u_next;
        particle.time += dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{ALPHA_CHARGE, ALPHA_MASS};

    #[test]
    fn test_relativistic_energy_conservation_in_pure_b_field() {
        // In pure magnetic field (E = 0), kinetic energy must be conserved exactly
        let pos = Vector3D::new(6.0, 0.0, 0.0);
        let vel = Vector3D::new(1.0e6, 2.0e6, 0.5e6);
        let mut particle = ParticleState::new(pos, vel, ALPHA_MASS, ALPHA_CHARGE);

        let initial_energy = particle.kinetic_energy();
        let b_uniform = Vector3D::new(0.0, 0.0, 3.0); // 3 T vertical field
        let dt = 1.0e-9; // 1 ns

        for _ in 0..1000 {
            RelativisticBorisIntegrator::step(&mut particle, dt, |_pos| {
                (Vector3D::zero(), b_uniform)
            });
        }

        let final_energy = particle.kinetic_energy();
        assert!(
            (final_energy - initial_energy).abs() / initial_energy < 1.0e-12,
            "Boris integrator must preserve kinetic energy in pure B field to machine precision"
        );
    }

    #[test]
    fn test_cyclotron_frequency_and_gyroradius() {
        // Gyromotion of alpha particle in uniform 5 Tesla field:
        // omega_c = q B / (gamma m), r_L = v_perp / omega_c
        let b0 = 5.0;
        let v_perp = 1.0e6;
        let pos = Vector3D::new(1.0, 0.0, 0.0);
        let vel = Vector3D::new(0.0, v_perp, 0.0);
        let mut particle = ParticleState::new(pos, vel, ALPHA_MASS, ALPHA_CHARGE);

        let gamma = particle.gamma();
        let omega_c = (ALPHA_CHARGE * b0) / (gamma * ALPHA_MASS);
        let gyro_period = 2.0 * std::f64::consts::PI / omega_c;
        let expected_rl = v_perp / omega_c;

        let steps_per_orbit = 100;
        let dt = gyro_period / (steps_per_orbit as f64);

        for _ in 0..steps_per_orbit {
            RelativisticBorisIntegrator::step(&mut particle, dt, |_pos| {
                (Vector3D::zero(), Vector3D::new(0.0, 0.0, b0))
            });
        }

        // After one full period, position should return to initial position
        assert!((particle.position.x - 1.0).abs() < expected_rl * 0.05);
        assert!(particle.position.y.abs() < expected_rl * 0.05);
    }
}

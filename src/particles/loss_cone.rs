//! Magnetic mirror trapping, banana orbit kinetics, and prompt loss cone kinetics.

use serde::{Deserialize, Serialize};

/// Classification of particle orbit topology in a tokamak.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrbitTopology {
    /// Passing (circulating) along the toroidal magnetic field.
    Passing,
    /// Trapped in magnetic well, executing banana bounce motion.
    TrappedBanana,
    /// Promptly lost to the first wall / limiter.
    PromptLoss,
}

/// Loss cone and banana orbit kinematic analyzer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossConeAnalyzer {
    pub r0: f64,
    pub a: f64,
    pub b0: f64,
    pub q_safety: f64,
}

impl LossConeAnalyzer {
    /// Creates a loss cone analyzer for a tokamak with major radius R0, minor radius a, on-axis B0, and safety factor q.
    pub fn new(r0: f64, a: f64, b0: f64, q_safety: f64) -> Self {
        Self { r0, a, b0, q_safety }
    }

    /// Inverse aspect ratio \epsilon = r / R_0.
    pub fn inverse_aspect_ratio(&self, r_minor: f64) -> f64 {
        (r_minor / self.r0).clamp(0.0, 0.99)
    }

    /// Mirror ratio R_M = B_{max} / B_{min} = (1 + \epsilon) / (1 - \epsilon).
    pub fn mirror_ratio(&self, r_minor: f64) -> f64 {
        let eps = self.inverse_aspect_ratio(r_minor);
        (1.0 + eps) / (1.0 - eps).max(1.0e-6)
    }

    /// Critical pitch angle pitch parameter \xi_c = (v_\parallel / v)_{crit} separating passing from trapped particles:
    /// \xi_c = \sqrt{1 - B_{min} / B_{max}} \approx \sqrt{2\epsilon / (1 + \epsilon)}.
    pub fn critical_pitch(&self, r_minor: f64) -> f64 {
        let eps = self.inverse_aspect_ratio(r_minor);
        (2.0 * eps / (1.0 + eps)).sqrt()
    }

    /// Theoretical trapped particle fraction for isotropic velocity distribution:
    /// f_{trapped} \approx \sqrt{2\epsilon}.
    pub fn trapped_fraction(&self, r_minor: f64) -> f64 {
        let eps = self.inverse_aspect_ratio(r_minor);
        (2.0 * eps).sqrt().min(1.0)
    }

    /// Banana orbit half-width \Delta_b for trapped particles:
    /// \Delta_b \approx \frac{q \rho_L}{\sqrt{\epsilon}}
    /// where \rho_L = v / \Omega_c is the Larmor radius.
    pub fn banana_orbit_width(
        &self,
        r_minor: f64,
        particle_speed: f64,
        mass: f64,
        charge: f64,
    ) -> f64 {
        let eps = self.inverse_aspect_ratio(r_minor).max(0.01);
        let omega_c = (charge.abs() * self.b0) / mass;
        let larmor_radius = particle_speed / omega_c;
        (self.q_safety * larmor_radius) / eps.sqrt()
    }

    /// Banana bounce frequency \omega_b for trapped particles:
    /// \omega_b \approx \frac{v \sqrt{\epsilon}}{2 \sqrt{2} q R_0}.
    pub fn bounce_frequency(&self, r_minor: f64, particle_speed: f64) -> f64 {
        let eps = self.inverse_aspect_ratio(r_minor).max(0.001);
        (particle_speed * eps.sqrt()) / (2.0 * (2.0_f64).sqrt() * self.q_safety * self.r0)
    }

    /// Classifies orbit topology given particle position and pitch \xi = v_\parallel / v.
    pub fn classify_orbit(
        &self,
        r_minor: f64,
        pitch: f64,
        particle_speed: f64,
        mass: f64,
        charge: f64,
    ) -> OrbitTopology {
        let xi_c = self.critical_pitch(r_minor);
        let is_trapped = pitch.abs() < xi_c;

        if is_trapped {
            let width = self.banana_orbit_width(r_minor, particle_speed, mass, charge);
            // If the outer banana tip exceeds the minor radius a, it hits the wall!
            if r_minor + width >= self.a {
                OrbitTopology::PromptLoss
            } else {
                OrbitTopology::TrappedBanana
            }
        } else {
            // Passing particle drift orbit shift: \Delta_p \approx q \rho_L
            let omega_c = (charge.abs() * self.b0) / mass;
            let larmor_radius = particle_speed / omega_c;
            let drift_shift = self.q_safety * larmor_radius;
            if r_minor + drift_shift >= self.a {
                OrbitTopology::PromptLoss
            } else {
                OrbitTopology::Passing
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{ALPHA_BIRTH_ENERGY_JOULES, ALPHA_CHARGE, ALPHA_MASS, SPEED_OF_LIGHT};

    #[test]
    fn test_loss_cone_trapped_fraction() {
        let analyzer = LossConeAnalyzer::new(6.2, 2.0, 5.3, 2.0);
        let r_mid = 1.0; // r/a = 0.5, eps = 1.0 / 6.2 ~ 0.161
        let f_trap = analyzer.trapped_fraction(r_mid);
        assert!(f_trap > 0.4 && f_trap < 0.7);

        let xi_c = analyzer.critical_pitch(r_mid);
        assert!(xi_c > 0.4 && xi_c < 0.6);
    }

    #[test]
    fn test_alpha_banana_orbit_and_prompt_loss() {
        let analyzer = LossConeAnalyzer::new(6.2, 2.0, 5.3, 2.0);

        // Alpha particle at 3.52 MeV
        let v_alpha = (2.0 * ALPHA_BIRTH_ENERGY_JOULES / ALPHA_MASS).sqrt();
        assert!(v_alpha > 1.0e7 && v_alpha < SPEED_OF_LIGHT);

        // Deep in the core: r = 0.2 m -> trapped particle should be well confined
        let topo_core = analyzer.classify_orbit(0.2, 0.1, v_alpha, ALPHA_MASS, ALPHA_CHARGE);
        assert_eq!(topo_core, OrbitTopology::TrappedBanana);

        // Near the edge: r = 1.95 m -> banana orbit width exceeds boundary -> prompt loss!
        let topo_edge = analyzer.classify_orbit(1.95, 0.1, v_alpha, ALPHA_MASS, ALPHA_CHARGE);
        assert_eq!(topo_edge, OrbitTopology::PromptLoss);
    }
}

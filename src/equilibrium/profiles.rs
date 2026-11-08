//! Plasma radial profiles: kinetic pressure, poloidal current stream function,
//! safety factor q(\psi), magnetic shear, and Shafranov shift.

use serde::{Deserialize, Serialize};
use crate::constants::MU_0;

/// Radial plasma equilibrium profiles defined as functions of normalized poloidal flux \bar{\psi} \in [0, 1].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasmaProfiles {
    /// Central kinetic pressure p_0 (Pa).
    pub p0: f64,
    /// Profile exponent \alpha_p for p(\bar{\psi}) = p_0 (1 - \bar{\psi}^{\alpha_p})^{\beta_p}.
    pub alpha_p: f64,
    /// Profile exponent \beta_p.
    pub beta_p: f64,
    /// Edge pedestal pressure p_{edge} (Pa).
    pub p_edge: f64,
    /// Central vacuum toroidal flux function F_0 = R_0 B_0 (T \cdot m).
    pub f0: f64,
    /// Diamagnetic/paramagnetic factor \delta_F in F^2(\bar{\psi}) = F_0^2 [1 + 2 \mu_0 \delta_F (p(\bar{\psi}) - p(0)) / B_0^2].
    pub delta_f: f64,
    /// Major radius of reference axis R_0 (m).
    pub r0: f64,
    /// Minor radius of plasma boundary a (m).
    pub a: f64,
    /// On-axis magnetic field B_0 (T).
    pub b0: f64,
    /// Absolute flux on magnetic axis \psi_{axis} (Wb/rad).
    pub psi_axis: f64,
    /// Absolute flux at plasma boundary \psi_{edge} (Wb/rad).
    pub psi_edge: f64,
}

impl PlasmaProfiles {
    /// Creates a standard tokamak profile configuration.
    pub fn new(
        p0: f64,
        alpha_p: f64,
        beta_p: f64,
        r0: f64,
        a: f64,
        b0: f64,
        psi_axis: f64,
        psi_edge: f64,
    ) -> Self {
        let f0 = r0 * b0;
        Self {
            p0,
            alpha_p,
            beta_p,
            p_edge: p0 * 0.01,
            f0,
            delta_f: -0.15, // mild diamagnetism
            r0,
            a,
            b0,
            psi_axis,
            psi_edge,
        }
    }

    /// Converts absolute poloidal flux \psi to normalized flux \bar{\psi} \in [0, 1].
    pub fn normalize_psi(&self, psi: f64) -> f64 {
        let denom = self.psi_edge - self.psi_axis;
        if denom.abs() < 1.0e-12 {
            return 0.0;
        }
        let psi_n = (psi - self.psi_axis) / denom;
        psi_n.clamp(0.0, 1.0)
    }

    /// Evaluates thermal pressure p(\psi) in Pascals.
    pub fn pressure(&self, psi_norm: f64) -> f64 {
        let s = psi_norm.clamp(0.0, 1.0);
        let base = (1.0 - s.powf(self.alpha_p)).max(0.0);
        self.p_edge + (self.p0 - self.p_edge) * base.powf(self.beta_p)
    }

    /// Evaluates the pressure derivative dp/d\psi in Pa / (Wb/rad).
    pub fn pressure_prime(&self, psi_norm: f64) -> f64 {
        let s = psi_norm.clamp(1.0e-5, 0.99999);
        let dpsi = self.psi_edge - self.psi_axis;
        if dpsi.abs() < 1.0e-12 {
            return 0.0;
        }

        let base = (1.0 - s.powf(self.alpha_p)).max(0.0);
        if base <= 1.0e-12 {
            return 0.0;
        }

        let dp_ds = -(self.p0 - self.p_edge) * self.beta_p * base.powf(self.beta_p - 1.0) * self.alpha_p * s.powf(self.alpha_p - 1.0);
        dp_ds / dpsi
    }

    /// Evaluates toroidal flux function F(\psi) = R B_\phi.
    pub fn f_function(&self, psi_norm: f64) -> f64 {
        let p = self.pressure(psi_norm);
        let p_core = self.pressure(0.0);
        let term = 1.0 + 2.0 * MU_0 * self.delta_f * (p - p_core) / (self.b0 * self.b0);
        self.f0 * term.max(0.01).sqrt()
    }

    /// Evaluates F(\psi) F'(\psi) = (1/2) d(F^2)/d\psi.
    pub fn ff_prime(&self, psi_norm: f64) -> f64 {
        let dpsi = self.psi_edge - self.psi_axis;
        if dpsi.abs() < 1.0e-12 {
            return 0.0;
        }

        let dp_dpsi = self.pressure_prime(psi_norm);
        // From F^2 = F0^2 [ 1 + 2 mu_0 delta_F (p - p0) / B0^2 ]
        // (1/2) d(F^2)/dpsi = F0^2 * mu_0 * delta_F * (dp/dpsi) / B0^2
        // Since F0 = R0 * B0, (F0/B0)^2 = R0^2
        self.r0 * self.r0 * MU_0 * self.delta_f * dp_dpsi
    }

    /// Computes toroidal current density J_\phi(R, \psi) = R p'(\psi) + \frac{1}{\mu_0 R} F F'(\psi).
    pub fn current_density(&self, r: f64, psi_norm: f64) -> f64 {
        if r <= 1.0e-6 {
            return 0.0;
        }
        let pp = self.pressure_prime(psi_norm);
        let ffp = self.ff_prime(psi_norm);
        r * pp + ffp / (MU_0 * r)
    }

    /// Analytical safety factor profile q(r/a) parameterization:
    /// q(\rho) = q_0 + (q_a - q_0) \rho^2 / (1 - \lambda \rho^2).
    pub fn safety_factor_model(&self, rho: f64, q0: f64, qa: f64) -> f64 {
        let r = rho.clamp(0.0, 1.0);
        q0 + (qa - q0) * r * r
    }

    /// Magnetic shear s(\rho) = (\rho / q) dq/d\rho for parabolic q profile:
    /// s(\rho) = \frac{2 (q_a - q_0) \rho^2}{q_0 + (q_a - q_0) \rho^2}.
    pub fn magnetic_shear_model(&self, rho: f64, q0: f64, qa: f64) -> f64 {
        let r = rho.clamp(0.0, 1.0);
        let q = self.safety_factor_model(r, q0, qa);
        if q <= 1.0e-6 {
            return 0.0;
        }
        let dq_drho = 2.0 * (qa - q0) * r;
        (r / q) * dq_drho
    }

    /// Analytical Shafranov shift \Delta(r):
    /// \Delta(r) = \Delta_0 (1 - (r/a)^2), where \Delta_0 \approx \frac{a^2}{2 R_0} (\beta_p + \ell_i / 2).
    pub fn shafranov_shift_model(&self, r_minor: f64, beta_p: f64, li: f64) -> f64 {
        let delta_max = (self.a * self.a / (2.0 * self.r0)) * (beta_p + 0.5 * li);
        let frac = (r_minor / self.a).clamp(0.0, 1.0);
        delta_max * (1.0 - frac * frac)
    }

    /// Calculates volume-averaged pressure <p> = \int p dV / V for standard profiles.
    pub fn volume_averaged_pressure(&self) -> f64 {
        // Simpson integration over normalized radius rho \in [0, 1] with dV \propto rho d\rho
        let n = 101;
        let drho = 1.0 / ((n - 1) as f64);
        let mut p_weight_sum = 0.0;
        let mut vol_weight_sum = 0.0;

        for i in 0..n {
            let rho = i as f64 * drho;
            let weight = if i == 0 || i == n - 1 { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 };
            let psi_n = rho * rho; // flux roughly scales with rho^2
            let p = self.pressure(psi_n);
            p_weight_sum += weight * p * rho;
            vol_weight_sum += weight * rho;
        }

        p_weight_sum / vol_weight_sum
    }

    /// Toroidal beta \beta_t = 2 \mu_0 <p> / B_0^2.
    pub fn toroidal_beta(&self) -> f64 {
        let p_avg = self.volume_averaged_pressure();
        2.0 * MU_0 * p_avg / (self.b0 * self.b0)
    }

    /// Troyon normalized beta \beta_N = \beta_t [%] / (I_p [MA] / (a [m] B_0 [T])).
    pub fn troyon_normalized_beta(&self, ip_amperes: f64) -> f64 {
        let beta_t_percent = self.toroidal_beta() * 100.0;
        let ip_ma = ip_amperes * 1.0e-6;
        if ip_ma <= 1.0e-6 {
            return 0.0;
        }
        beta_t_percent / (ip_ma / (self.a * self.b0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiles_pressure_and_derivative() {
        let profiles = PlasmaProfiles::new(
            1.0e5, // 100 kPa
            2.0,
            1.5,
            6.2,
            2.0,
            5.3,
            0.0,
            10.0,
        );

        let p_core = profiles.pressure(0.0);
        assert!((p_core - 1.0e5).abs() < 1.0e-3);

        let p_edge = profiles.pressure(1.0);
        assert!((p_edge - 1.0e3).abs() < 1.0e-3);

        let dp = profiles.pressure_prime(0.5);
        // Pressure is decreasing with radius, derivative should be non-positive
        assert!(dp <= 0.0, "dp/dpsi must be negative inside plasma: {}", dp);
    }

    #[test]
    fn test_safety_factor_and_shear() {
        let profiles = PlasmaProfiles::new(1.0e5, 2.0, 1.5, 6.2, 2.0, 5.3, 0.0, 10.0);
        let q0 = 1.05;
        let qa = 3.5;

        let q_axis = profiles.safety_factor_model(0.0, q0, qa);
        assert_eq!(q_axis, q0);

        let q_edge = profiles.safety_factor_model(1.0, q0, qa);
        assert_eq!(q_edge, qa);

        let shear_axis = profiles.magnetic_shear_model(0.0, q0, qa);
        assert_eq!(shear_axis, 0.0);

        let shear_edge = profiles.magnetic_shear_model(1.0, q0, qa);
        assert!(shear_edge > 0.0, "Shear at edge should be positive: {}", shear_edge);
    }
}

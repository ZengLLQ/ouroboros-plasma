//! Shear Alfvén wave dispersion relation and continuous spectrum profiles.

use serde::{Deserialize, Serialize};
use crate::constants::MU_0;

/// Shear Alfvén dispersion relation and continuum branches in tokamak geometry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlfvenContinuum {
    pub r0: f64,
    pub a: f64,
    pub b0: f64,
    pub toroidal_n: i32,
    pub poloidal_m_list: Vec<i32>,
    pub num_radial_points: usize,
}

impl AlfvenContinuum {
    /// Creates a new continuum analyzer for toroidal mode number n.
    pub fn new(r0: f64, a: f64, b0: f64, toroidal_n: i32, poloidal_m_list: Vec<i32>) -> Self {
        Self {
            r0,
            a,
            b0,
            toroidal_n,
            poloidal_m_list,
            num_radial_points: 200,
        }
    }

    /// Evaluates Alfvén speed v_A(r) = B_0 / \sqrt{\mu_0 \rho(r)}.
    /// Assumes parabolic density profile: \rho(r) = \rho_0 [1 - 0.8 (r/a)^2].
    pub fn alfven_speed(&self, r_minor: f64, rho_core: f64) -> f64 {
        let frac = (r_minor / self.a).clamp(0.0, 0.999);
        let rho_local = rho_core * (1.0 - 0.8 * frac * frac).max(0.1);
        self.b0 / (MU_0 * rho_local).sqrt()
    }

    /// Parallel wave vector k_{\parallel m, n}(r) = (n - m / q(r)) / R_0.
    pub fn parallel_wave_vector(&self, _r_minor: f64, m: i32, q_val: f64) -> f64 {
        if q_val <= 1.0e-5 {
            return 0.0;
        }
        (self.toroidal_n as f64 - m as f64 / q_val) / self.r0
    }

    /// Evaluates continuous spectrum frequency \omega_{A, m}(r) = |k_{\parallel m, n}(r)| v_A(r).
    pub fn continuum_frequency(&self, r_minor: f64, m: i32, q_val: f64, rho_core: f64) -> f64 {
        let k_par = self.parallel_wave_vector(r_minor, m, q_val);
        let va = self.alfven_speed(r_minor, rho_core);
        k_par.abs() * va
    }

    /// Computes continuum frequency curves \omega_m(r) across radial mesh for all specified m harmonics.
    pub fn compute_continuum_curves<FQ>(
        &self,
        rho_core: f64,
        q_profile: FQ,
    ) -> (Vec<f64>, Vec<(i32, Vec<f64>)>)
    where
        FQ: Fn(f64) -> f64,
    {
        let nr = self.num_radial_points;
        let dr = self.a / ((nr - 1) as f64);
        let r_coords: Vec<f64> = (0..nr).map(|i| i as f64 * dr).collect();

        let mut curves = Vec::with_capacity(self.poloidal_m_list.len());

        for &m in &self.poloidal_m_list {
            let mut freqs = Vec::with_capacity(nr);
            for &r in &r_coords {
                let q = q_profile(r / self.a);
                let w = self.continuum_frequency(r, m, q, rho_core);
                freqs.push(w);
            }
            curves.push((m, freqs));
        }

        (r_coords, curves)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alfven_speed_and_wave_vector() {
        let continuum = AlfvenContinuum::new(6.2, 2.0, 5.3, 2, vec![2, 3]);
        let rho_core = 1.0e-7; // ~ 10^20 m^-3 D-T ions

        let va_axis = continuum.alfven_speed(0.0, rho_core);
        assert!(va_axis > 1.0e7 && va_axis < 2.0e7, "Alfvén speed should be ~ 1.5e7 m/s: {}", va_axis);

        // At q = 1.5, n = 2, m = 3: k_parallel = (2 - 3/1.5) / R0 = (2 - 2) / R0 = 0!
        let k_par = continuum.parallel_wave_vector(1.0, 3, 1.5);
        assert!(k_par.abs() < 1e-12, "Parallel wave vector should vanish at resonance");
    }
}

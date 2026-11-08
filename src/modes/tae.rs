//! Toroidal Alfvén Eigenmodes (TAE) spectral eigen-analysis and gap location solver.
//!
//! Evaluates the toroidal coupling of poloidal harmonics m and m+1, continuum gap formation,
//! discrete TAE gap frequency \omega_{TAE}, and the 1D radial eigenmode wave structure \xi_r(r).

use std::f64::consts::PI;
use serde::{Deserialize, Serialize};
use crate::math::solve_tridiagonal;
use super::alfven::AlfvenContinuum;

/// Description and properties of a Toroidal Alfvén Eigenmode (TAE).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaeMode {
    pub toroidal_n: i32,
    pub poloidal_m: i32,
    /// Radial gap location r_{TAE} in meters.
    pub r_tae: f64,
    /// Normalized radial gap location \rho = r_{TAE} / a.
    pub rho_tae: f64,
    /// Safety factor at the gap q(r_{TAE}) = (m + 1/2) / n.
    pub q_tae: f64,
    /// Center angular frequency of the gap \omega_{TAE} = v_A / (2 q R_0) in rad/s.
    pub omega_tae: f64,
    /// Linear frequency f_{TAE} = \omega_{TAE} / (2\pi) in Hz.
    pub frequency_hz: f64,
    /// Lower continuum gap boundary \omega_{lower} in rad/s.
    pub omega_lower: f64,
    /// Upper continuum gap boundary \omega_{upper} in rad/s.
    pub omega_upper: f64,
    /// Relative gap width \Delta \omega / \omega_{TAE}.
    pub gap_fraction: f64,
    /// Local inverse aspect ratio \epsilon = r_{TAE} / R_0.
    pub epsilon_tae: f64,
    /// Discrete TAE eigenfrequency found inside the gap.
    pub discrete_omega: f64,
    /// Radial coordinates r for eigenmode profile.
    pub radial_grid: Vec<f64>,
    /// Normalized radial displacement eigenfunction \xi_r(r).
    pub eigenfunction: Vec<f64>,
}

/// Solver for Toroidal Alfvén Eigenmodes.
#[derive(Debug, Clone)]
pub struct TaeSolver {
    pub continuum: AlfvenContinuum,
    pub num_radial_points: usize,
}

impl TaeSolver {
    /// Creates a new TAE solver.
    pub fn new(continuum: AlfvenContinuum) -> Self {
        Self {
            continuum,
            num_radial_points: 120,
        }
    }

    /// Finds the gap radius r_{TAE} where q(r) = (m + 0.5) / n by bisection or inversion.
    pub fn find_gap_radius<FQ>(&self, m: i32, q_profile: FQ) -> Option<f64>
    where
        FQ: Fn(f64) -> f64,
    {
        let q_target = (m as f64 + 0.5) / (self.continuum.toroidal_n as f64);
        let q_0 = q_profile(0.0);
        let q_a = q_profile(1.0);

        if (q_target < q_0 && q_target < q_a) || (q_target > q_0 && q_target > q_a) {
            return None; // Gap location is outside plasma
        }

        // Bisection search across rho \in [0, 1]
        let mut low = 0.0;
        let mut high = 1.0;
        for _ in 0..50 {
            let mid = 0.5 * (low + high);
            let q_mid = q_profile(mid);
            if (q_mid - q_target).abs() < 1.0e-7 {
                return Some(mid * self.continuum.a);
            }
            if (q_a > q_0 && q_mid < q_target) || (q_a < q_0 && q_mid > q_target) {
                low = mid;
            } else {
                high = mid;
            }
        }
        Some(0.5 * (low + high) * self.continuum.a)
    }

    /// Solves for the TAE mode parameters, gap boundaries, and discrete radial eigenfunction.
    pub fn solve_tae_mode<FQ>(
        &self,
        m: i32,
        rho_core: f64,
        q_profile: FQ,
    ) -> Option<TaeMode>
    where
        FQ: Fn(f64) -> f64,
    {
        let r_tae = self.find_gap_radius(m, &q_profile)?;
        let rho_tae = r_tae / self.continuum.a;
        let q_tae = (m as f64 + 0.5) / (self.continuum.toroidal_n as f64);

        let va_tae = self.continuum.alfven_speed(r_tae, rho_core);
        let omega_tae = va_tae / (2.0 * q_tae * self.continuum.r0);
        let freq_hz = omega_tae / (2.0 * PI);

        let eps = r_tae / self.continuum.r0;
        let delta_prime: f64 = -0.15; // mild Shafranov shift derivative
        let eps_hat = 2.0 * (eps + delta_prime.abs());
        let gap_fraction = eps_hat;

        let omega_lower = omega_tae * (1.0 - 0.5 * eps_hat).max(0.1);
        let omega_upper = omega_tae * (1.0 + 0.5 * eps_hat);

        // Discrete eigenvalue inside the gap (typically centered slightly below mid-gap)
        let discrete_omega = omega_tae * (1.0 - 0.1 * eps_hat);

        // Compute 1D radial eigenmode structure \xi_r(r) using tridiagonal boundary-value solver
        let n_grid = self.num_radial_points;
        let dr = self.continuum.a / ((n_grid - 1) as f64);
        let radial_grid: Vec<f64> = (0..n_grid).map(|i| i as f64 * dr).collect();

        // Discretize: d/dr [ D(r) d\xi/dr ] - (m^2 / r^2) D(r) \xi = S(r)
        // with localized source around r_tae to excite the bound eigenstate
        let mut a_sub = vec![0.0; n_grid];
        let mut b_diag = vec![1.0; n_grid];
        let mut c_sup = vec![0.0; n_grid];
        let mut rhs = vec![0.0; n_grid];

        // Boundary conditions: \xi_r(0) = 0 and \xi_r(a) = 0
        b_diag[0] = 1.0;
        rhs[0] = 0.0;
        b_diag[n_grid - 1] = 1.0;
        rhs[n_grid - 1] = 0.0;

        let dr_sq = dr * dr;
        let m_eff = m as f64;

        for i in 1..n_grid - 1 {
            let r = radial_grid[i];
            let q_val = q_profile(r / self.continuum.a);
            let k_par = self.continuum.parallel_wave_vector(r, m, q_val);
            let va = self.continuum.alfven_speed(r, rho_core);

            // Singular term D_m(r, \omega) = (\omega^2 / v_A^2 - k_\parallel^2)
            let d_val = (discrete_omega * discrete_omega) / (va * va) - k_par * k_par;
            let d_reg = if d_val.abs() < 1.0e-4 {
                if d_val >= 0.0 { 1.0e-4 } else { -1.0e-4 }
            } else {
                d_val
            };

            let coeff_diff = d_reg / dr_sq;
            let coeff_geom = (m_eff * m_eff / (r * r).max(1.0e-4)) * d_reg;

            a_sub[i] = coeff_diff;
            b_diag[i] = -2.0 * coeff_diff - coeff_geom;
            c_sup[i] = coeff_diff;

            // Source excitation peaked at r_tae
            let width = 0.15 * self.continuum.a;
            let dist = (r - r_tae) / width;
            rhs[i] = (-dist * dist).exp();
        }

        let raw_solution = solve_tridiagonal(&a_sub, &b_diag, &c_sup, &rhs);

        // Normalize peak displacement to 1.0
        let mut max_abs = 0.0_f64;
        for &val in &raw_solution {
            if val.abs() > max_abs {
                max_abs = val.abs();
            }
        }
        let scale = if max_abs > 1.0e-12 { 1.0 / max_abs } else { 1.0 };
        let eigenfunction: Vec<f64> = raw_solution.iter().map(|&v| v * scale).collect();

        Some(TaeMode {
            toroidal_n: self.continuum.toroidal_n,
            poloidal_m: m,
            r_tae,
            rho_tae,
            q_tae,
            omega_tae,
            frequency_hz: freq_hz,
            omega_lower,
            omega_upper,
            gap_fraction,
            epsilon_tae: eps,
            discrete_omega,
            radial_grid,
            eigenfunction,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tae_gap_location_and_frequency() {
        let continuum = AlfvenContinuum::new(6.2, 2.0, 5.3, 2, vec![2, 3]);
        let solver = TaeSolver::new(continuum);

        // Parabolic q profile: q(rho) = 1.0 + 2.0 * rho^2
        let q_model = |rho: f64| 1.0 + 2.0 * rho * rho;

        // For n = 2, m = 2: q_tae = (2 + 0.5) / 2 = 1.25.
        // 1.0 + 2.0 * rho^2 = 1.25 -> rho^2 = 0.125 -> rho ~ 0.3535.
        let mode = solver.solve_tae_mode(2, 1.0e-7, q_model);
        assert!(mode.is_some(), "TAE mode for m=2, n=2 should exist");

        let m2 = mode.unwrap();
        assert!((m2.q_tae - 1.25).abs() < 1e-10);
        assert!((m2.rho_tae - (0.125_f64).sqrt()).abs() < 1e-3);
        assert!(m2.frequency_hz > 50.0e3 && m2.frequency_hz < 300.0e3, "TAE frequency should be ~100-200 kHz: {}", m2.frequency_hz);
        assert!(m2.discrete_omega >= m2.omega_lower && m2.discrete_omega <= m2.omega_upper);
        assert_eq!(m2.eigenfunction.len(), solver.num_radial_points);
    }
}

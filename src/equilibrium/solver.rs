//! 2D Grad-Shafranov elliptic PDE solver and Soloviev analytical equilibrium.
//!
//! Solves the fundamental axisymmetric magnetostatic equilibrium equation:
//! \Delta^* \psi = -\mu_0 R^2 p'(\psi) - F(\psi) F'(\psi)
//! where \Delta^* \psi = \frac{\partial^2 \psi}{\partial R^2} - \frac{1}{R} \frac{\partial \psi}{\partial R} + \frac{\partial^2 \psi}{\partial Z^2}.

use serde::{Deserialize, Serialize};
use crate::constants::MU_0;
use crate::math::{interpolate_2d_bilinear, Vector3D};
use super::flux_surface::MillerGeometry;
use super::profiles::PlasmaProfiles;

/// Analytical Soloviev equilibrium with exact polynomial solution for verification.
///
/// Under constant p'(\psi) = -p_1 and FF'(\psi) = -f_1, an exact solution exists:
/// \psi(R, Z) = \frac{\mu_0 p_1}{2(1+\kappa^2)} [R^2 Z^2 + \frac{\kappa^2}{4}(R^2 - R_0^2)^2] + \psi_0
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolovievEquilibrium {
    pub r0: f64,
    pub z0: f64,
    pub kappa: f64,
    pub b0: f64,
    pub p1: f64,
    pub f1: f64,
    pub psi_axis: f64,
    pub psi_edge: f64,
}

impl SolovievEquilibrium {
    /// Creates a Soloviev equilibrium configuration.
    pub fn new(r0: f64, kappa: f64, b0: f64, p_core: f64, minor_radius: f64) -> Self {
        // Approximate p1 to give roughly p_core in the center
        let p1 = 2.0 * p_core / (minor_radius * minor_radius);
        let f1 = 0.0;
        let c = MU_0 * p1 / (2.0 * (1.0 + kappa * kappa));
        let psi_axis = 0.0;
        let psi_edge = c * (kappa * kappa / 4.0) * (2.0 * r0 * minor_radius).powi(2);

        Self {
            r0,
            z0: 0.0,
            kappa,
            b0,
            p1,
            f1,
            psi_axis,
            psi_edge: psi_edge.max(1.0),
        }
    }

    /// Evaluates poloidal flux \psi(R, Z) analytically.
    pub fn psi(&self, r: f64, z: f64) -> f64 {
        let dz = z - self.z0;
        let c = MU_0 * self.p1 / (2.0 * (1.0 + self.kappa * self.kappa));
        let r2 = r * r;
        let r0_2 = self.r0 * self.r0;
        let delta_r2 = r2 - r0_2;

        c * (r2 * dz * dz + (self.kappa * self.kappa / 4.0) * delta_r2 * delta_r2) + self.psi_axis
    }

    /// Evaluates the gradient (\partial \psi / \partial R, \partial \psi / \partial Z).
    pub fn grad_psi(&self, r: f64, z: f64) -> (f64, f64) {
        let dz = z - self.z0;
        let c = MU_0 * self.p1 / (2.0 * (1.0 + self.kappa * self.kappa));
        let r2 = r * r;
        let r0_2 = self.r0 * self.r0;
        let delta_r2 = r2 - r0_2;

        let dpsi_dr = c * (2.0 * r * dz * dz + self.kappa * self.kappa * delta_r2 * r);
        let dpsi_dz = c * (2.0 * r2 * dz);
        (dpsi_dr, dpsi_dz)
    }

    /// Evaluates magnetic field vector (B_R, B_\phi, B_Z) in cylindrical coordinates.
    pub fn magnetic_field(&self, r: f64, z: f64) -> Vector3D {
        if r <= 1.0e-5 {
            return Vector3D::new(0.0, self.b0, 0.0);
        }
        let (dpsi_dr, dpsi_dz) = self.grad_psi(r, z);
        let br = -dpsi_dz / r;
        let bz = dpsi_dr / r;
        let bphi = self.r0 * self.b0 / r; // vacuum 1/R approximation
        Vector3D::new(br, bphi, bz)
    }

    /// Analytically computes \Delta^* \psi = \partial^2 \psi / \partial R^2 - (1/R) \partial \psi / \partial R + \partial^2 \psi / \partial Z^2.
    pub fn delta_star_psi(&self, r: f64, z: f64) -> f64 {
        let _dz = z - self.z0;
        let c = MU_0 * self.p1 / (2.0 * (1.0 + self.kappa * self.kappa));
        let r2 = r * r;
        let _r0_2 = self.r0 * self.r0;

        // dpsi_dr = c * [2 r dz^2 + kappa^2 (r^2 - r0^2) r]
        // d2psi_dr2 = c * [2 dz^2 + kappa^2 (3 r^2 - r0^2)]
        // (1/r) dpsi_dr = c * [2 dz^2 + kappa^2 (r^2 - r0^2)]
        // d2psi_dr2 - (1/r) dpsi_dr = c * kappa^2 [3 r^2 - r0^2 - (r^2 - r0^2)] = c * kappa^2 * 2 r^2
        // d2psi_dz2 = c * 2 r^2
        // Sum = c * 2 r^2 (1 + kappa^2)
        // With c = mu_0 p1 / (2(1 + kappa^2)), this equals exactly mu_0 p1 r^2!
        let diff_r = c * (self.kappa * self.kappa * 2.0 * r2);
        let diff_z = c * (2.0 * r2);
        diff_r + diff_z
    }
}

/// 2D Numerical Finite Difference Grad-Shafranov Solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradShafranovSolver {
    pub nr: usize,
    pub nz: usize,
    pub r_min: f64,
    pub r_max: f64,
    pub z_min: f64,
    pub z_max: f64,
    pub dr: f64,
    pub dz: f64,
    pub r_coords: Vec<f64>,
    pub z_coords: Vec<f64>,
    pub psi: Vec<Vec<f64>>, // [nr][nz]
    pub magnetic_axis: (f64, f64),
    pub psi_axis: f64,
    pub psi_edge: f64,
    pub total_current_ip: f64,
}

impl GradShafranovSolver {
    /// Initializes a grid for the Grad-Shafranov solver.
    pub fn new(nr: usize, nz: usize, r_min: f64, r_max: f64, z_min: f64, z_max: f64) -> Self {
        assert!(nr >= 5 && nz >= 5, "Grid dimensions must be at least 5x5");
        assert!(r_max > r_min && z_max > z_min, "Invalid spatial bounds");

        let dr = (r_max - r_min) / ((nr - 1) as f64);
        let dz = (z_max - z_min) / ((nz - 1) as f64);

        let r_coords: Vec<f64> = (0..nr).map(|i| r_min + i as f64 * dr).collect();
        let z_coords: Vec<f64> = (0..nz).map(|j| z_min + j as f64 * dz).collect();

        let psi = vec![vec![0.0; nz]; nr];
        let r0 = 0.5 * (r_min + r_max);
        let z0 = 0.5 * (z_min + z_max);

        Self {
            nr,
            nz,
            r_min,
            r_max,
            z_min,
            z_max,
            dr,
            dz,
            r_coords,
            z_coords,
            psi,
            magnetic_axis: (r0, z0),
            psi_axis: 0.0,
            psi_edge: 1.0,
            total_current_ip: 0.0,
        }
    }

    /// Initializes boundary conditions and initial flux guess from a Miller shape boundary.
    pub fn initialize_from_miller(&mut self, miller: &MillerGeometry, target_psi_edge: f64) {
        self.psi_edge = target_psi_edge;
        self.psi_axis = 0.0;

        let a = miller.minor_radius;
        let kappa = miller.elongation;
        let r0 = miller.r0;
        let z0 = miller.z0;

        for i in 0..self.nr {
            let r = self.r_coords[i];
            for j in 0..self.nz {
                let z = self.z_coords[j];
                let dr_rel = (r - r0) / a;
                let dz_rel = (z - z0) / (kappa * a);
                let rho2 = dr_rel * dr_rel + dz_rel * dz_rel;
                if rho2 <= 1.0 {
                    // Parabolic core flux
                    self.psi[i][j] = target_psi_edge * rho2;
                } else {
                    self.psi[i][j] = target_psi_edge * (1.0 + (rho2 - 1.0).sqrt() * 0.5);
                }
            }
        }
    }

    /// Executes Picard non-linear iteration with Successive Over-Relaxation (SOR).
    pub fn solve(&mut self, profiles: &PlasmaProfiles, max_iterations: usize, tolerance: f64) -> (usize, f64) {
        let omega = 1.35; // Relaxation parameter for elliptic PDE
        let dr2 = self.dr * self.dr;
        let dz2 = self.dz * self.dz;
        let center_coeff = 2.0 / dr2 + 2.0 / dz2;

        let mut final_residual = 0.0;
        let mut iter_count = 0;

        for iter in 0..max_iterations {
            iter_count = iter + 1;
            let mut max_diff: f64 = 0.0;

            // Find current magnetic axis (minimum or maximum of psi inside domain)
            self.update_magnetic_axis();

            for i in 1..self.nr - 1 {
                let r = self.r_coords[i];
                let coeff_r_plus = 1.0 / dr2 - 1.0 / (2.0 * r * self.dr);
                let coeff_r_minus = 1.0 / dr2 + 1.0 / (2.0 * r * self.dr);
                let coeff_z = 1.0 / dz2;

                for j in 1..self.nz - 1 {
                    let psi_val = self.psi[i][j];
                    let psi_n = profiles.normalize_psi(psi_val);

                    // Right hand side source: -\mu_0 R^2 p'(\psi) - F F'(\psi)
                    let pp = profiles.pressure_prime(psi_n);
                    let ffp = profiles.ff_prime(psi_n);
                    let source = -MU_0 * r * r * pp - ffp;

                    let stencil_sum = coeff_r_plus * self.psi[i + 1][j]
                        + coeff_r_minus * self.psi[i - 1][j]
                        + coeff_z * (self.psi[i][j + 1] + self.psi[i][j - 1]);

                    let psi_target = (stencil_sum - source) / center_coeff;
                    let psi_new = (1.0 - omega) * psi_val + omega * psi_target;

                    let diff = (psi_new - psi_val).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                    self.psi[i][j] = psi_new;
                }
            }

            final_residual = max_diff;
            if max_diff < tolerance {
                break;
            }
        }

        self.update_magnetic_axis();
        self.compute_total_plasma_current(profiles);
        (iter_count, final_residual)
    }

    /// Finds the location of the magnetic axis (extrema of \psi).
    fn update_magnetic_axis(&mut self) {
        let mut min_val = f64::INFINITY;
        let mut min_i = self.nr / 2;
        let mut min_j = self.nz / 2;

        for i in 1..self.nr - 1 {
            for j in 1..self.nz - 1 {
                if self.psi[i][j] < min_val {
                    min_val = self.psi[i][j];
                    min_i = i;
                    min_j = j;
                }
            }
        }

        self.magnetic_axis = (self.r_coords[min_i], self.z_coords[min_j]);
        self.psi_axis = min_val;
    }

    /// Integrates total toroidal plasma current: I_p = \int J_\phi dR dZ.
    fn compute_total_plasma_current(&mut self, profiles: &PlasmaProfiles) {
        let mut ip_sum = 0.0;
        let d_area = self.dr * self.dz;

        for i in 1..self.nr - 1 {
            let r = self.r_coords[i];
            for j in 1..self.nz - 1 {
                let psi_val = self.psi[i][j];
                let psi_n = profiles.normalize_psi(psi_val);
                if psi_n <= 1.0 {
                    let j_phi = profiles.current_density(r, psi_n);
                    ip_sum += j_phi * d_area;
                }
            }
        }
        self.total_current_ip = ip_sum.abs();
    }

    /// Evaluates poloidal flux \psi at arbitrary point (R, Z).
    pub fn evaluate_psi(&self, r: f64, z: f64) -> f64 {
        interpolate_2d_bilinear(r, z, &self.r_coords, &self.z_coords, &self.psi)
    }

    /// Evaluates the gradient (\partial \psi / \partial R, \partial \psi / \partial Z) numerically.
    pub fn evaluate_grad_psi(&self, r: f64, z: f64) -> (f64, f64) {
        let eps_r = self.dr * 0.5;
        let eps_z = self.dz * 0.5;

        let psi_r_plus = self.evaluate_psi(r + eps_r, z);
        let psi_r_minus = self.evaluate_psi(r - eps_r, z);
        let psi_z_plus = self.evaluate_psi(r, z + eps_z);
        let psi_z_minus = self.evaluate_psi(r, z - eps_z);

        let dpsi_dr = (psi_r_plus - psi_r_minus) / (2.0 * eps_r);
        let dpsi_dz = (psi_z_plus - psi_z_minus) / (2.0 * eps_z);
        (dpsi_dr, dpsi_dz)
    }

    /// Evaluates the local magnetic field vector \mathbf{B}(R, Z).
    pub fn evaluate_b_field(&self, r: f64, z: f64, profiles: &PlasmaProfiles) -> Vector3D {
        if r <= 1.0e-5 {
            return Vector3D::new(0.0, profiles.f0 / self.r_min, 0.0);
        }

        let (dpsi_dr, dpsi_dz) = self.evaluate_grad_psi(r, z);
        let br = -dpsi_dz / r;
        let bz = dpsi_dr / r;

        let psi_val = self.evaluate_psi(r, z);
        let psi_n = profiles.normalize_psi(psi_val);
        let bphi = profiles.f_function(psi_n) / r;

        Vector3D::new(br, bphi, bz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soloviev_analytical_grad_shafranov() {
        let sol = SolovievEquilibrium::new(6.2, 1.7, 5.3, 1.5e5, 2.0);
        let r = 6.5;
        let z = 0.5;

        // Verify \Delta^* \psi = \mu_0 p_1 R^2
        let delta_star = sol.delta_star_psi(r, z);
        let expected_rhs = MU_0 * sol.p1 * r * r;
        assert!((delta_star - expected_rhs).abs() / expected_rhs < 1.0e-10);

        // Verify magnetic field at magnetic axis
        let b_axis = sol.magnetic_field(6.2, 0.0);
        assert!(b_axis.x.abs() < 1e-10); // B_R = 0 on midplane at axis
        assert!(b_axis.z.abs() < 1e-10); // B_Z = 0 at magnetic axis
        assert!((b_axis.y - 5.3).abs() < 1e-10); // B_\phi = B_0 at axis
    }

    #[test]
    fn test_numerical_solver_convergence() {
        let mut solver = GradShafranovSolver::new(31, 31, 4.0, 8.4, -3.0, 3.0);
        let miller = MillerGeometry::new(6.2, 0.0, 2.0, 1.6, 0.3, 0.1);
        solver.initialize_from_miller(&miller, 10.0);

        let profiles = PlasmaProfiles::new(2.0e5, 2.0, 1.5, 6.2, 2.0, 5.3, 0.0, 10.0);
        let (iters, res) = solver.solve(&profiles, 200, 1.0e-4);

        assert!(iters > 0);
        assert!(res < 1.0e-2, "Final residual should be small: {}", res);
        assert!(solver.total_current_ip > 0.0, "Total plasma current must be positive");
    }
}

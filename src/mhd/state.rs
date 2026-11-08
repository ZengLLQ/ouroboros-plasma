//! Magnetohydrodynamic plasma state variables on 2D computational domain.

use serde::{Deserialize, Serialize};
use crate::constants::{ADIABATIC_INDEX_GAMMA, MU_0};
use crate::math::Vector3D;
use super::grid::MhdGrid;

/// Discrete 2D field container for MHD variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MhdState {
    pub nr: usize,
    pub nz: usize,
    /// Mass density \rho (kg / m^3).
    pub rho: Vec<Vec<f64>>,
    /// Radial velocity v_R (m/s).
    pub v_r: Vec<Vec<f64>>,
    /// Toroidal velocity v_\phi (m/s).
    pub v_phi: Vec<Vec<f64>>,
    /// Vertical velocity v_Z (m/s).
    pub v_z: Vec<Vec<f64>>,
    /// Kinetic plasma pressure p (Pa).
    pub pressure: Vec<Vec<f64>>,
    /// Radial magnetic field B_R (T).
    pub b_r: Vec<Vec<f64>>,
    /// Toroidal magnetic field B_\phi (T).
    pub b_phi: Vec<Vec<f64>>,
    /// Vertical magnetic field B_Z (T).
    pub b_z: Vec<Vec<f64>>,
    /// Physical simulation time elapsed (s).
    pub time: f64,
}

impl MhdState {
    /// Initializes an MHD state with uniform background values on the specified grid.
    pub fn new_uniform(
        grid: &MhdGrid,
        rho_val: f64,
        pressure_val: f64,
        b_toroidal: f64,
    ) -> Self {
        let nr = grid.nr;
        let nz = grid.nz;

        let rho = vec![vec![rho_val; nz]; nr];
        let v_r = vec![vec![0.0; nz]; nr];
        let v_phi = vec![vec![0.0; nz]; nr];
        let v_z = vec![vec![0.0; nz]; nr];
        let pressure = vec![vec![pressure_val; nz]; nr];
        let b_r = vec![vec![0.0; nz]; nr];
        let b_z = vec![vec![0.0; nz]; nr];

        // 1/R vacuum toroidal field
        let r0 = grid.r_coords[nr / 2];
        let mut b_phi = vec![vec![0.0; nz]; nr];
        for i in 0..nr {
            let r = grid.r_coords[i];
            let val = b_toroidal * r0 / r;
            for j in 0..nz {
                b_phi[i][j] = val;
            }
        }

        Self {
            nr,
            nz,
            rho,
            v_r,
            v_phi,
            v_z,
            pressure,
            b_r,
            b_phi,
            b_z,
            time: 0.0,
        }
    }

    /// Evaluates sound speed c_s = \sqrt{\gamma p / \rho} at (i, j).
    pub fn sound_speed(&self, i: usize, j: usize) -> f64 {
        let p = self.pressure[i][j].max(1.0e-10);
        let rho = self.rho[i][j].max(1.0e-15);
        (ADIABATIC_INDEX_GAMMA * p / rho).sqrt()
    }

    /// Evaluates Alfvén speed v_A = |\mathbf{B}| / \sqrt{\mu_0 \rho} at (i, j).
    pub fn alfven_speed(&self, i: usize, j: usize) -> f64 {
        let br = self.b_r[i][j];
        let bphi = self.b_phi[i][j];
        let bz = self.b_z[i][j];
        let b_mag2 = br * br + bphi * bphi + bz * bz;
        let rho = self.rho[i][j].max(1.0e-15);
        (b_mag2 / (MU_0 * rho)).sqrt()
    }

    /// Evaluates fast magnetosonic speed v_{fast} = \sqrt{c_s^2 + v_A^2} at (i, j).
    pub fn fast_magnetosonic_speed(&self, i: usize, j: usize) -> f64 {
        let cs = self.sound_speed(i, j);
        let va = self.alfven_speed(i, j);
        (cs * cs + va * va).sqrt()
    }

    /// Evaluates current density \mathbf{J} = \frac{1}{\mu_0} \nabla \times \mathbf{B} at cell (i, j).
    pub fn current_density(&self, i: usize, j: usize, grid: &MhdGrid) -> Vector3D {
        if grid.is_boundary(i, j) {
            return Vector3D::zero();
        }

        let r = grid.r_coords[i];
        let dr2 = 2.0 * grid.dr;
        let dz2 = 2.0 * grid.dz;

        // J_R = - (1/\mu_0) \partial B_\phi / \partial Z
        let dbphi_dz = (self.b_phi[i][j + 1] - self.b_phi[i][j - 1]) / dz2;
        let jr = -dbphi_dz / MU_0;

        // J_\phi = (1/\mu_0) (\partial B_R / \partial Z - \partial B_Z / \partial R)
        let dbr_dz = (self.b_r[i][j + 1] - self.b_r[i][j - 1]) / dz2;
        let dbz_dr = (self.b_z[i + 1][j] - self.b_z[i - 1][j]) / dr2;
        let jphi = (dbr_dz - dbz_dr) / MU_0;

        // J_Z = (1/\mu_0) (1/R) \partial (R B_\phi) / \partial R
        let r_p = grid.r_coords[i + 1];
        let r_m = grid.r_coords[i - 1];
        let rb_p = r_p * self.b_phi[i + 1][j];
        let rb_m = r_m * self.b_phi[i - 1][j];
        let jz = (rb_p - rb_m) / (dr2 * r * MU_0);

        Vector3D::new(jr, jphi, jz)
    }

    /// Calculates total stored thermal energy W_{th} = \int \frac{p}{\gamma - 1} dV.
    pub fn total_thermal_energy(&self, grid: &MhdGrid) -> f64 {
        let mut total = 0.0;
        let denom = ADIABATIC_INDEX_GAMMA - 1.0;
        for i in 1..self.nr - 1 {
            let v_cell = grid.cell_volume(i);
            for j in 1..self.nz - 1 {
                total += (self.pressure[i][j] / denom) * v_cell;
            }
        }
        total
    }

    /// Calculates total magnetic energy W_B = \int \frac{|\mathbf{B}|^2}{2\mu_0} dV.
    pub fn total_magnetic_energy(&self, grid: &MhdGrid) -> f64 {
        let mut total = 0.0;
        for i in 1..self.nr - 1 {
            let v_cell = grid.cell_volume(i);
            for j in 1..self.nz - 1 {
                let br = self.b_r[i][j];
                let bphi = self.b_phi[i][j];
                let bz = self.b_z[i][j];
                let b2 = br * br + bphi * bphi + bz * bz;
                total += (b2 / (2.0 * MU_0)) * v_cell;
            }
        }
        total
    }

    /// Calculates total kinetic energy W_{kin} = \int \frac{1}{2} \rho |\mathbf{v}|^2 dV.
    pub fn total_kinetic_energy(&self, grid: &MhdGrid) -> f64 {
        let mut total = 0.0;
        for i in 1..self.nr - 1 {
            let v_cell = grid.cell_volume(i);
            for j in 1..self.nz - 1 {
                let vr = self.v_r[i][j];
                let vphi = self.v_phi[i][j];
                let vz = self.v_z[i][j];
                let v2 = vr * vr + vphi * vphi + vz * vz;
                total += 0.5 * self.rho[i][j] * v2 * v_cell;
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mhd_characteristic_speeds() {
        let grid = MhdGrid::new(11, 11, 5.0, 7.0, -1.0, 1.0);
        let rho_val = 1.0e-7;
        let p_val = 1.0e4;
        let bt = 5.0;
        let state = MhdState::new_uniform(&grid, rho_val, p_val, bt);

        let cs = state.sound_speed(5, 5);
        let va = state.alfven_speed(5, 5);
        let vfast = state.fast_magnetosonic_speed(5, 5);

        assert!(cs > 0.0);
        assert!(va > 0.0);
        assert!(vfast > cs && vfast > va);
    }
}

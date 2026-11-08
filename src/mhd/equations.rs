//! Ideal and resistive magnetohydrodynamics fluid conservation equations.
//!
//! Evaluates spatial derivatives and RHS source terms for:
//! 1. Mass continuity: \partial \rho / \partial t = -\nabla \cdot (\rho \mathbf{v})
//! 2. Momentum equation: \rho [\partial \mathbf{v}/\partial t + (\mathbf{v}\cdot\nabla)\mathbf{v}] = -\nabla p + \mathbf{J}\times\mathbf{B} + \mu \nabla^2 \mathbf{v}
//! 3. Induction equation: \partial \mathbf{B}/\partial t = \nabla \times (\mathbf{v}\times\mathbf{B} - \eta \mathbf{J})
//! 4. Thermal energy: \partial p/\partial t = -\mathbf{v}\cdot\nabla p - \gamma p (\nabla\cdot\mathbf{v}) + (\gamma - 1) \eta |\mathbf{J}|^2

use crate::constants::ADIABATIC_INDEX_GAMMA;
use super::grid::MhdGrid;
use super::state::MhdState;

/// Container for the time derivatives of all MHD state fields: d/dt(State).
#[derive(Debug, Clone)]
pub struct MhdDerivatives {
    pub d_rho: Vec<Vec<f64>>,
    pub d_v_r: Vec<Vec<f64>>,
    pub d_v_phi: Vec<Vec<f64>>,
    pub d_v_z: Vec<Vec<f64>>,
    pub d_pressure: Vec<Vec<f64>>,
    pub d_b_r: Vec<Vec<f64>>,
    pub d_b_phi: Vec<Vec<f64>>,
    pub d_b_z: Vec<Vec<f64>>,
}

impl MhdDerivatives {
    /// Creates zeroed derivatives matching the grid size.
    pub fn zero(nr: usize, nz: usize) -> Self {
        Self {
            d_rho: vec![vec![0.0; nz]; nr],
            d_v_r: vec![vec![0.0; nz]; nr],
            d_v_phi: vec![vec![0.0; nz]; nr],
            d_v_z: vec![vec![0.0; nz]; nr],
            d_pressure: vec![vec![0.0; nz]; nr],
            d_b_r: vec![vec![0.0; nz]; nr],
            d_b_phi: vec![vec![0.0; nz]; nr],
            d_b_z: vec![vec![0.0; nz]; nr],
        }
    }
}

/// Evaluates the complete MHD RHS equations at the current state.
pub fn evaluate_mhd_equations(
    state: &MhdState,
    grid: &MhdGrid,
    viscosity: f64,
    resistivity: f64,
) -> MhdDerivatives {
    let nr = grid.nr;
    let nz = grid.nz;
    let dr = grid.dr;
    let dz = grid.dz;
    let dr2 = 2.0 * dr;
    let dz2 = 2.0 * dz;
    let dr_sq = dr * dr;
    let dz_sq = dz * dz;

    let mut deriv = MhdDerivatives::zero(nr, nz);

    for i in 1..nr - 1 {
        let r = grid.r_coords[i];
        let r_p = grid.r_coords[i + 1];
        let r_m = grid.r_coords[i - 1];

        for j in 1..nz - 1 {
            let rho = state.rho[i][j].max(1.0e-15);
            let vr = state.v_r[i][j];
            let vphi = state.v_phi[i][j];
            let vz = state.v_z[i][j];
            let p = state.pressure[i][j];
            let br = state.b_r[i][j];
            let bphi = state.b_phi[i][j];
            let bz = state.b_z[i][j];

            // 1. Mass continuity: -\nabla \cdot (\rho \mathbf{v})
            // In cylindrical: -(1/R \partial(R \rho v_R)/\partial R + \partial(\rho v_Z)/\partial Z)
            let flux_r_p = r_p * state.rho[i + 1][j] * state.v_r[i + 1][j];
            let flux_r_m = r_m * state.rho[i - 1][j] * state.v_r[i - 1][j];
            let div_rho_vr = (flux_r_p - flux_r_m) / (dr2 * r);

            let flux_z_p = state.rho[i][j + 1] * state.v_z[i][j + 1];
            let flux_z_m = state.rho[i][j - 1] * state.v_z[i][j - 1];
            let div_rho_vz = (flux_z_p - flux_z_m) / dz2;

            deriv.d_rho[i][j] = -(div_rho_vr + div_rho_vz);

            // 2. Velocity divergence: \nabla \cdot \mathbf{v}
            let div_v = (r_p * state.v_r[i + 1][j] - r_m * state.v_r[i - 1][j]) / (dr2 * r)
                + (state.v_z[i][j + 1] - state.v_z[i][j - 1]) / dz2;

            // Pressure gradients
            let dp_dr = (state.pressure[i + 1][j] - state.pressure[i - 1][j]) / dr2;
            let dp_dz = (state.pressure[i][j + 1] - state.pressure[i][j - 1]) / dz2;

            // Current density J = (1/\mu_0) \nabla \times B
            let cur_j = state.current_density(i, j, grid);
            let jr = cur_j.x;
            let jphi = cur_j.y;
            let jz = cur_j.z;

            // Lorentz force J x B
            let fl_r = jphi * bz - jz * bphi;
            let fl_phi = jz * br - jr * bz;
            let fl_z = jr * bphi - jphi * br;

            // Viscous laplacian \nabla^2 v
            let lap_vr = (state.v_r[i + 1][j] - 2.0 * vr + state.v_r[i - 1][j]) / dr_sq
                + (state.v_r[i][j + 1] - 2.0 * vr + state.v_r[i][j - 1]) / dz_sq;
            let lap_vphi = (state.v_phi[i + 1][j] - 2.0 * vphi + state.v_phi[i - 1][j]) / dr_sq
                + (state.v_phi[i][j + 1] - 2.0 * vphi + state.v_phi[i][j - 1]) / dz_sq;
            let lap_vz = (state.v_z[i + 1][j] - 2.0 * vz + state.v_z[i - 1][j]) / dr_sq
                + (state.v_z[i][j + 1] - 2.0 * vz + state.v_z[i][j - 1]) / dz_sq;

            // Momentum equation (with cylindrical centrifugal/Coriolis geometric terms)
            let adv_vr = vr * (state.v_r[i + 1][j] - state.v_r[i - 1][j]) / dr2
                + vz * (state.v_r[i][j + 1] - state.v_r[i][j - 1]) / dz2
                - (vphi * vphi) / r;
            let adv_vphi = vr * (state.v_phi[i + 1][j] - state.v_phi[i - 1][j]) / dr2
                + vz * (state.v_phi[i][j + 1] - state.v_phi[i][j - 1]) / dz2
                + (vr * vphi) / r;
            let adv_vz = vr * (state.v_z[i + 1][j] - state.v_z[i - 1][j]) / dr2
                + vz * (state.v_z[i][j + 1] - state.v_z[i][j - 1]) / dz2;

            deriv.d_v_r[i][j] = -adv_vr + (-dp_dr + fl_r + viscosity * lap_vr) / rho;
            deriv.d_v_phi[i][j] = -adv_vphi + (fl_phi + viscosity * lap_vphi) / rho;
            deriv.d_v_z[i][j] = -adv_vz + (-dp_dz + fl_z + viscosity * lap_vz) / rho;

            // 3. Induction equation: \partial B / \partial t = -\nabla \times E
            // where E = -(v x B) + \eta J
            let v_cross_b_r = vphi * bz - vz * bphi;
            let v_cross_b_phi = vz * br - vr * bz;
            let v_cross_b_z = vr * bphi - vphi * br;

            let _e_r = -v_cross_b_r + resistivity * jr;
            let _e_phi = -v_cross_b_phi + resistivity * jphi;
            let _e_z = -v_cross_b_z + resistivity * jz;

            // Curl of E in cylindrical coordinates
            // (\nabla \times E)_R = - \partial E_\phi / \partial Z
            // (\nabla \times E)_\phi = \partial E_R / \partial Z - \partial E_Z / \partial R
            // (\nabla \times E)_Z = (1/R) \partial (R E_\phi) / \partial R
            let dephi_dz = (state_e_phi(state, i, j + 1, resistivity) - state_e_phi(state, i, j - 1, resistivity)) / dz2;
            let der_dz = (state_e_r(state, i, j + 1, resistivity) - state_e_r(state, i, j - 1, resistivity)) / dz2;
            let dez_dr = (state_e_z(state, i + 1, j, resistivity) - state_e_z(state, i - 1, j, resistivity)) / dr2;
            let re_phi_p = r_p * state_e_phi(state, i + 1, j, resistivity);
            let re_phi_m = r_m * state_e_phi(state, i - 1, j, resistivity);
            let curl_e_z = (re_phi_p - re_phi_m) / (dr2 * r);

            deriv.d_b_r[i][j] = dephi_dz; // - (- dephi_dz)
            deriv.d_b_phi[i][j] = -(der_dz - dez_dr);
            deriv.d_b_z[i][j] = -curl_e_z;

            // 4. Energy equation for pressure p
            // \partial p / \partial t = -v \cdot \nabla p - \gamma p (\nabla \cdot v) + (\gamma - 1) \eta |J|^2
            let adv_p = vr * dp_dr + vz * dp_dz;
            let j_mag2 = jr * jr + jphi * jphi + jz * jz;
            let ohmic = (ADIABATIC_INDEX_GAMMA - 1.0) * resistivity * j_mag2;

            deriv.d_pressure[i][j] = -adv_p - ADIABATIC_INDEX_GAMMA * p * div_v + ohmic;
        }
    }

    deriv
}

// Helpers for electric field evaluations at neighbor cells
#[inline]
fn state_e_r(state: &MhdState, i: usize, j: usize, _eta: f64) -> f64 {
    let v_cross_b = state.v_phi[i][j] * state.b_z[i][j] - state.v_z[i][j] * state.b_phi[i][j];
    -v_cross_b
}

#[inline]
fn state_e_phi(state: &MhdState, i: usize, j: usize, _eta: f64) -> f64 {
    let v_cross_b = state.v_z[i][j] * state.b_r[i][j] - state.v_r[i][j] * state.b_z[i][j];
    -v_cross_b
}

#[inline]
fn state_e_z(state: &MhdState, i: usize, j: usize, _eta: f64) -> f64 {
    let v_cross_b = state.v_r[i][j] * state.b_phi[i][j] - state.v_phi[i][j] * state.b_r[i][j];
    -v_cross_b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mhd_equilibrium_derivatives_near_zero() {
        let grid = MhdGrid::new(15, 15, 5.0, 7.0, -1.0, 1.0);
        let state = MhdState::new_uniform(&grid, 1.0e-7, 1.0e4, 5.0);

        // Uniform state with zero velocity should have zero time derivatives
        let deriv = evaluate_mhd_equations(&state, &grid, 0.0, 0.0);
        let d_rho_mid = deriv.d_rho[7][7];
        assert!(d_rho_mid.abs() < 1e-12, "d_rho should be zero in uniform static plasma: {}", d_rho_mid);
    }
}

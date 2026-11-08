//! Time integration algorithms and boundary condition enforcement for MHD simulation.

use serde::{Deserialize, Serialize};
use super::equations::{evaluate_mhd_equations, MhdDerivatives};
use super::grid::{BoundaryCondition, MhdGrid};
use super::state::MhdState;

/// Configuration options for the MHD time integrator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MhdStepperConfig {
    /// CFL safety factor (typically 0.2 to 0.5).
    pub cfl_factor: f64,
    /// Plasma kinematic shear viscosity \mu (m^2 / s).
    pub viscosity: f64,
    /// Plasma electrical resistivity \eta (\Omega \cdot m).
    pub resistivity: f64,
    /// Maximum allowed time step (s).
    pub max_dt: f64,
}

impl Default for MhdStepperConfig {
    fn default() -> Self {
        Self {
            cfl_factor: 0.35,
            viscosity: 1.0e-3,
            resistivity: 1.0e-6,
            max_dt: 1.0e-4,
        }
    }
}

/// Advanced time stepper implementing 4th-order Runge-Kutta (RK4) integration.
#[derive(Debug, Clone)]
pub struct MhdStepper {
    pub config: MhdStepperConfig,
}

impl MhdStepper {
    /// Creates a new stepper with configuration.
    pub fn new(config: MhdStepperConfig) -> Self {
        Self { config }
    }

    /// Computes stable CFL time step \Delta t based on local fast magnetosonic wave speeds.
    pub fn compute_cfl_time_step(&self, state: &MhdState, grid: &MhdGrid) -> f64 {
        let mut min_dt = self.config.max_dt;

        for i in 1..grid.nr - 1 {
            for j in 1..grid.nz - 1 {
                let vfast = state.fast_magnetosonic_speed(i, j);
                let vr = state.v_r[i][j].abs();
                let vz = state.v_z[i][j].abs();

                let dt_r = grid.dr / (vr + vfast);
                let dt_z = grid.dz / (vz + vfast);
                let dt_local = dt_r.min(dt_z);

                if dt_local < min_dt {
                    min_dt = dt_local;
                }
            }
        }

        (self.config.cfl_factor * min_dt).min(self.config.max_dt).max(1.0e-12)
    }

    /// Executes a single 4th-order Runge-Kutta (RK4) time step.
    pub fn step_rk4(&self, state: &mut MhdState, grid: &MhdGrid, dt: f64) {
        let nr = grid.nr;
        let nz = grid.nz;

        // k1 = f(y_n)
        let k1 = evaluate_mhd_equations(state, grid, self.config.viscosity, self.config.resistivity);

        // y_temp = y_n + dt/2 * k1
        let mut temp_state = state.clone();
        Self::apply_update(&mut temp_state, state, &k1, 0.5 * dt);
        self.apply_boundary_conditions(&mut temp_state, grid);

        // k2 = f(y_n + dt/2 * k1)
        let k2 = evaluate_mhd_equations(&temp_state, grid, self.config.viscosity, self.config.resistivity);

        // y_temp = y_n + dt/2 * k2
        Self::apply_update(&mut temp_state, state, &k2, 0.5 * dt);
        self.apply_boundary_conditions(&mut temp_state, grid);

        // k3 = f(y_n + dt/2 * k2)
        let k3 = evaluate_mhd_equations(&temp_state, grid, self.config.viscosity, self.config.resistivity);

        // y_temp = y_n + dt * k3
        Self::apply_update(&mut temp_state, state, &k3, dt);
        self.apply_boundary_conditions(&mut temp_state, grid);

        // k4 = f(y_n + dt * k3)
        let k4 = evaluate_mhd_equations(&temp_state, grid, self.config.viscosity, self.config.resistivity);

        // Final weighted update: y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        for i in 1..nr - 1 {
            for j in 1..nz - 1 {
                let factor = dt / 6.0;
                state.rho[i][j] += factor * (k1.d_rho[i][j] + 2.0 * k2.d_rho[i][j] + 2.0 * k3.d_rho[i][j] + k4.d_rho[i][j]);
                state.rho[i][j] = state.rho[i][j].max(1.0e-15);

                state.v_r[i][j] += factor * (k1.d_v_r[i][j] + 2.0 * k2.d_v_r[i][j] + 2.0 * k3.d_v_r[i][j] + k4.d_v_r[i][j]);
                state.v_phi[i][j] += factor * (k1.d_v_phi[i][j] + 2.0 * k2.d_v_phi[i][j] + 2.0 * k3.d_v_phi[i][j] + k4.d_v_phi[i][j]);
                state.v_z[i][j] += factor * (k1.d_v_z[i][j] + 2.0 * k2.d_v_z[i][j] + 2.0 * k3.d_v_z[i][j] + k4.d_v_z[i][j]);

                state.pressure[i][j] += factor * (k1.d_pressure[i][j] + 2.0 * k2.d_pressure[i][j] + 2.0 * k3.d_pressure[i][j] + k4.d_pressure[i][j]);
                state.pressure[i][j] = state.pressure[i][j].max(1.0e-10);

                state.b_r[i][j] += factor * (k1.d_b_r[i][j] + 2.0 * k2.d_b_r[i][j] + 2.0 * k3.d_b_r[i][j] + k4.d_b_r[i][j]);
                state.b_phi[i][j] += factor * (k1.d_b_phi[i][j] + 2.0 * k2.d_b_phi[i][j] + 2.0 * k3.d_b_phi[i][j] + k4.d_b_phi[i][j]);
                state.b_z[i][j] += factor * (k1.d_b_z[i][j] + 2.0 * k2.d_b_z[i][j] + 2.0 * k3.d_b_z[i][j] + k4.d_b_z[i][j]);
            }
        }

        state.time += dt;
        self.apply_boundary_conditions(state, grid);
    }

    /// Internal helper to update fields by dt * derivative.
    fn apply_update(target: &mut MhdState, base: &MhdState, deriv: &MhdDerivatives, dt: f64) {
        let nr = target.nr;
        let nz = target.nz;
        for i in 1..nr - 1 {
            for j in 1..nz - 1 {
                target.rho[i][j] = (base.rho[i][j] + dt * deriv.d_rho[i][j]).max(1.0e-15);
                target.v_r[i][j] = base.v_r[i][j] + dt * deriv.d_v_r[i][j];
                target.v_phi[i][j] = base.v_phi[i][j] + dt * deriv.d_v_phi[i][j];
                target.v_z[i][j] = base.v_z[i][j] + dt * deriv.d_v_z[i][j];
                target.pressure[i][j] = (base.pressure[i][j] + dt * deriv.d_pressure[i][j]).max(1.0e-10);
                target.b_r[i][j] = base.b_r[i][j] + dt * deriv.d_b_r[i][j];
                target.b_phi[i][j] = base.b_phi[i][j] + dt * deriv.d_b_phi[i][j];
                target.b_z[i][j] = base.b_z[i][j] + dt * deriv.d_b_z[i][j];
            }
        }
    }

    /// Enforces boundary conditions on grid edges.
    pub fn apply_boundary_conditions(&self, state: &mut MhdState, grid: &MhdGrid) {
        let nr = grid.nr;
        let nz = grid.nz;

        // Inner and Outer radial boundaries (i = 0, i = nr - 1)
        for j in 0..nz {
            // Inner radial wall (i = 0)
            match grid.bc_r_inner {
                BoundaryCondition::ConductingWall => {
                    state.v_r[0][j] = 0.0;
                    state.b_r[0][j] = 0.0;
                    state.rho[0][j] = state.rho[1][j];
                    state.pressure[0][j] = state.pressure[1][j];
                    state.v_phi[0][j] = state.v_phi[1][j];
                    state.v_z[0][j] = 0.0; // no-slip or free-slip
                    state.b_phi[0][j] = state.b_phi[1][j];
                    state.b_z[0][j] = state.b_z[1][j];
                }
                _ => {
                    state.rho[0][j] = state.rho[1][j];
                    state.pressure[0][j] = state.pressure[1][j];
                }
            }

            // Outer radial wall (i = nr - 1)
            match grid.bc_r_outer {
                BoundaryCondition::ConductingWall => {
                    state.v_r[nr - 1][j] = 0.0;
                    state.b_r[nr - 1][j] = 0.0;
                    state.rho[nr - 1][j] = state.rho[nr - 2][j];
                    state.pressure[nr - 1][j] = state.pressure[nr - 2][j];
                    state.v_phi[nr - 1][j] = state.v_phi[nr - 2][j];
                    state.v_z[nr - 1][j] = 0.0;
                    state.b_phi[nr - 1][j] = state.b_phi[nr - 2][j];
                    state.b_z[nr - 1][j] = state.b_z[nr - 2][j];
                }
                _ => {
                    state.rho[nr - 1][j] = state.rho[nr - 2][j];
                    state.pressure[nr - 1][j] = state.pressure[nr - 2][j];
                }
            }
        }

        // Bottom and Top vertical boundaries (j = 0, j = nz - 1)
        for i in 0..nr {
            // Bottom wall (j = 0)
            state.v_z[i][0] = 0.0;
            state.b_z[i][0] = 0.0;
            state.rho[i][0] = state.rho[i][1];
            state.pressure[i][0] = state.pressure[i][1];
            state.v_r[i][0] = 0.0;
            state.v_phi[i][0] = state.v_phi[i][1];
            state.b_r[i][0] = state.b_r[i][1];
            state.b_phi[i][0] = state.b_phi[i][1];

            // Top wall (j = nz - 1)
            state.v_z[i][nz - 1] = 0.0;
            state.b_z[i][nz - 1] = 0.0;
            state.rho[i][nz - 1] = state.rho[i][nz - 2];
            state.pressure[i][nz - 1] = state.pressure[i][nz - 2];
            state.v_r[i][nz - 1] = 0.0;
            state.v_phi[i][nz - 1] = state.v_phi[i][nz - 2];
            state.b_r[i][nz - 1] = state.b_r[i][nz - 2];
            state.b_phi[i][nz - 1] = state.b_phi[i][nz - 2];
        }
    }

    /// Evaluates the maximum divergence of magnetic field \nabla \cdot \mathbf{B} across domain.
    pub fn compute_max_div_b(&self, state: &MhdState, grid: &MhdGrid) -> f64 {
        let mut max_div = 0.0;
        let dr2 = 2.0 * grid.dr;
        let dz2 = 2.0 * grid.dz;

        for i in 1..grid.nr - 1 {
            let r = grid.r_coords[i];
            let r_p = grid.r_coords[i + 1];
            let r_m = grid.r_coords[i - 1];

            for j in 1..grid.nz - 1 {
                // (1/R) \partial(R B_R)/\partial R + \partial B_Z / \partial Z
                let div_r = (r_p * state.b_r[i + 1][j] - r_m * state.b_r[i - 1][j]) / (dr2 * r);
                let div_z = (state.b_z[i][j + 1] - state.b_z[i][j - 1]) / dz2;
                let div = (div_r + div_z).abs();
                if div > max_div {
                    max_div = div;
                }
            }
        }
        max_div
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mhd_stepper_cfl_and_step() {
        let grid = MhdGrid::new(15, 15, 5.0, 7.0, -1.0, 1.0);
        let mut state = MhdState::new_uniform(&grid, 1.0e-7, 1.0e4, 5.0);

        let stepper = MhdStepper::new(MhdStepperConfig::default());
        let dt = stepper.compute_cfl_time_step(&state, &grid);
        assert!(dt > 0.0 && dt <= 1.0e-4);

        let t_initial = state.time;
        stepper.step_rk4(&mut state, &grid, dt);
        assert!((state.time - (t_initial + dt)).abs() < 1e-15);

        let div_b = stepper.compute_max_div_b(&state, &grid);
        assert!(div_b.is_finite());
    }
}

//! Integration and numerical physics tests for magnetohydrodynamics (MHD) module.

use ouroboros_plasma::mhd::{
    evaluate_mhd_equations, MhdGrid, MhdState, MhdStepper, MhdStepperConfig,
};

#[test]
fn test_mhd_uniform_background_static() {
    let grid = MhdGrid::new(15, 15, 5.0, 7.5, -1.2, 1.2);
    let state = MhdState::new_uniform(&grid, 1.2e-7, 1.5e4, 5.2);

    let deriv = evaluate_mhd_equations(&state, &grid, 1.0e-3, 1.0e-6);

    for i in 1..grid.nr - 1 {
        for j in 1..grid.nz - 1 {
            assert!(deriv.d_rho[i][j].abs() < 1e-10);
            assert!(deriv.d_pressure[i][j].abs() < 1e-10);
            assert!(deriv.d_b_r[i][j].abs() < 1e-10);
            assert!(deriv.d_b_z[i][j].abs() < 1e-10);
        }
    }
}

#[test]
fn test_mhd_energy_integrals() {
    let grid = MhdGrid::new(17, 17, 4.5, 7.5, -1.5, 1.5);
    let state = MhdState::new_uniform(&grid, 2.0e-7, 3.0e4, 5.0);

    let w_th = state.total_thermal_energy(&grid);
    let w_mag = state.total_magnetic_energy(&grid);
    let w_kin = state.total_kinetic_energy(&grid);

    assert!(w_th > 0.0);
    assert!(w_mag > 0.0);
    assert_eq!(w_kin, 0.0); // static initial plasma
    assert!(w_mag > w_th, "Magnetic energy dominates in low-beta tokamak");
}

#[test]
fn test_mhd_stepper_cfl_and_divergence() {
    let grid = MhdGrid::new(15, 15, 4.0, 8.0, -2.0, 2.0);
    let mut state = MhdState::new_uniform(&grid, 1.0e-7, 1.0e4, 5.0);

    let config = MhdStepperConfig {
        cfl_factor: 0.3,
        viscosity: 1.0e-3,
        resistivity: 1.0e-6,
        max_dt: 1.0e-4,
    };
    let stepper = MhdStepper::new(config);

    for _ in 0..5 {
        let dt = stepper.compute_cfl_time_step(&state, &grid);
        assert!(dt > 0.0);
        stepper.step_rk4(&mut state, &grid, dt);
    }

    let div_b = stepper.compute_max_div_b(&state, &grid);
    assert!(div_b.is_finite());
    assert!(div_b < 1.0);
}

//! Integration tests for Toroidal Alfvén Eigenmodes (TAE) and spectrum stability.

use approx::assert_relative_eq;
use ouroboros_plasma::constants::DT_AVERAGE_ION_MASS;
use ouroboros_plasma::modes::{
    AlfvenContinuum, DampingCalculator, FastIonDriveCalculator, ModeStability, TaeSolver,
};

#[test]
fn test_tae_gap_properties() {
    let continuum = AlfvenContinuum::new(6.2, 2.0, 5.3, 2, vec![2]);
    let solver = TaeSolver::new(continuum);

    let q_profile = |rho: f64| 1.05 + 2.0 * rho * rho;
    let rho_mass = 1.0e20 * DT_AVERAGE_ION_MASS;

    let mode = solver.solve_tae_mode(2, rho_mass, q_profile).expect("Mode should exist");

    // For n=2, m=2: q = 2.5 / 2 = 1.25
    assert_relative_eq!(mode.q_tae, 1.25, max_relative = 1e-10);

    // 1.05 + 2.0 * rho^2 = 1.25 => 2.0 * rho^2 = 0.20 => rho^2 = 0.10 => rho = sqrt(0.10) ~ 0.3162
    let expected_rho = (0.10_f64).sqrt();
    assert_relative_eq!(mode.rho_tae, expected_rho, max_relative = 1e-3);

    // Verify frequency is in expected tokamak acoustic/Alfvén gap range (50-200 kHz)
    assert!(mode.frequency_hz > 40.0e3 && mode.frequency_hz < 250.0e3);
    assert!(mode.omega_lower < mode.omega_upper);
    assert!(mode.gap_fraction > 0.0);
}

#[test]
fn test_tae_damping_and_drive_stability() {
    let continuum = AlfvenContinuum::new(6.2, 2.0, 5.3, 2, vec![2]);
    let solver = TaeSolver::new(continuum);
    let rho_mass = 1.0e20 * DT_AVERAGE_ION_MASS;
    let mode = solver.solve_tae_mode(2, rho_mass, |rho| 1.05 + 2.0 * rho * rho).unwrap();

    let damp_calc = DampingCalculator::new(12.0e3, 12.0e3, 1.0e20, DT_AVERAGE_ION_MASS);
    let va = solver.continuum.alfven_speed(mode.r_tae, rho_mass);
    let damping = damp_calc.compute_damping(&mode, 0.6, va, 0.02, 0.02, 5.3);

    assert!(damping.gamma_total_normalized > 0.0);

    let drive_calc = FastIonDriveCalculator::new();

    // With zero alpha beta: completely stable
    let rep_zero = drive_calc.analyze_stability(mode.clone(), 0.0, va, 5.3, 0.3, damping.clone());
    assert_eq!(rep_zero.stability, ModeStability::Stable);
    assert_eq!(rep_zero.gamma_drive_normalized, 0.0);

    // With high alpha beta (3.0%): unstable
    let rep_unstable = drive_calc.analyze_stability(mode, 0.03, va, 5.3, 0.15, damping);
    assert_eq!(rep_unstable.stability, ModeStability::Unstable);
    assert!(rep_unstable.gamma_net_normalized > 0.0);
}

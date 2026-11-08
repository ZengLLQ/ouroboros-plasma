//! Integration and verification tests for tokamak equilibrium module.

use approx::assert_relative_eq;
use ouroboros_plasma::constants::MU_0;
use ouroboros_plasma::equilibrium::{
    GradShafranovSolver, MillerGeometry, PlasmaProfiles, SolovievEquilibrium,
};

#[test]
fn test_soloviev_exact_pde_identity() {
    let sol = SolovievEquilibrium::new(6.2, 1.7, 5.3, 2.0e5, 2.0);

    // Verify at multiple spatial sample points inside plasma
    let test_points = [(6.0, 0.0), (6.5, 0.5), (5.5, -0.8), (7.0, 1.2)];
    for &(r, z) in &test_points {
        let delta_star = sol.delta_star_psi(r, z);
        let expected = MU_0 * sol.p1 * r * r;
        assert_relative_eq!(delta_star, expected, max_relative = 1.0e-9);
    }
}

#[test]
fn test_miller_geometry_and_volumes() {
    let r0 = 6.2;
    let a = 2.0;
    let kappa = 1.7;
    let delta = 0.33;
    let shift = 0.15;

    let miller = MillerGeometry::new(r0, 0.0, a, kappa, delta, shift);
    let n_pts = 300;

    let area = miller.poloidal_area(n_pts);
    let expected_area = std::f64::consts::PI * a * a * kappa;
    // Triangularity introduces small distortion (~2-5%)
    assert!((area - expected_area).abs() / expected_area < 0.06);

    let volume = miller.plasma_volume(n_pts);
    let expected_volume = 2.0 * std::f64::consts::PI * (r0 + shift) * expected_area;
    assert!((volume - expected_volume).abs() / expected_volume < 0.06);
}

#[test]
fn test_plasma_profiles_derivatives_and_current() {
    let p0 = 200.0e3;
    let profiles = PlasmaProfiles::new(p0, 2.0, 1.5, 6.2, 2.0, 5.3, 0.0, 10.0);

    // Test monotonicity of pressure
    let p_0 = profiles.pressure(0.0);
    let p_half = profiles.pressure(0.5);
    let p_edge = profiles.pressure(1.0);

    assert!(p_0 > p_half);
    assert!(p_half > p_edge);

    // Current density at mid-radius should exceed edge
    let j_mid = profiles.current_density(6.2, 0.5);
    let j_edge = profiles.current_density(8.0, 0.999);
    assert!(j_mid.abs() > j_edge.abs(), "Mid-plasma current density should exceed boundary edge: j_mid={}, j_edge={}", j_mid, j_edge);
}

#[test]
fn test_grad_shafranov_solver_convergence_and_fields() {
    let mut solver = GradShafranovSolver::new(25, 25, 4.0, 8.4, -3.0, 3.0);
    let miller = MillerGeometry::new(6.2, 0.0, 2.0, 1.6, 0.3, 0.1);
    solver.initialize_from_miller(&miller, 10.0);

    let profiles = PlasmaProfiles::new(2.5e5, 2.0, 1.5, 6.2, 2.0, 5.3, 0.0, 10.0);
    let (iters, res) = solver.solve(&profiles, 150, 1.0e-4);

    assert!(iters > 0);
    assert!(res < 1.0e-2);

    // Magnetic field evaluation
    let b_core = solver.evaluate_b_field(solver.magnetic_axis.0, solver.magnetic_axis.1, &profiles);
    assert!(b_core.x.abs() < 1.0, "B_R near axis should be small: {}", b_core.x);
    assert!(b_core.z.abs() < 1.0, "B_Z near axis should be small: {}", b_core.z);
    assert!(b_core.y > 4.5 && b_core.y < 6.0, "Toroidal field should match B0: {}", b_core.y);
}

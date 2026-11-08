//! Tests for 7-bit ASCII visualization and fusion performance diagnostic tables.

use ouroboros_plasma::equilibrium::{GradShafranovSolver, MillerGeometry, PlasmaProfiles};
use ouroboros_plasma::visualizer::AsciiVisualizer;

#[test]
fn test_ascii_contour_rendering_7bit_clean() {
    let mut solver = GradShafranovSolver::new(21, 21, 4.0, 8.4, -2.5, 2.5);
    let miller = MillerGeometry::new(6.2, 0.0, 2.0, 1.6, 0.3, 0.1);
    solver.initialize_from_miller(&miller, 10.0);
    let profiles = PlasmaProfiles::new(2.0e5, 2.0, 1.5, 6.2, 2.0, 5.3, 0.0, 10.0);
    solver.solve(&profiles, 50, 1.0e-3);

    let ascii_map = AsciiVisualizer::render_flux_contour(&solver, &miller, 50, 20);

    assert!(ascii_map.contains("OUROBOROS-PLASMA"));
    assert!(ascii_map.contains('*')); // axis
    assert!(ascii_map.contains('#')); // limiter wall

    // Strict 7-bit ASCII check
    for byte in ascii_map.bytes() {
        assert!(byte <= 127, "Must be valid 7-bit ASCII character, found: {}", byte);
    }
}

#[test]
fn test_ascii_q_profile_7bit_clean() {
    let q_plot = AsciiVisualizer::render_q_profile(1.1, 4.0, 55, 15);
    assert!(q_plot.contains("SAFETY FACTOR"));
    assert!(q_plot.contains('*'));

    for byte in q_plot.bytes() {
        assert!(byte <= 127, "Must be valid 7-bit ASCII character, found: {}", byte);
    }
}

#[test]
fn test_fusion_performance_diagnostics() {
    let rep = AsciiVisualizer::evaluate_fusion_performance(
        "ITER Reference 15MA",
        6.2,
        2.0,
        5.3,
        15.0e6,
        1.7,
        0.33,
        1.0e20,
        16.0,
        1.05,
        3.3,
    );

    assert!(rep.total_fusion_power_mw > 400.0 && rep.total_fusion_power_mw < 650.0);
    assert!(rep.alpha_heating_power_mw > 80.0 && rep.alpha_heating_power_mw < 140.0);
    assert!(rep.stored_thermal_energy_mj > 200.0 && rep.stored_thermal_energy_mj < 400.0);
    assert!(rep.tau_e_iter98y2_seconds > 2.0 && rep.tau_e_iter98y2_seconds < 5.0);

    let table = AsciiVisualizer::render_performance_table(&rep);
    for byte in table.bytes() {
        assert!(byte <= 127, "Performance table must be 7-bit ASCII only");
    }
}

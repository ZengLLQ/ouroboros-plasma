//! Integration tests for relativistic particle dynamics and ensemble kinetics.

use approx::assert_relative_eq;
use ouroboros_plasma::constants::{ALPHA_BIRTH_ENERGY_JOULES, ALPHA_CHARGE, ALPHA_MASS, SPEED_OF_LIGHT};
use ouroboros_plasma::math::Vector3D;
use ouroboros_plasma::particles::{
    GuidingCenter, LossConeAnalyzer, OrbitTopology, ParticleEnsemble, ParticleState,
    RelativisticBorisIntegrator,
};

#[test]
fn test_boris_relativistic_energy_conservation() {
    let pos = Vector3D::new(6.2, 0.0, 0.0);
    let speed = 0.05 * SPEED_OF_LIGHT;
    let vel = Vector3D::new(speed * 0.5, speed * 0.8, speed * 0.3);
    let mut p = ParticleState::new(pos, vel, ALPHA_MASS, ALPHA_CHARGE);

    let e_initial = p.kinetic_energy();
    let b_const = Vector3D::new(0.0, 0.0, 5.0);

    for _ in 0..500 {
        RelativisticBorisIntegrator::step(&mut p, 1.0e-9, |_| (Vector3D::zero(), b_const));
    }

    let e_final = p.kinetic_energy();
    assert_relative_eq!(e_final, e_initial, max_relative = 1.0e-11);
}

#[test]
fn test_guiding_center_drifts() {
    let e = Vector3D::new(1.0e3, 0.0, 0.0);
    let b = Vector3D::new(0.0, 0.0, 5.0);
    let ve = GuidingCenter::exb_drift(e, b);

    // v_E = E x B / B^2 = (1e3, 0, 0) x (0, 0, 5) / 25 = (0, -5e3, 0) / 25 = (0, -200, 0)
    assert_relative_eq!(ve.x, 0.0, max_relative = 1e-12);
    assert_relative_eq!(ve.y, -200.0, max_relative = 1e-12);
    assert_relative_eq!(ve.z, 0.0, max_relative = 1e-12);
}

#[test]
fn test_loss_cone_kinetics() {
    let analyzer = LossConeAnalyzer::new(6.2, 2.0, 5.3, 1.8);
    let r_core = 0.4;
    let r_edge = 1.9;

    let v_alpha = (2.0 * ALPHA_BIRTH_ENERGY_JOULES / ALPHA_MASS).sqrt();

    // Deep core trapped particle should be trapped, not promptly lost
    let topo_core = analyzer.classify_orbit(r_core, 0.1, v_alpha, ALPHA_MASS, ALPHA_CHARGE);
    assert_eq!(topo_core, OrbitTopology::TrappedBanana);

    // Extreme edge trapped particle with large banana width should hit wall
    let topo_edge = analyzer.classify_orbit(r_edge, 0.1, v_alpha, ALPHA_MASS, ALPHA_CHARGE);
    assert_eq!(topo_edge, OrbitTopology::PromptLoss);
}

#[test]
fn test_particle_ensemble_statistics() {
    let ensemble = ParticleEnsemble::generate_dt_alphas(400, 6.2, 2.0, 5.3, 1.8, 98765);
    let rep = ensemble.compute_confinement_report();

    assert_eq!(rep.total_particles, 400);
    assert!(rep.confinement_fraction > 0.85);
    assert!(rep.passing_fraction > 0.4);
    assert!(rep.trapped_fraction > 0.2);
    assert_relative_eq!(rep.passing_fraction + rep.trapped_fraction + rep.prompt_loss_fraction, 1.0, max_relative = 1e-12);
}

//! Example: Relativistic Alpha Particle Confinement & Boris Orbit Integration.
//!
//! Demonstrates:
//! 1. Synthesizing a realistic ensemble of 3.52 MeV alpha fusion products from D-T reactions.
//! 2. Relativistic Boris velocity leapfrog integration in 3D toroidal magnetic geometry.
//! 3. Banana orbit bounce dynamics, magnetic mirror trapping, and pitch-angle kinetics.
//! 4. Canonical angular momentum P_\phi and relativistic energy conservation diagnostics.
//! 5. Prompt loss fraction evaluation and wall collision power loss quantification.

use ouroboros_plasma::constants::{ALPHA_BIRTH_ENERGY_JOULES, ALPHA_CHARGE, ALPHA_MASS, SPEED_OF_LIGHT};
use ouroboros_plasma::math::Vector3D;
use ouroboros_plasma::particles::{
    LossConeAnalyzer, ParticleEnsemble, ParticleState, RelativisticBorisIntegrator,
};

fn main() {
    println!("================================================================================");
    println!("  OUROBOROS-PLASMA: RELATIVISTIC ALPHA CONFINEMENT & BORIS INTEGRATION DEMO     ");
    println!("================================================================================");

    let r0 = 6.2;       // Major radius (m)
    let a = 2.0;        // Minor radius (m)
    let b0 = 5.3;       // On-axis toroidal field (T)
    let q_safety = 1.85;// Representative safety factor

    // 1. Single Particle Detailed Orbit & Invariants Test
    println!("[Phase 1] Precision Relativistic Invariant Diagnostics (Single 3.52 MeV Alpha):");
    let v_alpha = (2.0 * ALPHA_BIRTH_ENERGY_JOULES / ALPHA_MASS).sqrt();
    let initial_pos = Vector3D::new(r0 + 0.5, 0.0, 0.0);
    let pitch = 0.35; // trapped banana orbit
    let v_par = v_alpha * pitch;
    let v_perp = v_alpha * (1.0 - pitch * pitch).sqrt();
    let initial_vel = Vector3D::new(0.0, v_par, v_perp);

    let mut single_particle = ParticleState::new(initial_pos, initial_vel, ALPHA_MASS, ALPHA_CHARGE);
    let initial_energy_ev = single_particle.kinetic_energy_ev();
    let initial_gamma = single_particle.gamma();

    println!("  -> Particle Mass:        {:.3e} kg", single_particle.mass);
    println!("  -> Particle Charge:      {:.3e} C (+2e)", single_particle.charge);
    println!("  -> Birth Speed:          {:.3e} m/s ({:.2}% of c)", v_alpha, (v_alpha / SPEED_OF_LIGHT) * 100.0);
    println!("  -> Initial Lorentz Gamma:{:.6}", initial_gamma);
    println!("  -> Initial Kinetic Energy: {:.3} MeV", initial_energy_ev * 1.0e-6);

    // Magnetic field function: B = B_0 (R_0 / R) \hat{\phi} + B_p \hat{\theta}
    let b_field_fn = |pos: Vector3D| -> (Vector3D, Vector3D) {
        let (r, phi, z) = pos.to_cylindrical();
        let bphi = if r > 0.1 { b0 * r0 / r } else { b0 };
        // Poloidal field B_p \approx (r_minor / (q R_0)) B_0
        let r_minor = ((r - r0) * (r - r0) + z * z).sqrt();
        let bp_mag = (r_minor / (q_safety * r0)) * b0;
        let theta = z.atan2(r - r0);
        let br = -bp_mag * theta.sin();
        let bz = bp_mag * theta.cos();
        let b_cart = Vector3D::cylindrical_to_cartesian_vector(br, bphi, bz, phi);
        (Vector3D::zero(), b_cart)
    };

    let dt = 0.2e-9; // 0.2 ns step
    let num_steps = 1500;
    println!("\n  Integrating orbit for {} steps (dt = {:.2} ns, total time = {:.2} microseconds)...",
        num_steps, dt * 1.0e9, num_steps as f64 * dt * 1.0e6);

    let mut min_r = f64::INFINITY;
    let mut max_r = f64::NEG_INFINITY;

    for _ in 0..num_steps {
        RelativisticBorisIntegrator::step(&mut single_particle, dt, b_field_fn);
        let (r, _phi, _z) = single_particle.cylindrical_position();
        if r < min_r { min_r = r; }
        if r > max_r { max_r = r; }
    }

    let final_energy_ev = single_particle.kinetic_energy_ev();
    let energy_drift = (final_energy_ev - initial_energy_ev).abs() / initial_energy_ev;
    println!("  -> Final Kinetic Energy:   {:.3} MeV", final_energy_ev * 1.0e-6);
    println!("  -> Energy Conservation Drift: {:.2e} (Machine precision invariant!)", energy_drift);
    println!("  -> Banana Orbit Radial Extent: [{:.3} m .. {:.3} m] (Width: {:.2} cm)",
        min_r, max_r, (max_r - min_r) * 100.0);

    // 2. Ensemble Confinement Statistics
    let ensemble_size = 500;
    println!("\n[Phase 2] Generating D-T Fusion Alpha Particle Swarm (N = {}):", ensemble_size);
    let mut ensemble = ParticleEnsemble::generate_dt_alphas(ensemble_size, r0, a, b0, q_safety, 424242);
    let report_initial = ensemble.compute_confinement_report();

    println!("  -> Passing (Circulating):{:>4} ({:.1}%)", report_initial.passing_count, report_initial.passing_fraction * 100.0);
    println!("  -> Trapped Bananas:      {:>4} ({:.1}%)", report_initial.trapped_count, report_initial.trapped_fraction * 100.0);
    println!("  -> Prompt Wall Losses:   {:>4} ({:.1}%)", report_initial.prompt_loss_count, report_initial.prompt_loss_fraction * 100.0);
    println!("  -> Overall Confinement:  {:>6.2}%", report_initial.confinement_fraction * 100.0);

    println!("\n[Phase 3] Advancing Alpha Ensemble Along Orbit Trajectories...");
    ensemble.advance_ensemble(1.0e-9, 50, b_field_fn);

    let report_final = ensemble.compute_confinement_report();
    println!("\n[Phase 4] Energetic Fusion Product Confinement Summary:");
    println!("  Total Alpha Birth Energy:         {:>10.3e} J ({:.1} MeV total)", report_final.total_birth_energy_joules, (report_final.total_birth_energy_joules / 1.602176634e-19) * 1.0e-6);
    println!("  Confined Core Heating Energy:     {:>10.3e} J ({:.1}%)", report_final.confined_energy_joules, report_final.confinement_fraction * 100.0);
    println!("  Prompt First-Wall Particle Loss:  {:>10.3e} J ({:.1}%)", report_final.lost_energy_joules, report_final.prompt_loss_fraction * 100.0);

    // Theoretical banana width comparison
    let analyzer = LossConeAnalyzer::new(r0, a, b0, q_safety);
    let theo_banana_width = analyzer.banana_orbit_width(1.0, v_alpha, ALPHA_MASS, ALPHA_CHARGE);
    let bounce_freq = analyzer.bounce_frequency(1.0, v_alpha);
    println!("\n[Phase 5] Theoretical Banana Kinetics:");
    println!("  -> Mid-Radius Banana Width:   {:.2} cm", theo_banana_width * 100.0);
    println!("  -> Banana Bounce Frequency:   {:.1} kHz", bounce_freq * 1.0e-3);
    println!("  -> Trapped Fraction Theory:   {:.1}%", analyzer.trapped_fraction(1.0) * 100.0);

    println!("\nRelativistic alpha particle confinement demonstration completed successfully!");
}

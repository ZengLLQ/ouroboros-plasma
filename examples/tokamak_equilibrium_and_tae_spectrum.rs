//! Example: Tokamak Equilibrium & Toroidal Alfvén Eigenmode (TAE) Spectrum.
//!
//! Demonstrates:
//! 1. Solving the 2D Grad-Shafranov elliptic PDE for an ITER-like burning plasma equilibrium.
//! 2. Rendering 7-bit ASCII flux surface contours and safety factor q(r) profiles.
//! 3. Computing shear Alfvén continuum branches and identifying the TAE frequency gap.
//! 4. Solving for discrete TAE eigenfrequency, damping rates, and fast alpha resonant drive.
//! 5. Outputting a comprehensive thermonuclear fusion performance report card.

use std::f64::consts::PI;
use ouroboros_plasma::constants::DT_AVERAGE_ION_MASS;
use ouroboros_plasma::equilibrium::{GradShafranovSolver, MillerGeometry, PlasmaProfiles};
use ouroboros_plasma::modes::{
    AlfvenContinuum, DampingCalculator, FastIonDriveCalculator, TaeSolver,
};
use ouroboros_plasma::visualizer::AsciiVisualizer;

fn main() {
    println!("================================================================================");
    println!("     OUROBOROS-PLASMA: TOKAMAK EQUILIBRIUM & TAE SPECTRUM DEMONSTRATION         ");
    println!("================================================================================");

    // 1. Plasma Geometry & Machine Parameters (ITER Baseline 15 MA)
    let r0 = 6.2;       // Major radius (m)
    let a = 2.0;        // Minor radius (m)
    let b0 = 5.3;       // Toroidal magnetic field on axis (T)
    let ip = 15.0e6;    // Plasma current (A)
    let kappa = 1.7;    // Elongation
    let delta = 0.33;   // Triangularity
    let p0 = 260.0e3;   // Central pressure (Pa) ~ 260 kPa

    println!("[Phase 1] Initializing 2D Grad-Shafranov Grid & Shaped Boundary...");
    let r_min = 3.6;
    let r_max = 8.8;
    let z_min = -3.8;
    let z_max = 3.8;
    let grid_dim = 33;

    let mut solver = GradShafranovSolver::new(grid_dim, grid_dim, r_min, r_max, z_min, z_max);
    let miller = MillerGeometry::new(r0, 0.0, a, kappa, delta, 0.14);
    solver.initialize_from_miller(&miller, 12.0);

    let profiles = PlasmaProfiles::new(p0, 2.0, 1.5, r0, a, b0, 0.0, 12.0);

    println!("[Phase 2] Solving Non-linear Grad-Shafranov PDE with Successive Over-Relaxation...");
    let (iters, res) = solver.solve(&profiles, 200, 1.0e-4);
    println!("  -> Iterations: {}, Final L2 Residual: {:.3e}", iters, res);
    println!("  -> Magnetic Axis: R = {:.3} m, Z = {:.3} m", solver.magnetic_axis.0, solver.magnetic_axis.1);
    println!("  -> Total Toroidal Current: {:.2} MA", solver.total_current_ip * 1.0e-6);

    println!("\n[Phase 3] ASCII Cross-Section Flux Contour Map:");
    println!("{}", AsciiVisualizer::render_flux_contour(&solver, &miller, 65, 24));

    println!("\n[Phase 4] ASCII Safety Factor q(rho) Profile:");
    let q0 = 1.05;
    let qa = 3.40;
    println!("{}", AsciiVisualizer::render_q_profile(q0, qa, 65, 14));

    println!("\n[Phase 5] Analyzing Toroidal Alfvén Eigenmodes (TAE)...");
    let toroidal_n = 2;
    let poloidal_m = 2;
    let continuum = AlfvenContinuum::new(r0, a, b0, toroidal_n, vec![poloidal_m, poloidal_m + 1]);
    let tae_solver = TaeSolver::new(continuum);

    let ne0 = 1.05e20; // Core electron density (m^-3)
    let rho_mass = ne0 * DT_AVERAGE_ION_MASS;
    let q_profile = |rho: f64| q0 + (qa - q0) * rho * rho;

    if let Some(mode) = tae_solver.solve_tae_mode(poloidal_m, rho_mass, q_profile) {
        println!("  TAE Mode Identified: (n={}, m={})", mode.toroidal_n, mode.poloidal_m);
        println!("  -> Radial Gap Location: r_TAE = {:.3} m (rho = {:.3})", mode.r_tae, mode.rho_tae);
        println!("  -> Safety Factor at Gap: q(r_TAE) = {:.3}", mode.q_tae);
        println!("  -> Center Gap Frequency: f_TAE = {:.1} kHz", mode.frequency_hz * 1.0e-3);
        println!("  -> Continuum Gap Range:  [{:.1} .. {:.1}] kHz",
            mode.omega_lower / (2.0 * PI) * 1.0e-3,
            mode.omega_upper / (2.0 * PI) * 1.0e-3
        );
        println!("  -> Gap Width Fraction:   {:.1}%", mode.gap_fraction * 100.0);

        let va = tae_solver.continuum.alfven_speed(mode.r_tae, rho_mass);
        let damp_calc = DampingCalculator::new(15.0e3, 15.0e3, ne0, DT_AVERAGE_ION_MASS);
        let damping = damp_calc.compute_damping(&mode, 0.65, va, 0.02, 0.02, b0);

        println!("\n[Phase 6] Damping vs Energetic Alpha Resonant Drive:");
        println!("  -> Continuum Damping:   gamma_cont / omega = {:>9.2e}", damping.gamma_continuum_normalized);
        println!("  -> Electron Landau:     gamma_e / omega    = {:>9.2e}", damping.gamma_electron_landau_normalized);
        println!("  -> Ion Landau:          gamma_i / omega    = {:>9.2e}", damping.gamma_ion_landau_normalized);
        println!("  -> Radiative Damping:   gamma_rad / omega  = {:>9.2e}", damping.gamma_radiative_normalized);
        println!("  -> Total Damping:       gamma_damp / omega = {:>9.2e} ({:.1} s^-1)", damping.gamma_total_normalized, damping.gamma_total_s1);

        let beta_alpha = 0.0055; // 0.55% fast alpha beta
        let drive_calc = FastIonDriveCalculator::new();
        let report = drive_calc.analyze_stability(mode, beta_alpha, va, b0, 0.45, damping);

        println!("  -> Alpha Resonant Drive:gamma_drive / omega= {:>9.2e}", report.gamma_drive_normalized);
        println!("  -> Net Linear Growth:   gamma_net / omega  = {:>9.2e} ({:.1} s^-1)", report.gamma_net_normalized, report.gamma_net_s1);
        println!("  -> Critical Alpha Beta: beta_alpha_crit    = {:.4}%", report.beta_alpha_critical * 100.0);
        println!("  -> Stability Verdict:   {:?}", report.stability);
    }

    println!("\n[Phase 7] Full Fusion Performance Diagnostic Table:");
    let perf = AsciiVisualizer::evaluate_fusion_performance(
        "ITER 15MA Q=10 Reference Demonstration",
        r0,
        a,
        b0,
        ip,
        kappa,
        delta,
        ne0,
        16.5, // 16.5 keV core temperature
        q0,
        qa,
    );
    println!("{}", AsciiVisualizer::render_performance_table(&perf));
    println!("Demonstration completed successfully!");
}

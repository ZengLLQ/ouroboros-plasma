//! Command-line interface definitions and execution handlers for OuroborosPlasma.

use clap::{Args, Parser, Subcommand};
use crate::constants::DT_AVERAGE_ION_MASS;
use crate::equilibrium::{GradShafranovSolver, MillerGeometry, PlasmaProfiles};
use crate::mhd::{MhdGrid, MhdState, MhdStepper, MhdStepperConfig};
use crate::modes::{AlfvenContinuum, DampingCalculator, FastIonDriveCalculator, TaeSolver};
use crate::particles::ParticleEnsemble;
use crate::visualizer::AsciiVisualizer;

/// High-performance computational fusion plasma transport and MHD simulator.
#[derive(Debug, Parser)]
#[command(name = "ouroboros-plasma")]
#[command(author = "ZengLLQ")]
#[command(version = "0.1.0")]
#[command(about = "Relativistic Tokamak Fusion Transport, Grad-Shafranov Equilibrium & TAE Analysis")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Solves the 2D Grad-Shafranov elliptic PDE for tokamak magnetostatic equilibrium.
    SolveEquilibrium(SolveEquilibriumArgs),
    /// Simulates relativistic Boris particle integration for energetic fusion alpha particles.
    TrackParticles(TrackParticlesArgs),
    /// Analyzes the Toroidal Alfvén Eigenmodes (TAE) spectrum and energetic alpha drive.
    TaeSpectrum(TaeSpectrumArgs),
    /// Runs 2D ideal/resistive magnetohydrodynamics (MHD) time integration steps.
    RunMhdStep(RunMhdStepArgs),
    /// Runs a full end-to-end multi-physics tokamak fusion demonstration.
    Demo(DemoArgs),
}

#[derive(Debug, Args)]
pub struct SolveEquilibriumArgs {
    /// Major radius R_0 in meters.
    #[arg(long, default_value_t = 6.2)]
    pub r0: f64,
    /// Minor radius a in meters.
    #[arg(long, default_value_t = 2.0)]
    pub a: f64,
    /// On-axis toroidal magnetic field B_0 in Tesla.
    #[arg(long, default_value_t = 5.3)]
    pub b0: f64,
    /// Plasma current in Mega-Amperes (MA).
    #[arg(long, default_value_t = 15.0)]
    pub ip_ma: f64,
    /// Plasma elongation kappa.
    #[arg(long, default_value_t = 1.7)]
    pub kappa: f64,
    /// Plasma triangularity delta.
    #[arg(long, default_value_t = 0.33)]
    pub delta: f64,
    /// Central kinetic pressure in kPa.
    #[arg(long, default_value_t = 250.0)]
    pub p_core_kpa: f64,
    /// Number of grid points along R and Z.
    #[arg(long, default_value_t = 35)]
    pub grid_size: usize,
    /// Maximum SOR relaxation iterations.
    #[arg(long, default_value_t = 200)]
    pub max_iters: usize,
    /// Whether to render ASCII flux surface cross-section map.
    #[arg(long, default_value_t = true)]
    pub ascii: bool,
}

#[derive(Debug, Args)]
pub struct TrackParticlesArgs {
    /// Number of alpha particles to generate in ensemble.
    #[arg(long, default_value_t = 400)]
    pub particles: usize,
    /// Kinetic energy in MeV.
    #[arg(long, default_value_t = 3.52)]
    pub energy_mev: f64,
    /// Major radius R_0 in meters.
    #[arg(long, default_value_t = 6.2)]
    pub r0: f64,
    /// Minor radius a in meters.
    #[arg(long, default_value_t = 2.0)]
    pub a: f64,
    /// Magnetic field B_0 in Tesla.
    #[arg(long, default_value_t = 5.3)]
    pub b0: f64,
    /// Safety factor q.
    #[arg(long, default_value_t = 1.8)]
    pub q_safety: f64,
    /// Number of Boris leapfrog steps per particle.
    #[arg(long, default_value_t = 40)]
    pub steps: usize,
    /// Integration time step in nanoseconds.
    #[arg(long, default_value_t = 0.5)]
    pub dt_ns: f64,
}

#[derive(Debug, Args)]
pub struct TaeSpectrumArgs {
    /// Toroidal mode number n.
    #[arg(long, default_value_t = 2)]
    pub toroidal_n: i32,
    /// Poloidal mode number m.
    #[arg(long, default_value_t = 2)]
    pub poloidal_m: i32,
    /// Major radius R_0 in meters.
    #[arg(long, default_value_t = 6.2)]
    pub r0: f64,
    /// Minor radius a in meters.
    #[arg(long, default_value_t = 2.0)]
    pub a: f64,
    /// On-axis field B_0 in Tesla.
    #[arg(long, default_value_t = 5.3)]
    pub b0: f64,
    /// Energetic alpha particle beta \beta_\alpha.
    #[arg(long, default_value_t = 0.005)]
    pub beta_alpha: f64,
    /// Electron temperature in keV.
    #[arg(long, default_value_t = 15.0)]
    pub te_kev: f64,
    /// Ion temperature in keV.
    #[arg(long, default_value_t = 15.0)]
    pub ti_kev: f64,
    /// Electron density in m^-3.
    #[arg(long, default_value_t = 1.0e20)]
    pub ne: f64,
}

#[derive(Debug, Args)]
pub struct RunMhdStepArgs {
    /// Number of RK4 time steps.
    #[arg(long, default_value_t = 10)]
    pub steps: usize,
    /// Grid resolution along R and Z.
    #[arg(long, default_value_t = 17)]
    pub grid_size: usize,
    /// Kinematic viscosity.
    #[arg(long, default_value_t = 1.0e-3)]
    pub viscosity: f64,
    /// Electrical resistivity.
    #[arg(long, default_value_t = 1.0e-6)]
    pub resistivity: f64,
}

#[derive(Debug, Args)]
pub struct DemoArgs {
    /// Facility or design configuration name.
    #[arg(long, default_value = "ITER Baseline 15MA Burning Plasma")]
    pub facility: String,
}

impl Cli {
    /// Executes the parsed CLI command.
    pub fn run(self) {
        match self.command {
            Commands::SolveEquilibrium(args) => run_solve_equilibrium(args),
            Commands::TrackParticles(args) => run_track_particles(args),
            Commands::TaeSpectrum(args) => run_tae_spectrum(args),
            Commands::RunMhdStep(args) => run_mhd_step(args),
            Commands::Demo(args) => run_demo(args),
        }
    }
}

fn run_solve_equilibrium(args: SolveEquilibriumArgs) {
    println!("=== OuroborosPlasma: 2D Grad-Shafranov Equilibrium Solver ===");
    println!("Facility Geometry: R0 = {:.2} m, a = {:.2} m, B0 = {:.2} T, Ip = {:.1} MA", args.r0, args.a, args.b0, args.ip_ma);
    println!("Shaping: kappa = {:.2}, delta = {:.2}", args.kappa, args.delta);

    let r_min = args.r0 - args.a * 1.35;
    let r_max = args.r0 + args.a * 1.35;
    let z_min = -args.a * args.kappa * 1.35;
    let z_max = args.a * args.kappa * 1.35;

    let mut solver = GradShafranovSolver::new(args.grid_size, args.grid_size, r_min, r_max, z_min, z_max);
    let miller = MillerGeometry::new(args.r0, 0.0, args.a, args.kappa, args.delta, 0.12);
    solver.initialize_from_miller(&miller, 10.0);

    let p0 = args.p_core_kpa * 1.0e3;
    let profiles = PlasmaProfiles::new(p0, 2.0, 1.5, args.r0, args.a, args.b0, 0.0, 10.0);

    let (iters, res) = solver.solve(&profiles, args.max_iters, 1.0e-4);
    println!("Convergence achieved in {} iterations, final residual: {:.3e}", iters, res);
    println!("Magnetic axis: ({:.3}, {:.3}) m, flux at axis: {:.4} Wb/rad", solver.magnetic_axis.0, solver.magnetic_axis.1, solver.psi_axis);
    println!("Computed toroidal plasma current: {:.2} MA", solver.total_current_ip * 1.0e-6);

    if args.ascii {
        println!("\n{}", AsciiVisualizer::render_flux_contour(&solver, &miller, 65, 25));
        println!("\n{}", AsciiVisualizer::render_q_profile(1.05, 3.5, 65, 14));
    }
}

fn run_track_particles(args: TrackParticlesArgs) {
    println!("=== OuroborosPlasma: Relativistic Boris Alpha Confinement ===");
    println!("Particle ensemble size: {} alphas at {:.2} MeV", args.particles, args.energy_mev);
    println!("Tokamak field: R0 = {:.2} m, B0 = {:.2} T, q = {:.2}", args.r0, args.b0, args.q_safety);

    let mut ensemble = ParticleEnsemble::generate_dt_alphas(args.particles, args.r0, args.a, args.b0, args.q_safety, 12345);
    let initial_report = ensemble.compute_confinement_report();

    println!("Initial Classification:");
    println!("  Passing particles:   {:>4} ({:.1} %)", initial_report.passing_count, initial_report.passing_fraction * 100.0);
    println!("  Trapped bananas:     {:>4} ({:.1} %)", initial_report.trapped_count, initial_report.trapped_fraction * 100.0);
    println!("  Prompt loss to wall: {:>4} ({:.1} %)", initial_report.prompt_loss_count, initial_report.prompt_loss_fraction * 100.0);
    println!("  Confinement Fraction:{:>6.2} %", initial_report.confinement_fraction * 100.0);

    let dt = args.dt_ns * 1.0e-9;
    println!("\nAdvancing ensemble by {} Boris leapfrog steps (dt = {:.2} ns)...", args.steps, args.dt_ns);
    ensemble.advance_ensemble(dt, args.steps, |pos| {
        let (r, phi, _z) = pos.to_cylindrical();
        let bphi = if r > 0.1 { args.b0 * args.r0 / r } else { args.b0 };
        let b_cart = crate::math::Vector3D::cylindrical_to_cartesian_vector(0.0, bphi, 0.0, phi);
        (crate::math::Vector3D::zero(), b_cart)
    });

    let final_report = ensemble.compute_confinement_report();
    println!("Energetics Summary:");
    println!("  Total alpha birth energy:     {:.3e} J ({:.1} MeV)", final_report.total_birth_energy_joules, (final_report.total_birth_energy_joules / 1.602176634e-19) * 1.0e-6);
    println!("  Confined heating energy:      {:.3e} J ({:.1} %)", final_report.confined_energy_joules, final_report.confinement_fraction * 100.0);
    println!("  Promptly lost power/energy:   {:.3e} J ({:.1} %)", final_report.lost_energy_joules, final_report.prompt_loss_fraction * 100.0);
}

fn run_tae_spectrum(args: TaeSpectrumArgs) {
    println!("=== OuroborosPlasma: Toroidal Alfvén Eigenmodes (TAE) Spectral Analysis ===");
    println!("Mode numbers: n = {}, m = {}", args.toroidal_n, args.poloidal_m);
    println!("Facility parameters: R0 = {:.2} m, a = {:.2} m, B0 = {:.2} T", args.r0, args.a, args.b0);

    let continuum = AlfvenContinuum::new(args.r0, args.a, args.b0, args.toroidal_n, vec![args.poloidal_m, args.poloidal_m + 1]);
    let solver = TaeSolver::new(continuum);

    let q_profile = |rho: f64| 1.05 + 2.4 * rho * rho;
    let rho_mass = args.ne * DT_AVERAGE_ION_MASS;

    if let Some(mode) = solver.solve_tae_mode(args.poloidal_m, rho_mass, q_profile) {
        println!("\n[1] Continuum Gap Analysis:");
        println!("  TAE Gap Location:        r = {:.3} m  (rho = {:.3})", mode.r_tae, mode.rho_tae);
        println!("  Safety Factor at Gap:    q(r_TAE) = {:.3}", mode.q_tae);
        println!("  Gap Center Frequency:    f_TAE = {:.1} kHz  (omega = {:.2e} rad/s)", mode.frequency_hz * 1.0e-3, mode.omega_tae);
        println!("  Continuum Gap Width:     [{:.1} .. {:.1}] kHz  (Delta omega / omega = {:.1} %)", mode.omega_lower / (2.0 * std::f64::consts::PI) * 1.0e-3, mode.omega_upper / (2.0 * std::f64::consts::PI) * 1.0e-3, mode.gap_fraction * 100.0);

        let va = solver.continuum.alfven_speed(mode.r_tae, rho_mass);
        let damp_calc = DampingCalculator::new(args.te_kev * 1.0e3, args.ti_kev * 1.0e3, args.ne, DT_AVERAGE_ION_MASS);
        let damping = damp_calc.compute_damping(&mode, 0.65, va, 0.025, 0.025, args.b0);

        println!("\n[2] Thermal Damping Breakdown:");
        println!("  Continuum Damping:       gamma_cont / omega = {:>9.2e}", damping.gamma_continuum_normalized);
        println!("  Electron Landau Damping: gamma_e / omega    = {:>9.2e}", damping.gamma_electron_landau_normalized);
        println!("  Thermal Ion Landau:      gamma_i / omega    = {:>9.2e}", damping.gamma_ion_landau_normalized);
        println!("  Radiative Damping:       gamma_rad / omega  = {:>9.2e}", damping.gamma_radiative_normalized);
        println!("  Total Damping Rate:      gamma_damp / omega = {:>9.2e}  ({:.1} s^-1)", damping.gamma_total_normalized, damping.gamma_total_s1);

        let drive_calc = FastIonDriveCalculator::new();
        let report = drive_calc.analyze_stability(mode, args.beta_alpha, va, args.b0, 0.45, damping);

        println!("\n[3] Energetic Alpha Particle Resonant Drive & Stability:");
        println!("  Alpha Beta (beta_alpha): {:.4} %", report.beta_alpha * 100.0);
        println!("  Resonant Drive:          gamma_drive / omega = {:>9.2e}", report.gamma_drive_normalized);
        println!("  Net Growth Rate:         gamma_net / omega   = {:>9.2e}  ({:.1} s^-1)", report.gamma_net_normalized, report.gamma_net_s1);
        println!("  Critical Threshold Beta: beta_alpha_crit     = {:.4} %", report.beta_alpha_critical * 100.0);
        println!("  Stability Verdict:       {:?}", report.stability);
    } else {
        println!("No TAE gap found inside plasma for m = {}, n = {}", args.poloidal_m, args.toroidal_n);
    }
}

fn run_mhd_step(args: RunMhdStepArgs) {
    println!("=== OuroborosPlasma: 2D Magnetohydrodynamics Fluid Stepper ===");
    println!("Grid resolution: {}x{}", args.grid_size, args.grid_size);

    let grid = MhdGrid::new(args.grid_size, args.grid_size, 4.0, 8.4, -2.5, 2.5);
    let mut state = MhdState::new_uniform(&grid, 1.0e-7, 2.0e4, 5.0);

    let config = MhdStepperConfig {
        viscosity: args.viscosity,
        resistivity: args.resistivity,
        ..Default::default()
    };
    let stepper = MhdStepper::new(config);

    println!("Initial Energies: W_th = {:.2} MJ, W_B = {:.2} MJ, W_kin = {:.2} kJ",
        state.total_thermal_energy(&grid) * 1.0e-6,
        state.total_magnetic_energy(&grid) * 1.0e-6,
        state.total_kinetic_energy(&grid) * 1.0e-3
    );

    for s in 0..args.steps {
        let dt = stepper.compute_cfl_time_step(&state, &grid);
        stepper.step_rk4(&mut state, &grid, dt);
        if s == 0 || s == args.steps - 1 {
            println!("  Step {:>3}: t = {:.4e} s, dt = {:.4e} s, max div(B) = {:.2e}", s + 1, state.time, dt, stepper.compute_max_div_b(&state, &grid));
        }
    }

    println!("Final Energies:   W_th = {:.2} MJ, W_B = {:.2} MJ, W_kin = {:.2} kJ",
        state.total_thermal_energy(&grid) * 1.0e-6,
        state.total_magnetic_energy(&grid) * 1.0e-6,
        state.total_kinetic_energy(&grid) * 1.0e-3
    );
}

fn run_demo(args: DemoArgs) {
    println!("================================================================================");
    println!("           OUROBOROS-PLASMA: INTEGRATED TOKAMAK FUSION SIMULATOR                ");
    println!("================================================================================");

    // 1. Solve Equilibrium
    let r0 = 6.2;
    let a = 2.0;
    let b0 = 5.3;
    let ip = 15.0e6;
    let kappa = 1.7;
    let delta = 0.33;

    let mut solver = GradShafranovSolver::new(31, 31, 4.0, 8.4, -3.2, 3.2);
    let miller = MillerGeometry::new(r0, 0.0, a, kappa, delta, 0.15);
    solver.initialize_from_miller(&miller, 12.0);

    let p0 = 250.0e3; // 250 kPa
    let profiles = PlasmaProfiles::new(p0, 2.0, 1.5, r0, a, b0, 0.0, 12.0);
    let (iters, res) = solver.solve(&profiles, 150, 1.0e-4);

    println!("[1] Grad-Shafranov 2D Equilibrium Solved ({} iters, res: {:.2e})", iters, res);
    println!("\n{}", AsciiVisualizer::render_flux_contour(&solver, &miller, 60, 22));
    println!("\n{}", AsciiVisualizer::render_q_profile(1.05, 3.4, 60, 12));

    // 2. Alpha Particle Confinement
    let ensemble = ParticleEnsemble::generate_dt_alphas(300, r0, a, b0, 1.8, 999);
    let conf_rep = ensemble.compute_confinement_report();
    println!("\n[2] Energetic Alpha Particle Confinement (3.52 MeV DT Products):");
    println!("    Passing: {:.1}% | Trapped Banana: {:.1}% | Prompt Loss: {:.1}% | Confinement: {:.1}%",
        conf_rep.passing_fraction * 100.0,
        conf_rep.trapped_fraction * 100.0,
        conf_rep.prompt_loss_fraction * 100.0,
        conf_rep.confinement_fraction * 100.0
    );

    // 3. Toroidal Alfvén Eigenmodes (TAE)
    let continuum = AlfvenContinuum::new(r0, a, b0, 2, vec![2]);
    let tae_solver = TaeSolver::new(continuum);
    let rho_mass = 1.0e20 * DT_AVERAGE_ION_MASS;
    if let Some(mode) = tae_solver.solve_tae_mode(2, rho_mass, |rho| 1.05 + 2.35 * rho * rho) {
        let va = tae_solver.continuum.alfven_speed(mode.r_tae, rho_mass);
        let damp_calc = DampingCalculator::new(15.0e3, 15.0e3, 1.0e20, DT_AVERAGE_ION_MASS);
        let damping = damp_calc.compute_damping(&mode, 0.6, va, 0.02, 0.02, b0);
        let drive_calc = FastIonDriveCalculator::new();
        let stab = drive_calc.analyze_stability(mode, 0.006, va, b0, 0.45, damping);

        println!("\n[3] Toroidal Alfvén Eigenmode (TAE) Spectral Analysis (n=2, m=2):");
        println!("    f_TAE: {:.1} kHz | Gap: [{:.1}..{:.1}] kHz | gamma_net/omega: {:+.2e} ({:?})",
            stab.mode.frequency_hz * 1.0e-3,
            stab.mode.omega_lower / (2.0 * std::f64::consts::PI) * 1.0e-3,
            stab.mode.omega_upper / (2.0 * std::f64::consts::PI) * 1.0e-3,
            stab.gamma_net_normalized,
            stab.stability
        );
    }

    // 4. Fusion Performance Table
    let perf = AsciiVisualizer::evaluate_fusion_performance(
        &args.facility,
        r0,
        a,
        b0,
        ip,
        kappa,
        delta,
        1.0e20,
        15.0,
        1.05,
        3.4,
    );
    println!("\n{}", AsciiVisualizer::render_performance_table(&perf));
}

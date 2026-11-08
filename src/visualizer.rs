//! 7-bit ASCII visualization and fusion performance diagnostic reporting.
//!
//! Generates 100% portable 7-bit ASCII cross-sectional poloidal flux contour maps,
//! safety factor q(\rho) profile plots, and fusion performance reports.

use std::fmt::Write;
use crate::constants::{dt_fusion_reactivity, BOLTZMANN_CONSTANT, DT_FUSION_ENERGY_JOULES, MU_0};
use crate::equilibrium::{GradShafranovSolver, MillerGeometry};

/// Comprehensive fusion performance diagnostics summary.
#[derive(Debug, Clone)]
pub struct FusionPerformanceReport {
    pub machine_name: String,
    pub r0: f64,
    pub a: f64,
    pub aspect_ratio: f64,
    pub elongation: f64,
    pub triangularity: f64,
    pub plasma_volume_m3: f64,
    pub b0: f64,
    pub ip_amperes: f64,
    pub q0: f64,
    pub qa: f64,
    pub central_density_m3: f64,
    pub central_temperature_kev: f64,
    pub stored_thermal_energy_mj: f64,
    pub beta_toroidal_percent: f64,
    pub beta_poloidal: f64,
    pub beta_normalized_troyon: f64,
    pub greenwald_density_limit_m3: f64,
    pub greenwald_fraction: f64,
    pub fusion_reactivity_m3_s: f64,
    pub total_fusion_power_mw: f64,
    pub alpha_heating_power_mw: f64,
    pub tau_e_iter98y2_seconds: f64,
    pub fusion_gain_q: f64,
}

/// ASCII Visualizer for tokamak equilibria and fusion performance.
pub struct AsciiVisualizer;

impl AsciiVisualizer {
    /// Renders a 2D poloidal cross-section contour map in 7-bit ASCII.
    ///
    /// Characters used:
    /// - `*`: Magnetic axis
    /// - `@`: Core high-flux region (\bar{\psi} <= 0.15)
    /// - `%`: (\bar{\psi} <= 0.35)
    /// - `+`: (\bar{\psi} <= 0.55)
    /// - `=`: (\bar{\psi} <= 0.75)
    /// - `-`: (\bar{\psi} <= 0.90)
    /// - `.`: Edge plasma boundary (\bar{\psi} <= 1.00)
    /// - `#`: Vacuum vessel / limiter wall
    /// - ` `: Vacuum / outside wall
    pub fn render_flux_contour(
        solver: &GradShafranovSolver,
        miller: &MillerGeometry,
        width: usize,
        height: usize,
    ) -> String {
        let width = width.clamp(30, 100);
        let height = height.clamp(15, 60);

        let mut out = String::new();
        writeln!(&mut out, "+{}+", "-".repeat(width)).unwrap();
        writeln!(&mut out, "|  OUROBOROS-PLASMA 2D POLOIDAL FLUX CROSS-SECTION MAP (R vs Z)  |").unwrap();
        writeln!(&mut out, "+{}+", "-".repeat(width)).unwrap();

        let dr = (solver.r_max - solver.r_min) / (width as f64);
        let dz = (solver.z_max - solver.z_min) / (height as f64);

        let (r_axis, z_axis) = solver.magnetic_axis;

        for row in 0..height {
            let j = height - 1 - row;
            let z = solver.z_min + (j as f64 + 0.5) * dz;

            // Coordinate label on left
            let mut line = String::with_capacity(width + 10);
            line.push('|');

            for col in 0..width {
                let r = solver.r_min + (col as f64 + 0.5) * dr;

                // Check distance to magnetic axis
                let dist_axis = ((r - r_axis) * (r - r_axis) + (z - z_axis) * (z - z_axis)).sqrt();
                if dist_axis < 0.5 * dr.max(dz) {
                    line.push('*');
                    continue;
                }

                // Check Miller boundary for limiter outline
                let (r_rel, z_rel) = (r - miller.r0 - miller.shafranov_shift, z - miller.z0);
                let rho2 = (r_rel / miller.minor_radius).powi(2)
                    + (z_rel / (miller.elongation * miller.minor_radius)).powi(2);

                if (rho2 - 1.0).abs() < 0.08 {
                    line.push('#');
                    continue;
                }

                if rho2 > 1.05 {
                    line.push(' ');
                    continue;
                }

                // Inside plasma: sample normalized flux
                let psi = solver.evaluate_psi(r, z);
                let psi_denom = solver.psi_edge - solver.psi_axis;
                let psi_norm = if psi_denom.abs() > 1.0e-12 {
                    ((psi - solver.psi_axis) / psi_denom).clamp(0.0, 1.0)
                } else {
                    rho2.clamp(0.0, 1.0)
                };

                let ch = if psi_norm < 0.15 {
                    '@'
                } else if psi_norm < 0.35 {
                    '%'
                } else if psi_norm < 0.55 {
                    '+'
                } else if psi_norm < 0.75 {
                    '='
                } else if psi_norm < 0.90 {
                    '-'
                } else {
                    '.'
                };

                line.push(ch);
            }

            line.push('|');
            writeln!(&mut out, "{}", line).unwrap();
        }

        writeln!(&mut out, "+{}+", "-".repeat(width)).unwrap();
        writeln!(
            &mut out,
            "  R: [{:.2} m .. {:.2} m]   Z: [{:.2} m .. {:.2} m]   Axis: ({:.2}, {:.2}) m",
            solver.r_min, solver.r_max, solver.z_min, solver.z_max, r_axis, z_axis
        )
        .unwrap();
        writeln!(
            &mut out,
            "  Legend: [*] Axis  [@] Core  [%] Mid-Core  [+] Mid  [=] Outer  [-] Edge  [#] Wall"
        )
        .unwrap();

        out
    }

    /// Renders the safety factor q(\rho) profile curve plot in 7-bit ASCII.
    pub fn render_q_profile(
        q0: f64,
        qa: f64,
        width: usize,
        height: usize,
    ) -> String {
        let width = width.clamp(35, 80);
        let height = height.clamp(10, 30);

        let mut out = String::new();
        writeln!(&mut out, "+{}+", "-".repeat(width)).unwrap();
        writeln!(&mut out, "|              SAFETY FACTOR q(rho) RADIAL PROFILE PLOT            |").unwrap();
        writeln!(&mut out, "+{}+", "-".repeat(width)).unwrap();

        let q_min = (q0 * 0.9).floor();
        let q_max = (qa * 1.1).ceil();
        let q_range = q_max - q_min;

        for row in 0..height {
            let j = height - 1 - row;
            let q_level = q_min + (j as f64 + 0.5) * (q_range / height as f64);

            let mut line = String::new();
            write!(&mut line, "{:4.1} |", q_level).unwrap();

            for col in 0..width - 6 {
                let rho = col as f64 / ((width - 7) as f64);
                // Parabolic q model: q(rho) = q0 + (qa - q0) rho^2
                let q_val = q0 + (qa - q0) * rho * rho;

                let row_matching = ((q_val - q_min) / q_range * (height as f64)).floor() as usize;
                if row_matching == j {
                    line.push('*');
                } else if (q_level - 1.0).abs() < 0.5 * (q_range / height as f64) {
                    line.push(':'); // Sawtooth boundary q = 1
                } else if (q_level - 2.0).abs() < 0.5 * (q_range / height as f64) {
                    line.push('-'); // Disruption limit q = 2
                } else {
                    line.push(' ');
                }
            }

            writeln!(&mut out, "{}", line).unwrap();
        }

        writeln!(&mut out, "     +{}+", "-".repeat(width - 6)).unwrap();
        writeln!(&mut out, "     0.0          0.25         0.50         0.75         1.00").unwrap();
        writeln!(&mut out, "                               rho = r / a").unwrap();
        writeln!(
            &mut out,
            "  q(0) = {:.2}   q_a = {:.2}   q=1.0 sawteeth[:]   q=2.0 disruption[-]",
            q0, qa
        )
        .unwrap();

        out
    }

    /// Evaluates fusion power, alpha heating, and energy confinement to build a full performance report.
    pub fn evaluate_fusion_performance(
        name: &str,
        r0: f64,
        a: f64,
        b0: f64,
        ip_amperes: f64,
        elongation: f64,
        triangularity: f64,
        central_density_m3: f64,
        central_temperature_kev: f64,
        q0: f64,
        qa: f64,
    ) -> FusionPerformanceReport {
        let aspect_ratio = r0 / a;
        // Plasma volume: V \approx 2 \pi^2 R_0 \kappa a^2
        let plasma_vol = 2.0 * std::f64::consts::PI * std::f64::consts::PI * r0 * elongation * a * a;

        // Bosch-Hale reactivity for D-T: <\sigma v> in m^3 / s
        let sigv = dt_fusion_reactivity(central_temperature_kev);

        // Volumetric fusion reaction rate in core: R_fus = n_D n_T <\sigma v> = (n_e / 2)^2 <\sigma v>
        let n_fuel = central_density_m3 * 0.5;
        let reaction_rate_core = n_fuel * n_fuel * sigv;

        // Volume-integrated fusion power: P_fus \approx 0.35 * R_fus,core * V * E_DT (for parabolic-squared profile)
        let profile_vol_factor = 0.32;
        let total_reactions_s = reaction_rate_core * plasma_vol * profile_vol_factor;

        let p_fusion_watts = total_reactions_s * DT_FUSION_ENERGY_JOULES;
        let p_alpha_watts = p_fusion_watts * (3.52 / 17.59);

        let p_fusion_mw = p_fusion_watts * 1.0e-6;
        let p_alpha_mw = p_alpha_watts * 1.0e-6;

        // Stored thermal energy: W_th = 3/2 \int (n_e T_e + n_i T_i) dV
        // In Joules: 3 n_e k_B T * V * profile_factor
        let temp_kelvin = central_temperature_kev * crate::constants::KEV_TO_KELVIN;
        let w_th_joules = 3.0 * central_density_m3 * BOLTZMANN_CONSTANT * temp_kelvin * plasma_vol * 0.38;
        let w_th_mj = w_th_joules * 1.0e-6;

        // Pressure in Pascals: p \approx 2 n_e k_B T
        let p_avg = (w_th_joules / (1.5 * plasma_vol)).max(1.0);
        let beta_t = (2.0 * MU_0 * p_avg / (b0 * b0)) * 100.0; // percent

        let ip_ma = ip_amperes * 1.0e-6;
        let beta_n = if ip_ma > 0.0 { beta_t / (ip_ma / (a * b0)) } else { 0.0 };

        // Poloidal beta: \beta_p \approx \beta_t * (q_a * R_0 / a)^2 / \kappa
        let beta_p = (beta_t / 100.0) * (qa * aspect_ratio).powi(2) / elongation;

        // Greenwald density limit: n_G = I_p [MA] / (\pi a^2) * 10^20 m^-3
        let n_g = (ip_ma / (std::f64::consts::PI * a * a)) * 1.0e20;
        let greenwald_frac = central_density_m3 / n_g.max(1.0);

        // ITER98y2 H-mode energy confinement scaling:
        // \tau_{E, 98y2} = 0.0562 * I_p^{0.93} * B_0^{0.15} * P_{net}^{-0.69} * n_e^{0.41} * M^{0.19} * R_0^{1.97} * \epsilon^{0.58} * \kappa^{0.78}
        let eps = a / r0;
        let n_19 = central_density_m3 * 1.0e-19;
        let mass_amu: f64 = 2.5; // D-T
        let p_aux_mw = (p_fusion_mw / 10.0).max(20.0); // Assume auxiliary heating
        let p_net_mw = p_aux_mw + p_alpha_mw * 0.8; // Net heating

        let tau_e = 0.0562
            * ip_ma.powf(0.93)
            * b0.powf(0.15)
            * p_net_mw.powf(-0.69)
            * n_19.powf(0.41)
            * mass_amu.powf(0.19)
            * r0.powf(1.97)
            * eps.powf(0.58)
            * elongation.powf(0.78);

        // Fusion gain factor: Q = P_fus / P_aux
        let q_gain = p_fusion_mw / p_aux_mw;

        FusionPerformanceReport {
            machine_name: name.to_string(),
            r0,
            a,
            aspect_ratio,
            elongation,
            triangularity,
            plasma_volume_m3: plasma_vol,
            b0,
            ip_amperes,
            q0,
            qa,
            central_density_m3,
            central_temperature_kev,
            stored_thermal_energy_mj: w_th_mj,
            beta_toroidal_percent: beta_t,
            beta_poloidal: beta_p,
            beta_normalized_troyon: beta_n,
            greenwald_density_limit_m3: n_g,
            greenwald_fraction: greenwald_frac,
            fusion_reactivity_m3_s: sigv,
            total_fusion_power_mw: p_fusion_mw,
            alpha_heating_power_mw: p_alpha_mw,
            tau_e_iter98y2_seconds: tau_e,
            fusion_gain_q: q_gain,
        }
    }

    /// Formats the complete fusion performance report as a 7-bit ASCII table.
    pub fn render_performance_table(rep: &FusionPerformanceReport) -> String {
        let mut out = String::new();
        let border = "+----------------------------------------------------------------------+";
        writeln!(&mut out, "{}", border).unwrap();
        writeln!(&mut out, "|       OUROBOROS-PLASMA COMPREHENSIVE FUSION PERFORMANCE REPORT       |").unwrap();
        writeln!(&mut out, "|  Facility / Benchmark: {:<45} |", rep.machine_name).unwrap();
        writeln!(&mut out, "{}", border).unwrap();

        writeln!(&mut out, "|  1. MACHINE GEOMETRY & MAGNETIC EQUILIBRIUM                          |").unwrap();
        writeln!(&mut out, "|     Major Radius (R_0):          {:>8.2} m   Aspect Ratio (A):     {:>6.2} |", rep.r0, rep.aspect_ratio).unwrap();
        writeln!(&mut out, "|     Minor Radius (a):            {:>8.2} m   Elongation (kappa):   {:>6.2} |", rep.a, rep.elongation).unwrap();
        writeln!(&mut out, "|     Triangularity (delta):       {:>8.2}     Plasma Volume:     {:>8.1} m^3 |", rep.triangularity, rep.plasma_volume_m3).unwrap();
        writeln!(&mut out, "|     Toroidal Field (B_0):        {:>8.2} T   Plasma Current (I_p): {:>6.1} MA |", rep.b0, rep.ip_amperes * 1.0e-6).unwrap();
        writeln!(&mut out, "|     Safety Factor On-Axis (q_0): {:>8.2}     Edge Safety (q_a):    {:>6.2} |", rep.q0, rep.qa).unwrap();
        writeln!(&mut out, "{}", border).unwrap();

        writeln!(&mut out, "|  2. KINETIC STATE & STABILITY METRICS                                |").unwrap();
        writeln!(&mut out, "|     Central Density (n_e0):    {:>10.2e} m^-3                      |", rep.central_density_m3).unwrap();
        writeln!(&mut out, "|     Central Temperature (T_0):   {:>8.2} keV                       |", rep.central_temperature_kev).unwrap();
        writeln!(&mut out, "|     Stored Thermal Energy (W_th):{:>8.1} MJ                        |", rep.stored_thermal_energy_mj).unwrap();
        writeln!(&mut out, "|     Toroidal Beta (beta_t):      {:>8.2} %   Poloidal Beta (beta_p):{:>5.2} |", rep.beta_toroidal_percent, rep.beta_poloidal).unwrap();
        writeln!(&mut out, "|     Normalized Troyon (beta_N):  {:>8.2}     Greenwald Limit (n_G): {:>4.1e} |", rep.beta_normalized_troyon, rep.greenwald_density_limit_m3).unwrap();
        writeln!(&mut out, "|     Greenwald Fraction (f_G):    {:>8.2} %                         |", rep.greenwald_fraction * 100.0).unwrap();
        writeln!(&mut out, "{}", border).unwrap();

        writeln!(&mut out, "|  3. FUSION ENERGETICS & CONFINEMENT                                  |").unwrap();
        writeln!(&mut out, "|     D-T Reactivity <sigma v>:  {:>10.2e} m^3/s                     |", rep.fusion_reactivity_m3_s).unwrap();
        writeln!(&mut out, "|     Total Fusion Power (P_fus):  {:>8.1} MW                        |", rep.total_fusion_power_mw).unwrap();
        writeln!(&mut out, "|     Alpha Heating Power (P_alpha):{:>7.1} MW                       |", rep.alpha_heating_power_mw).unwrap();
        writeln!(&mut out, "|     Energy Confinement (tau_E):  {:>8.2} s  (ITER98y2 H-mode)      |", rep.tau_e_iter98y2_seconds).unwrap();
        writeln!(&mut out, "|     Fusion Energy Gain (Q):      {:>8.1}                           |", rep.fusion_gain_q).unwrap();
        writeln!(&mut out, "{}", border).unwrap();

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iter_fusion_performance_calculation() {
        let rep = AsciiVisualizer::evaluate_fusion_performance(
            "ITER Baseline 15MA Q=10",
            6.2,
            2.0,
            5.3,
            15.0e6,
            1.7,
            0.33,
            1.0e20,
            15.0, // 15 keV core
            1.05,
            3.2,
        );

        assert!(rep.total_fusion_power_mw > 300.0 && rep.total_fusion_power_mw < 700.0, "ITER fusion power should be ~ 400-600 MW: {}", rep.total_fusion_power_mw);
        assert!(rep.alpha_heating_power_mw > 70.0 && rep.alpha_heating_power_mw < 150.0);
        assert!(rep.tau_e_iter98y2_seconds > 2.0 && rep.tau_e_iter98y2_seconds < 6.0, "ITER confinement time should be ~ 3-4 s: {}", rep.tau_e_iter98y2_seconds);
        assert!(rep.fusion_gain_q > 5.0, "ITER Q factor should exceed 5: {}", rep.fusion_gain_q);

        let table = AsciiVisualizer::render_performance_table(&rep);
        assert!(table.contains("OUROBOROS-PLASMA"));
        assert!(table.contains("ITER Baseline"));
    }

    #[test]
    fn test_7bit_ascii_only() {
        let rep = AsciiVisualizer::evaluate_fusion_performance(
            "Test Tokamak",
            3.0,
            1.0,
            3.0,
            2.0e6,
            1.5,
            0.2,
            5.0e19,
            10.0,
            1.0,
            3.0,
        );
        let table = AsciiVisualizer::render_performance_table(&rep);
        for b in table.bytes() {
            assert!(b <= 127, "Must contain 7-bit ASCII characters only! Found: {}", b);
        }

        let q_plot = AsciiVisualizer::render_q_profile(1.05, 3.5, 40, 15);
        for b in q_plot.bytes() {
            assert!(b <= 127, "q plot must be 7-bit ASCII only! Found: {}", b);
        }
    }
}

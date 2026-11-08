//! Energetic alpha particle resonant drive and TAE linear stability threshold analysis.
//!
//! Evaluates the kinetic resonant drive:
//! \frac{\gamma_\alpha}{\omega} \approx \frac{9}{4} q^2 \beta_\alpha \left( \frac{\omega_{*\alpha}}{\omega} - 1 \right) F_{res}\left( \frac{v_A}{v_{\alpha 0}} \right)
//! and determines whether the net growth rate \gamma_{net} = \gamma_\alpha - \gamma_{damp} exceeds zero.

use serde::{Deserialize, Serialize};
use crate::constants::{ALPHA_BIRTH_ENERGY_JOULES, ALPHA_CHARGE, ALPHA_MASS};
use super::damping::DampingBreakdown;
use super::tae::TaeMode;

/// Stability status of a Toroidal Alfvén Eigenmode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModeStability {
    /// Net growth rate is negative (\gamma_{net} < 0), mode is damped.
    Stable,
    /// Mode is near marginal stability (-1e-4 <= \gamma_{net}/\omega <= 1e-4).
    Marginal,
    /// Mode is linearly unstable (\gamma_{net} > 0), leading to alpha transport.
    Unstable,
}

/// Analysis report of fast-ion drive and linear stability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaeStabilityReport {
    pub mode: TaeMode,
    /// Alpha diamagnetic frequency \omega_{*\alpha} (rad/s).
    pub omega_star_alpha: f64,
    /// Ratio of diamagnetic to mode frequency \omega_{*\alpha} / \omega_{TAE}.
    pub drive_ratio: f64,
    /// Energetic alpha particle beta \beta_\alpha.
    pub beta_alpha: f64,
    /// Normalized alpha resonant drive rate \gamma_\alpha / \omega.
    pub gamma_drive_normalized: f64,
    /// Damping breakdown components.
    pub damping: DampingBreakdown,
    /// Net growth rate \gamma_{net} / \omega = \gamma_\alpha / \omega - \gamma_{damp} / \omega.
    pub gamma_net_normalized: f64,
    /// Net linear growth rate in s^-1.
    pub gamma_net_s1: f64,
    /// Critical threshold alpha beta \beta_{\alpha, crit} for onset of instability.
    pub beta_alpha_critical: f64,
    /// Final linear stability classification.
    pub stability: ModeStability,
}

/// Evaluates fast-ion drive and mode stability.
#[derive(Debug, Clone)]
pub struct FastIonDriveCalculator {
    pub alpha_birth_energy_joules: f64,
}

impl Default for FastIonDriveCalculator {
    fn default() -> Self {
        Self {
            alpha_birth_energy_joules: ALPHA_BIRTH_ENERGY_JOULES,
        }
    }
}

impl FastIonDriveCalculator {
    /// Creates a new fast ion drive calculator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Evaluates the alpha particle birth speed v_{\alpha 0} = \sqrt{2 E_{\alpha 0} / m_\alpha}.
    pub fn alpha_birth_speed(&self) -> f64 {
        (2.0 * self.alpha_birth_energy_joules / ALPHA_MASS).sqrt()
    }

    /// Resonance velocity function F_{res}(y) = y (1 + 2 y^2 + 2 y^4) \exp(-y^2) where y = v_A / v_{\alpha 0}.
    pub fn resonance_function(&self, va: f64) -> f64 {
        let v0 = self.alpha_birth_speed();
        let y = (va / v0).clamp(0.01, 5.0);
        let y2 = y * y;
        let y4 = y2 * y2;
        y * (1.0 + 2.0 * y2 + 2.0 * y4) * (-y2).exp()
    }

    /// Evaluates alpha diamagnetic frequency:
    /// \omega_{*\alpha} = - \frac{m}{q_\alpha B r} \frac{T_\alpha}{n_\alpha} \frac{dn_\alpha}{dr} \approx \frac{m T_\alpha}{q_\alpha B r L_{n\alpha}}.
    pub fn alpha_diamagnetic_frequency(
        &self,
        mode: &TaeMode,
        b0: f64,
        alpha_scale_length: f64,
    ) -> f64 {
        let m = mode.poloidal_m.abs() as f64;
        let r = mode.r_tae.max(0.1);
        let l_n = alpha_scale_length.max(0.1);
        // Effective alpha temperature: T_\alpha \approx 1.0 MeV
        let t_alpha_joules = 1.0e6 * crate::constants::ELEMENTARY_CHARGE;
        (m * t_alpha_joules) / (ALPHA_CHARGE * b0 * r * l_n)
    }

    /// Evaluates normalized fast-ion resonant drive rate \gamma_\alpha / \omega:
    /// \gamma_\alpha / \omega = \frac{9}{4} q^2 \beta_\alpha (\omega_{*\alpha} / \omega - 1) F_{res}(v_A / v_{\alpha 0}).
    pub fn evaluate_resonant_drive(
        &self,
        mode: &TaeMode,
        beta_alpha: f64,
        va: f64,
        omega_star: f64,
    ) -> f64 {
        if beta_alpha <= 0.0 {
            return 0.0;
        }
        let q = mode.q_tae;
        let f_res = self.resonance_function(va);
        let drive_factor = (omega_star / mode.omega_tae - 1.0).max(0.0);

        (9.0 / 4.0) * q * q * beta_alpha * drive_factor * f_res
    }

    /// Analyzes the full stability of a TAE mode against fast alpha resonant drive and damping.
    pub fn analyze_stability(
        &self,
        mode: TaeMode,
        beta_alpha: f64,
        va: f64,
        b0: f64,
        alpha_scale_length: f64,
        damping: DampingBreakdown,
    ) -> TaeStabilityReport {
        let omega_star = self.alpha_diamagnetic_frequency(&mode, b0, alpha_scale_length);
        let drive_ratio = omega_star / mode.omega_tae;
        let gamma_drive = self.evaluate_resonant_drive(&mode, beta_alpha, va, omega_star);

        let gamma_damp = damping.gamma_total_normalized;
        let gamma_net = gamma_drive - gamma_damp;
        let gamma_net_s1 = gamma_net * mode.omega_tae;

        // Critical beta: \beta_{\alpha, crit} = \gamma_{damp} / [ (9/4) q^2 (\omega_* / \omega - 1) F_res ]
        let q = mode.q_tae;
        let f_res = self.resonance_function(va);
        let drive_factor = (drive_ratio - 1.0).max(1.0e-5);
        let denom = (9.0 / 4.0) * q * q * drive_factor * f_res;
        let beta_crit = if denom > 1.0e-12 { gamma_damp / denom } else { f64::INFINITY };

        let stability = if gamma_net > 1.0e-4 {
            ModeStability::Unstable
        } else if gamma_net < -1.0e-4 {
            ModeStability::Stable
        } else {
            ModeStability::Marginal
        };

        TaeStabilityReport {
            mode,
            omega_star_alpha: omega_star,
            drive_ratio,
            beta_alpha,
            gamma_drive_normalized: gamma_drive,
            damping,
            gamma_net_normalized: gamma_net,
            gamma_net_s1,
            beta_alpha_critical: beta_crit,
            stability,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modes::alfven::AlfvenContinuum;
    use crate::modes::damping::DampingCalculator;
    use crate::modes::tae::TaeSolver;

    #[test]
    fn test_tae_stability_threshold() {
        let continuum = AlfvenContinuum::new(6.2, 2.0, 5.3, 2, vec![2]);
        let solver = TaeSolver::new(continuum);
        let mode = solver.solve_tae_mode(2, 1.0e-7, |rho| 1.0 + 2.0 * rho * rho).unwrap();

        let calc_damp = DampingCalculator::new(10.0e3, 10.0e3, 1.0e20, crate::constants::DT_AVERAGE_ION_MASS);
        let damping = calc_damp.compute_damping(&mode, 0.5, 1.5e7, 0.02, 0.02, 5.3);

        let drive_calc = FastIonDriveCalculator::new();

        // Low alpha beta (0.01%): mode should be stable
        let rep_stable = drive_calc.analyze_stability(mode.clone(), 0.0001, 1.5e7, 5.3, 0.15, damping.clone());
        assert_eq!(rep_stable.stability, ModeStability::Stable);
        assert!(rep_stable.gamma_net_normalized < 0.0);

        // High alpha beta (2.0%): mode should be unstable
        let rep_unstable = drive_calc.analyze_stability(mode, 0.02, 1.5e7, 5.3, 0.15, damping);
        assert_eq!(rep_unstable.stability, ModeStability::Unstable);
        assert!(rep_unstable.gamma_net_normalized > 0.0);
    }
}

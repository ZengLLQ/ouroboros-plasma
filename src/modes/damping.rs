//! Damping mechanisms for Toroidal Alfvén Eigenmodes.
//!
//! Evaluates:
//! 1. Continuum damping: \gamma_{cont} / \omega
//! 2. Electron Landau damping: \gamma_e / \omega
//! 3. Ion Landau damping: \gamma_i / \omega
//! 4. Radiative damping: \gamma_{rad} / \omega
//! 5. Total composite damping rate: \gamma_{damp} = \sum \gamma_k

use std::f64::consts::PI;
use serde::{Deserialize, Serialize};
use crate::constants::ELECTRON_MASS;
use super::tae::TaeMode;

/// Comprehensive breakdown of all damping rates acting on a TAE mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DampingBreakdown {
    /// Normalized continuum damping rate \gamma_{cont} / \omega.
    pub gamma_continuum_normalized: f64,
    /// Normalized electron Landau damping rate \gamma_e / \omega.
    pub gamma_electron_landau_normalized: f64,
    /// Normalized ion Landau damping rate \gamma_i / \omega.
    pub gamma_ion_landau_normalized: f64,
    /// Normalized radiative damping rate \gamma_{rad} / \omega.
    pub gamma_radiative_normalized: f64,
    /// Total normalized damping rate \gamma_{total} / \omega.
    pub gamma_total_normalized: f64,
    /// Total damping rate \gamma_{total} in s^-1.
    pub gamma_total_s1: f64,
}

/// Damping rate calculator for TAE eigenmodes.
#[derive(Debug, Clone)]
pub struct DampingCalculator {
    pub electron_temperature_ev: f64,
    pub ion_temperature_ev: f64,
    pub electron_density_m3: f64,
    pub ion_mass_kg: f64,
}

impl DampingCalculator {
    /// Creates a damping calculator for given plasma thermal background.
    pub fn new(
        electron_temperature_ev: f64,
        ion_temperature_ev: f64,
        electron_density_m3: f64,
        ion_mass_kg: f64,
    ) -> Self {
        Self {
            electron_temperature_ev,
            ion_temperature_ev,
            electron_density_m3,
            ion_mass_kg,
        }
    }

    /// Evaluates continuum damping rate:
    /// \gamma_{cont} / \omega \approx \frac{\pi}{4} \hat{s}^2 \exp\left( - \frac{2}{\hat{\epsilon} \hat{s}} \right).
    pub fn continuum_damping(&self, mode: &TaeMode, magnetic_shear: f64) -> f64 {
        let s = magnetic_shear.max(0.01);
        let eps_hat = mode.gap_fraction.max(0.01);
        let arg = 2.0 / (eps_hat * s);
        if arg > 80.0 {
            return 0.0;
        }
        (PI / 4.0) * s * s * (-arg).exp()
    }

    /// Evaluates electron Landau damping rate:
    /// \gamma_e / \omega \approx q^2 \beta_e \sqrt{\pi} \frac{v_A}{v_{te}} \exp\left( - \frac{v_A^2}{v_{te}^2} \right).
    pub fn electron_landau_damping(
        &self,
        mode: &TaeMode,
        alfven_speed: f64,
        beta_electron: f64,
    ) -> f64 {
        // Electron thermal speed: v_{te} = \sqrt{2 T_e / m_e}
        let e_joules = self.electron_temperature_ev * crate::constants::ELEMENTARY_CHARGE;
        let v_te = (2.0 * e_joules / ELECTRON_MASS).sqrt();

        let ratio = alfven_speed / v_te;
        let q = mode.q_tae;

        q * q * beta_electron * PI.sqrt() * ratio * (-ratio * ratio).exp()
    }

    /// Evaluates thermal ion Landau damping rate:
    /// \gamma_i / \omega \approx q^2 \beta_i \sqrt{\pi} (v_A / v_{ti})^3 \exp\left( - \frac{v_A^2}{v_{ti}^2} \right).
    pub fn ion_landau_damping(
        &self,
        mode: &TaeMode,
        alfven_speed: f64,
        beta_ion: f64,
    ) -> f64 {
        let i_joules = self.ion_temperature_ev * crate::constants::ELEMENTARY_CHARGE;
        let v_ti = (2.0 * i_joules / self.ion_mass_kg).sqrt();

        let ratio = (alfven_speed / v_ti).min(30.0);
        let q = mode.q_tae;

        q * q * beta_ion * PI.sqrt() * ratio.powi(3) * (-ratio * ratio).exp()
    }

    /// Evaluates radiative damping rate due to tunneling into kinetic Alfvén waves:
    /// \gamma_{rad} / \omega \approx 0.5 \hat{s}^{3/2} (\rho_i / r_{TAE}).
    pub fn radiative_damping(&self, mode: &TaeMode, magnetic_shear: f64, b0: f64) -> f64 {
        let i_joules = self.ion_temperature_ev * crate::constants::ELEMENTARY_CHARGE;
        let v_ti = (2.0 * i_joules / self.ion_mass_kg).sqrt();
        let omega_ci = (crate::constants::ELEMENTARY_CHARGE * b0) / self.ion_mass_kg;
        let rho_i = v_ti / omega_ci;

        let s = magnetic_shear.max(0.01);
        let r = mode.r_tae.max(0.1);
        0.5 * s.powf(1.5) * (rho_i / r)
    }

    /// Computes full damping rate breakdown for a given TAE mode.
    pub fn compute_damping(
        &self,
        mode: &TaeMode,
        magnetic_shear: f64,
        alfven_speed: f64,
        beta_electron: f64,
        beta_ion: f64,
        b0: f64,
    ) -> DampingBreakdown {
        let g_cont = self.continuum_damping(mode, magnetic_shear);
        let g_e = self.electron_landau_damping(mode, alfven_speed, beta_electron);
        let g_i = self.ion_landau_damping(mode, alfven_speed, beta_ion);
        let g_rad = self.radiative_damping(mode, magnetic_shear, b0);

        let g_total_norm = g_cont + g_e + g_i + g_rad;
        let g_total_s1 = g_total_norm * mode.omega_tae;

        DampingBreakdown {
            gamma_continuum_normalized: g_cont,
            gamma_electron_landau_normalized: g_e,
            gamma_ion_landau_normalized: g_i,
            gamma_radiative_normalized: g_rad,
            gamma_total_normalized: g_total_norm,
            gamma_total_s1: g_total_s1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::DT_AVERAGE_ION_MASS;
    use crate::modes::alfven::AlfvenContinuum;
    use crate::modes::tae::TaeSolver;

    #[test]
    fn test_damping_rates() {
        let continuum = AlfvenContinuum::new(6.2, 2.0, 5.3, 2, vec![2]);
        let solver = TaeSolver::new(continuum);
        let mode = solver.solve_tae_mode(2, 1.0e-7, |rho| 1.0 + 2.0 * rho * rho).unwrap();

        let calc = DampingCalculator::new(10.0e3, 10.0e3, 1.0e20, DT_AVERAGE_ION_MASS);
        let damping = calc.compute_damping(&mode, 0.5, 1.5e7, 0.02, 0.02, 5.3);

        assert!(damping.gamma_total_normalized > 0.0);
        assert!(damping.gamma_total_normalized < 0.1, "TAE damping is typically ~ 0.1% to 5% of omega");
        assert!(damping.gamma_total_s1 > 0.0);
    }
}

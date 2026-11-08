//! Physical and computational constants for fusion plasma physics.
//!
//! All quantities are defined in SI units (m, kg, s, A, K, J, T, V, etc.)
//! unless explicitly noted otherwise.

use std::f64::consts::PI;

/// Speed of light in vacuum (m/s).
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// Permeability of free space \mu_0 (H/m or N/A^2).
pub const MU_0: f64 = 4.0 * PI * 1.0e-7;

/// Permittivity of free space \epsilon_0 (F/m).
pub const EPSILON_0: f64 = 1.0 / (MU_0 * SPEED_OF_LIGHT * SPEED_OF_LIGHT);

/// Elementary electric charge (C).
pub const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19;

/// Boltzmann constant (J/K).
pub const BOLTZMANN_CONSTANT: f64 = 1.380_649e-23;

/// Electron mass (kg).
pub const ELECTRON_MASS: f64 = 9.109_383_7015e-31;

/// Proton mass (kg).
pub const PROTON_MASS: f64 = 1.672_621_923_69e-27;

/// Deuteron mass (kg).
pub const DEUTERON_MASS: f64 = 3.343_583_7724e-27;

/// Triton mass (kg).
pub const TRITON_MASS: f64 = 5.007_356_7446e-27;

/// Alpha particle (^4He nucleus) mass (kg).
pub const ALPHA_MASS: f64 = 6.644_657_230e-27;

/// Alpha particle charge (C) = +2e.
pub const ALPHA_CHARGE: f64 = 2.0 * ELEMENTARY_CHARGE;

/// Average 50:50 Deuterium-Tritium plasma ion mass (kg).
pub const DT_AVERAGE_ION_MASS: f64 = 0.5 * (DEUTERON_MASS + TRITON_MASS);

/// Alpha particle birth kinetic energy from D-T fusion: 3.52 MeV (Joules).
pub const ALPHA_BIRTH_ENERGY_JOULES: f64 = 3.52e6 * ELEMENTARY_CHARGE;

/// Total energy released in single D-T fusion event: 17.59 MeV (Joules).
pub const DT_FUSION_ENERGY_JOULES: f64 = 17.59e6 * ELEMENTARY_CHARGE;

/// Ratio of specific heats for fully ionized monoatomic plasma \gamma = 5/3.
pub const ADIABATIC_INDEX_GAMMA: f64 = 5.0 / 3.0;

/// Conversion factor from keV to Kelvin: 1 keV = e / k_B K.
pub const KEV_TO_KELVIN: f64 = (1.0e3 * ELEMENTARY_CHARGE) / BOLTZMANN_CONSTANT;

/// Conversion factor from keV to Joules.
pub const KEV_TO_JOULES: f64 = 1.0e3 * ELEMENTARY_CHARGE;

/// Computes the thermonuclear Maxwellian D-T fusion reactivity <\sigma v>_DT.
///
/// Temperature `t_kev` is the ion temperature in keV (valid range ~0.5 keV to 100 keV).
/// Parameterized according to the standard NRL Plasma Formulary & Bosch-Hale formulation.
/// Returns reactivity in m^3 / s.
pub fn dt_fusion_reactivity(t_kev: f64) -> f64 {
    if t_kev <= 0.3 {
        return 0.0;
    }

    // Standard NRL parameterization for D-T:
    // <\sigma v> \approx 3.68e-18 * T^(-2/3) * exp(-19.94 / T^(1/3)) * C(T)  [m^3/s]
    let t_13 = t_kev.powf(1.0 / 3.0);
    let t_23 = t_13 * t_13;
    let exp_term = (-19.94 / t_13).exp();

    // High-temperature correction factor C(T)
    let c_t = 1.0 + 0.0035 * t_kev.powf(1.25);
    let sigv = (3.68e-18 / t_23) * exp_term * c_t;

    sigv.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dt_reactivity_peak() {
        // At 10 keV, <\sigma v>_DT is roughly ~ 1.1e-22 m^3/s
        let sigv_10 = dt_fusion_reactivity(10.0);
        assert!(sigv_10 > 5.0e-23 && sigv_10 < 2.0e-22, "Reactivity at 10 keV was: {}", sigv_10);

        // At 20 keV, <\sigma v>_DT reaches around ~ 4.2e-22 m^3/s
        let sigv_20 = dt_fusion_reactivity(20.0);
        assert!(sigv_20 > sigv_10, "Reactivity at 20 keV should exceed 10 keV");
        assert!(sigv_20 > 2.0e-22 && sigv_20 < 8.0e-22, "Reactivity at 20 keV was: {}", sigv_20);
    }
}

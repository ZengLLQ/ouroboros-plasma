//! Guiding center drift theory and adiabatic invariant diagnostics.
//!
//! Evaluates guiding center trajectory, magnetic moment \mu, \mathbf{E}\times\mathbf{B} drift,
//! \nabla B drift, and magnetic field curvature drift:
//! \mathbf{v}_D = \mathbf{v}_E + \mathbf{v}_{\nabla B} + \mathbf{v}_c

use serde::{Deserialize, Serialize};
use crate::math::Vector3D;

/// Guiding center state and drift velocities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidingCenter {
    /// Guiding center position \mathbf{X} (m).
    pub position: Vector3D,
    /// Parallel velocity v_\parallel (m/s).
    pub v_parallel: f64,
    /// Perpendicular velocity v_\perp (m/s).
    pub v_perp: f64,
    /// Magnetic moment \mu = m v_\perp^2 / (2 B) (J / T).
    pub magnetic_moment: f64,
    /// Pitch angle parameter \lambda = \mu B_0 / E_{kin}.
    pub pitch_parameter: f64,
}

impl GuidingCenter {
    /// Computes the guiding center from full particle position and velocity in field B.
    pub fn from_particle(
        pos: Vector3D,
        vel: Vector3D,
        b_field: Vector3D,
        mass: f64,
        charge: f64,
    ) -> Self {
        let b_mag = b_field.norm();
        let b_unit = b_field.normalized();

        // Decompose velocity into parallel and perpendicular components
        let v_par = vel.dot(&b_unit);
        let v_par_vec = b_unit * v_par;
        let v_perp_vec = vel - v_par_vec;
        let v_perp = v_perp_vec.norm();

        // Gyroradius vector: \mathbf{r}_L = (m / (q B^2)) (\mathbf{v} \times \mathbf{B})
        let r_gyro = vel.cross(&b_field) * (mass / (charge * b_mag * b_mag));

        // Guiding center position: \mathbf{X} = \mathbf{x} - \mathbf{r}_L
        let gc_pos = pos - r_gyro;

        // First adiabatic invariant: \mu = m v_\perp^2 / (2 B)
        let mu = (mass * v_perp * v_perp) / (2.0 * b_mag);

        let e_kin = 0.5 * mass * vel.norm_squared();
        let pitch_param = if e_kin > 0.0 { mu * b_mag / e_kin } else { 0.0 };

        Self {
            position: gc_pos,
            v_parallel: v_par,
            v_perp,
            magnetic_moment: mu,
            pitch_parameter: pitch_param,
        }
    }

    /// Evaluates the \mathbf{E}\times\mathbf{B} drift velocity:
    /// \mathbf{v}_E = (\mathbf{E} \times \mathbf{B}) / B^2.
    pub fn exb_drift(e_field: Vector3D, b_field: Vector3D) -> Vector3D {
        let b2 = b_field.norm_squared();
        if b2 < 1.0e-12 {
            return Vector3D::zero();
        }
        e_field.cross(&b_field) / b2
    }

    /// Evaluates the gradient-B drift velocity:
    /// \mathbf{v}_{\nabla B} = \frac{m v_\perp^2}{2 q B^3} (\mathbf{B} \times \nabla B).
    pub fn grad_b_drift(
        b_field: Vector3D,
        grad_b_mag: Vector3D,
        v_perp: f64,
        mass: f64,
        charge: f64,
    ) -> Vector3D {
        let b_mag = b_field.norm();
        let b3 = b_mag.powi(3);
        if b3 < 1.0e-12 {
            return Vector3D::zero();
        }
        let coeff = (mass * v_perp * v_perp) / (2.0 * charge * b3);
        b_field.cross(&grad_b_mag) * coeff
    }

    /// Evaluates the curvature drift velocity in a torus:
    /// \mathbf{v}_c = \frac{m v_\parallel^2}{q B^4} (\mathbf{B} \times (\mathbf{B} \cdot \nabla)\mathbf{B})
    /// For tokamak with toroidal field B_\phi \propto 1/R, curvature is \boldsymbol{\kappa} = -\hat{R} / R.
    pub fn curvature_drift_tokamak(
        r_major: f64,
        b_field: Vector3D,
        v_parallel: f64,
        mass: f64,
        charge: f64,
    ) -> Vector3D {
        let b_mag = b_field.norm();
        if b_mag < 1.0e-6 || r_major < 0.1 {
            return Vector3D::zero();
        }
        // In tokamak geometry, combined curvature and grad-B drift is directed vertically:
        // v_D \approx \frac{m}{q B_0 R_0} (v_\parallel^2 + v_\perp^2 / 2) \hat{z}
        let v_drift_z = (mass / (charge * b_mag * r_major)) * (v_parallel * v_parallel);
        Vector3D::new(0.0, 0.0, v_drift_z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{ALPHA_CHARGE, ALPHA_MASS};

    #[test]
    fn test_guiding_center_decomposition() {
        let pos = Vector3D::new(6.0, 0.0, 0.0);
        let vel = Vector3D::new(1.0e6, 2.0e6, 0.0);
        let b_field = Vector3D::new(0.0, 0.0, 5.0);

        let gc = GuidingCenter::from_particle(pos, vel, b_field, ALPHA_MASS, ALPHA_CHARGE);
        assert_eq!(gc.v_parallel, 0.0);
        assert!((gc.v_perp - vel.norm()).abs() < 1e-6);
        assert!(gc.magnetic_moment > 0.0);

        // Guiding center x offset should correspond to gyroradius
        assert!((gc.position.y - 0.0).abs() > 0.0);
    }

    #[test]
    fn test_exb_drift_direction() {
        let e = Vector3D::new(1.0e3, 0.0, 0.0); // radial electric field
        let b = Vector3D::new(0.0, 0.0, 5.0);   // vertical B
        let v_e = GuidingCenter::exb_drift(e, b);
        // ExB should be along -y
        assert_eq!(v_e.x, 0.0);
        assert!(v_e.y < 0.0);
        assert_eq!(v_e.z, 0.0);
    }
}

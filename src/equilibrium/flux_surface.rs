//! Poloidal flux surface geometry, contour mapping, and Miller parameterization.

use std::f64::consts::PI;
use serde::{Deserialize, Serialize};

/// Miller parameterization of an elongated, shaped tokamak flux surface.
///
/// Flux surface boundary coordinates are given by:
/// R(\theta) = R_0 + \Delta + r \cos(\theta + \arcsin(\delta) \sin\theta)
/// Z(\theta) = Z_0 + \kappa r \sin\theta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MillerGeometry {
    /// Major radius of the reference geometric axis (m).
    pub r0: f64,
    /// Vertical position of the reference geometric axis (m).
    pub z0: f64,
    /// Minor radius of the plasma boundary / surface (m).
    pub minor_radius: f64,
    /// Elongation parameter \kappa (ratio of vertical to horizontal semi-axis).
    pub elongation: f64,
    /// Triangularity parameter \delta.
    pub triangularity: f64,
    /// Shafranov horizontal shift \Delta (m).
    pub shafranov_shift: f64,
    /// Squareness parameter \zeta (defaults to 0.0).
    pub squareness: f64,
}

impl MillerGeometry {
    /// Creates a new Miller geometry configuration.
    pub fn new(
        r0: f64,
        z0: f64,
        minor_radius: f64,
        elongation: f64,
        triangularity: f64,
        shafranov_shift: f64,
    ) -> Self {
        Self {
            r0,
            z0,
            minor_radius,
            elongation,
            triangularity,
            shafranov_shift,
            squareness: 0.0,
        }
    }

    /// Evaluates the (R, Z) coordinates along the flux surface at poloidal angle \theta \in [0, 2\pi).
    pub fn position_at_angle(&self, theta: f64) -> (f64, f64) {
        let delta_eff = self.triangularity.clamp(-0.99, 0.99);
        let asin_delta = delta_eff.asin();
        let r = self.r0 + self.shafranov_shift + self.minor_radius * (theta + asin_delta * theta.sin()).cos();
        let z = self.z0 + self.elongation * self.minor_radius * theta.sin();
        (r, z)
    }

    /// Evaluates the tangents (dR/d\theta, dZ/d\theta).
    pub fn tangent_at_angle(&self, theta: f64) -> (f64, f64) {
        let delta_eff = self.triangularity.clamp(-0.99, 0.99);
        let asin_delta = delta_eff.asin();
        let sin_part = theta + asin_delta * theta.sin();
        let d_arg = 1.0 + asin_delta * theta.cos();
        let dr_dtheta = -self.minor_radius * sin_part.sin() * d_arg;
        let dz_dtheta = self.elongation * self.minor_radius * theta.cos();
        (dr_dtheta, dz_dtheta)
    }

    /// Computes the poloidal perimeter L_p = \oint d\ell_p.
    pub fn poloidal_perimeter(&self, num_points: usize) -> f64 {
        let n = num_points.max(64);
        let dtheta = 2.0 * PI / (n as f64);
        let mut sum = 0.0;
        for i in 0..n {
            let theta = (i as f64 + 0.5) * dtheta;
            let (dr, dz) = self.tangent_at_angle(theta);
            sum += (dr * dr + dz * dz).sqrt() * dtheta;
        }
        sum
    }

    /// Computes the poloidal cross-sectional area S_p = \int dR dZ.
    pub fn poloidal_area(&self, num_points: usize) -> f64 {
        let n = num_points.max(64);
        let dtheta = 2.0 * PI / (n as f64);
        let mut area = 0.0;
        for i in 0..n {
            let theta = i as f64 * dtheta;
            let (r, z) = self.position_at_angle(theta);
            let (dr, dz) = self.tangent_at_angle(theta);
            // Green's theorem: 1/2 \oint (R dZ - Z dR)
            area += 0.5 * (r * dz - z * dr) * dtheta;
        }
        area.abs()
    }

    /// Computes the enclosed plasma volume V = 2\pi \int R dR dZ.
    pub fn plasma_volume(&self, num_points: usize) -> f64 {
        let n = num_points.max(64);
        let dtheta = 2.0 * PI / (n as f64);
        let mut vol = 0.0;
        for i in 0..n {
            let theta = i as f64 * dtheta;
            let (r, _z) = self.position_at_angle(theta);
            let (_dr, dz) = self.tangent_at_angle(theta);
            // Divergence theorem: \int R dR dZ = \oint (R^2 / 2) dZ
            vol += PI * (r * r) * dz * dtheta;
        }
        vol.abs()
    }
}

/// A discretized 2D poloidal flux surface contour at normalized flux \bar{\psi}.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxSurface {
    /// Normalized poloidal flux \bar{\psi} \in [0, 1].
    pub psi_norm: f64,
    /// Absolute poloidal flux value (Wb / rad).
    pub psi_value: f64,
    /// Discretized boundary coordinates (R, Z) around the full poloidal loop.
    pub contour_points: Vec<(f64, f64)>,
    /// Poloidal magnetic field magnitude |B_p| at each contour point.
    pub bp_values: Vec<f64>,
    /// Major radius of geometric center of this surface (m).
    pub r_center: f64,
    /// Vertical position of geometric center of this surface (m).
    pub z_center: f64,
    /// Effective minor radius of this surface (m).
    pub r_minor: f64,
    /// Poloidal perimeter L_p (m).
    pub perimeter: f64,
    /// Cross-sectional area S_p (m^2).
    pub area: f64,
    /// Enclosed volume V (m^3).
    pub volume: f64,
    /// Safety factor q(\psi) on this surface.
    pub safety_factor: f64,
}

impl FluxSurface {
    /// Generates a flux surface from a Miller geometry and safety factor.
    pub fn from_miller(
        psi_norm: f64,
        psi_value: f64,
        geom: &MillerGeometry,
        safety_factor: f64,
        num_points: usize,
    ) -> Self {
        let n = num_points.max(64);
        let dtheta = 2.0 * PI / (n as f64);
        let mut points = Vec::with_capacity(n);
        for i in 0..n {
            let theta = i as f64 * dtheta;
            points.push(geom.position_at_angle(theta));
        }

        let perimeter = geom.poloidal_perimeter(n);
        let area = geom.poloidal_area(n);
        let volume = geom.plasma_volume(n);

        Self {
            psi_norm,
            psi_value,
            contour_points: points,
            bp_values: vec![0.0; n],
            r_center: geom.r0 + geom.shafranov_shift,
            z_center: geom.z0,
            r_minor: geom.minor_radius,
            perimeter,
            area,
            volume,
            safety_factor,
        }
    }

    /// Checks if a cylindrical point (R, Z) is enclosed within this flux surface.
    pub fn contains_point(&self, r: f64, z: f64) -> bool {
        // Ray casting algorithm for polygon point-in-polygon
        let n = self.contour_points.len();
        if n < 3 {
            return false;
        }
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let (ri, zi) = self.contour_points[i];
            let (rj, zj) = self.contour_points[j];
            if ((zi > z) != (zj > z)) && (r < (rj - ri) * (z - zi) / (zj - zi) + ri) {
                inside = !inside;
            }
            j = i;
        }
        inside
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_miller_circular_geometry() {
        // Circular cross-section: kappa = 1, delta = 0, delta_shift = 0
        let r0 = 6.2;
        let z0 = 0.0;
        let a = 2.0;
        let geom = MillerGeometry::new(r0, z0, a, 1.0, 0.0, 0.0);

        let perimeter = geom.poloidal_perimeter(200);
        let expected_perimeter = 2.0 * PI * a;
        assert!((perimeter - expected_perimeter).abs() / expected_perimeter < 0.001);

        let area = geom.poloidal_area(200);
        let expected_area = PI * a * a;
        assert!((area - expected_area).abs() / expected_area < 0.001);

        let volume = geom.plasma_volume(200);
        let expected_vol = 2.0 * PI * PI * r0 * a * a;
        assert!((volume - expected_vol).abs() / expected_vol < 0.005);
    }

    #[test]
    fn test_flux_surface_containment() {
        let geom = MillerGeometry::new(3.0, 0.0, 1.0, 1.5, 0.3, 0.05);
        let surf = FluxSurface::from_miller(1.0, 1.0, &geom, 3.0, 128);

        // Center must be contained
        assert!(surf.contains_point(3.05, 0.0));
        // Far away point must not be contained
        assert!(!surf.contains_point(10.0, 5.0));
        assert!(!surf.contains_point(1.0, 0.0));
    }
}

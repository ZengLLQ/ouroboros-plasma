//! Vector mathematics, coordinate transformations, and numerical algorithms.

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use serde::{Deserialize, Serialize};

/// A 3D spatial vector with double-precision floating-point components.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vector3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector3D {
    /// Creates a new 3D vector.
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Zero vector (0, 0, 0).
    pub const fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }

    /// Euclidean norm (magnitude) of the vector: \sqrt{x^2 + y^2 + z^2}.
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Squared Euclidean norm: x^2 + y^2 + z^2.
    pub fn norm_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Dot product between self and another vector.
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product \mathbf{a} \times \mathbf{b}.
    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Returns a unit vector in the same direction, or zero vector if norm is near zero.
    pub fn normalized(&self) -> Self {
        let n = self.norm();
        if n < 1.0e-15 {
            Self::zero()
        } else {
            *self / n
        }
    }

    /// Converts cylindrical coordinates (R, \phi, Z) to Cartesian vector (x, y, z).
    pub fn from_cylindrical(r: f64, phi: f64, z: f64) -> Self {
        Self {
            x: r * phi.cos(),
            y: r * phi.sin(),
            z,
        }
    }

    /// Converts Cartesian position (x, y, z) into cylindrical coordinates (R, \phi, Z).
    pub fn to_cylindrical(&self) -> (f64, f64, f64) {
        let r = (self.x * self.x + self.y * self.y).sqrt();
        let phi = self.y.atan2(self.x);
        (r, phi, self.z)
    }

    /// Converts cylindrical vector components (v_R, v_\phi, v_Z) at toroidal angle \phi
    /// into Cartesian vector components (v_x, v_y, v_z).
    pub fn cylindrical_to_cartesian_vector(v_r: f64, v_phi: f64, v_z: f64, phi: f64) -> Self {
        let cos_p = phi.cos();
        let sin_p = phi.sin();
        Self {
            x: v_r * cos_p - v_phi * sin_p,
            y: v_r * sin_p + v_phi * cos_p,
            z: v_z,
        }
    }

    /// Converts Cartesian vector components (v_x, v_y, v_z) at toroidal angle \phi
    /// into cylindrical vector components (v_R, v_\phi, v_Z).
    pub fn cartesian_to_cylindrical_vector(&self, phi: f64) -> (f64, f64, f64) {
        let cos_p = phi.cos();
        let sin_p = phi.sin();
        let v_r = self.x * cos_p + self.y * sin_p;
        let v_phi = -self.x * sin_p + self.y * cos_p;
        (v_r, v_phi, self.z)
    }
}

impl Add for Vector3D {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl AddAssign for Vector3D {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Sub for Vector3D {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl SubAssign for Vector3D {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl Mul<f64> for Vector3D {
    type Output = Self;
    fn mul(self, scalar: f64) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

impl Mul<Vector3D> for f64 {
    type Output = Vector3D;
    fn mul(self, v: Vector3D) -> Vector3D {
        v * self
    }
}

impl MulAssign<f64> for Vector3D {
    fn mul_assign(&mut self, scalar: f64) {
        self.x *= scalar;
        self.y *= scalar;
        self.z *= scalar;
    }
}

impl Div<f64> for Vector3D {
    type Output = Self;
    fn div(self, scalar: f64) -> Self {
        Self {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
        }
    }
}

impl DivAssign<f64> for Vector3D {
    fn div_assign(&mut self, scalar: f64) {
        self.x /= scalar;
        self.y /= scalar;
        self.z /= scalar;
    }
}

impl Neg for Vector3D {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// Bilinear interpolation of a 2D scalar field defined on a regular rectangular grid.
pub fn interpolate_2d_bilinear(
    r: f64,
    z: f64,
    r_coords: &[f64],
    z_coords: &[f64],
    values: &[Vec<f64>], // [nr][nz]
) -> f64 {
    let nr = r_coords.len();
    let nz = z_coords.len();
    if nr < 2 || nz < 2 {
        return 0.0;
    }

    let r_min = r_coords[0];
    let r_max = r_coords[nr - 1];
    let z_min = z_coords[0];
    let z_max = z_coords[nz - 1];

    if r <= r_min {
        return interpolate_1d_clamped(z, z_coords, &values[0]);
    }
    if r >= r_max {
        return interpolate_1d_clamped(z, z_coords, &values[nr - 1]);
    }

    // Binary search or direct indexing if uniform
    let dr = (r_max - r_min) / ((nr - 1) as f64);
    let dz = (z_max - z_min) / ((nz - 1) as f64);

    let i = ((r - r_min) / dr).floor() as usize;
    let j = ((z - z_min) / dz).floor() as usize;

    let i = i.min(nr - 2);
    let j = j.min(nz - 2);

    let r1 = r_coords[i];
    let r2 = r_coords[i + 1];
    let z1 = z_coords[j];
    let z2 = z_coords[j + 1];

    let t = (r - r1) / (r2 - r1);
    let u = (z - z1) / (z2 - z1);

    let v00 = values[i][j];
    let v10 = values[i + 1][j];
    let v01 = values[i][j + 1];
    let v11 = values[i + 1][j + 1];

    (1.0 - t) * (1.0 - u) * v00 + t * (1.0 - u) * v10 + (1.0 - t) * u * v01 + t * u * v11
}

/// 1D interpolation helper clamped to boundary.
fn interpolate_1d_clamped(z: f64, z_coords: &[f64], vals: &[f64]) -> f64 {
    let nz = z_coords.len();
    if z <= z_coords[0] {
        return vals[0];
    }
    if z >= z_coords[nz - 1] {
        return vals[nz - 1];
    }
    let dz = (z_coords[nz - 1] - z_coords[0]) / ((nz - 1) as f64);
    let j = (((z - z_coords[0]) / dz).floor() as usize).min(nz - 2);
    let u = (z - z_coords[j]) / (z_coords[j + 1] - z_coords[j]);
    (1.0 - u) * vals[j] + u * vals[j + 1]
}

/// Solves a tridiagonal linear system:
/// a[i] * x[i-1] + b[i] * x[i] + c[i] * x[i+1] = d[i]
/// for i in 0..n using the Thomas algorithm (O(N) complexity).
pub fn solve_tridiagonal(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = b.len();
    assert_eq!(d.len(), n, "Right-hand side vector must match matrix dimension");
    assert_eq!(a.len(), n, "Subdiagonal array must have length n (with a[0] unused)");
    assert_eq!(c.len(), n, "Superdiagonal array must have length n (with c[n-1] unused)");

    let mut cp = vec![0.0; n];
    let mut dp = vec![0.0; n];
    let mut x = vec![0.0; n];

    if b[0].abs() < 1.0e-30 {
        panic!("Singular matrix encountered in Thomas algorithm at index 0");
    }

    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];

    for i in 1..n {
        let m = b[i] - a[i] * cp[i - 1];
        if m.abs() < 1.0e-30 {
            // Regularize slightly to prevent hard zero division
            let m_safe = if m >= 0.0 { 1.0e-30 } else { -1.0e-30 };
            cp[i] = c[i] / m_safe;
            dp[i] = (d[i] - a[i] * dp[i - 1]) / m_safe;
        } else {
            cp[i] = c[i] / m;
            dp[i] = (d[i] - a[i] * dp[i - 1]) / m;
        }
    }

    x[n - 1] = dp[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }

    x
}

/// Numerical integration using Simpson's composite rule on evenly spaced points.
pub fn simpson_integrate(values: &[f64], dx: f64) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    if n == 2 {
        return 0.5 * dx * (values[0] + values[1]);
    }

    let mut sum = values[0] + values[n - 1];
    let is_odd = (n - 1) % 2 == 1;
    let limit = if is_odd { n - 2 } else { n - 1 };

    for i in 1..limit {
        if i % 2 == 1 {
            sum += 4.0 * values[i];
        } else {
            sum += 2.0 * values[i];
        }
    }
    let mut result = sum * dx / 3.0;

    // Handle odd number of intervals with trapezoidal end-point rule
    if is_odd {
        result += 0.5 * dx * (values[n - 2] + values[n - 1]);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector3d_operations() {
        let v1 = Vector3D::new(1.0, 2.0, 3.0);
        let v2 = Vector3D::new(4.0, -1.0, 0.5);

        let v_add = v1 + v2;
        assert_eq!(v_add, Vector3D::new(5.0, 1.0, 3.5));

        let dot = v1.dot(&v2);
        assert_eq!(dot, 1.0 * 4.0 + 2.0 * (-1.0) + 3.0 * 0.5);

        let cross = v1.cross(&v2);
        assert_eq!(cross, Vector3D::new(2.0 * 0.5 - 3.0 * (-1.0), 3.0 * 4.0 - 1.0 * 0.5, 1.0 * (-1.0) - 2.0 * 4.0));
        assert_eq!(cross.dot(&v1), 0.0);
        assert_eq!(cross.dot(&v2), 0.0);
    }

    #[test]
    fn test_tridiagonal_solver() {
        // Solve system:
        //  2 x0 - x1       = 1
        // -x0 + 2 x1 - x2  = 2
        //      -x1 + 2 x2  = 3
        // Solution: x0 = 2, x1 = 3, x2 = 3.
        let a = vec![0.0, -1.0, -1.0];
        let b = vec![2.0, 2.0, 2.0];
        let c = vec![-1.0, -1.0, 0.0];
        let d = vec![1.0, 2.0, 3.0];

        let sol = solve_tridiagonal(&a, &b, &c, &d);
        assert!((sol[0] - 2.5).abs() < 1e-12);
        assert!((sol[1] - 4.0).abs() < 1e-12);
        assert!((sol[2] - 3.5).abs() < 1e-12);
    }

    #[test]
    fn test_simpson_integrate() {
        // Integrate f(x) = x^2 from 0 to 1 -> 1/3
        let n = 101;
        let dx = 1.0 / ((n - 1) as f64);
        let vals: Vec<f64> = (0..n).map(|i| {
            let x = i as f64 * dx;
            x * x
        }).collect();

        let integral = simpson_integrate(&vals, dx);
        assert!((integral - 1.0 / 3.0).abs() < 1.0e-5);
    }
}

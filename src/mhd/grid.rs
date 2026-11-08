//! 2D/3D computational grid representation and boundary conditions for MHD simulation.

use serde::{Deserialize, Serialize};

/// Type of boundary condition applied at grid limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryCondition {
    /// Perfectly conducting rigid wall (v_n = 0, B_n = 0, \partial p / \partial n = 0).
    ConductingWall,
    /// Fixed Dirichlet values.
    Fixed,
    /// Zero-gradient Neumann boundary condition (\partial \Phi / \partial n = 0).
    Neumann,
    /// Periodic boundary conditions.
    Periodic,
}

/// 2D rectangular computational grid in cylindrical coordinates (R, Z).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MhdGrid {
    pub nr: usize,
    pub nz: usize,
    pub r_min: f64,
    pub r_max: f64,
    pub z_min: f64,
    pub z_max: f64,
    pub dr: f64,
    pub dz: f64,
    pub r_coords: Vec<f64>,
    pub z_coords: Vec<f64>,
    pub bc_r_inner: BoundaryCondition,
    pub bc_r_outer: BoundaryCondition,
    pub bc_z_bottom: BoundaryCondition,
    pub bc_z_top: BoundaryCondition,
}

impl MhdGrid {
    /// Creates a new MHD computational grid.
    pub fn new(
        nr: usize,
        nz: usize,
        r_min: f64,
        r_max: f64,
        z_min: f64,
        z_max: f64,
    ) -> Self {
        assert!(nr >= 5 && nz >= 5, "Grid points must be >= 5 in each dimension");
        assert!(r_max > r_min && z_max > z_min, "Spatial coordinates must be strictly increasing");

        let dr = (r_max - r_min) / ((nr - 1) as f64);
        let dz = (z_max - z_min) / ((nz - 1) as f64);

        let r_coords: Vec<f64> = (0..nr).map(|i| r_min + i as f64 * dr).collect();
        let z_coords: Vec<f64> = (0..nz).map(|j| z_min + j as f64 * dz).collect();

        Self {
            nr,
            nz,
            r_min,
            r_max,
            z_min,
            z_max,
            dr,
            dz,
            r_coords,
            z_coords,
            bc_r_inner: BoundaryCondition::ConductingWall,
            bc_r_outer: BoundaryCondition::ConductingWall,
            bc_z_bottom: BoundaryCondition::ConductingWall,
            bc_z_top: BoundaryCondition::ConductingWall,
        }
    }

    /// Returns the cell volume at radial index i: V_cell = 2\pi R_i \Delta R \Delta Z.
    pub fn cell_volume(&self, i: usize) -> f64 {
        std::f64::consts::PI * 2.0 * self.r_coords[i] * self.dr * self.dz
    }

    /// Returns whether the index (i, j) lies on the computational boundary.
    pub fn is_boundary(&self, i: usize, j: usize) -> bool {
        i == 0 || i == self.nr - 1 || j == 0 || j == self.nz - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_initialization() {
        let grid = MhdGrid::new(21, 31, 4.0, 8.0, -2.0, 2.0);
        assert_eq!(grid.nr, 21);
        assert_eq!(grid.nz, 31);
        assert!((grid.dr - 0.2).abs() < 1e-12);
        assert!((grid.dz - (4.0 / 30.0)).abs() < 1e-12);
        assert!(grid.is_boundary(0, 5));
        assert!(grid.is_boundary(10, 0));
        assert!(!grid.is_boundary(10, 15));
    }
}

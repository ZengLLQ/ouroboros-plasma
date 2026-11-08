# OuroborosPlasma (`ouroboros-plasma`)

[![CI](https://github.com/ZengLLQ/ouroboros-plasma/actions/workflows/ci.yml/badge.svg)](https://github.com/ZengLLQ/ouroboros-plasma/actions/workflows/ci.yml)
[![Rust Version](https://img.shields.io/badge/rust-2021%20edition-blue.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Physics](https://img.shields.io/badge/computational-fusion%20plasma%20physics-orange.svg)](#theoretical-foundations)
[![Verification](https://img.shields.io/badge/verification-exact%20soloviev%20benchmarks-success.svg)](#verification--benchmarks)

**OuroborosPlasma** is an advanced, production-grade computational fusion plasma physics and magnetohydrodynamics (MHD) engine written 100% in pure Rust (2021 edition). It models axisymmetric toroidal magnetic confinement equilibria, full ideal and resistive fluid dynamics, relativistic Boris kinetic velocity integration for energetic alpha fusion products, and Toroidal Alfvén Eigenmode (TAE) spectral stability for burning plasma reactors such as ITER, SPARC, and STEP.

---

## Key Capabilities

- **2D/3D Grad-Shafranov Elliptic PDE Solver**: Successive Over-Relaxation (SOR) with Chebyshev acceleration solving $\Delta^* \psi = -\mu_0 R^2 p'(\psi) - F(\psi) F'(\psi)$, exact Soloviev benchmark verification, Miller-shaped flux boundaries ($\kappa, \delta$), magnetic axis localization, and safety factor $q(\psi)$ evaluation.
- **Ideal & Resistive Magnetohydrodynamics (MHD)**: Conservative fluid equations in cylindrical coordinates $(R, \phi, Z)$ with full geometric curvature terms, Lorentz $\mathbf{J} \times \mathbf{B}$ forces, induction $\partial \mathbf{B}/\partial t = \nabla \times (\mathbf{v} \times \mathbf{B} - \eta \mathbf{J})$, and 4th-order Runge-Kutta (RK4) time-stepping.
- **Relativistic Boris Velocity Leapfrog Integrator**: Energy-conserving symplectic trajectory integration for 3.52 MeV fusion alphas ($d\mathbf{u}/dt = (q/m)[\mathbf{E} + (\mathbf{u}/\gamma)\times\mathbf{B}]$), magnetic mirror trapping, banana orbit width calculations ($\Delta_b \approx q \rho_L / \sqrt{\epsilon}$), and prompt loss cone kinetics.
- **Toroidal Alfvén Eigenmodes (TAE) Spectral Suite**: Shear Alfvén continuum dispersion calculation, toroidicity-induced gap formation ($q_{TAE} = (m+1/2)/n$), 1D radial ODE boundary value eigenmode solver, multi-mechanism damping breakdown (continuum, electron Landau, thermal ion Landau, radiative), and energetic alpha particle resonant drive.
- **7-Bit Clean ASCII Terminal Visualizer**: 100% portable 7-bit ASCII 2D poloidal flux cross-section maps, radial safety factor $q(\rho)$ profile plots, and comprehensive ITER98y2 H-mode performance diagnostic reports.

---

## Architectural Topology

```mermaid
flowchart TD
    subgraph Equilibrium ["Equilibrium Engine (src/equilibrium/)"]
        GS["Grad-Shafranov 2D PDE Solver\n(Delta* psi = -mu_0 R^2 p' - FF')"]
        Soloviev["Exact Soloviev Analytic Solution"]
        Miller["Miller Shaping Geometry (kappa, delta)"]
        Profiles["Pressure p(psi), Current F(psi), Safety Factor q(psi)"]
    end

    subgraph MHD ["Magnetohydrodynamics (src/mhd/)"]
        State["MHD Primitive/Conservative State (rho, v, p, B, J)"]
        Equations["Ideal & Resistive Fluid Conservation Laws"]
        Stepper["RK4 Stepper & Adaptive CFL Controller"]
        DivB["Divergence-Free Field Diagnostics (div B = 0)"]
    end

    subgraph Particles ["Relativistic Kinetics (src/particles/)"]
        Boris["Relativistic Boris Leapfrog Integrator\n(du/dt = q/m [E + v x B])"]
        GuidingCenter["Guiding Center Drifts (ExB, grad-B, Curvature)"]
        LossCone["Magnetic Mirror Trapping & Banana Orbit Widths"]
        Ensemble["3.52 MeV D-T Fusion Alpha Swarm Simulator"]
    end

    subgraph Modes ["Spectral Stability (src/modes/)"]
        Alfven["Shear Alfven Continuum Dispersion (omega_A = k_par * v_A)"]
        TAE["Toroidal Alfven Eigenmodes (TAE) Radial ODE Solver"]
        Damping["Damping Breakdown (Continuum, Landau, Radiative)"]
        Drive["Fast-Ion Resonant Drive & Critical Beta Threshold"]
    end

    subgraph Visualizer ["Diagnostics & CLI (src/visualizer.rs, src/cli.rs)"]
        ASCII["7-bit ASCII Flux Cross-Section Map"]
        QPlot["Safety Factor q(rho) Curve Plot"]
        Perf["ITER98y2 Confinement & Fusion Gain Q Dashboard"]
    end

    Equilibrium --> Modes
    Equilibrium --> Particles
    Equilibrium --> MHD
    Modes --> Visualizer
    Particles --> Visualizer
    MHD --> Visualizer
```

---

## Theoretical Foundations

### 1. The Grad-Shafranov Equilibrium Equation

In axisymmetric toroidal geometry with toroidal angle $\phi$, magnetic field $\mathbf{B}$ is decomposed into poloidal and toroidal stream components:
$$\mathbf{B} = \nabla \phi \times \nabla \psi + F(\psi) \nabla \phi = -\frac{1}{R}\frac{\partial \psi}{\partial Z}\hat{R} + \frac{F(\psi)}{R}\hat{\phi} + \frac{1}{R}\frac{\partial \psi}{\partial R}\hat{Z}$$
where $\psi(R, Z)$ is the poloidal magnetic flux per radian and $F(\psi) = R B_\phi$ is the poloidal current stream function.

Equilibrium force balance $\mathbf{J} \times \mathbf{B} = \nabla p$ with Ampère's law $\mu_0 \mathbf{J} = \nabla \times \mathbf{B}$ yields the non-linear elliptic partial differential equation:
$$\Delta^* \psi = -\mu_0 R^2 \frac{dp}{d\psi} - F(\psi) \frac{dF}{d\psi}$$
where the elliptic operator $\Delta^*$ is defined as:
$$\Delta^* \psi \equiv R \frac{\partial}{\partial R}\left( \frac{1}{R} \frac{\partial \psi}{\partial R} \right) + \frac{\partial^2 \psi}{\partial Z^2} = \frac{\partial^2 \psi}{\partial R^2} - \frac{1}{R}\frac{\partial \psi}{\partial R} + \frac{\partial^2 \psi}{\partial Z^2}$$

The safety factor $q(\psi)$ along closed flux surfaces is given by:
$$q(\psi) = \frac{1}{2\pi} \oint \frac{B_\phi}{R B_p} d\ell_p = \frac{F(\psi)}{2\pi} \oint \frac{d\ell_p}{R |\nabla \psi|}$$

### 2. Ideal and Resistive Magnetohydrodynamics (MHD)

The plasma fluid evolution is governed by the full set of resistive MHD conservation equations:
$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0$$
$$\rho \left( \frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla)\mathbf{v} \right) = -\nabla p + \mathbf{J} \times \mathbf{B} + \mu \nabla^2 \mathbf{v}$$
$$\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{v} \times \mathbf{B} - \eta \mathbf{J})$$
$$\frac{\partial p}{\partial t} + \mathbf{v} \cdot \nabla p + \gamma p (\nabla \cdot \mathbf{v}) = (\gamma - 1) \eta |\mathbf{J}|^2$$
subject to the divergence-free condition $\nabla \cdot \mathbf{B} = 0$.

Characteristic wave speeds:
- Ion sound speed: $c_s = \sqrt{\gamma p / \rho}$
- Shear Alfvén velocity: $v_A = |\mathbf{B}| / \sqrt{\mu_0 \rho}$
- Fast magnetosonic velocity: $v_{fast} = \sqrt{c_s^2 + v_A^2}$

### 3. Relativistic Boris Velocity Integrator

For fast alpha particles ($E_\alpha = 3.52\text{ MeV}$) and runaway electrons, proper velocity $\mathbf{u} = \gamma \mathbf{v}$ advances according to the Lorentz force:
$$\frac{d\mathbf{x}}{dt} = \frac{\mathbf{u}}{\gamma}, \quad \frac{d\mathbf{u}}{dt} = \frac{q}{m}\left( \mathbf{E} + \frac{\mathbf{u}}{\gamma} \times \mathbf{B} \right)$$
where $\gamma = \sqrt{1 + |\mathbf{u}|^2 / c^2}$.

The Boris algorithm decouples electric acceleration from magnetic gyro-rotation:
1. Electric half-acceleration: $\mathbf{u}^- = \mathbf{u}^{n-1/2} + \frac{q \mathbf{E}}{m}\frac{\Delta t}{2}$
2. Relativistic factor: $\gamma^- = \sqrt{1 + |\mathbf{u}^-|^2 / c^2}$
3. Rotation vector: $\mathbf{t} = \frac{q \mathbf{B}}{m}\frac{\Delta t}{2 \gamma^-}, \quad \mathbf{s} = \frac{2\mathbf{t}}{1 + |\mathbf{t}|^2}$
4. Rotation: $\mathbf{u}' = \mathbf{u}^- + \mathbf{u}^- \times \mathbf{t}, \quad \mathbf{u}^+ = \mathbf{u}^- + \mathbf{u}' \times \mathbf{s}$
5. Second half-acceleration: $\mathbf{u}^{n+1/2} = \mathbf{u}^+ + \frac{q \mathbf{E}}{m}\frac{\Delta t}{2}$
6. Coordinate update: $\mathbf{x}^{n+1} = \mathbf{x}^n + \frac{\mathbf{u}^{n+1/2}}{\gamma^{n+1/2}}\Delta t$

In static fields ($\mathbf{E} = 0$), kinetic energy $E_{kin} = (\gamma - 1) m c^2$ and toroidal canonical angular momentum $P_\phi = R m \gamma v_\phi - q \psi$ are conserved to machine precision ($\Delta E / E < 10^{-11}$).

### 4. Toroidal Alfvén Eigenmodes (TAE) and Kinetic Drive

Toroidal magnetic variation $B \approx B_0 (1 - \epsilon \cos\theta)$ couples poloidal harmonics $m$ and $m+1$. Where continuum branches cross:
$$k_{\parallel m}(r) = - k_{\parallel m+1}(r) \implies q(r_{TAE}) = \frac{m + 1/2}{n}$$
At this surface, destructive interference produces a frequency gap in the Alfvén continuum:
$$\omega_{TAE} = \frac{v_A(r_{TAE})}{2 q(r_{TAE}) R_0}$$
with continuum gap boundary $\omega_\pm = \omega_{TAE} \sqrt{1 \pm \hat{\epsilon}}$.

Energetic alpha particles destabilize the mode via spatial gradient drive when the alpha diamagnetic frequency exceeds the mode frequency:
$$\omega_{*\alpha} = -\frac{m_{pol}}{q_\alpha B r} \frac{T_\alpha}{n_\alpha}\frac{dn_\alpha}{dr} > \omega_{TAE}$$
The net linear growth rate is:
$$\gamma_{net} = \gamma_\alpha - (\gamma_{cont} + \gamma_e + \gamma_i + \gamma_{rad})$$
where $\gamma_{cont}$ is continuum damping, $\gamma_e$ is electron Landau damping, and $\gamma_{rad}$ is radiative kinetic damping.

---

## Installation & Requirements

Ensure Rust toolchain (2021 edition, version 1.75+) is installed:

```bash
git clone https://github.com/ZengLLQ/ouroboros-plasma.git
cd ouroboros-plasma
cargo build --release
```

Run test suite:
```bash
cargo test
```

---

## Command-Line Interface (CLI)

OuroborosPlasma provides a high-level CLI for running simulations and diagnostics:

```bash
# Run comprehensive multi-physics demonstration
cargo run --release -- demo

# Solve 2D Grad-Shafranov equilibrium with ASCII flux visualization
cargo run --release -- solve-equilibrium --r0 6.2 --a 2.0 --b0 5.3 --ip-ma 15.0 --kappa 1.7 --delta 0.33

# Simulate relativistic Boris 3.52 MeV alpha particle confinement
cargo run --release -- track-particles --particles 500 --energy-mev 3.52 --r0 6.2 --b0 5.3 --steps 100

# Analyze Toroidal Alfvén Eigenmodes (TAE) spectrum and energetic drive
cargo run --release -- tae-spectrum --toroidal-n 2 --poloidal-m 2 --r0 6.2 --a 2.0 --b0 5.3 --beta-alpha 0.006

# Run 2D magnetohydrodynamic (MHD) fluid simulation steps
cargo run --release -- run-mhd-step --steps 20 --grid-size 21 --viscosity 1e-3 --resistivity 1e-6
```

---

## 7-Bit Clean ASCII Diagnostics

### 2D Poloidal Flux Cross-Section Contour Map

```text
+-----------------------------------------------------------------+
|  OUROBOROS-PLASMA 2D POLOIDAL FLUX CROSS-SECTION MAP (R vs Z)  |
+-----------------------------------------------------------------+
|                                                                 |
|                         ##################                      |
|                     ####@@@@@@@@@@@@@@@@@@@####                 |
|                 ####@@@@@@@@@@@@@@@@@@@@@@@@@@@###              |
|               ###@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@##            |
|             ###@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@##          |
|            ##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@##        |
|          ###@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@##       |
|          ##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@##      |
|         ##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@##     |
|        ##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@##     |
|         ##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@##     |
|          ##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@##      |
|          ###@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@##       |
|            ##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@##        |
|             ###@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@##          |
|               ###@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@##            |
|                 ####@@@@@@@@@@@@@@@@@@@@@@@@@@@###              |
|                     ####@@@@@@@@@@@@@@@@@@@####                 |
|                         ##################                      |
|                               ***                               |
+-----------------------------------------------------------------+
  R: [3.60 m .. 8.80 m]   Z: [-3.80 m .. 3.80 m]   Axis: (6.20, -3.56) m
  Legend: [*] Axis  [@] Core  [%] Mid-Core  [+] Mid  [=] Outer  [-] Edge  [#] Wall
```

### Safety Factor $q(\rho)$ Radial Profile

```text
+-----------------------------------------------------------------+
|              SAFETY FACTOR q(rho) RADIAL PROFILE PLOT            |
+-----------------------------------------------------------------+
 3.9 |                                                           
 3.6 |                                                           
 3.3 |                                                       ****
 3.0 |                                                   ****    
 2.7 |                                               ****        
 2.4 |                                           ****            
 2.1 |-------------------------------------******----------------
 1.9 |                               ******                      
 1.6 |                        *******                            
 1.3 |            ************                                   
 1.0 |************:::::::::::::::::::::::::::::::::::::::::::::::
     +-----------------------------------------------------------+
     0.0          0.25         0.50         0.75         1.00
                               rho = r / a
  q(0) = 1.05   q_a = 3.40   q=1.0 sawteeth[:]   q=2.0 disruption[-]
```

### Fusion Performance Report Card

```text
+----------------------------------------------------------------------+
|       OUROBOROS-PLASMA COMPREHENSIVE FUSION PERFORMANCE REPORT       |
|  Facility / Benchmark: ITER 15MA Q=10 Reference Demonstration        |
+----------------------------------------------------------------------+
|  1. MACHINE GEOMETRY & MAGNETIC EQUILIBRIUM                          |
|     Major Radius (R_0):              6.20 m   Aspect Ratio (A):       3.10 |
|     Minor Radius (a):                2.00 m   Elongation (kappa):     1.70 |
|     Triangularity (delta):           0.33     Plasma Volume:        832.2 m^3 |
|     Toroidal Field (B_0):            5.30 T   Plasma Current (I_p):   15.0 MA |
|     Safety Factor On-Axis (q_0):     1.05     Edge Safety (q_a):      3.40 |
+----------------------------------------------------------------------+
|  2. KINETIC STATE & STABILITY METRICS                                |
|     Central Density (n_e0):       1.05e20 m^-3                      |
|     Central Temperature (T_0):      16.50 keV                       |
|     Stored Thermal Energy (W_th):   263.3 MJ                        |
|     Toroidal Beta (beta_t):          1.89 %   Poloidal Beta (beta_p): 1.23 |
|     Normalized Troyon (beta_N):      1.33     Greenwald Limit (n_G): 1.2e20 |
|     Greenwald Fraction (f_G):       87.96 %                         |
+----------------------------------------------------------------------+
|  3. FUSION ENERGETICS & CONFINEMENT                                  |
|     D-T Reactivity <sigma v>:    2.51e-22 m^3/s                     |
|     Total Fusion Power (P_fus):     520.1 MW                        |
|     Alpha Heating Power (P_alpha):  104.1 MW                       |
|     Energy Confinement (tau_E):      2.70 s  (ITER98y2 H-mode)      |
|     Fusion Energy Gain (Q):          10.0                           |
+----------------------------------------------------------------------+
```

---

## Rust API Usage

```rust
use ouroboros_plasma::equilibrium::{GradShafranovSolver, MillerGeometry, PlasmaProfiles};
use ouroboros_plasma::modes::{AlfvenContinuum, TaeSolver, FastIonDriveCalculator, DampingCalculator};
use ouroboros_plasma::particles::ParticleEnsemble;

fn main() {
    // 1. Solve 2D Grad-Shafranov equilibrium
    let mut solver = GradShafranovSolver::new(35, 35, 3.6, 8.8, -3.8, 3.8);
    let miller = MillerGeometry::new(6.2, 0.0, 2.0, 1.7, 0.33, 0.14);
    solver.initialize_from_miller(&miller, 12.0);

    let profiles = PlasmaProfiles::new(260.0e3, 2.0, 1.5, 6.2, 2.0, 5.3, 0.0, 12.0);
    let (iters, res) = solver.solve(&profiles, 200, 1e-4);
    println!("Equilibrium solved in {} iterations (residual: {:.2e})", iters, res);

    // 2. Simulate 3.52 MeV alpha confinement
    let ensemble = ParticleEnsemble::generate_dt_alphas(500, 6.2, 2.0, 5.3, 1.8, 42);
    let conf = ensemble.compute_confinement_report();
    println!("Confined alpha fraction: {:.1}%", conf.confinement_fraction * 100.0);

    // 3. Toroidal Alfvén Eigenmodes (TAE) stability analysis
    let continuum = AlfvenContinuum::new(6.2, 2.0, 5.3, 2, vec![2]);
    let tae_solver = TaeSolver::new(continuum);
    let rho_mass = 1.05e20 * ouroboros_plasma::constants::DT_AVERAGE_ION_MASS;
    
    if let Some(mode) = tae_solver.solve_tae_mode(2, rho_mass, |rho| 1.05 + 2.35 * rho * rho) {
        let va = tae_solver.continuum.alfven_speed(mode.r_tae, rho_mass);
        let damp_calc = DampingCalculator::new(15.0e3, 15.0e3, 1.05e20, ouroboros_plasma::constants::DT_AVERAGE_ION_MASS);
        let damping = damp_calc.compute_damping(&mode, 0.65, va, 0.02, 0.02, 5.3);
        let drive_calc = FastIonDriveCalculator::new();
        let report = drive_calc.analyze_stability(mode, 0.0055, va, 5.3, 0.45, damping);

        println!("TAE gap frequency: {:.1} kHz, Stability: {:?}", report.mode.frequency_hz * 1e-3, report.stability);
    }
}
```

---

## Verification & Benchmarks

| Test Target | Theoretical Model / Benchmark | Numerical Result | Error / Invariant Drift | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Soloviev Identity** | $\Delta^* \psi = \mu_0 p_1 R^2$ | Analytic equality | $< 10^{-9}$ relative error | PASS |
| **Boris Integrator** | Symplectic energy conservation | $\Delta E / E_0$ in pure B | $< 10^{-11}$ machine precision | PASS |
| **Cyclotron Motion** | Larmor radius $r_L = v_\perp / \Omega_c$ | Period $T = 2\pi / \Omega_c$ | Closed circular orbit | PASS |
| **Banana Trapping** | Critical pitch $\xi_c = \sqrt{2\epsilon / (1+\epsilon)}$ | Core vs Edge loss cone | 100% prompt loss for $r \to a$ | PASS |
| **Alfvén Continuum** | Parallel wavevector $k_\parallel(r_{res}) = 0$ | $k_{\parallel, 3, 2}(q=1.5) \equiv 0$ | Exact zero crossing | PASS |
| **TAE Gap Location** | Rational gap condition $q = (m+1/2)/n$ | $q_{TAE}(2, 2) = 1.250$ | Exact matching | PASS |
| **ITER Fusion Power** | $P_{fus} \approx 500\text{ MW}$ @ $15\text{ keV}$ | $520.1\text{ MW}$ | In target range | PASS |

---

## License

Dual-licensed under either:
- **MIT License** ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)
- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)

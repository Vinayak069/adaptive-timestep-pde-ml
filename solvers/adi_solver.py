"""
solvers/adi_solver.py
─────────────────────────────────────────────────────────────────────────────
Alternating Direction Implicit (ADI) Solver for the 2D Heat / Parabolic PDE:

    ∂u/∂t = α (∂²u/∂x² + ∂²u/∂y²)   on [0,Lx] × [0,Ly], t ∈ [0,T]

Numerical Scheme — Peaceman–Rachford ADI:
  Step 1 (half-step, implicit in x):
    (I - α Δt/2 Aₓ) u* = (I + α Δt/2 Aᵧ) uⁿ

  Step 2 (full-step, implicit in y):
    (I - α Δt/2 Aᵧ) u^{n+1} = (I + α Δt/2 Aₓ) u*

  where Aₓ, Aᵧ are second-order finite-difference operators along x and y.

Properties:
  • Second-order accurate in both space (O(Δx², Δy²)) and time (O(Δt²))
  • Unconditionally stable (no CFL restriction on diffusion number)
  • Each half-step solves independent tridiagonal systems → O(N) Thomas algorithm
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from typing import Tuple
import config


def _thomas_algorithm(a: np.ndarray, b: np.ndarray,
                       c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Solves a tridiagonal system  A·x = d  via the Thomas algorithm (TDMA).

    Parameters
    ----------
    a : lower diagonal  (length n, a[0] unused)
    b : main  diagonal  (length n)
    c : upper diagonal  (length n, c[-1] unused)
    d : right-hand side (length n)

    Returns
    -------
    x : solution vector (length n)

    Complexity: O(n)
    """
    n = len(b)
    c_ = np.empty(n)
    d_ = np.empty(n)
    x  = np.empty(n)

    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]

    for i in range(1, n):
        denom   = b[i] - a[i] * c_[i - 1]
        c_[i]   = c[i] / denom
        d_[i]   = (d[i] - a[i] * d_[i - 1]) / denom

    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]

    return x


def _thomas_vectorized(a: float, b: np.ndarray,
                        c: float, D: np.ndarray) -> np.ndarray:
    """
    Solves multiple tridiagonal systems simultaneously.
    All systems share the same sub/super diagonal coefficients a, c
    but have different RHS columns in D (shape: n_systems × n_eq).

    Returns X of shape (n_systems, n_eq).
    """
    n_sys, n = D.shape
    c_ = np.empty(n)
    X  = np.empty_like(D)

    # Forward sweep — coefficients only depend on a, b (uniform), c (uniform)
    c_[0] = c / b[0]
    for i in range(1, n):
        denom  = b[i] - a * c_[i - 1]
        c_[i]  = c / denom

    # Forward sweep for RHS (vectorized over all systems)
    D_ = D.copy()
    D_[:, 0] /= b[0]
    for i in range(1, n):
        denom_i   = b[i] - a * c_[i - 1]
        D_[:, i]  = (D_[:, i] - a * D_[:, i - 1]) / denom_i

    # Backward substitution
    X[:, -1] = D_[:, -1]
    for i in range(n - 2, -1, -1):
        X[:, i] = D_[:, i] - c_[i] * X[:, i + 1]

    return X


class ADISolver:
    """
    Peaceman–Rachford ADI solver for the 2D heat equation.

    Parameters
    ----------
    nx, ny  : number of interior grid points (excluding Dirichlet boundaries)
    dx, dy  : spatial step sizes
    alpha   : thermal diffusivity coefficient
    bc      : boundary condition type ('dirichlet' | 'neumann' | 'periodic')
    """

    def __init__(self,
                 nx: int   = config.NX,
                 ny: int   = config.NY,
                 dx: float = config.DX,
                 dy: float = config.DY,
                 alpha: float = config.ALPHA,
                 bc: str = 'dirichlet'):

        self.nx    = nx
        self.ny    = ny
        self.dx    = dx
        self.dy    = dy
        self.alpha = alpha
        self.bc    = bc

    def _build_rhs_x(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Build RHS for the x-implicit half-step:
            RHS = (I + α Δt/2 Aᵧ) u

        Interior stencil in y:   u[i,j-1] - 2u[i,j] + u[i,j+1]
        """
        ry = self.alpha * dt / (2.0 * self.dy**2)
        rhs = np.zeros_like(u)
        rhs[:, 1:-1] = (u[:, 1:-1]
                        + ry * (u[:, :-2] - 2*u[:, 1:-1] + u[:, 2:]))
        # Dirichlet boundaries remain 0
        return rhs

    def _build_rhs_y(self, u_star: np.ndarray, dt: float) -> np.ndarray:
        """
        Build RHS for the y-implicit half-step:
            RHS = (I + α Δt/2 Aₓ) u*
        """
        rx = self.alpha * dt / (2.0 * self.dx**2)
        rhs = np.zeros_like(u_star)
        rhs[1:-1, :] = (u_star[1:-1, :]
                        + rx * (u_star[:-2, :] - 2*u_star[1:-1, :] + u_star[2:, :]))
        return rhs

    def step(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Advance solution by one ADI time-step Δt.

        Parameters
        ----------
        u  : current solution array, shape (ny, nx)   [row = y, col = x]
        dt : time step size

        Returns
        -------
        u_new : updated solution, shape (ny, nx)
        """
        nx, ny = self.nx, self.ny
        rx = self.alpha * dt / (2.0 * self.dx**2)
        ry = self.alpha * dt / (2.0 * self.dy**2)

        # ── Half-step: implicit in x ──────────────────────────────────────
        rhs_x = self._build_rhs_x(u, dt)   # shape (ny, nx)

        # Tridiagonal coefficients for x-direction systems
        ax = -rx
        cx = -rx
        bx = np.ones(nx)
        bx[1:-1] = 1.0 + 2*rx
        # Boundary: u=0 (Dirichlet) → trivial row
        bx[0] = 1.0; bx[-1] = 1.0
        ax_arr = np.zeros(nx); ax_arr[1:] = ax
        cx_arr = np.zeros(nx); cx_arr[:-1] = cx

        u_star = np.zeros_like(u)
        # Solve nx tridiagonal systems (one per row = y-index)
        for j in range(1, ny - 1):
            u_star[j, :] = _thomas_algorithm(ax_arr, bx, cx_arr, rhs_x[j, :])

        # ── Full-step: implicit in y ──────────────────────────────────────
        rhs_y = self._build_rhs_y(u_star, dt)   # shape (ny, nx)

        ay = -ry
        cy = -ry
        by = np.ones(ny)
        by[1:-1] = 1.0 + 2*ry
        by[0] = 1.0; by[-1] = 1.0
        ay_arr = np.zeros(ny); ay_arr[1:] = ay
        cy_arr = np.zeros(ny); cy_arr[:-1] = cy

        u_new = np.zeros_like(u)
        # Solve ny tridiagonal systems (one per column = x-index)
        for i in range(1, nx - 1):
            u_new[:, i] = _thomas_algorithm(ay_arr, by, cy_arr, rhs_y[:, i])

        return u_new

    def diffusion_number(self, dt: float) -> float:
        """Compute the von Neumann diffusion number r = α Δt / Δx²."""
        return self.alpha * dt / self.dx**2

    def compute_residual(self, u_prev: np.ndarray,
                          u_curr: np.ndarray, dt: float) -> float:
        """
        Estimate solution residual ‖(u_curr - u_prev)/dt - α∇²u_curr‖₂.
        Used as a feature for the ML timestep predictor.
        """
        laplacian = (
            (np.roll(u_curr, 1, axis=1) - 2*u_curr + np.roll(u_curr, -1, axis=1)) / self.dx**2 +
            (np.roll(u_curr, 1, axis=0) - 2*u_curr + np.roll(u_curr, -1, axis=0)) / self.dy**2
        )
        time_deriv = (u_curr - u_prev) / dt
        residual   = time_deriv - self.alpha * laplacian
        return float(np.linalg.norm(residual) / (self.nx * self.ny))

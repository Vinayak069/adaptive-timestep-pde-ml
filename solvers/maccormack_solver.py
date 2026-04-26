"""
solvers/maccormack_solver.py
─────────────────────────────────────────────────────────────────────────────
MacCormack Predictor-Corrector Solver for the 2D Linear Advection Equation:

    ∂u/∂t + cₓ ∂u/∂x + cᵧ ∂u/∂y = 0

Scheme (MacCormack, 1969):
  Predictor  (forward differences):
    ũ[i,j] = uⁿ[i,j]
             − (cₓ Δt/Δx)(uⁿ[i+1,j] − uⁿ[i,j])
             − (cᵧ Δt/Δy)(uⁿ[i,j+1] − uⁿ[i,j])

  Corrector  (backward differences on predictor):
    u^{n+1}[i,j] = ½(uⁿ[i,j] + ũ[i,j]
                   − (cₓ Δt/Δx)(ũ[i,j] − ũ[i−1,j])
                   − (cᵧ Δt/Δy)(ũ[i,j] − ũ[i,j−1]))

Properties:
  • Second-order accurate in space and time: O(Δx², Δy², Δt²)
  • Explicit → requires CFL condition:
      Δt ≤ CFL / (|cₓ|/Δx + |cᵧ|/Δy)      [2D CFL]
  • Directional alternation prevents directional bias
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from typing import Tuple
import config


class MacCormackSolver:
    """
    MacCormack predictor-corrector solver for 2D advection.

    Also supports optional artificial viscosity term to suppress
    Gibbs-type oscillations near steep gradients.

    Parameters
    ----------
    nx, ny : grid dimensions
    dx, dy : spatial step sizes
    cx, cy : advection velocities
    bc     : boundary type ('periodic' recommended for advection)
    nu_art : artificial viscosity coefficient (set 0 to disable)
    """

    def __init__(self,
                 nx: int   = config.NX,
                 ny: int   = config.NY,
                 dx: float = config.DX,
                 dy: float = config.DY,
                 cx: float = config.CX,
                 cy: float = config.CY,
                 bc: str = 'periodic',
                 nu_art: float = 0.0):

        self.nx     = nx
        self.ny     = ny
        self.dx     = dx
        self.dy     = dy
        self.cx     = cx
        self.cy     = cy
        self.bc     = bc
        self.nu_art = nu_art

    def _apply_bc(self, u: np.ndarray) -> np.ndarray:
        """Apply boundary conditions (periodic by default)."""
        if self.bc == 'periodic':
            return u  # np.roll handles periodicity natively
        elif self.bc == 'dirichlet':
            u[0, :]  = 0.0
            u[-1, :] = 0.0
            u[:, 0]  = 0.0
            u[:, -1] = 0.0
        return u

    def cfl_number(self, dt: float) -> float:
        """Compute 2D CFL number = dt * (|cx|/dx + |cy|/dy)."""
        return dt * (abs(self.cx) / self.dx + abs(self.cy) / self.dy)

    def max_stable_dt(self, cfl: float = config.CFL_TARGET) -> float:
        """Return maximum stable Δt for given CFL target."""
        return cfl / (abs(self.cx) / self.dx + abs(self.cy) / self.dy + 1e-12)

    def step(self, u: np.ndarray, dt: float,
             alternate: bool = True, step_idx: int = 0) -> np.ndarray:
        """
        Advance solution one step using the MacCormack scheme.

        Parameters
        ----------
        u         : current solution, shape (ny, nx)
        dt        : time step size
        alternate : alternate predictor direction each step to reduce bias
        step_idx  : current step index (used for alternation)

        Returns
        -------
        u_new : updated solution
        """
        dtdx = dt / self.dx
        dtdy = dt / self.dy

        if self.bc == 'periodic':
            u_pad_x_fwd  = np.roll(u, -1, axis=1)   # u[i, j+1]
            u_pad_x_bwd  = np.roll(u,  1, axis=1)   # u[i, j-1]
            u_pad_y_fwd  = np.roll(u, -1, axis=0)   # u[i+1, j]
            u_pad_y_bwd  = np.roll(u,  1, axis=0)   # u[i-1, j]
        else:
            # Zero-gradient at boundaries for non-periodic
            u_pad_x_fwd = np.pad(u[:, 1:],  ((0,0),(0,1)), mode='edge')
            u_pad_x_bwd = np.pad(u[:, :-1], ((0,0),(1,0)), mode='edge')
            u_pad_y_fwd = np.pad(u[1:, :],  ((0,1),(0,0)), mode='edge')
            u_pad_y_bwd = np.pad(u[:-1, :], ((1,0),(0,0)), mode='edge')

        # ── Predictor (forward differences) ─────────────────────────────
        if not alternate or (step_idx % 2 == 0):
            u_pred = (u
                      - self.cx * dtdx * (u_pad_x_fwd - u)
                      - self.cy * dtdy * (u_pad_y_fwd - u))
        else:
            # Alternate: start with backward on predictor
            u_pred = (u
                      - self.cx * dtdx * (u - u_pad_x_bwd)
                      - self.cy * dtdy * (u - u_pad_y_bwd))

        u_pred = self._apply_bc(u_pred)

        # ── Recompute neighbors of predictor field ────────────────────────
        if self.bc == 'periodic':
            up_x_bwd = np.roll(u_pred, 1, axis=1)
            up_y_bwd = np.roll(u_pred, 1, axis=0)
            up_x_fwd = np.roll(u_pred, -1, axis=1)
            up_y_fwd = np.roll(u_pred, -1, axis=0)
        else:
            up_x_bwd = np.pad(u_pred[:, :-1], ((0,0),(1,0)), mode='edge')
            up_y_bwd = np.pad(u_pred[:-1, :], ((1,0),(0,0)), mode='edge')
            up_x_fwd = np.pad(u_pred[:, 1:],  ((0,0),(0,1)), mode='edge')
            up_y_fwd = np.pad(u_pred[1:, :],  ((0,1),(0,0)), mode='edge')

        # ── Corrector (backward differences on predicted) ─────────────────
        if not alternate or (step_idx % 2 == 0):
            u_corr = 0.5 * (u + u_pred
                            - self.cx * dtdx * (u_pred - up_x_bwd)
                            - self.cy * dtdy * (u_pred - up_y_bwd))
        else:
            u_corr = 0.5 * (u + u_pred
                            - self.cx * dtdx * (up_x_fwd - u_pred)
                            - self.cy * dtdy * (up_y_fwd - u_pred))

        # ── Artificial viscosity (optional smoothing) ─────────────────────
        if self.nu_art > 0:
            lap = (u_pad_x_fwd - 2*u + u_pad_x_bwd) / self.dx**2 \
                + (u_pad_y_fwd - 2*u + u_pad_y_bwd) / self.dy**2
            u_corr = u_corr + self.nu_art * dt * lap

        u_corr = self._apply_bc(u_corr)
        return u_corr

    def compute_residual(self, u_prev: np.ndarray,
                          u_curr: np.ndarray, dt: float) -> float:
        """
        Estimate PDE residual ‖∂u/∂t + c·∇u‖₂ at current state.
        """
        dudx = (np.roll(u_curr, -1, axis=1) - np.roll(u_curr, 1, axis=1)) / (2*self.dx)
        dudy = (np.roll(u_curr, -1, axis=0) - np.roll(u_curr, 1, axis=0)) / (2*self.dy)
        dudt = (u_curr - u_prev) / dt
        residual = dudt + self.cx * dudx + self.cy * dudy
        return float(np.linalg.norm(residual) / (self.nx * self.ny))

    def check_stability(self, dt: float, warn: bool = True) -> bool:
        """Return True if CFL condition is satisfied."""
        cfl = self.cfl_number(dt)
        stable = cfl <= 1.0
        if not stable and warn:
            print(f"[WARNING] CFL = {cfl:.4f} > 1 → UNSTABLE! dt={dt:.2e}")
        return stable

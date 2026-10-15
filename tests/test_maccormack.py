"""MacCormack solver on 2D linear advection with periodic boundaries."""

import numpy as np
import pytest

from solvers import MacCormackSolver

N      = 64
DX     = 1.0 / N            # periodic grid: x_i = i·Δx, i = 0..N-1
CX, CY = 0.5, 0.3
SIGMA  = 0.08


def gaussian(t=0.0):
    """Gaussian pulse starting at (0.5, 0.5), advected by (CX, CY) and wrapped."""
    x = np.arange(N) * DX
    X, Y = np.meshgrid(x, x)
    dx = (X - 0.5 - CX * t + 0.5) % 1.0 - 0.5
    dy = (Y - 0.5 - CY * t + 0.5) % 1.0 - 0.5
    return np.exp(-(dx**2 + dy**2) / (2 * SIGMA**2))


@pytest.fixture
def solver():
    return MacCormackSolver(nx=N, ny=N, dx=DX, dy=DX, cx=CX, cy=CY)


def advect(solver, u, dt, n_steps):
    for k in range(n_steps):
        u = solver.step(u, dt, step_idx=k)
    return u


@pytest.mark.parametrize("cfl", [0.5, 0.9])
def test_max_stable_dt_gives_requested_cfl(solver, cfl):
    dt = solver.max_stable_dt(cfl)
    assert solver.cfl_number(dt) == pytest.approx(cfl, rel=1e-9)
    assert solver.check_stability(dt, warn=False)


@pytest.mark.parametrize("cfl", [0.5, 0.9])
def test_conserves_mass_and_tracks_exact_solution(solver, cfl):
    dt = solver.max_stable_dt(cfl)
    n_steps = 40
    u0 = gaussian()
    u = advect(solver, u0, dt, n_steps)

    # Periodic, conservative differences → total mass is preserved
    assert u.sum() == pytest.approx(u0.sum(), rel=1e-10)
    # Stable and close to the exactly shifted pulse
    assert np.isfinite(u).all()
    u_exact = gaussian(t=n_steps * dt)
    assert np.linalg.norm(u - u_exact) / np.linalg.norm(u_exact) < 0.05

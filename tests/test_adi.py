"""
ADI solver on the 2D heat equation with a known analytic solution:

    u(x, y, 0) = sin(πx) sin(πy)   →   u(x, y, t) = exp(-2απ²t) sin(πx) sin(πy)
"""

import numpy as np
import pytest

from solvers import ADISolver

N     = 33          # small grid keeps the test fast
DX    = 1.0 / (N - 1)
ALPHA = 1.0         # large α so the solution decays noticeably by T
T     = 0.05


@pytest.fixture
def setup():
    x = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, x)
    u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)
    solver = ADISolver(nx=N, ny=N, dx=DX, dy=DX, alpha=ALPHA)
    return solver, u0


def run(solver, u0, n_steps):
    dt = T / n_steps
    u = u0.copy()
    for _ in range(n_steps):
        u = solver.step(u, dt)
    return u


def rel_l2(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def test_matches_analytic_solution(setup):
    solver, u0 = setup
    u = run(solver, u0, n_steps=40)
    u_exact = np.exp(-2 * ALPHA * np.pi**2 * T) * u0
    assert rel_l2(u, u_exact) < 2e-3


def test_dirichlet_boundaries_stay_zero(setup):
    solver, u0 = setup
    u = run(solver, u0, n_steps=10)
    for edge in (u[0, :], u[-1, :], u[:, 0], u[:, -1]):
        assert np.max(np.abs(edge)) < 1e-12


def test_temporal_error_shrinks_second_order(setup):
    """Halving Δt should cut the time-discretisation error by ~4x."""
    solver, u0 = setup
    u_ref = run(solver, u0, n_steps=640)     # fine-Δt reference, same grid
    errors = [rel_l2(run(solver, u0, n), u_ref) for n in (5, 10, 20)]
    assert errors[0] > errors[1] > errors[2]
    for coarse, fine in zip(errors, errors[1:]):
        assert 3.0 < coarse / fine < 5.0

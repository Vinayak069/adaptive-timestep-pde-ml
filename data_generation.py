"""
data_generation.py
─────────────────────────────────────────────────────────────────────────────
Simulation Data Generator for ML Training.

Runs fixed-timestep PDE simulations and collects:
  • Feature vectors at each step
  • Optimal Δt targets (CFL-based ground truth)

Two PDE types:
  1. Heat equation   → ADI solver
  2. 2D Advection    → MacCormack solver

Initial conditions include:
  • Single Gaussian pulses
  • Double Gaussian (interaction test)
  • Step functions (sharp gradients)
  • Sinusoidal profiles
─────────────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
from tqdm import tqdm
import config
from solvers.adi_solver import ADISolver
from solvers.maccormack_solver import MacCormackSolver
from ml.feature_extractor import extract_features, compute_optimal_dt


# ──────────────────────────────────────────────────────────────────────────
# Initial Conditions
# ──────────────────────────────────────────────────────────────────────────

def ic_gaussian(nx, ny, dx, dy, cx=0.5, cy=0.5,
                sigma=0.08, amp=1.0) -> np.ndarray:
    """Single Gaussian pulse centered at (cx, cy)."""
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    return amp * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))


def ic_double_gaussian(nx, ny, dx, dy) -> np.ndarray:
    """Two overlapping Gaussian pulses."""
    return (ic_gaussian(nx, ny, dx, dy, cx=0.35, cy=0.35, sigma=0.07, amp=1.0) +
            ic_gaussian(nx, ny, dx, dy, cx=0.65, cy=0.65, sigma=0.05, amp=0.8))


def ic_sinusoidal(nx, ny, dx, dy, k=2) -> np.ndarray:
    """Sinusoidal initial condition."""
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    return np.sin(k * np.pi * X) * np.sin(k * np.pi * Y)


def ic_step(nx, ny, dx, dy) -> np.ndarray:
    """Step function — tests sharp gradient handling."""
    u = np.zeros((ny, nx))
    u[ny//4:3*ny//4, nx//4:3*nx//4] = 1.0
    return u


def ic_ring(nx, ny, dx, dy) -> np.ndarray:
    """Ring / annulus initial condition."""
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
    return np.exp(-((r - 0.25)**2) / (2 * 0.03**2))


IC_REGISTRY = {
    'gaussian':        ic_gaussian,
    'double_gaussian': ic_double_gaussian,
    'sinusoidal':      ic_sinusoidal,
    'step':            ic_step,
    'ring':            ic_ring,
}


# ──────────────────────────────────────────────────────────────────────────
# Data Collection Simulation
# ──────────────────────────────────────────────────────────────────────────

def _run_adi_simulation(ic_name: str,
                         t_end: float = config.T_END,
                         dt_fixed: float = config.DT_FIXED,
                         nx: int = config.NX,
                         ny: int = config.NY,
                         dx: float = config.DX,
                         dy: float = config.DY,
                         alpha: float = config.ALPHA) -> tuple:
    """
    Run one heat-equation simulation using ADI with fixed timestep.

    Returns
    -------
    (features_list, targets_list)  — lists of arrays at each step
    """
    solver = ADISolver(nx=nx, ny=ny, dx=dx, dy=dy, alpha=alpha)
    ic_fn  = IC_REGISTRY.get(ic_name, ic_gaussian)

    u = ic_fn(nx, ny, dx, dy)
    u_prev = u.copy()
    dt = dt_fixed
    t  = 0.0

    features_list = []
    targets_list  = []

    while t < t_end:
        dt = min(dt, t_end - t)

        feats = extract_features(
            u, u_prev, dt_prev=dt,
            cx=0.0, cy=0.0,
            dx=dx, dy=dy,
            solver_type='adi'
        )
        dt_opt = compute_optimal_dt(u, dx, dy,
                                    cx=0.0, cy=0.0,
                                    alpha=alpha,
                                    solver_type='adi')

        features_list.append(feats)
        targets_list.append(np.float32(np.log(dt_opt / config.DT_FIXED + 1e-12)))

        u_prev = u.copy()
        u = solver.step(u, dt)
        t += dt

    return features_list, targets_list


def _run_maccormack_simulation(ic_name: str,
                                t_end: float = config.T_END,
                                dt_fixed: float = None,
                                nx: int = config.NX,
                                ny: int = config.NY,
                                dx: float = config.DX,
                                dy: float = config.DY,
                                cx: float = config.CX,
                                cy: float = config.CY) -> tuple:
    """
    Run one advection simulation using MacCormack with fixed CFL-limited dt.

    Returns
    -------
    (features_list, targets_list)
    """
    solver = MacCormackSolver(nx=nx, ny=ny, dx=dx, dy=dy, cx=cx, cy=cy)

    # Fixed dt = CFL-limited dt (safe for explicit scheme)
    if dt_fixed is None:
        dt_fixed = solver.max_stable_dt(cfl=0.4)

    ic_fn = IC_REGISTRY.get(ic_name, ic_gaussian)
    u     = ic_fn(nx, ny, dx, dy)
    u_prev = u.copy()
    t  = 0.0
    step_idx = 0

    features_list = []
    targets_list  = []

    while t < t_end:
        dt = min(dt_fixed, t_end - t)

        feats = extract_features(
            u, u_prev, dt_prev=dt,
            cx=cx, cy=cy,
            dx=dx, dy=dy,
            solver_type='maccormack'
        )
        dt_opt = compute_optimal_dt(u, dx, dy, cx=cx, cy=cy,
                                    alpha=config.ALPHA,
                                    solver_type='maccormack')

        features_list.append(feats)
        dt_ref = solver.max_stable_dt(cfl=config.CFL_TARGET)
        targets_list.append(np.float32(np.log(dt_opt / (dt_ref + 1e-12))))

        u_prev = u.copy()
        u = solver.step(u, dt, step_idx=step_idx)
        t += dt
        step_idx += 1

    return features_list, targets_list


# ──────────────────────────────────────────────────────────────────────────
# Master Dataset Generator
# ──────────────────────────────────────────────────────────────────────────

def generate_dataset(pde_type:   str   = 'maccormack',
                     n_sims:     int   = config.N_SIMULATIONS,
                     t_end:      float = config.T_END,
                     save:       bool  = True,
                     data_dir:   str   = config.DATA_DIR,
                     verbose:    bool  = True) -> tuple:
    """
    Generate training dataset by running multiple simulations.

    Parameters
    ----------
    pde_type : 'adi' | 'maccormack'
    n_sims   : number of simulations (varied IC parameters)
    t_end    : final time for each simulation
    save     : whether to save arrays to disk

    Returns
    -------
    (X, y) — feature matrix (N, 7) and log-ratio targets (N,)
    """
    ic_names = list(IC_REGISTRY.keys())
    all_features = []
    all_targets  = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Generating {pde_type.upper()} Training Data")
        print(f"  {n_sims} simulations × ~{t_end/config.DT_FIXED:.0f} steps/sim")
        print(f"{'='*60}")

    sim_iter = tqdm(range(n_sims), desc=f"[{pde_type}] Simulating",
                    disable=not verbose)

    for i in sim_iter:
        ic_name = ic_names[i % len(ic_names)]

        try:
            if pde_type == 'adi':
                # Vary thermal diffusivity slightly per simulation
                alpha_var = config.ALPHA * (0.5 + np.random.rand())
                feats, tgts = _run_adi_simulation(
                    ic_name=ic_name,
                    t_end=t_end,
                    alpha=alpha_var
                )
            else:
                # Vary advection velocities slightly
                cx_var = config.CX * (0.7 + 0.6 * np.random.rand())
                cy_var = config.CY * (0.7 + 0.6 * np.random.rand())
                feats, tgts = _run_maccormack_simulation(
                    ic_name=ic_name,
                    t_end=t_end,
                    cx=cx_var, cy=cy_var
                )

            all_features.extend(feats)
            all_targets.extend(tgts)

        except Exception as e:
            if verbose:
                print(f"\n[Warning] Simulation {i} failed: {e}")

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_targets,  dtype=np.float32)

    if verbose:
        print(f"\n[✓] Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"    y ∈ [{y.min():.4f}, {y.max():.4f}]   mean={y.mean():.4f}")

    if save:
        os.makedirs(data_dir, exist_ok=True)
        np.save(os.path.join(data_dir, f'X_{pde_type}.npy'), X)
        np.save(os.path.join(data_dir, f'y_{pde_type}.npy'), y)
        if verbose:
            print(f"    Saved to '{data_dir}/'")

    return X, y


def build_lstm_sequences(X: np.ndarray, y: np.ndarray,
                          seq_len: int = config.SEQ_LEN) -> tuple:
    """
    Reshape flat feature array into overlapping sequences for LSTM training.

    Parameters
    ----------
    X       : (N, input_dim)
    y       : (N,)
    seq_len : length of each input sequence

    Returns
    -------
    (X_seq, y_seq) — shapes (N-seq_len, seq_len, input_dim) and (N-seq_len,)
    """
    N = len(X)
    X_seq = np.stack([X[i:i+seq_len] for i in range(N - seq_len)])
    y_seq = y[seq_len:]
    return X_seq.astype(np.float32), y_seq.astype(np.float32)


if __name__ == '__main__':
    np.random.seed(config.SEED)
    print("Generating MacCormack (advection) dataset...")
    X_mac, y_mac = generate_dataset('maccormack', n_sims=30)

    print("\nGenerating ADI (heat equation) dataset...")
    X_adi, y_adi = generate_dataset('adi', n_sims=20)

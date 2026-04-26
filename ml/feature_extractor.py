"""
ml/feature_extractor.py
─────────────────────────────────────────────────────────────────────────────
Feature Engineering for the ML Timestep Predictor.

At each simulation step, we extract a fixed-length feature vector from the
current PDE state u(x,y,t) that encodes information relevant to numerical
stability and optimal timestep selection.

Feature Vector (dim = 7):
  [0] mean_grad_x   — Mean |∂u/∂x|  over domain
  [1] mean_grad_y   — Mean |∂u/∂y|  over domain
  [2] max_grad      — Max(|∇u|)  — sharpest gradient peak
  [3] u_rms         — RMS of solution field (energy measure)
  [4] residual      — PDE residual estimate
  [5] dt_prev       — Previous timestep (temporal memory)
  [6] cfl_est       — Estimated CFL number at current state

These features capture:
  1. Spatial variability  → gradient-based features
  2. Solution amplitude   → RMS
  3. Temporal consistency → residual + dt_prev
  4. Stability margin     → CFL estimate
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import config


def extract_features(u: np.ndarray,
                     u_prev: np.ndarray,
                     dt_prev: float,
                     cx: float = config.CX,
                     cy: float = config.CY,
                     dx: float = config.DX,
                     dy: float = config.DY,
                     solver_type: str = 'maccormack') -> np.ndarray:
    """
    Extract a 7-dimensional feature vector from the current PDE state.

    Parameters
    ----------
    u           : current solution field, shape (ny, nx)
    u_prev      : solution at previous timestep
    dt_prev     : previous timestep Δt
    cx, cy      : advection velocities (for CFL estimation)
    dx, dy      : spatial step sizes
    solver_type : 'adi' or 'maccormack' (affects residual computation)

    Returns
    -------
    features : np.ndarray of shape (7,), normalized to reasonable scale
    """
    # ── Gradient features (central differences) ─────────────────────────
    grad_x = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dx)
    grad_y = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * dy)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    mean_grad_x = float(np.mean(np.abs(grad_x)))
    mean_grad_y = float(np.mean(np.abs(grad_y)))
    max_grad    = float(np.max(grad_mag))

    # ── RMS of solution ──────────────────────────────────────────────────
    u_rms = float(np.sqrt(np.mean(u**2)))

    # ── Residual estimate  ───────────────────────────────────────────────
    if dt_prev > 0:
        dudt = (u - u_prev) / dt_prev
        if solver_type == 'adi':
            lap = (
                (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dx**2 +
                (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / dy**2
            )
            residual = float(np.linalg.norm(dudt - config.ALPHA * lap)
                             / (u.size + 1e-12))
        else:  # maccormack / advection
            adv = cx * grad_x + cy * grad_y
            residual = float(np.linalg.norm(dudt + adv) / (u.size + 1e-12))
    else:
        residual = 0.0

    # ── CFL estimate ─────────────────────────────────────────────────────
    # Use max wave speed * current grad amplitude as proxy
    lambda_max = max(abs(cx), abs(cy))
    cfl_est    = dt_prev * lambda_max * max_grad / (min(dx, dy) + 1e-12)

    features = np.array([
        mean_grad_x,
        mean_grad_y,
        max_grad,
        u_rms,
        residual,
        dt_prev,
        cfl_est
    ], dtype=np.float32)

    return features


def normalize_features(features: np.ndarray,
                        stats: dict = None) -> tuple:
    """
    Standardize features: x̂ = (x - μ) / σ.

    Parameters
    ----------
    features : shape (N, 7) dataset array
    stats    : optional dict with pre-computed {'mean', 'std'}

    Returns
    -------
    (normalized_features, stats_dict)
    """
    if stats is None:
        mu  = features.mean(axis=0)
        sig = features.std(axis=0) + 1e-8
        stats = {'mean': mu, 'std': sig}
    norm = (features - stats['mean']) / stats['std']
    return norm.astype(np.float32), stats


def compute_optimal_dt(u: np.ndarray,
                        dx: float, dy: float,
                        cx: float, cy: float,
                        alpha: float,
                        solver_type: str = 'maccormack',
                        safety_factor: float = 0.9) -> float:
    """
    Compute the ground-truth optimal Δt based on CFL + diffusion constraints.
    Used as the regression target during training.

    For MacCormack (advection):
        Δt_CFL = CFL_target / (|cx|/dx + |cy|/dy)

    For ADI (diffusion):
        ADI is unconditionally stable, but we still bound Δt to avoid
        large temporal errors. We use:
        Δt_target = 0.1 * Δx² / (2α)

    Parameters
    ----------
    safety_factor : multiplier < 1 applied to the theoretical maximum
    """
    if solver_type == 'maccormack':
        # 2D CFL condition
        dt_cfl = config.CFL_TARGET / (
            abs(cx)/dx + abs(cy)/dy + 1e-12)
        dt_opt = safety_factor * dt_cfl
    else:  # adi / diffusion
        # Accuracy-based bound for diffusion
        dt_diff = 0.5 * min(dx, dy)**2 / (2.0 * alpha + 1e-12)
        dt_opt  = safety_factor * dt_diff

    # Gradient-based refinement: tighten near sharp gradients
    grad_x = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dx)
    grad_y = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * dy)
    max_grad = float(np.max(np.sqrt(grad_x**2 + grad_y**2)))

    if max_grad > 10.0:
        dt_opt *= 0.5     # Tighten near strong gradients
    elif max_grad < 0.1:
        dt_opt *= 1.2     # Allow larger step in smooth regions

    return float(np.clip(dt_opt, config.DT_MIN, config.DT_MAX))

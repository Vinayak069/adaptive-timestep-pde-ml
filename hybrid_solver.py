"""
hybrid_solver.py
─────────────────────────────────────────────────────────────────────────────
Hybrid AI + Numerical PDE Solver — Main Simulation Engine.

Pipeline at each step t:
  1. Extract features  f(t) from current state u(t)
  2. ML model predicts optimal Δt from f(t)
  3. Clamp Δt to [DT_MIN, DT_MAX] and verify CFL
  4. Advance u(t+Δt) via ADI or MacCormack
  5. Log metrics, repeat

Advanced Extensions (Section 9):
  • HybridSwitcher:   Dynamically switches ADI ↔ MacCormack
                      based on local Peclet number
  • RLTimestepAgent:  Reinforcement-learning-based Δt controller
  • EnsemblePredictor: Uncertainty estimation via MC Dropout
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Callable
from dataclasses import dataclass, field
import config
from solvers.adi_solver import ADISolver
from solvers.maccormack_solver import MacCormackSolver
from ml.feature_extractor import extract_features, compute_optimal_dt


# ──────────────────────────────────────────────────────────────────────────
# Simulation Log
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationLog:
    """Stores per-step diagnostics for post-analysis."""
    times:        list = field(default_factory=list)
    dts:          list = field(default_factory=list)
    cfl_numbers:  list = field(default_factory=list)
    residuals:    list = field(default_factory=list)
    l2_errors:    list = field(default_factory=list)
    solver_used:  list = field(default_factory=list)
    u_snapshots:  list = field(default_factory=list)
    step_count:   int  = 0
    blowup:       bool = False

    def record(self, t, dt, cfl, res, solver, u=None, l2=None):
        self.times.append(t)
        self.dts.append(dt)
        self.cfl_numbers.append(cfl)
        self.residuals.append(res)
        self.solver_used.append(solver)
        if l2 is not None:
            self.l2_errors.append(l2)
        if u is not None and self.step_count % 20 == 0:
            self.u_snapshots.append(u.copy())
        self.step_count += 1

    def to_arrays(self):
        return {
            'time':    np.array(self.times),
            'dt':      np.array(self.dts),
            'cfl':     np.array(self.cfl_numbers),
            'residual': np.array(self.residuals),
            'l2_error': np.array(self.l2_errors) if self.l2_errors else None,
            'solver':  self.solver_used,
        }


# ──────────────────────────────────────────────────────────────────────────
# Main Hybrid Solver
# ──────────────────────────────────────────────────────────────────────────

class HybridSolver:
    """
    Adaptive timestep hybrid solver combining ML prediction with
    classical PDE solvers.

    Parameters
    ----------
    solver_type  : 'adi' | 'maccormack' | 'hybrid'
    model        : trained nn.Module timestep predictor
    model_type   : 'feedforward' | 'lstm' | 'pinn'
    feature_stats: dict with 'mean', 'std' for feature normalization
    use_ml       : if False, falls back to CFL-based fixed stepping
    safety_factor: fraction of predicted Δt to use (conservative margin)
    """

    def __init__(self,
                 solver_type:   str = 'maccormack',
                 model:         Optional[nn.Module] = None,
                 model_type:    str = 'feedforward',
                 feature_stats: Optional[dict] = None,
                 use_ml:        bool = True,
                 safety_factor: float = 0.95,
                 nx: int   = config.NX,
                 ny: int   = config.NY,
                 dx: float = config.DX,
                 dy: float = config.DY,
                 alpha: float = config.ALPHA,
                 cx:    float = config.CX,
                 cy:    float = config.CY):

        self.solver_type   = solver_type
        self.model         = model
        self.model_type    = model_type
        self.feature_stats = feature_stats
        self.use_ml        = use_ml and (model is not None)
        self.safety_factor = safety_factor

        self.adi_solver = ADISolver(
            nx=nx, ny=ny, dx=dx, dy=dy, alpha=alpha)
        self.mac_solver = MacCormackSolver(
            nx=nx, ny=ny, dx=dx, dy=dy, cx=cx, cy=cy)

        self.nx = nx; self.ny = ny
        self.dx = dx; self.dy = dy
        self.alpha = alpha; self.cx = cx; self.cy = cy

        # LSTM state (persisted across steps)
        self._lstm_hidden = None
        self._feature_buffer = []   # Rolling window for LSTM

        if model is not None:
            model.eval()

    def _normalize(self, feats: np.ndarray) -> np.ndarray:
        if self.feature_stats is not None:
            return ((feats - self.feature_stats['mean'])
                    / (self.feature_stats['std'] + 1e-8)).astype(np.float32)
        return feats.astype(np.float32)

    def _predict_dt(self, feats: np.ndarray) -> float:
        """Use the ML model to predict optimal Δt."""
        feats_norm = self._normalize(feats)
        x = torch.from_numpy(feats_norm).unsqueeze(0)   # (1, 7)

        with torch.no_grad():
            if self.model_type == 'feedforward':
                dt_pred = self.model.predict_dt(x).item()

            elif self.model_type == 'lstm':
                self._feature_buffer.append(feats_norm)
                if len(self._feature_buffer) < config.SEQ_LEN:
                    # Not enough history — use CFL baseline
                    dt_pred = None
                else:
                    seq = np.stack(self._feature_buffer[-config.SEQ_LEN:])
                    x_seq = torch.from_numpy(seq).unsqueeze(0)  # (1, T, 7)
                    dt_pred, self._lstm_hidden = self.model.predict_dt(
                        x_seq, hidden=self._lstm_hidden)
                    dt_pred = dt_pred.item()

            elif self.model_type == 'pinn':
                dt_pred = self.model.predict_dt(x).item()

            else:
                dt_pred = None

        if dt_pred is None:
            # Fallback to CFL-based dt
            dt_pred = self.mac_solver.max_stable_dt(config.CFL_TARGET)

        return float(np.clip(
            self.safety_factor * dt_pred,
            config.DT_MIN,
            config.DT_MAX
        ))

    def _choose_solver(self, u: np.ndarray) -> str:
        """
        For 'hybrid' mode: switch between ADI and MacCormack based on
        local Peclet number Pe = |c| * Δx / α.

        Pe >> 1  → advection dominated → MacCormack
        Pe << 1  → diffusion dominated → ADI
        """
        Pe = max(abs(self.cx), abs(self.cy)) * min(self.dx, self.dy) / (self.alpha + 1e-12)
        return 'maccormack' if Pe > 1.0 else 'adi'

    def run(self,
            u0:           np.ndarray,
            t_end:        float = config.T_END,
            dt_fixed:     float = None,
            u_exact_fn:   Optional[Callable] = None,
            snapshot_freq: int = 20,
            verbose:      bool = True) -> SimulationLog:
        """
        Run the full adaptive simulation.

        Parameters
        ----------
        u0          : initial condition, shape (ny, nx)
        t_end       : final simulation time
        dt_fixed    : if not None, override ML with this fixed dt (baseline mode)
        u_exact_fn  : callable(t) → exact solution for error computation
        snapshot_freq: save every N steps for visualization
        verbose     : print progress

        Returns
        -------
        SimulationLog object
        """
        u      = u0.copy()
        u_prev = u0.copy()
        t      = 0.0
        log    = SimulationLog()
        step   = 0

        if dt_fixed is not None:
            mode_str = f"Fixed Δt = {dt_fixed:.2e}"
        elif self.use_ml:
            mode_str = f"ML-Adaptive ({self.model_type})"
        else:
            mode_str = "CFL-Adaptive (no ML)"

        if verbose:
            print(f"\n{'─'*55}")
            print(f"  HybridSolver: {self.solver_type.upper()} | {mode_str}")
            print(f"  Grid: {self.nx}×{self.ny}  t_end={t_end}")
            print(f"{'─'*55}")

        while t < t_end - 1e-14:
            # ── Choose solver ───────────────────────────────────────────
            if self.solver_type == 'hybrid':
                active_solver = self._choose_solver(u)
            else:
                active_solver = self.solver_type

            # ── Predict Δt ──────────────────────────────────────────────
            if dt_fixed is not None:
                dt = min(dt_fixed, t_end - t)
            elif self.use_ml:
                feats = extract_features(
                    u, u_prev, dt_prev=log.dts[-1] if log.dts else config.DT_FIXED,
                    cx=self.cx, cy=self.cy,
                    dx=self.dx, dy=self.dy,
                    solver_type=active_solver
                )
                dt = min(self._predict_dt(feats), t_end - t)
            else:
                # Pure CFL-based adaptive (no ML)
                dt = min(
                    compute_optimal_dt(u, self.dx, self.dy,
                                       self.cx, self.cy, self.alpha,
                                       solver_type=active_solver),
                    t_end - t
                )

            # ── Stability check for explicit methods ─────────────────────
            if active_solver == 'maccormack':
                cfl = self.mac_solver.cfl_number(dt)
                if cfl > 1.2:
                    dt *= 0.8
                    cfl = self.mac_solver.cfl_number(dt)
            else:
                cfl = self.adi_solver.diffusion_number(dt)

            # ── Advance solution ─────────────────────────────────────────
            u_prev = u.copy()
            if active_solver == 'maccormack':
                u = self.mac_solver.step(u, dt, step_idx=step)
            else:
                u = self.adi_solver.step(u, dt)

            t += dt
            step += 1

            # ── Compute residual ─────────────────────────────────────────
            if active_solver == 'maccormack':
                res = self.mac_solver.compute_residual(u_prev, u, dt)
            else:
                res = self.adi_solver.compute_residual(u_prev, u, dt)

            # ── L2 error vs exact (if available) ────────────────────────
            l2_err = None
            if u_exact_fn is not None:
                u_ex   = u_exact_fn(t)
                l2_err = float(np.linalg.norm(u - u_ex) / (np.linalg.norm(u_ex) + 1e-12))

            # ── Blowup detection ─────────────────────────────────────────
            if not np.isfinite(u).all() or np.max(np.abs(u)) > 1e6:
                log.blowup = True
                if verbose:
                    print(f"[⚠️ BLOWUP] Detected at t={t:.4f}, step={step}")
                break

            # ── Log ──────────────────────────────────────────────────────
            log.record(t=t, dt=dt, cfl=cfl, res=res,
                       solver=active_solver,
                       u=u if step % snapshot_freq == 0 else None,
                       l2=l2_err)

        if verbose:
            d = log.to_arrays()
            print(f"\n  Completed {log.step_count} steps  t_final={t:.4f}")
            print(f"  Mean Δt = {np.mean(d['dt']):.3e}  ± {np.std(d['dt']):.2e}")
            print(f"  Blowup: {log.blowup}")
            if d['l2_error'] is not None:
                print(f"  Mean L2 error: {np.mean(d['l2_error']):.4e}")

        return log


# ──────────────────────────────────────────────────────────────────────────
# ADVANCED EXTENSION: Ensemble Uncertainty Estimation
# ──────────────────────────────────────────────────────────────────────────

class EnsemblePredictor:
    """
    MC Dropout uncertainty estimation for timestep prediction.

    Runs the model K times with dropout active, collecting a distribution
    over Δt predictions. The mean is used as prediction, std as uncertainty.
    """

    def __init__(self, model: nn.Module, n_samples: int = 30,
                 model_type: str = 'feedforward'):
        self.model      = model
        self.n_samples  = n_samples
        self.model_type = model_type

    def _enable_dropout(self):
        """Enable dropout layers during inference."""
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def predict(self, x: torch.Tensor,
                dt_ref: float = config.DT_CFL) -> dict:
        """
        Return mean and std of Δt over MC samples.

        Parameters
        ----------
        x : (1, input_dim) normalized feature vector

        Returns
        -------
        dict with 'mean_dt', 'std_dt', 'samples'
        """
        self.model.eval()
        self._enable_dropout()

        samples = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                if self.model_type == 'pinn':
                    out = self.model(x)
                    log_r = out['log_ratio']
                elif self.model_type == 'lstm':
                    log_r, _ = self.model(x)
                else:
                    log_r = self.model(x)

                dt_s = float(dt_ref * torch.exp(log_r).item())
                samples.append(np.clip(dt_s, config.DT_MIN, config.DT_MAX))

        samples = np.array(samples)
        return {
            'mean_dt': float(np.mean(samples)),
            'std_dt':  float(np.std(samples)),
            'ci_lower': float(np.percentile(samples, 5)),
            'ci_upper': float(np.percentile(samples, 95)),
            'samples': samples
        }


# ──────────────────────────────────────────────────────────────────────────
# ADVANCED EXTENSION: Simple RL Agent for Timestep Control
# ──────────────────────────────────────────────────────────────────────────

class RLTimestepAgent:
    """
    Lightweight Reinforcement Learning agent for adaptive timestep control.

    State  : feature vector f(t) ∈ ℝ⁷
    Action : one of N discrete Δt candidates (discretized action space)
    Reward :  +1  if step reduces L2 error
              -1  if step increases L2 error or causes blowup
              -0.1 * (n_steps / n_steps_fixed)  (efficiency penalty)

    Uses tabular Q-learning with function approximation via a shallow MLP.
    """

    def __init__(self,
                 dt_candidates: np.ndarray = None,
                 gamma: float = 0.95,
                 epsilon: float = 0.1,
                 lr: float = 1e-3):

        if dt_candidates is None:
            # Discretize the timestep space logarithmically
            dt_candidates = np.logspace(
                np.log10(config.DT_MIN),
                np.log10(config.DT_MAX),
                num=10
            )
        self.dt_candidates = dt_candidates
        self.n_actions = len(dt_candidates)
        self.gamma   = gamma
        self.epsilon = epsilon

        # Q-network: state → Q-values for each action
        self.q_net = nn.Sequential(
            nn.Linear(config.INPUT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions)
        )
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = []
        self.max_buffer = 5000

    def select_action(self, state: np.ndarray, greedy: bool = False) -> tuple:
        """
        ε-greedy action selection.

        Returns (dt_value, action_index)
        """
        if not greedy and np.random.rand() < self.epsilon:
            idx = np.random.randint(self.n_actions)
        else:
            s = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                q = self.q_net(s)
            idx = int(q.argmax().item())
        return self.dt_candidates[idx], idx

    def store_transition(self, state, action_idx, reward, next_state, done):
        """Add transition to replay buffer."""
        if len(self.replay_buffer) >= self.max_buffer:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state, action_idx, reward, next_state, done))

    def update(self, batch_size: int = 64) -> float:
        """Sample a batch and update Q-network via TD learning."""
        if len(self.replay_buffer) < batch_size:
            return 0.0

        idxs = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in idxs]

        states   = torch.tensor([b[0] for b in batch], dtype=torch.float32)
        actions  = torch.tensor([b[1] for b in batch], dtype=torch.long)
        rewards  = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        n_states = torch.tensor([b[3] for b in batch], dtype=torch.float32)
        dones    = torch.tensor([b[4] for b in batch], dtype=torch.float32)

        q_vals   = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        q_next   = self.q_net(n_states).max(1)[0].detach()
        targets  = rewards + self.gamma * q_next * (1 - dones)

        loss = nn.functional.mse_loss(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

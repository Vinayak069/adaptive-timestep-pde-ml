"""
visualization.py
─────────────────────────────────────────────────────────────────────────────
Publication-quality visualization module.

Plots:
  1. Solution evolution (heatmap snapshots at different times)
  2. Adaptive Δt behavior over time (with CFL reference line)
  3. Comparison: Fixed vs Adaptive stepping (L2 error + step count)
  4. Training loss curves for all three models
  5. CFL history and stability analysis
  6. Residual evolution
  7. Uncertainty bounds (for ensemble predictor)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as ticker
from typing import Optional
import config

# ── Global Plot Style ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Serif',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.labelsize':    11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
    'lines.linewidth':   2.0,
    'figure.dpi':        120,
    'savefig.dpi':       150,
    'savefig.bbox':      'tight',
})

COLORS = {
    'fixed':   '#E63946',   # Red
    'cfl':     '#457B9D',   # Steel blue
    'ml_ff':   '#2EC4B6',   # Teal
    'ml_lstm': '#F4A261',   # Orange
    'ml_pinn': '#A8DADC',   # Light teal
    'exact':   '#264653',   # Dark teal
    'cfl_ref': '#6D6875',   # Mauve
}


def save_fig(fig, name: str, results_dir: str = config.RESULTS_DIR):
    """Save figure to results directory."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [✓] Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────
# 1. Solution Evolution Snapshots
# ──────────────────────────────────────────────────────────────────────────

def plot_solution_evolution(snapshots: list,
                             times: list,
                             title: str = "Solution Evolution",
                             results_dir: str = config.RESULTS_DIR) -> str:
    """
    Plot solution heatmaps at different timesteps.

    Parameters
    ----------
    snapshots : list of 2D arrays (ny, nx)
    times     : list of float time values
    """
    n = min(len(snapshots), 6)
    snap_idx = np.linspace(0, len(snapshots)-1, n, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    vmin = min(s.min() for s in snapshots)
    vmax = max(s.max() for s in snapshots)
    vcen = 0.0 if (vmin < 0 < vmax) else (vmin + vmax) / 2

    for k, idx in enumerate(snap_idx):
        ax = axes[k]
        snap = snapshots[idx]
        t_snap = times[idx] if idx < len(times) else 0.0

        try:
            norm = TwoSlopeNorm(vcenter=vcen, vmin=vmin, vmax=vmax)
            im = ax.imshow(snap, cmap='RdBu_r', norm=norm,
                           origin='lower', aspect='equal',
                           extent=[0, config.LX, 0, config.LY])
        except Exception:
            im = ax.imshow(snap, cmap='viridis',
                           origin='lower', aspect='equal',
                           extent=[0, config.LX, 0, config.LY])

        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"t = {t_snap:.4f}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.grid(False)

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    return save_fig(fig, "solution_evolution", results_dir)


# ──────────────────────────────────────────────────────────────────────────
# 2. Adaptive Timestep Behavior
# ──────────────────────────────────────────────────────────────────────────

def plot_adaptive_dt(logs_dict: dict,
                     dt_cfl_ref: float = config.DT_CFL,
                     dt_fixed:   float = config.DT_FIXED,
                     results_dir: str = config.RESULTS_DIR) -> str:
    """
    Plot Δt(t) for all solver modes side by side.

    Parameters
    ----------
    logs_dict : {'label': SimulationLog.to_arrays()}
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    # ── Top: Raw Δt traces ───────────────────────────────────────────────
    ax = axes[0]
    color_cycle = list(COLORS.values())
    for i, (label, data) in enumerate(logs_dict.items()):
        c = color_cycle[i % len(color_cycle)]
        ax.plot(data['time'], data['dt'],
                label=label, color=c, alpha=0.8, lw=1.5)

    ax.axhline(dt_cfl_ref, color=COLORS['cfl_ref'], ls='--', lw=1.5,
               label=f'CFL Reference Δt = {dt_cfl_ref:.2e}')
    ax.axhline(dt_fixed,   color=COLORS['fixed'],   ls=':', lw=1.5,
               label=f'Fixed Δt = {dt_fixed:.2e}')
    ax.set_ylabel("Timestep Δt (s)")
    ax.set_title("Adaptive Timestep Behavior")
    ax.legend(fontsize=9)
    ax.set_yscale('log')

    # ── Bottom: Cumulative step efficiency ───────────────────────────────
    ax2 = axes[1]
    for i, (label, data) in enumerate(logs_dict.items()):
        c = color_cycle[i % len(color_cycle)]
        cumtime = data['time']
        n_steps = np.arange(1, len(cumtime)+1)
        ax2.plot(cumtime, n_steps, label=label, color=c, lw=1.5)

    # Fixed reference line
    ref_steps = int(config.T_END / dt_fixed)
    ax2.axhline(ref_steps, color=COLORS['fixed'], ls=':', lw=1.2,
                label=f'Fixed Δt ({ref_steps} steps)')
    ax2.set_xlabel("Simulation Time t")
    ax2.set_ylabel("Cumulative Steps")
    ax2.set_title("Step Count Comparison (lower = more efficient)")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    return save_fig(fig, "adaptive_dt", results_dir)


# ──────────────────────────────────────────────────────────────────────────
# 3. Accuracy Comparison (L2 Error)
# ──────────────────────────────────────────────────────────────────────────

def plot_accuracy_comparison(logs_dict: dict,
                              results_dir: str = config.RESULTS_DIR) -> str:
    """
    Plot L2 error vs time for all solver modes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    color_cycle = list(COLORS.values())

    # ── L2 Error ─────────────────────────────────────────────────────────
    ax = axes[0]
    for i, (label, data) in enumerate(logs_dict.items()):
        if data.get('l2_error') is not None and len(data['l2_error']) > 0:
            c = color_cycle[i % len(color_cycle)]
            ax.semilogy(data['time'][:len(data['l2_error'])],
                        data['l2_error'], label=label, color=c, lw=1.8)

    ax.set_xlabel("Time t"); ax.set_ylabel("Relative L2 Error")
    ax.set_title("Solution Accuracy: Relative L2 Error vs Time")
    ax.legend(fontsize=9)

    # ── Residual ──────────────────────────────────────────────────────────
    ax2 = axes[1]
    for i, (label, data) in enumerate(logs_dict.items()):
        if len(data['residual']) > 0:
            c = color_cycle[i % len(color_cycle)]
            ax2.semilogy(data['time'], data['residual'] + 1e-16,
                         label=label, color=c, lw=1.8, alpha=0.85)

    ax2.set_xlabel("Time t"); ax2.set_ylabel("PDE Residual")
    ax2.set_title("PDE Residual Norm")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    return save_fig(fig, "accuracy_comparison", results_dir)


# ──────────────────────────────────────────────────────────────────────────
# 4. Training Loss Curves
# ──────────────────────────────────────────────────────────────────────────

def plot_training_history(histories: dict,
                           results_dir: str = config.RESULTS_DIR) -> str:
    """
    Plot train/val loss curves for all models.

    Parameters
    ----------
    histories : {'model_name': {'train_loss': [...], 'val_loss': [...], 'lr': [...]}}
    """
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]

    color_cycle = list(COLORS.values())

    for ax, (model_name, hist) in zip(axes, histories.items()):
        epochs = range(1, len(hist['train_loss'])+1)
        ax.semilogy(epochs, hist['train_loss'],
                    label='Train', color=color_cycle[0], lw=2)
        ax.semilogy(epochs, hist['val_loss'],
                    label='Validation', color=color_cycle[1], lw=2, ls='--')
        ax.set_title(f"{model_name} — Loss Curve")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (Huber)")
        ax.legend()

    plt.suptitle("ML Model Training Histories", fontsize=13, fontweight='bold')
    plt.tight_layout()
    return save_fig(fig, "training_history", results_dir)


# ──────────────────────────────────────────────────────────────────────────
# 5. Summary Bar Chart (Metrics Comparison)
# ──────────────────────────────────────────────────────────────────────────

def plot_summary_metrics(metrics: dict,
                          results_dir: str = config.RESULTS_DIR) -> str:
    """
    Bar chart comparing key metrics across all solver modes.

    Parameters
    ----------
    metrics : {
        'method_name': {
          'mean_l2':   float,
          'n_steps':   int,
          'speedup':   float,
          'blowup':    bool
        }
    }
    """
    methods = list(metrics.keys())
    n = len(methods)
    color_cycle = list(COLORS.values())

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # L2 error
    vals = [metrics[m].get('mean_l2', 0) for m in methods]
    axes[0].bar(methods, vals,
                color=[color_cycle[i % len(color_cycle)] for i in range(n)],
                edgecolor='black', linewidth=0.8)
    axes[0].set_title("Mean Relative L2 Error\n(lower = better)")
    axes[0].set_ylabel("L2 Error")
    axes[0].tick_params(axis='x', rotation=20)

    # Step count
    vals = [metrics[m].get('n_steps', 0) for m in methods]
    axes[1].bar(methods, vals,
                color=[color_cycle[i % len(color_cycle)] for i in range(n)],
                edgecolor='black', linewidth=0.8)
    axes[1].set_title("Total Step Count\n(lower = more efficient)")
    axes[1].set_ylabel("# Steps")
    axes[1].tick_params(axis='x', rotation=20)

    # Speedup
    vals = [metrics[m].get('speedup', 1.0) for m in methods]
    bars = axes[2].bar(methods, vals,
                       color=[color_cycle[i % len(color_cycle)] for i in range(n)],
                       edgecolor='black', linewidth=0.8)
    axes[2].axhline(1.0, color='gray', ls='--', lw=1.2, label='Baseline (×1)')
    axes[2].set_title("Speedup vs Fixed Δt\n(higher = faster)")
    axes[2].set_ylabel("Speedup (×)")
    axes[2].tick_params(axis='x', rotation=20)
    axes[2].legend()

    # Mark blowup
    for ax in axes:
        for i, m in enumerate(methods):
            if metrics[m].get('blowup', False):
                ax.patches[i].set_hatch('//')
                ax.patches[i].set_facecolor('#FF9999')

    plt.suptitle("Performance Summary: Fixed vs Adaptive Timestep",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return save_fig(fig, "summary_metrics", results_dir)


# ──────────────────────────────────────────────────────────────────────────
# 6. CFL Number History
# ──────────────────────────────────────────────────────────────────────────

def plot_cfl_history(logs_dict: dict,
                     results_dir: str = config.RESULTS_DIR) -> str:
    """Plot CFL number evolution for all adaptive solvers."""
    fig, ax = plt.subplots(figsize=(10, 4))
    color_cycle = list(COLORS.values())

    for i, (label, data) in enumerate(logs_dict.items()):
        c = color_cycle[i % len(color_cycle)]
        ax.plot(data['time'], data['cfl'],
                label=label, color=c, lw=1.5, alpha=0.8)

    ax.axhline(1.0, color='red', ls='--', lw=1.5, label='CFL = 1 (stability limit)')
    ax.axhline(config.CFL_TARGET, color='green', ls=':', lw=1.5,
               label=f'CFL target = {config.CFL_TARGET}')
    ax.set_xlabel("Time t")
    ax.set_ylabel("CFL Number")
    ax.set_title("CFL Number History — Stability Verification")
    ax.legend(fontsize=9)
    ax.set_ylim(0, None)

    plt.tight_layout()
    return save_fig(fig, "cfl_history", results_dir)


# ──────────────────────────────────────────────────────────────────────────
# 7. Uncertainty Bounds (Ensemble Predictor)
# ──────────────────────────────────────────────────────────────────────────

def plot_uncertainty(times: np.ndarray,
                     dt_mean: np.ndarray,
                     dt_lower: np.ndarray,
                     dt_upper: np.ndarray,
                     dt_fixed: float = config.DT_FIXED,
                     results_dir: str = config.RESULTS_DIR) -> str:
    """
    Plot predicted Δt with uncertainty bounds from ensemble predictor.
    """
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(times, dt_lower, dt_upper,
                    alpha=0.25, color=COLORS['ml_lstm'],
                    label='90% CI (MC Dropout)')
    ax.plot(times, dt_mean, color=COLORS['ml_lstm'], lw=2,
            label='Mean Δt (Ensemble)')
    ax.axhline(dt_fixed, color=COLORS['fixed'], ls='--', lw=1.5,
               label=f'Fixed Δt = {dt_fixed:.2e}')
    ax.set_xlabel("Time t")
    ax.set_ylabel("Timestep Δt")
    ax.set_yscale('log')
    ax.set_title("Timestep Prediction with Uncertainty Bounds (MC Dropout)")
    ax.legend()
    plt.tight_layout()
    return save_fig(fig, "uncertainty_bounds", results_dir)

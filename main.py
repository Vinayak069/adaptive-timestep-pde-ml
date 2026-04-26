"""
main.py
─────────────────────────────────────────────────────────────────────────────
Master Orchestration Script — AI-Assisted Adaptive Time-Stepping

Full pipeline:
  Phase 0  — Setup / reproducibility
  Phase 1  — Data generation (ADI + MacCormack)
  Phase 2  — Train all ML models (FeedforwardNet, LSTM, PINN)
  Phase 3  — Run comparative simulations
  Phase 4  — Generate all plots + metrics report
  Phase 5  — Print summary table

Run: python main.py [--pde {adi|maccormack|both}] [--no-train]
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

import config
from data_generation  import generate_dataset, build_lstm_sequences
from ml.models        import FeedforwardNet, LSTMPredictor, PINNPredictor, get_model
from ml.trainer       import train_model
from ml.feature_extractor import normalize_features, compute_optimal_dt
from hybrid_solver    import HybridSolver, EnsemblePredictor
from solvers          import ADISolver, MacCormackSolver
from visualization    import (plot_solution_evolution, plot_adaptive_dt,
                               plot_accuracy_comparison, plot_training_history,
                               plot_summary_metrics, plot_cfl_history,
                               plot_uncertainty)


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="AI-Assisted Adaptive Time-Stepping for PDEs")
    p.add_argument('--pde',      choices=['adi','maccormack','both'],
                   default='both',   help='Which PDE type to run')
    p.add_argument('--no-train', action='store_true',
                   help='Skip training and load saved models')
    p.add_argument('--n-sims',   type=int, default=config.N_SIMULATIONS,
                   help='Number of training simulations')
    p.add_argument('--epochs',   type=int, default=config.EPOCHS,
                   help='Training epochs per model')
    p.add_argument('--device',   default='cpu',
                   help='PyTorch device (cpu|cuda)')
    p.add_argument('--quick',    action='store_true',
                   help='Quick mode: fewer sims and epochs (for testing)')
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────
# Exact Solutions (for error computation)
# ──────────────────────────────────────────────────────────────────────────

def heat_exact(t: float) -> np.ndarray:
    """
    Exact solution to 2D heat equation with homogeneous Dirichlet BC:
    u(x,y,0) = sin(πx)sin(πy)

    u(x,y,t) = exp(-2α π² t) sin(πx) sin(πy)
    """
    x = np.linspace(0, config.LX, config.NX)
    y = np.linspace(0, config.LY, config.NY)
    X, Y = np.meshgrid(x, y)
    return np.exp(-2 * config.ALPHA * np.pi**2 * t) * np.sin(np.pi*X) * np.sin(np.pi*Y)


def advection_exact(t: float) -> np.ndarray:
    """
    Exact solution to 2D periodic advection (Gaussian pulse):
    u(x,y,t) = u₀(x - cx*t, y - cy*t)  (mod 1, periodic)
    """
    from data_generation import ic_gaussian
    x = np.linspace(0, 1, config.NX)
    y = np.linspace(0, 1, config.NY)
    X, Y = np.meshgrid(x, y)
    # Shift center, wrap periodically
    cx_t = (0.5 + config.CX * t) % 1.0
    cy_t = (0.5 + config.CY * t) % 1.0
    sigma = 0.08
    dx_wrap = (X - cx_t + 0.5) % 1.0 - 0.5
    dy_wrap = (Y - cy_t + 0.5) % 1.0 - 0.5
    return np.exp(-(dx_wrap**2 + dy_wrap**2) / (2 * sigma**2))


# ──────────────────────────────────────────────────────────────────────────
# Phase 1: Data Generation
# ──────────────────────────────────────────────────────────────────────────

def run_data_generation(pde_type: str, n_sims: int) -> tuple:
    """Generate or load dataset for given PDE type."""
    X_path = os.path.join(config.DATA_DIR, f'X_{pde_type}.npy')
    y_path = os.path.join(config.DATA_DIR, f'y_{pde_type}.npy')

    if os.path.exists(X_path) and os.path.exists(y_path):
        print(f"\n[✓] Loading cached {pde_type} dataset...")
        X = np.load(X_path)
        y = np.load(y_path)
        print(f"    {X.shape[0]} samples loaded.")
    else:
        X, y = generate_dataset(pde_type=pde_type, n_sims=n_sims)

    return X, y


# ──────────────────────────────────────────────────────────────────────────
# Phase 2: Train All Models
# ──────────────────────────────────────────────────────────────────────────

def run_training(X: np.ndarray, y: np.ndarray,
                 pde_type: str, epochs: int, device: str,
                 skip: bool = False) -> tuple:
    """
    Train FeedforwardNet, LSTMPredictor, PINNPredictor.

    Returns
    -------
    (models_dict, histories_dict, feature_stats)
    """
    # Normalize features
    X_norm, stats = normalize_features(X)
    np.save(os.path.join(config.DATA_DIR, f'stats_{pde_type}.npy'),
            stats, allow_pickle=True)

    models    = {}
    histories = {}

    # ── Feedforward Net ──────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Training FeedforwardNet [{pde_type}]")
    print(f"{'='*55}")
    ff_path = os.path.join(config.MODEL_DIR, f'ff_{pde_type}.pt')

    ff_model = FeedforwardNet()
    if skip and os.path.exists(ff_path):
        from ml.trainer import load_model
        hist = load_model(ff_model, ff_path)
        print(f"  [Loaded from {ff_path}]")
    else:
        hist = train_model(ff_model, X_norm, y,
                           model_type='feedforward',
                           epochs=epochs, device=device,
                           save_path=ff_path)
    models['feedforward']    = ff_model
    histories['Feedforward'] = hist

    # ── LSTM ─────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Training LSTMPredictor [{pde_type}]")
    print(f"{'='*55}")
    lstm_path = os.path.join(config.MODEL_DIR, f'lstm_{pde_type}.pt')

    X_seq, y_seq = build_lstm_sequences(X_norm, y, seq_len=config.SEQ_LEN)
    lstm_model   = LSTMPredictor()
    if skip and os.path.exists(lstm_path):
        from ml.trainer import load_model
        hist = load_model(lstm_model, lstm_path)
        print(f"  [Loaded from {lstm_path}]")
    else:
        hist = train_model(lstm_model, X_seq, y_seq,
                           model_type='lstm',
                           epochs=epochs, device=device,
                           save_path=lstm_path)
    models['lstm']    = lstm_model
    histories['LSTM'] = hist

    # ── PINN ─────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Training PINNPredictor [{pde_type}]")
    print(f"{'='*55}")
    pinn_path = os.path.join(config.MODEL_DIR, f'pinn_{pde_type}.pt')

    pinn_model = PINNPredictor()
    if skip and os.path.exists(pinn_path):
        from ml.trainer import load_model
        hist = load_model(pinn_model, pinn_path)
        print(f"  [Loaded from {pinn_path}]")
    else:
        hist = train_model(pinn_model, X_norm, y,
                           model_type='pinn',
                           epochs=epochs, device=device,
                           save_path=pinn_path)
    models['pinn']    = pinn_model
    histories['PINN'] = hist

    return models, histories, stats


# ──────────────────────────────────────────────────────────────────────────
# Phase 3: Comparative Simulations
# ──────────────────────────────────────────────────────────────────────────

def run_simulations(pde_type: str,
                    models: dict,
                    stats: dict,
                    quick: bool = False) -> dict:
    """
    Run all solver modes and collect SimulationLogs.

    Modes tested:
      1. Fixed Δt (baseline)
      2. CFL-adaptive (no ML)
      3. ML-Feedforward
      4. ML-LSTM
      5. ML-PINN
      6. Hybrid (dynamic ADI ↔ MacCormack)
    """
    t_end = config.T_END if not quick else 0.1

    # ── Initial Condition ────────────────────────────────────────────────
    if pde_type == 'adi':
        u0 = heat_exact(0.0)          # IC: sin(πx)sin(πy)
        exact_fn = heat_exact
    else:
        from data_generation import ic_gaussian
        u0 = ic_gaussian(config.NX, config.NY,
                         config.DX, config.DY,
                         cx=0.5, cy=0.5, sigma=0.08)
        exact_fn = advection_exact

    print(f"\n{'='*55}")
    print(f"  Running Comparative Simulations [{pde_type.upper()}]")
    print(f"{'='*55}")

    all_logs = {}

    # 1. Fixed Δt (Baseline)
    print("\n[1/6] Fixed Δt solver...")
    solver_fixed = HybridSolver(solver_type=pde_type, use_ml=False)
    log_fixed = solver_fixed.run(u0, t_end=t_end,
                                  dt_fixed=config.DT_FIXED,
                                  u_exact_fn=exact_fn,
                                  verbose=True)
    all_logs['Fixed Δt'] = log_fixed.to_arrays()
    all_logs['Fixed Δt']['snapshots'] = log_fixed.u_snapshots
    all_logs['Fixed Δt']['blowup']    = log_fixed.blowup

    # 2. CFL-Adaptive (no ML)
    print("\n[2/6] CFL-Adaptive solver...")
    solver_cfl = HybridSolver(solver_type=pde_type, use_ml=False)
    log_cfl = solver_cfl.run(u0, t_end=t_end,
                              u_exact_fn=exact_fn, verbose=True)
    all_logs['CFL-Adaptive'] = log_cfl.to_arrays()
    all_logs['CFL-Adaptive']['blowup'] = log_cfl.blowup

    # 3. ML-Feedforward
    print("\n[3/6] ML-Feedforward solver...")
    solver_ff = HybridSolver(solver_type=pde_type,
                              model=models['feedforward'],
                              model_type='feedforward',
                              feature_stats=stats,
                              use_ml=True)
    log_ff = solver_ff.run(u0, t_end=t_end,
                            u_exact_fn=exact_fn, verbose=True)
    all_logs['ML-Feedforward'] = log_ff.to_arrays()
    all_logs['ML-Feedforward']['blowup'] = log_ff.blowup

    # 4. ML-LSTM
    print("\n[4/6] ML-LSTM solver...")
    solver_lstm = HybridSolver(solver_type=pde_type,
                                model=models['lstm'],
                                model_type='lstm',
                                feature_stats=stats,
                                use_ml=True)
    log_lstm = solver_lstm.run(u0, t_end=t_end,
                                u_exact_fn=exact_fn, verbose=True)
    all_logs['ML-LSTM'] = log_lstm.to_arrays()
    all_logs['ML-LSTM']['blowup'] = log_lstm.blowup

    # 5. ML-PINN
    print("\n[5/6] ML-PINN solver...")
    solver_pinn = HybridSolver(solver_type=pde_type,
                                model=models['pinn'],
                                model_type='pinn',
                                feature_stats=stats,
                                use_ml=True)
    log_pinn = solver_pinn.run(u0, t_end=t_end,
                                u_exact_fn=exact_fn, verbose=True)
    all_logs['ML-PINN'] = log_pinn.to_arrays()
    all_logs['ML-PINN']['blowup'] = log_pinn.blowup

    # 6. Hybrid Solver (dynamic switching)
    print("\n[6/6] Dynamic Hybrid solver...")
    solver_hybrid = HybridSolver(solver_type='hybrid',
                                  model=models['feedforward'],
                                  model_type='feedforward',
                                  feature_stats=stats,
                                  use_ml=True)
    log_hybrid = solver_hybrid.run(u0, t_end=t_end,
                                    u_exact_fn=exact_fn, verbose=True)
    all_logs['Hybrid (Switch)'] = log_hybrid.to_arrays()
    all_logs['Hybrid (Switch)']['blowup'] = log_hybrid.blowup

    return all_logs, u0


# ──────────────────────────────────────────────────────────────────────────
# Phase 4: Visualization & Metrics
# ──────────────────────────────────────────────────────────────────────────

def generate_visualizations(all_logs: dict, histories: dict,
                             pde_type: str, u0: np.ndarray):
    """Generate all plots."""
    res_dir = os.path.join(config.RESULTS_DIR, pde_type)
    os.makedirs(res_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Generating Visualizations [{pde_type.upper()}]")
    print(f"{'='*55}")

    # Solution snapshots
    ref_log = all_logs.get('ML-Feedforward') or all_logs.get('Fixed Δt')
    if 'snapshots' in ref_log and ref_log['snapshots']:
        plot_solution_evolution(
            ref_log['snapshots'], ref_log['time'],
            title=f"{pde_type.upper()} Solution Evolution (ML-Adaptive)",
            results_dir=res_dir)

    # Adaptive Δt behavior
    logs_for_dt = {k: v for k, v in all_logs.items()}
    plot_adaptive_dt(logs_for_dt, results_dir=res_dir)

    # Accuracy comparison
    plot_accuracy_comparison(all_logs, results_dir=res_dir)

    # Training histories
    plot_training_history(histories, results_dir=res_dir)

    # CFL history
    plot_cfl_history(all_logs, results_dir=res_dir)


def compute_summary_metrics(all_logs: dict) -> dict:
    """Compute per-method performance metrics."""
    ref_steps = int(config.T_END / config.DT_FIXED)
    metrics = {}

    for method, data in all_logs.items():
        n_steps = len(data['dt'])
        l2_arr  = data.get('l2_error')
        mean_l2 = float(np.mean(l2_arr)) if l2_arr is not None and len(l2_arr) > 0 else float('nan')
        speedup = ref_steps / max(n_steps, 1)

        metrics[method] = {
            'mean_l2':  mean_l2,
            'n_steps':  n_steps,
            'speedup':  speedup,
            'mean_dt':  float(np.mean(data['dt'])),
            'blowup':   data.get('blowup', False),
        }

    return metrics


def print_summary_table(metrics: dict, pde_type: str):
    """Print formatted summary table."""
    print(f"\n{'='*75}")
    print(f"  RESULTS SUMMARY — {pde_type.upper()}")
    print(f"{'='*75}")
    print(f"  {'Method':<22} {'Steps':>8} {'Speedup':>9} {'Mean L2':>12} {'Mean Δt':>12} {'Blowup':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*9} {'-'*12} {'-'*12} {'-'*8}")

    for method, m in metrics.items():
        blowup_str = '⚠️  YES' if m['blowup'] else '  No'
        l2_str = f"{m['mean_l2']:.4e}" if not np.isnan(m['mean_l2']) else '    N/A'
        print(f"  {method:<22} {m['n_steps']:>8d} {m['speedup']:>8.2f}x "
              f"{l2_str:>12}  {m['mean_dt']:>10.2e}  {blowup_str}")

    print(f"{'='*75}")


# ──────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    os.makedirs(config.DATA_DIR,    exist_ok=True)
    os.makedirs(config.MODEL_DIR,   exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Quick mode overrides
    if args.quick:
        n_sims  = 5
        epochs  = 30
        t_short = 0.05
        print("\n[Quick Mode] Using reduced settings for fast testing.")
    else:
        n_sims = args.n_sims
        epochs = args.epochs

    pde_types = ['adi', 'maccormack'] if args.pde == 'both' else [args.pde]

    all_results = {}

    for pde_type in pde_types:
        print(f"\n{'#'*65}")
        print(f"#  PDE TYPE: {pde_type.upper()}")
        print(f"{'#'*65}")

        # ── Phase 1: Data ────────────────────────────────────────────────
        t0 = time.time()
        X, y = run_data_generation(pde_type, n_sims)
        print(f"  Data generation: {time.time()-t0:.1f}s")

        # ── Phase 2: Train ───────────────────────────────────────────────
        t0 = time.time()
        models, histories, stats = run_training(
            X, y, pde_type, epochs,
            device=args.device,
            skip=args.no_train
        )
        print(f"\n  Training: {time.time()-t0:.1f}s")

        # ── Phase 3: Simulate ────────────────────────────────────────────
        t0 = time.time()
        all_logs, u0 = run_simulations(
            pde_type, models, stats, quick=args.quick)
        print(f"\n  Simulations: {time.time()-t0:.1f}s")

        # ── Phase 4: Visualize ───────────────────────────────────────────
        generate_visualizations(all_logs, histories, pde_type, u0)

        # ── Phase 5: Summary ─────────────────────────────────────────────
        metrics = compute_summary_metrics(all_logs)
        print_summary_table(metrics, pde_type)

        res_dir = os.path.join(config.RESULTS_DIR, pde_type)
        plot_summary_metrics(metrics, results_dir=res_dir)

        all_results[pde_type] = metrics

    print(f"\n{'='*55}")
    print(f"  ✅  Pipeline Complete!")
    print(f"  📁  Results saved to: {config.RESULTS_DIR}/")
    print(f"{'='*55}\n")

    return all_results


if __name__ == '__main__':
    main()

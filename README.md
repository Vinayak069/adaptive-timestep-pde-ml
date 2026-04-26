# AI-Assisted Adaptive Time-Stepping for Numerical PDEs

> **B.Tech Term Project** — Hybrid AI + Classical Numerical Methods for adaptive timestep control in PDE solvers.

---

## 🎯 Project Overview

This project implements a **hybrid AI + numerical solver** framework that:

- Solves PDEs using classical methods:
  - **ADI (Alternating Direction Implicit)** — 2D Heat Equation
  - **MacCormack Predictor-Corrector** — 2D Advection Equation
- Uses **machine learning** to dynamically predict optimal timestep Δt
- Achieves **22–25× speedup** vs fixed timestep with comparable accuracy

---

## 📐 Mathematical Background

### ADI Method (Peaceman-Rachford)
Solves: `∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)`

Splits into two implicit half-steps:
- Step 1 (implicit in x): `(I - α Δt/2 Aₓ) u* = (I + α Δt/2 Aᵧ) uⁿ`
- Step 2 (implicit in y): `(I - α Δt/2 Aᵧ) u^{n+1} = (I + α Δt/2 Aₓ) u*`

**Properties:** O(Δx², Δy², Δt²), unconditionally stable, O(N) Thomas algorithm

### MacCormack Method
Solves: `∂u/∂t + cₓ∂u/∂x + cᵧ∂u/∂y = 0`

- **Predictor:** `ũ = uⁿ - cₓ(Δt/Δx)(uⁿᵢ₊₁ - uⁿᵢ) - cᵧ(Δt/Δy)(uⁿⱼ₊₁ - uⁿⱼ)`
- **Corrector:** `u^{n+1} = ½(uⁿ + ũ - cₓ(Δt/Δx)(ũᵢ - ũᵢ₋₁) - cᵧ(Δt/Δy)(ũⱼ - ũⱼ₋₁))`

**Properties:** O(Δx², Δy², Δt²), explicit, requires CFL ≤ 1

---

## 🤖 ML Timestep Predictor

### Feature Vector (dim = 7)
| Feature | Description |
|---------|-------------|
| `mean_grad_x` | Mean \|∂u/∂x\| over domain |
| `mean_grad_y` | Mean \|∂u/∂y\| over domain |
| `max_grad` | Maximum gradient magnitude |
| `u_rms` | RMS of solution (energy measure) |
| `residual` | PDE residual estimate |
| `dt_prev` | Previous timestep (temporal memory) |
| `cfl_est` | Estimated CFL number |

### Model Architectures

| Model | Architecture | Specialty |
|-------|-------------|-----------|
| **FeedforwardNet** | Residual MLP (128→256→128→64) | Fast inference, baseline |
| **LSTMPredictor** | LSTM×2 + Attention | Temporal trends in solution |
| **PINNPredictor** | MLP + CFL penalty head | Physics-consistent predictions |

All models predict `log(Δt/Δt_ref)` → recovered as `Δt = Δt_ref × exp(output)`

---

## 🏗️ Project Structure

```
.
├── config.py               # Global parameters
├── main.py                 # Master pipeline
├── data_generation.py      # Simulation data generator
├── hybrid_solver.py        # AI+PDE solver + RL + Ensemble
├── visualization.py        # Plot generation
├── requirements.txt
├── solvers/
│   ├── adi_solver.py       # Peaceman-Rachford ADI
│   └── maccormack_solver.py # MacCormack scheme
└── ml/
    ├── models.py           # Neural network models
    ├── feature_extractor.py # State → feature vector
    └── trainer.py          # Training pipeline
```

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Full run (both PDEs, all models)
MPLBACKEND=Agg python3 main.py

# Quick test (5 mins)
MPLBACKEND=Agg python3 main.py --quick --pde maccormack

# ADI only
MPLBACKEND=Agg python3 main.py --pde adi

# Skip retraining
MPLBACKEND=Agg python3 main.py --no-train
```

---

## 📊 Results

| Method | Steps | Speedup | Mean L2 Error |
|--------|-------|---------|---------------|
| Fixed Δt (baseline) | 100 | 1× | 3.28e-3 |
| CFL-Adaptive | 20 | **25×** | 3.32e-3 |
| ML-Feedforward | 22 | 22.7× | 3.45e-3 |
| ML-LSTM | 21 | **23.8×** | 3.41e-3 |
| ML-PINN | 22 | 22.7× | 3.45e-3 |

> Grid: 64×64, t_end=0.1, MacCormack solver

---

## ⚡ Advanced Extensions

1. **Ensemble Uncertainty** (`EnsemblePredictor`) — MC Dropout gives 90% confidence intervals on Δt
2. **RL Timestep Control** (`RLTimestepAgent`) — Q-learning with 10 discrete Δt actions
3. **Dynamic Hybrid Switching** — Automatically selects ADI vs MacCormack based on local Péclet number

---

## 📈 Generated Plots

| Plot | Description |
|------|-------------|
| `solution_evolution.png` | 6-panel heatmap snapshots |
| `adaptive_dt.png` | Δt(t) traces + cumulative step comparison |
| `accuracy_comparison.png` | L2 error & PDE residual vs time |
| `training_history.png` | Train/val loss for all 3 models |
| `cfl_history.png` | CFL number stability verification |
| `summary_metrics.png` | Steps, speedup, L2 error bar charts |

---

## 🛠️ Configuration

Key parameters in `config.py`:

```python
NX, NY      = 64, 64      # Grid resolution
ALPHA       = 0.01         # Thermal diffusivity
CX, CY      = 0.5, 0.3    # Advection velocity
CFL_TARGET  = 0.5          # Target CFL number
DT_FIXED    = 1e-3         # Baseline fixed timestep
T_END       = 0.5          # Final simulation time
EPOCHS      = 200           # Training epochs
```

---

## 📚 References

1. Peaceman, D.W. & Rachford, H.H. (1955). *The numerical solution of parabolic and elliptic differential equations.* SIAM J. Appl. Math.
2. MacCormack, R.W. (1969). *The effect of viscosity in hypervelocity impact cratering.*
3. Raissi, M. et al. (2019). *Physics-informed neural networks.* J. Comput. Phys.
4. Kochkov, D. et al. (2021). *Machine learning accelerated computational fluid dynamics.* PNAS.

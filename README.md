# Adaptive Time-Stepping for 2D PDEs with ML Timestep Prediction

B.Tech term project. Two classical finite-difference solvers for 2D PDEs, plus
small neural networks that predict the timestep Δt from features of the current
solution. The predicted Δt is compared against a fixed timestep and against a
classical CFL-based adaptive controller.

- **ADI (Peaceman–Rachford)** for the 2D heat equation
- **MacCormack predictor–corrector** for 2D linear advection
- **Δt predictors:** feedforward MLP, LSTM, and an MLP with an auxiliary CFL head ("PINN")

---

## Mathematical background

### ADI method (Peaceman–Rachford)
Solves: `∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)` with homogeneous Dirichlet boundaries.

Splits each step into two implicit sweeps:
- Step 1 (implicit in x): `(I - α Δt/2 Aₓ) u* = (I + α Δt/2 Aᵧ) uⁿ`
- Step 2 (implicit in y): `(I - α Δt/2 Aᵧ) u^{n+1} = (I + α Δt/2 Aₓ) u*`

**Properties:** O(Δx², Δy², Δt²), unconditionally stable, each sweep is a set
of tridiagonal systems solved with the Thomas algorithm in O(N).

### MacCormack method
Solves: `∂u/∂t + cₓ∂u/∂x + cᵧ∂u/∂y = 0` with periodic boundaries.

- **Predictor:** `ũ = uⁿ - cₓ(Δt/Δx)(uⁿᵢ₊₁ - uⁿᵢ) - cᵧ(Δt/Δy)(uⁿⱼ₊₁ - uⁿⱼ)`
- **Corrector:** `u^{n+1} = ½(uⁿ + ũ - cₓ(Δt/Δx)(ũᵢ - ũᵢ₋₁) - cᵧ(Δt/Δy)(ũⱼ - ũⱼ₋₁))`

The forward/backward direction alternates every step to avoid directional bias.

**Properties:** O(Δx², Δy², Δt²), explicit, stable for
`CFL = Δt(|cₓ|/Δx + |cᵧ|/Δy) ≤ 1`.

---

## ML timestep predictor

### Feature vector (dim = 7)
| Feature | Description |
|---------|-------------|
| `mean_grad_x` | Mean \|∂u/∂x\| over the domain |
| `mean_grad_y` | Mean \|∂u/∂y\| over the domain |
| `max_grad` | Maximum gradient magnitude |
| `u_rms` | RMS of the solution |
| `residual` | PDE residual estimate from the last step |
| `dt_prev` | Previous timestep |
| `cfl_est` | Gradient-weighted CFL estimate |

### Training target
Training data comes from fixed-Δt simulations over five initial conditions
(Gaussian, double Gaussian, sinusoid, step, ring) with randomly perturbed α or
(cₓ, cᵧ). The regression target is `log(Δt_opt / Δt_ref)` where `Δt_opt` is the
rule-based value from `compute_optimal_dt` (CFL limit for MacCormack, a
diffusion-number bound for ADI, halved near steep gradients) and
`Δt_ref = config.DT_CFL`. At inference `Δt = Δt_ref · exp(output)`, then
multiplied by a 0.95 safety factor and clamped to `[DT_MIN, DT_MAX]`.

So the networks learn to imitate the classical rule; they are not trained on
accuracy or cost directly.

### Models
| Model | Architecture |
|-------|-------------|
| `FeedforwardNet` | Residual MLP (128→256→128→64) |
| `LSTMPredictor` | 2-layer LSTM + attention over the last 10 feature vectors |
| `PINNPredictor` | MLP with an extra CFL head and a penalty on CFL > target |

---

## Project structure

```
.
├── config.py               # Grid, physics, Δt bounds, training settings
├── main.py                 # Pipeline: data → training → comparison runs → plots
├── data_generation.py      # Training data from fixed-Δt simulations
├── hybrid_solver.py        # Adaptive time loop (fixed / CFL / ML Δt)
├── visualization.py        # Plots
├── solvers/
│   ├── adi_solver.py       # Peaceman–Rachford ADI
│   └── maccormack_solver.py
├── ml/
│   ├── models.py
│   ├── feature_extractor.py
│   └── trainer.py
├── tests/                  # pytest suite
└── docs/results/           # Plots and tables from the run reported below
```

---

## How to run

```bash
pip install -r requirements.txt

# Full run (both PDEs, t_end = 0.5, 50 training simulations, up to 200 epochs)
MPLBACKEND=Agg python3 main.py

# Quick run (t_end = 0.1, 5 simulations, 30 epochs)
MPLBACKEND=Agg python3 main.py --quick --pde maccormack

# One PDE only / reuse saved models
MPLBACKEND=Agg python3 main.py --pde adi
MPLBACKEND=Agg python3 main.py --no-train
```

Generated data, models and plots go to `data/`, `models/` and `results/`
(all git-ignored). A cached dataset in `data/` is reused on the next run, so
delete `data/` and `models/` when switching between `--quick` and a full run.

## How to run tests

```bash
pip install -r requirements-dev.txt
python3 -m pytest -v
```

The tests (about 5 s on a CPU) check:
- ADI against the analytic heat-equation solution
  `u = exp(-2απ²t) sin(πx) sin(πy)`, that Dirichlet boundaries stay zero, and
  that the temporal error falls by ~4× when Δt is halved (second order);
- MacCormack mass conservation and agreement with the exactly shifted Gaussian
  at CFL 0.5 and 0.9;
- that the feature vector has length `INPUT_DIM` with finite values, that the
  rule-based Δt respects `[DT_MIN, DT_MAX]`, and that speedup is computed
  against the fixed-Δt run actually performed.

The same tests run on GitHub Actions (`.github/workflows/tests.yml`).

---

## Results

Measured with `MPLBACKEND=Agg python3 main.py` (fresh `data/` and `models/`,
64×64 grid, t_end = 0.5, CPU). Plots and the raw CSV tables are in
[`docs/results/`](docs/results/results.md). Step counts and errors are
deterministic for a given seed; wall times are from a single run and will vary
by machine.

"Mean L2" is the relative L2 error against the analytic solution, averaged
over all steps. Speedup is the step-count ratio against `Fixed Δt = 1e-3`.

**MacCormack, 2D advection of a Gaussian pulse**

| Method | Steps | Speedup vs Δt=1e-3 | Mean L2 | Wall [s] |
|--------|------:|------:|--------:|------:|
| Fixed Δt = 1.0e-3 | 500 | 1.00× | 1.625e-2 | 0.218 |
| Fixed Δt = 9.9e-3 (CFL 0.5) | 51 | 9.80× | 1.487e-2 | 0.022 |
| CFL-adaptive (no ML) | 100 | 5.00× | 1.591e-2 | 0.051 |
| ML-Feedforward | 106 | 4.72× | 1.606e-2 | 0.114 |
| ML-LSTM | 106 | 4.72× | 1.609e-2 | 0.215 |
| ML-PINN | 106 | 4.72× | 1.602e-2 | 0.113 |

**ADI, 2D heat equation, u₀ = sin(πx) sin(πy)**

| Method | Steps | Speedup vs Δt=1e-3 | Mean L2 | Wall [s] |
|--------|------:|------:|--------:|------:|
| Fixed Δt = 1.0e-3 | 500 | 1.00× | 1.025e-5 | 5.037 |
| Fixed Δt = 5.0e-3 (= DT_MAX) | 100 | 5.00× | 1.033e-5 | 1.000 |
| CFL-adaptive (no ML) | 100 | 5.00× | 1.033e-5 | 0.972 |
| ML-Feedforward | 106 | 4.72× | 1.039e-5 | 1.184 |
| ML-LSTM | 118 | 4.24× | 1.056e-5 | 1.428 |
| ML-PINN | 106 | 4.72× | 1.039e-5 | 1.160 |

### What the numbers show

- Against the small `DT_FIXED = 1e-3` baseline, the adaptive controllers take
  about 5× fewer steps at the same accuracy. That baseline is 10× below the
  MacCormack CFL-0.5 limit (9.9e-3), so this ratio mostly reflects the choice
  of baseline.
- The ML controllers do not beat the classical CFL-adaptive rule: they take
  106–118 steps against 100 and are slower in wall time, because each step also
  runs a network. This is expected: the networks are trained to imitate that
  rule, and for these smooth problems every adaptive controller sits at or just below the
  `DT_MAX = 5e-3` cap (the ML runs at 0.95 × DT_MAX because of the safety
  factor).
- For MacCormack, a plain fixed step at CFL 0.5 needs about half the steps of
  any adaptive controller here, with slightly lower error, because `DT_MAX` is
  below the CFL limit.
- ADI is unconditionally stable and its error on this problem is dominated by
  the spatial grid (about 1e-5 for every Δt tested), so larger steps cost
  essentially nothing in accuracy.

### Limitations

- `Hybrid (Switch)` is also run by `main.py`, but it decides between ADI and
  MacCormack from a global Péclet number `Pe = |c|·Δx/α`, which is ≈ 0.79 for
  the default config. It therefore always picks ADI: on the heat problem it
  matches ML-Feedforward, and on the advection problem it solves the wrong
  equation (mean L2 = 0.78). It is left in the code but should not be read as
  a result.
- The PINN's CFL penalty acts on a separate output head, not on the predicted
  Δt, so in practice it behaves like the feedforward model.
- `hybrid_solver.py` also contains an MC-dropout `EnsemblePredictor` and an
  `RLTimestepAgent` class; neither is trained or used by the pipeline.
- Only one initial condition per PDE is evaluated, on a single 64×64 grid.

---

## Configuration

Key parameters in `config.py`:

```python
NX, NY      = 64, 64      # Grid resolution
ALPHA       = 0.01        # Thermal diffusivity
CX, CY      = 0.5, 0.3    # Advection velocity
CFL_TARGET  = 0.5         # Target CFL number
DT_FIXED    = 1e-3        # Baseline fixed timestep
DT_MIN, DT_MAX = 1e-5, 5e-3   # Clamp applied to every adaptive Δt
T_END       = 0.5         # Final simulation time (0.1 with --quick)
EPOCHS      = 200         # Training epochs (early stopping)
```

---

## References

1. Peaceman, D.W. & Rachford, H.H. (1955). *The numerical solution of parabolic and elliptic differential equations.* SIAM J. Appl. Math.
2. MacCormack, R.W. (1969). *The effect of viscosity in hypervelocity impact cratering.*
3. Raissi, M. et al. (2019). *Physics-informed neural networks.* J. Comput. Phys.
4. Kochkov, D. et al. (2021). *Machine learning accelerated computational fluid dynamics.* PNAS.

"""
config.py — Global configuration for PDE-ML Adaptive Timestep System
"""

import numpy as np

# ──────────────────────────────────────────────
# Grid Configuration
# ──────────────────────────────────────────────
NX = 64          # Grid points in x
NY = 64          # Grid points in y
LX = 1.0         # Domain length in x  [0, LX]
LY = 1.0         # Domain length in y  [0, LY]

DX = LX / (NX - 1)
DY = LY / (NY - 1)

# ──────────────────────────────────────────────
# Physical Parameters
# ──────────────────────────────────────────────
ALPHA = 0.01     # Thermal diffusivity (heat equation)
CX    = 0.5      # Advection velocity x-component
CY    = 0.3      # Advection velocity y-component

# ──────────────────────────────────────────────
# Simulation Time
# ──────────────────────────────────────────────
T_END   = 0.5    # Final simulation time
DT_FIXED = 1e-3  # Fixed timestep (baseline)

# ──────────────────────────────────────────────
# CFL / Stability Constraints
# ──────────────────────────────────────────────
CFL_TARGET  = 0.5                       # Target CFL number
LAMBDA_MAX  = max(abs(CX), abs(CY))     # Max wave speed
DT_CFL      = CFL_TARGET * min(DX, DY) / (LAMBDA_MAX + 1e-12)
DT_DIFFUSE  = 0.4 * min(DX, DY)**2 / (2 * ALPHA)  # Von Neumann stability
DT_MIN      = 1e-5
DT_MAX      = 5e-3

# ──────────────────────────────────────────────
# ML Model Configuration
# ──────────────────────────────────────────────
INPUT_DIM     = 7    # Feature vector dimension
HIDDEN_DIMS   = [128, 256, 128, 64]
LSTM_HIDDEN   = 128
LSTM_LAYERS   = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
BATCH_SIZE    = 256
EPOCHS        = 200
SEQ_LEN       = 10   # LSTM sequence length

# ──────────────────────────────────────────────
# Data Generation
# ──────────────────────────────────────────────
N_SIMULATIONS = 50    # Number of independent simulations for training data
N_ICS         = 10    # Number of different initial conditions per PDE type

# ──────────────────────────────────────────────
# Output / Logging
# ──────────────────────────────────────────────
RESULTS_DIR = "results"
DATA_DIR    = "data"
MODEL_DIR   = "models"

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
SEED = 42

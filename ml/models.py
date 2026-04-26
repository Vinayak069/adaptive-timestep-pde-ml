"""
ml/models.py
─────────────────────────────────────────────────────────────────────────────
Neural Network Models for Adaptive Timestep Prediction.

Three model architectures are implemented and compared:

1. FeedforwardNet  — Baseline MLP with residual connections
2. LSTMPredictor   — Temporal model for time-series PDE features
3. PINNPredictor   — Physics-Informed NN with CFL-aware loss term

All models output log(Δt/Δt_cfl) which is then exponentiated and clipped,
ensuring the output is always a physically valid positive timestep.
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config


# ──────────────────────────────────────────────────────────────────────────
# 1. Feedforward Neural Network (Baseline MLP with Residual Connections)
# ──────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """A residual block: y = F(x) + x, with layer norm."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class FeedforwardNet(nn.Module):
    """
    Residual MLP for timestep prediction.

    Architecture:
        Input (7) → Linear → [ResidualBlock × N] → Linear → output (1)

    Output: log-ratio  log(Δt / Δt_cfl_reference)
    Final Δt is recovered as: Δt = Δt_ref * exp(output)
    """

    def __init__(self,
                 input_dim:   int   = config.INPUT_DIM,
                 hidden_dims: list  = config.HIDDEN_DIMS,
                 dropout:     float = 0.1):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dims[0]),
                  nn.LayerNorm(hidden_dims[0]),
                  nn.GELU()]

        for i in range(len(hidden_dims) - 1):
            if hidden_dims[i] == hidden_dims[i+1]:
                layers.append(ResidualBlock(hidden_dims[i], dropout))
            else:
                layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                           nn.LayerNorm(hidden_dims[i+1]),
                           nn.GELU(),
                           nn.Dropout(dropout)]

        self.encoder = nn.Sequential(*layers)
        self.head    = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, input_dim)
        returns : (batch,) — predicted log(Δt_ratio)
        """
        h = self.encoder(x)
        return self.head(h).squeeze(-1)

    def predict_dt(self, x: torch.Tensor,
                   dt_ref: float = config.DT_CFL) -> torch.Tensor:
        """Return actual Δt in physical units, clamped to [DT_MIN, DT_MAX]."""
        log_ratio = self.forward(x)
        dt = dt_ref * torch.exp(log_ratio.detach())
        return dt.clamp(config.DT_MIN, config.DT_MAX)


# ──────────────────────────────────────────────────────────────────────────
# 2. LSTM Temporal Model
# ──────────────────────────────────────────────────────────────────────────

class LSTMPredictor(nn.Module):
    """
    LSTM-based predictor that processes a sequence of feature vectors
    from past timesteps to predict the next optimal Δt.

    Architecture:
        Input sequence (seq_len, 7) → LSTM → FC layers → output (1)

    The temporal context captures trends in solution evolution (e.g.,
    growing gradients, oscillating residuals) that a stateless MLP misses.
    """

    def __init__(self,
                 input_dim:    int = config.INPUT_DIM,
                 hidden_size:  int = config.LSTM_HIDDEN,
                 num_layers:   int = config.LSTM_LAYERS,
                 dropout:      float = 0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
            bidirectional = False
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

    def forward(self, x: torch.Tensor,
                hidden: tuple = None) -> tuple:
        """
        x      : (batch, seq_len, input_dim)
        hidden : optional (h_0, c_0) for stateful inference

        Returns
        -------
        (output, hidden_state)
        output : (batch,) — predicted log(Δt_ratio)
        """
        out, hidden = self.lstm(x, hidden)   # out: (B, T, H)

        # Attention pooling over time dimension
        attn_scores = self.attention(out).squeeze(-1)   # (B, T)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, T)
        context = torch.bmm(attn_weights.unsqueeze(1), out).squeeze(1)  # (B, H)

        return self.head(context).squeeze(-1), hidden

    def predict_dt(self, x: torch.Tensor,
                   dt_ref: float = config.DT_CFL,
                   hidden: tuple = None) -> tuple:
        """Return actual Δt in physical units, clamped."""
        log_ratio, hidden = self.forward(x, hidden)
        dt = dt_ref * torch.exp(log_ratio.detach())
        return dt.clamp(config.DT_MIN, config.DT_MAX), hidden


# ──────────────────────────────────────────────────────────────────────────
# 3. Physics-Informed Neural Network (PINN-style)
# ──────────────────────────────────────────────────────────────────────────

class PINNPredictor(nn.Module):
    """
    Physics-Informed Neural Network for timestep prediction.

    Augments the standard MSE regression loss with a physics-based
    penalty term that penalizes CFL violations:

        L_total = L_data + λ_phys * L_physics

    where:
        L_data    = MSE(Δt_pred, Δt_true)
        L_physics = mean(ReLU(CFL_pred - 1.0)²)
              (penalizes predicted Δt that violates CFL stability)

    The physics loss is computed inside the model forward pass so it
    can be included in training without a custom training loop.
    """

    def __init__(self,
                 input_dim:    int   = config.INPUT_DIM,
                 hidden_dims:  list  = None,
                 lambda_phys:  float = 1.0,
                 cx:           float = config.CX,
                 cy:           float = config.CY,
                 dx:           float = config.DX,
                 dy:           float = config.DY):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 256, 128, 64]

        self.lambda_phys = lambda_phys
        self.cx = cx; self.cy = cy
        self.dx = dx; self.dy = dy

        # Backbone MLP
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1]),
                       nn.LayerNorm(dims[i+1]),
                       nn.GELU()]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], 1)

        # Physics coefficient predictor: predicts log(CFL_used)
        self.cfl_head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> dict:
        """
        x : (batch, input_dim)

        Returns dict with keys:
          'log_ratio' : predicted log(Δt/Δt_ref)
          'cfl_pred'  : predicted CFL number (for physics loss)
          'phys_loss' : CFL violation penalty
        """
        h = self.backbone(x)
        log_ratio = self.head(h).squeeze(-1)
        cfl_pred  = torch.sigmoid(self.cfl_head(h).squeeze(-1))  # ∈ (0,1)

        # Physics loss: penalize if predicted CFL exceeds 1.0
        phys_loss = F.relu(cfl_pred - config.CFL_TARGET).pow(2).mean()

        return {
            'log_ratio': log_ratio,
            'cfl_pred':  cfl_pred,
            'phys_loss': phys_loss
        }

    def predict_dt(self, x: torch.Tensor,
                   dt_ref: float = config.DT_CFL) -> torch.Tensor:
        out = self.forward(x)
        dt  = dt_ref * torch.exp(out['log_ratio'].detach())
        return dt.clamp(config.DT_MIN, config.DT_MAX)


# ──────────────────────────────────────────────────────────────────────────
# Utility: Model Factory
# ──────────────────────────────────────────────────────────────────────────

def get_model(model_type: str = 'feedforward', **kwargs) -> nn.Module:
    """
    Factory function to instantiate a model by name.

    Parameters
    ----------
    model_type : 'feedforward' | 'lstm' | 'pinn'

    Returns
    -------
    nn.Module
    """
    models = {
        'feedforward': FeedforwardNet,
        'lstm':        LSTMPredictor,
        'pinn':        PINNPredictor,
    }
    if model_type not in models:
        raise ValueError(f"Unknown model type '{model_type}'. "
                         f"Choose from: {list(models.keys())}")
    return models[model_type](**kwargs)

"""
ml/trainer.py
─────────────────────────────────────────────────────────────────────────────
Training pipeline for all ML models (FeedforwardNet, LSTMPredictor, PINN).

Features:
  • Unified train/val split with reproducible shuffling
  • Cosine annealing LR schedule with warm restarts
  • Gradient clipping to prevent exploding gradients
  • Early stopping with patience
  • PINN-specific physics loss weighting
  • Per-epoch metrics logging
─────────────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Optional
import config
from ml.models import FeedforwardNet, LSTMPredictor, PINNPredictor


class TimestepDataset(torch.utils.data.Dataset):
    """
    Dataset wrapping (features, targets) arrays.
    For LSTM mode, features should be (N, seq_len, input_dim).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(model:        nn.Module,
                X_train:      np.ndarray,
                y_train:      np.ndarray,
                model_type:   str   = 'feedforward',
                epochs:       int   = config.EPOCHS,
                batch_size:   int   = config.BATCH_SIZE,
                lr:           float = config.LEARNING_RATE,
                weight_decay: float = config.WEIGHT_DECAY,
                val_split:    float = 0.15,
                patience:     int   = 20,
                lambda_phys:  float = 1.0,
                device:       str   = 'cpu',
                save_path:    Optional[str] = None,
                verbose:      bool  = True) -> dict:
    """
    Train a timestep prediction model.

    Parameters
    ----------
    model      : nn.Module (FeedforwardNet | LSTMPredictor | PINNPredictor)
    X_train    : feature array, shape (N, input_dim) or (N, seq_len, input_dim)
    y_train    : log-ratio targets, shape (N,)
    model_type : 'feedforward' | 'lstm' | 'pinn'
    val_split  : fraction of data to use for validation
    patience   : early stopping patience (epochs without improvement)
    save_path  : if provided, saves best model state dict here

    Returns
    -------
    history : dict with keys 'train_loss', 'val_loss', 'lr'
    """
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    device = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
    model  = model.to(device)

    dataset   = TimestepDataset(X_train, y_train)
    n_val     = int(len(dataset) * val_split)
    n_train   = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(config.SEED)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    mse_loss  = nn.MSELoss()
    huber_loss = nn.HuberLoss(delta=0.5)

    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val  = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # ── Training ────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            if model_type == 'lstm':
                pred, _ = model(X_batch)
            elif model_type == 'pinn':
                out  = model(X_batch)
                pred = out['log_ratio']
                phys = out['phys_loss']
            else:
                pred = model(X_batch)

            loss = huber_loss(pred, y_batch)

            if model_type == 'pinn':
                loss = loss + lambda_phys * phys

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(X_batch)

        train_loss /= n_train
        scheduler.step()

        # ── Validation ──────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                if model_type == 'lstm':
                    pred, _ = model(X_batch)
                elif model_type == 'pinn':
                    out  = model(X_batch)
                    pred = out['log_ratio']
                else:
                    pred = model(X_batch)

                val_loss += mse_loss(pred, y_batch).item() * len(X_batch)

        val_loss /= n_val
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        # ── Early Stopping ──────────────────────────────────────────────
        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"Epoch [{epoch:4d}/{epochs}] "
                  f"Train: {train_loss:.6f}  Val: {val_loss:.6f}  "
                  f"LR: {current_lr:.2e}  "
                  f"Best Val: {best_val:.6f}  Patience: {no_improve}/{patience}")

        if no_improve >= patience:
            if verbose:
                print(f"\n[EarlyStopping] No improvement for {patience} epochs. "
                      f"Stopping at epoch {epoch}.")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({'model_state': model.state_dict(),
                    'history':     history,
                    'best_val':    best_val}, save_path)
        if verbose:
            print(f"\n[✓] Model saved to {save_path}")

    return history


def load_model(model: nn.Module, path: str) -> dict:
    """Load model weights and return training history."""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    return checkpoint.get('history', {})

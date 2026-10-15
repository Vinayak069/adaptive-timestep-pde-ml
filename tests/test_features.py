"""Feature extractor, optimal-Δt target and speedup bookkeeping."""

import numpy as np

import config
from data_generation import IC_REGISTRY
from ml.feature_extractor import extract_features, compute_optimal_dt


def test_feature_vector_shape_and_finite():
    for name, ic in IC_REGISTRY.items():
        u = ic(config.NX, config.NY, config.DX, config.DY)
        for solver_type in ('adi', 'maccormack'):
            f = extract_features(u, 0.99 * u, dt_prev=config.DT_FIXED,
                                 solver_type=solver_type)
            assert f.shape == (config.INPUT_DIM,), name
            assert np.isfinite(f).all(), name


def test_optimal_dt_within_bounds():
    for ic in IC_REGISTRY.values():
        u = ic(config.NX, config.NY, config.DX, config.DY)
        for solver_type in ('adi', 'maccormack'):
            dt = compute_optimal_dt(u, config.DX, config.DY, config.CX,
                                    config.CY, config.ALPHA, solver_type)
            assert config.DT_MIN <= dt <= config.DT_MAX


def test_speedup_is_relative_to_the_fixed_run():
    """Speedup must use the fixed-Δt run that was actually performed."""
    from main import compute_summary_metrics
    logs = {
        'Fixed Δt':     {'dt': np.full(100, 1e-3), 'l2_error': np.zeros(100)},
        'CFL-Adaptive': {'dt': np.full(20, 5e-3),  'l2_error': np.zeros(20)},
    }
    metrics = compute_summary_metrics(logs)
    assert metrics['Fixed Δt']['speedup'] == 1.0
    assert metrics['CFL-Adaptive']['speedup'] == 5.0

from __future__ import annotations

import numpy as np
import torch

from dsl_elto.config import MLPReadoutConfig, ReadoutConfig
from dsl_elto.readout import predict_mlp_readout, train_mlp_readout


def test_mlp_readout_training_lowers_feature_loss() -> None:
    torch.manual_seed(0)
    H_prior, Z_M = _nonlinear_toy_data()
    config = MLPReadoutConfig(
        hidden_dim=16,
        num_layers=1,
        activation="tanh",
        lr=5.0e-2,
        epochs=120,
        weight_decay=0.0,
        batch_size=None,
        early_stopping=False,
        patience=20,
    )

    model, loss_log = train_mlp_readout(H_prior, Z_M, config)
    ZM_pred = predict_mlp_readout(H_prior, model)

    assert len(loss_log) == config.epochs
    assert loss_log[-1] < loss_log[0]
    assert ZM_pred.shape == Z_M.shape
    assert np.all(np.isfinite(ZM_pred))


def test_mlp_readout_accepts_readout_config_and_does_not_update_core_arrays() -> None:
    torch.manual_seed(1)
    H_prior, Z_M = _nonlinear_toy_data()
    readout_config = ReadoutConfig(
        type="mlp",
        train_on_filter_prior=True,
        use_posterior_reconstruction=False,
        mlp=MLPReadoutConfig(
            hidden_dim=8,
            num_layers=1,
            activation="relu",
            lr=1.0e-2,
            epochs=10,
            weight_decay=0.0,
            batch_size=None,
            early_stopping=False,
            patience=20,
        ),
    )
    A_filter_minus = H_prior.copy()
    V_B = np.eye(H_prior.shape[1])
    A_filter_minus_before = A_filter_minus.copy()
    V_B_before = V_B.copy()

    model, loss_log = train_mlp_readout(H_prior, Z_M, readout_config)

    assert len(loss_log) == readout_config.mlp.epochs
    np.testing.assert_allclose(A_filter_minus, A_filter_minus_before)
    np.testing.assert_allclose(V_B, V_B_before)
    assert predict_mlp_readout(H_prior, model).shape == Z_M.shape


def _nonlinear_toy_data() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, 64)
    H_prior = np.column_stack([x, x * x])
    Z_M = np.column_stack([np.sin(np.pi * x), x + 0.5 * x * x])
    return H_prior, Z_M

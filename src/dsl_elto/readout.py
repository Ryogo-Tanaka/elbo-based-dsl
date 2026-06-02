"""Readout models for DSL/ELTO Version 0 predictions.

All feature arrays use the code convention `[T, d]`.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .config import MLPReadoutConfig, ReadoutConfig
from .shapes import assert_same_T, assert_shape, check_finite, make_eye


def fit_linear_readout(H_prior: np.ndarray, Z_M: np.ndarray, lambda_readout: float) -> np.ndarray:
    """Fit `R_readout: [m, d_B]` by normalized covariance ridge regression."""

    H_prior = check_finite("H_prior", assert_shape("H_prior", H_prior, (None, None)))
    Z_M = check_finite("Z_M", assert_shape("Z_M", Z_M, (None, None)))
    T = assert_same_T(("H_prior", H_prior), ("Z_M", Z_M))
    if T == 0:
        raise ValueError("H_prior and Z_M must contain at least one time step")
    if lambda_readout <= 0:
        raise ValueError("lambda_readout must be positive")

    _, d_B = H_prior.shape
    C_yx = (Z_M.T @ H_prior) / T
    C_xx = (H_prior.T @ H_prior) / T
    ridge = C_xx + lambda_readout * make_eye(d_B, dtype=H_prior.dtype)
    return np.linalg.solve(ridge.T, C_yx.T).T


def predict_linear_readout(H_prior: np.ndarray, R_readout: np.ndarray) -> np.ndarray:
    """Predict standardized target features as `H_prior @ R_readout.T`."""

    H_prior = check_finite("H_prior", assert_shape("H_prior", H_prior, (None, None)))
    _, d_B = H_prior.shape
    R_readout = check_finite("R_readout", assert_shape("R_readout", R_readout, (None, d_B)))
    return H_prior @ R_readout.T


class MLPReadout(nn.Module):
    """Small MLP mapping prior prediction features `[T, d_B]` to `[T, m]`."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        _require_positive_int("input_dim", input_dim)
        _require_positive_int("output_dim", output_dim)
        _require_positive_int("hidden_dim", hidden_dim)
        _require_positive_int("num_layers", num_layers)

        activation_factory = _activation_factory(activation)
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation_factory())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, H_prior: torch.Tensor) -> torch.Tensor:
        return self.network(H_prior)


def train_mlp_readout(
    H_prior: np.ndarray,
    Z_M: np.ndarray,
    readout_config: ReadoutConfig | MLPReadoutConfig,
) -> tuple[MLPReadout, list[float]]:
    """Train only an MLP readout module on fixed prior prediction features."""

    H_prior = check_finite("H_prior", assert_shape("H_prior", H_prior, (None, None)))
    Z_M = check_finite("Z_M", assert_shape("Z_M", Z_M, (None, None)))
    T = assert_same_T(("H_prior", H_prior), ("Z_M", Z_M))
    if T == 0:
        raise ValueError("H_prior and Z_M must contain at least one time step")

    mlp_config = readout_config.mlp if isinstance(readout_config, ReadoutConfig) else readout_config
    _validate_mlp_runtime_config(mlp_config)
    _, d_B = H_prior.shape
    _, m = Z_M.shape

    model = MLPReadout(
        input_dim=d_B,
        output_dim=m,
        hidden_dim=mlp_config.hidden_dim,
        num_layers=mlp_config.num_layers,
        activation=mlp_config.activation,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=mlp_config.lr, weight_decay=mlp_config.weight_decay)
    H_tensor = torch.as_tensor(np.asarray(H_prior, dtype=np.float32))
    Z_tensor = torch.as_tensor(np.asarray(Z_M, dtype=np.float32))
    loss_log: list[float] = []

    if mlp_config.batch_size is None or mlp_config.batch_size >= T:
        _train_full_batch(model, optimizer, H_tensor, Z_tensor, mlp_config.epochs, loss_log)
    else:
        _train_mini_batch(model, optimizer, H_tensor, Z_tensor, mlp_config, loss_log)

    return model, loss_log


def predict_mlp_readout(H_prior: np.ndarray, model: MLPReadout) -> np.ndarray:
    """Predict standardized target features with a trained MLP readout."""

    H_prior = check_finite("H_prior", assert_shape("H_prior", H_prior, (None, None)))
    H_tensor = torch.as_tensor(np.asarray(H_prior, dtype=np.float32))
    model.eval()
    with torch.no_grad():
        ZM_pred = model(H_tensor).detach().cpu().numpy()
    return ZM_pred


def _train_full_batch(
    model: MLPReadout,
    optimizer: torch.optim.Optimizer,
    H_tensor: torch.Tensor,
    Z_tensor: torch.Tensor,
    epochs: int,
    loss_log: list[float],
) -> None:
    for _ in range(epochs):
        optimizer.zero_grad()
        prediction = model(H_tensor)
        loss = torch.nn.functional.mse_loss(prediction, Z_tensor)
        loss.backward()
        optimizer.step()
        loss_log.append(float(loss.detach().cpu()))


def _train_mini_batch(
    model: MLPReadout,
    optimizer: torch.optim.Optimizer,
    H_tensor: torch.Tensor,
    Z_tensor: torch.Tensor,
    config: MLPReadoutConfig,
    loss_log: list[float],
) -> None:
    T = H_tensor.shape[0]
    batch_size = int(config.batch_size)
    for _ in range(config.epochs):
        permutation = torch.randperm(T)
        for start in range(0, T, batch_size):
            batch = permutation[start : start + batch_size]
            optimizer.zero_grad()
            prediction = model(H_tensor[batch])
            loss = torch.nn.functional.mse_loss(prediction, Z_tensor[batch])
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            full_loss = torch.nn.functional.mse_loss(model(H_tensor), Z_tensor)
        loss_log.append(float(full_loss.detach().cpu()))


def _activation_factory(name: str) -> type[nn.Module]:
    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
    }
    try:
        return activations[name]
    except KeyError as exc:
        supported = ", ".join(sorted(activations))
        raise ValueError(f"readout.mlp.activation must be one of {supported}") from exc


def _validate_mlp_runtime_config(config: MLPReadoutConfig) -> None:
    _require_positive_int("readout.mlp.hidden_dim", config.hidden_dim)
    _require_positive_int("readout.mlp.num_layers", config.num_layers)
    _require_positive("readout.mlp.lr", config.lr)
    _require_positive_int("readout.mlp.epochs", config.epochs)
    if config.weight_decay < 0:
        raise ValueError("readout.mlp.weight_decay must be nonnegative")
    if config.batch_size is not None:
        _require_positive_int("readout.mlp.batch_size", config.batch_size)
    _activation_factory(config.activation)


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")

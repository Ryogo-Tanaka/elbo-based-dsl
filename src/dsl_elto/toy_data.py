"""Minimal toy data for DSL/ELTO Version 0 solver sanity checks.

This module provides synthetic precomputed features. It is not a benchmark.
All arrays use the code convention `[T, d]`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .shapes import check_finite


@dataclass(frozen=True)
class ToyDataResult:
    """Toy latent states, observations, and precomputed feature arrays."""

    X_true: np.ndarray
    Y: np.ndarray
    M: np.ndarray
    B: np.ndarray


def generate_damped_rotation(
    T: int,
    radius: float,
    omega: float,
    process_noise_std: float,
    observation_noise_std: float,
    seed: int | None = None,
) -> ToyDataResult:
    """Generate a stable 2D damped rotation with precomputed `M` and `B` features."""

    _validate_inputs(
        T=T,
        radius=radius,
        omega=omega,
        process_noise_std=process_noise_std,
        observation_noise_std=observation_noise_std,
    )
    rng = np.random.default_rng(seed)
    transition = radius * np.array(
        [
            [np.cos(omega), -np.sin(omega)],
            [np.sin(omega), np.cos(omega)],
        ],
        dtype=float,
    )

    X_true = np.empty((T, 2), dtype=float)
    X_true[0] = np.array([1.0, 0.0], dtype=float)
    for t in range(1, T):
        process_noise = rng.normal(scale=process_noise_std, size=2)
        X_true[t] = transition @ X_true[t - 1] + process_noise

    Y = np.column_stack([X_true[:, 0], X_true[:, 1], X_true[:, 0] * X_true[:, 0]])
    if observation_noise_std > 0:
        Y = Y + rng.normal(scale=observation_noise_std, size=Y.shape)

    M = Y.copy()
    B = Y.copy()
    check_finite("X_true", X_true)
    check_finite("Y", Y)
    check_finite("M", M)
    check_finite("B", B)
    return ToyDataResult(X_true=X_true, Y=Y, M=M, B=B)


def _validate_inputs(
    T: int,
    radius: float,
    omega: float,
    process_noise_std: float,
    observation_noise_std: float,
) -> None:
    if not isinstance(T, int) or T < 2:
        raise ValueError("T must be an integer >= 2")
    if not 0.0 < radius < 1.0:
        raise ValueError("radius must satisfy 0 < radius < 1")
    if not np.isfinite(omega):
        raise ValueError("omega must be finite")
    if process_noise_std < 0 or not np.isfinite(process_noise_std):
        raise ValueError("process_noise_std must be finite and nonnegative")
    if observation_noise_std < 0 or not np.isfinite(observation_noise_std):
        raise ValueError("observation_noise_std must be finite and nonnegative")

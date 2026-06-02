"""Correction-based warm start for DSL/ELTO Version 0.

All coordinate arrays use the code convention `[T, d]`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .shapes import assert_shape, check_finite, make_eye


@dataclass(frozen=True)
class WarmStartResult:
    """Initial Phase 1 coordinates and operator seeds."""

    A_train_minus_init: np.ndarray
    A_train_plus_init: np.ndarray
    V_A_init: np.ndarray
    V_B_init: np.ndarray


def initialize_warm_start(
    Z_B: np.ndarray,
    d_A: int,
    a_init: np.ndarray | None,
    sigma_b2: float,
    p: float,
    seed: int,
    eps: float,
) -> WarmStartResult:
    """Build correction-based numerical seeds without direct state construction."""

    Z_B = check_finite("Z_B", assert_shape("Z_B", Z_B, (None, None)))
    T, d_B = Z_B.shape
    if T == 0:
        raise ValueError("Z_B must contain at least one time step")
    if not isinstance(d_A, int) or d_A <= 0:
        raise ValueError("d_A must be a positive integer")
    if sigma_b2 <= 0:
        raise ValueError("sigma_b2 must be positive")
    if p <= 0:
        raise ValueError("p must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")

    if a_init is None:
        a_init_array = np.zeros(d_A, dtype=Z_B.dtype)
    else:
        a_init_array = check_finite("a_init", assert_shape("a_init", a_init, (d_A,)))

    rng = np.random.default_rng(seed)
    V_B_init = rng.normal(loc=0.0, scale=1.0 / np.sqrt(d_A), size=(d_B, d_A))
    V_B_init = _normalize_columns(V_B_init, eps)
    V_A_init = np.zeros((d_A, d_A), dtype=V_B_init.dtype)

    w_B = 1.0 / sigma_b2
    w_P = 1.0 / p
    eye = make_eye(d_A, dtype=V_B_init.dtype)
    H_0 = w_B * (V_B_init.T @ V_B_init) + w_P * eye
    H_0 = H_0 + eps * eye
    rhs = w_B * (Z_B @ V_B_init) + w_P * a_init_array[None, :]
    A_train_plus_init = np.linalg.solve(H_0, rhs.T).T

    A_train_minus_init = np.empty((T, d_A), dtype=A_train_plus_init.dtype)
    A_train_minus_init[0] = a_init_array
    if T > 1:
        A_train_minus_init[1:] = A_train_plus_init[1:]

    check_finite("A_train_minus_init", A_train_minus_init)
    check_finite("A_train_plus_init", A_train_plus_init)
    check_finite("V_A_init", V_A_init)
    check_finite("V_B_init", V_B_init)

    return WarmStartResult(
        A_train_minus_init=A_train_minus_init,
        A_train_plus_init=A_train_plus_init,
        V_A_init=V_A_init,
        V_B_init=V_B_init,
    )


def _normalize_columns(X: np.ndarray, eps: float) -> np.ndarray:
    norms = np.linalg.norm(X, axis=0)
    if np.any(norms <= eps):
        raise ValueError("V_B_init contains a near-zero column")
    return X / norms[None, :]


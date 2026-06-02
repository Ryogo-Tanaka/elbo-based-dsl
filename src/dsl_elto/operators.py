"""Normalized covariance ridge operator updates for DSL/ELTO Version 0."""

from __future__ import annotations

import numpy as np

from .shapes import assert_same_T, assert_shape, check_finite, make_eye


def update_V_B(Z_B: np.ndarray, A_plus: np.ndarray, lambda_b: float) -> np.ndarray:
    """Update observable operator `V_B` with normalized covariance ridge form."""

    Z_B = check_finite("Z_B", assert_shape("Z_B", Z_B, (None, None)))
    A_plus = check_finite("A_plus", assert_shape("A_plus", A_plus, (None, None)))
    T = assert_same_T(("Z_B", Z_B), ("A_plus", A_plus))
    if T == 0:
        raise ValueError("Z_B and A_plus must contain at least one time step")
    _require_positive("lambda_b", lambda_b)

    d_A = A_plus.shape[1]
    C_yx = (Z_B.T @ A_plus) / T
    C_xx = (A_plus.T @ A_plus) / T
    return np.linalg.solve(C_xx + lambda_b * make_eye(d_A, dtype=A_plus.dtype), C_yx.T).T


def update_V_A(A_minus: np.ndarray, A_plus: np.ndarray, lambda_a: float) -> np.ndarray:
    """Update transfer operator `V_A` with normalized covariance ridge form."""

    A_minus = check_finite("A_minus", assert_shape("A_minus", A_minus, (None, None)))
    A_plus = check_finite("A_plus", assert_shape("A_plus", A_plus, (None, None)))
    T = assert_same_T(("A_minus", A_minus), ("A_plus", A_plus))
    if T < 2:
        raise ValueError("A_minus and A_plus must contain at least two time steps")
    if A_minus.shape[1] != A_plus.shape[1]:
        raise ValueError("A_minus and A_plus must share d_A")
    _require_positive("lambda_a", lambda_a)

    X = A_plus[:-1]
    Y = A_minus[1:]
    d_A = X.shape[1]
    C_yx = (Y.T @ X) / (T - 1)
    C_xx = (X.T @ X) / (T - 1)
    return np.linalg.solve(C_xx + lambda_a * make_eye(d_A, dtype=A_plus.dtype), C_yx.T).T


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


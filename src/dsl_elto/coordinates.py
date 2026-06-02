"""Scalar-metric coordinate updates for DSL/ELTO Version 0."""

from __future__ import annotations

import numpy as np

from .shapes import assert_same_T, assert_shape, check_finite, make_eye


def update_A_plus(
    Z_B: np.ndarray,
    A_minus: np.ndarray,
    V_A: np.ndarray,
    V_B: np.ndarray,
    sigma_b2: float,
    sigma_a2: float,
    p: float,
) -> np.ndarray:
    """Update posterior training coordinates `A_train_plus` for scalar metrics."""

    Z_B, A_minus, V_A, V_B = _validate_plus_inputs(Z_B, A_minus, V_A, V_B)
    _require_positive("sigma_b2", sigma_b2)
    _require_positive("sigma_a2", sigma_a2)
    _require_positive("p", p)

    T, d_A = A_minus.shape
    w_B = 1.0 / sigma_b2
    w_A = 1.0 / sigma_a2
    w_P = 1.0 / p
    eye = make_eye(d_A, dtype=A_minus.dtype)

    H_common = w_B * (V_B.T @ V_B) + w_P * eye + w_A * (V_A.T @ V_A)
    H_final = w_B * (V_B.T @ V_B) + w_P * eye

    A_plus = np.empty_like(A_minus, dtype=np.result_type(Z_B, A_minus, V_A, V_B))
    for t in range(T - 1):
        h = w_B * (V_B.T @ Z_B[t]) + w_P * A_minus[t] + w_A * (V_A.T @ A_minus[t + 1])
        A_plus[t] = np.linalg.solve(H_common, h)

    h_final = w_B * (V_B.T @ Z_B[-1]) + w_P * A_minus[-1]
    A_plus[-1] = np.linalg.solve(H_final, h_final)
    return check_finite("A_plus", A_plus)


def update_A_minus(
    A_plus: np.ndarray,
    V_A: np.ndarray,
    a_init: np.ndarray,
    sigma_a2: float,
    p: float,
    c0: float,
) -> np.ndarray:
    """Update prior training coordinates `A_train_minus` for scalar metrics."""

    A_plus = check_finite("A_plus", assert_shape("A_plus", A_plus, (None, None)))
    T, d_A = A_plus.shape
    if T == 0:
        raise ValueError("A_plus must contain at least one time step")
    V_A = check_finite("V_A", assert_shape("V_A", V_A, (d_A, d_A)))
    a_init = check_finite("a_init", assert_shape("a_init", a_init, (d_A,)))
    _require_positive("sigma_a2", sigma_a2)
    _require_positive("p", p)
    _require_positive("c0", c0)

    w_A = 1.0 / sigma_a2
    w_P = 1.0 / p
    w_0 = 1.0 / (c0 * p)

    A_minus = np.empty_like(A_plus)
    A_minus[0] = (w_P * A_plus[0] + w_0 * a_init) / (w_P + w_0)
    if T > 1:
        propagated = A_plus[:-1] @ V_A.T
        A_minus[1:] = (w_P * A_plus[1:] + w_A * propagated) / (w_P + w_A)
    return check_finite("A_minus", A_minus)


def _validate_plus_inputs(
    Z_B: np.ndarray,
    A_minus: np.ndarray,
    V_A: np.ndarray,
    V_B: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Z_B = check_finite("Z_B", assert_shape("Z_B", Z_B, (None, None)))
    A_minus = check_finite("A_minus", assert_shape("A_minus", A_minus, (None, None)))
    T = assert_same_T(("Z_B", Z_B), ("A_minus", A_minus))
    if T == 0:
        raise ValueError("Z_B and A_minus must contain at least one time step")
    d_B = Z_B.shape[1]
    d_A = A_minus.shape[1]
    V_A = check_finite("V_A", assert_shape("V_A", V_A, (d_A, d_A)))
    V_B = check_finite("V_B", assert_shape("V_B", V_B, (d_B, d_A)))
    return Z_B, A_minus, V_A, V_B


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


"""Phase 2 causal filtering path for DSL/ELTO Version 0."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .shapes import assert_shape, check_finite, make_eye


@dataclass(frozen=True)
class FilterResult:
    """Causal filtering coordinates and prior prediction diagnostics."""

    A_filter_minus: np.ndarray
    A_filter_plus: np.ndarray
    ZB_pred_prior: np.ndarray
    innovations: np.ndarray
    correction_norms: np.ndarray
    K: np.ndarray


def run_causal_filter(
    Z_B: np.ndarray,
    V_A: np.ndarray,
    V_B: np.ndarray,
    a_init: np.ndarray,
    sigma_b2: float,
    p: float,
) -> FilterResult:
    """Construct a causal prediction-correction path from learned operators."""

    Z_B = check_finite("Z_B", assert_shape("Z_B", Z_B, (None, None)))
    T, d_B = Z_B.shape
    if T == 0:
        raise ValueError("Z_B must contain at least one time step")
    V_B = check_finite("V_B", assert_shape("V_B", V_B, (d_B, None)))
    d_A = V_B.shape[1]
    V_A = check_finite("V_A", assert_shape("V_A", V_A, (d_A, d_A)))
    a_init = check_finite("a_init", assert_shape("a_init", a_init, (d_A,)))
    if sigma_b2 <= 0:
        raise ValueError("sigma_b2 must be positive")
    if p <= 0:
        raise ValueError("p must be positive")

    K = _compute_fixed_gain(V_B, sigma_b2=sigma_b2, p=p)
    A_filter_minus = np.empty((T, d_A), dtype=np.result_type(Z_B, V_A, V_B, a_init))
    A_filter_plus = np.empty_like(A_filter_minus)
    ZB_pred_prior = np.empty((T, d_B), dtype=A_filter_minus.dtype)
    innovations = np.empty_like(ZB_pred_prior)
    correction_norms = np.empty(T, dtype=A_filter_minus.dtype)

    A_filter_minus[0] = a_init
    for t in range(T):
        ZB_pred_prior[t] = A_filter_minus[t] @ V_B.T
        innovations[t] = Z_B[t] - ZB_pred_prior[t]
        correction = innovations[t] @ K.T
        correction_norms[t] = np.linalg.norm(correction)
        A_filter_plus[t] = A_filter_minus[t] + correction
        if t < T - 1:
            A_filter_minus[t + 1] = A_filter_plus[t] @ V_A.T

    check_finite("A_filter_minus", A_filter_minus)
    check_finite("A_filter_plus", A_filter_plus)
    check_finite("ZB_pred_prior", ZB_pred_prior)
    check_finite("innovations", innovations)
    check_finite("correction_norms", correction_norms)
    check_finite("K", K)
    return FilterResult(
        A_filter_minus=A_filter_minus,
        A_filter_plus=A_filter_plus,
        ZB_pred_prior=ZB_pred_prior,
        innovations=innovations,
        correction_norms=correction_norms,
        K=K,
    )


def _compute_fixed_gain(V_B: np.ndarray, sigma_b2: float, p: float) -> np.ndarray:
    d_B = V_B.shape[0]
    innovation_cov = p * (V_B @ V_B.T) + sigma_b2 * make_eye(d_B, dtype=V_B.dtype)
    right = (p * V_B.T).T
    return np.linalg.solve(innovation_cov, right).T


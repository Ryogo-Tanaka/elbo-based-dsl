"""Embedding free-energy objective components for DSL/ELTO Version 0."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .shapes import assert_same_T, assert_shape, check_finite


@dataclass(frozen=True)
class ObjectiveComponents:
    """Scalar objective components for fixed scalar metrics."""

    init: float
    obs: float
    corr: float
    trans: float
    total: float


def compute_objective_components(
    Z_B: np.ndarray,
    A_train_minus: np.ndarray,
    A_train_plus: np.ndarray,
    V_A: np.ndarray,
    V_B: np.ndarray,
    a_init: np.ndarray,
    sigma_b2: float,
    sigma_a2: float,
    p: float,
    c0: float,
) -> ObjectiveComponents:
    """Compute Version 0 embedding free-energy components for `[T, d]` arrays."""

    Z_B, A_train_minus, A_train_plus, V_A, V_B, a_init = _validate_objective_inputs(
        Z_B,
        A_train_minus,
        A_train_plus,
        V_A,
        V_B,
        a_init,
    )
    _require_positive("sigma_b2", sigma_b2)
    _require_positive("sigma_a2", sigma_a2)
    _require_positive("p", p)
    _require_positive("c0", c0)

    residual_init = A_train_minus[0] - a_init
    residual_obs = Z_B - A_train_plus @ V_B.T
    residual_corr = A_train_plus - A_train_minus
    residual_trans = A_train_minus[1:] - A_train_plus[:-1] @ V_A.T

    init = 0.5 / (c0 * p) * float(np.sum(residual_init * residual_init))
    obs = 0.5 / sigma_b2 * float(np.sum(residual_obs * residual_obs))
    corr = 0.5 / p * float(np.sum(residual_corr * residual_corr))
    trans = 0.5 / sigma_a2 * float(np.sum(residual_trans * residual_trans))
    return compute_total_objective(init=init, obs=obs, corr=corr, trans=trans)


def compute_total_objective(init: float, obs: float, corr: float, trans: float) -> ObjectiveComponents:
    """Return objective components with `total = init + obs + corr + trans`."""

    for name, value in {"init": init, "obs": obs, "corr": corr, "trans": trans}.items():
        if value < 0:
            raise ValueError(f"{name} objective component must be nonnegative")
        if not np.isfinite(value):
            raise ValueError(f"{name} objective component must be finite")
    total = float(init + obs + corr + trans)
    return ObjectiveComponents(init=float(init), obs=float(obs), corr=float(corr), trans=float(trans), total=total)


def _validate_objective_inputs(
    Z_B: np.ndarray,
    A_train_minus: np.ndarray,
    A_train_plus: np.ndarray,
    V_A: np.ndarray,
    V_B: np.ndarray,
    a_init: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Z_B = check_finite("Z_B", assert_shape("Z_B", Z_B, (None, None)))
    A_train_minus = check_finite("A_train_minus", assert_shape("A_train_minus", A_train_minus, (None, None)))
    A_train_plus = check_finite("A_train_plus", assert_shape("A_train_plus", A_train_plus, (None, None)))
    assert_same_T(("Z_B", Z_B), ("A_train_minus", A_train_minus), ("A_train_plus", A_train_plus))

    T, d_B = Z_B.shape
    _, d_A = A_train_plus.shape
    if T < 2:
        raise ValueError("objective requires T >= 2")

    if A_train_minus.shape[1] != d_A:
        raise ValueError("A_train_minus and A_train_plus must share d_A")
    V_A = check_finite("V_A", assert_shape("V_A", V_A, (d_A, d_A)))
    V_B = check_finite("V_B", assert_shape("V_B", V_B, (d_B, d_A)))
    a_init = check_finite("a_init", assert_shape("a_init", a_init, (d_A,)))
    return Z_B, A_train_minus, A_train_plus, V_A, V_B, a_init


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


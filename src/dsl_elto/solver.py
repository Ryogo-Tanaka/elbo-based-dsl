"""Phase 1 core solver loop for DSL/ELTO Version 0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import Version0Config, validate_config
from .coordinates import update_A_minus, update_A_plus
from .initialization import initialize_warm_start
from .objective import ObjectiveComponents, compute_objective_components
from .operators import update_V_A, update_V_B
from .shapes import assert_shape, check_finite


@dataclass(frozen=True)
class CoreFitResult:
    """Phase 1 training-time coordinates and learned operators."""

    A_train_minus: np.ndarray
    A_train_plus: np.ndarray
    V_A: np.ndarray
    V_B: np.ndarray
    a_init: np.ndarray
    objective_log: list[ObjectiveComponents]
    diagnostics: dict[str, Any]


def fit_core(Z_B: np.ndarray, config: Version0Config, a_init: np.ndarray | None = None) -> CoreFitResult:
    """Fit Phase 1 training coordinates and operators for standardized `Z_B`."""

    validate_config(config)
    Z_B = check_finite("Z_B", assert_shape("Z_B", Z_B, (None, None)))
    T, _ = Z_B.shape
    if T < 2:
        raise ValueError("fit_core requires at least two time steps")

    d_A = config.model.d_A
    if a_init is None:
        a_init_array = np.zeros(d_A, dtype=Z_B.dtype)
    else:
        a_init_array = check_finite("a_init", assert_shape("a_init", a_init, (d_A,)))

    warm_start = initialize_warm_start(
        Z_B=Z_B,
        d_A=d_A,
        a_init=a_init_array,
        sigma_b2=config.metrics.sigma_b2,
        p=config.metrics.p,
        seed=config.project.random_seed,
        eps=config.features.eps,
    )
    A_train_minus = warm_start.A_train_minus_init.copy()
    A_train_plus = warm_start.A_train_plus_init.copy()
    V_A = warm_start.V_A_init.copy()
    V_B = warm_start.V_B_init.copy()
    objective_log: list[ObjectiveComponents] = []
    converged = False
    stop_reason = "max_outer_iter"

    for iteration in range(config.solver.max_outer_iter):
        V_A = update_V_A(A_train_minus, A_train_plus, config.ridge.lambda_a)
        V_B = update_V_B(Z_B, A_train_plus, config.ridge.lambda_b)
        A_train_plus = update_A_plus(
            Z_B,
            A_train_minus,
            V_A,
            V_B,
            sigma_b2=config.metrics.sigma_b2,
            sigma_a2=config.metrics.sigma_a2,
            p=config.metrics.p,
        )
        A_train_minus = update_A_minus(
            A_train_plus,
            V_A,
            a_init_array,
            sigma_a2=config.metrics.sigma_a2,
            p=config.metrics.p,
            c0=config.metrics.c0,
        )
        _check_coordinate_bounds(A_train_minus, A_train_plus, config.solver.max_abs_A)
        components = compute_objective_components(
            Z_B=Z_B,
            A_train_minus=A_train_minus,
            A_train_plus=A_train_plus,
            V_A=V_A,
            V_B=V_B,
            a_init=a_init_array,
            sigma_b2=config.metrics.sigma_b2,
            sigma_a2=config.metrics.sigma_a2,
            p=config.metrics.p,
            c0=config.metrics.c0,
        )
        objective_log.append(components)

        if _has_converged(objective_log, config.solver.min_outer_iter, config.solver.tol_rel_loss):
            converged = True
            stop_reason = "tol_rel_loss"
            break

    diagnostics = {
        "num_outer_iter": len(objective_log),
        "converged": converged,
        "final_total": objective_log[-1].total,
        "stop_reason": stop_reason,
    }
    return CoreFitResult(
        A_train_minus=A_train_minus,
        A_train_plus=A_train_plus,
        V_A=V_A,
        V_B=V_B,
        a_init=a_init_array.copy(),
        objective_log=objective_log,
        diagnostics=diagnostics,
    )


def _has_converged(log: list[ObjectiveComponents], min_outer_iter: int, tol_rel_loss: float) -> bool:
    if len(log) < max(2, min_outer_iter):
        return False
    previous = log[-2].total
    current = log[-1].total
    rel_decrease = abs(current - previous) / (1.0 + abs(previous))
    return rel_decrease < tol_rel_loss


def _check_coordinate_bounds(A_train_minus: np.ndarray, A_train_plus: np.ndarray, max_abs_A: float) -> None:
    max_abs = max(float(np.max(np.abs(A_train_minus))), float(np.max(np.abs(A_train_plus))))
    if max_abs > max_abs_A:
        raise ValueError(f"coordinate magnitude exceeded max_abs_A={max_abs_A}")


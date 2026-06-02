from __future__ import annotations

import numpy as np
import pytest

from dsl_elto.objective import compute_objective_components, compute_total_objective


def test_objective_components_are_nonnegative_and_total_is_sum() -> None:
    Z_B = np.array([[1.0, 0.0], [0.5, -0.5], [0.0, 1.0]])
    A_minus = np.array([[0.0], [0.2], [0.1]])
    A_plus = np.array([[0.1], [0.3], [0.4]])
    V_A = np.array([[0.8]])
    V_B = np.array([[1.0], [-0.5]])
    a_init = np.array([0.0])

    components = compute_objective_components(
        Z_B,
        A_minus,
        A_plus,
        V_A,
        V_B,
        a_init,
        sigma_b2=1.0,
        sigma_a2=2.0,
        p=1.5,
        c0=10.0,
    )

    assert components.init >= 0.0
    assert components.obs >= 0.0
    assert components.corr >= 0.0
    assert components.trans >= 0.0
    assert components.total == pytest.approx(components.init + components.obs + components.corr + components.trans)


def test_zero_residual_artificial_case_has_zero_loss() -> None:
    A_plus = np.array([[1.0], [2.0], [4.0]])
    A_minus = A_plus.copy()
    V_A = np.array([[2.0]])
    V_B = np.array([[3.0], [-1.0]])
    Z_B = A_plus @ V_B.T
    A_minus[1:] = A_plus[:-1] @ V_A.T
    A_minus[0] = A_plus[0]
    a_init = A_minus[0].copy()

    components = compute_objective_components(
        Z_B,
        A_minus,
        A_plus,
        V_A,
        V_B,
        a_init,
        sigma_b2=1.0,
        sigma_a2=1.0,
        p=1.0,
        c0=1.0,
    )

    assert components.init == pytest.approx(0.0)
    assert components.obs == pytest.approx(0.0)
    assert components.trans == pytest.approx(0.0)


def test_compute_total_rejects_negative_component() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        compute_total_objective(init=-1.0, obs=0.0, corr=0.0, trans=0.0)


def test_objective_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="share T"):
        compute_objective_components(
            Z_B=np.zeros((3, 2)),
            A_train_minus=np.zeros((2, 1)),
            A_train_plus=np.zeros((3, 1)),
            V_A=np.zeros((1, 1)),
            V_B=np.zeros((2, 1)),
            a_init=np.zeros(1),
            sigma_b2=1.0,
            sigma_a2=1.0,
            p=1.0,
            c0=1.0,
        )


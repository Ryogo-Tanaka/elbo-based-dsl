from __future__ import annotations

import numpy as np
import pytest

from dsl_elto.operators import update_V_A, update_V_B


def test_operator_update_shapes() -> None:
    rng = np.random.default_rng(0)
    A_plus = rng.normal(size=(8, 3))
    A_minus = rng.normal(size=(8, 3))
    Z_B = rng.normal(size=(8, 4))

    V_B = update_V_B(Z_B, A_plus, lambda_b=1.0e-3)
    V_A = update_V_A(A_minus, A_plus, lambda_a=1.0e-3)

    assert V_B.shape == (4, 3)
    assert V_A.shape == (3, 3)
    assert np.all(np.isfinite(V_B))
    assert np.all(np.isfinite(V_A))


def test_update_V_B_recovers_known_linear_map_with_small_ridge() -> None:
    rng = np.random.default_rng(1)
    A_plus = rng.normal(size=(200, 2))
    V_true = np.array([[2.0, -1.0], [0.5, 1.5], [-0.25, 0.75]])
    Z_B = A_plus @ V_true.T

    V_B = update_V_B(Z_B, A_plus, lambda_b=1.0e-10)

    np.testing.assert_allclose(V_B, V_true, rtol=1.0e-7, atol=1.0e-7)


def test_update_V_A_recovers_known_linear_map_with_small_ridge() -> None:
    rng = np.random.default_rng(2)
    A_plus = rng.normal(size=(201, 2))
    V_true = np.array([[0.9, 0.2], [-0.1, 0.8]])
    A_minus = np.zeros_like(A_plus)
    A_minus[1:] = A_plus[:-1] @ V_true.T

    V_A = update_V_A(A_minus, A_plus, lambda_a=1.0e-10)

    np.testing.assert_allclose(V_A, V_true, rtol=1.0e-7, atol=1.0e-7)


def test_larger_lambda_shrinks_operator_norm() -> None:
    rng = np.random.default_rng(3)
    A_plus = rng.normal(size=(80, 3))
    V_true = np.array([[1.0, 0.5, -0.5], [0.2, 1.5, 0.25]])
    Z_B = A_plus @ V_true.T

    small = update_V_B(Z_B, A_plus, lambda_b=1.0e-8)
    large = update_V_B(Z_B, A_plus, lambda_b=10.0)

    assert np.linalg.norm(large) < np.linalg.norm(small)


def test_operator_updates_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="at least two time steps"):
        update_V_A(np.zeros((1, 2)), np.zeros((1, 2)), lambda_a=1.0e-3)

    with pytest.raises(ValueError, match="lambda_b"):
        update_V_B(np.zeros((2, 2)), np.zeros((2, 2)), lambda_b=0.0)


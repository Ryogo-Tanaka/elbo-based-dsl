from __future__ import annotations

import numpy as np
import pytest

from dsl_elto.coordinates import update_A_minus, update_A_plus


def test_coordinate_update_shapes_and_finite_values() -> None:
    rng = np.random.default_rng(0)
    Z_B = rng.normal(size=(5, 3))
    A_minus = rng.normal(size=(5, 2))
    V_A = np.array([[0.8, 0.1], [-0.2, 0.7]])
    V_B = rng.normal(size=(3, 2))
    a_init = np.zeros(2)

    A_plus = update_A_plus(Z_B, A_minus, V_A, V_B, sigma_b2=1.0, sigma_a2=1.5, p=2.0)
    A_minus_new = update_A_minus(A_plus, V_A, a_init, sigma_a2=1.5, p=2.0, c0=10.0)

    assert A_plus.shape == (5, 2)
    assert A_minus_new.shape == (5, 2)
    assert np.all(np.isfinite(A_plus))
    assert np.all(np.isfinite(A_minus_new))


def test_update_A_plus_matches_scalar_hand_computation() -> None:
    Z_B = np.array([[2.0], [4.0]])
    A_minus = np.array([[1.0], [3.0]])
    V_A = np.array([[2.0]])
    V_B = np.array([[1.5]])
    sigma_b2 = 2.0
    sigma_a2 = 4.0
    p = 5.0

    A_plus = update_A_plus(Z_B, A_minus, V_A, V_B, sigma_b2, sigma_a2, p)

    w_B = 1.0 / sigma_b2
    w_A = 1.0 / sigma_a2
    w_P = 1.0 / p
    H_0 = w_B * V_B[0, 0] ** 2 + w_P + w_A * V_A[0, 0] ** 2
    h_0 = w_B * V_B[0, 0] * Z_B[0, 0] + w_P * A_minus[0, 0] + w_A * V_A[0, 0] * A_minus[1, 0]
    H_1 = w_B * V_B[0, 0] ** 2 + w_P
    h_1 = w_B * V_B[0, 0] * Z_B[1, 0] + w_P * A_minus[1, 0]

    np.testing.assert_allclose(A_plus[:, 0], np.array([h_0 / H_0, h_1 / H_1]))


def test_final_time_posterior_update_omits_future_transfer_term() -> None:
    Z_B = np.array([[0.0], [4.0]])
    A_minus_base = np.array([[0.0], [3.0]])
    A_minus_changed_future = np.array([[0.0], [300.0]])
    V_A = np.array([[100.0]])
    V_B = np.array([[1.5]])

    base = update_A_plus(Z_B, A_minus_base, V_A, V_B, sigma_b2=2.0, sigma_a2=4.0, p=5.0)
    changed = update_A_plus(Z_B, A_minus_changed_future, V_A, V_B, sigma_b2=2.0, sigma_a2=4.0, p=5.0)

    assert base[-1, 0] != pytest.approx(changed[-1, 0])
    expected_changed_final = ((1.0 / 2.0) * 1.5 * 4.0 + (1.0 / 5.0) * 300.0) / (
        (1.0 / 2.0) * 1.5**2 + (1.0 / 5.0)
    )
    assert changed[-1, 0] == pytest.approx(expected_changed_final)


def test_update_A_minus_matches_scalar_hand_computation() -> None:
    A_plus = np.array([[2.0], [6.0]])
    V_A = np.array([[3.0]])
    a_init = np.array([10.0])
    sigma_a2 = 4.0
    p = 2.0
    c0 = 5.0

    A_minus = update_A_minus(A_plus, V_A, a_init, sigma_a2, p, c0)

    w_A = 1.0 / sigma_a2
    w_P = 1.0 / p
    w_0 = 1.0 / (c0 * p)
    expected_first = (w_P * 2.0 + w_0 * 10.0) / (w_P + w_0)
    expected_second = (w_P * 6.0 + w_A * 3.0 * 2.0) / (w_P + w_A)
    np.testing.assert_allclose(A_minus[:, 0], np.array([expected_first, expected_second]))


def test_coordinate_updates_reject_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="share T"):
        update_A_plus(
            Z_B=np.zeros((3, 2)),
            A_minus=np.zeros((2, 1)),
            V_A=np.zeros((1, 1)),
            V_B=np.zeros((2, 1)),
            sigma_b2=1.0,
            sigma_a2=1.0,
            p=1.0,
        )

    with pytest.raises(ValueError, match="a_init"):
        update_A_minus(np.zeros((3, 2)), np.zeros((2, 2)), np.zeros(3), sigma_a2=1.0, p=1.0, c0=1.0)


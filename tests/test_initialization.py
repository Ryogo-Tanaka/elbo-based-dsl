from __future__ import annotations

import numpy as np
import pytest

from dsl_elto.initialization import initialize_warm_start


def test_warm_start_shapes_and_finite_values() -> None:
    rng = np.random.default_rng(0)
    Z_B = rng.normal(size=(12, 5))
    d_A = 3
    a_init = np.array([0.1, -0.2, 0.3])

    result = initialize_warm_start(
        Z_B=Z_B,
        d_A=d_A,
        a_init=a_init,
        sigma_b2=1.0,
        p=2.0,
        seed=7,
        eps=1.0e-8,
    )

    assert result.A_train_minus_init.shape == (12, d_A)
    assert result.A_train_plus_init.shape == (12, d_A)
    assert result.V_A_init.shape == (d_A, d_A)
    assert result.V_B_init.shape == (5, d_A)
    assert np.all(np.isfinite(result.A_train_minus_init))
    assert np.all(np.isfinite(result.A_train_plus_init))
    assert np.all(np.isfinite(result.V_A_init))
    assert np.all(np.isfinite(result.V_B_init))


def test_warm_start_initial_prior_and_nonzero_posterior() -> None:
    Z_B = np.array(
        [
            [0.5, -1.0],
            [1.5, 0.25],
            [-0.75, 2.0],
            [0.25, -0.5],
        ]
    )
    a_init = np.zeros(2)

    result = initialize_warm_start(
        Z_B=Z_B,
        d_A=2,
        a_init=a_init,
        sigma_b2=1.0,
        p=1.0,
        seed=11,
        eps=1.0e-8,
    )

    np.testing.assert_allclose(result.A_train_minus_init[0], a_init)
    assert not np.allclose(result.A_train_plus_init, 0.0)
    np.testing.assert_allclose(result.A_train_minus_init[1:], result.A_train_plus_init[1:])
    np.testing.assert_allclose(result.V_A_init, np.zeros((2, 2)))


def test_warm_start_reproducible_with_same_seed() -> None:
    rng = np.random.default_rng(2)
    Z_B = rng.normal(size=(6, 4))
    kwargs = dict(Z_B=Z_B, d_A=3, a_init=np.zeros(3), sigma_b2=1.5, p=0.75, seed=5, eps=1.0e-8)

    first = initialize_warm_start(**kwargs)
    second = initialize_warm_start(**kwargs)

    np.testing.assert_allclose(first.A_train_minus_init, second.A_train_minus_init)
    np.testing.assert_allclose(first.A_train_plus_init, second.A_train_plus_init)
    np.testing.assert_allclose(first.V_A_init, second.V_A_init)
    np.testing.assert_allclose(first.V_B_init, second.V_B_init)


def test_warm_start_rejects_invalid_inputs() -> None:
    Z_B = np.zeros((3, 2))

    with pytest.raises(ValueError, match="a_init"):
        initialize_warm_start(Z_B, d_A=2, a_init=np.zeros(3), sigma_b2=1.0, p=1.0, seed=0, eps=1.0e-8)

    with pytest.raises(ValueError, match="d_A"):
        initialize_warm_start(Z_B, d_A=0, a_init=np.zeros(0), sigma_b2=1.0, p=1.0, seed=0, eps=1.0e-8)

    with pytest.raises(ValueError, match="at least one time step"):
        initialize_warm_start(
            np.zeros((0, 2)),
            d_A=2,
            a_init=np.zeros(2),
            sigma_b2=1.0,
            p=1.0,
            seed=0,
            eps=1.0e-8,
        )


def test_warm_start_uses_zero_reference_when_a_init_is_none() -> None:
    Z_B = np.array([[1.0, -1.0], [0.5, 0.25]])

    result = initialize_warm_start(
        Z_B=Z_B,
        d_A=2,
        a_init=None,
        sigma_b2=1.0,
        p=1.0,
        seed=3,
        eps=1.0e-8,
    )

    np.testing.assert_allclose(result.A_train_minus_init[0], np.zeros(2))


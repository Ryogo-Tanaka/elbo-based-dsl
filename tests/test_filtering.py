from __future__ import annotations

from dataclasses import replace

import numpy as np

from dsl_elto.config import SolverConfig, Version0Config
from dsl_elto.filtering import run_causal_filter
from dsl_elto.solver import fit_core


def test_causal_filter_shapes_and_initial_prior() -> None:
    Z_B = np.array([[1.0, -1.0], [0.5, 0.25], [-0.25, 0.75], [0.0, 1.0]])
    V_A = np.array([[0.8, 0.1], [-0.2, 0.7]])
    V_B = np.array([[1.0, 0.5], [-0.25, 0.75]])
    a_init = np.array([0.2, -0.1])

    result = run_causal_filter(Z_B, V_A, V_B, a_init, sigma_b2=1.5, p=0.8)

    assert result.A_filter_minus.shape == (4, 2)
    assert result.A_filter_plus.shape == (4, 2)
    assert result.ZB_pred_prior.shape == (4, 2)
    assert result.innovations.shape == (4, 2)
    assert result.correction_norms.shape == (4,)
    assert result.K.shape == (2, 2)
    np.testing.assert_allclose(result.A_filter_minus[0], a_init)
    assert np.all(np.isfinite(result.A_filter_minus))
    assert np.all(np.isfinite(result.A_filter_plus))
    assert np.all(np.isfinite(result.ZB_pred_prior))


def test_prediction_embedding_is_computed_from_A_filter_minus() -> None:
    Z_B = np.array([[1.0], [2.0], [3.0]])
    V_A = np.array([[0.5]])
    V_B = np.array([[2.0]])
    a_init = np.array([1.0])

    result = run_causal_filter(Z_B, V_A, V_B, a_init, sigma_b2=0.5, p=1.0)

    np.testing.assert_allclose(result.ZB_pred_prior, result.A_filter_minus @ V_B.T)
    posterior_reconstruction = result.A_filter_plus @ V_B.T
    assert not np.allclose(result.ZB_pred_prior, posterior_reconstruction)


def test_filter_arrays_are_distinct_from_each_other_and_training_arrays() -> None:
    rng = np.random.default_rng(0)
    Z_B = rng.normal(size=(14, 3))
    config = Version0Config()
    config = replace(
        config,
        model=replace(config.model, d_A=2),
        solver=SolverConfig(
            max_outer_iter=3,
            tol_rel_loss=1.0e-300,
            min_outer_iter=2,
            check_finite=True,
            fail_on_nan=True,
            fail_on_inf=True,
            max_abs_A=1.0e6,
            min_rms_A=1.0e-12,
        ),
    )
    core = fit_core(Z_B, config)
    filtered = run_causal_filter(
        Z_B,
        core.V_A,
        core.V_B,
        core.a_init,
        sigma_b2=config.metrics.sigma_b2,
        p=config.metrics.p,
    )

    assert filtered.A_filter_minus is not filtered.A_filter_plus
    assert filtered.A_filter_minus is not core.A_train_minus
    assert filtered.A_filter_plus is not core.A_train_plus
    assert not np.shares_memory(filtered.A_filter_minus, core.A_train_minus)
    assert not np.shares_memory(filtered.A_filter_plus, core.A_train_plus)

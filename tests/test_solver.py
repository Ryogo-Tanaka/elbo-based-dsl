from __future__ import annotations

from dataclasses import replace

import numpy as np

from dsl_elto.config import SolverConfig, Version0Config
from dsl_elto.solver import fit_core


def _small_config() -> Version0Config:
    config = Version0Config()
    return replace(
        config,
        model=replace(config.model, d_A=3),
        solver=SolverConfig(
            max_outer_iter=4,
            tol_rel_loss=1.0e-300,
            min_outer_iter=3,
            check_finite=True,
            fail_on_nan=True,
            fail_on_inf=True,
            max_abs_A=1.0e6,
            min_rms_A=1.0e-12,
        ),
    )


def test_fit_core_smoke_outputs_finite_training_arrays_and_logs_objectives() -> None:
    rng = np.random.default_rng(0)
    Z_B = rng.normal(size=(20, 4))
    config = _small_config()

    result = fit_core(Z_B, config)

    assert result.A_train_minus.shape == (20, 3)
    assert result.A_train_plus.shape == (20, 3)
    assert result.V_A.shape == (3, 3)
    assert result.V_B.shape == (4, 3)
    assert result.a_init.shape == (3,)
    assert np.all(np.isfinite(result.A_train_minus))
    assert np.all(np.isfinite(result.A_train_plus))
    assert np.all(np.isfinite(result.V_A))
    assert np.all(np.isfinite(result.V_B))
    assert len(result.objective_log) == result.diagnostics["num_outer_iter"]
    assert len(result.objective_log) == config.solver.max_outer_iter
    assert result.diagnostics["stop_reason"] == "max_outer_iter"
    assert result.objective_log[-1].total == result.diagnostics["final_total"]


def test_fit_core_keeps_train_arrays_distinct_and_returns_no_filter_arrays() -> None:
    rng = np.random.default_rng(1)
    Z_B = rng.normal(size=(12, 3))
    result = fit_core(Z_B, _small_config())

    assert result.A_train_minus is not result.A_train_plus
    assert not hasattr(result, "A_filter_minus")
    assert not hasattr(result, "A_filter_plus")


def test_fit_core_uses_provided_initial_coordinate() -> None:
    rng = np.random.default_rng(2)
    Z_B = rng.normal(size=(10, 3))
    a_init = np.array([0.1, -0.2, 0.3])

    result = fit_core(Z_B, _small_config(), a_init=a_init)

    np.testing.assert_allclose(result.a_init, a_init)

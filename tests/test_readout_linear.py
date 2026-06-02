from __future__ import annotations

import numpy as np
import pytest

from dsl_elto.readout import fit_linear_readout, predict_linear_readout


def test_linear_ridge_readout_shape() -> None:
    rng = np.random.default_rng(0)
    H_prior = rng.normal(size=(20, 3))
    Z_M = rng.normal(size=(20, 2))

    R_readout = fit_linear_readout(H_prior, Z_M, lambda_readout=1.0e-3)
    ZM_pred = predict_linear_readout(H_prior, R_readout)

    assert R_readout.shape == (2, 3)
    assert ZM_pred.shape == (20, 2)
    assert np.all(np.isfinite(R_readout))
    assert np.all(np.isfinite(ZM_pred))


def test_linear_ridge_recovers_toy_linear_map() -> None:
    rng = np.random.default_rng(1)
    H_prior = rng.normal(size=(80, 3))
    R_true = np.array([[1.0, -0.5, 0.25], [0.2, 0.75, -1.25]])
    Z_M = H_prior @ R_true.T

    R_readout = fit_linear_readout(H_prior, Z_M, lambda_readout=1.0e-10)
    ZM_pred = predict_linear_readout(H_prior, R_readout)

    np.testing.assert_allclose(R_readout, R_true, rtol=1.0e-8, atol=1.0e-8)
    np.testing.assert_allclose(ZM_pred, Z_M, rtol=1.0e-8, atol=1.0e-8)


def test_linear_readout_rejects_invalid_shape_and_nonfinite_input() -> None:
    with pytest.raises(ValueError, match="rank|2D"):
        fit_linear_readout(np.zeros(3), np.zeros((3, 1)), lambda_readout=1.0e-3)

    with pytest.raises(ValueError, match="share T"):
        fit_linear_readout(np.zeros((3, 2)), np.zeros((4, 1)), lambda_readout=1.0e-3)

    with pytest.raises(ValueError, match="finite"):
        fit_linear_readout(np.array([[0.0], [np.inf]]), np.zeros((2, 1)), lambda_readout=1.0e-3)

    with pytest.raises(ValueError, match="positive"):
        fit_linear_readout(np.zeros((2, 1)), np.zeros((2, 1)), lambda_readout=0.0)

    with pytest.raises(ValueError, match="shape mismatch"):
        predict_linear_readout(np.zeros((2, 3)), np.zeros((1, 2)))

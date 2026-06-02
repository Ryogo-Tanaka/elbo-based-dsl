from __future__ import annotations

import numpy as np
import pytest

from dsl_elto.preprocessing import (
    fit_standardizer,
    inverse_transform_standardizer,
    standardize_features,
    transform_standardizer,
)


def test_standardized_mean_is_nearly_zero_and_rms_nearly_one() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4)) * np.array([1.0, 2.0, 3.0, 4.0]) + np.array([2.0, -1.0, 0.5, 4.0])

    stats = fit_standardizer(X, eps=1.0e-12)
    Z = transform_standardizer(X, stats)
    rms = np.sqrt(np.mean(Z * Z, axis=0))

    np.testing.assert_allclose(Z.mean(axis=0), np.zeros(4), atol=1.0e-12)
    np.testing.assert_allclose(rms, np.ones(4), atol=1.0e-10)


def test_inverse_transform_reconstructs_input() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 3))

    stats = fit_standardizer(X, eps=1.0e-8)
    Z = transform_standardizer(X, stats)
    X_recovered = inverse_transform_standardizer(Z, stats)

    np.testing.assert_allclose(X_recovered, X, rtol=1.0e-12, atol=1.0e-12)


def test_test_time_transform_uses_saved_training_statistics() -> None:
    train = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    test = np.array([[10.0, 11.0], [12.0, 13.0]])

    stats = fit_standardizer(train, eps=1.0e-8)
    Z_test = transform_standardizer(test, stats)

    expected = (test - stats.mean) / stats.scale
    np.testing.assert_allclose(Z_test, expected)
    assert not np.allclose(Z_test.mean(axis=0), 0.0)


def test_standardize_features_returns_B_and_M_results() -> None:
    B = np.arange(12.0).reshape(4, 3)
    M = np.arange(8.0).reshape(4, 2)

    result = standardize_features(B, M, eps=1.0e-8)

    assert result.Z_B.shape == (4, 3)
    assert result.Z_M.shape == (4, 2)
    assert result.b_stats.mean.shape == (3,)
    assert result.m_stats.scale.shape == (2,)
    np.testing.assert_allclose(inverse_transform_standardizer(result.Z_M, result.m_stats), M)


def test_preprocessing_rejects_invalid_shape_and_nonfinite_input() -> None:
    with pytest.raises(ValueError, match="rank"):
        fit_standardizer(np.zeros(3), eps=1.0e-8)

    with pytest.raises(ValueError, match="finite"):
        fit_standardizer(np.array([[0.0], [np.nan]]), eps=1.0e-8)

    with pytest.raises(ValueError, match="positive"):
        fit_standardizer(np.zeros((2, 1)), eps=0.0)


def test_standardize_features_rejects_mismatched_time_dimension() -> None:
    B = np.zeros((3, 2))
    M = np.zeros((4, 2))

    with pytest.raises(ValueError, match="share T"):
        standardize_features(B, M, eps=1.0e-8)


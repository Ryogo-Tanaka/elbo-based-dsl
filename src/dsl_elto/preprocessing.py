"""Feature preprocessing for DSL/ELTO Version 0.

All feature arrays use the code convention `[T, d]`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .shapes import assert_same_T, assert_shape, check_finite


@dataclass(frozen=True)
class Standardizer:
    """Training statistics for center + RMS feature standardization."""

    mean: np.ndarray
    scale: np.ndarray
    eps: float


@dataclass(frozen=True)
class PreprocessResult:
    """Standardized observation and encoder features with their statistics."""

    Z_B: np.ndarray
    Z_M: np.ndarray
    b_stats: Standardizer
    m_stats: Standardizer


def fit_standardizer(X: np.ndarray, eps: float) -> Standardizer:
    """Fit center + RMS scale statistics from a `[T, d]` training array."""

    X = check_finite("X", assert_shape("X", X, (None, None)))
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one time step")
    if eps <= 0:
        raise ValueError("eps must be positive")

    mean = X.mean(axis=0)
    centered = X - mean
    scale = np.sqrt(np.mean(centered * centered, axis=0) + eps)
    return Standardizer(mean=mean, scale=scale, eps=eps)


def transform_standardizer(X: np.ndarray, stats: Standardizer) -> np.ndarray:
    """Standardize a `[T, d]` array using saved training statistics."""

    X = check_finite("X", assert_shape("X", X, (None, stats.mean.shape[0])))
    _validate_stats(stats)
    return (X - stats.mean) / stats.scale


def inverse_transform_standardizer(Z: np.ndarray, stats: Standardizer) -> np.ndarray:
    """Undo standardization with saved training statistics."""

    Z = check_finite("Z", assert_shape("Z", Z, (None, stats.mean.shape[0])))
    _validate_stats(stats)
    return Z * stats.scale + stats.mean


def standardize_features(B: np.ndarray, M: np.ndarray, eps: float) -> PreprocessResult:
    """Fit training statistics for `B` and `M`, then standardize both arrays."""

    assert_same_T(("B", B), ("M", M))
    b_stats = fit_standardizer(B, eps)
    m_stats = fit_standardizer(M, eps)
    Z_B = transform_standardizer(B, b_stats)
    Z_M = transform_standardizer(M, m_stats)
    return PreprocessResult(Z_B=Z_B, Z_M=Z_M, b_stats=b_stats, m_stats=m_stats)


def _validate_stats(stats: Standardizer) -> None:
    mean = check_finite("stats.mean", stats.mean)
    scale = check_finite("stats.scale", stats.scale)
    if mean.ndim != 1:
        raise ValueError(f"stats.mean must have shape [d], got {mean.shape}")
    if scale.shape != mean.shape:
        raise ValueError(f"stats.scale must have shape {mean.shape}, got {scale.shape}")
    if np.any(scale <= 0):
        raise ValueError("stats.scale must be strictly positive")
    if stats.eps <= 0:
        raise ValueError("stats.eps must be positive")


"""Prediction adapters for DSL/ELTO Version 0 readouts."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .preprocessing import Standardizer, inverse_transform_standardizer
from .shapes import assert_shape, check_finite


def unstandardize_M(ZM_pred: np.ndarray, m_stats: Standardizer) -> np.ndarray:
    """Map standardized predicted target features back to feature coordinates."""

    return inverse_transform_standardizer(ZM_pred, m_stats)


def decode_predictions(
    M_pred: np.ndarray,
    decoder: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Optionally pass predicted features through a fixed decoder adapter."""

    M_pred = check_finite("M_pred", assert_shape("M_pred", M_pred, (None, None)))
    if decoder is None:
        return M_pred
    if not callable(decoder):
        raise TypeError("decoder must be callable or None")
    decoded = decoder(M_pred)
    return check_finite("decoded", np.asarray(decoded))

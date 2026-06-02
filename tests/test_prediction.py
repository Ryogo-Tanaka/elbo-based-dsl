from __future__ import annotations

import numpy as np
import pytest

from dsl_elto.prediction import decode_predictions, unstandardize_M
from dsl_elto.preprocessing import Standardizer


def test_unstandardize_M_matches_saved_training_statistics() -> None:
    ZM_pred = np.array([[0.0, 1.0], [-1.0, 2.0], [0.5, -0.5]])
    m_stats = Standardizer(mean=np.array([10.0, -2.0]), scale=np.array([2.0, 0.5]), eps=1.0e-8)

    M_pred = unstandardize_M(ZM_pred, m_stats)

    assert M_pred.shape == (3, 2)
    np.testing.assert_allclose(M_pred, m_stats.mean + m_stats.scale * ZM_pred)


def test_decode_predictions_none_returns_input_features() -> None:
    M_pred = np.array([[1.0, 2.0], [3.0, 4.0]])

    decoded = decode_predictions(M_pred)

    assert decoded is M_pred
    np.testing.assert_allclose(decoded, M_pred)


def test_decode_predictions_uses_callable_decoder_adapter() -> None:
    M_pred = np.array([[1.0, 2.0], [3.0, 4.0]])

    decoded = decode_predictions(M_pred, decoder=lambda x: x[:, :1] + x[:, 1:])

    assert decoded.shape == (2, 1)
    np.testing.assert_allclose(decoded, np.array([[3.0], [7.0]]))


def test_decode_predictions_rejects_invalid_decoder_and_nonfinite_output() -> None:
    with pytest.raises(TypeError, match="callable"):
        decode_predictions(np.zeros((2, 1)), decoder=object())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="finite"):
        decode_predictions(np.zeros((2, 1)), decoder=lambda x: np.array([[np.nan]]))

from __future__ import annotations

import numpy as np
import pytest

from dsl_elto.shapes import (
    assert_2d,
    assert_last_dim,
    assert_same_T,
    assert_shape,
    check_finite,
    make_eye,
)


def test_assert_2d_accepts_T_by_d_array() -> None:
    x = np.zeros((5, 3))

    checked = assert_2d("Z_B", x)

    assert checked.shape == (5, 3)


def test_assert_2d_rejects_vector() -> None:
    with pytest.raises(ValueError, match=r"\[T, d\]"):
        assert_2d("B", np.zeros(3))


def test_assert_shape_supports_exact_shape_and_wildcard() -> None:
    x = np.zeros((7, 4))

    assert assert_shape("A_train_plus", x, (7, 4)).shape == (7, 4)
    assert assert_shape("A_train_plus", x, (None, 4)).shape == (7, 4)

    with pytest.raises(ValueError, match="shape mismatch"):
        assert_shape("A_train_plus", x, (8, 4))


def test_assert_same_T_returns_shared_time_dimension() -> None:
    B = np.zeros((6, 2))
    M = np.zeros((6, 3))

    assert assert_same_T(("B", B), ("M", M)) == 6

    with pytest.raises(ValueError, match="M has T=5"):
        assert_same_T(("B", B), ("M", np.zeros((5, 3))))


def test_assert_last_dim_checks_feature_dimension() -> None:
    A_filter_minus = np.zeros((10, 8))

    assert assert_last_dim("A_filter_minus", A_filter_minus, 8).shape == (10, 8)

    with pytest.raises(ValueError, match="last dimension"):
        assert_last_dim("A_filter_minus", A_filter_minus, 7)


def test_make_eye_and_check_finite() -> None:
    eye = make_eye(3, dtype=np.float64)

    assert eye.shape == (3, 3)
    assert eye.dtype == np.float64
    assert check_finite("V_A", eye).shape == (3, 3)

    with pytest.raises(ValueError, match="positive integer"):
        make_eye(0)

    with pytest.raises(ValueError, match="finite"):
        check_finite("bad", np.array([0.0, np.inf]))


"""Shape utilities for `[T, d]` DSL/ELTO Version 0 arrays."""

from __future__ import annotations

from typing import Any

import numpy as np


def assert_2d(name: str, x: Any) -> np.ndarray:
    """Return `x` as an array after checking it is rank-2."""

    array = np.asarray(x)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array with shape [T, d], got shape {array.shape}")
    return array


def assert_shape(name: str, x: Any, expected: tuple[int | None, ...]) -> np.ndarray:
    """Return `x` as an array after checking exact shape, with `None` as wildcard."""

    array = np.asarray(x)
    if array.ndim != len(expected):
        raise ValueError(f"{name} must have rank {len(expected)}, got shape {array.shape}")

    mismatches = []
    for axis, (actual_dim, expected_dim) in enumerate(zip(array.shape, expected, strict=True)):
        if expected_dim is not None and actual_dim != expected_dim:
            mismatches.append(f"axis {axis}: expected {expected_dim}, got {actual_dim}")
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(f"{name} shape mismatch: expected {expected}, got {array.shape} ({details})")
    return array


def assert_same_T(*arrays: Any) -> int:
    """Check that all `[T, d]` arrays share the same leading time dimension."""

    if not arrays:
        raise ValueError("assert_same_T requires at least one array")

    checked = [assert_2d(_array_name(index, item), _array_value(item)) for index, item in enumerate(arrays)]
    expected_T = checked[0].shape[0]
    for index, array in enumerate(checked[1:], start=1):
        if array.shape[0] != expected_T:
            raise ValueError(
                f"all arrays must share T={expected_T}; "
                f"{_array_name(index, arrays[index])} has T={array.shape[0]}"
            )
    return expected_T


def assert_last_dim(name: str, x: Any, d: int) -> np.ndarray:
    """Return `x` as a `[T, d]` array after checking its final dimension."""

    array = assert_2d(name, x)
    if array.shape[-1] != d:
        raise ValueError(f"{name} last dimension must be {d}, got shape {array.shape}")
    return array


def make_eye(d: int, dtype: Any = None) -> np.ndarray:
    """Create a square identity matrix for operator and ridge solves."""

    if not isinstance(d, int) or d <= 0:
        raise ValueError(f"d must be a positive integer, got {d!r}")
    return np.eye(d, dtype=dtype)


def check_finite(name: str, x: Any) -> np.ndarray:
    """Return `x` as an array after checking all values are finite."""

    array = np.asarray(x)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _array_name(index: int, item: Any) -> str:
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
        return item[0]
    return f"array_{index}"


def _array_value(item: Any) -> Any:
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
        return item[1]
    return item


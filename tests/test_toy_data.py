from __future__ import annotations

import numpy as np
import pytest

from dsl_elto.toy_data import generate_damped_rotation


def test_damped_rotation_reproducible_and_has_expected_shapes() -> None:
    result_a = generate_damped_rotation(
        T=12,
        radius=0.95,
        omega=0.2,
        process_noise_std=0.01,
        observation_noise_std=0.02,
        seed=7,
    )
    result_b = generate_damped_rotation(
        T=12,
        radius=0.95,
        omega=0.2,
        process_noise_std=0.01,
        observation_noise_std=0.02,
        seed=7,
    )

    assert result_a.X_true.shape == (12, 2)
    assert result_a.Y.shape == (12, 3)
    assert result_a.M.shape == (12, 3)
    assert result_a.B.shape == (12, 3)
    np.testing.assert_allclose(result_a.X_true, result_b.X_true)
    np.testing.assert_allclose(result_a.Y, result_b.Y)
    np.testing.assert_allclose(result_a.M, result_a.Y)
    np.testing.assert_allclose(result_a.B, result_a.Y)
    assert result_a.M is not result_a.Y
    assert result_a.B is not result_a.Y
    assert np.all(np.isfinite(result_a.X_true))
    assert np.all(np.isfinite(result_a.Y))


def test_damped_rotation_without_noise_matches_observation_formula() -> None:
    result = generate_damped_rotation(
        T=5,
        radius=0.9,
        omega=0.1,
        process_noise_std=0.0,
        observation_noise_std=0.0,
        seed=0,
    )

    expected_Y = np.column_stack(
        [result.X_true[:, 0], result.X_true[:, 1], result.X_true[:, 0] * result.X_true[:, 0]]
    )
    np.testing.assert_allclose(result.Y, expected_Y)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"T": 1}, "T"),
        ({"radius": 1.0}, "radius"),
        ({"radius": 0.0}, "radius"),
        ({"omega": np.inf}, "omega"),
        ({"process_noise_std": -1.0}, "process_noise_std"),
        ({"observation_noise_std": -1.0}, "observation_noise_std"),
    ],
)
def test_damped_rotation_rejects_invalid_inputs(kwargs: dict[str, float], match: str) -> None:
    params = {
        "T": 10,
        "radius": 0.95,
        "omega": 0.2,
        "process_noise_std": 0.01,
        "observation_noise_std": 0.02,
        "seed": 0,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=match):
        generate_damped_rotation(**params)

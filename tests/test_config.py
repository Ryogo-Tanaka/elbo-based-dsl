from __future__ import annotations

from pathlib import Path

import pytest

from dsl_elto.config import ConfigValidationError, Version0Config, load_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_config_loads() -> None:
    config = load_config(REPO_ROOT / "configs" / "version0_default.yaml")

    assert isinstance(config, Version0Config)
    assert config.scope.single_trajectory is True
    assert config.features.feature_array_layout == "T_by_d"
    assert config.solver.min_outer_iter == 3
    assert config.readout.type == "linear_ridge"


def test_toy_readout_configs_load() -> None:
    linear = load_config(REPO_ROOT / "configs" / "toy_sanity_linear.yaml")
    mlp = load_config(REPO_ROOT / "configs" / "toy_sanity_mlp.yaml")

    assert linear.readout.type == "linear_ridge"
    assert mlp.readout.type == "mlp"
    assert linear.solver.min_outer_iter == 3
    assert mlp.solver.min_outer_iter == 3


def test_invalid_readout_type_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_readout.yaml"
    config_path.write_text("readout:\n  type: posterior_reconstruction\n", encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="readout.type"):
        load_config(config_path)


def test_unsupported_svd_init_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "svd.yaml"
    config_path.write_text("scope:\n  allow_svd_init: true\n", encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="allow_svd_init"):
        load_config(config_path)


def test_negative_metric_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "negative_metric.yaml"
    config_path.write_text("metrics:\n  sigma_b2: -1.0\n", encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="metrics.sigma_b2"):
        load_config(config_path)


"""Configuration schema and validation for DSL/ELTO Version 0."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigValidationError(ValueError):
    """Raised when a Version 0 config requests unsupported behavior."""


@dataclass(frozen=True)
class ProjectConfig:
    name: str = "dsl_elto_version0"
    random_seed: int = 0
    dtype: str = "float64"
    device: str = "cpu"


@dataclass(frozen=True)
class ScopeConfig:
    single_trajectory: bool = True
    allow_multiple_trajectories: bool = False
    allow_svd_init: bool = False
    allow_direct_B_to_A_init: bool = False
    allow_inference_encoder: bool = False
    allow_residual_calibrated_metrics: bool = False
    allow_kkr_covariance_recursion: bool = False
    allow_encoder_refinement: bool = False
    allow_posterior_reconstruction_loss: bool = False


@dataclass(frozen=True)
class FeatureConfig:
    standardize: bool = True
    eps: float = 1.0e-8
    use_precomputed_features: bool = True
    feature_array_layout: str = "T_by_d"


@dataclass(frozen=True)
class ModelConfig:
    d_A: int = 8
    d_B: int | None = None
    m: int | None = None


@dataclass(frozen=True)
class MetricConfig:
    sigma_b2: float = 1.0
    sigma_a2: float = 1.0
    p: float = 1.0
    c0: float = 10.0


@dataclass(frozen=True)
class RidgeConfig:
    lambda_a: float = 1.0e-3
    lambda_b: float = 1.0e-3
    lambda_readout: float = 1.0e-3
    normalized_covariance_form: bool = True


@dataclass(frozen=True)
class InitializationConfig:
    type: str = "correction_warm_start"
    vb_seed: str = "random_column_normalized"
    va_seed: str = "zero"
    a_init_mode: str = "zero_reference"
    external_a_init_path: str | None = None
    seed_scale: float = 1.0


@dataclass(frozen=True)
class SolverConfig:
    max_outer_iter: int = 100
    tol_rel_loss: float = 1.0e-6
    min_outer_iter: int = 3
    check_finite: bool = True
    fail_on_nan: bool = True
    fail_on_inf: bool = True
    max_abs_A: float = 1.0e6
    min_rms_A: float = 1.0e-12


@dataclass(frozen=True)
class FilteringConfig:
    use_fixed_gain: bool = True
    use_same_p_all_times: bool = True
    separate_initial_filter_metric: bool = False


@dataclass(frozen=True)
class MLPReadoutConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    activation: str = "relu"
    lr: float = 1.0e-3
    epochs: int = 500
    weight_decay: float = 1.0e-4
    batch_size: int | None = None
    early_stopping: bool = False
    patience: int = 20


@dataclass(frozen=True)
class ReadoutConfig:
    type: str = "linear_ridge"
    train_on_filter_prior: bool = True
    use_posterior_reconstruction: bool = False
    mlp: MLPReadoutConfig = field(default_factory=MLPReadoutConfig)


@dataclass(frozen=True)
class ToyDataConfig:
    type: str = "damped_rotation"
    T: int = 300
    state_dim: int = 2
    obs_dim: int = 3
    radius: float = 0.98
    omega: float = 0.20
    process_noise_std: float = 0.01
    observation_noise_std: float = 0.05
    nonlinear_observation: bool = True


@dataclass(frozen=True)
class LoggingConfig:
    output_dir: str = "outputs/toy_sanity"
    save_config: bool = True
    save_losses_csv: bool = True
    save_metrics_json: bool = True
    save_arrays: bool = False
    save_plots: bool = True
    log_every: int = 1


@dataclass(frozen=True)
class TestsConfig:
    run_slow_mlp_tests: bool = False


@dataclass(frozen=True)
class Version0Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    scope: ScopeConfig = field(default_factory=ScopeConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    metrics: MetricConfig = field(default_factory=MetricConfig)
    ridge: RidgeConfig = field(default_factory=RidgeConfig)
    initialization: InitializationConfig = field(default_factory=InitializationConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    toy_data: ToyDataConfig = field(default_factory=ToyDataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tests: TestsConfig = field(default_factory=TestsConfig)


def load_config(path: str | Path) -> Version0Config:
    """Load a Version 0 YAML config and validate unsupported options."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ConfigValidationError("config root must be a mapping")

    config = _config_from_mapping(raw)
    validate_config(config)
    return config


def validate_config(config: Version0Config) -> None:
    """Validate Version 0 constraints and reject unsupported variants."""

    if not config.scope.single_trajectory:
        raise ConfigValidationError("scope.single_trajectory must be true")
    _require_false("scope.allow_multiple_trajectories", config.scope.allow_multiple_trajectories)
    _require_false("scope.allow_svd_init", config.scope.allow_svd_init)
    _require_false("scope.allow_direct_B_to_A_init", config.scope.allow_direct_B_to_A_init)
    _require_false("scope.allow_inference_encoder", config.scope.allow_inference_encoder)
    _require_false(
        "scope.allow_residual_calibrated_metrics",
        config.scope.allow_residual_calibrated_metrics,
    )
    _require_false("scope.allow_kkr_covariance_recursion", config.scope.allow_kkr_covariance_recursion)
    _require_false("scope.allow_encoder_refinement", config.scope.allow_encoder_refinement)
    _require_false(
        "scope.allow_posterior_reconstruction_loss",
        config.scope.allow_posterior_reconstruction_loss,
    )

    if not config.features.standardize:
        raise ConfigValidationError("features.standardize must be true")
    if config.features.feature_array_layout != "T_by_d":
        raise ConfigValidationError("features.feature_array_layout must be T_by_d")
    if not config.features.use_precomputed_features:
        raise ConfigValidationError("features.use_precomputed_features must be true for Version 0")
    _require_positive("features.eps", config.features.eps)

    _require_positive_int("model.d_A", config.model.d_A)
    _require_optional_positive_int("model.d_B", config.model.d_B)
    _require_optional_positive_int("model.m", config.model.m)

    _require_positive("metrics.sigma_b2", config.metrics.sigma_b2)
    _require_positive("metrics.sigma_a2", config.metrics.sigma_a2)
    _require_positive("metrics.p", config.metrics.p)
    _require_positive("metrics.c0", config.metrics.c0)

    _require_positive("ridge.lambda_a", config.ridge.lambda_a)
    _require_positive("ridge.lambda_b", config.ridge.lambda_b)
    _require_positive("ridge.lambda_readout", config.ridge.lambda_readout)
    if not config.ridge.normalized_covariance_form:
        raise ConfigValidationError("ridge.normalized_covariance_form must be true")

    if config.initialization.type != "correction_warm_start":
        raise ConfigValidationError("initialization.type must be correction_warm_start")
    if config.initialization.vb_seed != "random_column_normalized":
        raise ConfigValidationError("initialization.vb_seed must be random_column_normalized")
    if config.initialization.va_seed != "zero":
        raise ConfigValidationError("initialization.va_seed must be zero")
    _require_positive("initialization.seed_scale", config.initialization.seed_scale)

    _require_positive_int("solver.max_outer_iter", config.solver.max_outer_iter)
    _require_positive_int("solver.min_outer_iter", config.solver.min_outer_iter)
    _require_positive("solver.tol_rel_loss", config.solver.tol_rel_loss)
    _require_positive("solver.max_abs_A", config.solver.max_abs_A)
    _require_positive("solver.min_rms_A", config.solver.min_rms_A)

    if not config.filtering.use_fixed_gain:
        raise ConfigValidationError("filtering.use_fixed_gain must be true")
    if not config.filtering.use_same_p_all_times:
        raise ConfigValidationError("filtering.use_same_p_all_times must be true")
    _require_false(
        "filtering.separate_initial_filter_metric",
        config.filtering.separate_initial_filter_metric,
    )

    if config.readout.type not in {"linear_ridge", "mlp"}:
        raise ConfigValidationError("readout.type must be linear_ridge or mlp")
    if not config.readout.train_on_filter_prior:
        raise ConfigValidationError("readout.train_on_filter_prior must be true")
    _require_false("readout.use_posterior_reconstruction", config.readout.use_posterior_reconstruction)
    _validate_mlp_config(config.readout.mlp)


def _config_from_mapping(raw: Mapping[str, Any]) -> Version0Config:
    allowed = {field.name for field in fields(Version0Config)}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigValidationError(f"unknown top-level config sections: {sorted(unknown)}")

    readout_raw = _section(raw, "readout")
    mlp = _make_dataclass(MLPReadoutConfig, _section(readout_raw, "mlp"), "readout.mlp")
    readout_values = dict(readout_raw)
    readout_values.pop("mlp", None)
    readout = _make_dataclass(ReadoutConfig, readout_values, "readout", mlp=mlp)

    return Version0Config(
        project=_make_dataclass(ProjectConfig, _section(raw, "project"), "project"),
        scope=_make_dataclass(ScopeConfig, _section(raw, "scope"), "scope"),
        features=_make_dataclass(FeatureConfig, _section(raw, "features"), "features"),
        model=_make_dataclass(ModelConfig, _section(raw, "model"), "model"),
        metrics=_make_dataclass(MetricConfig, _section(raw, "metrics"), "metrics"),
        ridge=_make_dataclass(RidgeConfig, _section(raw, "ridge"), "ridge"),
        initialization=_make_dataclass(
            InitializationConfig,
            _section(raw, "initialization"),
            "initialization",
        ),
        solver=_make_dataclass(SolverConfig, _section(raw, "solver"), "solver"),
        filtering=_make_dataclass(FilteringConfig, _section(raw, "filtering"), "filtering"),
        readout=readout,
        toy_data=_make_dataclass(ToyDataConfig, _section(raw, "toy_data"), "toy_data"),
        logging=_make_dataclass(LoggingConfig, _section(raw, "logging"), "logging"),
        tests=_make_dataclass(TestsConfig, _section(raw, "tests"), "tests"),
    )


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f"{name} must be a mapping")
    return value


def _make_dataclass(cls: type[Any], raw: Mapping[str, Any], section_name: str, **overrides: Any) -> Any:
    allowed = {field.name for field in fields(cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigValidationError(f"unknown keys in {section_name}: {sorted(unknown)}")
    values = dict(raw)
    values.update(overrides)
    return cls(**values)


def _require_false(name: str, value: bool) -> None:
    if value:
        raise ConfigValidationError(f"{name} must be false for Version 0")


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ConfigValidationError(f"{name} must be positive")


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ConfigValidationError(f"{name} must be a positive integer")


def _require_optional_positive_int(name: str, value: int | None) -> None:
    if value is not None:
        _require_positive_int(name, value)


def _validate_mlp_config(config: MLPReadoutConfig) -> None:
    _require_positive_int("readout.mlp.hidden_dim", config.hidden_dim)
    _require_positive_int("readout.mlp.num_layers", config.num_layers)
    _require_positive("readout.mlp.lr", config.lr)
    _require_positive_int("readout.mlp.epochs", config.epochs)
    if config.weight_decay < 0:
        raise ConfigValidationError("readout.mlp.weight_decay must be nonnegative")
    _require_optional_positive_int("readout.mlp.batch_size", config.batch_size)
    _require_positive_int("readout.mlp.patience", config.patience)


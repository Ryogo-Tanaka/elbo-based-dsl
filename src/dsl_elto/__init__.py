"""DSL/ELTO Version 0 package scaffold."""

from .config import Version0Config, load_config, validate_config
from .coordinates import update_A_minus, update_A_plus
from .filtering import FilterResult, run_causal_filter
from .initialization import WarmStartResult, initialize_warm_start
from .objective import ObjectiveComponents, compute_objective_components, compute_total_objective
from .operators import update_V_A, update_V_B
from .preprocessing import (
    PreprocessResult,
    Standardizer,
    fit_standardizer,
    inverse_transform_standardizer,
    standardize_features,
    transform_standardizer,
)
from .prediction import decode_predictions, unstandardize_M
from .readout import (
    MLPReadout,
    fit_linear_readout,
    predict_linear_readout,
    predict_mlp_readout,
    train_mlp_readout,
)
from .solver import CoreFitResult, fit_core
from .toy_data import ToyDataResult, generate_damped_rotation

__all__ = [
    "PreprocessResult",
    "Standardizer",
    "Version0Config",
    "WarmStartResult",
    "ObjectiveComponents",
    "CoreFitResult",
    "FilterResult",
    "MLPReadout",
    "ToyDataResult",
    "compute_objective_components",
    "compute_total_objective",
    "decode_predictions",
    "fit_standardizer",
    "fit_linear_readout",
    "fit_core",
    "generate_damped_rotation",
    "initialize_warm_start",
    "inverse_transform_standardizer",
    "load_config",
    "predict_linear_readout",
    "predict_mlp_readout",
    "run_causal_filter",
    "standardize_features",
    "train_mlp_readout",
    "transform_standardizer",
    "unstandardize_M",
    "update_A_minus",
    "update_A_plus",
    "update_V_A",
    "update_V_B",
    "validate_config",
]

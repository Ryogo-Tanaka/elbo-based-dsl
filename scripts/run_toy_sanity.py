#!/usr/bin/env python
"""Run the DSL/ELTO Version 0 toy sanity pipeline.

This script is a solver sanity check, not a benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dsl_elto.config import Version0Config, load_config  # noqa: E402
from dsl_elto.filtering import FilterResult, run_causal_filter  # noqa: E402
from dsl_elto.preprocessing import PreprocessResult, standardize_features  # noqa: E402
from dsl_elto.prediction import unstandardize_M  # noqa: E402
from dsl_elto.readout import (  # noqa: E402
    fit_linear_readout,
    predict_linear_readout,
    predict_mlp_readout,
    train_mlp_readout,
)
from dsl_elto.solver import CoreFitResult, fit_core  # noqa: E402
from dsl_elto.toy_data import ToyDataResult, generate_damped_rotation  # noqa: E402


def run_toy_sanity(config_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Run toy sanity and save metrics/logs under the configured output directory."""

    config = load_config(config_path)
    out_dir = Path(output_dir) if output_dir is not None else REPO_ROOT / config.logging.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    toy = _generate_from_config(config)
    preprocess = standardize_features(toy.B, toy.M, eps=config.features.eps)
    core = fit_core(preprocess.Z_B, config)
    filtered = run_causal_filter(
        preprocess.Z_B,
        core.V_A,
        core.V_B,
        core.a_init,
        sigma_b2=config.metrics.sigma_b2,
        p=config.metrics.p,
    )
    H_prior = filtered.ZB_pred_prior
    ZM_pred, readout_diagnostics, R_readout = _fit_and_predict_readout(H_prior, preprocess.Z_M, config)
    M_pred = unstandardize_M(ZM_pred, preprocess.m_stats)

    metrics = _build_metrics(config, preprocess, core, filtered, ZM_pred, M_pred, toy.M)
    diagnostics = _build_diagnostics(core, filtered, readout_diagnostics)

    _save_config(config, out_dir / "config_used.yaml")
    _save_losses(core, out_dir / "losses.csv")
    _save_json(metrics, out_dir / "metrics.json")
    _save_json(diagnostics, out_dir / "diagnostics.json")
    _save_predictions(out_dir / "predictions.npz", toy, preprocess, core, filtered, ZM_pred, M_pred, R_readout)
    return metrics


def _generate_from_config(config: Version0Config) -> ToyDataResult:
    if config.toy_data.type != "damped_rotation":
        raise ValueError("toy_data.type must be damped_rotation")
    if config.toy_data.state_dim != 2:
        raise ValueError("toy_data.state_dim must be 2 for damped_rotation")
    if config.toy_data.obs_dim != 3:
        raise ValueError("toy_data.obs_dim must be 3 for Version 0 toy sanity")
    if not config.toy_data.nonlinear_observation:
        raise ValueError("toy_data.nonlinear_observation must be true for Version 0 toy sanity")
    return generate_damped_rotation(
        T=config.toy_data.T,
        radius=config.toy_data.radius,
        omega=config.toy_data.omega,
        process_noise_std=config.toy_data.process_noise_std,
        observation_noise_std=config.toy_data.observation_noise_std,
        seed=config.project.random_seed,
    )


def _fit_and_predict_readout(
    H_prior: np.ndarray,
    Z_M: np.ndarray,
    config: Version0Config,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray | None]:
    if config.readout.type == "linear_ridge":
        R_readout = fit_linear_readout(H_prior, Z_M, config.ridge.lambda_readout)
        ZM_pred = predict_linear_readout(H_prior, R_readout)
        return ZM_pred, {"type": "linear_ridge"}, R_readout
    if config.readout.type == "mlp":
        model, loss_log = train_mlp_readout(H_prior, Z_M, config.readout)
        ZM_pred = predict_mlp_readout(H_prior, model)
        diagnostics = {
            "type": "mlp",
            "loss_initial": float(loss_log[0]) if loss_log else None,
            "loss_final": float(loss_log[-1]) if loss_log else None,
            "num_epochs": len(loss_log),
        }
        return ZM_pred, diagnostics, None
    raise ValueError("readout.type must be linear_ridge or mlp")


def _build_metrics(
    config: Version0Config,
    preprocess: PreprocessResult,
    core: CoreFitResult,
    filtered: FilterResult,
    ZM_pred: np.ndarray,
    M_pred: np.ndarray,
    M_true: np.ndarray,
) -> dict[str, Any]:
    final_objective = core.objective_log[-1]
    innovation_norms = np.linalg.norm(filtered.innovations, axis=1)
    standardized_mse = float(np.mean((ZM_pred - preprocess.Z_M) ** 2))
    feature_mse = float(np.mean((M_pred - M_true) ** 2))
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(core.V_A))))
    metrics = {
        "readout_type": config.readout.type,
        "objective": {
            "L_init": float(final_objective.init),
            "L_obs": float(final_objective.obs),
            "L_corr": float(final_objective.corr),
            "L_trans": float(final_objective.trans),
            "L_total": float(final_objective.total),
        },
        "filtering": {
            "mean_innovation_norm": float(np.mean(innovation_norms)),
            "max_innovation_norm": float(np.max(innovation_norms)),
            "mean_correction_norm": float(np.mean(filtered.correction_norms)),
            "max_correction_norm": float(np.max(filtered.correction_norms)),
        },
        "prediction": {
            "standardized_feature_mse": standardized_mse,
            "feature_mse": feature_mse,
        },
        "operators": {
            "spectral_radius_V_A": spectral_radius,
        },
    }
    _assert_metrics_finite(metrics)
    return metrics


def _build_diagnostics(
    core: CoreFitResult,
    filtered: FilterResult,
    readout_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "core": core.diagnostics,
        "readout": readout_diagnostics,
        "filtering": {
            "K_shape": list(filtered.K.shape),
            "A_filter_minus_shape": list(filtered.A_filter_minus.shape),
            "A_filter_plus_shape": list(filtered.A_filter_plus.shape),
            "ZB_pred_prior_shape": list(filtered.ZB_pred_prior.shape),
        },
    }


def _save_config(config: Version0Config, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=True)


def _save_losses(core: CoreFitResult, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "L_init", "L_obs", "L_corr", "L_trans", "L_total"])
        for iteration, components in enumerate(core.objective_log):
            writer.writerow(
                [
                    iteration,
                    components.init,
                    components.obs,
                    components.corr,
                    components.trans,
                    components.total,
                ]
            )


def _save_json(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _save_predictions(
    path: Path,
    toy: ToyDataResult,
    preprocess: PreprocessResult,
    core: CoreFitResult,
    filtered: FilterResult,
    ZM_pred: np.ndarray,
    M_pred: np.ndarray,
    R_readout: np.ndarray | None,
) -> None:
    arrays = {
        "X_true": toy.X_true,
        "Y": toy.Y,
        "M": toy.M,
        "B": toy.B,
        "Z_B": preprocess.Z_B,
        "Z_M": preprocess.Z_M,
        "A_train_minus": core.A_train_minus,
        "A_train_plus": core.A_train_plus,
        "A_filter_minus": filtered.A_filter_minus,
        "A_filter_plus": filtered.A_filter_plus,
        "V_A": core.V_A,
        "V_B": core.V_B,
        "ZB_pred_prior": filtered.ZB_pred_prior,
        "innovations": filtered.innovations,
        "ZM_pred": ZM_pred,
        "M_pred": M_pred,
    }
    if R_readout is not None:
        arrays["R_readout"] = R_readout
    np.savez(path, **arrays)


def _assert_metrics_finite(payload: dict[str, Any]) -> None:
    for value in _walk_values(payload):
        if isinstance(value, (int, float)) and not np.isfinite(value):
            raise ValueError("metrics must contain only finite numeric values")


def _walk_values(payload: dict[str, Any]):
    for value in payload.values():
        if isinstance(value, dict):
            yield from _walk_values(value)
        else:
            yield value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DSL/ELTO Version 0 toy sanity check.")
    parser.add_argument("--config", required=True, help="Path to a Version 0 YAML config.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory override.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = run_toy_sanity(args.config, output_dir=args.output_dir)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

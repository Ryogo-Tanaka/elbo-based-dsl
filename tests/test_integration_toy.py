from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_TOY_SANITY_PATH = REPO_ROOT / "scripts" / "run_toy_sanity.py"


def _load_run_toy_sanity():
    spec = importlib.util.spec_from_file_location("run_toy_sanity", RUN_TOY_SANITY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load run_toy_sanity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_toy_sanity


def test_toy_sanity_linear_ridge_runs_end_to_end(tmp_path: Path) -> None:
    run_toy_sanity = _load_run_toy_sanity()
    config_path = _write_toy_config(tmp_path, readout_type="linear_ridge")
    output_dir = tmp_path / "linear_outputs"

    metrics = run_toy_sanity(config_path, output_dir=output_dir)

    _assert_toy_outputs(output_dir, metrics, readout_type="linear_ridge")
    predictions = np.load(output_dir / "predictions.npz")
    np.testing.assert_allclose(predictions["ZB_pred_prior"], predictions["A_filter_minus"] @ predictions["V_B"].T)
    posterior_embedding = predictions["A_filter_plus"] @ predictions["V_B"].T
    assert not np.allclose(predictions["ZB_pred_prior"], posterior_embedding)
    assert "R_readout" in predictions.files


def test_toy_sanity_mlp_runs_end_to_end(tmp_path: Path) -> None:
    run_toy_sanity = _load_run_toy_sanity()
    config_path = _write_toy_config(tmp_path, readout_type="mlp")
    output_dir = tmp_path / "mlp_outputs"

    metrics = run_toy_sanity(config_path, output_dir=output_dir)

    _assert_toy_outputs(output_dir, metrics, readout_type="mlp")
    diagnostics = json.loads((output_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["readout"]["type"] == "mlp"
    assert diagnostics["readout"]["num_epochs"] == 12
    assert np.isfinite(diagnostics["readout"]["loss_initial"])
    assert np.isfinite(diagnostics["readout"]["loss_final"])


def _write_toy_config(tmp_path: Path, readout_type: str) -> Path:
    config_path = tmp_path / f"toy_{readout_type}.yaml"
    config_path.write_text(
        f"""
project:
  random_seed: 0
model:
  d_A: 3
solver:
  max_outer_iter: 4
  min_outer_iter: 3
  tol_rel_loss: 1.0e-300
toy_data:
  T: 32
  radius: 0.92
  omega: 0.25
  process_noise_std: 0.005
  observation_noise_std: 0.01
readout:
  type: {readout_type}
  mlp:
    hidden_dim: 8
    num_layers: 1
    activation: relu
    lr: 1.0e-2
    epochs: 12
    weight_decay: 0.0
    batch_size: null
    early_stopping: false
    patience: 20
""",
        encoding="utf-8",
    )
    return config_path


def _assert_toy_outputs(output_dir: Path, metrics: dict, readout_type: str) -> None:
    expected_files = {
        "config_used.yaml",
        "diagnostics.json",
        "losses.csv",
        "metrics.json",
        "predictions.npz",
    }
    assert expected_files.issubset({path.name for path in output_dir.iterdir()})

    saved_metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved_metrics == metrics
    assert metrics["readout_type"] == readout_type
    assert set(metrics["objective"]) == {"L_init", "L_obs", "L_corr", "L_trans", "L_total"}
    assert np.isfinite(metrics["filtering"]["mean_innovation_norm"])
    assert np.isfinite(metrics["filtering"]["max_innovation_norm"])
    assert np.isfinite(metrics["prediction"]["standardized_feature_mse"])
    assert np.isfinite(metrics["prediction"]["feature_mse"])
    assert np.isfinite(metrics["operators"]["spectral_radius_V_A"])
    assert metrics["prediction"]["feature_mse"] >= 0.0

    losses = (output_dir / "losses.csv").read_text(encoding="utf-8").splitlines()
    assert losses[0] == "iteration,L_init,L_obs,L_corr,L_trans,L_total"
    assert len(losses) >= 2

    predictions = np.load(output_dir / "predictions.npz")
    assert predictions["ZM_pred"].shape == predictions["Z_M"].shape
    assert predictions["M_pred"].shape == predictions["M"].shape
    assert np.all(np.isfinite(predictions["ZM_pred"]))
    assert np.all(np.isfinite(predictions["M_pred"]))

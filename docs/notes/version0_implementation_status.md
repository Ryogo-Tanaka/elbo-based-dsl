# DSL / ELTO Version 0 Implementation Status

This note records the implementation status after completing Tasks 0-15.
It is an implementation status note, not a benchmark report.

Toy sanity results below are only sanity checks for wiring, shapes, finite objectives,
filtering recursion, and readout execution. They must not be used as performance claims.

## Source Of Truth

- Primary implementation source: `docs/specs/dsl_elto_version0_implementation_spec.md`
- Secondary task/source layout reference: `docs/specs/dsl_elto_codex_task_decomposition_v0.md`
- Mathematical background only: `docs/formulation/_formulation_memo (10).tex`

## Completed Task Groups

- Tasks 0-2: repository/package scaffold, config dataclasses/YAML loading, shape utilities.
- Tasks 3-4: center + RMS preprocessing, correction-based warm start.
- Tasks 5-7: objective components, normalized covariance ridge operator updates, scalar-metric coordinate updates.
- Tasks 8-9: Phase 1 core solver loop, Phase 2 causal filtering path.
- Tasks 10-12: linear ridge readout, MLP readout, inverse feature transform and decoder adapter.
- Tasks 13-15: minimal damped-rotation toy data, toy sanity script, integration tests.

## Implemented Modules

Core package modules under `src/dsl_elto/`:

- `config.py`: Version 0 config dataclasses, YAML loading, unsupported-option validation.
- `shapes.py`: `[T, d]` shape assertions and finite checks.
- `preprocessing.py`: center + RMS feature standardization and inverse transform.
- `initialization.py`: correction-based warm start.
- `objective.py`: objective components `L_init`, `L_obs`, `L_corr`, `L_trans`, `L_total`.
- `operators.py`: normalized covariance ridge updates for `V_A` and `V_B`.
- `coordinates.py`: scalar-metric closed-form coordinate updates.
- `solver.py`: Phase 1 core solver loop and objective logging.
- `filtering.py`: Phase 2 causal filtering with fixed scalar-metric gain.
- `readout.py`: `linear_ridge` and `mlp` readouts.
- `prediction.py`: inverse feature transform and optional decoder adapter.
- `toy_data.py`: minimal damped-rotation toy data generator.

Support files:

- `configs/version0_default.yaml`
- `configs/toy_sanity_linear.yaml`
- `configs/toy_sanity_mlp.yaml`
- `scripts/run_toy_sanity.py`
- `tests/*.py`

## Version 0 Conventions

- Code arrays use shape `[T, d]`.
- Version 0 is single-trajectory only.
- Encoder and observation dictionary are fixed.
- Metrics are fixed scalar metrics.
- Phase 1 training coordinates and Phase 2 filtering coordinates are distinct:
  - `A_train_minus`, `A_train_plus`
  - `A_filter_minus`, `A_filter_plus`
- Main one-step prediction uses `A_filter_minus` through the prior embedding:
  - `H_prior = ZB_pred_prior`
  - `ZB_pred_prior[t] = A_filter_minus[t] @ V_B.T`
- `A_filter_plus` is used only for correction and propagation, not main prediction.
- `A_train_*` is not used for final causal prediction.
- Toy sanity is not a benchmark.

## Preprocessing

Feature preprocessing uses training-set center + RMS scale.

For each feature dimension:

```text
mean_j = mean(x_j)
scale_j = sqrt(mean((x_j - mean_j)^2) + eps)
z_j = (x_j - mean_j) / scale_j
```

Saved training statistics are represented by `Standardizer(mean, scale, eps)`.

- `fit_standardizer` computes training statistics.
- `transform_standardizer` uses saved training statistics for new arrays.
- `inverse_transform_standardizer` reconstructs feature coordinates using saved statistics.
- `standardize_features(B, M, eps)` returns standardized `Z_B`, `Z_M`, plus separate `b_stats` and `m_stats`.

## Warm Start

Warm start is correction-based.

- `V_B_init` is random and column-normalized.
- `V_A_init` is the zero matrix.
- `A_train_plus_init` is computed by a correction-objective linear solve.
- `A_train_minus_init[0] = a_init`.
- For `t >= 1`, `A_train_minus_init[t] = A_train_plus_init[t]`.

This is not direct state construction. Version 0 does not use:

- `A = G @ B`
- `A = G @ Z_B`
- SVD/PCA state initialization
- DSE CCA state construction

## Core Solver

The Phase 1 core solver connects warm start, operator updates, coordinate updates, and objective logging.

Outer-iteration order:

1. Start from correction-based warm start.
2. Update `V_A`.
3. Update `V_B`.
4. Update `A_train_plus`.
5. Update `A_train_minus`.
6. Compute and append objective components.
7. Check stopping after `min_outer_iter`.

The objective log stores:

- `L_init`
- `L_obs`
- `L_corr`
- `L_trans`
- `L_total`

The solver returns Phase 1 quantities only. It does not return `A_filter_minus` or `A_filter_plus`.

## Operator Updates

Operator updates use normalized covariance ridge form.

For `V_B`:

```text
V_B = (Z_B.T @ A_plus / T) @ inv(A_plus.T @ A_plus / T + lambda_b I)
```

For `V_A`:

```text
V_A = (A_minus[1:].T @ A_plus[:-1] / (T - 1))
      @ inv(A_plus[:-1].T @ A_plus[:-1] / (T - 1) + lambda_a I)
```

The implementation uses linear solves rather than explicit matrix inverse.

## Coordinate Updates

Coordinate updates use scalar metrics:

- `w_B = 1 / sigma_b2`
- `w_A = 1 / sigma_a2`
- `w_P = 1 / p`
- `w_0 = 1 / (c0 * p)`

`A_train_plus` uses the future transfer term for `t < T - 1`.
The final-time update omits the future transfer term.

`A_train_minus[0]` is anchored by `a_init`.
For `t >= 1`, `A_train_minus[t]` uses propagation from `V_A @ A_train_plus[t - 1]` in code convention form.

## Causal Filtering Path

The Phase 2 filtering path is separate from Phase 1 training coordinates.

Filtering initializes:

```text
A_filter_minus[0] = a_init
```

The fixed gain is:

```text
K = p V_B.T (p V_B V_B.T + sigma_b2 I)^(-1)
```

The recursion is:

```text
ZB_pred_prior[t] = A_filter_minus[t] @ V_B.T
innovation[t] = Z_B[t] - ZB_pred_prior[t]
A_filter_plus[t] = A_filter_minus[t] + innovation[t] @ K.T
A_filter_minus[t + 1] = A_filter_plus[t] @ V_A.T
```

`ZB_pred_prior` is the main prediction embedding for readout.
`A_filter_plus` is not used as the main prediction state.

## Readout

Readout supports `linear_ridge` and `mlp`.

### linear_ridge

The linear readout stores:

```text
R_readout: [m, d_B]
```

Prediction is:

```text
ZM_pred = H_prior @ R_readout.T
```

where:

```text
H_prior = ZB_pred_prior
```

### mlp

The MLP readout maps:

```text
H_prior -> ZM_pred
```

Only the readout module is trained.
The core solver, `V_A`, `V_B`, `A_train_*`, and `A_filter_*` are fixed during MLP readout training.

### inverse feature transform

Predicted standardized target features are mapped back by:

```text
M_pred = m_mean + m_scale * ZM_pred
```

An optional decoder adapter may be applied after this step.
Version 0 does not train or refine the decoder.

## Toy Sanity

Toy sanity uses a stable 2D damped rotation:

```text
x[t + 1] = radius * R(omega) @ x[t] + process_noise
```

Observation:

```text
Y = [x1, x2, x1^2] + observation_noise
```

For this sanity check:

```text
M = Y
B = Y
```

This avoids neural encoder dependencies and checks only the Version 0 implementation path.

Run commands:

```bash
python scripts/run_toy_sanity.py --config configs/toy_sanity_linear.yaml
python scripts/run_toy_sanity.py --config configs/toy_sanity_mlp.yaml
```

The script saves:

- `config_used.yaml`
- `metrics.json`
- `losses.csv`
- `diagnostics.json`
- `predictions.npz`

By default these are saved under `outputs/toy_sanity/`.
The `outputs/` directory and generated array files are ignored by git.

## Verification Summary

Latest final verification command:

```bash
python -m pytest tests -o cache_dir=/tmp/dsl-elto-pytest-cache-final
```

Observed result:

```text
62 passed, 1 warning
```

The warning was a PyTorch CUDA initialization warning during CPU execution.
It did not indicate a test failure.

Toy sanity commands were also run with `/tmp` output directories:

```bash
python scripts/run_toy_sanity.py --config configs/toy_sanity_linear.yaml --output-dir /tmp/dsl-elto-final-linear
python scripts/run_toy_sanity.py --config configs/toy_sanity_mlp.yaml --output-dir /tmp/dsl-elto-final-mlp
```

Both runs completed with finite metrics and finite saved arrays.
The saved prediction embedding satisfied:

```text
ZB_pred_prior == A_filter_minus @ V_B.T
```

and differed from:

```text
A_filter_plus @ V_B.T
```

This confirms that the main prediction path uses `A_filter_minus`.

## Toy Sanity Metrics

These numbers are implementation sanity outputs only.
They are not performance claims.

### linear_ridge

```text
L_init  = 0.42123072129737765
L_obs   = 0.061481747765879315
L_corr  = 0.6005353168280735
L_trans = 0.5584122446983356
L_total = 1.6416600305896663

mean_innovation_norm = 0.5584300576335235
max_innovation_norm  = 7.724122176750088
mean_correction_norm = 0.20480569448664027
max_correction_norm  = 2.6690700144849853

feature_mse              = 0.005701218877506526
standardized_feature_mse = 0.19346281865982415
spectral_radius_V_A      = 0.9623525298874811
```

### mlp

```text
L_init  = 0.42123072129737765
L_obs   = 0.061481747765879315
L_corr  = 0.6005353168280735
L_trans = 0.5584122446983356
L_total = 1.6416600305896663

mean_innovation_norm = 0.5584300576335235
max_innovation_norm  = 7.724122176750088
mean_correction_norm = 0.20480569448664027
max_correction_norm  = 2.6690700144849853

feature_mse              = 0.004414205119712869
standardized_feature_mse = 0.14182002153645262
spectral_radius_V_A      = 0.9623525298874811
```

No NaN, inf, or obvious divergence was observed in these sanity runs.

## Known Limits

Version 0 intentionally does not include:

- DSE CCA state construction
- stochastic realization state construction
- SVD/PCA state initialization
- direct `A = G @ B` or `A = G @ Z_B` state construction
- inference encoder
- multiple trajectories
- residual-calibrated metrics
- KKR-style covariance recursion
- posterior reconstruction loss
- encoder or dictionary refinement
- hard whitening
- full non-isotropic metrics
- benchmark runners, baselines, or ablations

## Open Issues Before Experiment Design

- Define experiment goals separately from toy sanity.
- Keep toy sanity results out of performance claims.
- Decide synthetic experiment families only after documenting evaluation metrics.
- Decide train/test splits and leakage rules for real experiment settings.
- Decide whether experiment scripts need separate output layout beyond `outputs/toy_sanity/`.
- Decide whether MLP readout training should expose deterministic seed controls beyond current PyTorch global behavior.
- Consider adding config-load-time validation for unsupported MLP activations.
- Consider adding explicit forbidden-path tests for SVD/PCA, CCA, KKR recursion, and multiple trajectory support.

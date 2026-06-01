# DSL/ELTO Embedded-Belief Operator Learning — Codex Task Decomposition v0

作成日: 2026-06-01  
対象: `DSE / ELTO / KKR / KBR` に関連する embedded latent transfer-operator learning  
主参照:

- `_formulation_memo (10).tex` — mathematical formulation reference.
- `dsl_elto_version0_implementation_spec.md` — Version 0 implementation specification.
- `formulation_memo_algorithm_consensus_delta_20260601.md` — prior consensus / revision record.

目的: Codex に実装を依頼する前に、実装単位、依存関係、config 設計、acceptance criteria、禁止事項、toy sanity implementation の進め方を固定する。

---

## 0. How this document should be used

This document is **not** the formulation memo and **not** the final experiment plan. It is a Codex-facing task plan.

Use documents in this order:

1. `dsl_elto_version0_implementation_spec.md` as the source of implementation equations, shapes, and defaults.
2. `_formulation_memo (10).tex` as the source of mathematical terminology and conceptual constraints.
3. This task decomposition as the guide for splitting Codex work into small implementation steps.

Do **not** ask Codex to implement the whole project at once. Each task below should be implemented, tested, and reviewed before moving to the next group.

---

## 1. Global implementation policy

### 1.1 Target of the first implementation

The first implementation is a **single-trajectory numerical solver and prediction pipeline** for the Version 0 embedded-belief operator learning algorithm.

The goal is **solver sanity**, not paper-level performance.

The first implementation should verify that:

- Phase 1 batch variational coordinate learning runs without numerical failure.
- Phase 2 causal filtering path is generated separately from Phase 1 coordinates.
- Phase 3 prediction uses the prior filtering coordinate, not the posterior coordinate.
- Linear ridge and nonlinear MLP readouts both work.
- A minimal toy dynamical system can run end-to-end.

### 1.2 Core constraints

The implementation must preserve these constraints:

- Single trajectory only:
  \[
  \{y_t\}_{t=1}^T.
  \]
- Code stores arrays as `[T, d]` by default.
- Mathematical formulas in the memo may use `[d, T]`; code must document the `[T, d]` convention.
- Encoder, observation dictionary, and decoder are fixed during the core solver.
- Core solver can operate directly on precomputed feature matrices `B` and `M`.
- No DSE functional CCA / stochastic realization state construction.
- No SVD/PCA state-coordinate initialization.
- No direct observation-to-state warm start such as `A = G @ Z_B`.
- No inference encoder.
- No residual-calibrated metrics in the first implementation.
- No KKR-style covariance / inverse-curvature recursion in the first implementation.
- No posterior reconstruction loss.
- No multiple-trajectory boundary support.
- No hard whitening.
- No encoder / dictionary / decoder refinement inside the core solver.

### 1.3 Recommended technology stack

Recommended initial stack:

- Core linear algebra: `numpy`.
- MLP readout: `torch`.
- Config: YAML + dataclasses or Pydantic-like validation.
- Tests: `pytest`.
- Plots: optional `matplotlib` only in scripts / diagnostics, not in core logic.

Rationale:

- NumPy is simpler for closed-form linear algebra.
- PyTorch is useful only where gradient descent is required, namely MLP readout.
- The core algorithm should remain readable and easy to unit test.

---

## 2. Proposed repository layout

Recommended layout:

```text
dsl_elto/
  __init__.py
  config.py
  shapes.py
  preprocessing.py
  initialization.py
  objective.py
  operators.py
  coordinates.py
  solver.py
  filtering.py
  readout.py
  prediction.py
  toy_data.py
  diagnostics.py
  io.py

configs/
  version0_default.yaml
  toy_sanity_linear.yaml
  toy_sanity_mlp.yaml

scripts/
  run_toy_sanity.py
  inspect_toy_results.py

tests/
  test_config.py
  test_shapes.py
  test_preprocessing.py
  test_initialization.py
  test_objective.py
  test_operators.py
  test_coordinates.py
  test_solver.py
  test_filtering.py
  test_readout_linear.py
  test_readout_mlp.py
  test_prediction.py
  test_toy_data.py
  test_integration_toy.py

docs/
  implementation_notes.md
```

Optional but useful:

```text
outputs/
  toy_sanity/
```

Do not store large experiment outputs in the repository by default.

---

## 3. Config file design

The config is important because Codex should not invent algorithmic variants. The config should make allowed choices explicit and reject unsupported options.

### 3.1 Config files to create

Create at least three config files:

```text
configs/version0_default.yaml
configs/toy_sanity_linear.yaml
configs/toy_sanity_mlp.yaml
```

`version0_default.yaml` contains general defaults.  
`toy_sanity_linear.yaml` overrides readout to `linear_ridge`.  
`toy_sanity_mlp.yaml` overrides readout to `mlp`.

### 3.2 Top-level config schema

Recommended top-level YAML structure:

```yaml
project:
  name: dsl_elto_version0
  random_seed: 0
  dtype: float64
  device: cpu

scope:
  single_trajectory: true
  allow_multiple_trajectories: false
  allow_svd_init: false
  allow_direct_B_to_A_init: false
  allow_inference_encoder: false
  allow_residual_calibrated_metrics: false
  allow_kkr_covariance_recursion: false
  allow_encoder_refinement: false
  allow_posterior_reconstruction_loss: false

features:
  standardize: true
  eps: 1.0e-8
  use_precomputed_features: true
  feature_array_layout: T_by_d

model:
  d_A: 8
  d_B: null
  m: null

metrics:
  sigma_b2: 1.0
  sigma_a2: 1.0
  p: 1.0
  c0: 10.0

ridge:
  lambda_a: 1.0e-3
  lambda_b: 1.0e-3
  lambda_readout: 1.0e-3
  normalized_covariance_form: true

initialization:
  type: correction_warm_start
  vb_seed: random_column_normalized
  va_seed: zero
  a_init_mode: zero_reference
  external_a_init_path: null
  seed_scale: 1.0

solver:
  max_outer_iter: 100
  tol_rel_loss: 1.0e-6
  min_outer_iter: 1
  check_finite: true
  fail_on_nan: true
  fail_on_inf: true
  max_abs_A: 1.0e6
  min_rms_A: 1.0e-12

filtering:
  use_fixed_gain: true
  use_same_p_all_times: true
  separate_initial_filter_metric: false

readout:
  type: linear_ridge   # allowed: linear_ridge, mlp
  train_on_filter_prior: true
  use_posterior_reconstruction: false
  mlp:
    hidden_dim: 64
    num_layers: 2
    activation: relu
    lr: 1.0e-3
    epochs: 500
    weight_decay: 1.0e-4
    batch_size: null
    early_stopping: false
    patience: 20

toy_data:
  type: damped_rotation
  T: 300
  state_dim: 2
  obs_dim: 3
  radius: 0.98
  omega: 0.20
  process_noise_std: 0.01
  observation_noise_std: 0.05
  nonlinear_observation: true

logging:
  output_dir: outputs/toy_sanity
  save_config: true
  save_losses_csv: true
  save_metrics_json: true
  save_arrays: false
  save_plots: true
  log_every: 1

tests:
  run_slow_mlp_tests: false
```

### 3.3 Config validation rules

Implement strict validation:

- `scope.single_trajectory` must be `true`.
- `scope.allow_multiple_trajectories` must be `false`.
- `scope.allow_svd_init` must be `false`.
- `scope.allow_direct_B_to_A_init` must be `false`.
- `initialization.type` must be `correction_warm_start`.
- `readout.type` must be either `linear_ridge` or `mlp`.
- `ridge.normalized_covariance_form` must be `true`.
- `filtering.use_fixed_gain` must be `true`.
- `filtering.separate_initial_filter_metric` must be `false`.
- all scalar metrics and ridge parameters must be positive.
- `model.d_A` must be positive.
- if precomputed `B` and `M` are supplied, infer `d_B` and `m` from arrays and check consistency.

### 3.4 Why config is separate from formulation memo

The formulation memo should not include implementation details such as exact warm start, YAML keys, MLP optimizer, or toy data settings. The config files make those decisions executable without polluting the mathematical memo.

---

## 4. Task dependency graph

Recommended dependency order:

```text
Task 0  Repository scaffold
Task 1  Config schema and validation
Task 2  Shape utilities
Task 3  Feature preprocessing
Task 4  Correction-based warm start
Task 5  Objective components
Task 6  Closed-form operator updates
Task 7  Coordinate updates
Task 8  Core solver loop
Task 9  Causal filtering path
Task 10 Linear ridge readout
Task 11 MLP readout
Task 12 Prediction adapter
Task 13 Toy data generator
Task 14 Toy sanity script
Task 15 Unit / integration tests
Task 16 Documentation and acceptance checklist
```

Dependency graph:

```text
0 -> 1 -> 2
2 -> 3 -> 4
2 -> 5
2 -> 6
2 -> 7
4,5,6,7 -> 8
8 -> 9
9,3 -> 10
9,3 -> 11
10,11,3 -> 12
13 -> 14
3,4,5,6,7,8,9,10,11,12,13 -> 15
15 -> 16
```

---

## 5. Task 0 — Repository scaffold

### Goal

Create the package skeleton and ensure imports and tests can run.

### Files

Create:

```text
dsl_elto/__init__.py
dsl_elto/config.py
dsl_elto/shapes.py
dsl_elto/preprocessing.py
dsl_elto/initialization.py
dsl_elto/objective.py
dsl_elto/operators.py
dsl_elto/coordinates.py
dsl_elto/solver.py
dsl_elto/filtering.py
dsl_elto/readout.py
dsl_elto/prediction.py
dsl_elto/toy_data.py
dsl_elto/diagnostics.py
dsl_elto/io.py
configs/version0_default.yaml
scripts/run_toy_sanity.py
tests/test_imports.py
```

### Acceptance criteria

- `pytest` runs.
- `import dsl_elto` succeeds.
- No algorithmic code is required yet.

### Suggested Codex prompt

```text
Read dsl_elto_version0_implementation_spec.md and dsl_elto_codex_task_decomposition_v0.md.
Implement Task 0 only: create the package scaffold, placeholder modules, default config file, and a basic import test.
Do not implement algorithmic logic yet.
Do not add DSE CCA, SVD/PCA initialization, inference encoder, multiple trajectories, residual-calibrated metrics, KKR covariance recursion, posterior reconstruction loss, or encoder refinement.
```

---

## 6. Task 1 — Config schema and validation

### Goal

Implement config loading, dataclasses, defaults, and validation.

### Files

- `dsl_elto/config.py`
- `configs/version0_default.yaml`
- `configs/toy_sanity_linear.yaml`
- `configs/toy_sanity_mlp.yaml`
- `tests/test_config.py`

### Implementation requirements

Implement:

```python
load_config(path: str | Path) -> Version0Config
validate_config(config: Version0Config) -> None
```

Recommended dataclasses:

```python
ProjectConfig
ScopeConfig
FeatureConfig
ModelConfig
MetricConfig
RidgeConfig
InitializationConfig
SolverConfig
FilteringConfig
ReadoutConfig
ToyDataConfig
LoggingConfig
Version0Config
```

### Validation checks

Reject unsupported options:

- SVD initialization.
- direct `B_to_A` initialization.
- multiple trajectories.
- residual-calibrated metrics.
- KKR covariance recursion.
- posterior reconstruction loss.
- unknown readout type.

### Acceptance criteria

- default config loads.
- linear and MLP toy configs load.
- invalid readout type raises error.
- setting `allow_svd_init: true` raises error.
- negative metric parameters raise error.

---

## 7. Task 2 — Shape convention and utilities

### Goal

Make `[T, d]` layout explicit and enforce shape checks.

### Files

- `dsl_elto/shapes.py`
- `tests/test_shapes.py`

### Implementation requirements

Implement helpers:

```python
assert_2d(name, x)
assert_shape(name, x, expected)
assert_same_T(*arrays)
assert_last_dim(name, x, d)
make_eye(d, dtype=None)
check_finite(name, x)
```

### Code convention

All core arrays use `[T, d]`:

- `B`: `[T, d_B]`
- `M`: `[T, m]`
- `Z_B`: `[T, d_B]`
- `Z_M`: `[T, m]`
- `A_train_minus`: `[T, d_A]`
- `A_train_plus`: `[T, d_A]`
- `A_filter_minus`: `[T, d_A]`
- `A_filter_plus`: `[T, d_A]`

Operators:

- `V_A`: `[d_A, d_A]`
- `V_B`: `[d_B, d_A]`
- fixed gain `K`: `[d_A, d_B]`

### Acceptance criteria

- shape errors are informative.
- all tests use `[T, d]` convention.

---

## 8. Task 3 — Feature preprocessing

### Goal

Implement center + RMS scale preprocessing for `B` and `M`.

### Files

- `dsl_elto/preprocessing.py`
- `tests/test_preprocessing.py`

### Mathematical convention

For an array `X` of shape `[T, d]`:

\[
\bar x_j = \frac1T\sum_t x_{t,j},
\qquad
s_j = \sqrt{\frac1T\sum_t (x_{t,j}-\bar x_j)^2 + \epsilon}.
\]

\[
z_{t,j}=\frac{x_{t,j}-\bar x_j}{s_j}.
\]

### Implementation requirements

Implement:

```python
@dataclass
class Standardizer:
    mean: np.ndarray  # [d]
    scale: np.ndarray # [d]
    eps: float

fit_standardizer(X, eps) -> Standardizer
transform_standardizer(X, stats) -> Z
inverse_transform_standardizer(Z, stats) -> X
standardize_features(B, M, eps) -> PreprocessResult
```

`PreprocessResult` should include:

```python
Z_B
Z_M
b_stats
m_stats
```

### Acceptance criteria

- transformed arrays have mean approximately zero.
- RMS approximately one.
- inverse transform reconstructs original values within tolerance.
- test-time transform uses training statistics.

---

## 9. Task 4 — Correction-based warm start

### Goal

Implement numerical seeds for `A_train_minus` and `A_train_plus` without using SVD/PCA or direct `A=GZ_B` construction.

### Files

- `dsl_elto/initialization.py`
- `tests/test_initialization.py`

### Mathematical convention

Use a random nondegenerate seed for \(V_B^{(0)}\):

\[
V_B^{(0)}\in\mathbb R^{d_B\times d_A}.
\]

Set:

\[
V_A^{(0)}=0,
\qquad
 a_t^{-,(0)}=a_{\mathrm{init}}.
\]

Compute:

\[
H_0=w_B (V_B^{(0)})^\top V_B^{(0)}+w_P I.
\]

\[
a_t^{+,(0)}
=
H_0^{-1}
\left(w_B(V_B^{(0)})^\top z^b_t+w_Pa_{\mathrm{init}}\right).
\]

Then set:

\[
a_1^{-,(0)}=a_{\mathrm{init}},
\qquad
 a_t^{-,(0)}=a_t^{+,(0)}\quad(t\ge2).
\]

### Implementation requirements

Implement:

```python
initialize_warm_start(Z_B, d_A, a_init, sigma_b2, p, seed, eps)
```

Return:

```python
A_train_minus_init  # [T, d_A]
A_train_plus_init   # [T, d_A]
V_A_init            # [d_A, d_A]
V_B_init            # [d_B, d_A]
```

### Important prohibition

Do not implement:

```python
A_plus = Z_B @ G.T
```

or any direct observation-to-state construction.

The observation enters through the correction objective above.

### Acceptance criteria

- all shapes correct.
- all values finite.
- `A_train_minus_init[0] == a_init`.
- `A_train_plus_init` is not all zero.
- no SVD/PCA call.
- no direct `A = GZ_B` construction.

---

## 10. Task 5 — Objective components

### Goal

Evaluate embedding free-energy components separately.

### Files

- `dsl_elto/objective.py`
- `tests/test_objective.py`

### Components

\[
L_{\mathrm{init}}
=
\frac{1}{2c_0p}\|a_1^- - a_{\mathrm{init}}\|^2.
\]

\[
L_{\mathrm{obs}}
=
\frac{1}{2\sigma_B^2}
\sum_t
\|z^b_t - V_Ba_t^+\|^2.
\]

\[
L_{\mathrm{corr}}
=
\frac{1}{2p}
\sum_t
\|a_t^+ - a_t^-\|^2.
\]

\[
L_{\mathrm{trans}}
=
\frac{1}{2\sigma_A^2}
\sum_{t=1}^{T-1}
\|a_{t+1}^- - V_Aa_t^+\|^2.
\]

### Implementation requirements

Implement:

```python
compute_objective_components(...)
compute_total_objective(...)
```

Return a dict or dataclass:

```python
ObjectiveComponents(init, obs, corr, trans, total)
```

### Acceptance criteria

- all components nonnegative.
- total equals sum.
- zero-residual artificial case gives near-zero loss.
- invalid shapes raise errors.

---

## 11. Task 6 — Closed-form operator updates

### Goal

Implement normalized covariance ridge updates for `V_A` and `V_B`.

### Files

- `dsl_elto/operators.py`
- `tests/test_operators.py`

### Formulas with `[T, d]` convention

\[
V_B
=
\left(\frac1T Z_B^\top A^+\right)
\left(\frac1T (A^+)^\top A^+ + \lambda_BI\right)^{-1}.
\]

\[
V_A
=
\left(\frac1{T-1}(A^-_{2:T})^\top A^+_{1:T-1}\right)
\left(\frac1{T-1}(A^+_{1:T-1})^\top A^+_{1:T-1}+\lambda_AI\right)^{-1}.
\]

Shapes:

- `V_B`: `[d_B, d_A]`
- `V_A`: `[d_A, d_A]`

### Implementation requirements

Implement:

```python
update_V_B(Z_B, A_plus, lambda_b)
update_V_A(A_minus, A_plus, lambda_a)
```

Use `np.linalg.solve`, not explicit inverse, wherever practical.

### Acceptance criteria

- shapes correct.
- known ridge regression case recovered.
- lambda shrinkage works.
- no metric-weighted operator update in Version 0.

---

## 12. Task 7 — Coordinate updates

### Goal

Implement closed-form scalar-metric updates for `A_train_plus` and `A_train_minus`.

### Files

- `dsl_elto/coordinates.py`
- `tests/test_coordinates.py`

### Weights

\[
w_B=\sigma_B^{-2},
\qquad
w_A=\sigma_A^{-2},
\qquad
w_P=p^{-1},
\qquad
w_0=(c_0p)^{-1}.
\]

### Posterior update

For \(t<T\):

\[
H_t^+
=
w_BV_B^\top V_B+w_PI+w_AV_A^\top V_A.
\]

\[
h_t^+
=
w_BV_B^\top z^b_t+w_Pa_t^-+w_AV_A^\top a_{t+1}^-.
\]

For \(t=T\), omit the future term:

\[
H_T^+=w_BV_B^\top V_B+w_PI.
\]

\[
h_T^+=w_BV_B^\top z^b_T+w_Pa_T^-.
\]

### Prior update

\[
a_1^-=
\frac{w_Pa_1^+ + w_0a_{\mathrm{init}}}{w_P+w_0}.
\]

For \(t\ge2\):

\[
a_t^-=
\frac{w_Pa_t^+ + w_AV_Aa_{t-1}^+}{w_P+w_A}.
\]

### Implementation requirements

Implement:

```python
update_A_plus(Z_B, A_minus, V_A, V_B, sigma_b2, sigma_a2, p)
update_A_minus(A_plus, V_A, a_init, sigma_a2, p, c0)
```

### Acceptance criteria

- shapes correct.
- finite outputs.
- final time update omits future transfer term.
- initial prior update is weighted average of `a_init` and `a_1_plus`.
- scalar hand-computed case passes.

---

## 13. Task 8 — Core solver loop

### Goal

Implement Phase 1 training-time coordinate/operator learning.

### Files

- `dsl_elto/solver.py`
- `tests/test_solver.py`

### Implementation requirements

Implement:

```python
fit_core(Z_B, config, a_init=None) -> CoreFitResult
```

Recommended result dataclass:

```python
CoreFitResult:
  A_train_minus: np.ndarray
  A_train_plus: np.ndarray
  V_A: np.ndarray
  V_B: np.ndarray
  a_init: np.ndarray
  objective_log: list[ObjectiveComponents]
  diagnostics: dict
```

### Update order

1. set `a_init`:
   - if provided, use it;
   - otherwise zero reference prior.
2. correction-based warm start.
3. for `k=0,...,K-1`:
   - update `V_A` from current `A_train_minus`, `A_train_plus`;
   - update `V_B` from current `A_train_plus`, `Z_B`;
   - update `A_train_plus`;
   - update `A_train_minus`;
   - compute objective components;
   - run diagnostics and stopping checks.

### Acceptance criteria

- runs on toy arrays.
- outputs finite arrays.
- logs objective components.
- stops by max iteration or relative tolerance.
- does not produce `A_filter_*` objects; filtering is Task 9.

---

## 14. Task 9 — Causal filtering path

### Goal

Implement Phase 2 causal recursion from learned operators.

### Files

- `dsl_elto/filtering.py`
- `tests/test_filtering.py`

### Implementation requirements

Implement:

```python
run_causal_filter(Z_B, V_A, V_B, a_init, sigma_b2, p) -> FilterResult
```

Recommended result dataclass:

```python
FilterResult:
  A_filter_minus: np.ndarray  # [T, d_A]
  A_filter_plus: np.ndarray   # [T, d_A]
  ZB_pred_prior: np.ndarray   # [T, d_B]
  innovations: np.ndarray     # [T, d_B]
  correction_norms: np.ndarray # [T]
  K: np.ndarray               # [d_A, d_B]
```

### Recursion

\[
K=pV_B^\top(pV_BV_B^\top+\sigma_B^2I)^{-1}.
\]

\[
a_{\mathrm{filter},1}^- = a_{\mathrm{init}}.
\]

\[
\hat z^b_{t|t-1}=V_Ba_{\mathrm{filter},t}^-.
\]

\[
a_{\mathrm{filter},t}^+
=a_{\mathrm{filter},t}^-+K(z^b_t-\hat z^b_{t|t-1}).
\]

\[
a_{\mathrm{filter},t+1}^-
=V_Aa_{\mathrm{filter},t}^+.
\]

### Acceptance criteria

- `A_train_*` is not used or overwritten.
- `ZB_pred_prior` uses `A_filter_minus`.
- `A_filter_plus` is not used as the main prediction state.
- all arrays finite.
- shape tests pass.

---

## 15. Task 10 — Linear ridge readout

### Goal

Implement closed-form linear readout for solver sanity and baseline prediction.

### Files

- `dsl_elto/readout.py`
- `tests/test_readout_linear.py`

### Input

\[
H^-=[V_Ba_{\mathrm{filter},1}^-,\dots,V_Ba_{\mathrm{filter},T}^-].
\]

Code shape:

- `H_prior`: `[T, d_B]`
- `Z_M`: `[T, m]`

### Formula

Use weight matrix `W_readout` with shape `[d_B, m]` so that:

```python
Z_M_pred = H_prior @ W_readout
```

\[
W_{\mathrm{readout}}
=
\left(\frac1T (H^-)^\top H^- + \lambda_r I\right)^{-1}
\left(\frac1T (H^-)^\top Z_M\right).
\]

This is equivalent to the mathematical map \(R_{\xi_B}\in\mathbb R^{m\times d_B}\) with transposed storage.

### Implementation requirements

Implement:

```python
fit_linear_readout(H_prior, Z_M, lambda_readout)
predict_linear_readout(H_prior, W_readout)
```

### Acceptance criteria

- shape correct.
- ridge regression toy case passes.
- returns finite predictions.

---

## 16. Task 11 — Nonlinear MLP readout

### Goal

Implement nonlinear readout with regularization as a configurable option.

### Files

- `dsl_elto/readout.py`
- `tests/test_readout_mlp.py`

### Implementation requirements

Implement a PyTorch module:

```python
class MLPReadout(torch.nn.Module):
    ...
```

Implement:

```python
train_mlp_readout(H_prior, Z_M, config) -> trained module and logs
predict_mlp_readout(H_prior, module) -> Z_M_pred
```

Loss:

\[
\frac1T\sum_t
\|z^m_t-r_{\xi_B}^{\mathrm{MLP}}(H_t^-)\|^2
+
\lambda_r\Omega(\xi_B).
\]

### Acceptance criteria

- training loss decreases on a simple toy regression.
- core solver objects are not modified during MLP training.
- same prediction interface as linear readout.
- can be disabled via config.

---

## 17. Task 12 — Prediction adapter

### Goal

Map standardized feature predictions back to feature space and optionally raw observation space.

### Files

- `dsl_elto/prediction.py`
- `tests/test_prediction.py`

### Implementation requirements

Implement:

```python
unstandardize_M(ZM_pred, m_stats) -> M_pred
decode_predictions(M_pred, decoder=None) -> Y_pred_or_M_pred
```

Formula:

\[
\hat m_{t|t-1}=\bar m+D_m\hat z^m_{t|t-1}.
\]

If decoder is available:

\[
\hat y_{t|t-1}=g_\alpha(\hat m_{t|t-1}).
\]

### Acceptance criteria

- decoder optional.
- feature inverse transform correct.
- raw predictions shape correct if decoder supplied.

---

## 18. Task 13 — Minimal toy data generator

### Goal

Create solver sanity data, not a final benchmark.

### Files

- `dsl_elto/toy_data.py`
- `tests/test_toy_data.py`

### Default toy dynamics

Use damped rotation:

\[
x_{t+1}=A_{\mathrm{true}}x_t+\epsilon_t.
\]

\[
A_{\mathrm{true}}
=
r
\begin{pmatrix}
\cos\omega & -\sin\omega\\
\sin\omega & \cos\omega
\end{pmatrix},
\qquad 0<r<1.
\]

Observation:

\[
y_t=
\begin{bmatrix}
x_{1,t}\\
x_{2,t}\\
x_{1,t}^2
\end{bmatrix}
+
\eta_t.
\]

For first sanity run, set:

\[
M=Y,
\qquad B=Y.
\]

This avoids neural encoder complexity.

### Implementation requirements

Implement:

```python
generate_damped_rotation(T, radius, omega, process_noise_std, observation_noise_std, seed)
```

Return:

```python
X_true: [T, 2]
Y: [T, obs_dim]
M: [T, m]
B: [T, d_B]
```

### Acceptance criteria

- reproducible by seed.
- finite arrays.
- correct shapes.
- no neural dependency.

---

## 19. Task 14 — Toy sanity script

### Goal

Run end-to-end sanity check.

### Files

- `scripts/run_toy_sanity.py`
- `tests/test_integration_toy.py`

### Script steps

1. load config.
2. generate toy data.
3. preprocess `B`, `M`.
4. fit core solver.
5. run causal filter.
6. build `H_prior = ZB_pred_prior`.
7. fit readout:
   - linear ridge if `readout.type=linear_ridge`;
   - MLP if `readout.type=mlp`.
8. compute feature prediction metrics.
9. optionally unstandardize `M_pred`.
10. save logs and metrics.

### Outputs

```text
outputs/toy_sanity/
  config_used.yaml
  metrics.json
  losses.csv
  diagnostics.json
  predictions.npz
  plots.png        # optional
```

### Acceptance criteria

- script runs with linear config.
- script runs with MLP config.
- objective logs finite.
- feature MSE finite.
- no posterior reconstruction used as prediction.

---

## 20. Task 15 — Unit and integration tests

### Required tests

1. config load and rejection of unsupported options.
2. shape utility tests.
3. preprocessing inverse transform.
4. warm start finite and nondegenerate.
5. objective nonnegative and zero-residual artificial case.
6. operator ridge regression toy case.
7. coordinate update scalar hand-computed case.
8. core solver smoke test.
9. causal filter smoke test.
10. linear readout toy regression.
11. MLP readout loss decrease.
12. prediction adapter inverse transform.
13. toy data reproducibility.
14. end-to-end toy sanity.
15. forbidden paths test:
    - no SVD/PCA call;
    - no CCA module;
    - no multiple trajectory support;
    - no KKR covariance recursion.

### Acceptance criteria

- all fast tests pass.
- MLP test may be marked slow, but should run when enabled.
- no silent shape broadcasting.

---

## 21. Task 16 — Documentation and acceptance checklist

### Goal

Create minimal docs for running and reviewing the implementation.

### Files

- `README.md`
- `docs/implementation_notes.md`

### README should include

- install instructions;
- how to run tests;
- how to run toy sanity script;
- what is explicitly excluded;
- reference to formulation memo and implementation spec.

### Implementation notes should include

- shape convention `[T, d]`;
- explanation of `A_train_*` vs `A_filter_*`;
- correction-based warm start;
- readout types;
- known limitations.

---

## 22. Suggested Codex prompt sequence

### Prompt 1 — scaffold, config, shapes

```text
Use dsl_elto_version0_implementation_spec.md and dsl_elto_codex_task_decomposition_v0.md.
Implement Tasks 0, 1, and 2 only: repository scaffold, config loading/validation, and shape utilities.
Do not implement algorithmic updates yet.
Do not add DSE CCA, SVD/PCA state initialization, inference encoder, multiple trajectory support, residual-calibrated metrics, KKR-style covariance recursion, posterior reconstruction loss, or encoder/dictionary refinement.
Add pytest tests for the implemented pieces.
```

### Prompt 2 — preprocessing and initialization

```text
Implement Tasks 3 and 4 only: feature preprocessing with center + RMS scale, and correction-based warm start.
Use code layout [T, d].
Do not implement direct A=GZ_B initialization; observations may enter A_plus initialization only through the correction objective with random nondegenerate V_B seed.
Add tests for inverse preprocessing and warm start shapes/finite values.
```

### Prompt 3 — objective, operators, coordinates

```text
Implement Tasks 5, 6, and 7 only: objective components, normalized covariance ridge updates for V_A and V_B, and scalar-metric coordinate updates.
Use np.linalg.solve rather than explicit matrix inverse where possible.
Add unit tests, including scalar hand-computed coordinate update cases.
```

### Prompt 4 — core solver and causal filter

```text
Implement Tasks 8 and 9 only: Phase 1 core solver loop and Phase 2 causal filtering path.
Keep A_train_minus/A_train_plus separate from A_filter_minus/A_filter_plus.
Predictions must be based on A_filter_minus, not A_train coordinates and not posterior reconstruction.
Add smoke tests.
```

### Prompt 5 — readout and prediction

```text
Implement Tasks 10, 11, and 12 only: linear ridge readout, nonlinear MLP readout, and prediction adapter.
Linear ridge must be closed-form. MLP readout should train only the readout module and must not modify core solver outputs.
Add tests for both readout types.
```

### Prompt 6 — toy sanity and integration

```text
Implement Tasks 13, 14, and 15 only: damped-rotation toy data generator, end-to-end toy sanity script, and integration tests.
The toy run is a solver sanity check, not a performance benchmark.
Save metrics and logs to outputs/toy_sanity.
```

### Prompt 7 — docs and final review

```text
Implement Task 16 only: update README and implementation notes.
Document shape conventions, excluded items, config usage, and how to run toy sanity tests.
Do not change algorithmic code unless required to fix documentation inconsistencies.
```

---

## 23. Acceptance checklist before moving to experiment design

Do not move to full experiment design until all items below are satisfied:

- [ ] all unit tests pass;
- [ ] toy sanity script runs with linear readout;
- [ ] toy sanity script runs with MLP readout;
- [ ] objective components remain finite;
- [ ] causal filter creates `A_filter_minus` and `A_filter_plus` separately from `A_train_minus` and `A_train_plus`;
- [ ] readout prediction uses `A_filter_minus` only;
- [ ] no posterior reconstruction loss is used;
- [ ] no SVD/PCA or DSE CCA initialization exists in code;
- [ ] config validation rejects unsupported options;
- [ ] logs include objective components and key diagnostics;
- [ ] implementation notes explain known limitations.

---

## 24. After first toy sanity run

After a successful first toy run, inspect:

1. objective component curves;
2. observation consistency residual;
3. correction norms;
4. transfer residual;
5. spectral radius of `V_A`;
6. Phase 1 / Phase 2 coordinate gap;
7. feature prediction MSE for linear ridge readout;
8. feature prediction MSE for MLP readout;
9. whether MLP overfits compared with linear readout;
10. whether warm start produces stable initial coordinates.

Only after this inspection should we create the experiment setting md.

---

## 25. Future experiment-setting md scope

The experiment-setting md should be created after implementation sanity check. It should cover:

- minimum synthetic benchmark;
- linear SSM sanity check;
- nonlinear observation benchmark;
- non-Gaussian / switching benchmark;
- evaluation metrics;
- baselines;
- ablations;
- diagnostic plots;
- success and failure criteria.

Do not mix this with the current Codex task decomposition.

---

## 26. Explicit rationale for nonlinear readout support

The first solver sanity run should use linear ridge readout because it is closed-form and isolates the core solver. However, the implementation should also include nonlinear MLP readout because the DSE-style staged prediction path allows nonlinear read-out maps, while linear readout is the closed-form special case.

Therefore:

- `linear_ridge` is mandatory for tests and core sanity.
- `mlp` is mandatory as a configurable prediction module.
- The two readouts share the same input `H_prior = V_B A_filter_minus` and target `Z_M`.
- MLP readout must not update `V_A`, `V_B`, or `A_filter_*`.

---

## 27. Final instruction for Codex

Whenever giving a task to Codex, include:

```text
Follow dsl_elto_version0_implementation_spec.md and dsl_elto_codex_task_decomposition_v0.md.
Implement only the requested task group.
Do not introduce unsupported methods or additional algorithmic variants.
Keep mathematical shapes and naming conventions consistent with the spec.
Add tests for every implemented component.
```


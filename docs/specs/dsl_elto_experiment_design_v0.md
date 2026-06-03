# DSL / ELTO Version 0 Experiment Settings

This document defines the first experiment-setting plan after the Version 0
implementation and toy sanity check. It is not an experiment runner
specification, benchmark report, or performance claim.

The first experiments should verify core solver behavior and causal filtering
path wiring before moving toward benchmark-style comparisons.

## Source Context

- Implementation spec: `docs/specs/dsl_elto_version0_implementation_spec.md`
- Task decomposition: `docs/specs/dsl_elto_codex_task_decomposition_v0.md`
- Implementation status: `docs/notes/version0_implementation_status.md`
- Mathematical background: `docs/formulation/_formulation_memo (10).tex`

## Purpose

The experiment settings document fixes the order and interpretation of early
Version 0 experiments.

Goals:

- separate toy sanity, minimal sanity benchmark, linear SSM sanity, nonlinear
  observation benchmark, and later non-Gaussian / switching benchmarks;
- prevent leakage between Phase 1 training coordinates and Phase 2 causal
  prediction coordinates;
- define metrics, baselines, ablations, diagnostic plots, success criteria, and
  failure criteria before adding benchmark code;
- prevent toy sanity results from being interpreted as performance evidence.

## Claims Not Made In The First Experiments

The first experiment stage must not claim:

- method superiority;
- benchmark performance;
- SOTA or baseline superiority;
- general effectiveness on nonlinear dynamics;
- general effectiveness on non-Gaussian or switching dynamics;
- recovery or identification of the true physical latent state;
- equivalence to Gaussian latent SSM likelihood, standard ELBO, exact KKR,
  exact KKF, or DSE CCA.

## Stage 1: Minimal Sanity Benchmark

This stage is still a sanity check. The word benchmark here means a fixed,
repeatable sanity setting, not a performance benchmark.

Purpose:

- verify implementation wiring;
- verify `[T, d]` shapes;
- verify finite objective components;
- verify causal filtering recursion;
- verify readout prediction from `A_filter_minus`.

Setting:

- reuse the existing 2D damped rotation / oscillator sanity data;
- use precomputed `M = Y` and `B = Y`;
- use `linear_ridge` as the primary sanity readout;
- use `mlp` only as an optional readout sanity check.

Allowed claim:

- the pipeline runs without numerical failure on the fixed sanity setting.

Not allowed:

- treating toy sanity metrics as performance results;
- comparing against serious baselines;
- adding broad benchmark suites.

## Stage 2: Linear SSM Sanity Check

Purpose:

- check whether the Version 0 core solver and causal filtering path remain
  numerically stable on a simple known linear state-space structure.

Setting:

- low-dimensional stable linear dynamics;
- linear or mildly nonlinear observation features;
- synthetic data where true latent state and one-step targets are available for
  diagnostics;
- feature-level one-step prediction as the primary target.

Suggested checks:

- finite objective components across seeds;
- bounded filtering innovations and corrections;
- stable or interpretable `V_A` spectral radius;
- bounded Phase 1 / Phase 2 coordinate gap;
- prediction metrics compared with simple lightweight baselines.

Allowed claim:

- the implementation behaves stably on a known simple synthetic system.

Not allowed:

- claiming broad model accuracy or superiority.

## Stage 3: Nonlinear Observation Benchmark

Purpose:

- separate observation/readout nonlinearity from core dynamics.

Setting:

- stable linear or smooth low-dimensional latent dynamics;
- nonlinear observation map;
- compare `linear_ridge` and `mlp` readouts on the same `H_prior`;
- keep the core solver fixed while training MLP readout.

Suggested checks:

- MLP readout train loss decreases;
- feature prediction MSE is finite;
- held-out or validation loss is reported if splits are introduced;
- readout comparison is described as diagnostic, not final superiority.

Allowed claim:

- nonlinear readout may improve feature prediction diagnostics in this setting.

Not allowed:

- claiming that the core solver itself improved because MLP readout improved.

## Stage 4: Non-Gaussian / Switching Dynamics

This stage should be delayed.

Only introduce it after:

- minimal sanity benchmark is stable;
- linear SSM sanity check is stable;
- nonlinear observation benchmark is stable;
- evaluation metrics and failure criteria are fixed;
- leakage rules are documented;
- baseline scope is agreed.

Recommended interpretation:

- treat non-Gaussian / switching dynamics first as robustness and failure-mode
  diagnostics;
- do not treat it as the main Version 0 benchmark until scalar metrics and
  single-trajectory assumptions are clearly justified.

## Evaluation Metrics

Objective components:

- `L_init`
- `L_obs`
- `L_corr`
- `L_trans`
- `L_total`

Filtering diagnostics:

- mean innovation norm;
- max innovation norm;
- mean correction norm;
- max correction norm;
- prior prediction norm;
- Phase 1 / Phase 2 coordinate gap.

Prediction metrics:

- standardized feature MSE;
- unstandardized feature MSE;
- one-step prediction MSE;
- rollout MSE only after the one-step prediction path is stable.

Operator diagnostics:

- spectral radius of `V_A`;
- Frobenius norm of `V_A`;
- Frobenius norm of `V_B`;
- ridge solve condition diagnostics if available.

Readout diagnostics:

- linear ridge residual;
- MLP train loss;
- MLP validation loss if a split is introduced;
- MLP overfit gap.

## Baseline Candidates

Baselines should remain lightweight and must not pull in excluded Version 0
methods.

Allowed early baselines:

- persistence baseline: `M_pred[t + 1] = M[t]`;
- linear autoregressive feature baseline: ridge map from current features to
  next features;
- observation-space linear dynamics baseline: direct one-step ridge in feature
  space;
- oracle synthetic reference when true synthetic dynamics are known, used only
  as diagnostic reference.

Not included:

- DSE CCA;
- SVD/PCA state initialization;
- KKR covariance recursion;
- stochastic realization state construction;
- residual-calibrated metrics;
- learned inference encoder.

## Ablation Candidates

Initial ablations should be config-level or diagnostic-level.

Allowed early ablations:

- readout: `linear_ridge` vs `mlp`;
- regularization: `lambda_a`, `lambda_b`, `lambda_readout`;
- latent dimension: `d_A`;
- scalar metrics: `sigma_b2`, `sigma_a2`, `p`, `c0`;
- preprocessing: RMS scale `eps`;
- solver: max outer iteration and convergence tolerance;
- filtering: innovation and correction magnitude diagnostics.

Not included in Version 0 ablations:

- KKR recursion;
- residual calibration;
- SVD/PCA initialization;
- DSE CCA;
- multiple trajectory support.

## Diagnostic Plots

Recommended plots:

- objective component curves over outer iteration;
- `L_total` curve;
- innovation norm over time;
- correction norm over time;
- `A_train_minus` norm over time;
- `A_filter_minus` norm over time;
- Phase 1 / Phase 2 coordinate gap over time;
- one-step prediction traces against target features;
- prediction residual histogram;
- `V_A` eigenvalue plot or spectral radius summary;
- MLP train loss curve when MLP readout is used.

## Success Criteria

Minimal sanity:

- all saved arrays are finite;
- objective components are finite;
- no coordinate explosion;
- `A_filter_minus[0] = a_init`;
- prediction embedding uses `A_filter_minus`;
- `A_filter_plus` is not used for main prediction;
- `A_train_*` is not used for main prediction;
- script produces metrics and logs.

Linear SSM sanity:

- objective and prediction metrics are finite across seeds;
- `V_A` spectral radius is stable or interpretable;
- prediction improves over persistence or is at least competitive in simple
  cases;
- Phase 1 / Phase 2 coordinate gap remains bounded.

Nonlinear observation:

- MLP readout reduces feature loss compared with its own initial loss;
- comparison to linear readout is reported as readout diagnostic;
- held-out diagnostics are added before making stronger claims.

## Failure Criteria

Any of the following should block performance-style interpretation:

- NaN or inf in core arrays, objective components, filtering path, readout
  output, or metrics;
- `A_train_*` used for main prediction;
- `A_filter_plus` used for main prediction;
- coordinate explosion beyond configured threshold;
- `V_A` spectral radius far above 1 with unstable filtering rollout;
- MLP readout improves training loss but degrades held-out prediction badly;
- unexplained large Phase 1 / Phase 2 coordinate gap;
- toy sanity results described as benchmark performance;
- forbidden methods appear in implementation or experiment plan.

## Codex Task Decomposition For Future Work

1. Create or update experiment-setting documentation only.
   - No code.
   - No runner.
   - No benchmark outputs.

2. Add experiment config templates later.
   - Minimal sanity config.
   - Linear SSM sanity config.
   - Nonlinear observation config.
   - No non-Gaussian / switching config until explicitly approved.

3. Add data generators later, one group at a time.
   - Linear SSM synthetic generator.
   - Nonlinear observation synthetic generator.
   - Non-Gaussian / switching generator only after prior stages pass.

4. Add experiment runner later.
   - Preserve `A_filter_minus` prediction invariant.
   - Save metrics/logs without performance claims.
   - Exclude forbidden Version 0 methods.

5. Add baselines and ablations last.
   - Start with persistence and feature-space ridge baselines.
   - Keep ablations config-level.
   - Do not add DSE CCA, SVD/PCA initialization, KKR recursion, or residual
     calibration.

## Open Issues Before Implementation

- Whether to introduce train/validation/test splits before nonlinear observation
  benchmarks.
- Whether feature-level evaluation is sufficient before raw prediction.
- How to define leakage rules for readout fitting and evaluation windows.
- Whether MLP readout needs explicit deterministic seed controls.
- How to report Phase 1 / Phase 2 coordinate gap.
- Which baseline should be the first minimal baseline.
- Whether non-Gaussian / switching dynamics should be robustness diagnostics
  rather than benchmark.
- How to separate sanity outputs from future benchmark outputs on disk.
- Whether to add config-load-time validation for unsupported MLP activations
  before experiment code.

## Explicit Exclusions

The experiment setting must not introduce the following into Version 0:

- DSE CCA;
- stochastic realization state construction;
- SVD/PCA state initialization;
- direct `A = G @ B` or `A = G @ Z_B` state construction;
- inference encoder;
- multiple trajectories;
- residual-calibrated metrics;
- KKR-style covariance recursion;
- posterior reconstruction loss;
- encoder or dictionary refinement;
- hard whitening;
- full non-isotropic metrics.

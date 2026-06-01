# DSL/ELTO Embedded-Belief Operator Learning — Version 0 Implementation Specification

作成日: 2026-06-01  
対象: DSE / ELTO / KKR / KBR に関連する embedded latent transfer-operator learning  
参照 formulation memo: `_formulation_memo (10).tex`  
目的: Codex 実装へ進む前に、Version 0 の数理 convention、tensor shape、update、logging、test を固定する。

---

## 0. Status and intended use

This document is an implementation specification, not a replacement for the formulation memo.

The formulation memo defines the mathematical target: relaxed Galerkin coordinates of projected prior/posterior conditional mean embeddings. This specification fixes the concrete numerical conventions needed to implement the first solver and prediction pipeline.

Codex should use this file as the primary implementation guide. The formulation memo should be used as the mathematical reference, especially for terminology and motivation.

---

## 1. Scope and non-goals

### 1.1 Scope

Version 0 implements a single-trajectory embedded-state/operator learning pipeline:

\[
\{y_t\}_{t=1}^T.
\]

It includes:

1. fixed feature preparation from observations;
2. feature centering and standardization for numerical implementation;
3. batch learning of training-time variational coordinates
   \[
   A^-_{\mathrm{train}}, A^+_{\mathrm{train}};
   \]
4. closed-form Galerkin ridge updates for
   \[
   V_A, V_B;
   \]
5. scalar-metric closed-form coordinate updates;
6. causal filtering path construction
   \[
   A^-_{\mathrm{filter}}, A^+_{\mathrm{filter}};
   \]
7. linear ridge readout and nonlinear MLP readout from prior-coordinate prediction;
8. raw prediction through a fixed decoder when available;
9. objective logging, diagnostics, stopping, failure checks, and unit tests.

### 1.2 Non-goals

Version 0 explicitly excludes:

- DSE functional CCA / stochastic-realization state construction;
- DSE CCA state warm start;
- SVD / PCA initialization of state coordinates;
- direct observation-to-state warm starts such as \(A=G\tilde B\);
- inference encoder / amortized coordinate inference;
- multiple trajectories and trajectory boundary masks;
- residual-calibrated metrics;
- KKR-style covariance or inverse-curvature recursion;
- posterior reconstruction loss from \(V_B A^+_{\mathrm{filter}}\);
- encoder / observation dictionary refinement;
- decoder refinement in the core solver;
- hard whitening of latent coordinates;
- full non-isotropic metric-weighted operator updates.

### 1.3 Conceptual constraints

The implementation must not describe the method as:

- a Gaussian latent state-space model;
- a Gaussian latent SSM negative log-likelihood;
- a standard likelihood-based ELBO;
- an exact implementation of KKR / KKF;
- a reimplementation of KBR;
- a DSE CCA state-construction method.

Use the terminology:

- embedding free energy;
- loss-based variational free energy;
- relaxed Galerkin embedded-belief coordinates;
- KKR-like correction principle in learned Galerkin coordinates.

---

## 2. Mathematical conventions fixed for implementation

### 2.1 Observation feature path

Given observations \(y_t\), compute:

\[
m_t = u_\eta(y_t),
\qquad
b_t = \psi_\omega(m_t).
\]

Here:

- \(m_t\in\mathbb R^m\) is the encoder feature used for readout / decoder prediction;
- \(b_t\in\mathbb R^{d_B}\) is the observation embedding coordinate used by the core operator-learning problem.

The core solver should be able to operate on precomputed \(M=[m_t]\) and \(B=[b_t]\). It should not require training or modifying \(u_\eta\) or \(\psi_\omega\).

### 2.2 Standardized implementation variables

The formulation memo uses centered variables \(\tilde b_t,\tilde m_t\). The implementation uses standardized variables by default:

\[
z^b_t = D_b^{-1}(b_t-\bar b),
\qquad
z^m_t = D_m^{-1}(m_t-\bar m),
\]

where \(D_b\) and \(D_m\) are diagonal RMS-scale matrices.

In implementation equations, use:

\[
Z_B = [z^b_1,\dots,z^b_T]\in\mathbb R^{d_B\times T},
\qquad
Z_M = [z^m_1,\dots,z^m_T]\in\mathbb R^{m\times T}.
\]

This standardization is a numerical convention for scalar-metric optimization. It is not a Gaussian assumption.

### 2.3 Training coordinates versus causal filtering coordinates

Use distinct names in code.

Training-time variational coordinates:

\[
A^-_{\mathrm{train}}, A^+_{\mathrm{train}}\in\mathbb R^{d_A\times T}.
\]

Causal filtering coordinates:

\[
A^-_{\mathrm{filter}}, A^+_{\mathrm{filter}}\in\mathbb R^{d_A\times T}.
\]

Do not use the same object for both.

The bar notation \(\bar A^-,\bar A^+\) in the memo means causal filtering path, not an average. In code, prefer:

```python
A_train_minus
A_train_plus
A_filter_minus
A_filter_plus
```

### 2.4 Prior and posterior coordinate semantics

Training coordinates are variational variables with intended embedded-belief semantics:

\[
a_t^- \approx P_A\mathbb E[\phi_x(X_t)\mid Y_{1:t-1}],
\]

\[
a_t^+ \approx P_A\mathbb E[\phi_x(X_t)\mid Y_{1:t}].
\]

They are not arbitrary latent vectors and are not direct outputs of the observation encoder.

### 2.5 Observation embeddings do not define state coordinates

\(Z_B\) or \(\tilde B\) is not used to define \(A^-\) or \(A^+\). It enters the learning problem through observation consistency and correction terms.

The numerical seed for \(A^{\pm,(0)}\) does not define the latent coordinate system. The coordinate semantics are imposed by the embedding free energy and the learned causal prediction-correction recursion.

---

## 3. Inputs and outputs

### 3.1 Minimal core solver inputs

The core solver should support precomputed features:

- `B`: observation embedding matrix, shape `[T, d_B]` in code;
- `M`: encoder feature matrix, shape `[T, m]` in code, needed for readout;
- `d_A`: latent coordinate dimension;
- scalar metric hyperparameters;
- ridge parameters;
- optional `a_init`, shape `[d_A]`;
- random seed for initialization.

Mathematical notation uses columns \([d,T]\). Code may store arrays as `[T,d]`; all functions must document this choice and shape assertions must be included.

### 3.2 Optional full pipeline inputs

If the full pipeline is implemented, allow:

- `encoder`: fixed module \(u_\eta\);
- `obs_dictionary`: fixed module \(\psi_\omega\);
- `decoder`: fixed module \(g_\alpha\);
- raw observation sequence `Y`, shape `[T, ...]`.

If raw observations are not supplied or no decoder is available, the solver should still run and report feature-level predictions in \(Z_M\)-space or \(M\)-space.

### 3.3 Outputs

Core outputs:

- `A_train_minus`: `[T, d_A]`;
- `A_train_plus`: `[T, d_A]`;
- `V_A`: `[d_A, d_A]`;
- `V_B`: `[d_B, d_A]`;
- preprocessing statistics: `b_mean`, `b_scale`, `m_mean`, `m_scale`;
- objective logs and diagnostics.

Filtering outputs:

- `A_filter_minus`: `[T, d_A]`;
- `A_filter_plus`: `[T, d_A]`;
- standardized predicted observation embeddings `ZB_pred_prior`: `[T, d_B]`;
- innovation sequence: `[T, d_B]`;
- correction norms.

Readout / prediction outputs:

- `readout_type`: `linear_ridge` or `mlp`;
- linear readout matrix `R_readout` when applicable: `[m, d_B]`;
- MLP readout module when applicable;
- standardized feature predictions `ZM_pred`: `[T, m]`;
- unstandardized feature predictions `M_pred`: `[T, m]`;
- raw predictions `Y_pred` if decoder is available.

---

## 4. Tensor / matrix shapes

### 4.1 Mathematical convention

Formulas use column-major mathematical matrices:

\[
Z_B\in\mathbb R^{d_B\times T},
\qquad
Z_M\in\mathbb R^{m\times T},
\]

\[
A^-_{\mathrm{train}}, A^+_{\mathrm{train}}\in\mathbb R^{d_A\times T},
\]

\[
V_A\in\mathbb R^{d_A\times d_A},
\qquad
V_B\in\mathbb R^{d_B\times d_A}.
\]

### 4.2 Code convention

Recommended code convention:

```text
B:              [T, d_B]
M:              [T, m]
Z_B:            [T, d_B]
Z_M:            [T, m]
A_train_minus:  [T, d_A]
A_train_plus:   [T, d_A]
A_filter_minus: [T, d_A]
A_filter_plus:  [T, d_A]
V_A:            [d_A, d_A]
V_B:            [d_B, d_A]
R_readout:      [m, d_B]
```

When applying formulas, transpose as needed. For example, code prediction can use:

```python
ZB_pred = A_filter_minus @ V_B.T   # [T, d_B]
```

### 4.3 Shape assertions

Every public function must assert expected shapes. At minimum:

```python
assert Z_B.ndim == 2
assert Z_M.ndim == 2
assert Z_B.shape[0] == Z_M.shape[0] == T
assert V_A.shape == (d_A, d_A)
assert V_B.shape == (d_B, d_A)
assert A_train_plus.shape == (T, d_A)
```

---

## 5. Feature preprocessing

### 5.1 Center + RMS scale

Compute means:

\[
\bar b=\frac1T\sum_{t=1}^T b_t,
\qquad
\bar m=\frac1T\sum_{t=1}^T m_t.
\]

Compute RMS scales elementwise:

\[
s_{b,j}=\sqrt{\frac1T\sum_{t=1}^T (b_{j,t}-\bar b_j)^2+\epsilon_{\mathrm{scale}}},
\]

\[
s_{m,j}=\sqrt{\frac1T\sum_{t=1}^T (m_{j,t}-\bar m_j)^2+\epsilon_{\mathrm{scale}}}.
\]

Then:

\[
z^b_t=D_b^{-1}(b_t-\bar b),
\qquad
z^m_t=D_m^{-1}(m_t-\bar m).
\]

### 5.2 Why scaling is used

Scaling is used because Version 0 uses scalar metrics:

\[
R_B=\sigma_B^2 I,
\qquad
Q_A=\sigma_A^2 I,
\qquad
P_t^-=pI.
\]

If feature dimensions have very different scales, isotropic residual weights become numerically inappropriate. RMS scaling makes the default \(\sigma_B^2=1\) more meaningful.

This is not a Gaussian assumption.

### 5.3 Stored preprocessing state

Store:

```python
b_mean:  [d_B]
b_scale: [d_B]
m_mean:  [m]
m_scale: [m]
```

At test time, use training statistics only. Never compute test-sequence mean or test-sequence scale for causal prediction.

### 5.4 Inverse transform for feature prediction

If readout predicts \(\hat z^m_{t|t-1}\), recover encoder feature prediction by:

\[
\hat m_{t|t-1}=\bar m+D_m\hat z^m_{t|t-1}.
\]

In code:

```python
M_pred = ZM_pred * m_scale + m_mean
```

---

## 6. Initialization / warm start

### 6.1 Conceptual rule

The initialization of \(A^-\) and \(A^+\) is a numerical seed only. It is not a state construction principle and not a statistical target.

Do not implement direct warm starts of the form:

\[
A^{+,(0)}=GZ_B.
\]

Do not use SVD / PCA / DSE CCA coordinates.

### 6.2 Default initial prior

If no meaningful initial embedded law is provided:

\[
a_{\mathrm{init}}=0\in\mathbb R^{d_A}.
\]

This is a reference prior in the centered / standardized coordinate gauge, not a claim about the true initial distribution.

If a meaningful initial coordinate is provided, use it directly after verifying shape `[d_A]`.

### 6.3 Correction-based warm start

Use a correction-based warm start for Version 0.

Define:

\[
w_B=\sigma_B^{-2},
\qquad
w_P=p^{-1}.
\]

Initialize \(V_B^{(0)}\in\mathbb R^{d_B\times d_A}\) as a nondegenerate random matrix.

Recommended procedure:

1. draw entries from \(\mathcal N(0,1/d_A)\);
2. normalize columns to avoid near-zero columns;
3. optionally use QR if \(d_B\ge d_A\), but this is not required for Version 0.

Initialize:

\[
V_A^{(0)}=0.
\]

Initialize prior coordinates:

\[
a_t^{-,(0)}=a_{\mathrm{init}}
\quad\text{for all }t.
\]

Compute initial posterior coordinates by solving:

\[
a_t^{+,(0)}
=
\left(
 w_B(V_B^{(0)})^\top V_B^{(0)}+w_P I
\right)^{-1}
\left(
 w_B(V_B^{(0)})^\top z^b_t+w_Pa_{\mathrm{init}}
\right).
\]

Then set:

\[
a_1^{-,(0)}=a_{\mathrm{init}},
\qquad
 a_t^{-,(0)}=a_t^{+,(0)}\quad(t\ge2).
\]

This warm start lets observations influence \(A^+\) through the correction objective, without defining \(A\) directly from \(B\).

### 6.4 Numerical stability notes

Use `torch.linalg.solve` or a Cholesky-based solver. Do not explicitly invert matrices unless only for debugging.

Add a small diagonal jitter if necessary:

\[
H\leftarrow H+\epsilon_{\mathrm{solve}} I.
\]

### 6.5 No latent row centering

Do not row-center latent coordinates in Version 0.

Latent row-centering is affine and can break the meaning of \(a_{\mathrm{init}}\) unless all operators and offsets are transformed consistently.

Hard whitening is excluded.

Optional row RMS scaling may be added later as a numerical stabilization, but it is not part of the default Version 0 pipeline.

---

## 7. Objective components

The implementation objective uses standardized observation embeddings \(z^b_t\).

### 7.1 Initial boundary term

\[
L_{\mathrm{init}}
=
\frac12
\|a_1^- - a_{\mathrm{init}}\|_{P_0^{-1}}^2.
\]

For scalar metrics:

\[
P_0=c_0pI,
\qquad
L_{\mathrm{init}}
=
\frac{1}{2c_0p}
\|a_1^- - a_{\mathrm{init}}\|_2^2.
\]

### 7.2 Observation consistency

\[
L_{\mathrm{obs}}
=
\frac12\sum_{t=1}^T
\|z^b_t - V_Ba_t^+\|_{R_B^{-1}}^2.
\]

For scalar metrics:

\[
L_{\mathrm{obs}}
=
\frac{w_B}{2}\sum_{t=1}^T
\|z^b_t - V_Ba_t^+\|_2^2.
\]

### 7.3 Correction term

\[
L_{\mathrm{corr}}
=
\frac12\sum_{t=1}^T
\|a_t^+ - a_t^-\|_{(P_t^-)^\dagger}^2.
\]

For scalar metrics:

\[
L_{\mathrm{corr}}
=
\frac{w_P}{2}\sum_{t=1}^T
\|a_t^+ - a_t^-\|_2^2.
\]

### 7.4 Transfer consistency

\[
L_{\mathrm{trans}}
=
\frac12\sum_{t=1}^{T-1}
\|a_{t+1}^- - V_Aa_t^+\|_{Q_A^{-1}}^2.
\]

For scalar metrics:

\[
L_{\mathrm{trans}}
=
\frac{w_A}{2}\sum_{t=1}^{T-1}
\|a_{t+1}^- - V_Aa_t^+\|_2^2.
\]

### 7.5 Total embedding free energy

\[
L_{\mathrm{total}}
=
L_{\mathrm{init}}+L_{\mathrm{obs}}+L_{\mathrm{corr}}+L_{\mathrm{trans}}.
\]

Set \(\Omega_A=0\) in Version 0.

### 7.6 Operator ridge logs

Operator ridge losses should be logged separately and not added again to \(L_{\mathrm{total}}\):

\[
L_{\mathrm{ridge},A},
\qquad
L_{\mathrm{ridge},B}.
\]

---

## 8. Closed-form operator updates

Use normalized covariance form for ridge updates so that \(\lambda_A\) and \(\lambda_B\) do not depend strongly on sequence length.

### 8.1 Observable operator \(V_B\)

Mathematical form:

\[
V_B
=
\left(
\frac1T Z_B(A^+)^\top
\right)
\left(
\frac1T A^+(A^+)^\top+\lambda_BI
\right)^{-1}.
\]

Code convention with `[T,d]` arrays:

```python
C_yx = (Z_B.T @ A_train_plus) / T          # [d_B, d_A]
C_xx = (A_train_plus.T @ A_train_plus) / T # [d_A, d_A]
V_B = C_yx @ solve(C_xx + lambda_b * I, I)
```

Prefer solving linear systems rather than explicitly computing inverse.

### 8.2 Transfer operator \(V_A\)

Use:

\[
V_A
=
\left(
\frac1{T-1}A^-_{2:T}(A^+_{1:T-1})^\top
\right)
\left(
\frac1{T-1}A^+_{1:T-1}(A^+_{1:T-1})^\top+\lambda_AI
\right)^{-1}.
\]

Code convention:

```python
X = A_train_plus[:-1]   # [T-1, d_A]
Y = A_train_minus[1:]   # [T-1, d_A]
C_yx = (Y.T @ X) / (T - 1)  # [d_A, d_A]
C_xx = (X.T @ X) / (T - 1)  # [d_A, d_A]
V_A = C_yx @ solve(C_xx + lambda_a * I, I)
```

### 8.3 Minimum length

Require \(T\ge2\) for transfer learning. If \(T=1\), skip \(V_A\) learning and raise a clear error for the core solver unless a special single-step mode is explicitly requested.

---

## 9. Coordinate updates

Use scalar metrics only in Version 0.

Define:

\[
w_B=\sigma_B^{-2},
\qquad
w_A=\sigma_A^{-2},
\qquad
w_P=p^{-1},
\qquad
w_0=(c_0p)^{-1}.
\]

### 9.1 Posterior-coordinate update

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

\[
a_t^+=(H_t^+)^{-1}h_t^+.
\]

For \(t=T\):

\[
H_T^+
=
w_BV_B^\top V_B+w_PI,
\]

\[
h_T^+
=
w_BV_B^\top z^b_T+w_Pa_T^-.
\]

Implementation notes:

- cache factorizations of common \(H_t^+\) where possible;
- use linear solve;
- add jitter if Cholesky fails.

### 9.2 Prior-coordinate update

At \(t=1\):

\[
a_1^-
=
\frac{w_Pa_1^+ + w_0a_{\mathrm{init}}}{w_P+w_0}.
\]

For \(t=2,\dots,T\):

\[
a_t^-
=
\frac{w_Pa_t^+ + w_AV_Aa_{t-1}^+}{w_P+w_A}.
\]

### 9.3 Outer-loop order

Use the following order:

1. initialize / warm start;
2. update \(V_A,V_B\) from current \(A_{\mathrm{train}}^\pm\);
3. update all \(A^+_{\mathrm{train}}\);
4. update all \(A^-_{\mathrm{train}}\);
5. compute objective components;
6. log diagnostics;
7. check stopping.

Do not update metrics inside the Version 0 loop.

---

## 10. Causal filtering path

After Phase 1, construct a new causal filtering path. Do not use \(A_{\mathrm{train}}^\pm\) as prediction states.

### 10.1 Gain

For fixed scalar metrics:

\[
K=pV_B^\top(pV_BV_B^\top+\sigma_B^2I)^{-1}.
\]

Since metrics are fixed, \(K\) is time-independent.

### 10.2 Recursion

Initialize:

\[
a_{\mathrm{filter},1}^-=a_{\mathrm{init}}.
\]

For \(t=1,\dots,T\):

\[
\hat z^b_{t|t-1}=V_Ba_{\mathrm{filter},t}^-.
\]

\[
r_t=z^b_t-\hat z^b_{t|t-1}.
\]

\[
a_{\mathrm{filter},t}^+
=
a_{\mathrm{filter},t}^-+Kr_t.
\]

For \(t<T\):

\[
a_{\mathrm{filter},t+1}^-
=
V_Aa_{\mathrm{filter},t}^+.
\]

### 10.3 Code naming

Use:

```python
A_filter_minus
A_filter_plus
ZB_pred_prior
innovation
```

### 10.4 Prediction rule

Main prediction uses the prior coordinate:

\[
V_Ba_{\mathrm{filter},t}^-.
\]

Do not use \(V_Ba_{\mathrm{filter},t}^+\) as the main prediction. That is same-time posterior reconstruction.

---

## 11. Readout and raw prediction

Readout maps standardized observation-embedding predictions to standardized encoder-feature predictions.

Input to readout:

\[
h_t=V_Ba_{\mathrm{filter},t}^-\in\mathbb R^{d_B}.
\]

Target:

\[
z^m_t\in\mathbb R^m.
\]

### 11.1 Linear ridge readout

Always implement linear ridge readout.

Let:

\[
H^-=[h_1,\dots,h_T]\in\mathbb R^{d_B\times T}.
\]

The ridge solution is:

\[
R_{\xi_B}
=
\left(
\frac1T Z_M(H^-)^\top
\right)
\left(
\frac1T H^-(H^-)^\top+
\lambda_r I
\right)^{-1}.
\]

Prediction:

\[
\hat z^m_{t|t-1}=R_{\xi_B}h_t.
\]

Code convention:

```python
H = ZB_pred_prior          # [T, d_B]
C_yx = (Z_M.T @ H) / T     # [m, d_B]
C_xx = (H.T @ H) / T       # [d_B, d_B]
R_readout = C_yx @ solve(C_xx + lambda_readout * I, I)
ZM_pred = H @ R_readout.T  # [T, m]
```

### 11.2 Nonlinear MLP readout

Also implement nonlinear readout as a configurable main option.

Config:

```yaml
readout_type: "linear_ridge" | "mlp"
mlp_hidden_dims: [128, 128]
mlp_activation: "relu"
mlp_lr: 1e-3
mlp_weight_decay: 1e-4
mlp_epochs: 1000
mlp_batch_size: null  # null means full batch
```

MLP objective:

\[
L_{\mathrm{readout,feat}}
=
\frac1T\sum_{t=1}^T
\|z^m_t-r_{\xi_B}^{\mathrm{MLP}}(h_t)\|_2^2
+
\lambda_r\Omega(\xi_B).
\]

The core solver and causal filtering path are fixed while training the MLP readout.

### 11.3 Default readout usage

Use:

- `linear_ridge` for unit tests, core solver sanity checks, and fast diagnostics;
- `mlp` for the main nonlinear readout pipeline when raw prediction quality is evaluated.

### 11.4 Unstandardization

For either readout type:

\[
\hat m_{t|t-1}=\bar m+D_m\hat z^m_{t|t-1}.
\]

Code:

```python
M_pred = ZM_pred * m_scale + m_mean
```

### 11.5 Raw prediction

If a fixed decoder \(g_\alpha\) is available:

\[
\hat y_{t|t-1}=g_\alpha(\hat m_{t|t-1}).
\]

If no decoder is available, report feature-level prediction metrics only.

### 11.6 Do not train decoder in Version 0 core

Decoder refinement is excluded from Version 0 core implementation. A fixed pretrained decoder may be used.

---

## 12. Config defaults

Suggested YAML-style defaults:

```yaml
scope:
  single_trajectory: true
  allow_multiple_trajectories: false

preprocessing:
  standardize_features: true
  eps_scale: 1.0e-8

model_dims:
  d_A: null   # must be specified
  d_B: null   # inferred from B if precomputed
  m: null     # inferred from M if precomputed

initial_prior:
  use_provided_a_init: false
  c0: 10.0

metrics:
  sigma_b2: 1.0
  sigma_a2: 1.0
  p: 1.0

ridge:
  lambda_a: 1.0e-3
  lambda_b: 1.0e-3
  lambda_readout: 1.0e-3

warm_start:
  method: "correction_based"
  random_seed: 0
  vb0_column_normalize: true
  solve_jitter: 1.0e-8

optimization:
  max_outer_iter: 100
  tol_rel_loss: 1.0e-6
  min_outer_iter: 3

readout:
  readout_type: "linear_ridge"  # use "mlp" for nonlinear pipeline
  mlp_hidden_dims: [128, 128]
  mlp_activation: "relu"
  mlp_lr: 1.0e-3
  mlp_weight_decay: 1.0e-4
  mlp_epochs: 1000
  mlp_batch_size: null

diagnostics:
  compute_spectral_radius: true
  compute_phase1_phase2_gap: true
  burn_in_steps_for_reporting: 0
```

Notes:

- For unit tests, set `readout_type: linear_ridge`.
- For nonlinear prediction experiments, set `readout_type: mlp`.
- `c0=10` assumes fallback \(a_{\mathrm{init}}=0\). If a meaningful initial coordinate is provided, consider `c0=1`.

---

## 13. Logging and diagnostics

### 13.1 Objective logs per outer iteration

Log:

```text
L_init
L_obs
L_corr
L_trans
L_total
```

### 13.2 Operator diagnostics

Log:

```text
fro_norm_V_A
fro_norm_V_B
spectral_norm_V_A
spectral_radius_V_A
condition_number_ridge_A
condition_number_ridge_B
```

### 13.3 Coordinate diagnostics

Log:

```text
fro_norm_A_train_minus
fro_norm_A_train_plus
mean_norm_A_train_minus
mean_norm_A_train_plus
max_norm_A_train_minus
max_norm_A_train_plus
```

### 13.4 Filtering diagnostics

Log after Phase 2:

```text
mean_innovation_norm
max_innovation_norm
mean_correction_norm
max_correction_norm
mean_prior_prediction_norm
phase1_phase2_minus_gap
phase1_phase2_plus_gap
```

where:

\[
\mathrm{gap}^-=
\frac1T\sum_t
\|a_{\mathrm{train},t}^- - a_{\mathrm{filter},t}^-\|_2,
\]

\[
\mathrm{gap}^+=
\frac1T\sum_t
\|a_{\mathrm{train},t}^+ - a_{\mathrm{filter},t}^+\|_2.
\]

### 13.5 Prediction diagnostics

For linear or MLP readout:

```text
feature_mse_standardized
feature_mse_unstandardized
raw_prediction_loss  # if decoder and raw targets are available
```

Report both full-sequence and optional burn-in-excluded metrics.

---

## 14. Stopping criteria

### 14.1 Relative loss decrease

Stop if:

\[
\frac{|L^{(k)}-L^{(k-1)}|}{1+|L^{(k-1)}|}<\epsilon
\]

after at least `min_outer_iter` iterations.

Default:

```yaml
tol_rel_loss: 1e-6
min_outer_iter: 3
max_outer_iter: 100
```

### 14.2 Always stop on numerical failure

Stop and raise a clear error or warning if:

- loss is NaN or inf;
- any coordinate contains NaN or inf;
- ridge solve fails even after jitter;
- norm explosion exceeds configured threshold.

### 14.3 Non-monotonicity

Because operator updates and coordinate updates are interleaved, and because operator ridge losses are not added to the embedding free energy as separate terms, strict monotone decrease is not guaranteed. The implementation should log losses rather than assuming monotonicity.

---

## 15. Failure modes

### 15.1 Degenerate warm start

Symptoms:

- \(A^+\) nearly zero;
- \(V_B\) nearly zero;
- observation loss does not decrease.

Mitigation:

- check random seed;
- ensure \(V_B^{(0)}\) is nondegenerate;
- add solve jitter;
- inspect feature standardization.

### 15.2 Observation operator collapse

Symptoms:

- \(\|V_B\|\) near zero;
- prediction \(V_Ba\) near zero;
- high observation loss.

Mitigation:

- increase ridge stability only if ill-conditioned;
- verify \(A^+\) has nonzero variance;
- verify \(Z_B\) is not nearly zero after preprocessing.

### 15.3 Unstable transfer operator

Symptoms:

- spectral radius \(\rho(V_A)\gg1\);
- filtering path explodes;
- long rollout unstable.

Mitigation for Version 0:

- log only;
- do not add stability regularization yet;
- revisit after first run.

### 15.4 Incorrect use of training coordinates for prediction

Symptoms:

- prediction appears too good due to leakage;
- code uses `A_train_minus` or `A_train_plus` in readout.

Rule:

- readout and raw prediction must use `A_filter_minus` only.

### 15.5 Posterior reconstruction leakage

Symptoms:

- prediction uses \(V_BA^+_{\mathrm{filter}}\);
- same-time observation is already incorporated.

Rule:

- do not use posterior reconstruction as main prediction.

### 15.6 Standardization mismatch

Symptoms:

- test-time prediction degrades unexpectedly;
- different means/scales used for train and test.

Rule:

- always use training mean and scale.

### 15.7 MLP readout overfitting

Symptoms:

- MLP readout feature loss becomes very low but generalization poor;
- raw prediction unstable.

Mitigation:

- compare to linear ridge readout;
- use validation split if available;
- apply weight decay and early stopping;
- keep core solver fixed.

---

## 16. Unit tests

### 16.1 Preprocessing tests

1. Given random `B` and `M`, standardization returns `Z_B`, `Z_M` with correct shapes.
2. Means of standardized features are approximately zero.
3. RMS scales are approximately one for nonconstant dimensions.
4. Inverse transform reconstructs original `M` within tolerance.

### 16.2 Warm-start tests

1. Warm start returns finite `A_train_minus`, `A_train_plus`.
2. Shapes are `[T, d_A]`.
3. `A_train_minus[0] == a_init` within tolerance.
4. No all-zero coordinate arrays unless input data are degenerate.

### 16.3 Operator-update tests

1. `V_A` shape is `[d_A, d_A]`.
2. `V_B` shape is `[d_B, d_A]`.
3. Ridge update returns finite values.
4. Increasing `lambda_a` / `lambda_b` reduces operator norm on a fixed toy problem.

### 16.4 Coordinate-update tests

1. Posterior update returns finite `A_train_plus`.
2. Prior update returns finite `A_train_minus`.
3. Special handling of final time step \(t=T\) works.
4. Special handling of initial time step \(t=1\) works.

### 16.5 Objective tests

1. Objective components are nonnegative.
2. Total equals sum of components within tolerance.
3. Controlled simple update does not produce NaN / inf.

Monotone decrease should not be required as a hard unit test.

### 16.6 Causal filtering tests

1. `A_filter_minus`, `A_filter_plus` shapes are `[T, d_A]`.
2. `A_filter_minus[0] == a_init`.
3. Innovations have shape `[T, d_B]`.
4. Predictions use prior coordinates.
5. No dependency on future observations.

### 16.7 Readout tests

Linear ridge:

1. `R_readout` shape `[m, d_B]`.
2. `ZM_pred` shape `[T, m]`.
3. Closed-form solve finite.

MLP:

1. MLP accepts `[T, d_B]` and returns `[T, m]`.
2. One or more training epochs reduce feature MSE on a small toy dataset.
3. MLP training does not modify `V_A`, `V_B`, `A_filter_minus`, or `A_filter_plus`.

### 16.8 Exclusion tests

Ensure Version 0 code path does not call:

- SVD / PCA state initialization;
- DSE CCA state construction;
- residual metric calibration;
- KKR covariance recursion;
- posterior reconstruction loss.

---

## 17. Items explicitly excluded from Version 0

Do not implement in Version 0:

1. multiple trajectories;
2. trajectory boundary masks;
3. DSE CCA / stochastic realization state construction;
4. SVD / PCA initialization;
5. direct \(A=G\tilde B\) warm start;
6. inference encoder;
7. residual-calibrated metrics;
8. KKR-style covariance or inverse-curvature recursion;
9. full non-isotropic metrics;
10. hard whitening;
11. latent row centering;
12. encoder / observation dictionary refinement;
13. decoder refinement;
14. posterior reconstruction loss;
15. smoothing / backward filtering;
16. Koopman / spectrum experiments.

These may be added later as extensions or ablations.

---

## 18. Open issues to revisit after first run

After the first working implementation, revisit:

1. Is correction-based warm start stable enough?
2. Should feature scaling use RMS, robust scale, or no scale?
3. Is \(c_0=10\) appropriate for fallback \(a_{\mathrm{init}}=0\)?
4. Does nonlinear MLP readout overfit compared to linear ridge readout?
5. Does \(V_A\) require stability control?
6. Is residual calibration helpful after scalar metrics are stable?
7. Should KKR-style covariance recursion be added as an ablation?
8. Is multi-trajectory support needed?
9. Should encoder / dictionary refinement be introduced?
10. Is feature-level evaluation sufficient before raw prediction?
11. How large is the Phase 1 / Phase 2 coordinate gap?
12. Does the filtering path remain stable over long rollouts?

---

## 19. Suggested next steps after this spec

### Step 1: Human review

Review this spec and mark:

- accepted decisions;
- open implementation questions;
- equations requiring changes.

### Step 2: Codex task decomposition

Break implementation into small tasks:

1. preprocessing utilities;
2. warm-start initializer;
3. operator ridge updates;
4. coordinate updates;
5. objective/logging;
6. outer-loop solver;
7. causal filtering path;
8. linear readout;
9. MLP readout;
10. toy data generator;
11. unit tests.

### Step 3: Minimal toy sanity check

Before serious experiments, implement a very small synthetic dataset to verify:

- shapes;
- finite objectives;
- filtering recursion;
- readout training;
- one-step prediction pipeline.

This toy check is not a performance benchmark.

### Step 4: Experiment-design phase

Only after the solver and filtering path run should we design:

- synthetic linear SSM sanity check;
- nonlinear observation benchmark;
- non-Gaussian / switching benchmark;
- evaluation metrics;
- baselines;
- ablations;
- diagnostic plots.

---

## 20. Brief notes on first toy dynamics for implementation sanity

This section is intentionally preliminary. It is not the final experiment plan.

A minimal sanity check can use a scalar or two-dimensional latent process with nonlinear observation:

\[
x_{t+1}=\alpha x_t + \epsilon_t,
\qquad
 y_t = [x_t, x_t^2] + \eta_t.
\]

or a simple rotation / damped oscillator:

\[
x_{t+1}=R_\theta x_t + \epsilon_t,
\qquad
 y_t = h(x_t)+\eta_t.
\]

The goal is not to claim performance. The goal is to verify that:

- the core solver runs;
- the causal filtering path differs from but is compatible with the training path;
- prior-coordinate prediction uses only past information;
- readout recovers a sensible one-step feature prediction.

Detailed benchmark design should be handled after the implementation spec is approved.

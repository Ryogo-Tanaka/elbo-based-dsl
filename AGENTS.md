# AGENTS.md

## Project purpose

This repository implements Version 0 of an embedded latent transfer-operator learning method for the DSL / ELTO project.

The implementation is based on the formulation memo and the implementation specifications under `docs/`.

## Source of truth

Read these files in order before making implementation decisions:

1. `docs/specs/dsl_elto_version0_implementation_spec.md`
2. `docs/specs/dsl_elto_codex_task_decomposition_v0.md`
3. `docs/formulation/_formulation_memo_current.tex` only for mathematical background

The implementation spec overrides the formulation memo for implementation details.

Historical notes under `docs/notes/` are background only. Do not treat them as implementation source of truth unless the user explicitly says so.

## Language

- 回答、説明、進捗報告は日本語で行う。
- Code, variable names, function names, and docstrings may be written in English.
- Mathematical terms may use the notation from the implementation spec.

## Working rules

- 作業前に `git status` を確認する。
- 未コミット変更がある場合は、編集前にユーザーへ確認する。
- 変更前に関連ファイルを読み、短い計画を提示する。
- 大きな変更は小さな差分に分ける。
- 既存の設計、命名、フォーマットに合わせる。
- 新しい依存パッケージを追加する前に理由を説明する。
- 変更後は可能な範囲でテスト、lint、型チェックを実行する。
- `git commit`, `git push`, `git reset --hard`, ブランチ削除はユーザーの明示許可なしに実行しない。
- 最後に、変更ファイル、実行した検証、残課題を要約する。

## Security and private files

- `.env`, secret, private key, credential を読まない・表示しない。
- `_prompt/` はユーザー個人用メモ置き場なので、ユーザーが明示しない限り読まない。
- `AGENTS.local.md` が存在する場合も、ユーザーが明示しない限り読まない。
- リポジトリ外のファイルは、ユーザーが明示した場合のみ読む。

## Version 0 scope

Implement only the Version 0 single-trajectory solver.

Core assumptions:

- single trajectory only
- fixed encoder / fixed observation dictionary
- fixed scalar metrics
- center + RMS feature standardization in implementation
- correction-based warm start
- normalized covariance ridge updates for operators
- Phase 1 training coordinates and Phase 2 filtering coordinates are distinct
- readout supports both `linear_ridge` and `mlp`

## Explicitly excluded from Version 0

Do not implement:

- DSE CCA state construction
- stochastic realization state construction
- SVD/PCA state initialization
- direct `A = G @ B` or `A = G @ Z_B` state construction
- inference encoder
- multiple-trajectory support
- residual-calibrated metrics
- KKR-style covariance recursion
- posterior reconstruction loss
- encoder/dictionary refinement
- hard whitening
- full non-isotropic metrics
- experiment benchmark runners beyond the requested toy sanity script

## Matrix and naming convention

Use code arrays with shape `[T, d]`.

Use explicit names:

- `A_train_minus`, `A_train_plus` for Phase 1 variational coordinates
- `A_filter_minus`, `A_filter_plus` for Phase 2 causal filtering coordinates

Do not use `A_train_*` for final causal prediction.

Main one-step causal prediction must use `A_filter_minus`.

Do not use `A_filter_plus` as the main prediction state. It may be logged only for diagnostics.

## Allowed directories

Codex may create or edit these implementation directories when requested:

- `src/dsl_elto/`
- `tests/`
- `configs/`
- `scripts/`

Do not create notebooks, benchmark experiment folders, or large generated data directories unless explicitly requested.

## Readout

Implement both:

- `linear_ridge`
- `mlp`

Core solver sanity tests should use `linear_ridge`.

MLP readout is trained only in Phase 3 and must not update the core solver.

## Testing rule

Every implementation task must add or update tests.

Before reporting completion:

1. list changed files,
2. list tests run,
3. report whether tests passed,
4. mention any deviations from the implementation spec,
5. propose any minimal `AGENTS.md` update if a durable convention, invariant, or test command was discovered.

## AGENTS.md update policy

Do not edit `AGENTS.md` unless explicitly asked.

At the end of each task, propose minimal updates if needed. The user will decide whether to apply them.

When explicitly asked to update `AGENTS.md`, edit only `AGENTS.md` and do not change implementation files in the same task.
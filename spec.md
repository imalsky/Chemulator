# Autoregressive Chemical Kinetics Emulator Specification

Version: 1.6
Date: 2026-03-10
Audience: Professional collaborators and maintainers

## 0. Execution Environment
1. Canonical local development environment is Conda environment `nn`.
2. All canonical local commands and local validations must run via `nn`.
3. `run.pbs` uses an HPC-specific environment and currently defaults to `pyt2_8_gh`.
4. The local `nn` environment and the HPC batch environment are intentionally different and are not required to match.
5. Running local development commands outside `nn` is non-canonical and may be considered an unsupported setup.

## 1. Product Goal
1. Build and maintain an accurate, fast autoregressive chemical kinetics emulator.
2. Prioritize inference quality and runtime speed.
3. Use `val_loss` as the primary model-selection metric.

## 2. Project Scope
1. Scope includes the entire project lifecycle.
2. Lifecycle includes preprocessing, training, validation, export, and inference tooling.
3. Data/asset directories are not source-edit targets: `data/`, `figures/`, `misc/`, `models/`.
4. Runtime artifacts may still be generated under artifact directories (for example `models/` and `figures/`).

## 3. Engineering Principles
1. Fail fast in canonical preprocessing, training, and export contracts.
2. Diagnostic scripts under `testing/` may be more permissive when they are not release gates.
3. No silent behavior changes.
4. Hard-fail on malformed config, schema mismatch, ambiguous keys, and unsupported modes in canonical runtime paths.
5. Keep logic explicit and simple.
6. Avoid magic numbers in logic bodies.
7. Place stable defaults as named module-level constants in each executable script.

## 4. Canonical Workflow

## 4.1 Preprocessing
1. Preprocessing is canonical via `processing/preprocessing.py`.
2. Fixed `dt` mode is removed.
3. Variable `dt` is mandatory.
4. Required preprocessing `dt` config uses `dt_min` and `dt_max` only.
5. `dt_sampling` is fixed to `loguniform`.
6. Log-time/log-value interpolation is mandatory.
7. `drop_below` rejection is mandatory.
8. Raw schema is considered stable and non-evolving.
9. Species datasets must be scalar-per-time and only use shapes `[T]` or `[T, 1]`.
10. Multi-column or higher-rank species datasets are unsupported and must hard-fail.

## 4.2 Training
1. Training entrypoint is `src/main.py`.
2. Training mode is the only supported runtime mode.
3. Post-train automatic test execution is removed.
4. Validation remains enabled during training.
5. Training is a two-run workflow:
6. Stage 1 run uses one-jump training with `rollout_steps = 1`.
7. Stage 2 run uses autoregressive multi-step training with `rollout_steps = N`, where `N > 1`.
8. Stage 2 `N` is user-configured per run and not fixed by the codebase.
9. This two-run staging is a canonical workflow guideline and is not required to be hard-enforced by runtime guards in code.
10. Resume/checkpoint options are retained.
11. Fresh runs still require an empty work directory unless explicitly resuming.
12. Curriculum support is retained.
13. Autoregressive controls (`skip_steps`, detach behavior, backward mode) are retained.
14. Optimizer support is restricted to `AdamW` only.
15. Scheduler support is restricted to:
16. `reduce_on_plateau` (canonical config value; compatibility value `ReduceLROnPlateau` is accepted)
17. `cosine_with_warmup`
18. `torch.compile` support is retained and optional.
19. Compile default remains enabled with default compile settings.
20. Relative `runtime.checkpoint` paths are resolved against the config-file directory.
21. `runtime.checkpoint` must exist and be a file; otherwise training must hard-fail.
22. Training framework imports must use `lightning.pytorch`.
23. Direct source imports from `pytorch_lightning` are unsupported and must not appear in the repository.
24. Trainer/logger integration targets the current `lightning.pytorch` API directly; legacy PyTorch Lightning compatibility shims are not part of the supported codebase.

## 4.3 Export and Inference
1. Export contract is physical-space one-step inference.
2. Export signature is:
3. `y_next_phys = model(y_phys, dt_seconds, g_phys)`
4. Tensor shapes are:
5. `y_phys`: `[B, S]`
6. `dt_seconds`: `[B]`
7. `g_phys`: `[B, G]`
8. `y_next_phys`: `[B, S]`
9. Dynamic batch `B` is mandatory.
10. Variable per-step `dt` support is mandatory.
11. Baked normalization in export is mandatory.
12. Export naming standard is mandatory:
13. `export_{device}_dynB_1step_phys.pt2`
14. Canonical requested targets are CPU and CUDA.
15. `testing/export.py` may skip unavailable accelerator targets and may still emit any locally available target instead of failing the whole diagnostic run.
16. MPS support is retained as best-effort if low maintenance overhead.
17. Export metadata source of truth is embedded `metadata.json` inside the `.pt2` artifact.
18. No sidecar metadata JSON is required.
19. For `torch.export` artifacts, inference loaders must not call `model.eval()` or `model.train()` on `ep.module()` because these mode-switch APIs are unsupported; run inference under `torch.inference_mode()` instead.

## 4.4 Testing Script Canonicalization
1. `testing/` is the only place where temporary duplicate workflows are tolerated.
2. Scripts under `testing/` are diagnostic utilities and may use explicit best-effort behavior for portability and local usability.
3. The `new_` scripts become canonical and are renamed to remove the prefix.
4. Canonical script names after consolidation are:
5. `testing/export.py`
6. `testing/predictions.py`
7. Legacy duplicate scripts are removed.
8. Misspelled legacy script `testing/prections.py` is removed.

## 5. Data and Schema Contracts
1. Raw input data schema is stable and assumed unchanged.
2. Dataset path/key ambiguity is a hard error.
3. Globals are fixed per trajectory and scalar-only.
4. Global variable set is fixed and mandatory:
5. `["P", "T"]` in this exact order.
6. `G = 2` is mandatory for production workflows.
7. Species/global ordering must match exactly between config and `normalization.json`.
8. Processed shard schema remains fixed:
9. `y_mat` float32
10. `globals` float32
11. `dt_norm_mat` float32
12. `normalization.json` is generated by preprocessing and consumed downstream.
13. Split strategy should stay simple; leakage minimization beyond current behavior is not a requirement.
14. Global normalization methods are restricted to `standard`, `min-max`, `log-standard`, and `log-min-max`.
15. Global normalization method `identity` is unsupported and must hard-fail.

## 6. Metrics and Model Selection
1. Keep current loss and metric definitions.
2. Primary checkpoint/selection metric is `val_loss`.
3. No additional hard acceptance thresholds by species/horizon are required.
4. No physical-constraint enforcement metrics are required.
5. Benchmark and VULCAN scripts are diagnostic, not release gates.
6. For fully synthetic/randomly generated data workflows, stochastic validation window sampling is acceptable.
7. In that stochastic-validation setting, compare `val_loss` only within the same code/config/seed setup.

## 7. Logging and Outputs
1. Maintain high-quality operational logging.
2. Persist resolved config for each run.
3. Persist training metrics (`metrics.csv`) for each run.
4. Persist checkpoints according to configured checkpoint policy.

## 8. Non-Goals
1. Enforcing physical constraints in-model or post-hoc.
2. Supporting fixed-`dt` preprocessing mode.
3. Supporting multiple runtime modes beyond training.
4. Requiring unit tests or integration tests for this project.
5. Treating VULCAN comparison or benchmark scripts as mandatory acceptance gates.

## 9. Developer Tooling Notes
### 9.1 Canonical Environment Installs
1. Local developer tooling must be installed in Conda environment `nn`.
2. HPC batch scripts may activate a separate cluster-managed environment; `run.pbs` currently defaults to `pyt2_8_gh`.
3. The local `nn` environment and the HPC batch environment are expected to differ.
4. Canonical local install commands:
5. `conda activate nn`
6. `pip install lightning`
7. `pip install pyflakes`
8. `pip install vulture`
9. `pip install ruff`
10. The codebase targets the `lightning.pytorch` namespace; direct imports from `pytorch_lightning` are non-canonical even if that package is present transitively in the environment.

### 9.2 Dead Code Search Workflow
1. Run checks from project root under environment `nn`.
2. Canonical dead/unused checks:
3. `conda run -n nn pyflakes $(rg --files -g '*.py')`
4. `conda run -n nn ruff check $(rg --files -g '*.py')`
5. `conda run -n nn vulture $(rg --files -g '*.py')`
6. Optional high-confidence-only Vulture pass:
7. `conda run -n nn vulture --min-confidence 100 $(rg --files -g '*.py')`

### 9.3 Editing Rules For Dead Code Fixes
1. Remove truly unused locals/imports/functions rather than suppressing diagnostics.
2. If protocol signatures require unused parameters (for example context manager or framework hook signatures), rename them with leading underscores.
3. Treat framework callback methods and model `forward` methods as potential false positives in static dead-code tools; verify before deleting.
4. Keep parser argument removal policy: no `argparse`; use config values or explicit module-level constants.

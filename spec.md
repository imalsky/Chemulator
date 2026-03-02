# Flowmap Emulator Spec (Current Canonical)

## 0. High-Level Project Description

This project trains a one-step flow-map emulator from trajectory data.

- Preprocessing converts raw HDF5 trajectories into train/validation/test NPZ shards and writes normalization artifacts.
- Training learns a one-step map using species state, globals, and normalized `dt`.
- Inference (current mode) is one-step only and must be callable directly with physical inputs (species/globals/`dt` in seconds) via an exported physical-I/O artifact that has normalization + metadata baked in.

Out of scope for this spec:

- Multi-step rollout API.
- Serving/production deployment concerns.
- Physics-constraint enforcement.

## 0.1 Project Goal

The goal of this project is to produce a fully correct, reproducible one-step emulator workflow for scientific use.

This emulator is intended to replace the expensive local chemical-kinetics update normally computed by VULCAN in the target atmospheric workflow. Concretely, for each local state, the model should act as a surrogate for a single VULCAN chemistry advance over a specified `\Delta t`, while preserving the same input/output variable conventions defined by this spec.

The scientific purpose is to make stiff-chemistry calculations fast enough to integrate with broader exoplanet-atmosphere modeling loops without relying on classical per-step ODE integration at runtime. The replacement target is one-step state-to-state kinetics prediction (not long autoregressive rollout).

- Correct means no known bugs, no incorrect behavior versus this spec, and no data leakage.
- The workflow must remain simple to operate: preprocess data, train a one-step model, and run one-step inference.
- The primary success signals are stable training plus clear reporting of trainer loss outputs (`loss`, `phys`, `z`, `mult_err_proxy`).

## 0.2 Paper-Derived Context (Reference)

This section captures important context from the paper draft "Accelerating Chemical Kinetics for Exoplanet Atmospheres using Neural Networks."

- Scientific target: emulate local stiff chemical kinetics in exoplanet atmospheres as a surrogate for VULCAN one-step evolution.
- Task form: flow-map/state-to-state prediction `y_{t+\Delta t} = \Phi(y_t, g, \Delta t)`, where `g` includes pressure and temperature.
- Operating assumption: local 0D box update per grid cell, not a full spatial solver inside this model.
- Core motivations in the paper: time-step flexibility, high accuracy, low per-inference cost, and broad parameter-space coverage.
- Paper baseline data regime: VULCAN-generated trajectories from a 52-species thermochemical network (about 1200 reactions), with a 12-species tracked subset for the presented model.
- Paper baseline ranges: approximately `T in [300, 3000] K`, `P in [1e-6, 1e4] bar`, and `\Delta t in [1e-3, 1e8] s`.
- Paper baseline time grid: 100 points (initial `t=0` plus 99 log-spaced points).
- Paper architecture context: encoder + latent dynamics + decoder flow-map, with residual variants explored and strong results reported for residual flow-map settings.
- Paper context on rollout: model is primarily one-shot; long autoregressive rollout was shown to degrade and is not the primary target.

Interpretation rule:

- This section is scientific/background context.
- If a statement here conflicts with explicit requirements elsewhere in this spec, the explicit requirements elsewhere take precedence.

## 1. Scope

This document is the current source-of-truth spec for future agents and AI working in this codebase snapshot.

- Current workspace layout is canonical for now (`src/` for code, `config/` for config).
- Code was dumped for readability; future layered refactors are out of scope for this spec.
- Goal: fully correct emulator behavior.

## 2. Definition Of "Fully Correct"

Fully correct means:

- No known bugs.
- No incorrect behavior versus this spec.
- No data leakage.

Out of scope:

- Physics constraints/enforcement.

Engineering style requirements:

- Minimal defensive coding.
- Clean, concise, readable implementation.
- Fail fast on invalid states.

## 3. Canonical Paths And Config

- Canonical default config file: `/Users/imalsky/Desktop/Chemulator-Editing/config/config_job0.jsonc`.
- No CLI parser arguments are required.
- Default behavior should load the default config path directly.
- Multi-job usage (`job0`, `job1`, etc.) is supported by separate config files, but not via argparse.
- `paths.raw_data_files` must be explicitly configured and non-empty; there is no automatic `data/raw` scan fallback.
- `data.species_variables` must be explicitly configured and non-empty; there is no species auto-detection fallback.

## 4. Runtime Modes

- Inference mode is one-step only.
- No dedicated inference/serving API or CLI is required right now.
- Preprocessing may use MPI and is required to support MPI workflows.
- MPI initialization must be lazy: importing modules (including `main.py`) must not initialize MPI or require MPI runtime availability.
- Non-MPI workflows (training/inference and serial preprocessing) must remain usable even when MPI runtime initialization is unavailable.
- `preprocessing.use_mpi` may be used to control mode (`off` = force serial, `on` = require MPI, `auto` = enable only under detected MPI launcher context).
- Current MPI preprocessing scan aggregates scan metadata on rank 0; this is part of the current contract and may be a root-memory bottleneck at large scale.
- Training runtime should use a single trainer process/device (typically one GPU); CPU DataLoader workers are allowed.
- Multi-GPU/distributed training pathways are not part of the target behavior.

## 5. Inference Contract

### Inputs (required)

- All species variables in physical representation.
- All global variables in physical representation.
- `dt` in physical seconds.
- Callers must not be required to provide pre-normalized values.

### Outputs

- Return species only.
- Returned species must be in physical representation.
- Strict species ordering: exactly `cfg.data.species_variables`.

### `dt` behavior

- If physical `dt` maps outside the supported trained range, fail fast with an error (no clamping/silent extrapolation).

### Inference Packaging

- The exported/deployed one-step inference artifact must embed all required normalization metadata internally.
- External inference should require only physical tensors (`y_phys`, `dt_sec`, `g_phys`) plus documented variable ordering.

## 6. Data And Schema Contract

- Raw HDF5 schema is fixed/versioned and must be respected.
- Time grids must be identical across valid trajectories; mismatch is a hard error.
- `min_value_threshold` filtering is optional.
- Default `min_value_threshold` is `1e-30`.
- `skip_first_timestep` default behavior is `false`.
- Species sets may vary between runs.
- Configured species ordering is authoritative and must match processed artifacts exactly.
- Empty `train`/`validation`/`test` splits are hard preprocessing errors; no `allow_empty_splits` override.

## 7. Normalization Contract

- Per-key normalization methods are allowed to vary.
- Do not enforce species normalization as always `log-standard`.
- Normalization remains an internal model detail for training/runtime internals; exported physical-I/O inference must perform physical <-> normalized conversion internally.
- External inference callers must not need direct access to normalization manifests.
- Training loss uses:
  - `lambda_phys * weighted_MAE(log10)`.
  - `lambda_z * MSE(z)`.
- `training.adaptive_stiff_loss.use_weighting` defaults to `false` (uniform species weighting).
- If weighting is enabled, computed species weights must already be within `[w_min, w_max]`; out-of-range is a hard error (no weight clamping).
- MAE(log10) must be computed correctly for the active species normalization method, not by assuming `log-standard`.
- Log-domain computations must hard error on non-positive values (`<= 0`) rather than silently clamping for loss/metric math.
- If `data.time_variable` uses `log-standard` or `log-min-max`, retained time values must be strictly positive; non-positive time values are a hard preprocessing error.
- Runtime normalization must hard error if `std`/`log_std` violates `min_std`; do not silently clamp at train/inference time.
- This runtime `min_std` hard-error requirement applies to all active normalization paths, including model log-physical/softmax heads.
- Expected practical note: preprocessing should normally prevent this case.
- Preprocessing statistics finalization may clamp computed `std`/`log_std` up to `min_std`, but any such clamp must be logged as a warning for auditability.

## 8. Data Leakage Rules

Data leakage prevention requires:

- Train/validation/test split separation.
- Normalization statistics computed from train split only.

This is considered sufficient for leakage control in this project.

## 9. Split Behavior

- Deterministic hash-based split assignment is acceptable.
- Split/use hash keys must include fully resolved raw file path plus group identifier (basename-only hashing is not acceptable).
- Long-term split identity stability/versioning is not a strict requirement.

## 10. Model/Training Defaults

- Default model family: autoencoder flow-map.
- Default prediction head: `predict_delta`.
- Optimizer support target: AdamW only.
- Gradient accumulation is not supported.
- Remove/ignore LAMB support and references.
- Resume behavior is explicit-only via `training.resume`; implicit auto-resume from `work_dir/last.ckpt` is not part of target behavior.
- Sampling contract for training pairs:
  - Pair sampling is stochastic per access using worker-local RNG streams.
  - Worker-local RNG streams are seeded once from DataLoader worker seeds.
  - Determinism is guaranteed at the run level for fixed seed/config; there is no strict deterministic mapping from `(epoch, idx)` to a specific sampled pair.
  - `set_epoch()` does not reseed pair sampling.

## 11. Error Handling Policy

- Prefer fail-fast explicit errors.
- Keep checks minimal and essential.
- Avoid verbose defensive scaffolding beyond essential contract checks.
- Explicit user-intent settings should fail hard when they cannot be honored.
- Best-effort hardware/backend tuning hints may degrade to logged warnings when backend support is unavailable.
- Model runtime checks should be limited to contract-critical checks:
  - `dt` range enforcement (physical interface externally, normalized `[0, 1]` internally).
  - Log-domain non-positive hard errors for required log computations.
  - `min_std` hard errors for active runtime normalization paths.
- Shape/config mismatches outside those contract-critical checks may fail via underlying framework/runtime errors without dedicated model-side guard code.

## 12. Required Artifacts

All training/preprocessing runs are expected to produce full artifacts:

- Preprocessing outputs and metadata (including shards, shard index, summary/report, normalization manifest).
- Training outputs (checkpoints, metrics, logs, hydrated config snapshot at `work_dir/config.json`).
- Physical-I/O inference artifacts:
  - Exported model artifact with baked-in normalization/metadata (for example `physical_model_k1_cpu.pt2`).
  - Companion metadata file documenting variable order and reference values (for example `physical_model_metadata.json`).
- Testing/evaluation scripts must use the physical-I/O artifact interface (artifact + metadata) rather than requiring users to pass normalized tensors manually.
- Reusing existing processed artifacts must be an explicit user choice (`preprocessing.reuse_existing_data=true`); default behavior is to fail fast instead of implicitly reusing.

## 13. Metrics And Evaluation

Core reported metrics:

- `loss` (total objective).
- `phys` (lambda-scaled weighted MAE in log10 space).
- `z` (lambda-scaled z-space MSE term).
- `mult_err_proxy` (derived from unweighted mean absolute log10 error).

Current policy:

- These metrics are required outputs.
- Hard numeric pass/fail thresholds are not specified yet in this spec.

## 14. Environment And Hardware Requirements

Required software/runtime:

- Python 3.12+.
- Required Python packages: `torch`, `numpy`, `h5py`.
- Optional package: `mpi4py` (needed only for MPI preprocessing workflows).
- If `mpi4py` is installed but MPI runtime setup is invalid/unavailable, only MPI preprocessing mode may fail; import, training, inference, and serial preprocessing must still work.

Primary training hardware target:

- Full training is typically run on NVIDIA accelerators (A100 or GH200 class GPUs) with CUDA.

Local compatibility target:

- The same codebase must also run locally on a developer machine in single-process mode.
- Device selection priority is: CUDA, then MPS, then CPU.
- On CPU/MPS, precision must resolve to FP32 execution (`precision.amp` in FP32/off mode and `precision.dataset_dtype` set to `float32`, or `auto` that resolves to `float32`), otherwise fail fast.
- Local runs are expected to be slower; they are valid for correctness checks, preprocessing, smoke tests, and smaller training runs.

Current local reference environment for this workspace:

- macOS (Darwin) on arm64.
- Conda environment `nn`.
- Python 3.12.8.
- PyTorch 2.8.0.

Behavioral requirement across hardware:

- Changing hardware (A100/GH200 vs local machine) must not change the data/training/inference contracts defined in this spec; only throughput/performance may differ.

## 15. Static Analysis And Dead-Code Hygiene

- Development/runtime environment may include `ruff`, `pyflakes`, and `vulture` for static checks.
- Dead-code audits should run on `src/` and be triaged before claiming a correctness pass.
- `ruff`/`pyflakes` unused-code findings are expected to be fixed unless there is a concrete, documented reason to keep them.
- `vulture` findings must be reviewed manually: high-confidence findings should be fixed or explicitly justified; lower-confidence hits can include framework-dispatched methods (for example, PyTorch `forward` methods).

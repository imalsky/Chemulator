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

- Canonical default config file: `config/config_stage1.jsonc` (production
  stage-1 pretrain). The only other config is `config/config_stage2.jsonc`
  (autoregressive fine-tune; see §10.2).
- No CLI parser arguments are required.
- Default behavior should load that repository-local default config path directly.
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

- Default model family: latent-linear flow map (`model.architecture =
  "latent_linear"`); see §10.1. The autoencoder and MLP flow maps remain
  selectable for A/B comparison and back-compatibility.
- Default prediction head: `predict_delta`.
- Optimizer support target: AdamW only.
- Gradient accumulation is not supported.
- Resume behavior is explicit-only via `training.resume`; implicit auto-resume from `work_dir/last.ckpt` is not part of target behavior.
- Sampling contract for training pairs:
  - Pair sampling is stochastic per access using worker-local RNG streams.
  - Worker-local RNG streams are seeded once from DataLoader worker seeds.
  - Determinism is guaranteed at the run level for fixed seed/config; there is no strict deterministic mapping from `(epoch, idx)` to a specific sampled pair.
  - `set_epoch()` does not reseed pair sampling.

### 10.1 Model Architectures

`model.architecture` selects the flow-map family. All three obey the same
forward contract — `forward(y_i [B,S], dt_norm [B,K]|[B,K,1], g [B,G]) ->
y_pred_z [B,K,S]` in z-space, validating `dt_norm ∈ [0,1]` — and share the
same prediction heads (`predict_delta`, `predict_delta_log_phys`,
`softmax_head`), implemented once so cross-architecture comparisons are fair.

- `"autoencoder"` — Encoder → LatentDynamics → Decoder (`FlowMapAutoencoder`).
- `"mlp"` — single (residual) MLP over `[y_i, dt_norm, g]` (`FlowMapMLP`).
- `"latent_linear"` — the default; see below (`LatentLinearFlowMap`).

Selection and back-compatibility (fail-fast):

- If `model.architecture` is present it is authoritative, and `model.mlp_only`
  MUST be absent (specifying both is a hard error — no silent precedence).
- If `model.architecture` is absent, dispatch falls back to the legacy
  `model.mlp_only` switch unchanged (`true` → `"mlp"`, `false` →
  `"autoencoder"`). Legacy configs therefore keep working untouched.
- `model.architecture = "latent_linear"` forbids the `dynamics_hidden` /
  `dynamics_residual` keys (they are no-ops for this family — hard error
  rather than silently ignore them).

#### Latent-linear flow map

The latent-linear flow map is the campaign-recommended architecture (June 2026
Robertson sandbox study; condensed record in the umbrella-level
`EXPERIMENTS.md`, port design in `IMPLEMENTATION_PLAN.md`). A shared encoder
trunk reads the anchor state and emits an initial latent, a target
("equilibrium") latent, and per-mode log10 decay rates; the latent relaxes in
closed form and a decoder emits the species update:

```
feats        = trunk([y_i, g])                       # state-dependent features
h0, h_eq     = head_h0(feats), head_heq(feats)       # [B,L] each
log10_k      = head_rate(feats)                       # [B,L] per-mode log10 rates
log10_dt     = dt_log_min + dt_norm * (dt_log_max - dt_log_min)
decay        = exp(-10 ** clamp(log10_k + log10_dt, max=rate_clamp))
h(dt)        = h_eq + decay * (h0 - h_eq)             # exact diagonal linear ODE
out          = Decoder(h(dt))                          # then the shared head
```

Required behavior / fail-fast rules:

- Rates AND equilibrium MUST be state-dependent (both from the encoder trunk);
  a fixed/global operator was the single worst formulation in the campaign, so
  the trunk (`model.encoder_hidden`) must be non-empty.
- `dt_log_min` / `dt_log_max` MUST be read from the dataset's normalization
  manifest (`normalization.json` `dt.log_min`/`dt.log_max`) at build time and
  stored as model buffers — never hardcoded (they are dataset properties). A
  missing manifest or dt spec is a hard error.
- `model.decoder_mode = "mlp"` requires non-empty `model.decoder_hidden`;
  `"linear"` (the interpretable "lindec" deployment variant) requires
  `model.decoder_hidden = []`. Mismatch is a hard error.
- The relaxation path runs in fp32 regardless of AMP policy. `log10(k·dt)` is
  clamped from above at `model.rate_clamp` (default `3.0`) to keep `10**x`
  finite; there is no lower clamp (decay→1 underflow is exact).
- The rate-head bias is drawn from `U(model.rate_init)` (default `[-5, 5]`)
  using a generator seeded with `model.init_seed`, so builds are reproducible.

Required `model` keys for `architecture = "latent_linear"`: `latent_dim`,
`encoder_hidden`, `decoder_mode`, `decoder_hidden`, `rate_clamp`, `rate_init`,
`init_seed`, plus the common keys `activation`, `dropout`, `predict_delta`,
`predict_delta_log_phys`, `softmax_head`.

Recommended starting points (production scale): `latent_dim` ≈ 128 (latent
expansion `L >> S` is the load-bearing knob, not hidden width),
`encoder_hidden`/`decoder_hidden` `[1024, 1024]`, `activation` `"silu"`,
`predict_delta` true, and the pure log10-MAE loss
(`training.adaptive_stiff_loss.lambda_z = 0.0`; the MSE-z term was found
inert). The `"linear"` decoder variant (`decoder_mode = "linear"`, lindec) is a
deployment-tilted option (9–14% fixed-dt edge, p≈0.03) gated on a
production-scale fixed-dt spot-check — not the default.

Note (training scope): training defaults to strictly one-step pairs (§0). The
latent-linear architecture is adopted because it is the best one-step flow map
AND because it remains accurate when stepped autoregressively at fixed `dt` in a
host model. An **optional** autoregressive training regime is now available
(§10.2); it is off by default, so the one-step contract above is unchanged
unless explicitly enabled.

### 10.2 Autoregressive (rollout) training regime — stage 2

Production training is **two-stage** (the campaign-validated recipe; umbrella
`EXPERIMENTS.md`). Stage 1 (`config_stage1.jsonc`) is the one-step-from-t0 flow
map above. Stage 2 (`config_stage2.jsonc`) warm-starts from stage-1 weights
(`training.init_from`, weights only + fresh optimizer at lr/10) and fine-tunes
autoregressively. Off by default — when the `training.rollout` block is absent or
`enabled=false`, training is exactly the one-step contract.

When enabled, training switches to a **consecutive-step stepper**: a sample is a
random anchor plus up to `horizon` consecutive grid steps; the trainer unrolls
the current curriculum horizon feeding the model's own prediction forward.

Config (`training.rollout`, all required when `enabled=true`; fail-fast in
`main.validate_rollout_config` + the trainer):

- `enabled` (bool).
- `horizon` (int ≥ 1) — max consecutive steps. **K=10** (E3: inverted-U; K=1
  underfits, K=20 unstable).
- `curriculum_start` (int in [1, horizon]) + `curriculum_ramp_epochs` (int ≥ 0)
  — linearly ramp the horizon from `curriculum_start` to `horizon` over the ramp
  (validated **1→10**; constant if ramp=0).
- `detach_intermediate` (bool) — detach between steps (pushforward). Validated
  recipe = detached; E4 found detach ≈ full BPTT, detach is cheaper at scale.
- `discount_gamma` (float in (0,1]) — per-step discount `γ^i` on scored steps.
  Under detach, **γ<1 is required** (E3: uniform γ=1.0 was 4–14× worse);
  validated **0.9**. (Irrelevant under BPTT.)
- `pushforward_skip` (int ≥ 0) — leading no-grad warm steps (Brandstetter);
  validated **2**.
- `input_noise_std` (float ≥ 0) — **kept at 0**. GNS-style input noise was
  **rejected** (E6: 4/4 worse, 3/4 catastrophic 12–14 dex); the knob exists only
  for the record (if ever revisited, σ ≤ 1e-3).

`training.ema` ({`enabled`, `decay`}) maintains an EMA of the weights; when on,
`best.ckpt` holds the EMA weights (selected + exported), `last.ckpt` the raw
weights (resume). `training.init_from` (path) warm-starts model weights only
(fresh optimizer/schedule) and is mutually exclusive with `training.resume`.

Required behavior / fail-fast rules:

- `enabled=true` requires `dataset.use_first_anchor=false`. `min_steps`/`max_steps`
  become inert; `pairs_per_traj` = random anchors per trajectory per epoch. The
  unused one-step `dt_table` is not built in this mode.
- **Selection metric is fixed-dt rollout at the deployment dt** (the catastrophe
  detector), with the geometric-grid rollout as the fine discriminator — never
  one-step/val loss (campaign ρ(one-shot, rollout) = 0.41).
- Inference/export are unaffected — the trained model is still a one-step
  physical-I/O artifact (§5); only the training objective changes.

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

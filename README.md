# Chemulator

A one-step flow-map emulator for chemical kinetics in exoplanet atmospheres. The
network learns `y(t + dt) = Phi(y(t), g, dt)` for 12 species conditioned on the
globals `g = (P, T)` and on the time step `dt`, so it can stand in for a stiff
ODE integrator (VULCAN) inside a larger atmospheric model.

This is the code behind Malsky et al., "Accelerating Chemical Kinetics for
Exoplanet Atmospheres using Neural Networks".

## Install

```bash
python -m pip install -r requirements.txt
```

Python 3.12 and PyTorch 2.8 are the tested versions. Training needs one CUDA
device; evaluation and export run on CPU or MPS.

## Run

The pipeline is config-driven. `FLOWMAP_CONFIG` selects the config; `run.sh`
loads it, preprocesses if needed, trains, then exports the physical-I/O
artifact.

```bash
export FLOWMAP_CONFIG=config/config_stage1.jsonc
./run.sh
```

`python -m src.main` does the same without the launcher checks. `run.pbs` is the
HPC batch version.

Configs in `config/`:

- `config_paper.jsonc` reproduces the published model: an 11,577,356-parameter
  residual encoder-operator-decoder. Its header comment records where it differs
  from the archived run and why.
- `config_stage1.jsonc` and `config_stage2.jsonc` are the current two-stage
  production recipe (latent-linear pretrain, then autoregressive fine-tune).

## Tests

```bash
python -m pytest unit_tests/ -q
ruff check src/ testing/ unit_tests/
```

## Layout

- `src/` preprocessing, dataset, model, trainer, entry point
- `testing/` export to a physical-I/O artifact, accuracy, benchmarks, comparison
  against VULCAN
- `unit_tests/` config contracts, precision policy, runtime, model contracts
- `spec.md` the canonical behavioral spec
- `config/` run configurations, commented

## Data and model artifacts

`data/`, `models/`, `figures/` and `reports/` hold artifacts and are not tracked
here, with one exception noted below. A trained run directory contains
`best.ckpt`, `last.ckpt`, `metrics.jsonl`, `train.log`, the hydrated
`config.json`, and the exported `physical_model_k1_cpu.pt2` with its
`physical_model_metadata.json`.

The exception is the published run. `models/final_model/` carries four small text
records of it that are tracked here — the hydrated `config.json`, `train.log`,
`metrics.jsonl` and `physical_model_metadata.json` — because
`config/config_paper.jsonc` was reconstructed from the first two and this lets
that reconstruction be checked without downloading anything. The checkpoints and
the exported binary are large and ship in the Zenodo record instead.

To rebuild the processed dataset from raw HDF5 files, list them in
`paths.raw_data_files` and set `preprocessing.reuse_existing_data=false`. To
train on already-processed shards, point `paths.processed_data_dir` at them and
set `preprocessing.reuse_existing_data=true`; the shipped
`normalization.json`, `preprocessing_summary.json` and `shard_index.json` carry
the split assignment, the normalization statistics and the list of raw files the
data came from, so no re-preprocessing is required.

## Inference contract

The exported model takes physical species values, physical globals and `dt` in
seconds, and returns physical species values in exactly the configured species
order. Normalization is baked in. A `dt` outside the trained range is a hard
error, not an extrapolation.

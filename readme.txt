CONFIG MODES README (trainer config)
=================================

This project has two separate “mode” axes:
  (A) Training execution regime: training.train_mode
  (B) Evaluation execution regime: training.eval_mode

The goal is to control whether the model is trained and/or evaluated as parallel one-step transitions (fast) or as a true sequential rollout (realistic).


1) What train_mode and eval_mode mean
------------------------------------

train_mode: controls how the training step computes predictions and loss.

  • "one_jump"
    - Uses ground-truth states as inputs for every step in the window.
    - Predicts many one-step transitions in parallel (vectorized).
    - Highest throughput (no sequential dependency).
    - Does NOT train the model to recover from its own errors across time (because it never feeds its own outputs as inputs).

  • "autoregressive"
    - True sequential rollout.
    - Each step’s prediction becomes the next input state (open-loop dynamics inside the training step).
    - Lower throughput (sequential dependency).
    - Trains the model to stay stable and recover from error accumulation.


eval_mode: controls how validation/test is executed.

  • You almost always want eval_mode="autoregressive"
    - Validation metrics reflect true inference behavior (open-loop rollout).
    - Checkpoint selection is meaningful for deployment/inference.

  • If eval_mode="one_jump"
    - Validation metrics reflect teacher-forced one-step accuracy only.
    - This can look “better” numerically but is not a reliable proxy for rollout performance.


2) one_jump_k_roll: what it does
--------------------------------

one_jump_k_roll only affects training when train_mode="one_jump".

It caps how many one-step transitions (K steps) per window are included in the loss:

  - If one_jump_k_roll = 50:
      only the first 50 transitions in each sampled window are used for the one-step loss.
  - If one_jump_k_roll <= 0:
      the trainer uses the current scheduled rollout length (typically training.rollout_steps or a curriculum-derived value).

Why it exists:
  - It is a compute/throughput knob: fewer steps per window = faster + less memory.
  - It also controls which part of the window the model learns on (usually the earliest steps).


3) Recommended mode recipes
---------------------------

Recipe A: Fast baseline training (common)
-----------------------------------------
Use when: you want efficient training and will judge by autoregressive validation.

Set:
  training.train_mode = "one_jump"
  training.eval_mode  = "autoregressive"
  training.rollout_steps = 50
  training.one_jump_k_roll = 50
  training.teacher_forcing.start/end = 1.0 (kept explicit; one_jump is TF-by-construction)
  training.curriculum.enabled = false

Expected behavior:
  - train_loss measures one-step accuracy (teacher-forced).
  - val_loss measures open-loop rollout error (compounding).
  - Train/val losses will not be on the same scale. That is normal.


Recipe B: Autoregressive training (matching inference)
------------------------------------------------------
Use when: long-horizon stability is the primary objective and compute allows.

Set:
  training.train_mode = "autoregressive"
  training.eval_mode  = "autoregressive"
  training.rollout_steps = (target horizon, e.g., 50)

Teacher forcing options:
  - Stable start then taper:
      teacher_forcing.start = 1.0
      teacher_forcing.end   = 0.0
      teacher_forcing.decay_epochs = 50–200
  - Harder training (more open-loop early):
      teacher_forcing.start = 0.5
      teacher_forcing.end   = 0.0

Optional stabilization:
  - training.burn_in_steps = 1–5
  - training.burn_in_loss_weight = 0.0–0.2
  - training.burn_in_noise_std = small (only if trainer supports it for rollout)

Expected behavior:
  - train_loss and val_loss become more comparable (both rollout-based).
  - Slower wall-clock per epoch than one_jump.


Recipe C: Curriculum rollout length (if autoregressive is unstable)
-------------------------------------------------------------------
Use when: full-horizon autoregressive training diverges or is unstable.

Set:
  training.curriculum.enabled = true
  training.curriculum.start_steps = 2–8
  training.curriculum.ramp_epochs = 100–250
  training.rollout_steps = target horizon (e.g., 50)

Interpretation:
  - Training rollout length increases gradually over epochs.
  - Makes early training easier and more stable.


Recipe D: Long-rollout fine-tuning (late-stage specialization)
--------------------------------------------------------------
Use when: you have a good baseline (often from one_jump) and want to specialize to very long horizons.

Set:
  training.long_rollout.enabled = true
  training.long_rollout.long_rollout_steps = 150–500 (must be <= available transitions in data)
  training.long_rollout.long_ft_epochs = 10–50
  training.long_rollout.apply_to_validation = true/false depending on what you want to select checkpoints for
  training.long_rollout.apply_to_test = true (common)

Notes:
  - If apply_to_validation=true, checkpoint “best” should align with long-horizon metrics.
  - If apply_to_validation=false, validation remains comparable across most training and you do separate long-horizon eval.


4) Loss mode guidance (two-term v1-style loss)
----------------------------------------------

With the updated trainer, the loss is controlled by:

  training.loss.lambda_log10_mae  (primary accuracy term)
  training.loss.lambda_z_mse      (stability/regularization term)

Common patterns:
  - Accuracy-first:
      lambda_log10_mae = 1.0
      lambda_z_mse     = 0.01–0.2
  - Stability-heavy early:
      lambda_z_mse     = 0.1–0.5 early, lower later (manual schedule requires changing config between runs unless trainer supports scheduling)


5) Experiment hygiene: resume, work_dir, and preprocessing
----------------------------------------------------------

- paths.work_dir:
    Change this per experiment variant to avoid mixing checkpoints/metrics.
    Example:
      models/v1_onejump
      models/v1_autoreg_ft
      models/v1_longrollout

- training.resume=true:
    The trainer may auto-resume from a checkpoint in work_dir.
    Make sure work_dir contains the correct run artifacts.

- preprocessing/normalization changes:
    If you change normalization settings or preprocessing, regenerate processed data
    (or point to a new paths.processed_data_dir) to avoid mismatched normalization stats.


6) Minimal “mode switch” playbooks
----------------------------------

Playbook 1: One-jump baseline -> autoregressive fine-tune
  Stage 1 (fast baseline):
    train_mode="one_jump"
    eval_mode="autoregressive"
    one_jump_k_roll=50
    rollout_steps=50
  Stage 2 (fine-tune):
    train_mode="autoregressive"
    eval_mode="autoregressive"
    teacher_forcing: start=1.0 -> end=0.0 over 50–150 epochs
    (optional) curriculum.enabled=true if instability appears

Playbook 2: Autoregressive + curriculum -> long-rollout specialization
  Stage 1:
    train_mode="autoregressive"
    eval_mode="autoregressive"
    curriculum.enabled=true, start_steps=2, ramp_epochs=200, rollout_steps=50
  Stage 2:
    long_rollout.enabled=true
    long_rollout_steps=300
    long_ft_epochs=40
    apply_to_validation=true if you want to checkpoint on long-horizon behavior


END

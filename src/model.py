#!/usr/bin/env python3
"""
Neural Dynamics Autoencoder (Softmax head, cheap global conditioning)
====================================================================

- Single, strict output head: **softmax** over species (exact simplex).
- Optional **logit-delta** composition (predict_logit_delta=True) adds decoder
  output to the *natural-log probabilities* of the base state before softmax.
  This preserves positivity and ∑=1 while allowing residual updates.
- **Cheap decoder conditioning on globals (g)** via FiLM: a single Linear(g)->(γ,β)
  modulates latent z as z' = (1 + α·tanh(γ)) ⊙ z + β before the decoder.
  Enable via config: model.decoder_condition_on_g: true|false
- Interfaces preserved for trainer:
    forward(y_i, dt_norm, g)          -> [B, K, S_out]  (independent offsets)
    step(y, dt_step_norm, g)          -> [B, S_out]     (single-step, teacher forcing)
    rollout_vectorized(y0, g, dt_seq) -> [B, K, S_out]  (sequential K-step rollout)

Inputs/targets are in **normalized space**. Species targets use "log-standard":
    z = (log10(x) - log_mean) / log_std

This module outputs in the same normalized space. Internally the softmax head:
    logits            -> log_p = log_softmax(logits)
    log10_p           = log_p / ln(10)
    normalized_output = (log10_p - log_mean) / log_std
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Small helpers
# -------------------------
def get_activation(name: Union[str, nn.Module]) -> nn.Module:
    if isinstance(name, nn.Module):
        return name
    name = str(name).lower()
    if name in ("relu",):
        return nn.ReLU(inplace=True)
    if name in ("gelu",):
        return nn.GELU()
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    if name in ("tanh",):
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int], out_dim: int,
                 activation: Union[str, nn.Module] = "silu", dropout: float = 0.0):
        super().__init__()
        acts = []
        last = in_dim
        for h in hidden:
            acts += [nn.Linear(last, h), get_activation(activation)]
            if dropout and dropout > 0:
                acts.append(nn.Dropout(p=float(dropout)))
            last = h
        acts.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*acts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------
# Encoder / Dynamics / Decoder
# -------------------------
class Encoder(nn.Module):
    """
    Deterministic encoder: [y, g] -> z
    """
    def __init__(self, state_dim: int, global_dim: int, hidden_dims: Sequence[int],
                 latent_dim: int, activation: Union[str, nn.Module], dropout: float = 0.0):
        super().__init__()
        self.net = MLP(state_dim + global_dim, list(hidden_dims), latent_dim, activation, dropout)

    def forward(self, y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([y, g], dim=-1))


class LatentDynamics(nn.Module):
    """
    Residual latent flow-map: z' = z + f([z, dt_norm, g])  (if residual=True)
    If residual=False, z' = f([z, dt_norm, g]).
    dt_norm is a normalized scalar in [0, 1] corresponding to log10(dt_phys) normalized.
    """
    def __init__(self, latent_dim: int, global_dim: int, hidden_dims: Sequence[int],
                 activation: Union[str, nn.Module], dropout: float = 0.0,
                 residual: bool = True, dt_stats: Optional[Dict[str, float]] = None):
        super().__init__()
        self.residual = bool(residual)
        self.latent_dim = int(latent_dim)
        self.global_dim = int(global_dim)
        self.net = MLP(latent_dim + 1 + global_dim, list(hidden_dims), latent_dim, activation, dropout)

        # Normalize dt to physical inside the model (log-space bounds)
        self.dt_stats = dt_stats or {}
        self.register_buffer("dt_log_min", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("dt_log_max", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("ln10", torch.tensor(math.log(10.0), dtype=torch.float32))

        # ---- Validate Δt log-range so offsets->steps work correctly ----
        log_min = self.dt_stats.get("log_min", None)
        log_max = self.dt_stats.get("log_max", None)

        use_defaults = False
        if log_min is None or log_max is None:
            warnings.warn(
                "[LatentDynamics] dt_stats missing 'log_min'/'log_max'. "
                "Falling back to defaults (-9, 9). This can degrade semigroup/rollout losses.",
                RuntimeWarning
            )
            use_defaults = True
        else:
            try:
                log_min = float(log_min); log_max = float(log_max)
            except Exception:
                warnings.warn(
                    "[LatentDynamics] dt_stats values not numeric; overriding with (-9, 9).",
                    RuntimeWarning
                )
                use_defaults = True

        if use_defaults:
            log_min, log_max = -9.0, 9.0

        if not (math.isfinite(log_min) and math.isfinite(log_max)) or log_max <= log_min:
            warnings.warn(
                "[LatentDynamics] Invalid Δt log-bounds; overriding with (-9, 9).",
                RuntimeWarning
            )
            log_min, log_max = -9.0, 9.0

        self.dt_log_min.fill_(float(log_min))
        self.dt_log_max.fill_(float(log_max))
        self.dt_range = float(log_max - log_min)

    def _concat(self, z: torch.Tensor, dt_norm_scalar: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if dt_norm_scalar.ndim == 1:
            dt_norm_scalar = dt_norm_scalar.unsqueeze(-1)  # [B,1]
        return torch.cat([z, dt_norm_scalar, g], dim=-1)

    def forward(self, z: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        z:        [B, Z]
        dt_norm:  [B, K]
        g:        [B, G]
        Returns:  [B, K, Z]
        """
        B, K = dt_norm.shape
        z_rep = z.unsqueeze(1).expand(B, K, self.latent_dim)     # [B,K,Z]
        dt_flat = dt_norm.reshape(B * K, 1)                      # [BK,1]
        z_flat = z_rep.reshape(B * K, self.latent_dim)           # [BK,Z]
        g_flat = g.unsqueeze(1).expand(B, K, self.global_dim).reshape(B * K, self.global_dim)  # [BK,G]
        f_in = torch.cat([z_flat, dt_flat, g_flat], dim=-1)      # [BK,Z+1+G]
        dz = self.net(f_in).reshape(B, K, self.latent_dim)       # [B,K,Z]
        if self.residual:
            return z_rep + dz
        return dz

    def step(self, z: torch.Tensor, dt_step_norm: Union[torch.Tensor, float], g: torch.Tensor) -> torch.Tensor:
        """
        Single-step dynamics in latent space.
        z: [B,Z]; dt_step_norm: float or [B] or [B,1]; g: [B,G]
        Returns: [B,Z]
        """
        if not torch.is_tensor(dt_step_norm):
            dt_step_norm = torch.tensor(dt_step_norm, dtype=z.dtype, device=z.device)
        if dt_step_norm.ndim == 1:
            dt_step_norm = dt_step_norm.unsqueeze(-1)  # [B,1]
        f_in = torch.cat([z, dt_step_norm, g], dim=-1) # [B,Z+1+G]
        dz = self.net(f_in)                            # [B,Z]
        if self.residual:
            return z + dz
        return dz


# -------------------------
# FiLM from globals (decoder-side)
# -------------------------
class FiLMFromGlobals(nn.Module):
    """Apply z' = (1 + α·tanh(γ)) ⊙ z + β with a single linear projection from g."""
    def __init__(self, global_dim: int, latent_dim: int, alpha: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(global_dim, 2 * latent_dim, bias=True)
        self.alpha = float(alpha)

    def compute_affine(self, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (γ̃, β) where γ̃ = 1 + α·tanh(γ). Shapes: both [B, Z]."""
        gamma_beta = self.proj(g)  # [B,2Z]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        gamma = 1.0 + self.alpha * torch.tanh(gamma)
        return gamma, beta

    def forward(self, z: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Broadcast (γ̃,β) over time if needed and apply the affine."""
        gamma, beta = self.compute_affine(g)        # [B,Z], [B,Z]
        if z.ndim == 3:
            gamma = gamma.unsqueeze(1)              # [B,1,Z]
            beta = beta.unsqueeze(1)                # [B,1,Z]
        return gamma * z + beta


# -------------------------
# Full Model
# -------------------------
class FlowMapAutoencoder(nn.Module):
    def __init__(
        self,
        state_dim_in: int,
        state_dim_out: int,
        global_dim: int,
        latent_dim: int,
        encoder_hidden: Sequence[int],
        dynamics_hidden: Sequence[int],
        decoder_hidden: Sequence[int],
        activation: Union[str, nn.Module] = "silu",
        dropout: float = 0.0,
        dynamics_residual: bool = True,
        target_idx: Optional[torch.Tensor] = None,  # indices into species_in forming species_out
        target_log_mean: Optional[Sequence[float]] = None,
        target_log_std: Optional[Sequence[float]] = None,
        allow_partial_simplex: bool = False,
        dt_stats: Optional[Dict[str, float]] = None,
        # New options:
        decoder_condition_on_g: bool = True,
        predict_logit_delta: bool = False,
    ):
        super().__init__()
        self.S_in = int(state_dim_in)
        self.S_out = int(state_dim_out)
        self.global_dim = int(global_dim)
        self.latent_dim = int(latent_dim)

        if self.S_out != self.S_in and not allow_partial_simplex:
            raise ValueError(
                "Softmax head requires S_out == S_in for full simplex. "
                "Set allow_partial_simplex=True ONLY if you intentionally want a subset that sums to 1."
            )

        # Store stats for normalization<->physical conversions
        if target_log_mean is None or target_log_std is None:
            raise ValueError("target_log_mean and target_log_std are required for softmax head")
        self.register_buffer("log_mean", torch.tensor(target_log_mean, dtype=torch.float32))
        self.register_buffer("log_std", torch.clamp(torch.tensor(target_log_std, dtype=torch.float32), min=1e-10))
        self.register_buffer("ln10", torch.tensor(math.log(10.0), dtype=torch.float32))
        self.register_buffer("ln10_inv", torch.tensor(1.0 / math.log(10.0), dtype=torch.float32))
        self.register_buffer("target_idx", target_idx) if target_idx is not None else setattr(self, 'target_idx', None)
        self.dt_stats = dt_stats or {}

        # Modules
        self.encoder = Encoder(self.S_in, self.global_dim, list(encoder_hidden), self.latent_dim, activation, dropout)
        self.dynamics = LatentDynamics(self.latent_dim, self.global_dim, list(dynamics_hidden), activation, dropout,
                                       residual=dynamics_residual, dt_stats=dt_stats)
        self.decoder = Decoder(self.latent_dim, list(decoder_hidden), self.S_out, activation, dropout)

        # Cheap global conditioning for decoder
        self.decoder_condition_on_g = bool(decoder_condition_on_g)
        self.film = FiLMFromGlobals(self.global_dim, self.latent_dim) if self.decoder_condition_on_g else None

        # Head behavior
        self.predict_logit_delta = bool(predict_logit_delta)

        # Gentle init for stability with logit deltas
        with torch.no_grad():
            last = [m for m in self.decoder.net.modules() if isinstance(m, nn.Linear)][-1]
            last.weight.mul_(0.1)
            if last.bias is not None:
                last.bias.zero_()

    # ---- Head utilities (float32 numerics; cast back at the end) ----
    def _softmax_head_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Softmax → log10 → normalize with per-key stats. Numerics in float32."""
        import warnings
        if not torch.isfinite(logits).all():
            warnings.warn("[FlowMapAutoencoder] Non-finite logits detected before softmax.", RuntimeWarning)
        log_p = F.log_softmax(logits, dim=-1)                 # natural log p
        log_p_f = log_p.float()
        log10_p = log_p_f * self.ln10_inv                     # float32
        z_f = (log10_p - self.log_mean) / self.log_std        # float32
        if not torch.isfinite(z_f).all():
            warnings.warn("[FlowMapAutoencoder] Non-finite normalized outputs after head.", RuntimeWarning)
        return z_f.to(dtype=logits.dtype)

    def _head_from_logprobs(self, log_p: torch.Tensor) -> torch.Tensor:
        """Given natural log-probabilities, return normalized z; numerics in float32."""
        import warnings
        if not torch.isfinite(log_p).all():
            warnings.warn("[FlowMapAutoencoder] Non-finite log-probabilities provided to head.", RuntimeWarning)
        log10_p = log_p.float() * self.ln10_inv               # float32
        z_f = (log10_p - self.log_mean) / self.log_std        # float32
        if not torch.isfinite(z_f).all():
            warnings.warn("[FlowMapAutoencoder] Non-finite normalized outputs after logprob head.", RuntimeWarning)
        return z_f.to(dtype=log_p.dtype)

    def _denorm_to_logp(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Normalized y → natural log-probabilities; numerics in float32."""
        y_f = y_norm.float()
        log10_p = y_f * self.log_std + self.log_mean          # float32
        return log10_p * self.ln10                             # float32

    # ---- Core APIs ----
    def forward(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Vectorized independent offsets:
            y_i:    [B, S_in]   (normalized)
            dt_norm:[B, K]      (offsets from anchor; normalized)
            g:      [B, G]
        Returns:    [B, K, S_out] (normalized)
        """
        import warnings
        z_i = self.encoder(y_i, g)                      # [B,Z]
        z_k = self.dynamics(z_i, dt_norm, g)            # [B,K,Z]
        if self.decoder_condition_on_g:
            z_k = self.film(z_k, g)                     # [B,K,Z]
        logits = self.decoder(z_k)                      # [B,K,S_out]

        if not self.predict_logit_delta:
            return self._softmax_head_from_logits(logits)

        # Combine in log-probability space
        base = y_i if self.S_out == self.S_in else y_i.index_select(1, self.target_idx)
        base_logp = self._denorm_to_logp(base)          # [B,S_out]
        log_q = F.log_softmax(logits, dim=-1).float()   # [B,K,S_out]  (decoder distribution)
        combined = base_logp.unsqueeze(1) + log_q       # [B,K,S_out]
        log_p = combined - torch.logsumexp(combined, dim=-1, keepdim=True)

        if not torch.isfinite(log_p).all():
            warnings.warn("[FlowMapAutoencoder] Non-finite combined log-probs in forward().", RuntimeWarning)

        return self._head_from_logprobs(log_p)          # [B,K,S_out] (normalized)

    def step(self, y: torch.Tensor, dt_step_norm: Union[torch.Tensor, float], g: torch.Tensor,
             film_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        Single teacher-forced step:
            y:            [B, S_in]  (normalized)
            dt_step_norm: float or [B] or [B,1]
            g:            [B, G]
            film_cache:   optional (gamma_tilde, beta) to avoid recomputing FiLM
        Returns:          [B, S_out] (normalized)
        """
        import warnings
        z = self.encoder(y, g)                          # [B,Z]
        z_next = self.dynamics.step(z, dt_step_norm, g) # [B,Z]
        if self.decoder_condition_on_g:
            if film_cache is None:
                gamma_adj, beta = self.film.compute_affine(g)   # [B,Z], [B,Z]
            else:
                gamma_adj, beta = film_cache
            z_next = gamma_adj * z_next + beta                  # [B,Z]
        logits = self.decoder(z_next)                   # [B,S_out]

        if not self.predict_logit_delta:
            return self._softmax_head_from_logits(logits)

        # Compose base and decoder in log-prob space
        base = y if self.S_out == self.S_in else y.index_select(1, self.target_idx)
        base_logp = self._denorm_to_logp(base)          # [B,S_out]
        log_q = F.log_softmax(logits, dim=-1).float()   # [B,S_out]
        combined = base_logp + log_q                    # [B,S_out]
        log_p = combined - torch.logsumexp(combined, dim=-1, keepdim=True)

        if not torch.isfinite(log_p).all():
            warnings.warn("[FlowMapAutoencoder] Non-finite combined log-probs in step().", RuntimeWarning)

        return self._head_from_logprobs(log_p)          # [B,S_out] (normalized)

    def rollout_vectorized(self, y0: torch.Tensor, g: torch.Tensor, dt_steps_norm: torch.Tensor) -> torch.Tensor:
        """
        Sequential K-step rollout (autoregressive):
            y0:             [B, S_in]   (normalized)
            g:              [B, G]
            dt_steps_norm:  [B, K]      per-step increments
        Returns:            [B, K, S_out] (normalized)
        """
        import warnings
        B, K = y0.shape[0], dt_steps_norm.shape[1]
        z = self.encoder(y0, g)                         # [B,Z]
        outputs = []

        if self.predict_logit_delta:
            base = y0 if self.S_out == self.S_in else y0.index_select(1, self.target_idx)
            cur_logp = self._denorm_to_logp(base)        # [B,S_out]

        # Cache FiLM projection once per batch
        if self.decoder_condition_on_g:
            gamma_adj, beta = self.film.compute_affine(g)  # [B,Z], [B,Z]
        else:
            gamma_adj = beta = None

        for k in range(K):
            dt_k = dt_steps_norm[:, k]
            z = self.dynamics.step(z, dt_k, g)          # [B,Z]
            if self.decoder_condition_on_g:
                z = gamma_adj * z + beta                 # [B,Z]
            logits = self.decoder(z)                     # [B,S_out]

            if not self.predict_logit_delta:
                yk = self._softmax_head_from_logits(logits)      # normalized
            else:
                # Multiplicative update in log-probability space
                log_q = F.log_softmax(logits, dim=-1).float()    # [B,S_out]
                combined = cur_logp + log_q                      # [B,S_out]
                log_p = combined - torch.logsumexp(combined, dim=-1, keepdim=True)
                if not torch.isfinite(log_p).all():
                    warnings.warn("[FlowMapAutoencoder] Non-finite combined log-probs in rollout().", RuntimeWarning)
                yk = self._head_from_logprobs(log_p)             # normalized
                # Autoregressive base update in log-prob domain
                cur_logp = log_p

            outputs.append(yk)

        return torch.stack(outputs, dim=1)              # [B,K,S_out]


# -------------------------
# Decoder
# -------------------------
class Decoder(nn.Module):
    """
    Deterministic decoder: z -> logits over species.
    """
    def __init__(self, latent_dim: int, hidden_dims: Sequence[int], out_dim: int,
                 activation: Union[str, nn.Module] = "silu", dropout: float = 0.0):
        super().__init__()
        acts = []
        last = latent_dim
        for h in hidden_dims:
            acts += [nn.Linear(last, h), get_activation(activation)]
            if dropout and dropout > 0:
                acts.append(nn.Dropout(p=float(dropout)))
            last = h
        acts.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*acts)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# -------------------------
# Factory
# -------------------------
def create_model(config: dict) -> FlowMapAutoencoder:
    data_cfg = config.get("data", {})
    species_vars = list(data_cfg.get("species_variables") or [])
    global_vars = list(data_cfg.get("global_variables", []))
    if not species_vars:
        raise KeyError("data.species_variables required")

    target_vars = list(data_cfg.get("target_species") or species_vars)
    name_to_idx = {name: i for i, name in enumerate(species_vars)}
    try:
        target_idx = [name_to_idx[name] for name in target_vars]
    except KeyError as e:
        raise KeyError(f"target_species contains unknown name: {e.args[0]!r}")

    mcfg = config.get("model", {})
    state_dim_in = len(species_vars)
    state_dim_out = len(target_vars)
    global_dim = len(global_vars)
    latent_dim = int(mcfg.get("latent_dim", 128))
    encoder_hidden = mcfg.get("encoder_hidden", [256, 256])
    dynamics_hidden = mcfg.get("dynamics_hidden", [256, 256])
    decoder_hidden = mcfg.get("decoder_hidden", [128, 256])
    activation = mcfg.get("activation", "silu")
    dropout = float(mcfg.get("dropout", 0.0))
    dynamics_residual = bool(mcfg.get("dynamics_residual", True))
    allow_partial_simplex = bool(mcfg.get("allow_partial_simplex", False))

    # New flags
    decoder_condition_on_g = bool(mcfg.get("decoder_condition_on_g", True))
    predict_logit_delta = bool(mcfg.get("predict_logit_delta", False))

    # Load normalization manifest to get per-species log stats and dt stats
    target_log_mean, target_log_std, dt_stats = None, None, None
    paths = config.get("paths", {})
    norm_path = Path(paths.get("processed_data_dir", "data/processed")) / "normalization.json"

    if not norm_path.exists():
        raise FileNotFoundError(f"normalization.json not found at {norm_path}")

    with open(norm_path, "r") as f:
        manifest = json.load(f)
    if "dt" in manifest:
        dt_stats = {"log_min": float(manifest["dt"]["log_min"]), "log_max": float(manifest["dt"]["log_max"])}
    stats = manifest.get("per_key_stats", {})
    target_log_mean, target_log_std = [], []
    for name in target_vars:
        if name not in stats:
            raise KeyError(f"Target species '{name}' not in normalization stats")
        s = stats[name]
        target_log_mean.append(float(s.get("log_mean", 0.0)))
        target_log_std.append(float(s.get("log_std", 1.0)))

    return FlowMapAutoencoder(
        state_dim_in=state_dim_in,
        state_dim_out=state_dim_out,
        global_dim=global_dim,
        latent_dim=latent_dim,
        encoder_hidden=encoder_hidden,
        dynamics_hidden=dynamics_hidden,
        decoder_hidden=decoder_hidden,
        activation=activation,
        dropout=dropout,
        dynamics_residual=dynamics_residual,
        target_idx=torch.tensor(target_idx, dtype=torch.long),
        target_log_mean=target_log_mean,
        target_log_std=target_log_std,
        allow_partial_simplex=allow_partial_simplex,
        dt_stats=dt_stats,
        decoder_condition_on_g=decoder_condition_on_g,
        predict_logit_delta=predict_logit_delta,
    )


__all__ = ["FlowMapAutoencoder", "create_model"]

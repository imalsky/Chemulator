#!/usr/bin/env python3
"""
Stable LPV Koopman Autoencoder (trainer-compatible, hardened)
=============================================================

Key points:
- forward(y_i, dt_norm, g) returns [B, K, S] for multi-time prediction (K can be 1).
- Encoder/Decoder do a single sanitize at their boundaries (avoid repeated nan_to_num in hot loops).
- Decoder is numerically hardened:
  * Stats sanitized
  * Optional temperature with bounds
  * Logits clamped pre-softmax for bf16 stability
  * Output clamp applied both for softmax_head=True (standardized log space) and False (raw head)
  * Small-init on final decoder linear to keep early logits tame
- Stable LPV dynamics A(P,T) = S - L L^T - γ I, optional skew S
- Δt is denormalized from [0,1] using manifest log-space stats, with overflow guards
- predict_delta: if True, model predicts Δy in normalized space and returns y_i + Δy
- Latent propagation clamps λ·Δt to prevent exp overflow/underflow
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------- Constants / Utils -------------------------------

LOG_STD_MIN = 1e-9
Z_STD_CLIP_DEFAULT = 50.0
TEMP_INIT_DEFAULT = 1.0
TEMP_MIN_DEFAULT = 0.1
TEMP_MAX_DEFAULT = 10.0

# Clamp for eigen-exponent products to avoid overflow/underflow in exp
LAMDT_MAX_ABS = 50.0  # exp(±50) ~ [2e-22, 3e21]

# Clamp for logits before softmax to avoid bf16 overflow in extreme cases
LOGIT_CLAMP = 80.0

# Clamp for latent magnitude before decoding (protect Linear matmul)
Z_CLAMP = 100.0


def get_activation(name_or_mod: Union[str, nn.Module]) -> nn.Module:
    if isinstance(name_or_mod, nn.Module):
        return name_or_mod
    name = (name_or_mod or "silu").lower()
    table = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "elu": nn.ELU,
        "leakyrelu": nn.LeakyReLU,
        "leaky": nn.LeakyReLU,
        "mish": nn.Mish,
        "none": nn.Identity,
        "identity": nn.Identity,
    }
    if name not in table:
        raise ValueError(f"Unknown activation '{name}'.")
    return table[name]()


def _act_factory(activation: Union[str, nn.Module, Callable[[], nn.Module]]) -> Callable[[], nn.Module]:
    if isinstance(activation, nn.Module):
        return lambda: activation
    if isinstance(activation, str):
        return lambda: get_activation(activation)
    if callable(activation):
        return activation
    raise ValueError("activation must be str, nn.Module, or factory callable.")


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int], out_dim: int,
                 activation: Union[str, nn.Module, Callable[[], nn.Module]],
                 dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        dprev = int(in_dim)
        make_act = _act_factory(activation)
        for i, d in enumerate(list(hidden) + [int(out_dim)]):
            layers.append(nn.Linear(dprev, d))
            if i < len(hidden):
                layers.append(make_act())
                if dropout and dropout > 0:
                    layers.append(nn.Dropout(dropout))
            dprev = d
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Single sanitize at boundary; avoid per-layer nan_to_num
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return self.net(x)


# ------------------------------- Encoder/Decoder ------------------------------

class Encoder(nn.Module):
    def __init__(
        self,
        state_dim: int,
        global_dim: int,
        hidden: Sequence[int],
        latent_dim: int,
        activation: Union[str, nn.Module, Callable[[], nn.Module]],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.global_dim = int(global_dim)
        self.z_dim = int(latent_dim)
        self.net = MLP(self.state_dim + self.global_dim, hidden, self.z_dim, activation, dropout)

    def forward(self, y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x = torch.cat([y, g], dim=-1)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return self.net(x)


class Decoder(nn.Module):
    """
    Two modes:
      1) softmax_head=True:
         logits → softmax(p) → log10(p) → standardize((log10_p - mean)/std)
         Return standardized predictions (same space as targets).
      2) softmax_head=False:
         raw linear head in target space; NaN-to-num + clamp guardrails.

    z_std_clip (float or None): clamps the output in either mode.
    """
    def __init__(self, latent_dim: int, hidden: Sequence[int], state_dim: int,
                 activation: Union[str, nn.Module, Callable[[], nn.Module]], dropout: float = 0.0, *,
                 softmax_head: bool = True,
                 log_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                 temp_init: float = TEMP_INIT_DEFAULT,
                 temp_range: Tuple[float, float] = (TEMP_MIN_DEFAULT, TEMP_MAX_DEFAULT),
                 learn_temp: bool = True,
                 z_std_clip: Optional[float] = Z_STD_CLIP_DEFAULT):
        super().__init__()
        self.net = MLP(latent_dim, hidden, state_dim, activation, dropout)
        self.softmax_head = bool(softmax_head)
        self.state_dim = int(state_dim)
        self.learn_temp = bool(learn_temp)
        self.z_std_clip = float(z_std_clip) if z_std_clip is not None else None

        if self.softmax_head:
            if log_stats is None:
                raise ValueError("softmax_head=True requires (log_mean, log_std) per species.")
            log_mean, log_std = log_stats
            # sanitize stats aggressively
            log_mean = torch.nan_to_num(log_mean.detach().clone(), nan=0.0, neginf=-60.0, posinf=60.0)
            log_std = torch.nan_to_num(log_std.detach().clone(), nan=1.0, neginf=1.0, posinf=1.0).clamp(min=LOG_STD_MIN)
            self.register_buffer("log_mean", log_mean)
            self.register_buffer("log_std", log_std)
            if self.log_mean.shape[0] != self.state_dim or self.log_std.shape[0] != self.state_dim:
                raise ValueError("Decoder stats shape mismatch.")
            tmin, tmax = float(temp_range[0]), float(temp_range[1])
            self.tmin, self.tmax = tmin, tmax
            if self.learn_temp:
                self.temp = nn.Parameter(torch.tensor(float(temp_init)))
            else:
                self.register_buffer("temp_fixed", torch.tensor(float(temp_init)))

        # small-init the final linear to keep early logits near zero
        head = None
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                head = m
        if head is not None:
            nn.init.normal_(head.weight, mean=0.0, std=1e-4)  # tighter for bf16 stability
            nn.init.zeros_(head.bias)

    def _temperature(self) -> torch.Tensor:
        if not self.softmax_head:
            return torch.as_tensor(TEMP_INIT_DEFAULT, device=next(self.parameters()).device)
        if self.learn_temp:
            return self.temp.clamp(self.tmin, self.tmax)
        return self.temp_fixed.clamp(self.tmin, self.tmax)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Sanitize latent before first Linear to avoid Addmm NaN in backward
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-Z_CLAMP, Z_CLAMP)
        logits = self.net(z)

        if not self.softmax_head:
            out = torch.nan_to_num(logits, nan=0.0, neginf=-1e6, posinf=1e6)
            if self.z_std_clip is not None:
                out = out.clamp_(-self.z_std_clip, self.z_std_clip)
            return out

        # softmax path
        t = torch.nan_to_num(self._temperature(), nan=float(TEMP_MIN_DEFAULT)).clamp(TEMP_MIN_DEFAULT, TEMP_MAX_DEFAULT)
        x = torch.nan_to_num(logits / t).clamp_(-LOGIT_CLAMP, LOGIT_CLAMP)  # clamp for stability
        log10e = 1.0 / math.log(10.0)
        log10_p = F.log_softmax(x, dim=-1) * log10e
        log10_p = torch.nan_to_num(log10_p, nan=-1e6, neginf=-1e6, posinf=1e6)
        z_std = (log10_p - self.log_mean) / self.log_std
        z_std = torch.nan_to_num(z_std, nan=0.0, neginf=-1e6, posinf=1e6)
        if self.z_std_clip is not None:
            z_std = z_std.clamp_(-self.z_std_clip, self.z_std_clip)
        return z_std


# -------------------------- LPV Stable Latent Dynamics ------------------------

class _NoOpTimeWarp(nn.Module):
    def __init__(self, smax_init: float = 0.0):
        super().__init__()
        self.register_buffer("smax", torch.tensor(float(smax_init)))
        self.last_s_mean = 0.0


class LPVDynamics(nn.Module):
    """
    A(P,T) = S - L L^T - γ I; z_next = exp(A Δt) z
    - S is skew-symmetric (optional) scaled by tanh(scale_S)
    - L is low-rank with positive singular values via softplus
    """
    def __init__(self, latent_dim: int, global_dim: int,
                 cond_hidden: Sequence[int], rank_l: int, use_S: bool, gamma: float,
                 activation: Union[str, nn.Module, Callable[[], nn.Module]], dropout: float,
                 dt_stats: Dict[str, float]):
        super().__init__()
        self.z = int(latent_dim)
        self.g = int(global_dim)
        self.r = int(rank_l)
        self.use_S = bool(use_S)
        self.gamma = float(gamma)

        self.cond = MLP(self.g, cond_hidden, self.r + (self.z * self.z if self.use_S else 0),
                        activation, dropout)

        with torch.no_grad():
            Q, _ = torch.linalg.qr(torch.randn(self.z, self.r))
        self.register_buffer("base_U", Q)                 # [z,r]
        self.register_buffer("I", torch.eye(self.z))      # [z,z]
        self.timewarp = _NoOpTimeWarp(0.0)

        # Δt log range buffers (log10 space)
        if ("log_min" not in dt_stats) or ("log_max" not in dt_stats):
            raise ValueError("dt_stats must include 'log_min' and 'log_max' (log10 space).")
        log_min = float(dt_stats["log_min"])
        log_max = float(dt_stats["log_max"])
        if not math.isfinite(log_min) or not math.isfinite(log_max) or (log_max <= log_min):
            raise ValueError(f"Bad dt_stats: log_min={log_min}, log_max={log_max}")
        self.register_buffer("dt_log_min", torch.tensor(log_min, dtype=torch.float32))
        self.register_buffer("dt_log_max", torch.tensor(log_max, dtype=torch.float32))

        self.scale_S = nn.Parameter(torch.tensor(1.0))

    def _build_A(self, pt: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Build A(g): S - L L^T - γ I
        Single sanitize at input; one construction pass; no repeated to()/nan_to_num() churn.
        """
        B = pt.shape[0]
        pt = torch.nan_to_num(pt, nan=0.0, posinf=0.0, neginf=0.0)

        h = self.cond(pt)  # [B, r + (z*z if use_S)]
        if self.use_S:
            s, m_head = torch.split(h, [self.r, self.z * self.z], dim=-1)
        else:
            s, m_head = h, None

        s = F.softplus(s)  # [B,r]  (positive)
        U = self.base_U    # [z,r]
        L = U.unsqueeze(0) * s.unsqueeze(1)  # [B,z,r]
        LLt = torch.matmul(L, L.transpose(1, 2))  # [B,z,z], PSD

        if self.use_S:
            M = m_head.reshape(B, self.z, self.z)
            S = 0.5 * (M - M.transpose(1, 2))  # skew
            S = torch.tanh(self.scale_S) * S
        else:
            S = torch.zeros(B, self.z, self.z, device=pt.device, dtype=pt.dtype)

        A = S - LLt - self.gamma * self.I.to(device=pt.device, dtype=pt.dtype).expand(B, -1, -1)
        A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)  # final guard
        aux = {
            "damp_mean": float(s.mean().detach().cpu()),
            "gamma": self.gamma,
            "smax": float(self.timewarp.smax.detach().cpu()),
        }
        return A, aux

    @property
    def _dt_range_scalar(self) -> torch.Tensor:
        return (self.dt_log_max - self.dt_log_min).clamp_min(1e-12)

    def _denorm_dt(self, dt_norm: torch.Tensor) -> torch.Tensor:
        """
        Map normalized dt in [0,1] to physical seconds using stored log10 range.
        Guard against NaN/Inf before clamping; cap log10(dt) to 38 (~1e38).
        """
        dt_norm = torch.nan_to_num(dt_norm, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        log_dt = self.dt_log_min + dt_norm * self._dt_range_scalar
        log_dt = torch.clamp(log_dt, max=38.0)  # ~1e38 cap
        dt_phys = torch.exp(log_dt * math.log(10.0))
        dt_phys = torch.nan_to_num(dt_phys, nan=1e-30, posinf=1e38, neginf=1e-30).clamp_min(1e-30)
        return dt_phys

    def step(self, z: torch.Tensor, dt_step_norm: torch.Tensor | float, pt: torch.Tensor) -> torch.Tensor:
        """
        Backward-compatible latent step with normalized Δt.
        Converts to physical seconds and calls step_phys (no eigh here).
        """
        if not torch.is_tensor(dt_step_norm):
            dt_step_norm = torch.as_tensor(dt_step_norm, device=z.device, dtype=torch.float32)
        dt_step_norm = dt_step_norm.to(dtype=torch.float32, device=z.device)
        if dt_step_norm.ndim == 0:
            dt_step_norm = dt_step_norm.expand(z.shape[0])
        elif dt_step_norm.ndim != 1 or dt_step_norm.shape[0] != z.shape[0]:
            dt_step_norm = dt_step_norm.view(z.shape[0])
        dt_phys = self._denorm_dt(dt_step_norm)
        return self.step_phys(z, dt_phys, pt)

    def step_phys(self, z: torch.Tensor, dt_phys: torch.Tensor, pt: torch.Tensor) -> torch.Tensor:
        """
        Latent step with physical Δt provided (skips per-step denorm).
          z: [B,Z], dt_phys: [B], pt: [B,G]
        Uses fp32 matrix_exp for stable, differentiable propagation.
        """
        B, Z = z.shape
        if dt_phys.ndim != 1 or dt_phys.shape[0] != B:
            raise ValueError(f"dt_phys must be [B]; got {tuple(dt_phys.shape)} vs B={B}")

        with torch.amp.autocast("cuda", enabled=False):
            A, _ = self._build_A(pt)
            A32 = A.float()  # [B,Z,Z]
            dt32 = torch.nan_to_num(dt_phys.float(), nan=1e-30, posinf=1e38, neginf=1e-30).clamp_min(0.0).view(B, 1, 1)
            Phi = torch.matrix_exp(A32 * dt32)  # [B,Z,Z]
            z_next32 = torch.bmm(Phi, z.float().unsqueeze(-1)).squeeze(-1)

        z_next32 = torch.nan_to_num(z_next32, nan=0.0, posinf=0.0, neginf=0.0)
        return z_next32.to(z.dtype)


# ------------------------------ Full Model -----------------------------------

class StableLPVKoopmanAE(nn.Module):
    """
    Encoder → LPV Dynamics → Decoder.
    Works entirely in normalized target space; only Δt is de/renormalized inside dynamics.

    forward(y_i, dt_norm, g):
      y_i: [B,S], dt_norm: [B] or [B,K], g: [B,G]  →  [B,K,S]
    """
    def __init__(
        self,
        *,
        state_dim: int,
        global_dim: int,
        latent_dim: int,
        encoder_hidden: Sequence[int],
        decoder_hidden: Sequence[int],
        activation: Union[str, nn.Module, Callable[[], nn.Module]] = "silu",
        dropout: float = 0.0,
        softmax_head: bool = True,
        z_std_clip: Optional[float] = Z_STD_CLIP_DEFAULT,
        predict_delta: bool = False,
        # decoder stats (only if softmax_head=True)
        target_log_mean: Optional[Sequence[float]] = None,
        target_log_std: Optional[Sequence[float]] = None,
        # LPV dynamics
        cond_hidden: Sequence[int] = (64, 64),
        rank_l: int = 8,
        use_S: bool = False,
        gamma: float = 1e-2,
        dt_stats: Dict[str, float] = None,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            # ignore unrecognized kwargs for forward-compat
            pass

        self.S_in = int(state_dim)
        self.S_out = int(state_dim)
        self.G_in = int(global_dim)
        self.Z = int(latent_dim)
        self.use_S = bool(use_S)
        self.predict_delta = bool(predict_delta)

        act = get_activation(activation) if isinstance(activation, str) else activation

        self.encoder = Encoder(self.S_in, self.G_in, encoder_hidden, self.Z, act, dropout)
        self.dynamics = LPVDynamics(
            latent_dim=self.Z,
            global_dim=self.G_in,
            cond_hidden=cond_hidden,
            rank_l=rank_l,
            use_S=use_S,
            gamma=gamma,
            activation=act,
            dropout=dropout,
            dt_stats=dt_stats,
        )

        log_stats = None
        if softmax_head:
            if target_log_mean is None or target_log_std is None:
                raise ValueError("softmax_head=True requires target_log_mean/target_log_std arrays.")
            log_mean = torch.tensor(list(target_log_mean), dtype=torch.float32)
            log_std = torch.tensor(list(target_log_std), dtype=torch.float32)
            log_stats = (log_mean, log_std)

        self.decoder = Decoder(self.Z, decoder_hidden, self.S_out, act, dropout,
                               softmax_head=softmax_head, log_stats=log_stats,
                               z_std_clip=z_std_clip)

    # ---- Trainer API: multi-time forward ----
    def forward(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Efficient multi-time prediction.
          y_i: [B,S], dt_norm: [B] or [B,K], g: [B,G]  →  [B,K,S]
        Encodes once, propagates all K steps in latent space, decodes once (batched).
        For predict_delta=True, returns y_i + Δy_k for each k (anchor-relative).
        """
        B, S = y_i.shape
        if dt_norm.ndim == 1:
            if dt_norm.shape[0] != B:
                raise ValueError(f"dt_norm shape mismatch: got {tuple(dt_norm.shape)}, expected [{B}] or [{B},K]")
            dt_norm = dt_norm.reshape(B, 1)
        elif dt_norm.ndim == 2:
            if dt_norm.shape[0] != B:
                raise ValueError(f"dt_norm batch mismatch: got {tuple(dt_norm.shape)}, expected [{B},K]")
        else:
            raise ValueError(f"dt_norm must be [B] or [B,K], got {tuple(dt_norm.shape)}")
        K = dt_norm.shape[1]

        # Single sanitize at the boundary
        y_i = torch.nan_to_num(y_i, nan=0.0, posinf=0.0, neginf=0.0)
        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

        # Encode once
        z0 = self.encoder(y_i, g)  # [B,Z]

        # Build A(g) once
        A, _ = self.dynamics._build_A(g)  # [B,Z,Z]

        # Vectorized propagation for all K steps
        try:
            if not self.use_S:
                # Eigh path for symmetric A (fast). Operate in fp32.
                eigvals32, eigvecs32 = torch.linalg.eigh(A.float())  # [B,Z], [B,Z,Z]
                z0_eig32 = torch.bmm(eigvecs32.transpose(-2, -1), z0.float().unsqueeze(-1)).squeeze(-1)  # [B,Z]
                dt_phys32 = self.dynamics._denorm_dt(dt_norm).float()  # [B,K]
                lam_dt = eigvals32.unsqueeze(1) * dt_phys32.unsqueeze(-1)  # [B,K,Z]
                lam_dt = lam_dt.clamp_(-LAMDT_MAX_ABS, LAMDT_MAX_ABS)
                exp_lam_dt = torch.exp(lam_dt)  # [B,K,Z]
                z_eig_next = z0_eig32.unsqueeze(1) * exp_lam_dt  # [B,K,Z]
                z_next32 = torch.einsum('bij,bkj->bki', eigvecs32, z_eig_next)  # [B,K,Z]
                z_next32 = torch.nan_to_num(z_next32, nan=0.0, posinf=0.0, neginf=0.0)
                z_next = z_next32.to(z0.dtype)
            else:
                # Generic matrix_exp path in fp32
                dt_phys32 = self.dynamics._denorm_dt(dt_norm).float()  # [B,K]
                A32 = A.float().unsqueeze(1)  # [B,1,Z,Z]
                Phi32 = torch.matrix_exp(A32 * dt_phys32.reshape(B, K, 1, 1))  # [B,K,Z,Z]
                z0_32 = z0.float().reshape(B, 1, self.Z, 1)  # [B,1,Z,1]
                z_next32 = torch.matmul(Phi32, z0_32).squeeze(-1)  # [B,K,Z]
                z_next32 = torch.nan_to_num(z_next32, nan=0.0, posinf=0.0, neginf=0.0)
                z_next = z_next32.to(z0.dtype)
        except RuntimeError:
            # Fallback to matrix_exp if eigh path fails numerically
            dt_phys32 = self.dynamics._denorm_dt(dt_norm).float()  # [B,K]
            A32 = A.float().unsqueeze(1)  # [B,1,Z,Z]
            Phi32 = torch.matrix_exp(A32 * dt_phys32.reshape(B, K, 1, 1))  # [B,K,Z,Z]
            z0_32 = z0.float().reshape(B, 1, self.Z, 1)  # [B,1,Z,1]
            z_next32 = torch.matmul(Phi32, z0_32).squeeze(-1)  # [B,K,Z]
            z_next32 = torch.nan_to_num(z_next32, nan=0.0, posinf=0.0, neginf=0.0)
            z_next = z_next32.to(z0.dtype)

        # Single batched decode for all K latents
        z_next = z_next.clamp_(-Z_CLAMP, Z_CLAMP)
        y_out = self.decoder(z_next.reshape(B * K, self.Z)).reshape(B, K, S)  # [B,K,S]

        if self.predict_delta:
            y_out = y_i.unsqueeze(1) + y_out
        return y_out

    # ---- Convenience single-step (kept for callers that need it) ----
    def step(self, y: torch.Tensor, dt_step_norm: torch.Tensor | float, g: torch.Tensor) -> torch.Tensor:
        z = self.encoder(y, g)
        z_next = self.dynamics.step(z, dt_step_norm, g)
        y_next = self.decoder(z_next)
        if self.predict_delta:
            y_next = y + y_next
        return y_next

    @torch.no_grad()
    def rollout(self, y0: torch.Tensor, g: torch.Tensor, dt_step_norm: torch.Tensor | float, steps: int) -> torch.Tensor:
        """
        Returns [B, steps, S] (does NOT include y0).
        """
        B, _ = y0.shape
        y_t = y0
        out = []
        for _ in range(int(steps)):
            y_t = self.step(y_t, dt_step_norm, g)
            out.append(y_t)
        return torch.stack(out, dim=1)  # [B,steps,S]


# --------------------- Manifest helpers + model factory -----------------------

def _load_manifest(config: dict) -> dict:
    # Try config path first, then local fallback
    cand: list[Path] = []
    if "paths" in config and "processed_data_dir" in config["paths"]:
        cand.append(Path(config["paths"]["processed_data_dir"]) / "normalization.json")
    cand.append(Path("data/processed/normalization.json"))
    for p in cand:
        if p.exists():
            with open(p, "r") as f:
                return json.load(f)
    raise FileNotFoundError(f"Could not find normalization manifest at: {', '.join(map(str, cand))}")


def _gather_species_stats(methods: dict, stats: dict, species_vars: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    log_means, log_stds = [], []
    for k in species_vars:
        m = methods.get(k, "log-standard")
        if m != "log-standard":
            raise ValueError(f"Decoder softmax_head expects 'log-standard' for '{k}', got '{m}'.")
        st = stats.get(k, {})
        if ("log_mean" not in st) or ("log_std" not in st):
            raise KeyError(f"Missing log stats for species '{k}' in manifest.")
        lm = float(st.get("log_mean"))
        ls = max(float(st.get("log_std")), LOG_STD_MIN)
        log_means.append(lm)
        log_stds.append(ls)
    return torch.tensor(log_means, dtype=torch.float32), torch.tensor(log_stds, dtype=torch.float32)


def _get_dt_stats(manifest: dict, stats: dict) -> Dict[str, float]:
    # standard location
    if "dt" in manifest:
        st = manifest["dt"]
        if "log_min" in st and "log_max" in st:
            return {"log_min": float(st["log_min"]), "log_max": float(st["log_max"])}
    # stats fallback
    if "dt" in stats:
        st = stats["dt"]
        if "log_min" in st and "log_max" in st:
            return {"log_min": float(st["log_min"]), "log_max": float(st["log_max"])}
    # common aliases
    for k in ("Δt", "delta_t", "t_delta", "t_step", "dt_phys"):
        if k in stats and "log_min" in stats[k] and "log_max" in stats[k]:
            st = stats[k]
            return {"log_min": float(st["log_min"]), "log_max": float(st["log_max"])}
    raise KeyError("Could not find dt stats (expected dt.log_min/log_max) in manifest.")


# sensible defaults
LATENT_DIM_DEFAULT = 32
ENCODER_HIDDEN_DEFAULT = (256, 256)
DECODER_HIDDEN_DEFAULT = (256, 256)
COND_HIDDEN_DEFAULT = (64, 64)
RANK_L_DEFAULT = 8
GAMMA_DEFAULT = 1e-2


def create_model(config: dict) -> nn.Module:
    """
    Build model from config + normalization manifest.

    Expected shape resolution:
      - species_variables list comes from config["data"]["species_variables"]
        or is inferred from manifest stats by excluding globals and dt-like keys.
      - global_variables from config (recommended: ["P","T"]).
    """
    manifest = _load_manifest(config)
    stats = manifest.get("stats", {})
    methods = manifest.get("methods", {})
    data_cfg = config.get("data", {}) or {}

    species_vars = list(data_cfg.get("species_variables", []) or [])
    global_vars = list(data_cfg.get("global_variables", []) or [])
    if not global_vars:
        if "P" in stats and "T" in stats:
            global_vars = ["P", "T"]
        else:
            raise ValueError("No global_variables provided and P/T not found in manifest stats.")

    if not species_vars:
        time_var = data_cfg.get("time_variable", "t_time")
        reserved = {"dt", "Δt", "delta_t", "t_delta", "t_step", "dt_phys", time_var} | set(global_vars)
        species_vars = [k for k in stats.keys() if k not in reserved]

    if not species_vars:
        raise ValueError("No species_variables resolved from config or manifest.")

    # If softmax head is used, gather per-species log stats
    mcfg = config.get("model", {}) or {}
    softmax_head = bool(mcfg.get("softmax_head", True))
    if softmax_head:
        log_mean, log_std = _gather_species_stats(methods, stats, species_vars)
        target_log_mean = log_mean.tolist()
        target_log_std = log_std.tolist()
    else:
        target_log_mean = target_log_std = None

    # dynamics dt stats
    dt_stats = _get_dt_stats(manifest, stats)

    # model hyperparams
    latent_dim = int(mcfg.get("latent_dim", LATENT_DIM_DEFAULT))
    enc_h = tuple(mcfg.get("encoder_hidden", ENCODER_HIDDEN_DEFAULT))
    dec_h = tuple(mcfg.get("decoder_hidden", DECODER_HIDDEN_DEFAULT))
    activation = mcfg.get("activation", "silu")
    dropout = float(mcfg.get("dropout", 0.0))
    cond_h = tuple(mcfg.get("cond_hidden", COND_HIDDEN_DEFAULT))
    rank_l = int(mcfg.get("rank_l", RANK_L_DEFAULT))
    use_S = bool(mcfg.get("use_S", False))
    gamma = float(mcfg.get("gamma", GAMMA_DEFAULT))
    z_std_clip = float(mcfg.get("z_std_clip", Z_STD_CLIP_DEFAULT))
    predict_delta = bool(mcfg.get("predict_delta", False))

    model = StableLPVKoopmanAE(
        state_dim=len(species_vars),
        global_dim=len(global_vars),
        latent_dim=latent_dim,
        encoder_hidden=enc_h,
        decoder_hidden=dec_h,
        activation=activation,
        dropout=dropout,
        softmax_head=softmax_head,
        z_std_clip=z_std_clip,
        predict_delta=predict_delta,
        target_log_mean=target_log_mean,
        target_log_std=target_log_std,
        cond_hidden=cond_h,
        rank_l=rank_l,
        use_S=use_S,
        gamma=gamma,
        dt_stats=dt_stats,
    )
    return model


# Back-compat alias
FlowMapKoopman = StableLPVKoopmanAE

__all__ = [
    "MLP", "Encoder", "Decoder",
    "LPVDynamics", "StableLPVKoopmanAE",
    "create_model", "FlowMapKoopman",
]

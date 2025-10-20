#!/usr/bin/env python3
"""
Stable LPV Koopman Autoencoder (trainer-compatible, hardened)
=============================================================

Key points:
- forward(y_i, dt_norm, g) → [B, K, S] multi-time prediction.
- Encoder/Decoder sanitize only at boundaries (avoid repeated nan_to_num in hot loops).
- Decoder hardened for bf16: small-init, clamp logits, optional temperature.
- Stable LPV dynamics A(P,T) = S - L L^T - γ I, optional skew S.
- Δt is denormalized from [0,1] using manifest log-space stats (log10).
- predict_delta=True now returns cumulative deltas across K:
    y_pred[:, t] = y_i + sum_{j<=t} Δy_j
- Latent propagation clamps λ·Δt for exp stability.
- Eigh fast path is used only in eval/no-grad when use_S=False (safe gradients).
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
            nn.init.normal_(head.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(head.bias)

    def _temperature(self) -> torch.Tensor:
        if not self.softmax_head:
            return torch.as_tensor(TEMP_INIT_DEFAULT, device=next(self.parameters()).device)
        if self.learn_temp:
            return self.temp.clamp(self.tmin, self.tmax)
        return self.temp_fixed.clamp(self.tmin, self.tmax)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-Z_CLAMP, Z_CLAMP)
        logits = self.net(z)

        if not self.softmax_head:
            out = torch.nan_to_num(logits, nan=0.0, neginf=-1e6, posinf=1e6)
            if self.z_std_clip is not None:
                out = out.clamp_(-self.z_std_clip, self.z_std_clip)
            return out

        t = torch.nan_to_num(self._temperature(), nan=float(TEMP_MIN_DEFAULT)).clamp(TEMP_MIN_DEFAULT, TEMP_MAX_DEFAULT)
        x = torch.nan_to_num(logits / t).clamp_(-LOGIT_CLAMP, LOGIT_CLAMP)
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

        Returns:
          A:   [B, Z, Z] tensor (same device/dtype as pt)
          aux: dict with lightweight diagnostics
        """
        B = pt.shape[0]
        pt = torch.nan_to_num(pt, nan=0.0, posinf=0.0, neginf=0.0)

        h = self.cond(pt)  # [B, r + (z*z if use_S else 0)]
        if self.use_S:
            s, m_head = torch.split(h, [self.r, self.z * self.z], dim=-1)
        else:
            s, m_head = h, None

        s = F.softplus(s)  # [B, r] ≥ 0

        U = self.base_U  # [z, r]
        L = U.unsqueeze(0) * s.unsqueeze(1)  # [B, z, r]
        LLt = torch.matmul(L, L.transpose(1, 2))  # [B, z, z]

        if self.use_S:
            M = m_head.reshape(B, self.z, self.z)
            S = 0.5 * (M - M.transpose(1, 2))
            S = torch.tanh(self.scale_S) * S
        else:
            S = torch.zeros(B, self.z, self.z, device=pt.device, dtype=pt.dtype)

        A = S - LLt - self.gamma * self.I.to(device=pt.device, dtype=pt.dtype).expand(B, -1, -1)
        A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

        if not self.use_S:
            A = 0.5 * (A + A.transpose(1, 2))  # keep exact symmetry

        aux = {}
        try:
            import torch._dynamo as _dynamo  # type: ignore
            compiling = getattr(_dynamo, "is_compiling", None)
            if compiling is None or not compiling():
                aux = {
                    "damp_mean": float(s.mean().detach().cpu()),
                    "gamma": self.gamma,
                    "smax": float(getattr(self.timewarp, "smax", torch.tensor(0.0)).detach().cpu()),
                }
        except Exception:
            pass

        return A, aux

    @property
    def _dt_range_scalar(self) -> torch.Tensor:
        return (self.dt_log_max - self.dt_log_min).clamp_min(1e-12)

    def _denorm_dt(self, dt_norm: torch.Tensor) -> torch.Tensor:
        """
        Map normalized dt in [0,1] -> physical seconds using stored log10 range.
        Accepts [B] or [B,K]; returns same shape (fp32).
        """
        dt_norm = torch.nan_to_num(dt_norm, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)

        log_min = self.dt_log_min.to(torch.float64)
        log_max = self.dt_log_max.to(torch.float64)
        rng = (log_max - log_min).clamp(min=1e-12)
        log_dt = log_min + dt_norm.to(torch.float64) * rng

        dt_phys = torch.pow(torch.tensor(10.0, dtype=torch.float64, device=log_dt.device), log_dt)
        dt_phys = dt_phys.clamp_(min=1e-30)
        return dt_phys.to(torch.float32)

    # -------------------------- matrix exp helper ---------------------------

    @staticmethod
    def _apply_matrix_exp_bcast(A32: torch.Tensor, dt_phys32: torch.Tensor,
                                z0_32: Optional[torch.Tensor] = None,
                                target_mb: int = 300) -> torch.Tensor:
        """
        Compute exp(A*dt) (and optionally * z0) with adaptive chunking over the combined batch.

        A32:        [B, Z, Z]
        dt_phys32:  [B, K]
        z0_32:      [B, Z, 1] or None
        Returns:
          if z0_32 is None: [B, K, Z, Z]
          else            : [B, K, Z]   (result of Phi @ z0)
        """
        B, Z, _ = A32.shape
        K = int(dt_phys32.shape[1])

        # Work in blocks over K to keep memory bounded
        # bytes per (B, k, Z, Z) fp32 tensor:
        def bytes_for(b: int, k: int, z: int) -> int:
            return b * k * z * z * 4

        mb_limit = max(1, int(target_mb)) * 1024 * 1024
        # choose kchunk so that B * kchunk * Z * Z * 4 <= mb_limit
        kchunk = max(1, min(K, mb_limit // max(1, bytes_for(B, 1, Z))))

        outs = []
        for k0 in range(0, K, kchunk):
            k1 = min(K, k0 + kchunk)
            dt32 = dt_phys32[:, k0:k1].reshape(B, k1 - k0, 1, 1)  # [B, kc, 1, 1]
            Phi = torch.matrix_exp(A32.unsqueeze(1) * dt32)  # [B, kc, Z, Z]
            if z0_32 is None:
                outs.append(Phi)
            else:
                kc = k1 - k0
                # Expand z0 across the kc dimension so the matmul is well-defined
                zcol = z0_32.unsqueeze(1).expand(B, kc, Z, 1)  # [B, kc, Z, 1]
                outs.append(torch.matmul(Phi, zcol).squeeze(-1))  # [B, kc, Z]
        return torch.cat(outs, dim=1)

    # ------------------------------- stepping --------------------------------

    def step(self, z: torch.Tensor, dt_step_norm: torch.Tensor | float, pt: torch.Tensor) -> torch.Tensor:
        """
        Latent step with normalized Δt → physical seconds.
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
        z: [B,Z], dt_phys: [B], pt: [B,G]
        Chunked matrix_exp over the batch to keep memory bounded.
        """
        B, Z = z.shape
        if dt_phys.ndim != 1 or dt_phys.shape[0] != B:
            raise ValueError(f"dt_phys must be [B]; got {tuple(dt_phys.shape)} vs B={B}")

        with torch.amp.autocast("cuda", enabled=False):
            A, _ = self._build_A(pt)  # [B,Z,Z]
            A32 = A.float()
            zcol = z.float().unsqueeze(-1)  # [B,Z,1]
            dt32 = torch.nan_to_num(dt_phys.float(), nan=1e-30, posinf=1e30, neginf=1e-30).clamp_min_(0.0)

            # Adaptive chunk over B for ~300MB target
            bytes_per_mat = (Z * Z) * 4
            target_bytes = 300 * 1024 ** 2
            chunk_B = max(1, min(B, target_bytes // max(1, bytes_per_mat)))

            outs = []
            for i in range(0, B, chunk_B):
                Ai = A32[i:i + chunk_B]  # [b,Z,Z]
                dti = dt32[i:i + chunk_B].view(-1, 1, 1)  # [b,1,1]
                Phi = torch.matrix_exp(Ai * dti)  # [b,Z,Z]
                zi = torch.bmm(Phi, zcol[i:i + chunk_B]).squeeze(-1)
                outs.append(zi)

            z_next32 = torch.cat(outs, dim=0)

        z_next32 = torch.nan_to_num(z_next32, nan=0.0, posinf=0.0, neginf=0.0)
        return z_next32.to(z.dtype)


# ------------------------------ Full Model -----------------------------------

class StableLPVKoopmanAE(nn.Module):
    """
    Encoder → LPV Dynamics → Decoder.
    Works entirely in normalized target space; only Δt is de/renormalized inside dynamics.

    forward(y_i, dt_norm, g):
      y_i: [B,S], dt_norm: [B] or [B,K], g: [B,G]  →  [B,K,S]

    Notes:
    - Main forward does parallel propagation from the single encoding z0 at K offsets.
      (Matches multi-time dataset targets relative to anchor.)
    - For trainer rollouts, call .step() sequentially or use the helper in the trainer
      to cache A and z0 to avoid re-encoding.
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
        y_i: [B,S], dt_norm: [B] or [B,K], g: [B,G]  →  [B,K,S]
        TRAIN: matrix_exp path (fp32 inside autocast-disabled block), chunked adaptively over K.
        EVAL/NO-GRAD: eigh fast path when use_S=False (safer numerics, faster).
        """
        B, S = y_i.shape

        # Normalize dt to [B,K]
        if dt_norm.ndim == 1:
            if dt_norm.shape[0] != B:
                raise ValueError(f"dt_norm shape mismatch: got {tuple(dt_norm.shape)}, expected [{B}] or [{B},K]")
            dt_norm = dt_norm.reshape(B, 1)
        elif dt_norm.ndim == 2:
            if dt_norm.shape[0] != B:
                raise ValueError(f"dt_norm batch mismatch: got {tuple(dt_norm.shape)}, expected [{B},K]")
        else:
            raise ValueError(f"dt_norm must be [B] or [B,K], got {tuple(dt_norm.shape)}")
        K = int(dt_norm.shape[1])

        # Boundary sanitization
        y_i = torch.nan_to_num(y_i, nan=0.0, posinf=0.0, neginf=0.0)
        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

        # Encode & build A(g)
        z0 = self.encoder(y_i, g)  # [B,Z]
        A, _ = self.dynamics._build_A(g)  # [B,Z,Z]
        dt_phys32 = self.dynamics._denorm_dt(dt_norm).float()  # [B,K]

        use_fast_eigh = (not self.use_S) and (not torch.is_grad_enabled())  # eval-only
        if use_fast_eigh:
            with torch.amp.autocast("cuda", enabled=False):
                A32 = A.float()
                A32 = 0.5 * (A32 + A32.transpose(1, 2))  # ensure symmetry
                lam, U = torch.linalg.eigh(A32)  # lam:[B,Z], U:[B,Z,Z]

                v = torch.matmul(U.transpose(1, 2), z0.float().unsqueeze(-1)).squeeze(-1)  # [B,Z]
                lam = torch.nan_to_num(lam, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(1)  # [B,1,Z]
                dt = dt_phys32.unsqueeze(-1)  # [B,K,1]
                lamdt = (lam * dt).clamp_(-LAMDT_MAX_ABS, LAMDT_MAX_ABS)  # [B,K,Z]
                expfac = torch.exp(lamdt)  # [B,K,Z]
                w = expfac * v.unsqueeze(1)  # [B,K,Z]
                z_next32 = torch.matmul(U, w.transpose(1, 2)).transpose(1, 2)  # [B,K,Z]
        else:
            with torch.amp.autocast("cuda", enabled=False):
                A32 = A.float()                          # [B,Z,Z]
                z0_32 = z0.float().view(B, self.Z, 1)    # [B,Z,1]
                z_next32 = self.dynamics._apply_matrix_exp_bcast(A32, dt_phys32, z0_32, target_mb=300)  # [B,K,Z]

        z_next32 = torch.nan_to_num(z_next32, nan=0.0, posinf=0.0, neginf=0.0)
        z_next = z_next32.to(z0.dtype).clamp_(-Z_CLAMP, Z_CLAMP)

        # Decode batched
        y_out = self.decoder(z_next.reshape(B * K, self.Z)).reshape(B, K, S)

        # --- FIX: cumulative deltas for multi-step when predict_delta is True ---
        if self.predict_delta:
            # Each step's decoder output is Δy_t. Use cumulative sum over K.
            # y[:, t] = y_i + sum_{j<=t} Δy_j
            y_out = y_i.unsqueeze(1) + torch.cumsum(y_out, dim=1)

        return y_out

    # ---- Convenience single-step ----
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
        Returns [B, steps, S] (does NOT include y0). Uses single-step path.
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

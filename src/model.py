#!/usr/bin/env python3
"""
Flow-Map Semigroup Koopman Autoencoder
=======================================
Neural ODE-free Koopman operator learning with closed-form exponential propagation.
Uses low-rank linear parameter-varying (LPV) dynamics for efficient variable-time predictions.

Architecture:
- Encoder: Maps (state, globals) → latent codes
- Dynamics: Closed-form semigroup flow z(t) = exp(tA(g)) z(0) with:
  * Low-rank diagonalizable A(g) = U(g) Λ(g) U(g)^T + λ⊥(g) I⊥
  * Equilibrium point z_eq(g) for stability
  * Optional learnable orthonormal basis U(g) or fixed (DCT/identity)
- Decoder: Maps (latent, globals) → predicted state

Key Features:
- No ODE solver required - exact exponentials via eigendecomposition
- Numerical stability: optional FP64 for exp(), clipping, NaN protection
- Semigroup property: z(t1+t2) = exp(t2 A) exp(t1 A) z(0) by design
- Variable time prediction without retraining
- Efficient batched inference with K simultaneous time predictions
- Optional residual (delta) prediction mode

Critical: Requires dt normalization statistics (log_min, log_max) from preprocessing.
Model will exit with error if normalization.json is missing or incomplete.
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, Sequence, Union

import torch
import torch.nn as nn

LN10: float = math.log(10.0)
Z_STD_CLIP_DEFAULT: float = 10.0


# -------------------------
# Small helpers
# -------------------------
def get_activation(name: Union[str, nn.Module]) -> nn.Module:
    if isinstance(name, nn.Module):
        return name
    n = str(name).lower()
    return {
        "relu": nn.ReLU(inplace=False),
        "leakyrelu": nn.LeakyReLU(negative_slope=0.01, inplace=False),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "swish": nn.SiLU(),
        "tanh": nn.Tanh(),
        "elu": nn.ELU(),
    }.get(n, nn.SiLU())


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int], out_dim: int,
                 activation: Union[str, nn.Module] = "silu", dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        act = get_activation(activation)
        for h in hidden:
            layers += [nn.Linear(prev, h), act]
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))


# -------------------------
# Encoder / Decoder
# -------------------------
class Encoder(nn.Module):
    def __init__(self, state_dim: int, global_dim: int, hidden: Sequence[int], latent_dim: int,
                 activation: Union[str, nn.Module] = "silu", dropout: float = 0.0):
        super().__init__()
        self.net = MLP(state_dim + global_dim, hidden, latent_dim, activation, dropout)

    def forward(self, y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(self.net(torch.cat([y, g], dim=-1)),
                                nan=0.0, posinf=0.0, neginf=0.0)


class Decoder(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int], state_dim: int,
                 activation: Union[str, nn.Module] = "silu", dropout: float = 0.0,
                 z_std_clip: float = Z_STD_CLIP_DEFAULT):
        super().__init__()
        self.net = MLP(in_dim, hidden, state_dim, activation, dropout)
        self.z_std_clip = float(z_std_clip) if z_std_clip is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if self.z_std_clip is not None and self.z_std_clip > 0:
            x = x.clamp_(-self.z_std_clip, self.z_std_clip)
        y = self.net(x)
        return torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


# -------------------------
# Semigroup low-rank dynamics
# -------------------------
class LowRankSemigroupDynamics(nn.Module):
    """
    z' = A(g) z,  with  A(g) = U(g) diag(λ(g)) U(g)^T  +  λ⊥(g) * (I - U U^T)

    Closed-form flow for Δt_phys:
      x0 = z0 - z_eq(g)
      y0 = U^T x0,  x0⊥ = x0 - U y0
      yK = exp(Δt λ) ∘ y0
      xK⊥ = exp(Δt λ⊥) * x0⊥
      zK = z_eq + U yK + xK⊥
    """
    def __init__(
        self,
        *,
        latent_dim: int,
        global_dim: int,
        cond_hidden: Sequence[int],
        activation: Union[str, nn.Module] = "silu",
        dropout: float = 0.0,
        r: int = 16,
        learn_U: bool = False,
        basis: str = "dct",
        dt_stats: Dict[str, float],
        lambda_para_min: float = -20.0,
        lambda_para_max: float =  0.0,
        lambda_perp_min: float = -20.0,
        lambda_perp_max: float =  0.0,
        exp_clip: float | None = None,   # optional |λ Δt| clamp before exp; None = disabled
        exp_in_fp64: bool = True,        # compute exps in fp64 for wider dynamic range
    ):
        super().__init__()
        self.z = int(latent_dim)
        self.r = int(r)
        if not (1 <= self.r <= self.z):
            raise ValueError(f"r must be in [1, Z]; got r={self.r}, Z={self.z}")

        self.learn_U = bool(learn_U)
        self.use_S = False  # Trainer checks this to choose the low-rank path
        self.exp_clip = None if exp_clip is None else float(exp_clip)
        self.exp_in_fp64 = bool(exp_in_fp64)

        self.l_para_min = float(lambda_para_min)
        self.l_para_max = float(lambda_para_max)
        self.l_perp_min = float(lambda_perp_min)
        self.l_perp_max = float(lambda_perp_max)

        # Conditioners
        self.cond_eq = MLP(global_dim, cond_hidden, self.z, activation, dropout)     # z_eq(g)
        self.cond_lambda = MLP(global_dim, cond_hidden, self.r, activation, dropout) # λ(g)
        self.cond_lperp = MLP(global_dim, cond_hidden, 1,   activation, dropout)     # λ⊥(g)

        # U(g) conditioner (optional)
        if self.learn_U:
            self.cond_U = MLP(global_dim, cond_hidden, self.z * self.r, activation, dropout)
            # Initialize last layer near a fixed basis to help stability
            last = [m for m in self.cond_U.modules() if isinstance(m, nn.Linear)][-1]
            with torch.no_grad():
                last.weight.zero_()
                if last.bias is not None:
                    last.bias.zero_()
        else:
            self.cond_U = None

        # Base U for learn_U=False
        B = basis.lower()
        if B == "identity":
            Q = torch.eye(self.z, dtype=torch.float32)[:, : self.r]
        else:
            # Prefer library DCT (orthonormal) if available; fall back to classic construction + QR
            try:
                # PyTorch ≥ 2.5: type=2, norm='ortho' yields orthonormal columns
                I = torch.eye(self.z, dtype=torch.float32)
                M = torch.fft.dct(I, type=2, norm='ortho')  # [Z, Z]
                Q = M[:, : self.r].contiguous()
            except Exception:
                n = torch.arange(self.z, dtype=torch.float32).view(self.z, 1)
                k = torch.arange(self.r, dtype=torch.float32).view(1, self.r)
                M = torch.cos(math.pi * (n + 0.5) * k / float(self.z))
                M[:, 0] = M[:, 0] / math.sqrt(2.0)
                M = M * math.sqrt(2.0 / float(self.z))
                Q, _ = torch.linalg.qr(M)
                Q = Q[:, : self.r].contiguous()
        self.register_buffer("base_U", Q)                 # [Z,r]
        self.register_buffer("I", torch.eye(self.z))      # [Z,Z]

        # Δt stats (log10) for dt_norm ∈ [0,1]
        log_min = float(dt_stats["log_min"]); log_max = float(dt_stats["log_max"])
        self.register_buffer("dt_log_min", torch.tensor(log_min, dtype=torch.float32))
        self.register_buffer("dt_log_max", torch.tensor(log_max, dtype=torch.float32))
        self.register_buffer("_dt_range_scalar", (self.dt_log_max - self.dt_log_min).clamp_min(1e-9))

    # utilities
    def _compute_U(self, g: torch.Tensor) -> torch.Tensor:
        if not self.learn_U:
            return self.base_U.unsqueeze(0).expand(g.shape[0], -1, -1)  # [B,Z,r]
        raw = self.cond_U(g).view(g.shape[0], self.z, self.r).float()
        # Batched QR with sign stabilization
        Q, R = torch.linalg.qr(raw + 1e-6, mode="reduced")
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        sign = torch.sign(torch.where(diag == 0, torch.ones_like(diag), diag))
        return (Q * sign.unsqueeze(-2)).to(g.dtype)

    def _map_to_range(self, x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        # tanh -> (-1,1) → affine to [lo, hi]
        return 0.5 * (torch.tanh(x) + 1.0) * (hi - lo) + lo

    def _dt_phys_from_norm(self, dt_norm: torch.Tensor) -> torch.Tensor:
        dt_norm = dt_norm.clamp_(0.0, 1.0)
        if self.exp_in_fp64:
            dt_norm = dt_norm.to(torch.float64)
            log_dt = self.dt_log_min.to(torch.float64) + dt_norm * self._dt_range_scalar.to(torch.float64)
            return torch.exp(log_dt * LN10)  # fp64
        else:
            log_dt = self.dt_log_min + dt_norm * self._dt_range_scalar
            return torch.exp(log_dt * LN10)  # model dtype

    def _safe_exp(self, alpha: torch.Tensor) -> torch.Tensor:
        if self.exp_clip is not None and self.exp_clip > 0:
            alpha = alpha.clamp(min=-self.exp_clip, max=self.exp_clip)
        return torch.exp(alpha)

    # main propagators
    def propagate_K_lowrank(self, z0: torch.Tensor, g: torch.Tensor, dt_norm: torch.Tensor) -> torch.Tensor:
        """
        z0[B,Z], g[B,G], dt_norm[B,K]
        returns ZK[B,K,Z]
        """
        B, Z = z0.shape
        K = dt_norm.shape[1]
        z0 = torch.nan_to_num(z0, nan=0.0, posinf=0.0, neginf=0.0)
        g  = torch.nan_to_num(g,  nan=0.0, posinf=0.0, neginf=0.0)

        # Conditioners
        U = self._compute_U(g)                                              # [B,Z,r]
        z_eq = torch.nan_to_num(self.cond_eq(g), nan=0.0, posinf=0.0, neginf=0.0)  # [B,Z]
        x0 = z0 - z_eq                                                      # [B,Z]

        lam_para = self._map_to_range(self.cond_lambda(g), self.l_para_min, self.l_para_max)  # [B,r]
        lam_perp = self._map_to_range(self.cond_lperp(g), self.l_perp_min, self.l_perp_max)   # [B,1]

        dt_phys = self._dt_phys_from_norm(dt_norm).unsqueeze(-1)        # [B,K,1]

        alpha_para = dt_phys * lam_para.view(B, 1, -1).to(dt_phys.dtype)  # [B,K,r]
        alpha_perp = dt_phys * lam_perp.view(B, 1, 1).to(dt_phys.dtype)   # [B,K,1]

        lamK_para = self._safe_exp(alpha_para)                          # [B,K,r]
        lamK_perp = self._safe_exp(alpha_perp)                          # [B,K,1]

        # Decompose x0 into U and complement (einsum labels correct)
        y0  = torch.einsum('b z r, b z -> b r', U, x0)                  # [B,r]
        Uy0 = torch.einsum('b z r, b r -> b z', U, y0)                  # [B,Z]
        w0  = x0 - Uy0                                                  # [B,Z]

        # Evolve across K, reconstruct
        yK  = lamK_para.to(torch.float32) * y0.unsqueeze(1)             # [B,K,r]
        wK  = lamK_perp.to(torch.float32) * w0.view(B, 1, -1)           # [B,K,Z]
        UyK = torch.einsum('b z r, b k r -> b k z', U, yK)              # [B,K,Z]
        zK  = z_eq.unsqueeze(1) + UyK + wK                              # [B,K,Z]
        return torch.nan_to_num(zK, nan=0.0, posinf=0.0, neginf=0.0).to(z0.dtype)

    def step(self, z: torch.Tensor, dt_step_norm: torch.Tensor | float, g: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(dt_step_norm):
            dt_step_norm = torch.as_tensor(dt_step_norm, device=z.device, dtype=torch.float32)
        if dt_step_norm.ndim == 0:
            dt_step_norm = dt_step_norm.expand(z.shape[0])
        zK = self.propagate_K_lowrank(z, g, dt_step_norm.view(z.shape[0], 1))
        return zK[:, 0, :]


# -------------------------
# Full model
# -------------------------
class StableLPVKoopmanAE(nn.Module):
    """
    Name kept for back-compat with imports. Implements a semigroup latent AE.
    """
    def __init__(
        self,
        *,
        state_dim: int,
        global_dim: int,
        latent_dim: int,
        encoder_hidden: Sequence[int],
        decoder_hidden: Sequence[int],
        activation: Union[str, nn.Module] = "silu",
        dropout: float = 0.0,
        predict_delta: bool = True,
        z_std_clip: float = Z_STD_CLIP_DEFAULT,
        decoder_condition_on_g: bool = True,
        # Dynamics
        cond_hidden: Sequence[int] = (64, 64),
        rank_l: int = 16,
        learn_U: bool = False,
        basis: str = "dct",
        dt_stats: Dict[str, float] = None,
        # Eigenvalue range settings
        lambda_para_min: float = -20.0,
        lambda_para_max: float =  0.0,
        lambda_perp_min: float = -20.0,
        lambda_perp_max: float =  0.0,
        exp_clip: float | None = None,
    ):
        super().__init__()
        if dt_stats is None:
            raise ValueError("dt_stats required: {'log_min':..., 'log_max':...} (log10 of physical Δt)")

        self.state_dim = int(state_dim)
        self.global_dim = int(global_dim)
        self.latent_dim = int(latent_dim)
        self.predict_delta = bool(predict_delta)
        self.decoder_condition_on_g = bool(decoder_condition_on_g)

        self.encoder = Encoder(self.state_dim, self.global_dim, encoder_hidden, self.latent_dim, activation, dropout)
        dec_in = self.latent_dim + (self.global_dim if self.decoder_condition_on_g else 0)
        self.decoder = Decoder(dec_in, decoder_hidden, self.state_dim, activation, dropout, z_std_clip)

        self.dynamics = LowRankSemigroupDynamics(
            latent_dim=self.latent_dim,
            global_dim=self.global_dim,
            cond_hidden=cond_hidden,
            activation=activation,
            dropout=dropout,
            r=int(rank_l),
            learn_U=bool(learn_U),
            basis=str(basis),
            dt_stats=dt_stats,
            lambda_para_min=float(lambda_para_min),
            lambda_para_max=float(lambda_para_max),
            lambda_perp_min=float(lambda_perp_min),
            lambda_perp_max=float(lambda_perp_max),
            exp_clip=None if exp_clip is None else float(exp_clip),
        )

        # back-compat setter used by some tooling
        self.set_dt_log_stats = self._set_dt_log_stats

    # dt stats setter (rarely needed if manifest is correct)
    def _set_dt_log_stats(self, log_min: float, log_max: float) -> None:
        d = self.dynamics
        d.dt_log_min = torch.tensor(float(log_min), dtype=torch.float32, device=d.dt_log_min.device)
        d.dt_log_max = torch.tensor(float(log_max), dtype=torch.float32, device=d.dt_log_max.device)
        d._dt_range_scalar = (d.dt_log_max - d.dt_log_min).clamp_min(1e-9)

    # core API
    def forward(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if dt_norm.ndim == 3 and dt_norm.shape[-1] == 1:
            dt_norm = dt_norm.squeeze(-1)
        if dt_norm.ndim == 1:
            dt_norm = dt_norm.view(y_i.shape[0], 1)

        y_i = torch.nan_to_num(y_i, nan=0.0, posinf=0.0, neginf=0.0)
        g   = torch.nan_to_num(g,   nan=0.0, posinf=0.0, neginf=0.0)

        z0 = self.encoder(y_i, g)  # [B,Z]
        ZK = self.dynamics.propagate_K_lowrank(z0, g, dt_norm)  # [B,K,Z]

        B, K, _ = ZK.shape
        if self.decoder_condition_on_g:
            gK = g.unsqueeze(1).expand(B, K, -1)
            dec_in = torch.cat([ZK, gK], dim=-1).reshape(B * K, -1)
        else:
            dec_in = ZK.reshape(B * K, -1)
        dec_out = self.decoder(dec_in).reshape(B, K, -1)

        if self.predict_delta:
            y_out = y_i.unsqueeze(1) + torch.cumsum(dec_out, dim=1)
        else:
            y_out = dec_out
        return torch.nan_to_num(y_out, nan=0.0, posinf=0.0, neginf=0.0)

    def step(self, y: torch.Tensor, dt_step_norm: torch.Tensor | float, g: torch.Tensor) -> torch.Tensor:
        z = self.encoder(y, g)
        z_next = self.dynamics.step(z, dt_step_norm, g)
        dec_in = torch.cat([z_next, g], dim=-1) if self.decoder_condition_on_g else z_next
        y_next = self.decoder(dec_in)
        return torch.nan_to_num(y + y_next if self.predict_delta else y_next,
                                nan=0.0, posinf=0.0, neginf=0.0)

    @torch.no_grad()
    def rollout(self, y0: torch.Tensor, g: torch.Tensor, dt_step_norm: torch.Tensor | float, steps: int) -> torch.Tensor:
        B = y0.shape[0]
        if not torch.is_tensor(dt_step_norm):
            dt_step_norm = torch.as_tensor(dt_step_norm, device=y0.device, dtype=torch.float32)
        if dt_step_norm.ndim == 0:
            dt_step_norm = dt_step_norm.expand(B)

        z0 = self.encoder(y0, g)
        ZK = self.dynamics.propagate_K_lowrank(z0, g, dt_step_norm.unsqueeze(1).expand(B, steps))

        K = ZK.shape[1]
        if self.decoder_condition_on_g:
            gK = g.unsqueeze(1).expand(B, K, -1)
            dec_in = torch.cat([ZK, gK], dim=-1).reshape(B * K, -1)
        else:
            dec_in = ZK.reshape(B * K, -1)
        dec_out = self.decoder(dec_in).reshape(B, K, -1)

        if self.predict_delta:
            y_out = y0.unsqueeze(1) + torch.cumsum(dec_out, dim=1)
        else:
            y_out = dec_out
        return torch.nan_to_num(y_out, nan=0.0, posinf=0.0, neginf=0.0)


# -------------------------
# Factory + manifest helpers
# -------------------------
def _load_manifest_or_exit(cfg: dict) -> dict:
    paths = cfg.get("paths") or {}
    proc_dir = paths.get("processed_data_dir") or "data/processed"
    path = Path(proc_dir) / "normalization.json"
    try:
        manifest = json.loads(path.read_text())
    except Exception as e:
        warnings.warn(f"[normalization.json] not found or unreadable at {path}: {e}", RuntimeWarning)
        sys.exit(2)

    dt = manifest.get("dt") or {}
    if not isinstance(dt, dict) or ("log_min" not in dt) or ("log_max" not in dt):
        warnings.warn(f"[normalization.json] missing dt.log_min/log_max at {path}. Exiting.", RuntimeWarning)
        sys.exit(2)
    return manifest


def create_model(arg1, cfg: dict | None = None) -> StableLPVKoopmanAE:
    """
    Usage:
      - create_model(cfg)
      - create_model(manifest_path, cfg)  # deprecated path form retained for back-compat
    """
    if cfg is None:
        cfg = arg1
        manifest = _load_manifest_or_exit(cfg)
    else:
        # If a manifest path was passed directly, prefer cfg for dt checks
        manifest = _load_manifest_or_exit(cfg)

    data_cfg = cfg.get("data") or {}
    species = list(data_cfg.get("species_variables") or [])
    globals_ = list(data_cfg.get("global_variables") or [])
    S_out = len(species) if species else int((manifest.get("meta") or {}).get("num_species", 0))
    G_in = len(globals_) if globals_ else int((manifest.get("meta") or {}).get("num_globals", 0))
    if S_out <= 0:
        raise ValueError("Bad dimensions: config.data.species_variables must be set or manifest meta must specify num_species.")
    if G_in < 0:
        raise ValueError("Bad global_dim resolved from config/manifest.")

    dt_stats = {"log_min": float(manifest["dt"]["log_min"]), "log_max": float(manifest["dt"]["log_max"])}

    m = cfg.get("model") or {}
    return StableLPVKoopmanAE(
        state_dim=S_out,
        global_dim=G_in,
        latent_dim=int(m.get("latent_dim", 64)),
        encoder_hidden=list(m.get("encoder_hidden", [256, 256, 256])),
        decoder_hidden=list(m.get("decoder_hidden", [256, 256, 256])),
        activation=str(m.get("activation", "silu")),
        dropout=float(m.get("dropout", 0.0)),
        predict_delta=bool(m.get("predict_delta", True)),
        z_std_clip=float(m.get("z_std_clip", Z_STD_CLIP_DEFAULT)),
        decoder_condition_on_g=bool(m.get("decoder_condition_on_g", True)),

        # Dynamics
        cond_hidden=list(m.get("cond_hidden", [64, 64])),
        rank_l=int(m.get("rank_l", 16)),
        learn_U=bool(m.get("learn_U", False)),
        basis=str(m.get("basis", "dct")),
        dt_stats=dt_stats,

        # Eigenvalue ranges (defaults safe: non-positive)
        lambda_para_min=float(m.get("lambda_para_min", -20.0)),
        lambda_para_max=float(m.get("lambda_para_max",  0.0)),
        lambda_perp_min=float(m.get("lambda_perp_min", -20.0)),
        lambda_perp_max=float(m.get("lambda_perp_max",  0.0)),
        exp_clip=(None if m.get("exp_clip", None) is None else float(m["exp_clip"])),
    )


# Back-compat alias for any external references
FlowMapKoopman = StableLPVKoopmanAE

__all__ = [
    "MLP", "Encoder", "Decoder",
    "LowRankSemigroupDynamics", "StableLPVKoopmanAE",
    "create_model", "FlowMapKoopman",
]

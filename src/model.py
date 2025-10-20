#!/usr/bin/env python3
"""
Stable LPV Koopman Autoencoder (numerically safe, train-stable)
===============================================================

Design goals
------------
- **Training uses matrix_exp** (no eigen backprop at step 0).
- **Eval can use eigh** when `use_S=False` for speed.
- **Exponent safety**: Δt is capped *from γ* ⇒ |λ·Δt| ≤ LAMDT_MAX_ABS (default 50).
- **Bounded low‑rank damping**: s = s_bound * sigmoid(raw) ∈ [0, s_bound].
- **Optional Frobenius cap** on L to limit ||L Lᵀ|| relative to γ.
- **Predict‑delta is anchor‑relative** (no cumsum): ŷ = y_i + Δy.
- **No spectral_norm by default** to avoid step‑0 NaNs; can be enabled later.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------- Constants / Utils -------------------------------

LOG_STD_MIN = 1e-9
Z_STD_CLIP_DEFAULT = 50.0
TEMP_INIT_DEFAULT = 1.0
TEMP_MIN_DEFAULT = 0.1
TEMP_MAX_DEFAULT = 10.0

# Hard bound for exponent argument so exp(±x) stays finite/stable
LAMDT_MAX_ABS = 50.0  # exp(±50) ~ [1.9e-22, 3.0e21]

# Clamp for latent magnitude before decoding (protect Linear matmul)
Z_CLAMP = 100.0

LN10 = math.log(10.0)


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


# ----------------------------- Orthonormal basis ------------------------------

def _orthonormal_dct2(n: int) -> torch.Tensor:
    """Return n×n orthonormal DCT‑II matrix (float32)."""
    C = torch.empty(n, n, dtype=torch.float32)
    scale0 = math.sqrt(1.0 / n)
    scale = math.sqrt(2.0 / n)
    ns = torch.arange(n, dtype=torch.float32).view(1, n)
    for k in range(n):
        s = scale0 if k == 0 else scale
        C[k] = s * torch.cos(math.pi * (ns + 0.5) * (k / n))
    return C.t().contiguous()  # [n,n]


# --------------------------------- MLP ---------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int], out_dim: int,
                 activation: Union[str, nn.Module, Callable[[], nn.Module]],
                 dropout: float = 0.0, *, spectral_norm: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        dprev = int(in_dim)
        make_act = _act_factory(activation)
        for i, d in enumerate(list(hidden) + [int(out_dim)]):
            lin = nn.Linear(dprev, d)
            # Disabled by default; can be enabled from config later
            if spectral_norm:
                lin = nn.utils.spectral_norm(lin)
            layers.append(lin)
            if i < len(hidden):
                layers.append(make_act())
                if dropout and dropout > 0:
                    layers.append(nn.Dropout(dropout))
            dprev = d
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            tmin, tmax = float(temp_range[0]), float(temp_range[1])
            self.tmin, self.tmax = tmin, tmax
            if self.learn_temp:
                self.temp = nn.Parameter(torch.tensor(float(temp_init)))
            else:
                self.register_buffer("temp_fixed", torch.tensor(float(temp_init)))

        # small-normal on final linear
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
        x = torch.nan_to_num(logits / t)
        log10_p = F.log_softmax(x, dim=-1) * (1.0 / LN10)
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
    - L uses bounded singular values via sigmoid (s ∈ [0, s_bound])
    - Optional Frobenius cap on L to keep ||LLᵀ|| in check relative to γ
    """
    def __init__(self, latent_dim: int, global_dim: int,
                 cond_hidden: Sequence[int], rank_l: int, use_S: bool, gamma: float,
                 activation: Union[str, nn.Module, Callable[[], nn.Module]], dropout: float,
                 dt_stats: Dict[str, float], *,
                 s_bound: float = 2.0,
                 L_fro_factor: float = 0.5,
                 spectral_norm_cond: bool = False,
                 basis: str = "dct",
                 stability_projection_eps: Optional[float] = None):
        super().__init__()
        self.z = int(latent_dim)
        self.g = int(global_dim)
        self.r = int(rank_l)
        self.use_S = bool(use_S)
        self.gamma = float(gamma)
        self.s_bound = float(s_bound)
        self.L_fro_factor = float(L_fro_factor)
        self.stability_projection_eps = stability_projection_eps

        self.cond = MLP(self.g, cond_hidden, self.r + (self.z * self.z if self.use_S else 0),
                        activation, dropout, spectral_norm=spectral_norm_cond)

        # Orthonormal basis for L (prefer DCT low-frequency columns)
        with torch.no_grad():
            if basis.lower() == "dct":
                B = _orthonormal_dct2(self.z)
                Q = B[:, : self.r].contiguous()
            elif basis.lower() == "identity":
                Q = torch.eye(self.z, dtype=torch.float32)[:, : self.r]
            else:
                Q, _ = torch.linalg.qr(torch.randn(self.z, self.r))
        self.register_buffer("base_U", Q)                 # [z,r]
        self.register_buffer("I", torch.eye(self.z))      # [z,z]
        self.timewarp = _NoOpTimeWarp(0.0)

        # Δt log range (log10 space)
        if ("log_min" not in dt_stats) or ("log_max" not in dt_stats):
            raise ValueError("dt_stats must include 'log_min' and 'log_max' (log10 space).")
        log_min = float(dt_stats["log_min"])
        log_max = float(dt_stats["log_max"])
        if not math.isfinite(log_min) or not math.isfinite(log_max) or (log_max <= log_min):
            raise ValueError(f"Bad dt_stats: log_min={log_min}, log_max={log_max}")
        self.register_buffer("dt_log_min", torch.tensor(log_min, dtype=torch.float32))
        self.register_buffer("dt_log_max", torch.tensor(log_max, dtype=torch.float32))

        # Neutral skew at init
        self.scale_S = nn.Parameter(torch.tensor(0.0))

        # Init cond head so s ~ tiny at start (near-identity dynamics)
        with torch.no_grad():
            last_lin = None
            for m in self.cond.net.modules():
                if isinstance(m, nn.Linear):
                    last_lin = m
            if last_lin is not None:
                nn.init.zeros_(last_lin.weight)
                nn.init.zeros_(last_lin.bias)
                tiny = 1e-3
                y = max(min(tiny / max(self.s_bound, 1e-6), 1 - 1e-6), 1e-6)  # tiny/s_bound in (0,1)
                bias_init = math.log(y / (1.0 - y))
                last_lin.bias.data[: self.r].fill_(bias_init)

    @property
    def _dt_range_scalar(self) -> torch.Tensor:
        return (self.dt_log_max - self.dt_log_min).clamp_min(1e-12)

    def _build_A(self, pt: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B = pt.shape[0]
        pt = torch.nan_to_num(pt, nan=0.0, posinf=0.0, neginf=0.0)

        h = self.cond(pt)
        if self.use_S:
            s_raw, m_head = torch.split(h, [self.r, self.z * self.z], dim=-1)
        else:
            s_raw, m_head = h, None

        s = self.s_bound * torch.sigmoid(s_raw)  # [B,r] ∈ [0,s_bound]

        U = self.base_U  # [z,r]
        L = U.unsqueeze(0) * s.unsqueeze(1)  # [B,z,r]

        # Optional Frobenius cap: ||L||_F ≤ gamma * z * L_fro_factor
        if self.L_fro_factor > 0:
            with torch.no_grad():
                L_fro = torch.linalg.vector_norm(L, ord=2, dim=(1, 2))  # Frobenius
                cap = max(self.gamma, 1e-6) * self.z * self.L_fro_factor
                scale = torch.clamp(cap / (L_fro + 1e-12), max=1.0)
            L = L * scale.view(B, 1, 1)

        LLt = torch.matmul(L, L.transpose(1, 2))  # [B,z,z] PSD

        if self.use_S:
            M = m_head.reshape(B, self.z, self.z)
            S = 0.5 * (M - M.transpose(1, 2))
            S = torch.tanh(self.scale_S) * S
        else:
            S = torch.zeros(B, self.z, self.z, device=pt.device, dtype=pt.dtype)

        A = S - LLt - self.gamma * self.I.to(device=pt.device, dtype=pt.dtype).expand(B, -1, -1)
        A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

        if (not self.use_S) and (self.stability_projection_eps is not None):
            Asym = 0.5 * (A + A.transpose(1, 2))
            eigvals = torch.linalg.eigvalsh(Asym)
            max_eig = eigvals.max(dim=1).values
            mask = max_eig > -float(self.stability_projection_eps)
            if mask.any():
                shift = (max_eig[mask] + float(self.stability_projection_eps))
                A[mask] = A[mask] - shift.view(-1, 1, 1) * self.I.to(A)

        return A, {}

    def _denorm_dt(self, dt_norm: torch.Tensor) -> torch.Tensor:
        """[B] or [B,K] normalized → physical seconds, capped by γ to keep |λ·Δt| ≤ LAMDT_MAX_ABS."""
        dt_norm = torch.nan_to_num(dt_norm, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        log_dt = self.dt_log_min + dt_norm.to(torch.float64) * self._dt_range_scalar.to(torch.float64)
        dt_cap = LAMDT_MAX_ABS / max(self.gamma, 1e-12)  # seconds
        log_cap_gamma = math.log10(dt_cap)
        log_cap_manifest = float(self.dt_log_max.item())
        log_dt = log_dt.clamp(max=min(log_cap_gamma, log_cap_manifest))
        dt_phys = torch.exp(log_dt * LN10).clamp_(min=1e-30)
        return dt_phys.to(torch.float32)

    def step(self, z: torch.Tensor, dt_step_norm: torch.Tensor | float, pt: torch.Tensor) -> torch.Tensor:
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
        B, Z = z.shape
        if dt_phys.ndim != 1 or dt_phys.shape[0] != B:
            raise ValueError(f"dt_phys must be [B]; got {tuple(dt_phys.shape)} vs B={B}")

        with torch.amp.autocast("cuda", enabled=False):
            A, _ = self._build_A(pt)
            if (not self.use_S) and (not self.training) and (not torch.is_grad_enabled()):
                # Fast path for eval
                A32 = 0.5 * (A.float() + A.float().transpose(1, 2))
                lam, U = torch.linalg.eigh(A32)
                v = torch.matmul(U.transpose(1, 2), z.float().unsqueeze(-1)).squeeze(-1)
                lamdt = (lam * dt_phys.float().view(B, 1)).clamp_(-LAMDT_MAX_ABS, LAMDT_MAX_ABS)
                z_next32 = torch.matmul(U, (torch.exp(lamdt) * v).unsqueeze(-1)).squeeze(-1)
            else:
                # Generic path in train (or with skew): matrix_exp in fp32
                A32 = A.float()
                dt32 = torch.nan_to_num(dt_phys.float(), nan=1e-30, posinf=1e38, neginf=1e-30).clamp_min(0.0).view(B, 1, 1)
                Phi = torch.matrix_exp(A32 * dt32)
                z_next32 = torch.bmm(Phi, z.float().unsqueeze(-1)).squeeze(-1)

        z_next32 = torch.nan_to_num(z_next32, nan=0.0, posinf=0.0, neginf=0.0)
        return z_next32.to(z.dtype)


# ------------------------------ Full Model -----------------------------------

class StableLPVKoopmanAE(nn.Module):
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
        target_log_mean: Optional[Sequence[float]] = None,
        target_log_std: Optional[Sequence[float]] = None,
        cond_hidden: Sequence[int] = (64, 64),
        rank_l: int = 8,
        use_S: bool = False,
        gamma: float = 1e-2,
        dt_stats: Dict[str, float] = None,
        # extras
        s_bound: float = 2.0,
        L_fro_factor: float = 0.5,
        spectral_norm_cond: bool = False,
        basis: str = "dct",
        stability_projection_eps: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
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
            s_bound=s_bound,
            L_fro_factor=L_fro_factor,
            spectral_norm_cond=spectral_norm_cond,
            basis=basis,
            stability_projection_eps=stability_projection_eps,
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

    def forward(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
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

        y_i = torch.nan_to_num(y_i, nan=0.0, posinf=0.0, neginf=0.0)
        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

        z0 = self.encoder(y_i, g).clamp_(-Z_CLAMP, Z_CLAMP)

        A, _ = self.dynamics._build_A(g)
        dt_phys32 = self.dynamics._denorm_dt(dt_norm).float()

        try:
            if (not self.use_S) and (not self.training) and (not torch.is_grad_enabled()):
                with torch.amp.autocast("cuda", enabled=False):
                    A32 = 0.5 * (A.float() + A.float().transpose(1, 2))
                    lam, U = torch.linalg.eigh(A32)
                    v = torch.matmul(U.transpose(1, 2), z0.float().unsqueeze(-1)).squeeze(-1)
                    lamdt = (lam.unsqueeze(1) * dt_phys32.unsqueeze(-1)).clamp_(-LAMDT_MAX_ABS, LAMDT_MAX_ABS)
                    z_eig_next = v.unsqueeze(1) * torch.exp(lamdt)
                    z_next32 = torch.einsum('bij,bkj->bki', U, z_eig_next)
            else:
                with torch.amp.autocast("cuda", enabled=False):
                    A32 = A.float().unsqueeze(1)
                    Phi32 = torch.matrix_exp(A32 * dt_phys32.reshape(B, K, 1, 1))
                    z0_32 = z0.float().reshape(B, 1, self.Z, 1)
                    z_next32 = torch.matmul(Phi32, z0_32).squeeze(-1)
        except RuntimeError:
            with torch.amp.autocast("cuda", enabled=False):
                A32 = A.float().unsqueeze(1)
                Phi32 = torch.matrix_exp(A32 * dt_phys32.reshape(B, K, 1, 1))
                z0_32 = z0.float().reshape(B, 1, self.Z, 1)
                z_next32 = torch.matmul(Phi32, z0_32).squeeze(-1)

        z_next32 = torch.nan_to_num(z_next32, nan=0.0, posinf=0.0, neginf=0.0)
        z_next = z_next32.to(z0.dtype).clamp_(-Z_CLAMP, Z_CLAMP)

        y_out = self.decoder(z_next.reshape(B * K, self.Z)).reshape(B, K, S)
        if self.predict_delta:
            y_out = y_i.unsqueeze(1) + y_out
        return y_out

    def step(self, y: torch.Tensor, dt_step_norm: torch.Tensor | float, g: torch.Tensor) -> torch.Tensor:
        z = self.encoder(y, g)
        z_next = self.dynamics.step(z, dt_step_norm, g)
        y_next = self.decoder(z_next)
        if self.predict_delta:
            y_next = y + y_next
        return y_next

    @torch.no_grad()
    def rollout(self, y0: torch.Tensor, g: torch.Tensor, dt_step_norm: torch.Tensor | float, steps: int) -> torch.Tensor:
        B, _ = y0.shape
        y_t = y0
        out = []
        for _ in range(int(steps)):
            y_t = self.step(y_t, dt_step_norm, g)
            out.append(y_t)
        return torch.stack(out, dim=1)


# --------------------- Manifest helpers + model factory -----------------------

def _load_manifest(config: dict) -> dict:
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
    if "dt" in manifest:
        st = manifest["dt"]
        if "log_min" in st and "log_max" in st:
            return {"log_min": float(st["log_min"]), "log_max": float(st["log_max"])}
    if "per_key_stats" in manifest and "dt" in manifest["per_key_stats"]:
        st = manifest["per_key_stats"]["dt"]
        if "log_min" in st and "log_max" in st:
            return {"log_min": float(st["log_min"]), "log_max": float(st["log_max"])}
    if "dt" in stats:
        st = stats["dt"]
        if "log_min" in st and "log_max" in st:
            return {"log_min": float(st["log_min"]), "log_max": float(st["log_max"])}
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
    manifest = _load_manifest(config)

    stats = manifest.get("per_key_stats") or manifest.get("stats", {})
    methods = manifest.get("normalization_methods") or manifest.get("methods", {})
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

    mcfg = config.get("model", {}) or {}
    softmax_head = bool(mcfg.get("softmax_head", True))

    if softmax_head:
        log_mean, log_std = _gather_species_stats(methods, stats, species_vars)
        target_log_mean = log_mean.tolist()
        target_log_std = log_std.tolist()
    else:
        target_log_mean = target_log_std = None

    dt_stats = _get_dt_stats(manifest, stats)

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

    s_bound = float(mcfg.get("s_bound", 2.0))
    L_fro_factor = float(mcfg.get("L_fro_factor", 0.5))
    spectral_norm_cond = bool(mcfg.get("spectral_norm_cond", False))
    basis = str(mcfg.get("basis", "dct"))
    stability_projection_eps = mcfg.get("stability_projection_eps", None)
    if stability_projection_eps is not None:
        stability_projection_eps = float(stability_projection_eps)

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
        s_bound=s_bound,
        L_fro_factor=L_fro_factor,
        spectral_norm_cond=spectral_norm_cond,
        basis=basis,
        stability_projection_eps=stability_projection_eps,
    )
    return model


FlowMapKoopman = StableLPVKoopmanAE

__all__ = [
    "MLP", "Encoder", "Decoder",
    "LPVDynamics", "StableLPVKoopmanAE",
    "create_model", "FlowMapKoopman",
]
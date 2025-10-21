#!/usr/bin/env python3
"""
Stable LPV Koopman Autoencoder (trainer-compatible, numerically hardened)
=======================================================================

- use_S=False → closed-form propagation (no matrix_exp anywhere in model paths)
- Frobenius cap rescales s (and rebuilds L), so A and (U,s) are consistent
- Batched QR + sign-stabilization for learn_U=True
- No wasted A-build in forward/rollout when use_S=False
- step() now uses _build_info() when use_S=False (skips A bmm)
- Added propagate_with_info(...) so the trainer can avoid matrix_exp too
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

# ==============================
# Global numerics
# ==============================
DT_PHYS_MAX_GLOBAL: Optional[float] = None
LAMDT_MAX_ABS: float = 60.0
LN10 = math.log(10.0)

Z_STD_CLIP_DEFAULT: Optional[float] = 10.0
TEMP_INIT_DEFAULT = 1.0
TEMP_MIN_DEFAULT = 0.25
TEMP_MAX_DEFAULT = 4.0


# ==============================
# Small helpers
# ==============================
def get_activation(name: Union[str, nn.Module, Callable[[], nn.Module]]) -> nn.Module:
    if isinstance(name, nn.Module):
        return name
    if callable(name):
        return name()
    n = str(name).lower()
    return {
        "relu": nn.ReLU(inplace=False),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "swish": nn.SiLU(),
        "tanh": nn.Tanh(),
        "elu": nn.ELU(),
    }.get(n, nn.SiLU())


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: Sequence[int],
        out_dim: int,
        activation: Union[str, nn.Module, Callable[[], nn.Module]] = "silu",
        dropout: float = 0.0,
        *,
        spectral_norm: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        act = get_activation(activation)
        for h in hidden:
            lin = nn.Linear(prev, h)
            if spectral_norm:
                lin = nn.utils.parametrizations.spectral_norm(lin)
            layers += [lin, act]
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return self.net(x)


class Encoder(nn.Module):
    def __init__(
        self,
        state_dim: int,
        global_dim: int,
        hidden: Sequence[int],
        latent_dim: int,
        activation: Union[str, nn.Module, Callable[[], nn.Module]] = "silu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.global_dim = int(global_dim)
        self.latent_dim = int(latent_dim)
        self.net = MLP(self.state_dim + self.global_dim, hidden, self.latent_dim, activation, dropout)

    def forward(self, y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        z = self.net(torch.cat([y, g], dim=-1))
        return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


class Decoder(nn.Module):
    def __init__(self,
                 in_dim: int, hidden: Sequence[int], state_dim: int,
                 activation: Union[str, nn.Module, Callable[[], nn.Module]] = "silu",
                 dropout: float = 0.0,
                 *,
                 softmax_head: bool = False,
                 log_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                 z_std_clip: Optional[float] = Z_STD_CLIP_DEFAULT,
                 temp_init: float = TEMP_INIT_DEFAULT,
                 temp_range: Tuple[float, float] = (TEMP_MIN_DEFAULT, TEMP_MAX_DEFAULT),
                 learn_temp: bool = True):
        super().__init__()
        self.net = MLP(in_dim, hidden, state_dim, activation, dropout)
        self.softmax_head = bool(softmax_head)
        self.state_dim = int(state_dim)
        self.z_std_clip = None if (z_std_clip is None) else float(z_std_clip)

        if self.softmax_head:
            if log_stats is None:
                raise ValueError("softmax_head=True requires (log_mean, log_std).")
            lm, ls = log_stats
            self.register_buffer("log_mean", lm.clone())
            self.register_buffer("log_std", ls.clone())
            self.learn_temp = bool(learn_temp)
            t0 = float(max(min(temp_init, temp_range[1]), temp_range[0]))
            self.temp_min, self.temp_max = map(float, temp_range)
            self._log_temp = nn.Parameter(torch.tensor(math.log(t0))) if self.learn_temp else None

    @property
    def temperature(self) -> float:
        if (not self.softmax_head) or (self._log_temp is None):
            return 1.0
        t = float(torch.exp(self._log_temp).item())
        return max(min(t, self.temp_max), self.temp_min)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if self.z_std_clip is not None and self.z_std_clip > 0:
            x = x.clamp_(-self.z_std_clip, self.z_std_clip)
        y = self.net(x)

        if not self.softmax_head:
            return torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # Standardize → softmax → log → un-standardize (if you enable this head, re-validate targets/loss)
        T = self.temperature
        logits = (y - self.log_mean) / self.log_std.clamp_min(1e-8)
        logits = logits / max(T, 1e-8)
        probs = torch.softmax(logits, dim=-1).clamp_min(1e-30)
        logy = torch.log(probs)
        logy = logy * self.log_std + self.log_mean
        return torch.nan_to_num(logy, nan=0.0, posinf=0.0, neginf=0.0)


# ==============================
# LPV Dynamics
# ==============================
class LPVDynamics(nn.Module):
    """
    z' = A(g) z,  A(g) = S(g) - L(g)L(g)^T - γ I,  L = U(g) diag(s(g)),  rank(U)=r
    - use_S=False → symmetric negative semidefinite + shift (closed-form propagation available)
    - use_S=True  → optional skew added (matrix_exp / eig fast-path retained)
    """
    def __init__(
        self,
        *,
        latent_dim: int,
        cond_hidden: Sequence[int],
        global_dim: int,
        activation: Union[str, nn.Module, Callable[[], nn.Module]] = "silu",
        dropout: float = 0.0,
        r: int = 16,
        s_bound: float = 2.0,
        gamma: float = 0.1,
        use_S: bool = False,
        spectral_norm_cond: bool = False,
        decoder_condition_on_g: bool = True,
        cap_via_gamma: bool = True,
        learn_U: bool = False,
        U_method: str = "qr",
        basis: str = "dct",
        stability_projection_eps: Optional[float] = None,
        dt_stats: Dict[str, float] = None,
        L_fro_factor: float = 0.5,
        train_eigh_fastpath: bool = False,
    ):
        super().__init__()
        self.z = int(latent_dim)
        self.r = int(r)
        self.global_dim = int(global_dim)
        self.gamma = float(gamma)
        self.use_S = bool(use_S)
        self.decoder_condition_on_g = bool(decoder_condition_on_g)
        self.cap_via_gamma = bool(cap_via_gamma)
        self.learn_U = bool(learn_U)
        self.U_method = str(U_method).lower()
        self.s_bound = float(s_bound)
        self.L_fro_factor = float(L_fro_factor)
        self.train_eigh_fastpath = bool(train_eigh_fastpath)

        # Conditioner for s (and optional S tail)
        out_dim = self.r + (self.z * self.z if self.use_S else 0)
        self.cond = MLP(self.global_dim, cond_hidden, out_dim, activation, dropout,
                        spectral_norm=spectral_norm_cond)

        # Optional U(g) conditioner
        if self.learn_U:
            self.cond_U = MLP(self.global_dim, cond_hidden, self.z * self.r, activation, dropout,
                               spectral_norm=spectral_norm_cond)
        else:
            self.cond_U = None

        # Base orthonormal U for learn_U=False
        if basis.lower() == "dct":
            n = torch.arange(self.z, dtype=torch.float32).view(self.z, 1)
            k = torch.arange(self.r, dtype=torch.float32).view(1, self.r)
            B = torch.cos(math.pi * (n + 0.5) * k / float(self.z))
            B[:, 0] = B[:, 0] / math.sqrt(2.0)
            B = B * math.sqrt(2.0 / float(self.z))
            Q, _ = torch.linalg.qr(B)
            Q = Q[:, : self.r].contiguous()
        elif basis.lower() == "identity":
            Q = torch.eye(self.z, dtype=torch.float32)[:, : self.r]
        else:
            Q, _ = torch.linalg.qr(torch.randn(self.z, self.r))
            Q = Q[:, : self.r]
        self.register_buffer("base_U", Q)         # [z,r]
        self.register_buffer("I", torch.eye(self.z))

        # Δt stats (log10 space) are required
        if not (isinstance(dt_stats, dict) and "log_min" in dt_stats and "log_max" in dt_stats):
            raise ValueError("dt_stats must contain 'log_min' and 'log_max' (log10).")
        log_min = float(dt_stats["log_min"]); log_max = float(dt_stats["log_max"])
        if not (math.isfinite(log_min) and math.isfinite(log_max) and log_max > log_min):
            raise ValueError(f"Bad dt_stats: log_min={log_min}, log_max={log_max}")
        self.register_buffer("dt_log_min", torch.tensor(log_min, dtype=torch.float32))
        self.register_buffer("dt_log_max", torch.tensor(log_max, dtype=torch.float32))
        self.register_buffer("_dt_range_scalar", (self.dt_log_max - self.dt_log_min).clamp_min(1e-9))

        self.scale_S = nn.Parameter(torch.tensor(0.0))  # skew scale (safe if use_S=False)

    # ---------- utilities ----------
    def _compute_U(self, pt: torch.Tensor) -> torch.Tensor:
        """Return [B,z,r] with orthonormal columns. If learn_U, batched QR + sign stabilization."""
        B = pt.shape[0]
        if self.learn_U:
            raw = self.cond_U(pt).view(B, self.z, self.r).float()  # [B,z,r]
            Q, R = torch.linalg.qr(raw, mode="reduced")            # batched
            diag = torch.diagonal(R, dim1=-2, dim2=-1)
            sign = torch.sign(torch.where(diag == 0, torch.ones_like(diag), diag))
            Q = Q * sign.unsqueeze(-2)
            return Q[:, :, : self.r].to(pt.dtype)
        else:
            return self.base_U.unsqueeze(0).expand(B, -1, -1)

    def _denorm_dt(self, dt_norm: torch.Tensor) -> torch.Tensor:
        """[B] or [B,K] → physical seconds, capped by manifest + gamma-based cap."""
        dt_norm = torch.nan_to_num(dt_norm, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        log_dt = self.dt_log_min + dt_norm.to(torch.float64) * self._dt_range_scalar.to(torch.float64)

        max_cap = float(self.dt_log_max.item())
        if self.cap_via_gamma:
            dt_cap = LAMDT_MAX_ABS / max(self.gamma, 1e-12)
            max_cap = min(max_cap, math.log10(dt_cap))
        if DT_PHYS_MAX_GLOBAL is not None:
            max_cap = min(max_cap, math.log10(float(DT_PHYS_MAX_GLOBAL)))

        log_dt = log_dt.clamp(max=max_cap)
        dt_phys = torch.exp(log_dt * LN10).clamp_(min=1e-30)
        return dt_phys.to(torch.float32)

    # ---------- one-pass conditioner splits ----------
    def _split_h_to_US(self, pt: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given conditioner output h (computed once!), produce U, s, L with Frobenius cap applied.
        Returns U[B,z,r], s[B,r], L[B,z,r].
        """
        s_raw = h[:, : self.r]
        s = torch.sigmoid(s_raw).mul(float(self.s_bound))              # [B,r]
        U = self._compute_U(pt).float()                                # [B,z,r]
        L = U * s.unsqueeze(1)                                         # [B,z,r]

        if self.L_fro_factor > 0.0:
            fro = torch.linalg.norm(L, dim=(1, 2))                     # [B]
            cap = self.L_fro_factor * max(self.gamma, 1e-6) * math.sqrt(float(self.z * self.r))
            scale = torch.clamp(cap / torch.clamp(fro, min=1e-9), max=1.0)  # [B]
            s = s * scale.view(-1, 1)
            L = U * s.unsqueeze(1)
        return U, s, L

    # ---------- info/A builders ----------
    def _build_info(self, pt: torch.Tensor) -> dict:
        """Compute and return {'U','s'} without constructing A (use in model paths)."""
        with torch.amp.autocast("cuda", enabled=False):
            h = self.cond(torch.nan_to_num(pt, nan=0.0, posinf=0.0, neginf=0.0)).float()
            U, s, _L = self._split_h_to_US(pt, h)
        return {"U": U, "s": s}

    def _build_A(self, pt: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Build A once. Single conditioner call (no dropout inconsistency)."""
        B = pt.shape[0]
        with torch.amp.autocast("cuda", enabled=False):
            h = self.cond(torch.nan_to_num(pt, nan=0.0, posinf=0.0, neginf=0.0)).float()
            U, s, L = self._split_h_to_US(pt, h)

            A = -torch.bmm(L, L.transpose(1, 2))                      # [B,z,z]
            A = A - float(self.gamma) * self.I.to(A)
            info = {"U": U, "s": s} if not self.use_S else {}

            if self.use_S:
                M_raw = h[:, self.r:].reshape(B, self.z, self.z)      # same h (no second cond call)
                S = 0.5 * (M_raw - M_raw.transpose(1, 2))
                A = A + torch.tanh(self.scale_S) * S

        return A, info

    # ---------- closed-form step (use_S=False) ----------
    def _propagate_lowrank(self, z: torch.Tensor, dt_phys: torch.Tensor,
                           U: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        exp(AΔt)z for A = −U diag(s)^2 Uᵀ − γI.
        Shapes: z[B,z], dt_phys[B], U[B,z,r], s[B,r].
        """
        B, _Z = z.shape
        if dt_phys.ndim != 1 or dt_phys.shape[0] != B:
            raise ValueError(f"dt_phys must be [B]; got {tuple(dt_phys.shape)} vs B={B}")
        with torch.amp.autocast("cuda", enabled=False):
            z32 = torch.nan_to_num(z.float(), nan=0.0, posinf=0.0, neginf=0.0)
            U32 = torch.nan_to_num(U.float(), nan=0.0, posinf=0.0, neginf=0.0)
            s32 = torch.nan_to_num(s.float(), nan=0.0, posinf=0.0, neginf=0.0)

            uTz = torch.bmm(U32.transpose(1, 2), z32.unsqueeze(-1)).squeeze(-1)  # [B,r]
            proj = torch.bmm(U32, uTz.unsqueeze(-1)).squeeze(-1)                  # [B,z]

            dt = torch.nan_to_num(dt_phys.float(), nan=1e-30, posinf=1e38, neginf=1e-30).clamp_min(0.0)
            lam_g = torch.exp(-float(self.gamma) * dt).unsqueeze(-1)              # [B,1]
            lam_para = torch.exp(-(s32 * s32) * dt.unsqueeze(-1)) * lam_g         # [B,r]

            para = torch.bmm(U32, (lam_para * uTz).unsqueeze(-1)).squeeze(-1)     # [B,z]
            z_next32 = lam_g * (z32 - proj) + para
            z_next32 = torch.nan_to_num(z_next32, nan=0.0, posinf=0.0, neginf=0.0)
        return z_next32.to(z.dtype)

    def propagate_with_info(self, z: torch.Tensor, dt_phys: torch.Tensor, info: dict, A: torch.Tensor | None = None) -> torch.Tensor:
        """Prefer closed-form via (U,s). Fall back to A-based step if needed."""
        if (not self.use_S) and isinstance(info, dict) and ('U' in info) and ('s' in info):
            return self._propagate_lowrank(z, dt_phys, info['U'], info['s'])
        if A is None:
            raise RuntimeError("propagate_with_info needs (U,s) or a provided A for fallback.")
        return self.step_phys_with_A(z, dt_phys, A)

    # ---------- trainer & eval step helpers ----------
    def step(self, z: torch.Tensor, dt_step_norm: torch.Tensor | float, pt: torch.Tensor) -> torch.Tensor:
        """One normalized step using internal conditioning; closed-form when use_S=False."""
        if not torch.is_tensor(dt_step_norm):
            dt_step_norm = torch.as_tensor(dt_step_norm, device=z.device, dtype=torch.float32)
        dt_step_norm = dt_step_norm.to(dtype=torch.float32, device=z.device)
        if dt_step_norm.ndim == 0:
            dt_step_norm = dt_step_norm.expand(z.shape[0])
        elif not (dt_step_norm.ndim == 1 and dt_step_norm.shape[0] == z.shape[0]):
            raise ValueError(f"dt_step_norm must be scalar or [B], got {tuple(dt_step_norm.shape)}")

        dt_phys = self._denorm_dt(dt_step_norm)  # [B]
        with torch.amp.autocast("cuda", enabled=False):
            if not self.use_S:
                info = self._build_info(pt)
                return self._propagate_lowrank(z, dt_phys, info["U"], info["s"])

            A, _info = self._build_A(pt)
            if (not self.training) and (not torch.is_grad_enabled()):
                lam, Q = torch.linalg.eigh(A.float())
                v = torch.matmul(Q.transpose(1, 2), z.float().unsqueeze(-1)).squeeze(-1)
                lamdt = (lam * dt_phys.float().view(z.shape[0], 1)).clamp_(-LAMDT_MAX_ABS, LAMDT_MAX_ABS)
                z_next = torch.matmul(Q, (torch.exp(lamdt) * v).unsqueeze(-1)).squeeze(-1)
            else:
                Phi = torch.matrix_exp(A.float() * dt_phys.float().view(-1, 1, 1))
                z_next = torch.bmm(Phi, z.float().unsqueeze(-1)).squeeze(-1)
        return torch.nan_to_num(z_next, nan=0.0, posinf=0.0, neginf=0.0).to(z.dtype)

    def step_phys_with_A(self, z: torch.Tensor, dt_phys: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Trainer-facing path using a *provided* A.
        - use_S=False + eval(no grad): eig fast-path
        - otherwise: matrix_exp fallback
        """
        B, _Z = z.shape
        if dt_phys.ndim != 1 or dt_phys.shape[0] != B:
            raise ValueError(f"dt_phys must be [B]; got {tuple(dt_phys.shape)} vs B={B}")
        with torch.amp.autocast("cuda", enabled=False):
            if (not self.use_S) and (not self.training) and (not torch.is_grad_enabled()):
                lam, Q = torch.linalg.eigh(A.float())
                v = torch.matmul(Q.transpose(1, 2), z.float().unsqueeze(-1)).squeeze(-1)
                lamdt = (lam * dt_phys.float().view(B, 1)).clamp_(-LAMDT_MAX_ABS, LAMDT_MAX_ABS)
                z_next = torch.matmul(Q, (torch.exp(lamdt) * v).unsqueeze(-1)).squeeze(-1)
            else:
                Phi = torch.matrix_exp(A.float() * dt_phys.float().view(B, 1, 1))
                z_next = torch.bmm(Phi, z.float().unsqueeze(-1)).squeeze(-1)
        return torch.nan_to_num(z_next, nan=0.0, posinf=0.0, neginf=0.0).to(z.dtype)

    def precompute_A(self, pt: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Build once, return (A, info). Trainer can ignore 'info' safely."""
        A, info = self._build_A(pt)
        return A, info


# ==============================
# Top-level model
# ==============================
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
            softmax_head: bool = False,
            z_std_clip: Optional[float] = 10.0,
            predict_delta: bool = True,
            cond_hidden: Sequence[int] = (64, 64),
            rank_l: int = 16,
            use_S: bool = False,
            gamma: float = 0.3,
            s_bound: float = 2.0,
            L_fro_factor: float = 0.5,
            spectral_norm_cond: bool = False,
            decoder_condition_on_g: bool = True,
            cap_via_gamma: bool = True,
            learn_U: bool = False,
            U_method: str = "qr",
            basis: str = "dct",
            stability_projection_eps: Optional[float] = None,
            train_eigh_fastpath: bool = False,
            target_log_mean: Optional[Sequence[float]] = None,
            target_log_std: Optional[Sequence[float]] = None,
            dt_stats: Dict[str, float] = None,
            **kwargs,  # ← absorbs legacy/unknown keys (e.g., koopman_rank)
    ):
        """
        Stable LPV Koopman Autoencoder.

        Back-compat: accepts legacy kwarg `koopman_rank` and maps it to `rank_l`
        if `rank_l` isn't explicitly set by the caller.
        Unknown extra kwargs are ignored (safe no-ops).
        """
        # ---------- legacy alias handling ----------
        if "koopman_rank" in kwargs and (rank_l == 16):  # only override if user didn't set rank_l
            try:
                rank_l = int(kwargs.pop("koopman_rank"))
            except Exception:
                kwargs.pop("koopman_rank", None)  # drop bad value; keep default rank_l

        # (Optionally: log/ignore any remaining kwargs)
        # for k in list(kwargs.keys()):
        #     print(f"[StableLPVKoopmanAE] ignoring unknown kwarg: {k}")
        #     kwargs.pop(k)

        # ---------- core fields ----------
        super().__init__()
        self.S_out = int(state_dim)
        self.G_in = int(global_dim)
        self.Z = int(latent_dim)
        self.predict_delta = bool(predict_delta)
        self.decoder_condition_on_g = bool(decoder_condition_on_g)

        # ---------- encoder ----------
        act = get_activation(activation)
        self.encoder = Encoder(self.S_out, self.G_in, encoder_hidden, self.Z, act, dropout)

        # ---------- decoder (supports optional softmax_head) ----------
        log_stats = None
        if softmax_head:
            if target_log_mean is None or target_log_std is None:
                raise ValueError("softmax_head=True requires target_log_mean/target_log_std.")
            lm = torch.tensor(list(target_log_mean), dtype=torch.float32)
            ls = torch.tensor(list(target_log_std), dtype=torch.float32)
            log_stats = (lm, ls)

        dec_in_dim = self.Z + (self.G_in if self.decoder_condition_on_g else 0)
        self.decoder = Decoder(
            dec_in_dim, decoder_hidden, self.S_out, act, dropout,
            softmax_head=softmax_head, log_stats=log_stats, z_std_clip=z_std_clip
        )

        # ---------- dynamics ----------
        if rank_l > self.Z:
            raise ValueError(f"rank_l must be ≤ latent_dim; got rank_l={rank_l}, latent_dim={self.Z}")

        self.dynamics = LPVDynamics(
            latent_dim=self.Z,
            cond_hidden=cond_hidden,
            global_dim=self.G_in,
            activation=act,
            dropout=dropout,
            r=int(rank_l),
            s_bound=float(s_bound),
            gamma=float(gamma),
            use_S=bool(use_S),
            spectral_norm_cond=bool(spectral_norm_cond),
            decoder_condition_on_g=self.decoder_condition_on_g,
            cap_via_gamma=bool(cap_via_gamma),
            learn_U=bool(learn_U),
            U_method=str(U_method),
            basis=str(basis),
            stability_projection_eps=stability_projection_eps,
            dt_stats=dt_stats,
            L_fro_factor=float(L_fro_factor),
            train_eigh_fastpath=bool(train_eigh_fastpath),
        )

    def _propagate_K_lowrank(self, z0: torch.Tensor, dt_phys_all: torch.Tensor, U: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Efficient K-step propagation:
          y0 = Uᵀ z0, w0 = z0 - U y0
          y_{t+1} = exp(-(γ + s^2) Δt_t) ⊙ y_t
          w_{t+1} = exp(-γ Δt_t) * w_t
          z_t = U y_t + w_t
        """
        B, K = dt_phys_all.shape
        ZK = torch.empty(B, K, self.Z, device=z0.device, dtype=z0.dtype)

        with torch.amp.autocast("cuda", enabled=False):
            z = z0.float(); U32 = U.float(); s32 = s.float()
            y = torch.bmm(U32.transpose(1, 2), z.unsqueeze(-1)).squeeze(-1)  # [B,r]
            Uy = torch.bmm(U32, y.unsqueeze(-1)).squeeze(-1)                 # [B,Z]
            w = z - Uy                                                       # [B,Z]

            for t in range(K):
                dt = dt_phys_all[:, t].float()
                lam_g = torch.exp(-float(self.dynamics.gamma) * dt).unsqueeze(-1)      # [B,1]
                lam_y = lam_g * torch.exp(-(s32 * s32) * dt.unsqueeze(-1))             # [B,r]

                y = lam_y * y
                w = lam_g * w

                z_t = torch.bmm(U32, y.unsqueeze(-1)).squeeze(-1) + w
                ZK[:, t] = z_t.to(ZK.dtype)

        return ZK

    def forward(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # normalize dt to [B,K]
        if dt_norm.ndim == 3 and dt_norm.shape[-1] == 1:
            dt_norm = dt_norm.squeeze(-1)
        B, _S = y_i.shape
        if dt_norm.ndim == 1:
            if dt_norm.shape[0] != B:
                raise ValueError(f"dt_norm shape mismatch: got {tuple(dt_norm.shape)}, expected [{B}] or [{B},K]")
            dt_norm = dt_norm.view(B, 1)
        elif dt_norm.ndim != 2 or dt_norm.shape[0] != B:
            raise ValueError(f"dt_norm must be [B] or [B,K], got {tuple(dt_norm.shape)}")
        K = int(dt_norm.shape[1])

        y_i = torch.nan_to_num(y_i, nan=0.0, posinf=0.0, neginf=0.0)
        g   = torch.nan_to_num(g,   nan=0.0, posinf=0.0, neginf=0.0)

        # Encode once
        z0 = self.encoder(y_i, g)  # [B,Z]

        if not self.dynamics.use_S:
            info = self.dynamics._build_info(g)                 # no A build here
            dt_phys_all = self.dynamics._denorm_dt(dt_norm)    # [B,K]
            ZK = self._propagate_K_lowrank(z0, dt_phys_all, info["U"], info["s"])
        else:
            A, _info = self.dynamics._build_A(g)
            dt_phys_all = self.dynamics._denorm_dt(dt_norm)
            ZK = torch.empty(B, K, self.Z, device=z0.device, dtype=z0.dtype)
            z_t = z0
            for t in range(K):
                z_t = self.dynamics.step_phys_with_A(z_t, dt_phys_all[:, t], A)
                ZK[:, t] = z_t

        # Decode in a single batched call
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
        if self.predict_delta:
            y_next = y + y_next
        return torch.nan_to_num(y_next, nan=0.0, posinf=0.0, neginf=0.0)

    @torch.no_grad()
    def rollout(self, y0: torch.Tensor, g: torch.Tensor, dt_step_norm: torch.Tensor | float, steps: int) -> torch.Tensor:
        """Uniform-step rollout for visualization/eval (closed form when use_S=False)."""
        if steps <= 0:
            raise ValueError("steps must be > 0")
        B, _ = y0.shape
        if not torch.is_tensor(dt_step_norm):
            dt_step_norm = torch.as_tensor(dt_step_norm, device=y0.device, dtype=torch.float32)
        if dt_step_norm.ndim == 0:
            dt_step_norm = dt_step_norm.expand(B)
        elif not (dt_step_norm.ndim == 1 and dt_step_norm.shape[0] == B):
            raise ValueError(f"dt_step_norm must be scalar or [B], got {tuple(dt_step_norm.shape)} for B={B}")

        z0 = self.encoder(y0, g)
        if not self.dynamics.use_S:
            info = self.dynamics._build_info(g)
            dt_phys = self.dynamics._denorm_dt(dt_step_norm).unsqueeze(1).expand(B, steps)  # [B,steps]
            ZK = self._propagate_K_lowrank(z0, dt_phys, info["U"], info["s"])
        else:
            A, _info = self.dynamics._build_A(g)
            dt_phys = self.dynamics._denorm_dt(dt_step_norm)
            ZK = torch.empty(B, steps, self.Z, device=z0.device, dtype=z0.dtype)
            z_t = z0
            for t in range(int(steps)):
                z_t = self.dynamics.step_phys_with_A(z_t, dt_phys, A)
                ZK[:, t] = z_t

        if self.decoder_condition_on_g:
            gK = g.unsqueeze(1).expand(B, steps, -1)
            dec_in = torch.cat([ZK, gK], dim=-1).reshape(B * steps, -1)
        else:
            dec_in = ZK.reshape(B * steps, -1)
        dec_out = self.decoder(dec_in).reshape(B, steps, -1)

        if self.predict_delta:
            y_out = y0.unsqueeze(1) + torch.cumsum(dec_out, dim=1)
        else:
            y_out = dec_out
        return torch.nan_to_num(y_out, nan=0.0, posinf=0.0, neginf=0.0)


# ==============================
# Factory (+ back-compat)
# ==============================
def _get_target_log_stats(manifest: dict, keys: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
    per = manifest.get("per_key_stats") or {}
    LOG_STD_MIN = 1e-8
    log_means, log_stds = [], []
    for k in keys:
        st = per.get(k) or {}
        lm = float(st.get("log_mean"))
        ls = float(st.get("log_std"))
        if not (math.isfinite(lm) and math.isfinite(ls)):
            raise ValueError(f"Non-finite log stats for '{k}': mean={lm}, std={ls}")
        log_means.append(lm)
        log_stds.append(max(ls, LOG_STD_MIN))
    return torch.tensor(log_means, dtype=torch.float32), torch.tensor(log_stds, dtype=torch.float32)


def _get_dt_stats(m: dict) -> Dict[str, float]:
    d = m.get("dt")
    if isinstance(d, dict) and "log_min" in d and "log_max" in d:
        lo, hi = float(d["log_min"]), float(d["log_max"])
        if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
            return {"log_min": lo, "log_max": hi}
    tvar = (m.get("meta") or {}).get("time_variable", "t_time")
    s = (m.get("per_key_stats") or {}).get(tvar) or {}
    if "log_min" in s and "log_max" in s:
        lo, hi = float(s["log_min"]), float(s["log_max"])
        if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
            return {"log_min": lo, "log_max": hi}
    for k in ("dt_stats", "time_stats"):
        s = m.get(k) or {}
        if "log_min" in s and "log_max" in s:
            lo, hi = float(s["log_min"]), float(s["log_max"])
            if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
                return {"log_min": lo, "log_max": hi}
    raise ValueError("Normalization manifest missing dt stats (log_min/log_max).")


def _resolve_manifest_path(cfg: dict) -> Path:
    pdir = (cfg.get("paths") or {}).get("processed_data_dir")
    if not pdir:
        raise ValueError("cfg['paths']['processed_data_dir'] not set; cannot locate normalization.json")
    return Path(pdir) / "normalization.json"


def create_model(arg1, cfg: dict | None = None) -> StableLPVKoopmanAE:
    """
    Back-compat:
      - create_model(cfg)
      - create_model(manifest_path, cfg)
    """
    if cfg is None:
        cfg = arg1
        manifest_path = _resolve_manifest_path(cfg)
    else:
        manifest_path = Path(arg1)

    manifest = json.loads(Path(manifest_path).read_text())

    data_cfg = cfg.get("data") or {}
    species = list(data_cfg.get("species_variables") or [])
    globals_ = list(data_cfg.get("global_variables") or [])

    S_out = len(species) if species else int(manifest.get("meta", {}).get("num_species", 0))
    if S_out <= 0:
        raise ValueError("State dimension S_out must be > 0 (check manifest/species list).")
    G_in = len(globals_) if globals_ else int(manifest.get("meta", {}).get("num_globals", 0))
    if G_in <= 0:
        raise ValueError("Global dimension G_in must be > 0 (check manifest/global_variables).")

    target_log_mean = target_log_std = None
    if bool(cfg.get("model", {}).get("softmax_head", False)):
        per = manifest.get("per_key_stats") or {}
        LOG_STD_MIN = 1e-8
        log_means, log_stds = [], []
        for k in species:
            st = per.get(k) or {}
            lm = float(st.get("log_mean")); ls = float(st.get("log_std"))
            if not (math.isfinite(lm) and math.isfinite(ls)):
                raise ValueError(f"Non-finite log stats for '{k}': mean={lm}, std={ls}")
            log_means.append(lm); log_stds.append(max(ls, LOG_STD_MIN))
        target_log_mean, target_log_std = log_means, log_stds

    dt_stats = (lambda m: (
        {"log_min": float((m.get("dt") or {}).get("log_min")),
         "log_max": float((m.get("dt") or {}).get("log_max"))}
        if (m.get("dt") and "log_min" in m["dt"] and "log_max" in m["dt"])
        else _get_dt_stats(m)
    ))(manifest)

    m = cfg.get("model") or {}
    return StableLPVKoopmanAE(
        state_dim=S_out,
        global_dim=G_in,
        latent_dim=int(m.get("latent_dim", 48)),
        encoder_hidden=list(m.get("encoder_hidden", [256, 256, 256])),
        decoder_hidden=list(m.get("decoder_hidden", [256, 256, 256])),
        activation=str(m.get("activation", "silu")),
        dropout=float(m.get("dropout", 0.0)),
        softmax_head=bool(m.get("softmax_head", False)),
        z_std_clip=m.get("z_std_clip", 10.0),
        predict_delta=bool(m.get("predict_delta", True)),
        cond_hidden=list(m.get("cond_hidden", [64, 64])),
        rank_l=int(m.get("rank_l", m.get("koopman_rank", 16))),
        use_S=bool(m.get("use_S", False)),
        gamma=float(m.get("gamma", 0.3)),
        s_bound=float(m.get("s_bound", 2.0)),
        L_fro_factor=float(m.get("L_fro_factor", 0.5)),
        spectral_norm_cond=bool(m.get("spectral_norm_cond", False)),
        decoder_condition_on_g=bool(m.get("decoder_condition_on_g", True)),
        cap_via_gamma=bool(m.get("cap_via_gamma", True)),
        learn_U=bool(m.get("learn_U", False)),
        U_method=str(m.get("U_method", "qr")),
        basis=str(m.get("basis", "dct")),
        stability_projection_eps=m.get("stability_projection_eps", None),
        train_eigh_fastpath=bool(m.get("train_eigh_fastpath", False)),
        target_log_mean=target_log_mean,
        target_log_std=target_log_std,
        dt_stats=dt_stats,
    )


FlowMapKoopman = StableLPVKoopmanAE

__all__ = [
    "MLP", "Encoder", "Decoder",
    "LPVDynamics", "StableLPVKoopmanAE",
    "create_model", "FlowMapKoopman",
]

#!/usr/bin/env python3
"""
Flow-map Koopman Autoencoder with Stable Coupled Dynamics (PATCHED)
===================================================================
This version applies targeted correctness fixes and low-effort speed/robustness
improvements suggested in review:
- Fix mask shape bug: avoid squeezing `inv`; use boolean mask directly
- Fix broadcasting with unique-Δt scaling without `.item()` host sync
- Bucket on dt_norm (exact duplicates) and map uniques to physical once
- Fallback to full-batch path when bucketing brings no savings
- Reuse buffers per step to reduce allocations
- Add `step_with_K` to reuse cached K in rollout; actually use it
- Simpler decoder call in rollout; keep zero-step identity and φ₁ integration
- Log first dt clamp once (already wired)

Notes:
- Assumes `dt_norm` is in [0,1] from a log-min-max normalization of physical Δt.
- Uses diagonal damping to ensure stability: K = S - L L^T - γ I with γ>0.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------- Utilities --------------------------------------

def get_activation(name: str) -> nn.Module:
    name = (name or "silu").lower()
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "silu": nn.SiLU(),
        "swish": nn.SiLU(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: nn.Module,
        dropout: float = 0.0,
        zero_init_output: bool = False,
    ):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation.__class__())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        out = nn.Linear(prev, output_dim)
        if zero_init_output:
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
        layers.append(out)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.network(x)


# ---------------------------- Encoder ----------------------------------------

class Encoder(nn.Module):
    """Deterministic encoder: [y_i, g] -> z"""

    def __init__(
        self,
        state_dim: int,
        global_dim: int,
        hidden_dims: Sequence[int],
        latent_dim: int,
        activation: nn.Module,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.network = MLP(
            state_dim + global_dim, hidden_dims, latent_dim, activation, dropout
        )

    def forward(self, y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x = torch.cat([y, g], dim=-1)
        return self.network(x)


# ----------------------- Koopman Latent Dynamics -----------------------------

class KoopmanDynamics(nn.Module):
    """
    Stable coupled latent dynamics with single-shot Δt.
    Ensures f(y,0) = y through proper zero-step handling.
    """

    def __init__(
        self,
        latent_dim: int,
        global_dim: int,
        rank: int = 4,
        bias_hidden: int = 128,
        use_phi1_residual: bool = False,
        dt_stats: Optional[Dict[str, float]] = None,
        min_damping: float = 1e-4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.rank = min(rank, latent_dim)
        self.use_phi1_residual = use_phi1_residual
        self.min_damping = min_damping
        self.logger = logging.getLogger(__name__)

        # Networks for stable K(g) = S(g) - L(g)L(g)^T
        self.skew_net = MLP(global_dim, [bias_hidden], latent_dim * latent_dim, nn.SiLU(), 0.0)
        self.dissip_net = MLP(global_dim, [bias_hidden], latent_dim * self.rank, nn.SiLU(), 0.0)

        # Learnable diagonal damping parameter (slightly smaller initial value)
        self.gamma_param = nn.Parameter(torch.tensor(-6.0))  # ~0.0025 after softplus

        # Bias network with log-dt features: [dt_norm, log10_dt, globals]
        self.bias_net = MLP(
            2 + global_dim,
            [bias_hidden, bias_hidden],
            latent_dim,
            nn.SiLU(),
            0.0,
            zero_init_output=True,
        )

        # Optional residual network
        if use_phi1_residual:
            self.residual_net = MLP(
                latent_dim + global_dim,
                [bias_hidden],
                latent_dim,
                nn.SiLU(),
                0.0,
                zero_init_output=True,
            )
            # Apply spectral norm to first Linear only (guarded)
            applied = False
            for mod in self.residual_net.network.modules():
                if (
                    isinstance(mod, nn.Linear)
                    and not applied
                    and not hasattr(mod, "weight_orig")
                ):
                    nn.utils.spectral_norm(mod)
                    applied = True
                    break

        # Conservative initialization
        with torch.no_grad():
            for net in [self.skew_net, self.dissip_net]:
                for m in net.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, 0.0, 0.001)
                        nn.init.zeros_(m.bias)

        # Store dt normalization stats
        if dt_stats is None:
            raise ValueError("dt_stats required")

        self.register_buffer("dt_log_min", torch.tensor(float(dt_stats["log_min"])) )
        self.register_buffer("dt_log_max", torch.tensor(float(dt_stats["log_max"])) )

        # Physical dt bounds for clamping
        dt_min_phys = 10.0 ** float(dt_stats["log_min"])
        dt_max_phys = 10.0 ** float(dt_stats["log_max"])
        self.register_buffer("dt_min_phys", torch.tensor(dt_min_phys))
        self.register_buffer("dt_max_phys", torch.tensor(dt_max_phys))

        self._dt_clamp_warned = False

    def _build_stable_K(self, g: torch.Tensor) -> torch.Tensor:
        """Build stable generator K(g) = S(g) - L(g)L(g)^T - γI"""
        B = g.shape[0]
        d = self.latent_dim

        # Skew-symmetric part
        A_flat = self.skew_net(g)
        A = A_flat.view(B, d, d)
        S = 0.5 * (A - A.transpose(1, 2))

        # Dissipative part
        L = self.dissip_net(g).view(B, d, self.rank)
        D = torch.bmm(L, L.transpose(1, 2))

        # Diagonal damping
        gamma = F.softplus(self.gamma_param) + self.min_damping
        eye = torch.eye(d, device=g.device, dtype=g.dtype)

        K = S - D - gamma * eye.unsqueeze(0)
        return K

    def _denorm_dt_to_phys(self, dt_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized Δt to physical units with clamping"""
        dt_norm = dt_norm.clamp(0, 1)
        log_dt = self.dt_log_min + dt_norm * (self.dt_log_max - self.dt_log_min)
        dt_phys = (10.0 ** log_dt).clamp_min(1e-30)

        # Clamp and warn once
        dt_unclamped = dt_phys.clone()
        dt_phys = dt_phys.clamp(self.dt_min_phys, self.dt_max_phys)

        if not self._dt_clamp_warned and torch.any(dt_phys != dt_unclamped):
            self._dt_clamp_warned = True
            if self.training:
                self.logger.warning(
                    f"dt clamped to training range [{self.dt_min_phys.item():.3e}, {self.dt_max_phys.item():.3e}]"
                )

        return dt_phys

    def _compute_phi1(
        self, K_scaled: torch.Tensor, eye: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute φ₁(K) = (exp(K) - I) / K via block-matrix trick."""
        B, d, _ = K_scaled.shape
        device = K_scaled.device

        # Build augmented matrix [[K, I], [0, K]]
        aug = torch.zeros(B, 2 * d, 2 * d, device=device, dtype=K_scaled.dtype)
        aug[:, :d, :d] = K_scaled

        if eye is None:
            eye = torch.eye(d, device=device, dtype=K_scaled.dtype)
        aug[:, :d, d:] = eye.unsqueeze(0) if eye.ndim == 2 else eye
        aug[:, d:, d:] = K_scaled

        exp_aug = torch.linalg.matrix_exp(aug)
        phi1 = exp_aug[:, :d, d:]
        return phi1

    @torch.no_grad()
    def step_with_K(
        self,
        z: torch.Tensor,         # [B, Z]
        dt_norm: torch.Tensor,   # [B, 1] or [B]
        g: torch.Tensor,         # [B, G]
        K: torch.Tensor,         # [B, d, d] (precomputed)
    ) -> torch.Tensor:
        """Single-step evolution that reuses a precomputed K.
        Matches the per-step body of `forward` but skips rebuilding K.
        Returns z_next: [B, Z].
        """
        if dt_norm.ndim == 1:
            dt_norm = dt_norm.unsqueeze(1)  # [B,1]

        B = z.shape[0]
        d = self.latent_dim
        eye = torch.eye(d, device=K.device, dtype=K.dtype)

        # Zero-step fast path
        is_zero = dt_norm <= 0
        if torch.all(is_zero):
            return z

        # Unique on normalized dt (exact duplicates), then map to physical once
        vals_norm, inv = torch.unique(dt_norm, return_inverse=True, dim=0)  # [U,1], [B]
        vals_phys = torch.where(
            vals_norm <= 0,
            torch.zeros_like(vals_norm),
            self._denorm_dt_to_phys(vals_norm),
        )  # [U,1]

        # Heuristic fallback: if no dedup benefit, do full-batch once
        U = vals_phys.shape[0]
        exp_K = torch.empty(B, d, d, device=K.device, dtype=K.dtype)
        phi1 = torch.empty_like(exp_K)

        if U > 0.8 * B:
            dt_phys_full = torch.where(
                is_zero, torch.zeros_like(dt_norm), self._denorm_dt_to_phys(dt_norm)
            )  # [B,1]
            K_scaled = K * dt_phys_full.unsqueeze(-1)  # [B,d,d]
            exp_K[:] = torch.linalg.matrix_exp(K_scaled)
            phi1[:] = self._compute_phi1(K_scaled, eye)
        else:
            # Initialize to avoid stale values when filling by mask
            exp_K.zero_()
            phi1.zero_()
            for u in range(U):
                m = inv == u  # [B]
                v = vals_phys[u]  # [1] or [1,1]
                if torch.all(v == 0):
                    exp_K[m] = eye
                    phi1[m] = eye
                else:
                    v3 = v.view(1, 1, 1)
                    K_scaled = K[m] * v3  # [Nu,d,d]
                    exp_K[m] = torch.linalg.matrix_exp(K_scaled)
                    phi1[m] = self._compute_phi1(K_scaled, eye)

        # Linear evolution
        z_linear = torch.bmm(exp_K, z.unsqueeze(-1)).squeeze(-1)

        # Bias with log-dt features
        dt_phys_full = torch.where(
            is_zero, torch.zeros_like(dt_norm), self._denorm_dt_to_phys(dt_norm)
        )
        log_dt = torch.where(
            is_zero, torch.zeros_like(dt_phys_full), torch.log10(dt_phys_full.clamp(min=1e-30))
        )
        bias_input = torch.cat([dt_norm, log_dt, g], dim=-1)  # [B, 2+G]
        bias_vec = torch.tanh(self.bias_net(bias_input)) * dt_phys_full  # [B, Z]
        bias = torch.bmm(phi1, bias_vec.unsqueeze(-1)).squeeze(-1)  # [B, Z]

        # Optional residual
        if self.use_phi1_residual and hasattr(self, "residual_net"):
            r_raw = self.residual_net(torch.cat([z, g], dim=-1))  # [B, Z]
            r = r_raw * dt_phys_full
            r_integrated = torch.bmm(phi1, r.unsqueeze(-1)).squeeze(-1)
            z_next = z_linear + bias + r_integrated
        else:
            z_next = z_linear + bias

        return z_next

    def forward(
        self, z: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward dynamics with efficient bucketing for shared timesteps.

        z: [B, Z] or [B, 1, Z] initial latent state
        dt_norm: [B, K] or [B, K, 1] or [B] normalized time steps
        g: [B, G] global features
        Returns: [B, K, Z] evolved states
        """
        # Handle input shapes
        if z.ndim == 2:
            B, Z = z.shape
            z = z.unsqueeze(1)
        else:
            B, _, Z = z.shape

        assert z.shape[1] == 1, "Single initial state expected"
        z_curr = z[:, 0]

        # Normalize dt_norm to [B, K]
        if dt_norm.ndim == 1:
            dt_norm = dt_norm.unsqueeze(1)
        if dt_norm.ndim == 3 and dt_norm.shape[-1] == 1:
            dt_norm = dt_norm.squeeze(-1)
        K_steps = dt_norm.shape[1]

        # Build stable generator once
        K = self._build_stable_K(g)
        d = self.latent_dim

        # Precompute identity and reusable buffers once
        eye = torch.eye(d, device=K.device, dtype=K.dtype)
        exp_K_buf = torch.empty(B, d, d, device=K.device, dtype=K.dtype)
        phi1_buf = torch.empty_like(exp_K_buf)

        if self.training:
            assert (dt_norm >= 0).all() and (dt_norm <= 1).all(), "dt_norm out of bounds"

        outputs = []

        for k in range(K_steps):
            dt_k = dt_norm[:, k : k + 1]  # [B,1]

            # Fast path for all-zero step
            is_zero = dt_k <= 0
            if torch.all(is_zero):
                outputs.append(z_curr)
                continue

            # Unique on dt_norm, map to physical once
            vals_norm, inv = torch.unique(dt_k, return_inverse=True, dim=0)  # [U,1], [B]
            vals_phys = torch.where(
                vals_norm <= 0,
                torch.zeros_like(vals_norm),
                self._denorm_dt_to_phys(vals_norm),
            )  # [U,1]

            U = vals_phys.shape[0]
            exp_K = exp_K_buf
            phi1 = phi1_buf

            if U > 0.8 * B:
                # No real dedup benefit: do full-batch once
                dt_phys_full = torch.where(
                    is_zero, torch.zeros_like(dt_k), self._denorm_dt_to_phys(dt_k)
                )  # [B,1]
                K_scaled = K * dt_phys_full.unsqueeze(-1)  # [B,d,d]
                exp_K[:] = torch.linalg.matrix_exp(K_scaled)
                phi1[:] = self._compute_phi1(K_scaled, eye)
            else:
                # Fill by bucket
                exp_K.zero_()
                phi1.zero_()
                for u in range(U):
                    m = inv == u  # [B]
                    v = vals_phys[u]  # [1] or [1,1]
                    if torch.all(v == 0):
                        exp_K[m] = eye
                        phi1[m] = eye
                    else:
                        v3 = v.view(1, 1, 1)
                        K_scaled = K[m] * v3  # [Nu,d,d]
                        exp_K[m] = torch.linalg.matrix_exp(K_scaled)
                        phi1[m] = self._compute_phi1(K_scaled, eye)

            # Linear evolution
            z_linear = torch.bmm(exp_K, z_curr.unsqueeze(-1)).squeeze(-1)

            # Bias with log-dt features
            dt_phys_full = torch.where(
                is_zero, torch.zeros_like(dt_k), self._denorm_dt_to_phys(dt_k)
            )
            log_dt = torch.where(
                is_zero, torch.zeros_like(dt_phys_full), torch.log10(dt_phys_full.clamp(min=1e-30))
            )
            bias_input = torch.cat([dt_k, log_dt, g], dim=-1)  # [B, 2+G]
            bias_vec = torch.tanh(self.bias_net(bias_input)) * dt_phys_full  # [B, Z]
            bias = torch.bmm(phi1, bias_vec.unsqueeze(-1)).squeeze(-1)

            # Optional residual
            if self.use_phi1_residual and hasattr(self, "residual_net"):
                r_raw = self.residual_net(torch.cat([z_curr, g], dim=-1))  # [B, Z]
                r = r_raw * dt_phys_full
                r_integrated = torch.bmm(phi1, r.unsqueeze(-1)).squeeze(-1)
                z_next = z_linear + bias + r_integrated
            else:
                z_next = z_linear + bias

            outputs.append(z_next)
            z_curr = z_next

        return torch.stack(outputs, dim=1)


# ----------------------------- Decoder ---------------------------------------

class Decoder(nn.Module):
    """Decoder with optional simplex constraint for mole fractions"""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: Sequence[int],
        state_dim: int,
        activation: nn.Module,
        dropout: float = 0.0,
        softmax_head: bool = True,
        log_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        super().__init__()
        self.network = MLP(latent_dim, hidden_dims, state_dim, activation, dropout)
        self.softmax_head = softmax_head
        self.state_dim = state_dim

        if softmax_head:
            if log_stats is None:
                raise ValueError("softmax_head requires log_mean and log_std")

            # Ensure correct device placement
            device = next(self.network.parameters()).device
            self.register_buffer("log_mean", log_stats[0].to(device))
            self.register_buffer("log_std", log_stats[1].clamp(min=1e-10).to(device))

            assert self.log_mean.shape[0] == state_dim
            assert self.log_std.shape[0] == state_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # noqa: D401
        y_base = self.network(z)
        assert y_base.shape[-1] == self.state_dim, "Output shape mismatch"

        if self.softmax_head:
            ln10 = math.log(10.0)
            logits = y_base.clamp(-40, 40)
            log_p = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            log10_p = log_p / ln10

            mu = self.log_mean.view(1, 1, -1) if y_base.ndim == 3 else self.log_mean
            sig = self.log_std.view(1, 1, -1) if y_base.ndim == 3 else self.log_std
            return (log10_p - mu) / sig

        return y_base


# --------------------------- Full Model --------------------------------------

class FlowMapKoopman(nn.Module):
    """
    Flow-map Koopman autoencoder optimized for stiff chemical kinetics.
    Guarantees identity at zero timestep and efficient multi-scale dynamics.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        global_dim: int,
        latent_dim: int,
        encoder_hidden: Sequence[int],
        decoder_hidden: Sequence[int],
        koopman_rank: int = 4,
        koopman_bias_hidden: int = 128,
        use_phi1_residual: bool = False,
        dt_stats: Optional[Dict[str, float]] = None,
        activation: str = "silu",
        dropout: float = 0.0,
        softmax_head: bool = True,
        target_log_mean: Optional[Sequence[float]] = None,
        target_log_std: Optional[Sequence[float]] = None,
        min_damping: float = 1e-4,
    ):
        super().__init__()

        self.S = state_dim
        self.G = global_dim
        self.Z = latent_dim

        act = get_activation(activation)

        self.encoder = Encoder(state_dim, global_dim, encoder_hidden, latent_dim, act, dropout)
        self.dynamics = KoopmanDynamics(
            latent_dim=latent_dim,
            global_dim=global_dim,
            rank=koopman_rank,
            bias_hidden=koopman_bias_hidden,
            use_phi1_residual=use_phi1_residual,
            dt_stats=dt_stats,
            min_damping=min_damping,
        )

        log_stats = None
        if softmax_head:
            if target_log_mean is None or target_log_std is None:
                raise ValueError("softmax_head requires target_log_mean/std")
            log_stats = (
                torch.tensor(target_log_mean, dtype=torch.float32),
                torch.tensor(target_log_std, dtype=torch.float32).clamp(min=1e-10),
            )

        self.decoder = Decoder(
            latent_dim, decoder_hidden, state_dim, act, dropout, softmax_head, log_stats
        )

        self.softmax_head = softmax_head

    def forward(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Single-shot multi-timestep prediction"""
        z_i = self.encoder(y_i, g)
        z_j = self.dynamics(z_i, dt_norm, g)
        y_j = self.decoder(z_j)
        return y_j

    @torch.no_grad()
    def rollout(
        self,
        y_0: torch.Tensor,
        g: torch.Tensor,
        dt_step: torch.Tensor,  # normalized Δt per step; scalar or [B]
        max_steps: int = 100,
    ) -> torch.Tensor:
        """
        Efficient autoregressive rollout with cached K computation and step reuse.
        Returns: [B, T, S]
        """
        B = y_0.shape[0]

        dt_step = dt_step.reshape(-1)
        if dt_step.numel() == 1:
            dt_step = dt_step.expand(B)
        elif dt_step.numel() != B:
            raise ValueError("dt_step shape mismatch")

        # Precompute K once for entire rollout
        K = self.dynamics._build_stable_K(g)

        trajectory = []
        z_curr = self.encoder(y_0, g)  # [B,Z]

        for _ in range(max_steps):
            z_next = self.dynamics.step_with_K(z_curr, dt_step, g, K)  # [B,Z]
            y_next = self.decoder(z_next)  # [B,S]
            trajectory.append(y_next)
            z_curr = z_next

        return torch.stack(trajectory, dim=1)


# ------------------------------ Factory --------------------------------------

def create_model(config: dict) -> FlowMapKoopman:
    """Build model from configuration"""
    data_cfg = config.get("data", {})
    species_vars = list(data_cfg.get("species_variables", []))
    if not species_vars:
        raise KeyError("data.species_variables must be set")

    global_vars = list(data_cfg.get("global_variables", []))
    S = len(species_vars)
    G = len(global_vars)

    mcfg = config.get("model", {})
    latent_dim = int(mcfg.get("latent_dim", 12))
    encoder_hidden = mcfg.get("encoder_hidden", [128, 128])
    decoder_hidden = mcfg.get("decoder_hidden", [128, 128])

    kcfg = mcfg.get("koopman", {})
    koopman_rank = int(kcfg.get("rank", 4))
    bias_hidden = int(kcfg.get("bias_hidden", 128))
    use_phi1 = bool(kcfg.get("use_phi1_residual", False))
    min_damping = float(kcfg.get("min_damping", 1e-4))

    activation = str(mcfg.get("activation", "silu"))
    dropout = float(mcfg.get("dropout", 0.0))
    softmax_head = bool(mcfg.get("softmax_head", True))

    proc_dir = Path(config["paths"]["processed_data_dir"])
    manifest_path = proc_dir / "normalization.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    if "dt" not in manifest:
        raise ValueError("dt normalization spec missing")

    dt_stats = {
        "log_min": float(manifest["dt"]["log_min"]),
        "log_max": float(manifest["dt"]["log_max"]),
    }

    target_log_mean = None
    target_log_std = None
    if softmax_head:
        stats = manifest["per_key_stats"]
        target_log_mean = []
        target_log_std = []
        for name in species_vars:
            s = stats.get(name, {})
            target_log_mean.append(float(s.get("log_mean", 0.0)))
            target_log_std.append(float(s.get("log_std", 1.0)))

    return FlowMapKoopman(
        state_dim=S,
        global_dim=G,
        latent_dim=latent_dim,
        encoder_hidden=encoder_hidden,
        decoder_hidden=decoder_hidden,
        koopman_rank=koopman_rank,
        koopman_bias_hidden=bias_hidden,
        use_phi1_residual=use_phi1,
        dt_stats=dt_stats,
        activation=activation,
        dropout=dropout,
        softmax_head=softmax_head,
        target_log_mean=target_log_mean,
        target_log_std=target_log_std,
        min_damping=min_damping,
    )

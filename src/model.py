#!/usr/bin/env python3
"""
Flow-map Koopman Autoencoder for Mole Fractions (Simplex-Constrained Output)
============================================================================
Production-ready version with stability fixes:
- Conservative initialization for lambda network
- Safety clamping throughout
- Deterministic VAE at eval (no sampling noise)
- Softmax head for mole fraction constraints
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------- Utilities --------------------------------------

def get_activation(name: str) -> nn.Module:
    name = (name or "silu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name in ("silu", "swish"):
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dims: Sequence[int],
            output_dim: int,
            activation: nn.Module,
            dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation.__class__())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ---------------------------- Encoder ----------------------------------------

class Encoder(nn.Module):
    """
    [y_i, g] -> z. Supports deterministic AE and VAE.
    If vae_mode=True: samples during training, returns mean at eval (deterministic).
    """

    def __init__(
            self,
            state_dim: int,
            global_dim: int,
            hidden_dims: Sequence[int],
            latent_dim: int,
            activation: nn.Module,
            dropout: float = 0.0,
            vae_mode: bool = False,
    ):
        super().__init__()
        out_dim = latent_dim * (2 if vae_mode else 1)
        self.network = MLP(state_dim + global_dim, hidden_dims, out_dim, activation, dropout)
        self.latent_dim = latent_dim
        self.vae_mode = vae_mode

    def forward(self, y: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = torch.cat([y, g], dim=-1)
        out = self.network(x)
        if not self.vae_mode:
            return out, None

        mu, logvar = torch.chunk(out, 2, dim=-1)

        # FIX: Deterministic at eval to prevent noise accumulation in rollouts
        if not self.training:
            return mu, torch.tensor(0.0, device=mu.device)

        # Training: sample from distribution
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return z, kl


# ----------------------- Koopman Latent Dynamics -----------------------------

class KoopmanDynamics(nn.Module):
    """
    z_{t+Δt} = exp(λ(g[,Δt]) * Δt_phys) ⊙ z_t + gate(Δt_norm,g) ⊙ b(Δt_norm,g) * Δt_phys
    - λ is parameterized as -softplus(raw) to enforce non-positive rates (stable).
    - Δt_phys is reconstructed from normalized dt via log10 stats.
    - gate is scalar or per-latent in [0,1] (sigmoid).
    - bias is scaled by Δt_phys to prevent per-substep accumulation.
    """

    def __init__(
            self,
            latent_dim: int,
            global_dim: int,
            bias_hidden: int = 64,
            use_dt_gate: bool = True,
            gate_type: str = "scalar",  # "scalar" or "vector"
            lambda_depends_on_dt: bool = False,
            dt_stats: Optional[Dict[str, float]] = None,
            activation: nn.Module = nn.SiLU(),
            dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.global_dim = global_dim
        self.use_dt_gate = use_dt_gate
        self.gate_type = gate_type
        self.lambda_depends_on_dt = lambda_depends_on_dt

        # λ(g[, dt]) → latent_dim ; non-positive via -softplus
        lam_in = global_dim + (1 if lambda_depends_on_dt else 0)
        self.lambda_net = MLP(lam_in, [max(64, latent_dim)], latent_dim, activation, dropout)

        # CRITICAL FIX: Much safer initialization for stability
        with torch.no_grad():
            for m in self.lambda_net.modules():
                if isinstance(m, nn.Linear):
                    # 10x smaller weights than before
                    nn.init.normal_(m.weight, 0.0, 0.001)  # was 0.01
                    nn.init.zeros_(m.bias)

            # Set final layer bias to near-zero decay (near identity)
            last_linear = None
            for m in self.lambda_net.modules():
                if isinstance(m, nn.Linear):
                    last_linear = m
            if last_linear is not None:
                nn.init.constant_(last_linear.bias, -0.001)  # was -0.2 (way too aggressive!)

        # b(Δt_norm,g) → latent_dim
        self.bias_net = MLP(1 + global_dim, [bias_hidden, bias_hidden], latent_dim, activation, dropout)
        with torch.no_grad():
            for m in self.bias_net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0.0, 0.001)  # safer initialization
                    nn.init.zeros_(m.bias)

        # gate(Δt_norm,g) ∈ [0,1]^(1 or Z)
        if use_dt_gate:
            gate_out = 1 if gate_type == "scalar" else latent_dim
            self.gate_net = MLP(1 + global_dim, [bias_hidden], gate_out, activation, dropout)
            with torch.no_grad():
                for m in self.gate_net.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, 0.0, 0.001)
                        nn.init.constant_(m.bias, 0.0)  # Start at 0.5 open after sigmoid
        else:
            self.gate_net = None

        # dt normalization stats (log10 space)
        if dt_stats is not None:
            self.register_buffer("dt_log_min", torch.tensor(float(dt_stats["log_min"]), dtype=torch.float32))
            self.register_buffer("dt_log_max", torch.tensor(float(dt_stats["log_max"]), dtype=torch.float32))
            self.register_buffer("ln10", torch.tensor(math.log(10.0), dtype=torch.float32))
        else:
            self.dt_log_min = None
            self.dt_log_max = None
            self.ln10 = None

    def _denorm_dt_phys(self, dt_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized dt to physical in FP32, then clamp; safe under AMP."""
        if self.dt_log_min is None or self.dt_log_max is None:
            raise RuntimeError("dt_stats missing: cannot convert normalized dt to physical")

        # Ensure valid range and FP32 math
        dtn32 = dt_norm.to(torch.float32).clamp(0.0, 1.0)
        log10_dt32 = self.dt_log_min.to(torch.float32) + dtn32 * (
                    self.dt_log_max.to(torch.float32) - self.dt_log_min.to(torch.float32))
        dt_phys32 = (10.0 ** log10_dt32).clamp_min(1e-30)

        return dt_phys32  # keep FP32; downstream ops can cast as needed

    def forward(self, z: torch.Tensor, dt: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        z: [B,Z]; dt: [B,K] or [B,K,1] normalized; g: [B,G]
        returns z_future: [B,K,Z]
        """
        B, Z = z.shape
        if dt.ndim == 1:
            dt = dt.unsqueeze(1)  # [B,1]
        if dt.ndim == 3 and dt.shape[-1] == 1:
            dt = dt.squeeze(-1)  # [B,K]
        K = dt.shape[1]

        # Broadcast
        z = z.view(B, 1, Z).expand(B, K, Z)
        gk = g.view(B, 1, -1).expand(B, K, -1)
        dt_norm = dt.view(B, K, 1)

        # Physical step size with safety
        dt_phys = self._denorm_dt_phys(dt_norm)  # [B,K,1]

        # Additional safety: clamp physical dt to reasonable range
        dt_phys = torch.clamp(dt_phys, 1e-10, 1e2)

        # λ(g[,dt])
        if self.lambda_depends_on_dt:
            lam_in = torch.cat([gk, dt_norm], dim=-1)  # [B,K,G+1]
        else:
            lam_in = gk

        lam_raw = self.lambda_net(lam_in)  # [B,K,Z]

        # SAFETY: Clamp lambda BEFORE softplus to prevent explosion
        lam_raw = torch.clamp(lam_raw, -10, 5)  # Reasonable range

        lam = -F.softplus(lam_raw)  # ≤ 0 for stability

        # SAFETY: Additional clamp on final lambda
        lam = torch.clamp(lam, -10, 0)  # Max decay rate of 10

        # exp(λ Δt) with safety check
        exponent = lam * dt_phys
        exponent = torch.clamp(exponent, -20, 0)  # Prevent overflow
        A = torch.exp(exponent)  # [B,K,Z]

        # Bias and gate
        b = self.bias_net(torch.cat([dt_norm, gk], dim=-1))  # [B,K,Z]

        # SAFETY: Clamp bias magnitude
        b = torch.clamp(b, -10, 10)
        b = b * dt_phys  # scale by physical step

        if self.gate_net is not None:
            gate_raw = self.gate_net(torch.cat([dt_norm, gk], dim=-1))  # [B,K,1|Z]
            gate = torch.sigmoid(gate_raw)
            if gate.shape[-1] == 1:
                gate = gate.expand_as(b)
            b = gate * b

        # Evolve with safety check
        z_next = A * z + b  # [B,K,Z]

        # FINAL SAFETY: Clamp output to prevent explosion
        z_next = torch.clamp(z_next, -100, 100)

        return z_next

    @torch.no_grad()
    def gate_activity(self, dt: torch.Tensor, g: torch.Tensor) -> Optional[float]:
        """Monitor mean gate value over batch/steps."""
        if self.gate_net is None:
            return None
        if dt.ndim == 1:
            dt = dt.unsqueeze(1)
        if dt.ndim == 3 and dt.shape[-1] == 1:
            dt = dt.squeeze(-1)
        B, K = dt.shape
        gk = g.view(B, 1, -1).expand(B, K, -1)
        dt_norm = dt.view(B, K, 1)
        dt_norm = torch.clamp(dt_norm, 0.0, 1.0)  # Safety clamp
        gate_raw = self.gate_net(torch.cat([dt_norm, gk], dim=-1))
        gate = torch.sigmoid(gate_raw)
        return gate.mean().item()


# ----------------------------- Decoder ---------------------------------------

class Decoder(nn.Module):
    """
    z -> y_base. Output interpretation depends on head configuration.
    """

    def __init__(
            self,
            latent_dim: int,
            hidden_dims: Sequence[int],
            state_dim: int,
            activation: nn.Module,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.network = MLP(latent_dim, hidden_dims, state_dim, activation, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


# --------------------------- Full Model --------------------------------------

class FlowMapKoopman(nn.Module):
    """
    (y_i, dt_norm, g) → z_i → Koopman(dt_norm,g) → z_j → Decoder → y_j (z-space).
    Output heads:
      - softmax_head: apply softmax in physical space, then re-normalize to z-space.
      - predict_delta: residual in z-space, or if softmax_head=True, residual in logits.
      - predict_delta_log_phys: residual in log10 physical space.
    """

    def __init__(
            self,
            *,
            state_dim_in: int,
            state_dim_out: int,
            global_dim: int,
            latent_dim: int,
            encoder_hidden: Sequence[int],
            decoder_hidden: Sequence[int],
            koopman_bias_hidden: int = 64,
            use_dt_gate: bool = True,
            gate_type: str = "scalar",
            lambda_depends_on_dt: bool = False,
            dt_stats: Optional[Dict[str, float]] = None,
            activation: str = "silu",
            dropout: float = 0.0,
            vae_mode: bool = False,
            predict_delta: bool = False,
            predict_delta_log_phys: bool = False,
            softmax_head: bool = True,
            allow_partial_simplex: bool = False,
            target_idx: Optional[torch.Tensor] = None,
            target_log_mean: Optional[Sequence[float]] = None,
            target_log_std: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        act = get_activation(activation)

        # Sanity checks for softmax + subset
        if softmax_head and (state_dim_out != state_dim_in) and not allow_partial_simplex:
            raise ValueError("softmax_head with a subset of species requires allow_partial_simplex=True")

        self.S_in = state_dim_in
        self.S_out = state_dim_out
        self.G = global_dim
        self.Z = latent_dim

        # Modules
        self.encoder = Encoder(state_dim_in, global_dim, encoder_hidden, latent_dim, act, dropout, vae_mode)
        self.dynamics = KoopmanDynamics(
            latent_dim=latent_dim,
            global_dim=global_dim,
            bias_hidden=koopman_bias_hidden,
            use_dt_gate=use_dt_gate,
            gate_type=gate_type,
            lambda_depends_on_dt=lambda_depends_on_dt,
            dt_stats=dt_stats,
            activation=act,
            dropout=dropout,
        )
        self.decoder = Decoder(latent_dim, decoder_hidden, state_dim_out, act, dropout)

        # Heads / modes
        self.predict_delta = bool(predict_delta)
        self.predict_delta_log_phys = bool(predict_delta_log_phys)
        self.softmax_head = bool(softmax_head)
        self.allow_partial_simplex = bool(allow_partial_simplex)

        # Target subset index if S_out != S_in
        if target_idx is not None and (state_dim_out != state_dim_in):
            if not isinstance(target_idx, torch.Tensor):
                target_idx = torch.tensor(target_idx, dtype=torch.long)
            self.register_buffer("target_idx", target_idx)
        else:
            self.target_idx = None

        # Stats for log-phys transforms (needed for softmax or log_phys residual)
        if self.softmax_head or self.predict_delta_log_phys:
            if target_log_mean is None or target_log_std is None:
                raise ValueError("target_log_mean/std required for softmax or predict_delta_log_phys")
            self.register_buffer("log_mean", torch.tensor(target_log_mean, dtype=torch.float32))
            self.register_buffer("log_std", torch.clamp(torch.tensor(target_log_std, dtype=torch.float32), min=1e-10))
            self.register_buffer("ln10", torch.tensor(math.log(10.0), dtype=torch.float32))
        else:
            self.log_mean = None
            self.log_std = None
            self.ln10 = None

        # If residual modes are enabled, soften last decoder layer
        if self.predict_delta or self.predict_delta_log_phys:
            with torch.no_grad():
                last = self.decoder.network.network[-1]
                if isinstance(last, nn.Linear):
                    last.weight.mul_(0.1)
                    last.bias.zero_()

        self.kl_loss: Optional[torch.Tensor] = None

    def _base_subset(self, y_i: torch.Tensor) -> torch.Tensor:
        """Return base z matching S_out (subset if needed)."""
        if self.S_out != self.S_in:
            if self.target_idx is None:
                raise RuntimeError("target_idx required for S_out != S_in")
            return y_i.index_select(1, self.target_idx)  # [B,S_out]
        return y_i  # [B,S_out]

    def forward(self, y_i: torch.Tensor, dt_norm: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # Encode
        z_i, self.kl_loss = self.encoder(y_i, g)  # [B,Z]

        # Latent evolution for each requested Δt
        z_j = self.dynamics(z_i, dt_norm, g)  # [B,K,Z]

        # Decode
        y_base = self.decoder(z_j)  # [B,K,S_out]

        # -------- Softmax head (simplex constraint) --------
        if self.softmax_head:
            ln10 = self.ln10.to(y_base.dtype)

            if self.predict_delta:
                # Residual in logits (natural log) BEFORE softmax
                base_z = self._base_subset(y_i)  # [B,S_out]
                base_log10 = self.log_mean.view(1, -1) + self.log_std.view(1, -1) * base_z
                base_log10 = base_log10.view(y_base.shape[0], 1, -1).expand(-1, y_base.shape[1], -1)
                base_ln = base_log10 * ln10

                # ---- Clamp logits before logsumexp ----
                logits = (base_ln + y_base).clamp(-40, 40)
                log_p = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
                log10_p = log_p / ln10
            else:
                # Treat decoder output as logits directly
                logits = y_base.clamp(-40, 40)  # ---- Clamp logits ----
                log_p = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
                log10_p = log_p / ln10

            mu = self.log_mean.to(log10_p.dtype).view(1, 1, -1)
            sig = self.log_std.to(log10_p.dtype).view(1, 1, -1)
            y_pred = (log10_p - mu) / sig  # back to z-space
            return y_pred

        # -------- Residual in log10-physical (z-space targets) --------
        if self.predict_delta_log_phys:
            base_z = self._base_subset(y_i)  # [B,S_out]
            base_log10 = self.log_mean.view(1, -1) + self.log_std.view(1, -1) * base_z  # [B,S_out]
            base_log10 = base_log10.unsqueeze(1).expand(-1, z_j.shape[1], -1)  # [B,K,S_out]
            pred_log10 = base_log10 + y_base  # y_base is Δ log10
            y_pred = (pred_log10 - self.log_mean.view(1, 1, -1)) / self.log_std.view(1, 1, -1)
            return y_pred

        # -------- Residual / plain in z-space --------
        if self.predict_delta:
            base = self._base_subset(y_i)  # [B,S_out]
            return base.unsqueeze(1) + y_base  # [B,K,S_out]

        return y_base  # plain decoder output in z-space

    def get_latent(self, y_i: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        z, _ = self.encoder(y_i, g)
        return z

    @torch.no_grad()
    def check_stat_consistency(self, loss_log_mean: torch.Tensor, loss_log_std: torch.Tensor) -> None:
        """Verify model stats match loss stats at startup."""
        if not (self.softmax_head or self.predict_delta_log_phys):
            return
        if self.log_mean is None or self.log_std is None:
            raise RuntimeError("Model expects log stats but none are registered")
        rtol, atol = 1e-6, 1e-9
        if not torch.allclose(self.log_mean.cpu(), loss_log_mean.cpu(), rtol=rtol, atol=atol):
            raise ValueError("log_mean mismatch between model and loss")
        if not torch.allclose(self.log_std.cpu(), loss_log_std.cpu(), rtol=rtol, atol=atol):
            raise ValueError("log_std mismatch between model and loss")

    @torch.no_grad()
    def rollout(
            self,
            y_0: torch.Tensor,
            g: torch.Tensor,
            dt_step: torch.Tensor,
            max_steps: int = 100,
    ) -> torch.Tensor:
        """
        Autoregressive rollout from y_0 using a fixed incremental normalized step.
        Args:
            y_0: [B,S_in] in z-space
            g: [B,G]
            dt_step: scalar or [B] normalized dt increment
            max_steps: number of steps
        Returns:
            Trajectory [B,T,S_out] in z-space
        """
        B = y_0.shape[0]
        if dt_step.numel() == 1:
            dt_step = dt_step.expand(B)
        traj = []
        y_curr = y_0
        for _ in range(max_steps):
            y_next = self.forward(y_curr, dt_step.unsqueeze(1), g)  # [B,1,S_out]
            y_next = y_next.squeeze(1)  # [B,S_out]
            traj.append(y_next)

            # Handle subset case
            if self.S_out != self.S_in:
                y_upd = y_curr.clone()
                y_upd[:, self.target_idx] = y_next
                y_curr = y_upd
            else:
                y_curr = y_next
        return torch.stack(traj, dim=1)  # [B,T,S_out]


# ------------------------------ Factory --------------------------------------

def create_model(config: dict) -> FlowMapKoopman:
    """
    Build FlowMapKoopman from configuration and processed-data manifest.
    Expects processed_data_dir/normalization.json with per_key_stats and dt stats.
    """
    data_cfg = config.get("data", {})
    species_vars = list(data_cfg.get("species_variables", []))
    if not species_vars:
        raise KeyError("data.species_variables must be set and non-empty")
    target_vars = list(data_cfg.get("target_species", species_vars))
    global_vars = list(data_cfg.get("global_variables", []))

    name_to_idx = {n: i for i, n in enumerate(species_vars)}
    try:
        target_idx = [name_to_idx[n] for n in target_vars]
    except KeyError as e:
        raise KeyError(f"target_species contains unknown name: {e.args[0]!r}")

    S_in = len(species_vars)
    S_out = len(target_vars)
    G = len(global_vars)

    mcfg = config.get("model", {})
    latent_dim = int(mcfg.get("latent_dim", 256))
    encoder_hidden = mcfg.get("encoder_hidden", [256, 256])
    decoder_hidden = mcfg.get("decoder_hidden", [256, 256])

    kcfg = mcfg.get("koopman", {})
    bias_hidden = int(kcfg.get("bias_hidden", 64))
    use_dt_gate = bool(kcfg.get("use_dt_gate", True))
    gate_type = str(kcfg.get("gate_type", "scalar"))
    lambda_depends_on_dt = bool(kcfg.get("lambda_depends_on_dt", False))

    activation = str(mcfg.get("activation", "silu"))
    dropout = float(mcfg.get("dropout", 0.0))
    vae_mode = bool(mcfg.get("vae_mode", False))

    predict_delta = bool(mcfg.get("predict_delta", False))
    predict_delta_log_phys = bool(mcfg.get("predict_delta_log_phys", False))
    softmax_head = bool(mcfg.get("softmax_head", True))
    allow_partial_simplex = bool(mcfg.get("allow_partial_simplex", False))

    # Load normalization manifest for dt and per-key log stats
    proc_dir = Path(config["paths"]["processed_data_dir"])
    manifest_path = proc_dir / "normalization.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    dt_stats = None
    if "dt" in manifest:
        dtm = manifest["dt"]
        dt_stats = {"log_min": float(dtm["log_min"]), "log_max": float(dtm["log_max"])}

    need_stats = softmax_head or predict_delta_log_phys
    target_log_mean = None
    target_log_std = None
    if need_stats:
        stats = manifest["per_key_stats"]
        target_log_mean = []
        target_log_std = []
        for n in target_vars:
            if n not in stats:
                raise KeyError(f"Target species '{n}' missing in manifest per_key_stats")
            s = stats[n]
            target_log_mean.append(float(s.get("log_mean", 0.0)))
            target_log_std.append(float(s.get("log_std", 1.0)))

    model = FlowMapKoopman(
        state_dim_in=S_in,
        state_dim_out=S_out,
        global_dim=G,
        latent_dim=latent_dim,
        encoder_hidden=encoder_hidden,
        decoder_hidden=decoder_hidden,
        koopman_bias_hidden=bias_hidden,
        use_dt_gate=use_dt_gate,
        gate_type=gate_type,
        lambda_depends_on_dt=lambda_depends_on_dt,
        dt_stats=dt_stats,
        activation=activation,
        dropout=dropout,
        vae_mode=vae_mode,
        predict_delta=predict_delta,
        predict_delta_log_phys=predict_delta_log_phys,
        softmax_head=softmax_head,
        allow_partial_simplex=allow_partial_simplex,
        target_idx=torch.tensor(target_idx, dtype=torch.long) if S_out != S_in else None,
        target_log_mean=target_log_mean,
        target_log_std=target_log_std,
    )

    # Optionally check stat consistency at creation
    if need_stats:
        model.check_stat_consistency(
            torch.tensor(target_log_mean),
            torch.tensor(target_log_std)
        )

    return model
#!/usr/bin/env python3
"""
model.py — Linear Latent Network (LiLaN) with optional mixture-of-experts and learned time warping.

Core:
    z(t) = E(x0, p) + τ(t, ·) ∘ C(x0, p)      (constant-velocity latent dynamics)
    y(t) = D(z(t), p)

Supports two architectural variants:
  • "full": Single model for all output dimensions (shared encoder/decoder)
  • "independent": Separate model for each output dimension
"""

import logging
import math
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- helpers -----------------------------

def _get_activation(name: str) -> nn.Module:
    name = (name or "gelu").lower()
    table = {
        "gelu": nn.GELU(),
        "relu": nn.ReLU(inplace=True),
        "tanh": nn.Tanh(),
        "silu": nn.SiLU(inplace=True),
        "elu": nn.ELU(inplace=True),
    }
    return table.get(name, nn.GELU())


def _build_mlp(
    in_dim: int,
    hidden: List[int],
    out_dim: int,
    activation: nn.Module,
    dropout_p: float = 0.0,
    use_layernorm: bool = True,
    final_activation: bool = False,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for i, h in enumerate(hidden):
        if i > 0 and use_layernorm:
            layers.append(nn.LayerNorm(prev))
        layers += [nn.Linear(prev, h), activation]
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))
        prev = h
    if hidden and use_layernorm:
        layers.append(nn.LayerNorm(prev))
    layers.append(nn.Linear(prev, out_dim))
    if final_activation:
        layers.append(activation)
    return nn.Sequential(*layers)


# ----------------------------- time warp -----------------------------

class TimeWarp(nn.Module):
    """
    Monotonic per-latent time warp:
        τ_d(t) = s_d * t + Σ_{j=1..J} a_{d,j} * (1 - exp(-b_{d,j} * t))
    with s, a, b constrained positive via softplus.
    """
    def __init__(self, context_dim: int, latent_dim: int, J_terms: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.J_terms = J_terms
        out_dim = latent_dim * (1 + 2 * J_terms)  # s, {a_j}, {b_j}
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, t_norm: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        t_norm:  [B, M] in [0,1]
        context: [B, context_dim] (typically encoder features)
        returns: τ(t) of shape [B, M, D]
        """
        B, M = t_norm.shape
        params = self.net(context).view(B, self.latent_dim, 1 + 2 * self.J_terms)
        s = F.softplus(params[:, :, 0:1])                  # [B, D, 1]
        a = F.softplus(params[:, :, 1:1 + self.J_terms])   # [B, D, J]
        b = F.softplus(params[:, :, 1 + self.J_terms:])    # [B, D, J]

        t_exp = t_norm.unsqueeze(1).expand(B, self.latent_dim, M)  # [B, D, M]
        tau = s * t_exp
        # expm1 for stability: 1 - exp(-b t) = -expm1(-b t)
        exp_terms = (a.unsqueeze(-1) * (-torch.expm1(-b.unsqueeze(-1) * t_exp.unsqueeze(2)))).sum(dim=2)
        return (tau + exp_terms).transpose(1, 2)  # [B, M, D]


# ----------------------------- Base LiLaN model -----------------------------

class LinearLatentMixture(nn.Module):
    """
    LiLaN with optional mixture-of-experts and learned time warp.
    Inputs:  [x0_log (S), globals (G), times (M)]  →  outputs: [B, M, T]
    """
    def __init__(self, config: Dict[str, Any], output_dim: Optional[int] = None):
        super().__init__()
        self.log = logging.getLogger(__name__)
        data_cfg = config["data"]
        model_cfg = config["model"]

        # dims
        self.num_species = len(data_cfg["species_variables"])
        self.num_globals = len(data_cfg["global_variables"])
        # Allow override for independent models
        if output_dim is not None:
            self.num_targets = output_dim
        else:
            self.num_targets = len(data_cfg.get("target_species_variables", data_cfg["species_variables"]))
        self.latent_dim = int(model_cfg.get("latent_dim", 64))

        # activations / dropout
        self.activation = _get_activation(model_cfg.get("activation", "gelu"))
        self.dropout_p = float(model_cfg.get("dropout", 0.0))

        # mixture
        mix_cfg = model_cfg.get("mixture", {})
        self.K = int(mix_cfg.get("K", 1))
        self.gate_temp = float(mix_cfg.get("temperature", 1.0))
        # regularizer settings
        self.diversity_mode = str(mix_cfg.get("diversity_mode", "per_sample"))
        self.full_pair_threshold = int(mix_cfg.get("full_pair_threshold", 8))
        self.sample_factor = int(mix_cfg.get("sample_factor", 8))
        self.sample_pairs: Optional[int] = mix_cfg.get("sample_pairs", None)
        self.gate_use_features = bool(mix_cfg.get("use_encoder_features", True))

        # time warp
        tw_cfg = model_cfg.get("time_warp", {})
        self.use_time_warp = bool(tw_cfg.get("enabled", False))
        self.warp_use_features = bool(tw_cfg.get("use_encoder_features", True))
        self.warp_J = int(tw_cfg.get("J_terms", 3))
        self.warp_hidden = int(tw_cfg.get("hidden_dim", 64))

        # encoder backbone
        enc_layers = list(model_cfg.get("encoder_layers", [256, 256, 128]))
        enc_in = self.num_species + self.num_globals
        enc_out = enc_layers[-1] if enc_layers else enc_in
        self.encoder = _build_mlp(
            in_dim=enc_in, hidden=enc_layers, out_dim=enc_out,
            activation=self.activation, dropout_p=self.dropout_p,
            use_layernorm=True, final_activation=False,
        )

        # heads E (y0) and C (velocity)
        self.y0_head = nn.Linear(enc_out, self.latent_dim * self.K)
        self.c_head = nn.Linear(enc_out, self.latent_dim * self.K)

        # gate (optionally using encoder features)
        if self.K > 1:
            gate_layers = list(mix_cfg.get("gate_layers", [64, 32]))
            self.gate = _build_mlp(
                in_dim=(enc_out if self.gate_use_features else enc_in),
                hidden=gate_layers, out_dim=self.K,
                activation=self.activation, dropout_p=self.dropout_p,
                use_layernorm=bool(mix_cfg.get("gate_use_layernorm", False)),
                final_activation=False,
            )

        # time warp (context: encoder features or raw encoder input)
        if self.use_time_warp:
            warp_context_dim = enc_out if self.warp_use_features else enc_in
            self.time_warp = TimeWarp(warp_context_dim, self.latent_dim, self.warp_J, self.warp_hidden)

        # decoder over [z(t), globals]
        dec_layers = list(model_cfg.get("decoder_layers", [128, 256, 256]))
        dec_in = self.latent_dim + self.num_globals
        self.decoder = _build_mlp(
            in_dim=dec_in, hidden=dec_layers, out_dim=self.num_targets,
            activation=self.activation, dropout_p=self.dropout_p,
            use_layernorm=True, final_activation=False,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        # Xavier is a safe default across activations
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Gate: uniform at init (softmax(0)=uniform)
        if self.K > 1:
            for m in self.gate.modules():
                if isinstance(m, nn.Linear) and m.out_features == self.K:
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                    break

        # TimeWarp: start as identity-ish (no context dependence)
        if self.use_time_warp:
            last_linear = None
            for m in self.time_warp.net.modules():
                if isinstance(m, nn.Linear):
                    last_linear = m
            if last_linear is not None:
                nn.init.zeros_(last_linear.weight)
                with torch.no_grad():
                    J, D = self.time_warp.J_terms, self.latent_dim
                    bias = last_linear.bias.view(D, 1 + 2 * J)
                    bias[:, 0] = 0.54      # softplus(0.54) ~ 1  => s ≈ 1
                    bias[:, 1:1 + J] = -5  # softplus(-5)  ~ 0  => a ≈ 0
                    bias[:, 1 + J:] = 0.54 # softplus(0.54) ~ 1  => b ≈ 1

    def set_gate_temperature(self, temperature: float) -> None:
        if self.K > 1:
            self.gate_temp = max(1e-3, float(temperature))

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        inputs: [B, S + G + M] where
            S = num_species (already log-transformed in data pipeline)
            G = num_globals (raw, normalized by NormalizationHelper)
            M = number of normalized time points in [0,1]
        returns:
            predictions: [B, M, T]
            aux: dict with 'gate_p' and 'c_all' (for regularization)
        """
        B = inputs.size(0)
        x0_log = inputs[:, :self.num_species]
        g_vec = inputs[:, self.num_species:self.num_species + self.num_globals]
        t_norm = inputs[:, self.num_species + self.num_globals:]  # [B, M]
        M = t_norm.size(1)

        enc_input = torch.cat([x0_log, g_vec], dim=1)             # [B, S+G]
        features = self.encoder(enc_input)                         # [B, enc_out]

        y0_all = self.y0_head(features)                           # [B, K*D]
        c_all = self.c_head(features)                             # [B, K*D]

        # τ(t)
        if self.use_time_warp:
            warp_ctx = features if self.warp_use_features else enc_input
            tau = self.time_warp(t_norm, warp_ctx)                # [B, M, D]
        else:
            tau = t_norm.unsqueeze(-1).expand(B, M, self.latent_dim)

        aux: Dict[str, torch.Tensor] = {}

        if self.K > 1:
            y0_all = y0_all.view(B, self.K, self.latent_dim)      # [B, K, D]
            c_all = c_all.view(B, self.K, self.latent_dim)        # [B, K, D]

            gate_ctx = features if self.gate_use_features else enc_input
            logits = self.gate(gate_ctx)                          # [B, K]
            probs = F.softmax(logits / max(self.gate_temp, 1e-3), dim=-1)  # [B, K]

            aux["gate_p"] = probs
            aux["c_all"] = c_all

            # z_k(t) and gated blend
            z_all = y0_all.unsqueeze(2) + tau.unsqueeze(1) * c_all.unsqueeze(2)  # [B, K, M, D]
            z_t = (z_all * probs.view(B, self.K, 1, 1)).sum(dim=1)               # [B, M, D]
        else:
            y0 = y0_all.view(B, self.latent_dim)
            c = c_all.view(B, self.latent_dim)
            z_t = y0.unsqueeze(1) + tau * c.unsqueeze(1)          # [B, M, D]

        # decode
        g_rep = g_vec.unsqueeze(1).expand(B, M, self.num_globals)                 # [B, M, G]
        dec_in = torch.cat([z_t, g_rep], dim=-1).reshape(B * M, -1)               # [B*M, D+G]
        out = self.decoder(dec_in).view(B, M, self.num_targets)                   # [B, M, T]
        return out, aux

    def get_regularization_losses(self, aux: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Returns unweighted regularizers to be added by the Trainer:
          • gate_kl_to_uniform
          • generator_similarity
        """
        losses: Dict[str, torch.Tensor] = {}
        if not aux or self.K == 1:
            return losses

        # Gate KL(p || Uniform) = E_p[log p] + log K
        p = aux["gate_p"].clamp_min(1e-8)
        losses["gate_kl_to_uniform"] = (p * (p.log() + math.log(self.K))).sum(dim=-1).mean()

        # Generator diversity (actually C vectors diversity)
        c_all = aux["c_all"]                      # [B, K, D]
        if self.diversity_mode == "batch_mean":
            w = F.normalize(c_all.mean(dim=0), dim=-1)   # [K, D]
            sim = w @ w.T                                 # [K, K]
            i, j = torch.triu_indices(self.K, self.K, offset=1, device=sim.device)
            sims = sim[i, j].clamp_min(0)
            losses["generator_similarity"] = sims.mean() if sims.numel() else sim.new_tensor(0.0)
            return losses

        # per-sample view (default)
        w = F.normalize(c_all, dim=-1)                   # [B, K, D]
        sim = torch.einsum("bkd,bld->bkl", w, w)         # [B, K, K]
        i, j = torch.triu_indices(self.K, self.K, offset=1, device=sim.device)
        num_pairs = (self.K * (self.K - 1)) // 2

        if self.K <= self.full_pair_threshold or num_pairs == 0:
            sims = sim[:, i, j]                          # [B, P]
        else:
            if self.sample_pairs is None:
                S = min(self.sample_factor * self.K, num_pairs)
            else:
                S = min(int(self.sample_pairs), num_pairs)
            choice = torch.randint(0, num_pairs, (S,), device=sim.device)
            sims = sim[:, i[choice], j[choice]]          # [B, S]

        sims = sims.clamp_min(0)
        per_sample = sims.mean(dim=1)                    # [B]
        losses["generator_similarity"] = per_sample.mean()
        return losses


# ----------------------------- Independent Architecture -----------------------------

class IndependentLinearLatentMixture(nn.Module):
    """
    Independent architecture: separate LiLaN model for each output dimension.
    This typically performs better than the full architecture.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.log = logging.getLogger(__name__)
        
        data_cfg = config["data"]
        self.target_vars = data_cfg.get("target_species_variables", data_cfg["species_variables"])
        self.num_targets = len(self.target_vars)
        
        # Create independent model for each target
        self.models = nn.ModuleList()
        for i in range(self.num_targets):
            # Each model predicts only one output dimension
            model = LinearLatentMixture(config, output_dim=1)
            self.models.append(model)
        
        self.log.info(f"Created independent architecture with {self.num_targets} separate models")
    
    def set_gate_temperature(self, temperature: float) -> None:
        """Set gate temperature for all sub-models."""
        for model in self.models:
            model.set_gate_temperature(temperature)
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Run each model independently and concatenate outputs.
        
        inputs: [B, S + G + M]
        returns:
            predictions: [B, M, T] where T = num_targets
            aux: combined auxiliary data from all models
        """
        outputs = []
        all_aux = {}
        
        for i, model in enumerate(self.models):
            out, aux = model(inputs)  # out: [B, M, 1]
            outputs.append(out)
            
            # Combine auxiliary data with model index
            for key, val in aux.items():
                if key not in all_aux:
                    all_aux[key] = []
                all_aux[key].append(val)
        
        # Concatenate outputs
        predictions = torch.cat(outputs, dim=-1)  # [B, M, T]
        
        # Stack auxiliary tensors
        for key in all_aux:
            if len(all_aux[key]) > 0:
                all_aux[key] = torch.stack(all_aux[key], dim=0)  # [T, ...]
        
        return predictions, all_aux
    
    def get_regularization_losses(self, aux: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute regularization losses for all sub-models.
        """
        losses = {}
        
        # Average regularization losses across all models
        if "gate_p" in aux:
            gate_losses = []
            for i in range(self.num_targets):
                if aux["gate_p"].size(0) > i:
                    p = aux["gate_p"][i].clamp_min(1e-8)
                    K = p.size(-1)
                    gate_loss = (p * (p.log() + math.log(K))).sum(dim=-1).mean()
                    gate_losses.append(gate_loss)
            if gate_losses:
                losses["gate_kl_to_uniform"] = torch.stack(gate_losses).mean()
        
        if "c_all" in aux:
            sim_losses = []
            for i in range(self.num_targets):
                if aux["c_all"].size(0) > i:
                    c = aux["c_all"][i]
                    if c.dim() == 3 and c.size(1) > 1:  # [B, K, D] with K > 1
                        w = F.normalize(c, dim=-1)
                        sim = torch.einsum("bkd,bld->bkl", w, w)
                        K = c.size(1)
                        i_idx, j_idx = torch.triu_indices(K, K, offset=1, device=sim.device)
                        if i_idx.numel() > 0:
                            sims = sim[:, i_idx, j_idx].clamp_min(0)
                            sim_losses.append(sims.mean())
            if sim_losses:
                losses["generator_similarity"] = torch.stack(sim_losses).mean()
        
        return losses


# ----------------------------- Factory -----------------------------

def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """
    Factory for LiLaN models. Supports:
      • model.variant: "full" (default) or "independent"
      • model.type: "linear_latent" or "linear_latent_mixture"
      • optional torch.compile
      • dtype control (float64)
    """
    model_cfg = config["model"]
    model_type = model_cfg.get("type", "linear_latent_mixture").lower()
    model_variant = model_cfg.get("variant", "full").lower()
    
    if model_type not in {"linear_latent", "linear_latent_mixture"}:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if model_variant not in {"full", "independent"}:
        raise ValueError(f"Unknown model variant: {model_variant}. Choose 'full' or 'independent'")
    
    # Create model based on variant
    if model_variant == "independent":
        model = IndependentLinearLatentMixture(config)
        variant_str = "independent"
    else:
        model = LinearLatentMixture(config)
        variant_str = "full"
    
    # dtype
    if str(config["system"].get("dtype", "float32")).lower() == "float64":
        model = model.double()
    
    model = model.to(device)
    
    # optional compile
    logger = logging.getLogger(__name__)
    if config["system"].get("use_torch_compile", False) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=config["system"].get("compile_mode", "default"))
            logger.info("Model compilation successful.")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}. Running in eager mode.")
    
    # log summary
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Created {variant_str} {model_type}: params={n_params:,}")
    
    return model
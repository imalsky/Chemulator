#!/usr/bin/env python3
"""
Linear Latent Network models with constant-velocity latent dynamics and time warping.
LiLaN form: z(t) = E(x0, p) + τ(t, x0, p) ∘ C(x0, p); outputs are y(t) = D([z(t), p]).
"""

import logging
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeWarp(nn.Module):
    """Monotone time-warping module for multi-dimensional time τ(t)."""
    def __init__(
        self,
        n_species: int,
        n_globals: int,
        latent_dim: int,
        J_terms: int = 3,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.J_terms = J_terms
        self.latent_dim = latent_dim

        # Predict per-latent-dim parameters s, {a_j, b_j}
        input_dim = n_species + n_globals
        output_dim = latent_dim * (1 + 2 * J_terms)

        self.param_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, t_norm: torch.Tensor, x0_log: torch.Tensor, g_params: torch.Tensor) -> torch.Tensor:
        """
        Apply monotone time warp to produce multi-dimensional time.

        Args:
            t_norm: Normalized time in [0, 1], shape [B, M]
            x0_log: Initial log-species, shape [B, n_species]
            g_params: Global parameters, shape [B, n_globals]
        Returns:
            Warped time tau(t), shape [B, M, latent_dim]
        """
        B = x0_log.size(0)
        M = t_norm.size(1)

        # Predict warp params
        features = torch.cat([x0_log, g_params], dim=1)
        params = self.param_net(features).view(B, self.latent_dim, 1 + 2 * self.J_terms)

        # Positive parameters via softplus
        s = F.softplus(params[:, :, 0:1])                       # [B, D, 1]
        a = F.softplus(params[:, :, 1:1+self.J_terms])          # [B, D, J]
        b = F.softplus(params[:, :, 1+self.J_terms:])           # [B, D, J]

        # Broadcast time
        t_norm_expanded = t_norm.unsqueeze(1).expand(B, self.latent_dim, M)  # [B, D, M]

        # tau(t) = s*t + Σ_j a_j * (1 - exp(-b_j * t))
        # Use expm1 for better precision when b*t is small: 1 - exp(-x) = -expm1(-x)
        tau = s.expand(B, self.latent_dim, M) * t_norm_expanded
        t_be = t_norm_expanded.unsqueeze(2)                     # [B, D, 1, M]
        a_be = a.unsqueeze(-1)                                  # [B, D, J, 1]
        b_be = b.unsqueeze(-1)                                  # [B, D, J, 1]
        tau = tau + (a_be * (-torch.expm1(-b_be * t_be))).sum(dim=2)  # sum over J

        # [B, M, D]
        return tau.transpose(1, 2)


class LinearLatentMixture(nn.Module):
    """
    Linear Latent Network with constant-velocity dynamics and optional
    mixture-of-experts & monotone time-warp.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # ----- dimensions --------------------------------------------------
        self.num_species = len(config["data"]["species_variables"])
        self.num_globals = len(config["data"]["global_variables"])
        self.num_targets = len(
            config["data"].get("target_species_variables", config["data"]["species_variables"])
        )

        # ----- model-level hyper-params ------------------------------------
        m_cfg = config["model"]
        self.latent_dim = m_cfg.get("latent_dim", 64)
        self.K = m_cfg.get("mixture", {}).get("K", 1)
        self.use_time_warp = m_cfg.get("time_warp", {}).get("enabled", False)

        # layer sizes / activations
        self.encoder_layers = m_cfg.get("encoder_layers", [256, 256, 128])
        self.decoder_layers = m_cfg.get("decoder_layers", [128, 256, 256])
        self.activation = self._get_activation(m_cfg.get("activation", "gelu"))
        dropout_rate = m_cfg.get("dropout", 0.0)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        # ----- shared encoder backbone + heads -----------------------------
        enc_in = self.num_species + self.num_globals
        back_out = self.encoder_layers[-1] if self.encoder_layers else enc_in

        self.encoder_backbone = self._build_mlp(
            enc_in, self.encoder_layers, back_out, use_layernorm=True
        )
        self.y0_head = nn.Linear(back_out, self.latent_dim * self.K)
        self.c_head = nn.Linear(back_out, self.latent_dim * self.K)

        # ----- mixture gate -----------------------------------------------
        if self.K > 1:
            gate_layers = m_cfg.get("mixture", {}).get("gate_layers", [64, 32])
            layers, prev = [], enc_in
            for dim in gate_layers:
                layers.extend([nn.Linear(prev, dim), self.activation])
                prev = dim
            layers.append(nn.Linear(prev, self.K))
            self.gate_net = nn.Sequential(*layers)

            mix_cfg = m_cfg.get("mixture", {})
            self.gate_temperature = float(mix_cfg.get("temperature", 1.0))
            div_cfg = mix_cfg.get("diversity", {})
            self.full_pair_threshold = div_cfg.get("full_pair_threshold", 8)
            self.sample_factor = div_cfg.get("sample_factor", 8)

        # ----- time-warp module -------------------------------------------
        if self.use_time_warp:
            tw_cfg = m_cfg.get("time_warp", {})
            J_terms = tw_cfg.get("J_terms", 3)
            tw_hidden = tw_cfg.get("hidden_dim", 64)
            self.time_warp = TimeWarp(
                self.num_species, self.num_globals, self.latent_dim,
                J_terms=J_terms, hidden_dim=tw_hidden
            )

        # ----- decoder -----------------------------------------------------
        dec_in = self.latent_dim + self.num_globals
        self.decoder = self._build_mlp(
            dec_in, self.decoder_layers, self.num_targets,
            use_layernorm=True, final_activation=False
        )

        self._initialize_weights()

    def set_gate_temperature(self, temp: float):
        self.gate_temperature = float(max(1e-3, temp))

    def _get_activation(self, name: str):
        """Get activation function by name."""
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(inplace=True),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(inplace=True),
            "elu": nn.ELU(inplace=True)
        }
        return activations.get(name.lower(), nn.GELU())

    def _build_mlp(self, input_dim: int, hidden_layers: List[int],
                   output_dim: int, use_layernorm: bool = True,
                   final_activation: bool = True) -> nn.Sequential:
        """Build a multi-layer perceptron."""
        layers = []
        prev_dim = input_dim

        for i, dim in enumerate(hidden_layers):
            if i > 0 and use_layernorm:
                layers.append(nn.LayerNorm(prev_dim))

            layers.append(nn.Linear(prev_dim, dim))
            layers.append(self.activation)

            if self.dropout is not None:
                layers.append(self.dropout)

            prev_dim = dim

        # Output layer
        if use_layernorm:
            layers.append(nn.LayerNorm(prev_dim))
        layers.append(nn.Linear(prev_dim, output_dim))

        if final_activation:
            layers.append(self.activation)

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights for stability (encoders/decoder + TimeWarp if present)."""
        # Backbone and projection heads
        for mod in [self.encoder_backbone, self.y0_head, self.c_head, self.decoder]:
            for m in mod.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # TimeWarp (if enabled): Xavier on all linear layers, then set final bias via softplus-inverse
        if self.use_time_warp:
            # 1) Initialize all TimeWarp linear layers
            for m in self.time_warp.param_net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

            # 2) Overwrite the LAST layer so output == bias at init (input-independent)
            last = self.time_warp.param_net[-1]
            if isinstance(last, nn.Linear):
                # Make the last layer produce only the bias initially
                nn.init.zeros_(last.weight)
                if last.bias is None:
                    last.bias = nn.Parameter(
                        torch.zeros(self.latent_dim * (1 + 2 * self.time_warp.J_terms),
                                    dtype=last.weight.dtype, device=last.weight.device)
                    )

                with torch.no_grad():
                    J = self.time_warp.J_terms

                    # Desired *post-softplus* values (near identity)
                    # s ≈ 1, a_j ≈ 0.01 (small), b_j ≈ 1
                    desired_s = torch.full((self.latent_dim, 1), 1.0,
                                           dtype=last.weight.dtype, device=last.weight.device)
                    desired_a = torch.full((self.latent_dim, J), 0.01,
                                           dtype=last.weight.dtype, device=last.weight.device)
                    desired_b = torch.full((self.latent_dim, J), 1.0,
                                           dtype=last.weight.dtype, device=last.weight.device)

                    # Stable inverse softplus: x = log(exp(y) - 1)
                    def softplus_inv(y: torch.Tensor) -> torch.Tensor:
                        return torch.where(
                            y > 20,
                            y,  # for large y, softplus ~ y
                            torch.log(torch.expm1(y))
                        )

                    bias_view = last.bias.view(self.latent_dim, 1 + 2 * J)
                    bias_view[:, 0:1] = softplus_inv(desired_s)  # s pre-activations
                    bias_view[:, 1:1+J] = softplus_inv(desired_a)  # a_j pre-activations
                    bias_view[:, 1+J:1+2*J] = softplus_inv(desired_b)  # b_j pre-activations

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args
            inputs : [B, n_species + n_globals + M]  (last M entries are τ-normalised time)
        Returns
            Tuple of:
            - log-abundance predictions : [B, M, n_targets]
            - auxiliary dict with gate probabilities and velocities (if K > 1)
        """
        B = inputs.size(0)

        # -------- split inputs -------------------------------------------
        x0_log = inputs[:, :self.num_species]
        globals_vec = inputs[:, self.num_species : self.num_species + self.num_globals]
        t_norm = inputs[:, self.num_species + self.num_globals :]  # [B, M]
        M = t_norm.size(1)

        # -------- shared encoder -----------------------------------------
        enc_in = torch.cat([x0_log, globals_vec], dim=1)
        features = self.encoder_backbone(enc_in)
        y0_all = self.y0_head(features)  # [B, K*D] or [B, D]
        c_all = self.c_head(features)    # same

        # -------- time-warp ----------------------------------------------
        if self.use_time_warp:
            tau = self.time_warp(t_norm, x0_log, globals_vec)  # [B, M, D]
        else:
            tau = t_norm.unsqueeze(-1).expand(B, M, self.latent_dim)  # [B, M, D]

        # -------- mixture / single component -----------------------------
        aux = {}
        if self.K > 1:
            y0_all = y0_all.view(B, self.K, self.latent_dim)
            c_all = c_all.view(B, self.K, self.latent_dim)

            logits = self.gate_net(enc_in)
            p = F.softmax(logits / self.gate_temperature, dim=-1)  # [B, K]
            
            # Store auxiliary data for regularization
            aux["gate_p"] = p
            aux["c_all"] = c_all

            z_t = (
                (y0_all.unsqueeze(2) + tau.unsqueeze(1) * c_all.unsqueeze(2))  # [B, K, M, D]
                * p.unsqueeze(-1).unsqueeze(-1)
            ).sum(dim=1)  # [B, M, D]
        else:
            y0 = y0_all.view(B, self.latent_dim)
            c = c_all.view(B, self.latent_dim)
            z_t = y0.unsqueeze(1) + tau * c.unsqueeze(1)  # [B, M, D]

        # -------- decode --------------------------------------------------
        globals_exp = globals_vec.unsqueeze(1).expand(B, M, self.num_globals)
        dec_in = torch.cat([z_t, globals_exp], dim=-1)  # [B, M, D+G]
        out_flat = self.decoder(dec_in.reshape(B * M, -1))  # [B*M, T]
        
        return out_flat.view(B, M, self.num_targets), aux

    def get_regularization_losses(self, aux: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Returns dict of positive regularization terms.
        
        Args:
            aux: Auxiliary data from forward pass containing gate_p and c_all
        Returns:
            Dictionary with regularization losses
        """
        losses = {}
        
        if not aux or self.K == 1:
            return losses
        
        # Gate entropy penalty: KL(p || Uniform)
        p = aux["gate_p"]  # [B, K]
        K = p.size(-1)
        probs = p.clamp_min(1e-8)
        kl = (probs * (probs.log() + math.log(K))).sum(dim=-1).mean()
        losses["gate_kl_to_uniform"] = kl  # ≥0
        
        # Generator diversity penalty (mean cosine similarity between directions)
        c_all = aux["c_all"]  # [B, K, D]
        # Average over batch to get mean direction vectors
        w = F.normalize(c_all.mean(dim=0), dim=-1)  # [K, D]
        cos = (w @ w.T).triu(1)
        mean_sim = cos[cos != 0].mean() if cos.numel() > 0 else torch.tensor(0.0)
        losses["generator_similarity"] = mean_sim.clamp_min(0)
        
        return losses


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create model based on configuration."""
    model_type = config["model"]["type"].lower()

    if model_type in ["linear_latent", "linear_latent_mixture"]:
        model = LinearLatentMixture(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Set precision
    dtype_str = config["system"].get("dtype", "float32")
    if dtype_str == "float64":
        model = model.double()

    model = model.to(device)

    logger = logging.getLogger(__name__)
    logger.info(f"Created {model_type} model with constant velocity dynamics")
    logger.info(f"  Mixture components K: {model.K}")
    logger.info(f"  Time warping: {model.use_time_warp}")
    logger.info(f"  Latent dimension: {model.latent_dim}")

    # Compile if requested
    if config["system"].get("use_torch_compile", False) and hasattr(torch, 'compile'):
        compile_mode = config["system"].get("compile_mode", "default")
        try:
            model = torch.compile(model, mode=compile_mode, fullgraph=False, dynamic=False)
            logger.info("Model compilation successful")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}. Running in eager mode.")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params:,}")

    return model
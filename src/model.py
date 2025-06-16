#!/usr/bin/env python3
"""
model.py - A state-evolution predictor using a Multi-Layer Perceptron (MLP).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

import utils

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")

class StateEvolutionPredictor(nn.Module):
    def __init__(self, num_species: int, num_global_vars: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        
        input_dim = num_species + num_global_vars + 1
        output_dim = num_species

        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            current_dim = h_dim
            
        self.mlp_body = nn.Sequential(*layers)
        self.output_proj = nn.Linear(current_dim, output_dim)
        
        self._init_parameters()

    def _init_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        hidden_output = self.mlp_body(x)
        prediction = self.output_proj(hidden_output)
        return prediction

def create_prediction_model(
    config: Dict[str, Any], device: Optional[Union[str, torch.device]] = None
) -> StateEvolutionPredictor:
    logger.info("Configuration for MLP model loaded.")
    
    model = StateEvolutionPredictor(
        num_species=len(config["species_variables"]),
        num_global_vars=len(config["global_variables"]),
        hidden_dims=config.get("hidden_dims", [256, 256]),
        dropout=config.get("dropout", 0.1),
    )
    
    logger.info("StateEvolutionPredictor (MLP) instance created.")
    if device:
        model.to(torch.device(device))
        logger.info(f"Model moved to device: {device}")
    
    return model

__all__ = ["StateEvolutionPredictor", "create_prediction_model"]
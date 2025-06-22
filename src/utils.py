#!/usr/bin/env python3
"""
utils.py - A collection of shared helper functions for the project.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import sys

try:
    import json5 as _json_backend
    _HAS_JSON5 = True
except ImportError:
    _json_backend = json
    _HAS_JSON5 = False

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s"
DEFAULT_SEED = 42
UTF8_ENCODING = "utf-8"
UTF8_SIG_ENCODING = "utf-8-sig"

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO, log_file: Optional[Union[str, Path]] = None) -> None:
    """Sets up logging to console and an optional file, avoiding duplicate handlers."""
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
    root_logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, mode="a", encoding=UTF8_ENCODING)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logger.info(f"Logging to console and file: {log_path.resolve()}")
        except Exception as e:
            logger.error(f"File logging setup failed for {log_file}: {e}. Continuing with console only.")
    else:
        logger.info("Logging to console only.")

def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Loads and validates a configuration file (supports JSON with comments via JSON5)."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        config_text = config_path.read_text(encoding=UTF8_SIG_ENCODING)
        config_dict = _json_backend.loads(config_text)
        validate_config(config_dict)
        backend = "JSON5" if _HAS_JSON5 else "JSON"
        logger.info(f"Successfully loaded and validated {backend} config from {config_path}.")
        return config_dict
    except Exception as e:
        raise RuntimeError(f"Failed to load or validate config {config_path}: {e}") from e

def validate_config(config: Dict[str, Any]) -> None:
    """Validates the nested structure and key values of the configuration dictionary."""
    data_spec = config.get("data_specification")
    if not isinstance(data_spec, dict):
        raise ValueError("Config section 'data_specification' is missing or not a dictionary.")
    if not isinstance(data_spec.get("species_variables"), list) or not data_spec.get("species_variables"):
        raise ValueError("Config key 'species_variables' in 'data_specification' must be a non-empty list.")
    if not isinstance(data_spec.get("all_variables"), list) or not data_spec.get("all_variables"):
        raise ValueError("Config key 'all_variables' in 'data_specification' must be a non-empty list.")
    model_params = config.get("model_hyperparameters")
    if not isinstance(model_params, dict):
        raise ValueError("Config section 'model_hyperparameters' is missing or not a dictionary.")
    if not isinstance(model_params.get("hidden_dims"), list) or not model_params.get("hidden_dims"):
        raise ValueError("Config key 'hidden_dims' in 'model_hyperparameters' must be a non-empty list.")
    dropout = model_params.get("dropout", -1.0)
    if not isinstance(dropout, (float, int)) or not (0.0 <= dropout < 1.0):
        raise ValueError("'dropout' in 'model_hyperparameters' must be a float in the range [0.0, 1.0).")

def ensure_dirs(*paths: Union[str, Path]) -> bool:
    """Creates one or more directories if they do not already exist."""
    try:
        for path in paths: 
            Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except OSError as e:
        logger.error(f"Failed to create directories {paths}: {e}")
        return False

def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for handling numpy, torch, Path, and set objects."""
    if isinstance(obj, (np.integer, np.floating)): return obj.item()
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, torch.Tensor): return obj.detach().cpu().tolist()
    if isinstance(obj, Path): return str(obj)
    if isinstance(obj, set): return sorted(list(obj))
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable.")

def save_json(data: Dict[str, Any], path: Union[str, Path]) -> bool:
    """Saves a dictionary to a JSON file with pretty printing and robust serialization."""
    try:
        json_path = Path(path)
        if not ensure_dirs(json_path.parent): return False
        with json_path.open("w", encoding=UTF8_ENCODING) as f:
            json.dump(data, f, indent=2, default=_json_serializer, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON to {path}: {e}", exc_info=True)
        return False

def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Sets the random seed for Python, NumPy, and PyTorch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Global random seed set to {seed}.")

# CORRECTED: A much more robust parser for chemical formulas.
def _parse_formula_recursive(formula: str) -> Counter[str]:
    """Recursively parses a chemical formula, handling parentheses."""
    counts: Counter[str] = Counter()
    i = 0
    while i < len(formula):
        # Handle parentheses
        if formula[i] == '(':
            j = i
            balance = 1
            while balance > 0:
                j += 1
                if formula[j] == '(': balance += 1
                if formula[j] == ')': balance -= 1
            sub_counts = _parse_formula_recursive(formula[i+1:j])
            i = j + 1
            num_str = ''
            while i < len(formula) and formula[i].isdigit():
                num_str += formula[i]
                i += 1
            multiplier = int(num_str) if num_str else 1
            for elem, count in sub_counts.items():
                counts[elem] += count * multiplier
        # Handle elements
        else:
            j = i + 1
            while j < len(formula) and formula[j].islower():
                j += 1
            element = formula[i:j]
            i = j
            num_str = ''
            while i < len(formula) and formula[i].isdigit():
                num_str += formula[i]
                i += 1
            count = int(num_str) if num_str else 1
            counts[element] += count
    return counts

def parse_species_atoms(species_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Parses species names (e.g., "C2H2_evolution", "Ca(OH)2") to create an atom count matrix.
    """
    all_atom_counts: List[Counter[str]] = []
    unique_atoms: set[str] = set()

    for name in species_names:
        formula = name.split("_")[0]
        try:
            counts = _parse_formula_recursive(formula)
            all_atom_counts.append(counts)
            unique_atoms.update(counts.keys())
        except Exception as e:
            logger.error(f"Failed to parse formula '{formula}' from species '{name}': {e}")
            all_atom_counts.append(Counter())

    if not unique_atoms:
        logger.warning("Could not parse any atoms from any species names.")
        return np.array([[] for _ in species_names]), []

    sorted_atoms = sorted(list(unique_atoms))
    atom_map = {atom: i for i, atom in enumerate(sorted_atoms)}
    atom_matrix = np.zeros((len(species_names), len(sorted_atoms)), dtype=np.float32)

    for i, counts in enumerate(all_atom_counts):
        for atom, count in counts.items():
            if atom in atom_map:
                atom_matrix[i, atom_map[atom]] = count
    return atom_matrix, sorted_atoms

__all__ = [
    "setup_logging", "load_config", "validate_config", "ensure_dirs", 
    "save_json", "seed_everything", "parse_species_atoms"
]
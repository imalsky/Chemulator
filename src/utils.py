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
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Try to import json5 for more flexible JSON parsing
try:
    import json5 as _json_backend
    _HAS_JSON5 = True
except ImportError:
    _json_backend = json
    _HAS_JSON5 = False

# Constants
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_SEED = 42
MIN_DROPOUT = 0.0
MAX_DROPOUT = 1.0
JSON_INDENT = 2
UTF8_ENCODING = "utf-8"
UTF8_SIG_ENCODING = "utf-8-sig"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s"

logger = logging.getLogger(__name__)


def setup_logging(
    level: int = DEFAULT_LOG_LEVEL, 
    log_file: Optional[Union[str, Path]] = None
) -> None:
    """
    Sets up logging configuration for the application.
    
    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Optional path to log file. If provided, logs to both console and file.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    while root_logger.handlers:
        handler = root_logger.handlers.pop()
        handler.close()
    
    # Set up formatter
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if requested)
    if log_file:
        _setup_file_logging(root_logger, log_file, formatter)


def _setup_file_logging(
    root_logger: logging.Logger, 
    log_file: Union[str, Path], 
    formatter: logging.Formatter
) -> None:
    """Sets up file logging with proper error handling."""
    try:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_path, mode="a", encoding=UTF8_ENCODING
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        logger.info(f"Logging to console and file: {log_file}")
    except Exception as exc:
        logger.error(
            f"File logging setup failed for {log_file}: {exc}. "
            f"Continuing with console only."
        )


def load_config(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Loads and validates a configuration file.
    
    Args:
        path: Path to the configuration file (JSON or JSON5 format)
        
    Returns:
        Configuration dictionary if successful, None otherwise
    """
    config_path = Path(path)
    
    if not config_path.is_file():
        logger.critical(f"Configuration file not found: {config_path}")
        return None
    
    try:
        # Use UTF-8-SIG to handle potential BOM in config files
        config_text = config_path.read_text(encoding=UTF8_SIG_ENCODING)
        config_dict = _json_backend.loads(config_text)
        
        validate_config(config_dict)
        
        backend_name = "JSON5" if _HAS_JSON5 else "JSON"
        logger.info(
            f"Successfully loaded and validated {backend_name} configuration "
            f"from {config_path}."
        )
        return config_dict
    except Exception as e:
        logger.critical(
            f"Failed to load or validate config {config_path}: {e}", 
            exc_info=True
        )
        return None


def _validate_parameter(
    config: Dict[str, Any], 
    key: str, 
    expected_type: Union[type, tuple[type, ...]], 
    required: bool = True, 
    non_empty: bool = False, 
    min_value: Optional[Union[int, float]] = None
) -> Any:
    """
    Validates a single configuration parameter.
    
    Args:
        config: Configuration dictionary
        key: Parameter key to validate
        expected_type: Expected type(s) for the parameter
        required: Whether the parameter is required
        non_empty: Whether collections must be non-empty
        min_value: Minimum value for numeric parameters
        
    Returns:
        The validated parameter value
        
    Raises:
        ValueError: If required parameter is missing or invalid
        TypeError: If parameter has wrong type
    """
    # Check if key exists
    if key not in config:
        if required:
            raise ValueError(f"Required configuration key '{key}' is missing.")
        return None
    
    value = config[key]
    
    # Type validation
    if not isinstance(value, expected_type):
        type_names = (
            expected_type.__name__ 
            if isinstance(expected_type, type) 
            else [t.__name__ for t in expected_type]
        )
        raise TypeError(
            f"Configuration key '{key}' must be {type_names}, "
            f"got {type(value).__name__}."
        )
    
    # Non-empty validation for collections
    if non_empty and isinstance(value, (list, dict)) and not value:
        raise ValueError(f"Configuration key '{key}' must be non-empty.")
    
    # Minimum value validation for numbers
    if (min_value is not None and 
        isinstance(value, (int, float)) and 
        value < min_value):
        raise ValueError(
            f"Configuration key '{key}' ({value}) must be >= {min_value}."
        )
    
    return value


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validates the configuration dictionary for the model.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid
        TypeError: If configuration has wrong types
    """
    # Required list parameters
    _validate_parameter(
        config, "species_variables", list, 
        required=True, non_empty=True
    )
    _validate_parameter(
        config, "global_variables", list, 
        required=True  # Can be empty
    )
    _validate_parameter(
        config, "all_variables", list, 
        required=True, non_empty=True
    )
    
    # Model architecture parameters
    _validate_parameter(
        config, "hidden_dims", list, 
        required=True, non_empty=True
    )
    
    # Dropout validation with specific range checking
    _validate_dropout_parameter(config)


def _validate_dropout_parameter(config: Dict[str, Any]) -> None:
    """Validates that dropout is present and within valid range."""
    if "dropout" not in config:
        raise ValueError(
            "Configuration must explicitly specify 'dropout' parameter."
        )
    
    dropout = config["dropout"]
    
    if not isinstance(dropout, (float, int)):
        raise TypeError(
            f"'dropout' must be a float or int, got {type(dropout).__name__}."
        )
    
    dropout_float = float(dropout)
    if not (MIN_DROPOUT <= dropout_float < MAX_DROPOUT):
        raise ValueError(
            f"'dropout' must be in the range [{MIN_DROPOUT}, {MAX_DROPOUT}), "
            f"got {dropout_float}."
        )


def ensure_dirs(*paths: Union[str, Path]) -> bool:
    """
    Creates directories for the given paths if they don't exist.
    
    Args:
        *paths: Variable number of directory paths to create
        
    Returns:
        True if all directories were created successfully, False otherwise
    """
    try:
        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except OSError as e:
        logger.error(f"Failed to create directories {paths}: {e}")
        return False


# --- NEW FUNCTION for Physics-Informed Loss ---
def parse_species_atoms(species_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Parses a list of species names to create an atom count matrix.

    This function extracts the chemical formula from names like "C2H2_evolution"
    and builds a matrix for calculating atom conservation.

    Args:
        species_names: A list of species names from the configuration.

    Returns:
        A tuple containing:
        - A numpy array (atom_matrix) of shape (num_species, num_unique_atoms).
        - A list of unique atom names (e.g., ['C', 'H', 'N', 'O']).

    Example:
        >>> parse_species_atoms(["H2O_evolution", "CO2_evolution"])
        (array([[2., 1., 0.], [0., 2., 1.]]), ['H', 'O', 'C'])
    """
    # Regex to find an element (e.g., 'C', 'H', 'He') followed by an optional number
    atom_regex = re.compile(r"([A-Z][a-z]*)(\d*)")
    
    all_atom_counts = []
    unique_atoms = set()

    for name in species_names:
        formula = name.split("_")[0]  # Extract formula part, e.g., "C2H2"
        counts = {}
        
        for element, number in atom_regex.findall(formula):
            unique_atoms.add(element)
            # If no number follows, it's 1 atom (e.g., 'O' in 'H2O')
            count = int(number) if number else 1
            counts[element] = counts.get(element, 0) + count
        
        all_atom_counts.append(counts)

    if not unique_atoms:
        logger.warning("Could not parse any atoms from species names.")
        return np.array([[] for _ in species_names]), []

    # Create a sorted, consistent list of unique atoms
    sorted_atoms = sorted(list(unique_atoms))
    atom_map = {atom: i for i, atom in enumerate(sorted_atoms)}

    # Build the matrix
    num_species = len(species_names)
    num_atoms = len(sorted_atoms)
    atom_matrix = np.zeros((num_species, num_atoms), dtype=np.float32)

    for i, counts in enumerate(all_atom_counts):
        for atom, count in counts.items():
            if atom in atom_map:
                atom_matrix[i, atom_map[atom]] = count

    return atom_matrix, sorted_atoms


def _json_serializer(obj: Any) -> Any:
    """
    Custom JSON serializer for numpy and torch objects.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable representation of the object
        
    Raises:
        TypeError: If object cannot be serialized
    """
    # Handle numpy types
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Handle PyTorch tensors
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    
    # Handle Path objects
    if isinstance(obj, Path):
        return str(obj)
    
    # Handle sets (convert to sorted list for consistency)
    if isinstance(obj, set):
        return sorted(list(obj))
    
    # Cannot serialize this type
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable."
    )


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> bool:
    """
    Saves a dictionary to a JSON file with proper error handling.
    
    Args:
        data: Dictionary to save
        path: Path where to save the JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        json_path = Path(path)
        
        # Ensure parent directory exists
        if not ensure_dirs(json_path.parent):
            return False
        
        # Write JSON file
        with json_path.open("w", encoding=UTF8_ENCODING) as f:
            json.dump(
                data, 
                f, 
                indent=JSON_INDENT, 
                default=_json_serializer, 
                ensure_ascii=False
            )
        
        return True
    except Exception as exc:
        logger.error(f"Failed to save JSON to {path}: {exc}", exc_info=True)
        return False


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """
    Sets the random seed for reproducible results across all libraries.
    
    Args:
        seed: Random seed to use
    """
    # Set environment variable for Python hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Seed Python's random module
    random.seed(seed)
    
    # Seed NumPy
    np.random.seed(seed)
    
    # Seed PyTorch
    torch.manual_seed(seed)
    
    # Seed CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    logger.info(f"Global random seed set to {seed}.")


__all__ = [
    "setup_logging", 
    "load_config", 
    "validate_config", 
    "ensure_dirs", 
    "save_json", 
    "seed_everything",
    "parse_species_atoms"
]
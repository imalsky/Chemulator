import logging
import os
from typing import Dict, Any
import torch


def setup_device() -> torch.device:
    """Detect and configure the best available compute device."""
    logger = logging.getLogger(__name__)

    # CUDA first
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        device = torch.device(f"cuda:{idx}")
        try:
            torch.cuda.set_device(idx)
        except Exception:
            pass
        props = torch.cuda.get_device_properties(idx)
        mem_gib = props.total_memory / (1024**3)
        logger.info(f"Using CUDA device: {props.name} ({mem_gib:.1f} GiB)")
        return device

    # Apple Silicon
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available() and getattr(mps_backend, "is_built", lambda: True)():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS device")
        return device

    # CPU
    device = torch.device("cpu")
    logger.info(f"Using CPU device ({os.cpu_count()} cores)")
    return device


def optimize_hardware(config: Dict[str, Any], device: torch.device) -> None:
    """Apply hardware-specific optimizations with safe feature detection."""
    logger = logging.getLogger(__name__)

    tf32_enabled = bool(config.get("tf32", True))
    try:
        # Only meaningful on CUDA for float32; harmless elsewhere, but be explicit.
        if device.type == "cuda" and tf32_enabled:
            torch.set_float32_matmul_precision("high")  # enables TF32 paths
            # Redundant but explicit toggles:
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 enabled (matmul precision 'high').")
        else:
            # keep true FP32 when requested
            torch.set_float32_matmul_precision("highest")
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = False
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = False
            if device.type == "cuda":
                logger.info("TF32 disabled (matmul precision 'highest').")
    except Exception:
        pass

    if device.type == "cuda":
        try:
            if bool(config.get("cudnn_benchmark", True)) and hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = True
                logger.info("cuDNN autotuner enabled")
        except Exception:
            pass

        mem_frac = config.get("cuda_memory_fraction", None)
        if mem_frac is not None:
            try:
                mem_frac = float(mem_frac)
                if mem_frac < 1.0 and hasattr(torch.cuda, "set_per_process_memory_fraction"):
                    idx = device.index if device.index is not None else torch.cuda.current_device()
                    torch.cuda.set_per_process_memory_fraction(mem_frac, device=idx)
                    logger.info(f"CUDA memory fraction set to {mem_frac} on device {idx}")
                else:
                    logger.info("CUDA memory fraction not capped (using full device memory).")
            except Exception as e:
                logger.warning(f"Could not set CUDA memory fraction: {e}")

    if "OMP_NUM_THREADS" not in os.environ:
        try:
            n = min(32, os.cpu_count() or 1)
            torch.set_num_threads(n)
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(min(n, 8))
            logger.info(f"Using {torch.get_num_threads()} CPU threads (intra-op).")
        except Exception:
            pass
    else:
        logger.info(f"Using {os.environ['OMP_NUM_THREADS']} CPU threads (OMP_NUM_THREADS).")

    if bool(config.get("deterministic", False)):
        try:
            torch.use_deterministic_algorithms(True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            logger.info("Deterministic algorithms enabled.")
        except Exception:
            pass

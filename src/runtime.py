#!/usr/bin/env python3
"""Shared runtime helpers for platform setup and device selection.

These helpers intentionally avoid importing ``torch`` at module import time so
entrypoint scripts can apply environment tweaks before the PyTorch runtime is
loaded. That matters on some local macOS environments, where duplicate OpenMP
runtime detection can abort process startup.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import MutableMapping, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def prepare_platform_environment(
    *,
    platform: str | None = None,
    env: MutableMapping[str, str] | None = None,
) -> None:
    """Apply platform-specific environment tweaks before importing torch.

    Inputs:
      - ``platform``: optional platform string. Defaults to ``sys.platform``.
      - ``env``: optional environment mapping to mutate. Defaults to
        ``os.environ``.

    Output:
      - Mutates ``env`` in-place. On Darwin, sets
        ``KMP_DUPLICATE_LIB_OK=TRUE`` unless the caller already set it.
    """

    active_platform = sys.platform if platform is None else str(platform)
    target_env = os.environ if env is None else env
    if active_platform == "darwin":
        target_env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def mps_is_available() -> bool:
    """Return ``True`` when the current torch build can execute on Apple MPS."""

    import torch

    backends = getattr(torch, "backends", None)
    mps_backend = getattr(backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def select_best_device(
    logger: logging.Logger | None = None,
    *,
    force_cpu: bool = False,
) -> "torch.device":
    """Pick the canonical runtime device for this repository.

    Selection order:
      1. CPU immediately if ``force_cpu=True``.
      2. CUDA device 0.
      3. Apple MPS.
      4. CPU.
    """

    import torch

    if force_cpu:
        if logger is not None:
            logger.info("Using CPU (force_cpu=True)")
        return torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        dev_name = "unknown"
        try:
            dev_name = torch.cuda.get_device_name(0)
        except Exception as exc:
            if logger is not None:
                logger.warning("Could not query CUDA device name: %s", exc)
        if logger is not None:
            logger.info("Set CUDA device to cuda:0 (%s)", dev_name)
        return device

    if mps_is_available():
        if logger is not None:
            logger.info("Using Apple MPS")
        return torch.device("mps")

    if logger is not None:
        logger.info("Using CPU")
    return torch.device("cpu")

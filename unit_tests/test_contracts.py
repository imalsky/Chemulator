"""Unit tests for repository configuration contracts."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

import torch

from src.main import DEFAULT_CONFIG_PATH, reject_deprecated_config_keys
from src.utils import resolve_precision_policy


class ConfigContractsTests(unittest.TestCase):
    def test_default_config_path_is_repo_local(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        self.assertEqual(DEFAULT_CONFIG_PATH, repo_root / "config" / "config_job0.jsonc")

    def test_reject_deprecated_config_keys_fails_fast(self) -> None:
        cfg = {
            "dataset": {},
            "data": {},
            "preprocessing": {},
            "training": {"auto_resume": True},
            "model": {},
        }
        with self.assertRaises(KeyError):
            reject_deprecated_config_keys(cfg)

    def test_resolve_precision_policy_rejects_amp_on_cpu(self) -> None:
        cfg = {
            "precision": {
                "amp": "bf16",
                "dataset_dtype": "auto",
                "io_dtype": "float32",
                "time_io_dtype": "float32",
                "normalize_dtype": "float32",
                "tf32": False,
            }
        }
        with self.assertRaises(ValueError):
            resolve_precision_policy(cfg, torch.device("cpu"))

    def test_resolve_precision_policy_accepts_fp32_cpu_mode(self) -> None:
        cfg = {
            "precision": {
                "amp": "off",
                "dataset_dtype": "float32",
                "io_dtype": "float32",
                "time_io_dtype": "float32",
                "normalize_dtype": "float32",
                "tf32": False,
            }
        }
        policy = resolve_precision_policy(copy.deepcopy(cfg), torch.device("cpu"))
        self.assertEqual(policy.amp_mode, "off")
        self.assertEqual(policy.dataset_dtype, torch.float32)


if __name__ == "__main__":
    raise SystemExit(unittest.main())

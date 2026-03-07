"""Unit tests for shared runtime helpers."""

from __future__ import annotations

import unittest
from unittest import mock

from src.runtime import prepare_platform_environment, select_best_device


class RuntimeHelpersTests(unittest.TestCase):
    def test_prepare_platform_environment_sets_darwin_openmp_guard(self) -> None:
        env: dict[str, str] = {}
        prepare_platform_environment(platform="darwin", env=env)
        self.assertEqual(env["KMP_DUPLICATE_LIB_OK"], "TRUE")

    def test_prepare_platform_environment_does_not_override_existing_value(self) -> None:
        env = {"KMP_DUPLICATE_LIB_OK": "FALSE"}
        prepare_platform_environment(platform="darwin", env=env)
        self.assertEqual(env["KMP_DUPLICATE_LIB_OK"], "FALSE")

    def test_prepare_platform_environment_leaves_linux_untouched(self) -> None:
        env: dict[str, str] = {}
        prepare_platform_environment(platform="linux", env=env)
        self.assertNotIn("KMP_DUPLICATE_LIB_OK", env)

    def test_select_best_device_force_cpu(self) -> None:
        device = select_best_device(force_cpu=True)
        self.assertEqual(device.type, "cpu")

    def test_select_best_device_prefers_mps_after_cuda(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=False):
            with mock.patch("src.runtime.mps_is_available", return_value=True):
                device = select_best_device()
        self.assertEqual(device.type, "mps")


if __name__ == "__main__":
    raise SystemExit(unittest.main())

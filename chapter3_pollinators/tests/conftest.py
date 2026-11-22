"""Shared fixtures for the chapter3_pollinators test-suite."""

from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@pytest.fixture(scope="session")
def reference(tmp_path_factory: pytest.TempPathFactory):
    """The reference implementation, imported from a temporary copy (or skip)."""
    from chapter3_pollinators.tests.reference import ENV_VAR, load_reference, reference_source

    if reference_source() is None:
        pytest.skip(f"set {ENV_VAR} to the reference all_in_one folder to run this test")
    return load_reference(tmp_path_factory.mktemp("reference"))


@pytest.fixture(scope="session")
def cuda_required() -> None:
    """The learners place their networks on the GPU, as the reference does."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required by the P-MADDPG learners")

"""Seeding and generator state."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python's ``random``, NumPy's global generator and torch (CPU and CUDA)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generator_states() -> dict:
    """The state of every global generator the runs draw from."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_generator_states(states: dict) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    if torch.cuda.is_available() and states["cuda"]:
        torch.cuda.set_rng_state_all(states["cuda"])

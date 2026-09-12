"""Reproducibility helper: fix every source of randomness used across the
pipeline (Python's `random`, NumPy, PyTorch CPU/GPU RNGs) from one call.

OTI initializes v* randomly, samples random templates/concepts/GPT phrases;
Phi training shuffles the DataLoader and applies dropout; without a fixed
seed, two runs with identical data/hyperparameters can converge to
different pseudo-word tokens / weights, which is confusing when trying to
debug or compare against the paper's numbers.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fix all RNGs. Call once, as early as possible, in every entry point.

    `deterministic=True` also asks cuDNN for deterministic algorithms; this
    can be slightly slower but makes runs bit-for-bit reproducible on the
    same GPU/driver/CUDA version.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Needed for torch.use_deterministic_algorithms to not raise on ops
        # (e.g. some scatter/index ops) that only have nondeterministic
        # CUDA kernels; safe no-op on CPU-only setups.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

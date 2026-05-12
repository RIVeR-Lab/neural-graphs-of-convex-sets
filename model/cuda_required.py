"""
Global requirement: training and inference for the GNN require a GPU.
"""

from __future__ import annotations

import torch


def require_cuda() -> None:
    """Raise RuntimeError if CUDA is not available. Call at startup of train/inference scripts."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required but not available. "
            "This code requires a GPU. Install PyTorch with CUDA and ensure a GPU is visible."
        )

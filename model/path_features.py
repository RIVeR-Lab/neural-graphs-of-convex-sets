from __future__ import annotations

import math

import torch
from torch import Tensor


def sinusoidal_position_encoding(length: int, dim: int = 8, *, device: torch.device | None = None) -> Tensor:
    if dim % 2 != 0:
        raise ValueError("sinusoidal position dimension must be even")
    pos = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    freq_ids = torch.arange(dim // 2, dtype=torch.float32, device=device)
    div = torch.exp(-math.log(10000.0) * (2.0 * freq_ids / dim)).unsqueeze(0)
    angles = pos * div
    pe = torch.empty((length, dim), dtype=torch.float32, device=device)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class EncoderHParams:
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    ffn_hidden_mult: int = 2
    attn_negative_slope: float = 0.2
    dropout_p: float = 0.1


@dataclass(frozen=True)
class DecoderHParams:
    hidden_dims: Tuple[int, ...] = (256, 256)
    dropout_p: float = 0.1


@dataclass(frozen=True)
class TrainingHParams:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    num_workers: int = 0
    max_epochs: int = 500
    # LR schedule: linear warmup then cosine decay
    use_lr_schedule: bool = True
    warmup_epochs: int = 20
    lr_scheduler_eta_min: float = 1e-6
    # Early stopping
    use_early_stopping: bool = True
    early_stop_patience: int = 50
    early_stop_min_delta: float = 1e-4


@dataclass(frozen=True)
class InferenceHParams:
    rounding_samples: int = 64
    max_steps: int = 512


@dataclass(frozen=True)
class ModelHParams:
    encoder: EncoderHParams = EncoderHParams()
    decoder: DecoderHParams = DecoderHParams()
    training: TrainingHParams = TrainingHParams()
    inference: InferenceHParams = InferenceHParams()


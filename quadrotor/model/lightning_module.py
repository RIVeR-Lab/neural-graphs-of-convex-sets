from __future__ import annotations

import dataclasses
import math
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

from planning_through_contact.model.hparams import DecoderHParams, EncoderHParams, TrainingHParams
from planning_through_contact.model.model import GCSFlowPredictor


class QuadrotorGNNModule(pl.LightningModule):
    """Phase 1: Flow GNN trained with BCE loss against phi_star (SDP relaxation flows)."""

    def __init__(
        self,
        *,
        x_dim: int = 9,
        g_dim: int = 6,
        encoder_hp: EncoderHParams = EncoderHParams(),
        decoder_hp: DecoderHParams = DecoderHParams(),
        train_hp: TrainingHParams = TrainingHParams(),
    ):
        super().__init__()
        self.save_hyperparameters(
            {
                "x_dim": x_dim,
                "g_dim": g_dim,
                "encoder_hp": dataclasses.asdict(encoder_hp),
                "decoder_hp": dataclasses.asdict(decoder_hp),
                "train_hp": dataclasses.asdict(train_hp),
            }
        )
        self.model = GCSFlowPredictor(
            x_dim=x_dim, g_dim=g_dim, encoder_hp=encoder_hp, decoder_hp=decoder_hp
        )
        self.train_hp = train_hp
        # phi_star ∈ [0,1] so plain BCE (no class weighting needed)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, batch) -> Tensor:
        out = self.model(x=batch.x, edge_index=batch.edge_index, g=batch.g, batch=batch.batch)
        return out.edge_logits

    def _step(self, batch, stage: str) -> Tensor:
        logits = self.forward(batch)
        loss = self.loss_fn(logits, batch.y.float())
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True)

        with torch.no_grad():
            phi_hat = torch.sigmoid(logits)
            mae = (phi_hat - batch.y.float()).abs().mean()
            self.log(f"{stage}/mae", mae, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> None:
        self._step(batch, "val")

    def test_step(self, batch, batch_idx: int) -> None:
        self._step(batch, "test")

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.train_hp.lr),
            weight_decay=float(self.train_hp.weight_decay),
        )
        if not self.train_hp.use_lr_schedule:
            return optimizer

        base_lr = float(self.train_hp.lr)
        eta_min = float(self.train_hp.lr_scheduler_eta_min)
        warmup = int(self.train_hp.warmup_epochs)
        cosine_epochs = max(1, int(self.train_hp.max_epochs) - warmup)

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = min(1.0, (epoch - warmup + 1) / cosine_epochs)
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            return eta_min / base_lr + (1.0 - eta_min / base_lr) * cos

        return [optimizer], [{"scheduler": LambdaLR(optimizer, lr_lambda), "interval": "epoch"}]

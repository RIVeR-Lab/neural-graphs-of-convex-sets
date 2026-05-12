from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

from planning_through_contact.model.hparams import DecoderHParams, EncoderHParams, TrainingHParams
from planning_through_contact.model.model import GCSFlowPredictor


def _edge_metrics(logits: Tensor, y: Tensor, threshold: float = 0.5) -> dict[str, Tensor]:
    """
    Simple edge-level metrics (no external deps).
    """
    with torch.no_grad():
        p = torch.sigmoid(logits)
        y_hat = (p >= threshold).to(y.dtype)

        tp = ((y_hat == 1) & (y == 1)).sum().float()
        tn = ((y_hat == 0) & (y == 0)).sum().float()
        fp = ((y_hat == 1) & (y == 0)).sum().float()
        fn = ((y_hat == 0) & (y == 1)).sum().float()

        acc = (tp + tn) / torch.clamp(tp + tn + fp + fn, min=1.0)
        precision = tp / torch.clamp(tp + fp, min=1.0)
        recall = tp / torch.clamp(tp + fn, min=1.0)
        f1 = (2 * precision * recall) / torch.clamp(precision + recall, min=1e-8)
        pos_rate = (y == 1).float().mean()
        pred_pos_rate = (y_hat == 1).float().mean()

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pos_rate": pos_rate,
        "pred_pos_rate": pred_pos_rate,
    }


class GCSLightningModule(pl.LightningModule):
    """
    Paper training objective: weighted BCE-with-logits over edges (Eq. 36–37).
    """

    def __init__(
        self,
        *,
        x_dim: int,
        g_dim: int,
        pos_weight: Tensor,
        target: Literal["discrete", "sdp"] = "sdp",
        encoder_hp: EncoderHParams = EncoderHParams(),
        decoder_hp: DecoderHParams = DecoderHParams(),
        train_hp: TrainingHParams = TrainingHParams(),
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weight"])

        self.model = GCSFlowPredictor(
            x_dim=x_dim,
            g_dim=g_dim,
            encoder_hp=encoder_hp,
            decoder_hp=decoder_hp,
        )

        self.train_hp = train_hp
        self.target = target
        self.register_buffer("pos_weight", pos_weight.view(1).detach().float())
        # BCE: discrete uses pos_weight for class balance; sdp uses pos_weight=1 (no weighting).
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, batch) -> Tensor:  # type: ignore[override]
        return self.model(x=batch.x, edge_index=batch.edge_index, g=batch.g, batch=batch.batch)

    def _step(self, batch, stage: str) -> Tensor:
        logits = self.forward(batch)
        y = batch.y.float()
        loss = self.loss_fn(logits, y)

        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True)

        # For metrics: use binary labels (sdp targets binarized at 0.5)
        y_metric = (batch.y.float() >= 0.5).long() if self.target == "sdp" else batch.y.detach()
        m = _edge_metrics(logits.detach(), y_metric)
        for k, v in m.items():
            self.log(f"{stage}/{k}", v, prog_bar=(k in {"f1"}), on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx: int) -> Tensor:  # type: ignore[override]
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> None:  # type: ignore[override]
        _ = self._step(batch, "val")

    def test_step(self, batch, batch_idx: int) -> None:  # type: ignore[override]
        _ = self._step(batch, "test")

    def configure_optimizers(self) -> Any:  # type: ignore[override]
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.train_hp.lr),
            weight_decay=float(self.train_hp.weight_decay),
        )
        if not self.train_hp.use_lr_schedule:
            return optimizer

        base_lr = float(self.train_hp.lr)
        eta_min = float(self.train_hp.lr_scheduler_eta_min)
        warmup_epochs = int(self.train_hp.warmup_epochs)
        max_epochs = int(self.train_hp.max_epochs)
        cosine_epochs = max(1, max_epochs - warmup_epochs)

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = min(1.0, (epoch - warmup_epochs + 1) / cosine_epochs)
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return eta_min / base_lr + (1.0 - eta_min / base_lr) * cos_factor

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


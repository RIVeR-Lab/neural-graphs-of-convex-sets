from __future__ import annotations

import dataclasses
import math
from typing import Any

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

from planning_through_contact.model.hparams import TrainingHParams
from planning_through_contact.model.inference import project_flows_qp
from planning_through_contact.model.model import GCSFlowPredictor
from planning_through_contact.model.ranknet import PathRankNet, RankNetConfig, ranknet_pair_loss
from quadrotor.model.ranknet_dataset import RankNetSample


class QuadrotorRankNetModule(pl.LightningModule):
    """Phase 2: Frozen flow GNN + trainable PathRankNet with soft RankNet loss (τ=0.2)."""

    def __init__(
        self,
        *,
        flow_model: GCSFlowPredictor,
        ranker_cfg: RankNetConfig = RankNetConfig(),
        train_hp: TrainingHParams = TrainingHParams(),
    ):
        super().__init__()
        self.save_hyperparameters(
            {"ranker_cfg": dataclasses.asdict(ranker_cfg), "train_hp": dataclasses.asdict(train_hp)}
        )
        self.flow_model = flow_model
        self.flow_model.eval()
        for p in self.flow_model.parameters():
            p.requires_grad_(False)

        self.ranker = PathRankNet(ranker_cfg)
        self.train_hp = train_hp

    def _project_flows(self, sample: RankNetSample, logits: Tensor) -> Tensor:
        phi_hat = torch.sigmoid(logits).detach().cpu()
        phi_proj = project_flows_qp(
            edge_index=sample.edge_index.cpu(),
            phi_hat=phi_hat,
            num_nodes=int(sample.x.size(0)),
            source_idx=int(sample.source_idx),
            target_idx=int(sample.target_idx),
        )
        return phi_proj.to(device=logits.device, dtype=logits.dtype)

    def _score_sample(self, sample: RankNetSample) -> Tensor:
        device = self.device
        with torch.no_grad():
            out = self.flow_model(
                x=sample.x.to(device),
                edge_index=sample.edge_index.to(device),
                g=sample.g.to(device),
                batch=None,
            )
            edge_flows = self._project_flows(sample, out.edge_logits)

        return self.ranker(
            node_embeddings=out.node_embeddings.detach(),
            edge_flows=edge_flows.detach(),
            path_node_indices=sample.path_node_indices,
            path_edge_indices=sample.path_edge_indices,
            path_mask=sample.path_mask,
        )

    def _step(self, batch: list[RankNetSample], stage: str) -> Tensor:
        losses = []
        total_pairs = 0
        for sample in batch:
            scores = self._score_sample(sample)
            better = sample.better_idx.to(scores.device)
            worse = sample.worse_idx.to(scores.device)
            p_bar = sample.p_bar.to(scores.device) if sample.p_bar is not None else None
            losses.append(ranknet_pair_loss(scores, better, worse, p_bar=p_bar))
            total_pairs += int(better.numel())

        loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=self.device)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}/pairs", float(total_pairs), on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: list[RankNetSample], batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: list[RankNetSample], batch_idx: int) -> None:
        self._step(batch, "val")

    def test_step(self, batch: list[RankNetSample], batch_idx: int) -> None:
        self._step(batch, "test")

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.ranker.parameters(),
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

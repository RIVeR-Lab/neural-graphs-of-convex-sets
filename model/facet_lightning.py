"""Lightning modules for PointNet-over-facets flow GNN (Phase 1) and ranker (Phase 2)."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

from model.facet_dataset import FacetRankNetSample
from model.facet_pointnet import PointNetFlowPredictor
from model.hparams import DecoderHParams, EncoderHParams, TrainingHParams
from model.inference import project_flows_qp
from model.ranknet import PathRankNet, RankNetConfig, ranknet_pair_loss


def _cosine_warmup(optimizer, train_hp: TrainingHParams):
    base_lr = float(train_hp.lr)
    eta_min = float(train_hp.lr_scheduler_eta_min)
    warmup = int(train_hp.warmup_epochs)
    cosine_epochs = max(1, int(train_hp.max_epochs) - warmup)

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = min(1.0, (epoch - warmup + 1) / cosine_epochs)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min / base_lr + (1.0 - eta_min / base_lr) * cos

    return [optimizer], [{"scheduler": LambdaLR(optimizer, lr_lambda), "interval": "epoch"}]


class FacetFlowModule(pl.LightningModule):
    """Phase 1: PointNet + BiGATv2 flow GNN, BCE against phi_star."""

    def __init__(
        self,
        *,
        facet_dim: int,
        g_dim: int,
        encoder_hp: EncoderHParams = EncoderHParams(),
        decoder_hp: DecoderHParams = DecoderHParams(),
        train_hp: TrainingHParams = TrainingHParams(),
        pointnet_hidden: int = 64,
    ):
        super().__init__()
        self.save_hyperparameters(
            {
                "facet_dim": facet_dim,
                "g_dim": g_dim,
                "encoder_hp": dataclasses.asdict(encoder_hp),
                "decoder_hp": dataclasses.asdict(decoder_hp),
                "train_hp": dataclasses.asdict(train_hp),
                "pointnet_hidden": pointnet_hidden,
            }
        )
        self.model = PointNetFlowPredictor(
            facet_dim=facet_dim,
            g_dim=g_dim,
            encoder_hp=encoder_hp,
            decoder_hp=decoder_hp,
            pointnet_hidden=pointnet_hidden,
        )
        self.train_hp = train_hp
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, batch) -> Tensor:
        out = self.model(
            facets=batch.facets,
            facet_mask=batch.facet_mask,
            node_flags=batch.node_flags,
            edge_index=batch.edge_index,
            g=batch.g,
            batch=batch.batch,
        )
        return out.edge_logits

    def _step(self, batch, stage: str) -> Tensor:
        logits = self.forward(batch)
        loss = self.loss_fn(logits, batch.y.float())
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        with torch.no_grad():
            mae = (torch.sigmoid(logits) - batch.y.float()).abs().mean()
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
        return _cosine_warmup(optimizer, self.train_hp)


class FacetRankNetModule(pl.LightningModule):
    """Phase 2: frozen PointNet flow model + trainable PathRankNet."""

    def __init__(
        self,
        *,
        flow_model: PointNetFlowPredictor,
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

    def _project_flows(self, sample: FacetRankNetSample, logits: Tensor) -> Tensor:
        phi_hat = torch.sigmoid(logits).detach().cpu()
        phi_proj = project_flows_qp(
            edge_index=sample.edge_index.cpu(),
            phi_hat=phi_hat,
            num_nodes=int(sample.num_nodes),
            source_idx=int(sample.source_idx),
            target_idx=int(sample.target_idx),
        )
        return phi_proj.to(device=logits.device, dtype=logits.dtype)

    def _score_sample(self, sample: FacetRankNetSample) -> Tensor:
        device = self.device
        with torch.no_grad():
            out = self.flow_model(
                facets=sample.facets.to(device),
                facet_mask=sample.facet_mask.to(device),
                node_flags=sample.node_flags.to(device),
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

    def _step(self, batch: list[FacetRankNetSample], stage: str) -> Tensor:
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

    def training_step(self, batch, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> None:
        self._step(batch, "val")

    def test_step(self, batch, batch_idx: int) -> None:
        self._step(batch, "test")

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.ranker.parameters(),
            lr=float(self.train_hp.lr),
            weight_decay=float(self.train_hp.weight_decay),
        )
        if not self.train_hp.use_lr_schedule:
            return optimizer
        return _cosine_warmup(optimizer, self.train_hp)

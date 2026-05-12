from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from model.path_features import sinusoidal_position_encoding


@dataclass(frozen=True)
class RankNetConfig:
    d_model: int = 128
    pos_dim: int = 8
    num_layers: int = 3
    num_heads: int = 4
    ffn_hidden_dim: int = 256
    score_hidden_dim: int = 64
    dropout_p: float = 0.1
    soft_targets: bool = True
    # Scale applied to (c_worse - c_better) before sigmoid to compute P̄.
    # Auto-normalized at dataset construction time so tau=1.0 is meaningful.
    soft_targets_tau: float = 1.0


class PathRankNet(nn.Module):
    """
    RankNet-style path scorer.

    Per path position input is [h_v || phi_incoming_edge || pos_encoding].
    Scores are higher for better paths.
    """

    def __init__(self, cfg: RankNetConfig = RankNetConfig()):
        super().__init__()
        self.cfg = cfg
        token_dim = cfg.d_model + 1 + cfg.pos_dim
        self.input_proj = nn.Linear(token_dim, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_hidden_dim,
            dropout=cfg.dropout_p,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.score_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.score_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.score_hidden_dim, 1),
        )

    def forward(
        self,
        *,
        node_embeddings: Tensor,
        edge_flows: Tensor,
        path_node_indices: Tensor,
        path_edge_indices: Tensor,
        path_mask: Tensor,
    ) -> Tensor:
        if path_node_indices.numel() == 0:
            return torch.empty((0,), dtype=node_embeddings.dtype, device=node_embeddings.device)

        device = node_embeddings.device
        path_node_indices = path_node_indices.to(device)
        path_edge_indices = path_edge_indices.to(device)
        path_mask = path_mask.to(device)
        edge_flows = edge_flows.to(device)

        safe_node_idx = path_node_indices.clamp_min(0)
        h = node_embeddings[safe_node_idx]

        incoming_flow = torch.zeros(path_edge_indices.shape, dtype=node_embeddings.dtype, device=device)
        valid_edges = path_edge_indices >= 0
        incoming_flow[valid_edges] = edge_flows[path_edge_indices[valid_edges]]
        incoming_flow = incoming_flow.unsqueeze(-1)

        pos = sinusoidal_position_encoding(
            path_node_indices.size(1),
            self.cfg.pos_dim,
            device=device,
        )
        pos = pos.unsqueeze(0).expand(path_node_indices.size(0), -1, -1)

        tokens = torch.cat([h, incoming_flow, pos], dim=-1)
        tokens = self.input_proj(tokens)
        tokens = self.encoder(tokens, src_key_padding_mask=~path_mask.bool())

        mask_f = path_mask.to(dtype=tokens.dtype).unsqueeze(-1)
        pooled = (tokens * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        return self.score_head(pooled).squeeze(-1)


def ranknet_pair_loss(
    scores: Tensor,
    better_idx: Tensor,
    worse_idx: Tensor,
    p_bar: Tensor | None = None,
) -> Tensor:
    """RankNet cross-entropy loss (Eq. 3 from Burges et al. 2005).

    p_bar: per-pair target probability in (0,1]. None → hard labels (P̄=1).
    Hard pairs (feas vs infeas) should always pass p_bar=None or p_bar=1.
    """
    if better_idx.numel() == 0:
        return scores.sum() * 0.0
    margin = scores[better_idx] - scores[worse_idx]
    if p_bar is None:
        return -torch.nn.functional.logsigmoid(margin).mean()
    # Eq. 3: C_ij = -P̄·o_ij + log(1 + exp(o_ij))
    return torch.nn.functional.binary_cross_entropy_with_logits(margin, p_bar)

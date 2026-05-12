"""PointNet-over-facets region encoder (Quadrotor_ISRR.pdf, Section 2).

Encodes a convex region from its halfspace facets f_i = (a_i, b_i) via a shared
MLP + attention pooling into a fixed-size embedding, then feeds the result
(plus source/target flags) into the existing GCS flow predictor.

Works for any problem by varying ``facet_dim`` (4 for 3-D quadrotor boxes,
8 for 7-D manipulation IRIS regions).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from model.hparams import DecoderHParams, EncoderHParams
from model.model import GCSFlowOutput, GCSFlowPredictor

NUM_NODE_FLAGS = 3  # [is_region, is_source, is_target]


class FacetPointNet(nn.Module):
    """Shared MLP over facets + attention pooling -> per-region embedding.

    Input:
      facets: [N, F, facet_dim]  padded facet tokens (a_i normalized, b_i scaled)
      mask:   [N, F]             True where facet is real
    Output:
      h:      [N, out_dim]       region embeddings (zeros for facet-less nodes)
    """

    def __init__(self, facet_dim: int, hidden_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.facet_dim = facet_dim
        self.out_dim = out_dim
        self.phi = nn.Sequential(
            nn.Linear(facet_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.attn_proj = nn.Linear(out_dim, out_dim)
        self.attn_vec = nn.Parameter(torch.empty(out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.phi.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.attn_proj.weight)
        nn.init.zeros_(self.attn_proj.bias)
        nn.init.normal_(self.attn_vec, std=0.02)

    def forward(self, facets: Tensor, mask: Tensor) -> Tensor:
        z = self.phi(facets)  # [N, F, out_dim]
        scores = torch.tanh(self.attn_proj(z)) @ self.attn_vec  # [N, F]
        scores = scores.masked_fill(~mask, float("-inf"))
        # Nodes with no facets (source/target) -> all -inf; guard against NaN.
        has_facets = mask.any(dim=1, keepdim=True)  # [N, 1]
        alpha = torch.softmax(scores, dim=1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        h = (alpha.unsqueeze(-1) * z).sum(dim=1)  # [N, out_dim]
        return h * has_facets.to(h.dtype)


class PointNetFlowPredictor(nn.Module):
    """FacetPointNet region encoder + GCS flow predictor (BiGATv2 + edge MLP)."""

    def __init__(
        self,
        *,
        facet_dim: int,
        g_dim: int,
        encoder_hp: EncoderHParams = EncoderHParams(),
        decoder_hp: DecoderHParams = DecoderHParams(),
        pointnet_hidden: int = 64,
    ):
        super().__init__()
        d_model = encoder_hp.d_model
        self.pointnet = FacetPointNet(
            facet_dim=facet_dim, hidden_dim=pointnet_hidden, out_dim=d_model
        )
        self.flow = GCSFlowPredictor(
            x_dim=d_model + NUM_NODE_FLAGS,
            g_dim=g_dim,
            encoder_hp=encoder_hp,
            decoder_hp=decoder_hp,
        )

    def node_features(self, facets: Tensor, facet_mask: Tensor, node_flags: Tensor) -> Tensor:
        h_pn = self.pointnet(facets, facet_mask)  # [N, d_model]
        return torch.cat([h_pn, node_flags.to(h_pn.dtype)], dim=-1)  # [N, d_model + 3]

    def forward(
        self,
        *,
        facets: Tensor,
        facet_mask: Tensor,
        node_flags: Tensor,
        edge_index: Tensor,
        g: Tensor,
        batch: Optional[Tensor] = None,
    ) -> GCSFlowOutput:
        x = self.node_features(facets, facet_mask, node_flags)
        return self.flow(x=x, edge_index=edge_index, g=g, batch=batch)

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

try:
    from torch_scatter import scatter  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Missing dependency 'torch_scatter'. It should be installed with torch-geometric."
    ) from e

from torch_geometric.utils import softmax

from planning_through_contact.model.hparams import DecoderHParams, EncoderHParams


@dataclass(frozen=True)
class GCSFlowOutput:
    """
    Flow-model outputs needed by downstream path ranking.

    `node_embeddings` are the frozen GNN features h_v. `edge_logits` are the
    decoder outputs aligned with `edge_index`.
    """

    node_embeddings: Tensor
    edge_logits: Tensor


def _broadcast_global(g: Tensor, batch: Optional[Tensor], num_nodes: int) -> Tensor:
    """
    Broadcast per-graph global features to per-node features.

    - If g is [G, g_dim], batch must be [N] with values in [0, G).
    - If g is [g_dim], it is treated as a single-graph batch and expanded to [N, g_dim].
    """
    if g.dim() == 1:
        return g.view(1, -1).expand(num_nodes, -1)
    if batch is None:
        raise ValueError("batch must be provided when g is a batched tensor [num_graphs, g_dim].")
    return g[batch]


def _broadcast_global_to_edges(
    g: Tensor, node_batch: Optional[Tensor], src_nodes: Tensor
) -> Tensor:
    """
    Broadcast per-graph global features to per-edge features using the source node's graph id.
    """
    if g.dim() == 1:
        return g.view(1, -1).expand(src_nodes.size(0), -1)
    if node_batch is None:
        raise ValueError("node_batch must be provided when g is batched [num_graphs, g_dim].")
    return g[node_batch[src_nodes]]


class ConditionedInit(nn.Module):
    """
    Conditioned initialisation:

        h_v^(0) = LN( W_x x_v + U_x g + b_x )
    """

    def __init__(self, x_dim: int, g_dim: int, d_model: int, dropout_p: float = 0.1):
        super().__init__()
        self.x_proj = nn.Linear(x_dim, d_model, bias=False)
        self.g_proj = nn.Linear(g_dim, d_model, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x: Tensor, g: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        # Eq. (16): h_v^(0) = LN( W_x x_v + U_x g + b_x )
        g_node = _broadcast_global(g=g, batch=batch, num_nodes=x.size(0))
        h0 = self.x_proj(x) + self.g_proj(g_node) + self.bias
        h0 = self.ln(self.dropout(h0))
        return h0


@dataclass(frozen=True)
class BiGATv2Config:
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 8
    ffn_hidden_mult: int = 2
    attn_negative_slope: float = 0.2
    dropout_p: float = 0.1

    @staticmethod
    def from_hparams(hp: EncoderHParams) -> "BiGATv2Config":
        return BiGATv2Config(
            d_model=hp.d_model,
            num_layers=hp.num_layers,
            num_heads=hp.num_heads,
            ffn_hidden_mult=hp.ffn_hidden_mult,
            attn_negative_slope=hp.attn_negative_slope,
            dropout_p=hp.dropout_p,
        )


class _GATv2DirectionalStream(nn.Module):
    """
    One directional attention stream (incoming OR outgoing), Eq. 17–22 and 23–28.
    """

    def __init__(self, d_model: int, num_heads: int, negative_slope: float, dropout_p: float):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        self.d_model = d_model
        self.num_heads = num_heads
        self.dh = d_model // num_heads
        self.negative_slope = negative_slope

        self.W = nn.Linear(d_model, num_heads * self.dh, bias=False)
        self.Wa_T = nn.Parameter(torch.empty(num_heads, 2 * self.dh, self.dh))
        self.a = nn.Parameter(torch.empty(num_heads, self.dh))

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.Wa_T)
        nn.init.xavier_uniform_(self.a.unsqueeze(-1))
        nn.init.xavier_uniform_(self.out_proj.weight)

    def _project(self, h: Tensor) -> Tensor:
        # Eq. (17): z_v^{(ℓ,r,dir)} = W^{(ℓ,r,dir)} h_v^{(ℓ)}
        z = self.W(h)  # [N, H*dh]
        return z.view(h.size(0), self.num_heads, self.dh)  # [N, H, dh]

    def forward(self, h: Tensor, edge_index: Tensor, *, mode: str) -> Tensor:
        if mode not in {"in", "out"}:
            raise ValueError(f"mode must be 'in' or 'out', got {mode!r}.")

        src = edge_index[0]
        dst = edge_index[1]
        N = h.size(0)

        z = self._project(h)  # [N, H, dh]
        z_src = z[src]  # [E, H, dh]
        z_dst = z[dst]  # [E, H, dh]

        if mode == "in":
            z_left, z_right = z_src, z_dst  # (u, v)
            group_index = dst  # softmax over N_in(v)
            scatter_index = dst  # output at v
            values = z_src  # sum alpha * z_u
        else:
            z_left, z_right = z_src, z_dst  # (v, w)
            group_index = src  # softmax over N_out(v)
            scatter_index = src  # output at v
            values = z_dst  # sum alpha * z_w

        # Eq. (18)/(24): e_uv^{(ℓ,r,dir)} = a^{(ℓ,r,dir)T} φ( W_a^{(ℓ,r,dir)} [z_u || z_v] )
        cat = torch.cat([z_left, z_right], dim=-1)  # [E, H, 2dh]
        pre = torch.einsum("ehc,hcd->ehd", cat, self.Wa_T)  # [E, H, dh]
        pre = torch.nn.functional.leaky_relu(pre, negative_slope=self.negative_slope)
        e = (pre * self.a).sum(dim=-1)  # [E, H]

        # Eq. (19)/(25): α_uv^{(ℓ,r,dir)} = softmax_{neighbor}( e_uv^{(ℓ,r,dir)} )
        alpha = softmax(e, index=group_index, num_nodes=N)  # [E, H]
        alpha = self.dropout(alpha)

        # Eq. (20)/(26): m_v^{(ℓ,r,dir)} = Σ_{neighbor} α * z_neighbor
        msg = values * alpha.unsqueeze(-1)  # [E, H, dh]
        agg = scatter(msg, scatter_index, dim=0, dim_size=N, reduce="sum")  # [N, H, dh]

        # Eq. (21)/(27): m_v^{(ℓ,dir)} = [m_v^{(ℓ,1,dir)} || ... || m_v^{(ℓ,H,dir)}] ∈ R^d
        # Eq. (22)/(28): m̃_v^{(ℓ,dir)} = W_O^{(ℓ,dir)} m_v^{(ℓ,dir)}
        agg = agg.reshape(N, self.d_model)  # concat heads -> [N, d]
        agg = self.out_proj(agg)  # [N, d]
        return agg


class BiGATv2Layer(nn.Module):
    """
    One bidirectional GATv2 layer with fusion + residual FFN + LN (Eq. 29–30).
    """

    def __init__(self, cfg: BiGATv2Config):
        super().__init__()
        d = cfg.d_model
        self.in_stream = _GATv2DirectionalStream(
            d_model=d,
            num_heads=cfg.num_heads,
            negative_slope=cfg.attn_negative_slope,
            dropout_p=cfg.dropout_p,
        )
        self.out_stream = _GATv2DirectionalStream(
            d_model=d,
            num_heads=cfg.num_heads,
            negative_slope=cfg.attn_negative_slope,
            dropout_p=cfg.dropout_p,
        )

        self.combine = nn.Linear(2 * d, d, bias=False)

        ffn_hidden = cfg.ffn_hidden_mult * d
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout_p) if cfg.dropout_p > 0 else nn.Identity(),
            nn.Linear(ffn_hidden, d),
        )
        self.ln = nn.LayerNorm(d)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.combine.weight)
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: Tensor, edge_index: Tensor) -> Tensor:
        m_in = self.in_stream(h, edge_index, mode="in")  # [N, d]
        m_out = self.out_stream(h, edge_index, mode="out")  # [N, d]
        # Eq. (29): m̃_v^{(ℓ)} = W_combine^{(ℓ)} [ m̃_v^{(ℓ,in)} || m̃_v^{(ℓ,out)} ]
        m = self.combine(torch.cat([m_in, m_out], dim=-1))  # [N, d]
        # Eq. (30): h_v^{(ℓ+1)} = LN( h_v^{(ℓ)} + FFN^{(ℓ)}( m̃_v^{(ℓ)} ) )
        h_next = self.ln(h + self.ffn(m))
        return h_next


class BiGATv2Encoder(nn.Module):
    """
    Encoder: conditioned init + K bidirectional GATv2 layers (Eq. 16–31).
    """

    def __init__(self, *, x_dim: int, g_dim: int, cfg: BiGATv2Config):
        super().__init__()
        self.cfg = cfg
        self.init = ConditionedInit(
            x_dim=x_dim, g_dim=g_dim, d_model=cfg.d_model, dropout_p=cfg.dropout_p
        )
        self.layers = nn.ModuleList([BiGATv2Layer(cfg) for _ in range(cfg.num_layers)])

    def forward(self, x: Tensor, edge_index: Tensor, g: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        h = self.init(x=x, g=g, batch=batch)
        for layer in self.layers:
            h = layer(h, edge_index=edge_index)
        return h


def _make_mlp(in_dim: int, hidden_dims: Sequence[int], out_dim: int, dropout_p: float) -> nn.Sequential:
    dims = [in_dim, *hidden_dims, out_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
    return nn.Sequential(*layers)


class EdgeMLPDecoder(nn.Module):
    """
    Decoder: MLP per edge (Eq. 32–33).

      f_uv = [h_u || h_v || g]
      lhat_uv = MLP(f_uv)
    """

    def __init__(self, *, d_model: int, g_dim: int, hp: DecoderHParams):
        super().__init__()
        self.d_model = d_model
        self.g_dim = g_dim
        self.hp = hp
        self.mlp = _make_mlp(
            in_dim=2 * d_model + g_dim,
            hidden_dims=list(hp.hidden_dims),
            out_dim=1,
            dropout_p=hp.dropout_p,
        )

    def forward(
        self,
        h: Tensor,
        edge_index: Tensor,
        g: Tensor,
        node_batch: Optional[Tensor] = None,
    ) -> Tensor:
        src = edge_index[0]
        dst = edge_index[1]
        g_e = _broadcast_global_to_edges(g=g, node_batch=node_batch, src_nodes=src)  # [E, g_dim]
        # Eq. (32): f_uv = [h_u || h_v || g]
        f = torch.cat([h[src], h[dst], g_e], dim=-1)  # [E, 2d + g]
        # Eq. (33): lhat_uv = MLP(f_uv)
        logits = self.mlp(f).squeeze(-1)  # [E]
        return logits


class GCSFlowPredictor(nn.Module):
    """
    End-to-end model: encoder + edge decoder producing per-edge logits.
    """

    def __init__(
        self,
        *,
        x_dim: int,
        g_dim: int,
        encoder_hp: EncoderHParams = EncoderHParams(),
        decoder_hp: DecoderHParams = DecoderHParams(),
    ):
        super().__init__()
        enc_cfg = BiGATv2Config.from_hparams(encoder_hp)
        self.encoder = BiGATv2Encoder(x_dim=x_dim, g_dim=g_dim, cfg=enc_cfg)
        self.decoder = EdgeMLPDecoder(d_model=enc_cfg.d_model, g_dim=g_dim, hp=decoder_hp)

    def encode(
        self,
        *,
        x: Tensor,
        edge_index: Tensor,
        g: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """Return node embeddings h_v for the current graph/batch."""
        return self.encoder(x=x, edge_index=edge_index, g=g, batch=batch)

    def decode_edges(
        self,
        *,
        node_embeddings: Tensor,
        edge_index: Tensor,
        g: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """Return edge logits aligned with `edge_index` from precomputed node embeddings."""
        return self.decoder(
            h=node_embeddings,
            edge_index=edge_index,
            g=g,
            node_batch=batch,
        )

    def forward(
        self,
        *,
        x: Tensor,
        edge_index: Tensor,
        g: Tensor,
        batch: Optional[Tensor] = None,
    ) -> GCSFlowOutput:
        """Return both node embeddings and edge logits."""
        h = self.encode(x=x, edge_index=edge_index, g=g, batch=batch)
        logits = self.decode_edges(
            node_embeddings=h,
            edge_index=edge_index,
            g=g,
            batch=batch,
        )
        return GCSFlowOutput(node_embeddings=h, edge_logits=logits)


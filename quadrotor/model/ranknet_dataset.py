"""
RankNet dataset for the quadrotor HDF5.

Reads inline node_features per instance (no CSV needed).
Vertex names: "v{i}" for regions, "source", "target".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from quadrotor.model.dataset import _decode_str, _vertex_name_to_idx


@dataclass
class RankNetSample:
    x: Tensor               # [N+2, 9]
    edge_index: Tensor      # [2, E]
    g: Tensor               # [6]
    phi_star: Tensor        # [E]
    plan_id: Tensor
    source_idx: int
    target_idx: int
    path_node_indices: Tensor   # [P, L] padded with -1
    path_edge_indices: Tensor   # [P, L] padded with -1
    path_mask: Tensor           # [P, L] bool
    better_idx: Tensor          # [K]
    worse_idx: Tensor           # [K]
    p_bar: Tensor | None = None # [K] soft targets


@dataclass(frozen=True)
class _SampleIndex:
    instance_id: int
    split: str


class QuadrotorRankNetDataset(Dataset):
    def __init__(
        self,
        *,
        h5_path: str | Path,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
        soft_targets: bool = True,
        soft_targets_tau: float = 0.2,
        drop_infeasible_pairs: bool = True,
    ):
        self.h5_path = Path(h5_path)
        self.split = split
        self.max_samples = max_samples
        self.soft_targets = soft_targets
        self._tau = soft_targets_tau
        self.drop_infeasible_pairs = drop_infeasible_pairs

        if not self.h5_path.exists():
            raise FileNotFoundError(self.h5_path)

        self._samples = self._index_h5()

    def _index_h5(self) -> list[_SampleIndex]:
        import h5py

        out: list[_SampleIndex] = []
        with h5py.File(self.h5_path, "r") as h5:
            for k in h5["samples"].keys():
                grp = h5["samples"][k]
                if "candidate_paths" not in grp:
                    continue
                sp = _decode_str(grp.attrs.get("split", ""))
                if self.split is None or sp == self.split:
                    out.append(_SampleIndex(instance_id=int(k), split=sp))
        out.sort(key=lambda s: s.instance_id)
        if self.max_samples is not None:
            out = out[: int(self.max_samples)]
        return out

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> RankNetSample:
        import h5py

        sample = self._samples[idx]
        with h5py.File(self.h5_path, "r") as h5:
            grp = h5["samples"][str(sample.instance_id)]

            x_np = np.array(grp["node_features"][()], dtype=np.float32)
            g_np = np.array(grp["g"][()], dtype=np.float32)
            phi_np = np.array(grp["phi_star"][()], dtype=np.float32)
            edge_u = [_decode_str(s) for s in grp["edge_u"][()]]
            edge_v = [_decode_str(s) for s in grp["edge_v"][()]]

            source_idx = int(np.argmax(x_np[:, 7]))
            target_idx = int(np.argmax(x_np[:, 8]))

            src = [_vertex_name_to_idx(u, source_idx, target_idx) for u in edge_u]
            dst = [_vertex_name_to_idx(v, source_idx, target_idx) for v in edge_v]
            edge_index = torch.stack(
                [torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)], dim=0
            )
            edge_lookup = {(u, v): i for i, (u, v) in enumerate(zip(edge_u, edge_v))}

            path_nodes: list[list[int]] = []
            path_edges: list[list[int]] = []
            ranks: list[int] = []
            rounded_costs: list[float] = []

            for key in sorted(grp["candidate_paths"].keys()):
                cand = grp["candidate_paths"][key]
                cu = [_decode_str(s) for s in cand["edge_u"][()]]
                cv = [_decode_str(s) for s in cand["edge_v"][()]]
                if len(cu) == 0:
                    continue
                nodes = [_vertex_name_to_idx(cu[0], source_idx, target_idx)] + [
                    _vertex_name_to_idx(v, source_idx, target_idx) for v in cv
                ]
                edges = [-1] + [edge_lookup[(u, v)] for u, v in zip(cu, cv)]
                path_nodes.append(nodes)
                path_edges.append(edges)
                ranks.append(int(cand.attrs.get("rank", -1)))
                rounded_costs.append(float(cand.attrs.get("rounded_cost", np.nan)))

        if not path_nodes:
            raise RuntimeError(f"No candidate paths for instance_id={sample.instance_id}")

        max_len = max(len(p) for p in path_nodes)
        node_arr = torch.full((len(path_nodes), max_len), -1, dtype=torch.long)
        edge_arr = torch.full((len(path_edges), max_len), -1, dtype=torch.long)
        mask = torch.zeros((len(path_nodes), max_len), dtype=torch.bool)
        for i, (nodes, edges) in enumerate(zip(path_nodes, path_edges)):
            L = len(nodes)
            node_arr[i, :L] = torch.tensor(nodes, dtype=torch.long)
            edge_arr[i, :L] = torch.tensor(edges, dtype=torch.long)
            mask[i, :L] = True

        better: list[int] = []
        worse: list[int] = []
        p_bar_vals: list[float] = []
        for i in range(len(ranks)):
            for j in range(len(ranks)):
                if i == j:
                    continue
                i_feas = ranks[i] >= 0
                j_feas = ranks[j] >= 0
                if i_feas and not j_feas and not self.drop_infeasible_pairs:
                    better.append(i)
                    worse.append(j)
                    p_bar_vals.append(1.0)
                elif i_feas and j_feas and rounded_costs[i] < rounded_costs[j]:
                    better.append(i)
                    worse.append(j)
                    if self.soft_targets:
                        gap = rounded_costs[j] - rounded_costs[i]
                        p_bar_vals.append(float(1.0 / (1.0 + np.exp(-gap / self._tau))))
                    else:
                        p_bar_vals.append(1.0)

        p_bar = torch.tensor(p_bar_vals, dtype=torch.float32) if p_bar_vals else None

        return RankNetSample(
            x=torch.tensor(x_np),
            edge_index=edge_index,
            g=torch.tensor(g_np),
            phi_star=torch.tensor(phi_np),
            plan_id=torch.tensor(sample.instance_id, dtype=torch.long),
            source_idx=source_idx,
            target_idx=target_idx,
            path_node_indices=node_arr,
            path_edge_indices=edge_arr,
            path_mask=mask,
            better_idx=torch.tensor(better, dtype=torch.long),
            worse_idx=torch.tensor(worse, dtype=torch.long),
            p_bar=p_bar,
        )


def collate_ranknet_samples(samples: list[RankNetSample]) -> list[RankNetSample]:
    return samples

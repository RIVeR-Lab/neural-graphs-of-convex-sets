from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def _decode_str(x: Any) -> str:
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8")
    return str(x)


def _read_node_features_csv(path: Path) -> tuple[list[str], Tensor]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"Empty CSV: {path}")
        x_cols = [c for c in reader.fieldnames if c.startswith("x_")]
        names: list[str] = []
        rows: list[list[float]] = []
        for row in reader:
            names.append(row["node_name"])
            rows.append([float(row[c]) for c in x_cols])
    return names, torch.tensor(rows, dtype=torch.float32)


@dataclass(frozen=True)
class RankNetSampleIndex:
    plan_id: int
    split: str


@dataclass
class RankNetSample:
    x: Tensor
    edge_index: Tensor
    g: Tensor
    phi_star: Tensor
    plan_id: Tensor
    source_idx: int
    target_idx: int
    path_node_indices: Tensor
    path_edge_indices: Tensor
    path_mask: Tensor
    better_idx: Tensor
    worse_idx: Tensor
    # Per-pair soft target P̄ ∈ (0,1]. None means all pairs use hard labels (P̄=1).
    p_bar: Tensor | None = None


class RankNetH5Dataset(Dataset[RankNetSample]):
    def __init__(
        self,
        *,
        h5_path: str | Path,
        node_features_csv: str | Path,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
        soft_targets: bool = True,
        soft_targets_tau: float = 0.2,
        drop_infeasible_pairs: bool = True,
    ):
        self.h5_path = Path(h5_path)
        self.node_features_csv = Path(node_features_csv)
        self.split = split
        self.max_samples = max_samples
        self.soft_targets = soft_targets
        self.drop_infeasible_pairs = drop_infeasible_pairs
        self._tau: float = soft_targets_tau  # updated by _fit_tau after indexing

        if not self.h5_path.exists():
            raise FileNotFoundError(self.h5_path)
        if not self.node_features_csv.exists():
            raise FileNotFoundError(self.node_features_csv)

        self._node_names, self._x_static = _read_node_features_csv(self.node_features_csv)
        self._name_to_idx = {name: i for i, name in enumerate(self._node_names)}
        self._ensure_node("source")
        self._ensure_node("target")
        self._samples = self._index_h5()
        if self.soft_targets and soft_targets_tau == 1.0:
            self._fit_tau()

    def _ensure_node(self, name: str) -> None:
        if name in self._name_to_idx:
            return
        x_dim = int(self._x_static.size(1))
        self._name_to_idx[name] = int(self._x_static.size(0))
        self._node_names.append(name)
        self._x_static = torch.cat([self._x_static, torch.zeros((1, x_dim), dtype=torch.float32)], dim=0)

    def _fit_tau(self) -> None:
        """Set tau = std of feas-vs-feas cost gaps across all training samples so tau=1.0 is meaningful."""
        try:
            import h5py  # type: ignore
        except ModuleNotFoundError:
            return
        gaps: list[float] = []
        with h5py.File(self.h5_path, "r") as h5:
            for s in self._samples:
                grp = h5["samples"][str(s.plan_id)]
                if "candidate_paths" not in grp:
                    continue
                costs: list[float] = []
                feas: list[bool] = []
                for key in sorted(grp["candidate_paths"].keys()):
                    cand = grp["candidate_paths"][key]
                    rank = int(cand.attrs.get("rank", -1))
                    cost = float(cand.attrs.get("rounded_cost", float("nan")))
                    feas.append(rank >= 0)
                    costs.append(cost)
                for i in range(len(costs)):
                    for j in range(len(costs)):
                        if i != j and feas[i] and feas[j] and costs[i] < costs[j]:
                            gaps.append(costs[j] - costs[i])
        if len(gaps) > 1:
            std = float(np.std(gaps))
            if std > 0:
                self._tau = std

    def _index_h5(self) -> list[RankNetSampleIndex]:
        try:
            import h5py  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("Missing `h5py`. Install it with `pip install h5py`.") from e

        out: list[RankNetSampleIndex] = []
        with h5py.File(self.h5_path, "r") as h5:
            samples = h5["samples"]
            for key in samples.keys():
                grp = samples[key]
                if "candidate_paths" not in grp:
                    continue
                split = _decode_str(grp.attrs.get("split", ""))
                if self.split is None or split == self.split:
                    out.append(RankNetSampleIndex(plan_id=int(grp.attrs.get("plan_id", int(key))), split=split))
        out.sort(key=lambda s: s.plan_id)
        if self.max_samples is not None:
            out = out[: int(self.max_samples)]
        return out

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> RankNetSample:
        try:
            import h5py  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("Missing `h5py`. Install it with `pip install h5py`.") from e

        sample = self._samples[idx]
        with h5py.File(self.h5_path, "r") as h5:
            grp = h5["samples"][str(sample.plan_id)]
            edge_u = [_decode_str(s) for s in grp["edge_u"][()]]
            edge_v = [_decode_str(s) for s in grp["edge_v"][()]]

            for name in set(edge_u) | set(edge_v):
                self._ensure_node(name)

            src = torch.tensor([self._name_to_idx[u] for u in edge_u], dtype=torch.long)
            dst = torch.tensor([self._name_to_idx[v] for v in edge_v], dtype=torch.long)
            edge_index = torch.stack([src, dst], dim=0)
            edge_lookup = {(u, v): i for i, (u, v) in enumerate(zip(edge_u, edge_v))}

            g = torch.tensor(np.array(grp["g"][()], dtype=np.float32), dtype=torch.float32)
            phi_star = torch.tensor(np.array(grp["phi_star"][()], dtype=np.float32), dtype=torch.float32)

            path_nodes: list[list[int]] = []
            path_edges: list[list[int]] = []
            ranks: list[int] = []
            rounded_costs: list[float] = []
            candidates = grp["candidate_paths"]
            for key in sorted(candidates.keys()):
                cand = candidates[key]
                cand_u = [_decode_str(s) for s in cand["edge_u"][()]]
                cand_v = [_decode_str(s) for s in cand["edge_v"][()]]
                if len(cand_u) == 0:
                    continue
                nodes = [self._name_to_idx[cand_u[0]]] + [self._name_to_idx[v] for v in cand_v]
                edge_ids = [-1] + [edge_lookup[(u, v)] for u, v in zip(cand_u, cand_v)]
                path_nodes.append(nodes)
                path_edges.append(edge_ids)
                ranks.append(int(cand.attrs.get("rank", -1)))
                rounded_costs.append(float(cand.attrs.get("rounded_cost", np.nan)))

        if len(path_nodes) == 0:
            raise RuntimeError(f"No candidate paths found for plan_id={sample.plan_id}")

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
                    # Hard label: feasible always beats infeasible.
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

        p_bar: Tensor | None = None
        if p_bar_vals:
            p_bar = torch.tensor(p_bar_vals, dtype=torch.float32)

        return RankNetSample(
            x=self._x_static,
            edge_index=edge_index,
            g=g,
            phi_star=phi_star,
            plan_id=torch.tensor(sample.plan_id, dtype=torch.long),
            source_idx=int(self._name_to_idx["source"]),
            target_idx=int(self._name_to_idx["target"]),
            path_node_indices=node_arr,
            path_edge_indices=edge_arr,
            path_mask=mask,
            better_idx=torch.tensor(better, dtype=torch.long),
            worse_idx=torch.tensor(worse, dtype=torch.long),
            p_bar=p_bar,
        )


def collate_ranknet_samples(samples: list[RankNetSample]) -> list[RankNetSample]:
    return samples

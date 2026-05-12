"""Datasets that feed raw region halfspaces (A, b) to the PointNet flow model.

Works for both problems via a small config:
  - quadrotor: regions stored per instance (samples/<id>/regions/<i>), vertex
    names "v{i}" (convex) or "Subgraph0: Region{i}" (nonconvex).
  - manipulation: regions shared at file level (/regions/<name>), vertex names
    are region names (convex) or "Subgraph0: Region{i}" (nonconvex).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data, Dataset

_REGION_RE = re.compile(r"Region(\d+)")


def normalize_facets(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return [m, d+1] facet tokens with unit normals (b scaled by same norm)."""
    A = np.asarray(A, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 1)
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    return np.concatenate([A / norms, b / norms], axis=1).astype(np.float32)


def vertex_name_to_index(
    name: str, num_regions: int, name_to_idx: Optional[dict[str, int]]
) -> int:
    if name == "source" or name.startswith("source"):
        return num_regions
    if name == "target" or name.startswith("target"):
        return num_regions + 1
    m = _REGION_RE.search(name)
    if m is not None:
        return int(m.group(1))
    if name.startswith("v") and name[1:].isdigit():
        return int(name[1:])
    if name_to_idx is not None and name in name_to_idx:
        return name_to_idx[name]
    raise KeyError(f"Cannot map vertex name {name!r} to an index.")


def _decode(s: Any) -> str:
    return s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)


class FacetGCSData(Data):
    def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "g":
            return None  # stack -> [B, g_dim]
        return super().__cat_dim__(key, value, *args, **kwargs)


@dataclass(frozen=True)
class _Index:
    instance_id: int
    split: str


class _FacetH5Base:
    """Shared region/facet loading logic for both dataset variants."""

    def __init__(self, h5_path: str | Path, regions_scope: str):
        self.h5_path = Path(h5_path)
        if regions_scope not in ("shared", "per_instance"):
            raise ValueError(regions_scope)
        self.regions_scope = regions_scope
        if not self.h5_path.exists():
            raise FileNotFoundError(self.h5_path)

        import h5py

        with h5py.File(self.h5_path, "r") as h5:
            if "regions" not in h5 and self.regions_scope == "shared":
                raise RuntimeError(f"{self.h5_path} has no shared /regions group.")

            self._region_names: Optional[list[str]] = None
            self._shared_ab: Optional[list[np.ndarray]] = None
            self._name_to_idx: Optional[dict[str, int]] = None

            if self.regions_scope == "shared":
                names = [_decode(s) for s in h5["meta"]["region_names"][()]]
                self._region_names = names
                self._name_to_idx = {n: i for i, n in enumerate(names)}
                self._shared_ab = [
                    normalize_facets(h5["regions"][n]["A"][()], h5["regions"][n]["b"][()])
                    for n in names
                ]
                self.fmax = max(t.shape[0] for t in self._shared_ab)
                self.facet_dim = self._shared_ab[0].shape[1]
            else:
                fmax = 0
                facet_dim = None
                for k in h5["samples"].keys():
                    grp = h5["samples"][k]
                    if "regions" not in grp:
                        raise RuntimeError(
                            f"{self.h5_path} instance {k} has no regions group."
                        )
                    for rk in grp["regions"].keys():
                        a = grp["regions"][rk]["A"]
                        fmax = max(fmax, a.shape[0])
                        facet_dim = a.shape[1] + 1
                self.fmax = fmax
                self.facet_dim = int(facet_dim)

    def _load_region_ab(self, h5, instance_grp) -> list[np.ndarray]:
        if self.regions_scope == "shared":
            return self._shared_ab  # type: ignore[return-value]
        reg = instance_grp["regions"]
        keys = sorted(reg.keys(), key=lambda x: int(x))
        return [normalize_facets(reg[k]["A"][()], reg[k]["b"][()]) for k in keys]

    def _build_facets(self, ab_list: list[np.ndarray]) -> tuple[Tensor, Tensor, Tensor, int]:
        n_regions = len(ab_list)
        n_nodes = n_regions + 2  # + source + target
        facets = torch.zeros((n_nodes, self.fmax, self.facet_dim), dtype=torch.float32)
        mask = torch.zeros((n_nodes, self.fmax), dtype=torch.bool)
        flags = torch.zeros((n_nodes, 3), dtype=torch.float32)
        for i, tok in enumerate(ab_list):
            m = tok.shape[0]
            facets[i, :m] = torch.from_numpy(tok)
            mask[i, :m] = True
            flags[i, 0] = 1.0  # is_region
        flags[n_regions, 1] = 1.0      # source
        flags[n_regions + 1, 2] = 1.0  # target
        return facets, mask, flags, n_regions

    def _edge_index(self, edge_u, edge_v, n_regions: int) -> Tensor:
        src = [vertex_name_to_index(_decode(u), n_regions, self._name_to_idx) for u in edge_u]
        dst = [vertex_name_to_index(_decode(v), n_regions, self._name_to_idx) for v in edge_v]
        return torch.tensor([src, dst], dtype=torch.long)


class FacetGCSDataset(_FacetH5Base, Dataset):
    """Phase-1 flow dataset: per-edge phi_star regression target."""

    def __init__(
        self,
        *,
        h5_path: str | Path,
        regions_scope: str,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        _FacetH5Base.__init__(self, h5_path, regions_scope)
        Dataset.__init__(self)
        self.split = split
        self.max_samples = max_samples
        self._samples = self._index()

    def _index(self) -> list[_Index]:
        import h5py

        out: list[_Index] = []
        with h5py.File(self.h5_path, "r") as h5:
            for k in h5["samples"].keys():
                sp = _decode(h5["samples"][k].attrs.get("split", ""))
                if self.split is None or sp == self.split:
                    out.append(_Index(int(k), sp))
        out.sort(key=lambda s: s.instance_id)
        if self.max_samples is not None:
            out = out[: int(self.max_samples)]
        return out

    def len(self) -> int:
        return len(self._samples)

    def get(self, idx: int) -> FacetGCSData:
        import h5py

        sample = self._samples[idx]
        with h5py.File(self.h5_path, "r") as h5:
            grp = h5["samples"][str(sample.instance_id)]
            ab_list = self._load_region_ab(h5, grp)
            g_np = np.array(grp["g"][()], dtype=np.float32)
            phi_np = np.array(grp["phi_star"][()], dtype=np.float32)
            edge_u = grp["edge_u"][()]
            edge_v = grp["edge_v"][()]

        facets, mask, flags, n_regions = self._build_facets(ab_list)
        edge_index = self._edge_index(edge_u, edge_v, n_regions)

        return FacetGCSData(
            facets=facets,
            facet_mask=mask,
            node_flags=flags,
            edge_index=edge_index,
            y=torch.from_numpy(phi_np),
            phi_star=torch.from_numpy(phi_np),
            g=torch.from_numpy(g_np),
            num_nodes=facets.size(0),
            plan_id=torch.tensor(sample.instance_id, dtype=torch.long),
        )


@dataclass
class FacetRankNetSample:
    facets: Tensor
    facet_mask: Tensor
    node_flags: Tensor
    edge_index: Tensor
    g: Tensor
    source_idx: int
    target_idx: int
    num_nodes: int
    path_node_indices: Tensor
    path_edge_indices: Tensor
    path_mask: Tensor
    better_idx: Tensor
    worse_idx: Tensor
    p_bar: Tensor | None
    plan_id: Tensor


class FacetRankNetDataset(_FacetH5Base, TorchDataset):
    """Phase-2 dataset: candidate paths + RankNet pair labels."""

    def __init__(
        self,
        *,
        h5_path: str | Path,
        regions_scope: str,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
        soft_targets: bool = True,
        soft_targets_tau: float = 0.2,
        drop_infeasible_pairs: bool = True,
    ):
        _FacetH5Base.__init__(self, h5_path, regions_scope)
        TorchDataset.__init__(self)
        self.split = split
        self.max_samples = max_samples
        self.soft_targets = soft_targets
        self._tau = soft_targets_tau
        self.drop_infeasible_pairs = drop_infeasible_pairs
        self._samples = self._index()

    def _index(self) -> list[_Index]:
        import h5py

        out: list[_Index] = []
        with h5py.File(self.h5_path, "r") as h5:
            for k in h5["samples"].keys():
                grp = h5["samples"][k]
                if "candidate_paths" not in grp:
                    continue
                sp = _decode(grp.attrs.get("split", ""))
                if self.split is None or sp == self.split:
                    out.append(_Index(int(k), sp))
        out.sort(key=lambda s: s.instance_id)
        if self.max_samples is not None:
            out = out[: int(self.max_samples)]
        return out

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> FacetRankNetSample:
        import h5py

        sample = self._samples[idx]
        with h5py.File(self.h5_path, "r") as h5:
            grp = h5["samples"][str(sample.instance_id)]
            ab_list = self._load_region_ab(h5, grp)
            g_np = np.array(grp["g"][()], dtype=np.float32)
            edge_u = [_decode(s) for s in grp["edge_u"][()]]
            edge_v = [_decode(s) for s in grp["edge_v"][()]]

            facets, mask, flags, n_regions = self._build_facets(ab_list)
            source_idx, target_idx = n_regions, n_regions + 1
            src = [vertex_name_to_index(u, n_regions, self._name_to_idx) for u in edge_u]
            dst = [vertex_name_to_index(v, n_regions, self._name_to_idx) for v in edge_v]
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_lookup = {(u, v): i for i, (u, v) in enumerate(zip(edge_u, edge_v))}

            path_nodes: list[list[int]] = []
            path_edges: list[list[int]] = []
            ranks: list[int] = []
            costs: list[float] = []
            for key in sorted(grp["candidate_paths"].keys()):
                cand = grp["candidate_paths"][key]
                cu = [_decode(s) for s in cand["edge_u"][()]]
                cv = [_decode(s) for s in cand["edge_v"][()]]
                if len(cu) == 0:
                    continue
                nodes = [vertex_name_to_index(cu[0], n_regions, self._name_to_idx)] + [
                    vertex_name_to_index(v, n_regions, self._name_to_idx) for v in cv
                ]
                edges = [-1] + [edge_lookup[(u, v)] for u, v in zip(cu, cv)]
                path_nodes.append(nodes)
                path_edges.append(edges)
                ranks.append(int(cand.attrs.get("rank", -1)))
                costs.append(float(cand.attrs.get("rounded_cost", np.nan)))

        if not path_nodes:
            raise RuntimeError(f"No candidate paths for instance {sample.instance_id}")

        max_len = max(len(p) for p in path_nodes)
        node_arr = torch.full((len(path_nodes), max_len), -1, dtype=torch.long)
        edge_arr = torch.full((len(path_edges), max_len), -1, dtype=torch.long)
        pmask = torch.zeros((len(path_nodes), max_len), dtype=torch.bool)
        for i, (nodes, edges) in enumerate(zip(path_nodes, path_edges)):
            L = len(nodes)
            node_arr[i, :L] = torch.tensor(nodes, dtype=torch.long)
            edge_arr[i, :L] = torch.tensor(edges, dtype=torch.long)
            pmask[i, :L] = True

        better, worse, p_vals = [], [], []
        for i in range(len(ranks)):
            for j in range(len(ranks)):
                if i == j:
                    continue
                i_feas, j_feas = ranks[i] >= 0, ranks[j] >= 0
                if i_feas and not j_feas and not self.drop_infeasible_pairs:
                    better.append(i); worse.append(j); p_vals.append(1.0)
                elif i_feas and j_feas and costs[i] < costs[j]:
                    better.append(i); worse.append(j)
                    if self.soft_targets:
                        gap = costs[j] - costs[i]
                        p_vals.append(float(1.0 / (1.0 + np.exp(-gap / self._tau))))
                    else:
                        p_vals.append(1.0)

        p_bar = torch.tensor(p_vals, dtype=torch.float32) if p_vals else None

        return FacetRankNetSample(
            facets=facets,
            facet_mask=mask,
            node_flags=flags,
            edge_index=edge_index,
            g=torch.from_numpy(g_np),
            source_idx=source_idx,
            target_idx=target_idx,
            num_nodes=facets.size(0),
            path_node_indices=node_arr,
            path_edge_indices=edge_arr,
            path_mask=pmask,
            better_idx=torch.tensor(better, dtype=torch.long),
            worse_idx=torch.tensor(worse, dtype=torch.long),
            p_bar=p_bar,
            plan_id=torch.tensor(sample.instance_id, dtype=torch.long),
        )


def collate_ranknet(samples: list[FacetRankNetSample]) -> list[FacetRankNetSample]:
    return samples

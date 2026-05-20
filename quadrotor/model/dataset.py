"""
PyG dataset for the quadrotor GCS HDF5 produced by collect_quadrotor_data.py.

Node features are stored inline per instance as (N+2, 9):
  cols 0-5: [xmin,xmax,ymin,ymax,zmin,zmax] (box bounds; 0 for source/target rows)
  col  6:   is_region
  col  7:   is_source
  col  8:   is_target

Vertex names in the HDF5 are "v0".."vN-1" for regions, "source", "target".
Row index in node_features equals the vertex's position in gcs.Vertices() at collection time,
which matches: "vi" → row i, "source" → source_node_idx, "target" → target_node_idx.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data, Dataset


class GCSData(Data):
    def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "g":
            return None  # stack to [B, g_dim] instead of concatenate
        return super().__cat_dim__(key, value, *args, **kwargs)


def _decode_str(x: Any) -> str:
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8")
    return str(x)


def _vertex_name_to_idx(name: str, source_idx: int, target_idx: int) -> int:
    if name == "source":
        return source_idx
    if name == "target":
        return target_idx
    # Region vertices are named "v{i}" where i is their row in node_features
    return int(name[1:])


@dataclass(frozen=True)
class _SampleIndex:
    instance_id: int
    split: str


class QuadrotorGCSDataset(Dataset):
    """
    Each item is a GCSData with:
      x          [N+2, 9]  node features
      edge_index [2, E]    directed edges
      y          [E]       phi_star (regression target)
      g          [6]       global vec [p_start, p_goal]
      phi_star   [E]       same as y
      plan_id    scalar int64
    """

    def __init__(
        self,
        *,
        h5_path: str | Path,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.h5_path = Path(h5_path)
        self.split = split
        self.max_samples = max_samples
        if not self.h5_path.exists():
            raise FileNotFoundError(self.h5_path)
        self._samples: list[_SampleIndex] = self._index_h5()

    def _index_h5(self) -> list[_SampleIndex]:
        import h5py

        out: list[_SampleIndex] = []
        with h5py.File(self.h5_path, "r") as h5:
            for k in h5["samples"].keys():
                grp = h5["samples"][k]
                sp = _decode_str(grp.attrs.get("split", ""))
                if self.split is None or sp == self.split:
                    out.append(_SampleIndex(instance_id=int(k), split=sp))
        out.sort(key=lambda s: s.instance_id)
        if self.max_samples is not None:
            out = out[: int(self.max_samples)]
        return out

    def len(self) -> int:
        return len(self._samples)

    def get(self, idx: int) -> GCSData:
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

        src = torch.tensor(
            [_vertex_name_to_idx(u, source_idx, target_idx) for u in edge_u], dtype=torch.long
        )
        dst = torch.tensor(
            [_vertex_name_to_idx(v, source_idx, target_idx) for v in edge_v], dtype=torch.long
        )

        x = torch.tensor(x_np)
        g = torch.tensor(g_np)
        phi_star = torch.tensor(phi_np)

        return GCSData(
            x=x,
            edge_index=torch.stack([src, dst], dim=0),
            y=phi_star,
            g=g,
            phi_star=phi_star,
            plan_id=torch.tensor(sample.instance_id, dtype=torch.long),
        )

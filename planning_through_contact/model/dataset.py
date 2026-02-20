from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data, Dataset


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
        if len(x_cols) == 0:
            raise RuntimeError(f"No x_* columns found in {path}")

        names: list[str] = []
        rows: list[list[float]] = []
        for row in reader:
            names.append(row["node_name"])
            rows.append([float(row[c]) for c in x_cols])

    x = torch.tensor(rows, dtype=torch.float32)
    return names, x


@dataclass(frozen=True)
class H5SampleIndex:
    plan_id: int
    split: str


class GCSH5Dataset(Dataset):
    """
    PyG dataset for the logged solutions HDF5 produced by `collect_solutions.py`.

    Each item yields a `torch_geometric.data.Data` with:
      - x: [N, x_dim] static node features (plus optional source/target rows)
      - edge_index: [2, E] directed
      - y: [E] binary edge labels
      - g: [g_dim] global conditioning vector
      - phi_star: [E] teacher relaxation flows (optional; present in HDF5)
      - plan_id: scalar int64
    """

    def __init__(
        self,
        *,
        h5_path: str | Path,
        node_features_csv: str | Path = "planning_through_contact/dataset/data/node_features.csv",
        include_source_target: bool = True,
        split: Optional[str] = None,
    ):
        super().__init__()
        self.h5_path = Path(h5_path)
        self.node_features_csv = Path(node_features_csv)
        self.include_source_target = include_source_target
        self.split = split

        if not self.h5_path.exists():
            raise FileNotFoundError(self.h5_path)
        if not self.node_features_csv.exists():
            raise FileNotFoundError(self.node_features_csv)

        self._node_names, self._x_static = _read_node_features_csv(self.node_features_csv)
        self._name_to_idx = {n: i for i, n in enumerate(self._node_names)}

        if include_source_target:
            self._ensure_node("source")
            self._ensure_node("target")

        self._samples: list[H5SampleIndex] = self._index_h5()

    def _ensure_node(self, name: str) -> None:
        if name in self._name_to_idx:
            return
        x_dim = int(self._x_static.size(1))
        self._name_to_idx[name] = int(self._x_static.size(0))
        self._node_names.append(name)
        self._x_static = torch.cat([self._x_static, torch.zeros((1, x_dim), dtype=torch.float32)], dim=0)

    def _index_h5(self) -> list[H5SampleIndex]:
        try:
            import h5py  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("Missing `h5py`. Install it (pip install h5py).") from e

        out: list[H5SampleIndex] = []
        with h5py.File(self.h5_path, "r") as h5:
            if "samples" not in h5:
                raise RuntimeError(f"HDF5 missing group 'samples': {self.h5_path}")
            samples = h5["samples"]
            for k in samples.keys():
                grp = samples[k]
                pid = int(grp.attrs.get("plan_id", int(k)))
                split = _decode_str(grp.attrs.get("split", ""))
                if self.split is None or split == self.split:
                    out.append(H5SampleIndex(plan_id=pid, split=split))
        out.sort(key=lambda s: s.plan_id)
        return out

    @property
    def node_names(self) -> list[str]:
        return list(self._node_names)

    @property
    def x_static(self) -> Tensor:
        return self._x_static

    def len(self) -> int:  # type: ignore[override]
        return len(self._samples)

    def get(self, idx: int) -> Data:  # type: ignore[override]
        try:
            import h5py  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("Missing `h5py`. Install it (pip install h5py).") from e

        sample = self._samples[idx]
        with h5py.File(self.h5_path, "r") as h5:
            grp = h5["samples"][str(sample.plan_id)]

            edge_u = [_decode_str(s) for s in grp["edge_u"][()]]
            edge_v = [_decode_str(s) for s in grp["edge_v"][()]]

            for n in set(edge_u) | set(edge_v):
                if n not in self._name_to_idx:
                    # Be robust to unexpected nodes by adding zero features.
                    self._ensure_node(n)

            src = torch.tensor([self._name_to_idx[u] for u in edge_u], dtype=torch.long)
            dst = torch.tensor([self._name_to_idx[v] for v in edge_v], dtype=torch.long)
            edge_index = torch.stack([src, dst], dim=0)

            y = torch.tensor(np.array(grp["y"][()], dtype=np.int64), dtype=torch.long)  # [E]
            g = torch.tensor(np.array(grp["g"][()], dtype=np.float32), dtype=torch.float32)  # [g_dim]
            phi_star = torch.tensor(
                np.array(grp["phi_star"][()], dtype=np.float32), dtype=torch.float32
            )  # [E]

            plan_id_t = torch.tensor(int(sample.plan_id), dtype=torch.long)

        data = Data(
            x=self._x_static,
            edge_index=edge_index,
            y=y,
            g=g,
            phi_star=phi_star,
            plan_id=plan_id_t,
        )
        return data


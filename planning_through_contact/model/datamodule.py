from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

from planning_through_contact.model.dataset import GCSH5Dataset


@dataclass(frozen=True)
class DatasetPaths:
    h5_path: str | Path = "planning_through_contact/dataset/data/box_pushing/gcs_solutions.h5"
    node_features_csv: str | Path = "planning_through_contact/dataset/data/box_pushing/node_features.csv"


class GCSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        paths: DatasetPaths,
        batch_size: int = 32,
        num_workers: int = 0,
        include_source_target: bool = True,
        target: Literal["discrete", "sdp"] = "sdp",
    ):
        super().__init__()
        self.paths = paths
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.include_source_target = bool(include_source_target)
        self.target = target

        self.ds_train: Optional[GCSH5Dataset] = None
        self.ds_val: Optional[GCSH5Dataset] = None
        self.ds_test: Optional[GCSH5Dataset] = None

        self._pos_weight: Optional[torch.Tensor] = None

    @property
    def pos_weight(self) -> torch.Tensor:
        if self._pos_weight is None:
            raise RuntimeError("pos_weight not computed yet. Call setup('fit') first.")
        return self._pos_weight

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.ds_train = GCSH5Dataset(
                h5_path=self.paths.h5_path,
                node_features_csv=self.paths.node_features_csv,
                include_source_target=self.include_source_target,
                split="train",
                target=self.target,
            )
            self.ds_val = GCSH5Dataset(
                h5_path=self.paths.h5_path,
                node_features_csv=self.paths.node_features_csv,
                include_source_target=self.include_source_target,
                split="test",
                target=self.target,
            )
            self._pos_weight = (
                self._compute_pos_weight(self.ds_train)
                if self.target == "discrete"
                else torch.tensor(1.0, dtype=torch.float32)
            )

        if stage in (None, "test"):
            self.ds_test = GCSH5Dataset(
                h5_path=self.paths.h5_path,
                node_features_csv=self.paths.node_features_csv,
                include_source_target=self.include_source_target,
                split="test",
                target=self.target,
            )

    def _compute_pos_weight(self, ds: GCSH5Dataset) -> torch.Tensor:
        # Eq. (35): w_pos = |{e : y_e = 0}| / |{e : y_e = 1}|   (aggregated over the training set)
        pos = 0
        neg = 0
        for i in range(len(ds)):
            y = ds.get(i).y
            pos_i = int((y == 1).sum().item())
            neg_i = int((y == 0).sum().item())
            pos += pos_i
            neg += neg_i
        if pos <= 0:
            raise RuntimeError("No positive edges in training set; cannot compute pos_weight.")
        wpos = float(neg) / float(pos)
        return torch.tensor(wpos, dtype=torch.float32)

    def train_dataloader(self) -> DataLoader:
        if self.ds_train is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return PyGDataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        if self.ds_val is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return PyGDataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        if self.ds_test is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return PyGDataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


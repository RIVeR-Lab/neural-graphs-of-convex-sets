from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader

from quadrotor.model.dataset import QuadrotorGCSDataset


class QuadrotorGCSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        h5_path: str | Path,
        batch_size: int = 32,
        num_workers: int = 0,
        max_train_samples: Optional[int] = None,
    ):
        super().__init__()
        self.h5_path = Path(h5_path)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.max_train_samples = max_train_samples

        self.ds_train: Optional[QuadrotorGCSDataset] = None
        self.ds_val: Optional[QuadrotorGCSDataset] = None
        self.ds_test: Optional[QuadrotorGCSDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.ds_train = QuadrotorGCSDataset(
                h5_path=self.h5_path, split="train", max_samples=self.max_train_samples
            )
            self.ds_val = QuadrotorGCSDataset(h5_path=self.h5_path, split="val")

        if stage in (None, "test"):
            self.ds_test = QuadrotorGCSDataset(h5_path=self.h5_path, split="test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

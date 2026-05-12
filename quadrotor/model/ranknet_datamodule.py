from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from quadrotor.model.ranknet_dataset import QuadrotorRankNetDataset, collate_ranknet_samples


class QuadrotorRankNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        h5_path: str | Path,
        batch_size: int = 1,
        num_workers: int = 0,
        max_train_samples: Optional[int] = None,
        soft_targets: bool = True,
        soft_targets_tau: float = 0.2,
        drop_infeasible_pairs: bool = True,
    ):
        super().__init__()
        self.h5_path = Path(h5_path)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.max_train_samples = max_train_samples
        self.soft_targets = soft_targets
        self.soft_targets_tau = soft_targets_tau
        self.drop_infeasible_pairs = drop_infeasible_pairs

        self.ds_train: Optional[QuadrotorRankNetDataset] = None
        self.ds_val: Optional[QuadrotorRankNetDataset] = None
        self.ds_test: Optional[QuadrotorRankNetDataset] = None

    def _make_ds(self, split: str, max_samples: Optional[int] = None) -> QuadrotorRankNetDataset:
        return QuadrotorRankNetDataset(
            h5_path=self.h5_path,
            split=split,
            max_samples=max_samples,
            soft_targets=self.soft_targets,
            soft_targets_tau=self.soft_targets_tau,
            drop_infeasible_pairs=self.drop_infeasible_pairs,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.ds_train = self._make_ds("train", self.max_train_samples)
            self.ds_val = self._make_ds("val")
        if stage in (None, "test"):
            self.ds_test = self._make_ds("test")

    def _loader(self, ds: QuadrotorRankNetDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_ranknet_samples,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.ds_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.ds_val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.ds_test, shuffle=False)

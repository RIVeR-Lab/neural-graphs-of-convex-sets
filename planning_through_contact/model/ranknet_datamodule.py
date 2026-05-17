from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from planning_through_contact.model.datamodule import DatasetPaths
from planning_through_contact.model.ranknet_dataset import (
    RankNetH5Dataset,
    collate_ranknet_samples,
)


class RankNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        paths: DatasetPaths,
        batch_size: int = 1,
        num_workers: int = 0,
        max_train_samples: Optional[int] = None,
        soft_targets: bool = True,
        soft_targets_tau: float = 0.2,
        drop_infeasible_pairs: bool = True,
    ):
        super().__init__()
        self.paths = paths
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.max_train_samples = max_train_samples
        self.soft_targets = soft_targets
        self.soft_targets_tau = soft_targets_tau
        self.drop_infeasible_pairs = drop_infeasible_pairs

        self.ds_train: Optional[RankNetH5Dataset] = None
        self.ds_val: Optional[RankNetH5Dataset] = None
        self.ds_test: Optional[RankNetH5Dataset] = None

    @staticmethod
    def for_body(
        body: str,
        *,
        data_root: str | Path = "planning_through_contact/dataset/data",
        batch_size: int = 1,
        num_workers: int = 0,
        max_train_samples: Optional[int] = None,
        soft_targets: bool = True,
        soft_targets_tau: float = 0.2,
        drop_infeasible_pairs: bool = True,
    ) -> "RankNetDataModule":
        return RankNetDataModule(
            paths=DatasetPaths.for_body(body, data_root=data_root),
            batch_size=batch_size,
            num_workers=num_workers,
            max_train_samples=max_train_samples,
            soft_targets=soft_targets,
            soft_targets_tau=soft_targets_tau,
            drop_infeasible_pairs=drop_infeasible_pairs,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.ds_train = RankNetH5Dataset(
                h5_path=self.paths.h5_path,
                node_features_csv=self.paths.node_features_csv,
                split="train",
                max_samples=self.max_train_samples,
                soft_targets=self.soft_targets,
                soft_targets_tau=self.soft_targets_tau,
                drop_infeasible_pairs=self.drop_infeasible_pairs,
            )
            self.ds_val = RankNetH5Dataset(
                h5_path=self.paths.h5_path,
                node_features_csv=self.paths.node_features_csv,
                split="val",
                soft_targets=self.soft_targets,
                soft_targets_tau=self.ds_train._tau,  # use tau fitted on train
                drop_infeasible_pairs=self.drop_infeasible_pairs,
            )

        if stage in (None, "test"):
            train_tau = self.ds_train._tau if self.ds_train is not None else self.soft_targets_tau
            self.ds_test = RankNetH5Dataset(
                h5_path=self.paths.h5_path,
                node_features_csv=self.paths.node_features_csv,
                split="test",
                soft_targets=self.soft_targets,
                soft_targets_tau=train_tau,
                drop_infeasible_pairs=self.drop_infeasible_pairs,
            )

    def train_dataloader(self) -> DataLoader:
        if self.ds_train is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_ranknet_samples,
        )

    def val_dataloader(self) -> DataLoader:
        if self.ds_val is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_ranknet_samples,
        )

    def test_dataloader(self) -> DataLoader:
        if self.ds_test is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_ranknet_samples,
        )

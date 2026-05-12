from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model.checkpoint_utils import (
    CHECKPOINT_ROOT,
    dataset_paths_for_body,
    flow_checkpoint_path,
    ranker_checkpoint_name,
    validate_body,
)
from model.cuda_required import require_cuda
from model.hparams import DecoderHParams, EncoderHParams, TrainingHParams
from model.model import GCSFlowPredictor
from model.ranknet import RankNetConfig
from model.ranknet_datamodule import RankNetDataModule
from model.ranknet_lightning_module import RankNetLightningModule


def load_flow_model_from_checkpoint(
    ckpt_path: str | Path,
    *,
    x_dim: int,
    g_dim: int,
    encoder_hp: EncoderHParams,
    decoder_hp: DecoderHParams,
) -> GCSFlowPredictor:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model_state = {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    model = GCSFlowPredictor(
        x_dim=x_dim,
        g_dim=g_dim,
        encoder_hp=encoder_hp,
        decoder_hp=decoder_hp,
    )
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Could not load flow checkpoint: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model


def main() -> None:
    require_cuda()
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", type=str, default="sugar_box", choices=["sugar_box", "tee"])
    parser.add_argument("--data_root", type=str, default="planning_through_contact/dataset/data")
    parser.add_argument("--flow_ckpt_path", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=str(CHECKPOINT_ROOT))
    parser.add_argument("--experiment", type=str, default="ranknet")
    parser.add_argument("--ranker_ckpt_filename", type=str, default=None)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ffn_hidden_mult", type=int, default=2)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--decoder_hidden", type=str, default="256,256")
    parser.add_argument("--decoder_dropout_p", type=float, default=0.1)

    parser.add_argument("--ranker_layers", type=int, default=3)
    parser.add_argument("--ranker_heads", type=int, default=4)
    parser.add_argument("--ranker_ffn_hidden", type=int, default=256)
    parser.add_argument("--ranker_score_hidden", type=int, default=64)
    parser.add_argument("--ranker_dropout_p", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--no_lr_schedule", action="store_true")
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--lr_scheduler_eta_min", type=float, default=1e-6)
    parser.add_argument("--no_early_stopping", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=25)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)

    parser.add_argument("--no_soft_targets", action="store_true", help="Use hard 0/1 labels instead of soft P̄.")
    parser.add_argument("--soft_targets_tau", type=float, default=0.2, help="Temperature for soft targets.")
    parser.add_argument("--keep_infeasible_pairs", action="store_true", help="Include feas-vs-infeas pairs in training (default: dropped).")

    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="gnn-for-gcs")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="runs")
    args = parser.parse_args()

    body = validate_body(args.body)
    body_paths = dataset_paths_for_body(body, data_root=args.data_root)
    flow_ckpt_path = (
        Path(args.flow_ckpt_path)
        if args.flow_ckpt_path is not None
        else flow_checkpoint_path(body, root=args.ckpt_dir)
    )
    ranker_ckpt_filename = args.ranker_ckpt_filename or ranker_checkpoint_name(body)

    dm = RankNetDataModule.for_body(
        body,
        data_root=args.data_root,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        max_train_samples=args.max_train_samples,
        soft_targets=not bool(args.no_soft_targets),
        soft_targets_tau=float(args.soft_targets_tau),
        drop_infeasible_pairs=not bool(args.keep_infeasible_pairs),
    )
    dm.setup("fit")
    if dm.ds_train is None or len(dm.ds_train) == 0:
        raise RuntimeError("No RankNet training samples found.")
    sample = dm.ds_train[0]
    x_dim = int(sample.x.size(1))
    g_dim = int(sample.g.numel())

    encoder_hp = EncoderHParams(
        d_model=int(args.d_model),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        ffn_hidden_mult=int(args.ffn_hidden_mult),
        dropout_p=float(args.dropout_p),
    )
    hidden = tuple(int(s) for s in str(args.decoder_hidden).split(",") if s.strip())
    decoder_hp = DecoderHParams(hidden_dims=hidden, dropout_p=float(args.decoder_dropout_p))
    flow_model = load_flow_model_from_checkpoint(
        flow_ckpt_path,
        x_dim=x_dim,
        g_dim=g_dim,
        encoder_hp=encoder_hp,
        decoder_hp=decoder_hp,
    )

    train_hp = TrainingHParams(
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        max_epochs=int(args.max_epochs),
        use_lr_schedule=not bool(args.no_lr_schedule),
        warmup_epochs=int(args.warmup_epochs),
        lr_scheduler_eta_min=float(args.lr_scheduler_eta_min),
        use_early_stopping=not bool(args.no_early_stopping),
        early_stop_patience=int(args.early_stop_patience),
        early_stop_min_delta=float(args.early_stop_min_delta),
    )
    ranker_cfg = RankNetConfig(
        d_model=int(args.d_model),
        num_layers=int(args.ranker_layers),
        num_heads=int(args.ranker_heads),
        ffn_hidden_dim=int(args.ranker_ffn_hidden),
        score_hidden_dim=int(args.ranker_score_hidden),
        dropout_p=float(args.ranker_dropout_p),
    )
    lit = RankNetLightningModule(flow_model=flow_model, ranker_cfg=ranker_cfg, train_hp=train_hp)

    if not args.no_wandb:
        from pytorch_lightning.loggers import WandbLogger  # type: ignore

        logger = WandbLogger(
            project=args.wandb_project,
            entity=None if args.wandb_entity in (None, "", "none") else args.wandb_entity,
            name=args.wandb_run_name or f"{body}_ranknet",
        )
    else:
        logger = TensorBoardLogger(save_dir=args.log_dir, name=args.experiment)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(Path(args.ckpt_dir) / body),
            filename=ranker_ckpt_filename,
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            every_n_epochs=1,
            enable_version_counter=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if train_hp.use_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                mode="min",
                patience=int(train_hp.early_stop_patience),
                min_delta=float(train_hp.early_stop_min_delta),
                verbose=True,
            )
        )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=int(args.max_epochs),
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )
    trainer.fit(lit, datamodule=dm)


if __name__ == "__main__":
    main()

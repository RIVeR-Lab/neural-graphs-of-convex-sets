from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from planning_through_contact.model.cuda_required import require_cuda
from planning_through_contact.model.datamodule import DatasetPaths, GCSDataModule
from planning_through_contact.model.hparams import DecoderHParams, EncoderHParams, TrainingHParams
from planning_through_contact.model.lightning_module import GCSLightningModule


def main() -> None:
    require_cuda()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5_path",
        type=str,
        default="planning_through_contact/dataset/data/gcs_solutions.h5",
        help="Path to solutions HDF5.",
    )
    parser.add_argument(
        "--node_features_csv",
        type=str,
        default="planning_through_contact/dataset/data/node_features.csv",
        help="Static node features CSV.",
    )
    parser.add_argument("--x_dim", type=int, default=None, help="Override node feature dimension.")
    parser.add_argument("--g_dim", type=int, default=None, help="Override global feature dimension.")

    parser.add_argument(
        "--target",
        type=str,
        choices=["discrete", "sdp"],
        default="sdp",
        help="Regression target: discrete 0/1 path labels or SDP relaxation flows (phi_star).",
    )

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ffn_hidden_mult", type=int, default=2)
    parser.add_argument("--dropout_p", type=float, default=0.1)

    parser.add_argument("--decoder_hidden", type=str, default="256,256", help="Comma-separated dims.")
    parser.add_argument("--decoder_dropout_p", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--no_lr_schedule", action="store_true", help="Disable cosine warmup LR schedule.")
    parser.add_argument("--warmup_epochs", type=int, default=20, help="Epochs of linear LR warmup.")
    parser.add_argument("--lr_scheduler_eta_min", type=float, default=1e-6, help="Min LR for cosine phase.")
    parser.add_argument("--no_early_stopping", action="store_true", help="Disable early stopping.")
    parser.add_argument("--early_stop_patience", type=int, default=50, help="Epochs to wait before stopping.")
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4, help="Min change to qualify as improvement.")

    parser.add_argument("--log_dir", type=str, default="runs", help="TensorBoard log directory.")
    parser.add_argument("--experiment", type=str, default="gcs_gnn")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Where to save .ckpt files.")

    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases; use TensorBoard instead.")
    parser.add_argument("--wandb_project", type=str, default="gnn-for-gcs")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name (e.g. 'exp1_sugar_box'). If not set, uses --experiment.",
    )
    args = parser.parse_args()

    paths = DatasetPaths(h5_path=Path(args.h5_path), node_features_csv=Path(args.node_features_csv))
    dm = GCSDataModule(
        paths=paths,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        include_source_target=True,
        target=args.target,
    )
    dm.setup("fit")

    # Infer x_dim and g_dim from a sample unless overridden.
    sample = dm.ds_train.get(0)  # type: ignore[union-attr]
    x_dim = int(sample.x.size(1)) if args.x_dim is None else int(args.x_dim)
    g_dim = int(sample.g.numel()) if args.g_dim is None else int(args.g_dim)

    encoder_hp = EncoderHParams(
        d_model=int(args.d_model),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        ffn_hidden_mult=int(args.ffn_hidden_mult),
        dropout_p=float(args.dropout_p),
    )
    hidden = tuple(int(s) for s in str(args.decoder_hidden).split(",") if s.strip() != "")
    decoder_hp = DecoderHParams(hidden_dims=hidden, dropout_p=float(args.decoder_dropout_p))
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

    lit = GCSLightningModule(
        x_dim=x_dim,
        g_dim=g_dim,
        pos_weight=dm.pos_weight,
        target=args.target,
        encoder_hp=encoder_hp,
        decoder_hp=decoder_hp,
        train_hp=train_hp,
    )

    if not args.no_wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ModuleNotFoundError(
                "W&B is on by default but WandbLogger is unavailable. "
                "Install `wandb` and retry, or run with --no_wandb to use TensorBoard."
            ) from e

        run_name = (args.wandb_run_name or args.experiment).strip() or args.experiment
        logger = WandbLogger(
            project=str(args.wandb_project),
            entity=None if args.wandb_entity in (None, "", "none") else str(args.wandb_entity),
            name=run_name,
        )
        logger.log_hyperparams(
            {
                "target": args.target,
                "x_dim": x_dim,
                "g_dim": g_dim,
                "pos_weight": float(dm.pos_weight.item()),
                "encoder": encoder_hp.__dict__,
                "decoder": decoder_hp.__dict__,
                "train": train_hp.__dict__,
                "data": {
                    "h5_path": str(args.h5_path),
                    "node_features_csv": str(args.node_features_csv),
                },
            }
        )
    else:
        logger = TensorBoardLogger(save_dir=args.log_dir, name=args.experiment)

    ckpt = ModelCheckpoint(
        dirpath=str(Path(args.ckpt_dir) / args.experiment),
        filename="epoch{epoch:03d}-val_loss{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
    )
    callbacks = [ckpt]
    if train_hp.use_early_stopping:
        early_stop = EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=int(train_hp.early_stop_patience),
            min_delta=float(train_hp.early_stop_min_delta),
            verbose=True,
        )
        callbacks.append(early_stop)
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


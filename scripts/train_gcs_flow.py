#!/usr/bin/env python3
"""
Unified GCS training: PointNet-over-facets flow GNN (Phase 1) + PathRankNet (Phase 2).

Trains one model family selected by --problem and --planner:
  quadrotor/convex      -> quadrotor/dataset/quadrotor_gcs_convex.h5     (per-instance regions)
  quadrotor/nonlinear   -> quadrotor/dataset/quadrotor_gcs_nonlinear.h5  (per-instance regions)
  manipulation/convex   -> manipulation/data/iiwa_gcs_linear.h5          (shared regions)
  manipulation/nonlinear-> manipulation/data/iiwa_gcs_nonlinear.h5       (shared regions)

Examples:
  python scripts/train_gcs_flow.py --problem manipulation --planner convex --no_wandb
  python scripts/train_gcs_flow.py --problem quadrotor --planner nonlinear --phase flow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader

from planning_through_contact.model.facet_dataset import (
    FacetGCSDataset,
    FacetRankNetDataset,
    collate_ranknet,
)
from planning_through_contact.model.facet_lightning import FacetFlowModule, FacetRankNetModule
from planning_through_contact.model.facet_pointnet import PointNetFlowPredictor
from planning_through_contact.model.hparams import DecoderHParams, EncoderHParams, TrainingHParams
from planning_through_contact.model.ranknet import RankNetConfig

DATASETS = {
    ("quadrotor", "convex"): ("quadrotor/dataset/quadrotor_gcs_convex.h5", "per_instance"),
    ("quadrotor", "nonlinear"): ("quadrotor/dataset/quadrotor_gcs_nonlinear.h5", "per_instance"),
    ("manipulation", "convex"): ("manipulation/data/iiwa_gcs_linear.h5", "shared"),
    ("manipulation", "nonlinear"): ("manipulation/data/iiwa_gcs_nonlinear.h5", "shared"),
}


def make_encoder_hp(args) -> EncoderHParams:
    return EncoderHParams(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_hidden_mult=args.ffn_hidden_mult,
        dropout_p=args.dropout_p,
    )


def make_decoder_hp(args) -> DecoderHParams:
    hidden = tuple(int(s) for s in args.decoder_hidden.split(",") if s.strip())
    return DecoderHParams(hidden_dims=hidden, dropout_p=args.decoder_dropout_p)


def make_train_hp(args, *, max_epochs, lr, warmup, patience) -> TrainingHParams:
    return TrainingHParams(
        lr=lr,
        weight_decay=args.weight_decay,
        max_epochs=max_epochs,
        use_lr_schedule=not args.no_lr_schedule,
        warmup_epochs=warmup,
        use_early_stopping=not args.no_early_stopping,
        early_stop_patience=patience,
    )


def _callbacks(ckpt_dir: Path, filename: str, patience: int, use_es: bool):
    cbs = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename=filename,
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=f"{filename}_last",
            enable_version_counter=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if use_es:
        cbs.append(EarlyStopping(monitor="val/loss", mode="min", patience=patience, verbose=True))
    return cbs


def _wandb_run_name(tag: str, phase: str) -> str:
    """e.g. tag='quadrotor_convex', phase='ranknet' -> 'quadrotor convex ranknet'."""
    problem, planner = tag.rsplit("_", 1)
    return f"{problem} {planner} {phase}"


def _finish_wandb(args) -> None:
    """Close the active W&B run so the next phase gets its own run."""
    if args.no_wandb:
        return
    import wandb

    if wandb.run is not None:
        wandb.finish()


def _logger(args, experiment: str):
    if not args.no_wandb:
        from pytorch_lightning.loggers import WandbLogger

        _finish_wandb(args)
        return WandbLogger(project=args.wandb_project, name=experiment)
    return TensorBoardLogger(save_dir=args.log_dir, name=experiment)


def run_flow_phase(args, h5_path, scope, tag, ckpt_dir) -> Path:
    ds_train = FacetGCSDataset(
        h5_path=h5_path, regions_scope=scope, split="train", max_samples=args.max_train_samples
    )
    ds_val = FacetGCSDataset(h5_path=h5_path, regions_scope=scope, split="val")
    facet_dim = ds_train.facet_dim
    g_dim = int(ds_train.get(0).g.numel())

    train_loader = GeoDataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = GeoDataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    train_hp = make_train_hp(args, max_epochs=args.flow_epochs, lr=args.flow_lr,
                             warmup=args.flow_warmup, patience=args.flow_patience)
    module = FacetFlowModule(
        facet_dim=facet_dim, g_dim=g_dim,
        encoder_hp=make_encoder_hp(args), decoder_hp=make_decoder_hp(args),
        train_hp=train_hp, pointnet_hidden=args.pointnet_hidden,
    )
    filename = f"{tag}_flow_gnn"
    trainer = pl.Trainer(
        accelerator=args.accelerator, devices=1, max_epochs=args.flow_epochs,
        logger=_logger(args, _wandb_run_name(tag, "flow")),
        callbacks=_callbacks(ckpt_dir, filename, args.flow_patience, not args.no_early_stopping),
        log_every_n_steps=10,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best = trainer.checkpoint_callback.best_model_path
    print(f"[{tag}] flow checkpoint: {best}")
    _finish_wandb(args)
    return Path(best)


def load_flow_model(ckpt_path, *, facet_dim, g_dim, encoder_hp, decoder_hp, pointnet_hidden):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    model_state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    model = PointNetFlowPredictor(
        facet_dim=facet_dim, g_dim=g_dim, encoder_hp=encoder_hp,
        decoder_hp=decoder_hp, pointnet_hidden=pointnet_hidden,
    )
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Flow ckpt mismatch: missing={missing[:4]} unexpected={unexpected[:4]}")
    return model


def run_ranker_phase(args, h5_path, scope, tag, ckpt_dir, flow_ckpt) -> Path:
    ds_train = FacetRankNetDataset(
        h5_path=h5_path, regions_scope=scope, split="train",
        max_samples=args.max_train_samples,
        soft_targets=not args.no_soft_targets, soft_targets_tau=args.soft_targets_tau,
    )
    ds_val = FacetRankNetDataset(
        h5_path=h5_path, regions_scope=scope, split="val",
        soft_targets=not args.no_soft_targets, soft_targets_tau=args.soft_targets_tau,
    )
    facet_dim = ds_train.facet_dim
    g_dim = int(ds_train[0].g.numel())

    train_loader = DataLoader(ds_train, batch_size=args.ranker_batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_ranknet)
    val_loader = DataLoader(ds_val, batch_size=args.ranker_batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_ranknet)

    encoder_hp, decoder_hp = make_encoder_hp(args), make_decoder_hp(args)
    flow_model = load_flow_model(
        flow_ckpt, facet_dim=facet_dim, g_dim=g_dim,
        encoder_hp=encoder_hp, decoder_hp=decoder_hp, pointnet_hidden=args.pointnet_hidden,
    )
    ranker_cfg = RankNetConfig(
        d_model=args.d_model, num_layers=args.ranker_layers, num_heads=args.ranker_heads,
    )
    train_hp = make_train_hp(args, max_epochs=args.ranker_epochs, lr=args.ranker_lr,
                             warmup=args.ranker_warmup, patience=args.ranker_patience)
    module = FacetRankNetModule(flow_model=flow_model, ranker_cfg=ranker_cfg, train_hp=train_hp)

    filename = f"{tag}_ranknet"
    trainer = pl.Trainer(
        accelerator=args.accelerator, devices=1, max_epochs=args.ranker_epochs,
        logger=_logger(args, _wandb_run_name(tag, "ranknet")),
        callbacks=_callbacks(ckpt_dir, filename, args.ranker_patience, not args.no_early_stopping),
        log_every_n_steps=10,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best = trainer.checkpoint_callback.best_model_path
    print(f"[{tag}] ranknet checkpoint: {best}")
    _finish_wandb(args)
    return Path(best)


def main() -> None:
    p = argparse.ArgumentParser(description="Train GCS PointNet flow GNN + RankNet.")
    p.add_argument("--problem", required=True, choices=("quadrotor", "manipulation"))
    p.add_argument("--planner", required=True, choices=("convex", "nonlinear"))
    p.add_argument("--phase", default="both", choices=("flow", "ranker", "both"))
    p.add_argument("--h5_path", type=str, default=None, help="Override default dataset path.")
    p.add_argument("--ckpt_dir", type=str, default=None)
    p.add_argument("--flow_ckpt", type=str, default=None, help="Phase-1 ckpt for --phase ranker.")

    # architecture
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--ffn_hidden_mult", type=int, default=2)
    p.add_argument("--dropout_p", type=float, default=0.1)
    p.add_argument("--decoder_hidden", type=str, default="256,256")
    p.add_argument("--decoder_dropout_p", type=float, default=0.1)
    p.add_argument("--pointnet_hidden", type=int, default=64)
    p.add_argument("--ranker_layers", type=int, default=3)
    p.add_argument("--ranker_heads", type=int, default=4)

    # training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--ranker_batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--flow_epochs", type=int, default=300)
    p.add_argument("--flow_lr", type=float, default=1e-3)
    p.add_argument("--flow_warmup", type=int, default=20)
    p.add_argument("--flow_patience", type=int, default=50)
    p.add_argument("--ranker_epochs", type=int, default=100)
    p.add_argument("--ranker_lr", type=float, default=3e-3)
    p.add_argument("--ranker_warmup", type=int, default=10)
    p.add_argument("--ranker_patience", type=int, default=25)
    p.add_argument("--soft_targets_tau", type=float, default=0.2)
    p.add_argument("--no_soft_targets", action="store_true")
    p.add_argument("--no_lr_schedule", action="store_true")
    p.add_argument("--no_early_stopping", action="store_true")

    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--log_dir", type=str, default="runs")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="gnn-for-gcs")
    args = p.parse_args()

    default_h5, scope = DATASETS[(args.problem, args.planner)]
    h5_path = args.h5_path or str(REPO_ROOT / default_h5)
    if not Path(h5_path).is_file():
        print(f"Dataset not found: {h5_path}", file=sys.stderr)
        sys.exit(1)

    tag = f"{args.problem}_{args.planner}"
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else REPO_ROOT / "checkpoints" / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Training {tag} | data={h5_path} | scope={scope} ===")

    flow_ckpt = Path(args.flow_ckpt) if args.flow_ckpt else None
    if args.phase in ("flow", "both"):
        flow_ckpt = run_flow_phase(args, h5_path, scope, tag, ckpt_dir)

    if args.phase in ("ranker", "both"):
        if flow_ckpt is None or not Path(flow_ckpt).is_file():
            print("Need --flow_ckpt for ranker phase.", file=sys.stderr)
            sys.exit(1)
        run_ranker_phase(args, h5_path, scope, tag, ckpt_dir, flow_ckpt)

    print(f"\nDone: {tag}. Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()

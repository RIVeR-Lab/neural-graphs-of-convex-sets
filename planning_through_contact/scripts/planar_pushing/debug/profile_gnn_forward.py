"""
Profile the GNN forward pass only: run N times, discard first `warmup`, average the rest.
Run from repo root: python3 planning_through_contact/scripts/planar_pushing/debug/profile_gnn_forward.py --ckpt_path checkpoints/gcs_gnn/sugar_box_flow.ckpt [--num_runs 10] [--warmup 5]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from planning_through_contact.model.drake_rounding import build_gnn_batch_for_planner
from planning_through_contact.model.hparams import DecoderHParams, EncoderHParams
from planning_through_contact.model.model import GCSFlowPredictor
from planning_through_contact.planning.planar.planar_pushing_planner import PlanarPushingPlanner
from planning_through_contact.experiments.utils import get_default_experiment_plans, get_default_plan_config
from planning_through_contact.scripts.planar_pushing.plan_with_gnn import (
    compute_g,
    load_gcs_flow_predictor_from_lightning_ckpt,
    resolve_torch_device,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile GNN forward pass (no QP, no rounding).")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Lightning .ckpt from train_gcs_gnn.py")
    parser.add_argument(
        "--node_features_csv",
        type=str,
        default=None,
    )
    parser.add_argument("--body", type=str, default="sugar_box")
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="cpu",
        help="Device for model inference. Use 'cuda' for GPU or 'auto' to use CUDA when available.",
    )
    parser.add_argument("--num_runs", type=int, default=10, help="Total forward passes (warmup + timed).")
    parser.add_argument("--warmup", type=int, default=5, help="Discard first N runs; average the rest.")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ffn_hidden_mult", type=int, default=2)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--decoder_hidden", type=str, default="256,256")
    parser.add_argument("--decoder_dropout_p", type=float, default=0.1)
    args = parser.parse_args()

    if args.warmup >= args.num_runs:
        raise ValueError("warmup must be < num_runs")
    if args.node_features_csv is None:
        args.node_features_csv = str(Path("planning_through_contact/dataset/data") / args.body / "node_features.csv")

    config = get_default_plan_config(slider_type=args.body, pusher_radius=0.015, use_case="normal")
    plans = get_default_experiment_plans(seed=0, num_trajs=1, config=config)
    plan = plans[0]

    # Infer x_dim from CSV
    import csv as _csv
    with open(args.node_features_csv, newline="") as f:
        r = _csv.DictReader(f)
        if r.fieldnames is None:
            raise RuntimeError(f"Empty CSV: {args.node_features_csv}")
        x_dim = len([c for c in r.fieldnames if c.startswith("x_")])
    g_dim = 6 + int(config.slider_geometry.num_collision_free_regions)

    encoder_hp = EncoderHParams(
        d_model=int(args.d_model),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        ffn_hidden_mult=int(args.ffn_hidden_mult),
        dropout_p=float(args.dropout_p),
    )
    hidden = tuple(int(s) for s in str(args.decoder_hidden).split(",") if s.strip() != "")
    decoder_hp = DecoderHParams(hidden_dims=hidden, dropout_p=float(args.decoder_dropout_p))

    model = load_gcs_flow_predictor_from_lightning_ckpt(
        args.ckpt_path,
        x_dim=x_dim,
        g_dim=g_dim,
        encoder_hp=encoder_hp,
        decoder_hp=decoder_hp,
        map_location="cpu",
    )
    device = resolve_torch_device(args.device)
    model.to(device)
    model.eval()

    config.start_and_goal = plan
    planner = PlanarPushingPlanner(config)
    planner.formulate_problem()
    g = compute_g(config, plan)

    x, edge_index, g_t = build_gnn_batch_for_planner(
        planner=planner,
        g=g,
        node_features_csv=args.node_features_csv,
    )
    x = x.to(device)
    edge_index = edge_index.to(device)
    g_t = g_t.to(device)

    # Single loop: first `warmup` runs discarded for stats, but we record and print all runs
    all_times_ms: list[float] = []
    for i in range(args.num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x=x, edge_index=edge_index, g=g_t, batch=None)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        all_times_ms.append(elapsed_ms)

    print(f"GNN forward pass profile (device={device}, {args.num_runs} runs, first {args.warmup} warmup)")
    for i, t in enumerate(all_times_ms):
        warmup_tag = " (warmup)" if i < args.warmup else ""
        print(f"  run {i:2d}: {t:.3f} ms{warmup_tag}")

    times_ms = all_times_ms[args.warmup:]
    n = len(times_ms)
    mean_ms = sum(times_ms) / n
    variance = sum((t - mean_ms) ** 2 for t in times_ms) / n
    std_ms = variance ** 0.5

    print(f"  --- stats over runs {args.warmup}..{args.num_runs - 1} ---")
    print(f"  mean:   {mean_ms:.3f} ms")
    print(f"  std:    {std_ms:.3f} ms")
    print(f"  min:    {min(times_ms):.3f} ms")
    print(f"  max:    {max(times_ms):.3f} ms")


if __name__ == "__main__":
    main()

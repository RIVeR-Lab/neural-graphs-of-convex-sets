"""Render paper-style trajectory figures (one panel per contact/non-collision segment).

Runs GNN + RankNet on test-set instances and saves a PDF per instance using the
same make_traj_figure style as the paper: ghosted slider + pusher circles with
fading opacity, split by mode type, with a contact/non-contact legend.
"""
from __future__ import annotations

import argparse
import csv as _csv
import logging
import time
from pathlib import Path
from typing import Any

from planning_through_contact.visualize.planar_pushing import make_traj_figure


def main() -> None:
    from planning_through_contact.experiments.utils import (
        get_default_plan_config,
        get_default_solver_params,
    )
    from planning_through_contact.model.ranknet import RankNetConfig
    from planning_through_contact.model.ranknet_inference import (
        load_ranknet_from_checkpoint,
        ranknet_round_from_flow_model,
    )
    from planning_through_contact.model.drake_rounding import (
        predict_edge_flows_for_planner,
        round_from_predicted_flows,
    )
    from planning_through_contact.model.hparams import DecoderHParams, EncoderHParams
    from planning_through_contact.planning.planar.planar_pushing_planner import PlanarPushingPlanner
    from planning_through_contact.scripts.planar_pushing.plan_with_gnn import (
        compute_g,
        load_gcs_flow_predictor_from_lightning_ckpt,
        load_test_plans_from_csv,
        resolve_torch_device,
    )

    logging.getLogger("drake").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--body", type=str, default="sugar_box", choices=["sugar_box", "tee"])
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--max_paths", type=int, default=100)
    parser.add_argument("--num_contact_frames", type=int, default=4)
    parser.add_argument("--num_non_collision_frames", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="figures")
    parser.add_argument("--without_ranknet", action="store_true")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="cuda")
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
    args = parser.parse_args()

    data_dir      = Path(f"planning_through_contact/dataset/data/{args.body}")
    flow_ckpt     = Path(f"checkpoints/gcs_gnn/{args.body}_flow.ckpt")
    ranker_ckpt   = Path(f"checkpoints/ranknet/{args.body}_ranker.ckpt")
    node_feat_csv = str(data_dir / "node_features.csv")
    plan_idx_csv  = data_dir / "global_features.csv"

    test_plans, _ = load_test_plans_from_csv(plan_idx_csv)
    test_plans = test_plans[: args.num]
    if not test_plans:
        raise SystemExit("No test plans found.")

    config        = get_default_plan_config(slider_type=args.body, pusher_radius=0.015, use_case="normal")
    solver_params = get_default_solver_params(False, clarabel=False)
    device        = resolve_torch_device(args.device)

    with open(node_feat_csv, newline="") as f:
        r = _csv.DictReader(f)
        x_dim = len([c for c in r.fieldnames if c.startswith("x_")])
    g_dim = 6 + int(config.slider_geometry.num_collision_free_regions)

    enc_hp = EncoderHParams(
        d_model=args.d_model, num_layers=args.num_layers,
        num_heads=args.num_heads, ffn_hidden_mult=args.ffn_hidden_mult,
        dropout_p=args.dropout_p,
    )
    hidden = tuple(int(s) for s in args.decoder_hidden.split(",") if s.strip())
    dec_hp = DecoderHParams(hidden_dims=hidden, dropout_p=args.decoder_dropout_p)
    flow_model = load_gcs_flow_predictor_from_lightning_ckpt(
        flow_ckpt, x_dim=x_dim, g_dim=g_dim,
        encoder_hp=enc_hp, decoder_hp=dec_hp, map_location="cpu",
    )
    flow_model.to(device)

    ranker = None
    if not args.without_ranknet and ranker_ckpt.exists():
        ranker_cfg = RankNetConfig(
            d_model=args.d_model, num_layers=args.ranker_layers,
            num_heads=args.ranker_heads, ffn_hidden_dim=args.ranker_ffn_hidden,
            score_hidden_dim=args.ranker_score_hidden, dropout_p=args.ranker_dropout_p,
        )
        ranker = load_ranknet_from_checkpoint(str(ranker_ckpt), cfg=ranker_cfg, map_location="cpu")
        ranker.to(device)
        print(f"RankNet loaded: {ranker_ckpt}")
    elif not args.without_ranknet:
        print(f"[warn] RankNet checkpoint not found: {ranker_ckpt} — using GNN only.")

    base = Path(args.output_dir) / args.body
    base.mkdir(parents=True, exist_ok=True)
    existing = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("figure_")]
    next_n = max((int(d.name.split("_")[-1]) for d in existing if d.name.split("_")[-1].isdigit()), default=0) + 1
    out_dir = base / f"figure_{next_n}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (plan_id, plan) in enumerate(test_plans):
        config.start_and_goal = plan
        planner = PlanarPushingPlanner(config)
        planner.formulate_problem()
        g = compute_g(config, plan)

        t0 = time.perf_counter()
        profile: dict[str, Any] = {}
        if ranker is not None:
            path, _ = ranknet_round_from_flow_model(
                planner=planner, flow_model=flow_model, ranker=ranker,
                g=g, node_features_csv=node_feat_csv,
                solver_params=solver_params, max_paths=args.max_paths,
                max_steps=512, seed=idx, device=device, profile=profile,
            )
        else:
            edge_flows = predict_edge_flows_for_planner(
                planner=planner, model=flow_model, g=g,
                node_features_csv=node_feat_csv, device=device,
                enforce_flow_conservation=True, timings=profile,
            )
            path = round_from_predicted_flows(
                planner=planner, edge_flows=edge_flows,
                solver_params=solver_params, max_paths=args.max_paths,
                max_steps=512, seed=idx, profile=profile,
            )
        total_time = time.perf_counter() - t0

        label = f"plan_{plan_id}" if plan_id is not None else f"traj_{idx}"
        if path is None or path.rounded_result is None or not path.rounded_result.is_success():
            print(f"[{label}] planning failed — skipping.")
            continue

        traj = path.to_traj(rounded=True)
        out_path = str(out_dir / f"{label}.pdf")
        make_traj_figure(
            traj,
            filename=out_path,
            split_on_mode_type=True,
            start_end_legend=False,
            plot_lims=None,
            plot_knot_points=False,
            plot_forces=False,
            show_start_pose=False,
            show_goal_pusher=False,
            num_contact_frames=args.num_contact_frames,
            num_non_collision_frames=args.num_non_collision_frames,
            save_individual_panels=True,
            show_contact_legend=True,
        )
        print(f"[{label}] ok  t={total_time:.2f}s  → {out_path}")


if __name__ == "__main__":
    main()

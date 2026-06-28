#!/usr/bin/env python3
"""Plan planar pushing with Neural GCS and render full-IIWA Meshcat HTML.

Mirrors ``demo_quadrotor_neural_motion.py``: load checkpoints, plan one test
instance, then visualize with the IK-only IIWA + tabletop playback hack.

Examples:
  python scripts/demo_planar_pushing_neural_motion.py
  python scripts/demo_planar_pushing_neural_motion.py --body tee --traj 0
  python scripts/demo_planar_pushing_neural_motion.py --body sugar_box --traj 1 --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from planning_through_contact.experiments.utils import (
    get_default_plan_config,
    get_default_solver_params,
)
from planning_through_contact.model.hparams import DecoderHParams, EncoderHParams
from planning_through_contact.model.ranknet import RankNetConfig
from planning_through_contact.model.ranknet_inference import (
    load_ranknet_from_checkpoint,
    ranknet_round_from_flow_model,
)
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPushingPlanner,
)
from planning_through_contact.scripts.planar_pushing.plan_with_gnn import (
    compute_g,
    load_gcs_flow_predictor_from_lightning_ckpt,
    load_test_plans_from_csv,
    resolve_torch_device,
)
from planning_through_contact.visualize.ik_only_playback import render_tabletop_playback

logging.getLogger("drake").setLevel(logging.WARNING)


def plan_with_neural_gcs(
    *,
    body: str,
    plan,
    flow_ckpt: Path,
    ranker_ckpt: Path,
    node_features_csv: Path,
    device: torch.device,
    max_paths: int,
    max_steps: int,
    seed: int,
    encoder_hp: EncoderHParams,
    decoder_hp: DecoderHParams,
    ranker_cfg: RankNetConfig,
    debug: bool,
) -> tuple[Any, float, float]:
    with open(node_features_csv, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"Empty CSV: {node_features_csv}")
        x_dim = len([c for c in reader.fieldnames if c.startswith("x_")])

    config = get_default_plan_config(slider_type=body, pusher_radius=0.015, use_case="normal")
    config.start_and_goal = plan
    solver_params = get_default_solver_params(debug, clarabel=False)

    g_dim = 6 + int(config.slider_geometry.num_collision_free_regions)
    flow_model = load_gcs_flow_predictor_from_lightning_ckpt(
        flow_ckpt,
        x_dim=x_dim,
        g_dim=g_dim,
        encoder_hp=encoder_hp,
        decoder_hp=decoder_hp,
        map_location="cpu",
    )
    flow_model.to(device)
    ranker = load_ranknet_from_checkpoint(ranker_ckpt, cfg=ranker_cfg, map_location="cpu")
    ranker.to(device)

    planner = PlanarPushingPlanner(config)
    planner.formulate_problem()
    g = compute_g(config, plan)

    profile: dict[str, Any] = {}
    t0 = time.perf_counter()
    path, _edge_flows = ranknet_round_from_flow_model(
        planner=planner,
        flow_model=flow_model,
        ranker=ranker,
        g=g,
        node_features_csv=str(node_features_csv),
        solver_params=solver_params,
        max_paths=max_paths,
        max_steps=max_steps,
        seed=seed,
        device=device,
        profile=profile,
    )
    elapsed = time.perf_counter() - t0
    gnn_s = float(profile.get("gnn_s", elapsed))

    ok = path is not None and path.rounded_result is not None and path.rounded_result.is_success()
    if not ok:
        raise RuntimeError("Neural GCS planning failed (rounding/restriction did not succeed).")

    traj = path.to_traj(rounded=True)
    cost = float(path.rounded_result.get_optimal_cost())
    return traj, cost, gnn_s


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neural GCS planar pushing motion demo (IIWA IK playback HTML)."
    )
    parser.add_argument("--body", type=str, default="sugar_box", choices=["sugar_box", "tee"])
    parser.add_argument("--traj", type=int, default=0, help="Index into test-split plans.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--flow_ckpt",
        type=Path,
        default=REPO_ROOT / "checkpoints/gcs_gnn/sugar_box_flow.ckpt",
    )
    parser.add_argument(
        "--ranker_ckpt",
        type=Path,
        default=REPO_ROOT / "checkpoints/ranknet/sugar_box_ranker.ckpt",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "planning_through_contact/results/motion_demo",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max_paths", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--check_this_out_ik",
        action="store_true",
        help=(
            "Use exact check_this_out run_ik_only_playback (per-frame solve_ik). "
            "Fails on dataset plans parked at (-0.3, 0); default is diff-IK sim playback."
        ),
    )
    parser.add_argument(
        "--render_only",
        type=Path,
        default=None,
        help="Skip planning; render Meshcat HTML from an existing traj .pkl.",
    )
    parser.add_argument("--debug", action="store_true")
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

    data_dir = REPO_ROOT / "planning_through_contact/dataset/data" / args.body
    plan_index_csv = data_dir / "global_features.csv"
    node_features_csv = data_dir / "node_features.csv"

    if args.body == "tee":
        if args.flow_ckpt == REPO_ROOT / "checkpoints/gcs_gnn/sugar_box_flow.ckpt":
            args.flow_ckpt = REPO_ROOT / "checkpoints/gcs_gnn/tee_flow.ckpt"
        if args.ranker_ckpt == REPO_ROOT / "checkpoints/ranknet/sugar_box_ranker.ckpt":
            args.ranker_ckpt = REPO_ROOT / "checkpoints/ranknet/tee_ranker.ckpt"

    for path in (args.flow_ckpt, args.ranker_ckpt, plan_index_csv, node_features_csv):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    test_plans, body = load_test_plans_from_csv(plan_index_csv)
    if not test_plans:
        raise SystemExit("No test plans found in global_features.csv (split=='test').")
    if args.traj < 0 or args.traj >= len(test_plans):
        raise SystemExit(f"--traj {args.traj} out of range for {len(test_plans)} test plan(s).")

    plan_id, plan = test_plans[args.traj]
    device = resolve_torch_device(args.device)
    print(f"Body={body}  plan_id={plan_id}  device={device}")

    encoder_hp = EncoderHParams(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_hidden_mult=args.ffn_hidden_mult,
        dropout_p=args.dropout_p,
    )
    hidden = tuple(int(s) for s in args.decoder_hidden.split(",") if s.strip())
    decoder_hp = DecoderHParams(hidden_dims=hidden, dropout_p=args.decoder_dropout_p)
    ranker_cfg = RankNetConfig(
        d_model=args.d_model,
        num_layers=args.ranker_layers,
        num_heads=args.ranker_heads,
        ffn_hidden_dim=args.ranker_ffn_hidden,
        score_hidden_dim=args.ranker_score_hidden,
        dropout_p=args.ranker_dropout_p,
    )

    print("Planning: Neural GCS (GNN + RankNet)...")
    t_plan = time.perf_counter()
    if args.render_only is not None:
        from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
            PlanarPushingTrajectory,
        )

        traj = PlanarPushingTrajectory.load(str(args.render_only))
        cost = float("nan")
        gnn_s = 0.0
        plan_elapsed = 0.0
        plan_id = plan_id if plan_id is not None else args.traj
    else:
        traj, cost, gnn_s = plan_with_neural_gcs(
            body=body,
            plan=plan,
            flow_ckpt=args.flow_ckpt,
            ranker_ckpt=args.ranker_ckpt,
            node_features_csv=node_features_csv,
            device=device,
            max_paths=args.max_paths,
            max_steps=args.max_steps,
            seed=args.seed,
            encoder_hp=encoder_hp,
            decoder_hp=decoder_hp,
            ranker_cfg=ranker_cfg,
            debug=args.debug,
        )
        plan_elapsed = time.perf_counter() - t_plan
    print(
        f"  success  cost={cost:.4f}  duration={traj.end_time:.2f}s  "
        f"gnn={gnn_s:.2f}s  total={plan_elapsed:.2f}s"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"motion_{args.body}_plan{plan_id}_seed{args.seed}"
    traj_pkl = args.output_dir / f"{stem}_traj.pkl"
    traj.save(str(traj_pkl))

    summary = {
        "body": body,
        "plan_id": plan_id,
        "seed": args.seed,
        "cost": cost,
        "duration_s": float(traj.end_time),
        "gnn_s": gnn_s,
        "plan_elapsed_s": plan_elapsed,
        "flow_ckpt": str(args.flow_ckpt),
        "ranker_ckpt": str(args.ranker_ckpt),
        "traj_pkl": str(traj_pkl),
    }
    json_path = args.output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary → {json_path}")

    html_path = args.output_dir / f"{stem}.html"
    playback_mode = "check_this_out_ik" if args.check_this_out_ik else "diff_ik"
    print(f"Rendering IIWA tabletop Meshcat playback ({playback_mode})...")
    render_tabletop_playback(
        traj,
        html_path,
        fps=args.fps,
        mode=playback_mode,
    )
    print(f"Saved Meshcat HTML → {html_path}")
    print("Done.")


if __name__ == "__main__":
    main()

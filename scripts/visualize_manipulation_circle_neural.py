#!/usr/bin/env python3
"""Visualize Neural GCS on the single-arm IIWA shelf circle demo.

The scene is the manipulation environment with one shelf and two bins. The
script plans the circle waypoint sequence using only neural GCS and exports a
play-once Meshcat HTML animation suitable for paper figures or videos.

Examples:
  python scripts/visualize_manipulation_circle_neural.py
  python scripts/visualize_manipulation_circle_neural.py --planner nonlinear --show-line
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from pydrake.geometry import StartMeshcat

from manipulation.iiwa_helpers import build_shelf_plant, visualize_trajectory
from manipulation.paths import DEFAULT_OUTPUT_DIR, DEFAULT_REGIONS_PATH
from manipulation.shelf_gcs import (
    build_demo_sequences,
    build_seed_points,
    load_regions,
    planning_configurations,
)
from manipulation.trajopt import (
    build_nonlinear_gcs_problem,
    build_region_edges,
    iiwa_kinematic_limits,
    region_list,
)
from planning_through_contact.model.hparams import DecoderHParams, EncoderHParams
from planning_through_contact.model.ranknet import RankNetConfig
from quadrotor.gcs.linear import LinearGCS


_eval_spec = importlib.util.spec_from_file_location(
    "eval_manipulation_circle_demo",
    REPO_ROOT / "scripts" / "eval_manipulation_circle_demo.py",
)
eval_demo = importlib.util.module_from_spec(_eval_spec)
assert _eval_spec.loader is not None
sys.modules["eval_manipulation_circle_demo"] = eval_demo
_eval_spec.loader.exec_module(eval_demo)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Neural GCS Meshcat visualization for the IIWA shelf circle demo."
    )
    parser.add_argument("--planner", choices=("convex", "nonlinear"), default="convex")
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "circle_neural",
        help="Directory for the Meshcat HTML and JSON summary.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--max-paths", type=int, default=eval_demo.MAX_PATHS)
    parser.add_argument("--max-trials", type=int, default=eval_demo.MAX_ROUNDING_TRIALS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--flow-ckpt", default=None)
    parser.add_argument("--ranknet-ckpt", default=None)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-hidden-mult", type=int, default=2)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--decoder-hidden", default="256,256")
    parser.add_argument("--decoder-dropout-p", type=float, default=0.1)
    parser.add_argument("--pointnet-hidden", type=int, default=64)
    parser.add_argument("--ranker-layers", type=int, default=3)
    parser.add_argument("--ranker-heads", type=int, default=4)
    parser.add_argument(
        "--show-line",
        action="store_true",
        help="Draw the end-effector path in the exported Meshcat scene.",
    )
    parser.add_argument(
        "--no-ghosts",
        action="store_true",
        help="Do not show translucent robots at the circle waypoints.",
    )
    return parser.parse_args()


def facet_dim_for_first_segment(args, regions, sequence, plant, device: torch.device) -> int:
    if args.planner == "convex":
        gcs = LinearGCS(regions.copy())
        gcs.addSourceTarget(sequence[0], sequence[1])
        graph = gcs.gcs
    else:
        polys = region_list(regions)
        vel_limits, accel_limits = iiwa_kinematic_limits(plant)
        _, graph, _, _ = build_nonlinear_gcs_problem(
            polys,
            build_region_edges(polys),
            sequence[0],
            sequence[1],
            vel_limits=vel_limits,
            accel_limits=accel_limits,
        )
    return eval_demo.build_graph_tensors(graph, regions, device=device).facet_dim


def main() -> None:
    args = parse_args()
    eval_demo.MAX_PATHS = int(args.max_paths)
    eval_demo.MAX_ROUNDING_TRIALS = int(args.max_trials)

    from manipulation.paths import manipulation_models_hint, manipulation_models_ready

    if not manipulation_models_ready():
        print(manipulation_models_hint(), file=sys.stderr)
        sys.exit(1)
    if not args.regions.exists():
        print(f"Missing IRIS regions: {args.regions}", file=sys.stderr)
        print("Run: python scripts/iiwa_shelf_scenes.py --generate-regions --regions-only", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_dir = f"checkpoints/manipulation_{args.planner}"
    flow_ckpt = args.flow_ckpt or f"{ckpt_dir}/manipulation_{args.planner}_flow_gnn.ckpt"
    ranknet_ckpt = args.ranknet_ckpt or f"{ckpt_dir}/manipulation_{args.planner}_ranknet.ckpt"

    print("Building IIWA shelf scene...")
    plant, _, _, _ = build_shelf_plant()
    print(f"Loading regions: {args.regions}")
    regions = load_regions(args.regions)

    demo_configs = planning_configurations(regions)
    sequence = build_demo_sequences(demo_configs, build_seed_points())["circle"]
    print(f"Circle sequence: {len(sequence)} waypoints, {len(sequence) - 1} segments")
    print(f"Planner family: {args.planner}")
    print(f"Device: {device}")

    encoder_hp = EncoderHParams(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_hidden_mult=args.ffn_hidden_mult,
        dropout_p=args.dropout_p,
    )
    decoder_hp = DecoderHParams(
        hidden_dims=tuple(int(s) for s in args.decoder_hidden.split(",") if s.strip()),
        dropout_p=args.decoder_dropout_p,
    )
    facet_dim = facet_dim_for_first_segment(args, regions, sequence, plant, device)
    flow_model = eval_demo.load_flow_model(
        flow_ckpt,
        facet_dim=facet_dim,
        g_dim=14,
        encoder_hp=encoder_hp,
        decoder_hp=decoder_hp,
        pointnet_hidden=args.pointnet_hidden,
        device=device,
    )
    ranker = eval_demo.load_ranknet(
        ranknet_ckpt,
        cfg=RankNetConfig(
            d_model=args.d_model,
            num_layers=args.ranker_layers,
            num_heads=args.ranker_heads,
        ),
        device=device,
    )
    print(f"Loaded flow: {flow_ckpt}")
    print(f"Loaded ranknet: {ranknet_ckpt}")

    print("\nPlanning: neural GCS (PointNet + RankNet)...")
    neural = eval_demo.plan_circle(
        planner=args.planner,
        mode="neural",
        regions=regions,
        sequence=sequence,
        plant=plant,
        seed=args.seed,
        speed=args.speed,
        flow_model=flow_model,
        ranker=ranker,
        device=device,
    )
    print(
        f"  success={neural.success} cost={neural.cost:.4f} "
        f"length={neural.path_length:.3f} elapsed={neural.elapsed_s:.2f}s"
    )
    if not neural.success:
        print("Neural GCS planning failed.", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"circle_neural_{args.planner}_seed{args.seed}"
    summary = {
        "planner": args.planner,
        "demo": "circle",
        "seed": args.seed,
        "device": str(device),
        "flow_ckpt": str(flow_ckpt),
        "ranknet_ckpt": str(ranknet_ckpt),
        "neural": eval_demo.plan_to_json(neural),
    }
    json_path = args.output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary -> {json_path}")

    print("Rendering neural GCS Meshcat HTML...")
    meshcat = StartMeshcat()
    visualize_trajectory(
        meshcat,
        neural.trajectory,
        show_line=args.show_line,
        ghost_configs=[] if args.no_ghosts else sequence,
        alpha=0.3,
        plan_wait=2.0,
    )
    html_path = args.output_dir / f"{stem}.html"
    html_path.write_text(eval_demo.patch_meshcat_html_play_once(meshcat.StaticHtml()))
    print(f"Saved HTML -> {html_path}")


if __name__ == "__main__":
    main()

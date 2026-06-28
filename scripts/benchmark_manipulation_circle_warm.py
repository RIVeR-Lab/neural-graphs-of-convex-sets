#!/usr/bin/env python3
"""Warm-process benchmark for the manipulation circle demo.

Unlike running ``eval_manipulation_circle_demo.py`` twice from the shell, this
script keeps one Python process alive: the IIWA scene, IRIS regions, checkpoints,
and CUDA context are initialized once, then the same vanilla/neural comparison is
repeated.

Example:
  python scripts/benchmark_manipulation_circle_warm.py --planner convex --device cuda --repeats 3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from manipulation.iiwa_helpers import build_shelf_plant
from manipulation.paths import DEFAULT_OUTPUT_DIR, DEFAULT_REGIONS_PATH
from manipulation.shelf_gcs import build_demo_sequences, build_seed_points, load_regions, planning_configurations
from manipulation.trajopt import build_nonlinear_gcs_problem, build_region_edges, iiwa_kinematic_limits, region_list
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
    parser = argparse.ArgumentParser(description="Warm-process manipulation circle benchmark.")
    parser.add_argument("--planner", choices=("convex", "nonlinear"), default="convex")
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "circle_demo")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--max-paths", type=int, default=eval_demo.MAX_PATHS)
    parser.add_argument("--max-trials", type=int, default=eval_demo.MAX_ROUNDING_TRIALS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=2)
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
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def aggregate_stage_timings(plan) -> dict[str, float]:
    totals: dict[str, float] = {}
    for segment in plan.segments:
        for name, value in segment.stage_timings.items():
            totals[name] = totals.get(name, 0.0) + float(value)
    return totals


def facet_dim_for_first_segment(args, regions, sequence, plant, device):
    if args.planner == "convex":
        tmp = LinearGCS(regions.copy())
        tmp.addSourceTarget(sequence[0], sequence[1])
        graph = tmp.gcs
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
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")

    ckpt_dir = f"checkpoints/manipulation_{args.planner}"
    flow_ckpt = args.flow_ckpt or f"{ckpt_dir}/manipulation_{args.planner}_flow_gnn.ckpt"
    ranknet_ckpt = args.ranknet_ckpt or f"{ckpt_dir}/manipulation_{args.planner}_ranknet.ckpt"

    print("Building IIWA shelf scene once...")
    plant, _, _, _ = build_shelf_plant()
    print(f"Loading regions once: {args.regions}")
    regions = load_regions(args.regions)
    demo_configs = planning_configurations(regions)
    sequence = build_demo_sequences(demo_configs, build_seed_points())["circle"]
    print(f"Circle sequence: {len(sequence)} waypoints, {len(sequence) - 1} segments")

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
    print(f"Loaded flow once: {flow_ckpt}")
    print(f"Loaded ranknet once: {ranknet_ckpt}")

    # One tiny CUDA op initializes the runtime before timing repeats.
    if device.type == "cuda":
        torch.empty((1,), device=device) + 1.0
        synchronize(device)

    runs = []
    for repeat in range(1, int(args.repeats) + 1):
        print(f"\n=== Repeat {repeat}/{args.repeats} ===")

        synchronize(device)
        t0 = time.perf_counter()
        print("Planning: vanilla GCS...")
        vanilla = eval_demo.plan_circle(
            planner=args.planner,
            mode="vanilla",
            regions=regions,
            sequence=sequence,
            plant=plant,
            seed=args.seed,
            speed=args.speed,
        )
        synchronize(device)
        vanilla_wall = time.perf_counter() - t0
        vanilla_stages = aggregate_stage_timings(vanilla)
        print(
            f"  vanilla success={vanilla.success} cost={vanilla.cost:.4f} "
            f"planner_elapsed={vanilla.elapsed_s:.2f}s wall={vanilla_wall:.2f}s"
        )
        if vanilla_stages:
            stage_msg = " ".join(f"{k}={v:.3f}s" for k, v in sorted(vanilla_stages.items()))
            print(f"  vanilla stages: {stage_msg}")

        synchronize(device)
        t0 = time.perf_counter()
        print("Planning: neural GCS (PointNet + RankNet, no CR)...")
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
        synchronize(device)
        neural_wall = time.perf_counter() - t0
        neural_stages = aggregate_stage_timings(neural)
        print(
            f"  neural success={neural.success} cost={neural.cost:.4f} "
            f"planner_elapsed={neural.elapsed_s:.2f}s wall={neural_wall:.2f}s"
        )
        if neural_stages:
            stage_msg = " ".join(f"{k}={v:.3f}s" for k, v in sorted(neural_stages.items()))
            print(f"  neural stages: {stage_msg}")

        runs.append(
            {
                "repeat": repeat,
                "vanilla": eval_demo.plan_to_json(vanilla) | {
                    "wall_s": vanilla_wall,
                    "stage_totals": vanilla_stages,
                },
                "neural": eval_demo.plan_to_json(neural) | {
                    "wall_s": neural_wall,
                    "stage_totals": neural_stages,
                },
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"circle_{args.planner}_warm_repeats{args.repeats}_seed{args.seed}.json"
    out.write_text(
        json.dumps(
            {
                "planner": args.planner,
                "device": str(device),
                "repeats": args.repeats,
                "max_paths": args.max_paths,
                "max_trials": args.max_trials,
                "flow_ckpt": str(flow_ckpt),
                "ranknet_ckpt": str(ranknet_ckpt),
                "runs": runs,
            },
            indent=2,
        )
    )
    print(f"\nWrote warm benchmark summary -> {out}")


if __name__ == "__main__":
    main()

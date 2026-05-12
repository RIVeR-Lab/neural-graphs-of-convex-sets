#!/usr/bin/env python3
"""Unified quadrotor motion visualization: pick scenario + planning methods, export Meshcat HTML.

Loads a scenario from HDF5 or scene seeds, plans with selected methods, and writes one HTML per
method (regions overlay, START/GOAL markers, trajectory trace, corner legend, 0.5× playback).

Methods (--methods, comma-separated):
  vanilla_convex     BezierGCS + rounding
  vanilla_nonconvex  GcsTrajectoryOptimization
  neural_gnn         PointNet flow GNN (convex or nonconvex per --planner)
  neural_ranknet     PointNet + RankNet (convex or nonconvex per --planner)

Examples:
  python quadrotor/scripts/visualize_quadrotor_motion.py \\
      --h5_path quadrotor/dataset/quadrotor_gcs_convex.h5 --instance-id 0 \\
      --methods neural_ranknet --planner convex

  python quadrotor/scripts/visualize_quadrotor_motion.py \\
      --grid-size 4 --building-seed 12 --query-seed 99 \\
      --methods vanilla_convex,neural_gnn --planner nonconvex
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pydrake.geometry.optimization import HPolyhedron
from pydrake.solvers import MosekSolver

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from quadrotor.scripts._bootstrap import REPO_ROOT, SCRIPTS_DIR  # noqa: F401

logging.getLogger("drake").setLevel(logging.WARNING)

from model.dataset import _decode_str
from model.facet_dataset import vertex_name_to_index
from model.hparams import DecoderHParams, EncoderHParams
from model.inference import project_flows_qp
from model.ranknet import RankNetConfig
from quadrotor.building_generation import MODELS_DIR, with_repaired_exterior_regions
from quadrotor.checkpoint_paths import (
    add_planner_checkpoint_args,
    resolve_flow_ckpt,
    resolve_ranknet_ckpt,
)
from quadrotor.gcs.data_generation import (
    build_adjacency,
    generate_mostly_indoor_scene,
    sample_start_goal_indoor,
    solve_relaxation,
    solve_restriction,
)
from quadrotor.gcs.trajopt import (
    adjacency_to_edges,
    build_nonlinear_gcs_problem,
    solve_nonlinear_relaxation,
    solve_nonlinear_restriction_trajectory,
)
from quadrotor.helpers import build_bezier_gcs
import quadrotor.motion_visualization as motion_viz

ALL_METHODS = (
    "vanilla_convex",
    "vanilla_nonconvex",
    "neural_gnn",
    "neural_ranknet",
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "quadrotor/results/motion"


def _dataset_missing_message(h5_path: Path) -> str:
    return (
        f"Dataset not found: {h5_path}\n"
        "See README.md (Quadrotor Motion Planning) for how to obtain training data:\n"
        "  - Download: bash quadrotor/scripts/quadrotor_download.sh\n"
        "  - Generate: bash quadrotor/scripts/quadrotor_collect.sh"
    )


def parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(m.strip() for m in value.split(",") if m.strip())
    unknown = sorted(set(methods) - set(ALL_METHODS))
    if unknown:
        raise ValueError(f"Unknown method(s): {', '.join(unknown)}. Choose from {ALL_METHODS}.")
    if not methods:
        raise ValueError("Provide at least one method.")
    return methods


def list_instance_ids(h5_path: Path, split: str | None) -> list[int]:
    import h5py

    with h5py.File(h5_path, "r") as h5:
        ids: list[int] = []
        for key in h5["samples"].keys():
            grp = h5["samples"][key]
            sp = _decode_str(grp.attrs.get("split", ""))
            if split is None or sp == split:
                ids.append(int(key))
    return sorted(ids)


def load_h5_instance(h5_path: Path, instance_id: int) -> dict:
    import h5py

    with h5py.File(h5_path, "r") as h5:
        grp = h5["samples"][str(instance_id)]
        g = np.asarray(grp["g"][()], dtype=np.float64)
        regions = None
        if "regions" in grp:
            regions = [
                HPolyhedron(
                    grp["regions"][key]["A"][()],
                    grp["regions"][key]["b"][()],
                )
                for key in sorted(grp["regions"].keys(), key=lambda x: int(x))
            ]
        grid_size = int(grp.attrs["grid_size"])
        regions = with_repaired_exterior_regions(regions, (grid_size, grid_size))
        out = {
            "instance_id": instance_id,
            "split": _decode_str(grp.attrs.get("split", "")),
            "grid_size": grid_size,
            "building_seed": int(grp.attrs["building_seed"]),
            "query_seed": int(grp.attrs.get("query_seed", 0)),
            "start_pose": g[:3],
            "goal_pose": g[3:],
            "regions": regions,
        }
        return out


def resolve_scene(args) -> dict:
    if args.h5_path:
        h5_path = Path(args.h5_path)
        if not h5_path.is_file():
            raise SystemExit(_dataset_missing_message(h5_path))
        ids = list_instance_ids(h5_path, args.split)
        if not ids:
            raise SystemExit(f"No instances in {h5_path}" + (f" split={args.split}" if args.split else ""))
        instance_id = args.instance_id if args.instance_id is not None else ids[0]
        if instance_id not in ids:
            raise SystemExit(f"Instance {instance_id} not in dataset.")
        inst = load_h5_instance(h5_path, instance_id)
        inst["h5_path"] = h5_path
        return inst

    if args.grid_size is None or args.building_seed is None:
        raise SystemExit("Provide --h5_path + --instance-id, or --grid-size + --building-seed.")
    if args.query_seed is None:
        raise SystemExit("--query-seed is required when not using --h5_path.")

    return {
        "instance_id": None,
        "split": None,
        "grid_size": int(args.grid_size),
        "building_seed": int(args.building_seed),
        "query_seed": int(args.query_seed),
        "start_pose": None,
        "goal_pose": None,
        "h5_path": None,
    }


def build_scene(scene: dict) -> tuple[list, dict, Path, np.ndarray, np.ndarray]:
    grid_size = scene["grid_size"]
    building_seed = scene["building_seed"]
    sdf_dir = MODELS_DIR / "room_gen"
    sdf_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = sdf_dir / f"motion_{grid_size}x{grid_size}_b{building_seed}.sdf"

    generated_regions, generated_adj, _, _ = generate_mostly_indoor_scene(
        grid_size, building_seed, sdf_path,
    )
    regions = scene.get("regions") or generated_regions
    adj = build_adjacency(regions) if scene.get("regions") is not None else generated_adj

    if scene["start_pose"] is not None:
        start_pose = np.asarray(scene["start_pose"], dtype=np.float64)
        goal_pose = np.asarray(scene["goal_pose"], dtype=np.float64)
    else:
        rng = np.random.default_rng(scene["query_seed"])
        sampled = sample_start_goal_indoor(regions, adj, rng)
        if sampled is None:
            raise SystemExit("Failed to sample start/goal for scene seeds.")
        start_pose, goal_pose, _, _ = sampled

    return regions, adj, sdf_path, start_pose, goal_pose


def plan_vanilla_convex(regions, start_pose, goal_pose, solver) -> motion_viz.PlanResult:
    t0 = time.perf_counter()
    gcs = build_bezier_gcs(regions, solver)
    try:
        gcs.addSourceTarget(start_pose, goal_pose, zero_deriv_boundary=3)
    except ValueError:
        return motion_viz.PlanResult(False, None, float("nan"), float("nan"), 0, time.perf_counter() - t0)

    traj, results = gcs.SolvePath(rounding=True, verbose=False, preprocessing=True)
    path = results.get("best_path") if results else None
    region_indices = (
        motion_viz.path_edges_to_region_indices(path, len(regions)) if path else None
    )
    cost = float(results.get("rounded_cost", float("nan"))) if results else float("nan")
    duration, _ = motion_viz.trajectory_stats(traj)
    return motion_viz.PlanResult(
        success=traj is not None,
        trajectory=traj,
        cost=cost,
        duration_s=duration,
        paths_tried=1,
        elapsed_s=time.perf_counter() - t0,
        region_indices=region_indices,
    )


def plan_vanilla_nonlinear(regions, adj, start_pose, goal_pose) -> motion_viz.PlanResult:
    from quadrotor.gcs.trajopt import plan_nonlinear_gcs

    t0 = time.perf_counter()
    traj, result = plan_nonlinear_gcs(
        regions, adjacency_to_edges(adj), start_pose, goal_pose,
    )
    duration, _ = motion_viz.trajectory_stats(traj)
    return motion_viz.PlanResult(
        success=traj is not None and result.is_success(),
        trajectory=traj,
        cost=float(result.get_optimal_cost()) if traj is not None and result.is_success() else float("nan"),
        duration_s=duration,
        paths_tried=1,
        elapsed_s=time.perf_counter() - t0,
        region_indices=None,
    )


def _neural_round_convex(
    gcs_obj,
    regions,
    graph_tensors,
    flow_out,
    phi_proj,
    *,
    ranker,
    device,
) -> motion_viz.PlanResult:
    t0 = time.perf_counter()
    relaxed = solve_relaxation(gcs_obj)
    if not relaxed.is_success():
        return motion_viz.PlanResult(False, None, float("nan"), float("nan"), 0, time.perf_counter() - t0)

    candidates = motion_viz.sample_rounded_paths(
        gcs_obj.gcs, relaxed, gcs_obj.source, gcs_obj.target,
    )
    if not candidates:
        return motion_viz.PlanResult(False, None, float("nan"), float("nan"), 0, time.perf_counter() - t0)

    n_regions = len(regions)
    name_to_idx = {
        name: vertex_name_to_index(name, n_regions, None)
        for name in graph_tensors["vertex_names"]
    }
    node_arr, edge_arr, path_mask = motion_viz.build_path_tensors(
        candidates, graph_tensors["edge_lookup"], name_to_idx,
    )
    if node_arr is None:
        return motion_viz.PlanResult(False, None, float("nan"), float("nan"), 0, time.perf_counter() - t0)

    if ranker is not None:
        with torch.no_grad():
            scores = ranker(
                node_embeddings=flow_out.node_embeddings,
                edge_flows=phi_proj.to(device),
                path_node_indices=node_arr.to(device),
                path_edge_indices=edge_arr.to(device),
                path_mask=path_mask.to(device),
            )
        order = torch.argsort(scores, descending=True).cpu().tolist()
    else:
        order = list(range(len(candidates)))

    tried = 0
    best_traj = None
    best_cost = float("inf")
    best_path = None
    for ri in order:
        path_edges = candidates[ri]
        if path_edges is None:
            continue
        tried += 1
        res = solve_restriction(gcs_obj, path_edges)
        if not res.is_success():
            continue
        cost = float(res.get_optimal_cost())
        traj = motion_viz.extract_trajectory(gcs_obj, path_edges, res)
        region_indices = motion_viz.path_edges_to_region_indices(path_edges, n_regions)
        if not motion_viz.trajectory_stays_in_path_regions(traj, regions, region_indices):
            continue
        if ranker is not None:
            duration, _ = motion_viz.trajectory_stats(traj)
            return motion_viz.PlanResult(
                success=True,
                trajectory=traj,
                cost=cost,
                duration_s=duration,
                paths_tried=tried,
                elapsed_s=time.perf_counter() - t0,
                region_indices=region_indices,
            )
        if cost < best_cost:
            best_cost = cost
            best_traj = traj
            best_path = path_edges

    if best_traj is None:
        return motion_viz.PlanResult(False, None, float("nan"), float("nan"), tried, time.perf_counter() - t0)

    duration, _ = motion_viz.trajectory_stats(best_traj)
    return motion_viz.PlanResult(
        success=True,
        trajectory=best_traj,
        cost=best_cost,
        duration_s=duration,
        paths_tried=tried,
        elapsed_s=time.perf_counter() - t0,
        region_indices=motion_viz.path_edges_to_region_indices(best_path, n_regions),
    )


def plan_neural_convex(
    regions, start_pose, goal_pose, flow_model, ranker, solver, device,
) -> motion_viz.PlanResult:
    gcs_obj = build_bezier_gcs(regions, solver)
    try:
        gcs_obj.addSourceTarget(start_pose, goal_pose, zero_deriv_boundary=3)
    except ValueError:
        return motion_viz.PlanResult(False, None, float("nan"), float("nan"), 0, 0.0)

    graph_tensors = motion_viz.build_graph_tensors(gcs_obj.gcs, regions, device)
    g_t = torch.from_numpy(np.concatenate([start_pose, goal_pose]).astype(np.float32)).to(device)
    with torch.no_grad():
        flow_out = flow_model(
            facets=graph_tensors["facets"],
            facet_mask=graph_tensors["facet_mask"],
            node_flags=graph_tensors["node_flags"],
            edge_index=graph_tensors["edge_index"],
            g=g_t,
            batch=None,
        )
        phi_hat = torch.sigmoid(flow_out.edge_logits).detach().cpu()
    phi_proj = project_flows_qp(
        edge_index=graph_tensors["edge_index"].cpu(),
        phi_hat=phi_hat,
        num_nodes=int(graph_tensors["n_nodes"]),
        source_idx=int(graph_tensors["source_idx"]),
        target_idx=int(graph_tensors["target_idx"]),
    )
    return _neural_round_convex(
        gcs_obj, regions, graph_tensors, flow_out, phi_proj,
        ranker=ranker, device=device,
    )


def plan_neural_nonlinear(
    regions, adj, start_pose, goal_pose, flow_model, ranker, device,
) -> motion_viz.PlanResult:
    gcs, graph, source, target = build_nonlinear_gcs_problem(
        regions, adjacency_to_edges(adj), start_pose, goal_pose,
    )
    relaxed = solve_nonlinear_relaxation(graph, source, target)
    if not relaxed.is_success():
        return motion_viz.PlanResult(False, None, float("nan"), float("nan"), 0, 0.0)

    graph_tensors = motion_viz.build_graph_tensors(graph, regions, device)
    return motion_viz.plan_neural_nonlinear(
        graph_tensors=graph_tensors,
        planning_graph=graph,
        regions=regions,
        relaxed=relaxed,
        source=source,
        target=target,
        gcs=gcs,
        flow_model=flow_model,
        ranker=ranker,
        start_pose=start_pose,
        goal_pose=goal_pose,
        device=device,
    )


def plan_neural_gnn_only_nonlinear(
    regions, adj, start_pose, goal_pose, flow_model, device,
) -> motion_viz.PlanResult:
    """Nonlinear neural GNN without RankNet: best-cost rounded path."""
    gcs, graph, source, target = build_nonlinear_gcs_problem(
        regions, adjacency_to_edges(adj), start_pose, goal_pose,
    )
    relaxed = solve_nonlinear_relaxation(graph, source, target)
    if not relaxed.is_success():
        return motion_viz.PlanResult(False, None, float("nan"), float("nan"), 0, 0.0)

    graph_tensors = motion_viz.build_graph_tensors(graph, regions, device)
    g_t = torch.from_numpy(np.concatenate([start_pose, goal_pose]).astype(np.float32)).to(device)
    with torch.no_grad():
        flow_out = flow_model(
            facets=graph_tensors["facets"],
            facet_mask=graph_tensors["facet_mask"],
            node_flags=graph_tensors["node_flags"],
            edge_index=graph_tensors["edge_index"],
            g=g_t,
            batch=None,
        )
        phi_hat = torch.sigmoid(flow_out.edge_logits).detach().cpu()
    phi_proj = project_flows_qp(
        edge_index=graph_tensors["edge_index"].cpu(),
        phi_hat=phi_hat,
        num_nodes=int(graph_tensors["n_nodes"]),
        source_idx=int(graph_tensors["source_idx"]),
        target_idx=int(graph_tensors["target_idx"]),
    )

    candidates = motion_viz.sample_rounded_paths(graph, relaxed, source, target)
    if not candidates:
        return motion_viz.PlanResult(False, None, float("nan"), float("nan"), 0, 0.0)

    tried = 0
    best_traj = None
    best_cost = float("inf")
    best_path = None
    for path_edges in candidates:
        if path_edges is None:
            continue
        tried += 1
        try:
            traj, res = solve_nonlinear_restriction_trajectory(gcs, path_edges)
        except RuntimeError:
            continue
        if res.is_success() and traj is not None:
            region_indices = motion_viz.path_edges_to_region_indices(path_edges, len(regions))
            if not motion_viz.trajectory_stays_in_path_regions(traj, regions, region_indices):
                continue
            cost = float(res.get_optimal_cost())
            if cost < best_cost:
                best_cost = cost
                best_traj = traj
                best_path = path_edges

    if best_traj is None:
        return motion_viz.PlanResult(False, None, float("nan"), float("nan"), tried, 0.0)

    duration, _ = motion_viz.trajectory_stats(best_traj)
    return motion_viz.PlanResult(
        success=True,
        trajectory=best_traj,
        cost=best_cost,
        duration_s=duration,
        paths_tried=tried,
        elapsed_s=0.0,
        region_indices=motion_viz.path_edges_to_region_indices(best_path, len(regions)),
    )


def load_models(args, facet_dim: int, device: torch.device):
    encoder_hp = EncoderHParams(
        d_model=args.d_model, num_layers=args.num_layers, num_heads=args.num_heads,
        ffn_hidden_mult=args.ffn_hidden_mult, dropout_p=args.dropout_p,
    )
    hidden = tuple(int(s) for s in args.decoder_hidden.split(",") if s.strip())
    decoder_hp = DecoderHParams(hidden_dims=hidden, dropout_p=args.decoder_dropout_p)

    flow_model = motion_viz.load_flow_model(
        args.flow_ckpt,
        facet_dim=facet_dim,
        g_dim=6,
        encoder_hp=encoder_hp,
        decoder_hp=decoder_hp,
        pointnet_hidden=args.pointnet_hidden,
        device=device,
    )

    ranker = None
    if args.ranknet_ckpt and Path(args.ranknet_ckpt).is_file():
        ranker = motion_viz.load_ranknet(
            args.ranknet_ckpt,
            cfg=RankNetConfig(
                d_model=args.d_model,
                num_layers=args.ranker_layers,
                num_heads=args.ranker_heads,
                ffn_hidden_dim=args.ranker_ffn_hidden,
                score_hidden_dim=args.ranker_score_hidden,
                dropout_p=args.ranker_dropout_p,
            ),
            device=device,
        )
    return flow_model, ranker


def run_method(
    method: str,
    *,
    planner: str,
    regions,
    adj,
    start_pose,
    goal_pose,
    solver,
    flow_model,
    ranker,
    device,
) -> motion_viz.PlanResult:
    if method == "vanilla_convex":
        return plan_vanilla_convex(regions, start_pose, goal_pose, solver)
    if method == "vanilla_nonconvex":
        return plan_vanilla_nonlinear(regions, adj, start_pose, goal_pose)
    if method == "neural_gnn":
        if planner == "convex":
            return plan_neural_convex(regions, start_pose, goal_pose, flow_model, None, solver, device)
        return plan_neural_gnn_only_nonlinear(regions, adj, start_pose, goal_pose, flow_model, device)
    if method == "neural_ranknet":
        if ranker is None:
            raise SystemExit("neural_ranknet requires a RankNet checkpoint (--ranknet_ckpt).")
        if planner == "convex":
            return plan_neural_convex(regions, start_pose, goal_pose, flow_model, ranker, solver, device)
        return plan_neural_nonlinear(regions, adj, start_pose, goal_pose, flow_model, ranker, device)
    raise ValueError(method)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified quadrotor motion visualization.")
    parser.add_argument(
        "--methods",
        type=str,
        default="neural_ranknet",
        help=f"Comma-separated methods: {', '.join(ALL_METHODS)}",
    )
    add_planner_checkpoint_args(parser, default_planner="convex")
    parser.add_argument("--h5_path", type=str, default=None)
    parser.add_argument("--instance-id", type=int, default=None)
    parser.add_argument("--split", type=str, default=None, choices=("train", "val", "test"))
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--building-seed", type=int, default=None)
    parser.add_argument("--query-seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ffn_hidden_mult", type=int, default=2)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--decoder_hidden", default="256,256")
    parser.add_argument("--decoder_dropout_p", type=float, default=0.1)
    parser.add_argument("--pointnet_hidden", type=int, default=64)
    parser.add_argument("--ranker_layers", type=int, default=3)
    parser.add_argument("--ranker_heads", type=int, default=4)
    parser.add_argument("--ranker_ffn_hidden", type=int, default=256)
    parser.add_argument("--ranker_score_hidden", type=int, default=64)
    parser.add_argument("--ranker_dropout_p", type=float, default=0.1)
    parser.add_argument("--playback-speed", type=float, default=0.5)
    parser.add_argument("--intro-hold-s", type=float, default=2.0)
    parser.add_argument(
        "--legend",
        dest="legend",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    needs_neural = any(m.startswith("neural_") for m in methods)

    scene = resolve_scene(args)
    regions, adj, sdf_path, start_pose, goal_pose = build_scene(scene)

    print("=== Quadrotor motion visualization ===")
    if scene.get("h5_path"):
        print(f"  HDF5          : {scene['h5_path']}")
        print(f"  instance      : {scene['instance_id']} ({scene['split']})")
    print(f"  grid          : {scene['grid_size']}×{scene['grid_size']}")
    print(f"  building_seed : {scene['building_seed']}")
    print(f"  query_seed    : {scene['query_seed']}")
    print(f"  planner       : {args.planner}")
    print(f"  methods       : {', '.join(methods)}")
    print(f"  start         : {start_pose.round(3)}")
    print(f"  goal          : {goal_pose.round(3)}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    solver = MosekSolver()

    flow_model = ranker = None
    if needs_neural:
        args.flow_ckpt = resolve_flow_ckpt(args)
        if "neural_ranknet" in methods:
            args.ranknet_ckpt = resolve_ranknet_ckpt(args)
        gcs_probe = build_bezier_gcs(regions, solver)
        gcs_probe.addSourceTarget(start_pose, goal_pose, zero_deriv_boundary=3)
        facet_dim = motion_viz.build_graph_tensors(gcs_probe.gcs, regions, device)["facet_dim"]
        flow_model, ranker = load_models(args, facet_dim, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag_parts = []
    if scene.get("instance_id") is not None:
        tag_parts.append(f"id{scene['instance_id']}")
    else:
        tag_parts.append(f"b{scene['building_seed']}_q{scene['query_seed']}")

    any_ok = False
    for method in methods:
        print(f"\n--- {method} ---")
        result = run_method(
            method,
            planner=args.planner,
            regions=regions,
            adj=adj,
            start_pose=start_pose,
            goal_pose=goal_pose,
            solver=solver,
            flow_model=flow_model,
            ranker=ranker,
            device=device,
        )
        print(
            f"  success={result.success}  cost={result.cost:.4f}  "
            f"duration={result.duration_s:.2f}s  paths_tried={result.paths_tried}"
        )
        if not result.success or result.trajectory is None:
            print(f"  skipped (planning failed)")
            continue

        any_ok = True
        out_name = f"motion_{'_'.join(tag_parts)}_{args.planner}_{method}.html"
        out_path = args.output_dir / out_name

        motion_viz.render_motion_html(
            sdf_path=sdf_path,
            traj=result.trajectory,
            regions=regions,
            region_indices=result.region_indices,
            start_pose=start_pose,
            goal_pose=goal_pose,
            out_path=out_path,
            grid_size=scene["grid_size"],
            intro_hold_s=args.intro_hold_s,
            playback_speed=args.playback_speed,
            show_legend=args.legend,
        )

    if not any_ok:
        print("All selected methods failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

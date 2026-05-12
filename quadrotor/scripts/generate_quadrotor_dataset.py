#!/usr/bin/env python3
"""Generate quadrotor GCS datasets.

The default release layout writes two HDF5 files:

  quadrotor/dataset/quadrotor_gcs_convex.h5
  quadrotor/dataset/quadrotor_gcs_nonconvex.h5

Default split targets, per file:
  train: 500 x 4x4 scenes
  val:    50 x 3x3 scenes + 50 x 5x5 scenes
  test:   50 x 3x3 scenes + 50 x 5x5 scenes

Each instance stores graph solutions (phi_star, candidate_paths), metadata,
and per-region halfspaces (samples/<id>/regions/<i>/A, b) for PointNet training.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from quadrotor.scripts._bootstrap import REPO_ROOT, SCRIPTS_DIR  # noqa: F401

logging.getLogger("drake").setLevel(logging.WARNING)

from pydrake.solvers import MosekSolver

from quadrotor.building_generation import MODELS_DIR
from quadrotor.gcs.data_generation import (
    MAX_CANDIDATE_PATHS,
    MAX_ROUNDING_TRIALS,
    generate_mostly_indoor_scene,
    node_features,
    sample_start_goal_indoor,
    solve_relaxation,
    solve_restriction,
    stable_seed,
)
from quadrotor.gcs.rounding import randomForwardPathSearch
from quadrotor.helpers import build_bezier_gcs

CONVEX_FILENAME = "quadrotor_gcs_convex.h5"
NONCONVEX_FILENAME = "quadrotor_gcs_nonconvex.h5"
PLANNERS = ("convex", "nonconvex")


def write_region_halfspaces(reg_grp, regions) -> None:
    """Per-region convex halfspaces for PointNet training (index i -> node v{i})."""
    for i, region in enumerate(regions):
        rg = reg_grp.create_group(f"{i:04d}")
        rg.create_dataset("A", data=np.asarray(region.A(), dtype=np.float32))
        rg.create_dataset("b", data=np.asarray(region.b(), dtype=np.float32))


def write_instance(
    h5_samples,
    str_dtype,
    instance_id,
    split,
    grid_size,
    building_seed,
    query_seed,
    start_pose,
    goal_pose,
    regions,
    source_node_idx,
    target_node_idx,
    edge_u_names,
    edge_v_names,
    phi_star,
    candidate_records,
):
    gname = str(instance_id)
    if gname in h5_samples:
        return False

    feasible_sorted = sorted(
        [(i, r["cost"]) for i, r in enumerate(candidate_records) if r["feasible"]],
        key=lambda x: x[1],
    )
    rank_map = {idx: rank for rank, (idx, _) in enumerate(feasible_sorted)}

    grp = h5_samples.create_group(gname)
    grp.attrs["instance_id"] = instance_id
    grp.attrs["split"] = split
    grp.attrs["grid_size"] = grid_size
    grp.attrs["building_seed"] = building_seed
    grp.attrs["query_seed"] = query_seed

    g_vec = np.concatenate([start_pose, goal_pose]).astype(np.float32)
    grp.create_dataset("g", data=g_vec)
    grp.create_dataset("node_features", data=node_features(regions, source_node_idx, target_node_idx))
    grp.create_dataset("edge_u", data=np.array(edge_u_names, dtype=object), dtype=str_dtype)
    grp.create_dataset("edge_v", data=np.array(edge_v_names, dtype=object), dtype=str_dtype)
    grp.create_dataset("phi_star", data=phi_star)

    write_region_halfspaces(grp.create_group("regions"), regions)

    cands_grp = grp.create_group("candidate_paths")
    for ci, rec in enumerate(candidate_records):
        cg = cands_grp.create_group(f"{ci:03d}")
        cg.attrs["feasible"] = rec["feasible"]
        cg.attrs["rank"] = int(rank_map.get(ci, -1))
        cg.attrs["rounded_cost"] = rec["cost"]
        cg.attrs["solve_time"] = rec["solve_time"]
        cg.create_dataset("edge_u", data=np.array(rec["edge_u"], dtype=object), dtype=str_dtype)
        cg.create_dataset("edge_v", data=np.array(rec["edge_v"], dtype=object), dtype=str_dtype)

    return True


def collect_convex(
    regions,
    start_pose,
    goal_pose,
    solver,
    str_dtype,
    h5_samples,
    instance_id,
    split,
    grid_size,
    building_seed,
    query_seed,
):
    gcs_obj = build_bezier_gcs(regions, solver)
    try:
        gcs_obj.addSourceTarget(start_pose, goal_pose, zero_deriv_boundary=3)
    except ValueError:
        return False

    all_vertices = gcs_obj.gcs.Vertices()
    vertex_names = [v.name() for v in all_vertices]
    source_node_idx = vertex_names.index("source")
    target_node_idx = vertex_names.index("target")

    relaxed_result = solve_relaxation(gcs_obj)
    if not relaxed_result.is_success():
        return False

    all_edges = list(gcs_obj.gcs.Edges())
    edge_u_names = [e.u().name() for e in all_edges]
    edge_v_names = [e.v().name() for e in all_edges]
    phi_star = np.array(
        [float(relaxed_result.GetSolution(e.phi())) for e in all_edges],
        dtype=np.float32,
    )

    candidate_edge_lists = randomForwardPathSearch(
        gcs_obj.gcs, relaxed_result, gcs_obj.source, gcs_obj.target,
        max_paths=MAX_CANDIDATE_PATHS, max_trials=MAX_ROUNDING_TRIALS, seed=0,
    )
    if not candidate_edge_lists:
        return False

    candidate_records = []
    for path_edges in candidate_edge_lists:
        if path_edges is None:
            continue
        t0 = time.time()
        res = solve_restriction(gcs_obj, path_edges)
        elapsed = time.time() - t0
        feasible = res.is_success()
        cost = float(res.get_optimal_cost()) if feasible else float("nan")
        candidate_records.append({
            "feasible": feasible,
            "cost": cost,
            "edge_u": [e.u().name() for e in path_edges],
            "edge_v": [e.v().name() for e in path_edges],
            "solve_time": elapsed,
        })

    if not any(r["feasible"] for r in candidate_records):
        return False

    return write_instance(
        h5_samples, str_dtype, instance_id, split, grid_size,
        building_seed, query_seed, start_pose, goal_pose, regions,
        source_node_idx, target_node_idx, edge_u_names, edge_v_names,
        phi_star, candidate_records,
    )


def collect_nonlinear(
    regions,
    adj,
    start_pose,
    goal_pose,
    str_dtype,
    h5_samples,
    instance_id,
    split,
    grid_size,
    building_seed,
    query_seed,
):
    from quadrotor.gcs.trajopt import (
        adjacency_to_edges,
        build_nonlinear_gcs_problem,
        solve_nonlinear_relaxation,
        solve_nonlinear_restriction,
    )

    _, graph, source_vertex, target_vertex = build_nonlinear_gcs_problem(
        regions, adjacency_to_edges(adj), start_pose, goal_pose,
    )

    all_vertices = list(graph.Vertices())
    vertex_names = [v.name() for v in all_vertices]
    source_node_idx = vertex_names.index(source_vertex.name())
    target_node_idx = vertex_names.index(target_vertex.name())

    relaxed_result = solve_nonlinear_relaxation(graph, source_vertex, target_vertex)
    if not relaxed_result.is_success():
        return False

    all_edges = list(graph.Edges())
    edge_u_names = [e.u().name() for e in all_edges]
    edge_v_names = [e.v().name() for e in all_edges]
    phi_star = np.array(
        [float(relaxed_result.GetSolution(e.phi())) for e in all_edges],
        dtype=np.float32,
    )

    candidate_edge_lists = randomForwardPathSearch(
        graph, relaxed_result, source_vertex, target_vertex,
        max_paths=MAX_CANDIDATE_PATHS, max_trials=MAX_ROUNDING_TRIALS, seed=0,
    )
    if not candidate_edge_lists:
        return False

    candidate_records = []
    for path_edges in candidate_edge_lists:
        if path_edges is None:
            continue
        t0 = time.time()
        res = solve_nonlinear_restriction(graph, source_vertex, target_vertex, path_edges)
        elapsed = time.time() - t0
        feasible = res.is_success()
        cost = float(res.get_optimal_cost()) if feasible else float("nan")
        candidate_records.append({
            "feasible": feasible,
            "cost": cost,
            "edge_u": [e.u().name() for e in path_edges],
            "edge_v": [e.v().name() for e in path_edges],
            "solve_time": elapsed,
        })

    if not any(r["feasible"] for r in candidate_records):
        return False

    return write_instance(
        h5_samples, str_dtype, instance_id, split, grid_size,
        building_seed, query_seed, start_pose, goal_pose, regions,
        source_node_idx, target_node_idx, edge_u_names, edge_v_names,
        phi_star, candidate_records,
    )


def build_split_specs(args) -> list[tuple[str, int, int]]:
    specs: list[tuple[str, int, int]] = []
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    for split in splits:
        if split == "train":
            if args.train_count > 0:
                specs.append((split, args.train_size, args.train_count))
        elif split == "val":
            if args.val_3_count > 0:
                specs.append((split, 3, args.val_3_count))
            if args.val_5_count > 0:
                specs.append((split, 5, args.val_5_count))
        elif split == "test":
            if args.test_3_count > 0:
                specs.append((split, 3, args.test_3_count))
            if args.test_5_count > 0:
                specs.append((split, 5, args.test_5_count))
        else:
            raise ValueError(f"Unknown split: {split}")
    return specs


def parse_planners(value: str) -> tuple[str, ...]:
    if value == "both":
        return PLANNERS
    planners = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(planners) - set(PLANNERS))
    if unknown:
        raise ValueError(f"Unknown planner(s): {', '.join(unknown)}")
    return planners


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate quadrotor datasets for convex and nonconvex GCS.",
    )
    parser.add_argument("--output_dir", type=str, default="quadrotor/dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--planners",
        type=str,
        default="both",
        help="Dataset(s) to generate: convex, nonconvex, convex,nonconvex, or both.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits to generate.",
    )
    parser.add_argument("--train-size", type=int, default=4)
    parser.add_argument("--train-count", type=int, default=500)
    parser.add_argument("--val-3-count", type=int, default=50)
    parser.add_argument("--val-5-count", type=int, default=50)
    parser.add_argument("--test-3-count", type=int, default=50)
    parser.add_argument("--test-5-count", type=int, default=50)
    parser.add_argument(
        "--max-building-attempts",
        type=int,
        default=5,
        help="Max start/goal query attempts per generated scene.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output HDF5 files for selected planners.",
    )
    return parser.parse_args()


def main() -> None:
    import h5py

    args = parse_args()
    selected_planners = parse_planners(args.planners)
    split_specs = build_split_specs(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "convex": out_dir / CONVEX_FILENAME,
        "nonconvex": out_dir / NONCONVEX_FILENAME,
    }
    if args.overwrite:
        for planner in selected_planners:
            paths[planner].unlink(missing_ok=True)

    solver = MosekSolver()
    sdf_dir = MODELS_DIR / "room_gen"
    sdf_dir.mkdir(parents=True, exist_ok=True)

    files = {planner: h5py.File(paths[planner], "a") for planner in selected_planners}
    try:
        str_dtype = h5py.string_dtype(encoding="utf-8")
        samples = {}
        ids = {}
        for planner, h5 in files.items():
            meta = h5.require_group("meta").attrs
            meta["base_seed"] = int(args.seed)
            meta["model"] = planner
            meta["train_size"] = int(args.train_size)
            meta["val_sizes"] = "3,5"
            meta["test_sizes"] = "3,5"
            meta["has_region_halfspaces"] = True
            samples[planner] = h5.require_group("samples")
            existing = [int(k) for k in samples[planner].keys()]
            ids[planner] = max(existing) + 1 if existing else 0

        for split, grid_size, target_count in split_specs:
            label = f"{split} {grid_size}x{grid_size}"
            print(f"\n=== {label.upper()} — target {target_count} per selected planner ===")

            saved = {planner: 0 for planner in selected_planners}
            building_idx = 0
            attempts_without_progress = 0
            stats = {"query_fail": 0, "convex_fail": 0, "nonconvex_fail": 0}
            pbar = tqdm(total=target_count, desc=label)

            while any(saved[planner] < target_count for planner in selected_planners):
                building_seed = stable_seed(args.seed, split, grid_size, building_idx)
                sdf_path = sdf_dir / f"building_gen_{grid_size}x{grid_size}_{building_seed}.sdf"

                try:
                    regions, adj, _, _ = generate_mostly_indoor_scene(grid_size, building_seed, sdf_path)
                except Exception as exc:
                    print(f"  Scene generation failed (seed={building_seed}): {exc}")
                    building_idx += 1
                    continue

                saved_this_building = False
                for q_attempt in range(args.max_building_attempts):
                    if all(saved[planner] >= target_count for planner in selected_planners):
                        break

                    query_seed = stable_seed(args.seed, split, grid_size, building_idx, q_attempt)
                    rng = np.random.default_rng(query_seed)
                    sampled = sample_start_goal_indoor(regions, adj, rng)
                    if sampled is None:
                        stats["query_fail"] += 1
                        continue

                    start_pose, goal_pose, _, _ = sampled

                    if "convex" in selected_planners and saved["convex"] < target_count:
                        ok = collect_convex(
                            regions, start_pose, goal_pose, solver, str_dtype,
                            samples["convex"], ids["convex"], split, grid_size,
                            building_seed, query_seed,
                        )
                        if ok:
                            ids["convex"] += 1
                            saved["convex"] += 1
                            saved_this_building = True
                        else:
                            stats["convex_fail"] += 1

                    if "nonconvex" in selected_planners and saved["nonconvex"] < target_count:
                        ok = collect_nonlinear(
                            regions, adj, start_pose, goal_pose, str_dtype,
                            samples["nonconvex"], ids["nonconvex"], split, grid_size,
                            building_seed, query_seed,
                        )
                        if ok:
                            ids["nonconvex"] += 1
                            saved["nonconvex"] += 1
                            saved_this_building = True
                        else:
                            stats["nonconvex_fail"] += 1

                    pbar.n = min(saved.values()) if saved else 0
                    pbar.refresh()

                if saved_this_building:
                    attempts_without_progress = 0
                else:
                    attempts_without_progress += 1

                building_idx += 1
                if attempts_without_progress > target_count * 50:
                    print(f"  Stopping {label}: too many consecutive failures.")
                    break

            pbar.close()
            for planner in selected_planners:
                print(f"  {planner}: {saved[planner]}/{target_count}")
            print(
                f"  Failures — query: {stats['query_fail']}, "
                f"convex: {stats['convex_fail']}, nonconvex: {stats['nonconvex_fail']}"
            )
    finally:
        for h5 in files.values():
            h5.close()

    for planner in selected_planners:
        print(f"{planner.capitalize()} dataset: {paths[planner]}")


if __name__ == "__main__":
    main()

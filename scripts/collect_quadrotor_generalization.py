"""
Collect GCS generalization data for quadrotor motion planning.

Mostly-indoor scenes at multiple grid sizes; convex (BezierGCS) and nonlinear
(GcsTrajectoryOptimization) labels are written to separate HDF5 files with
the same schema as collect_quadrotor_data.py.

Default split targets (700 instances per file):
  train: 500 x 4x4
  val:    50 x 3x3 + 50 x 5x5
  test:   50 x 3x3 + 50 x 5x5
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import logging
import numpy as np
from tqdm import tqdm

logging.getLogger("drake").setLevel(logging.WARNING)

_collect_spec = importlib.util.spec_from_file_location(
    "collect_quadrotor_data",
    Path(__file__).resolve().parent / "collect_quadrotor_data.py",
)
_collect = importlib.util.module_from_spec(_collect_spec)
assert _collect_spec.loader is not None
_collect_spec.loader.exec_module(_collect)

from quadrotor.building_generation import (
    DOORS_WINDOWS_INDOOR_OPTIONS,
    DOORS_WINDOWS_WALL_OPTIONS,
    MODELS_DIR,
    MOSTLY_INDOOR_GROW_PROBABILITY,
    MOSTLY_INDOOR_START_INDOOR,
    MOSTLY_INDOOR_TREE_PROBABILITY,
    compile_sdf,
    diagonal_start_goal,
    generate_grid_world,
)
from quadrotor.gcs.rounding import randomForwardPathSearch
from quadrotor.gcs.trajopt import (
    adjacency_to_edges,
    build_nonlinear_gcs_problem,
    solve_nonlinear_relaxation,
    solve_nonlinear_restriction,
)
from quadrotor.helpers import build_bezier_gcs
from pydrake.solvers import MosekSolver

MAX_CANDIDATE_PATHS = _collect.MAX_CANDIDATE_PATHS
MAX_ROUNDING_TRIALS = _collect.MAX_ROUNDING_TRIALS
D_MIN = _collect.D_MIN
build_adjacency = _collect.build_adjacency
bfs_distance = _collect.bfs_distance
box_bounds = _collect.box_bounds
node_features = _collect.node_features
region_volume = _collect.region_volume
sample_point_in_region = _collect.sample_point_in_region
solve_relaxation = _collect.solve_relaxation
solve_restriction = _collect.solve_restriction

NUM_EXTERIOR_REGIONS = 4
CONVEX_FILENAME = "quadrotor_gcs_convex.h5"
NONLINEAR_FILENAME = "quadrotor_gcs_nonlinear.h5"


def sample_start_goal_indoor(regions, adj, rng, max_attempts=200):
    """Volume-weighted start/goal sampling restricted to non-exterior regions."""
    eligible = set(range(NUM_EXTERIOR_REGIONS, len(regions)))
    if not eligible:
        return None

    bounds = [box_bounds(r) for r in regions]
    vols = np.array([region_volume(lb, ub) for lb, ub in bounds])
    vols = np.maximum(vols, 0)
    eligible_list = sorted(eligible)
    eligible_vols = vols[eligible_list]
    if eligible_vols.sum() == 0:
        return None
    eligible_probs = eligible_vols / eligible_vols.sum()

    for _ in range(max_attempts):
        pick = rng.choice(len(eligible_list), p=eligible_probs)
        s_idx = eligible_list[pick]
        s_pt = sample_point_in_region(regions[s_idx], rng)
        if s_pt is None:
            continue

        dist = bfs_distance(adj, s_idx)
        candidates = [
            i for i, d in dist.items()
            if i != s_idx and i in eligible and d >= D_MIN
        ]
        if not candidates:
            continue

        cand_vols = vols[candidates]
        if cand_vols.sum() == 0:
            continue
        cand_probs = cand_vols / cand_vols.sum()
        g_idx = candidates[rng.choice(len(candidates), p=cand_probs)]
        g_pt = sample_point_in_region(regions[g_idx], rng)
        if g_pt is None:
            continue

        return s_pt, g_pt, s_idx, g_idx

    return None


def generate_scene(grid_size: int, building_seed: int, sdf_path: Path):
    shape = (grid_size, grid_size)
    grid_start, grid_goal = diagonal_start_goal(shape)
    grid, indoor_edges, outdoor_edges = generate_grid_world(
        shape=shape,
        start=grid_start,
        goal=grid_goal,
        seed=building_seed,
        grow_probability=MOSTLY_INDOOR_GROW_PROBABILITY,
        start_indoor=MOSTLY_INDOOR_START_INDOOR,
    )
    regions = compile_sdf(
        str(sdf_path),
        grid,
        grid_start,
        grid_goal,
        indoor_edges,
        outdoor_edges,
        seed=building_seed,
        indoor_options=DOORS_WINDOWS_INDOOR_OPTIONS,
        wall_options=DOORS_WINDOWS_WALL_OPTIONS,
        tree_probability=MOSTLY_INDOOR_TREE_PROBABILITY,
    )
    adj = build_adjacency(regions)
    return regions, adj, grid_start, grid_goal


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
    import h5py

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
    splits = args.splits.split(",")
    for split in splits:
        split = split.strip()
        if split == "train":
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


def main():
    parser = argparse.ArgumentParser(
        description="Collect quadrotor GCS generalization dataset (convex + nonlinear).",
    )
    parser.add_argument("--output_dir", type=str, default="quadrotor/dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--splits", type=str, default="train,val,test",
        help="Comma-separated splits to generate.",
    )
    parser.add_argument("--train-size", type=int, default=4)
    parser.add_argument("--train-count", type=int, default=500)
    parser.add_argument("--val-3-count", type=int, default=50)
    parser.add_argument("--val-5-count", type=int, default=50)
    parser.add_argument("--test-3-count", type=int, default=50)
    parser.add_argument("--test-5-count", type=int, default=50)
    parser.add_argument(
        "--max-building-attempts", type=int, default=5,
        help="Max query attempts per building before moving on.",
    )
    args = parser.parse_args()

    import h5py

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    convex_path = out_dir / CONVEX_FILENAME
    nonlinear_path = out_dir / NONLINEAR_FILENAME
    split_specs = build_split_specs(args)
    solver = MosekSolver()
    sdf_dir = MODELS_DIR / "room_gen"
    sdf_dir.mkdir(parents=True, exist_ok=True)

    convex_id = 0
    nonlinear_id = 0
    base_seed = args.seed

    with h5py.File(convex_path, "a") as h5_convex, h5py.File(nonlinear_path, "a") as h5_nc:
        str_dtype = h5py.string_dtype(encoding="utf-8")
        h5_convex.require_group("meta").attrs["base_seed"] = base_seed
        h5_convex.require_group("meta").attrs["model"] = "convex"
        h5_nc.require_group("meta").attrs["base_seed"] = base_seed
        h5_nc.require_group("meta").attrs["model"] = "nonlinear"
        convex_samples = h5_convex.require_group("samples")
        nonlinear_samples = h5_nc.require_group("samples")

        for split, grid_size, target_count in split_specs:
            label = f"{split} {grid_size}x{grid_size}"
            print(f"\n=== {label.upper()} — target {target_count} per file ===")

            convex_saved = 0
            nonlinear_saved = 0
            building_idx = 0
            attempts_without_progress = 0
            stats = {"query_fail": 0, "convex_fail": 0, "nc_fail": 0}

            pbar = tqdm(total=target_count, desc=f"{label} (convex)")
            while convex_saved < target_count or nonlinear_saved < target_count:
                building_seed = base_seed + hash((split, grid_size, building_idx)) % (2**31)
                sdf_path = sdf_dir / f"building_gen_{grid_size}x{grid_size}_{building_seed}.sdf"

                try:
                    regions, adj, _, _ = generate_scene(grid_size, building_seed, sdf_path)
                except Exception as exc:
                    print(f"  Scene generation failed (seed={building_seed}): {exc}")
                    building_idx += 1
                    continue

                saved_this_building = False
                for q_attempt in range(args.max_building_attempts):
                    if convex_saved >= target_count and nonlinear_saved >= target_count:
                        break

                    query_seed = base_seed + hash((split, grid_size, building_idx, q_attempt)) % (2**31)
                    rng = np.random.default_rng(query_seed)
                    sampled = sample_start_goal_indoor(regions, adj, rng)
                    if sampled is None:
                        stats["query_fail"] += 1
                        continue

                    start_pose, goal_pose, _, _ = sampled

                    if convex_saved < target_count:
                        ok = collect_convex(
                            regions, start_pose, goal_pose, solver, str_dtype,
                            convex_samples, convex_id, split, grid_size,
                            building_seed, query_seed,
                        )
                        if ok:
                            convex_id += 1
                            convex_saved += 1
                            pbar.update(1)
                            saved_this_building = True
                        else:
                            stats["convex_fail"] += 1

                    if nonlinear_saved < target_count:
                        ok = collect_nonlinear(
                            regions, adj, start_pose, goal_pose, str_dtype,
                            nonlinear_samples, nonlinear_id, split, grid_size,
                            building_seed, query_seed,
                        )
                        if ok:
                            nonlinear_id += 1
                            nonlinear_saved += 1
                            saved_this_building = True
                        else:
                            stats["nc_fail"] += 1

                if saved_this_building:
                    attempts_without_progress = 0
                else:
                    attempts_without_progress += 1

                building_idx += 1
                if attempts_without_progress > target_count * 50:
                    print(f"  Stopping {label}: too many consecutive failures.")
                    break

            pbar.close()
            print(
                f"  Convex: {convex_saved}/{target_count}  |  "
                f"Nonlinear: {nonlinear_saved}/{target_count}"
            )
            print(
                f"  Failures — query: {stats['query_fail']}, "
                f"convex: {stats['convex_fail']}, nc: {stats['nc_fail']}"
            )

    print(f"\nConvex dataset:    {convex_path} ({convex_id} instances)")
    print(f"Nonlinear dataset: {nonlinear_path} ({nonlinear_id} instances)")


if __name__ == "__main__":
    main()

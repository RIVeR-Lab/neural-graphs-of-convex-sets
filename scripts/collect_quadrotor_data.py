"""
Collect GCS training data for the quadrotor motion planning problem.

For each building:
  - Generate a random 3x3 environment and decompose into convex regions
  - Sample N_QUERIES start/goal pairs (randomized, 3D)
  - Per query:
      1. Solve convex relaxation -> get per-edge flows phi_star
      2. Run randomized rounding -> get ~10 unique candidate paths
      3. Solve SOCP per candidate path -> record feasibility + cost
  - Save everything to HDF5

Dataset splits:
  - Train: 500 buildings
  - Val:   100 buildings
  - Test:  100 buildings (held out)

HDF5 schema (one group per instance = one building + one query):
  samples/<instance_id>/
    attrs: instance_id, building_id, query_id, split, building_seed, query_seed
    g            float32 (6,)     [p_start, p_goal]
    node_features float32 (N, 9)  [xmin,xmax,ymin,ymax,zmin,zmax, is_region,is_source,is_target]
    edge_u        str    (E,)     source node name per directed edge
    edge_v        str    (E,)     target node name per directed edge
    phi_star      float32 (E,)    CR flow per edge
    candidate_paths/<000>/
      attrs: feasible, rank, rounded_cost
      edge_u  str (L,)
      edge_v  str (L,)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import logging
import numpy as np
from tqdm import tqdm

logging.getLogger("drake").setLevel(logging.WARNING)

from quadrotor.building_generation import generate_grid_world, compile_sdf, MODELS_DIR
from quadrotor.gcs.rounding import randomForwardPathSearch
from quadrotor.helpers import build_bezier_gcs
from pydrake.solvers import MosekSolver
from pydrake.geometry.optimization import GraphOfConvexSetsOptions

# ---------- constants ----------
N_QUERIES = 1
DELTA = 0.3
D_MIN = 3
MAX_CANDIDATE_PATHS = 10
MAX_ROUNDING_TRIALS = 100
GRID_START = np.array([-1, -1])
GRID_GOAL = np.array([2, 1])
GRID_SHAPE = (3, 3)

SPLITS = {
    "train": 500,
    "val":   100,
    "test":  100,
}


# ---------- geometry helpers ----------

def box_bounds(region):
    A, b = region.A(), region.b()
    lb = np.full(3, -np.inf)
    ub = np.full(3, np.inf)
    for i in range(A.shape[0]):
        row = A[i]
        nz = np.nonzero(row)[0]
        if len(nz) == 1:
            dim = nz[0]
            if row[dim] > 0:
                ub[dim] = min(ub[dim], b[i] / row[dim])
            else:
                lb[dim] = max(lb[dim], b[i] / row[dim])
    return lb, ub


def region_volume(lb, ub):
    dims = np.where(np.isinf(ub - lb), 1.0, ub - lb)
    return float(np.prod(np.maximum(dims, 0)))


def sample_point_in_region(region, rng):
    lb, ub = box_bounds(region)
    lo, hi = lb + DELTA, ub - DELTA
    if np.any(lo >= hi):
        return None
    return rng.uniform(lo, hi)


def node_features(regions, source_idx, target_idx):
    """Build (N+2, 9) node feature matrix including source and target nodes."""
    N = len(regions)
    feats = np.zeros((N + 2, 9), dtype=np.float32)
    for i, r in enumerate(regions):
        lb, ub = box_bounds(r)
        feats[i, :6] = np.concatenate([lb, ub])
        feats[i, 6] = 1.0  # is_region
    # source node
    feats[source_idx, 6] = 0.0
    feats[source_idx, 7] = 1.0  # is_source
    # target node
    feats[target_idx, 6] = 0.0
    feats[target_idx, 8] = 1.0  # is_target
    return feats


def build_adjacency(regions):
    n = len(regions)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if regions[i].IntersectsWith(regions[j]):
                adj[i].append(j)
                adj[j].append(i)
    return adj


def bfs_distance(adj, src):
    dist = {src: 0}
    queue = [src]
    while queue:
        cur = queue.pop(0)
        for nb in adj[cur]:
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                queue.append(nb)
    return dist


def sample_start_goal(regions, adj, rng, max_attempts=200):
    bounds = [box_bounds(r) for r in regions]
    vols = np.array([region_volume(lb, ub) for lb, ub in bounds])
    vols = np.maximum(vols, 0)
    if vols.sum() == 0:
        return None
    probs = vols / vols.sum()

    for _ in range(max_attempts):
        s_idx = rng.choice(len(regions), p=probs)
        s_pt = sample_point_in_region(regions[s_idx], rng)
        if s_pt is None:
            continue

        dist = bfs_distance(adj, s_idx)
        candidates = [i for i, d in dist.items() if i != s_idx and d >= D_MIN]
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


# ---------- GCS helpers ----------

def solve_relaxation(gcs_obj):
    opts = GraphOfConvexSetsOptions()
    opts.convex_relaxation = True
    opts.max_rounded_paths = 0
    opts.preprocessing = False
    if gcs_obj.solver is not None:
        opts.solver = gcs_obj.solver
    if gcs_obj.options is not None:
        opts.solver_options = gcs_obj.options
    return gcs_obj.gcs.SolveShortestPath(gcs_obj.source, gcs_obj.target, opts)


def solve_restriction(gcs_obj, path_edges):
    """Fix phi to a specific path and solve the convex restriction."""
    for edge in gcs_obj.gcs.Edges():
        edge.AddPhiConstraint(edge in path_edges)
    opts = GraphOfConvexSetsOptions()
    opts.convex_relaxation = True
    opts.max_rounded_paths = 0
    opts.preprocessing = False
    if gcs_obj.solver is not None:
        opts.solver = gcs_obj.solver
    if gcs_obj.options is not None:
        opts.solver_options = gcs_obj.options
    result = gcs_obj.gcs.SolveShortestPath(gcs_obj.source, gcs_obj.target, opts)
    for edge in gcs_obj.gcs.Edges():
        edge.ClearPhiConstraints()
    return result


# ---------- main ----------

def collect_instance(
    regions, adj, building_seed, query_seed, solver, str_dtype, h5_samples, instance_id, split
):
    import h5py

    rng = np.random.default_rng(query_seed)
    sampled = sample_start_goal(regions, adj, rng)
    if sampled is None:
        return False

    start_pose, goal_pose, s_reg_idx, g_reg_idx = sampled

    # Build GCS and add source/target
    gcs_obj = build_bezier_gcs(regions, solver)
    try:
        gcs_obj.addSourceTarget(start_pose, goal_pose, zero_deriv_boundary=3)
    except ValueError:
        return False

    # Identify source/target vertex indices in node feature matrix
    all_vertices = gcs_obj.gcs.Vertices()
    vertex_names = [v.name() for v in all_vertices]
    source_node_idx = vertex_names.index("source")
    target_node_idx = vertex_names.index("target")

    # Solve convex relaxation
    relaxed_result = solve_relaxation(gcs_obj)
    if not relaxed_result.is_success():
        return False

    # Extract edge info
    all_edges = list(gcs_obj.gcs.Edges())
    edge_u_names = [e.u().name() for e in all_edges]
    edge_v_names = [e.v().name() for e in all_edges]
    phi_star = np.array(
        [float(relaxed_result.GetSolution(e.phi())) for e in all_edges],
        dtype=np.float32,
    )

    # Randomized rounding -> candidate paths
    candidate_edge_lists = randomForwardPathSearch(
        gcs_obj.gcs, relaxed_result, gcs_obj.source, gcs_obj.target,
        max_paths=MAX_CANDIDATE_PATHS, max_trials=MAX_ROUNDING_TRIALS, seed=0,
    )
    if not candidate_edge_lists:
        return False

    # Solve SOCP per candidate path
    candidate_records = []
    for path_edges in candidate_edge_lists:
        if path_edges is None:
            continue
        t0 = time.time()
        res = solve_restriction(gcs_obj, path_edges)
        elapsed = time.time() - t0
        feasible = res.is_success()
        cost = float(res.get_optimal_cost()) if feasible else float("nan")
        path_eu = [e.u().name() for e in path_edges]
        path_ev = [e.v().name() for e in path_edges]
        candidate_records.append({
            "feasible": feasible,
            "cost": cost,
            "edge_u": path_eu,
            "edge_v": path_ev,
            "solve_time": elapsed,
        })

    if not any(r["feasible"] for r in candidate_records):
        return False

    # Rank feasible paths by cost ascending; infeasible get rank -1
    feasible_sorted = sorted(
        [(i, r["cost"]) for i, r in enumerate(candidate_records) if r["feasible"]],
        key=lambda x: x[1],
    )
    rank_map = {idx: rank for rank, (idx, _) in enumerate(feasible_sorted)}

    # Node features: N regions + source + target
    x_feats = node_features(regions, source_node_idx, target_node_idx)

    # Write to HDF5
    gname = str(instance_id)
    if gname in h5_samples:
        return False
    grp = h5_samples.create_group(gname)
    grp.attrs["instance_id"] = instance_id
    grp.attrs["split"] = split
    grp.attrs["building_seed"] = building_seed
    grp.attrs["query_seed"] = query_seed

    g_vec = np.concatenate([start_pose, goal_pose]).astype(np.float32)
    grp.create_dataset("g", data=g_vec)
    grp.create_dataset("node_features", data=x_feats)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="quadrotor/dataset",
                        help="Directory to write the HDF5 dataset.")
    parser.add_argument("--n_queries", type=int, default=N_QUERIES,
                        help="Start/goal queries per building.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base RNG seed.")
    parser.add_argument("--splits", type=str, default=None,
                        help="Comma-separated list of splits to generate, e.g. train,val,test. Default: all.")
    args = parser.parse_args()

    import h5py

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "quadrotor_gcs_dataset.h5"

    splits_to_run = args.splits.split(",") if args.splits else list(SPLITS.keys())
    solver = MosekSolver()
    sdf_path = str(MODELS_DIR / "room_gen" / "building.sdf")

    instance_id = 0
    base_seed = args.seed

    with h5py.File(h5_path, "a") as h5:
        str_dtype = h5py.string_dtype(encoding="utf-8")
        h5.require_group("meta").attrs["base_seed"] = base_seed
        samples_grp = h5.require_group("samples")

        for split in splits_to_run:
            n_buildings = SPLITS[split]
            n_target = n_buildings * args.n_queries
            print(f"\n=== {split.upper()} — {n_buildings} buildings x {args.n_queries} queries (target: {n_target}) ===")

            split_saved = 0
            skipped_buildings = 0

            for b_idx in tqdm(range(n_buildings), desc=split):
                building_seed = base_seed + hash((split, b_idx)) % (2**31)

                # Generate building
                grid, indoor_edges, outdoor_edges = generate_grid_world(
                    shape=GRID_SHAPE, start=GRID_START, goal=GRID_GOAL, seed=building_seed)
                regions = compile_sdf(
                    sdf_path, grid, GRID_START, GRID_GOAL,
                    indoor_edges, outdoor_edges, seed=building_seed)

                # Build adjacency once per building
                adj = build_adjacency(regions)

                saved = 0
                q_attempt = 0
                while saved < args.n_queries and q_attempt < args.n_queries * 5:
                    query_seed = base_seed + hash((split, b_idx, q_attempt)) % (2**31)
                    ok = collect_instance(
                        regions, adj, building_seed, query_seed,
                        solver, str_dtype, samples_grp, instance_id, split,
                    )
                    if ok:
                        saved += 1
                        instance_id += 1
                    q_attempt += 1

                split_saved += saved
                if saved < args.n_queries:
                    skipped_buildings += 1

            print(f"  Collected {split_saved} / {n_target} instances "
                  f"({skipped_buildings} buildings yielded fewer than {args.n_queries} query)")

    print(f"\nDataset saved to {h5_path}")
    print(f"Total instances: {instance_id} / {sum(SPLITS[s] for s in splits_to_run)} target")


if __name__ == "__main__":
    main()

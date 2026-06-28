"""Helpers for IIWA shelf GCS dataset collection."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from pydrake.common import RandomGenerator
from pydrake.geometry.optimization import GraphOfConvexSetsOptions, HPolyhedron
from pydrake.solvers import MosekSolver

from quadrotor.gcs.linear import LinearGCS
from quadrotor.gcs.preprocessing import removeRedundancies
from quadrotor.gcs.rounding import randomForwardPathSearch

from manipulation.trajopt import (
    build_nonlinear_gcs_problem,
    build_region_edges,
    iiwa_kinematic_limits,
    region_list,
    solve_nonlinear_relaxation,
    solve_nonlinear_restriction,
)

NUM_POSITIONS = 7
D_MIN = 2
MAX_CANDIDATE_PATHS = 10
MAX_ROUNDING_TRIALS = 100
ROUNDING_SEED = 0

SPLITS = {"train": 500, "val": 100, "test": 100}

LINEAR_FILENAME = "iiwa_gcs_linear.h5"
NONLINEAR_FILENAME = "iiwa_gcs_nonlinear.h5"


def region_names(regions: dict) -> list[str]:
    return list(regions.keys())


def build_adjacency(regions: dict) -> dict[int, list[int]]:
    names = region_names(regions)
    polys = region_list(regions)
    n = len(polys)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if polys[i].IntersectsWith(polys[j]):
                adj[i].append(j)
                adj[j].append(i)
    return adj


def bfs_distance(adj: dict[int, list[int]], src: int) -> dict[int, int]:
    dist = {src: 0}
    queue = [src]
    while queue:
        cur = queue.pop(0)
        for nb in adj[cur]:
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                queue.append(nb)
    return dist


def _region_weights(regions: dict) -> np.ndarray:
    """Proxy volume weights via max-inscribed ellipsoid volume."""
    weights = []
    for poly in region_list(regions):
        try:
            ellipsoid = poly.MaximumVolumeInscribedEllipsoid()
            weights.append(max(float(ellipsoid.volume()), 1e-12))
        except Exception:
            weights.append(1.0)
    w = np.asarray(weights, dtype=float)
    w = np.maximum(w, 1e-12)
    return w / w.sum()


def sample_point_in_region(region: HPolyhedron, seed: int) -> np.ndarray | None:
    gen = RandomGenerator(seed)
    try:
        q = region.UniformSample(gen)
    except RuntimeError:
        return None
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.shape[0] != NUM_POSITIONS or not region.PointInSet(q):
        return None
    return q


def sample_start_goal(
    regions: dict,
    adj: dict[int, list[int]],
    rng: np.random.Generator,
    *,
    d_min: int = D_MIN,
    max_attempts: int = 200,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    weights = _region_weights(regions)
    polys = region_list(regions)

    for attempt in range(max_attempts):
        s_idx = int(rng.choice(len(polys), p=weights))
        s_pt = sample_point_in_region(polys[s_idx], seed=int(rng.integers(0, 2**31)))
        if s_pt is None:
            continue

        dist = bfs_distance(adj, s_idx)
        candidates = [i for i, d in dist.items() if i != s_idx and d >= d_min]
        if not candidates:
            continue

        cand_w = weights[candidates]
        cand_w = cand_w / cand_w.sum()
        g_idx = int(candidates[int(rng.choice(len(candidates), p=cand_w))])
        g_pt = sample_point_in_region(polys[g_idx], seed=int(rng.integers(0, 2**31)))
        if g_pt is None:
            continue

        return s_pt, g_pt, s_idx, g_idx

    return None


def write_region_meta(h5_file, regions: dict, *, str_dtype) -> None:
    import h5py

    meta = h5_file.require_group("meta")
    meta.attrs["scene"] = "iiwa_shelf"
    meta.attrs["num_positions"] = NUM_POSITIONS
    names = region_names(regions)
    meta.create_dataset("region_names", data=np.array(names, dtype=object), dtype=str_dtype)

    reg_grp = h5_file.require_group("regions")
    for name, poly in regions.items():
        g = reg_grp.require_group(name)
        if "A" in g and "b" in g:
            continue
        g.create_dataset("A", data=np.asarray(poly.A(), dtype=np.float32))
        g.create_dataset("b", data=np.asarray(poly.b(), dtype=np.float32))


def write_instance(
    h5_samples,
    str_dtype,
    instance_id: int,
    split: str,
    query_seed: int,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    source_node_idx: int,
    target_node_idx: int,
    edge_u_names: list[str],
    edge_v_names: list[str],
    phi_star: np.ndarray,
    candidate_records: list[dict[str, Any]],
) -> bool:
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
    grp.attrs["query_seed"] = query_seed
    grp.attrs["source_node_idx"] = source_node_idx
    grp.attrs["target_node_idx"] = target_node_idx

    g_vec = np.concatenate([q_start, q_goal]).astype(np.float32)
    grp.create_dataset("g", data=g_vec)
    grp.create_dataset("edge_u", data=np.array(edge_u_names, dtype=object), dtype=str_dtype)
    grp.create_dataset("edge_v", data=np.array(edge_v_names, dtype=object), dtype=str_dtype)
    grp.create_dataset("phi_star", data=phi_star.astype(np.float32))

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


def _linear_gcs_options(gcs_obj: LinearGCS) -> GraphOfConvexSetsOptions:
    opts = GraphOfConvexSetsOptions()
    opts.convex_relaxation = True
    opts.max_rounded_paths = 0
    opts.preprocessing = False
    if gcs_obj.solver is not None:
        opts.solver = gcs_obj.solver
    if gcs_obj.options is not None:
        opts.solver_options = gcs_obj.options
    return opts


def _solve_linear_restriction(gcs_obj: LinearGCS, path_edges) -> Any:
    for edge in gcs_obj.gcs.Edges():
        edge.AddPhiConstraint(edge in path_edges)
    result = gcs_obj.gcs.SolveShortestPath(
        gcs_obj.source, gcs_obj.target, _linear_gcs_options(gcs_obj),
    )
    for edge in gcs_obj.gcs.Edges():
        edge.ClearPhiConstraints()
    return result


def collect_linear_instance(
    regions: dict,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    *,
    solver,
    str_dtype,
    h5_samples,
    instance_id: int,
    split: str,
    query_seed: int,
) -> bool:
    gcs = LinearGCS(regions.copy())
    gcs.setSolver(solver)
    try:
        gcs.addSourceTarget(q_start, q_goal)
    except ValueError:
        return False

    removeRedundancies(gcs.gcs, gcs.source, gcs.target, verbose=False)
    relaxed_result = gcs.gcs.SolveShortestPath(
        gcs.source, gcs.target, _linear_gcs_options(gcs),
    )
    if not relaxed_result.is_success():
        return False

    all_vertices = gcs.gcs.Vertices()
    vertex_names = [v.name() for v in all_vertices]
    source_node_idx = vertex_names.index("source")
    target_node_idx = vertex_names.index("target")

    all_edges = list(gcs.gcs.Edges())
    edge_u_names = [e.u().name() for e in all_edges]
    edge_v_names = [e.v().name() for e in all_edges]
    phi_star = np.array(
        [float(relaxed_result.GetSolution(e.phi())) for e in all_edges],
        dtype=np.float32,
    )

    candidate_edge_lists = randomForwardPathSearch(
        gcs.gcs, relaxed_result, gcs.source, gcs.target,
        max_paths=MAX_CANDIDATE_PATHS,
        max_trials=MAX_ROUNDING_TRIALS,
        seed=ROUNDING_SEED,
    )
    if not candidate_edge_lists:
        return False

    candidate_records = []
    for path_edges in candidate_edge_lists:
        if path_edges is None:
            continue
        t0 = time.time()
        res = _solve_linear_restriction(gcs, path_edges)
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
        h5_samples, str_dtype, instance_id, split, query_seed,
        q_start, q_goal, source_node_idx, target_node_idx,
        edge_u_names, edge_v_names, phi_star, candidate_records,
    )


def collect_nonlinear_instance(
    regions: dict,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    *,
    plant,
    str_dtype,
    h5_samples,
    instance_id: int,
    split: str,
    query_seed: int,
) -> bool:
    polys = region_list(regions)
    edges = build_region_edges(polys)
    vel_limits, accel_limits = iiwa_kinematic_limits(plant)

    _, graph, source_vertex, target_vertex = build_nonlinear_gcs_problem(
        polys, edges, q_start, q_goal,
        vel_limits=vel_limits, accel_limits=accel_limits,
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
        max_paths=MAX_CANDIDATE_PATHS,
        max_trials=MAX_ROUNDING_TRIALS,
        seed=ROUNDING_SEED,
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
        h5_samples, str_dtype, instance_id, split, query_seed,
        q_start, q_goal, source_node_idx, target_node_idx,
        edge_u_names, edge_v_names, phi_star, candidate_records,
    )


def count_split(h5_samples, split: str) -> int:
    n = 0
    for key in h5_samples.keys():
        if h5_samples[key].attrs.get("split") == split:
            n += 1
    return n


def next_instance_id(h5_samples) -> int:
    if len(h5_samples) == 0:
        return 0
    return max(int(k) for k in h5_samples.keys()) + 1

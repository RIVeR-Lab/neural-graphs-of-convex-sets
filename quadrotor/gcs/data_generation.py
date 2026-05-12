"""Shared helpers for quadrotor GCS dataset generation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from pydrake.geometry.optimization import GraphOfConvexSetsOptions

from quadrotor.building_generation import (
    DOORS_WINDOWS_INDOOR_OPTIONS,
    DOORS_WINDOWS_WALL_OPTIONS,
    MOSTLY_INDOOR_GROW_PROBABILITY,
    MOSTLY_INDOOR_START_INDOOR,
    MOSTLY_INDOOR_TREE_PROBABILITY,
    compile_sdf,
    diagonal_start_goal,
    generate_grid_world,
)

DELTA = 0.3
D_MIN = 3
MAX_CANDIDATE_PATHS = 10
MAX_ROUNDING_TRIALS = 100
NUM_EXTERIOR_REGIONS = 4


def stable_seed(base_seed: int, *parts: Any) -> int:
    """Derive a deterministic 31-bit seed from structured parts."""
    payload = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


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
    n_regions = len(regions)
    feats = np.zeros((n_regions + 2, 9), dtype=np.float32)
    for i, region in enumerate(regions):
        lb, ub = box_bounds(region)
        feats[i, :6] = np.concatenate([lb, ub])
        feats[i, 6] = 1.0
    feats[source_idx, 6] = 0.0
    feats[source_idx, 7] = 1.0
    feats[target_idx, 6] = 0.0
    feats[target_idx, 8] = 1.0
    return feats


def build_adjacency(regions):
    n_regions = len(regions)
    adj = {i: [] for i in range(n_regions)}
    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            if regions[i].IntersectsWith(regions[j]):
                adj[i].append(j)
                adj[j].append(i)
    return adj


def bfs_distance(adj, src):
    dist = {src: 0}
    queue = [src]
    while queue:
        cur = queue.pop(0)
        for neighbor in adj[cur]:
            if neighbor not in dist:
                dist[neighbor] = dist[cur] + 1
                queue.append(neighbor)
    return dist


def sample_start_goal_indoor(regions, adj, rng, max_attempts=200):
    """Volume-weighted start/goal sampling restricted to non-exterior regions."""
    eligible = set(range(NUM_EXTERIOR_REGIONS, len(regions)))
    if not eligible:
        return None

    bounds = [box_bounds(region) for region in regions]
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


def generate_mostly_indoor_scene(grid_size: int, building_seed: int, sdf_path: Path):
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


def generate_mostly_indoor_regions(grid_size: int, building_seed: int, sdf_path: Path):
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
    return compile_sdf(
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

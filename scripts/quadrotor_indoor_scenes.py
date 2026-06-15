"""
Generate mostly-indoor quadrotor environments (3x3, 4x4, 5x5), plan with convex
BezierGCS and nonlinear GcsTrajectoryOptimization on the same start/goal, and
export Meshcat HTML visualizations.

Example:
  python scripts/quadrotor_indoor_scenes.py
  python scripts/quadrotor_indoor_scenes.py --sizes 3,5 --seed 0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logging.getLogger("drake").setLevel(logging.WARNING)

from pydrake.examples import QuadrotorGeometry
from pydrake.geometry import MeshcatVisualizer, Rgba, StartMeshcat
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.parsing import Parser
from pydrake.solvers import MosekSolver
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from underactuated.uav_environment import (
    NONLINEAR_GCS_OPTION,
    FlatQuadrotorTrajectorySource,
    _QuadrotorGeometry,
)

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
from quadrotor.helpers import FlatnessInverter, build_bezier_gcs
from quadrotor.gcs.trajopt import adjacency_to_edges, plan_nonlinear_gcs
from quadrotor.gcs.viz_utils import DelayedTrajectory, add_trajectory_traces

DEFAULT_SIZES = (3, 4, 5)
DEFAULT_OUTPUT_DIR = Path("quadrotor/results/indoor_viz")
DELTA = 0.3
D_MIN = 3
TRAJ_DELAY_S = 1.0
CONVEX_TRACE = Rgba(1.0, 0.85, 0.0, 1.0)   # yellow
NONLINEAR_TRACE = Rgba(0.2, 0.55, 1.0, 1.0)  # blue


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
    """Same procedure as collect_quadrotor_data.py: volume-weighted regions, D_MIN hops."""
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

        return s_pt, g_pt, s_idx, g_idx, dist[g_idx]

    return None


def build_and_plan(
    size: int,
    seed: int,
    query_seed: int,
    sdf_path: Path,
) -> dict | None:
    shape = (size, size)
    grid_start, grid_goal = diagonal_start_goal(shape)

    grid, indoor_edges, outdoor_edges = generate_grid_world(
        shape=shape,
        start=grid_start,
        goal=grid_goal,
        seed=seed,
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
        seed=seed,
        indoor_options=DOORS_WINDOWS_INDOOR_OPTIONS,
        wall_options=DOORS_WINDOWS_WALL_OPTIONS,
        tree_probability=MOSTLY_INDOOR_TREE_PROBABILITY,
    )

    adj = build_adjacency(regions)
    rng = np.random.default_rng(query_seed)
    sampled = sample_start_goal(regions, adj, rng)
    if sampled is None:
        return None
    start_pose, goal_pose, s_idx, g_idx, hop_dist = sampled

    solver = MosekSolver()
    gcs = build_bezier_gcs(regions, solver)
    gcs.addSourceTarget(start_pose, goal_pose, zero_deriv_boundary=3)
    convex_traj, convex_results = gcs.SolvePath(rounding=True, verbose=False, preprocessing=True)

    nonlinear_traj, nonlinear_result = plan_nonlinear_gcs(
        regions,
        adjacency_to_edges(adj),
        start_pose,
        goal_pose,
    )

    return {
        "start_pose": start_pose,
        "goal_pose": goal_pose,
        "s_idx": s_idx,
        "g_idx": g_idx,
        "hop_dist": hop_dist,
        "convex_traj": convex_traj,
        "convex_results": convex_results,
        "nonlinear_traj": nonlinear_traj,
        "nonlinear_result": nonlinear_result,
        "n_regions": len(regions),
    }


def export_meshcat_html(
    sdf_path: Path,
    convex_traj,
    nonlinear_traj,
    html_path: Path,
) -> None:
    meshcat = StartMeshcat()
    meshcat.SetProperty("/Grid", "visible", False)
    meshcat.SetProperty("/Axes", "visible", False)
    meshcat.SetProperty("/Lights/AmbientLight/<object>", "intensity", 0.8)
    meshcat.SetProperty("/Lights/PointLightNegativeX/<object>", "intensity", 0)
    meshcat.SetProperty("/Lights/PointLightPositiveX/<object>", "intensity", 0)

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

    parser_drake = Parser(plant, scene_graph)
    parser_drake.AddModels(str(sdf_path))
    plant.Finalize()

    meshcat_viz = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    animator = meshcat_viz.StartRecording()

    add_trajectory_traces(
        meshcat,
        [convex_traj] if convex_traj is not None else [],
        [CONVEX_TRACE],
        "/drake/traces/convex",
    )
    if nonlinear_traj is not None:
        add_trajectory_traces(
            meshcat,
            [nonlinear_traj],
            [NONLINEAR_TRACE],
            "/drake/traces/nonlinear",
        )

    total_duration = 0.0
    if convex_traj is not None:
        convex_system = builder.AddSystem(FlatnessInverter(convex_traj, animator))
        QuadrotorGeometry.AddToBuilder(
            builder, convex_system.get_output_port(0), scene_graph,
        )
        total_duration = max(total_duration, convex_traj.end_time())

    if nonlinear_traj is not None:
        nc_delay = total_duration + TRAJ_DELAY_S
        delayed_nc = DelayedTrajectory(nonlinear_traj, delay=nc_delay)
        nc_source = builder.AddSystem(FlatQuadrotorTrajectorySource(delayed_nc))
        _QuadrotorGeometry.AddToBuilder(
            builder, nc_source.get_output_port(0), scene_graph, "nonlinear_",
        )
        total_duration = max(total_duration, delayed_nc.end_time())

    diagram = builder.Build()

    meshcat.Delete()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(0.0)
    if total_duration > 0:
        simulator.AdvanceTo(total_duration + 0.05)
    meshcat_viz.PublishRecording()

    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(meshcat.StaticHtml())


def parse_sizes(raw: str) -> list[int]:
    sizes = [int(s.strip()) for s in raw.split(",") if s.strip()]
    if not sizes:
        raise ValueError("Expected at least one grid size, e.g. --sizes 3,4,5")
    for size in sizes:
        if size < 2:
            raise ValueError(f"Grid size must be >= 2, got {size}")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize mostly-indoor scenes with convex + nonlinear GCS.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for scene generation.")
    parser.add_argument(
        "--query-seed",
        type=int,
        default=None,
        help="RNG seed for start/goal sampling (default: --seed + 1000 * grid size).",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SIZES),
        help="Comma-separated square grid sizes (default: 3,4,5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for SDF + HTML outputs.",
    )
    args = parser.parse_args()

    sizes = parse_sizes(args.sizes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sdf_dir = MODELS_DIR / "room_gen"

    print(f"Generating {len(sizes)} mostly-indoor scene(s) (seed={args.seed})...")
    for size in sizes:
        label = f"{size}x{size}"
        sdf_path = sdf_dir / f"building_indoor_{label}_seed{args.seed}.sdf"
        html_path = args.output_dir / f"indoor_{label}_seed{args.seed}.html"

        query_seed = args.query_seed if args.query_seed is not None else args.seed + size * 1000

        print(f"\n=== {label} (query_seed={query_seed}) ===")
        planned = build_and_plan(size, args.seed, query_seed, sdf_path)
        if planned is None:
            print(f"  Could not sample valid start/goal for {label}. Try a different --query-seed.")
            continue

        start_pose = planned["start_pose"]
        goal_pose = planned["goal_pose"]
        s_idx = planned["s_idx"]
        g_idx = planned["g_idx"]
        hop_dist = planned["hop_dist"]
        convex_traj = planned["convex_traj"]
        convex_results = planned["convex_results"]
        nonlinear_traj = planned["nonlinear_traj"]
        nonlinear_result = planned["nonlinear_result"]
        n_regions = planned["n_regions"]

        if convex_traj is None and nonlinear_traj is None:
            print(f"  Both planners failed for {label}. Try a different --seed or --query-seed.")
            continue

        print(f"  {n_regions} convex regions → {sdf_path.name}")
        print(f"  Start region {s_idx} @ {start_pose.round(2)}, "
              f"goal region {g_idx} @ {goal_pose.round(2)} ({hop_dist} hops)")

        if convex_traj is not None:
            cost = convex_results.get("rounded_cost", float("nan"))
            print(f"  Convex GCS: cost={cost:.3f}, "
                  f"duration={convex_traj.start_time():.2f}s → {convex_traj.end_time():.2f}s")
        else:
            print("  Convex GCS: failed")

        if nonlinear_traj is not None:
            print(f"  Nonlinear GCS: success, "
                  f"duration={nonlinear_traj.start_time():.2f}s → {nonlinear_traj.end_time():.2f}s")
        else:
            msg = nonlinear_result.get_solution_result() if nonlinear_result else "unknown"
            print(f"  Nonlinear GCS: failed ({msg})")

        if convex_traj is None:
            print("  Skipping Meshcat export (convex trajectory required for building viz).")
            continue

        print("  Rendering Meshcat HTML (yellow=convex, blue=nonlinear)...")
        export_meshcat_html(sdf_path, convex_traj, nonlinear_traj, html_path)
        print(f"  Saved {html_path}")

    print(f"\nDone. Open HTML files in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

"""
Run vanilla GCS on both the nominal and chance-constraint-inflated free-space
layouts for a 3x3 building and render each trajectory in Meshcat.

Reuses build_regions from visualize_chance_constraint to get both region sets,
then calls the standard plan_through_building for each.

Usage:
    python scripts/gcs_chance_constraint.py --seed 0 --sigma 0.15 --delta 0.05
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import norm

from pydrake.geometry import MeshcatVisualizer, Rgba, Sphere, StartMeshcat
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.solvers import MosekSolver
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder

from quadrotor.building_generation import compile_sdf, generate_grid_world, MODELS_DIR
from quadrotor.helpers import FlatnessInverter, build_bezier_gcs, plan_through_building
from pydrake.examples import QuadrotorGeometry

# reuse the region builder and viz helpers from visualize_chance_constraint
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from visualize_chance_constraint import (
    build_regions, draw_region, draw_graph,
    _box_from_hpoly, NOMINAL_WALL_OFFSET,
    GRID_SHAPE, GRID_START, GRID_GOAL, SDF_PATH,
)

DELTA_MARGIN = 0.3   # margin from box boundaries when sampling start/goal


def sample_start_goal(regions, rng, margin=DELTA_MARGIN):
    """Sample start and goal in opposite-corner rooms with a margin from walls."""
    finite = []
    for r in regions:
        got = _box_from_hpoly(r)
        if got is None:
            continue
        c, s = got
        if np.any(np.isinf(c)) or np.any(s <= 2 * margin):
            continue
        finite.append((c, s))

    if len(finite) < 2:
        raise RuntimeError("Not enough finite regions to sample start/goal.")

    centers = np.array([c for c, s in finite])
    scores  = centers[:, 0] + centers[:, 1]   # bottom-left → top-right diagonal

    def sample_in(c, s):
        lo = c - s / 2 + margin
        hi = c + s / 2 - margin
        return rng.uniform(lo, hi)

    start = sample_in(*finite[int(np.argmin(scores))])
    goal  = sample_in(*finite[int(np.argmax(scores))])
    return start, goal


def render_html(traj, regions, start_pose, goal_pose, label, out_path,
                region_fill, region_edge):
    """Build a Drake diagram, simulate the trajectory, and export StaticHtml."""
    meshcat = StartMeshcat()
    meshcat.SetProperty("/Grid", "visible", True)
    meshcat.SetProperty("/Axes", "visible", True)
    meshcat.SetProperty("/Lights/AmbientLight/<object>", "intensity", 0.8)
    meshcat.SetProperty("/Lights/PointLightNegativeX/<object>", "intensity", 0)
    meshcat.SetProperty("/Lights/PointLightPositiveX/<object>", "intensity", 0)

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    Parser(plant, scene_graph).AddModels(SDF_PATH)
    plant.Finalize()

    meshcat_viz = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    animator    = meshcat_viz.StartRecording()
    traj_system = builder.AddSystem(FlatnessInverter(traj, animator))
    QuadrotorGeometry.AddToBuilder(builder, traj_system.get_output_port(0), scene_graph)
    diagram = builder.Build()

    meshcat.Delete()

    # Draw free-space regions
    for k, r in enumerate(regions):
        draw_region(meshcat, f"/regions/r{k}", r,
                    fill_rgba=region_fill, edge_rgba=region_edge, edge_width=1.5)

    # Draw GCS graph
    draw_graph(meshcat, regions, path="/graph",
               node_rgba=Rgba(1.0, 0.55, 0.0, 1.0),
               edge_rgba=Rgba(1.0, 0.95, 0.05, 1.0),
               node_r=0.20, edge_width=3.5)

    # Start / goal markers
    for name, pos, rgba in (
        ("start", start_pose, Rgba(0.1, 0.9, 0.2, 1.0)),
        ("goal",  goal_pose,  Rgba(0.9, 0.1, 0.1, 1.0)),
    ):
        meshcat.SetObject(f"/markers/{name}", Sphere(0.18), rgba)
        meshcat.SetTransform(f"/markers/{name}", RigidTransform(pos))

    # Trajectory trace
    n_samples = 300
    ts       = np.linspace(traj.start_time(), traj.end_time(), n_samples)
    vertices = np.array([np.asarray(traj.value(t)).reshape(-1) for t in ts]).T
    meshcat.SetLine("/trace", vertices, line_width=3.0,
                    rgba=Rgba(1.0, 0.85, 0.1, 1.0))

    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(0.0)
    simulator.AdvanceTo(traj.end_time() + 0.05)
    meshcat_viz.PublishRecording()

    out_path.write_text(meshcat.StaticHtml())
    print(f"  [{label}] saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",   type=int,   default=0)
    ap.add_argument("--sigma",  type=float, default=0.15)
    ap.add_argument("--delta",  type=float, default=0.05)
    ap.add_argument("--outdir", type=str,   default="quadrotor/results/viz")
    args = ap.parse_args()

    assert 0.0 < args.delta < 0.5

    z      = float(norm.ppf(1.0 - args.delta))
    margin = z * args.sigma
    wall_offset_nom  = NOMINAL_WALL_OFFSET
    wall_offset_infl = NOMINAL_WALL_OFFSET + margin

    print(f"sigma={args.sigma}  delta={args.delta}  margin={margin:.4f} m")
    print(f"wall_offset  nominal={wall_offset_nom:.4f}  "
          f"inflated={wall_offset_infl:.4f}")

    # Build grid + SDF (needed for the Drake building model in render_html)
    print(f"\nGenerating building (seed={args.seed})...")
    grid, indoor_edges, outdoor_edges = generate_grid_world(
        shape=GRID_SHAPE, start=GRID_START, goal=GRID_GOAL, seed=args.seed)
    compile_sdf(SDF_PATH, grid, GRID_START, GRID_GOAL,
                indoor_edges, outdoor_edges, seed=args.seed)

    # Build both region sets
    regions_nom  = build_regions(grid, GRID_START, wall_offset_nom,  seed=args.seed)
    regions_infl = build_regions(grid, GRID_START, wall_offset_infl, seed=args.seed)

    rng = np.random.default_rng(args.seed)
    start_pose, goal_pose = sample_start_goal(regions_nom, rng)
    print(f"Start: {start_pose.round(3)}  Goal: {goal_pose.round(3)}")

    solver  = MosekSolver()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, regions, fill, edge, tag in [
        ("Nominal",
         regions_nom,
         Rgba(0.0,  0.75, 0.95, 0.15),
         Rgba(0.0,  0.7,  0.9,  1.0),
         "nominal"),
        ("Inflated",
         regions_infl,
         Rgba(0.45, 0.95, 0.15, 0.15),
         Rgba(0.3,  0.9,  0.05, 1.0),
         "inflated"),
    ]:
        print(f"\n[{label}] Planning...")
        traj, results = plan_through_building(
            regions, start_pose, goal_pose, solver=solver)

        if traj is None:
            print(f"  [{label}] GCS failed — no trajectory found.")
            continue

        cost = results.get("rounded_cost", float("nan"))
        print(f"  cost={cost:.3f}  "
              f"solve={results['solve_time']:.2f}s  "
              f"setup={results['setup_time']:.2f}s")

        out_path = out_dir / f"gcs_{tag}_seed{args.seed}.html"
        render_html(traj, regions, start_pose, goal_pose,
                    label, out_path, fill, edge)


if __name__ == "__main__":
    main()
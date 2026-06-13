"""
GCS trajectory sweep over sensing uncertainty (sigma values).

For each sigma in SIGMAS:
  - Inflate wall_offset by Phi^{-1}(1-delta)*sigma
  - Build free-space regions
  - Solve GCS
  - Draw trajectory as a colored polyline

All trajectories + the nominal baseline are overlaid in one Meshcat HTML.
The building geometry (walls) is drawn once from the SDF.
Start/goal are fixed across all sigmas (sampled from nominal regions).

Usage:
    python scripts/gcs_sigma_sweep.py --seed 0 --delta 0.05
"""
from __future__ import annotations

import argparse
import sys
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
from pydrake.examples import QuadrotorGeometry

sys.path.insert(0, str(Path(__file__).parent))
from visualize_chance_constraint import (
    build_regions, draw_region, draw_graph,
    _box_from_hpoly, NOMINAL_WALL_OFFSET,
    GRID_SHAPE, GRID_START, GRID_GOAL, SDF_PATH,
)
from quadrotor.building_generation import compile_sdf, generate_grid_world
from quadrotor.helpers import FlatnessInverter, plan_through_building

DELTA       = 0.05
SIGMAS      = [0.0, 0.05, 0.15, 0.30]   # 0.0 = nominal baseline

# One distinct color per sigma — chosen to pop on a dark background.
TRAJ_COLORS = [
    Rgba(0.2,  0.8,  1.0,  1.0),   # cyan    — nominal (sigma=0)
    Rgba(0.4,  1.0,  0.2,  1.0),   # green   — sigma=0.05
    Rgba(1.0,  0.85, 0.05, 1.0),   # yellow  — sigma=0.15
    Rgba(1.0,  0.25, 0.25, 1.0),   # red     — sigma=0.30
]

SAMPLE_MARGIN = 0.35   # margin from box walls when sampling start/goal


def sample_start_goal(regions, rng, margin=SAMPLE_MARGIN):
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
        raise RuntimeError("Not enough finite regions.")
    centers = np.array([c for c, s in finite])
    scores  = centers[:, 0] + centers[:, 1]

    def sample_in(c, s):
        return rng.uniform(c - s / 2 + margin, c + s / 2 - margin)

    # Fix z: start near floor, goal near ceiling (mirrors high_to_low style)
    sc, ss = finite[int(np.argmin(scores))]
    gc, gs = finite[int(np.argmax(scores))]
    start = sample_in(sc, ss)
    goal  = sample_in(gc, gs)
    start[2] = sc[2] - ss[2] / 2 + margin + 0.1
    goal[2]  = gc[2] + gs[2] / 2 - margin - 0.1
    return start, goal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",   type=int,   default=0)
    ap.add_argument("--delta",  type=float, default=DELTA)
    ap.add_argument("--outdir", type=str,   default="quadrotor/results/viz")
    args = ap.parse_args()

    assert 0.0 < args.delta < 0.5

    print(f"delta={args.delta}  sigmas={SIGMAS}  seed={args.seed}")

    # Build SDF once (needed for the Drake building model)
    print(f"\nGenerating building (seed={args.seed})...")
    grid, indoor_edges, outdoor_edges = generate_grid_world(
        shape=GRID_SHAPE, start=GRID_START, goal=GRID_GOAL, seed=args.seed)
    compile_sdf(SDF_PATH, grid, GRID_START, GRID_GOAL,
                indoor_edges, outdoor_edges, seed=args.seed)

    # Sample start/goal from nominal regions (fixed for all sigmas)
    regions_nom = build_regions(grid, GRID_START, NOMINAL_WALL_OFFSET, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    start_pose, goal_pose = sample_start_goal(regions_nom, rng)
    print(f"Start: {start_pose.round(3)}  Goal: {goal_pose.round(3)}")

    solver = MosekSolver()

    # Solve GCS for each sigma and collect (traj, regions, color, label)
    results = []
    for sigma, color in zip(SIGMAS, TRAJ_COLORS):
        margin = float(norm.ppf(1.0 - args.delta)) * sigma if sigma > 0 else 0.0
        wall_offset = NOMINAL_WALL_OFFSET + margin
        label = f"sigma={sigma}" + (" (nominal)" if sigma == 0 else
                                     f"  margin={margin:.3f}m")
        print(f"\n[{label}]  wall_offset={wall_offset:.4f}")

        regions = build_regions(grid, GRID_START, wall_offset, seed=args.seed)
        traj, res = plan_through_building(regions, start_pose, goal_pose, solver=solver)
        if traj is None:
            print(f"  GCS failed.")
            continue
        cost = res.get("rounded_cost", float("nan"))
        print(f"  cost={cost:.3f}  solve={res['solve_time']:.2f}s")
        results.append((traj, regions, color, label, sigma, margin))

    if not results:
        print("All GCS solves failed.")
        return

    # --- Build single Meshcat HTML with all trajectories ---
    print("\nRendering combined HTML...")
    meshcat = StartMeshcat()
    meshcat.SetProperty("/Grid",  "visible", True)
    meshcat.SetProperty("/Axes",  "visible", True)
    meshcat.SetProperty("/Lights/AmbientLight/<object>", "intensity", 0.8)
    meshcat.SetProperty("/Lights/PointLightNegativeX/<object>", "intensity", 0)
    meshcat.SetProperty("/Lights/PointLightPositiveX/<object>", "intensity", 0)

    # Build Drake diagram with the building model (for walls/floors visuals)
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    Parser(plant, scene_graph).AddModels(SDF_PATH)
    plant.Finalize()
    meshcat_viz = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    ctx = diagram.CreateDefaultContext()
    meshcat.Delete()
    meshcat_viz.ForcedPublish(meshcat_viz.GetMyContextFromRoot(ctx))

    # Draw nominal free-space regions once (translucent grey outlines only)
    for k, r in enumerate(regions_nom):
        draw_region(meshcat, f"/regions/r{k}", r,
                    fill_rgba=Rgba(0.6, 0.6, 0.6, 0.08),
                    edge_rgba=Rgba(0.6, 0.6, 0.6, 0.5),
                    edge_width=1.0)

    # Start / goal markers
    for name, pos, rgba in (
        ("start", start_pose, Rgba(0.1, 0.9, 0.2, 1.0)),
        ("goal",  goal_pose,  Rgba(0.9, 0.1, 0.1, 1.0)),
    ):
        meshcat.SetObject(f"/markers/{name}", Sphere(0.22), rgba)
        meshcat.SetTransform(f"/markers/{name}", RigidTransform(pos))

    # Draw each trajectory as a dense polyline in its assigned color
    n_samples = 400
    for traj, regions, color, label, sigma, margin in results:
        ts       = np.linspace(traj.start_time(), traj.end_time(), n_samples)
        vertices = np.array([np.asarray(traj.value(t)).reshape(-1) for t in ts]).T
        tag = f"sigma{sigma:.2f}".replace(".", "p")
        meshcat.SetLine(f"/trajectories/{tag}", vertices,
                        line_width=4.0, rgba=color)

    out_dir  = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gcs_sigma_sweep_seed{args.seed}.html"
    out_path.write_text(meshcat.StaticHtml())
    print(f"\nSaved -> {out_path}")
    print("Trajectories:")
    for _, _, color, label, sigma, margin in results:
        print(f"  sigma={sigma:.2f}  margin={margin:.3f}m  — {label}")


if __name__ == "__main__":
    main()

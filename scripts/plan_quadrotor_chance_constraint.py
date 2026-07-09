#!/usr/bin/env python3
"""
Vanilla/neural GCS motion plans with obstacle-first chance constraints on a fully
indoor 3x3 building (all rooms, no trees, start inside, goal at opposite end).

Pipeline:
  1. Build full compile_sdf regions (doors on interior/perimeter walls)
  2. Extract and inflate 3D wall box obstacles
  3. Tighten compile_sdf regions by that margin (obstacle-facing faces only)
  4. Plan nominal vs inflated with convex or nonconvex vanilla/neural GCS
  5. Export one static Meshcat HTML (building + both traces + legend)

Usage:
    python scripts/plan_quadrotor_chance_constraint.py
    python scripts/plan_quadrotor_chance_constraint.py --planner nonconvex
    python scripts/plan_quadrotor_chance_constraint.py --planner nonconvex --method neural_ranknet
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from pydrake.geometry import Mesh, MeshcatVisualizer, Rgba, Sphere, StartMeshcat
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.solvers import MosekSolver
from pydrake.systems.framework import DiagramBuilder

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from quadrotor.chance_constraints import inflate_regions, write_legend_gltf
from quadrotor.checkpoint_paths import flow_ckpt_path, ranknet_ckpt_path
from quadrotor.gcs.data_generation import build_adjacency
from quadrotor.gcs.trajopt import adjacency_to_edges, plan_nonlinear_gcs
from quadrotor.helpers import build_bezier_gcs, plan_through_building
from quadrotor.motion_visualization import build_graph_tensors, load_flow_model, load_ranknet
from quadrotor.obstacle_inflation import (
    extract_wall_obstacles,
    inflate_box_faces,
    inflation_margin,
)
from quadrotor.scene_sampling import pick_corner_regions

QUAD_SCRIPTS = REPO_ROOT / "quadrotor" / "scripts"
if str(QUAD_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(QUAD_SCRIPTS))
import visualize_quadrotor_motion as quad_motion  # noqa: E402

from model.hparams import DecoderHParams, EncoderHParams
from model.ranknet import RankNetConfig

logging.getLogger("drake").setLevel(logging.WARNING)

GRID_SHAPE = (3, 3)
SDF_PATH = str(MODELS_DIR / "room_gen" / "building.sdf")

SEED = 0
SIGMA = 0.15
DELTA = 0.1
OUTDIR = Path("quadrotor/results/viz")
METHODS = ("vanilla", "neural_gnn", "neural_ranknet")

TRAJ_NOMINAL = Rgba(0.20, 0.80, 1.00, 1.0)
TRAJ_INFLATED = Rgba(0.95, 0.45, 0.10, 1.0)


def trace_vertices(traj, n_samples: int = 400) -> np.ndarray:
    ts = np.linspace(traj.start_time(), traj.end_time(), n_samples)
    return np.array([np.asarray(traj.value(t)).reshape(3) for t in ts]).T


def build_fully_indoor_scene(seed: int):
    grid_start, grid_goal = diagonal_start_goal(GRID_SHAPE)
    grid, indoor_edges, outdoor_edges = generate_grid_world(
        shape=GRID_SHAPE,
        start=grid_start,
        goal=grid_goal,
        seed=seed,
        grow_probability=MOSTLY_INDOOR_GROW_PROBABILITY,
        start_indoor=MOSTLY_INDOOR_START_INDOOR,
    )
    regions = compile_sdf(
        SDF_PATH,
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
    indoor_cells = int(np.sum(grid > 0.5))
    return grid, grid_start, indoor_edges, outdoor_edges, regions, indoor_cells


def render_comparison_html(
    *,
    traj_nom,
    traj_infl,
    start_pose,
    goal_pose,
    out_path: Path,
) -> None:
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
    diagram = builder.Build()
    ctx = diagram.CreateDefaultContext()
    meshcat.Delete()
    meshcat_viz.ForcedPublish(meshcat_viz.GetMyContextFromRoot(ctx))

    for name, pos, rgba in (
        ("start", start_pose, Rgba(0.1, 0.9, 0.2, 1.0)),
        ("goal", goal_pose, Rgba(0.9, 0.1, 0.1, 1.0)),
    ):
        meshcat.SetObject(f"/markers/{name}", Sphere(0.22), rgba)
        meshcat.SetTransform(f"/markers/{name}", RigidTransform(pos))

    meshcat.SetLine("/trajectories/nominal", trace_vertices(traj_nom), line_width=4.0, rgba=TRAJ_NOMINAL)
    meshcat.SetLine("/trajectories/inflated", trace_vertices(traj_infl), line_width=4.0, rgba=TRAJ_INFLATED)

    assets_dir = Path(tempfile.mkdtemp(prefix="cc_motion_legend_"))
    try:
        legend_entries = [
            ("Nominal GCS", TRAJ_NOMINAL),
            ("Chance constraint GCS", TRAJ_INFLATED),
        ]
        gltf_path = write_legend_gltf(legend_entries, assets_dir, quad_w=6.0, quad_h=2.5)
        legend_pos = start_pose + np.array([0.0, 0.0, 4.5])
        meshcat.SetObject("/legend", Mesh(str(gltf_path), 1.0))
        meshcat.SetTransform("/legend", RigidTransform(legend_pos))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(meshcat.StaticHtml())
        print(f"  [Combined] saved -> {out_path}")
    finally:
        shutil.rmtree(assets_dir, ignore_errors=True)


def load_neural_models(planner: str, regions, start_pose, goal_pose, *, solver, device):
    flow_ckpt = flow_ckpt_path(planner)
    ranknet_ckpt = ranknet_ckpt_path(planner)
    gcs_probe = build_bezier_gcs(regions, solver)
    gcs_probe.addSourceTarget(start_pose, goal_pose, zero_deriv_boundary=3)
    facet_dim = build_graph_tensors(gcs_probe.gcs, regions, device)["facet_dim"]

    encoder_hp = EncoderHParams(d_model=128, num_layers=4, num_heads=4, ffn_hidden_mult=2, dropout_p=0.1)
    decoder_hp = DecoderHParams(hidden_dims=(256, 256), dropout_p=0.1)
    flow_model = load_flow_model(
        flow_ckpt,
        facet_dim=facet_dim,
        g_dim=6,
        encoder_hp=encoder_hp,
        decoder_hp=decoder_hp,
        pointnet_hidden=64,
        device=device,
    )
    ranker = load_ranknet(
        ranknet_ckpt,
        cfg=RankNetConfig(d_model=128, num_layers=3, num_heads=4),
        device=device,
    )
    print(f"  Loaded flow: {flow_ckpt}")
    print(f"  Loaded ranknet: {ranknet_ckpt}")
    return flow_model, ranker


def plan_scene(
    regions,
    adj,
    start_pose,
    goal_pose,
    *,
    planner: str,
    method: str,
    solver,
    flow_model=None,
    ranker=None,
    device=None,
):
    if method != "vanilla":
        neural_method = method
        result = quad_motion.run_method(
            neural_method,
            planner=planner,
            regions=regions,
            adj=adj,
            start_pose=start_pose,
            goal_pose=goal_pose,
            solver=solver,
            flow_model=flow_model,
            ranker=ranker,
            device=device,
        )
        if not result.success:
            return None, float("nan"), result.elapsed_s
        return result.trajectory, result.cost, result.elapsed_s

    if planner == "convex":
        traj, results = plan_through_building(
            regions, start_pose, goal_pose, solver=solver,
        )
        cost = results.get("rounded_cost", float("nan"))
        solve_time = results.get("solve_time", float("nan"))
        return traj, cost, solve_time

    t0 = time.time()
    traj, result = plan_nonlinear_gcs(
        regions, adjacency_to_edges(adj), start_pose, goal_pose,
    )
    solve_time = time.time() - t0
    if traj is None:
        return None, float("nan"), solve_time
    return traj, float(result.get_optimal_cost()), solve_time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", choices=("convex", "nonconvex"), default="convex")
    ap.add_argument("--method", choices=METHODS, default="vanilla")
    args = ap.parse_args()

    assert 0.0 < DELTA < 0.5

    print(f"Generating fully indoor 3x3 building (seed={SEED})...")
    grid, grid_start, indoor_edges, outdoor_edges, regions_nom, indoor_cells = (
        build_fully_indoor_scene(SEED)
    )
    print(f"  {indoor_cells}/{grid.size} indoor cells  |  {len(regions_nom)} compile_sdf regions")

    obstacles = extract_wall_obstacles(
        grid, grid_start, indoor_edges, outdoor_edges,
        include_perimeter=False, include_cell_walls=False,
    )
    margin_m = inflation_margin(SIGMA, DELTA)
    print(f"Extracted {len(obstacles)} wall box obstacles")
    print(f"sigma={SIGMA}  delta={DELTA}  inflation margin={margin_m:.4f} m")
    if obstacles:
        _, b_nom = inflate_box_faces(obstacles[0], np.zeros((3, 3)), DELTA)
        _, b_infl = inflate_box_faces(obstacles[0], np.diag([SIGMA**2] * 3), DELTA)
        print(f"Example obstacle +x face shift: {b_infl[0] - b_nom[0]:.4f} m")

    regions_infl = inflate_regions(regions_nom, margin_m)
    print(f"  {len(regions_infl)} inflated regions")

    rng = np.random.default_rng(SEED)
    sampled = pick_corner_regions(regions_nom, rng, diagonal="bl_to_tr")
    if sampled is None:
        raise RuntimeError("Could not sample start/goal from indoor room regions.")
    start_pose, goal_pose = sampled
    print(f"Start: {start_pose.round(3)}  Goal: {goal_pose.round(3)}")

    solver = MosekSolver()
    adj_nom = build_adjacency(regions_nom)
    adj_infl = build_adjacency(regions_infl)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow_model = ranker = None
    if args.method != "vanilla":
        print(f"\nLoading neural models ({args.method}, {args.planner})...")
        flow_model, ranker = load_neural_models(
            args.planner, regions_nom, start_pose, goal_pose, solver=solver, device=device,
        )
        if args.method == "neural_gnn":
            ranker = None

    out_dir = OUTDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    plans = {}
    costs = {}

    for label, regions, adj, tag in (
        ("Nominal", regions_nom, adj_nom, "nominal"),
        ("Chance-constrained", regions_infl, adj_infl, "inflated"),
    ):
        print(f"\n[{label}] Planning {args.method} {args.planner} GCS...")
        try:
            traj, cost, solve_time = plan_scene(
                regions, adj, start_pose, goal_pose,
                planner=args.planner,
                method=args.method,
                solver=solver,
                flow_model=flow_model,
                ranker=ranker,
                device=device,
            )
        except RuntimeError as exc:
            print(f"  [{label}] planning failed: {exc}")
            continue
        if traj is None:
            print(f"  [{label}] planning failed.")
            continue
        costs[tag] = cost
        print(f"  GCS path cost={cost:.3f}  solve={solve_time:.2f}s")
        plans[tag] = traj

    if costs:
        print("\n--- Path costs ---")
        if "nominal" in costs:
            print(f"  nominal:              {costs['nominal']:.3f}")
        if "inflated" in costs:
            print(f"  chance-constrained:   {costs['inflated']:.3f}")

    if "nominal" in plans and "inflated" in plans:
        render_comparison_html(
            traj_nom=plans["nominal"],
            traj_infl=plans["inflated"],
            start_pose=start_pose,
            goal_pose=goal_pose,
            out_path=out_dir / f"cc_motion_indoor3x3_{args.planner}_{args.method}_seed{SEED}.html",
        )
        print(
            f"\nDone. Open {out_dir}/cc_motion_indoor3x3_{args.planner}_{args.method}_seed{SEED}.html "
            "(cyan=Nominal GCS, orange=Chance constraint GCS)"
        )
    else:
        print("\nDone, but combined viz skipped because one plan failed.")


if __name__ == "__main__":
    main()

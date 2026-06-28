#!/usr/bin/env python3
"""
Plan with neural nonlinear GCS and render quadrotor motion in Meshcat.

Same mostly-indoor 5×5 environment (building_seed=42, query_seed=17 by default).
Scene geometry/regions are cached next to the SDF (``*.scene.pkl``) after first build.
Path regions drawn only where the trajectory actually flies (clipped boxes).

Usage:
  python scripts/demo_quadrotor_neural_motion.py
  python scripts/demo_quadrotor_neural_motion.py --device cpu
  python scripts/demo_quadrotor_neural_motion.py --regenerate-scene  # force rebuild
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pydrake.examples import QuadrotorGeometry
from pydrake.geometry import Mesh, MeshcatVisualizer, Rgba, Sphere, StartMeshcat
from pydrake.geometry.optimization import HPolyhedron
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

logging.getLogger("drake").setLevel(logging.WARNING)

from collect_quadrotor_data import build_adjacency
from planning_through_contact.model.facet_dataset import normalize_facets, vertex_name_to_index
from planning_through_contact.model.facet_pointnet import PointNetFlowPredictor
from planning_through_contact.model.hparams import DecoderHParams, EncoderHParams
from planning_through_contact.model.inference import project_flows_qp
from planning_through_contact.model.ranknet import PathRankNet, RankNetConfig
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
    solve_nonlinear_restriction_trajectory,
)
from quadrotor.gcs.viz_utils import (
    DelayedTrajectory,
    add_quadrotor_pose_trail,
    apply_trail_opacities,
)
from quadrotor.helpers import FlatnessInverter

import visualize_quadrotor as viz
import visualize_chance_constraint as cc

MAX_PATHS = 10
MAX_ROUNDING_TRIALS = 100
SCENE_CACHE_VERSION = 1
TRAJ_REGION_SAMPLES = 400
TRAJ_BOX_MARGIN = 1.2
TRAJ_BOX_MIN_EXTENT = 0.9
INTRO_HOLD_S = 45.0
PLAYBACK_SPEED = 0.5
LEGEND_FONT_SCALE = 2.0


def patch_meshcat_html_legend_overlay(
    html: str,
    *,
    font_scale: float = LEGEND_FONT_SCALE,
) -> str:
    """Persistent corner legend (convex sets + trajectory line)."""
    font_px = 14.7 * font_scale
    swatch_px = 14 * font_scale
    line_w = 18 * font_scale
    line_h = 2.5 * font_scale
    gap_px = 8 * font_scale
    margin_px = 8 * font_scale
    top_px = 36 * font_scale
    left_px = 24 * font_scale
    border_px = 1.5 * font_scale
    overlay = f"""
<div id="ngcs-legend" style="position:fixed;top:{top_px}px;left:{left_px}px;right:auto;z-index:10000;pointer-events:none;color:#fff;font-family:'DejaVu Sans',Arial,sans-serif;font-weight:700;font-size:{font_px}px;line-height:1.35;text-shadow:0 1px 4px rgba(0,0,0,0.9);text-align:left;">
  <div style="display:flex;align-items:center;gap:{gap_px}px;">
    <span style="display:inline-block;width:{swatch_px}px;height:{swatch_px}px;background:rgba(255,217,51,0.55);border:{border_px}px solid rgba(255,235,51,1);"></span>
    <span>Chosen convex sets</span>
  </div>
  <div style="margin-top:{margin_px}px;display:flex;align-items:center;gap:{gap_px}px;">
    <span style="display:inline-block;width:{line_w}px;height:{line_h}px;background:#fff;"></span>
    <span>Trajectory</span>
  </div>
</div>
"""
    if "</body>" not in html:
        return html
    return html.replace("</body>", overlay + "\n</body>", 1)


@dataclass
class PlanResult:
    success: bool
    trajectory: object | None
    cost: float
    duration_s: float
    paths_tried: int
    elapsed_s: float
    region_indices: list[int] | None = None


def path_edges_to_region_indices(path_edges, n_regions: int) -> list[int]:
    """Region indices along the GCS path in visit order."""
    if not path_edges:
        return []

    ordered: list[int] = []

    def maybe_add(name: str) -> None:
        try:
            idx = vertex_name_to_index(name, n_regions, None)
        except KeyError:
            return
        if idx < n_regions:
            ordered.append(idx)

    maybe_add(path_edges[0].u().name())
    for edge in path_edges:
        maybe_add(edge.v().name())
    return ordered


def _trajectory_points_in_region(traj, region, n_samples: int = TRAJ_REGION_SAMPLES) -> np.ndarray:
    ts = np.linspace(traj.start_time(), traj.end_time(), n_samples)
    pts = []
    for t in ts:
        p = np.asarray(traj.value(t)).reshape(3)
        if region.PointInSet(p):
            pts.append(p)
    return np.array(pts) if pts else np.empty((0, 3))


def _region_bounds(region) -> tuple[np.ndarray, np.ndarray] | None:
    got = cc._box_from_hpoly(region)
    if got is None:
        return None
    center, size = got
    return center - size / 2.0, center + size / 2.0


def _display_box_for_region(region, traj) -> HPolyhedron | None:
    """Clip a set to the trajectory samples that actually lie inside it."""
    bounds = _region_bounds(region)
    if bounds is None:
        return None
    rlb, rub = bounds
    pts = _trajectory_points_in_region(traj, region)
    if len(pts) == 0:
        return None

    lb = pts.min(axis=0) - TRAJ_BOX_MARGIN
    ub = pts.max(axis=0) + TRAJ_BOX_MARGIN
    lb = np.maximum(lb, rlb)
    ub = np.minimum(ub, rub)
    for d in range(3):
        if ub[d] - lb[d] < TRAJ_BOX_MIN_EXTENT:
            mid = 0.5 * (lb[d] + ub[d])
            half = TRAJ_BOX_MIN_EXTENT / 2.0
            lb[d] = max(rlb[d], mid - half)
            ub[d] = min(rub[d], mid + half)
    if np.any(ub <= lb):
        return None
    return HPolyhedron.MakeBox(lb, ub)


def trajectory_region_indices(traj, regions, path_region_indices: list[int]) -> list[int]:
    """Path regions that the trajectory actually samples, in visit order."""
    traversed: list[int] = []
    seen: set[int] = set()
    for ri in path_region_indices:
        if ri in seen or not (0 <= ri < len(regions)):
            continue
        if len(_trajectory_points_in_region(traj, regions[ri])) > 0:
            traversed.append(ri)
            seen.add(ri)
    return traversed


def draw_path_regions(meshcat, regions, region_indices: list[int], traj) -> int:
    """Draw only regions the trajectory passes through (clipped to path usage)."""
    fill = Rgba(1.0, 0.85, 0.1, 0.22)
    edge = Rgba(1.0, 0.92, 0.2, 1.0)
    drawn = 0
    for order, ri in enumerate(region_indices):
        display = _display_box_for_region(regions[ri], traj)
        if display is None:
            continue
        if cc.draw_region(
            meshcat,
            f"/drake/path_regions/{order}",
            display,
            fill,
            edge,
            edge_width=1.25,
        ):
            drawn += 1
    return drawn


def scene_cache_path(sdf_path: Path) -> Path:
    return sdf_path.with_suffix(".scene.pkl")


def save_scene_cache(
    cache_path: Path,
    *,
    grid_size: int,
    building_seed: int,
    regions,
    adj,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": SCENE_CACHE_VERSION,
        "grid_size": grid_size,
        "building_seed": building_seed,
        "regions": regions,
        "adj": adj,
    }
    with cache_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_scene_cache(
    cache_path: Path,
    *,
    grid_size: int,
    building_seed: int,
) -> tuple[list, dict] | None:
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
    except (OSError, pickle.UnpicklingError):
        return None
    if payload.get("version") != SCENE_CACHE_VERSION:
        return None
    if payload.get("grid_size") != grid_size or payload.get("building_seed") != building_seed:
        return None
    regions = payload.get("regions")
    adj = payload.get("adj")
    if not isinstance(regions, list) or not isinstance(adj, dict):
        return None
    return regions, adj


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
    return regions, adj


def load_or_generate_scene(
    grid_size: int,
    building_seed: int,
    sdf_path: Path,
    *,
    regenerate: bool = False,
) -> tuple[list, dict]:
    cache_path = scene_cache_path(sdf_path)
    if not regenerate:
        cached = load_scene_cache(cache_path, grid_size=grid_size, building_seed=building_seed)
        if cached is not None and sdf_path.is_file():
            print(f"Loaded cached scene → {cache_path}")
            return cached

    print(f"Generating {grid_size}×{grid_size} scene (seed={building_seed})...")
    regions, adj = generate_scene(grid_size, building_seed, sdf_path)
    save_scene_cache(
        cache_path,
        grid_size=grid_size,
        building_seed=building_seed,
        regions=regions,
        adj=adj,
    )
    print(f"  saved scene cache → {cache_path}")
    return regions, adj


def sample_corner_start_goal(regions, rng, diagonal: str = "bl_to_tr"):
    corner = viz.pick_corner_regions(regions, rng, diagonal=diagonal)
    if corner is None:
        return None
    start_pose, goal_pose = corner
    s_idx = g_idx = None
    for i, r in enumerate(regions):
        if s_idx is None and r.PointInSet(start_pose):
            s_idx = i
        if g_idx is None and r.PointInSet(goal_pose):
            g_idx = i
    if s_idx is None or g_idx is None:
        return None
    return start_pose, goal_pose, s_idx, g_idx


def build_facet_tensors(regions, source_idx: int, target_idx: int):
    ab_list = [normalize_facets(r.A(), r.b()) for r in regions]
    n_regions = len(regions)
    n_nodes = n_regions + 2
    fmax = max(t.shape[0] for t in ab_list)
    facet_dim = ab_list[0].shape[1]

    facets = torch.zeros((n_nodes, fmax, facet_dim), dtype=torch.float32)
    mask = torch.zeros((n_nodes, fmax), dtype=torch.bool)
    flags = torch.zeros((n_nodes, 3), dtype=torch.float32)
    for i, tok in enumerate(ab_list):
        m = tok.shape[0]
        facets[i, :m] = torch.from_numpy(tok)
        mask[i, :m] = True
        flags[i, 0] = 1.0
    flags[source_idx, 0] = 0.0
    flags[source_idx, 1] = 1.0
    flags[target_idx, 0] = 0.0
    flags[target_idx, 2] = 1.0
    return facets, mask, flags, facet_dim


def build_graph_tensors(graph, regions, device: torch.device):
    all_vertices = list(graph.Vertices())
    vertex_names = [v.name() for v in all_vertices]
    n_regions = len(regions)
    source_idx = vertex_names.index(
        next(n for n in vertex_names if n == "source" or n.startswith("source"))
    )
    target_idx = vertex_names.index(
        next(n for n in vertex_names if n == "target" or n.startswith("target"))
    )

    all_edges = list(graph.Edges())
    edge_u_names = [e.u().name() for e in all_edges]
    edge_v_names = [e.v().name() for e in all_edges]
    edge_lookup = {(u, v): i for i, (u, v) in enumerate(zip(edge_u_names, edge_v_names))}

    src = [vertex_name_to_index(u, n_regions, None) for u in edge_u_names]
    dst = [vertex_name_to_index(v, n_regions, None) for v in edge_v_names]
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)

    facets, facet_mask, node_flags, facet_dim = build_facet_tensors(regions, source_idx, target_idx)
    return {
        "facets": facets.to(device),
        "facet_mask": facet_mask.to(device),
        "node_flags": node_flags.to(device),
        "edge_index": edge_index,
        "edge_lookup": edge_lookup,
        "source_idx": source_idx,
        "target_idx": target_idx,
        "n_nodes": len(vertex_names),
        "facet_dim": facet_dim,
        "vertex_names": vertex_names,
    }


def load_flow_model(ckpt_path, *, facet_dim, g_dim, encoder_hp, decoder_hp, pointnet_hidden, device):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    model_state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    model = PointNetFlowPredictor(
        facet_dim=facet_dim,
        g_dim=g_dim,
        encoder_hp=encoder_hp,
        decoder_hp=decoder_hp,
        pointnet_hidden=pointnet_hidden,
    )
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Flow ckpt mismatch: missing={missing[:4]} unexpected={unexpected[:4]}")
    return model.eval().to(device)


def load_ranknet(ckpt_path, *, cfg: RankNetConfig, device):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    ranker_state = {k.removeprefix("ranker."): v for k, v in state.items() if k.startswith("ranker.")}
    if not ranker_state:
        ranker_state = state
    ranker = PathRankNet(cfg)
    ranker.load_state_dict(ranker_state, strict=False)
    return ranker.eval().to(device)


def _trajectory_stats(traj) -> tuple[float, float]:
    if traj is None:
        return float("nan"), float("nan")
    duration = float(traj.end_time() - traj.start_time())
    ts = np.linspace(traj.start_time(), traj.end_time(), 200)
    pts = np.array([np.asarray(traj.value(t)).reshape(3) for t in ts])
    length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    return duration, length


def _sample_rounded_paths(planning_graph, relaxed, source, target):
    return randomForwardPathSearch(
        planning_graph,
        relaxed,
        source,
        target,
        max_paths=MAX_PATHS,
        max_trials=MAX_ROUNDING_TRIALS,
        seed=0,
    )


def plan_neural_nonlinear(
    *,
    graph_tensors,
    planning_graph,
    regions,
    relaxed,
    source,
    target,
    gcs,
    flow_model,
    ranker,
    start_pose,
    goal_pose,
    device,
) -> PlanResult:
    t0 = time.perf_counter()
    g_np = np.concatenate([start_pose, goal_pose]).astype(np.float32)
    g_t = torch.from_numpy(g_np).to(device)

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

    candidates = _sample_rounded_paths(planning_graph, relaxed, source, target)
    if not candidates:
        return PlanResult(False, None, float("nan"), float("nan"), 0, time.perf_counter() - t0)

    n_regions = len(regions)
    name_to_idx = {
        name: vertex_name_to_index(name, n_regions, None)
        for name in graph_tensors["vertex_names"]
    }

    node_arr, edge_arr, path_mask = viz._build_path_tensors(
        candidates, graph_tensors["edge_lookup"], name_to_idx
    )
    if node_arr is None:
        return PlanResult(False, None, float("nan"), float("nan"), 0, time.perf_counter() - t0)

    with torch.no_grad():
        scores = ranker(
            node_embeddings=flow_out.node_embeddings,
            edge_flows=phi_proj.to(device),
            path_node_indices=node_arr.to(device),
            path_edge_indices=edge_arr.to(device),
            path_mask=path_mask.to(device),
        )
    ranked = torch.argsort(scores, descending=True).cpu().tolist()

    tried = 0
    for ri in ranked:
        path_edges = candidates[ri]
        if path_edges is None:
            continue
        tried += 1
        try:
            traj, res = solve_nonlinear_restriction_trajectory(gcs, path_edges)
        except RuntimeError:
            continue
        if res.is_success() and traj is not None:
            duration, _ = _trajectory_stats(traj)
            return PlanResult(
                success=True,
                trajectory=traj,
                cost=float(res.get_optimal_cost()),
                duration_s=duration,
                paths_tried=tried,
                elapsed_s=time.perf_counter() - t0,
                region_indices=path_edges_to_region_indices(path_edges, len(regions)),
            )

    return PlanResult(False, None, float("nan"), float("nan"), tried, time.perf_counter() - t0)


def _trace_vertices(traj, n_samples: int = 400) -> np.ndarray:
    ts = np.linspace(traj.start_time(), traj.end_time(), n_samples)
    return np.array([np.asarray(traj.value(t)).reshape(3) for t in ts]).T


def render_motion_html(
    *,
    sdf_path: Path,
    traj,
    regions,
    region_indices: list[int] | None,
    start_pose,
    goal_pose,
    out_path: Path,
    grid_size: int,
    intro_hold_s: float = INTRO_HOLD_S,
    trail_seconds: float = 5.8,
    trail_poses: int = 6,
    show_legend: bool = True,
) -> None:
    meshcat = StartMeshcat()
    meshcat.SetProperty("/Grid", "visible", False)
    meshcat.SetProperty("/Axes", "visible", False)
    meshcat.SetProperty("/Lights/AmbientLight/<object>", "intensity", 0.8)
    meshcat.SetProperty("/Lights/PointLightNegativeX/<object>", "intensity", 0)
    meshcat.SetProperty("/Lights/PointLightPositiveX/<object>", "intensity", 0)

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    Parser(plant, scene_graph).AddModels(str(sdf_path))
    plant.Finalize()

    meshcat_viz = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    animator = meshcat_viz.StartRecording()
    play_traj = DelayedTrajectory(traj, delay=intro_hold_s, speed=PLAYBACK_SPEED)
    traj_system = builder.AddSystem(FlatnessInverter(play_traj, animator))
    QuadrotorGeometry.AddToBuilder(builder, traj_system.get_output_port(0), scene_graph)
    trail_prefixes = add_quadrotor_pose_trail(
        builder,
        scene_graph,
        play_traj,
        trail_seconds=trail_seconds,
        trail_poses=trail_poses,
    )
    diagram = builder.Build()

    meshcat.Delete()
    context = diagram.CreateDefaultContext()
    meshcat_viz.ForcedPublish(meshcat_viz.GetMyContextFromRoot(context))
    if trail_prefixes:
        apply_trail_opacities(meshcat, trail_prefixes)
        print(f"  pose trail: {trail_poses} ghosts over last {trail_seconds:g}s")

    # Overlays under /drake before sim so chase-cam /drake transforms keep them
    # aligned with the building during playback (see visualize_quadrotor.render_html).
    assets_dir = Path(tempfile.mkdtemp(prefix="meshcat_labels_"))
    try:
        if region_indices:
            traversed = trajectory_region_indices(traj, regions, region_indices)
            n = draw_path_regions(meshcat, regions, traversed, traj)
            print(f"  drew {n} regions traversed by trajectory (of {len(region_indices)} on graph path)")

        meshcat.SetLine(
            "/drake/trace/path",
            _trace_vertices(traj),
            line_width=2.5,
            rgba=Rgba(1.0, 1.0, 1.0, 1.0),
        )

        viz.draw_start_goal_markers(meshcat, start_pose, goal_pose, assets_dir)

        sim = Simulator(diagram)
        sim.set_target_realtime_rate(0.0)
        sim.AdvanceTo(play_traj.end_time() + 0.05)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        viz.publish_meshcat_recording_play_once(meshcat_viz)
        html = viz.patch_meshcat_html_play_once(meshcat.StaticHtml())
        if show_legend:
            html = patch_meshcat_html_legend_overlay(html)
        out_path.write_text(html)
        print(f"Saved Meshcat HTML → {out_path}")
    finally:
        shutil.rmtree(assets_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural GCS quadrotor motion demo (no set viz).")
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--building_seed", type=int, default=42)
    parser.add_argument("--query_seed", type=int, default=17)
    parser.add_argument(
        "--diagonal",
        type=str,
        default="bl_to_tr",
        choices=("bl_to_tr", "br_to_tl"),
    )
    parser.add_argument("--flow_ckpt", default="checkpoints/quadrotor_nonlinear/quadrotor_nonlinear_flow_gnn.ckpt")
    parser.add_argument("--ranknet_ckpt", default="checkpoints/quadrotor_nonlinear/quadrotor_nonlinear_ranknet.ckpt")
    parser.add_argument("--output_dir", default="quadrotor/results/motion_demo")
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
    parser.add_argument(
        "--regenerate-scene",
        action="store_true",
        help="Rebuild SDF/regions even if a scene cache exists.",
    )
    parser.add_argument(
        "--intro-hold-s",
        type=float,
        default=INTRO_HOLD_S,
        help="Seconds to hold start pose before trajectory playback.",
    )
    parser.add_argument(
        "--trail-seconds",
        type=float,
        default=5.8,
        help="Ghost trail length in simulation seconds (0 disables trail).",
    )
    parser.add_argument(
        "--trail-poses",
        type=int,
        default=6,
        help="Number of ghost quad poses in the trail window.",
    )
    parser.add_argument(
        "--legend",
        dest="legend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay corner legend for convex sets and trajectory (default: on).",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sdf_path = MODELS_DIR / "room_gen" / f"motion_demo_{args.grid_size}x{args.grid_size}_{args.building_seed}.sdf"

    regions, adj = load_or_generate_scene(
        args.grid_size,
        args.building_seed,
        sdf_path,
        regenerate=args.regenerate_scene,
    )
    print(f"  {len(regions)} regions")

    rng = np.random.default_rng(args.query_seed)
    sampled = sample_corner_start_goal(regions, rng, diagonal=args.diagonal)
    if sampled is None:
        print("Failed to sample corner start/goal; try another --query_seed.", file=sys.stderr)
        sys.exit(1)
    start_pose, goal_pose, s_idx, g_idx = sampled
    print(f"  start region v{s_idx}: {start_pose.round(2)}")
    print(f"  goal  region v{g_idx}: {goal_pose.round(2)}")

    gcs, graph, source, target = build_nonlinear_gcs_problem(
        regions, adjacency_to_edges(adj), start_pose, goal_pose,
    )
    relaxed = solve_nonlinear_relaxation(graph, source, target)
    if not relaxed.is_success():
        print("Convex relaxation failed.", file=sys.stderr)
        sys.exit(1)
    print(f"  relaxed cost: {relaxed.get_optimal_cost():.4f}")

    encoder_hp = EncoderHParams(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_hidden_mult=args.ffn_hidden_mult,
        dropout_p=args.dropout_p,
    )
    hidden = tuple(int(s) for s in args.decoder_hidden.split(",") if s.strip())
    decoder_hp = DecoderHParams(hidden_dims=hidden, dropout_p=args.decoder_dropout_p)

    graph_tensors = build_graph_tensors(graph, regions, device)
    flow_model = load_flow_model(
        args.flow_ckpt,
        facet_dim=graph_tensors["facet_dim"],
        g_dim=6,
        encoder_hp=encoder_hp,
        decoder_hp=decoder_hp,
        pointnet_hidden=args.pointnet_hidden,
        device=device,
    )
    ranker = load_ranknet(
        args.ranknet_ckpt,
        cfg=RankNetConfig(
            d_model=args.d_model,
            num_layers=args.ranker_layers,
            num_heads=args.ranker_heads,
        ),
        device=device,
    )

    print("Planning: Neural GCS (PointNet + RankNet)...")
    result = plan_neural_nonlinear(
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
    print(
        f"  success={result.success}  cost={result.cost:.4f}  "
        f"duration={result.duration_s:.2f}s  paths_tried={result.paths_tried}  "
        f"elapsed={result.elapsed_s:.2f}s"
    )
    if not result.success:
        print("Neural planner failed.", file=sys.stderr)
        sys.exit(1)

    _, path_len = _trajectory_stats(result.trajectory)
    stem = f"motion_{args.grid_size}x{args.grid_size}_b{args.building_seed}_q{args.query_seed}"
    summary = {
        "grid_size": args.grid_size,
        "building_seed": args.building_seed,
        "query_seed": args.query_seed,
        "diagonal": args.diagonal,
        "start_pose": start_pose.tolist(),
        "goal_pose": goal_pose.tolist(),
        "start_region": int(s_idx),
        "goal_region": int(g_idx),
        "relaxed_cost": float(relaxed.get_optimal_cost()),
        "cost": result.cost,
        "duration_s": result.duration_s,
        "path_length_m": path_len,
        "paths_tried": result.paths_tried,
        "elapsed_s": result.elapsed_s,
        "region_indices": result.region_indices or [],
        "traversed_region_indices": trajectory_region_indices(
            result.trajectory, regions, result.region_indices or []
        ) if result.trajectory is not None else [],
    }
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary → {json_path}")

    html_path = out_dir / f"{stem}.html"
    print("Rendering Meshcat...")
    render_motion_html(
        sdf_path=sdf_path,
        traj=result.trajectory,
        regions=regions,
        region_indices=result.region_indices,
        start_pose=start_pose,
        goal_pose=goal_pose,
        out_path=html_path,
        grid_size=args.grid_size,
        intro_hold_s=args.intro_hold_s,
        trail_seconds=args.trail_seconds,
        trail_poses=args.trail_poses,
        show_legend=args.legend,
    )
    print("Done.")


if __name__ == "__main__":
    main()

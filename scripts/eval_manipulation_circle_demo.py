#!/usr/bin/env python3
"""Evaluate vanilla vs neural GCS on the IIWA shelf circle demo.

The script mirrors the quadrotor motion demo, but uses the fixed manipulation
scene and circle waypoint sequence from ``iiwa_shelf_scenes.py``.

Examples:
  python scripts/eval_manipulation_circle_demo.py --planner convex
  python scripts/eval_manipulation_circle_demo.py --planner nonconvex
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from pydrake.geometry import StartMeshcat
from pydrake.geometry.optimization import GraphOfConvexSetsOptions
from pydrake.solvers import MosekSolver

logging.getLogger("drake").setLevel(logging.WARNING)

from manipulation.iiwa_helpers import (
    build_shelf_plant,
    combine_trajectory,
    make_traj,
    trajectory_length,
    visualize_trajectory,
)
from manipulation.paths import DEFAULT_OUTPUT_DIR, DEFAULT_REGIONS_PATH
from manipulation.shelf_gcs import (
    build_demo_sequences,
    build_seed_points,
    load_regions,
    planning_configurations,
)
from manipulation.trajopt import (
    build_nonlinear_gcs_problem,
    build_region_edges,
    iiwa_kinematic_limits,
    region_list,
    solve_nonlinear_relaxation,
)
from model.facet_dataset import normalize_facets, vertex_name_to_index
from model.facet_pointnet import PointNetFlowPredictor
from model.hparams import DecoderHParams, EncoderHParams
from model.inference import project_flows_qp
from model.ranknet import PathRankNet, RankNetConfig
from quadrotor.gcs.linear import LinearGCS
from quadrotor.gcs.rounding import randomForwardPathSearch

MAX_PATHS = 10
MAX_ROUNDING_TRIALS = 100
_MESHCAT_FINISH_RESET_BUG = (
    "if(this.actions.every((A=>A.paused))){this.pause();for(let A of this.actions)A.reset()}"
)
_MESHCAT_FINISH_RESET_FIX = "if(this.actions.every((A=>A.paused))){this.pause()}"


def patch_meshcat_html_play_once(html: str) -> str:
    return html.replace(_MESHCAT_FINISH_RESET_BUG, _MESHCAT_FINISH_RESET_FIX, 1)


def sync_device(device) -> None:
    if device is not None and getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def elapsed_since(t0: float, device=None) -> float:
    sync_device(device)
    return time.perf_counter() - t0


@dataclass
class SegmentResult:
    success: bool
    trajectory: object | None
    cost: float
    paths_tried: int
    elapsed_s: float
    region_names: list[str]
    stage_timings: dict[str, float] = field(default_factory=dict)


@dataclass
class PlanResult:
    success: bool
    trajectory: object | None
    cost: float
    path_length: float
    duration_s: float
    paths_tried: int
    elapsed_s: float
    segments: list[SegmentResult]


@dataclass
class GraphTensors:
    facets: torch.Tensor
    facet_mask: torch.Tensor
    node_flags: torch.Tensor
    edge_index: torch.Tensor
    edge_lookup: dict[tuple[str, str], int]
    source_idx: int
    target_idx: int
    n_nodes: int
    facet_dim: int
    vertex_names: list[str]
    edge_objects: list


def _make_gcs_options(solver=None, solver_options=None) -> GraphOfConvexSetsOptions:
    opts = GraphOfConvexSetsOptions()
    opts.convex_relaxation = True
    opts.max_rounded_paths = 0
    opts.preprocessing = False
    if solver is not None:
        opts.solver = solver
    if solver_options is not None:
        opts.solver_options = solver_options
    return opts


def solve_restriction(graph, source, target, path_edges, *, solver=None, solver_options=None):
    for edge in graph.Edges():
        edge.AddPhiConstraint(edge in path_edges)
    result = graph.SolveShortestPath(source, target, _make_gcs_options(solver, solver_options))
    for edge in graph.Edges():
        edge.ClearPhiConstraints()
    return result


def path_edges_to_region_names(path_edges: list, region_names: list[str]) -> list[str]:
    names = set(region_names)
    out: list[str] = []
    seen: set[str] = set()
    for edge in path_edges:
        for vertex in (edge.u(), edge.v()):
            name = vertex.name()
            if name in names and name not in seen:
                out.append(name)
                seen.add(name)
    return out


def path_edges_to_active_vertices(path_edges: list) -> list:
    vertices = []
    seen: set[int] = set()
    for edge in path_edges:
        for vertex in (edge.u(), edge.v()):
            vid = vertex.id()
            if vid not in seen:
                vertices.append(vertex)
                seen.add(vid)
    return vertices


def _build_path_tensors(candidate_edge_lists, edge_lookup, name_to_idx):
    path_nodes, path_edges = [], []
    for edge_list in candidate_edge_lists:
        if not edge_list:
            continue
        nodes = [name_to_idx[edge_list[0].u().name()]] + [
            name_to_idx[e.v().name()] for e in edge_list
        ]
        edges = [-1] + [edge_lookup[(e.u().name(), e.v().name())] for e in edge_list]
        path_nodes.append(nodes)
        path_edges.append(edges)
    if not path_nodes:
        return None, None, None
    max_len = max(len(p) for p in path_nodes)
    node_arr = torch.full((len(path_nodes), max_len), -1, dtype=torch.long)
    edge_arr = torch.full((len(path_edges), max_len), -1, dtype=torch.long)
    mask = torch.zeros((len(path_nodes), max_len), dtype=torch.bool)
    for i, (nodes, edges) in enumerate(zip(path_nodes, path_edges)):
        n = len(nodes)
        node_arr[i, :n] = torch.tensor(nodes, dtype=torch.long)
        edge_arr[i, :n] = torch.tensor(edges, dtype=torch.long)
        mask[i, :n] = True
    return node_arr, edge_arr, mask


def build_facet_tensors(regions: dict, source_idx: int, target_idx: int, *, device) -> GraphTensors:
    region_names = list(regions.keys())
    ab_list = [normalize_facets(regions[name].A(), regions[name].b()) for name in region_names]
    n_regions = len(region_names)
    fmax = max(t.shape[0] for t in ab_list)
    facet_dim = ab_list[0].shape[1]

    facets = torch.zeros((n_regions + 2, fmax, facet_dim), dtype=torch.float32)
    mask = torch.zeros((n_regions + 2, fmax), dtype=torch.bool)
    flags = torch.zeros((n_regions + 2, 3), dtype=torch.float32)
    for i, tok in enumerate(ab_list):
        facets[i, : tok.shape[0]] = torch.from_numpy(tok)
        mask[i, : tok.shape[0]] = True
        flags[i, 0] = 1.0
    flags[source_idx, 1] = 1.0
    flags[target_idx, 2] = 1.0
    return facets.to(device), mask.to(device), flags.to(device), facet_dim


def build_graph_tensors(graph, regions: dict, *, device) -> GraphTensors:
    n_regions = len(regions)
    region_name_to_idx = {name: i for i, name in enumerate(regions.keys())}
    vertex_names = [v.name() for v in graph.Vertices()]
    source_name = next(n for n in vertex_names if n == "source" or n.startswith("source"))
    target_name = next(n for n in vertex_names if n == "target" or n.startswith("target"))
    source_idx = vertex_name_to_index(source_name, n_regions, region_name_to_idx)
    target_idx = vertex_name_to_index(target_name, n_regions, region_name_to_idx)

    edge_u = [e.u().name() for e in graph.Edges()]
    edge_v = [e.v().name() for e in graph.Edges()]
    edge_lookup = {(u, v): i for i, (u, v) in enumerate(zip(edge_u, edge_v))}
    src = [vertex_name_to_index(u, n_regions, region_name_to_idx) for u in edge_u]
    dst = [vertex_name_to_index(v, n_regions, region_name_to_idx) for v in edge_v]
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    facets, facet_mask, node_flags, facet_dim = build_facet_tensors(
        regions, source_idx, target_idx, device=device
    )
    return GraphTensors(
        facets=facets,
        facet_mask=facet_mask,
        node_flags=node_flags,
        edge_index=edge_index,
        edge_lookup=edge_lookup,
        source_idx=source_idx,
        target_idx=target_idx,
        n_nodes=n_regions + 2,
        facet_dim=facet_dim,
        vertex_names=vertex_names,
        edge_objects=list(graph.Edges()),
    )


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


def predict_projected_flows(
    *,
    graph_tensors: GraphTensors,
    flow_model,
    start_q,
    goal_q,
    device,
    timings: dict[str, float] | None = None,
):
    g_t = torch.from_numpy(np.concatenate([start_q, goal_q]).astype(np.float32)).to(device)
    sync_device(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        flow_out = flow_model(
            facets=graph_tensors.facets,
            facet_mask=graph_tensors.facet_mask,
            node_flags=graph_tensors.node_flags,
            edge_index=graph_tensors.edge_index,
            g=g_t,
            batch=None,
        )
        phi_hat = torch.sigmoid(flow_out.edge_logits).detach().cpu()
    if timings is not None:
        timings["gnn_s"] = elapsed_since(t0, device)

    sync_device(device)
    t0 = time.perf_counter()
    phi_proj = project_flows_qp(
        edge_index=graph_tensors.edge_index.cpu(),
        phi_hat=phi_hat,
        num_nodes=graph_tensors.n_nodes,
        source_idx=graph_tensors.source_idx,
        target_idx=graph_tensors.target_idx,
    )
    if timings is not None:
        timings["qp_s"] = elapsed_since(t0, device)
    return flow_out, phi_proj


def sample_candidate_paths_from_flows(
    *,
    graph_tensors: GraphTensors,
    phi: torch.Tensor,
    seed: int,
    flow_tol: float = 1e-6,
) -> list[list]:
    rng = np.random.default_rng(seed)
    edge_index = graph_tensors.edge_index.detach().cpu()
    phi_np = phi.detach().cpu().float().numpy().reshape(-1)
    src = edge_index[0].numpy().astype(int)
    dst = edge_index[1].numpy().astype(int)

    outgoing: list[list[int]] = [[] for _ in range(graph_tensors.n_nodes)]
    for edge_idx, u in enumerate(src):
        outgoing[int(u)].append(edge_idx)

    paths: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    trials = 0
    while len(paths) < MAX_PATHS and trials < MAX_ROUNDING_TRIALS:
        trials += 1
        cur = int(graph_tensors.source_idx)
        visited = {cur}
        edge_path: list[int] = []

        for _ in range(graph_tensors.n_nodes + 1):
            if cur == int(graph_tensors.target_idx):
                key = tuple(edge_path)
                if key and key not in seen:
                    seen.add(key)
                    paths.append(edge_path)
                break

            candidates = [
                e for e in outgoing[cur]
                if dst[e] not in visited and phi_np[e] > flow_tol
            ]
            if not candidates:
                break
            probs = np.asarray([phi_np[e] for e in candidates], dtype=float)
            probs = probs / probs.sum()
            chosen = int(rng.choice(candidates, p=probs))
            edge_path.append(chosen)
            cur = int(dst[chosen])
            visited.add(cur)

    return [[graph_tensors.edge_objects[i] for i in edge_indices] for edge_indices in paths]


def rank_candidate_paths(
    *,
    graph_tensors: GraphTensors,
    candidates,
    flow_out,
    phi_proj: torch.Tensor,
    ranker,
    regions: dict,
    device,
    timings: dict[str, float] | None = None,
) -> list[int]:
    sync_device(device)
    t0 = time.perf_counter()
    name_to_idx = {
        name: vertex_name_to_index(name, len(regions), {k: i for i, k in enumerate(regions.keys())})
        for name in graph_tensors.vertex_names
    }
    node_arr, edge_arr, path_mask = _build_path_tensors(
        candidates, graph_tensors.edge_lookup, name_to_idx
    )
    if node_arr is None:
        return []
    if timings is not None:
        timings["path_tensor_s"] = elapsed_since(t0, device)

    sync_device(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        scores = ranker(
            node_embeddings=flow_out.node_embeddings,
            edge_flows=phi_proj.to(device),
            path_node_indices=node_arr.to(device),
            path_edge_indices=edge_arr.to(device),
            path_mask=path_mask.to(device),
        )
    if timings is not None:
        timings["ranknet_s"] = elapsed_since(t0, device)
    return torch.argsort(scores, descending=True).cpu().tolist()


def plan_linear_segment_vanilla(regions: dict, start_q, goal_q, *, seed: int, speed: float) -> SegmentResult:
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    stage_t0 = time.perf_counter()
    gcs = LinearGCS(regions.copy())
    gcs.addSourceTarget(start_q, goal_q)
    gcs.setSolver(MosekSolver())
    timings["gcs_build_s"] = time.perf_counter() - stage_t0
    stage_t0 = time.perf_counter()
    relaxed = gcs.gcs.SolveShortestPath(gcs.source, gcs.target, _make_gcs_options(gcs.solver, gcs.options))
    timings["relaxation_s"] = time.perf_counter() - stage_t0
    if not relaxed.is_success():
        return SegmentResult(False, None, float("nan"), 0, time.perf_counter() - t0, [], timings)
    stage_t0 = time.perf_counter()
    candidates = randomForwardPathSearch(
        gcs.gcs, relaxed, gcs.source, gcs.target,
        max_paths=MAX_PATHS, max_trials=MAX_ROUNDING_TRIALS, seed=seed,
    )
    timings["rounding_s"] = time.perf_counter() - stage_t0
    best_traj, best_cost, best_regions = None, float("inf"), []
    tried = 0
    restriction_s = 0.0
    for path_edges in candidates or []:
        tried += 1
        stage_t0 = time.perf_counter()
        res = solve_restriction(gcs.gcs, gcs.source, gcs.target, path_edges, solver=gcs.solver)
        restriction_s += time.perf_counter() - stage_t0
        if not res.is_success():
            continue
        cost = float(res.get_optimal_cost())
        if cost < best_cost:
            waypoints = np.column_stack([res.GetSolution(e.xv()) for e in path_edges])
            best_traj = make_traj(waypoints, speed=speed)
            best_cost = cost
            best_regions = path_edges_to_region_names(path_edges, list(regions.keys()))
    timings["restriction_s"] = restriction_s
    return SegmentResult(
        best_traj is not None,
        best_traj,
        best_cost,
        tried,
        time.perf_counter() - t0,
        best_regions,
        timings,
    )


def plan_linear_segment_neural(
    regions: dict,
    start_q,
    goal_q,
    *,
    seed: int,
    speed: float,
    flow_model,
    ranker,
    device,
) -> SegmentResult:
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    stage_t0 = time.perf_counter()
    gcs = LinearGCS(regions.copy())
    gcs.addSourceTarget(start_q, goal_q)
    gcs.setSolver(MosekSolver())
    timings["gcs_build_s"] = time.perf_counter() - stage_t0
    # Neural GCS does not solve the convex relaxation. It predicts edge flows,
    # projects them to the flow polytope, and samples candidate paths directly.
    sync_device(device)
    stage_t0 = time.perf_counter()
    graph_tensors = build_graph_tensors(gcs.gcs, regions, device=device)
    timings["graph_tensors_s"] = elapsed_since(stage_t0, device)
    flow_out, phi_proj = predict_projected_flows(
        graph_tensors=graph_tensors,
        flow_model=flow_model,
        start_q=start_q,
        goal_q=goal_q,
        device=device,
        timings=timings,
    )
    sync_device(device)
    stage_t0 = time.perf_counter()
    candidates = sample_candidate_paths_from_flows(
        graph_tensors=graph_tensors,
        phi=phi_proj,
        seed=seed,
    )
    timings["sampling_s"] = elapsed_since(stage_t0, device)
    if not candidates:
        return SegmentResult(False, None, float("nan"), 0, time.perf_counter() - t0, [], timings)
    ranked = rank_candidate_paths(
        graph_tensors=graph_tensors,
        candidates=candidates,
        flow_out=flow_out,
        phi_proj=phi_proj,
        ranker=ranker,
        regions=regions,
        device=device,
        timings=timings,
    )
    tried = 0
    restriction_s = 0.0
    for idx in ranked:
        path_edges = candidates[idx]
        tried += 1
        stage_t0 = time.perf_counter()
        res = solve_restriction(gcs.gcs, gcs.source, gcs.target, path_edges, solver=gcs.solver)
        restriction_s += time.perf_counter() - stage_t0
        if res.is_success():
            timings["restriction_s"] = restriction_s
            waypoints = np.column_stack([res.GetSolution(e.xv()) for e in path_edges])
            traj = make_traj(waypoints, speed=speed)
            return SegmentResult(
                True,
                traj,
                float(res.get_optimal_cost()),
                tried,
                time.perf_counter() - t0,
                path_edges_to_region_names(path_edges, list(regions.keys())),
                timings,
            )
    timings["restriction_s"] = restriction_s
    return SegmentResult(False, None, float("nan"), tried, time.perf_counter() - t0, [], timings)


def solve_nonlinear_restriction_trajectory(gcs, path_edges, options):
    vertices = path_edges_to_active_vertices(path_edges)
    return gcs.SolveConvexRestriction(vertices, options)


def plan_nonlinear_segment_vanilla(regions: dict, start_q, goal_q, *, plant, seed: int) -> SegmentResult:
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    stage_t0 = time.perf_counter()
    polys = region_list(regions)
    edges = build_region_edges(polys)
    vel_limits, accel_limits = iiwa_kinematic_limits(plant)
    gcs, graph, source, target = build_nonlinear_gcs_problem(
        polys, edges, start_q, goal_q, vel_limits=vel_limits, accel_limits=accel_limits,
    )
    timings["gcs_build_s"] = time.perf_counter() - stage_t0
    stage_t0 = time.perf_counter()
    relaxed = solve_nonlinear_relaxation(graph, source, target)
    timings["relaxation_s"] = time.perf_counter() - stage_t0
    if not relaxed.is_success():
        return SegmentResult(False, None, float("nan"), 0, time.perf_counter() - t0, [], timings)
    stage_t0 = time.perf_counter()
    candidates = randomForwardPathSearch(
        graph, relaxed, source, target,
        max_paths=MAX_PATHS, max_trials=MAX_ROUNDING_TRIALS, seed=seed,
    )
    timings["rounding_s"] = time.perf_counter() - stage_t0
    best_traj, best_cost, best_regions = None, float("inf"), []
    tried = 0
    from manipulation.trajopt import _nonlinear_gcs_options

    options = _nonlinear_gcs_options(restriction=True)
    restriction_s = 0.0
    for path_edges in candidates or []:
        tried += 1
        try:
            stage_t0 = time.perf_counter()
            traj, res = solve_nonlinear_restriction_trajectory(gcs, path_edges, options)
            restriction_s += time.perf_counter() - stage_t0
        except RuntimeError:
            restriction_s += time.perf_counter() - stage_t0
            continue
        if not res.is_success() or traj is None:
            continue
        cost = float(res.get_optimal_cost())
        if cost < best_cost:
            best_traj, best_cost = traj, cost
            best_regions = path_edges_to_region_names(path_edges, [f"Subgraph0: Region{i}" for i in range(len(polys))])
    timings["restriction_s"] = restriction_s
    return SegmentResult(
        best_traj is not None,
        best_traj,
        best_cost,
        tried,
        time.perf_counter() - t0,
        best_regions,
        timings,
    )


def plan_nonlinear_segment_neural(
    regions: dict,
    start_q,
    goal_q,
    *,
    plant,
    seed: int,
    flow_model,
    ranker,
    device,
) -> SegmentResult:
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    stage_t0 = time.perf_counter()
    polys = region_list(regions)
    edges = build_region_edges(polys)
    vel_limits, accel_limits = iiwa_kinematic_limits(plant)
    gcs, graph, source, target = build_nonlinear_gcs_problem(
        polys, edges, start_q, goal_q, vel_limits=vel_limits, accel_limits=accel_limits,
    )
    timings["gcs_build_s"] = time.perf_counter() - stage_t0
    # Neural GCS does not solve the convex relaxation. It predicts edge flows,
    # projects them to the flow polytope, and samples candidate paths directly.
    sync_device(device)
    stage_t0 = time.perf_counter()
    graph_tensors = build_graph_tensors(graph, regions, device=device)
    timings["graph_tensors_s"] = elapsed_since(stage_t0, device)
    flow_out, phi_proj = predict_projected_flows(
        graph_tensors=graph_tensors,
        flow_model=flow_model,
        start_q=start_q,
        goal_q=goal_q,
        device=device,
        timings=timings,
    )
    sync_device(device)
    stage_t0 = time.perf_counter()
    candidates = sample_candidate_paths_from_flows(
        graph_tensors=graph_tensors,
        phi=phi_proj,
        seed=seed,
    )
    timings["sampling_s"] = elapsed_since(stage_t0, device)
    if not candidates:
        return SegmentResult(False, None, float("nan"), 0, time.perf_counter() - t0, [], timings)
    ranked = rank_candidate_paths(
        graph_tensors=graph_tensors,
        candidates=candidates,
        flow_out=flow_out,
        phi_proj=phi_proj,
        ranker=ranker,
        regions=regions,
        device=device,
        timings=timings,
    )
    from manipulation.trajopt import _nonlinear_gcs_options

    options = _nonlinear_gcs_options(restriction=True)
    tried = 0
    restriction_s = 0.0
    region_labels = [f"Subgraph0: Region{i}" for i in range(len(polys))]
    for idx in ranked:
        path_edges = candidates[idx]
        tried += 1
        try:
            stage_t0 = time.perf_counter()
            traj, res = solve_nonlinear_restriction_trajectory(gcs, path_edges, options)
            restriction_s += time.perf_counter() - stage_t0
        except RuntimeError:
            restriction_s += time.perf_counter() - stage_t0
            continue
        if res.is_success() and traj is not None:
            timings["restriction_s"] = restriction_s
            return SegmentResult(
                True,
                traj,
                float(res.get_optimal_cost()),
                tried,
                time.perf_counter() - t0,
                path_edges_to_region_names(path_edges, region_labels),
                timings,
            )
    timings["restriction_s"] = restriction_s
    return SegmentResult(False, None, float("nan"), tried, time.perf_counter() - t0, [], timings)


def combine_segments(results: list[SegmentResult]) -> object | None:
    if not all(r.success and r.trajectory is not None for r in results):
        return None
    return combine_trajectory([r.trajectory for r in results], wait=0.0)


def summarize_plan(results: list[SegmentResult], combined) -> PlanResult:
    elapsed = float(sum(r.elapsed_s for r in results))
    cost = float(sum(r.cost for r in results if np.isfinite(r.cost)))
    paths_tried = int(sum(r.paths_tried for r in results))
    if combined is None:
        return PlanResult(False, None, float("nan"), float("nan"), float("nan"), paths_tried, elapsed, results)
    return PlanResult(
        True,
        combined,
        cost,
        trajectory_length(combined),
        float(combined.end_time() - combined.start_time()),
        paths_tried,
        elapsed,
        results,
    )


def plan_circle(
    *,
    planner: str,
    mode: str,
    regions: dict,
    sequence: list[np.ndarray],
    plant,
    seed: int,
    speed: float,
    flow_model=None,
    ranker=None,
    device=None,
) -> PlanResult:
    segment_results = []
    for i, (start_q, goal_q) in enumerate(zip(sequence[:-1], sequence[1:])):
        print(f"    segment {i + 1}/{len(sequence) - 1}...", flush=True)
        if planner == "convex" and mode == "vanilla":
            res = plan_linear_segment_vanilla(regions, start_q, goal_q, seed=seed, speed=speed)
        elif planner == "convex":
            res = plan_linear_segment_neural(
                regions, start_q, goal_q, seed=seed, speed=speed,
                flow_model=flow_model, ranker=ranker, device=device,
            )
        elif mode == "vanilla":
            res = plan_nonlinear_segment_vanilla(regions, start_q, goal_q, plant=plant, seed=seed)
        else:
            res = plan_nonlinear_segment_neural(
                regions, start_q, goal_q, plant=plant, seed=seed,
                flow_model=flow_model, ranker=ranker, device=device,
            )
        segment_results.append(res)
        if not res.success:
            print(f"    failed after trying {res.paths_tried} path(s)", flush=True)
            break
        print(
            f"    cost={res.cost:.4f} paths_tried={res.paths_tried} elapsed={res.elapsed_s:.2f}s",
            flush=True,
        )
    return summarize_plan(segment_results, combine_segments(segment_results))


def plan_to_json(plan: PlanResult) -> dict:
    return {
        "success": plan.success,
        "cost": plan.cost,
        "path_length": plan.path_length,
        "duration_s": plan.duration_s,
        "paths_tried": plan.paths_tried,
        "elapsed_s": plan.elapsed_s,
        "segments": [
            {
                "success": s.success,
                "cost": s.cost,
                "paths_tried": s.paths_tried,
                "elapsed_s": s.elapsed_s,
                "region_names": s.region_names,
                "stage_timings": s.stage_timings,
            }
            for s in plan.segments
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manipulation circle demo: vanilla vs neural GCS.")
    parser.add_argument("--planner", choices=("convex", "nonconvex"), default="convex")
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "circle_demo")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--max-paths", type=int, default=MAX_PATHS)
    parser.add_argument("--max-trials", type=int, default=MAX_ROUNDING_TRIALS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--flow-ckpt", default=None)
    parser.add_argument("--ranknet-ckpt", default=None)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-hidden-mult", type=int, default=2)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--decoder-hidden", default="256,256")
    parser.add_argument("--decoder-dropout-p", type=float, default=0.1)
    parser.add_argument("--pointnet-hidden", type=int, default=64)
    parser.add_argument("--ranker-layers", type=int, default=3)
    parser.add_argument("--ranker-heads", type=int, default=4)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--show-line", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global MAX_PATHS, MAX_ROUNDING_TRIALS
    MAX_PATHS = int(args.max_paths)
    MAX_ROUNDING_TRIALS = int(args.max_trials)

    from manipulation.paths import manipulation_models_hint, manipulation_models_ready

    if not manipulation_models_ready():
        print(manipulation_models_hint(), file=sys.stderr)
        sys.exit(1)
    if not args.regions.exists():
        print(f"Missing IRIS regions: {args.regions}", file=sys.stderr)
        print("Run: python scripts/iiwa_shelf_scenes.py --generate-regions --regions-only", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_dir = f"manipulation/checkpoints/manipulation_{args.planner}"
    flow_ckpt = args.flow_ckpt or f"{ckpt_dir}/manipulation_{args.planner}_flow_gnn.ckpt"
    ranknet_ckpt = args.ranknet_ckpt or f"{ckpt_dir}/manipulation_{args.planner}_ranknet.ckpt"

    print("Building IIWA shelf scene...")
    plant, _, diagram, _ = build_shelf_plant()
    print(f"Loading regions: {args.regions}")
    regions = load_regions(args.regions)

    demo_configs = planning_configurations(regions)
    sequence = build_demo_sequences(demo_configs, build_seed_points())["circle"]
    print(f"Circle sequence: {len(sequence)} waypoints, {len(sequence) - 1} segments")
    print(f"Planner family: {args.planner}")

    encoder_hp = EncoderHParams(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_hidden_mult=args.ffn_hidden_mult,
        dropout_p=args.dropout_p,
    )
    decoder_hp = DecoderHParams(
        hidden_dims=tuple(int(s) for s in args.decoder_hidden.split(",") if s.strip()),
        dropout_p=args.decoder_dropout_p,
    )
    # Build a small temporary graph to determine the facet dimension.
    if args.planner == "convex":
        tmp = LinearGCS(regions.copy())
        tmp.addSourceTarget(sequence[0], sequence[1])
        tmp_graph = tmp.gcs
    else:
        polys = region_list(regions)
        vel_limits, accel_limits = iiwa_kinematic_limits(plant)
        _, tmp_graph, _, _ = build_nonlinear_gcs_problem(
            polys,
            build_region_edges(polys),
            sequence[0],
            sequence[1],
            vel_limits=vel_limits,
            accel_limits=accel_limits,
        )
    facet_dim = build_graph_tensors(tmp_graph, regions, device=device).facet_dim

    flow_model = load_flow_model(
        flow_ckpt,
        facet_dim=facet_dim,
        g_dim=14,
        encoder_hp=encoder_hp,
        decoder_hp=decoder_hp,
        pointnet_hidden=args.pointnet_hidden,
        device=device,
    )
    ranker = load_ranknet(
        ranknet_ckpt,
        cfg=RankNetConfig(
            d_model=args.d_model,
            num_layers=args.ranker_layers,
            num_heads=args.ranker_heads,
        ),
        device=device,
    )
    print(f"Loaded flow: {flow_ckpt}")
    print(f"Loaded ranknet: {ranknet_ckpt}")

    print("\nPlanning: vanilla GCS...")
    vanilla = plan_circle(
        planner=args.planner,
        mode="vanilla",
        regions=regions,
        sequence=sequence,
        plant=plant,
        seed=args.seed,
        speed=args.speed,
    )
    print(
        f"  success={vanilla.success} cost={vanilla.cost:.4f} "
        f"length={vanilla.path_length:.3f} elapsed={vanilla.elapsed_s:.2f}s"
    )

    print("\nPlanning: neural GCS (PointNet + RankNet)...")
    neural = plan_circle(
        planner=args.planner,
        mode="neural",
        regions=regions,
        sequence=sequence,
        plant=plant,
        seed=args.seed,
        speed=args.speed,
        flow_model=flow_model,
        ranker=ranker,
        device=device,
    )
    print(
        f"  success={neural.success} cost={neural.cost:.4f} "
        f"length={neural.path_length:.3f} elapsed={neural.elapsed_s:.2f}s"
    )

    if not vanilla.success and not neural.success:
        print("Both planners failed.", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"circle_{args.planner}_seed{args.seed}"
    summary = {
        "planner": args.planner,
        "demo": "circle",
        "seed": args.seed,
        "flow_ckpt": str(flow_ckpt),
        "ranknet_ckpt": str(ranknet_ckpt),
        "vanilla": plan_to_json(vanilla),
        "neural": plan_to_json(neural),
    }
    json_path = args.output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary -> {json_path}")

    if args.no_viz:
        return

    viz_trajs = [p.trajectory for p in (vanilla, neural) if p.trajectory is not None]
    print("Rendering Meshcat HTML (vanilla then neural, delayed)...")
    meshcat = StartMeshcat()
    visualize_trajectory(
        meshcat,
        viz_trajs,
        show_line=args.show_line,
        ghost_configs=sequence,
        alpha=0.3,
        plan_wait=3.0 if len(viz_trajs) > 1 else 2.0,
    )
    html_path = args.output_dir / f"{stem}.html"
    html_path.write_text(patch_meshcat_html_play_once(meshcat.StaticHtml()))
    print(f"Saved HTML -> {html_path}")


if __name__ == "__main__":
    main()

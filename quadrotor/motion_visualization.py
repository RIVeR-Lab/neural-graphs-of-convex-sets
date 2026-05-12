from __future__ import annotations

import base64
import io
import json
import shutil
import struct
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pydrake.examples import QuadrotorGeometry
from pydrake.geometry import (
    Box as DrakeBox,
    Cylinder,
    Mesh,
    MeshcatAnimation,
    MeshcatVisualizer,
    Rgba,
    Sphere,
    StartMeshcat,
)
from pydrake.geometry.optimization import HPolyhedron
from pydrake.math import BsplineBasis, RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.trajectories import BsplineTrajectory

from model.facet_dataset import normalize_facets, vertex_name_to_index
from model.facet_pointnet import PointNetFlowPredictor
from model.ranknet import PathRankNet, RankNetConfig
from quadrotor.gcs.bezier import BezierTrajectory
from quadrotor.gcs.rounding import randomForwardPathSearch
from quadrotor.gcs.trajopt import solve_nonlinear_restriction_trajectory
from quadrotor.gcs.viz_utils import DelayedTrajectory
from quadrotor.helpers import FlatnessInverter

MAX_PATHS = 10
MAX_ROUNDING_TRIALS = 100
TRAJ_REGION_SAMPLES = 400
TRAJ_CONTAINMENT_SAMPLES = 1200
TRAJ_CONTAINMENT_TOL = 1e-5
TRAJ_BOX_MARGIN = 1.2
TRAJ_BOX_MIN_EXTENT = 0.9
PLAYBACK_SPEED = 0.5
LEGEND_FONT_SCALE = 2.0
LABEL_FONT_SIZE = 72

_MESHCAT_FINISH_RESET_BUG = (
    "if(this.actions.every((A=>A.paused))){this.pause();for(let A of this.actions)A.reset()}"
)
_MESHCAT_FINISH_RESET_FIX = "if(this.actions.every((A=>A.paused))){this.pause()}"


@dataclass
class PlanResult:
    success: bool
    trajectory: object | None
    cost: float
    duration_s: float
    paths_tried: int
    elapsed_s: float
    region_indices: list[int] | None = None


def configure_meshcat_recording_play_once(recording: MeshcatAnimation) -> None:
    recording.set_loop_mode(MeshcatAnimation.LoopMode.kLoopOnce)
    recording.set_repetitions(1)
    recording.set_clamp_when_finished(True)
    recording.set_autoplay(True)


def publish_meshcat_recording_play_once(meshcat_viz: MeshcatVisualizer) -> None:
    configure_meshcat_recording_play_once(meshcat_viz.get_mutable_recording())
    meshcat_viz.PublishRecording()


def patch_meshcat_html_play_once(html: str) -> str:
    return html.replace(_MESHCAT_FINISH_RESET_BUG, _MESHCAT_FINISH_RESET_FIX, 1)


def patch_meshcat_html_legend_overlay(
    html: str,
    *,
    font_scale: float = LEGEND_FONT_SCALE,
) -> str:
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


def path_edges_to_region_indices(path_edges, n_regions: int) -> list[int]:
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


def _box_from_hpoly(region: HPolyhedron):
    A, b = region.A(), region.b()
    lb = np.full(3, -np.inf)
    ub = np.full(3, np.inf)
    for i in range(A.shape[0]):
        row = A[i]
        nz = np.nonzero(row)[0]
        if len(nz) != 1:
            continue
        dim = nz[0]
        if row[dim] > 0:
            ub[dim] = min(ub[dim], b[i] / row[dim])
        else:
            lb[dim] = max(lb[dim], b[i] / row[dim])
    if np.any(np.isinf(lb)) or np.any(np.isinf(ub)) or np.any(ub <= lb):
        return None
    return (lb + ub) / 2.0, ub - lb


def _box_edges(center, size):
    cx, cy, cz = center
    hx, hy, hz = np.asarray(size) / 2.0
    signs = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    corners = {s: np.array([cx + s[0] * hx, cy + s[1] * hy, cz + s[2] * hz]) for s in signs}
    starts, ends = [], []
    for a in signs:
        for axis in range(3):
            b = list(a)
            if b[axis] == -1:
                b[axis] = 1
                starts.append(corners[a])
                ends.append(corners[tuple(b)])
    return np.array(starts).T, np.array(ends).T


def draw_region(meshcat, path, region, fill_rgba, edge_rgba, edge_width=2.0):
    got = _box_from_hpoly(region)
    if got is None:
        return False
    c, s = got
    if np.any(s <= 0):
        return False
    meshcat.SetObject(f"{path}/fill", DrakeBox(*s), fill_rgba)
    meshcat.SetTransform(f"{path}/fill", RigidTransform(c))
    ss, es = _box_edges(c, s)
    meshcat.SetLineSegments(f"{path}/edges", ss, es, line_width=edge_width, rgba=edge_rgba)
    return True


def _region_bounds(region) -> tuple[np.ndarray, np.ndarray] | None:
    got = _box_from_hpoly(region)
    if got is None:
        return None
    center, size = got
    return center - size / 2.0, center + size / 2.0


def _display_box_for_region(region, traj) -> HPolyhedron | None:
    bounds = _region_bounds(region)
    if bounds is None:
        return None
    rlb, rub = bounds
    pts = _trajectory_points_in_region(traj, region)
    if len(pts) == 0:
        return None

    lb = np.maximum(pts.min(axis=0) - TRAJ_BOX_MARGIN, rlb)
    ub = np.minimum(pts.max(axis=0) + TRAJ_BOX_MARGIN, rub)
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
    traversed: list[int] = []
    seen: set[int] = set()
    for ri in path_region_indices:
        if ri in seen or not (0 <= ri < len(regions)):
            continue
        if len(_trajectory_points_in_region(traj, regions[ri])) > 0:
            traversed.append(ri)
            seen.add(ri)
    return traversed


def trajectory_stays_in_path_regions(
    traj,
    regions,
    path_region_indices: list[int],
    *,
    n_samples: int = TRAJ_CONTAINMENT_SAMPLES,
    tol: float = TRAJ_CONTAINMENT_TOL,
) -> bool:
    """Check that the displayed trajectory is contained in the selected GCS regions."""
    if not path_region_indices:
        return False

    path_regions = [
        regions[ri]
        for ri in path_region_indices
        if 0 <= ri < len(regions)
    ]
    if not path_regions:
        return False

    for t in np.linspace(traj.start_time(), traj.end_time(), n_samples):
        p = np.asarray(traj.value(t)).reshape(3)
        if not any(region.PointInSet(p, tol) for region in path_regions):
            return False
    return True


def draw_path_regions(meshcat, regions, region_indices: list[int], traj) -> int:
    fill = Rgba(1.0, 0.85, 0.1, 0.22)
    edge = Rgba(1.0, 0.92, 0.2, 1.0)
    drawn = 0
    for order, ri in enumerate(region_indices):
        display = _display_box_for_region(regions[ri], traj)
        if display is None:
            continue
        if draw_region(meshcat, f"/drake/path_regions/{order}", display, fill, edge, edge_width=1.25):
            drawn += 1
    return drawn


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
    source_idx = vertex_names.index(next(n for n in vertex_names if n == "source" or n.startswith("source")))
    target_idx = vertex_names.index(next(n for n in vertex_names if n == "target" or n.startswith("target")))

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


def trajectory_stats(traj) -> tuple[float, float]:
    if traj is None:
        return float("nan"), float("nan")
    duration = float(traj.end_time() - traj.start_time())
    ts = np.linspace(traj.start_time(), traj.end_time(), 200)
    pts = np.array([np.asarray(traj.value(t)).reshape(3) for t in ts])
    length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    return duration, length


def sample_rounded_paths(planning_graph, relaxed, source, target):
    return randomForwardPathSearch(
        planning_graph,
        relaxed,
        source,
        target,
        max_paths=MAX_PATHS,
        max_trials=MAX_ROUNDING_TRIALS,
        seed=0,
    )


def build_path_tensors(candidate_edge_lists, edge_lookup, name_to_idx):
    path_nodes_list, path_edges_list = [], []
    for path_edges in candidate_edge_lists:
        if not path_edges:
            continue
        node_seq = [name_to_idx[path_edges[0].u().name()]] + [
            name_to_idx[e.v().name()] for e in path_edges
        ]
        edge_seq = [-1] + [edge_lookup[(e.u().name(), e.v().name())] for e in path_edges]
        path_nodes_list.append(node_seq)
        path_edges_list.append(edge_seq)
    if not path_nodes_list:
        return None, None, None
    max_len = max(len(p) for p in path_nodes_list)
    node_arr = torch.full((len(path_nodes_list), max_len), -1, dtype=torch.long)
    edge_arr = torch.full((len(path_edges_list), max_len), -1, dtype=torch.long)
    mask = torch.zeros((len(path_nodes_list), max_len), dtype=torch.bool)
    for i, (nodes, edges) in enumerate(zip(path_nodes_list, path_edges_list)):
        L = len(nodes)
        node_arr[i, :L] = torch.tensor(nodes, dtype=torch.long)
        edge_arr[i, :L] = torch.tensor(edges, dtype=torch.long)
        mask[i, :L] = True
    return node_arr, edge_arr, mask


def extract_trajectory(gcs_obj, path_edges, result):
    order = gcs_obj.order
    knots = np.zeros(order + 1)
    path_ctrl, time_ctrl = [], []

    for edge in path_edges:
        if edge.v() == gcs_obj.target:
            knots = np.concatenate((knots, [knots[-1]]))
            path_ctrl.append(result.GetSolution(edge.xv()).reshape(-1, 1))
            time_ctrl.append(np.array([[result.GetSolution(edge.xu())[-1]]]))
            break
        edge_time = knots[-1] + 1.0
        knots = np.concatenate((knots, np.full(order, edge_time)))
        ep = np.reshape(result.GetSolution(edge.xv())[: -(order + 1)], (gcs_obj.dimension, order + 1), "F")
        et = result.GetSolution(edge.xv())[-(order + 1) :]
        for ii in range(order):
            path_ctrl.append(ep[:, ii : ii + 1])
            time_ctrl.append(np.array([[et[ii]]]))

    offset = time_ctrl[0].copy()
    for ii in range(len(time_ctrl)):
        time_ctrl[ii] -= offset

    path_spline = BsplineTrajectory(BsplineBasis(order + 1, knots), path_ctrl)
    time_spline = BsplineTrajectory(BsplineBasis(order + 1, knots), time_ctrl)
    return BezierTrajectory(path_spline, time_spline)


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
    g_t = torch.from_numpy(np.concatenate([start_pose, goal_pose]).astype(np.float32)).to(device)

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

    from model.inference import project_flows_qp

    phi_proj = project_flows_qp(
        edge_index=graph_tensors["edge_index"].cpu(),
        phi_hat=phi_hat,
        num_nodes=int(graph_tensors["n_nodes"]),
        source_idx=int(graph_tensors["source_idx"]),
        target_idx=int(graph_tensors["target_idx"]),
    )

    candidates = sample_rounded_paths(planning_graph, relaxed, source, target)
    if not candidates:
        return PlanResult(False, None, float("nan"), float("nan"), 0, time.perf_counter() - t0)

    n_regions = len(regions)
    name_to_idx = {
        name: vertex_name_to_index(name, n_regions, None)
        for name in graph_tensors["vertex_names"]
    }

    node_arr, edge_arr, path_mask = build_path_tensors(candidates, graph_tensors["edge_lookup"], name_to_idx)
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
            region_indices = path_edges_to_region_indices(path_edges, len(regions))
            if not trajectory_stays_in_path_regions(traj, regions, region_indices):
                continue
            duration, _ = trajectory_stats(traj)
            return PlanResult(
                success=True,
                trajectory=traj,
                cost=float(res.get_optimal_cost()),
                duration_s=duration,
                paths_tried=tried,
                elapsed_s=time.perf_counter() - t0,
                region_indices=region_indices,
            )

    return PlanResult(False, None, float("nan"), float("nan"), tried, time.perf_counter() - t0)


def trace_vertices(traj, n_samples: int = 400) -> np.ndarray:
    ts = np.linspace(traj.start_time(), traj.end_time(), n_samples)
    return np.array([np.asarray(traj.value(t)).reshape(3) for t in ts]).T


def _png_dimensions(png_bytes: bytes) -> tuple[int, int]:
    if len(png_bytes) < 24 or png_bytes[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    return struct.unpack(">II", png_bytes[16:24])


def _render_label_png(text: str, color_hex: str) -> tuple[bytes, float]:
    fig, ax = plt.subplots(figsize=(4, 1.5), dpi=150)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=LABEL_FONT_SIZE,
        fontweight="bold",
        color=color_hex,
        family="sans-serif",
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    png_bytes = buf.getvalue()
    width, height = _png_dimensions(png_bytes)
    return png_bytes, width / height


def _write_text_label_gltf(text: str, color_hex: str, assets_dir: Path, base_name: str, quad_height: float = 0.6):
    png_bytes, aspect = _render_label_png(text, color_hex)
    width = quad_height * aspect
    half_w, half_h = width / 2, quad_height / 2

    positions = [-half_w, 0, -half_h, half_w, 0, -half_h, half_w, 0, half_h, -half_w, 0, half_h]
    uvs = [1, 1, 0, 1, 0, 0, 1, 0]
    normals = [0, 1, 0] * 4
    indices = [0, 1, 2, 0, 2, 3]

    pos_b = struct.pack("<12f", *positions)
    uv_b = struct.pack("<8f", *uvs)
    norm_b = struct.pack("<12f", *normals)
    idx_b = struct.pack("<6H", *indices)
    buf = pos_b + uv_b + norm_b + idx_b + b"\x00" * ((-len(idx_b)) % 4)

    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "TEXCOORD_0": 1, "NORMAL": 2},
                        "indices": 3,
                        "material": 0,
                    }
                ]
            }
        ],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
                "doubleSided": True,
            }
        ],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{}],
        "images": [
            {
                "uri": "data:image/png;base64," + base64.b64encode(png_bytes).decode(),
                "mimeType": "image/png",
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 4,
                "type": "VEC3",
                "min": [-half_w, 0, -half_h],
                "max": [half_w, 0, half_h],
            },
            {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC2"},
            {"bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 3, "componentType": 5123, "count": 6, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 48},
            {"buffer": 0, "byteOffset": 48, "byteLength": 32},
            {"buffer": 0, "byteOffset": 80, "byteLength": 48},
            {"buffer": 0, "byteOffset": 128, "byteLength": 12},
        ],
        "buffers": [
            {
                "byteLength": len(buf),
                "uri": "data:application/octet-stream;base64," + base64.b64encode(buf).decode(),
            }
        ],
    }

    gltf_path = assets_dir / f"{base_name}.gltf"
    gltf_path.write_text(json.dumps(gltf))
    return gltf_path, width


def draw_start_goal_markers(
    meshcat,
    start_pose,
    goal_pose,
    assets_dir: Path,
    *,
    label_offsets=None,
    trace_prefix: str = "/drake/trace",
) -> None:
    pole_h = 1.2
    label_lift = 0.5
    for name, pos, rgba, color_hex in (
        ("START", np.asarray(start_pose), Rgba(0.1, 0.9, 0.2, 1.0), "#0e8a26"),
        ("GOAL", np.asarray(goal_pose), Rgba(0.95, 0.15, 0.15, 1.0), "#c81616"),
    ):
        root = f"{trace_prefix}/{name}"
        meshcat.SetObject(f"{root}/marker", Sphere(0.14), rgba)
        meshcat.SetTransform(f"{root}/marker", RigidTransform(pos))
        meshcat.SetObject(f"{root}/pole", Cylinder(0.02, pole_h), rgba)
        meshcat.SetTransform(f"{root}/pole", RigidTransform(pos + np.array([0.0, 0.0, pole_h / 2])))
        meshcat.SetObject(f"{root}/flag", Sphere(0.18), rgba)
        meshcat.SetTransform(f"{root}/flag", RigidTransform(pos + np.array([0.0, 0.0, pole_h])))
        gltf_path, _ = _write_text_label_gltf(name, color_hex, assets_dir, name.lower())
        meshcat.SetObject(f"{root}/label", Mesh(str(gltf_path), 1.0))
        extra = np.zeros(3)
        if label_offsets and name in label_offsets:
            extra = np.asarray(label_offsets[name], dtype=float)
        meshcat.SetTransform(
            f"{root}/label",
            RigidTransform(pos + np.array([0.0, 0.0, pole_h + label_lift]) + extra),
        )


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
    intro_hold_s: float,
    playback_speed: float = PLAYBACK_SPEED,
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
    play_traj = DelayedTrajectory(traj, delay=intro_hold_s, speed=playback_speed)
    traj_system = builder.AddSystem(FlatnessInverter(play_traj, animator))
    QuadrotorGeometry.AddToBuilder(builder, traj_system.get_output_port(0), scene_graph)
    diagram = builder.Build()

    meshcat.Delete()
    context = diagram.CreateDefaultContext()
    meshcat_viz.ForcedPublish(meshcat_viz.GetMyContextFromRoot(context))

    assets_dir = Path(tempfile.mkdtemp(prefix="meshcat_labels_"))
    try:
        if region_indices:
            traversed = trajectory_region_indices(traj, regions, region_indices)
            n = draw_path_regions(meshcat, regions, traversed, traj)
            print(f"  drew {n} regions traversed by trajectory (of {len(region_indices)} on graph path)")

        meshcat.SetLine(
            "/drake/trace/path",
            trace_vertices(traj),
            line_width=2.5,
            rgba=Rgba(1.0, 1.0, 1.0, 1.0),
        )
        draw_start_goal_markers(meshcat, start_pose, goal_pose, assets_dir)

        sim = Simulator(diagram)
        sim.set_target_realtime_rate(0.0)
        sim.AdvanceTo(play_traj.end_time() + 0.05)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        publish_meshcat_recording_play_once(meshcat_viz)
        html = patch_meshcat_html_play_once(meshcat.StaticHtml())
        if show_legend:
            html = patch_meshcat_html_legend_overlay(html)
        out_path.write_text(html)
        print(f"Saved Meshcat HTML -> {out_path}")
    finally:
        shutil.rmtree(assets_dir, ignore_errors=True)

"""
Visualize Neural GCS quadrotor trajectories in a large custom environment.

Uses a 6x6 grid (≈30m x 30m) so the drone navigates through many rooms.
Start and goal are placed at diagonally opposite corners of the building.
Start and goal poses are shown as colored spheres in Meshcat.

Output: quadrotor/results/viz/demo_<seed>.html
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import shutil
import struct
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pydrake.examples import QuadrotorGeometry
from pydrake.geometry import Cylinder, Mesh, MeshcatVisualizer, Rgba, Sphere, StartMeshcat
from pydrake.math import BsplineBasis, RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.solvers import MosekSolver
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.trajectories import BsplineTrajectory

logging.getLogger("drake").setLevel(logging.WARNING)

from planning_through_contact.model.hparams import DecoderHParams, EncoderHParams
from planning_through_contact.model.inference import project_flows_qp
from planning_through_contact.model.model import GCSFlowPredictor
from planning_through_contact.model.ranknet import PathRankNet, RankNetConfig
from pydrake.geometry.optimization import GraphOfConvexSetsOptions
from quadrotor.building_generation import compile_sdf, generate_grid_world, MODELS_DIR
from quadrotor.gcs.bezier import BezierTrajectory
from quadrotor.gcs.rounding import randomForwardPathSearch
from quadrotor.helpers import build_bezier_gcs, FlatnessInverter

# 3x3 grid — same as training distribution
GRID_SHAPE = (3, 3)
GRID_START = np.array([-1, -1])
GRID_GOAL  = np.array([2, 1])
SDF_PATH = str(MODELS_DIR / "room_gen" / "building.sdf")
DELTA = 0.3
MAX_PATHS = 10
MAX_ROUNDING_TRIALS = 100


# ---------- model ----------

def load_flow_model(ckpt_path, *, x_dim, g_dim, encoder_hp, decoder_hp, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model_state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    model = GCSFlowPredictor(x_dim=x_dim, g_dim=g_dim, encoder_hp=encoder_hp, decoder_hp=decoder_hp)
    model.load_state_dict(model_state, strict=False)
    return model.eval().to(device)


def load_ranknet(ckpt_path, *, cfg, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    ranker_state = {k.removeprefix("ranker."): v for k, v in state.items() if k.startswith("ranker.")}
    if not ranker_state:
        ranker_state = state
    ranker = PathRankNet(cfg)
    ranker.load_state_dict(ranker_state, strict=False)
    return ranker.eval().to(device)


def _build_path_tensors(candidate_edge_lists, edge_lookup, name_to_idx):
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
    P = len(path_nodes_list)
    node_arr = torch.full((P, max_len), -1, dtype=torch.long)
    edge_arr = torch.full((P, max_len), -1, dtype=torch.long)
    mask = torch.zeros((P, max_len), dtype=torch.bool)
    for i, (nodes, edges) in enumerate(zip(path_nodes_list, path_edges_list)):
        L = len(nodes)
        node_arr[i, :L] = torch.tensor(nodes, dtype=torch.long)
        edge_arr[i, :L] = torch.tensor(edges, dtype=torch.long)
        mask[i, :L] = True
    return node_arr, edge_arr, mask


# ---------- GCS helpers ----------

def _gcs_options(gcs_obj):
    opts = GraphOfConvexSetsOptions()
    opts.convex_relaxation = True
    opts.max_rounded_paths = 0
    opts.preprocessing = False
    if gcs_obj.solver is not None:
        opts.solver = gcs_obj.solver
    if gcs_obj.options is not None:
        opts.solver_options = gcs_obj.options
    return opts


def solve_relaxation(gcs_obj):
    return gcs_obj.gcs.SolveShortestPath(
        gcs_obj.source, gcs_obj.target, _gcs_options(gcs_obj))


def solve_restriction(gcs_obj, path_edges):
    edge_set = set(path_edges)
    for e in gcs_obj.gcs.Edges():
        e.AddPhiConstraint(e in edge_set)
    result = gcs_obj.gcs.SolveShortestPath(
        gcs_obj.source, gcs_obj.target, _gcs_options(gcs_obj))
    for e in gcs_obj.gcs.Edges():
        e.ClearPhiConstraints()
    return result


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
        ep = np.reshape(result.GetSolution(edge.xv())[:-(order + 1)],
                        (gcs_obj.dimension, order + 1), "F")
        et = result.GetSolution(edge.xv())[-(order + 1):]
        for ii in range(order):
            path_ctrl.append(ep[:, ii:ii+1])
            time_ctrl.append(np.array([[et[ii]]]))

    offset = time_ctrl[0].copy()
    for ii in range(len(time_ctrl)):
        time_ctrl[ii] -= offset

    path_spline = BsplineTrajectory(BsplineBasis(order + 1, knots), path_ctrl)
    time_spline = BsplineTrajectory(BsplineBasis(order + 1, knots), time_ctrl)
    return BezierTrajectory(path_spline, time_spline)


# ---------- start/goal sampling ----------

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


def sample_inside(region, rng):
    lb, ub = box_bounds(region)
    lo, hi = lb + DELTA, ub - DELTA
    if np.any(lo >= hi):
        return None
    return rng.uniform(lo, hi)


def pick_corner_regions(regions, rng, diagonal="bl_to_tr"):
    """
    Pick start/goal from opposite corners of the building.
    diagonal: 'bl_to_tr' (bottom-left → top-right) or 'br_to_tl' (bottom-right → top-left)
    """
    # First 4 regions are always outdoor corridors — skip them
    finite = []
    for i, r in enumerate(regions[4:], start=4):
        lb, ub = box_bounds(r)
        if not np.any(np.isinf(lb)) and not np.any(np.isinf(ub)):
            finite.append((i, lb, ub))

    if not finite:
        return None

    centers = np.array([(lb + ub) / 2 for _, lb, ub in finite])

    if diagonal == "bl_to_tr":
        scores = centers[:, 0] + centers[:, 1]
        start_reg_idx = finite[int(np.argmin(scores))][0]
        goal_reg_idx  = finite[int(np.argmax(scores))][0]
    elif diagonal == "br_to_tl":
        # Start at top-left, goal at bottom-right (reversed from the label).
        scores = centers[:, 0] - centers[:, 1]
        start_reg_idx = finite[int(np.argmin(scores))][0]
        goal_reg_idx  = finite[int(np.argmax(scores))][0]
    else:  # high_to_low — far apart XY, start near ceiling, goal near floor
        scores = centers[:, 0] + centers[:, 1]
        start_reg_idx = finite[int(np.argmax(scores))][0]
        goal_reg_idx  = finite[int(np.argmin(scores))][0]

    start_lb, start_ub = box_bounds(regions[start_reg_idx])
    goal_lb,  goal_ub  = box_bounds(regions[goal_reg_idx])

    start_pt = sample_inside(regions[start_reg_idx], rng)
    goal_pt  = sample_inside(regions[goal_reg_idx],  rng)
    if start_pt is None or goal_pt is None:
        return None

    if diagonal == "high_to_low":
        # Start near ceiling, goal near floor
        start_pt[2] = np.clip(start_ub[2] - DELTA - 0.1, start_lb[2] + DELTA, start_ub[2] - DELTA)
        goal_pt[2]  = np.clip(start_lb[2] + DELTA + 0.1, goal_lb[2]  + DELTA, goal_ub[2]  - DELTA)
    else:
        # Nudge start: left and back
        start_pt = np.clip(start_pt + np.array([-0.8, -0.6, -0.2]),
                           start_lb + DELTA, start_ub - DELTA)
        # Nudge goal: further and up
        direction = goal_pt - start_pt
        direction /= np.linalg.norm(direction) + 1e-6
        goal_pt = np.clip(goal_pt + direction * 0.5 + np.array([0.0, 0.0, 0.7]),
                          goal_lb + DELTA, goal_ub - DELTA)

    return start_pt, goal_pt


# ---------- planning ----------

def plan(regions, start_pose, goal_pose, flow_model, solver, device, ranker=None):
    """Run Neural GCS and return a BezierTrajectory, or None.

    If `ranker` is given, candidate paths are scored with RankNet and tried in
    descending score order; the first successful restriction is returned (mirrors
    the "GNN + RankNet" benchmark row). Otherwise all candidates are evaluated
    and the lowest-cost trajectory is returned ("GNN only").
    """
    gcs_obj = build_bezier_gcs(regions, solver)
    try:
        gcs_obj.addSourceTarget(start_pose, goal_pose, zero_deriv_boundary=3)
    except ValueError:
        return None

    relaxed = solve_relaxation(gcs_obj)
    if not relaxed.is_success():
        return None

    # Build name→index mapping from the GCS graph
    all_edges = list(gcs_obj.gcs.Edges())
    edge_u_names = [e.u().name() for e in all_edges]
    edge_v_names = [e.v().name() for e in all_edges]
    all_vertices = gcs_obj.gcs.Vertices()
    vertex_names = [v.name() for v in all_vertices]
    source_idx = vertex_names.index("source")
    target_idx = vertex_names.index("target")

    name_to_idx: dict[str, int] = {"source": source_idx, "target": target_idx}
    region_counter = 0
    for name in dict.fromkeys(edge_u_names + edge_v_names):
        if name in name_to_idx:
            continue
        while region_counter in (source_idx, target_idx):
            region_counter += 1
        name_to_idx[name] = region_counter
        region_counter += 1

    edge_lookup = {(u, v): i for i, (u, v) in enumerate(zip(edge_u_names, edge_v_names))}

    N = len(all_vertices)
    x_np = np.zeros((N, 9), dtype=np.float32)
    for i, r in enumerate(regions):
        if i >= N:
            break
        lb, ub = box_bounds(r)
        lb = np.where(np.isinf(lb), -50, lb)
        ub = np.where(np.isinf(ub),  50, ub)
        x_np[i, :6] = np.concatenate([lb, ub])
        x_np[i, 6] = 1.0
    x_np[source_idx, 6] = 0.0
    x_np[source_idx, 7] = 1.0
    x_np[target_idx, 6] = 0.0
    x_np[target_idx, 8] = 1.0

    g_np = np.concatenate([start_pose, goal_pose]).astype(np.float32)
    x_t = torch.tensor(x_np).to(device)
    g_t = torch.tensor(g_np).to(device)
    src_t = torch.tensor([name_to_idx.get(u, 0) for u in edge_u_names], dtype=torch.long)
    dst_t = torch.tensor([name_to_idx.get(v, 0) for v in edge_v_names], dtype=torch.long)
    edge_index_t = torch.stack([src_t, dst_t], dim=0).to(device)

    with torch.no_grad():
        flow_out = flow_model(x=x_t, edge_index=edge_index_t, g=g_t, batch=None)
        phi_hat = torch.sigmoid(flow_out.edge_logits).detach().cpu()

    phi_proj = project_flows_qp(
        edge_index=edge_index_t.cpu(), phi_hat=phi_hat,
        num_nodes=N, source_idx=source_idx, target_idx=target_idx,
    )

    candidates = randomForwardPathSearch(
        gcs_obj.gcs, relaxed, gcs_obj.source, gcs_obj.target,
        max_paths=MAX_PATHS, max_trials=MAX_ROUNDING_TRIALS, seed=0,
    )
    if not candidates:
        return None

    if ranker is not None:
        node_arr, edge_arr, path_mask = _build_path_tensors(candidates, edge_lookup, name_to_idx)
        if node_arr is None:
            return None
        with torch.no_grad():
            scores = ranker(
                node_embeddings=flow_out.node_embeddings.to(device),
                edge_flows=phi_proj.to(device),
                path_node_indices=node_arr,
                path_edge_indices=edge_arr,
                path_mask=path_mask,
            )
        ranked_indices = torch.argsort(scores, descending=True).cpu().tolist()
        for ri in ranked_indices:
            path_edges = candidates[ri]
            if path_edges is None:
                continue
            res = solve_restriction(gcs_obj, path_edges)
            if res.is_success():
                return extract_trajectory(gcs_obj, path_edges, res)
        return None

    best_traj, best_cost = None, float("inf")
    for path_edges in candidates:
        if path_edges is None:
            continue
        res = solve_restriction(gcs_obj, path_edges)
        if not res.is_success():
            continue
        cost = float(res.get_optimal_cost())
        if cost < best_cost:
            best_traj = extract_trajectory(gcs_obj, path_edges, res)
            best_cost = cost

    return best_traj


# ---------- meshcat rendering ----------


def _render_label_png(text: str, color_hex: str) -> bytes:
    """Render `text` to an in-memory PNG (bold black-bordered text on white)."""
    fig, ax = plt.subplots(figsize=(6, 2), dpi=150)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=110, fontweight="bold", color=color_hex, family="sans-serif")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _write_text_label_gltf(text: str, color_hex: str, assets_dir: Path,
                           base_name: str, quad_height: float = 0.6):
    """Render `text` to a PNG and emit a self-contained glTF with the texture
    embedded inline (data URI). Drake's Meshcat reliably reads textures from
    glTF; OBJ + MTL + PNG paths are not embedded in StaticHtml output.

    The quad lies in the X-Z plane with normal +Y, so it reads correctly from
    the default chase-cam viewing direction.
    """
    png_bytes = _render_label_png(text, color_hex)

    aspect = 3.0
    width = quad_height * aspect
    half_w, half_h = width / 2, quad_height / 2

    # Vertex data: 4 vertices, positions + UVs + normals, then triangle indices.
    positions = [-half_w, 0, -half_h,
                  half_w, 0, -half_h,
                  half_w, 0,  half_h,
                 -half_w, 0,  half_h]
    # World +X appears on the viewer's LEFT when looking down -Y (right-handed
    # coordinates), so the U axis must be mirrored relative to vertex X.
    # PNG: U=0 is left of texture, V=0 is top; vertices are v0=bot-right of view,
    # v1=bot-left, v2=top-left, v3=top-right.
    uvs       = [1, 1,  0, 1,  0, 0,  1, 0]
    normals   = [0, 1, 0] * 4
    indices   = [0, 1, 2, 0, 2, 3]

    pos_b   = struct.pack("<12f", *positions)
    uv_b    = struct.pack("<8f", *uvs)
    norm_b  = struct.pack("<12f", *normals)
    idx_b   = struct.pack("<6H", *indices)
    pad     = (-len(idx_b)) % 4  # align next section / total to 4 bytes

    buf = pos_b + uv_b + norm_b + idx_b + b"\x00" * pad
    off_pos, off_uv, off_norm, off_idx = 0, 48, 80, 128

    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "TEXCOORD_0": 1, "NORMAL": 2},
            "indices": 3,
            "material": 0,
        }]}],
        "materials": [{
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "doubleSided": True,
        }],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{}],
        "images": [{
            "uri": "data:image/png;base64," + base64.b64encode(png_bytes).decode(),
            "mimeType": "image/png",
        }],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
             "min": [-half_w, 0, -half_h], "max": [half_w, 0, half_h]},
            {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC2"},
            {"bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 3, "componentType": 5123, "count": 6, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": off_pos,  "byteLength": 48},
            {"buffer": 0, "byteOffset": off_uv,   "byteLength": 32},
            {"buffer": 0, "byteOffset": off_norm, "byteLength": 48},
            {"buffer": 0, "byteOffset": off_idx,  "byteLength": 12},
        ],
        "buffers": [{
            "byteLength": len(buf),
            "uri": "data:application/octet-stream;base64," + base64.b64encode(buf).decode(),
        }],
    }

    gltf_path = assets_dir / f"{base_name}.gltf"
    gltf_path.write_text(json.dumps(gltf))
    return gltf_path, width


def render_html(traj, start_pose, goal_pose, seed, out_path, show_trace=True,
                label_offsets=None):
    meshcat = StartMeshcat()
    meshcat.SetProperty("/Grid", "visible", False)
    meshcat.SetProperty("/Axes", "visible", False)
    meshcat.SetProperty("/Lights/AmbientLight/<object>", "intensity", 0.8)
    meshcat.SetProperty("/Lights/PointLightNegativeX/<object>", "intensity", 0)
    meshcat.SetProperty("/Lights/PointLightPositiveX/<object>", "intensity", 0)

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    Parser(plant, scene_graph).AddModels(SDF_PATH)
    plant.Finalize()

    meshcat_viz = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    animator = meshcat_viz.StartRecording()
    traj_system = builder.AddSystem(FlatnessInverter(traj, animator))
    QuadrotorGeometry.AddToBuilder(builder, traj_system.get_output_port(0), scene_graph)

    diagram = builder.Build()
    meshcat.Delete()

    # Trace: sample the trajectory densely and draw as a polyline + endpoint markers.
    # Parent under /drake so the chase-cam transform applied to /drake each frame
    # (see FlatnessInverter) also translates the trace — keeping it locked to the
    # building rather than drifting through the world.
    # Drake's Meshcat has no native text primitive, so we name the scene-tree
    # entries START / GOAL (visible in the Meshcat sidebar) and stack a
    # ball-on-flagpole marker so each endpoint reads clearly from any angle.
    if show_trace:
        n_samples = 400
        ts = np.linspace(traj.start_time(), traj.end_time(), n_samples)
        vertices = np.array([np.asarray(traj.value(t)).reshape(-1) for t in ts]).T  # (3, N)
        meshcat.SetLine("/drake/trace/path", vertices, line_width=3.0,
                        rgba=Rgba(1.0, 0.85, 0.1, 1.0))

    pole_h = 1.2
    label_lift = 0.5  # how far above the flag the text sits
    assets_dir = Path(tempfile.mkdtemp(prefix="meshcat_labels_"))
    try:
        for name, pos, rgba, color_hex in (
            ("START", np.asarray(start_pose), Rgba(0.1, 0.9, 0.2, 1.0),  "#0e8a26"),
            ("GOAL",  np.asarray(goal_pose),  Rgba(0.95, 0.15, 0.15, 1.0), "#c81616"),
        ):
            root = f"/drake/trace/{name}"
            meshcat.SetObject(f"{root}/marker", Sphere(0.14), rgba)
            meshcat.SetTransform(f"{root}/marker", RigidTransform(pos))
            meshcat.SetObject(f"{root}/pole", Cylinder(0.02, pole_h), rgba)
            meshcat.SetTransform(f"{root}/pole",
                                 RigidTransform(pos + np.array([0.0, 0.0, pole_h / 2])))
            meshcat.SetObject(f"{root}/flag", Sphere(0.18), rgba)
            meshcat.SetTransform(f"{root}/flag",
                                 RigidTransform(pos + np.array([0.0, 0.0, pole_h])))

            gltf_path, _ = _write_text_label_gltf(name, color_hex, assets_dir, name.lower())
            meshcat.SetObject(f"{root}/label", Mesh(str(gltf_path), 1.0))
            extra_offset = np.zeros(3)
            if label_offsets and name in label_offsets:
                extra_offset = np.asarray(label_offsets[name], dtype=float)
            meshcat.SetTransform(
                f"{root}/label",
                RigidTransform(pos + np.array([0.0, 0.0, pole_h + label_lift]) + extra_offset),
            )

        simulator = Simulator(diagram)
        simulator.set_target_realtime_rate(0.0)
        simulator.AdvanceTo(traj.end_time() + 0.05)
        meshcat_viz.PublishRecording()

        html = meshcat.StaticHtml()
        out_path.write_text(html)
        print(f"Saved → {out_path}")
    finally:
        shutil.rmtree(assets_dir, ignore_errors=True)


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow_ckpt", default="quadrotor/checkpoints/quadrotor_gnn/quadrotor_flow_gnn.ckpt")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 4],
                        help="Building seeds to visualize (one HTML per seed).")
    parser.add_argument("--output_dir", default="quadrotor/results/viz")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ffn_hidden_mult", type=int, default=2)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--decoder_hidden", default="256,256")
    parser.add_argument("--decoder_dropout_p", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    # RankNet (optional): enable with --use_ranknet
    parser.add_argument("--use_ranknet", action="store_true",
                        help="Rank candidate paths with RankNet and round only the top-ranked one.")
    parser.add_argument("--ranknet_ckpt",
                        default="quadrotor/checkpoints/quadrotor_ranknet/quadrotor_ranknet.ckpt")
    parser.add_argument("--ranker_layers", type=int, default=3)
    parser.add_argument("--ranker_heads", type=int, default=4)
    parser.add_argument("--ranker_ffn_hidden", type=int, default=256)
    parser.add_argument("--ranker_score_hidden", type=int, default=64)
    parser.add_argument("--ranker_dropout_p", type=float, default=0.1)
    parser.add_argument("--trace", dest="trace", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Draw the yellow trajectory polyline (default: on). "
                             "Use --no-trace to hide it.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Grid: {GRID_SHAPE}  |  Seeds: {args.seeds}")

    encoder_hp = EncoderHParams(
        d_model=args.d_model, num_layers=args.num_layers, num_heads=args.num_heads,
        ffn_hidden_mult=args.ffn_hidden_mult, dropout_p=args.dropout_p,
    )
    hidden = tuple(int(s) for s in args.decoder_hidden.split(",") if s.strip())
    decoder_hp = DecoderHParams(hidden_dims=hidden, dropout_p=args.decoder_dropout_p)

    flow_model = load_flow_model(
        args.flow_ckpt, x_dim=9, g_dim=6,
        encoder_hp=encoder_hp, decoder_hp=decoder_hp, device=device,
    )

    ranker = None
    if args.use_ranknet:
        ranker_cfg = RankNetConfig(
            d_model=args.d_model, num_layers=args.ranker_layers, num_heads=args.ranker_heads,
            ffn_hidden_dim=args.ranker_ffn_hidden, score_hidden_dim=args.ranker_score_hidden,
            dropout_p=args.ranker_dropout_p,
        )
        ranker = load_ranknet(args.ranknet_ckpt, cfg=ranker_cfg, device=device)
        print(f"RankNet loaded from {args.ranknet_ckpt}")
    else:
        print("RankNet disabled (pass --use_ranknet to enable).")

    method_tag = "ranknet" if ranker is not None else "gnn"
    solver = MosekSolver()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        print(f"\n── Seed {seed} ──────────────────────────────")
        rng = np.random.default_rng(seed)

        grid, indoor_edges, outdoor_edges = generate_grid_world(
            shape=GRID_SHAPE, start=GRID_START, goal=GRID_GOAL, seed=seed)
        regions = compile_sdf(
            SDF_PATH, grid, GRID_START, GRID_GOAL, indoor_edges, outdoor_edges, seed=seed)
        print(f"  {len(regions)} convex regions")

        diagonal_map = {0: ("br_to_tl", "bottom_right_to_top_left"),
                        4: ("high_to_low", "high_to_low")}
        diagonal, label = diagonal_map.get(seed, ("br_to_tl", "bottom_right_to_top_left"))
        for diagonal, label in [(diagonal, label)]:
            rng = np.random.default_rng(seed)
            sampled = pick_corner_regions(regions, rng, diagonal=diagonal)
            if sampled is None:
                print(f"  [SKIP] could not sample start/goal for {label}")
                continue
            start_pose, goal_pose = sampled
            dist = float(np.linalg.norm(goal_pose - start_pose))
            print(f"  {label}: start={start_pose.round(2)}  goal={goal_pose.round(2)}  dist={dist:.1f}m")

            method_name = "Neural GCS + RankNet" if ranker is not None else "Neural GCS"
            print(f"  Planning with {method_name}...")
            traj = plan(regions, start_pose, goal_pose, flow_model, solver, device, ranker=ranker)
            if traj is None:
                print(f"  [SKIP] planning failed")
                continue
            print(f"  Trajectory: {traj.start_time():.2f}s → {traj.end_time():.2f}s")

            # Per-seed label nudges in world frame [dx, dy, dz]. Top-down view in
            # Meshcat has +X to the right, so -X moves a label "to the left".
            # Seed 0's GOAL was sitting on top of a window — shove it -X off-axis.
            label_offsets_by_seed = {
                # Seed 0: GOAL sat on a window; START was buried in a wall.
                # Building ceiling is at z=3, so lifting START by +1.0 m puts it
                # above the roof where nothing can occlude it.
                0: {
                    "GOAL":  np.array([-1.2, 0.0, 0.0]),
                    "START": np.array([ 0.0, 0.0, 1.2]),
                },
            }
            out_path = out_dir / f"demo_seed{seed}_{label}_{method_tag}.html"
            print(f"  Rendering Meshcat HTML... (trace: {'on' if args.trace else 'off'})")
            render_html(traj, start_pose, goal_pose, seed, out_path,
                        show_trace=args.trace,
                        label_offsets=label_offsets_by_seed.get(seed))


if __name__ == "__main__":
    main()

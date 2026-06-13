"""
Neural GCS (GNN + RankNet) trajectory sweep over sensing uncertainty (sigma).

For each sigma in SIGMAS:
  - Inflate wall_offset by Phi^{-1}(1-delta)*sigma
  - Build free-space regions
  - Solve with Neural GCS + RankNet
  - Draw trajectory as a colored polyline

All trajectories + nominal baseline overlaid in one Meshcat HTML.

Usage:
    python scripts/neural_gcs_sigma_sweep.py --seed 0 --delta 0.05
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from scipy.stats import norm

from pydrake.geometry import Mesh, MeshcatVisualizer, Rgba, Sphere, StartMeshcat
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.solvers import MosekSolver
from pydrake.systems.framework import DiagramBuilder
from pydrake.examples import QuadrotorGeometry

sys.path.insert(0, str(Path(__file__).parent))
from visualize_chance_constraint import (
    build_regions, draw_region, draw_graph,
    _box_from_hpoly, NOMINAL_WALL_OFFSET,
    GRID_SHAPE, GRID_START, GRID_GOAL, SDF_PATH,
)
from visualize_quadrotor import load_flow_model, load_ranknet, plan
from quadrotor.building_generation import compile_sdf, generate_grid_world
from quadrotor.helpers import FlatnessInverter
from planning_through_contact.model.hparams import DecoderHParams, EncoderHParams
from planning_through_contact.model.ranknet import RankNetConfig

FLOW_CKPT    = "quadrotor/checkpoints/quadrotor_gnn/quadrotor_flow_gnn.ckpt"
RANKNET_CKPT = "quadrotor/checkpoints/quadrotor_ranknet/quadrotor_ranknet.ckpt"

SIGMAS      = [0.0, 0.05, 0.15, 0.30]
TRAJ_COLORS = [
    Rgba(0.2,  0.8,  1.0,  1.0),   # cyan   — nominal
    Rgba(0.4,  1.0,  0.2,  1.0),   # green
    Rgba(1.0,  0.85, 0.05, 1.0),   # yellow
    Rgba(1.0,  0.25, 0.25, 1.0),   # red
]

SAMPLE_MARGIN = 0.35


def _rgba_to_mpl(rgba: Rgba):
    return (rgba.r(), rgba.g(), rgba.b(), rgba.a())


def write_legend_gltf(entries, assets_dir, quad_w=5.0, quad_h=3.0):
    """Render a legend PNG (colored swatches + labels) and embed it in a glTF quad.
    entries: list of (label_str, Rgba)
    Returns path to the .gltf file.
    """
    fig, ax = plt.subplots(figsize=(quad_w, quad_h), dpi=120)
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_axis_off()
    handles = [
        mpatches.Patch(color=_rgba_to_mpl(rgba)[:3], label=label)
        for label, rgba in entries
    ]
    legend = ax.legend(
        handles=handles, loc="center", fontsize=14, framealpha=0.85,
        facecolor="#1a1a1a", edgecolor="white", labelcolor="white",
        handlelength=2.5, handleheight=1.4,
    )
    for text in legend.get_texts():
        text.set_color("white")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    png_bytes = buf.getvalue()

    # glTF quad in the X-Z plane (+Y normal), sized quad_w x quad_h metres.
    hw, hh = quad_w / 2.0, quad_h / 2.0
    positions = [-hw, 0, -hh,  hw, 0, -hh,  hw, 0, hh,  -hw, 0, hh]
    uvs       = [1, 1,  0, 1,  0, 0,  1, 0]
    normals   = [0, 1, 0] * 4
    indices   = [0, 1, 2, 0, 2, 3]

    pos_b  = struct.pack("<12f", *positions)
    uv_b   = struct.pack("<8f",  *uvs)
    norm_b = struct.pack("<12f", *normals)
    idx_b  = struct.pack("<6H",  *indices)
    pad    = (-len(idx_b)) % 4
    buf_bin = pos_b + uv_b + norm_b + idx_b + b"\x00" * pad

    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1, "NORMAL": 2},
                                    "indices": 3, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {
                            "baseColorTexture": {"index": 0},
                            "metallicFactor": 0.0, "roughnessFactor": 1.0},
                       "alphaMode": "BLEND", "doubleSided": True}],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{}],
        "images": [{"uri": "data:image/png;base64," + base64.b64encode(png_bytes).decode(),
                    "mimeType": "image/png"}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
             "min": [-hw, 0, -hh], "max": [hw, 0, hh]},
            {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC2"},
            {"bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 3, "componentType": 5123, "count": 6, "type": "SCALAR"}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0,  "byteLength": 48},
            {"buffer": 0, "byteOffset": 48, "byteLength": 32},
            {"buffer": 0, "byteOffset": 80, "byteLength": 48},
            {"buffer": 0, "byteOffset": 128,"byteLength": 12}],
        "buffers": [{"byteLength": len(buf_bin),
                     "uri": "data:application/octet-stream;base64,"
                            + base64.b64encode(buf_bin).decode()}],
    }
    gltf_path = assets_dir / "legend.gltf"
    gltf_path.write_text(json.dumps(gltf))
    return gltf_path


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

    sc, ss = finite[int(np.argmin(scores))]
    gc, gs = finite[int(np.argmax(scores))]

    start = rng.uniform(sc - ss / 2 + margin, sc + ss / 2 - margin)
    goal  = rng.uniform(gc - gs / 2 + margin, gc + gs / 2 - margin)
    start[2] = sc[2] - ss[2] / 2 + margin + 0.1
    goal[2]  = gc[2] + gs[2] / 2 - margin - 0.1
    return start, goal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",            type=int,   default=0)
    ap.add_argument("--delta",           type=float, default=0.05)
    ap.add_argument("--flow_ckpt",       type=str,   default=FLOW_CKPT)
    ap.add_argument("--ranknet_ckpt",    type=str,   default=RANKNET_CKPT)
    ap.add_argument("--device",          type=str,   default="cuda")
    ap.add_argument("--d_model",         type=int,   default=128)
    ap.add_argument("--num_layers",      type=int,   default=4)
    ap.add_argument("--num_heads",       type=int,   default=4)
    ap.add_argument("--ffn_hidden_mult", type=int,   default=2)
    ap.add_argument("--dropout_p",       type=float, default=0.1)
    ap.add_argument("--decoder_hidden",  type=str,   default="256,256")
    ap.add_argument("--outdir",          type=str,   default="quadrotor/results/viz")
    args = ap.parse_args()

    assert 0.0 < args.delta < 0.5
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  delta={args.delta}  |  sigmas={SIGMAS}")

    encoder_hp = EncoderHParams(
        d_model=args.d_model, num_layers=args.num_layers,
        num_heads=args.num_heads, ffn_hidden_mult=args.ffn_hidden_mult,
        dropout_p=args.dropout_p)
    hidden     = tuple(int(s) for s in args.decoder_hidden.split(",") if s.strip())
    decoder_hp = DecoderHParams(hidden_dims=hidden, dropout_p=0.1)

    print("Loading GNN flow model...")
    flow_model = load_flow_model(
        args.flow_ckpt, x_dim=9, g_dim=6,
        encoder_hp=encoder_hp, decoder_hp=decoder_hp, device=device)

    print("Loading RankNet...")
    ranker_cfg = RankNetConfig(
        d_model=args.d_model, num_layers=3, num_heads=4,
        ffn_hidden_dim=256, score_hidden_dim=64, dropout_p=0.1)
    ranker = load_ranknet(args.ranknet_ckpt, cfg=ranker_cfg, device=device)

    print(f"\nGenerating building (seed={args.seed})...")
    grid, indoor_edges, outdoor_edges = generate_grid_world(
        shape=GRID_SHAPE, start=GRID_START, goal=GRID_GOAL, seed=args.seed)
    compile_sdf(SDF_PATH, grid, GRID_START, GRID_GOAL,
                indoor_edges, outdoor_edges, seed=args.seed)

    regions_nom = build_regions(grid, GRID_START, NOMINAL_WALL_OFFSET, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    start_pose, goal_pose = sample_start_goal(regions_nom, rng)
    print(f"Start: {start_pose.round(3)}  Goal: {goal_pose.round(3)}")

    solver = MosekSolver()

    results = []
    for sigma, color in zip(SIGMAS, TRAJ_COLORS):
        margin      = float(norm.ppf(1.0 - args.delta)) * sigma if sigma > 0 else 0.0
        wall_offset = NOMINAL_WALL_OFFSET + margin
        label       = f"sigma={sigma}" + (" (nominal)" if sigma == 0
                                           else f"  margin={margin:.3f}m")
        print(f"\n[{label}]  wall_offset={wall_offset:.4f}")

        regions = build_regions(grid, GRID_START, wall_offset, seed=args.seed)
        traj = plan(regions, start_pose, goal_pose,
                    flow_model, solver, device, ranker=ranker)
        if traj is None:
            print(f"  Neural GCS failed.")
            continue
        print(f"  Trajectory: {traj.start_time():.2f}s -> {traj.end_time():.2f}s")
        results.append((traj, regions, color, label, sigma, margin))

    if not results:
        print("All solves failed.")
        return

    print("\nRendering combined HTML...")
    meshcat = StartMeshcat()
    meshcat.SetProperty("/Grid",  "visible", True)
    meshcat.SetProperty("/Axes",  "visible", True)
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

    for k, r in enumerate(regions_nom):
        draw_region(meshcat, f"/regions/r{k}", r,
                    fill_rgba=Rgba(0.6, 0.6, 0.6, 0.08),
                    edge_rgba=Rgba(0.6, 0.6, 0.6, 0.45),
                    edge_width=1.0)

    for name, pos, rgba in (
        ("start", start_pose, Rgba(0.1, 0.9, 0.2, 1.0)),
        ("goal",  goal_pose,  Rgba(0.9, 0.1, 0.1, 1.0)),
    ):
        meshcat.SetObject(f"/markers/{name}", Sphere(0.22), rgba)
        meshcat.SetTransform(f"/markers/{name}", RigidTransform(pos))

    n_samples = 400
    for traj, regions, color, label, sigma, margin in results:
        ts       = np.linspace(traj.start_time(), traj.end_time(), n_samples)
        vertices = np.array([np.asarray(traj.value(t)).reshape(-1) for t in ts]).T
        tag      = f"sigma{sigma:.2f}".replace(".", "p")
        meshcat.SetLine(f"/trajectories/{tag}", vertices,
                        line_width=4.0, rgba=color)

    # Legend — glTF quad placed above the start marker, facing +Y.
    assets_dir = Path(tempfile.mkdtemp(prefix="legend_"))
    try:
        legend_entries = [
            (f"sigma={s:.2f}  margin={float(norm.ppf(1-args.delta))*s:.3f}m"
             if s > 0 else f"sigma={s:.2f}  (nominal)", c)
            for s, c in zip(SIGMAS, TRAJ_COLORS)
            if any(r[4] == s for r in results)
        ]
        gltf_path = write_legend_gltf(legend_entries, assets_dir,
                                      quad_w=6.0, quad_h=3.5)
        legend_pos = start_pose + np.array([0.0, 0.0, 4.5])
        meshcat.SetObject("/legend", Mesh(str(gltf_path), 1.0))
        meshcat.SetTransform("/legend", RigidTransform(legend_pos))

        out_dir = Path(args.outdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"neural_gcs_sigma_sweep_seed{args.seed}.html"
        out_path.write_text(meshcat.StaticHtml())
        print(f"\nSaved -> {out_path}")
        print("Trajectories:")
        for _, _, _, label, sigma, margin in results:
            print(f"  sigma={sigma:.2f}  margin={margin:.3f}m")
    finally:
        shutil.rmtree(assets_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

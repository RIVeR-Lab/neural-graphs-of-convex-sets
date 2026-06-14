"""
Overlay the four chance-constrained free-space region sets (one per sigma)
in a single Meshcat scene — no trajectories, just the convex sets.

Each sigma gets its own color; regions are translucent so overlaps are visible.
The building walls are rendered once from the SDF.

Usage:
    python scripts/cc_regions_viz.py --seed 4 --delta 0.1
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
from scipy.stats import norm

from pydrake.geometry import Mesh, MeshcatVisualizer, Rgba, StartMeshcat
from pydrake.geometry.optimization import HPolyhedron
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder

sys.path.insert(0, str(Path(__file__).parent))
from visualize_quadrotor import GRID_SHAPE, GRID_START, GRID_GOAL, SDF_PATH
from visualize_chance_constraint import draw_region
from quadrotor.building_generation import compile_sdf, generate_grid_world

SIGMAS = [0.0, 0.15, 0.2, 0.25]

# Fill and edge colors per sigma — increasingly warm
REGION_STYLES = [
    (Rgba(0.2,  0.8,  1.0,  0.08), Rgba(0.2,  0.8,  1.0,  0.7)),  # cyan
    (Rgba(0.4,  1.0,  0.2,  0.08), Rgba(0.4,  1.0,  0.2,  0.7)),  # green
    (Rgba(1.0,  0.85, 0.05, 0.08), Rgba(1.0,  0.85, 0.05, 0.7)),  # yellow
    (Rgba(1.0,  0.25, 0.25, 0.08), Rgba(1.0,  0.25, 0.25, 0.7)),  # red
]


def inflate_regions(regions: list, margin: float) -> list[HPolyhedron]:
    all_faces: set[tuple] = set()
    for r in regions:
        A, b = r.A(), r.b()
        for i in range(A.shape[0]):
            all_faces.add(tuple(np.round(A[i], 4)) + (round(float(b[i]), 3),))
    inflated = []
    for r in regions:
        A, b = r.A(), r.b()
        b_new = b.copy()
        for i in range(A.shape[0]):
            opp_key = tuple(np.round(-A[i], 4)) + (round(float(-b[i]), 3),)
            if opp_key not in all_faces:
                b_new[i] -= margin * np.linalg.norm(A[i])
        inflated.append(HPolyhedron(A, b_new))
    return inflated


def _rgba_to_mpl(rgba: Rgba):
    return (rgba.r(), rgba.g(), rgba.b(), rgba.a())


def write_legend_gltf(entries, assets_dir, quad_w=5.5, quad_h=3.2):
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

    hw, hh = quad_w / 2.0, quad_h / 2.0
    positions = [-hw, 0, -hh,  hw, 0, -hh,  hw, 0, hh,  -hw, 0, hh]
    uvs       = [1, 1,  0, 1,  0, 0,  1, 0]
    normals   = [0, 1, 0] * 4
    indices   = [0, 1, 2, 0, 2, 3]
    pos_b  = struct.pack("<12f", *positions)
    uv_b   = struct.pack("<8f",  *uvs)
    norm_b = struct.pack("<12f", *normals)
    idx_b  = struct.pack("<6H",  *indices)
    buf_bin = pos_b + uv_b + norm_b + idx_b + b"\x00" * ((-len(idx_b)) % 4)

    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1,
                                                    "NORMAL": 2},
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
            {"buffer": 0, "byteOffset": 0,   "byteLength": 48},
            {"buffer": 0, "byteOffset": 48,  "byteLength": 32},
            {"buffer": 0, "byteOffset": 80,  "byteLength": 48},
            {"buffer": 0, "byteOffset": 128, "byteLength": 12}],
        "buffers": [{"byteLength": len(buf_bin),
                     "uri": "data:application/octet-stream;base64,"
                            + base64.b64encode(buf_bin).decode()}],
    }
    gltf_path = assets_dir / "legend.gltf"
    gltf_path.write_text(json.dumps(gltf))
    return gltf_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",   type=int,   default=4)
    ap.add_argument("--delta",  type=float, default=0.1)
    ap.add_argument("--outdir", type=str,   default="quadrotor/results/viz")
    args = ap.parse_args()

    assert 0.0 < args.delta < 0.5
    z = float(norm.ppf(1.0 - args.delta))

    seed = args.seed
    print(f"seed={seed}  delta={args.delta}  sigmas={SIGMAS}")

    grid, indoor_edges, outdoor_edges = generate_grid_world(
        shape=GRID_SHAPE, start=GRID_START, goal=GRID_GOAL, seed=seed)
    regions_nom = compile_sdf(
        SDF_PATH, grid, GRID_START, GRID_GOAL,
        indoor_edges, outdoor_edges, seed=seed)
    print(f"  {len(regions_nom)} nominal regions")

    # Build all region sets
    all_region_sets = []
    for sigma in SIGMAS:
        margin = z * sigma if sigma > 0 else 0.0
        regions = inflate_regions(regions_nom, margin) if sigma > 0 else regions_nom
        label = (f"σ=0.00 (nominal)" if sigma == 0
                 else f"σ={sigma:.2f}  margin={margin:.3f}m")
        all_region_sets.append((sigma, margin, label, regions))
        print(f"  {label}: {len(regions)} regions")

    # Render
    meshcat = StartMeshcat()
    meshcat.SetProperty("/Grid",  "visible", True)
    meshcat.SetProperty("/Axes",  "visible", True)
    meshcat.SetProperty("/Lights/AmbientLight/<object>", "intensity", 0.8)
    meshcat.SetProperty("/Lights/PointLightNegativeX/<object>", "intensity", 0)
    meshcat.SetProperty("/Lights/PointLightPositiveX/<object>", "intensity", 0)

    # Building walls
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    Parser(plant, scene_graph).AddModels(SDF_PATH)
    plant.Finalize()
    meshcat_viz = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    ctx = diagram.CreateDefaultContext()
    meshcat.Delete()
    meshcat_viz.ForcedPublish(meshcat_viz.GetMyContextFromRoot(ctx))

    # Draw each sigma's regions in its own namespace
    for (sigma, margin, label, regions), (fill, edge) in zip(all_region_sets, REGION_STYLES):
        ns = f"/regions/sigma{sigma:.2f}".replace(".", "p")
        for k, r in enumerate(regions):
            draw_region(meshcat, f"{ns}/r{k}", r,
                        fill_rgba=fill, edge_rgba=edge, edge_width=1.5)
        print(f"  drew {len(regions)} regions for {label}")

    # Legend
    assets_dir = Path(tempfile.mkdtemp(prefix="cc_regions_legend_"))
    try:
        legend_entries = [
            (label, edge)
            for (sigma, margin, label, regions), (fill, edge) in zip(all_region_sets, REGION_STYLES)
        ]
        gltf_path = write_legend_gltf(legend_entries, assets_dir)
        meshcat.SetObject("/legend", Mesh(str(gltf_path), 1.0))
        meshcat.SetTransform("/legend", RigidTransform(np.array([10.0, 10.0, 7.0])))

        out_dir = Path(args.outdir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"cc_regions_seed{seed}.html"
        out_path.write_text(meshcat.StaticHtml())
        print(f"\nSaved -> {out_path}")
    finally:
        shutil.rmtree(assets_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

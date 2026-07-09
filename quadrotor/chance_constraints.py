"""Chance-constraint helpers: seam-preserving region tightening and Meshcat legend."""

from __future__ import annotations

import base64
import io
import json
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from pydrake.geometry import Rgba
from pydrake.geometry.optimization import HPolyhedron


def inflate_regions(regions: list, margin: float) -> list[HPolyhedron]:
    """Shrink only obstacle-facing faces; preserve open-air seam faces."""
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


def write_legend_gltf(entries, assets_dir, quad_w=5.0, quad_h=3.0):
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
    gltf_path = Path(assets_dir) / "legend.gltf"
    gltf_path.write_text(json.dumps(gltf))
    return gltf_path

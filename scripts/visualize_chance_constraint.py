"""
Visualize the effect of chance-constraint obstacle inflation on GCS free-space
regions for a 3x3 quadrotor building.

The nominal pipeline (compile_sdf) builds free-space boxes using:
    wall_offset = 0.125 + quad_radius   (= 0.325 m)

Inflating every obstacle facet outward by margin = Phi^{-1}(1-Delta)*sigma is
exactly equivalent to increasing wall_offset by that margin:
    wall_offset_inflated = wall_offset + margin

We call compile_sdf twice — once with each wall_offset — to get two sets of
free-space regions, then visualize both with GCS graph edges.

Output (two HTMLs):
    quadrotor/results/viz/chance_constraint_nominal_seed{N}.html
    quadrotor/results/viz/chance_constraint_inflated_seed{N}.html

Usage:
    python scripts/visualize_chance_constraint.py --seed 0 --sigma 0.15 --delta 0.05
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import norm

from pydrake.geometry import Box as DrakeBox, Rgba, Sphere, StartMeshcat
from pydrake.geometry.optimization import HPolyhedron
from pydrake.math import RigidTransform
import lxml.etree as ET

from quadrotor.building_generation import generate_grid_world, MODELS_DIR

GRID_SHAPE = (3, 3)
GRID_START = np.array([-1, -1])
GRID_GOAL  = np.array([ 2,  1])
SDF_PATH   = str(MODELS_DIR / "room_gen" / "building.sdf")

NOMINAL_QUAD_RADIUS = 0.2
NOMINAL_WALL_OFFSET = 0.125 + NOMINAL_QUAD_RADIUS   # 0.325 m


# ---------------------------------------------------------------------------
# compile_sdf variant that accepts a custom wall_offset and returns regions
# without touching the SDF file (we only need the regions, not the SDF for viz)
# ---------------------------------------------------------------------------

def build_regions(grid: np.ndarray, start: np.ndarray, wall_offset: float,
                  seed: int | None = None) -> list[HPolyhedron]:
    """Re-implement the region-building part of compile_sdf with a custom
    wall_offset. Does NOT write any SDF or place any visual geometry."""
    if seed is not None:
        np.random.seed(seed)

    quad_radius = NOMINAL_QUAD_RADIUS
    z_min = quad_radius
    z_max = 3 - quad_radius
    x_cells, y_cells = grid.shape
    tree_probability = 0.7

    regions = [
        HPolyhedron.MakeBox([-2.5, -2.5, z_min],
                            [x_cells * 5 + 7.5, 2.5 - wall_offset, z_max]),
        HPolyhedron.MakeBox([-2.5, 2.5 - wall_offset, z_min],
                            [2.5 - wall_offset, y_cells * 5 + 2.5 + wall_offset, z_max]),
        HPolyhedron.MakeBox([x_cells * 5 + 2.5 + wall_offset, 2.5 - wall_offset, z_min],
                            [x_cells * 5 + 7.5, y_cells * 5 + 2.5 + wall_offset, z_max]),
        HPolyhedron.MakeBox([-2.5, y_cells * 5 + 2.5 + wall_offset, z_min],
                            [x_cells * 5 + 7.5, y_cells * 5 + 7.5, z_max]),
    ]

    for i in range(-1, x_cells + 1):
        for j in range(-1, y_cells + 1):
            xy = (np.array([i, j]) - start) * 5
            if i >= 0 and j >= 0 and i < x_cells and j < y_cells and grid[i, j] > 0.5:
                regions.append(HPolyhedron.MakeBox(
                    [xy[0] - (2.5 - wall_offset), xy[1] - (2.5 - wall_offset), z_min],
                    [xy[0] + (2.5 - wall_offset), xy[1] + (2.5 - wall_offset), z_max]))
            else:
                if i < 0 or j < 0 or i == x_cells or j == y_cells:
                    continue
                lb = [xy[0] - 2.5, xy[1] - 2.5, z_min]
                ub = [xy[0] + 2.5, xy[1] + 2.5, z_max]
                if i == 0:
                    lb[0] -= wall_offset
                if j == 0:
                    lb[1] -= wall_offset
                if i == x_cells - 1:
                    ub[0] += wall_offset
                if j == y_cells - 1:
                    ub[1] += wall_offset
                if i > 0 and j >= 0 and j < y_cells and grid[i - 1, j] > 0.5:
                    lb[0] += wall_offset
                if j > 0 and i >= 0 and i < x_cells and grid[i, j - 1] > 0.5:
                    lb[1] += wall_offset
                if i < x_cells - 1 and j >= 0 and j < y_cells and grid[i + 1, j] > 0.5:
                    ub[0] -= wall_offset
                if j < y_cells - 1 and i >= 0 and i < x_cells and grid[i, j + 1] > 0.5:
                    ub[1] -= wall_offset

                if np.random.random() < 1 - tree_probability:
                    regions.append(HPolyhedron.MakeBox(lb, ub))
                else:
                    tree_pose = xy + 3.0 * np.random.rand(2) - 1.5
                    regions.append(HPolyhedron.MakeBox(lb, [ub[0], tree_pose[1] - 0.5, ub[2]]))
                    regions.append(HPolyhedron.MakeBox([lb[0], tree_pose[1] - 0.5, lb[2]],
                                                       [tree_pose[0] - 0.5, tree_pose[1] + 0.5, ub[2]]))
                    regions.append(HPolyhedron.MakeBox([tree_pose[0] + 0.5, tree_pose[1] - 0.5, lb[2]],
                                                       [ub[0], tree_pose[1] + 0.5, ub[2]]))
                    regions.append(HPolyhedron.MakeBox([lb[0], tree_pose[1] + 0.5, lb[2]], ub))

    # door / window regions from outdoor edges are omitted here — they depend on
    # SDF placement logic and do not directly use wall_offset for sizing.
    return regions


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _box_from_hpoly(region: HPolyhedron):
    A, b = region.A(), region.b()
    lb, ub = np.full(3, -np.inf), np.full(3, np.inf)
    for i in range(A.shape[0]):
        row = A[i]; nz = np.nonzero(row)[0]
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
    signs = [(sx, sy, sz) for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)]
    corners = {s: np.array([cx+s[0]*hx, cy+s[1]*hy, cz+s[2]*hz]) for s in signs}
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
    meshcat.SetLineSegments(f"{path}/edges", ss, es,
                            line_width=edge_width, rgba=edge_rgba)
    return True


def draw_graph(meshcat, regions, path="/graph",
               node_rgba=Rgba(1.0, 0.55, 0.0, 1.0),
               edge_rgba=Rgba(1.0, 0.95, 0.05, 1.0),
               node_r=0.20, edge_width=4.0):
    centers = []
    for k, region in enumerate(regions):
        got = _box_from_hpoly(region)
        c = got[0] if got is not None else None
        centers.append(c)
        if c is not None:
            meshcat.SetObject(f"{path}/nodes/n{k}", Sphere(node_r), node_rgba)
            meshcat.SetTransform(f"{path}/nodes/n{k}", RigidTransform(c))

    n_edges = 0
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            if not regions[i].IntersectsWith(regions[j]):
                continue
            ci, cj = centers[i], centers[j]
            if ci is None or cj is None:
                continue
            meshcat.SetLineSegments(
                f"{path}/edges/e{i}_{j}",
                np.array([ci]).T, np.array([cj]).T,
                line_width=edge_width, rgba=edge_rgba)
            n_edges += 1
    return n_edges


def make_meshcat():
    mc = StartMeshcat()
    mc.SetProperty("/Grid", "visible", True)
    mc.SetProperty("/Axes", "visible", True)
    mc.Delete()
    return mc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",   type=int,   default=0)
    ap.add_argument("--sigma",  type=float, default=0.15,
                    help="Isotropic sensing std dev [m].")
    ap.add_argument("--delta",  type=float, default=0.05,
                    help="Per-facet risk budget in (0, 0.5).")
    ap.add_argument("--outdir", type=str,   default="quadrotor/results/viz")
    args = ap.parse_args()

    assert 0.0 < args.delta < 0.5

    z      = float(norm.ppf(1.0 - args.delta))
    margin = z * args.sigma
    wall_offset_nom  = NOMINAL_WALL_OFFSET
    wall_offset_infl = NOMINAL_WALL_OFFSET + margin

    print(f"sigma={args.sigma}  delta={args.delta}  "
          f"Phi^-1(1-delta)={z:.4f}  margin={margin:.4f} m")
    print(f"wall_offset: nominal={wall_offset_nom:.4f}  "
          f"inflated={wall_offset_infl:.4f}")

    print(f"\nGenerating {GRID_SHAPE[0]}x{GRID_SHAPE[1]} grid (seed={args.seed})...")
    grid, _, _ = generate_grid_world(
        shape=GRID_SHAPE, start=GRID_START, goal=GRID_GOAL, seed=args.seed)

    print("Building nominal free-space regions...")
    regions_nom  = build_regions(grid, GRID_START, wall_offset_nom,  seed=args.seed)
    print("Building inflated free-space regions...")
    regions_infl = build_regions(grid, GRID_START, wall_offset_infl, seed=args.seed)

    # Count drawable (bounded) regions and GCS edges for each
    drawable_nom  = [r for r in regions_nom  if _box_from_hpoly(r) is not None]
    drawable_infl = [r for r in regions_infl if _box_from_hpoly(r) is not None]

    def count_edges(regs):
        return sum(1 for i in range(len(regs))
                   for j in range(i+1, len(regs))
                   if regs[i].IntersectsWith(regs[j]))

    print(f"  Nominal:  {len(drawable_nom)} regions, "
          f"{count_edges(drawable_nom)} edges")
    print(f"  Inflated: {len(drawable_infl)} regions, "
          f"{count_edges(drawable_infl)} edges")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- HTML 1: nominal ---
    print("\nRendering nominal HTML...")
    mc = make_meshcat()
    drawn = sum(draw_region(mc, f"/regions/r{k}", r,
                            fill_rgba=Rgba(0.0, 0.75, 0.95, 0.18),
                            edge_rgba=Rgba(0.0, 0.7,  0.9,  1.0),
                            edge_width=2.0)
                for k, r in enumerate(regions_nom))
    n_e = draw_graph(mc, regions_nom, path="/graph",
                     node_rgba=Rgba(1.0, 0.55, 0.0, 1.0),
                     edge_rgba=Rgba(1.0, 0.95, 0.05, 1.0),
                     node_r=0.22, edge_width=4.0)
    p1 = out_dir / f"chance_constraint_nominal_seed{args.seed}.html"
    p1.write_text(mc.StaticHtml())
    print(f"  {drawn} regions drawn, {n_e} GCS edges  ->  {p1}")

    # --- HTML 2: inflated ---
    print("Rendering inflated HTML...")
    mc = make_meshcat()
    drawn = sum(draw_region(mc, f"/regions/r{k}", r,
                            fill_rgba=Rgba(0.45, 0.95, 0.15, 0.18),
                            edge_rgba=Rgba(0.3,  0.9,  0.05, 1.0),
                            edge_width=2.0)
                for k, r in enumerate(regions_infl))
    n_e = draw_graph(mc, regions_infl, path="/graph",
                     node_rgba=Rgba(1.0, 0.55, 0.0, 1.0),
                     edge_rgba=Rgba(1.0, 0.95, 0.05, 1.0),
                     node_r=0.22, edge_width=4.0)
    p2 = out_dir / f"chance_constraint_inflated_seed{args.seed}.html"
    p2.write_text(mc.StaticHtml())
    print(f"  {drawn} regions drawn, {n_e} GCS edges  ->  {p2}")

    print(f"\nDone.")
    print(f"  Cyan  regions + graph  ->  {p1.name}  (nominal wall_offset={wall_offset_nom:.3f})")
    print(f"  Green regions + graph  ->  {p2.name}  (inflated wall_offset={wall_offset_infl:.3f})")


if __name__ == "__main__":
    main()
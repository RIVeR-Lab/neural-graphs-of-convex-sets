"""
Create static (instance-independent) node features x_v for the planar pushing GCS graph.

Based on `/home/ananya/Downloads/ICRA2027_Ananya.pdf`:
  x_v ∈ R^(11 + 2*NF_max) with blocks:
    - type (2)
    - contact-geometry (5)
    - halfspace normals (4): two unit normals (n1x, n1y, n2x, n2y)
    - transition-context (2*NF_max)

We exclude the dynamically constructed `source` and `target` vertices.

Transition-context convention for ENTRY/EXIT non-contact nodes (NF=4 default):
  - ENTRY_NON_COLL_k: x_src = 0, x_tgt = onehot(k)
  - EXIT_NON_COLL_k:  x_src = onehot(k), x_tgt = 0

Output: a CSV with columns:
  - node_name
  - x_0 ... x_{D-1}   (D = 11 + 2*NF_max)
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from planning_through_contact.experiments.utils import (
    get_default_experiment_plans,
    get_default_plan_config,
)
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.planning.planar.planar_pushing_planner import PlanarPushingPlanner


_RE_FACE_NODE = re.compile(r"^FACE_(\d+)$")
_RE_FACE_TO_FACE_NONCOLL = re.compile(r"^FACE_(\d+)_to_FACE_(\d+)_NON_COLL_(\d+)$")
_RE_ENTRY_NONCOLL = re.compile(r"^ENTRY_NON_COLL_(\d+)$")
_RE_EXIT_NONCOLL = re.compile(r"^EXIT_NON_COLL_(\d+)$")


def _onehot(idx: int, dim: int) -> np.ndarray:
    v = np.zeros((dim,), dtype=np.float32)
    if 0 <= idx < dim:
        v[idx] = 1.0
    return v


def _unit_normal(a: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=np.float64).reshape((2,))
    nrm = float(np.linalg.norm(a))
    if nrm <= 0:
        return 0.0, 0.0
    a = a / nrm
    return float(a[0]), float(a[1])


def _transition_context(name: str, nf_max: int) -> np.ndarray:
    """
    Return x_trans ∈ R^(2*nf_max) for a vertex name.
    """
    m = _RE_FACE_TO_FACE_NONCOLL.match(name)
    if m:
        i = int(m.group(1))
        j = int(m.group(2))
        return np.concatenate([_onehot(i, nf_max), _onehot(j, nf_max)], axis=0)

    m = _RE_ENTRY_NONCOLL.match(name)
    if m:
        k = int(m.group(1))
        return np.concatenate([np.zeros((nf_max,), dtype=np.float32), _onehot(k, nf_max)])

    m = _RE_EXIT_NONCOLL.match(name)
    if m:
        k = int(m.group(1))
        return np.concatenate([_onehot(k, nf_max), np.zeros((nf_max,), dtype=np.float32)])

    # Other nodes (including contact nodes): no transition context
    return np.zeros((2 * nf_max,), dtype=np.float32)


def _is_contact_node(name: str) -> bool:
    return _RE_FACE_NODE.match(name) is not None


def _contact_geom_for_face(slider_geometry, face_idx: int) -> np.ndarray:
    """
    Contact-geometry block (5 dims): [mx, my, L, tx, ty] for the face.
    """
    loc = PolytopeContactLocation(ContactLocation.FACE, face_idx)
    c0, c1 = slider_geometry.get_proximate_vertices_from_location(loc)
    c0 = np.asarray(c0, dtype=np.float64).reshape((2,))
    c1 = np.asarray(c1, dtype=np.float64).reshape((2,))
    m = 0.5 * (c0 + c1)
    L = float(np.linalg.norm(c1 - c0))
    if L <= 0:
        t = np.zeros((2,), dtype=np.float64)
    else:
        t = (c1 - c0) / L
    return np.array([m[0], m[1], L, t[0], t[1]], dtype=np.float32)


def _noncontact_halfspace_normals(slider_geometry, region_idx: int) -> np.ndarray:
    """
    Halfspace-normal block (4 dims): two unit normals (n1x,n1y,n2x,n2y).
    """
    planes = list(slider_geometry.get_planes_for_collision_free_region(region_idx))

    n = np.zeros((2, 2), dtype=np.float32)  # (2 planes, 2 dims)
    for k, plane in enumerate(planes[:2]):
        nx, ny = _unit_normal(plane.a)
        n[k, :] = np.array([nx, ny], dtype=np.float32)

    return n.reshape((-1,))  # (4,)


def _region_idx_from_name(name: str) -> int | None:
    """
    Extract the collision-free region index for non-contact nodes when possible.
    """
    m = _RE_FACE_TO_FACE_NONCOLL.match(name)
    if m:
        return int(m.group(3))
    m = _RE_ENTRY_NONCOLL.match(name)
    if m:
        return int(m.group(1))
    m = _RE_EXIT_NONCOLL.match(name)
    if m:
        return int(m.group(1))
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", type=str, default="sugar_box")
    parser.add_argument("--nf_max", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0, help="Only used to pick a valid dummy plan.")
    parser.add_argument("--pusher_radius", type=float, default=0.015)
    parser.add_argument(
        "--output_path",
        type=str,
        default="planning_through_contact/dataset/data/box_pushing/node_features.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--print_summary",
        action="store_true",
        help="Print a short summary (and optionally the first few rows).",
    )
    parser.add_argument(
        "--print_n",
        type=int,
        default=10,
        help="How many nodes to print when --print_summary is set.",
    )
    args = parser.parse_args()

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config = get_default_plan_config(
        slider_type=args.body, pusher_radius=args.pusher_radius, use_case="normal"
    )

    # Build graph once using any valid start/goal (source/target are excluded anyway).
    dummy_plan = get_default_experiment_plans(args.seed, 1, config)[0]
    config.start_and_goal = dummy_plan
    planner = PlanarPushingPlanner(config)
    planner.formulate_problem()

    # Build a stable list of vertex names, excluding dynamic nodes.
    excluded = {"source", "target"}
    names: list[str] = sorted([v.name() for v in planner.gcs.Vertices() if v.name() not in excluded])

    slider_geometry = config.slider_geometry

    feat_dim = 11 + 2 * args.nf_max
    x = np.zeros((len(names), feat_dim), dtype=np.float32)

    for idx, name in enumerate(names):
        # Block 1: type (2)
        if _is_contact_node(name):
            x[idx, 0:2] = np.array([1.0, 0.0], dtype=np.float32)
        else:
            x[idx, 0:2] = np.array([0.0, 1.0], dtype=np.float32)

        # Block 2: contact geometry (5) at [2:7]
        m_face = _RE_FACE_NODE.match(name)
        if m_face:
            face_idx = int(m_face.group(1))
            x[idx, 2:7] = _contact_geom_for_face(slider_geometry, face_idx)

        # Block 3: halfspace normals (4) at [7:11]
        region_idx = _region_idx_from_name(name)
        if region_idx is not None:
            hs = _noncontact_halfspace_normals(slider_geometry, region_idx)
            x[idx, 7:11] = hs

        # Block 4: transition context (2*NF_max) at [11:]
        x[idx, 11:] = _transition_context(name, args.nf_max)

    # Write CSV.
    fieldnames = ["node_name"] + [f"x_{i}" for i in range(feat_dim)]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for name, row_x in zip(names, x):
            row = {"node_name": name, **{f"x_{i}": float(row_x[i]) for i in range(feat_dim)}}
            w.writerow(row)

    print(f"Wrote node features for {len(names)} nodes to {out_path}")

    if args.print_summary:
        n_show = max(0, min(int(args.print_n), len(names)))
        print(f"body={args.body} nf_max={args.nf_max} feat_dim={feat_dim} nodes={len(names)}")
        if n_show > 0:
            print("First nodes:")
            for i in range(n_show):
                print(f"  {i:03d} {names[i]}")


if __name__ == "__main__":
    main()


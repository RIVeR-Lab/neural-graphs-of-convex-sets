"""
Recreate the graph_edge_labels SVG from the HDF5 dataset logs.

This is a debugging/validation tool: it reads one plan_id from the solutions HDF5
and renders a Graphviz SVG using:
  - edge_u / edge_v
  - y
  - phi_star
  - path_edge_u / path_edge_v / path_edge_order (optional, for traversal order)

Examples:
  # Recreate for a single plan_id
  python3 planning_through_contact/dataset/debug/recreate_svg_from_hdf5.py \
    --h5_path planning_through_contact/dataset/data/box_pushing/solutions_YYYYMMDDHHMMSS.h5 \
    --plan_id 0

  # Recreate for all plan_ids in the file
  python3 planning_through_contact/dataset/debug/recreate_svg_from_hdf5.py \
    --h5_path planning_through_contact/dataset/data/box_pushing/solutions_YYYYMMDDHHMMSS.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _fmt_phi(phi: float) -> str:
    if np.isnan(phi):
        return "?"
    return f"{phi:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=str, required=True)
    parser.add_argument(
        "--plan_id",
        type=int,
        default=None,
        help="If set, recreate only this plan_id. Otherwise recreate all plan_ids in the file.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="planning_through_contact/dataset/debug/out",
        help="Output directory for dot/svg.",
    )
    parser.add_argument(
        "--rankdir",
        type=str,
        default="LR",
        choices=["LR", "TB", "RL", "BT"],
        help="Graphviz rank direction.",
    )
    args = parser.parse_args()

    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Missing `h5py`. Install it and retry.") from e

    try:
        import pydot  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Missing `pydot`. Install it and retry.") from e

    h5_path = Path(args.h5_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _render_one(plan_id: int) -> None:
        with h5py.File(h5_path, "r") as h5:
            grp = h5["samples"][str(plan_id)]

            edge_u = [
                s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
                for s in grp["edge_u"][()]
            ]
            edge_v = [
                s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
                for s in grp["edge_v"][()]
            ]
            y = np.array(grp["y"][()], dtype=np.int32)
            phi = np.array(grp["phi_star"][()], dtype=np.float32)

            # Optional ordered path sequence.
            path_orders: dict[tuple[str, str], list[int]] = {}
            if "path_edge_u" in grp and "path_edge_v" in grp and "path_edge_order" in grp:
                pu = grp["path_edge_u"][()]
                pv = grp["path_edge_v"][()]
                po = np.array(grp["path_edge_order"][()], dtype=np.int32)
                pu = [
                    s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
                    for s in pu
                ]
                pv = [
                    s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
                    for s in pv
                ]
                for u, v, o in zip(pu, pv, po):
                    path_orders.setdefault((u, v), []).append(int(o))

        # Nodes are derived from edge endpoints.
        node_names = sorted(set(edge_u) | set(edge_v))

        dot = pydot.Dot(graph_type="digraph", rankdir=args.rankdir)
        for n in node_names:
            attrs: dict[str, str] = {}
            if n == "source":
                attrs.update({"shape": "doublecircle", "color": "darkgreen"})
            elif n == "target":
                attrs.update({"shape": "doublecircle", "color": "darkblue"})
            dot.add_node(pydot.Node(n, **attrs))

        for u, v, yi, phii in zip(edge_u, edge_v, y, phi):
            if yi == 1:
                orders = path_orders.get((u, v), [])
                order_str = ",".join(str(o) for o in orders) if len(orders) > 0 else "?"
                edge_attrs = {
                    "label": f'<<FONT COLOR="black">1</FONT><FONT COLOR="blue"> {order_str}</FONT><FONT COLOR="gray40">, {_fmt_phi(float(phii))}</FONT>>',
                    "color": "red",
                    "penwidth": "3",
                }
            else:
                edge_attrs = {
                    "label": f'<<FONT COLOR="black">0</FONT><FONT COLOR="gray40">, {_fmt_phi(float(phii))}</FONT>>',
                    "color": "gray70",
                    "penwidth": "1",
                }
            dot.add_edge(pydot.Edge(u, v, **edge_attrs))

        stem = out_dir / f"plan_{plan_id}_graph_edge_labels"
        dot_path = f"{stem}.dot"
        svg_path = f"{stem}.svg"

        dot.write_raw(dot_path)
        try:
            dot.write_svg(svg_path)
        except Exception as e:
            raise RuntimeError(
                "Failed to write SVG. Graphviz may be missing. "
                "Try installing it (e.g. `sudo apt-get install graphviz`)."
            ) from e

        print(f"Wrote {dot_path}")
        print(f"Wrote {svg_path}")

    if args.plan_id is not None:
        _render_one(int(args.plan_id))
        return

    # Otherwise, render all samples in the file.
    with h5py.File(h5_path, "r") as h5:
        plan_ids = sorted(int(k) for k in h5["samples"].keys())
    for pid in plan_ids:
        _render_one(pid)


if __name__ == "__main__":
    main()


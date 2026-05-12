"""
Visualize GCS graph edges and export 0/1 edge labels.

This script:
- builds a planar pushing GCS instance
- selects ONE discrete path (prefers a feasible rounded path if available; otherwise the
  best convex-restriction "binary" path)
- assigns y_e ∈ {0,1} for every directed edge in the full graph
- saves:
  - `edges.json`: edges in a fixed order + labels
  - `graph_edge_labels.dot`: Graphviz DOT file
  - `graph_edge_labels.svg` (best-effort; requires Graphviz installed)

Run from the `planning-through-contact/` directory, inside your venv:

    python scripts/planar_pushing/visualize_graph_edge_labels.py --body sugar_box --seed 0 --traj 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pydot

from planning_through_contact.experiments.utils import (
    create_output_folder,
    get_default_experiment_plans,
    get_default_plan_config,
    get_default_solver_params,
)
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPushingPlanner,
)


def _edge_key(edge) -> tuple[str, str]:
    return (edge.u().name(), edge.v().name())


def _safe_write_svg(graph: pydot.Dot, svg_path: Path) -> None:
    """
    Writes SVG if Graphviz is installed; otherwise silently skips.
    """
    try:
        graph.write_svg(str(svg_path))
    except Exception:
        # Common causes:
        # - `dot` binary missing (graphviz not installed)
        # - pydot cannot invoke graphviz
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--traj",
        type=int,
        default=0,
        help="Index into the generated start/goal list (like create_plans.py).",
    )
    parser.add_argument("--body", type=str, default="sugar_box")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="trajectories",
        help="High-level output directory (a timestamped subfolder is created).",
    )
    parser.add_argument(
        "--no_rounding",
        action="store_true",
        help="If set, do not attempt nonlinear rounding; use the best convex-restriction path.",
    )
    args = parser.parse_args()

    seed: int = args.seed
    traj_idx: int = args.traj
    slider_type: str = args.body
    output_dir: str = args.output_dir

    pusher_radius = 0.015
    config = get_default_plan_config(
        slider_type=slider_type,
        pusher_radius=pusher_radius,
        use_case="normal",
    )
    solver_params = get_default_solver_params(debug=False, clarabel=False)

    # Use the same boundary-condition generator as `create_plans.py`, but only plan one.
    plans = get_default_experiment_plans(seed, traj_idx + 1, config)
    if traj_idx < 0 or traj_idx >= len(plans):
        raise ValueError(f"--traj {traj_idx} is out of range (0..{len(plans)-1}).")
    start_and_goal = plans[traj_idx]

    run_folder = create_output_folder(output_dir, slider_type, traj_idx)
    out_dir = Path(run_folder) / f"traj_{traj_idx}" / "graph_edge_labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    planner = PlanarPushingPlanner(config)
    planner.config.start_and_goal = start_and_goal
    planner.formulate_problem()

    # Prefer a feasible rounded path if possible; otherwise fall back to the best "binary" path.
    chosen_path = None
    chosen_kind = "none"

    binary_paths = planner._plan_paths(solver_params)
    if binary_paths is None or len(binary_paths) == 0:
        chosen_path = None
        chosen_kind = "none"
    else:
        best_binary_path = binary_paths[0]  # already sorted by increasing cost
        chosen_path = best_binary_path
        chosen_kind = "binary"

        if not args.no_rounding:
            feasible_paths = planner._get_rounded_paths(solver_params, binary_paths)
            if feasible_paths is not None and len(feasible_paths) > 0:
                chosen_path = planner._pick_best_path(feasible_paths)
                chosen_kind = "feasible"

    # Build full edge list in a stable order (Drake's internal edge ordering).
    all_edges = list(planner.gcs.Edges())
    all_edge_keys = [_edge_key(e) for e in all_edges]

    path_edge_keys: set[tuple[str, str]] = set()
    path_vertex_names: list[str] = []
    if chosen_path is not None:
        path_edge_keys = {_edge_key(e) for e in chosen_path.edges}
        path_vertex_names = chosen_path.get_path_names()

    y = [1 if key in path_edge_keys else 0 for key in all_edge_keys]

    # Save machine-readable labels.
    payload: dict[str, Any] = {
        "seed": seed,
        "traj_index": traj_idx,
        "slider_type": slider_type,
        "chosen_path_kind": chosen_kind,  # "feasible" | "binary" | "none"
        "num_vertices": len(list(planner.gcs.Vertices())),
        "num_edges": len(all_edges),
        "num_positive_edges": int(sum(y)),
        "path_vertex_names": path_vertex_names,
        "edges": [
            {"u": u, "v": v, "y": int(label)}
            for (u, v), label in zip(all_edge_keys, y)
        ],
    }
    (out_dir / "edges.json").write_text(json.dumps(payload, indent=2))

    # Build a Graphviz graph with highlighted path edges.
    dot = pydot.Dot(graph_type="digraph", rankdir="LR")

    # Nodes
    for v in planner.gcs.Vertices():
        name = v.name()
        attrs: dict[str, str] = {}
        if name == "source":
            attrs.update({"shape": "doublecircle", "color": "darkgreen"})
        elif name == "target":
            attrs.update({"shape": "doublecircle", "color": "darkblue"})
        dot.add_node(pydot.Node(name, **attrs))

    # Edges
    for (u, v), label in zip(all_edge_keys, y):
        if label == 1:
            edge_attrs = {
                "label": "1",
                "color": "red",
                "penwidth": "3",
            }
        else:
            edge_attrs = {
                "label": "0",
                "color": "gray70",
                "penwidth": "1",
            }
        dot.add_edge(pydot.Edge(u, v, **edge_attrs))

    dot_path = out_dir / "graph_edge_labels.dot"
    dot.write_raw(str(dot_path))
    _safe_write_svg(dot, out_dir / "graph_edge_labels.svg")

    print(f"Wrote edge labels to: {out_dir / 'edges.json'}")
    print(f"Wrote DOT graph to:  {dot_path}")
    if (out_dir / "graph_edge_labels.svg").exists():
        print(f"Wrote SVG graph to:  {out_dir / 'graph_edge_labels.svg'}")
    print(f"Chosen path kind: {chosen_kind} (positive edges: {sum(y)}/{len(y)})")


if __name__ == "__main__":
    main()


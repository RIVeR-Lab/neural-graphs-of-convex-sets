"""
Solve GCS planning queries from a precomputed plan-index CSV and log training targets to HDF5.

Inputs:
- A plan-index CSV produced by `planning_through_contact/dataset/create_plan_index.py`.

For each selected plan_id:
- Run the nominal GCS solve + rounding (same machinery as create_plans.py)
- Append a training record into an HDF5 file
- (Optional) save trajectory/graph artifacts for debugging
    - g (R^10) from the CSV (g_pose + g_entry)
    - edge keys (u_name, v_name) for each directed edge
    - y (0/1) from the chosen feasible rounded path
    - phi_star from the convex relaxation solution
    - sdp_solve_time and path_cost
    - candidate_paths/*: sampled paths with feasibility, cost, timing, and rank labels

Default behavior solves plan_id range [0, 5) (first 5).
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from planning_through_contact.experiments.utils import (
    create_output_folder,
    get_default_plan_config,
    get_default_solver_params,
)
from tqdm import tqdm
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarPushingStartAndGoal,
)
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPushingPlanner,
)
from planning_through_contact.planning.planar.utils import create_plan


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def _pose_from_row(row: dict[str, str], prefix: str) -> PlanarPose:
    return PlanarPose(
        _float(row, f"{prefix}_x"),
        _float(row, f"{prefix}_y"),
        _float(row, f"{prefix}_theta"),
    )


def _extract_g(row: dict[str, str]) -> np.ndarray:
    # Updated PDF: g_pose is 6 dims, g_entry is 4 dims => g ∈ R^10
    g_pose = np.array([_float(row, f"g_pose_{i}") for i in range(6)], dtype=np.float32)
    g_entry_cols = sorted([k for k in row.keys() if k.startswith("g_entry_")])
    g_entry = np.array([_float(row, k) for k in g_entry_cols], dtype=np.float32)
    return np.concatenate([g_pose, g_entry], axis=0)


def _float_or_nan(value: Any) -> float:
    return float("nan") if value is None else float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan_index_csv",
        type=str,
        default="planning_through_contact/dataset/data/sugar_box/global_features.csv",
        help="Path to plan-index CSV.",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=None,
        help="First plan_id to solve (inclusive).",
    )
    parser.add_argument(
        "--end_id",
        type=int,
        default=None,
        help="Stop plan_id (exclusive).",
    )
    parser.add_argument(
        "--trajectories_dir",
        type=str,
        default="planning_through_contact/dataset/trajectories",
        help="High-level output directory for saved trajectory artifacts.",
    )
    parser.add_argument(
        "--save_artifacts",
        action="store_true",
        default=False,
        help="Also save trajectory artifacts (mp4/pdf/pkl/svg/json) for debugging.",
    )
    parser.add_argument(
        "--h5_path",
        type=str,
        default=None,
        help="Output HDF5 path. Defaults to <plan_index_csv parent>/gcs_solutions.h5",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable solver debug output (slower, verbose).",
    )
    parser.add_argument(
        "--num_candidate_paths",
        type=int,
        default=100,
        help="Number of unique GCS-sampled candidate paths to save for RankNet labels.",
    )
    args = parser.parse_args()

    if (args.start_id is None) != (args.end_id is None):
        raise ValueError("Provide both --start_id and --end_id, or provide neither.")
    if args.num_candidate_paths <= 0:
        raise ValueError("--num_candidate_paths must be positive")

    csv_path = Path(args.plan_index_csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    # Lazily import h5py so error message is clearer.
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency `h5py`. Install it (e.g. `pip install h5py`) and retry."
        ) from e

    h5_path = (
        Path(args.h5_path)
        if args.h5_path is not None
        else csv_path.parent / "gcs_solutions.h5"
    )
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    # Load rows from the plan index (and optionally range-filter them).
    all_rows: dict[int, dict[str, str]] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = _int(row, "plan_id")
            all_rows[pid] = row

    if len(all_rows) == 0:
        raise RuntimeError(f"No rows found in {csv_path}")

    if args.start_id is None and args.end_id is None:
        start_id = min(all_rows.keys())
        end_id = max(all_rows.keys()) + 1
    else:
        start_id = int(args.start_id)
        end_id = int(args.end_id)
        if end_id <= start_id:
            raise ValueError("--end_id must be > --start_id")

    rows = {pid: r for pid, r in all_rows.items() if start_id <= pid < end_id}
    if len(rows) == 0:
        raise RuntimeError(f"No rows found in range [{start_id}, {end_id})")

    # Use row metadata to pick slider type etc. (assumes consistent within the CSV).
    first_row = rows[min(rows.keys())]
    body = first_row["body"]
    pusher_radius = float(first_row["pusher_radius"])

    config = get_default_plan_config(
        slider_type=body, pusher_radius=pusher_radius, use_case="normal"
    )
    solver_params = get_default_solver_params(args.debug, clarabel=False)
    solver_params.rounding_steps = int(args.num_candidate_paths)

    out_folder = None
    if args.save_artifacts:
        # Create an output folder similar to create_plans.py (timestamped).
        out_folder = create_output_folder(args.trajectories_dir, body, traj_number=None)

    str_dt = datetime.now().isoformat()
    with h5py.File(h5_path, "a") as h5:
        meta = h5.require_group("meta")
        meta.attrs["created_at"] = str_dt
        meta.attrs["plan_index_csv"] = str(csv_path)
        meta.attrs["body"] = body
        meta.attrs["start_id"] = int(start_id)
        meta.attrs["end_id"] = int(end_id)
        if out_folder is not None:
            meta.attrs["trajectories_folder"] = out_folder

        samples_grp = h5.require_group("samples")

        str_dtype = h5py.string_dtype(encoding="utf-8")

        for plan_id in tqdm(sorted(rows.keys()), desc="Solving plans"):
            row = rows[plan_id]

            # Build the start/goal for this plan_id.
            start_and_goal = PlanarPushingStartAndGoal(
                slider_initial_pose=_pose_from_row(row, "slider_init"),
                slider_target_pose=_pose_from_row(row, "slider_goal"),
                pusher_initial_pose=_pose_from_row(row, "pusher_init"),
                pusher_target_pose=_pose_from_row(row, "pusher_goal"),
            )

            # Solve in-memory (fast path): no disk artifacts required for HDF5 logging.
            config.start_and_goal = start_and_goal
            planner = PlanarPushingPlanner(config)
            planner.formulate_problem()

            relaxed_res = planner._solve(solver_params)
            planner.relaxed_gcs_result = relaxed_res
            if relaxed_res is None or not relaxed_res.is_success():
                continue

            rounding_profile: dict[str, Any] = {}
            candidate_paths = planner.get_solution_paths(
                relaxed_res,
                solver_params,
                profile=rounding_profile,
            )
            if candidate_paths is None:
                continue

            feasible_paths = planner._get_rounded_paths(
                solver_params,
                candidate_paths,
                profile=rounding_profile,
            )
            if feasible_paths is None:
                continue

            path = planner._pick_best_path(feasible_paths)
            planner.path = path

            if path.rounded_result is None or not path.rounded_result.is_success():
                continue  # only keep instances with at least one feasible rounded path

            all_edges = list(planner.gcs.Edges())
            edge_u = [e.u().name() for e in all_edges]
            edge_v = [e.v().name() for e in all_edges]

            # Teacher flows from convex relaxation.
            phi_star = np.array(
                [float(relaxed_res.GetSolution(e.phi())) for e in all_edges],
                dtype=np.float32,
            )

            # Binary labels + traversal order.
            path_edge_keys = [(e.u().name(), e.v().name()) for e in path.edges]
            path_orders: dict[tuple[str, str], list[int]] = {}
            for k, (u, v) in enumerate(path_edge_keys, start=1):
                path_orders.setdefault((u, v), []).append(k)

            y = np.array(
                [1 if (u, v) in set(path_edge_keys) else 0 for u, v in zip(edge_u, edge_v)],
                dtype=np.uint8,
            )

            # Ordered path edges (expanded if repeats).
            path_u = [u for (u, _) in path_edge_keys]
            path_v = [v for (_, v) in path_edge_keys]
            path_order = list(range(1, len(path_edge_keys) + 1))

            g = _extract_g(row)

            relaxed_solve_time = float(relaxed_res.get_solver_details().optimizer_time)  # type: ignore
            path_cost = float(path.rounded_result.get_optimal_cost())

            # Write to HDF5 (one group per plan_id)
            gname = f"{plan_id}"
            if gname in samples_grp:
                # Avoid accidental overwrite; skip.
                continue
            grp = samples_grp.create_group(gname)
            grp.attrs["plan_id"] = int(plan_id)
            grp.attrs["split"] = row.get("split", "")
            grp.attrs["seed"] = _int(row, "seed")
            grp.attrs["split_seed"] = _int(row, "split_seed")
            grp.attrs["entry_idx0"] = _int(row, "entry_idx0")

            grp.create_dataset("g", data=g, dtype=np.float32)
            grp.create_dataset("edge_u", data=np.array(edge_u, dtype=object), dtype=str_dtype)
            grp.create_dataset("edge_v", data=np.array(edge_v, dtype=object), dtype=str_dtype)
            grp.create_dataset("y", data=y, dtype=np.uint8)
            grp.create_dataset("phi_star", data=phi_star, dtype=np.float32)
            grp.create_dataset("sdp_solve_time", data=np.array(relaxed_solve_time, dtype=np.float32))
            grp.create_dataset("path_cost", data=np.array(path_cost, dtype=np.float32))
            grp.create_dataset("path_edge_u", data=np.array(path_u, dtype=object), dtype=str_dtype)
            grp.create_dataset("path_edge_v", data=np.array(path_v, dtype=object), dtype=str_dtype)
            grp.create_dataset("path_edge_order", data=np.array(path_order, dtype=np.int32), dtype=np.int32)

            # RankNet labels: all sampled candidate paths from ground-truth relaxation flows.
            candidate_entries = list(rounding_profile.get("paths", []))
            candidates_grp = grp.create_group("candidate_paths")
            candidates_grp.attrs["num_candidate_paths"] = len(candidate_entries)
            candidates_grp.attrs["path_sampling_s"] = _float_or_nan(
                rounding_profile.get("path_sampling_s")
            )
            candidates_grp.attrs["num_unique_paths"] = int(
                rounding_profile.get("num_unique_paths", len(candidate_entries))
            )
            candidates_grp.attrs["rounding_steps"] = int(solver_params.rounding_steps)
            candidates_grp.attrs["max_rounding_trials"] = int(solver_params.max_rounding_trials)

            feasible_candidate_idxs = [
                i
                for i, entry in enumerate(candidate_entries)
                if bool(entry.get("rounding_success", False))
            ]
            feasible_candidate_idxs.sort(
                key=lambda i: _float_or_nan(candidate_entries[i].get("rounded_cost"))
            )
            rank_by_idx = {idx: rank for rank, idx in enumerate(feasible_candidate_idxs)}

            for candidate_idx, entry in enumerate(candidate_entries):
                cand_grp = candidates_grp.create_group(f"{candidate_idx:03d}")
                feasible = bool(entry.get("rounding_success", False))
                cand_grp.attrs["path_index"] = int(entry.get("path_index", candidate_idx))
                cand_grp.attrs["feasible"] = feasible
                cand_grp.attrs["rank"] = int(rank_by_idx.get(candidate_idx, -1))
                cand_grp.attrs["convex_restriction_success"] = bool(
                    entry.get("convex_restriction_success", False)
                )
                cand_grp.attrs["rounding_success"] = feasible
                cand_grp.attrs["convex_restriction_s"] = _float_or_nan(
                    entry.get("convex_restriction_s")
                )
                cand_grp.attrs["convex_restriction_solver_s"] = _float_or_nan(
                    entry.get("convex_restriction_solver_s")
                )
                cand_grp.attrs["rounding_s"] = _float_or_nan(entry.get("rounding_s"))
                cand_grp.attrs["relaxed_cost"] = _float_or_nan(entry.get("relaxed_cost"))
                cand_grp.attrs["rounded_cost"] = _float_or_nan(entry.get("rounded_cost"))

                cand_edge_u = entry.get("path_edge_u", [])
                cand_edge_v = entry.get("path_edge_v", [])
                cand_edge_order = entry.get(
                    "path_edge_order", list(range(1, len(cand_edge_u) + 1))
                )
                cand_grp.create_dataset(
                    "edge_u", data=np.array(cand_edge_u, dtype=object), dtype=str_dtype
                )
                cand_grp.create_dataset(
                    "edge_v", data=np.array(cand_edge_v, dtype=object), dtype=str_dtype
                )
                cand_grp.create_dataset(
                    "edge_order",
                    data=np.array(cand_edge_order, dtype=np.int32),
                    dtype=np.int32,
                )

            # Optional slow debug artifacts.
            if args.save_artifacts and out_folder is not None:
                create_plan(
                    start_and_goal,
                    config,
                    solver_params,
                    output_folder=out_folder,
                    output_name=f"traj_{plan_id}",
                    save_video=True,
                    save_traj=True,
                    animation_lims=None,
                    interpolate_video=False,
                    do_rounding=True,
                    save_relaxed=True,
                    debug=args.debug,
                    save_graph_edge_labels=True,
                )

        print("Solve loop complete. Finalizing and saving HDF5 to disk...")

    if out_folder is not None:
        print(f"Trajectories saved under: {out_folder}")
    print(f"HDF5 written/appended at: {h5_path}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Collect IIWA shelf GCS dataset (linear + nonlinear) on a fixed IRIS region graph.

Samples random (q_start, q_goal) pairs in configuration space, solves convex
relaxation + rounded candidate paths, and writes two HDF5 files:
  - iiwa_gcs_linear.h5
  - iiwa_gcs_nonlinear.h5

Region halfspaces (A, b) are stored once in meta for PointNet encoding at train time.

Default split: 500 train / 100 val / 100 test (700 instances per file).

Example:
  bash scripts/iiwa_collect_data.sh
  python scripts/collect_iiwa_shelf_data.py --splits train --train-count 2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from tqdm import tqdm

logging.getLogger("drake").setLevel(logging.WARNING)

from pydrake.solvers import MosekSolver

from manipulation.dataset_collect import (
    LINEAR_FILENAME,
    NONLINEAR_FILENAME,
    SPLITS,
    build_adjacency,
    collect_linear_instance,
    collect_nonlinear_instance,
    count_split,
    next_instance_id,
    sample_start_goal,
    write_region_meta,
)
from manipulation.iiwa_helpers import build_shelf_plant
from manipulation.paths import DEFAULT_REGIONS_PATH, manipulation_models_hint, manipulation_models_ready
from manipulation.shelf_gcs import load_regions

MAX_QUERY_ATTEMPTS = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect IIWA shelf GCS dataset.")
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "manipulation" / "data",
        help="Directory for HDF5 outputs.",
    )
    parser.add_argument(
        "--regions", type=Path, default=DEFAULT_REGIONS_PATH,
        help="Pickle file with IRIS regions.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument(
        "--splits", type=str, default="train,val,test",
        help="Comma-separated splits to fill.",
    )
    parser.add_argument("--train-count", type=int, default=SPLITS["train"])
    parser.add_argument("--val-count", type=int, default=SPLITS["val"])
    parser.add_argument("--test-count", type=int, default=SPLITS["test"])
    parser.add_argument(
        "--planner", type=str, default="both", choices=("linear", "nonlinear", "both"),
        help="Which HDF5 file(s) to populate.",
    )
    return parser.parse_args()


def split_targets(args: argparse.Namespace) -> dict[str, int]:
    return {
        "train": args.train_count,
        "val": args.val_count,
        "test": args.test_count,
    }


def main() -> None:
    args = parse_args()
    if not manipulation_models_ready():
        print(manipulation_models_hint(), file=sys.stderr)
        sys.exit(1)
    if not args.regions.is_file():
        print(f"IRIS regions not found: {args.regions}", file=sys.stderr)
        print("Run: python scripts/iiwa_shelf_scenes.py --generate-regions --regions-only", file=sys.stderr)
        sys.exit(1)

    import h5py

    args.output_dir.mkdir(parents=True, exist_ok=True)
    linear_path = args.output_dir / LINEAR_FILENAME
    nonlinear_path = args.output_dir / NONLINEAR_FILENAME
    targets = split_targets(args)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    solver = MosekSolver()

    run_linear = args.planner in ("linear", "both")
    run_nonlinear = args.planner in ("nonlinear", "both")
    require_both = args.planner == "both"

    print("Building IIWA shelf scene...")
    plant, _, _, _ = build_shelf_plant()
    print(f"Loading IRIS regions from {args.regions}")
    regions = load_regions(args.regions)
    adj = build_adjacency(regions)
    print(f"  {len(regions)} regions, graph built.")

    linear_file = h5py.File(linear_path, "a") if run_linear else None
    nonlinear_file = h5py.File(nonlinear_path, "a") if run_nonlinear else None
    try:
        str_dtype = h5py.string_dtype(encoding="utf-8")

        linear_samples = None
        nonlinear_samples = None

        if run_linear:
            write_region_meta(linear_file, regions, str_dtype=str_dtype)
            linear_file.require_group("meta").attrs["base_seed"] = args.seed
            linear_file.require_group("meta").attrs["model"] = "linear"
            linear_samples = linear_file.require_group("samples")

        if run_nonlinear:
            write_region_meta(nonlinear_file, regions, str_dtype=str_dtype)
            nonlinear_file.require_group("meta").attrs["base_seed"] = args.seed
            nonlinear_file.require_group("meta").attrs["model"] = "nonlinear"
            nonlinear_samples = nonlinear_file.require_group("samples")

        for split in splits:
            if split not in targets:
                raise ValueError(f"Unknown split {split!r}")
            target = targets[split]
            if target <= 0:
                continue

            linear_saved = count_split(linear_samples, split) if run_linear else 0
            nonlinear_saved = count_split(nonlinear_samples, split) if run_nonlinear else 0
            instance_id = max(
                next_instance_id(linear_samples) if run_linear else 0,
                next_instance_id(nonlinear_samples) if run_nonlinear else 0,
            )

            print(f"\n=== {split.upper()} — target {target} per file ===")
            if run_linear and linear_saved:
                print(f"  Resuming linear at {linear_saved}/{target}")
            if run_nonlinear and nonlinear_saved:
                print(f"  Resuming nonlinear at {nonlinear_saved}/{target}")

            stats = {"query_fail": 0, "linear_fail": 0, "nonlinear_fail": 0}
            pbar = tqdm(
                initial=linear_saved if run_linear and not run_nonlinear else (
                    nonlinear_saved if run_nonlinear and not run_linear else min(linear_saved, nonlinear_saved)
                ),
                total=target,
                desc=split,
            )

            attempt = 0
            while (run_linear and linear_saved < target) or (run_nonlinear and nonlinear_saved < target):
                if attempt >= MAX_QUERY_ATTEMPTS * target:
                    print(f"  Stopping {split}: exceeded attempt budget.")
                    break

                query_seed = args.seed + hash((split, attempt)) % (2**31)
                rng = np.random.default_rng(query_seed)
                sampled = sample_start_goal(regions, adj, rng)
                attempt += 1
                if sampled is None:
                    stats["query_fail"] += 1
                    continue

                q_start, q_goal, _, _ = sampled
                linear_ok = not run_linear
                nonlinear_ok = not run_nonlinear

                if run_linear and linear_saved < target:
                    linear_ok = collect_linear_instance(
                        regions, q_start, q_goal,
                        solver=solver, str_dtype=str_dtype,
                        h5_samples=linear_samples,
                        instance_id=instance_id,
                        split=split,
                        query_seed=query_seed,
                    )
                    if not linear_ok:
                        stats["linear_fail"] += 1

                if run_nonlinear and nonlinear_saved < target:
                    nonlinear_ok = collect_nonlinear_instance(
                        regions, q_start, q_goal,
                        plant=plant, str_dtype=str_dtype,
                        h5_samples=nonlinear_samples,
                        instance_id=instance_id,
                        split=split,
                        query_seed=query_seed,
                    )
                    if not nonlinear_ok:
                        stats["nonlinear_fail"] += 1

                if require_both:
                    if linear_ok and nonlinear_ok:
                        instance_id += 1
                        linear_saved += 1
                        nonlinear_saved += 1
                        pbar.update(1)
                    else:
                        if linear_ok and str(instance_id) in linear_samples:
                            del linear_samples[str(instance_id)]
                        if nonlinear_ok and str(instance_id) in nonlinear_samples:
                            del nonlinear_samples[str(instance_id)]
                else:
                    accepted = False
                    if run_linear and linear_ok and linear_saved < target:
                        linear_saved += 1
                        accepted = True
                    if run_nonlinear and nonlinear_ok and nonlinear_saved < target:
                        nonlinear_saved += 1
                        accepted = True
                    if accepted:
                        instance_id += 1
                        pbar.update(1)

                postfix = {}
                if run_linear:
                    postfix["linear"] = linear_saved
                if run_nonlinear:
                    postfix["nc"] = nonlinear_saved
                pbar.set_postfix(**postfix)

            pbar.close()
            parts = []
            if run_linear:
                parts.append(f"linear={linear_saved}/{target}")
            if run_nonlinear:
                parts.append(f"nonlinear={nonlinear_saved}/{target}")
            print(
                f"  Saved {', '.join(parts)}  |  "
                f"failures query={stats['query_fail']} linear={stats['linear_fail']} "
                f"nc={stats['nonlinear_fail']}"
            )
    finally:
        if linear_file is not None:
            linear_file.close()
        if nonlinear_file is not None:
            nonlinear_file.close()

    if run_linear:
        print(f"\nLinear HDF5:    {linear_path}")
    if run_nonlinear:
        print(f"Nonlinear HDF5: {nonlinear_path}")


if __name__ == "__main__":
    main()
